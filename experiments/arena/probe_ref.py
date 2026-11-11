"""Reference-point probes, by replay.

Built for the no-range condition, but runs identically on range-posted negotiations so the
two are directly comparable — that comparison IS the experiment.

## Why this probe exists

With the market range posted, $20,000 is visible arithmetic: the midpoint of a number both
sides can read. Remove it and neither agent can compute it — the seller knows only its own
$18,000, the buyer only its own $22,000, and the template tells the buyer not to disclose.
So landing on 50-50 stops being arithmetic and becomes an inference, and the question
becomes: what is the agent anchoring on instead?

## Three questions, three SEPARATE calls

    1. REFERENCE POINT   what is this worth, and what are you basing that on?
    2. FAIRNESS          what would a fair price be here?
    3. STRATEGY          what is your plan and your walk-away?

Separate because in Arm A the stated STRATEGY turned out to be decorative -- it did not
predict the agent's own outcome (r = -0.01 across 80 sellers). A stated fairness JUDGMENT
may not be. Asking them together lets each be rationalised to fit the others and destroys
exactly that comparison.

Probe 1 also asks what the agent thinks the COUNTERPARTY's walk-away is. With the range
posted that is nearly given; without it, it has to be inferred, and it is the belief the
range was silently supplying.

## Where we probe

    round 0   BEFORE the negotiation starts — an empty message list. A clean prior: what do
              you think this is worth, before anyone has said anything? Only questions 1
              and 2 apply; there is no plan yet.
    then      every Nth seller turn — shows whether the reference point MOVES as offers
              land, which is the interesting version and the reason not to ask only once.

## Safety

Re-derives each seller's history block and checks its sha against the one recorded at run
time. A mismatch means the assembly code changed and the replayed context is not the one
that produced the transcript; the probe refuses rather than reading the wrong agent.

Usage:
    .venv/bin/python -m experiments.arena.probe_ref --file <run.jsonl> --dry-run
    .venv/bin/python -m experiments.arena.probe_ref --file <run.jsonl>
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
from pathlib import Path
from typing import Dict, List, Optional

from anthropic import AsyncAnthropic
from pydantic import BaseModel, Field

from experiments.arena import history as H
from experiments.arena import negotiate as NEG
from experiments.arena import run as R
from experiments.pool.generate import load_dotenv
from experiments.pool.goods import BY_KEY

CONCURRENCY = 10
EVERY_NTH_SELLER_TURN = 2


class ReferencePoint(BaseModel):
    worth: int = Field(
        description="In whole dollars, what do you think this item is actually worth right "
                    "now? One number, your honest estimate — not an asking price and not a "
                    "negotiating position."
    )
    basis: str = Field(
        description="In one sentence, what are you basing that number on?"
    )
    their_walkaway: Optional[int] = Field(
        default=None,
        description="In whole dollars, what do you think the other side's best alternative "
                    "is — the most they could pay elsewhere. Null if you have no idea."
    )
    confidence: int = Field(
        description="1 to 5, how sure are you of your own valuation?", ge=1, le=5
    )


class FairnessRead(BaseModel):
    fair_price: int = Field(
        description="In whole dollars, what price would be FAIR here — not what is good for "
                    "you, and not what you expect to get. One number."
    )
    why: str = Field(
        description="In one sentence, what makes that price the fair one?"
    )
    fair_differs_from_good: bool = Field(
        description="True if the fair price differs from the price that would be best for "
                    "you."
    )


class StrategyRead(BaseModel):
    plan: str = Field(
        description="In one or two sentences, what is your plan for the rest of this "
                    "negotiation?"
    )
    walkaway: Optional[int] = Field(
        default=None,
        description="The lowest price in whole dollars you would accept rather than take "
                    "your fallback. Null if you have not fixed one."
    )
    if_they_hold: str = Field(
        description="In one sentence: what will you do if they refuse to move from their "
                    "current number?"
    )


_PRIVATE = "Privately, for your own benefit only — nobody else will see this. "
_Q_REF = _PRIVATE + "Set out how you are valuing this."
_Q_FAIR = _PRIVATE + "Set out what you think a fair outcome here would look like."
_Q_STRAT = _PRIVATE + "Set out where you stand and what you intend to do."


def rebuild(rec: Dict, blocks: Dict[str, str], upto_round: int):
    """Seller's exact system prompt and message list just after it spoke at `upto_round`.

    upto_round=0 gives an EMPTY message list — the state before the buyer has said
    anything, which is the clean prior.
    """
    block = blocks[rec["seller_id"]]
    if R._sha(block) != rec["seller_sha"]:
        raise ValueError(f"{rec['seller_id']}: history block no longer matches the sha "
                         f"recorded at run time — assembly code changed since the run")
    system = NEG.build_agent_system(BY_KEY[rec["good"]], "seller", history_block=block,
                                    state_range=rec.get("state_range", True))
    msgs = [{"role": "assistant" if t["role"] == "seller" else "user",
             "content": t["text"]}
            for t in rec["transcript"] if t["round"] <= upto_round]
    return system, msgs


ROUND0_ONLY = False


def probe_points(rec: Dict) -> List[int]:
    """Round 0 (the prior) plus every Nth seller turn — or round 0 alone.

    Round 0 is the cleanest reference-point measurement there is: the agent is asked what
    the item is worth and what would be fair BEFORE anyone has said anything, so nothing in
    the answer can have come from the negotiation. With the range posted, "fair" came back
    as exactly the midpoint every time. Blind, there is no midpoint to compute, and round 0
    is where we find out what fills the vacuum.

    The later points show whether the reference point MOVES as offers land. Interesting, but
    second-order, and four times the cost.
    """
    if ROUND0_ONLY:
        return [0]
    rounds = [t["round"] for t in rec["transcript"] if t["role"] == "seller"]
    return [0] + rounds[::EVERY_NTH_SELLER_TURN]


async def main_async(path: Path, dry: bool, limit: int) -> None:
    recs = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    recs = [r for r in recs if r.get("sellshare") is not None]

    # Read blocks from the run's own sidecar. Re-deriving them is wrong: assemble() draws
    # from the pool, so any transcript added since the run silently changes every history.
    ag_path = path.with_suffix(".agents.jsonl")
    blocks = {}
    for l in ag_path.read_text().splitlines():
        if l.strip():
            a = json.loads(l)
            if a["role"] == "seller":
                blocks[a["agent_id"]] = a["block"]

    jobs = [(r, t) for r in recs for t in probe_points(r)]
    if limit:
        jobs = jobs[:limit]
    out_path = path.with_suffix(".refprobes.jsonl")
    done = set()
    if out_path.exists():
        for l in out_path.read_text().splitlines():
            if l.strip():
                try:
                    p = json.loads(l)
                    done.add((p["seller_id"], p["buyer_id"], p["j"], p["round"]))
                except Exception:
                    pass
    jobs = [(r, t) for r, t in jobs
            if (r["seller_id"], r["buyer_id"], r["j"], t) not in done]

    n_calls = sum(2 if t == 0 else 3 for _, t in jobs)
    print(f"{len(recs)} negotiations -> {len(jobs)} probe points -> {n_calls} calls")
    print(f"~${n_calls*0.0075:.2f} estimated")
    rebuild(recs[0], blocks, 0)
    print("sha check passed — replayed context matches run time")
    if dry:
        print("\n--dry-run: nothing sent.")
        return

    print(f"Using API key {load_dotenv()}\nModel: {NEG.MODEL}\n")
    client = AsyncAnthropic(max_retries=5)
    sem, lock = asyncio.Semaphore(CONCURRENCY), asyncio.Lock()
    n, spend, t0 = 0, 0.0, time.time()

    async def ask(system, msgs, q, schema):
        r = await client.messages.parse(
            model=NEG.MODEL, max_tokens=1024, system=system,
            messages=msgs + [{"role": "user", "content": q}], output_format=schema)
        return r.parsed_output, r.usage

    async def guarded(rec, rnd):
        nonlocal n, spend
        async with sem:
            try:
                system, msgs = rebuild(rec, blocks, rnd)
                ref, u1 = await ask(system, msgs, _Q_REF, ReferencePoint)
                fair, u2 = await ask(system, msgs, _Q_FAIR, FairnessRead)
                us = [u1, u2]
                strat = None
                if rnd > 0:
                    strat, u3 = await ask(system, msgs, _Q_STRAT, StrategyRead)
                    us.append(u3)
                lo, hi = rec["range"]
                # everything an analyst needs, flat, no joins required
                p = {"seller_id": rec["seller_id"], "arena": rec["arena"],
                     "arm": rec.get("arm"), "buyer_id": rec["buyer_id"],
                     "buyer_comp": rec["buyer_comp"], "j": rec["j"], "round": rnd,
                     "state_range": rec.get("state_range", True),
                     "good": rec["good"], "range_lo": lo, "range_hi": hi,
                     "midpoint": (lo + hi) / 2,
                     "seller_fallback": lo, "buyer_fallback": hi,
                     "final_price": rec["transaction_price"],
                     "final_sellshare": rec["sellshare"],
                     "rounds_used": rec["rounds_used"],
                     "turn_text": next((t["text"] for t in rec["transcript"]
                                        if t["round"] == rnd), None),
                     "reference": ref.model_dump(),
                     "fairness": fair.model_dump(),
                     "strategy": strat.model_dump() if strat else None,
                     # pre-computed shares so plots need no arithmetic
                     "worth_share": (ref.worth - lo) / (hi - lo),
                     "fair_share": (fair.fair_price - lo) / (hi - lo),
                     "their_walkaway_share": ((ref.their_walkaway - lo) / (hi - lo)
                                              if ref.their_walkaway else None)}
                for u in us:
                    spend += (u.input_tokens * 2 + u.output_tokens * 10
                              + getattr(u, "cache_read_input_tokens", 0) * 0.2) / 1e6
            except Exception as exc:
                p = {"seller_id": rec["seller_id"], "j": rec["j"], "round": rnd,
                     "buyer_id": rec.get("buyer_id"),
                     "error": f"{type(exc).__name__}: {str(exc)[:140]}"}
            async with lock:
                with out_path.open("a") as fh:
                    fh.write(json.dumps(p) + "\n"); fh.flush(); os.fsync(fh.fileno())
                n += 1
                if n % 50 == 0 or n == len(jobs):
                    print(f"  {n}/{len(jobs)}  {time.time()-t0:5.0f}s  ~${spend:.2f}",
                          flush=True)

    await asyncio.gather(*(guarded(r, t) for r, t in jobs))
    print(f"\nDone. ~${spend:.2f} -> {out_path}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--file", type=Path, required=True)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--round0", action="store_true",
                   help="probe only the pre-negotiation prior — the clean reference point")
    a = p.parse_args()
    global ROUND0_ONLY
    ROUND0_ONLY = a.round0
    asyncio.run(main_async(a.file, a.dry_run, a.limit))


if __name__ == "__main__":
    main()
