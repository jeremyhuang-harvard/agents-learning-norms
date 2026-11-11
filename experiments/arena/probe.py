"""Read what the seller thought it was doing, by replaying finished negotiations.

The arena engine saved no reasoning: it only tried to capture the model's own thinking
block, and at `effort: low` there isn't one. The pool generator does it properly -- a
DELIBERATE off-thread call after each turn -- and 100% of its past-you turns carry a note.
This file carries that mechanism into the arena, after the fact.

## Why after the fact is legitimate

The API is stateless. An agent's entire state at turn t IS its prompt. So rebuilding the
seller's exact system prompt and message list up to turn t and asking a question there is
not an approximation of an inline probe -- it is the same request.

⚠️ That is emphatically NOT the same as showing a judge the finished transcript and asking
what happened. A judge sees the ending. This sees exactly what the agent saw, in first
person, with the future withheld.

The one thing replay cannot recover is a SIMULTANEOUS reading: a note drawn now is a fresh
sample from the same distribution, not the specific one that accompanied the observed move.
For tactical reasoning that is a minor loss.

## Two questions, asked in SEPARATE calls

    situation  -- what kind of negotiation is this, and what do you make of them?
    strategy   -- what is your plan, your walk-away, and what if they hold firm?

Separate on purpose. Asked together, the stated strategy gets rationalised to fit the
stated category and the four-way table below collapses:

    category holds  + strategy holds   -> neglect of discrepant evidence (BGLS Prop 1)
    category holds  + strategy adapts  -> tactical adjustment WITHIN a category
    category switches + strategy switches -> genuine recategorisation
    category switches + strategy doesn't  -> the stated category is decorative

## Safety

Every probe re-derives the seller's history block and checks its sha against the one stored
on the negotiation record. A mismatch means the assembly code changed since the run, so the
replayed context is not the context that produced the transcript -- the probe refuses
rather than silently reading the wrong agent.

Usage:
    .venv/bin/python -m experiments.arena.probe --file experiments/arena/data/armA_full.jsonl --dry-run
    .venv/bin/python -m experiments.arena.probe --file experiments/arena/data/armA_full.jsonl
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
EVERY_NTH_SELLER_TURN = 2      # probe every other seller turn


class SituationRead(BaseModel):
    """The ASSUMPTION. Deliberately open-ended -- offering a menu of categories would
    hand the agent our answer and measure recognition of our labels, not its own."""

    situation: str = Field(
        description="In one short phrase, in your own words: what kind of negotiation is "
                    "this? Do not describe the price or the item — describe the kind of "
                    "encounter it is."
    )
    counterparty: str = Field(
        description="In one sentence, what do you make of the other side so far?"
    )
    expect_next: str = Field(
        description="In one sentence, what do you expect them to do next?"
    )
    confidence: int = Field(
        description="1 to 5, how sure are you of that reading of them?", ge=1, le=5
    )


class StrategyRead(BaseModel):
    """The GUARANTEE. What the agent intends to do, independent of how it labels things."""

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


_SITUATION_Q = ("Before they reply — privately, for your own benefit only. Answer briefly "
                "and honestly; nobody else will see this.")
_STRATEGY_Q = ("Before they reply — privately, for your own benefit only. Set out where you "
               "stand. Nobody else will see this.")


def rebuild_seller_context(rec: Dict, blocks: Dict[str, str], upto_round: int):
    """The seller's exact system prompt and message list just after it spoke at `upto_round`.

    Mirrors the engine: the seller's list starts empty and receives the buyer's turns as
    `user` and its own as `assistant`. The buyer's kickoff never enters it.
    """
    block = blocks[rec["seller_id"]]
    if R._sha(block) != rec["seller_sha"]:
        raise ValueError(
            f"history block for {rec['seller_id']} no longer matches the sha recorded at "
            f"run time ({rec['seller_sha']}). The assembly code changed; this replay would "
            f"probe a different agent than the one that produced the transcript."
        )
    system = NEG.build_agent_system(BY_KEY[rec["good"]], "seller", history_block=block)
    msgs = []
    for t in rec["transcript"]:
        if t["round"] > upto_round:
            break
        msgs.append({"role": "assistant" if t["role"] == "seller" else "user",
                     "content": t["text"]})
    return system, msgs


def probe_points(rec: Dict) -> List[int]:
    rounds = [t["round"] for t in rec["transcript"] if t["role"] == "seller"]
    return rounds[::EVERY_NTH_SELLER_TURN]


async def one_probe(client, system, msgs, question: str, schema):
    r = await client.messages.parse(
        model=NEG.MODEL, max_tokens=1024, system=system,
        messages=msgs + [{"role": "user", "content": question}],
        output_format=schema,
    )
    return r.parsed_output, r.usage


async def main_async(path: Path, dry: bool, limit: int) -> None:
    recs = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    recs = [r for r in recs if r.get("sellshare") is not None]

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

    out_path = path.with_suffix(".probes.jsonl")
    done = set()
    if out_path.exists():
        for l in out_path.read_text().splitlines():
            if l.strip():
                p = json.loads(l)
                done.add((p["seller_id"], p["buyer_id"], p["j"], p["round"]))
    jobs = [(r, t) for r, t in jobs
            if (r["seller_id"], r["buyer_id"], r["j"], t) not in done]

    print(f"{len(recs)} negotiations -> {len(jobs)} probe points "
          f"(every {EVERY_NTH_SELLER_TURN}nd seller turn), 2 calls each")
    print(f"~${len(jobs)*2*0.0075:.2f} estimated")
    # fail fast on drift before spending anything
    rebuild_seller_context(recs[0], blocks, probe_points(recs[0])[0])
    print("sha check passed on the first record — replayed context matches run time")
    if dry:
        print("\n--dry-run: nothing sent.")
        return

    key = load_dotenv()
    print(f"Using API key {key}\nModel: {NEG.MODEL}\n")
    client = AsyncAnthropic(max_retries=5)
    sem, lock = asyncio.Semaphore(CONCURRENCY), asyncio.Lock()
    n, spend, t0 = 0, 0.0, time.time()

    async def guarded(rec, rnd):
        nonlocal n, spend
        async with sem:
            try:
                system, msgs = rebuild_seller_context(rec, blocks, rnd)
                sit, u1 = await one_probe(client, system, msgs, _SITUATION_Q, SituationRead)
                strat, u2 = await one_probe(client, system, msgs, _STRATEGY_Q, StrategyRead)
                p = {"seller_id": rec["seller_id"], "arena": rec["arena"],
                     "buyer_id": rec["buyer_id"], "buyer_comp": rec["buyer_comp"],
                     "j": rec["j"], "round": rnd, "sellshare": rec["sellshare"],
                     "situation": sit.model_dump(), "strategy": strat.model_dump()}
                for u in (u1, u2):
                    spend += (u.input_tokens * 2 + u.output_tokens * 10
                              + getattr(u, "cache_read_input_tokens", 0) * 0.2) / 1e6
            except Exception as exc:
                p = {"seller_id": rec["seller_id"], "j": rec["j"], "round": rnd,
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
    a = p.parse_args()
    asyncio.run(main_async(a.file, a.dry_run, a.limit))


if __name__ == "__main__":
    main()
