"""Generate the pool of past-negotiation transcripts.

Each transcript is one negotiation between a BRIEFED counterparty and an UNBRIEFED
"past-you". The transcript later becomes one episode in some agent's seeded history, and
that agent is always told it was past-you -- the unbriefed side.

Two mirrored sub-pools fall out of that:
    briefed BUYER  -> past-you was the seller -> usable as SELLER history   (pool "S")
    briefed SELLER -> past-you was the buyer  -> usable as BUYER  history   (pool "B")
A single transcript cannot serve both roles; briefing both sides would leave no neutral
chair to seat the agent in.

Transcripts are binned by REALISED signed outcome, not by which brief produced them.
Briefs create variation; outcomes define conditions. Nothing is discarded -- a fair brief
that happens to land lopsided is simply filed as lopsided. This avoids selecting on
brief-success, which would leave us with only the transcripts where the brief "worked",
i.e. the most explicit ones.

Usage (from repo root):
    .venv/bin/python -m experiments.pool.generate --check
    .venv/bin/python -m experiments.pool.generate --batch first    # ~24, read these
    .venv/bin/python -m experiments.pool.generate --batch full     # the rest
    .venv/bin/python -m experiments.pool.generate --report data/pool_<stamp>.jsonl
"""

from __future__ import annotations

import argparse
import asyncio
import itertools
import json
import os
import time
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional

from anthropic import AsyncAnthropic
from pydantic import BaseModel, Field

from experiments.pool.briefs import BRIEFS, FAMILIES, KICKOFF, build_system
from experiments.pool.goods import BY_KEY, GOODS, Good

MODEL = "claude-sonnet-5"
MAX_TOKENS = 2048
CONCURRENCY = 12
THINKING = {"type": "adaptive"}
EFFORT = "low"
MAX_ROUNDS = 12

DATA_DIR = Path(__file__).parent / "data"

# How close to an even split counts as "even". 0.10 of the surplus -- for the Camry's
# $4,000 that is $19,600-$20,400.
EVEN_BAND = 0.10


def load_dotenv() -> str:
    env_path = Path(__file__).resolve().parents[2] / ".env"
    source = "shell environment"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, _, v = line.partition("=")
            k, v = k.strip(), v.strip().strip("'\"")
            if k == "ANTHROPIC_API_KEY" and v:
                source = str(env_path)
            os.environ[k] = v
    key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not key:
        raise SystemExit("No ANTHROPIC_API_KEY. Add it to .env at the repo root.")
    return f"...{key[-4:]} (len {len(key)}, from {source})"


# --- structured reads --------------------------------------------------------


class AgreementCheck(BaseModel):
    agreement_reached: bool = Field(
        description=(
            "True only if one party made an offer AND the other explicitly accepted it. "
            "A proposal or counter-offer is not agreement."
        )
    )
    transaction_price: Optional[int] = Field(
        default=None, description="Agreed price in whole dollars, if agreement was reached."
    )


class Outcome(BaseModel):
    agreement_reached: bool = Field(description="Did the parties clearly agree on a price?")
    transaction_price: Optional[int] = Field(
        default=None,
        description=(
            "Price in whole dollars at the FIRST clear agreement. Ignore later "
            "renegotiation. Null if no agreement."
        ),
    )
    agreement_round: Optional[int] = Field(default=None, description="Round of first agreement.")
    evidence: str = Field(description="Short supporting quote from the transcript.")


class TacticalNote(BaseModel):
    """Past-you's private reasoning at one turn.

    Deliberately excludes any verdict on what KIND of negotiation this is. The main
    experiment asks agents "what kind of situation is this?" -- if past-you's recorded
    thinking answers that question, the history hands the agent our answer ten times over
    and the probe measures copying. Tactical reasoning is fair game; situational
    assessment is not.
    """

    note: str = Field(
        description=(
            "2-3 sentences of private tactical reasoning at this moment: what you make of "
            "their last move, what you plan to do next, and what you expect from them. "
            "Concrete and specific to this negotiation. Do NOT characterise what type or "
            "category of negotiation this is; do not use words like 'cooperative', "
            "'adversarial', 'fair-division' or 'competitive' to label the situation."
        )
    )


# Broad on purpose: a false positive costs one cheap confirming call that returns
# agreement_reached=False, while a false negative lets the negotiation run past its close.
# "deal at" alone missed "Deal - $28,200 works for me", which cost seven rounds of
# post-agreement pleasantries.
ACCEPT_KEYWORDS = (
    "accept", "deal", "agreed", "sold", "done at", "i'll take it", "works for me",
    "that works", "you have yourself", "let's finalize", "let's close", "confirmed",
)


# --- one negotiation ---------------------------------------------------------


async def generate_one(
    client: AsyncAnthropic,
    idx: int,
    good: Good,
    family: str,
    briefed_role: str,
    state_range: bool = True,
) -> Dict:
    past_you_role = "seller" if briefed_role == "buyer" else "buyer"
    sys_briefed = build_system(good, briefed_role, BRIEFS[family], state_range)
    sys_past_you = build_system(good, past_you_role, None, state_range)

    sys_by_role = {briefed_role: sys_briefed, past_you_role: sys_past_you}
    msgs = {"buyer": [{"role": "user", "content": KICKOFF}], "seller": []}

    transcript: List[Dict] = []
    usage = {"input_tokens": 0, "output_tokens": 0}
    closed_at: Optional[int] = None

    for turn in range(MAX_ROUNDS):
        speaker = "buyer" if turn % 2 == 0 else "seller"
        listener = "seller" if speaker == "buyer" else "buyer"

        resp = await client.messages.create(
            model=MODEL, max_tokens=MAX_TOKENS,
            system=sys_by_role[speaker], messages=msgs[speaker],
            thinking=THINKING, output_config={"effort": EFFORT},
        )
        usage["input_tokens"] += resp.usage.input_tokens
        usage["output_tokens"] += resp.usage.output_tokens
        text = "".join(b.text for b in resp.content if b.type == "text").strip()

        entry = {"round": turn + 1, "role": speaker, "text": text}

        # Capture past-you's private reasoning OFF-THREAD: same context, separate call,
        # answer never fed back. The negotiation is bit-for-bit what it would have been.
        if speaker == past_you_role:
            try:
                note = await client.messages.parse(
                    model=MODEL, max_tokens=1024,
                    system=sys_by_role[speaker],
                    messages=msgs[speaker] + [
                        {"role": "assistant", "content": text},
                        {"role": "user", "content":
                            "Before the other side replies — privately, in 2-3 sentences: "
                            "what do you make of their last move, what will you do next, "
                            "and what do you expect from them?"},
                    ],
                    output_format=TacticalNote,
                )
                entry["thinking"] = note.parsed_output.note
            except Exception as exc:
                entry["thinking_error"] = f"{type(exc).__name__}"

        msgs[speaker].append({"role": "assistant", "content": text})
        msgs[listener].append({"role": "user", "content": text})
        transcript.append(entry)

        if turn >= 1 and any(k in text.lower() for k in ACCEPT_KEYWORDS):
            chk = await _check(client, transcript)
            if chk.agreement_reached and chk.transaction_price is not None:
                closed_at = turn + 1
                break

    out = await _parse(client, transcript)
    price = out.transaction_price if out.agreement_reached else None

    sellshare = (price - good.low) / good.surplus if price is not None else None
    if sellshare is None:
        past_you_share, bin_ = None, "no_deal"
    else:
        past_you_share = sellshare if past_you_role == "seller" else 1.0 - sellshare
        if abs(past_you_share - 0.5) <= EVEN_BAND:
            bin_ = "even"
        elif past_you_share > 0.5:
            bin_ = "favourable"
        else:
            bin_ = "unfavourable"

    return {
        "idx": idx,
        "model": MODEL,
        "good": good.key,
        "range": [good.low, good.high],
        # stamped so load_pool can detect records generated against an older definition
        "condition_note": good.condition_note,
        "state_range": state_range,
        "description": good.description,
        "family": family,             # which brief was used (generation device)
        "briefed_role": briefed_role,
        "past_you_role": past_you_role,
        "sub_pool": "S" if past_you_role == "seller" else "B",
        "transaction_price": price,
        "agreement_round": out.agreement_round,
        "rounds_used": len(transcript),
        "closed_early_at": closed_at,
        "sellshare": sellshare,
        "past_you_share": past_you_share,
        "bin": bin_,                  # realised outcome (defines conditions)
        "parser_evidence": out.evidence,
        "usage": usage,
        "transcript": transcript,
    }


async def _check(client: AsyncAnthropic, transcript: List[Dict]) -> AgreementCheck:
    tail = "\n\n".join(f"Round {t['round']} ({t['role']}): {t['text']}" for t in transcript[-3:])
    r = await client.messages.parse(
        model=MODEL, max_tokens=1024,
        messages=[{"role": "user", "content":
                   "Most recent turns of a negotiation. Has a deal been definitively "
                   "closed — one side offered a price and the other explicitly accepted "
                   f"it?\n\n{tail}"}],
        output_format=AgreementCheck,
    )
    return r.parsed_output


async def _parse(client: AsyncAnthropic, transcript: List[Dict]) -> Outcome:
    body = "\n\n".join(f"Round {t['round']} ({t['role']}): {t['text']}" for t in transcript)
    r = await client.messages.parse(
        model=MODEL, max_tokens=2048,
        messages=[{"role": "user", "content":
                   "Transcript of a negotiation. Did the parties reach a clear agreement "
                   "on a price, and if so what was it? An offer never accepted is not an "
                   "agreement. Report the FIRST clear agreement and ignore any later "
                   f"renegotiation.\n\n{body}"}],
        output_format=Outcome,
    )
    return r.parsed_output


# --- batches -----------------------------------------------------------------


def plan(batch: str) -> List[tuple]:
    """(good, family, briefed_role) combinations to run."""
    roles = ["buyer", "seller"]
    if batch == "first":
        # 3 families x 2 roles x 4 goods = 24. Enough to read and judge.
        goods = [BY_KEY[k] for k in ("camry", "motorcycle", "espresso", "pickup")]
    else:
        goods = GOODS
    combos = list(itertools.product(goods, FAMILIES, roles))
    if batch == "full":
        # more draws per cell for the two mirrored pools
        combos = combos * 2
    return combos


async def main_async(batch: str, out_path: Path) -> None:
    key = load_dotenv()
    client = AsyncAnthropic(max_retries=5)
    sem = asyncio.Semaphore(CONCURRENCY)
    lock = asyncio.Lock()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    done_idx = set()
    if out_path.exists():
        for line in out_path.read_text().splitlines():
            if line.strip():
                try:
                    done_idx.add(json.loads(line)["idx"])
                except Exception:
                    pass

    combos = plan(batch)
    todo = [(i, c) for i, c in enumerate(combos) if i not in done_idx]

    print(f"Using API key {key}")
    print(f"Output: {out_path}")
    if done_idx:
        print(f"Resuming: {len(done_idx)} done, {len(todo)} to run.")
    print(f"Generating {len(todo)} transcripts on {MODEL} (concurrency {CONCURRENCY})...")

    n_done, spend, t0 = 0, 0.0, time.time()

    async def guarded(i: int, combo) -> None:
        nonlocal n_done, spend
        good, family, briefed_role = combo
        async with sem:
            try:
                rec = await generate_one(client, i, good, family, briefed_role)
            except Exception as exc:
                rec = {"idx": i, "good": good.key, "family": family,
                       "briefed_role": briefed_role,
                       "error": f"{type(exc).__name__}: {exc}"}
            async with lock:
                with out_path.open("a") as fh:
                    fh.write(json.dumps(rec) + "\n")
                    fh.flush()
                    os.fsync(fh.fileno())
            n_done += 1
            if "usage" in rec:
                spend += (rec["usage"]["input_tokens"] * 2
                          + rec["usage"]["output_tokens"] * 10) / 1e6
            tag = rec.get("error", "")[:34] or f"{rec['family']:<9} {rec['bin']:<13} {rec['past_you_share'] if rec['past_you_share'] is not None else float('nan'):.3f}"
            print(f"  [{n_done}/{len(todo)}] {time.time()-t0:5.0f}s ~${spend:5.2f}  {rec['good']:<11} {tag}", flush=True)

    await asyncio.gather(*(guarded(i, c) for i, c in todo))
    print(f"\nDone. ~${spend:.2f} this session → {out_path}")
    report(out_path)


# --- report ------------------------------------------------------------------


def report(path: Path) -> None:
    recs = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    ok = [r for r in recs if "error" not in r]
    err = len(recs) - len(ok)

    print("\n" + "=" * 68)
    print(f"POOL  n={len(ok)} usable" + (f", {err} errors" if err else ""))
    print("=" * 68)

    # THE crossover table: does each brief actually land where intended?
    bins = ["even", "favourable", "unfavourable", "no_deal"]
    print(f"\ncrossover — brief used (rows) x realised bin (cols)\n")
    print(f"{'':<11}" + "".join(f"{b:>14}" for b in bins))
    for fam in FAMILIES:
        row = [r for r in ok if r["family"] == fam]
        counts = Counter(r["bin"] for r in row)
        print(f"{fam:<11}" + "".join(f"{counts.get(b,0):>14}" for b in bins))
    print("\n  near-diagonal ⇒ briefs produce distinct experiences and the bins are clean.")
    print("  fair→even, shark→unfavourable, pushover→favourable is the intended pattern.")

    # pool sizes actually available for history assembly
    print(f"\nusable episodes by sub-pool and bin")
    print(f"{'':<11}" + "".join(f"{b:>14}" for b in bins[:3]))
    for sp, label in (("S", "S (seller)"), ("B", "B (buyer)")):
        row = [r for r in ok if r.get("sub_pool") == sp]
        c = Counter(r["bin"] for r in row)
        print(f"{label:<11}" + "".join(f"{c.get(b,0):>14}" for b in bins[:3]))

    deals = [r for r in ok if r["past_you_share"] is not None]
    if deals:
        shares = sorted(r["past_you_share"] for r in deals)
        mid = shares[len(shares)//2]
        print(f"\npast-you share: median {mid:.3f}  min {shares[0]:.3f}  max {shares[-1]:.3f}")
        print(f"no-deal rate: {1 - len(deals)/len(ok):.1%}")
        # competence check: if past-you is systematically crushed, the pushover briefs
        # will not produce wins and the whole direction contrast fails.
        print(f"share of deals where past-you did better than even: "
              f"{sum(1 for s in shares if s > 0.5)/len(shares):.1%}")

    tok = sum(r["usage"]["input_tokens"] + r["usage"]["output_tokens"] for r in ok)
    print(f"\ntotal tokens {tok:,}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--batch", choices=["first", "full"], default="first")
    p.add_argument("--out", type=Path)
    p.add_argument("--report", type=Path)
    p.add_argument("--check", action="store_true")
    a = p.parse_args()

    if a.report:
        report(a.report)
        return
    if a.check:
        print(f"Using API key {load_dotenv()}")
        from anthropic import Anthropic
        raw = Anthropic().models.with_raw_response.retrieve(MODEL)
        print(f"Organization: {raw.headers.get('anthropic-organization-id')}")
        print(f"Auth OK. {raw.parse().display_name}")
        print(f"\nplanned batches: first={len(plan('first'))}  full={len(plan('full'))}")
        return

    out = a.out or DATA_DIR / f"pool_{a.batch}_{time.strftime('%Y%m%d-%H%M%S')}.jsonl"
    asyncio.run(main_async(a.batch, out))


if __name__ == "__main__":
    main()
