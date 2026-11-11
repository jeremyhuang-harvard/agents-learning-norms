"""Does the buyer-favouring skew come from moving first?

Imas et al. report a buyer-favouring skew in their agentic distribution (mean sellshare
0.35, 80% below 0.5, a spike at zero surplus to sellers) and conjecture it comes from
participants having more experience buying used cars than selling them -- i.e. better
buyer prompts, machine fluency.

But the buyer opens in EVERY negotiation, in both arms of his design:
    template: "The negotiation starts with the buyer's first message (round 1)"
    method:   "Each simulated negotiation began with the Buyer making the first offer."
He never varies it, so first-mover advantage is untested and confounded with role.

Our pool run already found a ~0.14 buyer advantage with NO strategy prompts on either
side, which rules out machine fluency as the whole story. This isolates the cause.

Design: both sides completely unbriefed. The ONLY thing that varies is who speaks first.
    buyer_first   -- Imas's protocol
    seller_first  -- flipped

If the advantage tracks the OPENER rather than the ROLE, first-mover anchoring is the
mechanism and the experience conjecture is not needed to explain the skew.

Model is imported from the pool module so it cannot drift: claude-sonnet-5.

Usage:
    .venv/bin/python -m experiments.firstmover.run --n 4
    .venv/bin/python -m experiments.firstmover.run --report data/<file>.jsonl
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import statistics as st
import time
from pathlib import Path
from typing import Dict, List

from anthropic import AsyncAnthropic

from experiments.pool.generate import (
    ACCEPT_KEYWORDS, CONCURRENCY, EFFORT, MAX_ROUNDS, MAX_TOKENS, MODEL, THINKING,
    _check, _parse, load_dotenv,
)
from experiments.pool.goods import GOODS, Good

DATA_DIR = Path(__file__).parent / "data"

# Imas's structure block, with the opener parameterised. This one line is the treatment.
_STRUCTURE = """Negotiation structure:
- This negotiation will last up to 12 rounds (6 offers from each side)
- You must reach agreement within these 12 rounds or the deal fails
- The negotiation starts with the {opener}'s first message (round 1)
- Your final message should clearly state acceptance or rejection
- If you're close to agreement in the final round, consider accepting to avoid failure"""

KICKOFF = "Let's begin the negotiation. Please make your initial offer and inquiries."


def build(good: Good, role: str, opener: str) -> str:
    """Imas's template with NO strategy slot at all, and the opener parameterised.

    The disclosure asymmetry (seller DISCLOSE STRATEGICALLY / buyer DO NOT DISCLOSE) is
    his and is held constant across both conditions, so it cannot explain the difference.
    """
    if role == "seller":
        head = (f"You are my negotiation agent for selling {good.description}. Your goal is "
                f"to maximize my surplus (difference between final price and my fallback price).")
        disc = "DISCLOSE STRATEGICALLY"
        fb = f"I can sell to a dealer at ${good.low:,}"
    elif role == "buyer":
        head = (f"You are my negotiation agent for purchasing {good.description}. Your goal "
                f"is to maximize my surplus (difference between my fallback price and final price).")
        disc = "DO NOT DISCLOSE DIRECTLY"
        fb = f"I can buy a comparable one elsewhere for ${good.high:,}"
    else:
        raise ValueError(role)
    return "\n".join([
        head, "", _STRUCTURE.format(opener=opener), "",
        "Negotiation parameters (common knowledge):",
        f"- Market value range is ${good.low:,} to ${good.high:,} depending on condition",
        f"- {good.condition_note}", "",
        f"Private information ({disc}):", f"- Fallback option: {fb}",
    ])


async def one(client: AsyncAnthropic, idx: int, good: Good, opener: str) -> Dict:
    other = "seller" if opener == "buyer" else "buyer"
    sysmsg = {r: build(good, r, opener) for r in ("buyer", "seller")}
    msgs: Dict[str, List[Dict]] = {opener: [{"role": "user", "content": KICKOFF}], other: []}

    transcript: List[Dict] = []
    usage = {"input_tokens": 0, "output_tokens": 0}
    closed_at = None

    for turn in range(MAX_ROUNDS):
        speaker = opener if turn % 2 == 0 else other
        listener = other if speaker == opener else opener
        r = await client.messages.create(
            model=MODEL, max_tokens=MAX_TOKENS, system=sysmsg[speaker],
            messages=msgs[speaker], thinking=THINKING, output_config={"effort": EFFORT},
        )
        usage["input_tokens"] += r.usage.input_tokens
        usage["output_tokens"] += r.usage.output_tokens
        text = "".join(b.text for b in r.content if b.type == "text").strip()
        msgs[speaker].append({"role": "assistant", "content": text})
        msgs[listener].append({"role": "user", "content": text})
        transcript.append({"round": turn + 1, "role": speaker, "text": text})

        if turn >= 1 and any(k in text.lower() for k in ACCEPT_KEYWORDS):
            chk = await _check(client, transcript)
            if chk.agreement_reached and chk.transaction_price is not None:
                closed_at = turn + 1
                break

    out = await _parse(client, transcript)
    price = out.transaction_price if out.agreement_reached else None
    return {
        "idx": idx, "model": MODEL, "good": good.key, "range": [good.low, good.high],
        "opener": opener,
        "transaction_price": price,
        "sellshare": (price - good.low) / good.surplus if price is not None else None,
        "rounds_used": len(transcript), "closed_early_at": closed_at,
        "agreement_round": out.agreement_round, "parser_evidence": out.evidence,
        "usage": usage, "transcript": transcript,
    }


async def main_async(n: int, out_path: Path) -> None:
    key = load_dotenv()
    client = AsyncAnthropic(max_retries=5)
    sem, lock = asyncio.Semaphore(CONCURRENCY), asyncio.Lock()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    combos = [(g, op) for g in GOODS for op in ("buyer", "seller") for _ in range(n)]
    print(f"Using API key {key}")
    print(f"Model: {MODEL}")
    print(f"{len(combos)} negotiations — both sides unbriefed, only the opener varies\n")
    done, spend, t0 = 0, 0.0, time.time()

    async def guarded(i, combo):
        nonlocal done, spend
        g, op = combo
        async with sem:
            try:
                rec = await one(client, i, g, op)
            except Exception as e:
                rec = {"idx": i, "good": g.key, "opener": op, "error": type(e).__name__}
            async with lock:
                with out_path.open("a") as fh:
                    fh.write(json.dumps(rec) + "\n")
                    fh.flush()
                    os.fsync(fh.fileno())
            done += 1
            if "usage" in rec:
                spend += (rec["usage"]["input_tokens"] * 2
                          + rec["usage"]["output_tokens"] * 10) / 1e6
            if done % 8 == 0:
                print(f"  {done}/{len(combos)}  {time.time()-t0:5.0f}s  ~${spend:.2f}", flush=True)

    await asyncio.gather(*(guarded(i, c) for i, c in enumerate(combos)))
    print(f"\nDone. ~${spend:.2f} → {out_path}")
    report(out_path)


def report(path: Path) -> None:
    recs = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    ok = [r for r in recs if "error" not in r and r.get("sellshare") is not None]
    errs = len(recs) - len(ok)

    print("\n" + "=" * 64)
    print(f"WHO MOVES FIRST   n={len(ok)}" + (f"  ({errs} dropped)" if errs else ""))
    print("both sides unbriefed — only the opening turn differs")
    print("=" * 64)

    print(f"\n{'':<16}{'n':>4}{'mean sellshare':>17}{'median':>10}{'P(sell>0.5)':>14}")
    means = {}
    for op in ("buyer", "seller"):
        v = [r["sellshare"] for r in ok if r["opener"] == op]
        if not v:
            continue
        means[op] = st.mean(v)
        print(f"{op + ' opens':<16}{len(v):>4}{st.mean(v):>17.3f}{st.median(v):>10.3f}"
              f"{sum(1 for x in v if x > 0.5) / len(v):>14.0%}")

    if len(means) == 2:
        d = means["seller"] - means["buyer"]
        print(f"\n  seller's share moves {d:+.3f} when the SELLER opens instead of the buyer.")
        print(f"  → moving first is worth ~{abs(d):.3f} of the surplus to whoever does it.")
        print(f"\n  Imas agentic mean sellshare 0.350 (his buyer always opens)")
        print(f"  ours, buyer opens           {means['buyer']:.3f}")

    print(f"\nby good")
    print(f"  {'good':<12}{'buyer opens':>13}{'seller opens':>14}{'swing':>9}")
    for g in GOODS:
        b = [r["sellshare"] for r in ok if r["good"] == g.key and r["opener"] == "buyer"]
        s = [r["sellshare"] for r in ok if r["good"] == g.key and r["opener"] == "seller"]
        if b and s:
            print(f"  {g.key:<12}{st.mean(b):>13.3f}{st.mean(s):>14.3f}{st.mean(s)-st.mean(b):>+9.3f}")

    nod = sum(1 for r in recs if "error" not in r and r.get("sellshare") is None)
    print(f"\nno-deal: {nod}/{len(recs)-errs}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n", type=int, default=4, help="draws per (good, opener) cell")
    p.add_argument("--out", type=Path)
    p.add_argument("--report", type=Path)
    a = p.parse_args()
    if a.report:
        report(a.report)
        return
    out = a.out or DATA_DIR / f"firstmover_{time.strftime('%Y%m%d-%H%M%S')}.jsonl"
    asyncio.run(main_async(a.n, out))


if __name__ == "__main__":
    main()
