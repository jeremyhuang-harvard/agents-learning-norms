"""Fill the 3x3 interior of the buyer-type x seller-type grid.

The pool was built with exactly ONE side briefed, because a history episode needs an
unbriefed chair to seat the agent in. That constraint is right for history generation and
irrelevant for mapping the support -- so the nine cells where BOTH sides carry a
disposition have never been run, including fair-vs-fair, the most obvious source of even
splits.

Both sides get FAMILY briefs (fair / shark / pushover). The tactics vocabulary
(firm/moderate/yielding) is deliberately not used here -- it overlaps with families and
having two vocabularies for one prompt slot fragments the design space.

Sonnet 5, so results are comparable with the existing edge cells.

Usage:
    .venv/bin/python -m experiments.pool.interior --draws 3
    .venv/bin/python -m experiments.pool.interior --report
"""

from __future__ import annotations

import argparse
import asyncio
import itertools
import json
import os
import statistics as st
import time
from pathlib import Path
from typing import Dict, List

from anthropic import AsyncAnthropic

from experiments.pool.briefs import BRIEFS, FAMILIES, KICKOFF, build_system
from experiments.pool.generate import (
    ACCEPT_KEYWORDS, EFFORT, MAX_ROUNDS, MAX_TOKENS, MODEL, THINKING,
    _check, _parse, load_dotenv,
)
from experiments.pool.goods import GOODS, Good

DATA_DIR = Path(__file__).parent / "data"
OUT = DATA_DIR / "interior.jsonl"
CONCURRENCY = 12


async def one(client: AsyncAnthropic, idx: int, good: Good, bfam: str, sfam: str) -> Dict:
    sysm = {"buyer": build_system(good, "buyer", BRIEFS[bfam]),
            "seller": build_system(good, "seller", BRIEFS[sfam])}
    msgs: Dict[str, List[Dict]] = {"buyer": [{"role": "user", "content": KICKOFF}], "seller": []}
    transcript, usage, closed = [], {"input_tokens": 0, "output_tokens": 0}, None

    for turn in range(MAX_ROUNDS):
        sp = "buyer" if turn % 2 == 0 else "seller"
        li = "seller" if sp == "buyer" else "buyer"
        r = await client.messages.create(
            model=MODEL, max_tokens=MAX_TOKENS, system=sysm[sp], messages=msgs[sp],
            thinking=THINKING, output_config={"effort": EFFORT},
        )
        usage["input_tokens"] += r.usage.input_tokens
        usage["output_tokens"] += r.usage.output_tokens
        tx = "".join(b.text for b in r.content if b.type == "text").strip()
        msgs[sp].append({"role": "assistant", "content": tx})
        msgs[li].append({"role": "user", "content": tx})
        transcript.append({"round": turn + 1, "role": sp, "text": tx})
        if turn >= 1 and any(k in tx.lower() for k in ACCEPT_KEYWORDS):
            ck = await _check(client, transcript)
            if ck.agreement_reached and ck.transaction_price is not None:
                closed = turn + 1
                break

    o = await _parse(client, transcript)
    p = o.transaction_price if o.agreement_reached else None
    return {
        "idx": idx, "model": MODEL, "buyer_family": bfam, "seller_family": sfam,
        "good": good.key, "range": [good.low, good.high], "transaction_price": p,
        "sellshare": (p - good.low) / good.surplus if p is not None else None,
        "rounds_used": len(transcript), "closed_early_at": closed,
        "agreement_round": o.agreement_round, "parser_evidence": o.evidence,
        "usage": usage, "transcript": transcript,
    }


async def main_async(draws: int) -> None:
    key = load_dotenv()
    client = AsyncAnthropic(max_retries=5)
    sem, lock = asyncio.Semaphore(CONCURRENCY), asyncio.Lock()
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    done = set()
    if OUT.exists():
        for l in OUT.read_text().splitlines():
            if l.strip():
                try:
                    done.add(json.loads(l)["idx"])
                except Exception:
                    pass

    combos = [(g, b, s) for b, s in itertools.product(FAMILIES, FAMILIES) for g in GOODS
              for _ in range(draws)]
    todo = [(i, c) for i, c in enumerate(combos) if i not in done]
    print(f"Using API key {key}\nModel: {MODEL}")
    print(f"{len(todo)} negotiations ({len(FAMILIES)**2} cells x {len(GOODS)} goods x {draws})")
    n, spend, t0 = 0, 0.0, time.time()

    async def guarded(i, c):
        nonlocal n, spend
        g, b, s = c
        async with sem:
            try:
                rec = await one(client, i, g, b, s)
            except Exception as e:
                rec = {"idx": i, "buyer_family": b, "seller_family": s, "good": g.key,
                       "error": f"{type(e).__name__}: {str(e)[:100]}"}
            async with lock:
                with OUT.open("a") as fh:
                    fh.write(json.dumps(rec) + "\n"); fh.flush(); os.fsync(fh.fileno())
            n += 1
            if "usage" in rec:
                spend += (rec["usage"]["input_tokens"] * 2
                          + rec["usage"]["output_tokens"] * 10) / 1e6
            if n % 15 == 0 or n == len(todo):
                print(f"  {n}/{len(todo)}  {time.time()-t0:5.0f}s  ~${spend:.2f}", flush=True)

    await asyncio.gather(*(guarded(i, c) for i, c in todo))
    print(f"\nDone. ~${spend:.2f}")
    report()


def report() -> None:
    rs = [json.loads(l) for l in OUT.read_text().splitlines() if l.strip()] if OUT.exists() else []
    ok = [r for r in rs if "error" not in r and r.get("sellshare") is not None]
    print(f"\n{'buyer':<10}{'seller':<10}{'n':>4}{'mean':>8}{'sd':>7}{'focal':>8}{'min':>7}{'max':>7}")
    print("-" * 61)
    for b in FAMILIES:
        for s in FAMILIES:
            v = [r["sellshare"] for r in ok if r["buyer_family"] == b and r["seller_family"] == s]
            if v:
                f = sum(1 for x in v if abs(x - 0.5) < 1e-9) / len(v)
                print(f"{b:<10}{s:<10}{len(v):>4}{st.mean(v):>8.3f}{st.pstdev(v):>7.3f}"
                      f"{f:>8.0%}{min(v):>7.3f}{max(v):>7.3f}")
    nd = sum(1 for r in rs if "error" not in r and r.get("sellshare") is None)
    print(f"\nno-deal {nd}/{len(rs)}   errors {sum(1 for r in rs if 'error' in r)}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--draws", type=int, default=3)
    p.add_argument("--report", action="store_true")
    a = p.parse_args()
    if a.report:
        report()
        return
    asyncio.run(main_async(a.draws))


if __name__ == "__main__":
    main()
