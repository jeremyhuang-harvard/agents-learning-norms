"""Run the first-mover design across the model x reasoning 2x2.

Everything is held fixed except which model negotiates and whether it reasons:
same template, same six goods, both sides unbriefed, opener varied within each cell.
The transcript parser stays on Sonnet 5 throughout so measurement does not move.

Usage:
    .venv/bin/python -m experiments.firstmover.run2x2 --cells haiku-nothink --n 4
    .venv/bin/python -m experiments.firstmover.run2x2 --n 4          # all four
    .venv/bin/python -m experiments.firstmover.run2x2 --report
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

from experiments.firstmover.models import CONFIGS, PRICES, call_kwargs
from experiments.firstmover.run import KICKOFF, build
from experiments.pool.generate import ACCEPT_KEYWORDS, MAX_ROUNDS, _check, _parse, load_dotenv
from experiments.pool.goods import GOODS, Good

DATA_DIR = Path(__file__).parent / "data"
CONCURRENCY = 10


async def one(client: AsyncAnthropic, idx: int, good: Good, opener: str, cell: str) -> Dict:
    cfg = CONFIGS[cell]
    kw = call_kwargs(cfg)
    other = "seller" if opener == "buyer" else "buyer"
    sysmsg = {r: build(good, r, opener) for r in ("buyer", "seller")}
    msgs: Dict[str, List[Dict]] = {opener: [{"role": "user", "content": KICKOFF}], other: []}

    transcript, usage, closed_at = [], {"input_tokens": 0, "output_tokens": 0}, None
    for turn in range(MAX_ROUNDS):
        speaker = opener if turn % 2 == 0 else other
        listener = other if speaker == opener else opener
        r = await client.messages.create(system=sysmsg[speaker], messages=msgs[speaker], **kw)
        usage["input_tokens"] += r.usage.input_tokens
        usage["output_tokens"] += r.usage.output_tokens
        text = "".join(b.text for b in r.content if b.type == "text").strip()
        msgs[speaker].append({"role": "assistant", "content": text})
        msgs[listener].append({"role": "user", "content": text})
        transcript.append({"round": turn + 1, "role": speaker, "text": text})

        if turn >= 1 and any(k in text.lower() for k in ACCEPT_KEYWORDS):
            chk = await _check(client, transcript)          # parser: always Sonnet 5
            if chk.agreement_reached and chk.transaction_price is not None:
                closed_at = turn + 1
                break

    out = await _parse(client, transcript)                   # parser: always Sonnet 5
    price = out.transaction_price if out.agreement_reached else None
    return {
        "idx": idx, "cell": cell, "model": cfg["model"],
        "thinking": cfg["thinking"], "good": good.key, "range": [good.low, good.high],
        "opener": opener, "transaction_price": price,
        "sellshare": (price - good.low) / good.surplus if price is not None else None,
        "rounds_used": len(transcript), "closed_early_at": closed_at,
        "agreement_round": out.agreement_round, "parser_evidence": out.evidence,
        "usage": usage, "transcript": transcript,
    }


async def run_cell(cell: str, n: int, out_path: Path) -> None:
    client = AsyncAnthropic(max_retries=5)
    sem, lock = asyncio.Semaphore(CONCURRENCY), asyncio.Lock()
    combos = [(g, op) for g in GOODS for op in ("buyer", "seller") for _ in range(n)]
    cfg = CONFIGS[cell]
    print(f"\n▸ {cfg['label']}  ({len(combos)} negotiations)")
    done, spend, t0 = 0, 0.0, time.time()
    pin, pout = PRICES[cfg["model"]]

    async def guarded(i, combo):
        nonlocal done, spend
        g, op = combo
        async with sem:
            try:
                rec = await one(client, i, g, op, cell)
            except Exception as e:
                rec = {"idx": i, "cell": cell, "good": g.key, "opener": op,
                       "error": f"{type(e).__name__}: {str(e)[:120]}"}
            async with lock:
                with out_path.open("a") as fh:
                    fh.write(json.dumps(rec) + "\n"); fh.flush(); os.fsync(fh.fileno())
            done += 1
            if "usage" in rec:
                spend += (rec["usage"]["input_tokens"] * pin
                          + rec["usage"]["output_tokens"] * pout) / 1e6
            if done % 12 == 0 or done == len(combos):
                print(f"    {done}/{len(combos)}  {time.time()-t0:4.0f}s  ~${spend:.2f}", flush=True)

    await asyncio.gather(*(guarded(i, c) for i, c in enumerate(combos)))


def report(paths: List[Path]) -> None:
    recs = []
    for p in paths:
        if p.exists():
            recs += [json.loads(l) for l in p.read_text().splitlines() if l.strip()]

    print("\n" + "=" * 76)
    print("MODEL x REASONING — does weaker capability produce Imas's extreme outcomes?")
    print("=" * 76)
    print(f"\n{'cell':<38}{'n':>4}{'mean':>8}{'min':>8}{'=0':>6}{'<0.1':>7}{'nodeal':>8}")
    print("-" * 76)
    for cell, cfg in CONFIGS.items():
        rs = [r for r in recs if r.get("cell") == cell and "error" not in r]
        if not rs:
            continue
        deals = [r for r in rs if r.get("sellshare") is not None]
        v = [r["sellshare"] for r in deals]
        nz = sum(1 for x in v if x <= 0.001)
        lo = sum(1 for x in v if x < 0.10)
        nd = len(rs) - len(deals)
        print(f"{cfg['label']:<38}{len(rs):>4}{st.mean(v):>8.3f}{min(v):>8.3f}"
              f"{nz:>6}{lo:>7}{nd:>8}")
    print("-" * 76)
    print(f"{'Imas et al. (gpt-4.1-mini, ~89k negs)':<38}{'':>4}{0.350:>8.3f}{'':>8}"
          f"{'12.5%':>6}{'':>7}{'3.0%':>8}")

    print(f"\n{'cell':<38}{'buyer opens':>13}{'seller opens':>14}{'swing':>9}")
    print("-" * 76)
    for cell, cfg in CONFIGS.items():
        rs = [r for r in recs if r.get("cell") == cell and "error" not in r
              and r.get("sellshare") is not None]
        b = [r["sellshare"] for r in rs if r["opener"] == "buyer"]
        s = [r["sellshare"] for r in rs if r["opener"] == "seller"]
        if b and s:
            print(f"{cfg['label']:<38}{st.mean(b):>13.3f}{st.mean(s):>14.3f}"
                  f"{st.mean(s)-st.mean(b):>+9.3f}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cells", nargs="*", default=list(CONFIGS))
    p.add_argument("--n", type=int, default=4)
    p.add_argument("--report", action="store_true")
    a = p.parse_args()

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    paths = [DATA_DIR / f"cell_{c}.jsonl" for c in CONFIGS]
    if a.report:
        report(paths)
        return

    print(f"Using API key {load_dotenv()}")
    for cell in a.cells:
        asyncio.run(run_cell(cell, a.n, DATA_DIR / f"cell_{cell}.jsonl"))
    report(paths)


if __name__ == "__main__":
    main()
