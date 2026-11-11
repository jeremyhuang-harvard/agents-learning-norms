"""How far can a ONE-SIDED brief push the outcome?

The question this settles: to reach the extreme tail of seller experience, do we have to
brief BOTH sides -- giving up the blank chair that makes attribution clean -- or is there
room left in the counterparty brief alone?

    standard shark buyer vs unbriefed seller   mean 0.287, worst 0.150   (measured, n=36)
    shark buyer vs pushover seller             mean 0.184, worst 0.025   (both briefed)

If a harder one-sided brief lands near 0.15 or below, the remaining gap to the two-sided
floor is small and probably not worth the identification cost. If it barely moves, the case
for double-briefing is made empirically rather than by argument.

Run on the FIVE ORIGINAL goods only. The five goods added on 2026-08-12 have never been
negotiated, so using them here would confound "does the harder brief work" with "is this
new good strange." Same goods, same template, same model as the standard-shark transcripts
we are comparing against -- the brief is the only thing that changes.

Output goes to data/hardshark_*.jsonl, deliberately NOT matching the pool glob: these are
valid transcripts but a more extreme environment than the rest of the library, and folding
them in silently would add uncontrolled variation to histories. Promote them later if we
decide to.

Usage:
    .venv/bin/python -m experiments.pool.hardshark --draws 4
    .venv/bin/python -m experiments.pool.hardshark --report
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

from experiments.pool.briefs import BRIEFS
from experiments.pool.generate import MODEL, generate_one, load_dotenv
from experiments.pool.goods import BY_KEY

DATA_DIR = Path(__file__).parent / "data"
CONCURRENCY = 10
FAMILY = "hard_shark"   # overridden by --family
ROLE = "buyer"         # which side carries the brief; past-you is the other one
BASELINE_GOODS = ["motorcycle", "espresso", "pickup", "kitchen", "sailboat"]
GOODS_UNDER_TEST = BASELINE_GOODS   # overridden by --goods
STATE_RANGE = True                  # overridden by --no-range


async def main_async(draws: int, out: Path) -> None:
    key = load_dotenv()
    client = AsyncAnthropic(max_retries=5)
    sem, lock = asyncio.Semaphore(CONCURRENCY), asyncio.Lock()
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    done = set()
    if out.exists():
        for l in out.read_text().splitlines():
            if l.strip():
                try:
                    done.add(json.loads(l)["idx"])
                except Exception:
                    pass

    combos = [(BY_KEY[g], FAMILY, ROLE)
              for g in GOODS_UNDER_TEST for _ in range(draws)]
    todo = [(i, c) for i, c in enumerate(combos) if i not in done]

    print(f"Using API key {key}\nModel: {MODEL}\nOutput: {out}")
    print(f"\nBrief on the BUYER (seller's chair stays blank):\n  {BRIEFS[FAMILY]}\n")
    print(f"{len(todo)} negotiations across {len(GOODS_UNDER_TEST)} goods "
          f"(~${len(todo)*0.025:.2f})")

    n, spend, t0 = 0, 0.0, time.time()

    async def guarded(i, combo):
        nonlocal n, spend
        good, fam, role = combo
        async with sem:
            try:
                rec = await generate_one(client, i, good, fam, role, STATE_RANGE)
            except Exception as e:
                rec = {"idx": i, "good": good.key, "family": fam, "briefed_role": role,
                       "error": f"{type(e).__name__}: {str(e)[:120]}"}
            async with lock:
                with out.open("a") as fh:
                    fh.write(json.dumps(rec) + "\n"); fh.flush(); os.fsync(fh.fileno())
            n += 1
            if "usage" in rec:
                spend += (rec["usage"]["input_tokens"] * 2
                          + rec["usage"]["output_tokens"] * 10) / 1e6
            s = rec.get("past_you_share")
            print(f"  [{n}/{len(todo)}] {time.time()-t0:5.0f}s ~${spend:5.2f}  "
                  f"{rec['good']:<11} " +
                  (rec.get("error", "")[:40] if "error" in rec
                   else f"{rec['bin']:<13} seller kept "
                        f"{'no deal' if s is None else f'{s:.3f}'}"), flush=True)

    await asyncio.gather(*(guarded(i, c) for i, c in todo))
    print(f"\nDone. ~${spend:.2f}")
    report(out)


def report(out: Path) -> None:
    from experiments.arena.history import load_pool
    recs = [json.loads(l) for l in out.read_text().splitlines() if l.strip()]
    ok = [r for r in recs if "error" not in r and r.get("past_you_share") is not None]
    nd = sum(1 for r in recs if "error" not in r and r.get("past_you_share") is None)

    basefam = "shark" if "shark" in FAMILY else "pushover"
    sp = "S" if ROLE == "buyer" else "B"
    base = [r["past_you_share"] for r in load_pool()
            if r["sub_pool"] == sp and r["family"] == basefam
            and r["good"] in BASELINE_GOODS and r["past_you_share"] is not None]
    new = [r["past_you_share"] for r in ok]

    print("\n" + "=" * 66)
    print("HOW MUCH OF THE SURPLUS THE (UNBRIEFED) SELLER KEPT")
    print("=" * 66)
    print(f"{'brief on the buyer':<22}{'n':>4}{'mean':>8}{'sd':>7}{'worst':>8}{'best':>8}")
    for name, v in ((f"standard {basefam}", base), (FAMILY, new)):
        if v:
            print(f"{name:<22}{len(v):>4}{st.mean(v):>8.3f}{st.pstdev(v):>7.3f}"
                  f"{min(v):>8.3f}{max(v):>8.3f}")
    if "shark" not in FAMILY:
        print(f"{'both briefed (grid)':<22}{18:>4}{0.628:>8.3f}{0.047:>7.3f}"
              f"{0.529:>8.3f}{0.725:>8.3f}   <- pushover buyer x shark seller")
        won = sum(1 for x in new if x > 0.60)
        print(f"\nBINS AS 'sold and came out ahead' (share > 0.60): {won}/{len(new)}")
        print(f"previous count of that cell in the ENTIRE pool: 2")
    else:
        lo = sum(1 for x in new if x < 0.40)
        print(f"\nBINS AS lopsided-against-past-you (share < 0.40): {lo}/{len(new)}")
        print(f"previous count of that cell in the ENTIRE pool: "
              f"{36 if sp=='S' else 2}")
    if base and new:
        print(f"\nshift from the harder brief: {st.mean(new)-st.mean(base):+.3f}")
    print(f"\nno-deals {nd}/{len(recs)}  "
          f"(a no-deal is unusable as a memory — it bins as no_deal)")
    if new:
        print("\nper good:")
        for g in BASELINE_GOODS:
            v = [r["past_you_share"] for r in ok if r["good"] == g]
            b = [r["past_you_share"] for r in load_pool()
                 if r["sub_pool"] == sp and r["family"] == basefam and r["good"] == g
                 and r["past_you_share"] is not None]
            if v:
                print(f"  {g:<12} hard {st.mean(v):.3f} (n={len(v)})   "
                      f"standard {st.mean(b):.3f} (n={len(b)})" if b else
                      f"  {g:<12} hard {st.mean(v):.3f} (n={len(v)})")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--draws", type=int, default=4, help="per good")
    p.add_argument("--no-range", action="store_true",
                   help="drop the common-knowledge value range")
    p.add_argument("--goods", default="", help="comma-separated good keys")
    p.add_argument("--role", default="buyer", choices=["buyer", "seller"],
                   help="which side is briefed; past-you is the other one")
    p.add_argument("--family", default="hard_shark",
                   choices=["fair", "shark", "pushover", "hard_shark", "hard_shark_sell", "hard_pushover", "conceder", "high_anchor", "invite", "never_holds", "invite_never_holds", "ratchet_accel", "ratchet_long", "ratchet_patient", "urgent_haggler", "strong_open_two_counters", "explicit_open", "slow_close", "firm_price", "patient_seller"])
    p.add_argument("--report", type=Path, nargs="?", const=None)
    a = p.parse_args()
    global FAMILY, ROLE, GOODS_UNDER_TEST, STATE_RANGE
    FAMILY, ROLE = a.family, a.role
    STATE_RANGE = not a.no_range
    if a.goods:
        GOODS_UNDER_TEST = [k.strip() for k in a.goods.split(",") if k.strip()]
    existing = sorted(DATA_DIR.glob(f"{FAMILY}_{ROLE}*.jsonl"))
    if a.report is not None or ("--report" in os.sys.argv and existing):
        report(a.report or existing[-1])
        return
    tag = "" if STATE_RANGE else "norange_"
    out = DATA_DIR / f"{tag}{FAMILY}_{ROLE}_{time.strftime('%Y%m%d-%H%M%S')}.jsonl"
    asyncio.run(main_async(a.draws, out))


if __name__ == "__main__":
    main()
