"""Top up the pool until every history cell has enough transcripts.

A history is ten episodes, one per history good (see goods.py). So the pool needs, for
EVERY one of the ten goods, transcripts in five cells:

    S/even          past-you was the seller, deal landed near even
    S/favourable    past-you was the seller and came out ahead
    S/unfavourable  past-you was the seller and got beaten
    B/even          past-you was the buyer, deal landed near even
    B/favourable    past-you was the buyer and cleaned up

⚠️ B/unfavourable was written off on the strength of ONE seller-side brief (a word-for-word
mirror of the buyer version, 6/20) against seven attempts on the buyer side. Two
purpose-built briefs reach it 90% of the time, leaving the buyer at 0.277 -- right alongside
the 0.184-0.286 a beaten seller gets. The bank supports all six cells symmetrically.

⚠️ S/favourable was believed unreachable until 2026-08-12 and it nearly derailed the
design. That was an artifact of measuring against a 0.70 threshold rather than the actual
bin boundary of 0.60, using a brief (slow_close) that also produces 2-round non-
negotiations. Measured properly with never_holds, every one of the ten goods clears 0.60,
at rates from 12% (motorcycle) to 88% (kitchen, tractor).

## Why this is iterative rather than a fixed batch

Bins are REALISED, not chosen. A brief is a generation device: briefing the buyer as a
shark usually leaves past-you-as-seller beaten, but not always, and nothing is discarded
when it doesn't -- a fair brief that happens to land lopsided is simply filed as lopsided.
That is the binning principle from the original pool design, and it means you cannot order
"six S/unfavourable motorcycles" directly. You can only run briefs and see where they land.

So: measure, run the briefs that best fill what's missing, re-measure, repeat. Every
transcript counts toward whichever cell it actually lands in, so nothing is wasted.

## ⭐ ROUTE DIVERSITY IS A TARGET, NOT A BYPRODUCT

The naive version of this file mapped each cell to its single most likely brief. That is
wrong, and the existing pool shows why: S/even is more PUSHOVER than fair (33 vs 23), and
B/even is mostly SHARK (33 of 55). Filling a cell from one family would make every "even"
episode in a history an episode where the counterparty was explicitly fair-minded --
reintroducing caricature through the back door, which is the exact failure binning-on-
outcome was adopted to prevent.

It also matters substantively. A shark seller against an unbriefed buyer lands even 92% of
the time: past-you HELD ITS GROUND against an aggressive counterparty. A fair seller lands
even 62%: both sides were reasonable throughout. Same bin, different experience. If a
category is an assumption about the environment, those two should install different
assumptions despite identical outcomes -- and we can only ask that question if both are in
the pool.

So the allocator below targets a minimum count per (cell, FAMILY) pair, not per cell.

⚠️ ONE CELL CANNOT HAVE ROUTE DIVERSITY. S/unfavourable is 34/36 shark: nothing but an
aggressive buyer beats the seller. Structural, not a sampling artifact. State it as a
limitation rather than trying to engineer around it.

Usage:
    .venv/bin/python -m experiments.pool.topup --report
    .venv/bin/python -m experiments.pool.topup --target 6 --dry-run
    .venv/bin/python -m experiments.pool.topup --target 6
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
from collections import Counter
from pathlib import Path
from typing import Dict, List

from anthropic import AsyncAnthropic

from experiments.pool.generate import MODEL, generate_one, load_dotenv
from experiments.pool.goods import BY_KEY, HISTORY_GOODS

DATA_DIR = Path(__file__).parent / "data"
CONCURRENCY = 12
MAX_ROUNDS = 4
STATE_RANGE = True               # overridden by --no-range

CELLS = [("S", "even"), ("S", "favourable"), ("S", "unfavourable"),
         ("B", "even"), ("B", "favourable"), ("B", "unfavourable")]
FAMILIES = ["fair", "shark", "pushover",           # role-neutral, usable on either side
            "never_holds", "explicit_open", "conceder", "invite",   # buyer-phrased only
            "firm_price", "patient_seller"]                         # seller-phrased only

# Plain-English names for the four cells. "S" and "B" say which chair PAST-YOU sat in --
# the unbriefed side, the one the agent is later told it was -- so an S transcript can only
# ever appear in a seller's history and a B transcript only in a buyer's. The bin says how
# it ended FOR PAST-YOU.
CELL_NAME = {
    ("S", "even"):         "sold, landed even",
    ("S", "favourable"):   "sold, came out ahead",
    ("S", "unfavourable"): "sold, got beaten",
    ("B", "even"):         "bought, landed even",
    ("B", "favourable"):   "bought, got a good price",
    ("B", "unfavourable"): "bought, got beaten",
}

# Which role must be briefed to make past-you occupy a given chair.
BRIEFED_FOR = {"S": "buyer", "B": "seller"}

# Measured crossover, 2026-08-11, n=187 over the first five goods: given the counterparty's
# brief, where does past-you's outcome land? Used only as a PLANNING PRIOR -- the loop
# re-measures the real pool every round, so a bad prior costs an extra round, not accuracy.
YIELD = {
    # The four briefs below were built on 2026-08-12 specifically to reach S/favourable,
    # which the three core families essentially never produce (<=4%). They are phrased for a
    # BUYER and appear only here -- a brief must name a move the role can actually make, or
    # the model ignores it (hard_shark handed to a seller moved nothing at all).
    "S": {"fair":          {"even": .96, "favourable": .04, "unfavourable": .00},
          "shark":         {"even": .03, "favourable": .00, "unfavourable": .97},
          "pushover":      {"even": .92, "favourable": .03, "unfavourable": .06},
          "never_holds":   {"even": .39, "favourable": .61, "unfavourable": .00},
          "explicit_open": {"even": .53, "favourable": .47, "unfavourable": .00},
          "conceder":      {"even": .58, "favourable": .42, "unfavourable": .00},
          "invite":        {"even": .67, "favourable": .33, "unfavourable": .00}},
    # firm_price and patient_seller are SELLER-phrased and exist to reach B/unfavourable,
    # which the mirrored buyer brief only managed 6 times in 20. Both work by IMMOVABILITY
    # rather than aggression -- one price stated up front, or simply refusing to move --
    # which is the same lesson as never_holds on the other side: the lever is a rule about
    # the exchange, not a disposition. Anchoring is already spent.
    "B": {"fair":           {"even": .62, "favourable": .38, "unfavourable": .00},
          "shark":          {"even": .92, "favourable": .03, "unfavourable": .06},
          "pushover":       {"even": .26, "favourable": .74, "unfavourable": .00},
          "firm_price":     {"even": .10, "favourable": .00, "unfavourable": .90},
          "patient_seller": {"even": .10, "favourable": .00, "unfavourable": .90}},
}

# ⭐ BLIND yields are completely different, so the blind bank needs its own priors.
# Measured 2026-08-12, n=80, five items. With no posted range the whole distribution slides
# toward the seller — even a FAIR buyer leaves it at 0.628 — so the cell that is abundant
# with a range posted (even, 63% of the bank) becomes the scarce one (20%).
#
#   with the range posted    63% even · 25% favourable · 12% unfavourable
#   blind                    20% even · 57% favourable · 22% unfavourable
#
# Consequence: filling the even cells costs roughly 3x what it does in the range-posted
# bank, and slow_close is useless for it (0% even). Only `fair` and `hard_shark` reach it.
YIELD_BLIND = {
    "S": {"fair":       {"even": .35, "favourable": .65, "unfavourable": .00},
          "hard_shark": {"even": .30, "favourable": .13, "unfavourable": .57},
          "slow_close": {"even": .00, "favourable": .97, "unfavourable": .03}},
    # the seller-side briefs have not been measured blind; these are the range-posted
    # priors and the loop will correct them after one round
    "B": {"fair":           {"even": .62, "favourable": .38, "unfavourable": .00},
          "pushover":       {"even": .26, "favourable": .74, "unfavourable": .00},
          "firm_price":     {"even": .10, "favourable": .00, "unfavourable": .90},
          "patient_seller": {"even": .10, "favourable": .00, "unfavourable": .90}},
}
FAMILIES_BLIND = ["fair", "hard_shark", "slow_close",
                  "pushover", "firm_price", "patient_seller"]

# A family counts as a genuine ROUTE into a cell if it reaches it often enough to be worth
# generating deliberately. Below this it is crossover noise, not a route.
ROUTE_MIN_YIELD = 0.20
# ...and we want at least this many transcripts per (cell, family) route, per good.
ROUTE_MIN = 2


def _yields() -> Dict:
    return YIELD if STATE_RANGE else YIELD_BLIND


def _families() -> List[str]:
    return FAMILIES if STATE_RANGE else FAMILIES_BLIND


def _y(sp: str, f: str, bin_: str) -> float:
    """Yield of brief f into cell (sp, bin_). 0 if that brief is not usable on that side."""
    return _yields()[sp].get(f, {}).get(bin_, 0.0)


def routes(sp: str, bin_: str) -> List[str]:
    """Families that genuinely reach this cell."""
    return [f for f in _families() if _y(sp, f, bin_) >= ROUTE_MIN_YIELD]


def _load_all() -> List[Dict]:
    from experiments.arena.history import load_pool
    return load_pool(state_range=STATE_RANGE)


def _plan_good(g, have: Counter, have_rt: Counter, target: int) -> List[tuple]:
    """Greedily choose runs for one good, accounting for JOINT PRODUCTION.

    A run is not aimed at a cell -- it produces a distribution over cells. A pushover
    seller reaches B/even only 26% of the time, so ordering "enough pushover runs to fill
    B/even" would be absurdly wasteful. But the other 74% land in B/favourable, which is
    also needed. So the right question at each step is: which single run reduces total
    unmet need the most, counting every cell it could land in?

    Needs are of two kinds and both must be met:
      cell need  -- (sp, bin) below `target`
      route need -- (sp, bin, family) below ROUTE_MIN, for genuine routes only
    """
    out: List[tuple] = []
    unmet_cell = {(sp, b): max(0.0, target - have.get((sp, b, g.key), 0))
                  for sp, b in CELLS}
    unmet_rt = {(sp, b, f): max(0.0, ROUTE_MIN - have_rt.get((sp, b, g.key, f), 0))
                for sp, b in CELLS for f in routes(sp, b)}

    for _ in range(400):                      # backstop; real plans are far shorter
        if not any(unmet_cell.values()) and not any(unmet_rt.values()):
            break
        best, best_score = None, 0.0
        for sp in ("S", "B"):
            for f in _families():
                score = 0.0
                for _sp, b in CELLS:
                    if _sp != sp:
                        continue
                    y = _y(sp, f, b)
                    score += min(y, unmet_cell[(sp, b)])
                    if (sp, b, f) in unmet_rt:
                        # a route shortfall is worth extra: nothing else can fill it
                        score += 1.5 * min(y, unmet_rt[(sp, b, f)])
                if score > best_score:
                    best, best_score = (sp, f), score
        if best is None:
            break
        sp, f = best
        out.append((g, BRIEFED_FOR[sp], f))
        for _sp, b in CELLS:
            if _sp != sp:
                continue
            y = _y(sp, f, b)
            unmet_cell[(sp, b)] = max(0.0, unmet_cell[(sp, b)] - y)
            if (sp, b, f) in unmet_rt:
                unmet_rt[(sp, b, f)] = max(0.0, unmet_rt[(sp, b, f)] - y)
    return out


def table(target: int, verbose: bool = True, only: List[str] = None,
          limit: int = 0) -> List[tuple]:
    """Print coverage and return the (good, briefed_role, family) runs still needed.

    Two requirements, both per good:
      1. each cell holds at least `target` transcripts
      2. each genuine ROUTE into a cell holds at least ROUTE_MIN

    Requirement 2 is what keeps "even" from meaning one thing. Requirement 1 is what gives
    the draw enough distinct transcripts that two sellers in an arena do not end up with
    near-identical histories.
    """
    pool = _load_all()
    have = Counter((r["sub_pool"], r["bin"], r["good"]) for r in pool)
    have_rt = Counter((r["sub_pool"], r["bin"], r["good"], r["family"]) for r in pool)

    if verbose:
        print(f"\n{'good':<12}" + "".join(f"{sp}/{b[:6]:<11}" for sp, b in CELLS))
        print(f"{'':<12}" + "".join(f"{'  '.join(f[:2] for f in routes(sp,b)):<10}"
                                    for sp, b in CELLS))
        print("-" * (12 + 10 * len(CELLS)))

    jobs: List[tuple] = []
    goods = [g for g in HISTORY_GOODS if not only or g.key in only]
    for g in goods:
        row = ""
        for sp, b in CELLS:
            n = have.get((sp, b, g.key), 0)
            rts = routes(sp, b)
            per = [have_rt.get((sp, b, g.key, f), 0) for f in rts]
            ok = n >= target and all(p >= ROUTE_MIN for p in per)
            row += f"{n:<3}({'/'.join(str(p) for p in per)}){'' if ok else '!'}".ljust(11)

        if verbose:
            print(f"{g.key:<12}{row}")
        jobs += _plan_good(g, have, have_rt, target)

    if limit:
        jobs = jobs[:limit]
    if verbose:
        print("\n  WHAT THE COLUMNS MEAN — each is one kind of memory an agent can hold.")
        print("  S = past-you was the seller (usable only in a seller's history)")
        print("  B = past-you was the buyer  (usable only in a buyer's history)")
        for sp, b in CELLS:
            print(f"    {sp}/{b:<13} {CELL_NAME[(sp, b)]:<26} "
                  f"reached via: {', '.join(routes(sp, b))}")
        print(f"\n  Each entry reads  total(counts by counterparty type, in the order "
              f"listed above).")
        print(f"  '!' = short of the {target}-per-cell target or the {ROUTE_MIN}-per-route "
              f"minimum.")
        print("  S/unfavourable has ONE route: nothing but an aggressive buyer beats the")
        print("  seller. Structural, not a sampling artifact — report it as a limitation.")
        byc = Counter((r, f) for _, r, f in jobs)
        print(f"\n{len(jobs)} negotiations needed (~${len(jobs)*0.025:.2f} at $0.025 each)")
        for (r, f), n in sorted(byc.items()):
            print(f"    brief the {r:<7} as {f:<9} x{n}")
    return jobs


async def main_async(target: int, dry: bool, only: List[str], limit: int) -> None:
    key = load_dotenv()
    tag = "" if STATE_RANGE else "blind_"
    out_path = DATA_DIR / f"pool_{tag}topup_{time.strftime('%Y%m%d-%H%M%S')}.jsonl"
    client = AsyncAnthropic(max_retries=5)
    sem, lock = asyncio.Semaphore(CONCURRENCY), asyncio.Lock()
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Using API key {key}\nModel: {MODEL}")
    spend, idx = 0.0, 0

    for rnd in range(MAX_ROUNDS):
        print(f"\n{'='*70}\nROUND {rnd+1}\n{'='*70}")
        jobs = table(target, only=only, limit=limit)
        if not jobs:
            print("\nAll cells at target, all routes covered.")
            return
        if dry:
            print("\n--dry-run: nothing sent.")
            return

        t0, n_done = time.time(), 0

        async def guarded(i: int, good, briefed_role, family):
            nonlocal spend, n_done
            async with sem:
                try:
                    rec = await generate_one(client, i, good, family, briefed_role,
                                             STATE_RANGE)
                except Exception as exc:
                    rec = {"idx": i, "good": good.key, "family": family,
                           "briefed_role": briefed_role,
                           "error": f"{type(exc).__name__}: {str(exc)[:100]}"}
                async with lock:
                    with out_path.open("a") as fh:
                        fh.write(json.dumps(rec) + "\n")
                        fh.flush(); os.fsync(fh.fileno())
                    n_done += 1
                    if "usage" in rec:
                        spend += (rec["usage"]["input_tokens"] * 2
                                  + rec["usage"]["output_tokens"] * 10) / 1e6
                    if n_done % 15 == 0 or n_done == len(jobs):
                        print(f"  {n_done}/{len(jobs)}  {time.time()-t0:5.0f}s  "
                              f"~${spend:.2f}", flush=True)

        await asyncio.gather(*(guarded(idx + i, *j) for i, j in enumerate(jobs)))
        idx += len(jobs)

    print(f"\nStopped after {MAX_ROUNDS} rounds. ~${spend:.2f}")
    table(target)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--no-range", action="store_true",
                   help="build the BLIND bank — no posted market range. Partitioned from "
                        "the range-posted bank; the two must never mix.")
    p.add_argument("--target", type=int, default=6, help="transcripts per cell")
    p.add_argument("--report", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--only", type=str, default="",
                   help="comma-separated good keys, e.g. the five new ones")
    p.add_argument("--limit", type=int, default=0, help="cap runs (smoke test)")
    a = p.parse_args()
    global STATE_RANGE
    STATE_RANGE = not a.no_range
    only = [k.strip() for k in a.only.split(",") if k.strip()]
    if a.report:
        table(a.target, only=only)
    else:
        asyncio.run(main_async(a.target, a.dry_run, only, a.limit))


if __name__ == "__main__":
    main()
