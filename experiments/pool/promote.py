"""Fold the brief-development transcripts into the bank.

Building the escalation briefs on 2026-08-12 produced ~590 negotiations that were written
under their own filenames and so fell outside the pool glob. Most of them belong in the
bank: same template, same items, same model, exactly one side briefed, real negotiations.

Keeping them out would break the binning principle the pool was built on -- *briefs create
variation, outcomes define conditions* -- which exists precisely so that we never select on
which brief produced a transcript. A run made while tuning a prompt is not a different kind
of negotiation from a run made to stock a cell.

Four exclusions, each for a reason that is about the TRANSCRIPT, not about its provenance:

  no-range setting        the value range was withheld; a different game, kept separate
  hard_pushover           closes in 2-5 rounds -- the buyer opens near the top and accepts.
                          A memory of being handed money, not of winning a negotiation.
  buyer-phrased brief
    handed to a seller    the brief named moves the role cannot make, so the seller ignored
                          it entirely (0.495 vs 0.495 for the standard brief). The
                          transcripts are fine but the condition is meaningless.
  stale item definition   generated before the pickup/espresso condition notes were
                          neutralised; the dialogue argues about facts the item no longer
                          states. (load_pool drops these anyway; excluded here too so the
                          promoted file is self-consistent.)

Usage:
    .venv/bin/python -m experiments.pool.promote --dry-run
    .venv/bin/python -m experiments.pool.promote
"""

from __future__ import annotations

import argparse
import glob
import json
from collections import Counter
from pathlib import Path

from experiments.pool.goods import BY_KEY, REVISED_GOODS

DATA_DIR = Path(__file__).parent / "data"
OUT = DATA_DIR / "pool_promoted_20260812.jsonl"
# Blind transcripts go to their own file. They were previously DROPPED as "no-range
# setting" — correct when there was one bank, wrong now that the two are partitioned and
# the blind bank needs stocking. Both filenames match the pool glob; load_pool separates
# them on the `state_range` field, so they can never end up in the same history.
OUT_BLIND = DATA_DIR / "pool_blind_promoted_20260812.jsonl"


def eligible(rec: dict, filename: str) -> str | None:
    """Returns a reason for exclusion, or None if the record belongs in the bank."""
    if filename.startswith("pool_"):
        return "already in the bank"
    if "error" in rec:
        return "api error"
    if rec.get("bin") in (None, "no_deal"):
        return "no deal"
    if rec.get("family") == "hard_pushover":
        return "hard_pushover: 2-5 round walkover, not a negotiation"
    if rec.get("family") == "hard_shark" and rec.get("briefed_role") == "seller":
        return "buyer-phrased brief handed to a seller"
    g = rec.get("good")
    if g in REVISED_GOODS and rec.get("condition_note") != BY_KEY[g].condition_note:
        return "stale item definition"
    return None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    kept, blind, dropped, seen = [], [], Counter(), 0
    for f in sorted(glob.glob(str(DATA_DIR / "*.jsonl"))):
        name = Path(f).name
        if name == OUT.name:
            continue
        for line in Path(f).read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            seen += 1
            why = eligible(r, name)
            if why:
                dropped[why] += 1
            else:
                r["promoted_from"] = name
                # norange_* records predate the state_range field being stamped
                if name.startswith("norange") or r.get("state_range") is False:
                    r["state_range"] = False
                    blind.append(r)
                else:
                    kept.append(r)

    print(f"scanned {seen} records")
    print(f"  promoting {len(kept)} to the range-posted bank")
    print(f"  promoting {len(blind)} to the BLIND bank")
    for k, v in dropped.most_common():
        print(f"  skipped {v:>5}  {k}")

    c = Counter((r["sub_pool"], r["bin"]) for r in kept)
    print("\npromoted, by cell:")
    for k, v in sorted(c.items(), key=lambda x: -x[1]):
        print(f"  {k[0]}/{k[1]:<14}{v:>5}")

    if a.dry_run:
        print("\n--dry-run: nothing written.")
        return
    for path, rows in ((OUT, kept), (OUT_BLIND, blind)):
        if rows:
            with path.open("w") as fh:
                for r in rows:
                    fh.write(json.dumps(r) + "\n")
            print(f"wrote {len(rows)} -> {path}")


if __name__ == "__main__":
    main()
