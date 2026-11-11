"""Rebuild every results table in `results/` from the saved run data.

No API key, no network, standard library only:

    python analysis/make_results.py

Outcome measure throughout: `sellshare` = (price - seller's fallback) / (buyer's fallback -
seller's fallback), i.e. the seller's share of the bargaining surplus. 0.5 is an exact even
split. "Focal" = the deal landed exactly on 0.5.
"""

from __future__ import annotations

import csv
import json
import statistics as st
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARENA = ROOT / "experiments" / "arena" / "data"
FIRSTMOVER = ROOT / "experiments" / "firstmover" / "data"
POOL = ROOT / "experiments" / "pool" / "data"
OUT = ROOT / "results"

# Benchmarks from Imas, Lee & Misra (2025), "Agentic Interactions". Cited, not computed.
IMAS_HUMAN_FOCAL = 0.347
IMAS_AGENTIC_FOCAL = 0.143

ARENA_ORDER = ["10/0", "10/0f", "5/5", "0/10", "0/10w"]
ARENA_LABEL = {
    "10/0": "ten even deals (near 50-50)",
    "10/0f": "ten deals at exactly 50-50",
    "5/5": "five even, five losses",
    "0/10": "ten losses",
    "0/10w": "ten wins",
}


def load(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def shares(records: list[dict]) -> list[float]:
    return [r["sellshare"] for r in records if r.get("sellshare") is not None]


def summarize(values: list[float]) -> dict:
    n = len(values)
    return {
        "n": n,
        "mean_seller_share": round(st.mean(values), 3),
        "sd": round(st.stdev(values), 3) if n > 1 else "",
        "focal_share": round(sum(abs(v - 0.5) < 1e-9 for v in values) / n, 3),
        "below_0.40": round(sum(v < 0.40 for v in values) / n, 3),
    }


def by_arena(records: list[dict]) -> dict[str, list[float]]:
    out: dict[str, list[float]] = defaultdict(list)
    for r in records:
        if r.get("sellshare") is not None:
            out[r["arena"]].append(r["sellshare"])
    return out


def baseline_records() -> list[dict]:
    """No-memory, no-instruction Sonnet 5 (thinking on), buyer opens -- the arena's move order."""
    return [r for r in load(FIRSTMOVER / "cell_sonnet-think.jsonl") if r.get("opener") == "buyer"]


def write_csv(name: str, rows: list[dict]) -> None:
    OUT.mkdir(exist_ok=True)
    with (OUT / name).open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def table_memory() -> list[dict]:
    rows = [{"condition": "no memory (baseline)", "arena": "", **summarize(shares(baseline_records()))}]
    arms = by_arena(load(ARENA / "armA_v2.jsonl"))
    rows += [{"condition": ARENA_LABEL[a], "arena": a, **summarize(arms[a])} for a in ARENA_ORDER]
    return rows


def table_arms() -> list[dict]:
    a = by_arena(load(ARENA / "armA_v2.jsonl"))
    b = by_arena(load(ARENA / "armB_full.jsonl"))
    rows = []
    for arena in ARENA_ORDER:
        sa, sb = summarize(a[arena]), summarize(b[arena])
        rows.append({"arena": arena, "condition": ARENA_LABEL[arena],
                     "A_n": sa["n"], "A_mean": sa["mean_seller_share"], "A_focal": sa["focal_share"],
                     "B_n": sb["n"], "B_mean": sb["mean_seller_share"], "B_focal": sb["focal_share"]})
    for label, keep in (("pooled: original four arenas", ["10/0", "5/5", "0/10", "0/10w"]),
                        ("pooled: all five arenas", ARENA_ORDER)):
        sa = summarize([x for k in keep for x in a[k]])
        sb = summarize([x for k in keep for x in b[k]])
        rows.append({"arena": "", "condition": label,
                     "A_n": sa["n"], "A_mean": sa["mean_seller_share"], "A_focal": sa["focal_share"],
                     "B_n": sb["n"], "B_mean": sb["mean_seller_share"], "B_focal": sb["focal_share"]})
    return rows


def table_range() -> list[dict]:
    cells = [
        ("range posted in memory and in the arena", "armA_v2.jsonl"),
        ("range posted in memory, removed from the arena", "armA_norange_pilot.jsonl"),
        ("range removed from memory and from the arena", "armA_blind.jsonl"),
    ]
    rows = []
    for label, fname in cells:
        arms = by_arena(load(ARENA / fname))
        for arena in ARENA_ORDER:
            if arena in arms:
                rows.append({"cell": label, "arena": arena, "condition": ARENA_LABEL[arena],
                             **summarize(arms[arena])})
    return rows


def table_interior() -> list[dict]:
    grid: dict[tuple, list[float]] = defaultdict(list)
    for r in load(POOL / "interior.jsonl"):
        if r.get("sellshare") is not None:
            grid[(r["buyer_family"], r["seller_family"])].append(r["sellshare"])
    order = ["fair", "shark", "pushover"]
    return [{"buyer_brief": b, "seller_brief": s, **summarize(grid[(b, s)])}
            for b in order for s in order]


def table_models() -> list[dict]:
    rows = []
    for cell in ["sonnet-think", "sonnet-nothink", "haiku-think", "haiku-nothink"]:
        recs = load(FIRSTMOVER / f"cell_{cell}.jsonl")
        for opener in ["buyer", "seller"]:
            rows.append({"model_cell": cell, "opener": opener,
                         **summarize(shares([r for r in recs if r.get("opener") == opener]))})
    return rows


def table_bank() -> list[dict]:
    counts: dict[tuple, int] = defaultdict(int)
    for path in sorted(POOL.glob("pool_*.jsonl")):
        for r in load(path):
            if "error" in r or not r.get("transcript") or r.get("past_you_role") is None:
                continue
            if r.get("bin") in (None, "no_deal"):
                continue
            key = ("range posted" if r.get("state_range", True) else "range hidden",
                   r["past_you_role"], r["bin"])
            counts[key] += 1
    return [{"bank": k[0], "past_you_role": k[1], "outcome_bin": k[2], "transcripts": v}
            for k, v in sorted(counts.items())]


def svg_focal(rows: list[dict]) -> str:
    bars = [(r["condition"], r["focal_share"]) for r in rows]
    bars += [("Imas et al. humans (cited)", IMAS_HUMAN_FOCAL),
             ("Imas et al. AI delegates (cited)", IMAS_AGENTIC_FOCAL)]
    w, left, bar_h, gap, top = 760, 250, 22, 10, 50
    h = top + len(bars) * (bar_h + gap) + 40
    scale = (w - left - 60) / 0.40
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" '
             f'font-family="Helvetica, Arial, sans-serif" font-size="13">',
             f'<rect width="{w}" height="{h}" fill="white"/>',
             '<text x="20" y="28" font-size="16" font-weight="bold">'
             'Share of deals landing exactly on a 50-50 split, by what the seller remembers</text>']
    for i, (label, value) in enumerate(bars):
        y = top + i * (bar_h + gap)
        cited = "cited" in label
        color = "#9aa5b1" if cited else ("#c0392b" if i == 0 else "#2e86c1")
        parts.append(f'<text x="{left - 10}" y="{y + 16}" text-anchor="end">{label}</text>')
        parts.append(f'<rect x="{left}" y="{y}" width="{value * scale:.1f}" height="{bar_h}" fill="{color}"/>')
        parts.append(f'<text x="{left + value * scale + 6:.1f}" y="{y + 16}">{value * 100:.0f}%</text>')
    parts.append(f'<text x="20" y="{h - 12}" font-size="11" fill="#555">Sonnet 5 sellers, no '
                 'instructions in any prompt; only the remembered negotiations differ. '
                 'Grey bars are published benchmarks, not our data.</text>')
    parts.append("</svg>")
    return "\n".join(parts) + "\n"


def fmt(x) -> str:
    return f"{x * 100:.0f}%" if isinstance(x, float) else str(x)


def markdown(memory, arms, rng, interior, models, bank) -> str:
    lines = ["# Results", "",
             "Generated by `python analysis/make_results.py` from the saved runs in "
             "`experiments/*/data/`. Do not edit by hand.", "",
             "`seller share` = fraction of the surplus the seller captured (0.5 = even split). "
             "`focal` = deal landed exactly on 0.5.", "",
             "## 1. What the seller remembers vs. how the deal ends", "",
             "| seller remembers | n | mean seller share | focal | deals below 0.40 |",
             "|---|---:|---:|---:|---:|"]
    for r in memory:
        lines.append(f"| {r['condition']} | {r['n']} | {r['mean_seller_share']:.3f} | "
                     f"{fmt(r['focal_share'])} | {fmt(r['below_0.40'])} |")
    lines += [f"| *Imas et al. humans (cited)* | | | *{fmt(IMAS_HUMAN_FOCAL)}* | |",
              f"| *Imas et al. AI delegates (cited)* | | | *{fmt(IMAS_AGENTIC_FOCAL)}* | |", "",
              "![focal by memory](focal_by_memory.svg)", "",
              "## 2. Carrying the history (Arm A) vs. a playbook written from it (Arm B)", "",
              "| arena | A n | A mean | A focal | B n | B mean | B focal |",
              "|---|---:|---:|---:|---:|---:|---:|"]
    for r in arms:
        lines.append(f"| {r['condition']} | {r['A_n']} | {r['A_mean']:.3f} | {fmt(r['A_focal'])} | "
                     f"{r['B_n']} | {r['B_mean']:.3f} | {fmt(r['B_focal'])} |")
    lines += ["", "## 3. Removing the posted price range", "",
              "| cell | seller remembers | n | mean | focal |", "|---|---|---:|---:|---:|"]
    for r in rng:
        lines.append(f"| {r['cell']} | {r['condition']} | {r['n']} | "
                     f"{r['mean_seller_share']:.3f} | {fmt(r['focal_share'])} |")
    lines += ["", "## 4. Both sides briefed (pool interior grid)", "",
              "| buyer brief | seller brief | n | mean | sd | focal |", "|---|---|---:|---:|---:|---:|"]
    for r in interior:
        lines.append(f"| {r['buyer_brief']} | {r['seller_brief']} | {r['n']} | "
                     f"{r['mean_seller_share']:.3f} | {r['sd']} | {fmt(r['focal_share'])} |")
    lines += ["", "## 5. No-memory baselines by model and reasoning", "",
              "| cell | opener | n | mean | focal | below 0.40 |", "|---|---|---:|---:|---:|---:|"]
    for r in models:
        lines.append(f"| {r['model_cell']} | {r['opener']} | {r['n']} | {r['mean_seller_share']:.3f} | "
                     f"{fmt(r['focal_share'])} | {fmt(r['below_0.40'])} |")
    lines += ["", "## 6. The transcript bank histories are drawn from", "",
              "| bank | past-you role | outcome | transcripts |", "|---|---|---|---:|"]
    for r in bank:
        lines.append(f"| {r['bank']} | {r['past_you_role']} | {r['outcome_bin']} | {r['transcripts']} |")
    return "\n".join(lines) + "\n"


def main() -> None:
    memory, arms, rng = table_memory(), table_arms(), table_range()
    interior, models, bank = table_interior(), table_models(), table_bank()
    write_csv("memory_vs_outcome.csv", memory)
    write_csv("arm_a_vs_arm_b.csv", arms)
    write_csv("range_removed.csv", rng)
    write_csv("interior_grid.csv", interior)
    write_csv("baseline_models.csv", models)
    write_csv("transcript_bank.csv", bank)
    (OUT / "focal_by_memory.svg").write_text(svg_focal(memory))
    (OUT / "RESULTS.md").write_text(markdown(memory, arms, rng, interior, models, bank))
    print(f"wrote results to {OUT.relative_to(ROOT)}/")


if __name__ == "__main__":
    main()
