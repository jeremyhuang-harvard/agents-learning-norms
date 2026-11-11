"""Plots for Phase 1 output.

    ../../.venv/bin/python plot.py data/noise_floor_<stamp>.jsonl

Produces one PNG next to the input file. Everything here reads the JSONL, so it
can be re-run and re-styled without touching the API.
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # write files, don't try to open a window
import matplotlib.pyplot as plt
import pandas as pd

# Published comparison points (Imas et al. §3, §3.4).
IMAS_AGENTIC_MEAN = 0.35
IMAS_AGENTIC_MEDIAN = 0.375
IMAS_AGENTIC_P25 = 0.19
IMAS_AGENTIC_P75 = 0.50
IMAS_AGENTIC_FOCAL = 0.143
IMAS_HUMAN_FOCAL = 0.347

INK = "#1a1a1a"
OURS = "#2b6cb0"
IMAS = "#c05621"
MUTED = "#8a8a8a"


def load(path: Path) -> pd.DataFrame:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    df = pd.DataFrame([r for r in rows if "error" not in r])
    n_err = len(rows) - len(df)
    if n_err:
        print(f"note: {n_err} errored run(s) excluded")
    return df


def make_figure(df: pd.DataFrame, out_path: Path) -> None:
    deals = df[df["sellshare"].notna()]
    shares = deals["sellshare"]

    fig, axes = plt.subplots(2, 3, figsize=(17, 8))
    fig.suptitle(
        f"Phase 1 — noise floor: {len(df)} negotiations, one fixed prompt pair",
        fontsize=13,
        color=INK,
        y=0.98,
    )

    # (1) The headline: where did outcomes land, and how tightly.
    ax = axes[0][0]
    ax.hist(shares, bins=40, range=(0, 1), color=OURS, alpha=0.85)
    ax.axvline(0.5, color=MUTED, ls="--", lw=1, label="50-50 split")
    ax.axvline(shares.mean(), color=OURS, lw=1.5, label=f"our mean {shares.mean():.3f}")
    ax.axvline(IMAS_AGENTIC_MEAN, color=IMAS, lw=1.5, ls=":", label="Imas agentic mean 0.35")
    ax.set_xlabel("sellshare  (seller's share of the $4,000 surplus)")
    ax.set_ylabel("negotiations")
    ax.set_title("Outcome distribution", fontsize=10)
    ax.legend(fontsize=8)

    # (2) Spread against Imas's agentic quartiles. The whole point of Phase 1 is
    # that our box should be dramatically narrower than his.
    ax = axes[0][1]
    ax.boxplot(
        [shares, [IMAS_AGENTIC_P25, IMAS_AGENTIC_MEDIAN, IMAS_AGENTIC_P75]],
        tick_labels=["ours\n(fixed prompt)", "Imas agentic\n(299 prompts, quartiles)"],
        orientation="vertical",
        widths=0.5,
    )
    ax.set_ylabel("sellshare")
    ax.set_ylim(0, 1)
    ax.set_title(f"Dispersion  (our σ = {shares.std():.4f})", fontsize=10)
    ax.text(
        0.5,
        0.02,
        "right box is reconstructed from published quartiles, not raw data",
        transform=ax.transAxes,
        ha="center",
        fontsize=7,
        color=MUTED,
    )

    # (3) Focal-point adherence — the moment Phase 2 is aiming at.
    ax = axes[1][0]
    ours_focal = (shares.sub(0.5).abs() < 0.01).mean()
    bars = ["ours", "Imas\nagentic", "Imas\nhuman"]
    vals = [ours_focal, IMAS_AGENTIC_FOCAL, IMAS_HUMAN_FOCAL]
    ax.bar(bars, vals, color=[OURS, IMAS, MUTED])
    for i, v in enumerate(vals):
        ax.text(i, v + 0.008, f"{v:.1%}", ha="center", fontsize=9)
    ax.set_ylabel("share of deals at 50-50")
    ax.set_ylim(0, max(vals) * 1.25)
    ax.set_title("50-50 focal mass", fontsize=10)

    # (4) Convergence: is N large enough? Running estimate of sigma as
    # negotiations accumulate. A trace that has gone flat means more draws would
    # not move the headline number; one still wandering means extend the run.
    ax = axes[0][2]
    running = shares.expanding(min_periods=5).std()
    ax.plot(range(1, len(running) + 1), running, color=OURS, lw=1.5)
    final = shares.std()
    band = final * 0.04  # approx SE of sigma at n=300
    ax.axhline(final, color=MUTED, ls="--", lw=1)
    ax.fill_between(
        range(1, len(running) + 1), final - band, final + band, color=MUTED, alpha=0.15
    )
    ax.set_xlabel("negotiations")
    ax.set_ylabel("running σ")
    ax.set_title(f"Has σ settled?  (final {final:.4f})", fontsize=10)
    ax.text(
        0.97, 0.05, "shaded = ±4% (approx SE at n=300)",
        transform=ax.transAxes, ha="right", fontsize=7, color=MUTED,
    )

    # (5) Message length. The verbosity question, made empirical: Claude writes
    # far longer negotiation turns than gpt-4.1-mini would have.
    ax = axes[1][2]
    lengths = [len(t["text"]) for r in df["transcript"] for t in r]
    ax.hist(lengths, bins=30, color=OURS, alpha=0.85)
    ax.axvline(
        sum(lengths) / len(lengths), color=IMAS, lw=1.5,
        label=f"mean {sum(lengths)/len(lengths):.0f} chars",
    )
    ax.set_xlabel("characters per message")
    ax.set_ylabel("messages")
    ax.set_title("Message length (verbosity)", fontsize=10)
    ax.legend(fontsize=8)

    # (6) How long negotiations ran before closing.
    ax = axes[1][1]
    rounds = df["rounds_used"].dropna()
    ax.hist(rounds, bins=range(1, 14), color=OURS, alpha=0.85, align="left")
    ax.set_xlabel("rounds used")
    ax.set_ylabel("negotiations")
    ax.set_xticks(range(1, 13))
    no_deal = 1 - len(deals) / len(df)
    ax.set_title(f"Rounds to close  (no-deal rate {no_deal:.1%}, Imas 3%)", fontsize=10)

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_path, dpi=150)
    print(f"wrote {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path, help="JSONL produced by run_noise_floor.py")
    args = parser.parse_args()

    df = load(args.path)
    if df.empty:
        raise SystemExit("no valid records to plot")
    make_figure(df, args.path.with_suffix(".png"))


if __name__ == "__main__":
    main()
