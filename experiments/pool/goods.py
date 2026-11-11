"""The goods negotiated over in pool transcripts.

Imas's design uses one car with a $18k-$22k range, where the range endpoints ARE the
two outside options (seller can get the low end elsewhere, buyer can get the high end
elsewhere). We keep that structure and vary the good.

Why vary it: if every past negotiation in an agent's history is the same $18-22k Camry
and the target negotiation is also a $18-22k Camry, then "fair" is a *number* ($20,000)
the agent can copy forward without understanding anything. Varying the good forces
"fair" to be relational -- near the midpoint of whatever range is on the table -- which
is the kind of thing a category actually is.

Widths differ as well as levels, so the midpoint is never the same number twice.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Good:
    key: str
    description: str      # "a used 2020 Toyota Camry LE"
    low: int              # seller's outside option
    high: int             # buyer's outside option
    condition_note: str   # the "common knowledge" condition line

    @property
    def surplus(self) -> int:
        return self.high - self.low

    @property
    def midpoint(self) -> int:
        return (self.low + self.high) // 2


GOODS = [
    Good(
        key="camry",
        description="a used 2020 Toyota Camry LE",
        low=18_000, high=22_000,
        condition_note="Vehicle history: No accidents reported",
    ),
    Good(
        key="motorcycle",
        description="a used 2019 Honda CB500F motorcycle",
        low=4_200, high=5_800,
        condition_note="14,000 miles, one previous owner, no drops or damage",
    ),
    Good(
        key="espresso",
        description="a commercial espresso machine (2-group)",
        low=7_500, high=10_500,
        condition_note="Serviced within the last year, no known faults",
    ),
    Good(
        key="pickup",
        description="a used 2018 Ford F-150 XLT",
        low=24_000, high=31_000,
        condition_note="68,000 miles, no accidents reported, service records available",
    ),
    Good(
        key="kitchen",
        description="a lot of restaurant kitchen equipment (range, hood, prep tables)",
        low=12_000, high=16_000,
        condition_note="All units working; sold as a single lot, buyer arranges removal",
    ),
    Good(
        key="sailboat",
        description="a 26-foot sailboat with trailer",
        low=9_000, high=14_000,
        condition_note="Hull sound, sails replaced two seasons ago, engine runs",
    ),
    # --- added 2026-08-11 to bring the history set to ten -----------------------
    Good(
        key="trailer",
        description="a 2017 travel trailer (24-foot, sleeps six)",
        low=15_000, high=19_500,
        condition_note="No leaks or soft spots; appliances and awning all functional",
    ),
    Good(
        key="tractor",
        description="a compact utility tractor with front loader",
        low=21_000, high=26_000,
        condition_note="1,100 hours, recent hydraulic service, loader bucket included",
    ),
    Good(
        key="piano",
        description="a restored Steinway model B grand piano",
        low=28_000, high=38_000,
        condition_note="Rebuilt action and new strings; bench and delivery not included",
    ),
    Good(
        key="camera",
        description="a cinema camera package with three prime lenses",
        low=2_800, high=4_000,
        condition_note="Roughly 900 recorded hours; sensor clean, no dead pixels",
    ),
    Good(
        key="freezer",
        description="a walk-in freezer with refrigeration unit",
        low=6_000, high=8_200,
        condition_note="Holds temperature; buyer arranges disassembly and removal",
    ),
]

BY_KEY = {g.key: g for g in GOODS}

# ⭐ THE SPLIT (2026-08-11). The Camry is the ARENA and never appears in a history; the
# other ten are the HISTORY set and never appear in the arena.
#
# Why the Camry is excluded from histories: a past Camry that closed at $20,000 hands the
# agent a literal number for the exact deal it is about to do -- the number-copying channel
# this whole file was written to shut down. Excluding it makes the arena a genuine transfer.
#
# Why exactly ten, one appearance each: a history is ten episodes, so "each good exactly
# once" means good composition is IDENTICAL across every history and every arena. The only
# thing that differs between two sellers is which outcomes attached to which items. Any
# looser rule (draw ten of thirteen, or allow repeats) leaks "seller #3 happened to draw two
# motorcycles" into between-history variance -- which is the very quantity the mixture test
# is trying to measure.
#
# Levels and widths are deliberately spread (midpoints $3.4k to $33k, widths $1.2k to $10k)
# so no midpoint recurs and none is near the Camry's $20,000.
# ⚠️ REVISED 2026-08-12. The pickup carried "one accident on record, repaired" and the
# espresso machine was "refurbished". Both were throwaway flavour text, and both turned out
# to be load-bearing: they hand the buyer legitimate ammunition to push the price down, and
# those two goods produced ZERO seller-favourable outcomes in 30 draws each while the
# kitchen lot (whose note makes the BUYER bear removal costs) produced 7. A casual phrase
# was silently deciding which experimental conditions are reachable.
#
# Any record generated before this revision describes a DIFFERENT item -- its transcript
# text argues about the accident -- so it must not be rendered under the new description.
# generate_one now stamps condition_note into every record and load_pool drops mismatches.
REVISED_GOODS = {"pickup", "espresso"}

ARENA_GOOD_KEY = "camry"
HISTORY_GOODS = [g for g in GOODS if g.key != ARENA_GOOD_KEY]
assert len(HISTORY_GOODS) == 10, "a history is ten episodes, one per good"
assert len({g.midpoint for g in GOODS}) == len(GOODS), "midpoints must be distinct"
