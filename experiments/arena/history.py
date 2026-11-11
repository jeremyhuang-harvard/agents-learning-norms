"""Assemble and render a seeded experience history.

A history is 10 past negotiations drawn from the pool. Only ONE-SIDE-BRIEFED pool
transcripts qualify: the agent is always told it was the unbriefed party ("past-you"), so
a transcript with both sides briefed has no neutral chair to seat it in. The interior-grid
data is therefore support-mapping only, never history material.

Length is fixed at 10 in every condition, so length can never be the confound. Only the
MIX varies.

What a history episode shows the agent:
  - the good and its market range          (varies per episode -- forces "fair" to be
                                            relational rather than a memorised number)
  - the full dialogue, both sides
  - past-you's private tactical notes      (captured off-thread at generation time)
  - the counterparty's private notes       NEVER -- you cannot see inside the environment;
                                            that asymmetry IS the inference problem
  - the outcome

Past-you's notes deliberately contain no verdict on what KIND of negotiation it was. The
experiment asks agents "what kind of situation is this?" -- if the history answers that
question, the probe measures copying.
"""

from __future__ import annotations

import glob
import json
import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from experiments.pool.goods import BY_KEY, HISTORY_GOODS, REVISED_GOODS

POOL_GLOB = "experiments/pool/data/pool_*.jsonl"
HISTORY_LEN = 10


def load_pool(state_range: bool = True) -> List[Dict]:
    """One-side-briefed transcripts only, with an unbriefed past-you.

    ⚠️ `state_range` PARTITIONS the bank and must never be mixed. A range-posted transcript
    and a blind one are different games: in the first, both sides could see the whole
    bargaining zone (the range endpoints ARE the two reservation prices); in the second each
    side knew only its own. A history containing both would teach an incoherent world.
    """
    out = []
    for p in sorted(glob.glob(POOL_GLOB)):
        stem = Path(p).stem
        for line in Path(p).read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            if "error" in r or not r.get("transcript") or r.get("bin") in (None, "no_deal"):
                continue
            if r.get("past_you_role") is None:
                continue
            if r.get("state_range", True) != state_range:
                continue
            # A good's definition can change (see REVISED_GOODS in goods.py). Records made
            # against the old one argue about facts the new description no longer states,
            # so rendering them would produce a memory that contradicts its own header.
            if r["good"] in REVISED_GOODS and \
                    r.get("condition_note") != BY_KEY[r["good"]].condition_note:
                continue
            # idx is per-FILE, so it collides across pool batches. Stamp a globally
            # unique key -- overlap diagnostics and probe-replay provenance both depend on
            # it. NOTE promote.py copies records out of many source files into one, so the
            # containing file is NOT enough: use the ORIGINAL filename when present.
            src = r.get("promoted_from", stem)
            r["uid"] = f"{src}:{r['idx']}"
            out.append(r)

    # Provenance is load-bearing (a history records which transcripts fed it, and a probe
    # replay resolves them back). A duplicate uid silently makes that lookup ambiguous, so
    # fail loudly rather than carry on.
    dupes = [u for u, n in Counter(r["uid"] for r in out).items() if n > 1]
    if dupes:
        raise ValueError(
            f"{len(dupes)} duplicate transcript ids in the pool, e.g. {dupes[:3]}. "
            f"Every record needs a globally unique (source file, idx)."
        )
    return out


@dataclass(frozen=True)
class Composition:
    """How many of the 10 episodes ended evenly, and what the rest were.

    role        : which chair past-you occupied throughout ("seller" | "buyer")
    n_even      : episodes that closed near an even split
    lopsided    : "unfavourable" (past-you was rolled) or "favourable" (past-you cleaned up)

    NOTE on availability -- the pool cannot supply every combination:
        past-you = seller : 57 even, 36 unfavourable, only 2 favourable
        past-you = buyer  : 55 even, 35 favourable,   only 2 unfavourable
    Seller-favourable and buyer-unfavourable are essentially unreachable in this game (a
    seller never clears ~0.63; a buyer rarely falls below ~0.4). So the direction contrast
    cannot be run within a single role. Assembly will refuse rather than silently reuse a
    handful of episodes.
    """
    role: str
    n_even: int
    lopsided: str = "unfavourable"
    # ⭐ focal=True narrows "even" from a BAND to a POINT. Discovered 2026-08-12: our even
    # bin is +/-0.10, so ten "even" memories average 0.530 and only 18% sit at exactly
    # 50-50. An agent reads that as "I normally do a bit better than half" and aims ABOVE
    # the midpoint -- so the condition we ran was never the focal treatment we intended.
    # With focal=True every even episode closed at exactly the midpoint, which is the only
    # history that can install a genuine focal norm rather than a favourable expectation.
    focal: bool = False

    @property
    def n_lopsided(self) -> int:
        return HISTORY_LEN - self.n_even

    @property
    def label(self) -> str:
        return f"{self.n_even}/{self.n_lopsided}" + ("f" if self.focal else "")


def assemble(pool: List[Dict], comp: Composition, rng: random.Random) -> List[Dict]:
    """Draw one history: EXACTLY ONE episode per history good.

    Every history therefore contains the same ten items. What varies across histories is
    (a) which of the ten ended evenly and which ended lopsided, and (b) which particular
    transcript fills each slot. Good composition contributes nothing to between-history
    variance, which is the point -- see the note in goods.py.
    """
    want = "S" if comp.role == "seller" else "B"
    goods = [g.key for g in HISTORY_GOODS]
    rng.shuffle(goods)
    even_goods, lop_goods = goods[:comp.n_even], goods[comp.n_even:]

    out, missing = [], []
    for gs, b in ((even_goods, "even"), (lop_goods, comp.lopsided)):
        for gk in gs:
            cands = [r for r in pool
                     if r["sub_pool"] == want and r["bin"] == b and r["good"] == gk]
            if comp.focal and b == "even":
                cands = [r for r in cands if abs(r["past_you_share"] - 0.5) < 1e-9]
            if not cands:
                missing.append(f"{want}/{b}/{gk}")
            else:
                out.append(rng.choice(cands))
    if missing:
        raise ValueError(
            f"pool has no transcript for {len(missing)} cell(s) needed by "
            f"{comp.role} {comp.label}: {', '.join(sorted(missing))}\n"
            f"Run: .venv/bin/python -m experiments.pool.topup --report"
        )
    return out


def deficit(pool: List[Dict], target: int) -> Dict[tuple, int]:
    """How many transcripts each history cell is short of `target`.

    The four cells that matter are the ones histories are built from:
        S/even, S/unfavourable   (seller histories)
        B/even, B/favourable     (buyer histories)
    The other two -- S/favourable and B/unfavourable -- are unreachable in this game and
    are not used by any condition.
    """
    from collections import Counter
    have = Counter((r["sub_pool"], r["bin"], r["good"]) for r in pool)
    out = {}
    for sp, b in (("S", "even"), ("S", "unfavourable"),
                  ("B", "even"), ("B", "favourable")):
        for g in HISTORY_GOODS:
            short = target - have.get((sp, b, g.key), 0)
            if short > 0:
                out[(sp, b, g.key)] = short
    return out


def render_episode(r: Dict, n: int) -> str:
    """One past negotiation, from past-you's point of view.

    Two renderings, because the two banks are two different games.

    RANGE POSTED — the market range was common knowledge during the negotiation, so the
    header states it, exactly as past-you saw it.

    BLIND — neither side knew the other's reservation price. The header therefore states
    only what past-you knew going in: its OWN outside option. The counterparty's is revealed
    in the outcome line, as something learned AFTERWARDS.

    ⚠️ That post-hoc reveal is deliberate and it is what keeps the blind condition from
    being useless. Without it the agent sees ten deals above its own floor and cannot tell a
    triumph from a rout, so "ten wins" and "ten losses" stop being different treatments. With
    it, the agent can work out how it did relative to the zone — but the negotiation itself
    was still conducted blind, which is what separates this from the range-posted bank.

    ⚠️ The reveal states the counterparty's number as a BARE FACT and stops there. It does
    not say "so you took 48% of the zone." Computing the share is the inference we want to
    observe in the probes, not one we should supply.
    """
    lo, hi = r["range"]
    role = r["past_you_role"]
    other = "seller" if role == "buyer" else "buyer"
    desc = BY_KEY[r["good"]].description
    if r.get("state_range", True):
        opening = (f"You were the {role}. The item was {desc}; the market range was "
                   f"${lo:,} to ${hi:,}.")
    else:
        mine = (f"you could have sold it elsewhere for ${lo:,}" if role == "seller"
                else f"you could have bought a comparable one elsewhere for ${hi:,}")
        opening = f"You were the {role}. The item was {desc}, and {mine}."
    lines = [f"### Negotiation {n}", opening, ""]
    for t in r["transcript"]:
        who = "You" if t["role"] == role else f"The {other}"
        lines.append(f"**{who}:** {t['text']}")
        # past-you's private note only -- never the counterparty's
        if t["role"] == role and t.get("thinking"):
            lines.append(f"*(What you were thinking: {t['thinking']})*")
        lines.append("")
    price = r.get("transaction_price")
    if price is None:
        lines.append("**Outcome:** no deal was reached.")
    elif r.get("state_range", True):
        lines.append(f"**Outcome:** closed at ${price:,}.")
    else:
        theirs = (f"the {other} could have bought a comparable one elsewhere for ${hi:,}"
                  if role == "seller"
                  else f"the {other} could have sold it elsewhere for ${lo:,}")
        lines.append(f"**Outcome:** closed at ${price:,}. Afterwards you learned that "
                     f"{theirs}.")
    return "\n".join(lines)


def render(history: List[Dict], shuffle_rng: Optional[random.Random] = None) -> str:
    """The full history block that goes into the agent's system prompt.

    Framed autobiographically -- these are the agent's OWN past negotiations, which is
    what BGLS's F_c (frequency of category use) requires. Order is shuffled so that
    composition is not readable from position.
    """
    eps = list(history)
    if shuffle_rng is not None:
        shuffle_rng.shuffle(eps)
    body = "\n\n".join(render_episode(r, i + 1) for i, r in enumerate(eps))
    return (
        "## Your recent negotiating experience\n\n"
        "These are your last 10 negotiations, in no particular order. Each shows what was "
        "said, what you were thinking at the time, and how it ended.\n\n" + body
    )


def inventory(pool: List[Dict]) -> Dict:
    from collections import Counter
    return dict(Counter((r["sub_pool"], r["bin"]) for r in pool))
