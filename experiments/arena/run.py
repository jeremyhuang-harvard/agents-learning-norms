"""Arena runner — Arm A (experience) and Arm B (experience -> playbook -> negotiate).

    Arm A   seller carries its 10-episode history into the negotiation
    Arm B   seller compiles that same history into a playbook, then a FRESH agent
            negotiates from the playbook alone

The A-B gap is the tacit-knowledge measurement: whatever the history did that the playbook
does not do is experience that would not go into words.

Two things must be held fixed for that gap to mean anything, and both are enforced here by
construction rather than by hoping:

  1. **The same seller histories.** Arm B's seller h in arena a is compiled from the exact
     ten transcripts Arm A's seller h in arena a lived. The draw is seeded on
     (arena, h), not on the arm.
  2. **The same environment.** Identical buyer panel, identical seller x buyer pairing,
     identical good on every pairing. Only the seller's architecture changes. Buyers are
     Arm A in BOTH arms -- changing both sides at once would leave the gap unattributable.

## Compositions

A composition is "how many of the ten ended evenly, and what the rest were." Three arenas:
10/0 (all even), 5/5, 0/10 (all lopsided).

⚠️ **THE TWO ROLES' 0/10 CONDITIONS ARE NOT THE SAME TREATMENT.** The pool can only supply
seller-unfavourable and buyer-favourable lopsided episodes (a seller essentially never
clears ~0.63 of the surplus, a buyer essentially never falls below ~0.4). So a 0/10 seller
has ten losses and a 0/10 buyer has ten wins. Within the seller arenas -- which is where the
whole Arm A/Arm B comparison lives -- this is fine and the dose runs even -> beaten. Across
roles it is not a clean contrast and must not be reported as one.

## Run plan (defaults)

    3 arenas x 20 sellers x 6 negotiations = 360 negotiations per arm
    buyer panel: 24 histories (8 at each of 3 compositions), recycled everywhere
    every negotiation is the CAMRY -- Imas's setting, so our distribution is
      comparable to his human 34.7% / agentic 14.3% focal masses
    k=6 => one buyer per composition, each faced TWICE, so each seller yields
      3 true replications (same seller, same buyer, same car, different draw)

Usage:
    .venv/bin/python -m experiments.arena.run --preview          # no API calls at all
    .venv/bin/python -m experiments.arena.run --preview --full   # + full prompt dumps
    .venv/bin/python -m experiments.arena.run --arm B            # spends money
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import random
import statistics as st
import time
from pathlib import Path
from typing import Dict, List, Optional

from experiments.arena import history as H
from experiments.arena import negotiate as NEG
from experiments.arena import playbook as PB
from experiments.pool.generate import load_dotenv
from experiments.pool.goods import BY_KEY, GOODS

DATA_DIR = Path(__file__).parent / "data"
CONCURRENCY = 8
STATE_RANGE = True               # overridden by --no-range
# ⭐ CELL 3. --no-range removes the posted range from the ARENA only; the histories still
# come from the range-posted bank, so the agent has SEEN bargaining zones and now faces one
# it cannot see. --blind-history additionally draws the histories from the BLIND bank, so
# the agent has never had access to a zone at all — in memory or in front of it.
#
#   baseline   zone visible in memory, zone visible in the arena
#   Cell 2     zone visible in memory, arena blind
#   Cell 3     blind in memory, arena blind
BLIND_HISTORY = False            # overridden by --blind-history

# ⭐ THE ARENA IS ALWAYS THE CAMRY. Imas's setting, unchanged -- 2020 Camry, seller
# fallback $18k, buyer fallback $22k. His human (34.7% focal) and agentic (14.3%)
# distributions are defined on THIS setting and nowhere else, so rotating goods through the
# arena would leave our sellshare distribution non-comparable to the only benchmark the
# project is aimed at.
#
# The six goods exist for the HISTORIES, and for one reason: ten identical Camrys would
# make "fair" a number ($20,000) to copy, where varied goods and ranges force it to be
# abstracted as relational -- near the midpoint of whatever is on the table. Abstraction is
# what a category IS. Varying the good in the arena as well confuses the two jobs.
ARENA_GOOD = "camry"

# (n_even, what the other episodes were). The first three are the DOSE-RESPONSE: even ->
# half -> all lopsided. The fourth is the DIRECTION contrast, and it is the one place the
# two theories disagree outright -- payoff learning says a seller that lost ten times should
# concede more than one that won ten times; categorisation says both land in the same
# competitive category and behave alike.
#
# ⚠️ Both 0-even arenas would otherwise be labelled "0/10" and collide, so the wins arena
# carries a "w". The three original labels are unchanged on purpose: seller ids are derived
# from them, and renaming would orphan data already on disk.
# (n_even, what the lopsided ones were, whether "even" means EXACTLY 50-50)
#
# ⭐ The fifth arena is the focal treatment we thought we were running all along. Our even
# bin is a BAND (+/-0.10), so 10/0 memories average 0.530 and the agent aims ABOVE the
# midpoint -- it reads "I normally do a bit better than half", not "deals land at half".
# 10/0f uses only episodes that closed at exactly the midpoint. Comparing the two is a
# direct test of whether a genuine focal history installs a genuine focal norm.
SELLER_ARENAS = [(10, "unfavourable", False), (5, "unfavourable", False),
                 (0, "unfavourable", False), (0, "favourable", False),
                 (10, "unfavourable", True)]
SELLER_COMPS = SELLER_ARENAS      # kept as an alias; older code iterates this
BUYER_COMPS = [10, 5, 0]         # n_even; the rest are lopsided-favourable
SELLERS_PER_ARENA = 20
# ⭐ 8, not 4 (raised 2026-08-12). The panel size, NOT the negotiation count, is what limits
# precision on the buyer arm -- each seller plays 6 either way, so raising it is free.
#   4 per type -> buyer MDE 0.105, against 0.057 for the seller arm: badly underpowered
#   8 per type -> buyer MDE 0.079, and each buyer is still faced by 2-3 sellers per arena
#  20 per type -> buyer MDE 0.057, but 1:1 seller-buyer pairing, so seller and buyer effects
#                 become perfectly confounded and the S_ij = theta_i + theta_j + eps
#                 decomposition the panel exists for is lost
# 8 is the point where both the main effect and the crossing survive.
BUYERS_PER_COMP = 8
REPS = 2                         # true repeats per (seller, buyer) -- see pairings()
K = len(BUYER_COMPS) * REPS      # negotiations per seller

BASE_SEED = 20260811


# --- panel construction ------------------------------------------------------
#
# Every probe we might want later is a REPLAY: rebuild the agent's exact system prompt and
# its message list up to turn t, and ask it a question off-thread. The API is stateless, so
# a replayed request with identical inputs is the same request -- but only if the rendered
# history block is byte-identical. It is seeded, so it is reproducible today and silently
# is not the moment anyone edits history.py. So every run writes an `.agents.jsonl`
# sidecar holding each agent's FULL rendered block plus its sha, and every negotiation
# record carries the shas of the two blocks that produced it. Replay verifies the sha and
# refuses rather than probing the wrong context.


def _sha(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()[:12]


def build_buyers(pool: List[Dict]) -> List[Dict]:
    """The fixed buyer panel. Identical in every arena and in both arms."""
    out = []
    for comp_even in BUYER_COMPS:
        comp = H.Composition("buyer", comp_even, lopsided="favourable")
        for j in range(BUYERS_PER_COMP):
            rng = random.Random(BASE_SEED + 900_000 + comp_even * 100 + j)
            eps = H.assemble(pool, comp, rng)
            block = H.render(eps, random.Random(BASE_SEED + 1 + j))
            out.append({
                "buyer_id": f"b{comp.label.replace('/', '-')}#{j}",
                "comp": comp.label,
                "episodes": [e["uid"] for e in eps],
                "block": block,
                "sha": _sha(block),
            })
    return out


def arena_label(arena_even: int, lopsided: str = "unfavourable",
                focal: bool = False) -> str:
    """'0/10' losses, '0/10w' wins, '10/0f' exactly-at-the-midpoint."""
    base = f"{arena_even}/{H.HISTORY_LEN - arena_even}"
    return base + ("" if lopsided == "unfavourable" else "w") + ("f" if focal else "")


def build_sellers(pool: List[Dict], arena_even: int,
                  lopsided: str = "unfavourable", focal: bool = False) -> List[Dict]:
    """Sellers for one arena. Seeded on (arena, h) so both arms get the same ten.

    The seed for the losses arenas is deliberately unchanged from before the wins arena
    existed, so those sellers stay byte-identical and data already on disk stays valid.
    """
    comp = H.Composition("seller", arena_even, lopsided=lopsided, focal=focal)
    lab = arena_label(arena_even, lopsided, focal)
    out = []
    for h in range(SELLERS_PER_ARENA):
        off = (7_000_000 if lopsided == "favourable" else 0) + (11_000_000 if focal else 0)
        rng = random.Random(BASE_SEED + off + arena_even * 1000 + h)
        eps = H.assemble(pool, comp, rng)
        block = H.render(eps, random.Random(BASE_SEED + 500 + h))
        out.append({
            "seller_id": f"s{lab.replace('/', '-')}#{h}",
            "arena": lab,
            "episodes": [e["uid"] for e in eps],
            "block": block,
            "sha": _sha(block),
        })
    return out


def pairings(seller_ix: int) -> List[Dict]:
    """The k negotiations this seller plays. Identical in both arms.

    One buyer from each composition, each faced REPS times. The good is the Camry
    throughout, so the ONLY thing separating a seller's k negotiations is which buyer it
    meets -- and within a (seller, buyer) pair, nothing at all.

    That is what makes the mixture test identified. Two runs of the same seller against
    the same buyer over the same car differ only in the sampling draw, so if the outcomes
    split into two modes it is the SAME experience producing two readings of the situation
    -- BGLS's within-person multimodality, the prediction that is nearly untestable in
    humans. With a different good and a different opponent every time (the earlier
    version) that split is indistinguishable from motorcycles simply negotiating
    differently from pickups.

    Which buyer is drawn from each composition rotates with seller index, so all
    BUYERS_PER_COMP are exercised rather than every seller meeting #0.
    """
    out = []
    for c in range(len(BUYER_COMPS)):
        within = (seller_ix + c) % BUYERS_PER_COMP
        for rep in range(REPS):
            out.append({
                "j": c * REPS + rep,
                "rep": rep,
                "good": ARENA_GOOD,
                "buyer_slot": c * BUYERS_PER_COMP + within,
            })
    return out


# --- preview (zero API calls) ------------------------------------------------


def _tok(s: str) -> int:
    """Real token count via the free count_tokens endpoint; chars/4 if unavailable."""
    try:
        from anthropic import Anthropic
        load_dotenv()
        return Anthropic().messages.count_tokens(
            model=NEG.MODEL, messages=[{"role": "user", "content": s}]).input_tokens
    except Exception:
        return len(s) // 4


def preview(full: bool) -> None:
    pool = H.load_pool()
    print("=" * 78)
    print("POOL INVENTORY (one-side-briefed transcripts only)")
    print("=" * 78)
    for (sp, b), n in sorted(H.inventory(pool).items()):
        print(f"  sub_pool {sp}   {b:<14} {n:>4}")

    try:
        buyers = build_buyers(pool)
        arenas = {arena_label(e, d, f): build_sellers(pool, e, d, f)
                  for e, d, f in SELLER_ARENAS}
    except ValueError as exc:
        print(f"\n⚠️  CANNOT BUILD PANELS YET\n\n{exc}\n")
        print("Every history needs one episode per history good, so each of the four "
              "history\ncells must be non-empty for all ten goods. Top the pool up first:")
        print("\n    .venv/bin/python -m experiments.pool.topup --report")
        print("    .venv/bin/python -m experiments.pool.topup --target 8 --dry-run\n")
        return

    print("\n" + "=" * 78)
    print("PANELS")
    print("=" * 78)
    print(f"  buyer panel : {len(buyers)} histories "
          f"({BUYERS_PER_COMP} at each of {BUYER_COMPS} even/lopsided-FAVOURABLE)")
    for a, ss in arenas.items():
        print(f"  arena {ss[0]['arena']:<5}: {len(ss)} sellers "
              f"(lopsided = UNFAVOURABLE, i.e. past-you was beaten)")

    s0 = arenas["5/5"][0]
    b0 = buyers[0]
    print("\n  history block size by arena (real token counts):")
    sizes = {}
    for a, ss in arenas.items():
        sizes[a] = st.mean(_tok(s["block"]) for s in ss)
        print(f"    seller {ss[0]['arena']:<5} {sizes[a]:>7,.0f} tokens (mean over "
              f"{len(ss)} sellers)")
    print(f"    buyer  5/5   {_tok(b0['block']):>7,} tokens")
    swell = sizes["0/10"] / sizes["10/0"] - 1
    print(f"  ⚠️ LOPSIDED HISTORIES ARE LONGER — lopsided negotiations ran more rounds, so")
    print(f"     the 0/10 agent gets {swell:.0%} more context than the 10/0 agent.")
    print( "     Composition is confounded with history LENGTH. Report it; if it bites, the")
    print( "     fix is a length-matched control that truncates episodes to fixed rounds.")

    # how much do sellers within an arena actually differ? the pool is finite, so
    # histories overlap -- which attenuates between-history variance, the quantity the
    # mixture test needs power on.
    print("\n  episode overlap within each arena (mean shared episodes per seller pair):")
    for a, ss in arenas.items():
        sets = [set(s["episodes"]) for s in ss]
        pairs = [len(x & y) for i, x in enumerate(sets) for y in sets[i + 1:]]
        print(f"    arena {ss[0]['arena']:<5} {st.mean(pairs):>5.1f} of 10   "
              f"(drawing from {len(sets[0])} per history)")

    print("\n" + "=" * 78)
    print(f"PAIRING — arena good is always {ARENA_GOOD.upper()} (Imas's setting)")
    print("=" * 78)
    for si in (0, 1):
        print(f"  seller #{si}: " + ", ".join(
            f"{buyers[p['buyer_slot']]['buyer_id']}(r{p['rep']})" for p in pairings(si)))
    print(f"  => {len(BUYER_COMPS)} true replications per seller "
          f"(same seller, same buyer, same car, {REPS} draws)")

    n_neg = len(SELLER_ARENAS) * SELLERS_PER_ARENA * K
    n_pb = len(SELLER_ARENAS) * SELLERS_PER_ARENA
    print("\n" + "=" * 78)
    print("RUN PLAN")
    print("=" * 78)
    print(f"  Arm B : {n_pb} playbook writes + {n_neg} negotiations")
    print(f"  Arm A : {n_neg} negotiations")
    print(f"  seller carries history in Arm A only; buyer carries history in BOTH")

    # cost: per negotiation, 12 turns. Buyer turns carry the history block (cached after
    # the first negotiation of that buyer); Arm A seller turns too. Sonnet 5 intro
    # pricing $2/M in, $10/M out, cache read 0.1x, cache write 1.25x.
    hb = max(_tok(ss[0]["block"]) for ss in arenas.values())
    tmpl, outp = 400, 350
    def neg_cost(seller_has_history: bool) -> float:
        c = 0.0
        for turn in range(12):
            sysl = hb if (turn % 2 == 0 or seller_has_history) else 0
            convo = 300 * turn
            c += (sysl * 0.1 * 2 + (tmpl + convo) * 2 + outp * 10) / 1e6
        return c + 2 * (2500 * 2 + 200 * 10) / 1e6      # the two parser calls
    cb, ca = neg_cost(False), neg_cost(True)
    pbc = (hb * 2 + 600 * 10) / 1e6
    print(f"\n  ~${cb:.3f}/negotiation Arm B   ~${ca:.3f}/negotiation Arm A "
          f"  ~${pbc:.3f}/playbook")
    print(f"  ~${n_neg*cb + n_pb*pbc:.0f} Arm B     ~${n_neg*ca:.0f} Arm A"
          f"     ~${n_neg*(ca+cb) + n_pb*pbc:.0f} both")
    print("  (order-of-magnitude. Assumes the history block is a CACHE HIT after the first\n"
          "   negotiation each agent plays — a miss costs ~10x more on that turn. Sellers\n"
          "   are run serially within a seller to make that hold; buyers are shared and\n"
          "   warm up on their own.)")

    print("\n" + "=" * 78)
    print("ARM B PLAYBOOK-WRITER PROMPT  (seller, arena 5/5, history elided)")
    print("=" * 78)
    p = PB.build_writer_prompt("seller", "<<HISTORY BLOCK>>")
    print("--- system ---\n" + p["system"])
    print("\n--- user ---\n" + p["user"])

    print("\n" + "=" * 78)
    print("ARM B NEGOTIATOR SYSTEM PROMPT  (playbook in the {sellerPrompt} slot)")
    print("=" * 78)
    blocks = NEG.build_agent_system(BY_KEY[ARENA_GOOD], "seller", playbook="<<PLAYBOOK>>")
    print(blocks[0]["text"])

    if full:
        print("\n" + "=" * 78)
        print("ARM A SELLER SYSTEM PROMPT — FULL")
        print("=" * 78)
        for blk in NEG.build_agent_system(BY_KEY[ARENA_GOOD], "seller", history_block=s0["block"]):
            print(blk["text"])


# --- execution ---------------------------------------------------------------


async def run_arm(arm: str, out_path: Path) -> None:
    from anthropic import AsyncAnthropic

    key = load_dotenv()
    client = AsyncAnthropic(max_retries=5)
    pool = H.load_pool(state_range=not BLIND_HISTORY)
    buyers = build_buyers(pool)
    # An arena the bank cannot supply is SKIPPED, not fatal. The blind bank is younger and
    # thinner, and a missing cell there should cost that condition, not the whole run.
    arenas = {}
    for e, d, f in SELLER_ARENAS:
        lab = arena_label(e, d, f)
        try:
            arenas[lab] = build_sellers(pool, e, d, f)
        except ValueError as exc:
            print(f"  SKIP arena {lab}: {str(exc).splitlines()[0]}")
    if not arenas:
        raise SystemExit("no arena could be built from this bank")
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    done = set()
    if out_path.exists():
        for l in out_path.read_text().splitlines():
            if l.strip():
                try:
                    r = json.loads(l)
                    done.add((r["seller_id"], r["j"]))
                except Exception:
                    pass

    print(f"Using API key {key}\nModel: {NEG.MODEL}\nArm {arm} -> {out_path}")

    # Provenance sidecar: the full rendered block for every agent in this run, so a probe
    # replay never has to re-derive one and can verify it byte-for-byte.
    # ⚠️ APPEND what is missing; never skip because the file exists. A run once inherited a
    # copied sidecar covering only 12 of its 80 sellers, and the gap only surfaced when a
    # later probe could not rebuild them. Re-deriving is NOT a fallback: assemble() draws
    # from the pool, so ANY pool growth silently changes every history. The sidecar is the
    # only durable record of what an agent actually saw.
    ag_path = out_path.with_suffix(".agents.jsonl")
    have_ag = set()
    if ag_path.exists():
        for l in ag_path.read_text().splitlines():
            if l.strip():
                try:
                    have_ag.add(json.loads(l)["agent_id"])
                except Exception:
                    pass
    missing = ([b for b in buyers if b["buyer_id"] not in have_ag]
               + [s for ss in arenas.values() for s in ss
                  if s["seller_id"] not in have_ag])
    if missing:
        with ag_path.open("a") as fh:
            for a in missing:
                fh.write(json.dumps({
                    "agent_id": a.get("seller_id") or a["buyer_id"],
                    "role": "seller" if "seller_id" in a else "buyer",
                    "comp": a.get("arena") or a["comp"], "sha": a["sha"],
                    "episodes": a["episodes"], "block": a["block"]}) + "\n")
        print(f"Wrote {len(missing)} agent blocks -> {ag_path}")

    # Arm B: compile first. One playbook per seller history, written once and reused for
    # all k negotiations -- the playbook is the agent's persistent strategy, not a
    # per-deal note.
    books: Dict[str, Dict] = {}
    if arm == "B":
        pb_path = out_path.with_suffix(".playbooks.jsonl")
        if pb_path.exists():
            # tolerate a truncated final line: an abrupt suspend (laptop lid) can leave a
            # partial write, and one bad line must not cost 100 cached playbooks
            for l in pb_path.read_text().splitlines():
                if l.strip():
                    try:
                        r = json.loads(l)
                        books[r["seller_id"]] = r
                    except Exception:
                        pass
        todo = [s for ss in arenas.values() for s in ss if s["seller_id"] not in books]
        print(f"Writing {len(todo)} playbooks ({len(books)} cached)...")
        sem = asyncio.Semaphore(CONCURRENCY)

        async def compile_one(s):
            async with sem:
                try:
                    r = await PB.write_playbook(client, "seller", s["block"])
                except Exception as exc:
                    print(f"  {s['seller_id']:<12} FAILED {type(exc).__name__} — "
                          f"re-run to retry", flush=True)
                    return
            rec = {"seller_id": s["seller_id"], "arena": s["arena"],
                   "episodes": s["episodes"], **r}
            with pb_path.open("a") as fh:
                fh.write(json.dumps(rec) + "\n")
            books[s["seller_id"]] = rec
            print(f"  {s['seller_id']:<12} {r['playbook_words']:>4}w", flush=True)

        await asyncio.gather(*(compile_one(s) for s in todo))
        ws = [b["playbook_words"] for b in books.values()]
        print(f"playbook length: median {st.median(ws):.0f}w  "
              f"min {min(ws)}  max {max(ws)}  (Imas's printed examples: 26 and 40)")

    # One task PER SELLER, its k negotiations run SEQUENTIALLY. Fanning all k out at once
    # would have every one of them miss the seller's history cache simultaneously, which
    # is most of the bill in Arm A. Serial within a seller means j=0 writes the cache and
    # j=1..k-1 read it (and each read refreshes the 5-minute TTL).
    queues = []
    for a, ss in arenas.items():
        for si, s in enumerate(ss):
            if arm == "B" and s["seller_id"] not in books:
                continue                      # playbook failed; re-run picks it up
            todo_p = [p for p in pairings(si) if (s["seller_id"], p["j"]) not in done]
            if todo_p:
                queues.append((s, todo_p))
    total = sum(len(q) for _, q in queues)
    print(f"{total} negotiations to run across {len(queues)} sellers.")

    sem, lock = asyncio.Semaphore(CONCURRENCY), asyncio.Lock()
    n, spend, t0 = 0, 0.0, time.time()
    by_key = BY_KEY

    async def one(s, p):
        nonlocal n, spend
        good = by_key[p["good"]]
        b = buyers[p["buyer_slot"]]
        if arm == "A":
            seller_sys = NEG.build_agent_system(good, "seller", history_block=s["block"],
                                                state_range=STATE_RANGE)
        else:
            seller_sys = NEG.build_agent_system(
                good, "seller", playbook=books[s["seller_id"]]["playbook"],
                state_range=STATE_RANGE)
        buyer_sys = NEG.build_agent_system(good, "buyer", history_block=b["block"],
                                           state_range=STATE_RANGE)
        try:
            r = await NEG.negotiate(client, good, seller_sys, buyer_sys)
        except Exception as e:
            r = {"error": f"{type(e).__name__}: {str(e)[:120]}"}
        rec = {"arm": arm, "arena": s["arena"], "seller_id": s["seller_id"],
               "j": p["j"], "good": good.key, "range": [good.low, good.high],
               "buyer_id": b["buyer_id"], "buyer_comp": b["comp"],
               # replay keys: which blocks produced this negotiation
               "seller_sha": (s["sha"] if arm == "A"
                              else _sha(books[s["seller_id"]]["playbook"])),
               "buyer_sha": b["sha"], "state_range": STATE_RANGE, **r}
        async with lock:
            with out_path.open("a") as fh:
                fh.write(json.dumps(rec) + "\n"); fh.flush(); os.fsync(fh.fileno())
            n += 1
            if "usage" in rec:
                u = rec["usage"]
                spend += (u["input_tokens"] * 2 + u["cache_read_input_tokens"] * 0.2
                          + u["cache_creation_input_tokens"] * 2.5
                          + u["output_tokens"] * 10) / 1e6
            if n % 10 == 0 or n == total:
                print(f"  {n}/{total}  {time.time()-t0:5.0f}s  ~${spend:.2f}", flush=True)

    async def seller_queue(s, ps):
        async with sem:
            for p in ps:
                await one(s, p)

    await asyncio.gather(*(seller_queue(s, ps) for s, ps in queues))
    print(f"\nDone. ~${spend:.2f}")
    report(out_path)


def report(path: Path) -> None:
    rs = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    ok = [r for r in rs if r.get("sellshare") is not None]
    print(f"\n{'arena':<8}{'n':>4}{'mean':>8}{'sd':>7}{'focal':>8}{'min':>7}{'max':>7}")
    print("-" * 49)
    for a in [arena_label(e, d, f) for e, d, f in SELLER_ARENAS]:
        v = [r["sellshare"] for r in ok if r["arena"] == a]
        if v:
            f = sum(1 for x in v if abs(x - 0.5) < 1e-9) / len(v)
            print(f"{a:<8}{len(v):>4}{st.mean(v):>8.3f}{st.pstdev(v):>7.3f}"
                  f"{f:>8.0%}{min(v):>7.3f}{max(v):>7.3f}")
    print(f"\nno-deal {sum(1 for r in rs if 'error' not in r and r.get('sellshare') is None)}"
          f"/{len(rs)}   errors {sum(1 for r in rs if 'error' in r)}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--preview", action="store_true", help="no API calls")
    p.add_argument("--full", action="store_true", help="with --preview, dump full prompts")
    p.add_argument("--arm", choices=["A", "B"])
    p.add_argument("--blind-history", action="store_true",
                   help="draw histories from the BLIND bank too (Cell 3). Implies "
                        "--no-range: an agent whose memories contain no bargaining zone "
                        "must not then be shown one.")
    p.add_argument("--no-range", action="store_true",
                   help="drop the posted market value range from the ARENA. Histories keep "
                        "theirs — that asymmetry is the experiment.")
    p.add_argument("--sellers", type=int,
                   help="run only the first N sellers of each arena. A true PREFIX of the "
                        "full run -- histories are seeded on (arena, index), so seller 3 is "
                        "byte-identical whether you build 4 or 20 -- so a pilot resumes "
                        "into the full run instead of being redone.")
    p.add_argument("--out", type=Path)
    p.add_argument("--report", type=Path)
    a = p.parse_args()
    global STATE_RANGE, BLIND_HISTORY
    BLIND_HISTORY = a.blind_history
    STATE_RANGE = not (a.no_range or a.blind_history)
    if a.sellers:
        global SELLERS_PER_ARENA
        SELLERS_PER_ARENA = a.sellers

    if a.report:
        report(a.report)
    elif a.preview:
        preview(a.full)
    elif a.arm:
        out = a.out or DATA_DIR / f"arm{a.arm}_{time.strftime('%Y%m%d-%H%M%S')}.jsonl"
        asyncio.run(run_arm(a.arm, out))
    else:
        p.print_help()


if __name__ == "__main__":
    main()
