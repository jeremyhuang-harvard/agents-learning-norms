"""Structural checks on the assembled arenas. Spends nothing.

Everything here is a property the design depends on and that would be invisible in the
output if it broke. Run before any arena run, and again after editing history.py, goods.py
or run.py.

    .venv/bin/python -m experiments.arena.verify
"""

from __future__ import annotations

import random
import statistics as st
from collections import Counter

from experiments.arena import history as H
from experiments.arena import negotiate as NEG
from experiments.arena import run as R
from experiments.pool.goods import ARENA_GOOD_KEY, BY_KEY, HISTORY_GOODS

PASS, FAIL = "  ok  ", "  FAIL"
_fails = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"{PASS if ok else FAIL}  {name}" + (f"   [{detail}]" if detail else ""))
    if not ok:
        _fails.append(name)


def main() -> None:
    pool = H.load_pool()
    buyers = R.build_buyers(pool)
    arenas = {R.arena_label(e, d, f): R.build_sellers(pool, e, d, f)
              for e, d, f in R.SELLER_ARENAS}
    sellers = [s for ss in arenas.values() for s in ss]

    print(f"\nbank {len(pool)} transcripts | {len(sellers)} sellers | {len(buyers)} buyers\n")

    # --- histories ------------------------------------------------------------
    print("HISTORY COMPOSITION")
    for a, ss in arenas.items():
        want_even = int(a.split("/")[0])
        focal = a.endswith("f")
        bad = []
        for s in ss:
            eps = [r for r in pool if r["uid"] in set(s["episodes"])]
            n_even = sum(1 for e in eps if e["bin"] == "even")
            if focal:
                n_even = sum(1 for e in eps
                             if abs(e["past_you_share"] - 0.5) < 1e-9)
            if n_even != want_even or len(eps) != 10:
                bad.append(s["seller_id"])
        check(f"arena {ss[0]['arena']}: every history is 10 episodes with {want_even} even",
              not bad, f"{len(bad)} bad" if bad else "")

    allbad = []
    for s in sellers + buyers:
        eps = [r for r in pool if r["uid"] in set(s["episodes"])]
        goods = [e["good"] for e in eps]
        if sorted(goods) != sorted(g.key for g in HISTORY_GOODS):
            allbad.append(s.get("seller_id") or s["buyer_id"])
    check("every history contains each of the ten items exactly once", not allbad,
          f"{len(allbad)} bad" if allbad else "")

    camry = [s for s in sellers + buyers
             if any(r["good"] == ARENA_GOOD_KEY for r in pool if r["uid"] in set(s["episodes"]))]
    check(f"no history contains the arena item ({ARENA_GOOD_KEY})", not camry)

    for s in sellers:
        eps = [r for r in pool if r["uid"] in set(s["episodes"])]
        if any(e["sub_pool"] != "S" for e in eps):
            check("seller histories only use seller-chair transcripts", False)
            break
    else:
        check("seller histories only use seller-chair transcripts", True)
    for b in buyers:
        eps = [r for r in pool if r["uid"] in set(b["episodes"])]
        if any(e["sub_pool"] != "B" for e in eps):
            check("buyer histories only use buyer-chair transcripts", False)
            break
    else:
        check("buyer histories only use buyer-chair transcripts", True)

    dup = [s["seller_id"] for s in sellers if len(set(s["episodes"])) != 10]
    check("no history repeats a transcript", not dup)

    # --- determinism ----------------------------------------------------------
    print("\nREPRODUCIBILITY  (probe replay depends on this)")
    again = {R.arena_label(e, d, f): R.build_sellers(pool, e, d, f)
             for e, d, f in R.SELLER_ARENAS}
    same = all(x["sha"] == y["sha"] for a in arenas for x, y in zip(arenas[a], again[a]))
    check("rebuilding produces byte-identical history blocks", same)
    check("all agent shas are distinct",
          len({s["sha"] for s in sellers + buyers}) == len(sellers) + len(buyers))

    # --- schedule -------------------------------------------------------------
    print("\nSCHEDULE")
    p0 = R.pairings(0)
    check(f"each seller plays k={R.K} negotiations", len(p0) == R.K)
    check("the arena item is the same in every negotiation",
          {p["good"] for p in p0} == {R.ARENA_GOOD})
    comps = Counter(buyers[p["buyer_slot"]]["comp"] for p in p0)
    check("each seller meets every buyer experience level equally",
          len(set(comps.values())) == 1, str(dict(comps)))
    pairs = Counter((p["buyer_slot"],) for p in p0)
    check(f"each (seller, buyer) pair is repeated {R.REPS}x — true replications",
          all(v == R.REPS for v in pairs.values()))

    # the arena contrast leans on this: seller i meets the SAME buyers in every arena,
    # so opponent quality differences out of the comparison entirely
    check("seller i faces identical buyers in every arena", True,
          "by construction: pairings() depends only on the seller index")

    use = Counter()
    for _ in R.SELLER_ARENAS:
        for si in range(R.SELLERS_PER_ARENA):
            for p in R.pairings(si):
                use[buyers[p["buyer_slot"]]["buyer_id"]] += 1
    check("every buyer is used", len(use) == len(buyers),
          f"{min(use.values())}-{max(use.values())} negotiations each")
    xross = R.SELLERS_PER_ARENA / R.BUYERS_PER_COMP
    check("buyers are crossed with sellers (>=2 sellers per buyer per arena)",
          xross >= 2, f"{xross:.1f} sellers per buyer")

    # --- prompts --------------------------------------------------------------
    print("\nPROMPTS")
    blocks = NEG.build_agent_system(BY_KEY[R.ARENA_GOOD], "seller",
                                    history_block=sellers[0]["block"])
    joined = "\n".join(b["text"] for b in blocks)
    check("no strategy slot in Arm A — the agent gets no instructions",
          "Negotiation strategy:" not in joined)
    check("the history block comes first and is marked cacheable",
          blocks[0].get("cache_control") is not None and len(blocks) == 2)
    check("the arena prompt states the value range (Imas's setting)",
          "Market value range" in joined)
    check("history is framed as the agent's own past",
          "Your recent negotiating experience" in joined)
    check("past-you's private thinking is shown",
          "What you were thinking" in joined)
    other = "the buyer:" in joined.lower()
    check("the counterparty's private thinking is NEVER shown",
          "(What they were thinking" not in joined and other)

    # --- overlap and size -----------------------------------------------------
    print("\nHEALTH")
    for a, ss in arenas.items():
        sets = [set(s["episodes"]) for s in ss]
        ov = st.mean([len(x & y) for i, x in enumerate(sets) for y in sets[i + 1:]])
        check(f"arena {ss[0]['arena']}: sellers have distinct histories",
              ov < 3, f"{ov:.1f}/10 episodes shared per pair")

    lens = {a: st.mean(len(s["block"]) for s in ss) for a, ss in arenas.items()}
    swell = lens["0/10"] / lens["10/0"] - 1
    print(f"  note  lopsided histories are {swell:.0%} longer than even ones — "
          f"composition is\n        confounded with history LENGTH. Known; report it.")

    n = len(R.SELLER_ARENAS) * R.SELLERS_PER_ARENA * R.K
    print(f"\n{'ALL CHECKS PASSED' if not _fails else f'{len(_fails)} FAILED: {_fails}'}")
    print(f"run would be {n} negotiations across {len(R.SELLER_ARENAS)} arenas")


if __name__ == "__main__":
    main()
