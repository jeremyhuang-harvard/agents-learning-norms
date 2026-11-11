"""Can a zero-surplus close be produced AT ALL?

349 negotiations across every configuration have never ended at the seller's fallback,
while Imas reports 12.5% there. Two further conditions, both on Haiku 4.5 without
thinking (the weakest setup, most likely to produce it):

  "realistic"  -- an impatient seller who dislikes the hassle of the outside option,
                  against a buyer making a credible take-it-or-leave-it. No instruction
                  to accept any particular number; the conditions merely make a thin
                  deal plausible.
  "ceiling"    -- near-explicit. Tells the seller to close on essentially anything
                  workable. NOT a scientific condition -- a capability probe. If a zero
                  cannot be produced even here, something structural is preventing it.

Note the "realistic" seller brief adds friction to the outside option ("a drive and
paperwork") that is not in the payoff structure. That is a framing manipulation, not a
payoff change, and should be reported as such.
"""
import asyncio, json, os, statistics as st
from pathlib import Path
from anthropic import AsyncAnthropic
from experiments.pool.briefs import build_system
from experiments.pool.generate import ACCEPT_KEYWORDS, MAX_ROUNDS, _check, _parse, load_dotenv
from experiments.pool.goods import GOODS

KICK = "Let's begin the negotiation. Please make your initial offer and inquiries."
KW = {"model": "claude-haiku-4-5", "max_tokens": 2048}
OUT = Path("experiments/firstmover/data/extreme2.jsonl")

CONDS = {
    "realistic": {
        "buyer": ("You have decided what this is worth to you and you are not moving off it. "
                  "Name your number early, stay polite, and repeat it. You have other options "
                  "and you are willing to walk away rather than pay more."),
        "seller": ("You have been trying to sell this for months and you are tired of it. You "
                   "want it gone today. The dealer alternative means a drive across town and "
                   "paperwork you would rather avoid. If they hold firm, you would rather take "
                   "a thin deal than start over with someone new."),
    },
    "ceiling": {
        "buyer": ("Open at the very bottom of the range and do not move from it under any "
                  "circumstances, however they respond."),
        "seller": ("Close this today. If their offer is workable at all, take it rather than "
                   "counter. Do not let the negotiation run long."),
    },
}


async def one(c, i, good, cond):
    b = CONDS[cond]
    sysm = {"buyer": build_system(good, "buyer", b["buyer"]),
            "seller": build_system(good, "seller", b["seller"])}
    msgs = {"buyer": [{"role": "user", "content": KICK}], "seller": []}
    tr = []
    for turn in range(MAX_ROUNDS):
        sp = "buyer" if turn % 2 == 0 else "seller"
        li = "seller" if sp == "buyer" else "buyer"
        r = await c.messages.create(system=sysm[sp], messages=msgs[sp], **KW)
        tx = "".join(x.text for x in r.content if x.type == "text").strip()
        msgs[sp].append({"role": "assistant", "content": tx})
        msgs[li].append({"role": "user", "content": tx})
        tr.append({"round": turn + 1, "role": sp, "text": tx})
        if turn >= 1 and any(k in tx.lower() for k in ACCEPT_KEYWORDS):
            ck = await _check(c, tr)
            if ck.agreement_reached and ck.transaction_price is not None:
                break
    o = await _parse(c, tr)
    p = o.transaction_price if o.agreement_reached else None
    return {"idx": i, "cond": cond, "good": good.key, "range": [good.low, good.high],
            "transaction_price": p,
            "sellshare": (p - good.low) / good.surplus if p is not None else None,
            "rounds_used": len(tr), "agreement_round": o.agreement_round, "transcript": tr}


async def main():
    load_dotenv(); c = AsyncAnthropic(max_retries=5)
    sem, lock = asyncio.Semaphore(10), asyncio.Lock()
    combos = [(g, cd) for cd in CONDS for g in GOODS for _ in range(3)]
    print(f"{len(combos)} negotiations, Haiku 4.5 no thinking")

    async def go(i, gc):
        g, cd = gc
        async with sem:
            try: rec = await one(c, i, g, cd)
            except Exception as e: rec = {"idx": i, "cond": cd, "error": type(e).__name__}
            async with lock:
                with OUT.open("a") as f:
                    f.write(json.dumps(rec) + "\n"); f.flush(); os.fsync(f.fileno())
    await asyncio.gather(*(go(i, gc) for i, gc in enumerate(combos)))

    rs = [json.loads(l) for l in open(OUT) if l.strip()]
    for cd in CONDS:
        d = [r for r in rs if r.get("cond") == cd and r.get("sellshare") is not None]
        n = [r for r in rs if r.get("cond") == cd and "error" not in r and r.get("sellshare") is None]
        v = [r["sellshare"] for r in d]
        if not v: continue
        print(f"\n{cd.upper()}   deals={len(d)}  no-deal={len(n)}")
        print(f"  mean {st.mean(v):.3f}   min {min(v):.3f}   max {max(v):.3f}")
        print(f"  at exactly 0: {sum(1 for x in v if x<=0.001)}   below 0.05: {sum(1 for x in v if x<0.05)}"
              f"   below 0.15: {sum(1 for x in v if x<0.15)}")
        for r in sorted(d, key=lambda x: x["sellshare"])[:4]:
            lo, _ = r["range"]
            print(f"    {r['sellshare']:.3f}  ${r['transaction_price']:,} vs fallback ${lo:,} "
                  f"(+${r['transaction_price']-lo:,})  closed r{r['agreement_round']}")
