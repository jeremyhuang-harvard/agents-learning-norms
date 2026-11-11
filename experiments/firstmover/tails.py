"""Can the support reach the HIGH tail? (mirror of the capitulation test)

Matching Imas's human distribution needs sd ~0.21. Our conditional distributions top out
around 0.13 in a mixture because we have almost no tail mass: 1.1% of outcomes below 0.2,
0.2% above 0.8.

The LOW tail is reachable -- a seller told to close on anything hits exactly 0.000, and a
shark buyer vs pushover seller reaches 0.175. This tests the mirror: a BUYER told to close
on anything, against a firm seller. If sellshare goes near 1.0, the channel is symmetric
and the support can be widened at both ends. If it does not, specification failure is
one-directional -- which is the tacit-knowledge claim, and a finding in its own right.
"""
import asyncio, json, os, statistics as st
from pathlib import Path
from anthropic import AsyncAnthropic
from experiments.pool.briefs import build_system
from experiments.pool.generate import ACCEPT_KEYWORDS, MAX_ROUNDS, _check, _parse, load_dotenv
from experiments.pool.goods import GOODS

KICK = "Let's begin the negotiation. Please make your initial offer and inquiries."
KW = {"model": "claude-haiku-4-5", "max_tokens": 2048}
OUT = Path("experiments/firstmover/data/tails.jsonl")

# exact mirror of the "ceiling" condition that produced 14/16 zeros
CONDS = {
    "buyer_capitulates": {
        "buyer":  ("Close this today. If their price is workable at all, take it rather than "
                   "counter. Do not let the negotiation run long."),
        "seller": ("Open at the very top of the range and do not move from it under any "
                   "circumstances, however they respond."),
    },
}

async def one(c, i, good, cond):
    b = CONDS[cond]
    sysm = {r: build_system(good, r, b[r]) for r in ("buyer", "seller")}
    msgs = {"buyer": [{"role": "user", "content": KICK}], "seller": []}
    tr = []
    for turn in range(MAX_ROUNDS):
        sp = "buyer" if turn % 2 == 0 else "seller"
        li = "seller" if sp == "buyer" else "buyer"
        r = await c.messages.create(system=sysm[sp], messages=msgs[sp], **KW)
        tx = "".join(x.text for x in r.content if x.type == "text").strip()
        msgs[sp].append({"role": "assistant", "content": tx})
        msgs[li].append({"role": "user", "content": tx})
        tr.append({"round": turn+1, "role": sp, "text": tx})
        if turn >= 1 and any(k in tx.lower() for k in ACCEPT_KEYWORDS):
            ck = await _check(c, tr)
            if ck.agreement_reached and ck.transaction_price is not None: break
    o = await _parse(c, tr)
    p = o.transaction_price if o.agreement_reached else None
    return {"idx": i, "cond": cond, "good": good.key, "range": [good.low, good.high],
            "transaction_price": p,
            "sellshare": (p-good.low)/good.surplus if p is not None else None,
            "rounds_used": len(tr), "agreement_round": o.agreement_round, "transcript": tr}

async def main():
    load_dotenv(); c = AsyncAnthropic(max_retries=5)
    sem, lock = asyncio.Semaphore(10), asyncio.Lock()
    combos = [(g, cd) for cd in CONDS for g in GOODS for _ in range(3)]
    print(f"{len(combos)} negotiations — BUYER told to capitulate, firm seller")
    async def go(i, gc):
        g, cd = gc
        async with sem:
            try: rec = await one(c, i, g, cd)
            except Exception as e: rec = {"idx": i, "cond": cd, "error": type(e).__name__}
            async with lock:
                with OUT.open("a") as f: f.write(json.dumps(rec)+"\n"); f.flush(); os.fsync(f.fileno())
    await asyncio.gather(*(go(i, gc) for i, gc in enumerate(combos)))
    rs=[json.loads(l) for l in open(OUT) if l.strip()]
    d=[r for r in rs if r.get("sellshare") is not None]
    v=[r["sellshare"] for r in d]
    print(f"\nn={len(rs)}  deals={len(d)}  no-deal={len(rs)-len(d)}")
    print(f"mean {st.mean(v):.3f}  min {min(v):.3f}  max {max(v):.3f}")
    print(f"at exactly 1.0: {sum(1 for x in v if x>=0.999)}   above 0.95: {sum(1 for x in v if x>0.95)}"
          f"   above 0.8: {sum(1 for x in v if x>0.8)}")
    print("\nhighest 6:")
    for r in sorted(d, key=lambda x:-x["sellshare"])[:6]:
        lo,hi=r["range"]
        print(f"  {r['sellshare']:.3f}  ${r['transaction_price']:,} vs buyer's ceiling ${hi:,} "
              f"(-${hi-r['transaction_price']:,})  r{r['agreement_round']}")
