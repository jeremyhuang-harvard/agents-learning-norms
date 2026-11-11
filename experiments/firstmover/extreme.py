"""Maximally seller-adverse: shark BUYER vs pushover SELLER, on Haiku without thinking.
If a zero-surplus close is reachable anywhere, it is here."""
import asyncio, json, os, statistics as st, sys
from pathlib import Path
from anthropic import AsyncAnthropic
from experiments.pool.briefs import BRIEFS, build_system
from experiments.pool.generate import ACCEPT_KEYWORDS, MAX_ROUNDS, _check, _parse, load_dotenv
from experiments.pool.goods import GOODS

KICK = "Let's begin the negotiation. Please make your initial offer and inquiries."
KW = {"model": "claude-haiku-4-5", "max_tokens": 2048}   # no thinking, no effort
OUT = Path("experiments/firstmover/data/extreme_haiku.jsonl")

async def one(c, i, good):
    sysm = {"buyer":  build_system(good, "buyer",  BRIEFS["shark"]),
            "seller": build_system(good, "seller", BRIEFS["pushover"])}
    msgs = {"buyer": [{"role":"user","content":KICK}], "seller": []}
    tr, closed = [], None
    for turn in range(MAX_ROUNDS):
        sp = "buyer" if turn % 2 == 0 else "seller"
        li = "seller" if sp == "buyer" else "buyer"
        r = await c.messages.create(system=sysm[sp], messages=msgs[sp], **KW)
        tx = "".join(b.text for b in r.content if b.type == "text").strip()
        msgs[sp].append({"role":"assistant","content":tx}); msgs[li].append({"role":"user","content":tx})
        tr.append({"round":turn+1,"role":sp,"text":tx})
        if turn >= 1 and any(k in tx.lower() for k in ACCEPT_KEYWORDS):
            ck = await _check(c, tr)
            if ck.agreement_reached and ck.transaction_price is not None:
                closed = turn+1; break
    o = await _parse(c, tr)
    p = o.transaction_price if o.agreement_reached else None
    return {"idx":i,"good":good.key,"range":[good.low,good.high],"transaction_price":p,
            "sellshare":(p-good.low)/good.surplus if p is not None else None,
            "rounds_used":len(tr),"agreement_round":o.agreement_round,"transcript":tr}

async def main():
    load_dotenv(); c = AsyncAnthropic(max_retries=5); sem = asyncio.Semaphore(10); lock = asyncio.Lock()
    combos = [g for g in GOODS for _ in range(5)]
    print(f"{len(combos)} negotiations — shark BUYER vs pushover SELLER, Haiku 4.5 no thinking")
    async def go(i,g):
        async with sem:
            try: rec = await one(c,i,g)
            except Exception as e: rec = {"idx":i,"good":g.key,"error":type(e).__name__}
            async with lock:
                with OUT.open("a") as f: f.write(json.dumps(rec)+"\n"); f.flush(); os.fsync(f.fileno())
    await asyncio.gather(*(go(i,g) for i,g in enumerate(combos)))
    rs=[json.loads(l) for l in open(OUT) if l.strip()]
    d=[r for r in rs if r.get("sellshare") is not None]
    v=[r["sellshare"] for r in d]
    print(f"\nn={len(rs)}  deals={len(d)}  no-deal={len(rs)-len(d)}")
    print(f"mean {st.mean(v):.3f}  min {min(v):.3f}  max {max(v):.3f}")
    print(f"at exactly 0: {sum(1 for x in v if x<=0.001)}   below 0.1: {sum(1 for x in v if x<0.1)}")
    print(f"\nlowest 6:")
    for r in sorted(d,key=lambda x:x['sellshare'])[:6]:
        lo,_=r['range']
        print(f"  {r['sellshare']:.3f}  ${r['transaction_price']:,} vs fallback ${lo:,} "
              f"(+${r['transaction_price']-lo:,})  r{r['agreement_round']}")
