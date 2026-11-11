"""The negotiation engine shared by both arms.

Identical to pool generation in every respect except where the two sides' system prompts
come from. Keeping one engine is deliberate: if Arm A and Arm B ran different loops, any
A-B gap could be the loop rather than the compression.

Agent construction, by arm and role:

    Arm A seller   template(good, "seller", brief=None)   + 10-episode history
    Arm B seller   template(good, "seller", brief=PLAYBOOK)   and nothing else
    buyer, both    template(good, "buyer",  brief=None)   + 10-episode history

The Arm B playbook goes into the SAME slot a brief goes into -- Imas's
`{buyerPrompt}`/`{sellerPrompt}`. That is exactly his design: the participant's submitted
text sits inside a system prompt that separately supplies range, outside option, rounds and
role. So Arm B *is* Imas's architecture, with the playbook author replaced by an agent that
has lived ten negotiations instead of a person who has lived a life.

⚠️ ORDERING AND CACHING. The history block is placed BEFORE the template so that the
system prompt has a stable cacheable prefix across the k negotiations one agent plays (the
good changes; the history does not). At ~48k tokens of history per agent, without caching
this is the entire cost of the experiment. The cost of the choice is that the assignment
("You are my negotiation agent for purchasing X") arrives after the experience rather than
before it. Flag it; it applies identically to both arms and to every condition, so it
cannot generate an A-B gap.
"""

from __future__ import annotations

from typing import Dict, List, Optional

from anthropic import AsyncAnthropic

from experiments.pool.briefs import KICKOFF, build_system
from experiments.pool.generate import ACCEPT_KEYWORDS, MAX_ROUNDS, _check, _parse
from experiments.pool.goods import Good

MODEL = "claude-sonnet-5"
MAX_TOKENS = 2048
THINKING = {"type": "adaptive"}
EFFORT = "low"

_HANDOFF = (
    "---\n\n"
    "## Your current assignment\n\n"
)


def build_agent_system(
    good: Good,
    role: str,
    history_block: Optional[str] = None,
    playbook: Optional[str] = None,
    state_range: bool = True,
) -> List[Dict]:
    """One side's system prompt, as cache-aware blocks.

    Exactly one of history_block / playbook should normally be set:
      history_block -> Arm A (or an Arm A buyer inside Arm B)
      playbook      -> Arm B seller
      neither       -> unbriefed control
    """
    # state_range=False drops the posted market value range from the ARENA only. The
    # histories keep theirs -- they are past negotiations that did have a book value. That
    # asymmetry is the experiment: an agent that has done ten deals where a reference point
    # existed now faces one where it does not. If it behaves as though a reference point is
    # still there, EXPERIENCE IS THE REFERENCE POINT.
    #
    # It also changes what the focal point means. With the range posted, $20,000 is visible
    # arithmetic -- the midpoint of a number both sides can see. Without it neither agent
    # can compute it: the seller knows only its own $18,000, the buyer only its own
    # $22,000, and the template tells the buyer not to disclose. Landing on 50-50 stops
    # being arithmetic and becomes an inference.
    template = build_system(good, role, playbook, state_range)
    if history_block is None:
        return [{"type": "text", "text": template}]
    return [
        # cacheable prefix: constant across the k negotiations this agent plays
        {"type": "text", "text": history_block,
         "cache_control": {"type": "ephemeral"}},
        {"type": "text", "text": _HANDOFF + template},
    ]


async def negotiate(
    client: AsyncAnthropic,
    good: Good,
    seller_system: List[Dict],
    buyer_system: List[Dict],
) -> Dict:
    """Run one negotiation. Buyer opens, per Imas's protocol."""
    sysm = {"buyer": buyer_system, "seller": seller_system}
    msgs: Dict[str, List[Dict]] = {
        "buyer": [{"role": "user", "content": KICKOFF}], "seller": []}
    transcript: List[Dict] = []
    usage = {"input_tokens": 0, "output_tokens": 0,
             "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0}
    closed_at: Optional[int] = None

    for turn in range(MAX_ROUNDS):
        sp = "buyer" if turn % 2 == 0 else "seller"
        li = "seller" if sp == "buyer" else "buyer"
        r = await client.messages.create(
            model=MODEL, max_tokens=MAX_TOKENS, system=sysm[sp], messages=msgs[sp],
            thinking=THINKING, output_config={"effort": EFFORT},
        )
        for k in usage:
            usage[k] += getattr(r.usage, k, 0) or 0
        text = "".join(b.text for b in r.content if b.type == "text").strip()
        # Keep the reasoning. It is NOT fed back into either side's message list -- the
        # negotiation is bit-for-bit what it would be without this -- but it is the only
        # record of what the agent was attending to at the moment it made the move, and
        # no later replay recovers it (a replay draws NEW reasoning, not the reasoning
        # that produced the move we observed).
        thinking = "".join(b.thinking for b in r.content if b.type == "thinking").strip()
        msgs[sp].append({"role": "assistant", "content": text})
        msgs[li].append({"role": "user", "content": text})
        entry = {"round": turn + 1, "role": sp, "text": text}
        if thinking:
            entry["thinking"] = thinking
        transcript.append(entry)

        if turn >= 1 and any(k in text.lower() for k in ACCEPT_KEYWORDS):
            chk = await _check(client, transcript)
            if chk.agreement_reached and chk.transaction_price is not None:
                closed_at = turn + 1
                break

    out = await _parse(client, transcript)
    price = out.transaction_price if out.agreement_reached else None
    return {
        "transaction_price": price,
        "sellshare": (price - good.low) / good.surplus if price is not None else None,
        "agreement_round": out.agreement_round,
        "rounds_used": len(transcript),
        "closed_early_at": closed_at,
        "parser_evidence": out.evidence,
        "usage": usage,
        "transcript": transcript,
    }
