"""Arm B, step 1: compile a lived history into a written playbook.

Arm A  :  history  ->  negotiate
Arm B  :  history  ->  WRITE A PLAYBOOK  ->  a fresh agent negotiates from the playbook alone

The A-B gap is the measurement. Whatever the history did for Arm A that the playbook does
not do for Arm B is experience that could not be put into words -- the tacit-knowledge tax.
That gap is load-bearing for the headline claim, so the compile step has to be a fair
representation of what Imas's participants actually did, not a strawman.

What Imas's participants were told (paper §2.1, verbatim fragments):
  - write instructions for "an AI agent that would negotiate on their behalf"
  - the agent plays "a round-robin tournament against other participants' AI agents"
  - the prompt "should allow their AI agent to perform well in a variety of different
    situations" -- a "strategic playbook" the agent "would utilize for varying values of
    the negotiation parameters"
  - separate prompts for buying and selling
  - the submitted prompt is embedded in a system prompt that separately supplies the value
    range, the outside option, the round count, and the role

Three consequences for the writer prompt below:
  1. It is told it is writing FOR ANOTHER AGENT. His participants knew that; an agent
     writing a note to itself would compress differently.
  2. It writes GENERICALLY -- no specific car, no specific range. His participants wrote
     one playbook covering "varying values of the negotiation parameters." Handing our
     writer the good it is about to face would let it smuggle a number across, and the
     A-B gap would then be measuring number-passing rather than strategy compression.
  3. Length is SOFT-GUIDED, NOT CAPPED. The measured loss is a function of the compression
     budget, so the budget must be reported rather than silently chosen. His two printed
     examples run 26 and 40 words; the distribution over his 299 is unknown. We match his
     instruction (which sets no length) and RECORD every playbook's length, then compare
     long against short observationally. A hard-capped dose-response (25 / 100 / 400 words)
     is the natural follow-up, not the first pass.

⚠️ The writer never sees the counterparty it will face, and never sees an Arm A outcome.
Otherwise the playbook is written with hindsight Arm A did not have.
"""

from __future__ import annotations

import asyncio
from typing import Dict, List, Optional

from anthropic import AsyncAnthropic
from pydantic import BaseModel, Field

MODEL = "claude-sonnet-5"
MAX_TOKENS = 8192
THINKING = {"type": "adaptive"}
EFFORT = "low"

# Imas's structural facts, minus the good. His participants received "extensive
# instructions on both the bargaining setting" before writing, so withholding the protocol
# would handicap our writer relative to his.
_SETTING = """You have been negotiating over used vehicles, equipment and similar items. Each negotiation works the same way:

- Two sides, a buyer and a seller, negotiate over one item.
- Both sides are told a market value range for the item.
- Each side privately has an outside option: the seller can sell elsewhere at the bottom of the range, the buyer can buy elsewhere at the top of it. A deal is only worth doing between those two numbers.
- Your payoff is the difference between the agreed price and your own outside option. No deal pays nothing.
- The negotiation runs up to 12 rounds, alternating, and the buyer speaks first.
- The item, the range and the outside options are DIFFERENT every time."""

_TASK = """You are now going to hand this work over.

Another agent will negotiate in your place from here on. It will be told the item, the market value range, its outside option, the round limit and which side it is on — all of that is supplied to it separately, so you do not need to repeat it and you do not know which item it will get.

What it will NOT have is your experience. It has never negotiated before and it will never see the negotiations above.

Write it a strategic playbook: the instructions it should follow to negotiate well on your behalf. It must work for varying values of the negotiation parameters — different items, different ranges, different outside options — so write it to apply generally rather than to one particular deal.

Write only the playbook itself, addressed to that agent. Do not explain your reasoning or refer to the negotiations above."""


class Playbook(BaseModel):
    text: str = Field(
        description=(
            "The playbook, addressed to the agent that will negotiate. Instructions only "
            "— no preamble, no commentary, no reference to past negotiations."
        )
    )


def build_writer_prompt(role: str, history_block: str) -> Dict[str, str]:
    """Returns {"system": ..., "user": ...} for the compile call.

    role: the chair the agent occupied and will hand over ("seller" | "buyer").
    history_block: output of history.render().
    """
    if role not in ("seller", "buyer"):
        raise ValueError(role)
    other = "buyer" if role == "seller" else "seller"
    system = (
        f"You are an experienced {role}'s negotiating agent. You have handled a number of "
        f"these negotiations against {other}s of various kinds.\n\n" + _SETTING
    )
    user = history_block + "\n\n---\n\n" + _TASK
    return {"system": system, "user": user}


async def write_playbook(
    client: AsyncAnthropic, role: str, history_block: str, attempts: int = 4
) -> Dict[str, object]:
    """Compile one history into a playbook.

    Retries on its own: structured output intermittently comes back empty on these long
    (14-20k token) prompts, and the SDK's transport-level retries do not cover a response
    that arrives successfully but parses to nothing. One such failure previously killed a
    whole run, so failure is handled here rather than left to the caller.
    """
    p = build_writer_prompt(role, history_block)
    last = None
    for i in range(attempts):
        try:
            r = await client.messages.parse(
                model=MODEL, max_tokens=MAX_TOKENS,
                system=p["system"], messages=[{"role": "user", "content": p["user"]}],
                thinking=THINKING, output_config={"effort": EFFORT},
                output_format=Playbook,
            )
            if r.parsed_output and r.parsed_output.text.strip():
                break
            last = "empty parsed_output"
        except Exception as exc:
            last = f"{type(exc).__name__}: {str(exc)[:120]}"
        await asyncio.sleep(1.5 * (i + 1))
    else:
        raise RuntimeError(f"playbook failed after {attempts} attempts: {last}")
    text = r.parsed_output.text.strip()
    return {
        "playbook": text,
        "playbook_words": len(text.split()),
        "playbook_chars": len(text),
        "usage": {"input_tokens": r.usage.input_tokens,
                  "output_tokens": r.usage.output_tokens},
    }
