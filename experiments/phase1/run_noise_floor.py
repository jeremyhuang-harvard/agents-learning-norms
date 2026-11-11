"""Phase 1 — the noise floor.

Runs the Imas et al. used-car negotiation N times with a *fixed* buyer prompt and
a *fixed* seller prompt. Nothing varies between runs except the model's own
sampling. The resulting spread in transaction prices is therefore pure noise, not
heterogeneity — and it's the zero mark on the ruler for every later phase, which
all take the form "we varied experience histories and got spread X."

Usage (from the repo root, using the project venv):
    .venv/bin/python experiments/phase1/run_noise_floor.py --show-setup   # inspect, no API calls
    .venv/bin/python experiments/phase1/run_noise_floor.py --check        # verify credentials
    .venv/bin/python experiments/phase1/run_noise_floor.py --n 300
    .venv/bin/python experiments/phase1/run_noise_floor.py --n 300 --out data/run_a.jsonl  # resumes
    .venv/bin/python experiments/phase1/run_noise_floor.py --analyze data/run_a.jsonl
    .venv/bin/python experiments/phase1/plot.py data/run_a.jsonl

Records are appended and fsynced as each negotiation completes, so an interrupted
run loses only the negotiations still in flight. Re-running with the same --out
skips run_ids already present.
"""

import argparse
import asyncio
import json
import os
import statistics
import time
from pathlib import Path
from typing import Dict, List, Optional

from anthropic import AsyncAnthropic
from pydantic import BaseModel, Field

from prompts import (
    BUYER_TEMPLATE,
    SELLER_TEMPLATE,
    KICKOFF,
    STRATEGIES,
    SELLER_FALLBACK,
    TOTAL_SURPLUS,
    MAX_ROUNDS,
)

# --- configuration -----------------------------------------------------------

MODEL = "claude-opus-5"
PARSER_MODEL = "claude-opus-5"
# max_tokens caps thinking + visible text together. 1024 was tight enough that a
# long reasoning pass could silently truncate the negotiation message itself.
MAX_TOKENS = 2048
CONCURRENCY = 20

# Adaptive thinking at low effort rather than thinking disabled. Disabling
# thinking on Opus 5 can leak <thinking> tags into the visible response, which
# would contaminate transcripts we then parse as negotiation messages. Low effort
# keeps cost and latency near the disabled case without that failure mode.
THINKING: Dict[str, str] = {"type": "adaptive"}
EFFORT = "low"

DATA_DIR = Path(__file__).parent / "data"
# Default to the paper's own Appendix A pair, word for word. It is lopsided
# (see prompts.py) and the resulting MEAN is therefore uninterpretable — it
# reflects the prompt mismatch, not the model. That is acceptable because Phase
# 1's deliverable is the SPREAD, not the centre. Fidelity to real participant
# text beats a cleaner instrument we would have had to write ourselves.
STRATEGY = "imas"  # overridden by --strategy


def load_dotenv() -> str:
    """Load .env and report which credential is in play.

    The project .env deliberately *overrides* an ambient shell variable: an
    exported key left over from another project would otherwise silently win and
    bill the wrong account. Returns a masked identifier so the caller can eyeball
    which key is being used without ever printing it.
    """
    env_path = Path(__file__).resolve().parents[2] / ".env"
    source = "shell environment"

    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key, value = key.strip(), value.strip().strip("'\"")
            if key == "ANTHROPIC_API_KEY" and value:
                source = str(env_path)
            os.environ[key] = value

    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        raise SystemExit(
            "No ANTHROPIC_API_KEY found. Add it to .env at the repo root "
            "(already gitignored) or export it in your shell."
        )
    return f"...{api_key[-4:]} (len {len(api_key)}, from {source})"


# --- transcript parsing ------------------------------------------------------


class AgreementCheck(BaseModel):
    """Cheap mid-negotiation check: has a deal actually closed?"""

    agreement_reached: bool = Field(
        description=(
            "True only if one party has made an offer AND the other has explicitly "
            "accepted it. A proposal, a counter-offer, or a statement of intent to "
            "accept is not agreement."
        )
    )
    transaction_price: Optional[int] = Field(
        default=None, description="Agreed price in whole dollars, if agreement was reached."
    )


class NegotiationOutcome(BaseModel):
    """Structured read of a finished transcript.

    Imas et al. also parsed transcripts with a second LLM; using a schema-
    constrained call makes that step deterministic to read rather than regex-y.
    """

    agreement_reached: bool = Field(
        description="True only if both parties clearly agreed on a transaction price."
    )
    transaction_price: Optional[int] = Field(
        default=None,
        description=(
            "The price in whole dollars at the FIRST point both parties clearly agreed. "
            "Ignore any later renegotiation. Null if no agreement was reached."
        ),
    )
    agreement_round: Optional[int] = Field(
        default=None,
        description="Round number at which agreement was first reached. Null if none.",
    )
    evidence: str = Field(
        description="Short quote from the transcript supporting the determination."
    )


# --- the negotiation ---------------------------------------------------------


async def run_one_negotiation(
    client: AsyncAnthropic,
    run_id: int,
    buyer_prompt: str,
    seller_prompt: str,
    strategy: str,
) -> Dict:
    """Alternate buyer/seller messages for MAX_ROUNDS turns, starting with the buyer.

    Each side keeps its own message list where its own turns are 'assistant' and
    the counterparty's are 'user' — neither agent ever sees the other's system
    prompt or private fallback.
    """
    buyer_system = BUYER_TEMPLATE.format(buyer_prompt=buyer_prompt)
    seller_system = SELLER_TEMPLATE.format(seller_prompt=seller_prompt)

    buyer_msgs: List[Dict] = [{"role": "user", "content": KICKOFF}]
    seller_msgs: List[Dict] = []
    transcript: List[Dict] = []
    usage = {"input_tokens": 0, "output_tokens": 0}
    stop_reasons: List[Dict] = []
    closed_at: Optional[int] = None

    for turn in range(MAX_ROUNDS):
        is_buyer = turn % 2 == 0
        system = buyer_system if is_buyer else seller_system
        msgs = buyer_msgs if is_buyer else seller_msgs

        response = await client.messages.create(
            model=MODEL,
            max_tokens=MAX_TOKENS,
            system=system,
            messages=msgs,
            thinking=THINKING,
            output_config={"effort": EFFORT},
        )

        usage["input_tokens"] += response.usage.input_tokens
        usage["output_tokens"] += response.usage.output_tokens

        # stop_reason is how truncation and refusals surface. A "max_tokens" here
        # means the visible message was cut off mid-sentence, which the transcript
        # alone would not make obvious.
        if response.stop_reason != "end_turn":
            stop_reasons.append({"round": turn + 1, "stop_reason": response.stop_reason})

        text = "".join(b.text for b in response.content if b.type == "text").strip()

        # Feed this turn back to the speaker as their own, and to the other side
        # as the counterparty's message.
        msgs.append({"role": "assistant", "content": text})
        other = seller_msgs if is_buyer else buyer_msgs
        other.append({"role": "user", "content": text})

        transcript.append(
            {"round": turn + 1, "role": "buyer" if is_buyer else "seller", "text": text}
        )

        # Stop as soon as a deal closes, as Imas et al. did. Left running, agents
        # drift after agreement — writing deal summaries, and in observed cases
        # losing track of which side they represent — which wastes tokens and
        # pollutes the transcript. The keyword prefilter keeps this nearly free:
        # the confirming model call only fires on turns that look like a close.
        if turn >= 1 and any(k in text.lower() for k in ACCEPTANCE_KEYWORDS):
            check = await check_agreement(client, transcript)
            if check.agreement_reached and check.transaction_price is not None:
                closed_at = turn + 1
                break

    outcome = await parse_transcript(client, transcript)

    price = outcome.transaction_price if outcome.agreement_reached else None
    sellshare = (price - SELLER_FALLBACK) / TOTAL_SURPLUS if price is not None else None

    return {
        "run_id": run_id,
        "model": MODEL,
        "thinking": THINKING,
        "effort": EFFORT,
        "strategy_pair": strategy,
        "buyer_prompt": buyer_prompt,
        "seller_prompt": seller_prompt,
        "agreement_reached": outcome.agreement_reached,
        "transaction_price": price,
        "agreement_round": outcome.agreement_round,
        "rounds_used": len(transcript),
        "closed_early_at": closed_at,
        "sellshare": sellshare,
        "parser_evidence": outcome.evidence,
        "anomalous_stop_reasons": stop_reasons,
        "usage": usage,
        "transcript": transcript,
    }


# Cheap prefilter. Broad on purpose: a false positive costs one small model call
# that returns agreement_reached=False, while a false negative would let the
# negotiation run past its close.
ACCEPTANCE_KEYWORDS = (
    "accept",
    "we have a deal",
    "we've got a deal",
    "deal at",
    "agreed at",
    "sold",
    "done at",
    "i'll take it",
)


async def check_agreement(client: AsyncAnthropic, transcript: List[Dict]) -> AgreementCheck:
    """Confirm whether the last turn actually closed the deal."""
    tail = transcript[-3:]
    rendered = "\n\n".join("Round {round} ({role}): {text}".format(**t) for t in tail)
    response = await client.messages.parse(
        model=PARSER_MODEL,
        max_tokens=1024,
        messages=[
            {
                "role": "user",
                "content": (
                    "These are the most recent turns of a used-car negotiation. Has a "
                    "deal been definitively closed — one side offered a price and the "
                    "other explicitly accepted it?\n\n" + rendered
                ),
            }
        ],
        output_format=AgreementCheck,
    )
    return response.parsed_output


async def parse_transcript(
    client: AsyncAnthropic, transcript: List[Dict]
) -> NegotiationOutcome:
    rendered = "\n\n".join(
        "Round {round} ({role}): {text}".format(**t) for t in transcript
    )
    response = await client.messages.parse(
        model=PARSER_MODEL,
        max_tokens=2048,
        messages=[
            {
                "role": "user",
                "content": (
                    "Below is a transcript of a used-car negotiation between a buyer "
                    "agent and a seller agent. Determine whether the two parties "
                    "reached a clear agreement on a transaction price, and if so, what "
                    "that price was. An agreement requires explicit acceptance — an "
                    "offer that was never accepted is not an agreement.\n\n"
                    "The transcript always runs the full 12 rounds even when the parties "
                    "agree earlier, so it may contain conversation after a deal is "
                    "struck. Report the price at the FIRST clear agreement and the round "
                    "it occurred in; ignore any later renegotiation.\n\n"
                    f"{rendered}"
                ),
            }
        ],
        output_format=NegotiationOutcome,
    )
    return response.parsed_output


# --- driver ------------------------------------------------------------------


def completed_run_ids(path: Path) -> set:
    """run_ids already present in an output file, so a resumed run skips them."""
    if not path.exists():
        return set()
    ids = set()
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue  # a partial final line from an interrupted write
        if "error" not in record:
            ids.add(record["run_id"])
    return ids


async def main_async(n: int, out_path: Path, strategy: str) -> None:
    key_id = load_dotenv()
    client = AsyncAnthropic(max_retries=5)
    semaphore = asyncio.Semaphore(CONCURRENCY)
    write_lock = asyncio.Lock()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    already = completed_run_ids(out_path)
    todo = [i for i in range(n) if i not in already]

    print(f"Using API key {key_id}")
    print(f"Output: {out_path}")
    if already:
        print(f"Resuming: {len(already)} already complete, {len(todo)} to run.")

    done = 0
    spend = 0.0
    started = time.time()

    async def guarded(run_id: int) -> None:
        nonlocal done, spend
        async with semaphore:
            try:
                record = await run_one_negotiation(
                    client, run_id, *STRATEGIES[strategy], strategy=strategy
                )
            except Exception as exc:  # keep one bad run from killing the batch
                record = {"run_id": run_id, "error": f"{type(exc).__name__}: {exc}"}

            # Append immediately. An interrupted run then costs only the
            # negotiations still in flight, not everything spent so far.
            async with write_lock:
                with out_path.open("a") as fh:
                    fh.write(json.dumps(record) + "\n")
                    fh.flush()
                    os.fsync(fh.fileno())

            done += 1
            if "usage" in record:
                spend += (
                    record["usage"]["input_tokens"] * 5 / 1e6
                    + record["usage"]["output_tokens"] * 25 / 1e6
                )
            elapsed = time.time() - started
            detail = (
                "ERR " + record["error"][:40]
                if "error" in record
                else f"${record['transaction_price']:,}"
                if record["transaction_price"]
                else "no deal"
            )
            print(
                f"  [{done}/{len(todo)}] {elapsed:6.0f}s  ~${spend:5.2f}  {detail}",
                flush=True,
            )

    print(f"Running {len(todo)} negotiations on {MODEL}, strategy={strategy!r} (concurrency {CONCURRENCY})...")
    await asyncio.gather(*(guarded(i) for i in todo))

    print(f"\nDone. ~${spend:.2f} this session. Records in {out_path}")
    analyze(out_path)


# --- analysis ----------------------------------------------------------------

# Published comparison points from Imas et al. §3 and §3.4. "Agentic" is their
# 299 human-authored prompts on gpt-4.1-mini; "human" is their 304 direct
# negotiators. We are neither — we are a single fixed prompt on Claude, so the
# expectation is a much tighter distribution than both.
IMAS_AGENTIC = {
    "mean": 0.35,
    "median": 0.375,
    "p25": 0.19,
    "p75": 0.50,
    "zero_mass": 0.1249,
    "no_deal": 0.03,
    "focal_mass": 0.143,
    "share_below_half": 0.80,
}
IMAS_HUMAN_FOCAL_MASS = 0.347


def analyze(path: Path) -> None:
    records = [json.loads(line) for line in path.read_text().splitlines() if line]
    errors = [r for r in records if "error" in r]
    valid = [r for r in records if "error" not in r]
    deals = [r for r in valid if r["sellshare"] is not None]
    shares = [r["sellshare"] for r in deals]

    if not shares:
        print("No completed deals to analyze.")
        return

    # "Landed on 50-50" needs a tolerance: a $20,000 deal is exactly 0.5, but so
    # is anything a hair either side once agents round to hundreds.
    focal = sum(1 for s in shares if abs(s - 0.5) < 0.01) / len(shares)
    zero_mass = sum(1 for s in shares if s <= 0.0) / len(shares)
    quantiles = statistics.quantiles(shares, n=4) if len(shares) >= 4 else [float("nan")] * 3

    print("\n" + "=" * 66)
    print(f"PHASE 1 — NOISE FLOOR   n={len(valid)} valid, {len(errors)} errors")
    print("=" * 66)
    print(f"{'':22}{'ours':>10}{'Imas agentic':>15}{'Imas human':>13}")
    print("-" * 66)
    rows = [
        ("mean sellshare", statistics.mean(shares), IMAS_AGENTIC["mean"], None),
        ("median", statistics.median(shares), IMAS_AGENTIC["median"], None),
        ("p25", quantiles[0], IMAS_AGENTIC["p25"], None),
        ("p75", quantiles[2], IMAS_AGENTIC["p75"], None),
        ("std dev", statistics.pstdev(shares), None, None),
        ("50-50 focal mass", focal, IMAS_AGENTIC["focal_mass"], IMAS_HUMAN_FOCAL_MASS),
        ("zero-surplus mass", zero_mass, IMAS_AGENTIC["zero_mass"], None),
        ("no-deal rate", 1 - len(deals) / len(valid), IMAS_AGENTIC["no_deal"], None),
        (
            "share below 0.5",
            sum(1 for s in shares if s < 0.5) / len(shares),
            IMAS_AGENTIC["share_below_half"],
            None,
        ),
    ]
    for label, ours, agentic, human in rows:
        a = f"{agentic:>15.3f}" if agentic is not None else f"{'—':>15}"
        h = f"{human:>13.3f}" if human is not None else f"{'—':>13}"
        print(f"{label:22}{ours:>10.3f}{a}{h}")

    print("-" * 66)
    print(
        "Variance is the headline number: it is the spread produced by sampling\n"
        "alone. Every later phase's dispersion should be read against it."
    )
    tokens = sum(r["usage"]["input_tokens"] + r["usage"]["output_tokens"] for r in valid)
    print(f"\nTotal tokens across {len(valid)} negotiations: {tokens:,}")


def show_setup() -> None:
    """Dump the exact strings and parameters the run will use. No API calls."""
    rule = "=" * 78

    print(f"\n{rule}\nMODEL CALL PARAMETERS\n{rule}")
    print(f"  model            {MODEL}")
    print(f"  max_tokens       {MAX_TOKENS}")
    print(f"  thinking         {THINKING}")
    print(f"  output_config    {{'effort': '{EFFORT}'}}")
    print(f"  concurrency      {CONCURRENCY} negotiations in flight")
    print("  temperature      (not sent — rejected by Opus 5; Imas used 1.0 on gpt-4.1-mini)")

    for name, (bp, sp) in sorted(STRATEGIES.items()):
        marker = "  <-- DEFAULT" if name == STRATEGY else ""
        print(f"\n{rule}\nSTRATEGY PAIR: {name!r}{marker}\n{rule}")
        print(f"  buyer slot : {bp}\n")
        print(f"  seller slot: {sp}")

    bp, sp = STRATEGIES[STRATEGY]
    print(f"\n{rule}\nBUYER SYSTEM PROMPT, fully rendered (strategy={STRATEGY!r})\n{rule}")
    print(BUYER_TEMPLATE.format(buyer_prompt=bp))

    print(f"\n{rule}\nSELLER SYSTEM PROMPT, fully rendered (strategy={STRATEGY!r})\n{rule}")
    print(SELLER_TEMPLATE.format(seller_prompt=sp))

    print(f"\n{rule}\nTURN STRUCTURE\n{rule}")
    print(f"  Opening user message to the buyer: {KICKOFF!r}\n")
    print("  Two separate conversations. Each agent sees its own turns as")
    print("  'assistant' and the counterparty's as 'user'. Neither ever sees the")
    print("  other's system prompt or private fallback.\n")
    for turn in range(MAX_ROUNDS):
        who = "buyer " if turn % 2 == 0 else "seller"
        arrow = "->" if turn % 2 == 0 else "<-"
        print(f"    round {turn + 1:>2}  {who} {arrow} counterparty")
    print(f"\n  Fixed at {MAX_ROUNDS} turns. No early exit: even if the agents agree in")
    print("  round 4, the loop runs to 12 and the parser decides what happened.")

    print(f"\n{rule}\nPARSER CALL (one per negotiation, after the transcript is complete)\n{rule}")
    print(f"  model         {PARSER_MODEL}")
    print("  schema        NegotiationOutcome, enforced by output_format:")
    for name, field in NegotiationOutcome.model_fields.items():
        print(f"                  {name}: {field.description}")
    print("\n  Imas et al. also parsed transcripts with a second LLM; a schema-")
    print("  constrained call makes that step deterministic to read.")

    print(f"\n{rule}\nWHAT GETS SAVED\n{rule}")
    print(f"  One JSON object per line, to {DATA_DIR}/noise_floor_<timestamp>.jsonl")
    print("""
    run_id             which of the N runs this is
    model, thinking, effort, buyer_prompt, seller_prompt
                       full provenance — every knob, recorded per row
    agreement_reached  bool, from the parser
    transaction_price  int or null
    sellshare          (price - 18000) / 4000, or null on no deal
    parser_evidence    quote the parser used to justify its call
    usage              input/output token totals for the negotiation
    transcript         all 12 messages, verbatim
    """)
    print("  Nothing is discarded. Full transcripts mean any measure we didn't")
    print("  think of yet can be computed later without re-running.\n")
    print(f"  Economics: seller fallback ${SELLER_FALLBACK:,}, total surplus ${TOTAL_SURPLUS:,}.")
    print("  A $20,000 deal is sellshare 0.50 — the 50-50 focal point.\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=300, help="number of negotiations")
    parser.add_argument(
        "--strategy",
        default=STRATEGY,
        choices=sorted(STRATEGIES),
        help="which slot-filling strategy pair to use (default: symmetric)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        help="output JSONL; if it exists, completed run_ids are skipped (resume)",
    )
    parser.add_argument("--analyze", type=Path, help="analyze an existing JSONL file")
    parser.add_argument(
        "--check", action="store_true", help="verify credentials and exit (no negotiations)"
    )
    parser.add_argument(
        "--show-setup", action="store_true", help="print the exact pipeline and exit (no API calls)"
    )
    args = parser.parse_args()

    if args.analyze:
        analyze(args.analyze)
        return

    if args.show_setup:
        show_setup()
        return

    if args.check:
        print(f"Using API key {load_dotenv()}")
        from anthropic import Anthropic

        # with_raw_response exposes the headers, which carry the organization the
        # key belongs to. That's the thing worth checking before a big run: the
        # key suffix tells you which string you're holding, the org id tells you
        # which account actually gets billed.
        raw = Anthropic().models.with_raw_response.retrieve(MODEL)
        model = raw.parse()
        org = raw.headers.get("anthropic-organization-id", "<not reported>")
        print(f"Organization: {org}")
        print(f"Auth OK. {model.display_name}, context {model.max_input_tokens:,} tokens.")
        return

    out = args.out or DATA_DIR / f"noise_floor_{time.strftime('%Y%m%d-%H%M%S')}.jsonl"
    asyncio.run(main_async(args.n, out, args.strategy))


if __name__ == "__main__":
    main()
