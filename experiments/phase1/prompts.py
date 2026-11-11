"""Prompt templates from Imas, Lee & Misra (2025), "Agentic Interactions", Appendix A.

Transcribed verbatim from papers/agentic interactions.pdf. The only edits are
normalising the PDF's typographic minus (U+2212) to ASCII hyphen and its curly
apostrophes to straight ones.

The {buyer_prompt} / {seller_prompt} slots are where participant-authored
"strategic playbooks" go. In the original study each of 299 participants wrote
their own pair; Phase 1 holds a single pair fixed so the only thing varying
across runs is the model's own sampling.
"""

BUYER_TEMPLATE = """You are my negotiation agent for purchasing a used 2020 Toyota Camry LE. Your goal is to maximize my surplus (difference between my fallback price and final price).

Negotiation structure:
- This negotiation will last up to 12 rounds (6 offers from each side)
- You must reach agreement within these 12 rounds or the deal fails
- The negotiation starts with the buyer's first message (round 1)
- Your final message should clearly state acceptance or rejection
- If you're close to agreement in the final round, consider accepting to avoid failure

Negotiation parameters (common knowledge):
- Market value range is $18,000 to $22,000 depending on condition
- Car history: No accidents reported

Private information (DO NOT DISCLOSE DIRECTLY):
- Fallback option: I can buy a similar Camry from a dealer for $22,000

Negotiation strategy:
{buyer_prompt}"""

SELLER_TEMPLATE = """You are my negotiation agent for selling my used 2020 Toyota Camry LE. Your goal is to maximize my surplus (difference between final price and my fallback price).

Negotiation structure:
- This negotiation will last up to 12 rounds (6 offers from each side)
- You must reach agreement within these 12 rounds or the deal fails
- The negotiation starts with the buyer's first message (round 1)
- Your final message should clearly state acceptance or rejection
- If you're close to agreement in the final round, consider accepting to avoid failure

Negotiation parameters (common knowledge):
- Market value range is $18,000 to $22,000 depending on condition
- Car history: No accidents reported

Private information (DISCLOSE STRATEGICALLY):
- Fallback option: I can sell to a dealer at $18,000

Negotiation strategy:
{seller_prompt}"""

# Verbatim opener the authors used to start every negotiation.
KICKOFF = "Let's begin the negotiation. Please make your initial offer and inquiries."

# --- strategy slots -----------------------------------------------------------
#
# Two named pairs. Pick with --strategy on the runner.
#
# "imas" — the two prompts printed in Appendix A. The paper introduces these as
# "Here are examples of a buyer prompt and a seller prompt", immediately followed
# by an illustrative transcript. They are a snapshot of two participant
# submissions, NOT a matched or representative pair, and they are lopsided on all
# three dimensions a negotiation strategy has:
#
#              anchor                  concession      objective reference
#   buyer      below the *range*       a few hundred   "as low as possible"
#   seller     above the *midpoint*    $500-$1000      "above the dealer price"
#
# The buyer anchors further out AND concedes 2-5x more slowly, which structurally
# advantages the buyer. Any mean sellshare produced by this pair is a property of
# the mismatch, not of the model.

IMAS_BUYER_PROMPT = (
    "Offer a deal a few thousand under the KBB range and go up by a few hundred "
    "dollars as needed with the aim of closing as low as possible."
)

IMAS_SELLER_PROMPT = (
    "Start with an offer that's a few thousand dollars above the midpoint of the "
    "blue book value range. Go down by $500-$1000 each round if necessary. Aim to "
    "close a deal as far above the dealer price as possible"
)

# "symmetric" — a deliberate mirror, matched on anchor distance, concession rate,
# and objective phrasing. Note "the dealer price" resolves correctly for both
# sides: it is $22,000 in the buyer's private info and $18,000 in the seller's,
# i.e. each side's own fallback. With this pair the prompts impose no directional
# advantage, so any deviation of the outcome distribution from 0.50 isolates
# something real — a model-level bias, or a structural feature of the game such
# as the buyer opening while the seller speaks last.

SYMMETRIC_BUYER_PROMPT = (
    "Open a few thousand dollars below the blue book value range. Concede "
    "$500-$1000 each round if necessary. Aim to close a deal as far below the "
    "dealer price as possible."
)

SYMMETRIC_SELLER_PROMPT = (
    "Open a few thousand dollars above the blue book value range. Concede "
    "$500-$1000 each round if necessary. Aim to close a deal as far above the "
    "dealer price as possible."
)

STRATEGIES = {
    "symmetric": (SYMMETRIC_BUYER_PROMPT, SYMMETRIC_SELLER_PROMPT),
    "imas": (IMAS_BUYER_PROMPT, IMAS_SELLER_PROMPT),
}

# Game parameters (Imas et al. §2.1)
SELLER_FALLBACK = 18_000
BUYER_FALLBACK = 22_000
TOTAL_SURPLUS = BUYER_FALLBACK - SELLER_FALLBACK  # $4,000
MAX_ROUNDS = 12  # 6 messages per side
