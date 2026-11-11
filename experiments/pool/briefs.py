"""Counterparty briefs and the generalized negotiation template.

A brief is the text that fills Imas's `{buyerPrompt}` / `{sellerPrompt}` slot. In pool
generation exactly ONE side is briefed; the other side -- "past-you", whose history this
transcript will become -- gets no strategy at all.

Why only the counterparty is briefed: a history in which the OTHER party was fair-minded
tells the agent something about the world it operates in ("the people you meet are often
reasonable"). A history in which YOU were fair-minded is a self-instruction dressed as a
story. The theory is about what you assume regarding the other side, so that is where the
disposition belongs.

Three families, one brief each. Sub-variants were considered and dropped: no planned test
requires route variation, variety already comes from six goods plus sampling, and more
within-family spread is noise against the mixture test.
"""

from experiments.pool.goods import Good

# --- the three briefs --------------------------------------------------------
#
# FAIR is deliberately ASYMMETRIC -- firm against doing badly, relaxed about doing well.
# Symmetric fairness (refusing a generous deal) is the implausible assumption, and it
# contradicts Fehr-Schmidt inequity aversion, where resistance to disadvantageous
# inequality far exceeds resistance to advantageous.
#
# It is also framed as a NORMATIVE BELIEF, not a reservation value. "You won't settle for
# less than half the range" merely sets a number -- every negotiator has a reservation
# value; a shark's is just further out. That phrasing fails to separate FAIR from SHARK,
# and it produces transcripts about bargaining positions rather than about what's
# appropriate. Only the latter installs a fair-division category, and the language is the
# mechanism.
#
# Note "won't do much worse than that" rather than "hold out for at least the middle":
# the latter is correct for a seller and backwards for a buyer.

BRIEFS = {
    "fair": (
        "You think a deal near the middle of the range is about right, and you won't do "
        "much worse than that. You're not trying to squeeze more out of them, but you "
        "won't refuse it if it's offered."
    ),
    "shark": (
        "You want as much of the range as you can get. Open aggressively, concede in very "
        "small steps, and don't be the one who closes the remaining gap."
    ),
    # Rewritten 2026-08-10 after the first pool batch. The original read "Open close to
    # your own fallback and move toward the other side when pushed" -- and was simply
    # ignored. Opening at your own fallback means offering your worst case as an OPENING
    # move, which no negotiator does and the model would not do either: pushover buyers
    # opened at 0.00-0.21 of the range, indistinguishable from fair buyers.
    #
    # A pushover is not someone who opens badly. It is someone who CAVES. Specify the
    # concession behaviour, not the anchor.
    # v3 2026-08-10. v2 ("make a large move toward their number") still failed to produce
    # wins for past-you-as-seller: a pushover BUYER opens low like everyone else, so
    # conceding a lot only gets back to the middle. First-mover anchoring dominates. v3
    # therefore specifies a generous ANCHOR in terms the model will actually follow --
    # "clearly attractive to them" rather than "at your own fallback", which reads as an
    # absurd opening move and was ignored.
    "pushover": (
        "You want this done quickly and you're not going to fight over the last stretch. "
        "Open at a number that is clearly attractive to them, and close as soon as it's "
        "anywhere near reasonable."
    ),
}


# --- escalation briefs (2026-08-12) ------------------------------------------
#
# TWO directions, and they are NOT symmetric in difficulty.
#
# hard_pushover exists because of the binding constraint in the whole pool: "past-you sold
# and CAME OUT AHEAD" has n=2. The buyer moves first, so the buyer's opening sets the
# anchor, and an unbriefed seller has never cleared ~0.60 of the surplus. The even band is
# 0.40-0.60, so the seller ceiling sits exactly on the boundary -- it reaches even and
# essentially never crosses into winning. Without this cell the DIRECTION CONTRAST cannot
# be run, and that contrast is discriminating: ten memories of being crushed vs ten of
# winning big: payoff learning says the crushed agent softens, categorisation says both
# land in the same competitive category. Opposite predictions.
#
# ⚠️ The brief must NOT let the buyer pay above the top of the range -- that is the buyer's
# own outside option, so paying more is strictly irrational and the transcript becomes a
# memory of beating an incompetent, not of winning a negotiation.
#
# How far can a ONE-SIDED brief push the outcome? The standard shark leaves an unbriefed
# seller at 0.287 on average, worst case 0.150. Reaching the 0.025 floor is only known to
# be possible by briefing BOTH sides -- which costs clean attribution, because past-you's
# disposition then risks becoming a proxy for the treatment.
#
# So: before paying that price, find out what the one-sided floor actually is. If a harder
# buyer reaches ~0.10, the remaining gap to 0.025 may not be worth giving up the blank
# chair. If it doesn't, the case for double-briefing is settled empirically.
#
# The levers are chosen from what we have already MEASURED about this model, not from
# generic "be more aggressive" language:
#   - The anchor is nearly spent. Buyers already open at 1.03 of the range (below the
#     stated floor), so telling them to open lower adds little.
#   - The model will not make a move that is not a recognisable move -- the lesson from the
#     broken v1 pushover brief. So every instruction here names a move a negotiator makes.
#   - What is NOT spent: concession *pattern* and a credible walk-away. Those are where the
#     remaining room is.
#
# Deliberately stops short of instructing failure. A no-deal is unusable as a memory (it
# bins as no_deal and is discarded), so pushing to the point of collapse buys nothing.
BRIEFS["hard_pushover"] = (
    "You need this settled today and the price matters far less to you than getting it "
    "done. Open near the top of the range — a number they will find hard to refuse — and "
    "say you would rather pay a fair price than spend a week haggling. If they counter, "
    "accept it straight away unless it is above the top of the range. Do not talk them "
    "down, do not ask them to justify their number, and do not make a second counter."
)

# v2, after hard_pushover overshot. v1 said "accept straight away, do not make a second
# counter" -- which reached 0.939 but in THREE rounds: buyer opens near the top, seller
# counters, buyer accepts. That is not a memory of winning a negotiation, it is a memory of
# being handed money, and it would confound the direction contrast with whether a
# negotiation happened at all (hard_shark runs 8-10 rounds; hard_pushover v1 ran 3).
#
# The fix is not a weaker brief -- standard pushover only reaches 0.514, because the buyer
# opens first and anchors low, so conceding a lot merely returns to the middle. It needs a
# DECENT OPENING plus REAL BACK-AND-FORTH: open mid-range, engage, then concede more than
# the seller does, repeatedly. Target ~0.70-0.80 over 6-9 rounds.
BRIEFS["conceder"] = (
    "You want this deal and you have room to pay, but you don't want to look like a pushover. "
    "Open around the middle of the range and ask the normal questions about condition. "
    "After that, every time they hold their number, move most of the way toward it rather "
    "than splitting the difference — bigger steps than they are making. Keep going until "
    "you are close to their figure. Never walk away over the last stretch, but do make them "
    "take you through several rounds before you get there."
)

BRIEFS["hard_shark"] = (
    "Your outside option is genuinely good and you are willing to use it. Anchor low and "
    "justify it with the item's flaws. Concede in very small steps, never twice in a row, "
    "and let long stretches pass with no movement at all. Say plainly, more than once, "
    "that you will walk rather than pay near the top of the range. Never split the "
    "difference and never be the one who closes the final gap — make them come to you."
)

# --- three conceder variants (2026-08-12) ------------------------------------
#
# conceder reached 0.568 (best 0.700) over 5-7 rounds -- a real negotiation, but not a
# clear seller win. hard_pushover reached 0.939 but in 3 rounds, which is not a negotiation
# at all. The gap between them is where a usable "past-you sold and CAME OUT AHEAD" memory
# has to live.
#
# The binding constraint is the OPENING. The buyer speaks first, so its opening number sets
# the ceiling; a yielding buyer still opens like a buyer, and conceding a lot from a low
# anchor only walks back to the middle. Each variant below attacks a different part of that.
#
# All three also DROP the walk-away threat. Every brief so far let the buyer invoke its
# outside option, which is the seller's main reason to come down. Removing it costs nothing
# in realism -- plenty of buyers never mention alternatives -- and should raise the floor.

# A. Move the anchor up without abolishing the haggle.
BRIEFS["high_anchor"] = (
    "You want this deal and you can afford it. Open in the upper part of the range — a "
    "number that shows you are serious rather than a lowball — and ask the normal questions "
    "about condition. When they push back, keep improving your offer in steady steps rather "
    "than holding firm. Do not mention your outside option and never threaten to walk away."
)

# B. Refuse to anchor at all, and let the SELLER set the number. The buyer still speaks
# first, so Imas's protocol is untouched -- it just opens with a question instead of a
# figure, which is a strategy a real participant might well write.
BRIEFS["invite"] = (
    "Do not name a price first. Open by asking what they are looking to get for it, along "
    "with your questions about condition. Once they give you a number, push back once or "
    "twice on the price, but move up substantially each time and settle close to their "
    "asking figure. Do not mention your outside option and never threaten to walk away."
)

# C. Keep the low anchor but make holding firm strictly profitable for the seller: the
# buyer may never repeat a number, so waiting always pays.
BRIEFS["never_holds"] = (
    "You want this settled and you are willing to pay for it. Open around the middle of the "
    "range. From then on, never repeat the same number twice — every time they hold their "
    "position, come back with a better offer, moving more than halfway to their figure. "
    "Take four or five offers to get there. Do not mention your outside option and never "
    "threaten to walk away."
)

# D. A + B failed / worked respectively; C worked best. high_anchor did NOTHING (0.520,
# 0/20, max exactly 0.600 -- the same ceiling standard pushover hits), which says raising
# the opening is not the lever: the model converges back toward a fair split from whatever
# anchor it is given, unless something makes holding firm PAY. never_holds is that
# something. This stacks it with invite, the other lever that moved (0.589), since the two
# act on different things -- who anchors, and whether waiting is profitable.
BRIEFS["invite_never_holds"] = (
    "Do not name a price first. Open by asking what they are looking to get for it, along "
    "with your questions about condition. Once they name a number, push back once or twice "
    "— but never repeat the same offer twice: every time they hold their position, come "
    "back with a better one, moving more than halfway to their figure. Take four or five "
    "offers to get there and settle close to their asking price. Do not mention your "
    "outside option and never threaten to walk away."
)

# --- hillclimbing on never_holds (2026-08-12) --------------------------------
#
# never_holds (0.606 mean, 45% clearing 0.60, max 0.738) is the only lever that moved the
# seller meaningfully, and it works by MECHANISM rather than disposition: forbid the buyer
# from repeating a number and holding firm becomes strictly profitable for a seller who was
# told nothing. These three push on different parts of that mechanism.

# Make patience pay MORE the longer you wait, rather than at a constant rate.
BRIEFS["ratchet_accel"] = (
    "You want this settled and you are willing to pay for it. Open around the middle of the "
    "range. Never repeat the same number twice: every time they hold their position, come "
    "back with a better offer — and make each raise larger than your last one. Keep going "
    "until they accept. Do not mention your outside option and never threaten to walk away."
)

# More ratchet steps. Opening LOW is counter-intuitive but gives the climb more room, and it
# is also what a buyer naturally does, so it should read as more ordinary than opening high.
BRIEFS["ratchet_long"] = (
    "You want this settled and you are willing to pay for it. Open low, near the bottom of "
    "the range. Never repeat the same number twice: every time they hold their position, "
    "come back with a better offer. Do not try to close quickly — work up through six or "
    "seven offers and let them set the pace. Do not mention your outside option and never "
    "threaten to walk away."
)

# Neutralise the template's own deadline pressure. Imas's structure block tells BOTH sides
# "if you're close to agreement in the final round, consider accepting to avoid failure",
# which pushes the buyer to close early and cuts the ratchet short.
BRIEFS["ratchet_patient"] = (
    "You want this settled and you are willing to pay for it, and you have plenty of time — "
    "there is no penalty for using all twelve rounds. Open around the middle of the range. "
    "Never repeat the same number twice: every time they hold their position, come back with "
    "a better offer, moving more than halfway to their figure. Do not mention your outside "
    "option and never threaten to walk away."
)

# --- reverse-engineered from the transcripts (2026-08-12) ---------------------
#
# ⭐ THE OUTCOME IS THE MIDPOINT OF THE TWO OPENINGS. Across seven briefs the final split
# tracks (buyer's opening + seller's ask)/2 almost exactly:
#     ratchet_long .02/.77 -> mid .395, actual .373 | never_holds .35/.86 -> mid .605, actual .608
#     high_anchor  .22/.79 -> mid .505, actual .520 | hard_pushover .74/.97 -> mid .855, actual .939
# Every concession-behaviour instruction was near-irrelevant. What each one really did was
# move where the buyer OPENED, and the split followed.
#
# That explains the paradox: high_anchor TOLD the buyer to open high and it opened at .22;
# never_holds said "around the middle" and it opened at .35. Forbidding repeats makes a low
# opening costly, so the model anchors higher of its own accord. The ratchet never worked as
# a ratchet -- it worked as an incentive to anchor.
#
# And high_anchor failed for the same reason pushover v1 failed: a buyer opening near the top
# is not a recognisable move, so the model quietly declines. hard_pushover reached .74 only
# because urgency made a high opening make sense.
#
# So the brief needs BOTH: a justified high opening, AND an explicit instruction to keep
# haggling afterwards (which is what hard_pushover lacked, closing in 3 rounds).
# Target: buyer opens ~.60-.65 against a seller asking ~.85 => final ~.73-.75 over 5-7 rounds.

BRIEFS["urgent_haggler"] = (
    "You need this done this week and you would rather pay a bit over the odds than lose it "
    "and start again. Open with a strong offer — well above where a haggler would start — "
    "and tell them plainly that you are opening high to save both of you time. But do not "
    "just take their counter: ask your questions about condition, push back on the price "
    "twice, and get them to come down a little before you agree. Never repeat the same "
    "number twice. Do not mention your outside option and never threaten to walk away."
)

BRIEFS["strong_open_two_counters"] = (
    "Open with a genuinely strong offer — the kind you make when you want to be taken "
    "seriously immediately rather than trade lowballs — and say that is why you have opened "
    "there. Then negotiate properly: make exactly two more offers, each better than the "
    "last, before you accept anything. Do not accept their first counter no matter how "
    "reasonable it looks. Do not mention your outside option and never threaten to walk away."
)

# Round 2 of reverse-engineering. Both round-1 attempts FAILED, informatively:
#   urgent_haggler            buyer opened .29 -> final .509
#   strong_open_two_counters  buyer opened .06 -> final .431   <- LOWEST opening of any brief
# "Open with a genuinely strong offer" backfired completely: to a BUYER, a strong offer is a
# TOUGH one. And urgent_haggler opened low despite "well above where a haggler would start"
# -- adding "push back twice" pulled it straight back into haggler behaviour.
#
# ⭐ The model treats "open high" and "negotiate hard" as a PACKAGE. Every brief asking for
# genuine bargaining got a low opening; the only one that opened high (hard_pushover, .74)
# also told the buyer to stop negotiating. Not independent dials.
#
# Two ways left: state the opening as an explicit FRACTION rather than an adjective, or keep
# hard_pushover's opening and simply forbid an early close.

BRIEFS["explicit_open"] = (
    "Open at a number about two thirds of the way up the market range — so if the range were "
    "$10,000 to $14,000 you would open near $12,700. Say you are opening there because you "
    "want a serious conversation rather than a week of lowball offers. Then negotiate "
    "normally: ask about condition, push back once on their counter, and settle somewhere in "
    "the upper part of the range. Do not mention your outside option and never threaten to "
    "walk away."
)

BRIEFS["slow_close"] = (
    "You need this settled soon and the price matters far less to you than getting it done. "
    "Open near the top of the range — a number they will find hard to refuse. But do not "
    "agree to anything before the fifth round: until then keep asking about condition, "
    "history and what is included, and make at least two offers of your own. After that, "
    "accept their number unless it is above the top of the range."
)

# --- seller-side escalation (2026-08-12) -------------------------------------
#
# "Past-you bought and got beaten" is the one thin cell. Only ONE seller-side brief was ever
# tried (hard_shark_sell, a word-for-word mirror of the buyer version) against seven on the
# buyer side, and it hit 6/20. So the cell is under-attempted, not proven unreachable.
#
# Why it is genuinely harder: the buyer opens first and opens LOW, and the outcome lands
# near the midpoint of the two openings. An aggressive seller already asks ~0.86 of the
# range, so anchoring harder is nearly spent -- (0.25 + 0.90)/2 still lands near even. The
# seller cannot out-anchor a first mover. It has to win by being IMMOVABLE instead, forcing
# the buyer to close the whole gap.
#
# Note the mirror of never_holds would be exactly WRONG here: forbidding the seller to
# repeat a number makes the BUYER profit from waiting. Immovability is the mirror, not the
# ratchet.
#
# Neither brief instructs deception. Inventing a rival bidder would work but it is a lie,
# and Imas's template says DISCLOSE STRATEGICALLY, not fabricate.

BRIEFS["firm_price"] = (
    "You have priced this deliberately and you do not haggle. State your number at the "
    "outset, say plainly that it is not a negotiating position, and repeat that whenever "
    "they push. Answer every question about condition fully and stay friendly and easy to "
    "deal with — you are not being difficult, you simply have one price. Do not lower it at "
    "any point. If they want it, they will come to you."
)

BRIEFS["patient_seller"] = (
    "You are in no hurry at all and you are content to keep the item if this does not work "
    "out. Open near the top of the range. Concede at most once, in a very small step, and "
    "not until late. Do not fill silences with new offers, do not split the difference, and "
    "do not be the one who closes any gap. Let them do the moving."
)

# ⚠️ hard_shark above is phrased for a BUYER ("anchor low", "the item's flaws", "walk
# rather than PAY near the top"). The three core families are deliberately role-neutral;
# this one is not. Handing it to a seller produced literally zero effect (0.495 vs 0.495
# for the standard shark, 0/20 pushing the buyer into lopsided) -- the model will not
# follow an instruction that names no move it can make. Below is the exact mirror.
BRIEFS["hard_shark_sell"] = (
    "Your outside option is genuinely good and you are willing to use it. Anchor high and "
    "justify it with the item's strengths. Concede in very small steps, never twice in a "
    "row, and let long stretches pass with no movement at all. Say plainly, more than once, "
    "that you will walk rather than sell near the bottom of the range. Never split the "
    "difference and never be the one who closes the final gap — make them come to you."
)

# The three core families. Kept explicit rather than derived from BRIEFS so that adding an
# escalation brief does not silently change what generate.py, interior.py and topup.py run.
FAMILIES = ["fair", "shark", "pushover"]

# --- the template ------------------------------------------------------------
#
# Imas Appendix A, verbatim, with the good parameterised. The disclosure asymmetry
# (seller "DISCLOSE STRATEGICALLY" / buyer "DO NOT DISCLOSE DIRECTLY") is his and is kept
# -- past negotiations must look like the game the agent is about to play. It is measured
# rather than manipulated.

_STRUCTURE = """Negotiation structure:
- This negotiation will last up to 12 rounds (6 offers from each side)
- You must reach agreement within these 12 rounds or the deal fails
- The negotiation starts with the buyer's first message (round 1)
- Your final message should clearly state acceptance or rejection
- If you're close to agreement in the final round, consider accepting to avoid failure"""


def build_system(good: Good, role: str, brief: str | None,
                 state_range: bool = True) -> str:
    """Assemble one side's system prompt.

    role: "buyer" | "seller"
    brief: the strategy text, or None for the unbriefed side (past-you).

    When brief is None the entire "Negotiation strategy:" header is omitted -- a dangling
    header with nothing under it reads as an omission the model may try to fill.
    """
    if role == "seller":
        opening = (
            f"You are my negotiation agent for selling {good.description}. Your goal is to "
            f"maximize my surplus (difference between final price and my fallback price)."
        )
        disclosure = "DISCLOSE STRATEGICALLY"
        fallback = f"I can sell to a dealer at ${good.low:,}"
    elif role == "buyer":
        opening = (
            f"You are my negotiation agent for purchasing {good.description}. Your goal is "
            f"to maximize my surplus (difference between my fallback price and final price)."
        )
        disclosure = "DO NOT DISCLOSE DIRECTLY"
        fallback = f"I can buy a comparable one elsewhere for ${good.high:,}"
    else:
        raise ValueError(f"role must be 'buyer' or 'seller', got {role!r}")

    # state_range=False drops the common-knowledge value range. Imas states it, and it is
    # what manufactures the midpoint: both sides can see a number that is obviously "fair",
    # and every measurement we have says they converge on it (buyers open at 1.03 of the
    # range, sellers at 0.93, and the final split tracks the midpoint of the two openings).
    # Without it each side knows only its own walk-away, which is ordinary bargaining -- and
    # it turns "is the 50-50 focal point a fairness norm or an artifact of showing everyone
    # a range?" into a measurable question rather than an assumption.
    params = [f"- {good.condition_note}"]
    if state_range:
        params.insert(0, f"- Market value range is ${good.low:,} to ${good.high:,} "
                         f"depending on condition")
    parts = [
        opening,
        "",
        _STRUCTURE,
        "",
        "Negotiation parameters (common knowledge):",
        *params,
        "",
        f"Private information ({disclosure}):",
        f"- Fallback option: {fallback}",
    ]
    if brief:
        parts += ["", "Negotiation strategy:", brief]
    return "\n".join(parts)


KICKOFF = "Let's begin the negotiation. Please make your initial offer and inquiries."


# --- past-you tactics --------------------------------------------------------
#
# Added 2026-08-10. Unbriefed past-you plays like a competent moderate and never exceeds
# ~0.60 of the surplus as a seller, so "past-you won big" was unreachable and the
# direction contrast could not be run.
#
# The fix is NOT to give past-you a disposition -- ten transcripts of past-you being
# fair-minded reads as "you are a fair person", a self-instruction, which is exactly what
# briefing only the counterparty was meant to avoid.
#
# Instead give past-you a TACTIC, drawn at RANDOM and INDEPENDENTLY of the counterparty's
# disposition. A disposition repeated ten times is a character; a tactic that varies
# across the ten is just how those negotiations happened to go -- which is also how real
# negotiating histories look. The systematic variation stays on the counterparty, where
# the theory wants it; past-you's play becomes noise rather than signal.

TACTICS = {
    "firm":     "Open near the best end of the range for you and concede in small steps.",
    "moderate": "Open somewhat in your favour and concede at a steady pace.",
    "yielding": "Open close to the middle of the range and move readily when pushed.",
}
