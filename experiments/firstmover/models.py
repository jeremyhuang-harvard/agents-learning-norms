"""Model x reasoning 2x2 for the first-mover experiment.

Imas et al. used gpt-4.1-mini, which differs from our Sonnet 5 baseline on TWO axes at
once -- it is a small model AND it does no reasoning. Swapping models alone would change
both, so this crosses them.

    | thinking ON            | thinking OFF
Sonnet 5  | baseline               | isolates REASONING
Haiku 4.5 | isolates MODEL SIZE    | closest analogue to gpt-4.1-mini

If the zero-surplus spike appears only bottom-right it needs both. If it appears across
the whole thinking-off column, reasoning is doing the work -- the more interesting result,
because a seller accepting its own outside option is exactly the error a moment's
reflection prevents.

⚠️ The cells are NOT perfectly symmetric, and this must be stated as a limitation:
Haiku 4.5 has an older parameter surface -- it rejects `effort` and uses the legacy
`{type: "enabled", budget_tokens: N}` thinking config rather than adaptive. So "thinking
on" means something different for Haiku than for Sonnet. Haiku also accepts `temperature`
(Sonnet 5 rejects it); we leave it unset so both run at the default 1.0.

The transcript PARSER stays on Sonnet 5 in every cell, so measurement does not move --
only the negotiating agents change.
"""

from typing import Dict, Optional

CONFIGS: Dict[str, Dict] = {
    "sonnet-think": {
        "model": "claude-sonnet-5",
        "max_tokens": 2048,
        "thinking": {"type": "adaptive"},
        "effort": "low",
        "label": "Sonnet 5 · thinking",
    },
    "sonnet-nothink": {
        "model": "claude-sonnet-5",
        "max_tokens": 2048,
        "thinking": {"type": "disabled"},
        "effort": "low",
        "label": "Sonnet 5 · no thinking",
    },
    "haiku-think": {
        # legacy config: budget_tokens must be < max_tokens, minimum 1024
        "model": "claude-haiku-4-5",
        "max_tokens": 3072,
        "thinking": {"type": "enabled", "budget_tokens": 1024},
        "effort": None,          # Haiku 4.5 rejects `effort`
        "label": "Haiku 4.5 · thinking",
    },
    "haiku-nothink": {
        # omitting `thinking` on Haiku 4.5 means no thinking (unlike Sonnet 5, where
        # omitting it runs adaptive)
        "model": "claude-haiku-4-5",
        "max_tokens": 2048,
        "thinking": None,
        "effort": None,
        "label": "Haiku 4.5 · no thinking  ← closest to gpt-4.1-mini",
    },
}

# $/MTok (input, output). Sonnet 5 intro pricing through 2026-08-31.
PRICES = {"claude-sonnet-5": (2.0, 10.0), "claude-haiku-4-5": (1.0, 5.0)}


def call_kwargs(cfg: Dict) -> Dict:
    """Build the model-call kwargs for a config, omitting what the model rejects."""
    kw = {"model": cfg["model"], "max_tokens": cfg["max_tokens"]}
    if cfg["thinking"] is not None:
        kw["thinking"] = cfg["thinking"]
    if cfg["effort"] is not None:
        kw["output_config"] = {"effort": cfg["effort"]}
    return kw
