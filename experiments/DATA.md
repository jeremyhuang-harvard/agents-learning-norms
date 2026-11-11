# The data files

Everything is JSON Lines — one object per line, flat, no joins required for the common
analyses. ~42MB total, all tracked in git.

```python
import json
recs = [json.loads(l) for l in open("experiments/arena/data/armA_full.jsonl") if l.strip()]
```

---

## `experiments/pool/data/` — the transcript bank

The library of past negotiations that histories are built from. ~1,350 usable transcripts.

`pool_*.jsonl` is loaded by `experiments/arena/history.load_pool()`; other filenames in this
directory were brief-development runs and are folded in by `pool/promote.py`.

| field | meaning |
|---|---|
| `good`, `range`, `condition_note`, `description` | the item; `range` is `[low, high]` and the endpoints ARE the two outside options |
| `family`, `briefed_role` | which brief was given, and to which side. The OTHER side is "past-you" |
| `past_you_role`, `sub_pool` | which chair was blank. `S` = past-you sold, `B` = past-you bought |
| `transaction_price`, `sellshare` | the deal; `sellshare` = (price − low) / (high − low) |
| `past_you_share` | the same number from past-you's side. This is what the agent "remembers getting" |
| `bin` | `even` (within 0.10 of half) / `favourable` / `unfavourable`, signed from past-you |
| `transcript` | `[{round, role, text, thinking?}]`. `thinking` is past-you's private tactical note, captured off-thread and never fed back |
| `state_range` | whether the posted market range was shown. Absent = true (legacy) |
| `uid`, `promoted_from` | globally unique id; source file if folded in from a development run |

⚠️ A transcript can only be used for the role past-you played — an `S` record can never be a
buyer's memory.

---

## `experiments/arena/data/` — the experiment

### `<run>.jsonl` — one line per negotiation

| field | meaning |
|---|---|
| `arm` | `A` = the seller carries its history. `B` = the seller wrote a playbook and a fresh agent used it |
| `arena` | the seller's memory composition. `10/0` all even · `5/5` half · `0/10` ten losses · `0/10w` ten wins · `10/0f` ten deals at EXACTLY the midpoint |
| `seller_id`, `buyer_id`, `buyer_comp`, `j` | who played, and which of that seller's six negotiations this was |
| `state_range` | whether the posted market range was shown to the negotiators |
| `transaction_price`, `sellshare` | outcome. `sellshare` = 0.5 is the exact even split |
| `agreement_round`, `rounds_used`, `closed_early_at` | when it closed |
| `parser_evidence` | the quote the price was read from — every number is traceable to text |
| `seller_sha`, `buyer_sha` | which history block produced this. **Replay verifies these and refuses on mismatch** |
| `transcript` | `[{round, role, text}]`, complete |
| `usage` | input / output / cache-read / cache-write tokens |

### `<run>.agents.jsonl` — ⚠️ the irreplaceable one

One line per agent: `agent_id`, `role`, `comp`, `sha`, `episodes` (the ten source transcript
ids), and **`block`** — the complete rendered history exactly as the agent saw it.

**This cannot be regenerated.** Histories are drawn from the pool, so a single transcript
added later changes every agent. Delete this and no probe, replay, or provenance claim can
ever be reproduced.

### `<run>.playbooks.jsonl` — Arm B only

`seller_id`, `arena`, `episodes`, `playbook` (the text), `playbook_words`, `playbook_chars`.
Length matters: the measured compression loss is a function of the budget, so it is
recorded rather than assumed. Ours ran 115–802 words; Imas's two printed examples were 26
and 40.

### `<run>.probes.jsonl` — situation and strategy, by replay

Per probe point: `arena`, `seller_id`, `buyer_id`, `j`, `round`, `sellshare`, plus

- `situation`: `{situation, counterparty, expect_next, confidence}` — the ASSUMPTION
- `strategy`: `{plan, walkaway, if_they_hold}` — the GUARANTEE

Asked in **separate calls** on purpose: together, the stated strategy gets rationalised to
fit the stated category.

### `<run>.refprobes.jsonl` — reference points, by replay

Same keys plus `round: 0`, the prior taken *before the negotiation starts*.

- `reference`: `{worth, basis, their_walkaway, confidence}`
- `fairness`: `{fair_price, why, fair_differs_from_good}`
- `strategy`: as above, `null` at round 0
- `worth_share`, `fair_share`, `their_walkaway_share` — pre-computed so plots need no arithmetic
- `range_lo`, `range_hi`, `midpoint`, `seller_fallback`, `buyer_fallback`, `final_price`,
  `final_sellshare`, `turn_text`

---

## `experiments/firstmover/data/` — baselines

`cell_sonnet-think.jsonl` is **the baseline every arena result is measured against**: no
history, no instructions, buyer opens. Mean 0.409, lands exactly on 50-50 8% of the time.

`cell_haiku-*.jsonl` are the same game on Haiku 4.5 — the buyer skew present on Sonnet is
absent there, which is why no finding here should be stated as a property of "AI agents"
without a second model. `extreme*.jsonl` and `tails.jsonl` are the capitulation probes
behind the zero-spike result.

---

## Two things that will bite an analyst

**`sellshare` vs `past_you_share`.** In pool records, `sellshare` is always from the
seller's side; `past_you_share` flips it when past-you was the buyer. Arena records only
have `sellshare` because the treated agent is always the seller.

**Reasoning traces are absent from arena transcripts.** The engine tried to capture the
model's own thinking block, but at `effort: "low"` Sonnet 5 emits none. Everything about
what agents were thinking comes from the `.probes.jsonl` / `.refprobes.jsonl` replays
instead. Pool transcripts DO carry `thinking`, captured by a deliberate off-thread call.
