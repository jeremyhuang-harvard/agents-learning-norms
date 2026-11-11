# Smoke-test output — NOT for analysis

Two tiny runs used only to validate the pipeline on 2026-07-30. Kept for
provenance; never pool them with real results.

- `noise_floor_20260730-160928.jsonl` (n=2) — ran all 12 rounds with no early
  stopping. Agents drifted after agreement (in one case losing track of which
  side they represented). Superseded.
- `noise_floor_20260730-161357.jsonl` (n=3) — first run with early stopping.
  Pipeline correct, but n=3.

Real runs live one directory up.

- `resume_test.jsonl` (n=5) — resume-machinery validation, 2026-07-30. Ran 2,
  then resumed the same file to 5 and confirmed only the missing 3 executed.
  Data is pipeline-valid (current config) but far too small to analyze.
