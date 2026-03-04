# Segment Definitions (PAT-14)

## Shared Rules
- State unit: `(layer, query)` from PAT-13 layer-query rows.
- Query cohort A (answer/summary region): `query >= floor(0.75 * L_valid)`.
- Attention mass computed on causal keys `0..query` and averaged over target heads per layer.

## HotpotQA
- `question`: token range `[0, context_start)` where `context_start` is first token matching `Context` marker.
- `context`: `[context_start, generation_start)`.
- `generation`: `[generation_start, L_valid)`, `generation_start=floor(0.75*L_valid)`.

## XSUM
- `generation`: `[floor(0.75*L_valid), L_valid)`.
- Source prefix `[0, generation_start)` split as:
  - `lead`: first 20%
  - `mid`: 20%~60%
  - `tail`: 60%~100%
