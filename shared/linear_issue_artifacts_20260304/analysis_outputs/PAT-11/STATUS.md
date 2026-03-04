# PAT-14 Status

## Inputs
- layerquery rows: `/project/nlp-work5/hongyu-s/analysis_outputs/PAT-10/gate4_layerquery_rows.parquet`
- wave dump root: `/project/nlp-work5/hongyu-s/transformers/examples/pytorch/language-modeling/analysis_program/out/conditioning_hotpot_xsum_s64/__attn_wave`

## Counts
- total layer-query rows: 29018
- examples saved: 40

## Artifacts
- `analysis_outputs/PAT-11/segment_definitions.md`
- `analysis_outputs/PAT-11/segment_mass_hotpotqa.csv`
- `analysis_outputs/PAT-11/segment_mass_xsum.csv`
- `analysis_outputs/PAT-11/examples/`
- `analysis_outputs/PAT-11/conclusion.md`
- `analysis_outputs/PAT-11/STATUS.md`

## Decision
- Branch **A**. Segment-aware mass shift is semantically aligned (Hotpot→context/supporting, XSUM→lead).
