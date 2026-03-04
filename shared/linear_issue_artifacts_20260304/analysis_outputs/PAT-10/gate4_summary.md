# Gate4a Summary (PAT-13)

## Inputs
- gate2 dataset: `/project/nlp-work5/hongyu-s/analysis_outputs/PAT-6/gate2_router_choice_dataset.parquet`
- router state stats: `/project/nlp-work5/hongyu-s/transformers/examples/pytorch/language-modeling/analysis_program/out/conditioning_hotpot_xsum_s64/router_state_stats.json`
- base dump root: `/project/nlp-work5/hongyu-s/transformers/examples/pytorch/language-modeling/analysis_program/out/conditioning_hotpot_xsum_s64/__attn_base`
- wave dump root: `/project/nlp-work5/hongyu-s/transformers/examples/pytorch/language-modeling/analysis_program/out/conditioning_hotpot_xsum_s64/__attn_wave`

## Key Signals
- max SMALL ratio gap (HotpotQA vs XSUM): 0.043476
- max SMALL low-margin flip gap: 0.020321
- max SMALL-vs-baseline PrefixMass delta gap: 0.002605

## Decision
- Branch: **B**
- Reason: Task-split differences are weak/inconsistent.
- Rule: mark A if >=2 predefined task-gap signals are above threshold.

## Artifacts
- `gate4_task_split_state_stats.json`
- `gate4_task_split_flip_margin.csv`
- `gate4_task_split_pattern_metrics.csv`
- `gate4_layerquery_rows.parquet` (intermediate)
- `gate4_head_rows.parquet` (intermediate)
