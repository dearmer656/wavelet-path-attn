# PAT-13 Status

## Inputs
- `/project/nlp-work5/hongyu-s/analysis_outputs/PAT-6/gate2_router_choice_dataset.parquet`
- `/project/nlp-work5/hongyu-s/transformers/examples/pytorch/language-modeling/analysis_program/out/conditioning_hotpot_xsum_s64/router_state_stats.json`
- `/project/nlp-work5/hongyu-s/transformers/examples/pytorch/language-modeling/analysis_program/out/conditioning_hotpot_xsum_s64/__attn_base`
- `/project/nlp-work5/hongyu-s/transformers/examples/pytorch/language-modeling/analysis_program/out/conditioning_hotpot_xsum_s64/__attn_wave`

## Sample Counts
- layer-query rows: 29018
- head rows: 87054
- margin quantiles: q33=0.150510, q66=0.227166

## Artifacts
- `analysis_outputs/PAT-10/gate4_task_split_state_stats.json`
- `analysis_outputs/PAT-10/gate4_task_split_flip_margin.csv`
- `analysis_outputs/PAT-10/gate4_task_split_pattern_metrics.csv`
- `analysis_outputs/PAT-10/gate4_summary.md`
- `analysis_outputs/PAT-10/conclusion.md`
- `analysis_outputs/PAT-10/STATUS.md`

## Decision
- Branch **B**. Task-split differences are weak/inconsistent.
