# PAT-15 Status

## Command Inputs
- ckpt_wavelet: `/project/nlp-work5/hongyu-s/transformers/examples/pytorch/language-modeling/runs/token_even_mix_2epoch_wavelet_farhead_suite/exp9_scale_coupled_shift_alllayers_nolayergate_10epoch_a6000x4_seed42/checkpoint-15000`
- ckpt_baseline: `/project/nlp-work5/hongyu-s/transformers/examples/pytorch/language-modeling/runs/token_even_mix_PA/checkpoint-15000`
- layerquery_rows: `/project/nlp-work5/hongyu-s/analysis_outputs/PAT-10/gate4_layerquery_rows.parquet`
- cached_wave_dump_root: `/project/nlp-work5/hongyu-s/transformers/examples/pytorch/language-modeling/analysis_program/out/conditioning_hotpot_xsum_s64/__attn_wave`

## Artifacts
- `analysis_outputs/PAT-12/shortcut_test_config.json`
- `analysis_outputs/PAT-12/shortcut_test_results.json`
- `analysis_outputs/PAT-12/examples/`
- `analysis_outputs/PAT-12/conclusion.md`
- `analysis_outputs/PAT-12/STATUS.md`
- `analysis_outputs/PAT-12/run.log`

## Decision
- Branch **A**. Prefix perturbation degrades loss and disrupts mechanism signals; mechanism is unlikely a trivial shortcut.
