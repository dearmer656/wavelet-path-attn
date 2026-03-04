# PAT-1c-2b Status

## Command Run
```bash
srun -p gpu_intr --gres=gpu:a6000:1 --cpus-per-task=8 bash -lc 'source /home/is/hongyu-s/miniconda3/etc/profile.d/conda.sh && conda activate latest_transformers && python /cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling/analysis_program/pat1c2b_gate1c_hotfix.py --bootstrap_json /cl/work5/hongyu-s/analysis_outputs/PAT-1c-1/bootstrap_ci.json --profiles_json /cl/work5/hongyu-s/analysis_outputs/PAT-1c-1/deltaE_profiles_rel_abs_minmax.json --prev_decision_json /cl/work5/hongyu-s/analysis_outputs/PAT-1c-2/remediation_retest_gate1c.json --prev_spec_md /cl/work5/hongyu-s/analysis_outputs/PAT-1c-2/gate1c_spec_update.md --out_dir /cl/work5/hongyu-s/analysis_outputs/PAT-1c-2b --profile_l2_threshold 0.12'
```

## Input Artifacts Reused (No Heavy Recompute)
- `/cl/work5/hongyu-s/analysis_outputs/PAT-1c-1/bootstrap_ci.json`
- `/cl/work5/hongyu-s/analysis_outputs/PAT-1c-1/deltaE_profiles_rel_abs_minmax.json`
- `/cl/work5/hongyu-s/analysis_outputs/PAT-1c-2/remediation_retest_gate1c.json`
- `/cl/work5/hongyu-s/analysis_outputs/PAT-1c-2/gate1c_spec_update.md`

## Output Artifacts
- `analysis_outputs/PAT-1c-2b/gate1c_spec_update_hotfix.md`
- `analysis_outputs/PAT-1c-2b/remediation_retest_gate1c_hotfix.json`
- `analysis_outputs/PAT-1c-2b/remediation_report_hotfix.md`
- `analysis_outputs/PAT-1c-2b/conclusion.md`
- `analysis_outputs/PAT-1c-2b/STATUS.md`

## One-line Summary
- **Gate1c: PASS (branch A), by patched shape criterion B** (`profile_rel_l2 + CI` stable for both L10/L11).
