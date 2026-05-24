#!/bin/bash
# PAT-164: Submit HotpotQA-Long uniform eval for NoPE standard-attn model (checkpoint-15000)
# Reviewer h55n A6 baseline: standard attention + NoPE at L512/L2048/L4096
# Uses run_eval_hotpot_long_uniform_eager_a6000.sh (pe_method=no_pe arg6)
# Prerequisite: train_small_nope_std_10ep_s42.sh must have completed

set -euxo pipefail

BASE="/cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling"
SCRIPT="${BASE}/hotpot_long/run_eval_hotpot_long_uniform_eager_a6000.sh"
CKPT="${BASE}/runs/small_nope_std_10ep_s42/checkpoint-15000"
MODEL_NAME="nope_std_s42_ckpt15000"
CFG="${BASE}/runs/small_nope_std_10ep_s42/supply_model.cfg"
PE_METHOD="no_pe"

cd "${BASE}"
mkdir -p hotpot_long/logs

echo "=== Submitting NoPE standard-attn HotpotQA-Long eval (checkpoint-15000) ==="

for BSIZE in 512 2048 4096; do
    JID=$(sbatch "${SCRIPT}" "${CKPT}" "${MODEL_NAME}" "${BSIZE}" "${CFG}" "${BSIZE}" "${PE_METHOD}" | grep -oP '\d+')
    echo "  L${BSIZE}: job ${JID}"
done

echo "Done. Results will be in: hotpot_long/results_uniform/${MODEL_NAME}/"
squeue -u hongyu-s --format="%.10i %.20j %.8T %.12M %R" | grep -E "JOBID|hpunif_eager" || true
