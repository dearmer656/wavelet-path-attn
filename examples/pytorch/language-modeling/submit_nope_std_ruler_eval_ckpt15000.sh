#!/bin/bash
# PAT-164: submit RULER eval for NoPE standard-attn model (checkpoint-15000)
#
# Usage:
#   bash submit_nope_std_ruler_eval_ckpt15000.sh <RULER_JSONL> [PE_METHOD]
#
# Example:
#   bash submit_nope_std_ruler_eval_ckpt15000.sh /path/to/ruler_task/validation.jsonl no_pe

set -euxo pipefail

RULER_JSONL="${1:?RULER_JSONL required}"
PE_METHOD="${2:-no_pe}"

BASE="/cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling"
SCRIPT="${BASE}/hotpot_long/run_eval_ruler_eager_a6000.sh"
CKPT="${BASE}/runs/small_nope_std_10ep_s42/checkpoint-15000"
MODEL_NAME="nope_std_s42_ckpt15000"
CFG="${BASE}/runs/small_nope_std_10ep_s42/supply_model.cfg"

cd "${BASE}"
mkdir -p hotpot_long/logs

echo "=== Submitting RULER eval for ${MODEL_NAME} ==="
echo "RULER JSONL: ${RULER_JSONL}"
echo "PE method: ${PE_METHOD}"

RULER_BASENAME="$(basename "${RULER_JSONL}")"
RULER_STEM="${RULER_BASENAME%.jsonl}"
RULER_PARENT="$(basename "$(dirname "${RULER_JSONL}")")"
RULER_TAG="${RULER_PARENT}_${RULER_STEM}"
RULER_TAG="$(echo "${RULER_TAG}" | tr -cs '[:alnum:]_.-' '_')"

# Keep minimal by default for rebuttal speed; extend as needed.
for BSIZE in 4096; do
    JID=$(sbatch "${SCRIPT}" "${CKPT}" "${MODEL_NAME}" "${BSIZE}" "${CFG}" "${RULER_JSONL}" "${PE_METHOD}" | grep -oP '\d+')
    echo "  L${BSIZE}: job ${JID}"
done

echo "Done. Results will be in: hotpot_long/results_ruler/${MODEL_NAME}/L*/${RULER_TAG}/"
squeue -u hongyu-s --format="%.10i %.20j %.8T %.12M %R" | grep -E "JOBID|ruler_eval" || true
