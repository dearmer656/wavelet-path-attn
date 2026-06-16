#!/bin/bash
#SBATCH --job-name=pat200_dump_router_scale
#SBATCH --output=/cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling/hotpot_long/logs/%j_pat200_dump_router_scale.txt
#SBATCH --partition=gpu_long
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a6000:1
#SBATCH --time=100:00:00

# --- args (override on submit): NCASE, BUDGET, OUTSUB ---
NCASE="${NCASE:-150}"
BUDGET="${BUDGET:-384}"
OUTSUB="${OUTSUB:-full}"

_slack() {
    python3 /project/nlp-work5/hongyu-s/gate1/scripts/notify_slack.py \
        --exit-code "$1" \
        --job-id "${SLURM_JOB_ID}" \
        --node "${SLURMD_NODENAME}" \
        --issue "PAT-200" \
        --gpu "1xA6000" \
        --summary "PAT-200 Phase A query-level router-scale dump: QWAB s42 head-shared, HotpotQA-Long Uniform L4096, NCASE=${NCASE} BUDGET=${BUDGET} (${OUTSUB})" 2>/dev/null || true
}
trap '_slack $?' EXIT

set -euxo pipefail

if [ -f /home/is/hongyu-s/miniconda3/etc/profile.d/conda.sh ]; then
    set +u; source /home/is/hongyu-s/miniconda3/etc/profile.d/conda.sh; conda activate latest_transformers; set -u
fi

export PYTHONPATH=/cl/work5/hongyu-s/flash-linear-attention:/cl/work5/hongyu-s/transformers/src${PYTHONPATH:+:${PYTHONPATH}}
export HF_HOME=/cl/work5/hongyu-s/huggingfac
export HF_DATASETS_CACHE=/cl/work5/hongyu-s/huggingfac/datasets
export PYTHONUNBUFFERED=1

cd /cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling
mkdir -p hotpot_long/logs hotpot_long/analysis_outputs/qwab_scale_query_nmf

CKPT=runs/head_wise_scale_selection_vs_layer_wise/layer_wise/sigmoid_exp/s42_delta_detach/checkpoint-15900

python hotpot_long/dump_router_scale_query.py \
    --checkpoint ${CKPT} \
    --jsonl hotpot_long/data/hotpot_long_dev_uniform.jsonl \
    --out_dir hotpot_long/analysis_outputs/qwab_scale_query_nmf/${OUTSUB} \
    --seq_len 4096 \
    --n_case ${NCASE} \
    --query_budget ${BUDGET} \
    --min_pos 16 \
    --seed 42

echo "=== Done: PAT-200 router-scale dump (${OUTSUB}) ==="
