#!/bin/bash
#SBATCH --job-name=pat187_abl_path_L2048
#SBATCH --output=/cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling/hotpot_long/logs/%j_pat187_abl_path_L2048_R16.txt
#SBATCH --partition=gpu_long
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=8:00:00

_slack() {
    python3 /project/nlp-work5/hongyu-s/gate1/scripts/notify_slack.py \
        --exit-code "$1" \
        --job-id "${SLURM_JOB_ID}" \
        --node "${SLURMD_NODENAME}" \
        --issue "PAT-187" \
        --gpu "1xGPU" \
        --summary "Stage 1 head ablation eval: PaTH-only L2048 R16, pilot (max_cases=20, n_random=5, K=4)"
}
trap '_slack $?' EXIT

set -euxo pipefail

if [ -f /home/is/hongyu-s/miniconda3/etc/profile.d/conda.sh ]; then
    set +u; source /home/is/hongyu-s/miniconda3/etc/profile.d/conda.sh; conda activate latest_transformers; set -u
fi

export PYTHONPATH=/cl/work5/hongyu-s/flash-linear-attention:/cl/work5/hongyu-s/transformers/src${PYTHONPATH:+:${PYTHONPATH}}
export HF_HOME=/cl/work5/hongyu-s/huggingfac
export HF_DATASETS_CACHE=/cl/work5/hongyu-s/huggingfac/datasets

cd /cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling/hotpot_long
mkdir -p logs analysis_outputs/head_ablation_eval

python run_head_ablation_eval.py \
    --model PaTH-only \
    --ext_len 2048 \
    --rank 16 \
    --max_cases 20 \
    --n_random 5 \
    --ks 4 \
    --seed 42 \
    --max_new_tokens 32

echo "=== Done: PAT-187 PaTH-only L2048 R16 pilot ablation ==="
