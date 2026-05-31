#!/bin/bash
#SBATCH --job-name=pat176_rul_path
#SBATCH --output=examples/pytorch/rul_cmapss/outputs/fd001/%j_pat176_rul_path.out
#SBATCH --partition=gpu_long
#SBATCH --gres=gpu:a100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=1:00:00

_slack() {
    python3 /project/nlp-work5/hongyu-s/gate1/scripts/notify_slack.py \
        --exit-code "$1" \
        --job-id "${SLURM_JOB_ID}" \
        --node "${SLURMD_NODENAME}" \
        --issue "PAT-176" \
        --gpu "A100x1" \
        --summary "PAT-176 RUL smoke run: PaTH-only FD001 5 epochs"
}
trap '_slack $?' EXIT

export PYTHONPATH=/project/nlp-work5/hongyu-s/transformers/src:${PYTHONPATH:-}
PYTHON=/cl/work5/hongyu-s/conda/envs/latest_transformers/bin/python

cd /project/nlp-work5/hongyu-s/transformers

mkdir -p examples/pytorch/rul_cmapss/outputs/fd001

# DATA: place train_FD001.txt, test_FD001.txt, RUL_FD001.txt in data/cmapss/
# Download from: https://ti.arc.nasa.gov/tech/dash/groups/pcoe/prognostic-data-repository/#turbofan
if [ ! -f data/cmapss/train_FD001.txt ]; then
    echo "ERROR: data/cmapss/train_FD001.txt not found. Please download C-MAPSS FD001 data first."
    exit 1
fi

$PYTHON examples/pytorch/rul_cmapss/run_fd001.py \
    --model path \
    --data_dir data/cmapss \
    --output_dir examples/pytorch/rul_cmapss/outputs/fd001 \
    --window_size 30 \
    --stride 1 \
    --max_rul 125 \
    --epochs 5 \
    --batch_size 128 \
    --lr 1e-3 \
    --n_layer 2 \
    --num_heads 4 \
    --head_dim 16 \
    --seed 42 \
    --device cuda
