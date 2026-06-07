#!/bin/bash
#SBATCH --job-name=pat186_path_nmf_L512
#SBATCH --output=/cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling/hotpot_long/logs/%j_pat186_path_nmf_L512.txt
#SBATCH --partition=gpu_long
#SBATCH --gres=gpu:a6000:1
#SBATCH --time=4:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4

_slack() {
    python3 /project/nlp-work5/hongyu-s/gate1/scripts/notify_slack.py \
        --exit-code "$1" \
        --job-id "${SLURM_JOB_ID}" \
        --node "${SLURMD_NODENAME}" \
        --issue "PAT-186" \
        --gpu "a6000x1" \
        --summary "PaTH NMF L512 pilot: salient+dense R=16, 20 cases"
}
trap '_slack $?' EXIT

set -euxo pipefail

if [ -f /home/is/hongyu-s/miniconda3/etc/profile.d/conda.sh ]; then
    set +u; source /home/is/hongyu-s/miniconda3/etc/profile.d/conda.sh; conda activate latest_transformers; set -u
fi

export PYTHONPATH=/cl/work5/hongyu-s/flash-linear-attention:/cl/work5/hongyu-s/transformers/src${PYTHONPATH:+:${PYTHONPATH}}
export HF_HOME=/cl/work5/hongyu-s/huggingfac
export HF_DATASETS_CACHE=/cl/work5/hongyu-s/huggingfac/datasets
export WANDB_DISABLED=true

BASE=/cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling
CKPT=${BASE}/runs/PA_baseline_multi_seeds/token_even_mix_PA_s42/checkpoint-15900
OUT_ROOT=${BASE}/runs/PA_baseline_multi_seeds/nmf_logit_motifs
JSONL=${BASE}/hotpot_long/data/hotpot_long_dev_uniform.jsonl

mkdir -p ${BASE}/hotpot_long/logs

# L1 pilot: Salient-NMF (main run)
python ${BASE}/hotpot_long/dump_path_nmf.py \
    --checkpoint ${CKPT} \
    --jsonl ${JSONL} \
    --out_root ${OUT_ROOT} \
    --seq_len 512 \
    --n_case 20 \
    --rank 16 \
    --pool_size 128 \
    --preprocessing salient

# L1 control: Dense-NMF (no filtering)
python ${BASE}/hotpot_long/dump_path_nmf.py \
    --checkpoint ${CKPT} \
    --jsonl ${JSONL} \
    --out_root ${OUT_ROOT} \
    --seq_len 512 \
    --n_case 20 \
    --rank 16 \
    --pool_size 128 \
    --preprocessing dense

echo "=== Done: PAT-186 PaTH NMF L512 pilot ==="
echo "Output: ${OUT_ROOT}/"
