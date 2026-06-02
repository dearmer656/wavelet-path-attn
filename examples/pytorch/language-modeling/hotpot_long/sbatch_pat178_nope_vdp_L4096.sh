#!/bin/bash
#SBATCH --job-name=pat178_nope_vdp_L4096
#SBATCH --output=/cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling/hotpot_long/logs/%j_pat178_nope_vdp_L4096.txt
#SBATCH --partition=gpu_long
#SBATCH --gres=gpu:a6000:1
#SBATCH --time=8:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4

_slack() {
    python3 /project/nlp-work5/hongyu-s/gate1/scripts/notify_slack.py \
        --exit-code "$1" \
        --job-id "${SLURM_JOB_ID}" \
        --node "${SLURMD_NODENAME}" \
        --issue "PAT-178" \
        --gpu "a6000x1" \
        --summary "NoPE head-wise V/D/P dump L4096 (64 cases, s42 ckpt15900)"
}
trap '_slack $?' EXIT

set -euxo pipefail

if [ -f /home/is/hongyu-s/miniconda3/etc/profile.d/conda.sh ]; then
    set +u; source /home/is/hongyu-s/miniconda3/etc/profile.d/conda.sh; conda activate latest_transformers; set -u
fi

export PYTHONPATH=/project/nlp-work5/hongyu-s/flash-linear-attention:/project/nlp-work5/hongyu-s/transformers/src${PYTHONPATH:+:${PYTHONPATH}}
export HF_HOME=/cl/work5/hongyu-s/huggingfac
export HF_DATASETS_CACHE=/cl/work5/hongyu-s/huggingfac/datasets
export WANDB_DISABLED=true

BASE=/cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling
CKPT=${BASE}/runs/wikitext_pe_cmp/wavelet/finetune_eager_nope_seed42/checkpoint-15900
OUT_ROOT=${BASE}/runs/wikitext_pe_cmp/wavelet/finetune_eager_nope_seed42/attn_analysis_s42_headwise

mkdir -p ${BASE}/hotpot_long/logs

python ${BASE}/hotpot_long/dump_nope_head_vdp.py \
    --checkpoint ${CKPT} \
    --jsonl ${BASE}/hotpot_long/data/hotpot_long_dev_uniform.jsonl \
    --out_root ${OUT_ROOT} \
    --seq_len 4096 \
    --case_limit 64 \
    --query_stride 4

echo "=== Done: PAT-178 NoPE head-wise V/D/P L4096 ==="
echo "Features: ${OUT_ROOT}/block_4096/nope_s42_ckpt15900/"
