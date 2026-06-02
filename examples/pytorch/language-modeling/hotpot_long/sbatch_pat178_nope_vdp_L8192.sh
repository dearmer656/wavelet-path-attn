#!/bin/bash
#SBATCH --job-name=pat178_nope_vdp_L8192
#SBATCH --output=/cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling/hotpot_long/logs/%j_pat178_nope_vdp_L8192.txt
#SBATCH --partition=gpu_long
#SBATCH --gres=gpu:q6000:1
#SBATCH --nodelist=elm26
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4

_slack() {
    python3 /project/nlp-work5/hongyu-s/gate1/scripts/notify_slack.py \
        --exit-code "$1" \
        --job-id "${SLURM_JOB_ID}" \
        --node "${SLURMD_NODENAME}" \
        --issue "PAT-178" \
        --gpu "q6000x1" \
        --summary "NoPE head-wise V/D/P dump L8192 (64 cases, s42 ckpt15900, hooks+recv_attn)"
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
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

BASE=/cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling
CKPT=${BASE}/runs/wikitext_pe_cmp/wavelet/finetune_eager_nope_seed42/checkpoint-15900
OUT_ROOT=${BASE}/runs/wikitext_pe_cmp/wavelet/finetune_eager_nope_seed42/attn_analysis_s42_headwise

mkdir -p ${BASE}/hotpot_long/logs

python ${BASE}/hotpot_long/dump_nope_head_vdp.py \
    --checkpoint ${CKPT} \
    --jsonl ${BASE}/hotpot_long/data/hotpot_long_dev_uniform_8192only.jsonl \
    --out_root ${OUT_ROOT} \
    --seq_len 8192 \
    --case_limit 64 \
    --query_stride 8 \
    --use_hooks \
    --save_received_attn

echo "=== Done: PAT-178 NoPE head-wise V/D/P L8192 ==="
echo "Features: ${OUT_ROOT}/block_8192/nope_s42_ckpt15900/"
