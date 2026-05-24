#!/bin/bash
#SBATCH --job-name=rot_qwab_s44
#SBATCH --output=/cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling/runs/rotary_qwab_mix_finetune/s44/%j_train_rotary_qwab_mix_s44.txt
#SBATCH --partition=gpu_long
#SBATCH --time=100:00:00
#SBATCH --gres=gpu:a100-80:2
#SBATCH --nodelist=elm44
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4

# PAT-164: Rotary + QWAB mix fine-tune seed=44
# Baseline: standard Rotary attention (train_rotary_mix_s44.sh)
# Delta: wavelet_router=True → QWABBias module added to each attention layer

set -euxo pipefail

if [ -f /home/is/hongyu-s/miniconda3/etc/profile.d/conda.sh ]; then
  set +u; source /home/is/hongyu-s/miniconda3/etc/profile.d/conda.sh; conda activate latest_transformers; set -u
fi

WORKDIR=/cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling
cd "${WORKDIR}"

if [ -n "${SLURM_NODELIST:-}" ]; then
  export MASTER_ADDR="$(scontrol show hostname "${SLURM_NODELIST}" | head -n 1)"
  export MASTER_PORT=$(( 19500 + SLURM_JOB_ID % 1000 ))
  export WORLD_SIZE="${SLURM_NTASKS:-1}"
  export RANK="${SLURM_PROCID:-0}"
fi

export PYTHONPATH=/project/nlp-work5/hongyu-s/transformers/src${PYTHONPATH:+:${PYTHONPATH}}
export HF_HOME=/cl/work5/hongyu-s/huggingfac
export HF_DATASETS_CACHE=/cl/work5/hongyu-s/huggingfac/datasets
export WANDB_DISABLED=true
export WANDB_MODE=disabled

PRETRAIN_CKPT="${WORKDIR}/runs/wikitext_pe_cmp/rotary/checkpoint-80000"
RUN_OUT="${WORKDIR}/runs/rotary_qwab_mix_finetune/s44"
mkdir -p "${RUN_OUT}/train"

MASTER_PORT=$(( 19500 + SLURM_JOB_ID % 1000 ))

_JOB_TAG="rot_qwab_s44_${SLURM_JOB_ID:-local}"
_slack() { curl -s -X POST -H 'Content-type: application/json' --data "{\"text\":\"[${_JOB_TAG}] $1\"}" "${SLACK_WEBHOOK_URL:-}" > /dev/null 2>&1 || true; }
trap '_slack "FAILED (exit $?)"' ERR
trap '_slack "DONE"' EXIT

_slack "START on $(hostname)"
echo "================= BEGIN: Rotary+QWAB mix fine-tune s44 ================="

python -m torch.distributed.run --nproc_per_node=2 --master_port="${MASTER_PORT}" ./run_clm.py \
  --model_name_or_path "${PRETRAIN_CKPT}" \
  --tokenizer_name gpt2 \
  --dataset_name mix \
  --pe_method rotary \
  --attn_implementation eager \
  --block_size 512 \
  --per_device_train_batch_size 32 \
  --per_device_eval_batch_size 16 \
  --gradient_accumulation_steps 1 \
  --num_train_epochs 10 \
  --learning_rate 1e-4 \
  --weight_decay 0.0 \
  --warmup_ratio 0.05 \
  --do_train \
  --do_eval \
  --eval_strategy steps \
  --eval_steps 500 \
  --logging_steps 250 \
  --save_steps 2500 \
  --save_total_limit 5 \
  --output_dir "${RUN_OUT}" \
  --overwrite_output_dir \
  --logging_dir "${RUN_OUT}/train/tensorboard" \
  --wavelet_router True \
  --router_band_num 8 \
  --scale_range 0 16 \
  --wavelet_mode logit_bias_ctxscale_shift_v0 \
  --use_beta_modulation False \
  --use_soft_wavelet_fox False \
  --wavelet_baseline_use False \
  --single_A_B True \
  --num_harmonics 1 \
  --share_freq_across_heads True \
  --cfg_path "${WORKDIR}/runs/rotary_qwab_mix_finetune/supply_model.cfg" \
  --seed 44

echo "=== Done: Rotary+QWAB mix fine-tune s44 ==="
echo "Output: ${RUN_OUT}"
