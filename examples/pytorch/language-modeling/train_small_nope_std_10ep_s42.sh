#!/bin/bash
#SBATCH --job-name=nope_sm_mix_s42
#SBATCH --output=/cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling/runs/small_nope_std_10ep_s42/train/%j_small_nope_std_s42.txt
#SBATCH --partition=gpu_long
#SBATCH --gres=gpu:a6000:4
#SBATCH --time=100:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4

# PAT-164: NoPE standard-attention baseline — mix fine-tune seed=42
# Reviewer h55n (A6 baseline): standard attention + NoPE extrapolation baseline.
# One-factor delta from train_rotary_mix_s42.sh:
#   pe_method="no_pe"  (was "rotary")
# All other flags identical to rotary mix fine-tune.
# Global BS = 4 x 16 x 1 = 64
# Prerequisite: train_nope_wt103.sh → checkpoint-80000 must exist
# Branch: hongyusaatitech/pat-164-sine-basis-ablation

set -euxo pipefail

if [ -f /home/is/hongyu-s/miniconda3/etc/profile.d/conda.sh ]; then
  set +u; source /home/is/hongyu-s/miniconda3/etc/profile.d/conda.sh; conda activate latest_transformers; set -u
fi

WORKDIR=/cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling
cd "${WORKDIR}"

if [ -n "${SLURM_NODELIST:-}" ]; then
  export MASTER_ADDR="$(scontrol show hostname "${SLURM_NODELIST}" | head -n 1)"
  export MASTER_PORT=$(( 19200 + SLURM_JOB_ID % 1000 ))
  export WORLD_SIZE="${SLURM_NTASKS:-1}"
  export RANK="${SLURM_PROCID:-0}"
fi

export PYTHONPATH=/project/nlp-work5/hongyu-s/transformers/src${PYTHONPATH:+:${PYTHONPATH}}
export HF_HOME=/cl/work5/hongyu-s/huggingfac
export HF_DATASETS_CACHE=/cl/work5/hongyu-s/huggingfac/datasets
export WANDB_DISABLED=true
export WANDB_MODE=disabled

PRETRAIN_CKPT="${WORKDIR}/runs/wikitext_pe_cmp/nope/checkpoint-80000"
RUN_OUT="${WORKDIR}/runs/small_nope_std_10ep_s42"
mkdir -p "${RUN_OUT}/train"

MASTER_PORT=$(( 19200 + SLURM_JOB_ID % 1000 ))

echo "================= BEGIN: NoPE standard attn mix fine-tune s42 ================="

python -m torch.distributed.run --nproc_per_node=4 --master_port="${MASTER_PORT}" ./run_clm.py \
  --model_name_or_path "${PRETRAIN_CKPT}" \
  --tokenizer_name gpt2 \
  --dataset_name mix \
  --pe_method no_pe \
  --attn_implementation eager \
  --block_size 512 \
  --per_device_train_batch_size 16 \
  --per_device_eval_batch_size 16 \
  --gradient_accumulation_steps 1 \
  --num_train_epochs 10 \
  --learning_rate 1e-4 \
  --weight_decay 0.01 \
  --warmup_ratio 0.1 \
  --do_train \
  --do_eval \
  --eval_strategy steps \
  --eval_steps 5000 \
  --logging_steps 250 \
  --save_steps 5000 \
  --save_total_limit 10 \
  --output_dir "${RUN_OUT}" \
  --overwrite_output_dir \
  --logging_dir "${RUN_OUT}/train/tensorboard" \
  --wavelet_router False \
  --wavelet_mode logit_bias_ctxscale_shift_v0 \
  --scale_range 0 16 \
  --router_band_num 8 \
  --use_beta_modulation False \
  --use_soft_wavelet_fox False \
  --wavelet_baseline_use False \
  --single_A_B True \
  --num_harmonics 1 \
  --share_freq_across_heads True \
  --cfg_path "${RUN_OUT}/supply_model.cfg" \
  --seed 42

echo "=== Done: NoPE standard attn mix fine-tune s42 ==="
echo "Output: ${RUN_OUT}"
