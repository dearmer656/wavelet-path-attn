#!/bin/bash
#SBATCH --job-name=wt103_nope
#SBATCH --output=/cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling/runs/wikitext_pe_cmp/nope/train/%j_nope_wt103.txt
#SBATCH --partition=gpu_long
#SBATCH --time=100:00:00
#SBATCH --gres=gpu:a6000:4
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8

# PAT-164: NoPE pretraining on wikitext-103 (standard attention, no position encoding)
# One-factor delta from train_rotary_wt103.sh:
#   pe_method="no_pe"  (was "rotary")
#   attn_implementation="eager" (same)
# Output: runs/wikitext_pe_cmp/nope/checkpoint-80000  (used by train_small_nope_std_10ep_s42.sh)
# Branch: hongyusaatitech/pat-164-sine-basis-ablation

set -euxo pipefail

if [ -f /home/is/hongyu-s/miniconda3/etc/profile.d/conda.sh ]; then
  set +u; source /home/is/hongyu-s/miniconda3/etc/profile.d/conda.sh; conda activate latest_transformers; set -u
fi

WORKDIR=/cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling
cd "${WORKDIR}"

if [ -n "${SLURM_NODELIST:-}" ]; then
  export MASTER_ADDR="$(scontrol show hostname "${SLURM_NODELIST}" | head -n 1)"
  export MASTER_PORT=$(( 19000 + SLURM_JOB_ID % 1000 ))
  export WORLD_SIZE="${SLURM_NTASKS:-1}"
  export RANK="${SLURM_PROCID:-0}"
fi

export PYTHONPATH=/project/nlp-work5/hongyu-s/transformers/src${PYTHONPATH:+:${PYTHONPATH}}
export HF_HOME=/cl/work5/hongyu-s/huggingfac
export HF_DATASETS_CACHE=/cl/work5/hongyu-s/huggingfac/datasets
export WANDB_DISABLED=true
export WANDB_MODE=disabled

RUN_OUT="${WORKDIR}/runs/wikitext_pe_cmp/nope"
mkdir -p "${RUN_OUT}/train"

MASTER_PORT=$(( 19100 + SLURM_JOB_ID % 1000 ))

echo "================= BEGIN: NoPE pretrain on wikitext-103 ================="

python -m torch.distributed.run --nproc_per_node=4 --master_port="${MASTER_PORT}" ./run_clm.py \
  --model_type gpt2 \
  --config_name gpt2 \
  --tokenizer_name gpt2 \
  --dataset_name wikitext \
  --dataset_config_name wikitext-103-raw-v1 \
  --pe_method no_pe \
  --attn_implementation eager \
  --block_size 512 \
  --per_device_train_batch_size 16 \
  --per_device_eval_batch_size 16 \
  --gradient_accumulation_steps 1 \
  --num_train_epochs 30 \
  --learning_rate 6e-4 \
  --weight_decay 0.1 \
  --warmup_ratio 0.01 \
  --do_train \
  --do_eval \
  --eval_strategy steps \
  --eval_steps 5000 \
  --logging_steps 500 \
  --save_steps 20000 \
  --save_total_limit 10 \
  --output_dir "${RUN_OUT}" \
  --logging_dir "${RUN_OUT}/train" \
  --overwrite_output_dir \
  --overwrite_cache \
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
  --seed 42

echo "=== Done: NoPE wikitext-103 pretrain ==="
echo "Checkpoint for fine-tuning: ${RUN_OUT}/checkpoint-80000"
