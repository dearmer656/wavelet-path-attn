#! /bin/bash
#SBATCH --job-name=PAT105_A1
#SBATCH --output=log_file/train/%j_pat105_a1_fixed32.txt
#SBATCH --partition=gpu_long
#SBATCH --gres=gpu:a6000:4
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=48:00:00

# PAT-105: A1 — fixed-interval k=32 cumulative mask.
# Deterministic trigger: (t+1)%32==0 → 16 positions per T=512.
# No sparse gate, no force-open gate. path_lam learnable.
# Same recipe as A0 (PAT-102 baseline), one-factor delta: path_fixed_interval=32.

set -euxo pipefail
echo 'Workdir: /project/nlp-work5/hongyu-s/transformers/examples/pytorch/language-modeling'
cd /project/nlp-work5/hongyu-s/transformers/examples/pytorch/language-modeling

set +u; source /home/is/hongyu-s/miniconda3/etc/profile.d/conda.sh; conda activate latest_transformers; set -u
export PYTHONPATH=/project/nlp-work5/hongyu-s/transformers/src:/project/nlp-work5/hongyu-s/flash-linear-attention:${PYTHONPATH:-}
export HF_HOME=/cl/work5/hongyu-s/huggingfac
export HF_DATASETS_CACHE=/cl/work5/hongyu-s/huggingfac/datasets
export WANDB_DISABLED=true
export WANDB_MODE=disabled

MASTER_PORT=$(( 20200 + SLURM_JOB_ID % 1000 ))

echo '================= BEGIN RUN ================='

/cl/work5/hongyu-s/conda/envs/latest_transformers/bin/torchrun \
  --nproc_per_node=4 \
  --master_port="${MASTER_PORT}" \
  ./run_clm.py \
  --model_type gpt2 \
  --tokenizer_name gpt2 \
  --config_name openai-community/gpt2-medium \
  --model_name_or_path runs/gpt2_medium_0.01wd_0.1wu_pytorch_level_path_attn/checkpoint-30000 \
  --dataset_name hotpot_qa \
  --dataset_config_name distractor \
  --block_size 512 \
  --do_train \
  --do_eval \
  --max_steps 5000 \
  --eval_strategy steps \
  --eval_steps 500 \
  --logging_steps 250 \
  --save_steps 1000 \
  --load_best_model_at_end True \
  --metric_for_best_model eval_loss \
  --greater_is_better False \
  --per_device_train_batch_size 8 \
  --per_device_eval_batch_size 8 \
  --gradient_accumulation_steps 2 \
  --learning_rate 1e-4 \
  --weight_decay 0.01 \
  --warmup_ratio 0.05 \
  --attn_implementation path_attn \
  --path_use_qk_norm false \
  --path_use_low_rank_w true \
  --path_use_w_shortconv false \
  --path_conv_size 3 \
  --path_conv_bias false \
  --single_A_B True \
  --share_freq_across_heads True \
  --b_unfreeze_step 5000 \
  --pe_method vanilla \
  --num_harmonics 1 \
  --wavelet_pe_softmax_use False \
  --wavelet_mode db1 \
  --wavelet_baseline_use False \
  --wavelet_router False \
  --use_beta_modulation False \
  --use_soft_wavelet_fox False \
  --use_forget_gate False \
  --full_fine_tune False \
  --init_theta 0.847 \
  --sample_num 16 \
  --spectral_loss_coe 0.0 \
  --temp_loss_coe 0.0 \
  --distill_teacher wavelet \
  --distill_in_which_layers 0 \
  --distill_freq_scale 25 \
  --smooth_use False \
  --distilling_coe_warmup_use False \
  --scale_range 0 16 \
  --weight_alpha 0.0 \
  --loss_type cos \
  --qk_rotation False \
  --router_band_num 8 \
  --router_hidden_dim 32 \
  --rel_selection all \
  --path_blend_layers 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 \
  --path_sparse_gate False \
  --path_fixed_interval 32 \
  --preprocessing_num_workers 8 \
  --ddp_timeout 7200 \
  --seed 42 \
  --overwrite_output_dir \
  --output_dir runs/pat105_a1_fixed32_hotpotqa_5k_seed42 \
  --logging_dir ./pat105_a1_fixed32_hotpotqa_5k_seed42_log \
  --cfg_path runs/pat105_a1_fixed32_hotpotqa_5k_seed42/supply_model.cfg

echo "=== PAT-105 A1 fixed-32 done ==="
