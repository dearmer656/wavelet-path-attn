#! /bin/bash
#SBATCH --job-name=MedOWT_PA
#SBATCH --output=log_file/train/%j_gpt2_medium_owt_pytorch_level_path_attn.txt
#SBATCH --partition=gpu_long
#SBATCH --time=100:00:00
#SBATCH --gres=gpu:a100-80:2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4

export MASTER_ADDR=$(scontrol show hostname $SLURM_NODELIST | head -n 1)
export MASTER_PORT=12421
export WORLD_SIZE=$SLURM_NTASKS
export RANK=$SLURM_PROCID

echo "MASTER_ADDR: $MASTER_ADDR"
echo "MASTER_PORT: $MASTER_PORT"
echo "WORLD_SIZE: $WORLD_SIZE"
echo "RANK: $RANK"

set -euxo pipefail
echo 'Workdir: /project/nlp-work5/hongyu-s/transformers/examples/pytorch/language-modeling'
cd /project/nlp-work5/hongyu-s/transformers/examples/pytorch/language-modeling

# Use dev transformers from source; conda env provides torch/tokenizers
source /home/is/hongyu-s/miniconda3/etc/profile.d/conda.sh
conda activate latest_transformers
export PYTHONPATH=/project/nlp-work5/hongyu-s/transformers/src:${PYTHONPATH:-}

echo 'Launching OWT medium PaTH pretraining'
echo '================= BEGIN RUN ================='

# PAT-101: GPT-2 medium PaTH backbone pretrained on OpenWebText.
# One-factor delta from WikiText-103 run: dataset only. All other hyperparams frozen.
# Early stopping patience=8 at eval_steps=500 on val_loss (5% OWT train → val split).
# Global batch = 16 (per_device) × 2 (GPUs) × 2 (grad_accum) = 64.

/cl/work5/hongyu-s/conda/envs/latest_transformers/bin/torchrun --nproc_per_node=2 --master_port=12421 ./run_clm.py \
  --model_type gpt2 \
  --tokenizer_name gpt2 \
  --config_name openai-community/gpt2-medium \
  --dataset_name openwebtext \
  --validation_split_percentage 1 \
  --block_size 512 \
  --do_train \
  --do_eval \
  --num_train_epochs 30 \
  --eval_strategy steps \
  --eval_steps 500 \
  --logging_steps 500 \
  --save_steps 5000 \
  --load_best_model_at_end True \
  --metric_for_best_model eval_loss \
  --greater_is_better False \
  --early_stopping_patience 8 \
  --per_device_train_batch_size 16 \
  --per_device_eval_batch_size 16 \
  --gradient_accumulation_steps 2 \
  --learning_rate 1e-4 \
  --weight_decay 0.01 \
  --warmup_ratio 0.1 \
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
  --wavelet_pe_softmax_use True \
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
  --seed 42 \
  --overwrite_output_dir \
  --output_dir runs/gpt2_medium_owt_pytorch_level_path_attn \
  --logging_dir ./gpt2_medium_owt_pytorch_level_path_attn_log \
  --cfg_path runs/gpt2_medium_owt_pytorch_level_path_attn/supply_model.cfg
