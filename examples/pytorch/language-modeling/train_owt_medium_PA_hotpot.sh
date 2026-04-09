#! /bin/bash
#SBATCH --job-name=OWT_MedPA_HP
#SBATCH --output=log_file/train/%j_owt_medium_PA_hotpot.txt
#SBATCH --partition=gpu_long
#SBATCH --time=100:00:00
#SBATCH --gres=gpu:6000:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4

export MASTER_ADDR=$(scontrol show hostname $SLURM_NODELIST | head -n 1)
export MASTER_PORT=12345
export WORLD_SIZE=$SLURM_NTASKS
export RANK=$SLURM_PROCID

echo "MASTER_ADDR: $MASTER_ADDR"
echo "MASTER_PORT: $MASTER_PORT"
echo "WORLD_SIZE: $WORLD_SIZE"
echo "RANK: $RANK"

set -euxo pipefail
echo 'Workdir: /project/nlp-work5/hongyu-s/transformers/examples/pytorch/language-modeling'
cd /project/nlp-work5/hongyu-s/transformers/examples/pytorch/language-modeling
echo 'Launching training/eval command'
echo '================= BEGIN RUN ================='

# PAT-101: PA-only downstream fine-tuning on HotpotQA-Long, from OWT-pretrained medium backbone.
# Set BACKBONE_CKPT to the best val-loss checkpoint from train_gpt2_medium_owt_pytorch_level_path_attn.sh.
BACKBONE_CKPT="runs/gpt2_medium_owt_pytorch_level_path_attn/checkpoint-XXXXX"

torchrun --nproc_per_node=1 --master_port=12420 ./run_clm.py \
  --model_type gpt2 \
  --tokenizer_name gpt2 \
  --config_name openai-community/gpt2-medium \
  --model_name_or_path "${BACKBONE_CKPT}" \
  --dataset_name hotpot_qa \
  --dataset_config_name distractor \
  --block_size 512 \
  --do_train \
  --do_eval \
  --eval_strategy steps \
  --eval_steps 250 \
  --logging_steps 250 \
  --num_train_epochs 8 \
  --save_steps 5000 \
  --per_device_train_batch_size 16 \
  --per_device_eval_batch_size 16 \
  --gradient_accumulation_steps 4 \
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
  --spectral_loss_coe 0.1 \
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
  --output_dir runs/owt_medium_PA_hotpot \
  --logging_dir ./owt_medium_PA_hotpot_log \
  --cfg_path runs/owt_medium_PA_hotpot/supply_model.cfg
