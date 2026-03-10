#!/bin/bash
#SBATCH --job-name=lwreshw_s42_wr
#SBATCH --output=/cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling/runs/LW_residual_HW/train/%j_gpt2small_pathattn_lw_residual_hw_weakreg_seed42_4xa6000_20260310.txt
#SBATCH --partition=gpu_long
#SBATCH --time=100:00:00
#SBATCH --gres=gpu:a6000:4
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4

set -euxo pipefail

if [ -f /home/is/hongyu-s/miniconda3/etc/profile.d/conda.sh ]; then
  set +u
  source /home/is/hongyu-s/miniconda3/etc/profile.d/conda.sh
  conda activate latest_transformers
  set -u
fi

WORKDIR=/cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling
cd "${WORKDIR}"

export MASTER_ADDR="$(scontrol show hostname "${SLURM_NODELIST}" | head -n 1)"
export MASTER_PORT=16442
export WORLD_SIZE="${SLURM_NTASKS}"
export RANK="${SLURM_PROCID}"

RUN_NAME=gpt2small_pathattn_lw_residual_hw_weakreg_seed42_4xa6000_20260310
RUN_ROOT="runs/LW_residual_HW/checkpoints/${RUN_NAME}"
CFG_PATH="runs/LW_residual_HW/configs/${RUN_NAME}.cfg"
LOG_DIR="runs/LW_residual_HW/train_log/${RUN_NAME}"

python -m torch.distributed.run --nproc_per_node=4 --master_port=16442 ./run_clm.py --model_type gpt2 --tokenizer_name gpt2 --share_freq_across_heads True \
--learning_rate 1e-4 --weight_decay 0.0 --per_device_train_batch_size 8 --per_device_eval_batch_size 16 --block_size 512 --dataset_name mix \
--do_train --do_eval --eval_strategy steps --eval_steps 500 --logging_dir "${LOG_DIR}" --logging_steps 500 --num_train_epochs 10 --num_harmonics 1 --wavelet_pe_softmax_use False \
--save_steps 5000 --attn_implementation path_attn --path_use_qk_norm false \
--path_use_low_rank_w true --path_use_w_shortconv false --path_conv_size 3 --warmup_ratio 0.05 --path_conv_bias false --output_dir "${RUN_ROOT}" \
--gradient_accumulation_steps 2 --b_unfreeze_step 5000 --pe_method no_pe --single_A_B True \
--use_beta_modulation False --use_soft_wavelet_fox False --wavelet_mode logit_bias_ctxscale_shift_v0 --model_name_or_path runs/1r_baseline_from_s/checkpoint-80000 --full_fine_tune False \
--wavelet_baseline_use False --init_theta 0.847 --use_forget_gate False --sample_num 16 \
--spectral_loss_coe 0.1 --temp_loss_coe 0.0 --distill_teacher wavelet --distill_in_which_layers 0 \
--distill_freq_scale 25 --smooth_use False --distilling_coe_warmup_use False --scale_range 0 16 \
--weight_alpha 0.0 --loss_type cos --qk_rotation False --wavelet_router False \
--router_band_num 8 --router_hidden_dim 32 --rel_selection all \
--eval_rel_stats_enabled True --eval_rel_stats_layers 'all' --eval_rel_stats_bin_size 256 --eval_rel_stats_log_every 0 --eval_rel_stats_log_once True \
--eval_rel_stats_per_head False --eval_rel_stats_max_samples_per_bin 4096 --eval_rel_stats_eps 1e-06 --eval_rel_stats_anchor_layer 0 \
--log_rel_stats False --log_rel_every 500 --log_rel_sample_qpos '128,512,2048' --log_rel_sample_heads '0,3,7,11' \
--log_rel_sample_key_offsets '0,16,64,256,1024' --log_rel_tail_tau 1024 --log_rel_eval_every 0 \
--rel_param_keywords 'rel,wavelet,router,seq_pe,q_corr,wavelet_dtt' \
--router_norm_enable False --router_norm_mode pre_gate --router_norm_type rmsnorm --router_norm_affine False \
--router_norm_eps 1e-05 --router_norm_clamp_std_min 0.0001 --router_norm_log_every 500 --router_norm_log_heads '0,3,7' \
--router_norm_log_tokens '0,-1' --eval_router_heatmap_enable False --eval_router_heatmap_bin_size 256 --eval_router_heatmap_max_batches 0 \
--eval_router_heatmap_out_subdir 'router_heatmaps' \
--lw_residual_hw_enable True --lw_residual_hw_alpha 0.1 --lw_residual_hw_l2 1e-4 --lw_residual_hw_freeze_steps 1500 \
--cfg_path "${CFG_PATH}" \
--seed 42 --config_name gpt2 --overwrite_output_dir --coe_for_rel_lr 1e-4
