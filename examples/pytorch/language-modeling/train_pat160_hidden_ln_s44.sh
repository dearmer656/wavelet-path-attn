#! /bin/bash
#SBATCH --job-name=PAT160_hln_s44
#SBATCH --output=log_file/train/%j_pat160_hidden_ln_s44.txt
#SBATCH --partition=gpu_long
#SBATCH --gres=gpu:a6000:4
#SBATCH --time=100:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4

# PAT-160: QWAB-Hidden-LN ablation — router uses LN-normalized hidden state
# instead of PaTH q_corr delta. One-factor change from Full QWAB (s44):
#   wavelet_ctx_feat_mode="hidden_ln" (was "q_minus_qcorr_meanh")
# All other flags, backbone, data, LR, epochs identical to Full QWAB s44.
# Branch: hongyusaatitech/pat-160-ablate-qwab-router-conditioning-source-with-ln-hidden-input

set -euxo pipefail
echo 'Workdir: /project/nlp-work5/hongyu-s/transformers/examples/pytorch/language-modeling'
cd /project/nlp-work5/hongyu-s/transformers/examples/pytorch/language-modeling

set +u; source /home/is/hongyu-s/miniconda3/etc/profile.d/conda.sh; conda activate latest_transformers; set -u
export PYTHONPATH=/project/nlp-work5/hongyu-s/transformers/src:/project/nlp-work5/hongyu-s/flash-linear-attention:${PYTHONPATH:-}
export HF_HOME=/cl/work5/hongyu-s/huggingfac
export HF_DATASETS_CACHE=/cl/work5/hongyu-s/huggingfac/datasets
export WANDB_DISABLED=true
export WANDB_MODE=disabled

MASTER_PORT=$(( 24160 + SLURM_JOB_ID % 1000 ))

echo '================= BEGIN RUN PAT-160 hidden_ln s44 ================='

/cl/work5/hongyu-s/conda/envs/latest_transformers/bin/torchrun \
  --nproc_per_node=4 \
  --master_port="${MASTER_PORT}" \
  ./run_clm.py \
  --model_type gpt2 \
  --tokenizer_name gpt2 \
  --config_name openai-community/gpt2-medium \
  --model_name_or_path runs/gpt2_medium_owt_pytorch_level_path_attn/checkpoint-80000 \
  --dataset_name mix \
  --block_size 512 \
  --do_train \
  --num_train_epochs 10 \
  --logging_steps 500 \
  --save_steps 5000 \
  --per_device_train_batch_size 4 \
  --per_device_eval_batch_size 4 \
  --gradient_accumulation_steps 4 \
  --learning_rate 1e-4 \
  --weight_decay 0.0 \
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
  --wavelet_mode logit_bias_ctxscale_shift_v0 \
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
  --seed 44 \
  --overwrite_output_dir \
  --output_dir runs/mix_medium_owt_hidden_ln_10ep_s44 \
  --logging_dir ./log_file/mix_medium_owt_hidden_ln_10ep_s44 \
  --cfg_path runs/mix_medium_owt_hidden_ln_10ep_s44/supply_model.cfg

echo "=== PAT-160 QWAB-Hidden-LN s44 done ==="
