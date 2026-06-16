#!/bin/bash
#SBATCH --job-name=P160_hln_s43_xs
#SBATCH --output=/cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling/log_file/train/%j_pat160_hidden_ln_s43_xsum.txt
#SBATCH --partition=gpu_long
#SBATCH --gres=gpu:a100:4
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4

# PAT-160: XSum Filtered eval for QWAB-Hidden-LN s43
# Lengths: L512, L1024, L1536

set -euxo pipefail
export PYTHONPATH=/cl/work5/hongyu-s/transformers/src:/cl/work5/hongyu-s/flash-linear-attention:${PYTHONPATH:-}
export HF_HOME=/cl/work5/hongyu-s/huggingfac
export HF_DATASETS_CACHE=/cl/work5/hongyu-s/huggingfac/datasets
export WANDB_DISABLED=true

set +u; source /home/is/hongyu-s/miniconda3/etc/profile.d/conda.sh; conda activate latest_transformers; set -u

cd /cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling

CKPT_DIR="runs/mix_medium_owt_hidden_ln_10ep_s43/checkpoint-15000"
CFG_PATH="runs/mix_medium_owt_hidden_ln_10ep_s43/supply_model.cfg"
XSUM_VALIDATION_FILE="/cl/work5/hongyu-s/fact-check-summarization/xsum_test_filter_level2_official_style.jsonl"
RESULT_DIR="runs/mix_medium_owt_hidden_ln_10ep_s43/ckpt_eval_xsum_rouge"
RUN_TAG="${SLURM_JOB_ID}_$(date -u +%Y%m%dT%H%M%SZ)"
OUT_ROOT_DIR="${RESULT_DIR}/${RUN_TAG}"
mkdir -p "${RESULT_DIR}" "${OUT_ROOT_DIR}"

JOB_PORT=$(( 19160 + SLURM_JOB_ID % 1000 ))

declare -a BLOCK_SIZES=(512 1024 1536)
declare -a BATCH_SIZES=(16  8    4)

for i in "${!BLOCK_SIZES[@]}"; do
  BS="${BLOCK_SIZES[$i]}"
  BAT="${BATCH_SIZES[$i]}"
  OUT_DIR="${OUT_ROOT_DIR}/xsum_L${BS}"
  mkdir -p "${OUT_DIR}"
  echo "=== XSum L${BS} batch=${BAT} ==="

  /cl/work5/hongyu-s/conda/envs/latest_transformers/bin/torchrun \
    --nproc_per_node=4 \
    --master_port="${JOB_PORT}" \
    ./run_clm.py \
    --model_type gpt2 --tokenizer_name gpt2 \
    --model_name_or_path "${CKPT_DIR}" \
    --share_freq_across_heads True \
    --learning_rate 1e-4 \
    --per_device_eval_batch_size "${BAT}" \
    --block_size "${BS}" \
    --dataset_name xsum \
    --dataset_config_name default \
    --validation_file "${XSUM_VALIDATION_FILE}" \
    --do_eval \
    --output_dir "${OUT_DIR}" --overwrite_output_dir \
    --logging_dir "${RESULT_DIR}/log" \
    --attn_implementation path_attn \
    --path_use_qk_norm false --path_use_low_rank_w true \
    --path_use_w_shortconv false --path_conv_size 3 --path_conv_bias false \
    --num_harmonics 1 --single_A_B True \
    --use_beta_modulation False --use_soft_wavelet_fox False \
    --wavelet_mode logit_bias_ctxscale_shift_v0 \
    --wavelet_baseline_use False \
    --init_theta 0.847 \
    --use_forget_gate False \
    --xsum_bucket_size 512 \
    --xsum_bucket_apply_to eval_test \
    --spectral_loss_coe 0.1 --temp_loss_coe 0.0 \
    --scale_range 0 16 \
    --qk_rotation False --ablate_switch False \
    --wavelet_router False \
    --router_hidden_dim 32 --router_band_num 8 --rel_selection all \
    --cfg_path "${CFG_PATH}" \
    --load_best_model_at_end False \
    --seed 43

  RES="${OUT_DIR}/eval_results.json"
  if [ -f "${RES}" ]; then
    python3 -c "import json; d=json.load(open('${RES}')); print(f'L${BS}: rouge1={d.get(\"eval_rouge1\",\"N/A\"):.4f}')"
  fi
done

echo "=== Done: PAT-160 Hidden-LN s43 XSum L512/L1024/L1536 ==="
