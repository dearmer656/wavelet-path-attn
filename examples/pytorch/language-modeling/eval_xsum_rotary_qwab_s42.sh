#!/bin/bash
#SBATCH --job-name=xsum_rot_qwab
#SBATCH --output=/cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling/runs/small_rotary_qwab_10ep_s42/ckpt_eval_hm/%j_xsum_rotary_qwab_s42.txt
#SBATCH --partition=gpu_long
#SBATCH --gres=gpu:a6000:4
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4

# PAT-166: Filtered XSum eval for Rotary+QWAB small model, seed=42
# block_sizes: 512 / 1024 / 1536

set -euxo pipefail

if [ -f /home/is/hongyu-s/miniconda3/etc/profile.d/conda.sh ]; then
  set +u; source /home/is/hongyu-s/miniconda3/etc/profile.d/conda.sh; conda activate latest_transformers; set -u
fi

export PYTHONPATH=/project/nlp-work5/hongyu-s/transformers/src${PYTHONPATH:+:${PYTHONPATH}}
export HF_HOME=/cl/work5/hongyu-s/huggingfac
export HF_DATASETS_CACHE=/cl/work5/hongyu-s/huggingfac/datasets
export SKIP_FENICE=1
export SKIP_SUMMAC=1
export WANDB_DISABLED=true
export WANDB_MODE=disabled

WORKDIR=/cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling
cd "${WORKDIR}"

RUN_DIR="${WORKDIR}/runs/small_rotary_qwab_10ep_s42"
BEST_CKPT="${RUN_DIR}/checkpoint-15000"
MODEL_LABEL="rotary_qwab_s42"
XSUM_VALIDATION_FILE="/cl/work5/hongyu-s/fact-check-summarization/xsum_test_filter_level2_official_style.jsonl"
RESULT_DIR="${RUN_DIR}/ckpt_eval_hm"
RUN_TAG="${SLURM_JOB_ID:-manual}_$(date -u +%Y%m%dT%H%M%SZ)"
OUT_ROOT_DIR="${RESULT_DIR}/xsum_out_${RUN_TAG}"
XSUM_CSV="${RESULT_DIR}/xsum_filter_metrics_${MODEL_LABEL}_${RUN_TAG}.csv"
mkdir -p "${RESULT_DIR}" "${OUT_ROOT_DIR}"

echo "step,checkpoint,block_size,batch_size,rouge1,rouge2,rougeL,bertscore,count,eval_loss,timestamp_utc" > "${XSUM_CSV}"

declare -a XSUM_BLOCK_SIZES=(512 1024 1536)
declare -a XSUM_BATCH_SIZES=(16   8    4)

JOB_PORT_OFFSET=$(( ${SLURM_JOB_ID:-0} % 1000 ))
XSUM_PORT=$(( 19400 + JOB_PORT_OFFSET ))

extract_metric() {
  local key="$1" file="$2"
  awk -F: -v target="\"${key}\"" '$1 ~ target {gsub(/[ ,]/, "", $2); gsub(/"/, "", $2); print $2; exit}' "${file}"
}
extract_first_metric() {
  local file="$1"; shift
  local key="" value=""
  for key in "$@"; do
    value="$(extract_metric "${key}" "${file}")"
    if [ -n "${value}" ]; then echo "${value}"; return 0; fi
  done
  echo ""
}

echo "=== Rotary+QWAB XSum eval | ckpt: ${BEST_CKPT} ==="

for i in "${!XSUM_BLOCK_SIZES[@]}"; do
  block_size="${XSUM_BLOCK_SIZES[$i]}"
  batch_size="${XSUM_BATCH_SIZES[$i]}"
  OUT_DIR="${OUT_ROOT_DIR}/bs_${block_size}"
  mkdir -p "${OUT_DIR}"
  echo "  -> block_size=${block_size} batch=${batch_size}"

  python -m torch.distributed.run --nproc_per_node=4 --master_port="${XSUM_PORT}" ./run_clm.py \
    --model_type gpt2 \
    --tokenizer_name gpt2 \
    --model_name_or_path "${BEST_CKPT}" \
    --pe_method rotary \
    --attn_implementation eager \
    --wavelet_router True \
    --qwab_train_block_size 512 \
    --router_band_num 8 \
    --scale_range 0 16 \
    --wavelet_mode logit_bias_ctxscale_shift_v0 \
    --wavelet_baseline_use False \
    --use_beta_modulation False \
    --use_soft_wavelet_fox False \
    --single_A_B True \
    --num_harmonics 1 \
    --share_freq_across_heads True \
    --block_size "${block_size}" \
    --dataset_name xsum \
    --dataset_config_name default \
    --validation_file "${XSUM_VALIDATION_FILE}" \
    --do_eval \
    --per_device_eval_batch_size "${batch_size}" \
    --output_dir "${OUT_DIR}" \
    --overwrite_output_dir \
    --xsum_bucket_size 512 \
    --xsum_bucket_apply_to eval_test \
    --load_best_model_at_end False \
    --seed 42

  eval_results_file="${OUT_DIR}/eval_results.json"
  if [ ! -f "${eval_results_file}" ]; then
    echo "[WARN] Missing ${eval_results_file}; skip block=${block_size}"
    continue
  fi

  rouge1="$(extract_first_metric "${eval_results_file}" eval_rouge1 rouge1)"
  rouge2="$(extract_first_metric "${eval_results_file}" eval_rouge2 rouge2)"
  rougeL="$(extract_first_metric "${eval_results_file}" eval_rougeL rougeL)"
  bertscore="$(extract_first_metric "${eval_results_file}" eval_bertscore bertscore)"
  count="$(extract_first_metric "${eval_results_file}" eval_count count)"
  eval_loss="$(extract_first_metric "${eval_results_file}" eval_loss loss)"
  ts="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

  : "${rouge1:=nan}" "${rouge2:=nan}" "${rougeL:=nan}" "${bertscore:=nan}" "${count:=nan}" "${eval_loss:=nan}"

  echo "best,${BEST_CKPT},${block_size},${batch_size},${rouge1},${rouge2},${rougeL},${bertscore},${count},${eval_loss},${ts}" >> "${XSUM_CSV}"
  echo "  [DONE] L${block_size}: rouge1=${rouge1} rougeL=${rougeL} bertscore=${bertscore}"
done

echo "=== Done. CSV: ${XSUM_CSV} ==="
