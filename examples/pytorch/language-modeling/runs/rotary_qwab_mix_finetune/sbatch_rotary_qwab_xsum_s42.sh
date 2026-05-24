#!/bin/bash
#SBATCH --job-name=rqwab_xsum_s42
#SBATCH --output=/cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling/hotpot_long/logs/%j_rotary_qwab_xsum_s42.txt
#SBATCH --partition=gpu_long
#SBATCH --gres=gpu:a100-80:2
#SBATCH --nodelist=elm44
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8

# Eval: Rotary+QWAB s42/s43/s44 on Filtered XSum L512/L1024/L1536
# Model flags must match training (runs/rotary_qwab_mix_finetune/train_rotary_qwab_mix_s4*.sh)

set -euxo pipefail

if [ -f /home/is/hongyu-s/miniconda3/etc/profile.d/conda.sh ]; then
  set +u; source /home/is/hongyu-s/miniconda3/etc/profile.d/conda.sh; conda activate latest_transformers; set -u
fi

export PYTHONPATH=/project/nlp-work5/hongyu-s/transformers/src${PYTHONPATH:+:${PYTHONPATH}}
export HF_HOME=/cl/work5/hongyu-s/huggingfac
export HF_DATASETS_CACHE=/cl/work5/hongyu-s/huggingfac/datasets
export WANDB_DISABLED=true
export WANDB_MODE=disabled

BASE=/cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling
QWAB_BASE=${BASE}/runs/rotary_qwab_mix_finetune
XSUM_VALIDATION_FILE=/cl/work5/hongyu-s/fact-check-summarization/xsum_test_filter_level2_official_style.jsonl
CFG=${QWAB_BASE}/supply_model.cfg
CKPT_STEP=15000
RUN_TAG="${SLURM_JOB_ID}_$(date -u +%Y%m%dT%H%M%SZ)"
cd "${BASE}"

run_xsum() {
  local SEED=$1 SEED_NUM=$2 BLOCK_SIZE=$3
  local CKPT=${QWAB_BASE}/${SEED}/checkpoint-${CKPT_STEP}
  local OUT_DIR=${QWAB_BASE}/${SEED}/ckpt_eval_xsum_rouge/xsum_L${BLOCK_SIZE}_${RUN_TAG}
  mkdir -p "${OUT_DIR}"
  echo "=== rotary_qwab ${SEED} XSum L${BLOCK_SIZE} ==="
  MASTER_PORT=$(( 18000 + SLURM_JOB_ID % 1000 + BLOCK_SIZE % 100 + SEED_NUM ))
  python -m torch.distributed.run --nproc_per_node=2 --master_port=${MASTER_PORT} ./run_clm.py \
    --model_type gpt2 --tokenizer_name gpt2 \
    --model_name_or_path "${CKPT}" \
    --pe_method rotary \
    --attn_implementation eager \
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
    --cfg_path "${CFG}" \
    --dataset_name xsum --dataset_config_name default \
    --validation_file "${XSUM_VALIDATION_FILE}" \
    --do_eval \
    --block_size "${BLOCK_SIZE}" \
    --per_device_eval_batch_size 4 \
    --xsum_bucket_size 512 \
    --xsum_bucket_apply_to eval_test \
    --output_dir "${OUT_DIR}" --overwrite_output_dir \
    --logging_dir "${OUT_DIR}/log" \
    --seed ${SEED_NUM} --load_best_model_at_end False
  python3 -c "import json; d=json.load(open('${OUT_DIR}/eval_results.json')); print(f'rotary_qwab ${SEED} XSum L${BLOCK_SIZE}: ROUGE-L={d.get(\"eval_rougeL\",d.get(\"eval_rouge_l\",\"n/a\"))}')"
  echo "Done: ${OUT_DIR}"
}

for SEED_INFO in "s42:42"; do
  SEED=${SEED_INFO%%:*}
  SEED_NUM=${SEED_INFO##*:}
  for BSIZE in 512 1024 1536; do
    run_xsum "${SEED}" "${SEED_NUM}" "${BSIZE}"
  done
done

echo "=== All rotary_qwab XSum evals done ==="
