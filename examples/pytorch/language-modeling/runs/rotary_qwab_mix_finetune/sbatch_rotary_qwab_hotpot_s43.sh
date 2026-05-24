#!/bin/bash
#SBATCH --job-name=rqwab_hp_s43
#SBATCH --output=/cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling/hotpot_long/logs/%j_rotary_qwab_hotpot_s43.txt
#SBATCH --partition=gpu_long
#SBATCH --gres=gpu:a100-80:2
#SBATCH --nodelist=elm44
#SBATCH --time=36:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8

# Eval: Rotary+QWAB s42/s43/s44 on HotpotQA-Long uniform L512/L2048/L4096
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
JSONL=${BASE}/hotpot_long/data/hotpot_long_dev_uniform.jsonl
CFG=${QWAB_BASE}/supply_model.cfg
CKPT_STEP=15000
cd "${BASE}"

run_eval() {
  local SEED=$1 SEED_NUM=$2 BLOCK_SIZE=$3
  local CKPT=${QWAB_BASE}/${SEED}/checkpoint-${CKPT_STEP}
  local OUTPUT=${BASE}/hotpot_long/results_uniform/rotary_qwab_pe_${SEED}_ckpt${CKPT_STEP}/L${BLOCK_SIZE}
  mkdir -p "${OUTPUT}"
  echo "=== rotary_qwab ${SEED} L${BLOCK_SIZE} ==="
  MASTER_PORT=$(( 17000 + SLURM_JOB_ID % 1000 + BLOCK_SIZE % 100 + SEED_NUM ))
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
    --dataset_name hotpot_qa --dataset_config_name distractor \
    --hotpot_long_jsonl "${JSONL}" \
    --hotpot_long_lengths "${BLOCK_SIZE}" \
    --do_eval \
    --block_size "${BLOCK_SIZE}" \
    --per_device_eval_batch_size 4 \
    --output_dir "${OUTPUT}" --overwrite_output_dir \
    --logging_dir "${OUTPUT}/log" \
    --seed ${SEED_NUM} --load_best_model_at_end False
  python3 -c "import json; d=json.load(open('${OUTPUT}/eval_results.json')); print(f'rotary_qwab ${SEED} L${BLOCK_SIZE}: F1={d[\"eval_f1\"]:.4f} EM={d[\"eval_exact_match\"]:.4f}')"
  echo "Done: ${OUTPUT}"
}

for SEED_INFO in "s43:43"; do
  SEED=${SEED_INFO%%:*}
  SEED_NUM=${SEED_INFO##*:}
  for BSIZE in 512 2048 4096; do
    run_eval "${SEED}" "${SEED_NUM}" "${BSIZE}"
  done
done

echo "=== All rotary_qwab HotpotQA-Long evals done ==="
