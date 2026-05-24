#!/bin/bash
# run_eval_ruler_eager_a6000.sh
# Generic RULER eval entry for eager-attn models.
#
# Usage:
#   sbatch hotpot_long/run_eval_ruler_eager_a6000.sh \
#       <CHECKPOINT> <MODEL_NAME> <BLOCK_SIZE> <CFG_PATH> <RULER_JSONL> [PE_METHOD]
#
# RULER JSONL (official-compatible):
#   each line should contain at least:
#     - input   (prompt)
#     - outputs (gold answer list/string)
#   optional:
#     - length, ruler_config (for per-slice reporting)
#
# PE_METHOD: rotary | alibi | no_pe  (default: no_pe)

#SBATCH --job-name=ruler_eval
#SBATCH --output=/cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling/hotpot_long/logs/%j_ruler_eval_a6000.txt
#SBATCH --partition=gpu_long
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:a6000:4
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8

set -euxo pipefail

CHECKPOINT="${1:?CHECKPOINT required}"
MODEL_NAME="${2:?MODEL_NAME required}"
BLOCK_SIZE="${3:?BLOCK_SIZE required}"
CFG_PATH="${4:?CFG_PATH required}"
RULER_JSONL="${5:?RULER_JSONL required}"
PE_METHOD="${6:-no_pe}"

if [ -f /home/is/hongyu-s/miniconda3/etc/profile.d/conda.sh ]; then
  set +u
  source /home/is/hongyu-s/miniconda3/etc/profile.d/conda.sh
  conda activate latest_transformers
  set -u
fi

export PYTHONPATH=/project/nlp-work5/hongyu-s/transformers/src${PYTHONPATH:+:${PYTHONPATH}}
export HF_HOME=/cl/work5/hongyu-s/huggingfac
export HF_DATASETS_CACHE=/cl/work5/hongyu-s/huggingfac/datasets

LANG_MODEL_DIR="/cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling"
HOTPOT_LONG_DIR="${LANG_MODEL_DIR}/hotpot_long"

echo "Node: $(hostname) | Model: ${MODEL_NAME} | block_size: ${BLOCK_SIZE} | pe: ${PE_METHOD}"
echo "RULER JSONL: ${RULER_JSONL}"

mkdir -p "${HOTPOT_LONG_DIR}/logs"
OUTPUT_DIR="${HOTPOT_LONG_DIR}/results_ruler/${MODEL_NAME}/L${BLOCK_SIZE}"
mkdir -p "${OUTPUT_DIR}"

cd "${LANG_MODEL_DIR}"

MASTER_PORT=$(( 12000 + SLURM_JOB_ID % 10000 ))
python -m torch.distributed.run --nproc_per_node=4 --master_port=${MASTER_PORT} ./run_clm.py \
    --model_type gpt2 \
    --tokenizer_name gpt2 \
    --model_name_or_path "${CHECKPOINT}" \
    --pe_method "${PE_METHOD}" \
    --attn_implementation eager \
    --cfg_path "${CFG_PATH}" \
    --dataset_name ruler \
    --validation_file "${RULER_JSONL}" \
    --ruler_input_field input \
    --ruler_output_field outputs \
    --ruler_task_field ruler_config \
    --ruler_length_field length \
    --do_eval \
    --block_size "${BLOCK_SIZE}" \
    --per_device_eval_batch_size 4 \
    --output_dir "${OUTPUT_DIR}" \
    --overwrite_output_dir \
    --logging_dir "${OUTPUT_DIR}/log" \
    --seed 42 \
    --share_freq_across_heads True \
    --num_harmonics 1 \
    --single_A_B True \
    --use_beta_modulation False \
    --use_soft_wavelet_fox False \
    --wavelet_baseline_use False \
    --use_forget_gate False \
    --qk_rotation False \
    --ablate_switch False \
    --wavelet_router False \
    --scale_range 0 16 \
    --rel_selection all \
    --load_best_model_at_end False

echo "=== Done: ${MODEL_NAME} RULER L${BLOCK_SIZE} ==="
echo "Results: ${OUTPUT_DIR}"
