#!/bin/bash
# Evaluate QWAB RULER-512 finetuned checkpoint on official RULER sets (L2048, L4096), full-size.

#SBATCH --job-name=ev_ruler_qw_of
#SBATCH --output=/cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling/hotpot_long/logs/%j_ev_ruler_qwab_official_elm55.txt
#SBATCH --partition=gpu_long
#SBATCH --nodelist=elm55
#SBATCH --gres=gpu:3090:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=16:00:00

set -euxo pipefail

BASE="/cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling"
cd "${BASE}"

if [ -f /home/is/hongyu-s/miniconda3/etc/profile.d/conda.sh ]; then
  set +u
  source /home/is/hongyu-s/miniconda3/etc/profile.d/conda.sh
  conda activate latest_transformers
  set -u
fi

export PYTHONPATH=/cl/work5/hongyu-s/transformers/src${PYTHONPATH:+:${PYTHONPATH}}
export HF_HOME=/cl/work5/hongyu-s/huggingfac
export HF_DATASETS_CACHE=/cl/work5/hongyu-s/huggingfac/datasets
export TOKENIZERS_PARALLELISM=false
export WANDB_DISABLED=true

CKPT="${BASE}/runs/ruler_ft_len_generalize_20260526/path_qwab_s42_ruler512_ft1k/checkpoint-1000"
CFG="${BASE}/runs/mix_medium_owt_dd_10ep/supply_model.cfg"
DATA_2048="${BASE}/hotpot_long/data/ruler_official/ruler_official_eval_L2048.jsonl"
DATA_4096="${BASE}/hotpot_long/data/ruler_official/ruler_official_eval_L4096.jsonl"

run_one() {
  local tag="$1"
  local data_file="$2"
  local bs="$3"
  local out_dir="${BASE}/hotpot_long/results_ruler_ft_official/path_qwab_s42_ruler512_ft1k/${tag}"
  mkdir -p "${out_dir}"

  python ./run_clm.py \
    --model_type gpt2 \
    --tokenizer_name gpt2 \
    --model_name_or_path "${CKPT}" \
    --pe_method vanilla \
    --attn_implementation path_attn \
    --cfg_path "${CFG}" \
    --dataset_name ruler \
    --validation_file "${data_file}" \
    --ruler_input_field input \
    --ruler_output_field outputs \
    --ruler_task_field ruler_config \
    --ruler_length_field length \
    --ruler_eval_mode generate \
    --ruler_max_new_tokens 32 \
    --ruler_num_beams 1 \
    --ruler_do_sample false \
    --ruler_predictions_file "pred_${tag}.jsonl" \
    --do_eval \
    --block_size "${bs}" \
    --per_device_eval_batch_size 1 \
    --output_dir "${out_dir}" \
    --overwrite_output_dir \
    --logging_dir "${out_dir}/log" \
    --seed 42 \
    --load_best_model_at_end False
}

run_one "official_len2048" "${DATA_2048}" 2048
run_one "official_len4096" "${DATA_4096}" 4096

echo "Done. Results saved under ${BASE}/hotpot_long/results_ruler_ft_official/path_qwab_s42_ruler512_ft1k/"
