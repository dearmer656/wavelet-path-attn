#!/bin/bash
#SBATCH --job-name=P160_hln_hq2
#SBATCH --output=/cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling/hotpot_long/logs/%j_pat160_hidden_ln_s42_hotpot_L2048_4096.txt
#SBATCH --partition=gpu_long
#SBATCH --gres=gpu:a100:4
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8

# PAT-160: HotpotQA L2048 + L4096 only (L512 already done; elm52 NODE_FAIL)

set -euxo pipefail
if [ -f /home/is/hongyu-s/miniconda3/etc/profile.d/conda.sh ]; then
  set +u; source /home/is/hongyu-s/miniconda3/etc/profile.d/conda.sh; conda activate latest_transformers; set -u
fi
export PYTHONPATH=/project/nlp-work5/hongyu-s/transformers/src${PYTHONPATH:+:${PYTHONPATH}}
export HF_HOME=/cl/work5/hongyu-s/huggingfac
export HF_DATASETS_CACHE=/cl/work5/hongyu-s/huggingfac/datasets
export WANDB_DISABLED=true

BASE=/cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling
CKPT="${BASE}/runs/mix_medium_owt_hidden_ln_10ep_s42/checkpoint-15000"
CFG="${BASE}/runs/mix_medium_owt_hidden_ln_10ep_s42/supply_model.cfg"
JSONL="${BASE}/hotpot_long/data/hotpot_long_dev_uniform.jsonl"
cd "${BASE}"

for L in 2048 4096; do
  OUTPUT="${BASE}/hotpot_long/results_uniform/mix_medium_owt_hidden_ln_10ep_s42_ckpt15000/L${L}"
  mkdir -p "${OUTPUT}"
  echo "=== Hidden-LN s42 L${L} ==="
  MASTER_PORT=$(( 16200 + SLURM_JOB_ID % 1000 ))
  python -m torch.distributed.run --nproc_per_node=4 --master_port=${MASTER_PORT} ./run_clm.py \
    --model_type gpt2 --tokenizer_name gpt2 \
    --model_name_or_path "${CKPT}" \
    --attn_implementation path_attn --path_attn_impl pytorch \
    --cfg_path "${CFG}" \
    --dataset_name hotpot_qa --dataset_config_name distractor \
    --hotpot_long_jsonl "${JSONL}" \
    --hotpot_long_lengths ${L} \
    --do_eval --block_size ${L} \
    --per_device_eval_batch_size 1 \
    --output_dir "${OUTPUT}" --overwrite_output_dir \
    --logging_dir "${OUTPUT}/log" \
    --seed 42 --load_best_model_at_end False \
    --path_use_qk_norm false --path_use_low_rank_w true \
    --path_use_w_shortconv false --path_conv_size 3 --path_conv_bias false \
    --num_harmonics 1 --single_A_B True \
    --use_beta_modulation False --use_soft_wavelet_fox False \
    --wavelet_baseline_use False --use_forget_gate False \
    --qk_rotation False --ablate_switch False \
    --wavelet_router True --router_hidden_dim 32 --router_band_num 8 \
    --rel_selection all --wavelet_mode logit_bias_ctxscale_shift_v0 \
    --scale_range 0 16 --share_freq_across_heads True \
    --load_best_model_at_end False
  python3 -c "import json; d=json.load(open('${OUTPUT}/eval_results.json')); print(f'Hidden-LN s42 L${L}: F1={d[\"eval_f1\"]:.4f} EM={d[\"eval_em\"]:.4f}')"
done

echo "=== Done: PAT-160 Hidden-LN s42 HotpotQA L2048/L4096 ==="
