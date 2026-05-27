#!/bin/bash
#SBATCH --job-name=hp_nope_qwab
#SBATCH --output=/cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling/runs/small_nope_qwab_10ep_s42/ckpt_eval_hm/%j_hotpot_nope_qwab_s42.txt
#SBATCH --partition=gpu_long
#SBATCH --nodelist=elm73
#SBATCH --gres=gpu:6000:4
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4

# PAT-165: HotpotQA-Long eval for NoPE+QWAB small model, seed=42
# Re-run after beta clamp fix (f4a1a40f68) + _train_T cleanup.
# use_cache=False ensures QWAB bias is active during decode (no T_q!=T_k skip).
# block_sizes: 512 / 2048 / 4096

set -euxo pipefail

if [ -f /home/is/hongyu-s/miniconda3/etc/profile.d/conda.sh ]; then
  set +u; source /home/is/hongyu-s/miniconda3/etc/profile.d/conda.sh; conda activate latest_transformers; set -u
fi

export PYTHONPATH=/project/nlp-work5/hongyu-s/transformers/src${PYTHONPATH:+:${PYTHONPATH}}
export HF_HOME=/cl/work5/hongyu-s/huggingfac
export HF_DATASETS_CACHE=/cl/work5/hongyu-s/huggingfac/datasets
export WANDB_DISABLED=true
export WANDB_MODE=disabled

WORKDIR=/cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling
cd "${WORKDIR}"

RUN_DIR="${WORKDIR}/runs/small_nope_qwab_10ep_s42"
BEST_CKPT="${RUN_DIR}/checkpoint-15000"
JSONL="${WORKDIR}/hotpot_long/data/hotpot_long_dev_uniform.jsonl"
RESULT_DIR="${RUN_DIR}/ckpt_eval_hm_fixedbeta"
mkdir -p "${RESULT_DIR}"

echo "=== NoPE+QWAB HotpotQA-Long eval (beta fix + no_cache) | ckpt: ${BEST_CKPT} ==="

JOB_PORT_OFFSET=$(( ${SLURM_JOB_ID:-0} % 10000 ))

for BLOCK_SIZE in 512 2048 4096; do
  OUT_DIR="${RESULT_DIR}/hotpot_L${BLOCK_SIZE}"
  mkdir -p "${OUT_DIR}"
  MASTER_PORT=$(( 15000 + JOB_PORT_OFFSET ))
  echo "  -> block_size=${BLOCK_SIZE}"

  python -m torch.distributed.run --nproc_per_node=4 --master_port="${MASTER_PORT}" ./run_clm.py \
    --model_type gpt2 \
    --tokenizer_name gpt2 \
    --model_name_or_path "${BEST_CKPT}" \
    --pe_method no_pe \
    --attn_implementation eager \
    --use_qwab_bias True \
    --qwab_train_block_size 512 \
    --wavelet_router False \
    --router_band_num 8 \
    --scale_range 0 16 \
    --wavelet_mode logit_bias_ctxscale_shift_v0 \
    --wavelet_baseline_use False \
    --use_beta_modulation False \
    --use_soft_wavelet_fox False \
    --single_A_B True \
    --num_harmonics 1 \
    --share_freq_across_heads True \
    --dataset_name hotpot_qa \
    --dataset_config_name distractor \
    --hotpot_long_jsonl "${JSONL}" \
    --hotpot_long_lengths "${BLOCK_SIZE}" \
    --eval_generate_no_cache True \
    --do_eval \
    --block_size "${BLOCK_SIZE}" \
    --per_device_eval_batch_size 8 \
    --output_dir "${OUT_DIR}" \
    --overwrite_output_dir \
    --logging_dir "${OUT_DIR}/log" \
    --load_best_model_at_end False \
    --seed 42

  echo "  [DONE] L${BLOCK_SIZE}: $(cat ${OUT_DIR}/eval_results.json 2>/dev/null || echo 'no results')"
done

echo "=== Done: NoPE+QWAB HotpotQA eval (beta fix). Results: ${RESULT_DIR} ==="
