#!/bin/bash
#SBATCH --job-name=pat195_hp_u4096
#SBATCH --output=/cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling/hotpot_long/logs/%j_pat195_hotpot_uniform.txt
#SBATCH --partition=gpu_long
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:a6000:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4

_slack() {
    python3 /project/nlp-work5/hongyu-s/gate1/scripts/notify_slack.py \
        --exit-code "$?" \
        --job-id "${SLURM_JOB_ID}" \
        --node "${SLURMD_NODENAME}" \
        --issue "PAT-195" \
        --gpu "A6000x1" \
        --summary "hotpot L4096 entmax eval"
}
trap '_slack $?' EXIT

set -euxo pipefail

CHECKPOINT="${1:?CHECKPOINT required}"
ENTMAX_CFG="${2:?ENTMAX_CFG required}"
OUT_DIR="${3:?OUT_DIR required}"
PATH_ATTN_IMPL="${4:-}"

export PYTHONPATH=/project/nlp-work5/hongyu-s/transformers/src:/cl/work5/hongyu-s/flash-linear-attention${PYTHONPATH:+:${PYTHONPATH}}
export WANDB_DISABLED=true

WORKDIR=/cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling
PYTHON_BIN=/cl/work5/hongyu-s/conda/envs/path_ok/bin/python3
JSONL="${WORKDIR}/hotpot_long/data/hotpot_long_dev_uniform.jsonl"

mkdir -p "${WORKDIR}/hotpot_long/logs" "${OUT_DIR}"

RUNTIME_CFG="${OUT_DIR}/runtime_supply_model.cfg"
cp "${ENTMAX_CFG}" "${RUNTIME_CFG}"
if [ -n "${PATH_ATTN_IMPL}" ]; then
    printf 'path_attn_impl=%s\n' "${PATH_ATTN_IMPL}" >> "${RUNTIME_CFG}"
fi

cd "${WORKDIR}"

"${PYTHON_BIN}" hotpot_long/eval_hotpot_long.py \
    --model-path "${CHECKPOINT}" \
    --hotpot-long-jsonl "${JSONL}" \
    --tokenizer gpt2 \
    --output-dir "${OUT_DIR}" \
    --batch-size 2 \
    --target-lengths 4096 \
    --max-new-tokens 32 \
    --device cuda \
    --cfg-path "${RUNTIME_CFG}"
