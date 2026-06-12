#!/bin/bash
#SBATCH --job-name=pat195_xsum1536
#SBATCH --output=/cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling/hotpot_long/logs/%j_pat195_xsum_rouge.txt
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
        --summary "xsum L1536 entmax eval"
}
trap '_slack $?' EXIT

set -euxo pipefail

CHECKPOINT="${1:?CHECKPOINT required}"
ENTMAX_CFG="${2:?ENTMAX_CFG required}"
OUT_DIR="${3:?OUT_DIR required}"
PATH_ATTN_IMPL="${4:-}"
EVAL_BATCH_SIZE="${5:-8}"
SEED="${6:-42}"
XSUM_VALIDATION_FILE="${7:-/cl/work5/hongyu-s/fact-check-summarization/xsum_test_filter_level2_official_style.jsonl}"

export PYTHONPATH=/project/nlp-work5/hongyu-s/transformers/src:/cl/work5/hongyu-s/flash-linear-attention${PYTHONPATH:+:${PYTHONPATH}}
export SKIP_FENICE=1
export SKIP_SUMMAC=1
export WANDB_DISABLED=true
export TRITON_CACHE_DIR="/tmp/triton_cache_${SLURM_JOB_ID}"

WORKDIR=/cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling
PYTHON_BIN=/cl/work5/hongyu-s/conda/envs/latest_transformers/bin/python

mkdir -p "${WORKDIR}/hotpot_long/logs" "${OUT_DIR}"

RUNTIME_CFG="${OUT_DIR}/runtime_supply_model.cfg"
cp "${ENTMAX_CFG}" "${RUNTIME_CFG}"
if [ -n "${PATH_ATTN_IMPL}" ]; then
    printf 'path_attn_impl=%s\n' "${PATH_ATTN_IMPL}" >> "${RUNTIME_CFG}"
fi

CKPT_CONFIG="${CHECKPOINT}/config.json"
if [ ! -f "${CKPT_CONFIG}" ]; then
    echo "checkpoint config not found: ${CKPT_CONFIG}" >&2
    exit 1
fi

eval "$(
"${PYTHON_BIN}" - "${CKPT_CONFIG}" <<'PY'
import json
import shlex
import sys

cfg = json.load(open(sys.argv[1], "r", encoding="utf-8"))

def as_bool(v, default=False):
    if v is None:
        v = default
    if isinstance(v, str):
        v = v.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(v)

def emit(k, v):
    print(f"{k}={shlex.quote(str(v))}")

def b(name, default):
    return "true" if as_bool(cfg.get(name, default), default) else "false"

scale_range = cfg.get("scale_range", [0, 16])
if not isinstance(scale_range, list) or len(scale_range) < 2:
    scale_range = [0, 16]

emit("ATTN_IMPLEMENTATION", cfg.get("attn_implementation", "path_attn"))
emit("SHARE_FREQ_ACROSS_HEADS", b("share_freq_across_heads", True))
emit("PATH_USE_QK_NORM", b("path_use_qk_norm", False))
emit("PATH_USE_LOW_RANK_W", b("path_use_low_rank_w", True))
emit("PATH_USE_W_SHORTCONV", b("path_use_w_shortconv", False))
emit("PATH_CONV_SIZE", int(cfg.get("path_conv_size", 3)))
emit("PATH_CONV_BIAS", b("path_conv_bias", False))
emit("NUM_HARMONICS", int(cfg.get("num_harmonics", 1)))
emit("SINGLE_A_B", b("single_A_B", True))
emit("USE_BETA_MODULATION", b("use_beta_modulation", False))
emit("USE_SOFT_WAVELET_FOX", b("use_soft_wavelet_fox", False))
emit("WAVELET_MODE", cfg.get("wavelet_mode", "additive"))
emit("WAVELET_BASELINE_USE", b("wavelet_baseline_use", False))
emit("INIT_THETA", float(cfg.get("init_theta", 0.847)))
emit("USE_FORGET_GATE", b("use_forget_gate", False))
emit("SPECTRAL_LOSS_COE", float(cfg.get("spectral_loss_coe", 0.1)))
emit("TEMP_LOSS_COE", float(cfg.get("temp_loss_coe", 0.0)))
emit("SCALE_RANGE_0", int(scale_range[0]))
emit("SCALE_RANGE_1", int(scale_range[1]))
emit("QK_ROTATION", b("qk_rotation", False))
emit("ABLATE_SWITCH", b("ablate_switch", False))
emit("WAVELET_ROUTER", b("wavelet_router", False))
emit("ROUTER_HIDDEN_DIM", int(cfg.get("router_hidden_dim", 32)))
emit("ROUTER_BAND_NUM", int(cfg.get("router_band_num", 8)))
emit("REL_SELECTION", cfg.get("rel_selection", "all"))
PY
)"

PORT=$((19600 + (${SLURM_JOB_ID:-0} % 1000)))

cd "${WORKDIR}"

"${PYTHON_BIN}" -m torch.distributed.run --nproc_per_node=1 --master_port="${PORT}" ./run_clm_v_arc.py \
    --model_type gpt2 \
    --tokenizer_name gpt2 \
    --model_name_or_path "${CHECKPOINT}" \
    --share_freq_across_heads "${SHARE_FREQ_ACROSS_HEADS}" \
    --per_device_eval_batch_size "${EVAL_BATCH_SIZE}" \
    --block_size 1536 \
    --dataset_name xsum \
    --dataset_config_name default \
    --validation_file "${XSUM_VALIDATION_FILE}" \
    --do_eval \
    --output_dir "${OUT_DIR}" \
    --overwrite_output_dir \
    --attn_implementation "${ATTN_IMPLEMENTATION}" \
    --path_use_qk_norm "${PATH_USE_QK_NORM}" \
    --path_use_low_rank_w "${PATH_USE_LOW_RANK_W}" \
    --path_use_w_shortconv "${PATH_USE_W_SHORTCONV}" \
    --path_conv_size "${PATH_CONV_SIZE}" \
    --path_conv_bias "${PATH_CONV_BIAS}" \
    --num_harmonics "${NUM_HARMONICS}" \
    --single_A_B "${SINGLE_A_B}" \
    --use_beta_modulation "${USE_BETA_MODULATION}" \
    --use_soft_wavelet_fox "${USE_SOFT_WAVELET_FOX}" \
    --wavelet_mode "${WAVELET_MODE}" \
    --wavelet_baseline_use "${WAVELET_BASELINE_USE}" \
    --init_theta "${INIT_THETA}" \
    --use_forget_gate "${USE_FORGET_GATE}" \
    --spectral_loss_coe "${SPECTRAL_LOSS_COE}" \
    --temp_loss_coe "${TEMP_LOSS_COE}" \
    --scale_range "${SCALE_RANGE_0}" "${SCALE_RANGE_1}" \
    --qk_rotation "${QK_ROTATION}" \
    --ablate_switch "${ABLATE_SWITCH}" \
    --wavelet_router "${WAVELET_ROUTER}" \
    --router_hidden_dim "${ROUTER_HIDDEN_DIM}" \
    --router_band_num "${ROUTER_BAND_NUM}" \
    --rel_selection "${REL_SELECTION}" \
    --cfg_path "${RUNTIME_CFG}" \
    --seed "${SEED}"
