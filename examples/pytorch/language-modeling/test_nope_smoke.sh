#!/bin/bash
#SBATCH --job-name=nope_smoke
#SBATCH --output=/cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling/runs/nope_smoke_test/%j_nope_smoke.txt
#SBATCH --partition=gpu_long
#SBATCH --gres=gpu:3090:1
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4

# PAT-164: NoPE smoke test — verifies pe_method=no_pe + eager attn work end-to-end
#   1. Init model from HuggingFace gpt2 (no NoPE pretrain needed)
#   2. Run 10 training steps on wikitext
#   3. Verify saved config has pe_method=no_pe and no wpe in state dict
# Branch: hongyusaatitech/pat-164-sine-basis-ablation

set -euxo pipefail

if [ -f /home/is/hongyu-s/miniconda3/etc/profile.d/conda.sh ]; then
  set +u; source /home/is/hongyu-s/miniconda3/etc/profile.d/conda.sh; conda activate latest_transformers; set -u
fi

WORKDIR=/cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling
cd "${WORKDIR}"

export PYTHONPATH=/project/nlp-work5/hongyu-s/transformers/src${PYTHONPATH:+:${PYTHONPATH}}
export HF_HOME=/cl/work5/hongyu-s/huggingfac
export HF_DATASETS_CACHE=/cl/work5/hongyu-s/huggingfac/datasets
export WANDB_DISABLED=true
export WANDB_MODE=disabled

RUN_OUT="${WORKDIR}/runs/nope_smoke_test"
mkdir -p "${RUN_OUT}"

echo "===== [1] Verify model init: pe_method=no_pe should have NO wpe ====="
python3 - <<'PY'
import sys
sys.path.insert(0, '/project/nlp-work5/hongyu-s/transformers/src')
from transformers import GPT2Config
from transformers.models.gpt2.modeling_gpt2 import GPT2Model

cfg = GPT2Config(
    n_positions=1024, n_embd=768, n_layer=12, n_head=12,
    pe_method='no_pe', attn_implementation='eager',
    wavelet_router=False,
    wavelet_mode='logit_bias_ctxscale_shift_v0',
    wavelet_baseline_use=False,
    scale_range=[0, 16],
    router_band_num=8,
    use_beta_modulation=False,
    use_soft_wavelet_fox=False,
    single_A_B=True,
    num_harmonics=1,
    share_freq_across_heads=True,
    scale_type='none',
    analyzer=False,
    block_size=512,
)
cfg._attn_implementation = 'eager'

model = GPT2Model(cfg)

# Check 1: no wpe
assert not hasattr(model, 'wpe'), "FAIL: wpe should NOT exist for pe_method=no_pe"
print("[PASS] no wpe embedding")

# Check 2: attention class is GPT2Attention (not PaTH)
attn_class = type(model.h[0].attn).__name__
assert attn_class == 'GPT2Attention', f"FAIL: expected GPT2Attention, got {attn_class}"
print(f"[PASS] attention class = {attn_class}")

# Check 3: no router (wavelet_router=False)
assert model.h[0].attn.router is None, "FAIL: router should be None when wavelet_router=False"
print("[PASS] router is None")

# Check 4: _skip_wavelet_decay_table = True
assert model._skip_wavelet_decay_table, "FAIL: _skip_wavelet_decay_table should be True for logit_bias_ctxscale_shift_v0"
print("[PASS] _skip_wavelet_decay_table = True")

# Check 5: d_m (decay table) is None
assert model.d_m is None, f"FAIL: d_m should be None, got {model.d_m}"
print("[PASS] d_m is None (no wavelet decay table)")

print("\n===== ALL INIT CHECKS PASSED =====")
PY

echo "===== [2] Quick training run: 10 steps, gpt2 init, pe_method=no_pe ====="
python -m torch.distributed.run --nproc_per_node=1 --master_port=29900 ./run_clm.py \
  --model_type gpt2 \
  --config_name gpt2 \
  --tokenizer_name gpt2 \
  --dataset_name wikitext \
  --dataset_config_name wikitext-103-raw-v1 \
  --pe_method no_pe \
  --attn_implementation eager \
  --block_size 128 \
  --per_device_train_batch_size 4 \
  --per_device_eval_batch_size 4 \
  --gradient_accumulation_steps 1 \
  --max_steps 10 \
  --learning_rate 1e-4 \
  --do_train \
  --do_eval \
  --eval_strategy steps \
  --eval_steps 10 \
  --logging_steps 1 \
  --save_steps 10 \
  --save_total_limit 1 \
  --output_dir "${RUN_OUT}" \
  --overwrite_output_dir \
  --wavelet_router False \
  --wavelet_mode logit_bias_ctxscale_shift_v0 \
  --scale_range 0 16 \
  --router_band_num 8 \
  --use_beta_modulation False \
  --use_soft_wavelet_fox False \
  --wavelet_baseline_use False \
  --single_A_B True \
  --num_harmonics 1 \
  --share_freq_across_heads True \
  --cfg_path "${RUN_OUT}/supply_model.cfg" \
  --seed 42

echo "===== [3] Verify saved config has pe_method=no_pe and no wpe ====="
python3 - <<PY
import json, sys

cfg_path = "${RUN_OUT}/checkpoint-10/config.json"
try:
    cfg = json.load(open(cfg_path))
except FileNotFoundError:
    cfg_path = "${RUN_OUT}/config.json"
    cfg = json.load(open(cfg_path))

pe = cfg.get('pe_method', 'NOT_FOUND')
attn = cfg.get('attn_implementation', cfg.get('_attn_implementation', 'NOT_FOUND'))
print(f"pe_method in saved config   : {pe}")
print(f"attn_implementation in saved: {attn}")
assert pe == 'no_pe', f"FAIL: expected pe_method=no_pe, got {pe}"
assert attn == 'eager', f"FAIL: expected eager, got {attn}"
print("[PASS] config correctly saved pe_method=no_pe + attn_implementation=eager")

import torch, os
ckpt_dir = os.path.dirname(cfg_path)
# Check no wpe in state dict
for fname in ['model.safetensors', 'pytorch_model.bin']:
    fpath = os.path.join(ckpt_dir, fname)
    if os.path.exists(fpath):
        if fname.endswith('.safetensors'):
            from safetensors import safe_open
            with safe_open(fpath, framework='pt') as f:
                keys = list(f.keys())
        else:
            sd = torch.load(fpath, map_location='cpu')
            keys = list(sd.keys())
        wpe_keys = [k for k in keys if 'wpe' in k]
        assert len(wpe_keys) == 0, f"FAIL: found wpe in state dict: {wpe_keys}"
        print(f"[PASS] no wpe in state dict ({fname})")
        break

print("\n===== ALL CONFIG CHECKS PASSED =====")
PY

echo "===== NoPE smoke test COMPLETE ====="
