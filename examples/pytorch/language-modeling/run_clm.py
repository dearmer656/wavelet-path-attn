#!/usr/bin/env python
# Copyright 2020 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# /// script
# dependencies = [
#     "transformers @ git+https://github.com/huggingface/transformers.git",
#     "albumentations >= 1.4.16",
#     "accelerate >= 0.12.0",
#     "torch >= 1.3",
#     "datasets >= 2.14.0",
#     "sentencepiece != 0.1.92",
#     "protobuf",
#     "evaluate",
#     "scikit-learn",
# ]
# ///

"""
Fine-tuning the library models for causal language modeling (GPT, GPT-2, CTRL, ...) on a text file or a dataset.

Here is the full list of checkpoints on the hub that can be fine-tuned by this script:
https://huggingface.co/models?filter=text-generation
"""
# You can also adapt this script on your own causal language modeling task. Pointers for this are left as comments.

import logging
import math
import os
import sys
from dataclasses import dataclass, field
from itertools import chain
from typing import Optional
import numpy as np
import random
import datasets
import evaluate
import torch
import torch.distributed as dist
from datasets import IterableDataset, IterableDatasetDict, load_dataset, DatasetDict, concatenate_datasets, load_from_disk, interleave_datasets
import transformers
from typing import List
from transformers import (
    CONFIG_MAPPING,
    MODEL_FOR_CAUSAL_LM_MAPPING,
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    EarlyStoppingCallback,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed,
)
try:
    from transformers.testing_utils import CaptureLogger
except Exception:
    import contextlib
    class CaptureLogger:
        def __init__(self, *a, **kw): self.out = ""
        def __enter__(self): return self
        def __exit__(self, *a): pass
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version, send_example_telemetry
from transformers.utils.versions import require_version

try:
    from transformers import is_torch_xla_available
except ImportError:
    def is_torch_xla_available(): return False

import json, hashlib
from pathlib import Path

# Global reference for dataset map functions that need tokenizer in worker/main scope
GLOBAL_TOKENIZER = None


def _cli_flag_present(flag: str) -> bool:
    """Return True when a CLI flag was explicitly provided (supports --foo bar / --foo=bar)."""
    for arg in sys.argv[1:]:
        if arg == flag or arg.startswith(flag + "="):
            return True
    return False

# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.57.0.dev0")

require_version("datasets>=2.14.0", "To fix: pip install -r examples/pytorch/language-modeling/requirements.txt")

logger = logging.getLogger(__name__)


MODEL_CONFIG_CLASSES = list(MODEL_FOR_CAUSAL_LM_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """
    cfg_path: str = field(
        default=None,
        metadata={
            "help": "The config file path for model."
        },
    )
    rel_selection: str = field(
            default="all",
            metadata={
                "help": "Method to select relative contributions from different frequency bands: mean | max | router.",
                "choices": ["rel1", "rel2", "all"],
            },
        )
    router_hidden_dim: int = field(
        default=64,
        metadata={
            "help": "Hidden dimension for wavelet router.",
        },
    )
    router_band_num: int = field(
        default=8,
        metadata={
            "help": "Number of frequency bands for wavelet router.",
        },
    )
    wavelet_router: bool = field(
        default=False,
        metadata={
            "help": "Whether to use wavelet router.",
        },
    )
    qk_rotation: bool = field(
        default=False,
        metadata={
            "help": "Whether to use QK rotation.",
        },
    )
    attn_implementation: str = field(
        default="eager",
        metadata={
            "help": "Attention backend to use: eager | sdpa | flash_attention_2 | path_attn.",
            "choices": ["eager", "sdpa", "flash_attention_2", "path_attn", "path_attn_wfreq", "eager_paged"],
        },
    )
    path_attn_impl: str = field(
        default="pytorch",
        metadata={
            "help": "PaTH attention kernel implementation: pytorch (reference) | triton (parallel_path_attn).",
            "choices": ["pytorch", "triton"],
        },
    )
    pe_method: str = field(
        default="vanilla",
        metadata={
            "help": "Positional encoding method to use: vanilla | rotary | no_pe | wavelet.",
            "choices": ["vanilla", "rotary", "no_pe", "wavelet"],
        },
    )
    relative_type: Optional[str] = field(
        default=None,
        metadata={"help": "Relative PE sub-type. '4' = Ricker wavelet (used with pe_method=wavelet)."},
    )
    # ===== PaTHAttention 专用开关（仅在 --_attn_implementation path_attn 时生效） =====
    use_forget_gate: bool = field(
        default=False,
        metadata={"help": "Enable forget gate (g) for PaTH attention."},
    )
    ### a0=0.7 when theta=0.847, path_attn 初始化为0.8
    init_theta: float = field(
        default=0.847,
        metadata={"help": "Enable forget gate (g) for PaTH attention."},
    )    
    wavelet_baseline_use: bool = field(
        default=False,
        metadata={"help": "Enable wavelet baseline for PaTH attention."},
    )
    use_beta_modulation: bool = field(
        default=False,
        metadata={"help": "Whether use freq into beta."},
    )
    path_use_qk_norm: bool = field(
        default=False,
        metadata={"help": "Apply RMSNorm to Q and K in PaTH attention."},
    )
    share_freq_across_heads: bool = field(
        default=False,
        metadata={"help": "Apply RMSNorm to Q and K in PaTH attention."},
    )
    path_use_low_rank_w: bool = field(
        default=True,
        metadata={"help": "Use low-rank parameterization for W projection (32 bottleneck)."},
    )
    path_use_w_shortconv: bool = field(
        default=True,
        metadata={"help": "Use ShortConvolution on W path."},
    )
    single_A_B: bool = field(
        default=True,
        metadata={"help": "A B single."},
    )    
    path_conv_size: int = field(
        default=3,
        metadata={"help": "Kernel size for W ShortConvolution."},
    )
    num_harmonics: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "Number of harmonics for PaTH attention. "
                "If omitted, keep the value stored in the loaded checkpoint/config."
            )
        },
    )
    path_conv_bias: bool = field(
        default=False,
        metadata={"help": "Use bias in W ShortConvolution."},
    )
    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The model checkpoint for weights initialization. Don't set if you want to train a model from scratch."
            )
        },
    )
    model_type: Optional[str] = field(
        default=None,
        metadata={"help": "If training from scratch, pass a model type from the list: " + ", ".join(MODEL_TYPES)},
    )
    wavelet_mode: Optional[str] = field(
        default="router_rel",
        metadata={
            "help": (
                "Wavelet path mode: off | router_rel | key_inject | logit_bias | "
                "logit_bias_ctxscale_shift_v0 | logit_bias_ctxscale_shift_v0_film. "
                "Legacy values (e.g. additive/softmix/db1) are treated as router_rel."
            )
        },
    )
    wavelet_viz_export: bool = field(
        default=False,
        metadata={"help": "Enable compact eval-time wavelet mechanistic JSONL export."},
    )
    wavelet_viz_run_tag: Optional[str] = field(
        default=None,
        metadata={"help": "Run tag for wavelet_viz export (e.g., A/B1/B2/C/PA)."},
    )
    wavelet_viz_max_batches: int = field(
        default=8,
        metadata={"help": "Max exported eval batches per layer for wavelet_viz."},
    )
    wavelet_viz_sample_q: Optional[int] = field(
        default=None,
        metadata={"help": "Sampled query count for wavelet_viz export (default: wavelet_logit_bias_log_sample_tokens)."},
    )
    wavelet_viz_sample_k: int = field(
        default=256,
        metadata={"help": "Sampled key count for wavelet_viz width/energy statistics."},
    )
    wavelet_viz_outdir: Optional[str] = field(
        default=None,
        metadata={"help": "Output root for wavelet_viz export (default: output_dir)."},
    )
    wavelet_viz_model_size: Optional[str] = field(
        default=None,
        metadata={"help": "Model-size label written into wavelet_viz records (e.g., gpt2 / gpt2-medium)."},
    )
    use_soft_wavelet_fox: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether use wavelet freq into beta (only for path_attn_wfreq)."},
    )
    lw_residual_hw_enable: bool = field(
        default=False,
        metadata={"help": "Enable LW residual HW router specialization (shared LW logits + head-wise residual)."},
    )
    lw_residual_hw_alpha: float = field(
        default=0.1,
        metadata={"help": "Residual scale alpha for LW residual HW: z_res = z_lw + alpha * delta."},
    )
    lw_residual_hw_l2: float = field(
        default=1e-4,
        metadata={"help": "L2 penalty coefficient mu for LW residual delta logits."},
    )
    lw_residual_hw_freeze_steps: int = field(
        default=1500,
        metadata={"help": "Freeze delta for first N global steps, then unfreeze in the same run."},
    )
    config_overrides: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Override some existing default config settings when a model is trained from scratch. Example: "
                "n_embd=10,resid_pdrop=0.2,scale_attn_weights=false,summary_type=cls_index"
            )
        },
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    token: str = field(
        default=None,
        metadata={
            "help": (
                "The token to use as HTTP bearer authorization for remote files. If not specified, will use the token "
                "generated when running `hf auth login` (stored in `~/.huggingface`)."
            )
        },
    )
    trust_remote_code: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to trust the execution of code from datasets/models defined on the Hub."
                " This option should only be set to `True` for repositories you trust and in which you have read the"
                " code, as it will execute code present on the Hub on your local machine."
            )
        },
    )
    dtype: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Override the default `torch.dtype` and load the model under this dtype. If `auto` is passed, the "
                "dtype will be automatically derived from the model's weights."
            ),
            "choices": ["auto", "bfloat16", "float16", "float32"],
        },
    )

    def __post_init__(self):
        if self.config_overrides is not None and (self.config_name is not None or self.model_name_or_path is not None):
            raise ValueError(
                "--config_overrides can't be used in combination with --config_name or --model_name_or_path"
            )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    train_file: Optional[str] = field(default=None, metadata={"help": "The input training data file (a text file)."})
    validation_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input evaluation data file to evaluate the perplexity on (a text file)."},
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )
    streaming: bool = field(default=False, metadata={"help": "Enable streaming mode"})
    full_fine_tune: bool = field(default=False, metadata={"help": "Enable full fine-tuning mode"})
    block_size: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "Optional input sequence length after tokenization. "
                "The training dataset will be truncated in block of this size for training. "
                "Default to the model max input length for single sentence inputs (take into account special tokens)."
            )
        },
    )
    xsum_bucket_size: int = field(
        default=512,
        metadata={
            "help": (
                "XSUM only. If > 0, apply length-bucket filtering to avoid overlap across different --block_size runs. "
                "Keeps samples with total_token_len in (lower, upper] where buckets are aligned to xsum_bucket_size. "
                "Example: xsum_bucket_size=512 and block_size=2048 keeps (1536, 2048]. Set 0 to disable."
            )
        },
    )
    xsum_bucket_apply_to: str = field(
        default="eval_test",
        metadata={
            "help": (
                "XSUM only. Which splits to apply xsum_bucket_size filtering to. "
                "Choices: 'eval_test' (default), 'all', 'none'."
            )
        },
    )
    xsum_min_total_len: int = field(
        default=0,
        metadata={
            "help": (
                "XSUM only. Optional absolute minimum of total_token_len (before padding) to keep. "
                "Applied in addition to xsum_bucket_size filtering."
            )
        },
    )

    hotpot_long_jsonl: Optional[str] = field(
        default=None,
        metadata={"help": "Path to HotpotQA-Long augmented JSONL. If set, overrides HF hotpot_qa distractor loading."},
    )
    hotpot_long_lengths: Optional[str] = field(
        default=None,
        metadata={"help": "Comma-separated target lengths to filter from hotpot_long_jsonl (e.g. '2048,4096')."},
    )
    passkey_num_samples: int = field(
        default=50,
        metadata={"help": "Number of passkey retrieval examples to generate per evaluation run."},
    )
    passkey_num_digits: int = field(
        default=5,
        metadata={"help": "Number of digits in the hidden passkey number."},
    )

    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    validation_split_percentage: Optional[int] = field(
        default=5,
        metadata={
            "help": "The percentage of the train set used as validation set in case there's no validation split"
        },
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    keep_linebreaks: bool = field(
        default=True, metadata={"help": "Whether to keep line breaks when using TXT files or not."}
    )

    def __post_init__(self):
        if self.streaming:
            require_version("datasets>=2.0.0", "The streaming feature requires `datasets>=2.0.0`")

        if self.dataset_name is None and self.train_file is None and self.validation_file is None:
            raise ValueError("Need either a dataset name or a training/validation file.")
        else:
            if self.train_file is not None:
                extension = self.train_file.split(".")[-1]
                assert extension in ["csv", "json", "jsonl", "txt"], "`train_file` should be a csv, a json or a txt file."
            if self.validation_file is not None:
                extension = self.validation_file.split(".")[-1]
                assert extension in ["csv", "json", "jsonl", "txt"], "`validation_file` should be a csv, a json, a jsonl or a txt file."
@dataclass
class SupplyTrainingArguments(TrainingArguments):
    ablate_switch: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to ablate the wavelet path and only use vanilla attention"
            )
        },
    )

    analyzer: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to use analyzer to analyze the attention weights"
            )
        },
    )
    loss_type: str = field(
        default='mse',
        metadata={
            "help": (
                "The type of loss to use: default | spectral_distill | temp_distill | both"
            )
        },
    )
    coe_for_rel_lr: Optional[float] = field(
        default=None,
        metadata={"help": "LR for coe_for_rel parameter group"},
    )
    qwab_lr: Optional[float] = field(
        default=None,
        metadata={"help": "Peak LR for QWAB ctxscale parameter group (wavelet_ctx_*, wavelet_bias_film*, wavelet_logit_bias_*, mlp_bias_*). If None, falls back to the global learning_rate. Has no effect on PA-only runs (no matching params)."},
    )
    weight_alpha: float = field(
        default=0.0,
        metadata={"help": "The alpha value for weighting"}
    )
    rel_zoom_in_coe: float = field(
        default=1.0,
        metadata={"help": "The alpha value for weighting"}
    )    
    rel_alpha: float = field(
        default=1.0,
        metadata={"help": "Scalar alpha for final attention logits: z = base + alpha * rel. Use 0.0 for rel0."},
    )
    paired_eval_setting: str = field(
        default="auto",
        metadata={"help": "Record tag for paired eval output: auto | baseline | rel0."},
    )
    paired_eval_bootstrap_B: int = field(
        default=2000,
        metadata={"help": "Bootstrap iterations for paired mean-delta confidence intervals."},
    )
    paired_eval_seq_bin_size: int = field(
        default=256,
        metadata={"help": "Sequence length bucket size for paired delta statistics."},
    )
    wavelet_pe_softmax_use: bool = field(
        default=False,
        metadata={"help": "Whether to use softmax in wavelet positional encoding"}
    )
    b_unfreeze_step: int = field(
        default=5000,
        metadata={
            "help": (
                "The wB doesn't update until unfreeze_step"
            )
        },
    )
    sample_num: int = field(
        default=0,
        metadata={
            "help": (
                "The number of samples to use geom sampling"
            )
        },
    )
    spectral_loss_coe: float = field(
        default=1e-3,
        metadata={
            "help": (
                "The coefficient for distillation loss"
            )
        },
    )
    temp_loss_coe: float = field(
        default=5e-3,
        metadata={
            "help": (
                "The coefficient for distillation loss"
            )
        },
    )
    distill_teacher: str = field(
        default='wavelet',
        metadata={
            "help": (
                "The path for distillation teacher model"
            )
        },
    )
    distill_in_which_layers: int = field(
        default=1,
        metadata={
            "help": (
                "distill happen in layer less than the setted value."
            )
        },
    )
    distill_freq_scale: int = field(
        default=25,
        metadata={
            "help": (
                "The frequency scaling factor for distillation loss"
            )
        },
    )
    scale_range: list[int] = field(
        default_factory=lambda: [0, 16],
        metadata={
            "help": ("The scale range as two integers, e.g. --scale_range 0 16"),
        },
    )
    path_blend_layers: list[int] = field(
        default_factory=lambda: [5, 6],
        metadata={
            "help": ("Layer indices that use PaTH logit blending (e.g. --path_blend_layers 0 1). "
                     "Default [5, 6] per PAT-100 spec."),
        },
    )
    path_sparse_gate: bool = field(
        default=False,
        metadata={"help": "PAT-100: enable sparse query-conditioned gate on path logits. "
                          "When True, gate_eff scales each query token's path-logit row (conditioned on delta = q - q_corr)."},
    )
    gate_sparse_alpha: float = field(
        default=0.01,
        metadata={"help": "PAT-100: L_sparse = gate_sparse_alpha * gate.mean() added to training loss."},
    )
    gate_warmup_steps: int = field(
        default=2000,
        metadata={"help": "PAT-100: number of steps over which the gate eta ramps from 0 to 1."},
    )
    path_gate_force_open: bool = field(
        default=False,
        metadata={"help": "PAT-100: forced-open gate control. When True, g_i≡1 (gate bypassed), "
                          "giving full gradient access to path params. gate_eff = eta (warmup only). "
                          "No sparse loss. Used to isolate gate-training dynamics from path learning."},
    )
    early_stopping_patience: int = field(
        default=0,
        metadata={"help": "PAT-101: if > 0, add EarlyStoppingCallback with this patience (number of "
                          "eval steps with no improvement before stopping). Requires --load_best_model_at_end True."},
    )
    smooth_use: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to use smoothing in distillation loss"
            )
        },
    )
    distilling_coe_warmup_use: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to use warmup for distilling coefficient"
            )
        },
    )
    data_collection_style: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "collect data for router analysis."
            )
        },
    )
    head_mask_list: Optional[list[int]] = field(
        default=None,
        metadata={
            "help": "Mask one attention head as: --head_mask_list <layer> <head>.",
            "nargs": 2,
        },
    )
    eval_rel_stats_enabled: bool = field(
        default=True,
        metadata={"help": "Enable eval-only position-bin stats for E_base_raw/rel diagnostics."},
    )
    eval_rel_stats_layers: str = field(
        default="all",
        metadata={"help": "Comma-separated layer ids to track in eval (e.g. '0' or '0,6'). Use 'all' for all layers."},
    )
    eval_rel_stats_bin_size: int = field(
        default=256,
        metadata={"help": "Bin size over query index t for eval stats."},
    )
    eval_rel_stats_log_every: int = field(
        default=0,
        metadata={"help": "Print local eval stats every N eval batches from the anchor layer (0 disables periodic local print)."},
    )
    eval_rel_stats_log_once: bool = field(
        default=True,
        metadata={"help": "When eval_rel_stats_log_every=0, print once per eval run (rank0 merged summary)."},
    )
    eval_rel_stats_per_head: bool = field(
        default=False,
        metadata={"help": "Track means per-head (still prints head-averaged line summary)."},
    )
    eval_rel_stats_max_samples_per_bin: int = field(
        default=4096,
        metadata={"help": "Reservoir sample cap per bin for p50/p90/p99 estimates."},
    )
    eval_rel_stats_eps: float = field(
        default=1e-6,
        metadata={"help": "Epsilon for ratio computations in eval stats."},
    )
    eval_rel_stats_anchor_layer: int = field(
        default=0,
        metadata={"help": "Layer id used to count eval batches for periodic local logging."},
    )
    log_rel_stats: bool = field(
        default=False,
        metadata={"help": "Enable rel-branch gradient/eval diagnostics logging."},
    )
    log_rel_every: int = field(
        default=500,
        metadata={"help": "Log rel diagnostics every N optimizer steps in training."},
    )
    log_rel_sample_qpos: str = field(
        default="128,512,2048",
        metadata={"help": "CSV query positions used by rel diagnostics sampling."},
    )
    log_rel_sample_heads: str = field(
        default="0,3,7,11",
        metadata={"help": "CSV head ids used by rel diagnostics sampling."},
    )
    log_rel_sample_key_offsets: str = field(
        default="0,16,64,256,1024",
        metadata={"help": "CSV key offsets for n=m-offset sampling in rel diagnostics."},
    )
    log_rel_tail_tau: int = field(
        default=1024,
        metadata={"help": "Tail-mass tau for rel eval diagnostics (n <= m-tau)."},
    )
    log_rel_eval_every: int = field(
        default=0,
        metadata={"help": "Run eval rel diagnostics every K eval calls (<=0 disables eval rel diagnostics)."},
    )
    rel_param_keywords: str = field(
        default="rel,wavelet,router,seq_pe,q_corr,wavelet_dtt",
        metadata={"help": "CSV substrings used to match rel-branch parameters for grad norm logging."},
    )
    freeze_train_keywords: str = field(
        default="",
        metadata={"help": "CSV substrings for parameter names to keep trainable when freeze_backbone=True. Empty uses built-in defaults."},
    )
    router_norm_enable: bool = field(
        default=False,
        metadata={"help": "Enable router input/logit normalization for wavelet router branches."},
    )
    router_norm_mode: str = field(
        default="pre_gate",
        metadata={"help": "Router norm insertion mode: pre_gate | post_gate | logit | none."},
    )
    router_norm_type: str = field(
        default="rmsnorm",
        metadata={"help": "Router norm type: layernorm | rmsnorm | zscore | none."},
    )
    router_norm_affine: bool = field(
        default=False,
        metadata={"help": "Whether to use affine parameters for router normalization."},
    )
    router_norm_eps: float = field(
        default=1e-5,
        metadata={"help": "Epsilon for router normalization."},
    )
    router_norm_clamp_std_min: float = field(
        default=1e-4,
        metadata={"help": "Minimum clamped std (or rms) for router normalization."},
    )
    router_norm_log_every: int = field(
        default=500,
        metadata={"help": "Log sampled router norm stats every N steps."},
    )
    router_norm_log_heads: str = field(
        default="0,3,7",
        metadata={"help": "Sampled head ids for router norm stats logging (csv)."},
    )
    router_norm_log_tokens: str = field(
        default="0,-1",
        metadata={"help": "Sampled token ids for router norm stats logging (csv, supports 'mid' and -1)."},
    )
    eval_router_heatmap_enable: bool = field(
        default=False,
        metadata={"help": "Enable eval-time router position-bin heatmap analysis (extra eval forward pass)."},
    )
    eval_router_heatmap_bin_size: int = field(
        default=256,
        metadata={"help": "Query-position bin size used by router heatmap analysis."},
    )
    eval_router_heatmap_max_batches: int = field(
        default=0,
        metadata={"help": "Optional cap on eval batches for router heatmap analysis (<=0 means all)."},
    )
    eval_router_heatmap_out_subdir: str = field(
        default="router_heatmaps",
        metadata={"help": "Output subdirectory under output_dir for router heatmap figures and tensors."},
    )
# save_router_pt_callback_once.py
import os
from dataclasses import dataclass
from typing import Dict, List, Optional

import torch
from tqdm import tqdm
from transformers import TrainerCallback


def is_rank0() -> bool:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank() == 0
    return True


def to_cpu_detached(x: torch.Tensor) -> torch.Tensor:
    return x.detach().to("cpu")


@dataclass
class RouterPTDumpOnceConfig:
    dump_dir: str = "./router_pt_dump"
    file_prefix: str = "dump_once"
    # 按样本总数截断，而不是按 batch 数
    sampled_data_number: int = 1
    # 可选的 batch 上限，避免极端情况下无限迭代
    max_batches: Optional[int] = None
    # 是否把 input_ids 一起存
    save_input_ids: bool = True
    save_attention_mask: bool = False
    # router idx 的 dtype（K<=256 推荐 uint8）
    cast_idx_dtype: torch.dtype = torch.uint8
    # True: 只保存 labels!=-100 token 的 router（更省空间）
    only_label_tokens: bool = False
    # dump 完立刻退出程序
    exit_after_dump: bool = True
    # 仅用于生成文件名，方便区分不同 block_size 的缓存
    block_size: Optional[int] = None

class SaveRouterPTOnceCallback(TrainerCallback):
    """
    在第一次 evaluate 时：
    - 从 eval dataloader 按样本数抓取（sampled_data_number）
      - forward 一次（会刷新 model.last_router1_idx / last_router2_idx）
      - 保存 labels + router idx 到 .pt
      - 然后立刻退出程序（SystemExit）

    依赖你在 model.forward 里做了：
      self.last_router1_idx = router1_idx.detach()
      self.last_router2_idx = router2_idx.detach()
    """
    def __init__(self, cfg: RouterPTDumpOnceConfig):
        super().__init__()
        self.cfg = cfg
        os.makedirs(cfg.dump_dir, exist_ok=True)
        self._has_dumped = False  # 防止多次触发

    @torch.no_grad()
    def on_evaluate(self, args, state, control, model=None, **kwargs):
        if self._has_dumped:
            return control
        if not is_rank0():
            return control

        trainer = kwargs.get("trainer", None)
        if trainer is None:
            eval_dataloader = kwargs.get("eval_dataloader", None)
        else:
            eval_dataloader = trainer.get_eval_dataloader()

        if eval_dataloader is None:
            print("[SaveRouterPTOnceCallback] No eval_dataloader found; skip.")
            return control

        if model is None:
            raise RuntimeError("[SaveRouterPTOnceCallback] model is None")

        dump_path = os.path.join(
            self.cfg.dump_dir, f"L{self.cfg.block_size}_{self.cfg.file_prefix}.pt"
        )

        model.eval()
        device = args.device

        target_samples = max(1, int(self.cfg.sampled_data_number))
        batches_seen = 0
        collected = 0

        labels_chunks: List[torch.Tensor] = []
        input_ids_chunks: List[torch.Tensor] = []
        attn_chunks: List[torch.Tensor] = []
        router1_chunks: List[torch.Tensor | List[torch.Tensor]] = []
        router2_chunks: List[torch.Tensor | List[torch.Tensor]] = []
        pred_chunks: List[torch.Tensor] = []

        pbar = tqdm(total=target_samples, desc=f"Dump {target_samples} samples -> {os.path.basename(dump_path)}")

        for i, batch in enumerate(eval_dataloader):
            if self.cfg.max_batches is not None and i >= self.cfg.max_batches:
                break
            if collected >= target_samples:
                break

            # Move tensors to device
            batch = {k: v.to(device) for k, v in batch.items() if torch.is_tensor(v)}
            labels = batch.get("labels", None)
            if labels is None:
                raise RuntimeError("Batch has no 'labels'.")

            # forward 一次：这一步会让 model.last_router*_idx 刷新，并拿到 logits 做 preds
            outputs = model(**batch)
            logits = None
            if hasattr(outputs, "logits"):
                logits = outputs.logits
            elif isinstance(outputs, tuple) and len(outputs) > 0:
                logits = outputs[0]
            if logits is None:
                raise RuntimeError("Model forward did not return logits; cannot compute preds")

            # 1) DDP 解包
            m = model.module if hasattr(model, "module") else model

            # 2) GPT2LMHeadModel 的 transformer 子模块才是 GPT2Model
            if not hasattr(m, "transformer"):
                raise RuntimeError(f"Top model has no .transformer. type={type(m)}")

            tr = m.transformer  # GPT2Model 实例

            # 3) 从 GPT2Model 上拿缓存
            if not hasattr(tr, "last_router1_idx") or not hasattr(tr, "last_router2_idx"):
                raise RuntimeError(
                    "transformer (GPT2Model) does not have last_router1_idx/last_router2_idx. "
                    "Did you set them inside GPT2Model.forward()?"
                )

            if tr.last_router1_idx is None or tr.last_router2_idx is None:
                raise RuntimeError(
                    "Found last_router1_idx/last_router2_idx on transformer, but they are None. "
                    "Forward may not have set them on this call."
                )

            router1_idx = tr.last_router1_idx
            router2_idx = tr.last_router2_idx

            # 打印原始张量形状，方便确认是否全量保存
            try:
                print(
                    "[SaveRouterPTOnceCallback] shapes",
                    {
                        "labels": tuple(labels.shape),
                        "router1_idx": tuple(router1_idx.shape),
                        "router2_idx": tuple(router2_idx.shape),
                        "logits": tuple(logits.shape),
                    },
                )
            except Exception:
                pass  # 打印失败不影响后续流程

            batch_size = labels.shape[0]
            take = min(batch_size, target_samples - collected)
            if take <= 0:
                break

            labels_slice = labels[:take]

            if self.cfg.save_input_ids and "input_ids" in batch:
                input_ids_chunks.append(to_cpu_detached(batch["input_ids"][:take]))
            if self.cfg.save_attention_mask and "attention_mask" in batch:
                attn_chunks.append(to_cpu_detached(batch["attention_mask"][:take]))

            def process_router(v: torch.Tensor, label_ref: torch.Tensor) -> torch.Tensor | List[torch.Tensor]:
                v = v.detach()

                if self.cfg.only_label_tokens:
                    mask = label_ref.ne(-100)
                    if v.dim() == 5:  # [L,B,T,H,R]
                        out = []
                        for l in range(v.shape[0]):
                            # mask over B,T then flatten tokens -> [N,H,R]
                            sel = v[l][:take][mask]  # boolean mask on first two dims
                            out.append(to_cpu_detached(sel))
                        return out
                    if v.dim() == 4:  # [L,B,T,H]
                        out = []
                        for l in range(v.shape[0]):
                            sel = v[l][:take][mask]  # [N,H]
                            out.append(to_cpu_detached(sel))
                        return out
                    if v.dim() == 3:  # [B,T,H]
                        sel = v[:take][mask]
                        return to_cpu_detached(sel)
                    return to_cpu_detached(v[:take])

                # 不掩码，直接按 batch 维度截取
                if v.dim() >= 4:  # [L,B,...]
                    return to_cpu_detached(v[:, :take])
                return to_cpu_detached(v[:take])

            def process_preds(pred_ids: torch.Tensor, label_ref: torch.Tensor) -> torch.Tensor:
                pred_ids = pred_ids.detach()
                if self.cfg.only_label_tokens:
                    mask = label_ref.ne(-100)
                    return to_cpu_detached(pred_ids[:take][mask])
                return to_cpu_detached(pred_ids[:take])

            labels_chunks.append(to_cpu_detached(labels_slice))
            router1_chunks.append(process_router(router1_idx, labels_slice))
            router2_chunks.append(process_router(router2_idx, labels_slice))
            pred_chunks.append(process_preds(logits, labels_slice))

            collected += take
            batches_seen += 1
            pbar.update(take)

            if collected >= target_samples:
                break

        pbar.close()

        def _cat_if_tensor(seq):
            if not seq:
                return None
            if any(isinstance(x, list) for x in seq):
                # list 按 layer 存储时，逐 layer 拼接 batch 维
                L = len(seq[0])
                merged: List[torch.Tensor] = []
                for l in range(L):
                    merged.append(torch.cat([chunk[l] for chunk in seq], dim=0))
                return merged

            # tensor 情况：根据维度决定 batch 维位置
            sample = seq[0]
            cat_dim = 0 if sample.dim() < 4 else 1  # [B,...] 用 0；[L,B,...] 用 1
            return torch.cat(seq, dim=cat_dim)

        payload = {
            "global_step": int(state.global_step),
            "num_batches": batches_seen,
            "num_samples": collected,
            "sampled_data_number": target_samples,
        }

        labels_cat = _cat_if_tensor(labels_chunks)
        if labels_cat is not None:
            payload["labels"] = labels_cat

        input_cat = _cat_if_tensor(input_ids_chunks)
        if input_cat is not None:
            payload["input_ids"] = input_cat

        attn_cat = _cat_if_tensor(attn_chunks)
        if attn_cat is not None:
            payload["attention_mask"] = attn_cat

        router1_cat = _cat_if_tensor(router1_chunks)
        if router1_cat is not None:
            payload["router1_idx"] = router1_cat

        router2_cat = _cat_if_tensor(router2_chunks)
        if router2_cat is not None:
            payload["router2_idx"] = router2_cat

        pred_cat = _cat_if_tensor(pred_chunks)
        if pred_cat is not None:
            payload["pred_ids"] = pred_cat

        payload = {
            k: v for k, v in payload.items() if v is not None
        }
        torch.save(payload, dump_path)
        print(f"[SaveRouterPTOnceCallback] Saved: {dump_path}")

        self._has_dumped = True

        # 立刻退出整个程序（你要求“直走一次就停止”）
        if self.cfg.exit_after_dump:
            raise SystemExit(0)

        return control


@dataclass
class RouterPosBinHeatmapConfig:
    enable: bool = False
    bin_size: int = 256
    out_subdir: str = "router_heatmaps"
    max_batches: Optional[int] = None


class RouterPosBinHeatmapCallback(TrainerCallback):
    """
    Eval-only router analyzer:
      - Aggregate router1/router2 over valid tokens by query-position bins.
      - Metrics per (layer, bin, head, scale):
        1) mean router probability
        2) top1(argmax) frequency
      - Save raw tensors (.pt) and per-layer subplot heatmaps.

    Notes:
      - Runs an extra eval forward pass when enabled.
      - Uses model.transformer.last_router1_idx/last_router2_idx refreshed by each forward.
    """

    def __init__(self, cfg: RouterPosBinHeatmapConfig):
        super().__init__()
        self.cfg = cfg
        self._warned_missing_router = False

    @staticmethod
    def _world_rank() -> int:
        if dist.is_available() and dist.is_initialized():
            return dist.get_rank()
        return 0

    @staticmethod
    def _world_size() -> int:
        if dist.is_available() and dist.is_initialized():
            return dist.get_world_size()
        return 1

    @staticmethod
    def _unwrap_transformer(model):
        root = model.module if hasattr(model, "module") else model
        if not hasattr(root, "transformer"):
            return None
        return root.transformer

    @staticmethod
    def _to_prob_router(router: torch.Tensor, router_style: Optional[str]) -> torch.Tensor:
        """
        Convert stored router tensor to probabilities.
        - style='logit' => softmax
        - otherwise assume already prob, then renormalize for numerical safety
        """
        r = router.float()
        if router_style == "logit":
            return torch.softmax(r, dim=-1)
        denom = r.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        return r / denom

    @staticmethod
    def _select_valid_mask(batch_tensors, bsz: int, seq_len: int, device: torch.device) -> torch.Tensor:
        attn = batch_tensors.get("attention_mask", None)
        if attn is not None:
            return attn[:, :seq_len].to(device=device).bool()
        labels = batch_tensors.get("labels", None)
        if labels is not None:
            return labels[:, :seq_len].to(device=device).ne(-100)
        return torch.ones((bsz, seq_len), device=device, dtype=torch.bool)

    @staticmethod
    def _plot_layer_bin_grid(
        mat_lbhS: torch.Tensor,
        count_tokens_per_bin: torch.Tensor,
        bin_size: int,
        out_dir: Path,
        router_tag: str,
        metric_tag: str,
        global_step: int,
    ):
        import matplotlib.pyplot as _plt

        arr = mat_lbhS.detach().cpu().float().numpy()  # [L,NB,H,S]
        counts = count_tokens_per_bin.detach().cpu().numpy().astype(np.int64)
        L, NB, H, S = arr.shape
        paths = []
        if NB <= 0:
            return paths

        ncols = min(4, NB)
        nrows = int(math.ceil(NB / ncols))
        value_label = "mean_prob" if metric_tag == "prob" else "top1_freq"

        for li in range(L):
            fig, axes = _plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.2 * nrows), squeeze=False)
            im = None
            for bidx in range(NB):
                ax = axes[bidx // ncols][bidx % ncols]
                t0 = int(bidx) * int(bin_size)
                t1 = t0 + int(bin_size)
                # [H,S] -> [S,H]: y=scale, x=head
                hm = arr[li, bidx].transpose(1, 0)
                im = ax.imshow(hm, aspect="auto", origin="lower", vmin=0.0, vmax=1.0, cmap="viridis")
                ax.set_title(f"bin {t0}-{t1} | N={int(counts[bidx])}")
                ax.set_xlabel("head")
                ax.set_ylabel("scale")
                if H <= 16:
                    ax.set_xticks(np.arange(H))
                if S <= 16:
                    ax.set_yticks(np.arange(S))

            total_axes = nrows * ncols
            for k in range(NB, total_axes):
                axes[k // ncols][k % ncols].axis("off")

            if im is not None:
                cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.92)
                cbar.set_label(value_label)
            fig.suptitle(
                f"router={router_tag} metric={metric_tag} layer={li} step={int(global_step)}",
                fontsize=11,
            )
            fig.tight_layout()
            out_path = out_dir / f"{router_tag}_{metric_tag}_layer{li:02d}.png"
            fig.savefig(out_path, dpi=180, bbox_inches="tight")
            _plt.close(fig)
            paths.append(str(out_path))

        return paths

    @torch.no_grad()
    def on_evaluate(self, args, state, control, model=None, **kwargs):
        if not self.cfg.enable:
            return control
        if model is None:
            logger.warning("[RouterHeatmap] model is None, skip.")
            return control

        trainer = kwargs.get("trainer", None)
        eval_dataloader = kwargs.get("eval_dataloader", None)
        if eval_dataloader is None and trainer is not None:
            eval_dataloader = trainer.get_eval_dataloader()
        if eval_dataloader is None:
            logger.warning("[RouterHeatmap] eval_dataloader not found, skip.")
            return control

        rank = self._world_rank()
        world = self._world_size()
        is_rank0_local = (rank == 0)

        tr = self._unwrap_transformer(model)
        if tr is None:
            if is_rank0_local:
                logger.warning("[RouterHeatmap] model has no .transformer, skip.")
            return control

        # Router style controls whether cached tensor is logits or probabilities.
        router_style = str(getattr(getattr(tr, "config", None), "router_data_collection_style", "prob")).lower()
        if router_style not in {"prob", "logit", "none"}:
            router_style = "prob"

        bin_size = max(1, int(self.cfg.bin_size))
        max_batches = None if self.cfg.max_batches is None else int(self.cfg.max_batches)
        if max_batches is not None and max_batches <= 0:
            max_batches = None

        device = args.device
        model.eval()

        cfg_obj = getattr(tr, "config", None)
        n_layers = max(
            1,
            int(
                getattr(
                    cfg_obj,
                    "num_hidden_layers",
                    getattr(cfg_obj, "n_layer", 1),
                )
            ),
        )
        n_heads = max(
            1,
            int(
                getattr(
                    cfg_obj,
                    "num_attention_heads",
                    getattr(cfg_obj, "n_head", 1),
                )
            ),
        )
        n_scales = max(1, int(getattr(cfg_obj, "router_band_num", 8)))
        eval_seq_len = int(getattr(cfg_obj, "block_size", 0))
        if eval_seq_len <= 0:
            eval_seq_len = int(getattr(cfg_obj, "max_position_embeddings", 0))
        if eval_seq_len <= 0:
            eval_seq_len = int(getattr(cfg_obj, "n_positions", 0))
        if eval_seq_len <= 0:
            eval_seq_len = 3072

        n_bins = int(math.ceil(eval_seq_len / bin_size))
        shape = (n_layers, n_bins, n_heads, n_scales)
        sum_prob1 = torch.zeros(shape, device=device, dtype=torch.float64)
        sum_prob2 = torch.zeros(shape, device=device, dtype=torch.float64)
        top1_cnt1 = torch.zeros(shape, device=device, dtype=torch.float64)
        top1_cnt2 = torch.zeros(shape, device=device, dtype=torch.float64)
        tok_cnt = torch.zeros((n_bins,), device=device, dtype=torch.float64)
        if is_rank0_local:
            logger.info(
                "[RouterHeatmap] init shapes: L=%d H=%d S=%d T=%d bins=%d bin_size=%d",
                n_layers, n_heads, n_scales, eval_seq_len, n_bins, bin_size,
            )

        num_batches = 0
        warned_shape_mismatch = False

        progress = tqdm(
            eval_dataloader,
            disable=not is_rank0_local,
            desc=f"RouterPosBin eval scan (rank {rank}/{world})",
        )
        for batch_idx, batch in enumerate(progress):
            if max_batches is not None and batch_idx >= max_batches:
                break
            batch_tensors = {k: v.to(device) for k, v in batch.items() if torch.is_tensor(v)}
            if not batch_tensors:
                continue

            _ = model(**batch_tensors)
            router1_raw = getattr(tr, "last_router1_idx", None)
            router2_raw = getattr(tr, "last_router2_idx", None)
            if router1_raw is None or router2_raw is None:
                if (not self._warned_missing_router) and is_rank0_local:
                    logger.warning(
                        "[RouterHeatmap] last_router tensors are None. "
                        "Ensure wavelet router is enabled and model forward exports router outputs."
                    )
                    self._warned_missing_router = True
                continue

            if router1_raw.dim() != 5 or router2_raw.dim() != 5:
                if (not self._warned_missing_router) and is_rank0_local:
                    logger.warning(
                        "[RouterHeatmap] Expected router shape [L,B,T,H,S], got r1=%s r2=%s",
                        tuple(router1_raw.shape), tuple(router2_raw.shape)
                    )
                    self._warned_missing_router = True
                continue

            router1 = self._to_prob_router(router1_raw, router_style=router_style)
            router2 = self._to_prob_router(router2_raw, router_style=router_style)

            L, B, T, H, S = router1.shape
            L_use = min(int(L), int(n_layers))
            H_use = min(int(H), int(n_heads))
            S_use = min(int(S), int(n_scales))
            if (L_use <= 0) or (H_use <= 0) or (S_use <= 0):
                continue
            if (not warned_shape_mismatch) and is_rank0_local and (L_use != L or H_use != H or S_use != S):
                logger.warning(
                    "[RouterHeatmap] router dims (%d,%d,%d) differ from config (%d,%d,%d); using overlap dims.",
                    int(L), int(H), int(S), int(n_layers), int(n_heads), int(n_scales),
                )
                warned_shape_mismatch = True

            T_mask = T
            if "attention_mask" in batch_tensors:
                T_mask = min(T_mask, int(batch_tensors["attention_mask"].shape[1]))
            elif "labels" in batch_tensors:
                T_mask = min(T_mask, int(batch_tensors["labels"].shape[1]))
            if T_mask <= 0:
                continue

            T_use = min(int(T_mask), eval_seq_len)
            if T_use <= 0:
                continue

            valid_mask = self._select_valid_mask(batch_tensors, bsz=B, seq_len=T_use, device=device)  # [B,T_use]
            if valid_mask.sum().item() == 0:
                continue

            # [L,B,T,H,S] -> [L,H,B,T,S] for easier flattening over (B,T_bin)
            r1_lhbts = router1[:L_use, :, :T_use, :H_use, :S_use].permute(0, 3, 1, 2, 4).contiguous()
            r2_lhbts = router2[:L_use, :, :T_use, :H_use, :S_use].permute(0, 3, 1, 2, 4).contiguous()

            for start in range(0, T_use, bin_size):
                end = min(start + bin_size, T_use)
                bidx = start // bin_size
                v = valid_mask[:, start:end].reshape(-1)  # [B * Tb]
                n_valid = int(v.sum().item())
                if n_valid == 0:
                    continue
                tok_cnt[bidx] += float(n_valid)

                r1_bin = r1_lhbts[:, :, :, start:end, :].reshape(L_use, H_use, -1, S_use)
                r2_bin = r2_lhbts[:, :, :, start:end, :].reshape(L_use, H_use, -1, S_use)
                r1_sel = r1_bin[:, :, v, :]  # [L,H,N,S]
                r2_sel = r2_bin[:, :, v, :]

                sum_prob1[:L_use, bidx, :H_use, :S_use] += r1_sel.sum(dim=2, dtype=torch.float64)
                sum_prob2[:L_use, bidx, :H_use, :S_use] += r2_sel.sum(dim=2, dtype=torch.float64)

                top1_idx1 = r1_sel.argmax(dim=-1)  # [L,H,N]
                top1_idx2 = r2_sel.argmax(dim=-1)
                top1_oh1 = torch.nn.functional.one_hot(top1_idx1, num_classes=S_use).to(torch.float64)
                top1_oh2 = torch.nn.functional.one_hot(top1_idx2, num_classes=S_use).to(torch.float64)
                top1_cnt1[:L_use, bidx, :H_use, :S_use] += top1_oh1.sum(dim=2)
                top1_cnt2[:L_use, bidx, :H_use, :S_use] += top1_oh2.sum(dim=2)

            num_batches += 1

        if dist.is_available() and dist.is_initialized():
            for t in [sum_prob1, sum_prob2, top1_cnt1, top1_cnt2, tok_cnt]:
                dist.all_reduce(t, op=dist.ReduceOp.SUM)
            nb = torch.tensor([num_batches], device=device, dtype=torch.int64)
            dist.all_reduce(nb, op=dist.ReduceOp.SUM)
            num_batches = int(nb.item())

        if tok_cnt.sum().item() <= 0:
            if is_rank0_local:
                logger.warning("[RouterHeatmap] No valid router tokens collected across ranks; skip plotting.")
            return control
        if not is_rank0_local:
            return control

        denom = tok_cnt.clamp_min(1.0).view(1, n_bins, 1, 1)
        mean_prob1 = sum_prob1 / denom
        mean_prob2 = sum_prob2 / denom
        top1_freq1 = top1_cnt1 / denom
        top1_freq2 = top1_cnt2 / denom

        out_dir = Path(args.output_dir) / str(self.cfg.out_subdir) / f"step_{int(state.global_step)}"
        out_dir.mkdir(parents=True, exist_ok=True)

        stats_payload = {
            "global_step": int(state.global_step),
            "num_batches_scanned": int(num_batches),
            "world_size": int(world),
            "bin_size": int(bin_size),
            "eval_seq_len": int(eval_seq_len),
            "n_layers": int(n_layers),
            "n_heads": int(n_heads),
            "n_scales": int(n_scales),
            "bin_ranges": [(int(i * bin_size), int((i + 1) * bin_size)) for i in range(n_bins)],
            "count_tokens_per_bin": tok_cnt.detach().cpu(),
            "router1_mean_prob": mean_prob1.detach().cpu(),
            "router2_mean_prob": mean_prob2.detach().cpu(),
            "router1_top1_freq": top1_freq1.detach().cpu(),
            "router2_top1_freq": top1_freq2.detach().cpu(),
            "router1_sum_prob": sum_prob1.detach().cpu(),
            "router2_sum_prob": sum_prob2.detach().cpu(),
            "router1_top1_count": top1_cnt1.detach().cpu(),
            "router2_top1_count": top1_cnt2.detach().cpu(),
        }
        stats_path = out_dir / "router_posbin_stats.pt"
        torch.save(stats_payload, stats_path)

        fig_paths = []
        fig_paths.extend(
            self._plot_layer_bin_grid(
                mean_prob1, tok_cnt, bin_size, out_dir,
                router_tag="router1", metric_tag="prob", global_step=int(state.global_step)
            )
        )
        fig_paths.extend(
            self._plot_layer_bin_grid(
                top1_freq1, tok_cnt, bin_size, out_dir,
                router_tag="router1", metric_tag="top1freq", global_step=int(state.global_step)
            )
        )
        fig_paths.extend(
            self._plot_layer_bin_grid(
                mean_prob2, tok_cnt, bin_size, out_dir,
                router_tag="router2", metric_tag="prob", global_step=int(state.global_step)
            )
        )
        fig_paths.extend(
            self._plot_layer_bin_grid(
                top1_freq2, tok_cnt, bin_size, out_dir,
                router_tag="router2", metric_tag="top1freq", global_step=int(state.global_step)
            )
        )

        logger.info(
            "[RouterHeatmap] done. step=%d num_batches=%d bin_size=%d bins=%d saved_pt=%s saved_figs=%d",
            int(state.global_step),
            int(num_batches),
            int(bin_size),
            int(n_bins),
            str(stats_path),
            len(fig_paths),
        )
        logger.info("[RouterHeatmap] token_count_per_bin=%s", tok_cnt.detach().cpu().tolist())
        return control


def split_streaming_dataset(
    full_streaming_dataset,
    validation_percentage: int = 5,
) -> IterableDatasetDict:
    """
    Splits a streaming dataset into
    training and validation IterableDatasets, and supports methods like .map(), .filter(),
    .take() and properties like .features on the resulting streams.

    Args:
        full_streaming_dataset (Dataset): The name of the dataset to load (e.g., "HuggingFaceFW/fineweb").
        validation_percentage (int): The proportion of the dataset to be used for validation split.

    Returns:
        IterableDatasetDict: An IterableDatasetDict containing two IterableDataset objects: (train_stream, validation_stream).
    """
    if not (0 < validation_percentage < 100):
        raise ValueError(
            f"validation_percentage must be between 0 and 100 (exclusive). Passed: {validation_percentage}"
        )

    def split_generator(is_train: bool):
        for i, example in enumerate(full_streaming_dataset):
            if is_train:
                if i % 100 > validation_percentage:
                    yield example
            else:
                if i % 100 < validation_percentage:
                    yield example

    features = full_streaming_dataset.features
    train_stream = IterableDataset.from_generator(split_generator, gen_kwargs={"is_train": True}, features=features)
    validation_stream = IterableDataset.from_generator(
        split_generator, gen_kwargs={"is_train": False}, features=features
    )

    return IterableDatasetDict({"train": train_stream, "validation": validation_stream})
import wandb
def init_wandb(config, proj_name, run_name=None):
    import os
    import yaml
    api_key = os.getenv("WANDB_API_KEY")
    # config_json = yaml.safe_load(config.dump())
    # config_json['CODE_VERSION'] = get_git_commit_hash()
    wandb.login(key=api_key)
    wandb.init(
        # set the wandb project where this run will be logged
        project=proj_name,
        # track hyperparameters and run metadata
        config=config
    )
from transformers import TrainerCallback
import torch.nn as nn
def ddp_is_initialized():
    return dist.is_available() and dist.is_initialized()

def _iter_path_debug_modules(model):
    root = getattr(model, "module", model)
    for mod in root.modules():
        if hasattr(mod, "debug_accum") and hasattr(mod, "_debug_enabled"):
            yield mod

def _new_debug_layer_stats():
    return {
        "sum_abs_rel": 0.0,
        "sum_abs_eb": 0.0,
        "sum_batch_ratio": 0.0,
        "sum_elem_ratio": 0.0,
        "sum_entropy": 0.0,
        "sum_top1": 0.0,
        "count": 0.0,
        "samples_rel_abs": [],
        "samples_eb_abs": [],
        "samples_batch_ratio": [],
        "samples_elem_ratio": [],
        "samples_attn_top1": [],
        "samples_router_top1": [],
        "samples_router_margin": [],
        "by_seq_len": {},
        "eval_bin_state": None,
    }

def _to_float_list(values):
    if values is None:
        return []
    if isinstance(values, torch.Tensor):
        return values.detach().float().reshape(-1).cpu().tolist()
    if isinstance(values, (list, tuple)):
        out = []
        for v in values:
            try:
                out.append(float(v))
            except Exception:
                continue
        return out
    try:
        return [float(values)]
    except Exception:
        return []

_EVAL_BIN_SUM_KEYS = (
    "sum_mu_base",
    "sum_std_base",
    "sum_mu_rel",
    "sum_std_rel",
    "sum_r",
    "sum_kl",
    "count",
)
_EVAL_BIN_SAMPLE_KEYS = (
    "samples_std_base",
    "samples_std_rel",
    "samples_r",
    "samples_kl",
)


def _safe_int(x, default=0):
    try:
        return int(x)
    except Exception:
        return int(default)


def _normalize_vec(vals, vec_len: int):
    fl = _to_float_list(vals)
    if not fl:
        return [0.0] * vec_len
    if len(fl) == 1 and vec_len > 1:
        return [float(fl[0])] * vec_len
    if len(fl) < vec_len:
        fl = fl + [0.0] * (vec_len - len(fl))
    return [float(v) for v in fl[:vec_len]]


def _normalize_eval_bin_state(state, max_samples_per_metric=4096):
    if not isinstance(state, dict):
        return None
    vec_len = max(1, _safe_int(state.get("vec_len", 1), 1))
    out = {
        "bin_size": max(1, _safe_int(state.get("bin_size", 256), 256)),
        "eps": float(state.get("eps", 1e-6)),
        "per_head": bool(state.get("per_head", False)),
        "max_samples_per_bin": max(
            64, _safe_int(state.get("max_samples_per_bin", max_samples_per_metric), max_samples_per_metric)
        ),
        "vec_len": vec_len,
        "bins": {},
    }
    bins = state.get("bins", {})
    if not isinstance(bins, dict):
        return out

    cap = min(out["max_samples_per_bin"], int(max_samples_per_metric))
    for bidx_raw, src_bucket in bins.items():
        if not isinstance(src_bucket, dict):
            continue
        try:
            bidx = int(bidx_raw)
        except Exception:
            continue
        dst_bucket = {}
        for k in _EVAL_BIN_SUM_KEYS:
            dst_bucket[k] = _normalize_vec(src_bucket.get(k, 0.0), vec_len)
        for sk in _EVAL_BIN_SAMPLE_KEYS:
            vals = _to_float_list(src_bucket.get(sk, []))
            dst_bucket[sk] = [float(v) for v in vals[:cap]]
            seen_k = "seen_" + sk[len("samples_") :]
            dst_bucket[seen_k] = max(_safe_int(src_bucket.get(seen_k, len(dst_bucket[sk])), len(dst_bucket[sk])), len(dst_bucket[sk]))
        out["bins"][bidx] = dst_bucket
    return out


def _merge_eval_bin_state_inplace(dst_state, src_state, max_samples_per_metric=4096):
    src = _normalize_eval_bin_state(src_state, max_samples_per_metric=max_samples_per_metric)
    if src is None:
        return dst_state
    if dst_state is None:
        return src
    dst = _normalize_eval_bin_state(dst_state, max_samples_per_metric=max_samples_per_metric)
    if dst is None:
        return src

    if dst["vec_len"] != src["vec_len"]:
        # Unexpected shape mismatch; keep destination to avoid corrupting stats.
        return dst

    dst["per_head"] = bool(dst.get("per_head", False) or src.get("per_head", False))
    dst["bin_size"] = int(src.get("bin_size", dst.get("bin_size", 256)))
    dst["eps"] = float(src.get("eps", dst.get("eps", 1e-6)))
    dst["max_samples_per_bin"] = min(
        int(max_samples_per_metric),
        max(
            64,
            int(
                max(
                    dst.get("max_samples_per_bin", max_samples_per_metric),
                    src.get("max_samples_per_bin", max_samples_per_metric),
                )
            ),
        ),
    )
    cap = int(dst["max_samples_per_bin"])

    for bidx, src_bucket in src["bins"].items():
        dst_bucket = dst["bins"].get(bidx)
        if dst_bucket is None:
            dst["bins"][bidx] = src_bucket
            continue
        for k in _EVAL_BIN_SUM_KEYS:
            da = dst_bucket.get(k, [0.0] * dst["vec_len"])
            sa = src_bucket.get(k, [0.0] * dst["vec_len"])
            dst_bucket[k] = [float(a) + float(b) for a, b in zip(da, sa)]
        for sk in _EVAL_BIN_SAMPLE_KEYS:
            merged = dst_bucket.get(sk, []) + src_bucket.get(sk, [])
            if len(merged) > cap:
                merged = merged[:cap]
            dst_bucket[sk] = merged
            seen_k = "seen_" + sk[len("samples_") :]
            dst_bucket[seen_k] = int(dst_bucket.get(seen_k, 0)) + int(src_bucket.get(seen_k, 0))
    return dst


def _quantile_dict(values):
    vals = _to_float_list(values)
    if not vals:
        return {"p50": float("nan"), "p90": float("nan"), "p99": float("nan")}
    t = torch.tensor(vals, dtype=torch.float32)
    q = torch.quantile(t, torch.tensor([0.5, 0.9, 0.99], dtype=torch.float32))
    return {
        "p50": float(q[0].item()),
        "p90": float(q[1].item()),
        "p99": float(q[2].item()),
    }


def _finalize_eval_bin_state(state):
    s = _normalize_eval_bin_state(state)
    if s is None:
        return []

    bins_out = []
    bin_size = int(s.get("bin_size", 256))
    vec_len = int(s.get("vec_len", 1))
    per_head = bool(s.get("per_head", False))

    for bidx in sorted(s["bins"].keys()):
        b = s["bins"][bidx]
        count = _normalize_vec(b.get("count", [0.0]), vec_len)
        count = [max(float(c), 1.0) for c in count]

        def _metric(name):
            sums = _normalize_vec(b.get(f"sum_{name}", [0.0]), vec_len)
            per_head_vals = [float(sv) / float(cv) for sv, cv in zip(sums, count)]
            return per_head_vals

        mu_base_h = _metric("mu_base")
        std_base_h = _metric("std_base")
        mu_rel_h = _metric("mu_rel")
        std_rel_h = _metric("std_rel")
        r_h = _metric("r")
        kl_h = _metric("kl")

        rec = {
            "bin_start": int(bidx) * bin_size,
            "bin_end": int(bidx) * bin_size + bin_size,
            "count": float(sum(count) / len(count)),
            "mu_base": float(sum(mu_base_h) / len(mu_base_h)),
            "std_base": float(sum(std_base_h) / len(std_base_h)),
            "mu_rel": float(sum(mu_rel_h) / len(mu_rel_h)),
            "std_rel": float(sum(std_rel_h) / len(std_rel_h)),
            "r": float(sum(r_h) / len(r_h)),
            "kl": float(sum(kl_h) / len(kl_h)),
            "std_base_q": _quantile_dict(b.get("samples_std_base", [])),
            "std_rel_q": _quantile_dict(b.get("samples_std_rel", [])),
            "r_q": _quantile_dict(b.get("samples_r", [])),
            "kl_q": _quantile_dict(b.get("samples_kl", [])),
        }
        if per_head:
            rec["mu_base_per_head"] = mu_base_h
            rec["std_base_per_head"] = std_base_h
            rec["mu_rel_per_head"] = mu_rel_h
            rec["std_rel_per_head"] = std_rel_h
            rec["r_per_head"] = r_h
            rec["kl_per_head"] = kl_h
        bins_out.append(rec)
    return bins_out

def _merge_debug_layer_stats_inplace(dst, src, max_samples_per_layer=4096):
    if src is None:
        return

    sum_keys = (
        "sum_abs_rel",
        "sum_abs_eb",
        "sum_batch_ratio",
        "sum_elem_ratio",
        "sum_entropy",
        "sum_top1",
        "count",
    )
    for k in sum_keys:
        try:
            dst[k] += float(src.get(k, 0.0))
        except Exception:
            pass

    sample_keys = (
        "samples_rel_abs",
        "samples_eb_abs",
        "samples_batch_ratio",
        "samples_elem_ratio",
        "samples_attn_top1",
        "samples_router_top1",
        "samples_router_margin",
    )
    for k in sample_keys:
        vals = _to_float_list(src.get(k, []))
        if not vals:
            continue
        remain = max_samples_per_layer - len(dst[k])
        if remain > 0:
            dst[k].extend(vals[:remain])

    by_len = src.get("by_seq_len", {})
    if isinstance(by_len, dict):
        for seq_len, bst in by_len.items():
            try:
                seq_len_i = int(seq_len)
            except Exception:
                continue
            dst_bucket = dst["by_seq_len"].setdefault(
                seq_len_i,
                {
                    "sum_abs_rel": 0.0,
                    "sum_abs_eb": 0.0,
                    "sum_batch_ratio": 0.0,
                    "sum_elem_ratio": 0.0,
                    "count": 0.0,
                },
            )
            for k in ("sum_abs_rel", "sum_abs_eb", "sum_batch_ratio", "sum_elem_ratio", "count"):
                try:
                    dst_bucket[k] += float(bst.get(k, 0.0))
                except Exception:
                    pass

    dst["eval_bin_state"] = _merge_eval_bin_state_inplace(
        dst.get("eval_bin_state"),
        src.get("eval_bin_state"),
        max_samples_per_metric=max_samples_per_layer,
    )

def _collect_local_path_debug_accum(model, max_samples_per_layer=4096):
    merged = {}
    for mod in _iter_path_debug_modules(model):
        acc = getattr(mod, "debug_accum", None)
        if not isinstance(acc, dict):
            continue
        for layer, st in acc.items():
            layer_key = int(layer) if isinstance(layer, int) or (isinstance(layer, str) and layer.isdigit()) else layer
            dst = merged.get(layer_key)
            if dst is None:
                dst = _new_debug_layer_stats()
                merged[layer_key] = dst
            _merge_debug_layer_stats_inplace(dst, st, max_samples_per_layer=max_samples_per_layer)
    return merged

def _layer_sort_key(layer):
    if isinstance(layer, int):
        return (0, layer)
    return (1, str(layer))

def _finalize_debug_summary(merged):
    if not merged:
        return None

    eps = 1e-12
    out_all = {}
    for layer in sorted(merged.keys(), key=_layer_sort_key):
        st = merged[layer]
        count = max(float(st.get("count", 0.0)), 1.0)
        rel_abs_mean = float(st.get("sum_abs_rel", 0.0)) / count
        eb_abs_mean = float(st.get("sum_abs_eb", 0.0)) / count
        out = {
            "batch_count": int(round(float(st.get("count", 0.0)))),
            "rel_abs_mean": rel_abs_mean,
            "eb_abs_mean": eb_abs_mean,
            "rel_over_eb": rel_abs_mean / max(eb_abs_mean, eps),
            "batch_ratio_mean": float(st.get("sum_batch_ratio", 0.0)) / count,
            "elem_ratio_mean": float(st.get("sum_elem_ratio", 0.0)) / count,
            "entropy_mean": float(st.get("sum_entropy", 0.0)) / count,
            "top1_mean": float(st.get("sum_top1", 0.0)) / count,
        }

        by_len_out = {}
        by_len = st.get("by_seq_len", {})
        if isinstance(by_len, dict):
            for seq_len in sorted(by_len.keys()):
                bst = by_len[seq_len]
                c = max(float(bst.get("count", 0.0)), 1.0)
                rel_m = float(bst.get("sum_abs_rel", 0.0)) / c
                eb_m = float(bst.get("sum_abs_eb", 0.0)) / c
                by_len_out[int(seq_len)] = {
                    "batch_count": int(round(float(bst.get("count", 0.0)))),
                    "rel_abs_mean": rel_m,
                    "eb_abs_mean": eb_m,
                    "rel_over_eb": rel_m / max(eb_m, eps),
                    "batch_ratio_mean": float(bst.get("sum_batch_ratio", 0.0)) / c,
                    "elem_ratio_mean": float(bst.get("sum_elem_ratio", 0.0)) / c,
                }
        if by_len_out:
            out["by_seq_len"] = by_len_out

        for name, skey in (
            ("rel_abs", "samples_rel_abs"),
            ("eb_abs", "samples_eb_abs"),
            ("batch_ratio", "samples_batch_ratio"),
            ("elem_ratio", "samples_elem_ratio"),
        ):
            vals = _to_float_list(st.get(skey, []))
            if vals:
                t = torch.tensor(vals, dtype=torch.float32)
                q = torch.quantile(t, torch.tensor([0.9, 0.99], dtype=torch.float32))
                out[f"{name}_p90"] = float(q[0].item())
                out[f"{name}_p99"] = float(q[1].item())

        eval_bins = _finalize_eval_bin_state(st.get("eval_bin_state"))
        if eval_bins:
            out["eval_bins"] = eval_bins

        out_all[layer] = out
    return out_all

def _format_eval_debug_summary(summary):
    if not summary:
        return "[EVAL DEBUG SUMMARY] empty"

    lines = ["[EVAL DEBUG SUMMARY]"]
    layers = sorted(summary.keys(), key=_layer_sort_key)
    uniq_seq_lens = set()

    header = (
        "layer | batches | rel_over_eb | batch_ratio_mean | elem_ratio_mean | "
        "rel_abs_mean | eb_abs_mean"
    )
    lines.append(header)
    lines.append("-" * len(header))

    for layer in layers:
        st = summary[layer]
        lines.append(
            f"{layer:>5} | {int(st.get('batch_count', 0)):>7d} | "
            f"{float(st.get('rel_over_eb', 0.0)):.6e} | "
            f"{float(st.get('batch_ratio_mean', 0.0)):.6e} | "
            f"{float(st.get('elem_ratio_mean', 0.0)):.6e} | "
            f"{float(st.get('rel_abs_mean', 0.0)):.6e} | "
            f"{float(st.get('eb_abs_mean', 0.0)):.6e}"
        )

        by_len = st.get("by_seq_len", {})
        if isinstance(by_len, dict) and by_len:
            parts = []
            for seq_len in sorted(by_len.keys()):
                uniq_seq_lens.add(int(seq_len))
                bst = by_len[seq_len]
                parts.append(
                    f"T={int(seq_len)}:ratio={float(bst.get('batch_ratio_mean', 0.0)):.6e},n={int(bst.get('batch_count', 0))}"
                )
            lines.append("  by_seq_len: " + " | ".join(parts))

        eval_bins = st.get("eval_bins", [])
        if isinstance(eval_bins, list) and eval_bins:
            for b in eval_bins:
                lines.append(
                    f"[EvalStats] layer={layer} bin={int(b.get('bin_start', 0))}-{int(b.get('bin_end', 0))} "
                    f"mu_base={float(b.get('mu_base', 0.0)):.6e} std_base={float(b.get('std_base', 0.0)):.6e} "
                    f"mu_rel={float(b.get('mu_rel', 0.0)):.6e} std_rel={float(b.get('std_rel', 0.0)):.6e} "
                    f"R={float(b.get('r', 0.0)):.6e} KL={float(b.get('kl', 0.0)):.6e} "
                    f"std_base_p50={float(b.get('std_base_q', {}).get('p50', float('nan'))):.6e} "
                    f"std_base_p90={float(b.get('std_base_q', {}).get('p90', float('nan'))):.6e} "
                    f"std_base_p99={float(b.get('std_base_q', {}).get('p99', float('nan'))):.6e} "
                    f"std_rel_p50={float(b.get('std_rel_q', {}).get('p50', float('nan'))):.6e} "
                    f"std_rel_p90={float(b.get('std_rel_q', {}).get('p90', float('nan'))):.6e} "
                    f"std_rel_p99={float(b.get('std_rel_q', {}).get('p99', float('nan'))):.6e} "
                    f"R_p50={float(b.get('r_q', {}).get('p50', float('nan'))):.6e} "
                    f"R_p90={float(b.get('r_q', {}).get('p90', float('nan'))):.6e} "
                    f"R_p99={float(b.get('r_q', {}).get('p99', float('nan'))):.6e} "
                    f"KL_p50={float(b.get('kl_q', {}).get('p50', float('nan'))):.6e} "
                    f"KL_p90={float(b.get('kl_q', {}).get('p90', float('nan'))):.6e} "
                    f"KL_p99={float(b.get('kl_q', {}).get('p99', float('nan'))):.6e}"
                )

    if len(uniq_seq_lens) <= 1:
        seq_desc = "none" if len(uniq_seq_lens) == 0 else str(next(iter(uniq_seq_lens)))
        lines.append(
            f"note: only one sequence length in eval ({seq_desc}); cannot compare extended-length differences yet."
        )
    else:
        lines.append(f"note: compared sequence lengths = {sorted(uniq_seq_lens)}")

    return "\n".join(lines)

@torch.no_grad()
def merge_debug_across_ranks(model, device="cuda"):
    """
    Merge PaTHAttention eval debug accumulators across ranks into rank0.
    """
    _ = device  # keep signature stable
    root = getattr(model, "module", model)
    cfg = getattr(root, "config", None)
    rank0_only = True
    merge_cap = 256
    if cfg is not None:
        try:
            rank0_only = bool(getattr(cfg, "eval_rel_stats_rank0_only", True))
        except Exception:
            rank0_only = True
        try:
            merge_cap = max(16, int(getattr(cfg, "eval_rel_stats_merge_max_samples", 256)))
        except Exception:
            merge_cap = 256
    local_acc = _collect_local_path_debug_accum(model, max_samples_per_layer=int(merge_cap))

    if not ddp_is_initialized():
        return _finalize_debug_summary(local_acc)

    rank = dist.get_rank()
    if rank0_only:
        return _finalize_debug_summary(local_acc) if rank == 0 else None

    world = dist.get_world_size()
    gathered = [None for _ in range(world)]
    dist.all_gather_object(gathered, local_acc)

    if rank != 0:
        return None

    merged = {}
    for part in gathered:
        if not isinstance(part, dict):
            continue
        for layer, st in part.items():
            dst = merged.get(layer)
            if dst is None:
                dst = _new_debug_layer_stats()
                merged[layer] = dst
            _merge_debug_layer_stats_inplace(dst, st)

    out = _finalize_debug_summary(merged)
    return out
class ParamTrackerCallback(TrainerCallback):
    def __init__(
        self,
        param_name_list,
        log_every_n_steps: int = 50,
        tag_prefix: str = "",
    ):
        """
        param_name_list: 需要跟踪的参数名（与 model.named_parameters() 的 name 精确匹配）
        log_every_n_steps: 每隔多少个 global_step 记录一次（依赖 Trainer 的 logging_steps/on_log 触发）
        tag_prefix: 日志前缀（可为空），方便区分不同实验
        """
        self.param_name_list = list(dict.fromkeys(param_name_list))  # 去重保持顺序
        self.log_every_n_steps = log_every_n_steps
        self.tag_prefix = (tag_prefix.rstrip("/") + "/") if tag_prefix else ""
        self._resolved = None
        self._warned_missing: bool = False

    # -------- 内部工具 --------
    def _resolve_params(self, model: nn.Module):
        name_to_param = dict(model.named_parameters())
        resolved, missing = {}, []
        for name in self.param_name_list:
            p = name_to_param.get(name)
            if p is not None:
                resolved[name] = p
            else:
                missing.append(name)
        if missing and not self._warned_missing:
            self._warned_missing = True
        return resolved

    def _ensure_resolved(self, model: nn.Module):
        if self._resolved is None:
            self._resolved = self._resolve_params(model)

    @torch.no_grad()
    def _collect_stats(self, p: torch.nn.Parameter):
        d = p.data
        return dict(
            mean=float(d.mean().item()),
            std=float(d.std(unbiased=False).item()),
            l2=float(d.norm(2).item()),
            max_abs=float(d.abs().max().item()),
        )

    # -------- 训练起始时先解析一次，避免每步开销 --------
    def on_train_begin(self, args, state, control, **kwargs):
        model: nn.Module = kwargs["model"]
        self._ensure_resolved(model)

    # -------- 日志触发点：往 logs 里“就地”塞入自定义指标 --------
    def on_log(self, args, state, control, logs=None, **kwargs):
        # HF 会在 _maybe_log_save_evaluate 里构建 logs 并调用 on_log
        # 我们在这里追加自定义日志（且只在 rank0 打印/记录）
        if logs is None:
            return
        if state.global_step == 0 or (state.global_step % self.log_every_n_steps) != 0:
            return
        if hasattr(state, "is_world_process_zero") and not state.is_world_process_zero:
            return

        model: nn.Module = kwargs["model"]
        self._ensure_resolved(model)

        # 逐参数统计并注入 logs（这样会被正常写入到 TensorBoard/W&B 等）
        summary_parts = []
        for name, p in self._resolved.items():
            s = self._collect_stats(p)
            logs[f"{self.tag_prefix}{name}/mean"] = s["mean"]
            logs[f"{self.tag_prefix}{name}/std"] = s["std"]
            logs[f"{self.tag_prefix}{name}/l2"] = s["l2"]
            logs[f"{self.tag_prefix}{name}/max_abs"] = s["max_abs"]
            summary_parts.append(
                f"{name}: μ={s['mean']:.6f}, σ={s['std']:.6f}, ‖·‖₂={s['l2']:.6f}, max|·|={s['max_abs']:.6f}"
            )

    # -------- 可选：在保存前打印一份（on_save 没有 logs 可写）--------
    def on_save(self, args, state, control, **kwargs):
        if hasattr(state, "is_world_process_zero") and not state.is_world_process_zero:
            return
        if not self._resolved:
            return
        summary_parts = []
        for name, p in self._resolved.items():
            s = self._collect_stats(p)
            summary_parts.append(f"{name}: μ(save)={s['mean']:.6f}")
        if summary_parts:
            logger.info("[ParamTracker] on_save:\n" + "\n".join(summary_parts))
class PathDebugEvalCallback(TrainerCallback):
    def __init__(self):
        self._eval_active = False

    def on_prediction_step(self, args, state, control, **kwargs):
        model = kwargs["model"]
        # 进入 eval loop 的第一次 prediction_step
        if not self._eval_active:
            self._eval_active = True
            # reset all PaTHAttention-like debug modules
            for mod in _iter_path_debug_modules(model):
                mod.debug_accum = {}
                mod._debug_enabled = True
                if hasattr(mod, "_debug_probe_done"):
                    mod._debug_probe_done = False
                if hasattr(mod, "reset_eval_stats"):
                    mod.reset_eval_stats()
                else:
                    if hasattr(mod, "_eval_bin_stats"):
                        mod._eval_bin_stats = {}
                    if hasattr(mod, "_eval_batch_step"):
                        mod._eval_batch_step = 0
                    if hasattr(mod, "_eval_stats_logged_once"):
                        mod._eval_stats_logged_once = False
        return control

    def on_evaluate(self, args, state, control, **kwargs):
        model = kwargs["model"]
        # 这里做 merge + 打印
        merged = merge_debug_across_ranks(model, device=str(args.device))
        if merged is not None:
            # rank0 打印紧凑摘要（原始 dict 如需可用 PATH_DEBUG_RAW=1 打开）
            print(_format_eval_debug_summary(merged))
            if os.environ.get("PATH_DEBUG_RAW", "0") == "1":
                print("[EVAL DEBUG RAW]", merged)

        # 关闭 debug
        self._eval_active = False
        for mod in _iter_path_debug_modules(model):
            mod._debug_enabled = False
        return control            


class RelStatsCallback(TrainerCallback):
    def __init__(self):
        self.enabled = False
        self.log_every = 500
        self.eval_every = 0
        self.tail_tau = 1024
        self.param_keywords = ("rel", "wavelet", "router", "seq_pe", "q_corr", "wavelet_dtt")
        self.log_to_trainer = False
        self._eval_active = False
        self._eval_count = 0
        self._eval_collect_this_round = False
        self._trainer = None
        self._pending_train_metrics = {}
        self._pre_step_payload = {}
        self._config_logged = False

    def set_trainer(self, trainer):
        self._trainer = trainer

    @staticmethod
    def _cfg_or_arg(model, args, key: str, default=None):
        root = getattr(model, "module", model) if model is not None else None
        cfg = getattr(root, "config", None) if root is not None else None
        if cfg is not None and hasattr(cfg, key):
            val = getattr(cfg, key, None)
            if val is not None:
                return val
        return getattr(args, key, default)

    @staticmethod
    def _is_rank0(state) -> bool:
        if hasattr(state, "is_world_process_zero"):
            return bool(state.is_world_process_zero)
        if dist.is_available() and dist.is_initialized():
            return int(dist.get_rank()) == 0
        return True

    @staticmethod
    def _fmt(x):
        try:
            return f"{float(x):.6f}"
        except Exception:
            return "nan"

    @staticmethod
    def _fmt_sci(x):
        try:
            return f"{float(x):.6e}"
        except Exception:
            return "nan"

    @staticmethod
    def _parse_keywords(raw):
        if raw is None:
            return ("rel", "wavelet", "router", "seq_pe", "q_corr", "wavelet_dtt")
        if isinstance(raw, str):
            parts = [p.strip().lower() for p in raw.split(",") if p.strip()]
            return tuple(parts) if parts else ("rel", "wavelet", "router", "seq_pe", "q_corr", "wavelet_dtt")
        if isinstance(raw, (list, tuple, set)):
            parts = [str(p).strip().lower() for p in raw if str(p).strip()]
            return tuple(parts) if parts else ("rel", "wavelet", "router", "seq_pe", "q_corr", "wavelet_dtt")
        return ("rel", "wavelet", "router", "seq_pe", "q_corr", "wavelet_dtt")

    @staticmethod
    def _infer_layer_id(name: str):
        try:
            return _infer_layer_id_from_name(name)
        except Exception:
            return None

    @staticmethod
    def _parse_layer_set(v):
        if v is None:
            return None
        if isinstance(v, str):
            s = v.strip().lower()
            if s in ("all", "*"):
                return None
            out = set()
            for tok in s.replace("[", "").replace("]", "").split(","):
                tok = tok.strip()
                if not tok:
                    continue
                try:
                    out.add(int(tok))
                except Exception:
                    continue
            return out
        if isinstance(v, (list, tuple, set)):
            out = set()
            for item in v:
                try:
                    out.add(int(item))
                except Exception:
                    continue
            return out
        try:
            return {int(v)}
        except Exception:
            return None

    @staticmethod
    def _to_float(x, default=float("nan")):
        try:
            return float(x)
        except Exception:
            return float(default)

    def _layer_layout(self, model):
        root = getattr(model, "module", model)
        cfg = getattr(root, "config", None)
        n_layer = None
        if cfg is not None:
            n_layer = getattr(cfg, "n_layer", None)
            if n_layer is None:
                n_layer = getattr(cfg, "num_hidden_layers", None)
        if n_layer is None:
            n_layer = 0
            for mod in _iter_path_debug_modules(model):
                lid = getattr(mod, "layer_idx", None)
                if lid is None:
                    continue
                n_layer = max(int(n_layer), int(lid) + 1)
        n_layer = max(int(n_layer), 0)
        layer_ids = list(range(n_layer))

        active_set = None
        if cfg is not None:
            active_set = self._parse_layer_set(getattr(cfg, "rel_use_layer_list", None))
        if active_set is None:
            enabled = {l: True for l in layer_ids}
        else:
            enabled = {l: (int(l) in active_set) for l in layer_ids}
        mask = "".join(["1" if enabled.get(l, False) else "0" for l in layer_ids]) if layer_ids else ""
        return layer_ids, enabled, mask

    def _init_layer_rec(self, enabled: bool):
        return {
            "enabled": 1.0 if enabled else 0.0,
            "A": float("nan"),
            "corr_grad_rel": float("nan"),
            "grad_logits_abs_mean": float("nan"),
            "grad_logits_std": float("nan"),
            "grad_logits_max": float("nan"),
            "grad_norm_total_logits": float("nan"),
            "grad_norm_coe": float("nan"),
            "grad_norm_rel_params": float("nan"),
            "coe": float("nan"),
            "delta_coe": float("nan"),
            "delta_rel_params": float("nan"),
        }

    def _log_config_once(self, args, state, model):
        if self._config_logged or (not self.enabled) or (not self._is_rank0(state)):
            return
        layer_ids, enabled, mask = self._layer_layout(model)
        heads_cfg = self._cfg_or_arg(model, args, "log_rel_sample_heads", "0,3,7,11")
        qpos_cfg = self._cfg_or_arg(model, args, "log_rel_sample_qpos", "128,512,2048")
        key_offsets_cfg = self._cfg_or_arg(model, args, "log_rel_sample_key_offsets", "0,16,64,256,1024")
        msg = (
            f"[RELLOG][config] rel_layer_mask={mask} "
            f"log_every={int(self.log_every)} eval_every={int(self.eval_every)} "
            f"heads={heads_cfg} "
            f"qpos={qpos_cfg} "
            f"key_offsets={key_offsets_cfg} "
            f"tail_tau={int(self.tail_tau)} keywords={','.join(self.param_keywords)} "
            f"layers={len(layer_ids)} enabled_layers={sum(int(v) for v in enabled.values())}"
        )
        logger.info(msg)
        self._config_logged = True

    def on_train_begin(self, args, state, control, **kwargs):
        model = kwargs.get("model", None)
        log_rel_stats_cfg = self._cfg_or_arg(model, args, "log_rel_stats", False)
        log_rel_every_cfg = self._cfg_or_arg(model, args, "log_rel_every", 500)
        log_rel_eval_every_cfg = self._cfg_or_arg(model, args, "log_rel_eval_every", 0)
        log_rel_tail_tau_cfg = self._cfg_or_arg(model, args, "log_rel_tail_tau", 1024)
        rel_param_keywords_cfg = self._cfg_or_arg(model, args, "rel_param_keywords", None)
        log_rel_to_trainer_cfg = self._cfg_or_arg(model, args, "log_rel_to_trainer", False)

        self.enabled = bool(log_rel_stats_cfg)
        self.log_every = max(0, int(log_rel_every_cfg))
        self.eval_every = max(0, int(log_rel_eval_every_cfg))
        self.tail_tau = max(1, int(log_rel_tail_tau_cfg))
        self.param_keywords = self._parse_keywords(rel_param_keywords_cfg)
        self.log_to_trainer = bool(log_rel_to_trainer_cfg)
        if model is not None:
            for mod in _iter_path_debug_modules(model):
                if hasattr(mod, "_rel_reset_eval_buffer"):
                    mod._rel_reset_eval_buffer()
                if hasattr(mod, "_rel_eval_collect"):
                    mod._rel_eval_collect = False
        if model is not None:
            self._log_config_once(args, state, model)
        return control

    def _collect_train_layer_stats(self, model, step: int):
        layer_stats = {}
        for mod in _iter_path_debug_modules(model):
            if not hasattr(mod, "_rel_pop_train_bucket"):
                continue
            layer = int(getattr(mod, "layer_idx", -1))
            rec = layer_stats.setdefault(layer, {})
            bucket = mod._rel_pop_train_bucket(step)
            if isinstance(bucket, dict):
                n = max(float(bucket.get("n_elem", 0.0)), 1.0)
                sum_grad = float(bucket.get("sum_grad", 0.0))
                sum_grad_sq = float(bucket.get("sum_grad_sq", 0.0))
                grad_mean = sum_grad / n
                grad_var = max(sum_grad_sq / n - grad_mean * grad_mean, 0.0)
                rec["A"] = float(bucket.get("sum_A", 0.0)) / n
                rec["grad_logits_abs_mean"] = float(bucket.get("sum_abs_grad", 0.0)) / n
                rec["grad_logits_std"] = math.sqrt(grad_var)
                rec["grad_logits_max"] = float(bucket.get("max_abs_grad", 0.0))
                rec["grad_norm_total_logits"] = math.sqrt(max(sum_grad_sq, 0.0))
                coe_count = max(float(bucket.get("coe_count", 0.0)), 1.0)
                rec["coe"] = float(bucket.get("sum_coe", 0.0)) / coe_count
                corr_count = max(float(bucket.get("corr_count", 0.0)), 0.0)
                if corr_count > 0:
                    rec["corr_grad_rel"] = float(bucket.get("sum_corr_grad_rel", 0.0)) / corr_count

            coe_param = getattr(mod, "coe_for_rel", None)
            if isinstance(coe_param, torch.nn.Parameter):
                g = coe_param.grad
                if g is not None:
                    gf = g.detach().float().reshape(-1)
                    finite = torch.isfinite(gf)
                    if bool(finite.any().item()):
                        gf = gf[finite]
                        rec["grad_norm_coe"] = float(gf.norm(2).item())
                    else:
                        rec["grad_norm_coe"] = 0.0
                else:
                    rec["grad_norm_coe"] = 0.0
            elif torch.is_tensor(coe_param):
                rec["grad_norm_coe"] = 0.0
            elif coe_param is not None:
                rec["grad_norm_coe"] = 0.0
        return layer_stats

    def _match_rel_params(self, model):
        root = getattr(model, "module", model)
        kws = tuple(self.param_keywords)
        matched = []
        for name, p in root.named_parameters():
            lname = str(name).lower()
            if any(k in lname for k in kws):
                matched.append((name, p))
        return matched

    def _collect_rel_param_grad_norm_and_snapshot(self, model):
        sq_all = 0.0
        sq_by_layer = {}
        matched_snapshot = {}
        matched_layer = {}
        matched = self._match_rel_params(model)
        for name, p in matched:
            layer_id = self._infer_layer_id(name)
            matched_layer[name] = layer_id
            if p.requires_grad:
                matched_snapshot[name] = p.detach().float().clone()
            if p.grad is None:
                continue
            gf = p.grad.detach().float().reshape(-1)
            finite = torch.isfinite(gf)
            if not bool(finite.any().item()):
                continue
            if not bool(finite.all().item()):
                gf = gf[finite]
            sq = float((gf * gf).sum().item())
            sq_all += sq
            if layer_id is not None:
                sq_by_layer[layer_id] = sq_by_layer.get(layer_id, 0.0) + sq
        return (
            math.sqrt(max(sq_all, 0.0)),
            {k: math.sqrt(max(v, 0.0)) for k, v in sq_by_layer.items()},
            matched_snapshot,
            matched_layer,
        )

    def _snapshot_coe(self, model):
        out = {}
        for mod in _iter_path_debug_modules(model):
            layer = int(getattr(mod, "layer_idx", -1))
            coe = getattr(mod, "coe_for_rel", None)
            if torch.is_tensor(coe):
                out[layer] = coe.detach().float().clone()
            else:
                try:
                    out[layer] = torch.tensor(float(coe), dtype=torch.float32)
                except Exception:
                    continue
        return out

    def on_pre_optimizer_step(self, args, state, control, **kwargs):
        if (not self.enabled) or self.log_every <= 0:
            return control
        step = int(state.global_step) + 1
        if step <= 0 or (step % self.log_every != 0):
            return control
        model = kwargs.get("model", None)
        if model is None:
            return control
        self._log_config_once(args, state, model)

        layer_ids, enabled_map, _ = self._layer_layout(model)
        layer_stats = {l: self._init_layer_rec(enabled=enabled_map.get(l, True)) for l in layer_ids}

        hook_stats = self._collect_train_layer_stats(model, step=step)
        for layer, st in hook_stats.items():
            rec = layer_stats.setdefault(int(layer), self._init_layer_rec(enabled=True))
            rec.update(st)

        global_rel_grad_norm, rel_grad_by_layer, rel_snapshot, rel_snapshot_layer = (
            self._collect_rel_param_grad_norm_and_snapshot(model)
        )
        for layer, gnorm in rel_grad_by_layer.items():
            rec = layer_stats.setdefault(int(layer), self._init_layer_rec(enabled=True))
            rec["grad_norm_rel_params"] = float(gnorm)

        self._pre_step_payload[int(step)] = {
            "layer_stats": layer_stats,
            "global_grad_norm_rel_params": float(global_rel_grad_norm),
            "coe_snapshot": self._snapshot_coe(model),
            "rel_snapshot": rel_snapshot,
            "rel_snapshot_layer": rel_snapshot_layer,
        }
        return control

    def on_optimizer_step(self, args, state, control, **kwargs):
        if (not self.enabled) or self.log_every <= 0:
            return control
        step = int(state.global_step) + 1
        payload = self._pre_step_payload.pop(int(step), None)
        if payload is None:
            return control
        model = kwargs.get("model", None)
        if model is None:
            return control

        layer_stats = payload["layer_stats"]
        delta_rel_sq_global = 0.0
        base_rel_sq_global = 0.0
        delta_rel_sq_layer = {}
        base_rel_sq_layer = {}

        root = getattr(model, "module", model)
        name_to_param = dict(root.named_parameters())
        rel_snapshot = payload.get("rel_snapshot", {})
        rel_snapshot_layer = payload.get("rel_snapshot_layer", {})
        eps = 1e-12
        for name, prev in rel_snapshot.items():
            p = name_to_param.get(name, None)
            if p is None:
                continue
            cur = p.detach().float()
            diff = cur - prev
            delta_sq = float((diff * diff).sum().item())
            base_sq = float((prev * prev).sum().item())
            delta_rel_sq_global += delta_sq
            base_rel_sq_global += base_sq
            lid = rel_snapshot_layer.get(name, None)
            if lid is not None:
                delta_rel_sq_layer[lid] = delta_rel_sq_layer.get(lid, 0.0) + delta_sq
                base_rel_sq_layer[lid] = base_rel_sq_layer.get(lid, 0.0) + base_sq

        delta_rel_params_global = math.sqrt(max(delta_rel_sq_global, 0.0)) / (math.sqrt(max(base_rel_sq_global, 0.0)) + eps)
        for lid, dsq in delta_rel_sq_layer.items():
            bsq = base_rel_sq_layer.get(lid, 0.0)
            rec = layer_stats.setdefault(int(lid), self._init_layer_rec(enabled=True))
            rec["delta_rel_params"] = math.sqrt(max(dsq, 0.0)) / (math.sqrt(max(bsq, 0.0)) + eps)

        coe_snapshot = payload.get("coe_snapshot", {})
        for mod in _iter_path_debug_modules(model):
            lid = int(getattr(mod, "layer_idx", -1))
            prev = coe_snapshot.get(lid, None)
            if prev is None:
                continue
            coe = getattr(mod, "coe_for_rel", None)
            if torch.is_tensor(coe):
                cur = coe.detach().float()
            else:
                try:
                    cur = torch.tensor(float(coe), dtype=torch.float32, device=prev.device)
                except Exception:
                    continue
            delta = float((cur - prev).abs().mean().item())
            rec = layer_stats.setdefault(lid, self._init_layer_rec(enabled=True))
            rec["delta_coe"] = delta

        if not self._is_rank0(state):
            return control

        layer_ids, enabled_map, _ = self._layer_layout(model)
        metrics = {
            "rel/grad_norm_rel_params_global": float(payload.get("global_grad_norm_rel_params", float("nan"))),
            "rel/delta_rel_params_global": float(delta_rel_params_global),
        }
        lines = []
        for layer in layer_ids:
            st = layer_stats.get(layer, self._init_layer_rec(enabled=enabled_map.get(layer, True)))
            enabled = float(st.get("enabled", 1.0))
            metrics[f"rel/enabled_layer{layer}"] = enabled
            metrics[f"rel/A_layer{layer}"] = self._to_float(st.get("A", float("nan")))
            metrics[f"rel/corr_grad_rel_layer{layer}"] = self._to_float(st.get("corr_grad_rel", float("nan")))
            metrics[f"rel/grad_logits_abs_mean_layer{layer}"] = self._to_float(st.get("grad_logits_abs_mean", float("nan")))
            metrics[f"rel/grad_logits_std_layer{layer}"] = self._to_float(st.get("grad_logits_std", float("nan")))
            metrics[f"rel/grad_logits_max_layer{layer}"] = self._to_float(st.get("grad_logits_max", float("nan")))
            metrics[f"rel/grad_norm_total_logits_layer{layer}"] = self._to_float(st.get("grad_norm_total_logits", float("nan")))
            metrics[f"rel/grad_norm_coe_layer{layer}"] = self._to_float(st.get("grad_norm_coe", float("nan")))
            metrics[f"rel/grad_norm_rel_params_layer{layer}"] = self._to_float(st.get("grad_norm_rel_params", float("nan")))
            metrics[f"rel/delta_coe_layer{layer}"] = self._to_float(st.get("delta_coe", float("nan")))
            metrics[f"rel/delta_rel_params_layer{layer}"] = self._to_float(st.get("delta_rel_params", float("nan")))
            metrics[f"rel/coe_layer{layer}"] = self._to_float(st.get("coe", float("nan")))

            lines.append(
                f"[RELLOG][train_grad] step={step} layer={layer} enabled={int(enabled)} "
                f"A={self._fmt_sci(st.get('A', float('nan')))} "
                f"corr_grad_rel={self._fmt(st.get('corr_grad_rel', float('nan')))} "
                f"grad_norm_coe={self._fmt(st.get('grad_norm_coe', float('nan')))} "
                f"grad_norm_rel_params={self._fmt(st.get('grad_norm_rel_params', float('nan')))} "
                f"delta_coe={self._fmt(st.get('delta_coe', float('nan')))} "
                f"delta_rel_params={self._fmt(st.get('delta_rel_params', float('nan')))}"
            )

        for line in lines:
            logger.info(line)
        if self.log_to_trainer:
            self._pending_train_metrics[int(step)] = metrics
        return control

    def on_step_end(self, args, state, control, **kwargs):
        if not self.enabled:
            return control
        if not self.log_to_trainer:
            return control
        step = int(state.global_step)
        metrics = self._pending_train_metrics.pop(step, None)
        if metrics and self._trainer is not None and self._is_rank0(state):
            self._trainer.log(metrics)
        return control

    def on_prediction_step(self, args, state, control, **kwargs):
        if not self.enabled:
            return control
        if int(self.eval_every) <= 0:
            # Eval rel stats disabled: do one-time flag clear at eval start, then no-op.
            if not self._eval_active:
                self._eval_active = True
                model = kwargs.get("model", None)
                if model is not None:
                    for mod in _iter_path_debug_modules(model):
                        if hasattr(mod, "_rel_eval_collect"):
                            mod._rel_eval_collect = False
            return control
        model = kwargs.get("model", None)
        if model is None:
            return control
        if self._eval_active:
            return control
        self._eval_active = True
        self._eval_count += 1
        do_collect = (self._eval_count % int(self.eval_every) == 0)
        self._eval_collect_this_round = bool(do_collect)
        self._log_config_once(args, state, model)
        for mod in _iter_path_debug_modules(model):
            if hasattr(mod, "_rel_reset_eval_buffer"):
                mod._rel_reset_eval_buffer()
            if hasattr(mod, "_rel_eval_collect"):
                mod._rel_eval_collect = bool(do_collect)
        return control

    def on_evaluate(self, args, state, control, **kwargs):
        model = kwargs.get("model", None)
        self._eval_active = False
        if int(self.eval_every) <= 0:
            self._eval_collect_this_round = False
            return control
        collect_this_round = bool(self._eval_collect_this_round)
        self._eval_collect_this_round = False
        if (not self.enabled) or model is None:
            return control
        if not collect_this_round:
            return control

        layer_ids, enabled_map, _ = self._layer_layout(model)
        layer_stats = {l: {"enabled": 1.0 if enabled_map.get(l, True) else 0.0} for l in layer_ids}
        for mod in _iter_path_debug_modules(model):
            if not hasattr(mod, "_rel_pop_eval_buffer"):
                continue
            buf = mod._rel_pop_eval_buffer()
            if not isinstance(buf, dict):
                continue
            layer = int(buf.get("layer_id", getattr(mod, "layer_idx", -1)))
            count = max(float(buf.get("count", 0.0)), 1.0)
            rec = layer_stats.setdefault(layer, {"enabled": 1.0})
            rec["tail_tau"] = int(buf.get("tail_tau", self.tail_tau))
            rec["rms_base"] = float(buf.get("sum_rms_base", 0.0)) / count
            rec["rms_rel"] = float(buf.get("sum_rms_rel", 0.0)) / count
            rec["rms_relEff"] = float(buf.get("sum_rms_rel_eff", 0.0)) / count
            rec["rho"] = float(buf.get("sum_rho", 0.0)) / count
            rec["coe"] = float(buf.get("sum_coe", 0.0)) / count
            rec["logits_std"] = float(buf.get("sum_logits_std", 0.0)) / count
            rec["top1_gap"] = float(buf.get("sum_top1_gap", 0.0)) / count
            rec["logits_range"] = float(buf.get("sum_logits_range", 0.0)) / count
            rec["entropy"] = float(buf.get("sum_ent", 0.0)) / count
            rec["top1_mass"] = float(buf.get("sum_top1", 0.0)) / count
            rec["tail_mass"] = float(buf.get("sum_tail", 0.0)) / count
            corr_count = float(buf.get("corr_base_rel_count", 0.0))
            if corr_count > 0.0:
                rec["corr_base_rel"] = float(buf.get("sum_corr_base_rel", 0.0)) / corr_count
            else:
                rec["corr_base_rel"] = float("nan")
            rec["base_abs_mean"] = float(buf.get("sum_base_abs_mean", 0.0)) / count
            rec["rel_abs_mean"] = float(buf.get("sum_rel_abs_mean", 0.0)) / count
            rec["relEff_abs_mean"] = float(buf.get("sum_rel_eff_abs_mean", 0.0)) / count

        if not self._is_rank0(state):
            return control

        eval_metrics = {}
        lines = []
        for layer in layer_ids:
            st = layer_stats.get(layer, {"enabled": 1.0 if enabled_map.get(layer, True) else 0.0})
            enabled = float(st.get("enabled", 1.0))
            tau = int(st.get("tail_tau", self.tail_tau))
            eval_metrics[f"rel/enabled_layer{layer}"] = enabled
            eval_metrics[f"rel/rms_base_layer{layer}"] = self._to_float(st.get("rms_base", float("nan")))
            eval_metrics[f"rel/rms_rel_layer{layer}"] = self._to_float(st.get("rms_rel", float("nan")))
            eval_metrics[f"rel/rms_relEff_layer{layer}"] = self._to_float(st.get("rms_relEff", float("nan")))
            eval_metrics[f"rel/rho_layer{layer}"] = self._to_float(st.get("rho", float("nan")))
            eval_metrics[f"rel/coe_layer{layer}"] = self._to_float(st.get("coe", float("nan")))
            eval_metrics[f"rel/logits_std_layer{layer}"] = self._to_float(st.get("logits_std", float("nan")))
            eval_metrics[f"rel/top1_gap_layer{layer}"] = self._to_float(st.get("top1_gap", float("nan")))
            eval_metrics[f"rel/logits_range_layer{layer}"] = self._to_float(st.get("logits_range", float("nan")))
            eval_metrics[f"rel/ent_layer{layer}"] = self._to_float(st.get("entropy", float("nan")))
            eval_metrics[f"rel/top1_mass_layer{layer}"] = self._to_float(st.get("top1_mass", float("nan")))
            eval_metrics[f"rel/tail{tau}_layer{layer}"] = self._to_float(st.get("tail_mass", float("nan")))
            eval_metrics[f"rel/corr_base_rel_layer{layer}"] = self._to_float(st.get("corr_base_rel", float("nan")))
            eval_metrics[f"rel/base_abs_mean_layer{layer}"] = self._to_float(st.get("base_abs_mean", float("nan")))
            eval_metrics[f"rel/rel_abs_mean_layer{layer}"] = self._to_float(st.get("rel_abs_mean", float("nan")))
            eval_metrics[f"rel/relEff_abs_mean_layer{layer}"] = self._to_float(st.get("relEff_abs_mean", float("nan")))

            lines.append(
                f"[RELLOG][eval] step={int(state.global_step)} layer={layer} enabled={int(enabled)} "
                f"rms_base={self._fmt(st.get('rms_base', float('nan')))} "
                f"rms_rel={self._fmt(st.get('rms_rel', float('nan')))} "
                f"rms_relEff={self._fmt(st.get('rms_relEff', float('nan')))} "
                f"rho={self._fmt(st.get('rho', float('nan')))} "
                f"logits_std={self._fmt(st.get('logits_std', float('nan')))} "
                f"top1_gap={self._fmt(st.get('top1_gap', float('nan')))} "
                f"logits_range={self._fmt(st.get('logits_range', float('nan')))} "
                f"ent={self._fmt(st.get('entropy', float('nan')))} "
                f"top1_mass={self._fmt(st.get('top1_mass', float('nan')))} "
                f"tail{tau}={self._fmt(st.get('tail_mass', float('nan')))} "
                f"corr_base_rel={self._fmt(st.get('corr_base_rel', float('nan')))} "
                f"coe={self._fmt(st.get('coe', float('nan')))}"
            )

        for line in lines:
            logger.info(line)
        if eval_metrics and self._trainer is not None:
            self._trainer.log(eval_metrics)
        return control


class EvalProgressCallback(TrainerCallback):
    def __init__(self, log_every_batches: int = 0):
        self.log_every_batches = max(0, int(log_every_batches))
        self._active = False
        self._batches = 0

    @staticmethod
    def _is_rank0(state) -> bool:
        if hasattr(state, "is_world_process_zero"):
            return bool(state.is_world_process_zero)
        if dist.is_available() and dist.is_initialized():
            return int(dist.get_rank()) == 0
        return True

    def on_prediction_step(self, args, state, control, **kwargs):
        if self.log_every_batches <= 0:
            return control
        if not self._active:
            self._active = True
            self._batches = 0
            if self._is_rank0(state):
                logger.info(
                    "[EvalProgress] start global_step=%d log_every_batches=%d",
                    int(state.global_step),
                    int(self.log_every_batches),
                )
        self._batches += 1
        if self._is_rank0(state) and (self._batches % self.log_every_batches == 0):
            logger.info(
                "[EvalProgress] global_step=%d seen_eval_batches=%d",
                int(state.global_step),
                int(self._batches),
            )
        return control

    def on_evaluate(self, args, state, control, **kwargs):
        if self.log_every_batches > 0 and self._active and self._is_rank0(state):
            logger.info(
                "[EvalProgress] done global_step=%d total_eval_batches=%d",
                int(state.global_step),
                int(self._batches),
            )
        self._active = False
        self._batches = 0
        return control


class LrMonitorCallback(TrainerCallback):
    def __init__(self, group_names_expect=("main_decay","main_nodecay","B_decay","B_nodecay")):
        self.expect = set(group_names_expect)
        self._optimizer = None  # 训练开始后会缓存
        # 可选：如果你有自定义名字，初始化时传入元组即可

    # 任一钩子若提供了 optimizer，就缓存下来（不同版本/时机更稳妥）
    def on_train_begin(self, args, state, control, **kwargs):
        if "optimizer" in kwargs and kwargs["optimizer"] is not None:
            self._optimizer = kwargs["optimizer"]
        return control

    def on_step_begin(self, args, state, control, **kwargs):
        if self._optimizer is None and "optimizer" in kwargs and kwargs["optimizer"] is not None:
            self._optimizer = kwargs["optimizer"]
        return control

    def on_log(self, args, state, control, model=None, logs=None, **kwargs):
        # 部分版本在 on_log 里也会带 optimizer；若有则更新缓存
        if "optimizer" in kwargs and kwargs["optimizer"] is not None:
            self._optimizer = kwargs["optimizer"]

        if self._optimizer is None:
            # 还没拿到 optimizer，就先不打印，避免报错
            return control

        opt = self._optimizer
        lines, seen = [], set()

        for i, pg in enumerate(opt.param_groups):
            name = pg.get("name", f"group_{i}")
            lr = pg.get("lr", 0.0)
            # “active”：该组里是否有参数当前参与反传更新
            any_updating = any(getattr(p, "requires_grad", False) for p in pg["params"])
            lines.append(f"{name}: lr={lr:.6g}, active={any_updating}")
            seen.add(name)

            # 把 lr 写入日志（便于 tb/wandb）
            if logs is not None:
                logs[f"lr/{name}"] = float(lr)

        # 预期但尚未加入 optimizer 的组（比如 B 还没注册）→ 标注 N/A
        for name in (self.expect - seen):
            lines.append(f"{name}: lr=N/A, active=False (not in optimizer)")
            if logs is not None:
                logs[f"lr/{name}"] = float("nan")

        print("[LR]", " | ".join(lines))
        return control


class WaveletGateStepSyncCallback(TrainerCallback):
    """
    Sync optimizer-step counters into model config so wavelet gate logs are aligned
    to optimizer updates (not forward micro-steps).
    """

    def __init__(self, log_every_steps: int = 0):
        self.log_every_steps = max(0, int(log_every_steps))
        self._prev_gate_by_layer = {}

    @staticmethod
    def _root_model(model):
        return getattr(model, "module", model)

    @staticmethod
    def _is_rank0(state) -> bool:
        if hasattr(state, "is_world_process_zero"):
            return bool(state.is_world_process_zero)
        if dist.is_available() and dist.is_initialized():
            return int(dist.get_rank()) == 0
        return True

    @staticmethod
    def _iter_gate_scalars(root):
        for mod in root.modules():
            gate = getattr(mod, "wavelet_logit_bias_a", None)
            if torch.is_tensor(gate):
                lid = int(getattr(mod, "layer_idx", -1))
                yield lid, float(gate.detach().float().item())

    @staticmethod
    def _sync_step_fields(root, args, state):
        cfg = getattr(root, "config", None)
        if cfg is None:
            return
        setattr(cfg, "router_global_step", int(state.global_step) + 1)
        max_steps = int(getattr(state, "max_steps", 0) or getattr(args, "max_steps", 0) or 0)
        if max_steps > 0:
            setattr(cfg, "router_max_steps", max_steps)
        setattr(cfg, "router_grad_accum_steps", int(getattr(args, "gradient_accumulation_steps", 1)))

    def on_train_begin(self, args, state, control, **kwargs):
        model = kwargs.get("model", None)
        if model is None:
            return control
        root = self._root_model(model)
        self._sync_step_fields(root, args, state)
        self._prev_gate_by_layer = {lid: a for lid, a in self._iter_gate_scalars(root)}
        return control

    def on_step_begin(self, args, state, control, **kwargs):
        model = kwargs.get("model", None)
        if model is None:
            return control
        root = self._root_model(model)
        self._sync_step_fields(root, args, state)
        return control

    def on_optimizer_step(self, args, state, control, **kwargs):
        model = kwargs.get("model", None)
        if model is None:
            return control
        root = self._root_model(model)
        self._sync_step_fields(root, args, state)

        step = int(state.global_step) + 1
        rows = []
        for lid, cur in self._iter_gate_scalars(root):
            prev = float(self._prev_gate_by_layer.get(lid, cur))
            delta = float(cur - prev)
            upd_ratio = float(abs(delta) / (abs(prev) + 1e-12))
            self._prev_gate_by_layer[lid] = float(cur)
            if self.log_every_steps > 0 and (step % self.log_every_steps == 0):
                rows.append((lid, cur, delta, upd_ratio))

        if rows and self._is_rank0(state):
            rows.sort(key=lambda x: x[0])
            msg = " | ".join(
                f"l{lid}:a={cur:.6e},da={delta:.6e},ur={ur:.6e}" for lid, cur, delta, ur in rows
            )
            logger.info("[wavelet gate optstep] step=%d %s", step, msg)
        return control

from transformers import TrainerCallback
import torch
import math
import numpy as np

def _percentiles(x, ps=(0, 25, 50, 75, 100)):
    x = x.detach().float().reshape(-1).cpu()
    return {f"p{p}": float(torch.quantile(x, torch.tensor(p/100.0))) for p in ps}

class OmegaPhiMonitorCallback(TrainerCallback):
    """
    每隔 every_n_steps:
      - 遍历模型中所有 PaTHAttentionWfreq 模块
      - 计算当前有效 ω, φ （用模块的 get_omega()/get_phi()）
      - 记录均值/最值/分位数；若 report_to 包含 'tensorboard'，同时写 histogram
    """
    def __init__(self, every_n_steps=100, hist=False, group_tag="op", ps=(0, 25, 50, 75, 100)):
        self.every = every_n_steps
        self.hist = hist
        self.group_tag = group_tag  # 日志前缀
        self.ps = ps
        self.tb = None  # TensorBoard writer（按需惰性获取）

    def _maybe_get_tb(self, trainer):
        if self.tb is not None:
            return self.tb
        # 仅当 report_to 包含 tensorboard 时尝试获取
        report_to = getattr(trainer.args, "report_to", None) or []
        if isinstance(report_to, str):
            report_to = [report_to]
        if "tensorboard" in report_to:
            # _get_tb_writer 是私有接口，但目前 HF 最稳定的方式
            self.tb = getattr(trainer, "_get_tb_writer", lambda: None)()
        return self.tb

    @torch.no_grad()
    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step == 0 or state.global_step % self.every != 0:
            return

        model = kwargs["model"]
        trainer = kwargs["trainer"]
        step = state.global_step

        logs = {}
        tb = self._maybe_get_tb(trainer)

        for name, m in model.named_modules():
            # 只抓你的模块；若类名不同，请替换
            if m.__class__.__name__ != "PaTHAttentionWfreq":
                continue

            # 计算有效 ω, φ
            try:
                omega = m.get_omega()  # [1,1,Hf,r]
                phi   = m.get_phi()    # [1,1,Hf,r]
            except Exception as e:
                # 如果模块还没初始化好/有分支未覆盖，跳过该层
                continue

            # 统计信息
            w_stats = {
                "mean": float(omega.mean()),
                "min":  float(omega.min()),
                "max":  float(omega.max()),
            }
            w_stats.update(_percentiles(omega, self.ps))

            p_stats = {
                "mean": float(phi.mean()),
                "min":  float(phi.min()),
                "max":  float(phi.max()),
            }
            p_stats.update(_percentiles(phi, self.ps))

            # 写到 HF 的 log（控制台/记事本/追踪器都会收到）
            for k, v in w_stats.items():
                logs[f"{self.group_tag}/{name}/omega_{k}"] = v
            for k, v in p_stats.items():
                logs[f"{self.group_tag}/{name}/phi_{k}"] = v

            # 可选：TensorBoard histogram（更直观）
            if self.hist and tb is not None:
                tb.add_histogram(f"{self.group_tag}/{name}/omega_hist", omega.detach().cpu().reshape(-1), step)
                tb.add_histogram(f"{self.group_tag}/{name}/phi_hist",   phi.detach().cpu().reshape(-1), step)

        if logs:
            trainer.log(logs)
import pdb
import os
import re
import string
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm


def _infer_layer_id_from_name(name: str) -> Optional[int]:
    """
    Try to infer layer index from a parameter name.
    Supports common patterns: model.layers.<i>, transformer.h.<i>, decoder.layers.<i>, blocks.<i>, etc.
    """
    patterns = [
        r"(?:^|\.)(?:model\.)?layers\.(\d+)(?:\.|$)",          # LLaMA-like: model.layers.0.*
        r"(?:^|\.)(?:transformer\.)?h\.(\d+)(?:\.|$)",         # GPT2-like: transformer.h.0.*
        r"(?:^|\.)(?:decoder\.)?layers\.(\d+)(?:\.|$)",        # decoder.layers.0.*
        r"(?:^|\.)(?:encoder\.)?layer\.(\d+)(?:\.|$)",         # encoder.layer.0.*
        r"(?:^|\.)(?:blocks?)\.(\d+)(?:\.|$)",                 # blocks.0.* / block.0.*
    ]
    for pat in patterns:
        m = re.search(pat, name)
        if m:
            return int(m.group(1))
    return None


def _to_head_vector(x: torch.Tensor) -> torch.Tensor:
    """
    Convert wavelet_coe parameter tensor to a 1D [H] vector.
    Accepts scalar, [H], [1,H], [1,H,1,1], etc.
    """
    x = x.detach().float().cpu()
    if x.numel() == 1:
        return x.view(1)  # H=1
    # common per-head shapes: [H], [1,H], [H,1], [1,H,1,1] ...
    x = x.reshape(-1)
    return x


def parse_step_from_ckpt_path(ckpt_path: str) -> Optional[int]:
    """
    Best-effort step parsing from checkpoint path.
    Supports: .../checkpoint-30000, .../step_30000, .../global_step30000, etc.
    """
    pats = [
        r"checkpoint[-_](\d+)",
        r"(?:^|[/_])step[-_]?(\d+)(?:$|[/_])",
        r"global[_-]?step[-_]?(\d+)",
    ]
    for pat in pats:
        m = re.search(pat, ckpt_path)
        if m:
            return int(m.group(1))
    return None

def _get_tokenizer():
    if GLOBAL_TOKENIZER is None:
        raise NameError("GLOBAL_TOKENIZER is not set; set it before dataset.map")
    return GLOBAL_TOKENIZER


def _token_ids(text: str):
    tok = _get_tokenizer()
    return tok(text, add_special_tokens=False)["input_ids"]

def _build_title2sents(ex):
    ctx = ex.get("context", None)
    title2sents = {}
    if isinstance(ctx, dict) and "title" in ctx and "sentences" in ctx:
        for t, sents in zip(ctx["title"], ctx["sentences"]):
            title2sents[str(t)] = sents
    elif isinstance(ctx, list):
        for item in ctx:
            if isinstance(item, (list, tuple)) and len(item) == 2:
                t, sents = item
                title2sents[str(t)] = sents
            elif isinstance(item, dict):
                t = item.get("title", "")
                sents = item.get("sentences", item.get("sents", []))
                title2sents[str(t)] = sents
    return title2sents

def _get_supporting_pairs(ex):
    sf = ex.get("supporting_facts", None)
    if sf is None:
        return []
    # 你截图里是 dict: {"title":[...], "sent_id":[...]}
    if isinstance(sf, dict) and "title" in sf and "sent_id" in sf:
        return [(str(t), int(i)) for t, i in zip(sf["title"], sf["sent_id"])]
    # 常见另一种：list of [title, sent_id]
    pairs = []
    for item in sf:
        if isinstance(item, (list, tuple)) and len(item) == 2:
            pairs.append((str(item[0]), int(item[1])))
    return pairs

def build_context_budgeted(ex, budget_tokens: int, prefer_same_title: bool = True, min_tokens: int = 0):
    """
    Always include supporting sentences. Then add shortest sentences until budget is hit.
    If prefer_same_title: fill from titles that appear in supporting_facts first.
    If min_tokens>0, do a second pass (longest-first) to try reaching the minimum length without exceeding budget.
    Returns (context_text, context_token_ids, status).
    """
    # Clamp to avoid impossible targets (e.g., min_tokens > budget_tokens)
    min_tokens = min(max(min_tokens, 0), max(budget_tokens, 0))
    title2sents = _build_title2sents(ex)
    tok = _get_tokenizer()
    newline_ids = tok("\n", add_special_tokens=False)["input_ids"]
    newline_cost = len(newline_ids)
    sf_pairs = _get_supporting_pairs(ex)
    if not sf_pairs:
        return None, None, "no_supporting"

    # 1) supporting sentences (must include)
    selected = []  # (t, sid, text, ids, L)
    selected_keys = set()
    total = 0

    def add_sent(t, sid):
        nonlocal total
        sents = title2sents.get(t, [])
        if not (0 <= sid < len(sents)):
            return False
        sent = sents[sid]
        if not isinstance(sent, str):
            return False
        text = f"{t}: {sent}"
        ids = _token_ids(text)
        L = len(ids)
        add_cost = L + (newline_cost if len(selected) > 0 else 0)
        if total + add_cost > budget_tokens:
            return False
        key = (t, sid)
        if key in selected_keys:
            return True
        selected_keys.add(key)
        selected.append((t, sid, text, ids, L))
        total += add_cost
        return True

    # supporting 先加（若任意 supporting 句无法加入 -> discard）
    for t, sid in sf_pairs:
        ok = add_sent(t, sid)
        if not ok:
            return None, None, "supporting_over_budget"

    # 2) build candidate pool (exclude supporting)
    # 候选：所有句子 (t, sid, text, len)
    candidates = []  # (t, sid, text, ids, L)
    sf_titles = {t for t, _ in sf_pairs}

    for t, sents in title2sents.items():
        if not isinstance(sents, list):
            continue
        for sid, sent in enumerate(sents):
            key = (t, sid)
            if key in selected_keys:
                continue
            if not isinstance(sent, str):
                continue
            # 可选：你也可以只用 sent 本体统计长度；但拼 title 更稳
            text = f"{t}: {sent}"
            ids = _token_ids(text)
            L = len(ids)
            candidates.append((t, sid, text, ids, L))

    # 3) sort by length (shortest first), but optionally prioritize same titles
    if prefer_same_title:
        candidates.sort(key=lambda x: (0 if x[0] in sf_titles else 1, x[4]))
    else:
        candidates.sort(key=lambda x: x[4])

    # 4) add until overflow; drop the last that causes overflow (i.e., just don't add it)
    for t, sid, text, ids, L in candidates:
        add_cost = L + (newline_cost if len(selected) > 0 else 0)
        if total + add_cost > budget_tokens:
            continue
        selected_keys.add((t, sid))
        selected.append((t, sid, text, ids, L))
        total += add_cost
        if total >= budget_tokens:
            break

    # 5) 如果还没达到 min_tokens，尝试用剩余候选里较长的句子填充到下界
    if total < min_tokens and candidates:
        # 只保留未使用的候选，按长度降序以尽快逼近 min_tokens
        remaining = [(t, sid, text, ids, L) for t, sid, text, ids, L in candidates if (t, sid) not in selected_keys]
        remaining.sort(key=lambda x: x[4], reverse=True)
        for t, sid, text, ids, L in remaining:
            if total >= min_tokens:
                break
            add_cost = L + (newline_cost if len(selected) > 0 else 0)
            if total + add_cost > budget_tokens:
                continue
            selected_keys.add((t, sid))
            selected.append((t, sid, text, ids, L))
            total += add_cost

    # 6) optional: keep original order (more natural), or keep "added order"
    # 更自然：按 title 在原 context 中的顺序 + sid 排序（需要你提供 title 顺序）
    # 这里先保持“添加顺序”（supporting + shortest fill (+ longest top-up)），与你的设想一致
    context_text = "\n".join([x[2] for x in selected])
    context_ids = []
    for i, x in enumerate(selected):
        if i > 0:
            context_ids.extend(newline_ids)
        context_ids.extend(x[3])
    return context_text, context_ids, "ok"

@dataclass
class WaveletCoeSnapshot:
    step: int
    layer_ids: List[int]
    head_count: int
    coe_LH: torch.Tensor          # [L, H] (nan-filled for missing)
    param_names: List[str] = field(default_factory=list)


import os
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm


def _infer_layer_id_from_name(name: str) -> Optional[int]:
    patterns = [
        r"(?:^|\.)(?:model\.)?layers\.(\d+)(?:\.|$)",
        r"(?:^|\.)(?:transformer\.)?h\.(\d+)(?:\.|$)",
        r"(?:^|\.)(?:decoder\.)?layers\.(\d+)(?:\.|$)",
        r"(?:^|\.)(?:blocks?)\.(\d+)(?:\.|$)",
    ]
    for pat in patterns:
        m = re.search(pat, name)
        if m:
            return int(m.group(1))
    return None


def _to_head_vector(x: torch.Tensor) -> torch.Tensor:
    """
    wavelet_coe: [1,H,1,1] / [H] / scalar -> return [H] (or [1] if scalar)
    """
    x = x.detach().float().cpu()
    if x.numel() == 1:
        return x.view(1)
    return x.reshape(-1)  # e.g. [1,12,1,1] -> [12]


@dataclass
class WaveletCoeSnapshot:
    step: int
    layer_ids: List[int]
    head_count: int
    coe_LH: torch.Tensor  # [L,H]
    param_names: List[str] = field(default_factory=list)


class WaveletCoeTracker:
    def __init__(
        self,
        out_dir: str,
        param_key: str = "wavelet_coe",
        vmin: float = 0.5,
        vmax: float = 1.5,
        save_each_snapshot: bool = True,
    ):
        self.out_dir = out_dir
        self.param_key = param_key
        self.vmin = vmin
        self.vmax = vmax
        self.save_each_snapshot = save_each_snapshot
        os.makedirs(self.out_dir, exist_ok=True)
        self.snapshots: List[WaveletCoeSnapshot] = []

    def _extract_from_model(self, model: torch.nn.Module) -> Tuple[List[int], int, torch.Tensor, List[str]]:
        layer_to_vec: Dict[int, torch.Tensor] = {}
        names: List[str] = []

        for n, p in model.named_parameters():
            if self.param_key not in n:
                continue
            layer_id = _infer_layer_id_from_name(n)
            if layer_id is None:
                continue
            vec = _to_head_vector(p)  # [H]
            layer_to_vec[layer_id] = vec
            names.append(n)

        if not layer_to_vec:
            raise RuntimeError(
                f"没在 model.named_parameters() 里找到包含 '{self.param_key}' 且能解析 layer_id 的参数名。"
            )

        layer_ids = sorted(layer_to_vec.keys())
        head_count = max(int(layer_to_vec[l].numel()) for l in layer_ids)

        coe = torch.full((len(layer_ids), head_count), float("nan"))
        for i, lid in enumerate(layer_ids):
            v = layer_to_vec[lid]
            coe[i, : v.numel()] = v

        return layer_ids, head_count, coe, names

    def capture(self, model: torch.nn.Module, step: int, ckpt_tag: Optional[str] = None) -> WaveletCoeSnapshot:
        layer_ids, head_count, coe_LH, names = self._extract_from_model(model)

        snap = WaveletCoeSnapshot(
            step=int(step),
            layer_ids=layer_ids,
            head_count=head_count,
            coe_LH=coe_LH,
            param_names=names,
        )
        self.snapshots.append(snap)

        if self.save_each_snapshot:
            tag = ckpt_tag if ckpt_tag is not None else f"step{step:07d}"
            path = os.path.join(self.out_dir, f"wavelet_coe_{tag}.pt")
            torch.save(
                {
                    "step": snap.step,
                    "layer_ids": snap.layer_ids,
                    "head_count": snap.head_count,
                    "coe_LH": snap.coe_LH,
                    "param_names": snap.param_names,
                },
                path,
            )
        return snap

    def plot_each_step_layer_head(
        self,
        prefix: str = "wavelet_coe",
        dpi: int = 200,
        mode: str = "robust",   # "fixed" | "per_step" | "global" | "robust"
        fixed_vmin: float = 0.5,
        fixed_vmax: float = 1.5,
        q_low: float = 0.02,    # robust 下界分位数
        q_high: float = 0.98,   # robust 上界分位数
        center_at_1: bool = False,  # True 时以 1 为中心对称色域（看“偏离 1”很直观）
    ):
        """
        每 step 一张 heatmap：y=layers, x=heads, color=coe.
        mode:
          - fixed: 用 fixed_vmin/fixed_vmax
          - per_step: 每张图自己 min/max
          - global: 全部 step 用全局 min/max
          - robust: 全部 step 用分位数范围（默认 2%-98%）
        center_at_1:
          - 如果 coe 默认初始化是 1，打开它会让色域围绕 1 对称，更好观察“偏离 1”
        """
        if not self.snapshots:
            raise RuntimeError("还没有 capture 任何 snapshot。")

        snaps = sorted(self.snapshots, key=lambda s: s.step)

        # ---- 先决定全局 vmin/vmax（global/robust 时用）----
        all_vals = None
        if mode in ("global", "robust"):
            vals = []
            for s in snaps:
                mat = s.coe_LH[:, : s.head_count].detach().cpu().numpy()
                v = mat[np.isfinite(mat)]
                if v.size:
                    vals.append(v)
            if not vals:
                raise RuntimeError("所有 snapshot 都是 NaN/Inf，无法画图。")
            all_vals = np.concatenate(vals, axis=0)

            if mode == "global":
                vmin, vmax = float(np.min(all_vals)), float(np.max(all_vals))
            else:  # robust
                vmin, vmax = np.quantile(all_vals, [q_low, q_high]).astype(float)

            if center_at_1:
                radius = max(abs(vmax - 1.0), abs(1.0 - vmin))
                vmin, vmax = 1.0 - radius, 1.0 + radius

        saved_paths = []

        for s in tqdm(snaps, desc=f"Plotting heatmaps per step ({mode})"):
            mat = s.coe_LH[:, : s.head_count].detach().cpu().numpy()
            mat = np.ma.masked_invalid(mat)

            if mode == "fixed":
                vmin, vmax = fixed_vmin, fixed_vmax
                if center_at_1:
                    radius = max(abs(vmax - 1.0), abs(1.0 - vmin))
                    vmin, vmax = 1.0 - radius, 1.0 + radius
            elif mode == "per_step":
                vv = mat.compressed()
                if vv.size == 0:
                    continue
                vmin, vmax = float(vv.min()), float(vv.max())
                if center_at_1:
                    radius = max(abs(vmax - 1.0), abs(1.0 - vmin))
                    vmin, vmax = 1.0 - radius, 1.0 + radius
            else:
                # global/robust 已经算过 vmin,vmax
                pass

            fig = plt.figure(figsize=(max(6, s.head_count * 0.6), max(4, len(s.layer_ids) * 0.35)))
            ax = fig.add_subplot(111)
            im = ax.imshow(mat, aspect="auto", vmin=vmin, vmax=vmax, origin="upper")

            ax.set_title(f"{prefix} layer x head @ step {s.step} (vmin={vmin:.4f}, vmax={vmax:.4f})")
            ax.set_xlabel("head")
            ax.set_ylabel("layer")
            ax.set_xticks(range(s.head_count))
            ax.set_xticklabels([str(h) for h in range(s.head_count)])
            ax.set_yticks(range(len(s.layer_ids)))
            ax.set_yticklabels([str(lid) for lid in s.layer_ids])

            fig.colorbar(im, ax=ax)

            out_png = os.path.join(self.out_dir, f"{prefix}_step{s.step:07d}_LxH_{mode}.png")
            fig.tight_layout()
            fig.savefig(out_png, dpi=dpi)
            plt.close(fig)
            saved_paths.append(out_png)

        return saved_paths

def parse_step_from_name(name: str) -> Optional[int]:
    m = re.search(r"checkpoint[-_](\d+)", name)
    if m:
        return int(m.group(1))
    m = re.search(r"(?:^|[_-])step[_-]?(\d+)(?:$|[_-])", name)
    if m:
        return int(m.group(1))
    return None


def get_ckpt_paths(root_dir: str, only_checkpoint_like: bool = False) -> List[str]:
    """
    root_dir: 你的 checkpoint 根目录
    only_checkpoint_like:
      - False: 收集所有子文件夹
      - True : 只收集名字像 checkpoint-xxxxx / step_xxxxx 的文件夹
    """
    ckpt_paths = []
    for name in tqdm(os.listdir(root_dir), desc="Scanning ckpt dirs"):
        p = os.path.join(root_dir, name)
        if not os.path.isdir(p):
            continue
        if only_checkpoint_like and (parse_step_from_name(name) is None):
            continue
        ckpt_paths.append(p)

    # 按 step 排序（解析不到 step 的放最后）
    ckpt_paths.sort(
        key=lambda p: (parse_step_from_name(os.path.basename(p)) is None,
                       parse_step_from_name(os.path.basename(p)) or 10**18)
    )
    return ckpt_paths
import ast

def read_kv_config(path: str) -> dict:
    cfg = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                raise ValueError(f"Bad line (no '='): {line}")
            k, v = line.split("=", 1)
            k, v = k.strip(), v.strip()
            # 自动把 "8" -> int, "0.1" -> float, "True" -> bool, "[1,2]" -> list, '"topk"' -> str
            try:
                v = ast.literal_eval(v)
            except Exception:
                # 没加引号的纯字符串会落到这里
                pass
            cfg[k] = v
    return cfg

def normalize_rel_use_layer_list(v, n_layer: Optional[int] = None):
    """
    Normalize rel layer selection to:
      - None => all layers enabled
      - sorted List[int] => enabled layers
    Accepts csv string, bracket string, list/tuple/set, scalar int.
    """
    if v is None:
        return None
    if isinstance(v, str):
        s = v.strip().lower()
        if s in ("", "all", "*"):
            return None
        s = s.strip("[]()")
        items = [tok for tok in re.split(r"[,\s]+", s) if tok]
    elif isinstance(v, (list, tuple, set)):
        items = list(v)
    else:
        items = [v]

    out, seen = [], set()
    for it in items:
        try:
            lid = int(it)
        except Exception:
            continue
        if n_layer is not None and int(n_layer) > 0:
            if lid < 0:
                lid = int(n_layer) + lid
            if lid < 0 or lid >= int(n_layer):
                continue
        if lid not in seen:
            seen.add(lid)
            out.append(lid)
    if not out:
        return None
    return sorted(out)

def cfg_bool(cfg: dict, key: str, default: bool) -> bool:
    if key not in cfg:
        return bool(default)
    v = cfg.get(key)
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return bool(v)
    if isinstance(v, str):
        s = v.strip().lower()
        if s in ("1", "true", "yes", "y", "on"):
            return True
        if s in ("0", "false", "no", "n", "off"):
            return False
    return bool(default)

def cfg_int(cfg: dict, key: str, default: int) -> int:
    if key not in cfg:
        return int(default)
    try:
        return int(cfg.get(key))
    except Exception:
        return int(default)

def cfg_float(cfg: dict, key: str, default: float) -> float:
    if key not in cfg:
        return float(default)
    try:
        return float(cfg.get(key))
    except Exception:
        return float(default)

def cfg_str(cfg: dict, key: str, default: str) -> str:
    if key not in cfg:
        return str(default)
    v = cfg.get(key)
    return str(v)

def add_missing_to_hf_config(config, kv: dict):
    existing = set(config.to_dict().keys())  # 现有配置字段
    added, skipped = [], []
    for k, v in kv.items():
        if k in existing:
            skipped.append(k)   # 已存在：忽略，不覆盖
        else:
            setattr(config, k, v)  # 不存在：新增
            added.append(k)
    return added, skipped

def force_override_hf_config(config, kv: dict, key_prefix: str):
    """Force-override config keys whose name starts with key_prefix, regardless of whether they exist."""
    overridden = []
    for k, v in kv.items():
        if k.startswith(key_prefix):
            setattr(config, k, v)
            overridden.append(k)
    return overridden
def compute_pad_ratio_stats(train_dataset, block_size: int, n_samples: int = 5000, seed: int = 0):
    """
    Estimate pad ratio distribution on a HuggingFace-style dataset.
    Pad ratio is computed from attention_mask (preferred). Falls back to input_ids length if needed.

    pad_ratio = 1 - (#nonpad_tokens / block_size)

    Args:
        train_dataset: HF Dataset / list-like with dict samples. Each sample should have 'attention_mask'
                       (recommended) or at least 'input_ids'.
        block_size: sequence length after padding/truncation.
        n_samples: number of examples to sample for stats.
        seed: random seed for sampling indices.

    Returns:
        A dict of summary stats and a few quantiles.
    """
    rng = random.Random(seed)
    N = len(train_dataset)
    n = min(n_samples, N)

    # Sample indices without replacement (fast)
    idxs = list(range(N))
    rng.shuffle(idxs)
    idxs = idxs[:n]

    pad_ratios = []

    for i in tqdm(idxs, desc="Computing pad ratio"):
        ex = train_dataset[i]

        if "attention_mask" in ex and ex["attention_mask"] is not None:
            # attention_mask is usually a list[int] of length block_size
            nonpad = int(np.sum(ex["attention_mask"]))
        else:
            # Fallback: use input_ids length (only correct if your dataset stores unpadded sequences)
            # If your dataset already stores padded sequences, this fallback will just give ~0 pad ratio.
            nonpad = len(ex["input_ids"])

        # Clamp to [0, block_size] just in case
        nonpad = max(0, min(nonpad, block_size))
        pad_ratio = 1.0 - (nonpad / float(block_size))
        pad_ratios.append(pad_ratio)

    pad_ratios = np.array(pad_ratios, dtype=np.float32)

    stats = {
        "n_samples": int(n),
        "mean_pad_ratio": float(pad_ratios.mean()),
        "std_pad_ratio": float(pad_ratios.std()),
        "min_pad_ratio": float(pad_ratios.min()),
        "max_pad_ratio": float(pad_ratios.max()),
        "p10": float(np.quantile(pad_ratios, 0.10)),
        "p25": float(np.quantile(pad_ratios, 0.25)),
        "p50": float(np.quantile(pad_ratios, 0.50)),
        "p75": float(np.quantile(pad_ratios, 0.75)),
        "p90": float(np.quantile(pad_ratios, 0.90)),
        "p95": float(np.quantile(pad_ratios, 0.95)),
        "p99": float(np.quantile(pad_ratios, 0.99)),
    }
    return stats
from random import randint

def debug_supervision(ds, tok, n=10, block_size=512):
    for _ in range(n):
        ex = ds[randint(0, len(ds)-1)]
        ids = ex["input_ids"]
        labels = ex["labels"]
        sup_pos = [i for i,y in enumerate(labels) if y != -100]
        print("supervised_tokens:", len(sup_pos), "pad_ratio:", 1 - sum(ex["attention_mask"])/block_size)
        if not sup_pos:
            print("!! no supervision (all -100) -> likely discard or bug")
            continue
        sup_text = tok.decode([ids[i] for i in sup_pos], skip_special_tokens=False)
        print("SUP TEXT:", repr(sup_text[:200]))
        print("-"*60)
import numpy as np
from tqdm import tqdm

def summarize_lengths(arr, name=""):
    if len(arr) == 0:
        return {f"{name}count": 0}
    a = np.asarray(arr, dtype=np.int64)
    return {
        f"{name}count": int(a.size),
        f"{name}mean": float(a.mean()),
        f"{name}p50": float(np.percentile(a, 50)),
        f"{name}p90": float(np.percentile(a, 90)),
        f"{name}p95": float(np.percentile(a, 95)),
        f"{name}p99": float(np.percentile(a, 99)),
        f"{name}min": int(a.min()),
        f"{name}max": int(a.max()),
    }

def stats_from_builder_until_kept(
    dataset,
    builder_fn,
    max_kept=5000,
    max_seen_cap=None,
    desc="token-stats",
):
    """
    Stop when kept (discard=False) reaches max_kept.

    dataset: datasets.Dataset or iterable of examples
    builder_fn: function(ex, idx) -> dict with keys:
        - discard: bool
        - labels: List[int]
        - attention_mask: List[int]

    max_seen_cap: optional int, stop early if seen exceeds this cap (safety).
    """
    label_tok = []
    input_tok = []
    discarded = 0
    seen = 0
    kept = 0

    it = dataset  # do NOT pre-select by length; we want "until kept"
    pbar = tqdm(total=max_kept, desc=desc)

    for i, ex in enumerate(it):
        out = builder_fn(ex, i)
        seen += 1

        if out.get("discard", False):
            discarded += 1
        else:
            labels = np.asarray(out["labels"], dtype=np.int64)
            attn = np.asarray(out["attention_mask"], dtype=np.int64)
            label_tok.append(int((labels != -100).sum()))
            input_tok.append(int(attn.sum()))
            kept += 1
            pbar.update(1)

            if kept >= max_kept:
                break

        if max_seen_cap is not None and seen >= max_seen_cap:
            break

    pbar.close()

    res = {
        "seen": int(seen),
        "kept": int(kept),
        "discarded": int(discarded),
        "discard_rate": float(discarded / max(seen, 1)),
    }
    res.update(summarize_lengths(label_tok, name="label_tok_"))
    res.update(summarize_lengths(input_tok, name="input_tok_"))
    return res

def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, SupplyTrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Sending telemetry. Tracking the example usage helps us better allocate resources to maintain them. The
    # information sent is the one passed as arguments along with your Python/PyTorch versions.
    send_example_telemetry("run_clm", model_args, data_args)

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if training_args.should_log:
        # The default of training_args.log_level is passive, so we set log level at info here to have that default.
        transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}, "
        + f"distributed training: {training_args.parallel_mode.value == 'distributed'}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # Get the datasets: you can either provide your own CSV/JSON/TXT training and evaluation files (see below)
    # or just provide the name of one of the public datasets available on the hub at https://huggingface.co/datasets/
    # (the dataset will be downloaded automatically from the datasets Hub).
    #
    # For CSV/JSON files, this script will use the column called 'text' or the first column if no column called
    # 'text' is found. You can easily tweak this behavior (see below).
    #
    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    if data_args.dataset_name == 'mix':
        xsum_raw_datasets = load_dataset(
            'xsum',
            'default',
            cache_dir=model_args.cache_dir,
            token=model_args.token,
            streaming=data_args.streaming,
            trust_remote_code=model_args.trust_remote_code,
        )
        _hotpot_long_jsonl = getattr(data_args, "hotpot_long_jsonl", None)
        if _hotpot_long_jsonl:
            # Load HotpotQA-Long augmented JSONL instead of HF distractor split.
            import json as _json
            from datasets import Dataset as _Dataset, DatasetDict as _DatasetDict
            _target_lengths = None
            _hotpot_long_lengths = getattr(data_args, "hotpot_long_lengths", None)
            if _hotpot_long_lengths:
                _target_lengths = {int(x) for x in str(_hotpot_long_lengths).split(",")}
            _records = []
            with open(_hotpot_long_jsonl) as _f:
                for _line in _f:
                    _rec = _json.loads(_line)
                    if _target_lengths and _rec["meta"]["target_total_tokens"] not in _target_lengths:
                        continue
                    _ctx_titles = [c[0] for c in _rec["context"]]
                    _ctx_sents  = [c[1] for c in _rec["context"]]
                    _records.append({
                        "_id": _rec["_id"],
                        "question": _rec["question"],
                        "answer": _rec["answer"],
                        "supporting_facts": {
                            "title": [sf[0] for sf in _rec["supporting_facts"]],
                            "sent_id": [sf[1] for sf in _rec["supporting_facts"]],
                        },
                        "context": {"title": _ctx_titles, "sentences": _ctx_sents},
                        "type": "bridge",
                        "level": "hard",
                        "placement_pct": float(_rec["meta"].get("placement_actual_pct", -1.0)),
                    })
            _hf_ds = _Dataset.from_list(_records)
            hotpot_qa_raw_datasets = _DatasetDict({"validation": _hf_ds, "test": _hf_ds})
            logger.info(f"[hotpot_long] loaded {len(_records)} examples from {_hotpot_long_jsonl}")
        else:
            hotpot_qa_raw_datasets = load_dataset(
                'hotpot_qa',
                'distractor',
                cache_dir=model_args.cache_dir,
                token=model_args.token,
                streaming=data_args.streaming,
                trust_remote_code=model_args.trust_remote_code,
            )
    else:
        if data_args.dataset_name is not None:
            # If hotpot_long_jsonl is provided for hotpot_qa, load from JSONL instead of HF hub.
            _hotpot_long_jsonl = getattr(data_args, "hotpot_long_jsonl", None)
            xsum_eval_sources = None  # default; assigned below only for xsum+validation_file path
            if data_args.dataset_name == "xsum" and (data_args.validation_file is not None or data_args.train_file is not None):
                data_files = {}
                if data_args.train_file is not None:
                    data_files["train"] = data_args.train_file
                if data_args.validation_file is not None:
                    data_files["validation"] = data_args.validation_file
                raw_datasets = load_dataset("json", data_files=data_files, cache_dir=model_args.cache_dir)
                xsum_eval_sources = None

                if data_args.dataset_name == "xsum" and "validation" in raw_datasets:
                    try:
                        xsum_eval_sources = list(raw_datasets["validation"]["document"])
                        logger.info(f"[xsum] cached {len(xsum_eval_sources)} validation source documents for source-grounded metrics")
                    except Exception as e:
                        logger.warning(f"[xsum] failed to cache source documents: {e}")
                        xsum_eval_sources = None
            
            elif data_args.dataset_name == "hotpot_qa" and _hotpot_long_jsonl:
                import json as _json
                from datasets import Dataset as _Dataset, DatasetDict as _DatasetDict
                _target_lengths = None
                _hotpot_long_lengths = getattr(data_args, "hotpot_long_lengths", None)
                if _hotpot_long_lengths:
                    _target_lengths = {int(x) for x in str(_hotpot_long_lengths).split(",")}
                _records = []
                with open(_hotpot_long_jsonl) as _f:
                    for _line in _f:
                        _rec = _json.loads(_line)
                        if _target_lengths and _rec["meta"]["target_total_tokens"] not in _target_lengths:
                            continue
                        _ctx_titles = [c[0] for c in _rec["context"]]
                        _ctx_sents  = [c[1] for c in _rec["context"]]
                        _records.append({
                            "_id": _rec["_id"],
                            "question": _rec["question"],
                            "answer": _rec["answer"],
                            "supporting_facts": {
                                "title": [sf[0] for sf in _rec["supporting_facts"]],
                                "sent_id": [sf[1] for sf in _rec["supporting_facts"]],
                            },
                            "context": {"title": _ctx_titles, "sentences": _ctx_sents},
                            "type": "bridge",
                            "level": "hard",
                            "placement_pct": float(_rec["meta"].get("placement_actual_pct", -1.0)),
                        })
                _hf_ds = _Dataset.from_list(_records)
                raw_datasets = _DatasetDict({"validation": _hf_ds, "test": _hf_ds})
                logger.info(f"[hotpot_long] loaded {len(_records)} examples from {_hotpot_long_jsonl}")
            elif data_args.dataset_name == "passkey":
                # Passkey retrieval: fully synthetic, raw_datasets is just a placeholder.
                from datasets import Dataset as _Dataset, DatasetDict as _DatasetDict
                raw_datasets = _DatasetDict({"validation": _Dataset.from_dict({"text": ["passkey_placeholder"]})})
                logger.info("[passkey] using synthetic passkey retrieval evaluation; raw_datasets is a placeholder.")
            else:
            # Downloading and loading a dataset from the hub.
                raw_datasets = load_dataset(
                    data_args.dataset_name,
                    data_args.dataset_config_name,
                    cache_dir=model_args.cache_dir,
                    token=model_args.token,
                    streaming=data_args.streaming,
                    trust_remote_code=model_args.trust_remote_code,
                )
            if "validation" not in raw_datasets:
                if data_args.streaming:
                    dataset_stream = load_dataset(
                        data_args.dataset_name,
                        data_args.dataset_config_name,
                        split="train",
                        cache_dir=model_args.cache_dir,
                        token=model_args.token,
                        streaming=data_args.streaming,
                        trust_remote_code=model_args.trust_remote_code,
                    )
                    raw_datasets = split_streaming_dataset(dataset_stream, data_args.validation_split_percentage)
                else:
                    raw_datasets["validation"] = load_dataset(
                        data_args.dataset_name,
                        data_args.dataset_config_name,
                        split=f"train[:{data_args.validation_split_percentage}%]",
                        cache_dir=model_args.cache_dir,
                        token=model_args.token,
                        streaming=data_args.streaming,
                        trust_remote_code=model_args.trust_remote_code,
                    )
                    raw_datasets["train"] = load_dataset(
                        data_args.dataset_name,
                        data_args.dataset_config_name,
                        split=f"train[{data_args.validation_split_percentage}%:]",
                        cache_dir=model_args.cache_dir,
                        token=model_args.token,
                        streaming=data_args.streaming,
                        trust_remote_code=model_args.trust_remote_code,
                    )
        else:
            data_files = {}
            dataset_args = {}
            if data_args.train_file is not None:
                data_files["train"] = data_args.train_file
            if data_args.validation_file is not None:
                data_files["validation"] = data_args.validation_file
            extension = (
                data_args.train_file.split(".")[-1]
                if data_args.train_file is not None
                else data_args.validation_file.split(".")[-1]
            )
            # `datasets` uses the "json" builder for both .json and .jsonl files.
            if extension == "jsonl":
                extension = "json"
            if data_args.train_file is not None and data_args.validation_file is not None:
                train_ext = data_args.train_file.split(".")[-1]
                valid_ext = data_args.validation_file.split(".")[-1]
                train_norm = "json" if train_ext == "jsonl" else train_ext
                valid_norm = "json" if valid_ext == "jsonl" else valid_ext
                if train_norm != valid_norm:
                    raise ValueError(
                        f"`train_file` and `validation_file` must use the same file type; got "
                        f"{train_ext!r} and {valid_ext!r}."
                    )
            if extension == "txt":
                extension = "text"
                dataset_args["keep_linebreaks"] = data_args.keep_linebreaks
            raw_datasets = load_dataset(
                extension,
                data_files=data_files,
                cache_dir=model_args.cache_dir,
                token=model_args.token,
                **dataset_args,
            )
            # If no validation data is there, validation_split_percentage will be used to divide the dataset.
            if "validation" not in raw_datasets:
                if data_args.streaming:
                    dataset_stream = load_dataset(
                        extension,
                        data_files=data_files,
                        split="train",
                        cache_dir=model_args.cache_dir,
                        token=model_args.token,
                        **dataset_args,
                    )
                    raw_datasets = split_streaming_dataset(dataset_stream, data_args.validation_split_percentage)
                else:
                    raw_datasets["validation"] = load_dataset(
                        extension,
                        data_files=data_files,
                        split=f"train[:{data_args.validation_split_percentage}%]",
                        cache_dir=model_args.cache_dir,
                        token=model_args.token,
                        **dataset_args,
                    )

                    raw_datasets["train"] = load_dataset(
                        extension,
                        data_files=data_files,
                        split=f"train[{data_args.validation_split_percentage}%:]",
                        cache_dir=model_args.cache_dir,
                        token=model_args.token,
                        **dataset_args,
                    )

    # See more about loading any type of standard or custom dataset (from files, python dict, pandas DataFrame, etc) at
    # https://huggingface.co/docs/datasets/loading_datasets.

    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.

    config_kwargs = {
        "cache_dir": model_args.cache_dir,
        "revision": model_args.model_revision,
        "token": model_args.token,
        "trust_remote_code": model_args.trust_remote_code,
    }

    if model_args.config_name:
        config = AutoConfig.from_pretrained(model_args.config_name, **config_kwargs)
    elif model_args.model_name_or_path:
        config = AutoConfig.from_pretrained(model_args.model_name_or_path, **config_kwargs)
    else:
        config = CONFIG_MAPPING[model_args.model_type]()
        logger.warning("You are instantiating a new config instance from scratch.")
        if model_args.config_overrides is not None:
            logger.info(f"Overriding config: {model_args.config_overrides}")
            config.update_from_string(model_args.config_overrides)
            logger.info(f"New config: {config}")
    tokenizer_kwargs = {
        "cache_dir": model_args.cache_dir,
        "use_fast": model_args.use_fast_tokenizer,
        "revision": model_args.model_revision,
        "token": model_args.token,
        "trust_remote_code": model_args.trust_remote_code,
    }
    if model_args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(model_args.tokenizer_name, **tokenizer_kwargs)
    elif model_args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, **tokenizer_kwargs)
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script. "
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )
    cfg = {}
    try:
        cfg = read_kv_config(model_args.cfg_path)
        added, skipped = add_missing_to_hf_config(config, cfg)
        print("added:", added)
        print("skipped:", skipped)
        overridden = []
        for _prefix in ("eval_attn_heatmap", "wavelet_mode", "wavelet_ctxscale", "wavelet_router_sigmoid", "wavelet_ctx_feat", "rel_use_layer"):
            overridden += force_override_hf_config(config, cfg, _prefix)
        # Also force a single explicit key if present
        for _key in ("wavelet_mode", "rel_use_layer_list"):
            if _key in cfg and _key not in overridden:
                setattr(config, _key, cfg[_key])
                overridden.append(_key)
        if overridden:
            print("force_overridden:", overridden)
    except Exception as e:
        pass
    # Fast eval mode during training: disable expensive eval-time statistics/extra analysis by default.
    train_eval_disable_expensive_stats = bool(
        training_args.do_train
        and training_args.do_eval
        and cfg_bool(cfg, "train_eval_disable_expensive_stats", True)
    )
    if train_eval_disable_expensive_stats:
        logger.info(
            "[EvalFast] enabled for train-time eval: disabling heavy eval stats/callbacks and decode metrics."
        )
    ################################# generate tokenizer.json #################################
    # vocab = tokenizer.get_vocab()

    # # 2) invert: id(int) -> token(str)
    # id2token = [None] * (max(vocab.values()) + 1)
    # for tok, tid in tqdm(vocab.items(), total=len(vocab), desc="Build id2token"):
    #     id2token[tid] = tok

    # # 3) sanity check (可选)
    # missing = sum(t is None for t in id2token)
    # if missing:
    #     print(f"[warn] {missing} ids are missing in id2token (unexpected).")

    # # 4) save as JSON: {"0": "...", "1": "...", ...}
    # out_path = "tokenizer.json"
    # with open(out_path, "w", encoding="utf-8") as f:
    #     json.dump({str(i): t for i, t in enumerate(id2token)}, f, ensure_ascii=False, indent=2)
    # print(f"Saved id->token mapping to: {out_path}")
    # os._exit(0)
    ################################# generate tokenizer.json #################################

    config.attn_implementation = model_args.attn_implementation
    config.use_forget_gate = model_args.use_forget_gate
    config.path_use_qk_norm = model_args.path_use_qk_norm
    config.path_use_low_rank_w = model_args.path_use_low_rank_w
    config.path_use_w_shortconv = model_args.path_use_w_shortconv
    config.path_conv_size = model_args.path_conv_size
    config.path_conv_bias = model_args.path_conv_bias
    # Important for checkpoint compatibility: only override when explicitly provided.
    if model_args.num_harmonics is not None:
        config.num_harmonics = int(model_args.num_harmonics)
    config.share_freq_across_heads = model_args.share_freq_across_heads
    if _cli_flag_present("--pe_method"):
        config.pe_method = model_args.pe_method
    if model_args.relative_type is not None:
        config.relative_type = model_args.relative_type
    config.use_beta_modulation = model_args.use_beta_modulation
    wavelet_mode_cli_provided = _cli_flag_present("--wavelet_mode")
    if wavelet_mode_cli_provided:
        config.wavelet_mode = model_args.wavelet_mode
        wavelet_mode_source = "cli"
    else:
        config.wavelet_mode = cfg_str(cfg, "wavelet_mode", model_args.wavelet_mode)
        wavelet_mode_source = "cfg_or_default"
    config.path_attn_impl = model_args.path_attn_impl
    config.use_soft_wavelet_fox = model_args.use_soft_wavelet_fox
    config.logging_steps = training_args.logging_steps
    config.wavelet_baseline_use = model_args.wavelet_baseline_use
    config.init_theta = model_args.init_theta
    config.sample_num = training_args.sample_num
    config.spectral_loss_coe = training_args.spectral_loss_coe
    config.temp_loss_coe = training_args.temp_loss_coe
    config.distill_teacher = training_args.distill_teacher
    config.distill_in_which_layers = training_args.distill_in_which_layers
    config.distill_freq_scale = training_args.distill_freq_scale
    config.smooth_use = training_args.smooth_use
    config.distilling_coe_warmup_use = training_args.distilling_coe_warmup_use
    config.scale_range = training_args.scale_range
    if not hasattr(config, 'scale_type'):
        config.scale_type = 'custom'  # default; modeling_gpt2 requires this attr
    config.path_blend_layers = training_args.path_blend_layers if training_args.path_blend_layers else None
    config.path_sparse_gate = training_args.path_sparse_gate
    config.gate_sparse_alpha = training_args.gate_sparse_alpha
    config.gate_warmup_steps = training_args.gate_warmup_steps
    config.path_gate_force_open = training_args.path_gate_force_open
    config.dataset_name = data_args.dataset_name
    config.wavelet_pe_softmax_use = training_args.wavelet_pe_softmax_use
    config.weight_alpha = training_args.weight_alpha
    config.loss_type = training_args.loss_type
    config.model_name_or_path = model_args.model_name_or_path
    config.analyzer = training_args.analyzer
    config.qk_rotation = model_args.qk_rotation
    config.ablate_switch = training_args.ablate_switch
    config.wavelet_router = model_args.wavelet_router
    config.router_band_num = model_args.router_band_num
    config.router_hidden_dim = model_args.router_hidden_dim
    config.rel_selection = cfg_str(cfg, "rel_selection", model_args.rel_selection)
    config.wavelet_analysis_export = cfg_bool(
        cfg, "wavelet_analysis_export", bool(getattr(config, "wavelet_analysis_export", False))
    )
    config.wavelet_analysis_max_q = cfg_int(
        cfg,
        "wavelet_analysis_max_q",
        int(getattr(config, "wavelet_analysis_max_q", getattr(config, "wavelet_logit_bias_log_sample_tokens", 64))),
    )
    config.wavelet_analysis_max_batches = cfg_int(
        cfg, "wavelet_analysis_max_batches", int(getattr(config, "wavelet_analysis_max_batches", 64))
    )
    config.wavelet_analysis_run_tag = cfg_str(
        cfg, "wavelet_analysis_run_tag", str(getattr(config, "wavelet_analysis_run_tag", "default"))
    )
    config.wavelet_analysis_mode = cfg_str(
        cfg, "wavelet_analysis_mode", str(getattr(config, "wavelet_analysis_mode", config.wavelet_mode))
    )
    config.wavelet_analysis_output_dir = str(training_args.output_dir)
    config.wavelet_analysis_seed = int(getattr(training_args, "seed", -1))
    config.wavelet_analysis_dataset_config = str(getattr(data_args, "dataset_config_name", "default"))
    config.wavelet_analysis_bucket_size = int(getattr(data_args, "xsum_bucket_size", 0))
    wavelet_viz_model_size_default = str(getattr(model_args, "wavelet_viz_model_size", "") or "").strip()
    if not wavelet_viz_model_size_default:
        _ms_candidates = [
            str(getattr(model_args, "config_name", "") or ""),
            str(getattr(model_args, "model_name_or_path", "") or ""),
            str(getattr(config, "_name_or_path", "") or ""),
            str(getattr(config, "model_name_or_path", "") or ""),
        ]
        _ms_text = " ".join(_ms_candidates).lower()
        if "gpt2-medium" in _ms_text or "gpt2_medium" in _ms_text or "token_even_mix_medium" in _ms_text:
            wavelet_viz_model_size_default = "gpt2-medium"
        else:
            wavelet_viz_model_size_default = "gpt2"
    wavelet_viz_sample_q_default = getattr(model_args, "wavelet_viz_sample_q", None)
    if wavelet_viz_sample_q_default is None:
        wavelet_viz_sample_q_default = int(getattr(config, "wavelet_logit_bias_log_sample_tokens", 64))
    wavelet_viz_outdir_default = str(getattr(model_args, "wavelet_viz_outdir", "") or "").strip()
    if not wavelet_viz_outdir_default:
        wavelet_viz_outdir_default = str(training_args.output_dir)
    config.wavelet_viz_export = cfg_bool(
        cfg, "wavelet_viz_export", bool(getattr(model_args, "wavelet_viz_export", False))
    )
    config.wavelet_viz_run_tag = cfg_str(
        cfg, "wavelet_viz_run_tag", str(getattr(model_args, "wavelet_viz_run_tag", "default") or "default")
    )
    config.wavelet_viz_mode = cfg_str(
        cfg, "wavelet_viz_mode", str(getattr(config, "wavelet_mode", "unknown"))
    )
    config.wavelet_viz_max_batches = cfg_int(
        cfg, "wavelet_viz_max_batches", int(getattr(model_args, "wavelet_viz_max_batches", 8))
    )
    config.wavelet_viz_sample_q = cfg_int(
        cfg, "wavelet_viz_sample_q", int(wavelet_viz_sample_q_default)
    )
    config.wavelet_viz_sample_k = cfg_int(
        cfg, "wavelet_viz_sample_k", int(getattr(model_args, "wavelet_viz_sample_k", 256))
    )
    config.wavelet_viz_outdir = cfg_str(
        cfg, "wavelet_viz_outdir", str(wavelet_viz_outdir_default)
    )
    config.wavelet_viz_model_size = cfg_str(
        cfg, "wavelet_viz_model_size", str(wavelet_viz_model_size_default)
    )
    config.wavelet_viz_seed = int(getattr(training_args, "seed", -1))
    config.wavelet_viz_dataset_config = str(getattr(data_args, "dataset_config_name", "default"))
    config.wavelet_viz_bucket_size = int(getattr(data_args, "xsum_bucket_size", 0))
    config.wavelet_ctx_feat_mode = cfg_str(
        cfg, "wavelet_ctx_feat_mode", str(getattr(config, "wavelet_ctx_feat_mode", "q_meanH"))
    )
    config.wavelet_ctx_feat_rms_eps = cfg_float(
        cfg, "wavelet_ctx_feat_rms_eps", float(getattr(config, "wavelet_ctx_feat_rms_eps", 1e-6))
    )
    config.wavelet_ctx_feat_detach_delta = cfg_bool(
        cfg, "wavelet_ctx_feat_detach_delta", bool(getattr(config, "wavelet_ctx_feat_detach_delta", False))
    )
    config.router_jitter_flip_ratio = cfg_float(
        cfg,
        "router_jitter_flip_ratio",
        float(getattr(config, "router_jitter_flip_ratio", float(getattr(config, "router_jitter_std", 0.0)))),
    )
    # Backward-compat alias: old code/metrics may still read router_jitter_std.
    config.router_jitter_std = float(config.router_jitter_flip_ratio)
    config.wavelet_ctxscale_film_hidden = cfg_int(
        cfg, "wavelet_ctxscale_film_hidden", int(getattr(config, "wavelet_ctxscale_film_hidden", 64))
    )
    config.wavelet_ctxscale_film_alpha = cfg_float(
        cfg, "wavelet_ctxscale_film_alpha", float(getattr(config, "wavelet_ctxscale_film_alpha", 0.5))
    )
    config.wavelet_ctxscale_film_beta = cfg_float(
        cfg, "wavelet_ctxscale_film_beta", float(getattr(config, "wavelet_ctxscale_film_beta", 0.1))
    )
    config.wavelet_ctxscale_film_clamp = cfg_float(
        cfg, "wavelet_ctxscale_film_clamp", float(getattr(config, "wavelet_ctxscale_film_clamp", 8.0))
    )
    config.wavelet_gate_grad_clip = cfg_float(
        cfg, "wavelet_gate_grad_clip", float(getattr(config, "wavelet_gate_grad_clip", 1.0))
    )
    config.wavelet_ctxscale_lock_window = cfg_int(
        cfg, "wavelet_ctxscale_lock_window", int(getattr(config, "wavelet_ctxscale_lock_window", 300))
    )
    config.wavelet_ctxscale_lock_grad_eps = cfg_float(
        cfg, "wavelet_ctxscale_lock_grad_eps", float(getattr(config, "wavelet_ctxscale_lock_grad_eps", 1e-6))
    )
    config.wavelet_ctxscale_lock_update_eps = cfg_float(
        cfg, "wavelet_ctxscale_lock_update_eps", float(getattr(config, "wavelet_ctxscale_lock_update_eps", 1e-6))
    )
    config.wavelet_gate_autofix = cfg_bool(
        cfg, "wavelet_gate_autofix", bool(getattr(config, "wavelet_gate_autofix", False))
    )
    config.wavelet_gate_autofix_clamp_abs = cfg_float(
        cfg, "wavelet_gate_autofix_clamp_abs", float(getattr(config, "wavelet_gate_autofix_clamp_abs", 4.0))
    )
    config.wavelet_ctxscale_disable_layer_gate = cfg_bool(
        cfg,
        "wavelet_ctxscale_disable_layer_gate",
        bool(getattr(config, "wavelet_ctxscale_disable_layer_gate", False)),
    )
    # Causal rho intervention: override rho to a fixed constant for ablation (None = free rho).
    _rho_ov_raw = cfg.get("wavelet_ctxscale_rho_override", None)
    config.wavelet_ctxscale_rho_override = (
        float(_rho_ov_raw) if _rho_ov_raw is not None else getattr(config, "wavelet_ctxscale_rho_override", None)
    )
    # LW residual HW specialization: CLI + cfg_path configurable, defaults preserve legacy behavior.
    config.lw_residual_hw_enable = cfg_bool(
        cfg,
        "lw_residual_hw_enable",
        bool(getattr(model_args, "lw_residual_hw_enable", False)),
    )
    config.lw_residual_hw_alpha = cfg_float(
        cfg,
        "lw_residual_hw_alpha",
        float(getattr(model_args, "lw_residual_hw_alpha", 0.1)),
    )
    config.lw_residual_hw_l2 = cfg_float(
        cfg,
        "lw_residual_hw_l2",
        float(getattr(model_args, "lw_residual_hw_l2", 1e-4)),
    )
    config.lw_residual_hw_freeze_steps = cfg_int(
        cfg,
        "lw_residual_hw_freeze_steps",
        int(getattr(model_args, "lw_residual_hw_freeze_steps", 1500)),
    )
    logger.info(
        "[WaveletCfg] wavelet_ctxscale_disable_layer_gate=%s (cfg_has_key=%s, cfg_path=%s)",
        str(getattr(config, "wavelet_ctxscale_disable_layer_gate", False)),
        str("wavelet_ctxscale_disable_layer_gate" in cfg),
        str(model_args.cfg_path),
    )
    logger.info(
        "[WaveletCfg] wavelet_mode=%s (source=%s, cfg_has_key=%s, cli_flag=%s)",
        str(getattr(config, "wavelet_mode", "")),
        str(wavelet_mode_source),
        str("wavelet_mode" in cfg),
        str(wavelet_mode_cli_provided),
    )
    if "wavelet_ctxscale_disable_layer_gate" in cfg:
        logger.info(
            "[WaveletCfg] wavelet_ctxscale_disable_layer_gate is consumed by PaTH ctxscale gate routing."
        )
    num_layers = getattr(config, "num_hidden_layers", None)
    if num_layers is None:
        num_layers = getattr(config, "n_layer", None)
    rel_layers_src = cfg.get("rel_use_layer_list", getattr(config, "rel_use_layer_list", None))
    rel_layers_norm = normalize_rel_use_layer_list(rel_layers_src, n_layer=num_layers)
    config.rel_use_layer_list = "all" if rel_layers_norm is None else rel_layers_norm
    logger.info("[REL] rel_use_layer_list=%s", str(config.rel_use_layer_list))
    config.gradient_accumulation_steps = training_args.gradient_accumulation_steps
    config.data_collection_style = training_args.data_collection_style
    config.head_mask_list = training_args.head_mask_list
    config.eval_rel_stats_enabled = training_args.eval_rel_stats_enabled
    config.eval_rel_stats_layers = training_args.eval_rel_stats_layers
    config.eval_rel_stats_bin_size = training_args.eval_rel_stats_bin_size
    config.eval_rel_stats_log_every = training_args.eval_rel_stats_log_every
    config.eval_rel_stats_log_once = training_args.eval_rel_stats_log_once
    config.eval_rel_stats_per_head = training_args.eval_rel_stats_per_head
    config.eval_rel_stats_max_samples_per_bin = training_args.eval_rel_stats_max_samples_per_bin
    config.eval_rel_stats_eps = training_args.eval_rel_stats_eps
    config.eval_rel_stats_anchor_layer = training_args.eval_rel_stats_anchor_layer
    config.log_rel_stats = cfg_bool(cfg, "log_rel_stats", bool(training_args.log_rel_stats))
    config.log_rel_every = cfg_int(cfg, "log_rel_every", int(training_args.log_rel_every))
    config.log_rel_eval_every = cfg_int(cfg, "log_rel_eval_every", int(training_args.log_rel_eval_every))
    config.log_rel_sample_qpos = cfg_str(cfg, "log_rel_sample_qpos", str(training_args.log_rel_sample_qpos))
    config.log_rel_sample_heads = cfg_str(cfg, "log_rel_sample_heads", str(training_args.log_rel_sample_heads))
    config.log_rel_sample_key_offsets = cfg_str(
        cfg, "log_rel_sample_key_offsets", str(training_args.log_rel_sample_key_offsets)
    )
    config.log_rel_tail_tau = cfg_int(cfg, "log_rel_tail_tau", int(training_args.log_rel_tail_tau))
    config.rel_param_keywords = cfg_str(cfg, "rel_param_keywords", str(training_args.rel_param_keywords))
    # backward-compatible aliases
    config.log_rel_sample_tokens = str(training_args.log_rel_sample_qpos)
    config.log_rel_param_keywords = str(training_args.rel_param_keywords)
    config.rel_alpha = float(training_args.rel_alpha)
    config.attn_rel_alpha = float(training_args.rel_alpha)
    # Honor eval heatmap dump controls from cfg for PaTH attention export.
    config.eval_attn_heatmap_enabled = cfg_bool(
        cfg, "eval_attn_heatmap_enabled", bool(getattr(config, "eval_attn_heatmap_enabled", False))
    )
    config.eval_attn_heatmap_layers = cfg_str(
        cfg, "eval_attn_heatmap_layers", str(getattr(config, "eval_attn_heatmap_layers", "all"))
    )
    config.eval_attn_heatmap_case_limit = cfg_int(
        cfg, "eval_attn_heatmap_case_limit", int(getattr(config, "eval_attn_heatmap_case_limit", 1))
    )
    config.eval_attn_heatmap_case_index = cfg_int(
        cfg, "eval_attn_heatmap_case_index", int(getattr(config, "eval_attn_heatmap_case_index", 0))
    )
    config.eval_attn_heatmap_save_png = cfg_bool(
        cfg, "eval_attn_heatmap_save_png", bool(getattr(config, "eval_attn_heatmap_save_png", True))
    )
    config.eval_attn_heatmap_save_pt = cfg_bool(
        cfg, "eval_attn_heatmap_save_pt", bool(getattr(config, "eval_attn_heatmap_save_pt", False))
    )
    config.eval_attn_heatmap_save_pt_logits = cfg_bool(
        cfg,
        "eval_attn_heatmap_save_pt_logits",
        bool(getattr(config, "eval_attn_heatmap_save_pt_logits", False)),
    )
    config.eval_attn_heatmap_save_pt_outputs = cfg_bool(
        cfg,
        "eval_attn_heatmap_save_pt_outputs",
        bool(getattr(config, "eval_attn_heatmap_save_pt_outputs", False)),
    )
    config.eval_attn_heatmap_outdir = cfg_str(
        cfg, "eval_attn_heatmap_outdir", str(getattr(config, "eval_attn_heatmap_outdir", "analysis"))
    )
    config.eval_attn_heatmap_run_tag = cfg_str(
        cfg, "eval_attn_heatmap_run_tag", str(getattr(config, "eval_attn_heatmap_run_tag", "default"))
    )
    config.eval_attn_heatmap_separate_step = cfg_bool(
        cfg,
        "eval_attn_heatmap_separate_step",
        bool(getattr(config, "eval_attn_heatmap_separate_step", False)),
    )
    config.eval_attn_heatmap_stop_after_case = cfg_bool(
        cfg,
        "eval_attn_heatmap_stop_after_case",
        bool(getattr(config, "eval_attn_heatmap_stop_after_case", False)),
    )
    logger.info(
        "[HeatmapCfg] enabled=%s case_limit=%s save_pt=%s save_logits=%s outdir=%s run_tag=%s",
        str(bool(getattr(config, "eval_attn_heatmap_enabled", False))),
        str(int(getattr(config, "eval_attn_heatmap_case_limit", 1))),
        str(bool(getattr(config, "eval_attn_heatmap_save_pt", False))),
        str(bool(getattr(config, "eval_attn_heatmap_save_pt_logits", False))),
        str(getattr(config, "eval_attn_heatmap_outdir", "")),
        str(getattr(config, "eval_attn_heatmap_run_tag", "")),
    )
    if train_eval_disable_expensive_stats:
        # Force-disable eval-side rel stats collection in train-time eval.
        config.log_rel_eval_every = 0
        config.eval_rel_stats_enabled = False
        config.eval_rel_stats_log_every = 0
    config.router_norm = {
        "enable": bool(training_args.router_norm_enable),
        "mode": str(training_args.router_norm_mode),
        "type": str(training_args.router_norm_type),
        "affine": bool(training_args.router_norm_affine),
        "eps": float(training_args.router_norm_eps),
        "clamp_std_min": float(training_args.router_norm_clamp_std_min),
        "log_every": int(training_args.router_norm_log_every),
        "log_heads": str(training_args.router_norm_log_heads),
        "log_tokens": str(training_args.router_norm_log_tokens),
    }
    config.router_norm_enable = bool(training_args.router_norm_enable)
    config.router_norm_mode = str(training_args.router_norm_mode)
    config.router_norm_type = str(training_args.router_norm_type)
    # Allow supply_model.cfg to control Trainer-level eval visibility/perf knobs.
    if "disable_tqdm" in cfg:
        try:
            training_args.disable_tqdm = bool(cfg_bool(cfg, "disable_tqdm", False))
            logger.info("[TrainerCfg] disable_tqdm=%s (from cfg)", str(training_args.disable_tqdm))
        except Exception:
            pass
    if "dataloader_num_workers" in cfg:
        try:
            training_args.dataloader_num_workers = max(0, int(cfg_int(cfg, "dataloader_num_workers", int(training_args.dataloader_num_workers))))
            logger.info("[TrainerCfg] dataloader_num_workers=%d (from cfg)", int(training_args.dataloader_num_workers))
        except Exception:
            pass
    config.router_norm_affine = bool(training_args.router_norm_affine)
    config.router_norm_eps = float(training_args.router_norm_eps)
    config.router_norm_clamp_std_min = float(training_args.router_norm_clamp_std_min)
    config.router_norm_log_every = int(training_args.router_norm_log_every)
    config.router_norm_log_heads = str(training_args.router_norm_log_heads)
    config.router_norm_log_tokens = str(training_args.router_norm_log_tokens)
    config.eval_router_heatmap_enable = bool(training_args.eval_router_heatmap_enable)
    config.eval_router_heatmap_bin_size = int(training_args.eval_router_heatmap_bin_size)
    config.eval_router_heatmap_max_batches = int(training_args.eval_router_heatmap_max_batches)
    config.eval_router_heatmap_out_subdir = str(training_args.eval_router_heatmap_out_subdir)
    if training_args.resume_from_checkpoint is not None:
        config.resume_checkpoint = int(training_args.resume_from_checkpoint.split('-')[-1])
    if not training_args.do_train and training_args.do_eval:
        config.save_root =  model_args.model_name_or_path
    else:
        config.save_root = training_args.output_dir
    config.rel_zoom_in_coe = training_args.rel_zoom_in_coe
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if local_rank == 0 and training_args.do_train:
        init_wandb(config, 'gpt2 with path attn')        

    # We resize the embeddings only when necessary to avoid index errors. If you are creating a model from scratch
    # on a small vocab and want a smaller embedding size, remove this test.

    # Preprocessing the datasets.
    # First we tokenize all the texts.
    if data_args.dataset_name == 'mix':
        raw_datasets = hotpot_qa_raw_datasets
    if training_args.do_train:
        column_names = list(raw_datasets["train"].features)
    else:
        column_names = list(raw_datasets["validation"].features)
    text_column_name = "text" if "text" in column_names else column_names[0]

    # since this will be pickled to avoid _LazyModule error in Hasher force logger loading before tokenize_function
    tok_logger = transformers.utils.logging.get_logger("transformers.tokenization_utils_base")

    def tokenize_function(examples):
        with CaptureLogger(tok_logger) as cl:
            output = tokenizer(examples[text_column_name])
        # clm input could be much much longer than block_size
        if "Token indices sequence length is longer than the" in cl.out:
            tok_logger.warning(
                "^^^^^^^^^^^^^^^^ Please ignore the warning above - this long input will be chunked into smaller bits"
                " before being passed to the model."
            )
        return output
    if hasattr(config, "max_position_embeddings"):
        max_pos_embeddings = config.max_position_embeddings
    else:
        # Define a default value if the attribute is missing in the config.
        max_pos_embeddings = 1024

    if data_args.block_size is None:
        block_size = tokenizer.model_max_length
        if block_size > max_pos_embeddings:
            logger.warning(
                f"The tokenizer picked seems to have a very large `model_max_length` ({tokenizer.model_max_length}). "
                f"Using block_size={min(1024, max_pos_embeddings)} instead. You can change that default value by passing --block_size xxx."
            )
            if max_pos_embeddings > 0:
                block_size = min(1024, max_pos_embeddings)
            else:
                block_size = 1024
    else:
        block_size = data_args.block_size
    config.block_size = block_size
    config.wavelet_analysis_block_size = int(block_size)
    config.wavelet_viz_block_size = int(block_size)
    config.rope_theta = 10000
    # Expose tokenizer globally for dataset map helpers
    global GLOBAL_TOKENIZER
    GLOBAL_TOKENIZER = tokenizer

    def _log_model_param_stats(model_to_log):
        """Log per-parameter mean and std right after model construction."""
        for name, param in model_to_log.named_parameters():
            data = param.detach().float()
            if data.numel() == 0:
                logger.info(f"[ParamStats] {name}: empty tensor (numel=0), skip mean/std")
                continue
            mean = data.mean().item()
            std = data.std(unbiased=False).item()
            logger.info(f"[ParamStats] {name}: mean={mean:.6f}, std={std:.6f}")

    def _sanitize_wavelet_scalar_params(model_to_fix, cfg_obj, missing_keys=None):
        """
        Ensure newly-added scalar gate params are sane after from_pretrained load.
        This avoids pathological startup values when keys are missing in old checkpoints.
        """
        a_init = float(getattr(cfg_obj, "wavelet_logit_bias_a_init", -5.0))
        k1_init = float(getattr(cfg_obj, "wavelet_k1_gain_init", 0.0))
        reset_missing = bool(getattr(cfg_obj, "wavelet_reinit_missing_scalars", True))
        abs_max = float(getattr(cfg_obj, "wavelet_scalar_sanity_abs_max", 1e4))
        missing = set(missing_keys) if missing_keys is not None else set()

        n_reset_missing_a = 0
        n_reset_missing_mlp_a = 0
        n_reset_missing_k1 = 0
        n_sanitize_a = 0
        n_sanitize_mlp_a = 0
        n_sanitize_k1 = 0
        n_reqgrad_fix = 0

        with torch.no_grad():
            for name, p in model_to_fix.named_parameters():
                if name.endswith("wavelet_logit_bias_a"):
                    if reset_missing and (name in missing):
                        p.fill_(a_init)
                        n_reset_missing_a += 1
                    data = p.detach().float()
                    if (not torch.isfinite(data).all()) or (float(data.abs().max().item()) > abs_max):
                        p.fill_(a_init)
                        n_sanitize_a += 1
                    if not bool(p.requires_grad):
                        p.requires_grad_(True)
                        n_reqgrad_fix += 1
                elif name.endswith("mlp_bias_logit_bias_a") or name.endswith("mlp_bias_logit_bias_a_head"):
                    if reset_missing and (name in missing):
                        p.fill_(a_init)
                        n_reset_missing_mlp_a += 1
                    data = p.detach().float()
                    if (not torch.isfinite(data).all()) or (float(data.abs().max().item()) > abs_max):
                        p.fill_(a_init)
                        n_sanitize_mlp_a += 1
                    if not bool(p.requires_grad):
                        p.requires_grad_(True)
                        n_reqgrad_fix += 1
                elif name.endswith("wavelet_k1_gain"):
                    if reset_missing and (name in missing):
                        p.fill_(k1_init)
                        n_reset_missing_k1 += 1
                    data = p.detach().float()
                    if (not torch.isfinite(data).all()) or (float(data.abs().max().item()) > abs_max):
                        p.fill_(k1_init)
                        n_sanitize_k1 += 1
                    if not bool(p.requires_grad):
                        p.requires_grad_(True)
                        n_reqgrad_fix += 1
                elif name.endswith(".path_lam"):
                    # path_lam is always new (not in pretrained ckpt) — always zero-init
                    data = p.detach().float()
                    if (not torch.isfinite(data).all()) or (float(data.abs().max().item()) > 1.0):
                        logger.warning(
                            "[PathBlendInit] %s has abnormal value %.6g — force reset to 0.0", name, float(data.abs().max().item())
                        )
                        p.fill_(0.0)
                    elif name in missing:
                        p.fill_(0.0)
                    if not bool(p.requires_grad):
                        p.requires_grad_(True)
                        n_reqgrad_fix += 1

        logger.info(
            "[WaveletInitSanity] reset_missing_a=%d reset_missing_mlp_a=%d reset_missing_k1=%d "
            "sanitize_a=%d sanitize_mlp_a=%d sanitize_k1=%d reqgrad_fix=%d "
            "(a_init=%.6g k1_init=%.6g abs_max=%.6g missing_keys=%d)",
            int(n_reset_missing_a),
            int(n_reset_missing_mlp_a),
            int(n_reset_missing_k1),
            int(n_sanitize_a),
            int(n_sanitize_mlp_a),
            int(n_sanitize_k1),
            int(n_reqgrad_fix),
            float(a_init),
            float(k1_init),
            float(abs_max),
            int(len(missing)),
        )

    if model_args.model_name_or_path:
        dtype = model_args.dtype if model_args.dtype in ["auto", None] else getattr(torch, model_args.dtype)
        model_loaded = AutoModelForCausalLM.from_pretrained(
                model_args.model_name_or_path,
                # ckpt_path,
                from_tf=bool(".ckpt" in model_args.model_name_or_path),
                config=config,
                cache_dir=model_args.cache_dir,
                revision=model_args.model_revision,
                token=model_args.token,
                trust_remote_code=model_args.trust_remote_code,
                dtype=dtype,
                output_loading_info=True,
            )
        loading_info = {}
        if isinstance(model_loaded, tuple) and len(model_loaded) == 2:
            model, loading_info = model_loaded
        else:
            model = model_loaded
        if float(getattr(config, "coe_for_rel_init", -1)) != -1:
            with torch.no_grad():
                for _, module in model.named_modules():
                    if hasattr(module, "coe_for_rel") and isinstance(getattr(module, "coe_for_rel"), torch.nn.Parameter):
                        module.coe_for_rel.fill_(float(config.coe_for_rel_init))
        missing_keys = set(loading_info.get("missing_keys", [])) if isinstance(loading_info, dict) else set()
        _sanitize_wavelet_scalar_params(model, config, missing_keys=missing_keys)
        ###### wavelet coe 可视化分支 ######
        # tracker = WaveletCoeTracker(out_dir="wavelet_coe_logs_w_learnable_coe_0_8_scale_rel2_wavelet_path_attn", vmin=0.5, vmax=1.5)
        # dtype = model_args.dtype if model_args.dtype in ["auto", None] else getattr(torch, model_args.dtype)
        # ckpt_paths = get_ckpt_paths('runs/w_learnable_coe_0_8_scale_rel2_wavelet_path_attn', only_checkpoint_like=True)
        # for ckpt_path in tqdm(ckpt_paths):
        #     step = parse_step_from_ckpt_path(ckpt_path)
        #     assert step is not None, f"Cannot parse step from {ckpt_path}"        
        #     model = AutoModelForCausalLM.from_pretrained(
        #         # model_args.model_name_or_path,
        #         ckpt_path,
        #         from_tf=bool(".ckpt" in model_args.model_name_or_path),
        #         config=config,
        #         cache_dir=model_args.cache_dir,
        #         revision=model_args.model_revision,
        #         token=model_args.token,
        #         trust_remote_code=model_args.trust_remote_code,
        #         dtype=dtype,
        #     )
        #     tracker.capture(model, step=step, ckpt_tag=os.path.basename(ckpt_path))
        # pngs = tracker.plot_each_step_layer_head(prefix="wavelet_coe", dpi=200)
        # print("saved:", pngs[:3], "...", pngs[-1])
        # os._exit(0)
        ###### wavelet coe 可视化分支 ######
    else:
        model = AutoModelForCausalLM.from_config(config, trust_remote_code=model_args.trust_remote_code)
        n_params = sum({p.data_ptr(): p.numel() for p in model.parameters()}.values())
        logger.info(f"Training new model from scratch - Total size={n_params / 2**20:.2f}M params")    
        _sanitize_wavelet_scalar_params(model, config, missing_keys=None)

    _log_model_param_stats(model)
    embedding_size = model.get_input_embeddings().weight.shape[0]
    ######## freeze parameter #############
    if getattr(config, "freeze_backbone", False):
        custom_kw_csv = str(getattr(training_args, "freeze_train_keywords", "") or "").strip()
        if custom_kw_csv:
            keep_keywords = tuple(k.strip().lower() for k in custom_kw_csv.split(",") if k.strip())
            logger.info("[Freeze] use custom keep keywords: %s", ",".join(keep_keywords))
        else:
            keep_keywords = (
                "router1",
                "router2",
                "wavelet_coe",
                "rel1_coe",
                "ln_1",
                "ln_2",
                "ln_f",
            )

        for name, p in model.named_parameters():
            lname = name.lower()
            p.requires_grad = any(k in lname for k in keep_keywords)
        # sanity check
        trainable = [n for n, p in model.named_parameters() if p.requires_grad]
        print(f"[Freeze] trainable tensors = {len(trainable)}")
        for n in trainable[:50]:
            print("  ", n)
        model.enable_input_require_grads()
        if training_args.gradient_checkpointing:
            model.config.use_cache = False
    ######## freeze parameter #############
    if len(tokenizer) > embedding_size:
        model.resize_token_embeddings(len(tokenizer))
    bucket_size = getattr(data_args, "xsum_bucket_size", 0)
    hotpot_level_cache = {}
    hotpot_placement_cache = {}   # maps split → list[float] of placement_actual_pct per example
    mix_eval_task_cache = {}

    probe_split = "train" if (training_args.do_train and "train" in raw_datasets) else list(raw_datasets.keys())[0]
    probe_cols = raw_datasets[probe_split].column_names
    is_synth = ("input_ids" in probe_cols) and ("labels" in probe_cols)

    if data_args.dataset_name == "hotpot_qa" or (data_args.dataset_name == 'mix'):
        # =========================
        # hotpot_qa MCQ eval-only 分支
        # =========================
        tokenizer.pad_token = tokenizer.eos_token
        block_size = data_args.block_size
        q_use = bool(getattr(config, "question_use_for_train", False))
        q_use = q_use and training_args.do_train
        logger.info(f"[hotpot_qa] q_use={q_use}")
        if training_args.do_train and bucket_size == 0:
            # 训练默认使用 block_size 作为 bucket，避免长度过滤掉过多样本；若用户已指定则尊重用户值
            bucket_size = block_size
        def build_hotpot_qa(ex, idx):
            # Keep schema aligned even for discarded samples
            def make_discard(total_len: int = 0):
                return {
                    "discard": True,
                    "input_ids": [tokenizer.pad_token_id] * block_size,
                    "labels": [-100] * block_size,
                    "attention_mask": [0] * block_size,
                    "level": level,
                    "placement_pct": placement_pct,
                }

            # tokenizer is accessed via GLOBAL_TOKENIZER set below
            q = ex.get("question", "")
            gold = ex.get("answer", "")
            level = ex.get("level", "unknown")
            placement_pct = float(ex.get("placement_pct", -1.0))
            if not isinstance(level, str):
                level = str(level)
            if not isinstance(q, str) or not isinstance(gold, str) or len(gold.strip()) == 0:
                return make_discard(0)

            tok = _get_tokenizer()
            
            if config.hotpot_question_position == "before":
                q_part = f"Question: {q}\n"
                ctx_part = "Context:\n"
                suffix_text = "\nAnswer:"

                q_ids = tok(q_part, add_special_tokens=True, truncation=False)["input_ids"]
                ctx_prompt_ids = tok(ctx_part, add_special_tokens=False, truncation=False)["input_ids"]
                suffix_ids = tok(suffix_text, add_special_tokens=False, truncation=False)["input_ids"]

                prefix_ids = q_ids + ctx_prompt_ids
                labels_prefix = (q_ids if q_use else [-100] * len(q_ids)) + [-100] * len(ctx_prompt_ids)
            else:
                ctx_part = "Context:\n"
                q_part = f"\nQuestion: {q}\n"
                ans_prompt = "Answer:"

                prefix_ids = tok(ctx_part, add_special_tokens=True, truncation=False)["input_ids"]
                q_ids = tok(q_part, add_special_tokens=False, truncation=False)["input_ids"]
                ans_prompt_ids = tok(ans_prompt, add_special_tokens=False, truncation=False)["input_ids"]
                suffix_ids = q_ids + ans_prompt_ids

                labels_prefix = [-100] * len(prefix_ids)
                labels_suffix = (q_ids if q_use else [-100] * len(q_ids)) + [-100] * len(ans_prompt_ids)

            ans_ids = tok(" " + gold.strip(), add_special_tokens=False)["input_ids"]

            # 预算上下文长度
            if config.hotpot_question_position == "before":
                prompt_ids = prefix_ids
                prompt_labels = labels_prefix
            else:
                prompt_ids = prefix_ids
                prompt_labels = labels_prefix

            budget_context = block_size - len(prefix_ids) - len(suffix_ids) - len(ans_ids) - 1  # -1 给 eos
            lower = max(0, block_size - bucket_size) if bucket_size > 0 else 0
            min_ctx_tokens = max(0, (lower + 1) - (len(prefix_ids) + len(suffix_ids) + len(ans_ids) + 1))

            ctx_text, ctx_ids, status = build_context_budgeted(
                ex,
                budget_context,
                prefer_same_title=True,
                min_tokens=min_ctx_tokens,
            )
            if ctx_text is None:
                return make_discard(0)

            prompt_ids = prefix_ids + ctx_ids + suffix_ids

            # 目标答案 token（你可以决定要不要在 gold 前面加空格，保持一致就行）
            # ans_ids 已在上面计算

            # 拼接 input_ids / labels
            ids = prompt_ids + ans_ids + [tokenizer.eos_token_id]

            # labels：prompt 部分根据 q_use 决定，context/suffix 仍忽略；答案部分监督（eos 忽略）
            if config.hotpot_question_position == "before":
                labels = prompt_labels + [-100]*len(ctx_ids) + [-100]*len(suffix_ids) + ans_ids + [-100]
            else:
                labels = prompt_labels + [-100]*len(ctx_ids) + labels_suffix + ans_ids + [-100]

            L = len(ids)

            # 长度过滤：
            # 1) 超过 block_size 一律丢弃；
            # 2) 若设置了 bucket_size>0，则仅保留 (block_size-bucket_size, block_size] 区间，其余丢弃，避免不同 block_size 运行重叠。
            if L > block_size:
                return make_discard(L)

            if bucket_size > 0:
                lower = max(0, block_size - bucket_size)
                if L <= lower:
                    return make_discard(L)

            pad_len = block_size - L
            ids = ids + [tokenizer.pad_token_id]*pad_len
            labels = labels + [-100]*pad_len
            attn = [1]*L + [0]*pad_len

            return {
                "discard": False,
                "input_ids": ids,
                "labels": labels,
                "attention_mask": attn,
                "level": level,
                "placement_pct": placement_pct,
            }
        # hotpot_qa_stats = stats_from_builder(
        #     raw_datasets["train"],
        #     builder_fn=build_hotpot_qa,
        #     max_samples=5000,
        #     desc="hotpot_qa train stats",
        # )
        # print(hotpot_qa_stats)
        # os._exit(0)
        def _hotpot_cache_fingerprint():
            key = {
                "dataset": "hotpot_qa_mcq_lm_v1",
                "block_size": block_size,
                "q_use": q_use,
                "bucket_size": bucket_size,
                "tok": getattr(tokenizer, "name_or_path", None),
                "question_position": getattr(config, "hotpot_question_position", None),
                "hotpot_long_jsonl": getattr(data_args, "hotpot_long_jsonl", None),
                "hotpot_long_lengths": getattr(data_args, "hotpot_long_lengths", None),
            }
            return hashlib.md5(json.dumps(key, sort_keys=True).encode("utf-8")).hexdigest()
        if training_args.do_train and not training_args.do_eval:
            hotpot_cache_root = Path(getattr(data_args, "cache_dir", None) or os.path.expanduser("~/.cache/train_hotpot_qa_lm"))
        elif not training_args.do_train and training_args.do_eval:
            hotpot_cache_root = Path(getattr(data_args, "cache_dir", None) or os.path.expanduser("~/.cache/eval_hotpot_qa_lm"))
        else:
            hotpot_cache_root = Path(getattr(data_args, "cache_dir", None) or os.path.expanduser("~/.cache/hotpot_qa_lm"))
        hotpot_cache_root.mkdir(parents=True, exist_ok=True)
        hotpot_cache_dir = hotpot_cache_root / f"hotpot_qa_lm_{_hotpot_cache_fingerprint()}"
        logger.info(f"[hotpot_qa] cache_dir={hotpot_cache_dir}")

        need_splits = []
        if training_args.do_train and "train" in raw_datasets:
            need_splits.append("train")
        if training_args.do_eval:
            if "validation" in raw_datasets:
                need_splits.append("validation")
            elif "val" in raw_datasets:
                need_splits.append("val")
        if training_args.do_predict and "test" in raw_datasets:
            need_splits.append("test")

        if not need_splits:
            raise ValueError("[hotpot_qa] no split to process (need train/validation/test)")

        use_cache = hotpot_cache_dir.exists() and (not data_args.overwrite_cache)
        target_ds = DatasetDict()

        if use_cache:
            cached = load_from_disk(str(hotpot_cache_dir))
            logger.info(f"[hotpot_qa] cached splits={list(cached.keys())}")
            missing = [sp for sp in need_splits if sp not in cached]
            if missing:
                logger.warning(f"[hotpot_qa] cache missing splits {missing}, will rebuild.")
                use_cache = False
            else:
                target_ds = cached
                for sp in target_ds.keys():
                    target_ds[sp].set_format(type="python", columns=["input_ids", "attention_mask", "labels"])
                    hotpot_level_cache[sp] = None

        if not use_cache:
            processed = {}
            with training_args.main_process_first(desc="hotpot_qa build"):
                for split in need_splits:
                    processed[split] = raw_datasets[split].map(
                        build_hotpot_qa,
                        with_indices=True,
                        remove_columns=column_names,
                        num_proc=None,
                        desc=f"hotpot_qa map (split={split})",
                        load_from_cache_file=not data_args.overwrite_cache,
                    )

            for split in need_splits:
                filtered = processed[split].filter(lambda ex: not ex.get("discard", False))
                out_split = "validation" if split == "val" else split
                if "level" in filtered.column_names:
                    hotpot_level_cache[out_split] = list(filtered["level"])
                    filtered = filtered.remove_columns(["level"])
                else:
                    hotpot_level_cache[out_split] = None
                if hotpot_level_cache[out_split] is not None:
                    lvl_stats = Counter([str(x).lower() for x in hotpot_level_cache[out_split]])
                    logger.info(f"[hotpot_qa] split={out_split} level distribution: {dict(lvl_stats)} (n={len(hotpot_level_cache[out_split])})")
                if "placement_pct" in filtered.column_names:
                    hotpot_placement_cache[out_split] = [float(v) for v in filtered["placement_pct"]]
                    filtered = filtered.remove_columns(["placement_pct"])
                else:
                    hotpot_placement_cache[out_split] = None
                filtered.set_format(type="python", columns=["input_ids", "attention_mask", "labels"])
                target_ds[out_split] = filtered

            if getattr(training_args, "process_index", 0) == 0:
                hotpot_cache_dir.mkdir(parents=True, exist_ok=True)
                target_ds.save_to_disk(str(hotpot_cache_dir))

        if data_args.dataset_name == "mix":
            hotpot_qa_lm_datasets = target_ds
        else:
            lm_datasets = target_ds
    elif data_args.dataset_name == "passkey":
        # =====================================================================
        # Passkey retrieval: generate synthetic fixed-length examples in which
        # a random N-digit number is hidden at a random position in filler text.
        # The model is asked to recall the key at the END of the sequence.
        # Metric: Exact Match on the answer key tokens at the final position.
        # block_size determines the context length tested in this run.
        #
        # Sequence structure:
        #   [intro][filler_before][key_prefix][key][filler_after][question][answer_key]
        # Labels: -100 everywhere except the answer_key tokens at the very end.
        # =====================================================================
        from datasets import Dataset as _PkDataset, DatasetDict as _PkDatasetDict
        _pk_block_size = data_args.block_size
        _pk_num_samples = int(getattr(data_args, "passkey_num_samples", 50))
        _pk_num_digits  = int(getattr(data_args, "passkey_num_digits",  5))
        _pk_rng = random.Random(42)
        _pk_filler = "The grass is green. The sky is blue. The sun is yellow. Here we go. There and back again. "
        _pk_intro       = "There is an important info hidden inside a lot of irrelevant text. Find it and memorize them. I will quiz you about the important information there.\n\n"
        _pk_key_prefix  = "The special magic number is: "
        _pk_question    = "\n\nWhat is the special magic number?\n\nThe special magic number is:"
        passkey_length_cache = []
        tokenizer.pad_token = tokenizer.eos_token
        _pk_records = {"input_ids": [], "labels": [], "attention_mask": []}
        # Pre-tokenize fixed non-key parts (shared across samples)
        _intro_ids    = tokenizer(_pk_intro,       add_special_tokens=True,  truncation=False)["input_ids"]
        _prefix_ids   = tokenizer(_pk_key_prefix,  add_special_tokens=False, truncation=False)["input_ids"]
        _question_ids = tokenizer(_pk_question,    add_special_tokens=False, truncation=False)["input_ids"]
        _filler_one   = tokenizer(_pk_filler,      add_special_tokens=False, truncation=False)["input_ids"]
        for _pk_i in range(_pk_num_samples):
            _pk_key     = "".join([str(_pk_rng.randint(0, 9)) for _ in range(_pk_num_digits)])
            _pk_key_str = " " + _pk_key  # leading space for GPT-2 tokenisation
            _key_ids    = tokenizer(_pk_key_str, add_special_tokens=False, truncation=False)["input_ids"]
            # Total fixed token budget:
            #   intro + key_prefix(in doc) + key(in doc) + question + answer_key
            # answer_key is identical to _key_ids and sits at the very end.
            _fixed_tokens = (len(_intro_ids) + len(_prefix_ids) + len(_key_ids)
                             + len(_question_ids) + len(_key_ids))
            _filler_budget = max(0, _pk_block_size - _fixed_tokens)
            # Build full filler token sequence
            _filler_full = []
            while len(_filler_full) < _filler_budget:
                _filler_full += _filler_one
            _filler_full = _filler_full[:_filler_budget]
            # Randomly split filler into before/after the in-document key
            _split = _pk_rng.randint(0, max(0, len(_filler_full)))
            _filler_before = _filler_full[:_split]
            _filler_after  = _filler_full[_split:]
            # Assemble: intro | filler_before | key_prefix | key | filler_after | question | answer_key
            _input_ids = (_intro_ids + _filler_before + _prefix_ids + _key_ids
                          + _filler_after + _question_ids + _key_ids)
            # Truncate / pad to block_size
            if len(_input_ids) > _pk_block_size:
                _input_ids = _input_ids[:_pk_block_size]
            _attn_mask = [1] * len(_input_ids)
            while len(_input_ids) < _pk_block_size:
                _input_ids.append(tokenizer.pad_token_id)
                _attn_mask.append(0)
            # Labels: -100 everywhere; non-(-100) only on the answer_key at the end.
            _labels = [-100] * _pk_block_size
            _ans_start = (len(_intro_ids) + len(_filler_before) + len(_prefix_ids)
                          + len(_key_ids) + len(_filler_after) + len(_question_ids))
            for _ki, _tok in enumerate(_key_ids):
                _pos = _ans_start + _ki
                if _pos < _pk_block_size:
                    _labels[_pos] = _tok
            _pk_records["input_ids"].append(_input_ids)
            _pk_records["labels"].append(_labels)
            _pk_records["attention_mask"].append(_attn_mask)
            passkey_length_cache.append(_pk_block_size)
        _pk_ds = _PkDataset.from_dict(_pk_records)
        lm_datasets = _PkDatasetDict({"validation": _pk_ds})
        logger.info(
            "[passkey] generated %d examples at block_size=%d with %d-digit passkeys",
            _pk_num_samples, _pk_block_size, _pk_num_digits,
        )
    elif data_args.dataset_name in ('wikitext', 'openwebtext', 'Skylion007/openwebtext'):
        with training_args.main_process_first(desc="dataset map tokenization"):
            if not data_args.streaming:
                tokenized_datasets = raw_datasets.map(
                    tokenize_function,
                    batched=True,
                    num_proc=data_args.preprocessing_num_workers,
                    remove_columns=column_names,
                    load_from_cache_file=not data_args.overwrite_cache,
                    desc="Running tokenizer on dataset",
                )
            else:
                tokenized_datasets = raw_datasets.map(
                    tokenize_function,
                    batched=True,
                    remove_columns=column_names,
                )
 
        # Main data processing function that will concatenate all texts from our dataset and generate chunks of block_size.
        def group_texts(examples):
            # Concatenate all texts.
            concatenated_examples = {k: list(chain(*examples[k])) for k in examples}
            total_length = len(concatenated_examples[list(examples.keys())[0]])
            # We drop the small remainder, and if the total_length < block_size  we exclude this batch and return an empty dict.
            # We could add padding if the model supported it instead of this drop, you can customize this part to your needs.
            total_length = (total_length // block_size) * block_size
            # Split by chunks of max_len.
            result = {
                k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
                for k, t in concatenated_examples.items()
            }
            result["labels"] = result["input_ids"].copy()
            return result

        # Note that with `batched=True`, this map processes 1,000 texts together, so group_texts throws away a remainder
        # for each of those groups of 1,000 texts. You can adjust that batch_size here but a higher value might be slower
        # to preprocess.
        #
        # To speed up this part, we use multiprocessing. See the documentation of the map method for more information:
        # https://huggingface.co/docs/datasets/process#map

        with training_args.main_process_first(desc="grouping texts together"):
            if not data_args.streaming:
                lm_datasets = tokenized_datasets.map(
                    group_texts,
                    batched=True,
                    num_proc=data_args.preprocessing_num_workers,
                    load_from_cache_file=not data_args.overwrite_cache,
                    desc=f"Grouping texts in chunks of {block_size}",
                )
            else:
                lm_datasets = tokenized_datasets.map(
                    group_texts,
                    batched=True,
                )
    elif is_synth:
        # ============================================================
        # Synthetic pre-tokenized dataset:
        # expects fields: input_ids, labels (with -100 on non-answer tokens),
        # optional: attention_mask
        # ============================================================
        lm_datasets = raw_datasets

        # attention_mask not needed (causal mask generated by model); drop if present
        probe_split = "train" if (training_args.do_train and "train" in lm_datasets) else list(lm_datasets.keys())[0]
        if "attention_mask" in lm_datasets[probe_split].column_names:
            with training_args.main_process_first(desc="drop attention_mask (synthetic_task)"):
                lm_datasets = lm_datasets.remove_columns("attention_mask")

        # optional sanity checks
        required = {"input_ids", "labels"}
        missing = required - set(lm_datasets[probe_split].column_names)
        if missing:
            raise ValueError(f"synthetic_task dataset missing columns: {missing}")

        # (optional) check length equals block_size
        if (not data_args.streaming) and training_args.do_train and "train" in lm_datasets and len(lm_datasets["train"]) > 0:
            ex0 = lm_datasets["train"][0]
            if len(ex0["input_ids"]) != block_size:
                logger.warning(
                    f"[synthetic_task] input length {len(ex0['input_ids'])} != block_size {block_size}. "
                    "If you intended fixed length training, set seq_len accordingly."
                )

        # Ensure trainer keeps these columns
        # Move cached tensors directly to the desired device to avoid CPU->GPU copy in the training loop
        for sp in ("train", "validation", "test"):
            if sp in lm_datasets:
                keep_cols = [c for c in ["input_ids", "labels"] if c in lm_datasets[sp].column_names]
                lm_datasets[sp].set_format(type="torch", columns=keep_cols)
    if data_args.dataset_name in ('xsum', 'mix', 'govreport', 'ccdv/govreport-summarization'):
        if data_args.dataset_name == 'mix':
            raw_datasets = xsum_raw_datasets
            column_names = raw_datasets["train"].column_names 
        # =========================
        # XSUM prefix-LM 分支（简化版）
        # =========================
        PROMPT_TPL = "Summarize the following document:\n{doc}\n\nSummary:"
        tokenizer.pad_token = tokenizer.eos_token
        # ---- 根据 full_fine_tune 决定 cache 模式 ---- #
        def _lm_cache_fingerprint():
            mode = "fulltrain" if getattr(data_args, "full_fine_tune", False) else "masktrain"
            key = {
                "dataset": f"xsum_prefixlm_v4_bucket_select_{data_args.dataset_name}",  # 改个名字，避免和旧 cache 混淆
                "mode": mode,
                "tok_name": getattr(tokenizer, "name_or_path", None),
                "block_size": data_args.block_size,
                "xsum_bucket_size": getattr(data_args, "xsum_bucket_size", 0),
                "xsum_bucket_apply_to": getattr(data_args, "xsum_bucket_apply_to", "eval_test"),
                "xsum_min_total_len": getattr(data_args, "xsum_min_total_len", 0),
                "eos_id": tokenizer.eos_token_id,
                "train_file": str(getattr(data_args, "train_file", None) or ""),
            }
            return hashlib.md5(json.dumps(key, sort_keys=True).encode("utf-8")).hexdigest()

        mode = "fulltrain" if getattr(data_args, "full_fine_tune", False) else "masktrain"

        LM_CACHE_ROOT = Path(
            getattr(data_args, "cache_dir", None) or os.path.expanduser("~/.cache/xsum_prefixlm")
        ) / "lm_datasets_cache_simple" / mode
        LM_CACHE_ROOT.mkdir(parents=True, exist_ok=True)
        LM_CACHE_DIR = LM_CACHE_ROOT / f"xsum_prefixlm_{_lm_cache_fingerprint()}"

        block_size = data_args.block_size


        def _xsum_keep_for_split(ex, split_name: str) -> bool:

            """XSUM filter to (optionally) enforce disjoint length buckets without truncation."""

            if ex.get("discard", False):

                return False


            L = ex.get("total_token_len", 0)


            minL = getattr(data_args, "xsum_min_total_len", 0) or 0

            if minL > 0 and L < minL:

                return False
            
            apply_to = getattr(data_args, "xsum_bucket_apply_to", "eval_test") or "eval_test"


            if bucket_size > 0 and apply_to != "none":

                apply_bucket = (apply_to == "all") or (apply_to == "eval_test" and split_name in ("validation", "test"))

                if apply_bucket:

                    # Use xsum_bucket_size as the bucket length; block_size only decides which bucket index to keep
                    bucket_idx = ((block_size - 1) // bucket_size) + 1  # 1-based bucket index containing block_size

                    upper = bucket_idx * bucket_size

                    lower = upper - bucket_size

                    return (L > lower) and (L <= upper)


            return True
        # Arrow cache fingerprint is based on input data content, so different validation_files
        # produce different cache paths — no stale-cache risk. Always allow cache to prevent
        # SIGBUS from multiple DDP ranks writing the same arrow file simultaneously.
        disable_xsum_cache_for_validation_file = False
        use_cache = LM_CACHE_DIR.exists() and (not disable_xsum_cache_for_validation_file)
        lm_datasets = None
        if use_cache:
            lm_datasets = load_from_disk(str(LM_CACHE_DIR))

            # —— 过滤掉 discard，并（可选）按长度桶筛选，避免不同 block_size 的 eval/test 样本集合重叠 —— #

            need_len = (getattr(data_args, "xsum_bucket_size", 0) or 0) > 0 or (getattr(data_args, "xsum_min_total_len", 0) or 0) > 0

            if need_len:

                for split in list(lm_datasets.keys()):

                    if "total_token_len" not in lm_datasets[split].column_names:

                        logger.warning(

                            f"[XSUM] Cached dataset at {LM_CACHE_DIR} has no 'total_token_len' column (split={split}); will rebuild."

                        )

                        use_cache = False

                        break


            if use_cache:

                for split in list(lm_datasets.keys()):

                    lm_datasets[split] = lm_datasets[split].filter(

                        lambda ex, split=split: _xsum_keep_for_split(ex, split)

                    )


            # ⚠️ 检查 cache 里有没有 train / validation，没有就当作坏 cache，重新构建
            if training_args.do_train and "train" not in lm_datasets:
                logger.warning(
                    f"[XSUM] Cached dataset at {LM_CACHE_DIR} has no 'train' split, will rebuild."
                )
                use_cache = False
            if training_args.do_eval and "validation" not in lm_datasets:
                logger.warning(
                    f"[XSUM] Cached dataset at {LM_CACHE_DIR} has no 'validation' split, will rebuild."
                )
                use_cache = False

            if training_args.do_predict and "test" not in lm_datasets:
                logger.warning(
                    f"[XSUM] Cached dataset at {LM_CACHE_DIR} has no 'test' split, will rebuild."
                )
                use_cache = False

        def build_and_index_factory(split_name):
            def build_and_index(ex, idx):
                def make_discard(total_len: int):
                    # 给丢弃样本也返回同样的列（占位），避免 datasets.map schema 不一致
                    return {
                        "discard": True,
                        "input_ids": [tokenizer.pad_token_id] * block_size,
                        "labels": [-100] * block_size,
                        "attention_mask": [0] * block_size,
                        "total_token_len": total_len,
                    }

                enc_prompt = tokenizer(
                    PROMPT_TPL.format(doc=ex["document"]),
                    add_special_tokens=True,
                    truncation=False,
                )
                enc_summ = tokenizer(
                    ex["summary"],
                    add_special_tokens=False,
                    truncation=False,
                )

                prompt_ids = enc_prompt["input_ids"]
                summ_ids   = enc_summ["input_ids"]

                if len(summ_ids) == 0:
                    return make_discard(0)

                ids = prompt_ids + summ_ids + [tokenizer.eos_token_id]

                if split_name == "train" and getattr(data_args, "full_fine_tune", False):
                    labels = ids.copy()
                else:
                    labels = ([-100] * len(prompt_ids)) + summ_ids + [tokenizer.eos_token_id]

                L = len(ids)
                if L <= 1:
                    return make_discard(L)

                if L > block_size:
                    return make_discard(L)

                pad_len = block_size - L
                ids    = ids    + [tokenizer.pad_token_id] * pad_len
                labels = labels + [-100] * pad_len
                attn   = [1] * L + [0] * pad_len

                return {
                    "discard": False,            # ✅ 关键：正常样本也要有 discard 字段
                    "input_ids": ids,
                    "labels": labels,
                    "attention_mask": attn,
                    "total_token_len": L,
                }
            return build_and_index
            ####### length stat info ##############
        # xsum_stats = stats_from_builder_until_kept(
        #     raw_datasets["train"],
        #     builder_fn=build_and_index_factory("train"),
        #     max_kept=5000,
        #     max_seen_cap=20000, 
        #     desc="xsum train stats",
        # )
        # print(xsum_stats)
        # os._exit(0)
        if not use_cache:
            # ===== 无缓存：从 raw_datasets 构建 =====
            if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
                tokenizer.pad_token_id = tokenizer.eos_token_id
            if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
                tokenizer.pad_token_id = tokenizer.eos_token_id

            assert not data_args.streaming, "XSUM prefix-LM 分支暂不支持 streaming=True"
            def build_and_index_factory(split_name):
                def build_and_index(ex, idx):
                    def make_discard(total_len: int):
                        return {
                            "discard": True,
                            "input_ids": [tokenizer.pad_token_id] * block_size,
                            "labels": [-100] * block_size,
                            "attention_mask": [0] * block_size,
                            "total_token_len": total_len,
                        }

                    # document / summary 字段兼容
                    doc = ex.get("document", ex.get("article", ex.get("source", ex.get("report", ""))))
                    summ = ex.get("summary", ex.get("summary_filtered", ex.get("summary_original", ex.get("target", ex.get("abstract", ex.get("highlights", ""))))))

                    if not isinstance(doc, str):
                        doc = str(doc) if doc is not None else ""
                    if not isinstance(summ, str):
                        summ = str(summ) if summ is not None else ""

                    doc = doc.strip()
                    summ = summ.strip()
                    if len(doc) == 0 or len(summ) == 0:
                        return make_discard(0)

                    enc_prompt = tokenizer(
                        PROMPT_TPL.format(doc=doc),
                        add_special_tokens=True,
                        truncation=False,
                    )
                    enc_summ = tokenizer(
                        summ,
                        add_special_tokens=False,
                        truncation=False,
                    )

                    prompt_ids = enc_prompt["input_ids"]
                    summ_ids = enc_summ["input_ids"]

                    if len(summ_ids) == 0:
                        return make_discard(0)

                    ids = prompt_ids + summ_ids + [tokenizer.eos_token_id]

                    if split_name == "train" and getattr(data_args, "full_fine_tune", False):
                        labels = ids.copy()
                    else:
                        labels = ([-100] * len(prompt_ids)) + summ_ids + [tokenizer.eos_token_id]

                    L = len(ids)
                    if L <= 1:
                        return make_discard(L)
                    if L > block_size:
                        return make_discard(L)

                    pad_len = block_size - L
                    ids = ids + [tokenizer.pad_token_id] * pad_len
                    labels = labels + [-100] * pad_len
                    attn = [1] * L + [0] * pad_len

                    return {
                        "discard": False,
                        "input_ids": ids,
                        "labels": labels,
                        "attention_mask": attn,
                        "total_token_len": L,
                    }
                return build_and_index    
            # def build_and_index_factory(split_name):
            #     def build_and_index(ex, idx):
            #         def make_discard(total_len: int):
            #             # 给丢弃样本也返回同样的列（占位），避免 datasets.map schema 不一致
            #             return {
            #                 "discard": True,
            #                 "input_ids": [tokenizer.pad_token_id] * block_size,
            #                 "labels": [-100] * block_size,
            #                 "attention_mask": [0] * block_size,
            #                 "total_token_len": total_len,
            #             }

            #         enc_prompt = tokenizer(
            #             PROMPT_TPL.format(doc=ex["document"]),
            #             add_special_tokens=True,
            #             truncation=False,
            #         )
            #         enc_summ = tokenizer(
            #             ex["summary"],
            #             add_special_tokens=False,
            #             truncation=False,
            #         )

            #         prompt_ids = enc_prompt["input_ids"]
            #         summ_ids   = enc_summ["input_ids"]

            #         if len(summ_ids) == 0:
            #             return make_discard(0)

            #         ids = prompt_ids + summ_ids + [tokenizer.eos_token_id]

            #         if split_name == "train" and getattr(data_args, "full_fine_tune", False):
            #             labels = ids.copy()
            #         else:
            #             labels = ([-100] * len(prompt_ids)) + summ_ids + [tokenizer.eos_token_id]

            #         L = len(ids)
            #         if L <= 1:
            #             return make_discard(L)

            #         if L > block_size:
            #             return make_discard(L)

            #         pad_len = block_size - L
            #         ids    = ids    + [tokenizer.pad_token_id] * pad_len
            #         labels = labels + [-100] * pad_len
            #         attn   = [1] * L + [0] * pad_len

            #         return {
            #             "discard": False,            # ✅ 关键：正常样本也要有 discard 字段
            #             "input_ids": ids,
            #             "labels": labels,
            #             "attention_mask": attn,
            #             "total_token_len": L,
            #         }
            #     return build_and_index

            # —— 一次 map：构造 & 固定长度 —— #

            processed_splits = {}
            with training_args.main_process_first(desc="XSUM build & pad to block_size"):
                for split in raw_datasets.keys():
                    if split not in ("train", "validation", "test"):
                        continue
                    processed_splits[split] = raw_datasets[split].map(
                        build_and_index_factory(split),
                        with_indices=True,
                        remove_columns=column_names,
                        num_proc=1,
                        desc=f"Tokenize & pad (split={split})",
                        load_from_cache_file=(not data_args.overwrite_cache) and (not disable_xsum_cache_for_validation_file),
                    )

            lm_datasets = DatasetDict()
            if training_args.do_train:
                lm_datasets["train"] = processed_splits["train"].filter(
                    lambda ex, split="train": _xsum_keep_for_split(ex, split)
                )
            if training_args.do_eval:
                lm_datasets["validation"] = processed_splits["validation"].filter(
                    lambda ex, split="validation": _xsum_keep_for_split(ex, split)
                )

            if training_args.do_predict and "test" in processed_splits:
                lm_datasets["test"] = processed_splits["test"].filter(
                    lambda ex, split="test": _xsum_keep_for_split(ex, split)
                )

            # —— 缓存到磁盘（仅主进程写，其他进程等待） —— #
            with training_args.main_process_first(desc="save lm_datasets to disk (XSUM)"):
                if getattr(training_args, "process_index", 0) == 0:
                    lm_datasets.save_to_disk(str(LM_CACHE_DIR))

        # ====== 最后：统一设置格式 & 自检 ====== #
        need_cols = ["input_ids", "attention_mask", "labels"]
        for split in ("train", "validation"):
            if split in lm_datasets:
                cols = [c for c in need_cols if c in lm_datasets[split].column_names]
                lm_datasets[split].set_format(type="python", columns=cols)
                assert len(lm_datasets[split]) > 0, f"[XSUM] lm_datasets['{split}'] is empty."

        # 仅训练场景才混合 train；避免 do_eval-only 时缺少 train split 触发断言
        def _strip_to_lm_cols(ds):
            keep = {"input_ids", "attention_mask", "labels"}
            drop = [c for c in ds.column_names if c not in keep]
            return ds.remove_columns(drop) if drop else ds

        if data_args.dataset_name == 'mix':
            # 保留 xsum 处理后的数据集，后续与 hotpot_qa 组装
            xsum_lm_datasets = lm_datasets
            lm_datasets = DatasetDict()

            if training_args.do_train:
                # 仅 mix 模式才需要把 hotpot 与 xsum 的 train split 混合；纯 xsum 时跳过
                assert "train" in hotpot_qa_lm_datasets, "[mix] hotpot_qa_lm_datasets has no train split"
                assert "train" in xsum_lm_datasets, "[mix] xsum lm_datasets has no train split"

                hotpot_train = _strip_to_lm_cols(hotpot_qa_lm_datasets["train"])
                xsum_train   = _strip_to_lm_cols(xsum_lm_datasets["train"])

                mixed_train = interleave_datasets(
                    [hotpot_train, xsum_train],
                    probabilities=[0.889, 0.111],
                    seed=training_args.seed,
                    stopping_strategy="first_exhausted",
                )
                lm_datasets["train"] = mixed_train

            if training_args.do_eval:
                if ("validation" in hotpot_qa_lm_datasets) and ("validation" in xsum_lm_datasets):
                    hotpot_val = _strip_to_lm_cols(hotpot_qa_lm_datasets["validation"])
                    xsum_val   = _strip_to_lm_cols(xsum_lm_datasets["validation"])
                    mixed_val  = concatenate_datasets([hotpot_val, xsum_val])
                    lm_datasets["validation"] = mixed_val
                    mix_eval_task_cache["validation"] = (
                        ["hotpot_qa"] * len(hotpot_val)
                        + ["xsum"] * len(xsum_val)
                    )
                else:
                    logger.warning("[mix] validation split missing in hotpot_qa or xsum; skip eval mixing.")

            if training_args.do_predict:
                if ("test" in hotpot_qa_lm_datasets) and ("test" in xsum_lm_datasets):
                    hotpot_test = _strip_to_lm_cols(hotpot_qa_lm_datasets["test"])
                    xsum_test   = _strip_to_lm_cols(xsum_lm_datasets["test"])
                    mixed_test  = concatenate_datasets([hotpot_test, xsum_test])
                    lm_datasets["test"] = mixed_test
                    mix_eval_task_cache["test"] = (
                        ["hotpot_qa"] * len(hotpot_test)
                        + ["xsum"] * len(xsum_test)
                    )
                else:
                    logger.warning("[mix] test split missing in hotpot_qa or xsum; skip predict mixing.")
            
    if training_args.do_train:
        # if "train" not in tokenized_datasets:
        #     raise ValueError("--do_train requires a train dataset")
        train_dataset = lm_datasets["train"]
        if data_args.max_train_samples is not None:
            if data_args.streaming:
                train_dataset = train_dataset.take(data_args.max_train_samples)
            else:
                max_train_samples = min(len(train_dataset), data_args.max_train_samples)
                train_dataset = train_dataset.select(range(max_train_samples))
    # debug_supervision(train_dataset, tokenizer)
    # os._exit(0)
    # stats = compute_pad_ratio_stats(train_dataset, block_size=data_args.block_size, n_samples=5000, seed=0)
    # print("Pad ratio stats:", stats)
    if training_args.do_eval:
        # if "validation" not in tokenized_datasets:
        #     raise ValueError("--do_eval requires a validation dataset")
        IS_HOTPOTQA = (data_args.dataset_name == "hotpot_qa")
        IS_MIX = (data_args.dataset_name == "mix")
        IS_PASSKEY = (data_args.dataset_name == "passkey")
        eval_dataset = lm_datasets["validation"]
        if data_args.max_eval_samples is not None:
            if data_args.streaming:
                eval_dataset = eval_dataset.take(data_args.max_eval_samples)
            else:
                max_eval_samples = min(len(eval_dataset), data_args.max_eval_samples)
                eval_dataset = eval_dataset.select(range(max_eval_samples))
                if IS_HOTPOTQA and hotpot_level_cache.get("validation") is not None:
                    hotpot_level_cache["validation"] = hotpot_level_cache["validation"][:max_eval_samples]
                if IS_MIX and mix_eval_task_cache.get("validation") is not None:
                    mix_eval_task_cache["validation"] = mix_eval_task_cache["validation"][:max_eval_samples]
        elif IS_HOTPOTQA and hotpot_level_cache.get("validation") is not None and not data_args.streaming:
            # Align level cache length with potential dataset length changes (e.g., cache filtering)
            if len(hotpot_level_cache["validation"]) != len(eval_dataset):
                hotpot_level_cache["validation"] = hotpot_level_cache["validation"][: len(eval_dataset)]
        elif IS_MIX and mix_eval_task_cache.get("validation") is not None and not data_args.streaming:
            if len(mix_eval_task_cache["validation"]) != len(eval_dataset):
                mix_eval_task_cache["validation"] = mix_eval_task_cache["validation"][: len(eval_dataset)]

        #### hotpot_qa setting ########
        hotpot_eval_levels = hotpot_level_cache.get("validation") if IS_HOTPOTQA else None
        hotpot_eval_placements = hotpot_placement_cache.get("validation") if IS_HOTPOTQA else None
        mix_eval_tasks = mix_eval_task_cache.get("validation") if IS_MIX else None

        # Dump eval dataset (text + token ids) for downstream attention analysis
        # dump_dir = Path(training_args.output_dir)
        # dump_dir.mkdir(parents=True, exist_ok=True)
        # dump_path = dump_dir / "eval_dataset_dump.jsonl"
        # try:
        #     with dump_path.open("w", encoding="utf-8") as fout:
        #         for idx, ex in enumerate(eval_dataset):
        #             if not isinstance(ex, dict):
        #                 continue
        #             token_ids = ex.get("input_ids")
        #             text = None
        #             if token_ids is not None and tokenizer is not None:
        #                 text = tokenizer.decode(token_ids, skip_special_tokens=True)
        #             if text is None and "text" in ex:
        #                 text = str(ex["text"])

        #             record = {
        #                 "idx": idx,
        #                 "text": text,
        #                 "token_ids": token_ids,
        #             }
        #             if "attention_mask" in ex:
        #                 record["attention_mask"] = ex["attention_mask"]
        #             if "labels" in ex:
        #                 record["labels"] = ex["labels"]

        #             fout.write(json.dumps(record, ensure_ascii=False) + "\n")
        #             break
        #     logger.info("Saved eval dataset dump to %s", dump_path)
        # except Exception as e:
        #     logger.warning("Failed to dump eval dataset: %s", e)
        # Dump eval dataset (text + token ids) for downstream attention analysis
        # pdb.set_trace()
        def preprocess_logits_for_metrics(logits, labels):
            if isinstance(logits, tuple):
                logits = logits[0]
            return logits.argmax(dim=-1)
        
        #### hotpot_qa setting ########
        metric = evaluate.load("accuracy", cache_dir=model_args.cache_dir)
        IS_XSUM = (data_args.dataset_name == "xsum")
        rouge_metric = evaluate.load("rouge") if (IS_XSUM or IS_MIX) else None
        bertscore_metric = evaluate.load("bertscore") if IS_XSUM else None

        # extra reference-based metrics
        bleu_metric = evaluate.load("bleu") if IS_XSUM else None
        sacrebleu_metric = evaluate.load("sacrebleu") if IS_XSUM else None
        meteor_metric = evaluate.load("meteor") if IS_XSUM else None
        chrf_metric = evaluate.load("chrf") if IS_XSUM else None
        paired_setting_raw = str(getattr(training_args, "paired_eval_setting", "auto")).strip().lower()
        if paired_setting_raw not in {"auto", "baseline", "rel0"}:
            raise ValueError(f"paired_eval_setting must be one of: auto|baseline|rel0, got {paired_setting_raw}")
        rel_alpha_eval = float(getattr(config, "rel_alpha", getattr(training_args, "rel_alpha", 1.0)))
        rel_zoom_eval = float(getattr(config, "rel_zoom_in_coe", 1.0))
        is_rel0_effective = abs(rel_alpha_eval * rel_zoom_eval) <= 1e-12
        eval_setting = (
            paired_setting_raw if paired_setting_raw != "auto"
            else ("rel0" if is_rel0_effective else "baseline")
        )
        eval_records_path = Path(training_args.output_dir) / f"L{config.block_size}_{rel_zoom_eval}scale_eval_records_{eval_setting}.jsonl"
        paired_summary_path = Path(training_args.output_dir) / "paired_eval_summary.json"
        paired_bucket_path = Path(training_args.output_dir) / "paired_eval_bucket_summary.jsonl"
        bootstrap_B = max(1, int(getattr(training_args, "paired_eval_bootstrap_B", 2000)))
        seq_bucket_size = max(1, int(getattr(training_args, "paired_eval_seq_bin_size", 256)))
        _is_world_zero = int(getattr(training_args, "process_index", 0)) == 0
        save_eval_sample_records = bool((not training_args.do_train) and cfg_bool(cfg, "save_eval_sample_records", True))
        logger.info(
            "[PairedEval] setting=%s rel_alpha=%.6e rel_zoom_in_coe=%.6e effective_alpha=%.6e "
            "records_path=%s bootstrap_B=%d seq_bucket_size=%d",
            eval_setting, rel_alpha_eval, rel_zoom_eval, rel_alpha_eval * rel_zoom_eval,
            str(eval_records_path), bootstrap_B, seq_bucket_size,
        )
        if not save_eval_sample_records:
            logger.info(
                "[PairedEval] per-sample eval record saving disabled (do_train=%s, save_eval_sample_records=%s).",
                str(bool(training_args.do_train)),
                str(bool(save_eval_sample_records)),
            )

        def _stable_example_id(row_idx: int, ref_ids):
            payload = f"{int(row_idx)}|" + ",".join(str(int(x)) for x in ref_ids)
            return hashlib.md5(payload.encode("utf-8")).hexdigest()

        def _write_eval_records(records):
            if not save_eval_sample_records:
                return
            if not _is_world_zero:
                return
            eval_records_path.parent.mkdir(parents=True, exist_ok=True)
            with eval_records_path.open("w", encoding="utf-8") as fout:
                for rec in records:
                    fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            logger.info(
                "[PairedEval] wrote %d records to %s (setting=%s)",
                len(records),
                str(eval_records_path),
                eval_setting,
            )

        def _load_jsonl_records(path: Path):
            out = []
            if not path.exists():
                return out
            with path.open("r", encoding="utf-8") as fin:
                for line_no, line in enumerate(fin, start=1):
                    s = line.strip()
                    if not s:
                        continue
                    try:
                        out.append(json.loads(s))
                    except Exception as e:
                        logger.warning("[PairedEval] failed to parse %s:%d (%s)", str(path), line_no, e)
            return out

        def _bootstrap_ci_mean(delta_arr: np.ndarray, B: int = 2000, seed: int = 1234):
            n = int(delta_arr.shape[0])
            if n <= 0:
                return float("nan"), float("nan")
            rng = np.random.default_rng(seed)
            means = np.empty((B,), dtype=np.float64)
            for bi in range(B):
                sample_idx = rng.integers(0, n, size=n)
                means[bi] = float(delta_arr[sample_idx].mean())
            lo = float(np.quantile(means, 0.025))
            hi = float(np.quantile(means, 0.975))
            return lo, hi

        def _delta_stats(delta_arr: np.ndarray, B: int, seed: int):
            if delta_arr.size == 0:
                return {
                    "N": 0,
                    "mean_delta": float("nan"),
                    "median_delta": float("nan"),
                    "n_pos": 0,
                    "n_neg": 0,
                    "n_zero": 0,
                    "pos_ratio": float("nan"),
                    "ci_low": float("nan"),
                    "ci_high": float("nan"),
                    "bootstrap_B": int(B),
                }
            n_pos = int((delta_arr > 0.0).sum())
            n_neg = int((delta_arr < 0.0).sum())
            n_zero = int((delta_arr == 0.0).sum())
            ci_low, ci_high = _bootstrap_ci_mean(delta_arr, B=B, seed=seed)
            return {
                "N": int(delta_arr.size),
                "mean_delta": float(delta_arr.mean()),
                "median_delta": float(np.median(delta_arr)),
                "n_pos": n_pos,
                "n_neg": n_neg,
                "n_zero": n_zero,
                "pos_ratio": float(n_pos / max(int(delta_arr.size), 1)),
                "ci_low": ci_low,
                "ci_high": ci_high,
                "bootstrap_B": int(B),
            }

        def _records_to_unique_map(records, tag):
            mp = {}
            dup = []
            for rec in records:
                eid = str(rec.get("example_id", "")).strip()
                if not eid:
                    continue
                if eid in mp:
                    dup.append(eid)
                    continue
                mp[eid] = rec
            if dup:
                logger.warning("[PairedEval] %s duplicate example_id count=%d", tag, len(dup))
            return mp

        def run_paired_analysis_if_ready():
            if not _is_world_zero:
                return
            baseline_path = Path(training_args.output_dir) / "eval_records_baseline.jsonl"
            rel0_path = Path(training_args.output_dir) / "eval_records_rel0.jsonl"
            if not baseline_path.exists() or not rel0_path.exists():
                logger.info(
                    "[PairedEval] waiting for both files. baseline_exists=%s rel0_exists=%s",
                    baseline_path.exists(),
                    rel0_path.exists(),
                )
                return

            base_records = _load_jsonl_records(baseline_path)
            rel0_records = _load_jsonl_records(rel0_path)
            base_map = _records_to_unique_map(base_records, "baseline")
            rel0_map = _records_to_unique_map(rel0_records, "rel0")

            base_ids = set(base_map.keys())
            rel0_ids = set(rel0_map.keys())
            inter = sorted(base_ids & rel0_ids)
            missing_in_base = sorted(rel0_ids - base_ids)
            missing_in_rel0 = sorted(base_ids - rel0_ids)

            logger.info(
                "[PairedEval][Align] N_baseline=%d N_rel0=%d N_intersection=%d "
                "N_missing_baseline=%d N_missing_rel0=%d",
                len(base_map), len(rel0_map), len(inter), len(missing_in_base), len(missing_in_rel0),
            )
            logger.info("[PairedEval][Align] missing_in_baseline(first50)=%s", missing_in_base[:50])
            logger.info("[PairedEval][Align] missing_in_rel0(first50)=%s", missing_in_rel0[:50])

            if not inter:
                logger.warning("[PairedEval] empty intersection; skip paired stats")
                return

            deltas = []
            seq_lens = []
            for eid in inter:
                b = base_map[eid]
                r = rel0_map[eid]
                deltas.append(float(b.get("f1", 0.0)) - float(r.get("f1", 0.0)))
                seq_len_b = b.get("seq_len", None)
                seq_len_r = r.get("seq_len", None)
                seq_len = seq_len_b if seq_len_b is not None else seq_len_r
                try:
                    seq_lens.append(int(seq_len))
                except Exception:
                    seq_lens.append(0)

            delta_arr = np.asarray(deltas, dtype=np.float64)
            overall = _delta_stats(delta_arr, B=bootstrap_B, seed=2027)
            logger.info(
                "[PairedEval][Overall] N=%d mean_delta=%.6e median_delta=%.6e "
                "n_pos=%d n_neg=%d n_zero=%d pos_ratio=%.6f ci_low=%.6e ci_high=%.6e bootstrap_B=%d",
                overall["N"], overall["mean_delta"], overall["median_delta"],
                overall["n_pos"], overall["n_neg"], overall["n_zero"], overall["pos_ratio"],
                overall["ci_low"], overall["ci_high"], overall["bootstrap_B"],
            )

            bucket_map = {}
            for d, sl in zip(delta_arr.tolist(), seq_lens):
                b0 = (int(sl) // seq_bucket_size) * seq_bucket_size
                b1 = b0 + seq_bucket_size
                key = (b0, b1)
                bucket_map.setdefault(key, []).append(float(d))

            bucket_records = []
            for b0, b1 in sorted(bucket_map.keys()):
                arr = np.asarray(bucket_map[(b0, b1)], dtype=np.float64)
                st = _delta_stats(arr, B=bootstrap_B, seed=4000 + int(b0))
                rec = {
                    "bucket_range": f"{b0}-{b1}",
                    "N_bucket": st["N"],
                    "mean_delta_bucket": st["mean_delta"],
                    "median_delta_bucket": st["median_delta"],
                    "n_pos": st["n_pos"],
                    "n_neg": st["n_neg"],
                    "n_zero": st["n_zero"],
                    "pos_ratio": st["pos_ratio"],
                    "ci_low": st["ci_low"],
                    "ci_high": st["ci_high"],
                    "bootstrap_B": st["bootstrap_B"],
                }
                bucket_records.append(rec)
                logger.info(
                    "[PairedEval][Bucket] bucket=%s N=%d mean_delta=%.6e ci_low=%.6e ci_high=%.6e "
                    "n_pos=%d n_neg=%d n_zero=%d pos_ratio=%.6f",
                    rec["bucket_range"], rec["N_bucket"], rec["mean_delta_bucket"],
                    rec["ci_low"], rec["ci_high"], rec["n_pos"], rec["n_neg"], rec["n_zero"], rec["pos_ratio"],
                )

            paired_summary = {
                "alignment": {
                    "N_baseline": len(base_map),
                    "N_rel0": len(rel0_map),
                    "N_intersection": len(inter),
                    "N_missing_baseline": len(missing_in_base),
                    "N_missing_rel0": len(missing_in_rel0),
                    "missing_in_baseline_head50": missing_in_base[:50],
                    "missing_in_rel0_head50": missing_in_rel0[:50],
                },
                "overall": overall,
                "seq_bucket_size": int(seq_bucket_size),
                "bucket_count": len(bucket_records),
            }
            with paired_summary_path.open("w", encoding="utf-8") as fout:
                json.dump(paired_summary, fout, ensure_ascii=False, indent=2)
            with paired_bucket_path.open("w", encoding="utf-8") as fout:
                for rec in bucket_records:
                    fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            logger.info("[PairedEval] summary saved: %s", str(paired_summary_path))
            logger.info("[PairedEval] bucket summary saved: %s", str(paired_bucket_path))

        eval_output_js_enabled = bool((not training_args.do_train) and cfg_bool(cfg, "eval_output_js_export_enabled", False))
        eval_output_js_case_index = max(0, int(cfg_int(cfg, "eval_output_js_case_index", 0)))
        eval_output_js_chunk_size = max(1, int(cfg_int(cfg, "eval_output_js_chunk_size", 8)))
        eval_output_js_max_positions = max(0, int(cfg_int(cfg, "eval_output_js_max_positions", 0)))
        eval_output_js_sample_positions = max(1, int(cfg_int(cfg, "eval_output_js_sample_positions", 8)))
        eval_output_js_topk = max(1, int(cfg_int(cfg, "eval_output_js_topk", 12)))
        eval_output_js_save_pt = bool(cfg_bool(cfg, "eval_output_js_save_pt", False))
        eval_output_js_outdir = cfg_str(cfg, "eval_output_js_outdir", "eval_output_js")
        eval_output_js_compute_softmax = bool(cfg_bool(cfg, "eval_output_js_compute_softmax_metrics", False))
        if eval_output_js_enabled and _is_world_zero:
            logger.info(
                "[EvalOutputJS] enabled=1 case_index=%d outdir=%s chunk=%d max_positions=%d topk=%d sample_positions=%d save_pt=%d softmax_metrics=%d",
                int(eval_output_js_case_index),
                str(eval_output_js_outdir),
                int(eval_output_js_chunk_size),
                int(eval_output_js_max_positions),
                int(eval_output_js_topk),
                int(eval_output_js_sample_positions),
                int(eval_output_js_save_pt),
                int(eval_output_js_compute_softmax),
            )

        def _collect_attn_output_delta_by_layer(model_root):
            recs = []
            for mod in model_root.modules():
                st = getattr(mod, "_eval_attn_mech_last", None)
                if isinstance(st, dict) and ("layer" in st):
                    recs.append(dict(st))
            recs.sort(key=lambda x: int(x.get("layer", 10**9)))
            return recs

        def _vector_delta_metrics(base: torch.Tensor, wave: torch.Tensor, mask_bt: torch.Tensor, eps: float = 1e-12):
            base_f = torch.nan_to_num(base.detach().float(), nan=0.0, posinf=0.0, neginf=0.0)
            wave_f = torch.nan_to_num(wave.detach().float(), nan=0.0, posinf=0.0, neginf=0.0)
            delta_f = wave_f - base_f
            if mask_bt is None:
                mask_bt = torch.ones(base_f.shape[:2], dtype=torch.bool, device=base_f.device)
            else:
                mask_bt = mask_bt.to(device=base_f.device, dtype=torch.bool)
            if int(mask_bt.numel()) <= 0:
                mask_bt = torch.ones(base_f.shape[:2], dtype=torch.bool, device=base_f.device)
            mask_f = mask_bt.to(dtype=base_f.dtype)
            valid_cnt = mask_f.sum().clamp_min(1.0)

            base_tok = torch.linalg.vector_norm(base_f, ord=2, dim=-1)
            wave_tok = torch.linalg.vector_norm(wave_f, ord=2, dim=-1)
            delta_tok = torch.linalg.vector_norm(delta_f, ord=2, dim=-1)
            rel_tok = delta_tok / base_tok.clamp_min(eps)
            rel_vals = rel_tok[mask_bt]
            if int(rel_vals.numel()) > 0:
                q50 = float(torch.quantile(rel_vals, 0.50).item())
                q90 = float(torch.quantile(rel_vals, 0.90).item())
                q99 = float(torch.quantile(rel_vals, 0.99).item())
            else:
                q50 = 0.0
                q90 = 0.0
                q99 = 0.0

            mean_base_tok = (base_tok * mask_f).sum() / valid_cnt
            mean_delta_tok = (delta_tok * mask_f).sum() / valid_cnt
            mean_wave_tok = (wave_tok * mask_f).sum() / valid_cnt
            mean_rel_tok = (rel_tok * mask_f).sum() / valid_cnt

            base_sq = (base_f.square().sum(dim=-1) * mask_f).sum()
            wave_sq = (wave_f.square().sum(dim=-1) * mask_f).sum()
            delta_sq = (delta_f.square().sum(dim=-1) * mask_f).sum()
            dot = ((base_f * wave_f).sum(dim=-1) * mask_f).sum()
            base_l2 = torch.sqrt(base_sq.clamp_min(eps))
            wave_l2 = torch.sqrt(wave_sq.clamp_min(eps))
            delta_l2 = torch.sqrt(delta_sq.clamp_min(eps))
            cos = dot / (base_l2 * wave_l2).clamp_min(eps)

            return {
                "num_valid_positions": int(mask_bt.sum().item()),
                "E_norm_delta_over_E_norm_base": float((mean_delta_tok / mean_base_tok.clamp_min(eps)).item()),
                "E_norm_delta": float(mean_delta_tok.item()),
                "E_norm_base": float(mean_base_tok.item()),
                "E_norm_wave": float(mean_wave_tok.item()),
                "E_norm_delta_over_norm_base": float(mean_rel_tok.item()),
                "E_norm_delta_over_norm_base_p50": float(q50),
                "E_norm_delta_over_norm_base_p90": float(q90),
                "E_norm_delta_over_norm_base_p99": float(q99),
                "global_l2_rel": float((delta_l2 / base_l2.clamp_min(eps)).item()),
                "global_cosine": float(cos.item()),
            }

        def run_eval_output_js_probe_if_enabled(_trainer):
            if not eval_output_js_enabled:
                return
            if not _is_world_zero:
                return
            try:
                ds_len = len(eval_dataset)
            except Exception:
                logger.warning("[EvalOutputJS] eval_dataset has no __len__; skip probe.")
                return
            if ds_len <= 0:
                logger.warning("[EvalOutputJS] empty eval_dataset; skip probe.")
                return

            case_idx = min(int(eval_output_js_case_index), int(ds_len) - 1)
            ex = eval_dataset[int(case_idx)]
            if not isinstance(ex, dict):
                logger.warning("[EvalOutputJS] eval example is not dict at idx=%d; skip probe.", int(case_idx))
                return
            batch = default_data_collator([ex])
            if "input_ids" not in batch:
                logger.warning("[EvalOutputJS] missing input_ids in eval batch; skip probe.")
                return
            if "labels" not in batch:
                labels = batch["input_ids"].clone()
                if "attention_mask" in batch:
                    labels = labels.masked_fill(batch["attention_mask"] <= 0, -100)
                batch["labels"] = labels

            model_eval = _trainer.model
            device = _trainer.args.device
            for k, v in list(batch.items()):
                if torch.is_tensor(v):
                    batch[k] = v.to(device)

            root = getattr(model_eval, "module", model_eval)
            cfg_obj = getattr(root, "config", None)
            if cfg_obj is None:
                logger.warning("[EvalOutputJS] model config missing; skip probe.")
                return

            rel_alpha_orig = float(getattr(cfg_obj, "rel_alpha", 1.0))
            wavelet_mode_orig = getattr(cfg_obj, "wavelet_mode", None)
            heatmap_orig = getattr(cfg_obj, "eval_attn_heatmap_enabled", None)
            stop_orig = getattr(cfg_obj, "eval_attn_heatmap_stop_after_case", None)
            mech_orig = getattr(cfg_obj, "eval_attn_mech_enabled", None)

            was_training = bool(model_eval.training)
            model_eval.eval()
            attn_output_delta_by_layer = []
            mod_mech_restore = []
            try:
                if heatmap_orig is not None:
                    setattr(cfg_obj, "eval_attn_heatmap_enabled", False)
                if stop_orig is not None:
                    setattr(cfg_obj, "eval_attn_heatmap_stop_after_case", False)
                setattr(cfg_obj, "eval_attn_mech_enabled", True)
                for mod in root.modules():
                    if hasattr(mod, "eval_attn_mech_enabled"):
                        prev = bool(getattr(mod, "eval_attn_mech_enabled"))
                        mod_mech_restore.append((mod, prev))
                        setattr(mod, "eval_attn_mech_enabled", True)
                        if hasattr(mod, "_eval_attn_mech_last"):
                            setattr(mod, "_eval_attn_mech_last", None)

                with torch.no_grad():
                    out_wavelet = model_eval(
                        **batch,
                        output_hidden_states=True,
                        output_attentions=False,
                        use_cache=False,
                        return_dict=True,
                    )
                    logits_wavelet = out_wavelet.logits.detach().float()
                    hidden_wavelet = tuple(h.detach().float() for h in out_wavelet.hidden_states) if out_wavelet.hidden_states is not None else tuple()
                    attn_output_delta_by_layer = _collect_attn_output_delta_by_layer(root)

                    setattr(cfg_obj, "rel_alpha", 0.0)
                    # For ctxscale/logit-bias modes, rel_alpha does not disable wavelet path.
                    # Explicitly switch wavelet_mode off to build true baseline forward.
                    setattr(cfg_obj, "wavelet_mode", "off")
                    out_base = model_eval(
                        **batch,
                        output_hidden_states=True,
                        output_attentions=False,
                        use_cache=False,
                        return_dict=True,
                    )
                    logits_base = out_base.logits.detach().float()
                    hidden_base = tuple(h.detach().float() for h in out_base.hidden_states) if out_base.hidden_states is not None else tuple()
            finally:
                setattr(cfg_obj, "rel_alpha", rel_alpha_orig)
                if heatmap_orig is not None:
                    setattr(cfg_obj, "eval_attn_heatmap_enabled", heatmap_orig)
                if stop_orig is not None:
                    setattr(cfg_obj, "eval_attn_heatmap_stop_after_case", stop_orig)
                if mech_orig is not None:
                    setattr(cfg_obj, "eval_attn_mech_enabled", mech_orig)
                else:
                    try:
                        delattr(cfg_obj, "eval_attn_mech_enabled")
                    except Exception:
                        setattr(cfg_obj, "eval_attn_mech_enabled", False)
                if wavelet_mode_orig is not None:
                    setattr(cfg_obj, "wavelet_mode", wavelet_mode_orig)
                else:
                    try:
                        delattr(cfg_obj, "wavelet_mode")
                    except Exception:
                        setattr(cfg_obj, "wavelet_mode", "off")
                for mod, prev in mod_mech_restore:
                    try:
                        setattr(mod, "eval_attn_mech_enabled", bool(prev))
                    except Exception:
                        pass
                if was_training:
                    model_eval.train()

            z_base = logits_base[:, :-1, :].contiguous()
            z_wave = logits_wavelet[:, :-1, :].contiguous()
            labels = batch.get("labels", None)
            attn_mask = batch.get("attention_mask", None)
            if labels is not None:
                valid_mask_shift = labels[:, 1:] != -100
            elif attn_mask is not None:
                valid_mask_shift = attn_mask[:, 1:] > 0
            else:
                valid_mask_shift = torch.ones(z_base.shape[:2], dtype=torch.bool, device=z_base.device)
            if int(valid_mask_shift.sum().item()) <= 0:
                valid_mask_shift = torch.ones(z_base.shape[:2], dtype=torch.bool, device=z_base.device)
            valid_mask_hidden = (attn_mask > 0) if attn_mask is not None else torch.ones(
                (z_base.shape[0], z_base.shape[1] + 1), dtype=torch.bool, device=z_base.device
            )
            if int(valid_mask_hidden.sum().item()) <= 0:
                valid_mask_hidden = torch.ones_like(valid_mask_hidden, dtype=torch.bool)

            delta_logits = _vector_delta_metrics(z_base, z_wave, valid_mask_shift)
            top1_changed = ((z_base.argmax(dim=-1) != z_wave.argmax(dim=-1)) & valid_mask_shift).to(dtype=torch.float32)
            delta_logits["top1_changed_ratio"] = float(top1_changed.sum().item() / max(int(valid_mask_shift.sum().item()), 1))
            base_abs_tok = z_base.abs().mean(dim=-1)
            delta_abs_tok = (z_wave - z_base).abs().mean(dim=-1)
            mask_f_shift = valid_mask_shift.to(dtype=base_abs_tok.dtype)
            base_abs_mean = float((base_abs_tok * mask_f_shift).sum().item() / max(float(mask_f_shift.sum().item()), 1.0))
            delta_abs_mean = float((delta_abs_tok * mask_f_shift).sum().item() / max(float(mask_f_shift.sum().item()), 1.0))
            delta_logits["delta_logit_abs_over_base_abs_mean"] = float(delta_abs_mean / max(base_abs_mean, 1e-12))

            delta_h_by_layer = []
            n_hidden_layers = min(len(hidden_base), len(hidden_wavelet))
            for li in range(n_hidden_layers):
                h_metric = _vector_delta_metrics(hidden_base[li], hidden_wavelet[li], valid_mask_hidden)
                h_metric["layer"] = int(li)
                h_metric["name"] = "embedding" if li == 0 else f"block_{li - 1}_output"
                delta_h_by_layer.append(h_metric)

            valid_idx_all = valid_mask_shift.nonzero(as_tuple=False)
            sample_dump = []
            sample_take = min(int(eval_output_js_sample_positions), int(valid_idx_all.shape[0]))
            k_top = max(1, int(eval_output_js_topk))
            for i in range(sample_take):
                b_i = int(valid_idx_all[i, 0].item())
                t_i = int(valid_idx_all[i, 1].item())
                pb = torch.softmax(z_base[b_i, t_i, :], dim=-1)
                pw = torch.softmax(z_wave[b_i, t_i, :], dim=-1)
                kb = min(k_top, int(pb.numel()))
                kw = min(k_top, int(pw.numel()))
                b_topv, b_topi = torch.topk(pb, k=kb, dim=-1)
                w_topv, w_topi = torch.topk(pw, k=kw, dim=-1)
                sample_dump.append(
                    {
                        "batch_index": b_i,
                        "shift_pos": t_i,
                        "next_token_label": int(batch["labels"][b_i, t_i + 1].item()) if "labels" in batch else None,
                        "topk_base": [{"id": int(tid), "p": float(tp)} for tid, tp in zip(b_topi.tolist(), b_topv.tolist())],
                        "topk_wavelet": [{"id": int(tid), "p": float(tp)} for tid, tp in zip(w_topi.tolist(), w_topv.tolist())],
                    }
                )

            softmax_metrics = None
            if eval_output_js_compute_softmax:
                valid_idx = valid_idx_all
                if int(eval_output_js_max_positions) > 0 and int(valid_idx.shape[0]) > int(eval_output_js_max_positions):
                    keep = int(eval_output_js_max_positions)
                    sel = torch.linspace(
                        0,
                        int(valid_idx.shape[0]) - 1,
                        steps=keep,
                        device=valid_idx.device,
                    ).round().to(dtype=torch.long)
                    valid_idx = valid_idx.index_select(0, sel)
                eps = 1e-12
                js_nats_vals = []
                tv_vals = []
                for st in range(0, int(valid_idx.shape[0]), int(eval_output_js_chunk_size)):
                    ed = min(st + int(eval_output_js_chunk_size), int(valid_idx.shape[0]))
                    idx_chunk = valid_idx[st:ed]
                    b_idx = idx_chunk[:, 0]
                    t_idx = idx_chunk[:, 1]
                    zb = z_base[b_idx, t_idx, :]
                    zw = z_wave[b_idx, t_idx, :]
                    lpb = torch.log_softmax(zb, dim=-1)
                    lpw = torch.log_softmax(zw, dim=-1)
                    pb = lpb.exp()
                    pw = lpw.exp()
                    m = 0.5 * (pb + pw)
                    lm = torch.log(m.clamp_min(eps))
                    js_nats = 0.5 * ((pb * (lpb - lm)).sum(dim=-1) + (pw * (lpw - lm)).sum(dim=-1))
                    tv = 0.5 * (pb - pw).abs().sum(dim=-1)
                    js_nats_vals.extend([float(x) for x in js_nats.detach().cpu().tolist()])
                    tv_vals.extend([float(x) for x in tv.detach().cpu().tolist()])
                js_bits_arr = np.asarray([float(x / math.log(2.0)) for x in js_nats_vals], dtype=np.float64)
                tv_arr = np.asarray(tv_vals, dtype=np.float64)
                softmax_metrics = {
                    "num_positions": int(valid_idx.shape[0]),
                    "js_bits_mean": float(js_bits_arr.mean()) if js_bits_arr.size else 0.0,
                    "js_bits_p90": float(np.quantile(js_bits_arr, 0.90)) if js_bits_arr.size else 0.0,
                    "js_bits_p99": float(np.quantile(js_bits_arr, 0.99)) if js_bits_arr.size else 0.0,
                    "tv_mean": float(tv_arr.mean()) if tv_arr.size else 0.0,
                    "tv_p90": float(np.quantile(tv_arr, 0.90)) if tv_arr.size else 0.0,
                }

            block_tag = f"block_{int(getattr(config, 'block_size', -1))}"
            run_tag = str(getattr(config, "eval_attn_heatmap_run_tag", "default"))
            out_root = Path(str(eval_output_js_outdir))
            if not out_root.is_absolute():
                out_root = Path.cwd() / out_root
            out_root = out_root / block_tag / run_tag / f"case{int(case_idx):03d}"
            out_root.mkdir(parents=True, exist_ok=True)

            summary = {
                "probe_kind": "mechanism_probe_v2",
                "case_index": int(case_idx),
                "num_eval_examples": int(ds_len),
                "block_tag": str(block_tag),
                "run_tag": str(run_tag),
                "rel_alpha_wavelet": float(rel_alpha_orig),
                "rel_alpha_base": 0.0,
                "delta_logits": delta_logits,
                "delta_h_by_layer": delta_h_by_layer,
                "delta_o_by_layer": attn_output_delta_by_layer,
                "softmax_distribution_metrics": softmax_metrics,
                "sample_positions": sample_dump,
            }

            out_json_mech = out_root / "mechanism_probe.json"
            with out_json_mech.open("w", encoding="utf-8") as fout:
                json.dump(summary, fout, ensure_ascii=False, indent=2)
            out_json_legacy = out_root / "output_logits_js.json"
            with out_json_legacy.open("w", encoding="utf-8") as fout:
                json.dump(summary, fout, ensure_ascii=False, indent=2)

            if eval_output_js_save_pt:
                torch.save(
                    {
                        "meta": summary,
                        "logits_base": logits_base.cpu(),
                        "logits_wavelet": logits_wavelet.cpu(),
                        "valid_mask_shift": valid_mask_shift.detach().cpu(),
                        "valid_mask_hidden": valid_mask_hidden.detach().cpu(),
                    },
                    out_root / "output_logits_case.pt",
                )
            logger.info("[EvalOutputJS] saved mechanism probe to %s", str(out_json_mech))

        def _build_hotpot_sample_records(preds_np, labels_np, indices):
            labels_shift = labels_np[:, 1:]
            preds_shift = preds_np[:, :-1]
            records = []
            f1_list, em_list = [], []
            for i in indices:
                m = labels_shift[i] != -100
                seq_len = int(m.sum())
                if seq_len == 0:
                    continue
                ref_ids = labels_shift[i][m].tolist()
                pred_ids = preds_shift[i][m].tolist()
                ref_text = tokenizer.decode(ref_ids, skip_special_tokens=True)
                pred_text = tokenizer.decode(pred_ids, skip_special_tokens=True)
                f1_val = float(_f1(pred_text, ref_text))
                em_val = float(_normalize(pred_text) == _normalize(ref_text))
                eid = _stable_example_id(i, ref_ids)
                records.append(
                    {
                        "example_id": eid,
                        "sample_index": int(i),
                        "seq_len": seq_len,
                        "setting": eval_setting,
                        "f1": f1_val,
                        "em": em_val,
                        "pred_ids": [int(x) for x in pred_ids],
                        "label_ids": [int(x) for x in ref_ids],
                        "pred_text": pred_text,
                        "label_text": ref_text,
                        "gold_text": ref_text,
                    }
                )
                f1_list.append(f1_val)
                em_list.append(em_val)
            return records, f1_list, em_list

        def _normalize(s: str) -> str:
            s = s.lower()
            s = re.sub(r"\b(a|an|the)\b", " ", s)
            s = "".join(ch for ch in s if ch not in string.punctuation)
            s = " ".join(s.split())
            return s

        def _f1(pred: str, ref: str) -> float:
            pred_toks = _normalize(pred).split()
            ref_toks  = _normalize(ref).split()
            if len(pred_toks) == 0 and len(ref_toks) == 0:
                return 1.0
            if len(pred_toks) == 0 or len(ref_toks) == 0:
                return 0.0
            common = Counter(pred_toks) & Counter(ref_toks)
            num_same = sum(common.values())
            if num_same == 0:
                return 0.0
            p = num_same / len(pred_toks)
            r = num_same / len(ref_toks)
            return 2 * p * r / (p + r)        
        def _to_numpy(x):
            if hasattr(x, "detach"):
                return x.detach().cpu().numpy()
            return np.asarray(x)

        def _compute_hotpot_metrics(preds_np, labels_np, indices):
            if len(indices) == 0:
                return {"hotpot_f1": 0.0, "hotpot_em": 0.0, "hotpot_count": 0}, []
            records, f1s, ems = _build_hotpot_sample_records(preds_np, labels_np, indices)
            return {
                "hotpot_f1": float(np.mean(f1s)) if len(f1s) else 0.0,
                "hotpot_em": float(np.mean(ems)) if len(ems) else 0.0,
                "hotpot_count": len(f1s),
            }, records

        def _compute_xsum_metrics(preds_np, labels_np, indices):
            if len(indices) == 0:
                return {"xsum_count": 0}
            labels_shift = labels_np[:, 1:]
            preds_shift  = preds_np[:, :-1]
            pred_texts, ref_texts = [], []
            for i in indices:
                mask = labels_shift[i] != -100
                if mask.sum() == 0:
                    continue
                ref_ids  = labels_shift[i][mask].tolist()
                pred_ids = preds_shift[i][mask].tolist()
                ref_texts.append(tokenizer.decode(ref_ids,  skip_special_tokens=True))
                pred_texts.append(tokenizer.decode(pred_ids, skip_special_tokens=True))
            if len(pred_texts) == 0:
                return {"xsum_count": 0}
            rouge = rouge_metric.compute(predictions=pred_texts, references=ref_texts, use_stemmer=True)
            out = {
                "xsum_rouge1": float(rouge.get("rouge1", 0.0)),
                "xsum_rouge2": float(rouge.get("rouge2", 0.0)),
                "xsum_rougeL": float(rouge.get("rougeL", 0.0)),
                "xsum_count":  len(pred_texts),
            }
            return out
        def _safe_mean(xs):
            return float(np.mean(xs)) if len(xs) else 0.0

        def _safe_std(xs):
            return float(np.std(xs, ddof=1)) if len(xs) > 1 else 0.0
        xsum_alignscore_scorer = None
        xsum_fenice_scorer = None

        def _get_alignscore_scorer():
            nonlocal xsum_alignscore_scorer
            if xsum_alignscore_scorer is not None:
                return xsum_alignscore_scorer
            try:
                from alignscore import AlignScore
                ckpt_path = os.environ.get("ALIGNSCORE_CKPT", "").strip()
                if not ckpt_path:
                    logger.warning("[AlignScore] ALIGNSCORE_CKPT is empty, skip AlignScore.")
                    return None
                device = os.environ.get("ALIGNSCORE_DEVICE", "cuda:0")
                batch_size = int(os.environ.get("ALIGNSCORE_BATCH_SIZE", "8"))
                model_name = os.environ.get("ALIGNSCORE_MODEL", "roberta-base")
                eval_mode = os.environ.get("ALIGNSCORE_EVAL_MODE", "nli_sp")
                xsum_alignscore_scorer = AlignScore(
                    model=model_name,
                    batch_size=batch_size,
                    device=device,
                    ckpt_path=ckpt_path,
                    evaluation_mode=eval_mode,
                )
                logger.info(f"[AlignScore] initialized with ckpt={ckpt_path}")
                return xsum_alignscore_scorer
            except Exception as e:
                logger.warning(f"[AlignScore] init failed: {e}")
                return None

        def _get_fenice_scorer():
            nonlocal xsum_fenice_scorer
            if xsum_fenice_scorer is not None:
                return xsum_fenice_scorer
            try:
                from metric.FENICE import FENICE
                xsum_fenice_scorer = FENICE()
                logger.info("[FENICE] initialized")
                return xsum_fenice_scorer
            except Exception as e:
                logger.warning(f"[FENICE] init failed: {e}")
                return None    
        _SUMMAC_SCORER = None

        def _get_summac_scorer():
            nonlocal _SUMMAC_SCORER
            if _SUMMAC_SCORER is not None:
                return _SUMMAC_SCORER
            try:
                # Patch tokenizers.pre_tokenizers.Metaspace / decoders.Metaspace to drop
                # the 'prepend_scheme' kwarg that older tokenizers versions don't support.
                # convert_slow_tokenizer.py accesses these via module ref, so patching works.
                try:
                    from tokenizers import pre_tokenizers as _pt, decoders as _dec
                    _real_pt_ms = _pt.Metaspace
                    _real_dec_ms = _dec.Metaspace
                    def _safe_pt_ms(*a, prepend_scheme=None, **kw): return _real_pt_ms(*a, **kw)
                    def _safe_dec_ms(*a, prepend_scheme=None, **kw): return _real_dec_ms(*a, **kw)
                    _pt.Metaspace = _safe_pt_ms
                    _dec.Metaspace = _safe_dec_ms
                except Exception:
                    _real_pt_ms = _real_dec_ms = None

                from summac.model_summac import SummaCConv
                _SUMMAC_SCORER = SummaCConv(
                    models=["vitc"],
                    bins="percentile",
                    granularity="sentence",
                    nli_labels="e",
                    device="cuda" if torch.cuda.is_available() else "cpu"
                )
                logger.info("[SummaC] scorer initialized successfully")

                # tokenizer is loaded lazily in SummaCImager.load_nli().
                # Patch load_nli so that after tokenizer is created, batch_encode_plus
                # drops truncation_strategy (newer transformers already handles it via
                # truncation=True; passing both causes "multiple values" error).
                try:
                    from summac.model_summac import SummaCImager as _SummaCImager
                    _orig_load_nli = _SummaCImager.load_nli
                    def _patched_load_nli(_self):
                        _orig_load_nli(_self)
                        if hasattr(_self, 'tokenizer'):
                            _tok = _self.tokenizer
                            _orig_bep = _tok.batch_encode_plus  # bound method
                            def _safe_bep(*_a, truncation_strategy=None, _ob=_orig_bep, **_kw):
                                return _ob(*_a, **_kw)
                            _tok.batch_encode_plus = _safe_bep
                    _SummaCImager.load_nli = _patched_load_nli
                except Exception as _pe:
                    logger.warning(f"[SummaC] tokenizer patch failed: {_pe}")

                # Keep Metaspace patch applied permanently (just drops unknown kwarg)
            except Exception as e:
                logger.warning(f"[SummaC] init failed: {e}")
                _SUMMAC_SCORER = None
            return _SUMMAC_SCORER                        
        def compute_metrics(eval_preds):
            preds, labels = eval_preds
            if IS_MIX and mix_eval_tasks is not None:
                preds_np = _to_numpy(preds)
                labels_np = _to_numpy(labels)
                hotpot_indices = [i for i, t in enumerate(mix_eval_tasks) if t == "hotpot_qa"]
                xsum_indices = [i for i, t in enumerate(mix_eval_tasks) if t == "xsum"]
                out = {}
                hotpot_metrics, hotpot_records = _compute_hotpot_metrics(preds_np, labels_np, hotpot_indices)
                out.update(hotpot_metrics)
                out.update(_compute_xsum_metrics(preds_np, labels_np, xsum_indices))
                _write_eval_records(hotpot_records)
                return out
            if IS_HOTPOTQA:
                preds_np = _to_numpy(preds)
                labels_np = _to_numpy(labels)
                all_indices = list(range(labels_np.shape[0]))
                records, f1s, ems = _build_hotpot_sample_records(preds_np, labels_np, all_indices)
                labels_shift = labels_np[:, 1:]
                preds_shift  = preds_np[:, :-1]
                by_level = {"easy": [], "medium": [], "hard": []}
                # 5 placement buckets: [0,20%), [20,40%), [40,60%), [60,80%), [80,100%]
                _PLACEMENT_BUCKET_NAMES = ["p0_20", "p20_40", "p40_60", "p60_80", "p80_100"]
                by_placement: dict[str, list] = {k: [] for k in _PLACEMENT_BUCKET_NAMES}
                def _std(vals):
                    return float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
                for i in range(labels_shift.shape[0]):
                    m = labels_shift[i] != -100
                    if m.sum() == 0:
                        continue
                    ref_ids  = labels_shift[i][m].tolist()
                    pred_ids = preds_shift[i][m].tolist()
                    ref_text  = tokenizer.decode(ref_ids,  skip_special_tokens=True)
                    pred_text = tokenizer.decode(pred_ids, skip_special_tokens=True)
                    f1_val = _f1(pred_text, ref_text)
                    if hotpot_eval_levels is not None and i < len(hotpot_eval_levels):
                        lvl = str(hotpot_eval_levels[i]).lower()
                        if lvl in by_level:
                            by_level[lvl].append(f1_val)
                    if hotpot_eval_placements is not None and i < len(hotpot_eval_placements):
                        pct = float(hotpot_eval_placements[i])
                        if pct >= 0.0:  # -1.0 means unknown (non-JSONL example)
                            bucket_idx = min(4, int(pct * 5))
                            by_placement[_PLACEMENT_BUCKET_NAMES[bucket_idx]].append(f1_val)

                out = {
                    "f1": float(np.mean(f1s)) if len(f1s) else 0.0,
                    "f1_std": _std(f1s),
                    "em": float(np.mean(ems)) if len(ems) else 0.0,
                    "count": len(f1s),
                }
                for lvl_name, scores in by_level.items():
                    out[f"f1_{lvl_name}"] = float(np.mean(scores)) if scores else 0.0
                    out[f"f1_{lvl_name}_std"] = _std(scores)
                    out[f"count_{lvl_name}"] = len(scores)
                if hotpot_eval_placements is not None:
                    for bname, scores in by_placement.items():
                        out[f"f1_{bname}"] = float(np.mean(scores)) if scores else 0.0
                        out[f"count_{bname}"] = len(scores)
                _write_eval_records(records)
                return out
            if IS_PASSKEY:
                # -------------------------------------------------------
                # Passkey retrieval evaluation: Exact Match per sample.
                # Labels contain the passkey token ids at the key position;
                # everything else is -100. We compare causal-shifted preds.
                # -------------------------------------------------------
                preds_np  = _to_numpy(preds)
                labels_np = _to_numpy(labels)
                labels_shift = labels_np[:, 1:]
                preds_shift  = preds_np[:, :-1]
                em_list = []
                for _pki in range(labels_shift.shape[0]):
                    _m = labels_shift[_pki] != -100
                    if _m.sum() == 0:
                        continue
                    _ref_ids  = labels_shift[_pki][_m].tolist()
                    _pred_ids = preds_shift[_pki][_m].tolist()
                    _ref_text  = tokenizer.decode(_ref_ids,  skip_special_tokens=True).strip()
                    _pred_text = tokenizer.decode(_pred_ids, skip_special_tokens=True).strip()
                    em_list.append(float(_ref_text == _pred_text))
                _pk_em = float(np.mean(em_list)) if em_list else 0.0
                logger.info("[passkey] block_size=%d  EM=%.4f  N=%d", data_args.block_size, _pk_em, len(em_list))
                return {"passkey_em": _pk_em, "passkey_count": len(em_list)}
            if not IS_XSUM:
                if is_synth:
                    # to numpy
                    if hasattr(preds, "detach"):
                        preds = preds.detach().cpu().numpy()
                    if hasattr(labels, "detach"):
                        labels = labels.detach().cpu().numpy()

                    # causal align
                    labels_shift = labels[:, 1:]
                    preds_shift  = preds[:, :-1]

                    m = (labels_shift != -100)
                    denom = int(m.sum())
                    if denom == 0:
                        return {"accuracy": 0.0, "em": 0.0, "count": 0}

                    acc = float((preds_shift[m] == labels_shift[m]).mean())

                    # exact match per sample (only masked positions)
                    em_list = []
                    B = labels_shift.shape[0]
                    for i in range(B):
                        mi = m[i]
                        if mi.sum() == 0:
                            continue
                        em_list.append(float((preds_shift[i][mi] == labels_shift[i][mi]).all()))
                    em = float(sum(em_list) / len(em_list)) if em_list else 0.0

                    return {"accuracy": acc, "em": em, "count": denom}                    
                labels = labels[:, 1:].reshape(-1)
                preds = preds[:, :-1].reshape(-1)
                return metric.compute(predictions=preds, references=labels)
                # XSum：只评 summary（label != -100）的片段，计算 ROUGE 与 BLEU1-4
                # 先做 causal 的对齐（和你原来一致）
            labels_shift = labels[:, 1:]
            preds_shift  = preds[:, :-1]
            pred_texts, ref_texts, source_texts = [], [], []
            for i in range(labels_shift.shape[0]):
                mask = labels_shift[i] != -100
                if mask.sum() == 0:
                    continue
                ref_ids  = labels_shift[i][mask].tolist()
                pred_ids = preds_shift[i][mask].tolist()
                ref_texts.append(tokenizer.decode(ref_ids,  skip_special_tokens=True))
                pred_texts.append(tokenizer.decode(pred_ids, skip_special_tokens=True))
                if xsum_eval_sources is not None and i < len(xsum_eval_sources):
                    source_texts.append(xsum_eval_sources[i])
            # 仅计算 ROUGE（rouge1/2/L）
            rouge = rouge_metric.compute(predictions=pred_texts, references=ref_texts, use_stemmer=True)
            if training_args.do_train:
                if len(pred_texts) == 0:
                    return {
                        "rouge1": 0.0,
                        "rouge2": 0.0,
                        "rougeL": 0.0,
                        "bertscore": 0.0,
                        "count": 0,
                    }
                bertscore = bertscore_metric.compute(predictions=pred_texts, references=ref_texts, lang="en")
                out = {
                    "rouge1": float(rouge.get("rouge1", 0.0)),
                    "rouge2": float(rouge.get("rouge2", 0.0)),
                    "rougeL": float(rouge.get("rougeL", 0.0)),
                    "bertscore": float(np.mean(bertscore['f1'])),
                    "count":  len(pred_texts),
                }
            else:
                if len(pred_texts) == 0:
                    return {
                        "rouge1": 0.0,
                        "rouge2": 0.0,
                        "rougeL": 0.0,
                        "bertscore": 0.0,
                        "bleu": 0.0,
                        "sacrebleu": 0.0,
                        "meteor": 0.0,
                        "chrf": 0.0,
                        "gen_len": 0.0,
                        "ref_len": 0.0,
                        "compression_ratio": 0.0,
                        "empty_pred_ratio": 0.0,
                        "count": 0,
                    }

                # ---------------------------
                # reference-based metrics
                # ---------------------------
                rouge = rouge_metric.compute(
                    predictions=pred_texts,
                    references=ref_texts,
                    use_stemmer=True
                )

                bertscore = bertscore_metric.compute(
                    predictions=pred_texts,
                    references=ref_texts,
                    lang="en"
                )
                bert_f1 = np.asarray(bertscore["f1"], dtype=np.float32)

                bleu = bleu_metric.compute(
                    predictions=pred_texts,
                    references=[[r] for r in ref_texts],
                )

                sacrebleu = sacrebleu_metric.compute(
                    predictions=pred_texts,
                    references=[[r] for r in ref_texts],
                )

                meteor = meteor_metric.compute(
                    predictions=pred_texts,
                    references=ref_texts,
                )

                chrf = chrf_metric.compute(
                    predictions=pred_texts,
                    references=ref_texts,
                )

                # ---------------------------
                # length / sanity metrics
                # ---------------------------
                gen_lens = [
                    len(tokenizer.encode(t, add_special_tokens=False))
                    for t in pred_texts
                ]
                ref_lens = [
                    len(tokenizer.encode(t, add_special_tokens=False))
                    for t in ref_texts
                ]
                empty_pred_ratio = float(sum([1 for t in pred_texts if len(t.strip()) == 0])) / max(len(pred_texts), 1)

                out = {
                    "rouge1": float(rouge.get("rouge1", 0.0)),
                    "rouge2": float(rouge.get("rouge2", 0.0)),
                    "rougeL": float(rouge.get("rougeL", 0.0)),
                    "bertscore": float(bert_f1.mean()),
                    "bertscore_std": _safe_std(bert_f1.tolist()),

                    "bleu": float(bleu.get("bleu", 0.0)),
                    "sacrebleu": float(sacrebleu.get("score", 0.0)),
                    "meteor": float(meteor.get("meteor", 0.0)),
                    "chrf": float(chrf.get("score", 0.0)),

                    "gen_len": _safe_mean(gen_lens),
                    "ref_len": _safe_mean(ref_lens),
                    "compression_ratio": (
                        float(np.mean([g / max(r, 1) for g, r in zip(gen_lens, ref_lens)]))
                        if len(gen_lens) else 0.0
                    ),
                    "empty_pred_ratio": empty_pred_ratio,
                    "count": len(pred_texts),
                }

                # ---------------------------
                # source-grounded metrics
                # ---------------------------
                _local_rank = int(os.environ.get("LOCAL_RANK", training_args.local_rank if hasattr(training_args, "local_rank") else -1))
                _is_main = (_local_rank <= 0)
                if len(source_texts) == len(pred_texts) and len(source_texts) > 0 and _is_main:
                    # ---------------------------
                    # SummaC (recommended primary faithful metric)
                    # ---------------------------
                    try:
                        summac_scorer = None if os.environ.get("SKIP_SUMMAC", "0") == "1" else _get_summac_scorer()
                        if summac_scorer is not None:
                            # 常见接口：score(list_of_docs, list_of_summaries)
                            # 有些版本返回 dict，有些返回 list，下面做兼容
                            summac_results = summac_scorer.score(source_texts, pred_texts)

                            summac_scores = []
                            if isinstance(summac_results, dict):
                                # 常见字段可能是 "scores"
                                if "scores" in summac_results:
                                    summac_scores = [float(x) for x in summac_results["scores"]]
                            elif isinstance(summac_results, list):
                                summac_scores = [float(x) for x in summac_results]

                            if len(summac_scores):
                                out["summac"] = float(np.mean(summac_scores))
                                out["summac_std"] = _safe_std(summac_scores)
                                out["_summac_per_example"] = summac_scores

                            # ref_summac: reference summaries vs source (sanity baseline)
                            try:
                                ref_summac_results = summac_scorer.score(source_texts, ref_texts)
                                ref_summac_scores = []
                                if isinstance(ref_summac_results, dict) and "scores" in ref_summac_results:
                                    ref_summac_scores = [float(x) for x in ref_summac_results["scores"]]
                                elif isinstance(ref_summac_results, list):
                                    ref_summac_scores = [float(x) for x in ref_summac_results]
                                if ref_summac_scores:
                                    out["ref_summac"] = float(np.mean(ref_summac_scores))
                                    out["ref_summac_std"] = _safe_std(ref_summac_scores)
                                    out["_ref_summac_per_example_tmp"] = ref_summac_scores
                            except Exception as _re:
                                logger.warning(f"[SummaC] ref_summac failed: {_re}")
                    except Exception as e:
                        logger.warning(f"[SummaC] compute failed: {e}")

                    # ---------------------------
                    # AlignScore
                    # ---------------------------
                    try:
                        alignscore_scorer = _get_alignscore_scorer()
                        if alignscore_scorer is not None:
                            align_scores = alignscore_scorer.score(
                                contexts=source_texts,
                                claims=pred_texts
                            )
                            align_scores = np.asarray(align_scores, dtype=np.float32)
                            out["alignscore"] = float(align_scores.mean())
                            out["alignscore_std"] = _safe_std(align_scores.tolist())
                    except Exception as e:
                        logger.warning(f"[AlignScore] compute failed: {e}")

                    # ---------------------------
                    # FENICE
                    # ---------------------------
                    try:
                        fenice_scorer = None if os.environ.get("SKIP_FENICE", "0") == "1" else _get_fenice_scorer()
                        if fenice_scorer is not None:
                            batch = [
                                {"document": src, "summary": pred}
                                for src, pred in zip(source_texts, pred_texts)
                            ]
                            fenice_results = fenice_scorer.score_batch(batch)
                            fenice_scores = []
                            for item in fenice_results:
                                if isinstance(item, dict) and ("score" in item):
                                    fenice_scores.append(float(item["score"]))
                            if len(fenice_scores):
                                out["fenice"] = float(np.mean(fenice_scores))
                                out["fenice_std"] = _safe_std(fenice_scores)
                                out["_fenice_per_example"] = fenice_scores
                    except Exception as e:
                        logger.warning(f"[FENICE] compute failed: {e}")

                    # --- write per-example faithfulness scores to disk ---
                    _pe_summac = out.pop("_summac_per_example", None)
                    _pe_fenice = out.pop("_fenice_per_example", None)
                    _pe_ref_summac = out.pop("_ref_summac_per_example_tmp", None)
                    if _pe_summac is not None or _pe_fenice is not None or _pe_ref_summac is not None:
                        try:
                            import os as _os
                            _pe_path = _os.path.join(training_args.output_dir, "faithfulness_per_example.json")
                            _pe_data = {}
                            if _pe_summac is not None:
                                _pe_data["summac"] = _pe_summac
                            if _pe_fenice is not None:
                                _pe_data["fenice"] = _pe_fenice
                            if _pe_ref_summac is not None:
                                _pe_data["ref_summac"] = _pe_ref_summac
                            with open(_pe_path, "w") as _f:
                                import json as _json
                                _json.dump(_pe_data, _f)
                            logger.info(f"[faithfulness] per-example scores saved to {_pe_path}")
                        except Exception as _e:
                            logger.warning(f"[faithfulness] failed to save per-example scores: {_e}")
                else:
                    logger.info("[XSUM] source_texts unavailable or length mismatch, skip source-grounded metrics")

                logger.info(
                    "[XSUM_METRICS] "
                    f"count={out['count']} "
                    f"rouge1={out['rouge1']:.4f} rouge2={out['rouge2']:.4f} rougeL={out['rougeL']:.4f} "
                    f"bertscore={out['bertscore']:.4f} "
                    f"bleu={out['bleu']:.4f} sacrebleu={out['sacrebleu']:.4f} "
                    f"meteor={out['meteor']:.4f} chrf={out['chrf']:.4f} "
                    f"gen_len={out['gen_len']:.2f} ref_len={out['ref_len']:.2f} "
                    f"compression_ratio={out['compression_ratio']:.4f} "
                    f"empty_pred_ratio={out['empty_pred_ratio']:.4f} "
                    f"summac={out.get('summac', float('nan')):.4f} "
                    f"ref_summac={out.get('ref_summac', float('nan')):.4f} "
                    f"alignscore={out.get('alignscore', float('nan')):.4f} "
                    f"fenice={out.get('fenice', float('nan')):.4f}"
                )
            return out
    # Initialize our Trainer

    enable_path_debug_eval = cfg_bool(cfg, "path_debug_eval_enable", False)
    heatmap_or_mech_enabled = bool(getattr(config, "eval_attn_heatmap_enabled", False)) or bool(
        getattr(config, "eval_attn_mech_enabled", False)
    )
    if heatmap_or_mech_enabled and (not enable_path_debug_eval):
        enable_path_debug_eval = True
        logger.info(
            "[PathDebugEval] auto-enabled because eval_attn_heatmap_enabled=%s eval_attn_mech_enabled=%s",
            str(bool(getattr(config, "eval_attn_heatmap_enabled", False))),
            str(bool(getattr(config, "eval_attn_mech_enabled", False))),
        )
    if train_eval_disable_expensive_stats:
        enable_path_debug_eval = False
    callbacks = [
        # LrMonitorCallback(group_names_expect=("main_decay","main_nodecay","B_decay","B_nodecay")),
        # ParamTrackerCallback(['phi_raw', 'omega_raw'], training_args.logging_steps),
    ]
    wavelet_mode_name = str(getattr(config, "wavelet_mode", "")).strip().lower()
    if wavelet_mode_name in ("logit_bias_ctxscale_shift_v0", "logit_bias_ctxscale_shift_v0_film"):
        gate_sync_every = max(1, int(getattr(training_args, "logging_steps", 50)))
        callbacks.append(WaveletGateStepSyncCallback(log_every_steps=gate_sync_every))
        logger.info("[WaveletGateStepSync] enabled (log_every_steps=%d)", int(gate_sync_every))
    if enable_path_debug_eval:
        callbacks.append(PathDebugEvalCallback())
        logger.info("[PathDebugEval] enabled")
    else:
        logger.info("[PathDebugEval] disabled")
    eval_progress_log_every = cfg_int(cfg, "eval_progress_log_every", 0)
    if eval_progress_log_every > 0:
        callbacks.append(EvalProgressCallback(log_every_batches=eval_progress_log_every))
        logger.info("[EvalProgress] enabled: log_every_batches=%d", int(eval_progress_log_every))

    if getattr(training_args, 'early_stopping_patience', 0) > 0:
        callbacks.append(EarlyStoppingCallback(
            early_stopping_patience=training_args.early_stopping_patience,
        ))
        logger.info("[EarlyStopping] enabled: patience=%d", training_args.early_stopping_patience)

    rel_stats_callback = None
    rel_log_enabled = bool(getattr(config, "log_rel_stats", bool(training_args.log_rel_stats)))
    if rel_log_enabled:
        rel_stats_callback = RelStatsCallback()
        callbacks.append(rel_stats_callback)
    # 数据收集开关优先生效；否则仅 do_eval-only 时启用 router dump，避免训练阶段中途退出
    data_collection_style = str(training_args.data_collection_style).strip() if training_args.data_collection_style is not None else ""
    if train_eval_disable_expensive_stats:
        data_collection_style = ""
    if data_collection_style:
        router_path = Path(config.model_name_or_path)
        cfg = RouterPTDumpOnceConfig(
            dump_dir=f"./router_pt_dump",
            file_prefix=f"{config.dataset_name}_router_{config.data_collection_style}_{router_path.parent.name}_{router_path.name}",
            sampled_data_number=100,
            only_label_tokens=True,    # 需要只保留 label 位置就改成 True
            exit_after_dump=True,       # dump 完立刻退出程序
            block_size=config.block_size,
            
        )
        print(f'{config.model_name_or_path} Trainer init...')

        callbacks.append(SaveRouterPTOnceCallback(cfg))
        logger.warning(
            "[SaveRouterPTOnceCallback] enabled because data_collection_style='%s' (this adds an extra eval-pass and may look like eval is hanging).",
            data_collection_style,
        )

    if training_args.do_eval and (not train_eval_disable_expensive_stats) and bool(training_args.eval_router_heatmap_enable):
        heatmap_cfg = RouterPosBinHeatmapConfig(
            enable=True,
            bin_size=max(1, int(training_args.eval_router_heatmap_bin_size)),
            out_subdir=str(training_args.eval_router_heatmap_out_subdir),
            max_batches=(
                None
                if int(training_args.eval_router_heatmap_max_batches) <= 0
                else int(training_args.eval_router_heatmap_max_batches)
            ),
        )
        callbacks.append(RouterPosBinHeatmapCallback(heatmap_cfg))
        logger.info(
            "[RouterHeatmap] enabled: bin_size=%d out_subdir=%s max_batches=%s",
            int(heatmap_cfg.bin_size),
            str(heatmap_cfg.out_subdir),
            "all" if heatmap_cfg.max_batches is None else str(int(heatmap_cfg.max_batches)),
        )

    if training_args.do_eval and data_args.dataset_name == "mix":
        if train_eval_disable_expensive_stats:
            # compute_metrics is disabled in fast eval mode; keep checkpoint selection consistent.
            training_args.metric_for_best_model = "eval_loss"
            training_args.greater_is_better = False
            if training_args.load_best_model_at_end:
                logger.warning(
                    "[EvalFast] do_train=True: metric_for_best_model forced to eval_loss because decode metrics are disabled."
                )
            else:
                logger.info("[EvalFast] mix best-metric override: metric_for_best_model=eval_loss")
        else:
            if training_args.metric_for_best_model is None:
                training_args.metric_for_best_model = "hotpot_f1"
            if training_args.greater_is_better is None:
                training_args.greater_is_better = True
            if not training_args.load_best_model_at_end:
                logger.warning(
                    "[mix] metric_for_best_model is set to hotpot_f1 but load_best_model_at_end=False; "
                    "best checkpoint will not be auto-loaded."
                )

    use_eval_compute_metrics = bool(training_args.do_eval and (not is_torch_xla_available()) and (not train_eval_disable_expensive_stats))
    if train_eval_disable_expensive_stats:
        logger.info("[EvalFast] compute_metrics disabled; eval will report eval_loss/perplexity only.")

    # CanonicalConfigCallback: rewrite config.json in every checkpoint dir with the
    # fully-resolved runtime config (cfg_path overrides applied).  Prevents the
    # persistent bug where config.json keeps the base-checkpoint value (e.g.
    # wavelet_mode='off') while the actual training used a different value from
    # supply_model.cfg.  Without this, all intervention / ablation scripts that
    # read wavelet_mode from config.json get the wrong value and must patch it
    # manually.
    class CanonicalConfigCallback(TrainerCallback):
        def __init__(self, resolved_config):
            self._config = resolved_config

        def on_save(self, args, state, control, **kwargs):
            if not state.is_world_process_zero:
                return
            ckpt_dir = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
            if os.path.isdir(ckpt_dir):
                self._config.save_pretrained(ckpt_dir)
                logger.info("[CanonicalConfig] Rewrote config.json in %s", ckpt_dir)

    callbacks.append(CanonicalConfigCallback(config))
    # Also write canonical config to output_dir immediately (covers eval-only runs
    # and the case where training fails before the first checkpoint).
    if training_args.process_index == 0:
        os.makedirs(training_args.output_dir, exist_ok=True)
        config.save_pretrained(training_args.output_dir)
        logger.info("[CanonicalConfig] Wrote canonical config.json to %s", training_args.output_dir)

    callbacks_for_trainer = callbacks if len(callbacks) > 0 else None

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        processing_class=tokenizer,
        # Data collator will default to DataCollatorWithPadding, so we change it.
        data_collator=default_data_collator,
        compute_metrics=compute_metrics if use_eval_compute_metrics else None,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics if use_eval_compute_metrics else None,
        callbacks=callbacks_for_trainer,
    )
    if rel_stats_callback is not None:
        rel_stats_callback.set_trainer(trainer)

    if training_args.do_train:
        train_dataloader = trainer.get_train_dataloader()
        _, num_update_steps_per_epoch, _, _, epoch_based, len_dataloader, max_steps = trainer.set_initial_training_values(
            training_args,
            train_dataloader,
            training_args.train_batch_size,
        )
        logger.info(
            "[TrainSetup] len_dataloader=%s num_update_steps_per_epoch=%s epoch_based=%s max_steps=%s resume_from=%s",
            len_dataloader,
            num_update_steps_per_epoch,
            epoch_based,
            max_steps,
            training_args.resume_from_checkpoint,
        )

    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()  # Saves the tokenizer too for easy upload

        metrics = train_result.metrics

        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        )
        if data_args.streaming:
            metrics["train_samples"] = max_train_samples
        else:
            metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        run_eval_output_js_probe_if_enabled(trainer)
        try:
            eval_len = len(eval_dataset)
            world_size = 1
            if dist.is_available() and dist.is_initialized():
                world_size = max(1, int(dist.get_world_size()))
            global_eval_bs = max(1, int(training_args.per_device_eval_batch_size)) * int(world_size)
            est_eval_batches = int(math.ceil(float(eval_len) / float(global_eval_bs)))
            logger.info(
                "[EvalSetup] eval_examples=%d per_device_eval_batch_size=%d world_size=%d global_eval_batch_size=%d est_eval_batches=%d",
                int(eval_len),
                int(training_args.per_device_eval_batch_size),
                int(world_size),
                int(global_eval_bs),
                int(est_eval_batches),
            )
        except Exception:
            pass

        metrics = trainer.evaluate()

        max_eval_samples = data_args.max_eval_samples if data_args.max_eval_samples is not None else len(eval_dataset)
        if data_args.streaming:
            metrics["eval_samples"] = max_eval_samples
        else:
            metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))

        if "eval_loss" in metrics:
            try:
                perplexity = math.exp(metrics["eval_loss"])
            except OverflowError:
                perplexity = float("inf")
            metrics["perplexity"] = perplexity
        # else:
            # HotpotQA logits eval（或任何没有 labels 的 eval）会走到这里

            # metrics["perplexity"] = None

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)
        if data_args.dataset_name in {"hotpot_qa", "mix"} and (not train_eval_disable_expensive_stats):
            run_paired_analysis_if_ready()

    kwargs = {"finetuned_from": model_args.model_name_or_path, "tasks": "text-generation"}
    if data_args.dataset_name is not None:
        kwargs["dataset_tags"] = data_args.dataset_name
        if data_args.dataset_config_name is not None:
            kwargs["dataset_args"] = data_args.dataset_config_name
            kwargs["dataset"] = f"{data_args.dataset_name} {data_args.dataset_config_name}"
        else:
            kwargs["dataset"] = data_args.dataset_name

    if training_args.push_to_hub:
        trainer.push_to_hub(**kwargs)
    else:
        trainer.create_model_card(**kwargs)


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
