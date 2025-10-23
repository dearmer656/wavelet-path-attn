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

import datasets
import evaluate
import torch
from datasets import IterableDataset, IterableDatasetDict, load_dataset, DatasetDict, concatenate_datasets, load_from_disk

import transformers
from transformers import (
    CONFIG_MAPPING,
    MODEL_FOR_CAUSAL_LM_MAPPING,
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    default_data_collator,
    is_torch_xla_available,
    set_seed,
)
from transformers.testing_utils import CaptureLogger
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version, send_example_telemetry
from transformers.utils.versions import require_version

import json, hashlib
from pathlib import Path 

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
    attn_implementation: str = field(
        default="eager",
        metadata={
            "help": "Attention backend to use: eager | sdpa | flash_attention_2 | path_attn.",
            "choices": ["eager", "sdpa", "flash_attention_2", "path_attn", "path_attn_wfreq"],
        },
    )
    pe_method: str = field(
        default="vanilla",
        metadata={
            "help": "Positional encoding method to use: vanilla | rotary.",
            "choices": ["vanilla", "rotary"],
        },
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
    num_harmonics: int = field(
        default=2,
        metadata={"help": "number of harmonics"},
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
        default="additive",
        metadata={"help": "Wavelet mode to use: additive | softmix."},
    )
    use_soft_wavelet_fox: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether use wavelet freq into beta (only for path_attn_wfreq)."},
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
                assert extension in ["csv", "json", "txt"], "`train_file` should be a csv, a json or a txt file."
            if self.validation_file is not None:
                extension = self.validation_file.split(".")[-1]
                assert extension in ["csv", "json", "txt"], "`validation_file` should be a csv, a json or a txt file."
@dataclass
class SupplyTrainingArguments(TrainingArguments):
    b_unfreeze_step: int = field(
        default=5000,
        metadata={
            "help": (
                "The wB doesn't update until unfreeze_step"
            )
        },
    )

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
    if data_args.dataset_name is not None:
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
    config.attn_implementation = model_args.attn_implementation
    config.use_forget_gate = model_args.use_forget_gate
    config.path_use_qk_norm = model_args.path_use_qk_norm
    config.path_use_low_rank_w = model_args.path_use_low_rank_w
    config.path_use_w_shortconv = model_args.path_use_w_shortconv
    config.path_conv_size = model_args.path_conv_size
    config.path_conv_bias = model_args.path_conv_bias
    config.num_harmonics = model_args.num_harmonics
    config.share_freq_across_heads = model_args.share_freq_across_heads
    config.pe_method = model_args.pe_method
    config.use_beta_modulation = model_args.use_beta_modulation
    config.wavelet_mode = model_args.wavelet_mode
    config.use_soft_wavelet_fox = model_args.use_soft_wavelet_fox
    config.logging_steps = training_args.logging_steps
    config.wavelet_baseline_use = model_args.wavelet_baseline_use
    config.init_theta = model_args.init_theta    
    if torch.cuda.current_device() == 0:
        init_wandb(config, 'gpt2 with path attn')        

    # We resize the embeddings only when necessary to avoid index errors. If you are creating a model from scratch
    # on a small vocab and want a smaller embedding size, remove this test.

    # Preprocessing the datasets.
    # First we tokenize all the texts.
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
        if data_args.block_size > tokenizer.model_max_length:
            logger.warning(
                f"The block_size passed ({data_args.block_size}) is larger than the maximum length for the model "
                f"({tokenizer.model_max_length}). Using block_size={tokenizer.model_max_length}."
            )
        # block_size = min(data_args.block_size, tokenizer.model_max_length)
        block_size = data_args.block_size
    config.block_size = block_size
    config.rope_theta = 10000
    if model_args.model_name_or_path:
        dtype = model_args.dtype if model_args.dtype in ["auto", None] else getattr(torch, model_args.dtype)
        model = AutoModelForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            token=model_args.token,
            trust_remote_code=model_args.trust_remote_code,
            dtype=dtype,
        )
    else:
        model = AutoModelForCausalLM.from_config(config, trust_remote_code=model_args.trust_remote_code)
        n_params = sum({p.data_ptr(): p.numel() for p in model.parameters()}.values())
        logger.info(f"Training new model from scratch - Total size={n_params / 2**20:.2f}M params")    
    embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        model.resize_token_embeddings(len(tokenizer))
    if data_args.dataset_name != 'xsum':
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
    else:
                PROMPT_TPL     = "Summarize the following document:\n{doc}\n\nSummary:"
                MAX_TRAIN_DOC  = 512
                BUCKET_STEP    = 512
                UPPER          = 8192
                def _lm_cache_fingerprint():
                    key = {
                        "dataset": "xsum_prefixlm_v1",
                        "tok_name": getattr(tokenizer, "name_or_path", None),
                        "block_size": data_args.block_size,
                        "eos_id": tokenizer.eos_token_id,
                        "pad_id": tokenizer.pad_token_id,
                        # 如需更严谨，可把数据源版本号/筛选规则也放进来
                    }
                    return hashlib.sha1(json.dumps(key, sort_keys=True).encode("utf-8")).hexdigest()[:12]
                LM_CACHE_ROOT = Path(
                    getattr(data_args, "cache_dir", None) or os.path.expanduser("~/.cache/xsum_prefixlm")
                ) / "lm_datasets_cache"
                LM_CACHE_ROOT.mkdir(parents=True, exist_ok=True)
                LM_CACHE_DIR = LM_CACHE_ROOT / f"xsum_prefixlm_{_lm_cache_fingerprint()}"
                # pdb.set_trace()
                if LM_CACHE_DIR.exists():
                    lm_datasets = load_from_disk(str(LM_CACHE_DIR))

                    # 先丢弃 discard 行（如果你的 map 里可能返回了 discard 标记）
                    for split in list(lm_datasets.keys()):
                        lm_datasets[split] = lm_datasets[split].filter(lambda ex: not ex.get("discard", False))

                    if training_args.do_train:
                        train_hi = int(512)  # 或 int(MAX_TRAIN_DOC)
                        lm_datasets["train"] = lm_datasets["train"].filter(lambda ex: ex["hi"] == train_hi)

                    if training_args.do_eval:
                        val_hi = int(data_args.block_size)
                        lm_datasets["validation"] = lm_datasets["validation"].filter(lambda ex: ex["hi"] == val_hi)

                    # 只保留需要的列并固定返回 dict
                    need_cols = ["input_ids", "attention_mask", "labels"]
                    for split in ("train", "validation"):
                        if split in lm_datasets:
                            cols = [c for c in need_cols if c in lm_datasets[split].column_names]
                            lm_datasets[split].set_format(type="python", columns=cols)

                    # 自检（可留可删）
                    for split in ("train", "validation"):
                        if split in lm_datasets:
                            assert len(lm_datasets[split]) > 0, f"{split} bucket empty"
                            x = lm_datasets[split][0]
                            assert isinstance(x, dict) and all(k in x for k in ["input_ids","attention_mask","labels"])

                # ====== 无缓存 => 执行你原有逻辑并缓存 ======
                else:
                    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
                        tokenizer.pad_token_id = tokenizer.eos_token_id
                    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
                        tokenizer.pad_token_id = tokenizer.eos_token_id

                    assert not data_args.streaming, "XSUM prefix-LM 分支暂不支持 streaming=True"

                    from collections import defaultdict
                    IDX_BY_HI_ORIGIN = {"train": defaultdict(list), "validation": defaultdict(list)}
                    IDX_BY_HI_ORIGIN_FROM_TRAIN = {"validation": defaultdict(list)}
                    xsum_proc = {}
                    def build_and_index_factory(split_name):
                        def build_and_index(ex, idx):
                            enc_prompt = tokenizer(PROMPT_TPL.format(doc=ex["document"]), add_special_tokens=True, truncation=False)
                            enc_summ   = tokenizer(ex["summary"], add_special_tokens=False, truncation=False)

                            ids    = enc_prompt["input_ids"] + enc_summ["input_ids"] + [tokenizer.eos_token_id]
                            labels = ([-100] * len(enc_prompt["input_ids"])) + enc_summ["input_ids"] + [tokenizer.eos_token_id]
                            L = len(ids)
                            if L <= 1 or len(enc_summ["input_ids"]) == 0:
                                return {"discard": True}

                            if L <= MAX_TRAIN_DOC:
                                hi = MAX_TRAIN_DOC
                            else:
                                hi = ((L - 1) // BUCKET_STEP + 1) * BUCKET_STEP
                                hi = min(hi, UPPER)
                            pad_len = hi - L
                            if pad_len > 0:
                                ids    = ids    + [tokenizer.eos_token_id] * pad_len
                                labels = labels + [-100] * pad_len
                                attn   = [1] * L + [0] * pad_len
                            else:
                                attn   = [1] * hi

                            if L > MAX_TRAIN_DOC and split_name == 'train':
                                IDX_BY_HI_ORIGIN_FROM_TRAIN['validation'][hi].append(idx)
                            else:
                                IDX_BY_HI_ORIGIN[split_name][hi].append(idx)

                            return {
                                "input_ids": ids,
                                "labels": labels,
                                "attention_mask": attn,
                                "total_token_len": L,
                                "hi": hi,
                            }
                        return build_and_index
                    # —— 一次 map：构造 + pad 到桶 hi + 建索引 —— #
                    if training_args.do_train:
                        raw_datasets['train'] = raw_datasets['train']
                    raw_datasets['validation'] = raw_datasets['validation']
                    raw_datasets['test'] = raw_datasets['test']

                    with training_args.main_process_first(desc="XSUM build+pad+index (by hi)"):
                        for split in raw_datasets.keys():
                            if split not in ("train", "validation"):
                                continue
                            # if not training_args.do_train and split == 'train':
                                # continue
                            xsum_proc[split] = raw_datasets[split].map(
                                build_and_index_factory(split),
                                with_indices=True,
                                remove_columns=column_names,
                                num_proc=1,
                                desc=f"Tokenize & bucket (split={split})",
                                load_from_cache_file=False,
                            )
                    lm_datasets = DatasetDict()
                    if training_args.do_train:
                        lm_datasets["train"] = xsum_proc["train"].select(IDX_BY_HI_ORIGIN["train"][MAX_TRAIN_DOC])
                    if training_args.do_eval:
                        val_from_train = xsum_proc["train"].select(
                            IDX_BY_HI_ORIGIN_FROM_TRAIN["validation"][data_args.block_size]
                        )
                        val_from_val = xsum_proc["validation"].select(
                            IDX_BY_HI_ORIGIN["validation"][data_args.block_size]
                        )                        
                        lm_datasets["validation"] = concatenate_datasets([val_from_train, val_from_val])
                        # pdb.set_trace()
                    # —— 缓存到磁盘（仅主进程写，其他进程等待） —— #
                    # pdb.set_trace()
                    with training_args.main_process_first(desc="save lm_datasets to disk"):
                        # 只有主进程实际写盘；main_process_first 会屏障其它进程
                        if getattr(training_args, "process_index", 0) == 0:
                            lm_datasets.save_to_disk(str(LM_CACHE_DIR))
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

    if training_args.do_eval:
        # if "validation" not in tokenized_datasets:
        #     raise ValueError("--do_eval requires a validation dataset")
        eval_dataset = lm_datasets["validation"]
        if data_args.max_eval_samples is not None:
            if data_args.streaming:
                eval_dataset = eval_dataset.take(data_args.max_eval_samples)
            else:
                max_eval_samples = min(len(eval_dataset), data_args.max_eval_samples)
                eval_dataset = eval_dataset.select(range(max_eval_samples))

        def preprocess_logits_for_metrics(logits, labels):
            if isinstance(logits, tuple):
                # Depending on the model and config, logits may contain extra tensors,
                # like past_key_values, but logits always come first
                logits = logits[0]
            return logits.argmax(dim=-1)

        metric = evaluate.load("accuracy", cache_dir=model_args.cache_dir)
        IS_XSUM = (data_args.dataset_name == "xsum")
        rouge_metric = evaluate.load("rouge") if IS_XSUM else None
        bleu_metric  = evaluate.load("bleu")  if IS_XSUM else None
        bertscore_metric  = evaluate.load("bertscore")  if IS_XSUM else None
        def compute_metrics(eval_preds):
            preds, labels = eval_preds
            # preds have the same shape as the labels, after the argmax(-1) has been calculated
            # by preprocess_logits_for_metrics but we need to shift the labels
            if not IS_XSUM:
                labels = labels[:, 1:].reshape(-1)
                preds = preds[:, :-1].reshape(-1)
                return metric.compute(predictions=preds, references=labels)
                # XSum：只评 summary（label != -100）的片段，计算 ROUGE 与 BLEU1-4
                # 先做 causal 的对齐（和你原来一致）
            labels_shift = labels[:, 1:]
            preds_shift  = preds[:, :-1]
            pred_texts, ref_texts = [], []
            for i in range(labels_shift.shape[0]):
                mask = labels_shift[i] != -100
                if mask.sum() == 0:
                    continue
                ref_ids  = labels_shift[i][mask].tolist()
                pred_ids = preds_shift[i][mask].tolist()
                ref_texts.append(tokenizer.decode(ref_ids,  skip_special_tokens=True))
                pred_texts.append(tokenizer.decode(pred_ids, skip_special_tokens=True))
                # print(ref_texts)
                # print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!boundary,boundary,boundary,boundary!!!!!!!!!!!!!!!!!!!!!!')
                # print(pred_texts)
                # pdb.set_trace()
            # 计算 ROUGE（rouge1/2/L）与 BLEU（含 precisions -> BLEU1-4）
            rouge = rouge_metric.compute(predictions=pred_texts, references=ref_texts, use_stemmer=True)
            bleu  = bleu_metric.compute(predictions=pred_texts, references=[[r] for r in ref_texts])
            if not training_args.do_train:
                bertscore = bertscore_metric.compute(predictions=pred_texts, references=ref_texts, lang="en")
                out = {
                    "rouge1": float(rouge.get("rouge1", 0.0)),
                    "rouge2": float(rouge.get("rouge2", 0.0)),
                    "rougeL": float(rouge.get("rougeL", 0.0)),
                    "bleu":   float(bleu.get("bleu", 0.0)),
                    "bertscore": float(np.mean(bertscore['f1'])),
                    "count":  len(pred_texts),
                }
            else:
                out = {
                    "rouge1": float(rouge.get("rouge1", 0.0)),
                    "rouge2": float(rouge.get("rouge2", 0.0)),
                    "rougeL": float(rouge.get("rougeL", 0.0)),
                    "bleu":   float(bleu.get("bleu", 0.0)),
                    "count":  len(pred_texts),
                }
            # BLEU1-4：evaluate 的 "bleu" 会返回 n-gram precisions（通常是百分数）
            if "precisions" in bleu and len(bleu["precisions"]) >= 4:
                p = bleu["precisions"]
                out.update({
                    "bleu1": float(p[0]),
                    "bleu2": float(p[1]),
                    "bleu3": float(p[2]),
                    "bleu4": float(p[3]),
                })
            # pdb.set_trace()
            # if not training_args.do_train:
            #     p_ents = [ents(p) for p in pred_texts]
            #     r_ents = [ents(r) for r in ref_texts]

            #     tp = sum(len(pe & re) for pe, re in zip(p_ents, r_ents))
            #     fp = sum(len(pe - re) for pe, re in zip(p_ents, r_ents))
            #     fn = sum(len(re - pe) for pe, re in zip(p_ents, r_ents))

            #     precision = tp / (tp + fp + 1e-9)
            #     recall    = tp / (tp + fn + 1e-9)
            #     f1        = 2 * precision * recall / (precision + recall + 1e-9)

            #     out.update({
            #         "entity_precision": float(precision),
            #         "entity_recall":    float(recall),
            #         "entity_f1":        float(f1),
            #         "entity_tp":        int(tp),
            #         "entity_fp":        int(fp),
            #         "entity_fn":        int(fn),
            #     })
            return out
    # Initialize our Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        processing_class=tokenizer,
        # Data collator will default to DataCollatorWithPadding, so we change it.
        data_collator=default_data_collator,
        compute_metrics=compute_metrics if training_args.do_eval and not is_torch_xla_available() else None,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics
        if training_args.do_eval and not is_torch_xla_available()
        else None,
        callbacks = [
            LrMonitorCallback(group_names_expect=("main_decay","main_nodecay","B_decay","B_nodecay")),
            ParamTrackerCallback(['phi_raw', 'omega_raw'], training_args.logging_steps)
        ],
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

        metrics = trainer.evaluate()

        max_eval_samples = data_args.max_eval_samples if data_args.max_eval_samples is not None else len(eval_dataset)
        if data_args.streaming:
            metrics["eval_samples"] = max_eval_samples
        else:
            metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))

        try:
            perplexity = math.exp(metrics["eval_loss"])
        except OverflowError:
            perplexity = float("inf")
        metrics["perplexity"] = perplexity

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

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
