# coding=utf-8
# Copyright 2018 The OpenAI Team Authors and HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
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
"""PyTorch OpenAI GPT-2 model."""

import math
import warnings
from dataclasses import dataclass
from typing import Callable, Optional, Union

import torch
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
import torch.nn.functional as F
import os
from ...activations import ACT2FN, get_activation
from ...cache_utils import Cache, DynamicCache, EncoderDecoderCache
from ...generation import GenerationMixin
from ...masking_utils import create_causal_mask
from ...modeling_attn_mask_utils import _prepare_4d_attention_mask_for_sdpa
from ...modeling_layers import GradientCheckpointingLayer
from ...modeling_outputs import (
    BaseModelOutputWithPastAndCrossAttentions,
    CausalLMOutputWithCrossAttentions,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutputWithPast,
    TokenClassifierOutput,
)
from ...modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from ...pytorch_utils import Conv1D, find_pruneable_heads_and_indices, prune_conv1d_layer
from ...utils import (
    ModelOutput,
    add_start_docstrings,
    auto_docstring,
    logging,
)
from ...utils.deprecation import deprecate_kwarg
from ...utils.model_parallel_utils import assert_device_map, get_device_map
from .configuration_gpt2 import GPT2Config


from einops import rearrange

try:
    from fla.layers.path_attn import PaTHAttention as _PaTHAttention
    from fla.layers.path_attn import PaTHAttentionWfreq as _PaTHAttentionWfreq
    from fla.layers.path_attn import path_ut_base_raw as _path_ut_base_raw
except Exception as _e:
    _PaTHAttention = None
    _PaTHAttentionWfreq = None
    _path_ut_base_raw = None
try:
    from fla.layers.freq_analysis_utils import *
except Exception:
    pass
# from fla.layers.path_attn import PaTHAttention as _PaTHAttention
logger = logging.get_logger(__name__)

class PWavMeanLogger:
    """One-shot logger for per-layer P_wav mean heatmaps."""

    def __init__(self, save_dir: str = "analysis/pwav_mean", cmap: str = "hot") -> None:
        self.save_dir = save_dir
        self.cmap = cmap
        self.recorded_layers: set[int] = set()
        os.makedirs(save_dir, exist_ok=True)

    def update(self, layer_idx: int, p_wav: torch.Tensor) -> None:
        # Only log once per layer to avoid spamming the disk during long runs.
        if p_wav is None:
            return
        if layer_idx in self.recorded_layers:
            return
        self.recorded_layers.add(layer_idx)

        with torch.no_grad():
            # 平均掉 batch 和 head，得到 [T,T]
            mean_map = p_wav.mean(dim=(0, 1)).detach().float().cpu()

        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(mean_map, cmap=self.cmap, aspect="auto")
        ax.set_xlabel("Key position j")
        ax.set_ylabel("Query position i")
        ax.set_title(f"P_wav mean (layer {layer_idx})")
        fig.colorbar(im, ax=ax)
        fig.tight_layout()

        fig_path = os.path.join(self.save_dir, f"layer_{layer_idx:02d}_pwav_mean.png")
        fig.savefig(fig_path)
        plt.close(fig)

        tensor_path = os.path.join(self.save_dir, f"layer_{layer_idx:02d}_pwav_mean.pt")
        torch.save(mean_map, tensor_path)

def _get_alibi_slopes(n_heads: int, device) -> torch.Tensor:
    """Standard ALiBi slopes (Press et al. 2022), correct for non-power-of-2 head counts."""
    import math
    def _slopes_power_of_2(n):
        start = 2.0 ** (-(2.0 ** -(math.log2(n) - 3)))
        return [start * (start ** i) for i in range(n)]
    if math.log2(n_heads) % 1 == 0:
        slopes = _slopes_power_of_2(n_heads)
    else:
        closest_pow2 = 2 ** math.floor(math.log2(n_heads))
        slopes = _slopes_power_of_2(closest_pow2)
        extra = _slopes_power_of_2(2 * closest_pow2)[0::2]
        slopes += extra[: n_heads - closest_pow2]
    return torch.tensor(slopes, device=device, dtype=torch.float32)

class QWABBias(nn.Module):
    """Query-conditioned Wavelet Attention Bias for standard (non-path) attention.

    Functionally equivalent to logit_bias_ctxscale_shift_v0 with wavelet_ctx_feat_mode=hidden_ln:
    routes via LN-normalized hidden states only, no path-attention dependency.
    """

    def __init__(self, config):
        super().__init__()
        embed_dim = config.hidden_size
        num_heads = config.num_attention_heads
        head_dim = embed_dim // num_heads
        K = int(getattr(config, "router_band_num", 8))
        eps = float(getattr(config, "layer_norm_epsilon", 1e-5))

        # Router: mean-pooled hidden_states → LN → Linear(head_dim, K+1)
        self.feat_ln = nn.LayerNorm(head_dim, eps=eps)
        self.router = nn.Linear(head_dim, K + 1, bias=True)  # K scales + 1 null

        # Shift: full hidden_states → LN → Linear(E, 1) → sigmoid → token shift β
        self.shift_ln = nn.LayerNorm(embed_dim, eps=eps)
        self.shift_proj = nn.Linear(embed_dim, 1, bias=True)

        # Learnable layer gate; init to -2.0 so bias starts near-zero
        self.logit_bias_a = nn.Parameter(torch.tensor(-2.0))

        # Ricker wavelet scales: 2^(scale_range[0] + step*i) for i=0..K-1
        scale_range = list(getattr(config, "scale_range", [0, 16]))
        step = (scale_range[1] - scale_range[0]) // K
        scales = torch.tensor(
            [2.0 ** (scale_range[0] + step * i) for i in range(K)], dtype=torch.float32
        )
        self.register_buffer("scales", scales)

        self._num_heads = num_heads
        self._head_dim = head_dim
        self._K = K
        self._eps = 1e-6
        self._g_max = 0.5

    def _bias_chunk(
        self,
        q0: int,
        q1: int,
        pi_scale: torch.Tensor,
        beta: torch.Tensor,
        diff: torch.Tensor,
        eps: float,
    ) -> torch.Tensor:
        """Compute QWAB bias for query rows [q0, q1)."""
        T = diff.shape[0]
        chunk_bias = pi_scale.new_zeros(pi_scale.shape[0], q1 - q0, T)
        for s_idx in range(self._K):
            s = float(self.scales[s_idx].item())
            u = (diff.view(1, 1, T) - beta[:, q0:q1].unsqueeze(-1) * s) / s  # [B, q_len, T]
            basis = (1.0 - u.pow(2)) * torch.exp(-0.5 * u.pow(2))
            basis = basis / (basis.pow(2).mean(-1, keepdim=True) + eps).sqrt()
            basis = basis.clamp(-5.0, 5.0)
            chunk_bias = chunk_bias + pi_scale[:, q0:q1, s_idx].unsqueeze(-1) * basis
        return chunk_bias

    def forward(self, hidden_states: torch.Tensor, chunk_size: int = 128) -> torch.Tensor:
        """
        Args:
            hidden_states: [B, T, embed_dim]  (pre-attention LN output)
            chunk_size: query chunk size for memory-efficient long-context eval
        Returns:
            [B, 1, T, T]  additive bias for all heads
        """
        B, T, _ = hidden_states.shape
        H, D, K, eps = self._num_heads, self._head_dim, self._K, self._eps
        hs = hidden_states.float()

        # ---- routing ----
        h_feat = hs.view(B, T, H, D).mean(dim=2)        # [B, T, D]
        h_feat = self.feat_ln(h_feat)
        rlogits = self.router(h_feat)                    # [B, T, K+1]
        rlogits = rlogits / (rlogits.pow(2).mean(-1, keepdim=True) + eps).sqrt()
        g_scales = torch.sigmoid(rlogits[..., 1:])       # [B, T, K]
        g_null   = torch.sigmoid(rlogits[..., 0:1])      # [B, T, 1]  non-null mass
        pi_scale = g_null * (g_scales / g_scales.sum(-1, keepdim=True).clamp_min(eps))  # [B, T, K]

        # ---- shift ----
        rho  = torch.sigmoid(self.shift_proj(self.shift_ln(hs)).squeeze(-1))  # [B, T]
        beta = torch.round(rho * float(T - 1)).clamp_(0.0, float(T - 1))      # [B, T]

        # ---- layer gate ----
        g_layer = self._g_max * torch.sigmoid(self.logit_bias_a.float())

        # ---- Ricker wavelet basis + weighted sum (chunked over query dim) ----
        diff = torch.arange(T, device=hidden_states.device, dtype=torch.float32)
        if T <= chunk_size:
            bias = self._bias_chunk(0, T, pi_scale, beta, diff, eps)  # [B, T, T]
        else:
            chunks = []
            for q0 in range(0, T, chunk_size):
                q1 = min(q0 + chunk_size, T)
                chunks.append(self._bias_chunk(q0, q1, pi_scale, beta, diff, eps))
            bias = torch.cat(chunks, dim=1)  # [B, T, T]

        return (g_layer * bias).unsqueeze(1).to(dtype=hidden_states.dtype)  # [B, 1, T, T]


def eager_attention_forward(module, query, key, value, attention_mask, head_mask=None, **kwargs):
    attn_weights = torch.matmul(query, key.transpose(-1, -2))

    # Wavelet relative position bias (pe_method='wavelet', relative_type='4')
    wavelet_rel_buf = kwargs.get("wavelet_relative_tensor", None)
    if wavelet_rel_buf is not None:
        q_len = query.size(-2)
        k_len = key.size(-2)
        if hasattr(module, "_get_wavelet_relative_tensor"):
            W = module._get_wavelet_relative_tensor(
                q_len=q_len,
                k_len=k_len,
                device=query.device,
                dtype=query.dtype,
                base_tensor=wavelet_rel_buf,
            )
        else:
            W = wavelet_rel_buf[:, :q_len, :k_len].to(device=query.device, dtype=query.dtype)  # [D, q_len, k_len]
        rel = torch.einsum("bhld,dln->bhln", query, W)
        attn_weights = attn_weights + rel

    if module.scale_attn_weights:
        attn_weights = attn_weights / torch.full(
            [], value.size(-1) ** 0.5, dtype=attn_weights.dtype, device=attn_weights.device
        )

    # ALiBi linear position bias (pe_method='alibi') — applied AFTER QK scaling
    if getattr(getattr(module, 'config', None), 'pe_method', None) == 'alibi':
        q_len = query.size(-2)
        k_len = key.size(-2)
        num_heads = query.size(1)
        # Standard ALiBi slopes (Press et al. 2022), handles non-power-of-2 head counts
        slopes = _get_alibi_slopes(num_heads, query.device).to(dtype=attn_weights.dtype)
        # Correct query positions for KV-cache: query occupies positions [k_len-q_len, k_len)
        q_pos = torch.arange(k_len - q_len, k_len, device=query.device, dtype=torch.float32).unsqueeze(1)
        k_pos = torch.arange(k_len, device=query.device, dtype=torch.float32).unsqueeze(0)
        dist = (q_pos - k_pos).clamp(min=0).to(dtype=attn_weights.dtype)  # [q_len, k_len]
        alibi_bias = -slopes.view(num_heads, 1, 1) * dist.unsqueeze(0)    # [1, H, q_len, k_len]
        attn_weights = attn_weights + alibi_bias

    # QWAB logit bias (Rotary + QWAB mode)
    _qwab_bias = kwargs.get("qwab_bias", None)
    if _qwab_bias is not None:
        q_len, k_len = query.size(-2), key.size(-2)
        attn_weights = attn_weights + _qwab_bias[:, :, :q_len, :k_len].to(dtype=attn_weights.dtype)

    # Layer-wise attention scaling
    if module.scale_attn_by_inverse_layer_idx:
        attn_weights = attn_weights / float(module.layer_idx + 1)

    if not module.is_cross_attention:
        # if only "normal" attention layer implements causal mask
        query_length, key_length = query.size(-2), key.size(-2)
        if module.bias.size(-1) >= key_length:
            causal_mask = module.bias[:, :, key_length - query_length : key_length, :key_length]
        else:
            # Fallback for long-context eval where key_length exceeds the precomputed bias buffer.
            causal_mask = torch.tril(
                torch.ones((query_length, key_length), dtype=torch.bool, device=attn_weights.device),
                diagonal=key_length - query_length,
            ).view(1, 1, query_length, key_length)
        mask_value = torch.finfo(attn_weights.dtype).min
        # Need to be a tensor, otherwise we get error: `RuntimeError: expected scalar type float but found double`.
        # Need to be on the same device, otherwise `RuntimeError: ..., x and y to be on the same device`
        mask_value = torch.full([], mask_value, dtype=attn_weights.dtype, device=attn_weights.device)
        attn_weights = torch.where(causal_mask, attn_weights.to(attn_weights.dtype), mask_value)

    # PaTH logit blending: must happen BEFORE attention_mask to preserve padding/key-mask semantics.
    # attention_mask is additive (subtracts large value from masked positions); if we blend after it,
    # large-negative padding values get diluted by finite path_logits when lam > 0, effectively
    # unmasking padding tokens.  By blending here — after the causal mask (future = finfo.min) but
    # before the additive attention_mask — the padding mask is applied on top of the blended logit
    # and its semantics are preserved.
    # path_logits: [B,H,T,T] float32, natural scale, lower-triangular (upper=0).
    # After the causal mask above, lower triangle of attn_weights is finite; blending is safe.
    _path_logits = kwargs.get('_path_logits', None)
    _path_lam = kwargs.get('_path_lam', None)
    # Guard: skip for cross-attention — causal-triangle blend is meaningless there (q_len != k_len
    # in general, and no causal ordering applies between encoder and decoder sequences).
    if _path_logits is not None and _path_lam is not None and not module.is_cross_attention:
        _T = attn_weights.shape[-1]
        _causal_blend = torch.ones(_T, _T, device=attn_weights.device, dtype=torch.bool).tril()
        # nan_to_num + clamp: path_ut_base_raw with random init can produce NaN (ill-conditioned
        # triangular solve) or very large finite values (near float32 overflow ~1e38).
        # Even with _path_lam=0, 0*NaN=NaN in IEEE.  Very large finite values are not caught
        # by nan_to_num but still overflow path_lam.grad when summed over B*H*T*T elements,
        # producing grad_norm=inf from step 1 and preventing any learning.
        # Clamp to ±100 (>> typical pre-softmax attention logit range of ±50) so gradient
        # stays finite while preserving signal once path_w_proj has learned meaningful weights.
        _path_safe = _path_logits.to(attn_weights.dtype).nan_to_num(nan=0.0, posinf=0.0, neginf=0.0).clamp(-100.0, 100.0)
        _lam_safe = _path_lam.to(attn_weights.dtype).nan_to_num(nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
        attn_weights = torch.where(
            _causal_blend,
            (1.0 - _lam_safe) * attn_weights + _lam_safe * _path_safe,
            attn_weights,
        )

    if attention_mask is not None:
        # Apply the attention mask
        causal_mask = attention_mask[:, :, :, : key.shape[-2]]
        attn_weights = attn_weights + causal_mask

    # Final safety net before softmax: prevent NaN/Inf from poisoning the whole row.
    _attn_min = torch.finfo(attn_weights.dtype).min
    attn_weights = attn_weights.nan_to_num(nan=0.0, posinf=0.0, neginf=_attn_min)
    attn_weights = nn.functional.softmax(attn_weights, dim=-1)

    # Downcast (if necessary) back to V's dtype (if in mixed-precision) -- No-Op otherwise
    attn_weights = attn_weights.type(value.dtype)
    attn_weights = module.attn_dropout(attn_weights)

    # Mask heads if we want to
    if head_mask is not None:
        attn_weights = attn_weights * head_mask

    attn_output = torch.matmul(attn_weights, value)
    attn_output = attn_output.transpose(1, 2)

    return attn_output, attn_weights
import pdb
class GPT2PaTHAttention(nn.Module):
    """
    轻量适配器：复用 fla.layers.path_attn.PaTHAttention 的实现，
    但把入参/出参改成 GPT2Block 期望的样子。
    """
    def __init__(self, config, is_cross_attention: bool = False, layer_idx: int = None):
        super().__init__()
        if is_cross_attention:
            raise NotImplementedError("PaTHAttention 暂不支持 cross-attention（config.add_cross_attention=False 时使用）。")
        if _PaTHAttention is None:
            raise ImportError(
                "未能 import fla.PaTHAttention。请确认 fla 在 sys.path 上并已可用（建议用 .pth 方案或 pip install -e .）。"
            )

        self.layer_idx = layer_idx
        # 构造 PaTH 核心
        if getattr(config, "attn_implementation", None) == "path_attn_wfreq":
            self.core = _PaTHAttentionWfreq(
                hidden_size=config.hidden_size,
                num_heads=getattr(config, "num_attention_heads", getattr(config, "n_head", None)),
                num_kv_heads=getattr(config, "num_key_value_heads", None),
                layer_idx=layer_idx,
                # 可选开关，从 config 读取（不存在则给默认）
                use_forget_gate=getattr(config, "path_use_forget_gate", False),
                use_qk_norm=getattr(config, "path_use_qk_norm", False),
                use_low_rank_w=getattr(config, "path_use_low_rank_w", True),
                use_w_shortconv=getattr(config, "path_use_w_shortconv", True),
                conv_size=getattr(config, "path_conv_size", 3),
                conv_bias=getattr(config, "path_conv_bias", False),
                num_harmonics=getattr(config, "num_harmonics", 2),
                share_freq_across_heads=getattr(config, "share_freq_across_heads", False),
                single_A_B=getattr(config, "single_A_B", False),
                use_beta_modulation=getattr(config, "use_beta_modulation", False),
                use_soft_wavelet_fox=getattr(config, "use_soft_wavelet_fox", False),
                wavelet_mode=getattr(config, "wavelet_mode", "router_rel"),
            )
        elif getattr(config, "attn_implementation", None) == "path_attn":
            self.core = _PaTHAttention(
                hidden_size=config.hidden_size,
                num_heads=getattr(config, "num_attention_heads", getattr(config, "n_head", None)),
                num_kv_heads=getattr(config, "num_key_value_heads", None),
                layer_idx=layer_idx,
                # 可选开关，从 config 读取（不存在则给默认）
                use_qk_norm=getattr(config, "path_use_qk_norm", False),
                use_low_rank_w=getattr(config, "path_use_low_rank_w", True),
                use_w_shortconv=getattr(config, "path_use_w_shortconv", True),
                conv_size=getattr(config, "path_conv_size", 3),
                conv_bias=getattr(config, "path_conv_bias", False),
                num_harmonics=getattr(config, "num_harmonics", 2),
                use_soft_wavelet_fox=getattr(config, "use_soft_wavelet_fox", False),
                wavelet_mode=getattr(config, "wavelet_mode", "router_rel"),
                logging_steps = config.logging_steps,
                wavelet_baseline_use = config.wavelet_baseline_use,
                attn_pdrop = config.attn_pdrop,
                init_theta = config.init_theta,   # initial theta for path attention ratio
                use_forget_gate = config.use_forget_gate,
                config=config,
            )
        self.resid_dropout = nn.Dropout(config.resid_pdrop)

    def forward(
        self,
        hidden_states: torch.Tensor,
        past_key_values=None,                  # HF 的 Cache，这里不使用（最小改动）
        cache_position: Optional[torch.LongTensor] = None,  # 不使用
        attention_mask: Optional[torch.Tensor] = None,      # HF 传入的 4D 加性 mask（PaTH 不用）
        head_mask: Optional[torch.Tensor] = None,           # 不使用
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
        attention_mask_2d: Optional[torch.Tensor] = None,   # 我们在 GPT2Model 额外传入的 2D 0/1 mask
        wavelet_decay_table: Optional[torch.Tensor] = None, # 我们在 GPT2Model 额外传入的 wavelet 衰减表
        geom_p = 0,
        analyzer=None,
        input_ids=None,
        **kwargs,
    ):
        if encoder_hidden_states is not None:
            raise NotImplementedError("PaTHAttention 适配器暂不支持 cross-attention。")

        # 选择 PaTH 用的 2D mask：优先采用 attention_mask_2d；训练建议传 None
        mask_2d = attention_mask_2d if attention_mask_2d is not None else None

        # PaTH 约束：训练时 attention_mask 必须为 None，且不使用 cache（源码有断言）。:contentReference[oaicite:3]{index=3}
        mask_2d = None
        use_cache = False

        # 最小改动：PaTH 自己的 cache 语义与 HF 不同，这里不走 HF 的 past_key_values。
        # 如需解码加速，后续可扩展成模块内维护 self._path_cache。
        path_cache = getattr(self, "_path_cache", None) if use_cache else None
        # print(hidden_states, 'in GPT2PaTHAttention.')
        router1, router2 = None, None
        attn_out, _weights, path_cache, dis_loss, router1, router2 = self.core(
            hidden_states,
            attention_mask=mask_2d,          # 2D 0/1 mask（或 None）
            past_key_values=path_cache,      # PaTH 自己的 cache（dict）
            output_attentions=output_attentions,
            use_cache=use_cache,
            wavelet_decay_table=wavelet_decay_table,
            geom_p=geom_p,
            analyzer=analyzer,
            input_ids=input_ids,
            **kwargs,  # 透传如 global_step/max_steps 等额外上下文
        )
        if use_cache:
            self._path_cache = path_cache

        attn_out = self.resid_dropout(attn_out)
        # GPT2Block 期望返回 (attn_output, attn_weights)
        return attn_out, None, dis_loss, router1, router2
from rotary_embedding_torch import RotaryEmbedding
class GPT2Attention(nn.Module):
    def __init__(self, config, is_cross_attention=False, layer_idx=None):
        super().__init__()
        self.config = config
        max_positions = config.max_position_embeddings
        self.register_buffer(
            "bias",
            torch.tril(torch.ones((max_positions, max_positions), dtype=torch.bool)).view(
                1, 1, max_positions, max_positions
            ),
            persistent=False,
        )
   

        self.register_buffer("masked_bias", torch.tensor(-1e4), persistent=False)

        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.split_size = self.embed_dim
        if config.wavelet_router and config.pe_method == 'no_pe':
            self.router = nn.Sequential(
                nn.Linear(self.embed_dim, 32, bias=False),
                nn.Linear(32, self.num_heads * self.config.router_band_num, bias=False),
            )
        else:
            self.router = None
        # QWAB for Rotary PE: hidden-state-conditioned wavelet logit bias
        if getattr(config, 'wavelet_router', False) and config.pe_method == 'rotary':
            self.qwab_bias_module = QWABBias(config)
        else:
            self.qwab_bias_module = None            
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                f"`embed_dim` must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`:"
                f" {self.num_heads})."
            )

        self.scale_attn_weights = config.scale_attn_weights
        self.is_cross_attention = is_cross_attention

        # Layer-wise attention scaling, reordering, and upcasting
        self.scale_attn_by_inverse_layer_idx = config.scale_attn_by_inverse_layer_idx
        self.layer_idx = layer_idx
        self.reorder_and_upcast_attn = config.reorder_and_upcast_attn

        if self.is_cross_attention:
            self.c_attn = Conv1D(2 * self.embed_dim, self.embed_dim)
            self.q_attn = Conv1D(self.embed_dim, self.embed_dim)
        else:
            self.c_attn = Conv1D(3 * self.embed_dim, self.embed_dim)
        self.c_proj = Conv1D(self.embed_dim, self.embed_dim)

        self.attn_dropout = nn.Dropout(config.attn_pdrop)
        self.resid_dropout = nn.Dropout(config.resid_pdrop)
        self.is_causal = True
        if config.pe_method == 'rotary':
            self.rotary_emb = RotaryEmbedding(dim=self.head_dim, theta=getattr(config, 'rope_theta', 10000))

        # Wavelet relative PE: precompute (head_dim, block_size, block_size) buffer
        if config.pe_method == 'wavelet' and getattr(config, 'relative_type', None) == '4':
            scales = [1, 2, 4, 8, 16, 32, 64, 128]
            shifts = [0, 1, 2, 3, 4, 5, 6, 7]
            pairs = [(s, t) for s in scales for t in shifts]  # 64 pairs
            if len(pairs) != self.head_dim:
                raise ValueError(
                    f"wavelet PE expects head_dim={len(pairs)} (8 scales × 8 shifts), "
                    f"got head_dim={self.head_dim}"
                )
            block_size = config.max_position_embeddings
            i_idx = torch.arange(block_size, dtype=torch.float32).unsqueeze(1)  # [L, 1]
            j_idx = torch.arange(block_size, dtype=torch.float32).unsqueeze(0)  # [1, L]
            W = torch.zeros(self.head_dim, block_size, block_size)
            for d, (s, t) in enumerate(pairs):
                u = (i_idx - j_idx) / s - t  # [L, L]
                W[d] = (1.0 - u ** 2) * torch.exp(-0.5 * u ** 2)
            self.register_buffer("wavelet_relative_tensor", W, persistent=False)
            wavelet_scales = torch.tensor([float(s) for s, _ in pairs], dtype=torch.float32)
            wavelet_shifts = torch.tensor([float(t) for _, t in pairs], dtype=torch.float32)
            self.register_buffer("wavelet_scales", wavelet_scales, persistent=False)
            self.register_buffer("wavelet_shifts", wavelet_shifts, persistent=False)
            # PaTH logit blending parameters (only when wavelet PE is active, fla is available,
            # and this layer_idx is listed in config.path_blend_layers)
            _path_blend_layers = getattr(config, 'path_blend_layers', None)
            _use_path_blend = (
                _path_ut_base_raw is not None and
                _path_blend_layers is not None and
                layer_idx is not None and
                layer_idx in _path_blend_layers
            )
            if _use_path_blend:
                # w projection: hidden_states → [B,T,H,D] float32 weights for path attention
                self.path_w_proj = Conv1D(self.embed_dim, self.embed_dim)
                # std=1e-4 init via _path_small_init_std (applied by _init_weights / post_init).
                # F.normalize keeps ||w||=0.1 at every forward pass, bounding A off-diagonals
                # to ≤ beta*0.01 ≤ 0.02 and κ(A) ≈ 2.6 for T=512 — no NaN from solve backward.
                # Small init ensures A ≈ I before _init_weights fires (e.g. during __init__),
                # consistent with the runtime normalization.
                self.path_w_proj._path_small_init_std = 1e-4
                # beta projection: hidden_states → [B,T,H] gate scalar (sigmoid → ×2 → (0,2))
                # beta participates in A directly: A[i,j] = beta[i] * (w_i · w_j), so small
                # init (beta ≈ sigmoid(0)*2 = 1) keeps A well-conditioned at initialisation.
                self.path_beta_proj = Conv1D(self.num_heads, self.embed_dim)
                self.path_beta_proj._path_small_init_std = 1e-4
                # λ: learnable blend scalar, init=0 (pure wavelet PE at start), warms up to 1
                self.path_lam = nn.Parameter(torch.zeros(1))
                # step counter for linear warmup (not persistent — resets on reload, intentional)
                self.register_buffer('_path_warmup_step', torch.tensor(0, dtype=torch.long), persistent=False)
                # Safety gradient hooks: zero NaN/Inf gradients on all path parameters.
                # Covers weight AND bias of both projections (bias also gets NaN grad when
                # the triangular-solve backward is ill-conditioned).
                def _safe_grad_hook(grad):
                    return grad.nan_to_num(nan=0.0, posinf=0.0, neginf=0.0)
                self.path_w_proj.weight.register_hook(_safe_grad_hook)
                self.path_w_proj.bias.register_hook(_safe_grad_hook)
                self.path_beta_proj.weight.register_hook(_safe_grad_hook)
                self.path_beta_proj.bias.register_hook(_safe_grad_hook)
                self.path_lam.register_hook(_safe_grad_hook)
                # PAT-100: sparse query-conditioned gate on path logits (query-token-wise output control)
                if getattr(config, 'path_sparse_gate', False):
                    self.path_gate_ln = nn.LayerNorm(self.head_dim)
                    self.path_gate_proj = nn.Linear(self.head_dim, 1, bias=True)
                    nn.init.constant_(self.path_gate_proj.bias, -2.0)
                    self.register_buffer(
                        '_gate_warmup_step', torch.tensor(0, dtype=torch.long), persistent=False)
                    # Guard gate params against NaN gradients (M_base can be NaN from ill-conditioned solve)
                    for _p in list(self.path_gate_ln.parameters()) + list(self.path_gate_proj.parameters()):
                        _p.register_hook(_safe_grad_hook)
        else:
            self.wavelet_relative_tensor = None
            self.wavelet_scales = None
            self.wavelet_shifts = None

        self.pruned_heads = set()

    def _get_wavelet_relative_tensor(
        self,
        q_len: int,
        k_len: int,
        device: torch.device,
        dtype: torch.dtype,
        base_tensor: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        rel_buf = self.wavelet_relative_tensor if base_tensor is None else base_tensor
        if rel_buf is not None and rel_buf.size(1) >= q_len and rel_buf.size(2) >= k_len:
            return rel_buf[:, :q_len, :k_len].to(device=device, dtype=dtype)

        if self.wavelet_scales is None or self.wavelet_shifts is None:
            raise RuntimeError("wavelet_scales/wavelet_shifts are required for dynamic wavelet relative tensor.")

        i_idx = torch.arange(q_len, dtype=dtype, device=device).unsqueeze(1)  # [q_len, 1]
        j_idx = torch.arange(k_len, dtype=dtype, device=device).unsqueeze(0)  # [1, k_len]
        delta = (i_idx - j_idx).unsqueeze(0)  # [1, q_len, k_len]

        scales = self.wavelet_scales.to(device=device, dtype=dtype).view(-1, 1, 1)  # [D,1,1]
        shifts = self.wavelet_shifts.to(device=device, dtype=dtype).view(-1, 1, 1)  # [D,1,1]
        u = delta / scales - shifts
        return (1.0 - u * u) * torch.exp(-0.5 * u * u)

    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        heads, index = find_pruneable_heads_and_indices(heads, self.num_heads, self.head_dim, self.pruned_heads)
        index_attn = torch.cat([index, index + self.split_size, index + (2 * self.split_size)])

        # Prune conv1d layers
        self.c_attn = prune_conv1d_layer(self.c_attn, index_attn, dim=1)
        self.c_proj = prune_conv1d_layer(self.c_proj, index, dim=0)

        # Update hyper params
        self.split_size = (self.split_size // self.num_heads) * (self.num_heads - len(heads))
        self.num_heads = self.num_heads - len(heads)
        self.pruned_heads = self.pruned_heads.union(heads)

    def _upcast_and_reordered_attn(self, query, key, value, attention_mask=None, head_mask=None):
        # Use `torch.baddbmm` (a bit more efficient w/ alpha param for scaling -- from Megatron-LM)
        bsz, num_heads, q_seq_len, dk = query.size()
        _, _, k_seq_len, _ = key.size()

        # Preallocate attn_weights for `baddbmm`
        attn_weights = torch.empty(bsz * num_heads, q_seq_len, k_seq_len, dtype=torch.float32, device=query.device)

        # Compute Scale Factor
        scale_factor = 1.0
        if self.scale_attn_weights:
            scale_factor /= float(value.size(-1)) ** 0.5

        if self.scale_attn_by_inverse_layer_idx:
            scale_factor /= float(self.layer_idx + 1)

        # Upcast (turn off autocast) and reorder (Scale K by 1 / root(dk))
        with torch.autocast(query.device.type, enabled=False):
            q, k = query.reshape(-1, q_seq_len, dk), key.transpose(-1, -2).reshape(-1, dk, k_seq_len)
            attn_weights = torch.baddbmm(attn_weights, q.float(), k.float(), beta=0, alpha=scale_factor)
            attn_weights = attn_weights.reshape(bsz, num_heads, q_seq_len, k_seq_len)

        if not self.is_cross_attention:
            # if only "normal" attention layer implements causal mask
            query_length, key_length = query.size(-2), key.size(-2)
            if self.bias.size(-1) >= key_length:
                causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]
            else:
                causal_mask = torch.tril(
                    torch.ones((query_length, key_length), dtype=torch.bool, device=attn_weights.device),
                    diagonal=key_length - query_length,
                ).view(1, 1, query_length, key_length)
            mask_value = torch.finfo(attn_weights.dtype).min
            # Need to be a tensor, otherwise we get error: `RuntimeError: expected scalar type float but found double`.
            # Need to be on the same device, otherwise `RuntimeError: ..., x and y to be on the same device`
            mask_value = torch.tensor(mask_value, dtype=attn_weights.dtype, device=attn_weights.device)
            attn_weights = torch.where(causal_mask, attn_weights, mask_value)

        if attention_mask is not None:
            # Apply the attention mask
            attn_weights = attn_weights + attention_mask

        attn_weights = nn.functional.softmax(attn_weights, dim=-1)

        # Downcast (if necessary) back to V's dtype (if in mixed-precision) -- No-Op if otherwise
        if attn_weights.dtype != torch.float32:
            raise RuntimeError("Error with upcasting, attn_weights does not have dtype torch.float32")
        attn_weights = attn_weights.type(value.dtype)
        attn_weights = self.attn_dropout(attn_weights)

        # Mask heads if we want to
        if head_mask is not None:
            attn_weights = attn_weights * head_mask

        attn_output = torch.matmul(attn_weights, value)
        attn_output = attn_output.transpose(1, 2)

        return attn_output, attn_weights

    @deprecate_kwarg("past_key_value", new_name="past_key_values", version="4.58")
    def forward(
        self,
        hidden_states: Optional[tuple[torch.FloatTensor]],
        past_key_values: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = False,
        rope=None,
        wavelet_decay_table=None,
        **kwargs,
    ) -> tuple[Union[torch.Tensor, tuple[torch.Tensor]], ...]:
        is_cross_attention = encoder_hidden_states is not None
        if past_key_values is not None:
            if isinstance(past_key_values, EncoderDecoderCache):
                is_updated = past_key_values.is_updated.get(self.layer_idx)
                if is_cross_attention:
                    # after the first generated id, we can subsequently re-use all key/value_layer from cache
                    curr_past_key_value = past_key_values.cross_attention_cache
                else:
                    curr_past_key_value = past_key_values.self_attention_cache
            else:
                curr_past_key_value = past_key_values

        if is_cross_attention:
            if not hasattr(self, "q_attn"):
                raise ValueError(
                    "If class is used as cross attention, the weights `q_attn` have to be defined. "
                    "Please make sure to instantiate class with `GPT2Attention(..., is_cross_attention=True)`."
                )
            query_states = self.q_attn(hidden_states)
            attention_mask = encoder_attention_mask

            # Try to get key/value states from cache if possible
            if past_key_values is not None and is_updated:
                key_states = curr_past_key_value.layers[self.layer_idx].keys
                value_states = curr_past_key_value.layers[self.layer_idx].values
            else:
                key_states, value_states = self.c_attn(encoder_hidden_states).split(self.split_size, dim=2)
                shape_kv = (*key_states.shape[:-1], -1, self.head_dim)
                key_states = key_states.view(shape_kv).transpose(1, 2)
                value_states = value_states.view(shape_kv).transpose(1, 2)
        else:
            query_states, key_states, value_states = self.c_attn(hidden_states).split(self.split_size, dim=2)
            shape_kv = (*key_states.shape[:-1], -1, self.head_dim)
            key_states = key_states.view(shape_kv).transpose(1, 2)
            value_states = value_states.view(shape_kv).transpose(1, 2)

        shape_q = (*query_states.shape[:-1], -1, self.head_dim)
        query_states = query_states.view(shape_q).transpose(1, 2)

        if (past_key_values is not None and not is_cross_attention) or (
            past_key_values is not None and is_cross_attention and not is_updated
        ):
            # save all key/value_layer to cache to be re-used for fast auto-regressive generation
            cache_position = cache_position if not is_cross_attention else None
            key_states, value_states = curr_past_key_value.update(
                key_states, value_states, self.layer_idx, {"cache_position": cache_position}
            )
            # set flag that curr layer for cross-attn is already updated so we can re-use in subsequent calls
            if is_cross_attention:
                past_key_values.is_updated[self.layer_idx] = True

        is_causal = attention_mask is None and query_states.shape[-2] > 1 and not is_cross_attention

        using_eager = self.config._attn_implementation == "eager"
        attention_interface: Callable = eager_attention_forward
        # wavelet/alibi PE require eager (custom bias injected inside eager_attention_forward)
        if self.config.pe_method == 'wavelet' and getattr(self.config, 'relative_type', None) == '4':
            using_eager = True
        elif self.config.pe_method == 'alibi':
            using_eager = True
        elif self.config._attn_implementation != "eager" and self.config.pe_method != 'rotary':
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]
        if self.config.pe_method == 'rotary':
            query_states = self.rotary_emb.rotate_queries_or_keys(query_states)
            key_states = self.rotary_emb.rotate_queries_or_keys(key_states)
            # query_states = query_states.permute(0, 2, 1, 3)
            # key_states = key_states.permute(0, 2, 1, 3)
        # QWAB bias for Rotary PE (computed from hidden_states, no path-attention dependency)
        _qwab_bias = None
        if self.qwab_bias_module is not None and not is_cross_attention:
            _T_q, _T_k = query_states.shape[-2], key_states.shape[-2]
            if _T_q == _T_k:  # skip during KV-cache decode (T_q=1, T_k=full)
                _qwab_bias = self.qwab_bias_module(hidden_states)
        router1 = None
        if self.router is not None:
            jitter_std = getattr(self.config, "router_jitter_std", 0.0)   # e.g. 0.01
            jitter_apply_in_eval = getattr(self.config, "router_jitter_apply_in_eval", False)
            jitter_scale_by_logit_std = getattr(self.config, "router_jitter_scale_by_logit_std", True)      
            def _add_gaussian_jitter(logits: torch.Tensor, std: float) -> torch.Tensor:
                """
                logits: [B,T,H,S]
                std: base noise std in logit space
                """
                if std <= 0:
                    return logits
                if (not self.training) and (not jitter_apply_in_eval):
                    return logits

                if jitter_scale_by_logit_std:
                    # scale noise by per-(B,T,H) logit std over S to be robust to logit magnitude
                    # detach so the scaling factor doesn't backprop weirdly
                    scale = logits.detach().std(dim=-1, keepdim=True).clamp_min(1e-6)
                    noise = torch.randn_like(logits) * (std * scale)
                else:
                    noise = torch.randn_like(logits) * std

                return logits + noise
            B,H,T,D = query_states.shape
            router1_logits = self.router(hidden_states)          # [B,T,H*S]
            router1_logits = router1_logits.view(B, T, H, self.config.router_band_num)      # [B,T,H,S]
            router1_logits = _add_gaussian_jitter(router1_logits, jitter_std)
            router1 = torch.softmax(router1_logits / 1.0, dim=-1)  # [B,T,H,S]            

        _gate_sparse_loss = query_states.new_zeros(())
        if using_eager and self.reorder_and_upcast_attn and self.config.pe_method not in ('rotary', 'wavelet'):
            attn_output, attn_weights = self._upcast_and_reordered_attn(
                query_states, key_states, value_states, attention_mask, head_mask
            )
        else:
            wavelet_rel_kwarg = {}
            if self.wavelet_relative_tensor is not None:
                wavelet_rel_kwarg["wavelet_relative_tensor"] = self.wavelet_relative_tensor
            # PAT-100: sparse query-conditioned gate on path logits.
            # Single path_ut_base_raw call. M_base (returned by that call) provides q_corr
            # for gate conditioning; gate_eff then scales E_base_raw before lam-blending.
            if (self.wavelet_relative_tensor is not None and hasattr(self, 'path_lam')
                    and _path_ut_base_raw is not None and not self.is_cross_attention):
                _B, _H, _T_q, _D = query_states.shape
                _T_k = key_states.shape[-2]
                # Skip during KV-cache generation (T_q=1, T_k=full sequence length).
                if _T_q == _T_k:
                    # lam warmup
                    if self.training:
                        self._path_warmup_step.add_(1)
                        _warmup = min(float(self._path_warmup_step.item()) / 2000.0, 1.0)
                    else:
                        _warmup = 1.0
                    _lam_eff = (
                        self.path_lam.float()
                        .nan_to_num(nan=0.0, posinf=1.0, neginf=0.0)
                        .clamp(0.0, 1.0)
                        * _warmup
                    )

                    _q = query_states.permute(0, 2, 1, 3).contiguous()  # [B,T,H,D]
                    _k = key_states.permute(0, 2, 1, 3).contiguous()
                    _hs = hidden_states
                    _w_raw = self.path_w_proj(_hs).view(_B, _T_q, _H, _D).to(torch.float32)
                    # l2-normalise w to unit vectors — matches original PaTHAttention design
                    # (path_attn.py uses l2_norm(w) giving ||w||=1 before the path kernel).
                    _w = F.normalize(_w_raw, p=2, dim=-1, eps=1e-6)
                    _beta = torch.sigmoid(self.path_beta_proj(_hs)).to(torch.float32) * 2.0

                    # Single call — M_base also used for gate conditioning when sparse gate is on.
                    _E_base, _M_base, _, _ = _path_ut_base_raw(_q, _k, _w, _beta)
                    _E_base = _E_base.nan_to_num(nan=0.0, posinf=0.0, neginf=0.0)
                    _M_base = _M_base.nan_to_num(nan=0.0, posinf=0.0, neginf=0.0)

                    if getattr(self.config, 'path_sparse_gate', False) and hasattr(self, 'path_gate_proj'):
                        # Gate warmup coefficient (shared for both learned and forced-open gate)
                        if self.training:
                            self._gate_warmup_step.add_(1)
                            _eta = min(
                                float(self._gate_warmup_step.item()) /
                                float(getattr(self.config, 'gate_warmup_steps', 2000)),
                                1.0)
                        else:
                            _eta = 1.0
                        if getattr(self.config, 'path_gate_force_open', False):
                            # Forced-open control: g_i≡1, gate_eff = eta (full Route-A gradient)
                            # Bypasses LN/proj/sigmoid; path_gate_ln/proj params receive no gradient.
                            _gate_eff = torch.ones(
                                _B, _T_q, _H, 1, dtype=torch.float32, device=query_states.device) * _eta
                            # No sparse loss: constant penalty alpha*1 has no gradient signal
                        else:
                            # Learned gate conditioned on delta = q - q_corr
                            # nan_to_num guards: M_base can be NaN from ill-conditioned triangular solve
                            _q_corr = torch.einsum("bhij,bjhd->bihd", _M_base, _w).nan_to_num(nan=0.0, posinf=0.0, neginf=0.0)  # [B,T,H,D]
                            _delta = _q.to(torch.float32) - _q_corr              # [B,T,H,D]
                            _gate_feat = self.path_gate_ln(_delta)                # [B,T,H,D]
                            _gate = torch.sigmoid(self.path_gate_proj(_gate_feat)).nan_to_num(nan=0.0)  # [B,T,H,1]
                            _gate_eff = _gate * _eta                              # [B,T,H,1]
                            if self.training:
                                _alpha = float(getattr(self.config, 'gate_sparse_alpha', 0.01))
                                _gate_sparse_loss = _alpha * _gate.mean()
                        # Fold gate into λ to get per-query blend coefficient [B,H,T,1].
                        # Semantics: ã[i,j] = (1 - λ·g_i)·a_wav[i,j] + λ·g_i·E[i,j]
                        # When g_i=0 → pure wavelet (no dilution from λ).
                        # When g_i=1 → same as scalar-λ blend.
                        _lam_eff = _lam_eff * _gate_eff.permute(0, 2, 1, 3)      # [B,H,T,1]

                    _path_logits = (_E_base * (_D ** -0.5)).to(torch.float32)
                    wavelet_rel_kwarg['_path_logits'] = _path_logits
                    wavelet_rel_kwarg['_path_lam'] = _lam_eff
            attn_output, attn_weights = attention_interface(
                self,
                query_states,
                key_states,
                value_states,
                attention_mask,
                head_mask=head_mask,
                dropout=self.attn_dropout.p if self.training else 0.0,
                is_causal=is_causal,
                wavelet_decay_table=wavelet_decay_table,
                router=router1,
                qwab_bias=_qwab_bias,
                **wavelet_rel_kwarg,
                **kwargs,
            )
        attn_output = attn_output.reshape(*attn_output.shape[:-2], -1).contiguous()
        attn_output = self.c_proj(attn_output)
        attn_output = self.resid_dropout(attn_output)

        # Return 5-tuple to match GPT2Block.forward unpacking.
        # 3rd element: gate sparse loss (scalar) when PAT-100 is active, else zero.
        return attn_output, attn_weights, _gate_sparse_loss, None, None


class GPT2MLP(nn.Module):
    def __init__(self, intermediate_size, config):
        super().__init__()
        embed_dim = config.hidden_size
        self.c_fc = Conv1D(intermediate_size, embed_dim)
        self.c_proj = Conv1D(embed_dim, intermediate_size)
        self.act = ACT2FN[config.activation_function]
        self.dropout = nn.Dropout(config.resid_pdrop)

    def forward(self, hidden_states: Optional[tuple[torch.FloatTensor]]) -> torch.FloatTensor:
        hidden_states = self.c_fc(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.c_proj(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states


class GPT2Block(GradientCheckpointingLayer):
    def __init__(self, config, layer_idx=None):
        super().__init__()
        hidden_size = config.hidden_size
        inner_dim = config.n_inner if config.n_inner is not None else 4 * hidden_size

        self.ln_1 = nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)
        if getattr(config, "attn_implementation", None) in ["path_attn", "path_attn_wfreq"]:
            self.attn = GPT2PaTHAttention(config=config, layer_idx=layer_idx)
        else:
            self.attn = GPT2Attention(config=config, layer_idx=layer_idx)
        self.ln_2 = nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)

        if config.add_cross_attention:
            self.crossattention = GPT2Attention(config=config, is_cross_attention=True, layer_idx=layer_idx)
            self.ln_cross_attn = nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)

        self.mlp = GPT2MLP(inner_dim, config)

    @deprecate_kwarg("past_key_value", new_name="past_key_values", version="4.58")
    def forward(
        self,
        hidden_states: Optional[tuple[torch.FloatTensor]],
        past_key_values: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
        rope=None,
        wavelet_decay_table=None,
        geom_p=0,
        analyzer=None,
        input_ids=None,
        **kwargs,
    ) -> Union[tuple[torch.Tensor], Optional[tuple[torch.Tensor, tuple[torch.FloatTensor, ...]]]]:
        residual = hidden_states
        hidden_states = self.ln_1(hidden_states)
        # print(hidden_states, 'in GPT2Block.')
        router1, router2 = None, None
        attn_output, self_attn_weights, dis_loss, router1, router2 = self.attn(
            hidden_states,
            past_key_values=past_key_values,
            cache_position=cache_position,
            attention_mask=attention_mask,
            head_mask=head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            rope=rope,
            wavelet_decay_table=wavelet_decay_table,
            geom_p=geom_p,
            analyzer=analyzer,
            input_ids=input_ids,
            **kwargs,
        )
        # residual connection
        hidden_states = attn_output + residual

        if encoder_hidden_states is not None:
            # add one self-attention block for cross-attention
            if not hasattr(self, "crossattention"):
                raise ValueError(
                    f"If `encoder_hidden_states` are passed, {self} has to be instantiated with "
                    "cross-attention layers by setting `config.add_cross_attention=True`"
                )
            residual = hidden_states
            hidden_states = self.ln_cross_attn(hidden_states)
            cross_attn_output, cross_attn_weights = self.crossattention(
                hidden_states,
                past_key_values=past_key_values,
                attention_mask=attention_mask,
                head_mask=head_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                output_attentions=output_attentions,
            )
            # residual connection
            hidden_states = residual + cross_attn_output

        residual = hidden_states
        hidden_states = self.ln_2(hidden_states)
        feed_forward_hidden_states = self.mlp(hidden_states)
        # residual connection
        hidden_states = residual + feed_forward_hidden_states

        outputs = (hidden_states,)
        if output_attentions:
            outputs += (self_attn_weights,)
            if encoder_hidden_states is not None:
                outputs += (cross_attn_weights,)

        return outputs, dis_loss, router1, router2


# Copied from transformers.models.xlm.modeling_xlm.XLMSequenceSummary with XLM->GPT2
class GPT2SequenceSummary(nn.Module):
    r"""
    Compute a single vector summary of a sequence hidden states.

    Args:
        config ([`GPT2Config`]):
            The config used by the model. Relevant arguments in the config class of the model are (refer to the actual
            config class of your model for the default values it uses):

            - **summary_type** (`str`) -- The method to use to make this summary. Accepted values are:

                - `"last"` -- Take the last token hidden state (like XLNet)
                - `"first"` -- Take the first token hidden state (like Bert)
                - `"mean"` -- Take the mean of all tokens hidden states
                - `"cls_index"` -- Supply a Tensor of classification token position (GPT/GPT-2)
                - `"attn"` -- Not implemented now, use multi-head attention

            - **summary_use_proj** (`bool`) -- Add a projection after the vector extraction.
            - **summary_proj_to_labels** (`bool`) -- If `True`, the projection outputs to `config.num_labels` classes
              (otherwise to `config.hidden_size`).
            - **summary_activation** (`Optional[str]`) -- Set to `"tanh"` to add a tanh activation to the output,
              another string or `None` will add no activation.
            - **summary_first_dropout** (`float`) -- Optional dropout probability before the projection and activation.
            - **summary_last_dropout** (`float`)-- Optional dropout probability after the projection and activation.
    """

    def __init__(self, config: GPT2Config):
        super().__init__()

        self.summary_type = getattr(config, "summary_type", "last")
        if self.summary_type == "attn":
            # We should use a standard multi-head attention module with absolute positional embedding for that.
            # Cf. https://github.com/zihangdai/xlnet/blob/master/modeling.py#L253-L276
            # We can probably just use the multi-head attention module of PyTorch >=1.1.0
            raise NotImplementedError

        self.summary = nn.Identity()
        if hasattr(config, "summary_use_proj") and config.summary_use_proj:
            if hasattr(config, "summary_proj_to_labels") and config.summary_proj_to_labels and config.num_labels > 0:
                num_classes = config.num_labels
            else:
                num_classes = config.hidden_size
            self.summary = nn.Linear(config.hidden_size, num_classes)

        activation_string = getattr(config, "summary_activation", None)
        self.activation: Callable = get_activation(activation_string) if activation_string else nn.Identity()

        self.first_dropout = nn.Identity()
        if hasattr(config, "summary_first_dropout") and config.summary_first_dropout > 0:
            self.first_dropout = nn.Dropout(config.summary_first_dropout)

        self.last_dropout = nn.Identity()
        if hasattr(config, "summary_last_dropout") and config.summary_last_dropout > 0:
            self.last_dropout = nn.Dropout(config.summary_last_dropout)

    def forward(
        self, hidden_states: torch.FloatTensor, cls_index: Optional[torch.LongTensor] = None
    ) -> torch.FloatTensor:
        """
        Compute a single vector summary of a sequence hidden states.

        Args:
            hidden_states (`torch.FloatTensor` of shape `[batch_size, seq_len, hidden_size]`):
                The hidden states of the last layer.
            cls_index (`torch.LongTensor` of shape `[batch_size]` or `[batch_size, ...]` where ... are optional leading dimensions of `hidden_states`, *optional*):
                Used if `summary_type == "cls_index"` and takes the last token of the sequence as classification token.

        Returns:
            `torch.FloatTensor`: The summary of the sequence hidden states.
        """
        if self.summary_type == "last":
            output = hidden_states[:, -1]
        elif self.summary_type == "first":
            output = hidden_states[:, 0]
        elif self.summary_type == "mean":
            output = hidden_states.mean(dim=1)
        elif self.summary_type == "cls_index":
            if cls_index is None:
                cls_index = torch.full_like(
                    hidden_states[..., :1, :],
                    hidden_states.shape[-2] - 1,
                    dtype=torch.long,
                )
            else:
                cls_index = cls_index.unsqueeze(-1).unsqueeze(-1)
                cls_index = cls_index.expand((-1,) * (cls_index.dim() - 1) + (hidden_states.size(-1),))
            # shape of cls_index: (bsz, XX, 1, hidden_size) where XX are optional leading dim of hidden_states
            output = hidden_states.gather(-2, cls_index).squeeze(-2)  # shape (bsz, XX, hidden_size)
        elif self.summary_type == "attn":
            raise NotImplementedError

        output = self.first_dropout(output)
        output = self.summary(output)
        output = self.activation(output)
        output = self.last_dropout(output)

        return output


@auto_docstring
class GPT2PreTrainedModel(PreTrainedModel):
    config: GPT2Config
    base_model_prefix = "transformer"
    is_parallelizable = True
    supports_gradient_checkpointing = True
    _no_split_modules = ["GPT2Block"]
    _skip_keys_device_placement = "past_key_values"
    _supports_flash_attn = True
    _supports_sdpa = True
    _supports_attention_backend = True

    _can_compile_fullgraph = True

    def __init__(self, *inputs, **kwargs):
        super().__init__(*inputs, **kwargs)

    def _init_weights(self, module):
        """Initialize the weights."""
        if isinstance(module, (nn.Linear, Conv1D)):
            std = getattr(module, "_path_small_init_std", self.config.initializer_range)
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

        # Reinitialize selected weights subject to the OpenAI GPT-2 Paper Scheme:
        #   > A modified initialization which accounts for the accumulation on the residual path with model depth. Scale
        #   > the weights of residual layers at initialization by a factor of 1/√N where N is the # of residual layers.
        #   >   -- GPT-2 :: https://openai.com/blog/better-language-models/
        #
        # Reference (Megatron-LM): https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/model/gpt_model.py
        for name, p in module.named_parameters():
            if name == "c_proj.weight":
                # Special Scaled Initialization --> There are 2 Layer Norms per Transformer Block
                p.data.normal_(mean=0.0, std=(self.config.initializer_range / math.sqrt(2 * self.config.n_layer)))


@dataclass
@auto_docstring(
    custom_intro="""
    Base class for outputs of models predicting if two sentences are consecutive or not.
    """
)
class GPT2DoubleHeadsModelOutput(ModelOutput):
    r"""
    loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
        Language modeling loss.
    mc_loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `mc_labels` is provided):
        Multiple choice classification loss.
    logits (`torch.FloatTensor` of shape `(batch_size, num_choices, sequence_length, config.vocab_size)`):
        Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
    mc_logits (`torch.FloatTensor` of shape `(batch_size, num_choices)`):
        Prediction scores of the multiple choice classification head (scores for each choice before SoftMax).
    past_key_values (`Cache`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
        It is a [`~cache_utils.Cache`] instance. For more details, see our [kv cache guide](https://huggingface.co/docs/transformers/en/kv_cache).

        Contains pre-computed hidden-states (key and values in the attention blocks) that can be used (see
        `past_key_values` input) to speed up sequential decoding.
    """

    loss: Optional[torch.FloatTensor] = None
    mc_loss: Optional[torch.FloatTensor] = None
    logits: Optional[torch.FloatTensor] = None
    mc_logits: Optional[torch.FloatTensor] = None
    past_key_values: Optional[Cache] = None
    hidden_states: Optional[tuple[torch.FloatTensor]] = None
    attentions: Optional[tuple[torch.FloatTensor]] = None


PARALLELIZE_DOCSTRING = r"""
    This is an experimental feature and is a subject to change at a moment's notice.

    Uses a device map to distribute attention modules of the model across several devices. If no device map is given,
    it will evenly distribute blocks across all devices.

    Args:
        device_map (`dict[int, list]`, *optional*):
            A dictionary that maps attention modules to devices. Note that the embedding module and LMHead are always
            automatically mapped to the first device (for esoteric reasons). That means that the first device should
            have fewer attention modules mapped to it than other devices. For reference, the gpt2 models have the
            following number of attention modules:

                - openai-community/gpt2: 12
                - openai-community/gpt2-medium: 24
                - openai-community/gpt2-large: 36
                - openai-community/gpt2-xl: 48

    Example:

    ```python
    # Here is an example of a device map on a machine with 4 GPUs using gpt2-xl, which has a total of 48 attention modules:
    model = GPT2LMHeadModel.from_pretrained("openai-community/gpt2-xl")
    device_map = {
        0: [0, 1, 2, 3, 4, 5, 6, 7, 8],
        1: [9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21],
        2: [22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34],
        3: [35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47],
    }
    model.parallelize(device_map)
    ```
"""
DEPARALLELIZE_DOCSTRING = r"""
    Moves the model to cpu from a model parallel state.

    Example:

    ```python
    # On a 4 GPU machine with openai-community/gpt2-large:
    model = GPT2LMHeadModel.from_pretrained("openai-community/gpt2-large")
    device_map = {
        0: [0, 1, 2, 3, 4, 5, 6, 7],
        1: [8, 9, 10, 11, 12, 13, 14, 15],
        2: [16, 17, 18, 19, 20, 21, 22, 23],
        3: [24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35],
    }
    model.parallelize(device_map)  # Splits the model across several devices
    model.deparallelize()  # Put the model back on cpu and cleans memory by calling torch.cuda.empty_cache()
    ```
"""

import pdb
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

# 假设你的 tensor 叫 x，形状为 [64, 257]
# x = your_tensor

# 示例：如果你只是想测试，可以取消注释这句
# x = torch.randn(64, 257)
def plot_tensor_grouped_lines(x: torch.Tensor):
    plt.figure(figsize=(10, 6))

    num_groups = 64 // 16  # =4

    for i in tqdm(range(num_groups), desc="Plotting groups"):
        start = i * 16
        
        # 将这 16 条线合成为一条（可选），这里按照你的要求：每 16 条画一条折线
        # 你需要的是把这 16 行的每一个元素画出来，同时画 16 条线的话可以改成循环画
        # === 根据题意，我理解为：把每组16行平均后画一条线 ===
        # y = x[start:end].mean(dim=0).tolist()
        y = x[start].tolist()
        
        plt.plot(y, label=f"dim {start}")

    plt.legend()
    plt.title("Tensor grouped-by-16 line plot")
    plt.xlabel("Index (257 dim)")
    plt.ylabel("Value")
    os.makedirs("wavelet_spectrum", exist_ok=True)
    plt.savefig("wavelet_spectrum/tensor_64x257_group16_plot.png", dpi=300)
    print("Saved to wavelet_spectrum/tensor_64x257_group16_plot.png")
@torch.no_grad()
def build_wavelet_dtt_bands(
    wavelet_dtt: torch.Tensor,   # [D, T, T]
    K: int = 8,
    method: str = "freq_proxy",  # "freq_proxy" or "scale_d"
    scale_d: torch.Tensor | None = None,  # [D], optional
    compute_dtype: torch.dtype = torch.float32,
    chunk_d: int = 1024,         # for big D
):
    """
    Returns:
      wavelet_dtt_bands: [K, D, T, T]  (each band masks a subset of D)
      masks           : [K, D]         (0/1 float mask)
      score_d         : [D]            (used for sorting / bucketing)
      order           : [D]            (indices sorted by score_d ascending)
    """
    assert wavelet_dtt.dim() == 3, f"expect [D,T,T], got {tuple(wavelet_dtt.shape)}"
    D, T1, T2 = wavelet_dtt.shape
    assert T1 == T2, f"expect square [T,T], got {T1}x{T2}"

    device = wavelet_dtt.device
    wav = wavelet_dtt.to(dtype=compute_dtype)

    # ---- (A) build score_d for bucketing ----
    if method == "scale_d":
        assert scale_d is not None, "method='scale_d' needs scale_d: [D]"
        assert scale_d.shape == (D,), f"scale_d should be [D], got {tuple(scale_d.shape)}"
        score_d = scale_d.to(device=device, dtype=compute_dtype).clone()

    elif method == "freq_proxy":
        # proxy: average absolute finite differences along both axes
        # higher => more rapidly varying kernel => "higher-frequency-ish"
        score_d = torch.empty(D, device=device, dtype=compute_dtype)

        # chunk to avoid big intermediate allocations if D large
        for s in tqdm(range(0, D, chunk_d), desc="Scoring wavelet_dtt by freq proxy"):
            e = min(D, s + chunk_d)
            x = wav[s:e]  # [d',T,T]

            # diffs
            dx_t = (x[:, 1:, :] - x[:, :-1, :]).abs().mean(dim=(1, 2))    # [d']
            dx_n = (x[:, :, 1:] - x[:, :, :-1]).abs().mean(dim=(1, 2))    # [d']

            score_d[s:e] = dx_t + dx_n

    else:
        raise ValueError(f"Unknown method: {method}")

    # ---- (B) sort + split D into K buckets ----
    order = torch.argsort(score_d, dim=0)  # ascending
    chunks = torch.chunk(order, K)         # nearly equal size

    masks = torch.zeros((K, D), device=device, dtype=compute_dtype)
    for k, ids in enumerate(chunks):
        masks[k, ids] = 1.0

    # ---- (C) apply masks to create bands ----
    # [K,D,1,1] * [1,D,T,T] => [K,D,T,T]
    wavelet_dtt_bands = masks[:, :, None, None] * wavelet_dtt[None, :, :, :].to(dtype=compute_dtype)

    return wavelet_dtt_bands, masks, score_d, order
from pathlib import Path
def build_bucket_offsets(*, T: int, K: int, config) -> list:
    mode = str(getattr(config, "shift_offset_mode", "linear"))

    cap = int(getattr(config, "shift_offset_cap", 0))           # 0 => no cap
    off_min = int(getattr(config, "shift_offset_min", 0))

    def _apply_cap(x: int) -> int:
        if cap and cap > 0:
            x = min(x, cap)
        # 不能超过 T-1，否则必然全被 mask 掉
        x = min(x, T - 1)
        # 最小值约束
        x = max(x, off_min)
        return int(x)

    if mode == "linear":
        stride = int(getattr(config, "shift_stride", 16))
        center = int(getattr(config, "shift_center", 8))
        offsets = [0] + [stride * j + center for j in range(1, K)]
        return [_apply_cap(x) for x in offsets]

    if mode == "list":
        offsets = list(getattr(config, "shift_offsets", []))
        assert len(offsets) == K, f"shift_offsets must have length K={K}, got {len(offsets)}"
        return [_apply_cap(int(x)) for x in offsets]

    if mode == "ratio":
        ratios = list(getattr(config, "shift_offsets_ratio", []))
        assert len(ratios) == K, f"shift_offsets_ratio must have length K={K}, got {len(ratios)}"
        offsets = [int(round(r * T)) for r in ratios]
        return [_apply_cap(x) for x in offsets]

    if mode == "geom":
        a = float(getattr(config, "shift_geom_a", 32.0))
        r = float(getattr(config, "shift_geom_r", 2.0))
        offsets = [0]
        for j in range(1, K):
            offsets.append(int(round(a * (r ** (j - 1)))))
        return [_apply_cap(x) for x in offsets]

    raise ValueError(f"Unknown shift_offset_mode={mode}")   
@auto_docstring
class GPT2Model(GPT2PreTrainedModel):
    _supports_param_buffer_assignment = False

    def __init__(self, config):
        super().__init__(config)

        self.embed_dim = config.hidden_size
        self.wte = nn.Embedding(config.vocab_size, self.embed_dim)
        attn_impl = getattr(config, "attn_implementation", getattr(config, "_attn_implementation", "eager"))
        if config.pe_method in ('no_pe', 'wavelet', 'alibi'):
            pass
        else:
            if config.pe_method != 'rotary' and attn_impl == 'eager':
                self.wpe = nn.Embedding(config.max_position_embeddings, self.embed_dim)

        self.drop = nn.Dropout(config.embd_pdrop)
        self.h = nn.ModuleList([GPT2Block(config, layer_idx=i) for i in range(config.num_hidden_layers)])
        self.ln_f = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_epsilon)

        # Model parallel
        self.model_parallel = False
        self.device_map = None
        self.gradient_checkpointing = False
        self._attn_implementation = config._attn_implementation

        # Initialize weights and apply final processing

        self.post_init()
        self.config=config
        wavelet_mode = str(getattr(config, "wavelet_mode", "router_rel")).strip().lower()
        self._skip_wavelet_decay_table = wavelet_mode in ("logit_bias_ctxscale_shift_v0", "logit_bias_ctxscale_shift_v0_film")
        self.scale_range = config.scale_range
        self.s_tensor = None
        self.beta_tensor = None
        if not self._skip_wavelet_decay_table:
            if config.scale_type == 'learnable':
                self.S = 8  # 你现在 interval 那套基本就是 8 个 scale

                # 用你当前 custom 作为初始化
                # 例如 scale_range=(0,16), interval=2 -> i=[0,2,4,...14]
                init_exps = list(range(self.scale_range[0], self.scale_range[1], (self.scale_range[1]-self.scale_range[0])//self.S))
                init_scales = torch.tensor([2**i for i in init_exps], dtype=torch.float32)  # [S]

                # 初始化 a0 和 r：a0 = init_scales[0], r = (init_scales[-1]/init_scales[0])**(1/(S-1))
                a0_init = init_scales[0].item()
                r_init  = (init_scales[-1].item()/init_scales[0].item()) ** (1.0/(self.S-1))

                self.theta_a0 = torch.nn.Parameter(torch.tensor([math.log(a0_init)], dtype=torch.float32, device='cuda'))
                self.theta_r  = torch.nn.Parameter(torch.tensor([math.log(math.expm1(r_init-1))], dtype=torch.float32, device='cuda'))
            if config.scale_type == 'custom':
                self.s_tensor, self.beta_tensor = self.make_scale_shift_vectors()
            elif config.scale_type == 'uniform':
                self.s_tensor, self.beta_tensor = self.make_uniform_scale_shift_vectors()
            elif config.scale_type == 'learnable':
                pass
            elif config.scale_type == 'none':
                self.s_tensor = None
                self.beta_tensor = None
            else:
                raise ValueError(f"Unknown scale_type: {config.scale_type}")
        self.d_m = None
        if (not self._skip_wavelet_decay_table) and config.scale_type != 'learnable':
            use_time_shift = getattr(config, 'use_time_shift', False)
            if use_time_shift:
                K = getattr(config, 'shift_bucket_K', 4)
                # bucket_offsets = torch.tensor([0] + [16*j + 8 for j in range(1, K)], device=self.s_tensor.device)
                bucket_offsets = build_bucket_offsets(T=config.block_size, K=K, config=config)
                self.d_m = self.make_decay_per_dim_custom_bucketed(64, config.block_size, self.s_tensor, self.beta_tensor, bucket_offsets)
            else:
                self.d_m = self.make_decay_per_dim_custom(64, config.block_size, self.s_tensor, self.beta_tensor)
        # if config.wavelet_router:
        #     self.d_m, _, _, _ = build_wavelet_dtt_bands(self.d_m)
        if config.analyzer:
            # self.analyzer = LayerAttentionAnalyzer(num_layers=config.num_hidden_layers, num_heads=12)
            # self.analyzer.reset()
            # self.scale_wise_analyzer = ScaleWiseAnalyzer(n_layers=config.num_hidden_layers, n_heads=12, head_dim=64, device='cuda', save_dir=f'scale_wise_analyzer_logs_{config.model_name_or_path}')
            # self.router_analyzer = RouterAnalyzer(
            #                             n_layers=config.num_hidden_layers,
            #                             n_heads=config.num_attention_heads,
            #                             n_scales=8,
            #                             save_dir=f'router_analysis_{config.model_name_or_path}',
            #                             device="cuda",              # 统计放 CPU 更省显存
            #                             compute_dtype=torch.float32,
            #                         )
            self.analyzer_tools = {
                'layer_attention_analyzer': LayerAttentionAnalyzer(num_layers=config.num_hidden_layers, num_heads=12),
                'scale_wise_analyzer': ScaleWiseAnalyzer(n_layers=config.num_hidden_layers, n_heads=12, head_dim=64, device='cuda', save_dir=f'L_{config.block_size}_scale_wise_analyzer_logs_{config.model_name_or_path}'),
                'router_analyzer': RouterAnalyzer(
                                        n_layers=config.num_hidden_layers,
                                        n_heads=config.num_attention_heads,
                                        n_scales=8,
                                        save_dir=f'L{config.block_size}_router_analysis_{config.model_name_or_path}',
                                        device="cuda",              # 统计放 CPU 更省显存
                                        compute_dtype=torch.float32,
                                    ),
                'token_scale_dumper': TokenScaleDumper(TokenScaleDumperConfig(
                                                                                out_dir=f'L{config.block_size}_router_analysis_{config.model_name_or_path}',
                                                                                tag="exp_router",
                                                                                compress=False,
                                                                                max_rows_per_shard=2_000_000,
                                                                                flush_every=200_000,
                                                                                low_frac=0.25,        # last 25% scales are "low-freq"
                                                                                pos_segments=3,
                                                                            )),
                'pwav_mean_logger' : PWavMeanLogger(save_dir=f'L{config.block_size}_score_matrix_analysis_{config.model_name_or_path}'),
            }
        # X = torch.fft.rfft(self.d_m[:, -1,:], dim=-1, norm='ortho')   # [B, Q, K, H, D]
        # A = X.abs()
        # A_log = torch.log(A.clamp_min(1e-6))
        # # pdb.set_trace()
        # plot_tensor_grouped_lines(A_log)
    def make_scale_shift_vectors(self, learnable_switch: bool = False):
        shift_list = [0,1,2,3,4,5,6,7]
        # shift_list = [0] * 8
        device = 'cuda'

        if learnable_switch:
            a0 = self.theta_a0.exp().squeeze(0)                        # scalar
            r  = (1.0 + torch.nn.functional.softplus(self.theta_r)).squeeze(0)  # scalar >1
            k  = torch.arange(self.S, device=device, dtype=torch.float32)       # [S]
            scales = a0 * (r ** k)                                     # [S]
        else:
            interval = (self.scale_range[1] - self.scale_range[0]) // 8
            scale_list = [2**i for i in range(self.scale_range[0], self.scale_range[1], interval)]
            scales = torch.tensor(scale_list, dtype=torch.float32, device=device)  # [S]

        s_list = []
        beta_list = []
        for sc in scales:
            for sh in shift_list:
                s_list.append(sc)
                beta_list.append(float(sh))

        s_tensor = torch.stack(s_list).to(torch.float32)
        beta_tensor = torch.tensor(beta_list, dtype=torch.float32, device=device)
        return s_tensor, beta_tensor
    # def make_scale_shift_vectors(self, learnable_switch: bool = False):
    #     """
    #     1) 定义 scale_list = [2^1, 2^2, ..., 2^15]  共 15 个值
    #     2) 定义 shift_list = [0, 32, 64, 96]         共 4 个值
    #     3) 组合成 d_h=15*4=60 维的 s 向量与 beta 向量:
    #     s = [2^1,2^1,2^1,2^1, 2^2,2^2,2^2,2^2, …, 2^15,2^15,2^15,2^15]
    #     beta = [0,32,64,96, 0,32,64,96, …, 0,32,64,96]
    #     返回:
    #     - s_tensor:  shape=(d_h,) dtype=float32
    #     - beta_tensor:shape=(d_h,) dtype=float32
    #     """
    #     interval = (self.scale_range[1] - self.scale_range[0]) // 8
    #     scale_list = [2**i for i in range(self.scale_range[0], self.scale_range[1], interval)]  # [2^1, 2^2, …, 2^15]
    #     shift_list = [0, 1, 2, 3, 4, 5, 6, 7]
    #     s = []
    #     beta = []
    #     for sc in scale_list:
    #         for sh in shift_list:
    #             s.append(float(sc))
    #             if self.config.scale_use_for_shift:
    #                 beta.append(float(sc * sh))
    #             else:
    #                 beta.append(float(sh))

    #     # 现在 len(s) == len(beta) == 15*4 = 60
    #     s_tensor = torch.tensor(s, dtype=torch.float32, device='cuda')
    #     beta_tensor = torch.tensor(beta, dtype=torch.float32, device='cuda')
    #     return s_tensor, beta_tensor
    def make_uniform_scale_shift_vectors(self):
        """
        Log-uniform scale list + fixed shift list

        - scale 在 [2^scale_range[0], 2^scale_range[1]] 上 log-space 均匀
        - scale 个数 = d_h / len(shift_list)
        - 不改变 d_h，不改变 shift_list
        """

        device = 'cuda'

        # -------------------------
        # 1) shift list（保持不变）
        # -------------------------
        shift_list = [0, 1, 2, 3, 4, 5, 6, 7]
        # shift_list = [0] * 8
        num_shifts = len(shift_list)

        # -------------------------
        # 2) scale 个数由 d_h 决定
        # -------------------------
        num_scales = 8

        # -------------------------
        # 3) log-uniform scale list
        # -------------------------
        scale_min = 2 ** self.scale_range[0]
        scale_max = 2 ** self.scale_range[1]

        # log-space 均匀
        scale_list = torch.logspace(
            start=torch.log10(torch.tensor(scale_min, dtype=torch.float32)),
            end=torch.log10(torch.tensor(scale_max, dtype=torch.float32)),
            steps=num_scales,
            base=10.0,
            device=device
        )

        # （可选）如果你希望 scale 是“干净”的整数
        # scale_list = torch.round(scale_list)

        # -------------------------
        # 4) 组合 scale × shift
        # -------------------------
        s = []
        beta = []

        for sc in scale_list:
            for sh in shift_list:
                s.append(sc)
                beta.append(float(sh))

        s_tensor = torch.stack(s).to(device)           # (d_h,)
        beta_tensor = torch.tensor(beta, device=device, dtype=torch.float32)

        return s_tensor, beta_tensor    
    def make_decay_per_dim_custom_bucketed(
        self,
        d_h: int,
        L: int,
        s: torch.Tensor,        # [d_h]
        beta: torch.Tensor,     # [d_h]
        bucket_offsets: torch.Tensor,  # [K] (e.g., [0, 24, 40, ...])
        device: str = "cuda",
        dtype: torch.dtype = torch.float32,
        enable_tqdm: bool = False,
    ):
        """
        Return: p_bucketed [K, d_h, L, L], each is tril() causal.
        Implements: u = (diff - (beta + bucket_offset)) / s
        """
        assert s is not None and beta is not None
        assert s.shape[0] == d_h and beta.shape[0] == d_h

        s = s.to(device=device, dtype=dtype)
        beta = beta.to(device=device, dtype=dtype)
        bucket_offsets = torch.tensor(bucket_offsets, dtype=dtype, device=device)

        # diff[m,n] = m - n
        idx = torch.arange(L, device=device, dtype=dtype)
        diff = idx.view(L, 1) - idx.view(1, L)  # [L,L]

        a = s.view(d_h, 1, 1)       # [d_h,1,1]
        b = beta.view(d_h, 1, 1)    # [d_h,1,1]

        K = bucket_offsets.numel()
        out = torch.empty((K, d_h, L, L), device=device, dtype=dtype)

        it = range(K)
        if enable_tqdm:
            it = tqdm(it, desc="build wavelet_dtt buckets", total=K)

        for k in it:
            t = bucket_offsets[k].view(1, 1, 1)          # scalar broadcast
            u = (diff.unsqueeze(0) - (b + t)) / a        # [d_h,L,L]
            p = (1.0 - u**2) * torch.exp(-0.5 * u**2)    # [d_h,L,L]
            out[k] = p.tril()                            # causal
        return out.view(K, 8, d_h // 8, L, L).mean(dim=2)  # [K, d_h, L, L]
    def make_decay_per_dim_custom(self, d_h: int, L: int, s: torch.Tensor, beta: torch.Tensor, device='cuda'):
        """
         改造后直接输出 p_{m,n} 矩阵，shape=(d_h, L, L)
        """
         # 1) 构造位置差 diff_{m,n} = m - n
        if s is None or beta is None:
            return None
        idx = torch.arange(L, device=device).float()             # (L,)
        m = idx.view(L, 1)                                      # (L,1)
        n = idx.view(1, L)                                      # (1,L)
        diff = m - n                                            # (L,L)

         # 2) 扩展 scale a 和 shift b 到 (d_h,1,1)
        a = s.view(d_h, 1, 1)                                   # (d_h,1,1)
        b = beta.view(d_h, 1, 1)                                # (d_h,1,1)

         # 3) 计算 u = (diff - b) / a
        u = (diff.unsqueeze(0) - b) / a                         # (d_h,L,L)

         # 4) 公式 p_{m,n} = (1 - u^2) * exp(-0.5 * u^2)
        p = (1 - u**2) * torch.exp(-0.5 * u**2)                 # (d_h,L,L)

         # 返回 shape=(d_h, L, L)，自注意力里再做 sqrt(d) 缩放和合并
        return p.tril()
    # def make_decay_per_dim_custom(self,
    #                           d_h: int,
    #                           L: int,
    #                           s: torch.Tensor,
    #                           beta: torch.Tensor,
    #                           device: str = "cuda",
    #                           target_norm: float = 1.0):
    #     idx = torch.arange(L, device=device, dtype=torch.float32)
    #     m = idx.view(L, 1)
    #     n = idx.view(1, L)
    #     diff = m - n  # (L,L)

    #     a = s.view(d_h, 1, 1)
    #     b = beta.view(d_h, 1, 1)

    #     u = (diff.unsqueeze(0) - b) / a              # (d_h, L, L)
    #     p = (1.0 - u**2) * torch.exp(-0.5 * u**2)    # (d_h, L, L)

    #     tril_mask = torch.tril(torch.ones(L, L, device=device, dtype=torch.float32))
    #     mask = tril_mask.unsqueeze(0)                # (1, L, L)

    #     p_causal = p * mask                          # (d_h, L, L)

    #     # 关键一句：对最后一维做 L2 norm 归一
    #     w = torch.nn.functional.normalize(p_causal, p=2, dim=-1, eps=1e-8)
    #     w = w * target_norm

    #     return w   # [d_h, L, L]

    # def make_decay_per_dim_custom(self,
    #                             d_h: int,
    #                             L: int,
    #                             s: torch.Tensor,
    #                             beta: torch.Tensor,
    #                             device: str = "cuda",
    #                             target_m2: float = 1.0):
    #     """
    #     输出 w_{d,i,j}, shape = (d_h, L, L)
    #     - 先按 Ricker wavelet 公式得到 p
    #     - 施加 causal tril mask（只保留下三角 i>=j）
    #     - 再对 mask 后的有效区域做二阶矩归一，让 E[w^2] ~= target_m2
    #     """

    #     # 1) 位置差 diff_{m,n} = m - n
    #     idx = torch.arange(L, device=device, dtype=torch.float32)  # (L,)
    #     m = idx.view(L, 1)                                         # (L,1)
    #     n = idx.view(1, L)                                         # (1,L)
    #     diff = m - n                                               # (L,L)

    #     # 2) 扩展 scale a 和 shift b 到 (d_h,1,1)
    #     a = s.view(d_h, 1, 1)                                      # (d_h,1,1)
    #     b = beta.view(d_h, 1, 1)                                   # (d_h,1,1)

    #     # 3) u = (diff - b) / a
    #     u = (diff.unsqueeze(0) - b) / a                            # (d_h,L,L)

    #     # 4) 原始 Ricker wavelet 分数 p_{d,i,j}
    #     p = (1.0 - u**2) * torch.exp(-0.5 * u**2)                  # (d_h,L,L)

    #     # === 关键变化从这里开始 ===

    #     # 5) causal mask: 只保留下三角 (i >= j)
    #     tril_mask = torch.tril(torch.ones(L, L, device=device, dtype=torch.float32))  # (L,L)
    #     mask = tril_mask.unsqueeze(0)                                                 # (1,L,L)

    #     p_causal = p * mask  # 上三角直接置 0（不会被用到）

    #     # 6) 在 mask 后的有效区域上做「全局二阶矩归一」
    #     #    目标: E[w^2] ~= target_m2，避免整体坍缩或爆炸
    #     eps = 1e-8
    #     # valid 元素数量：d_h * 有效(i,j)个数
    #     num_valid = (mask > 0).sum() * d_h

    #     # 注意只统计有效区域
    #     m2 = (p_causal.pow(2) * mask).sum() / (num_valid + eps)    # 标量

    #     alpha = (target_m2 / (m2 + eps)).sqrt()                    # 标量

    #     w = p_causal * alpha                                       # [d_h,L,L]

    #     return w
    # def make_decay_per_dim_custom(self, d_h: int, L: int,
    #                             s: torch.Tensor, beta: torch.Tensor,
    #                             device='cuda'):
    #     """
    #     输出 p_{m,n}，shape=(d_h, L, L)。
    #     改动：
    #     1) 严格上三角位置（m < n）替换为“每一行的最小值”
    #     2) 对最后一维做 softmax
    #     """
    #     # 1) 位置差 diff_{m,n} = m - n
    #     idx = torch.arange(L, device=device, dtype=torch.float32)  # (L,)
    #     m = idx.view(L, 1)                                         # (L,1)
    #     n = idx.view(1, L)                                         # (1,L)
    #     diff = m - n                                               # (L,L)

    #     # 2) 扩展 scale a 和 shift b 到 (d_h,1,1)
    #     a = s.view(d_h, 1, 1)                                      # (d_h,1,1)
    #     b = beta.view(d_h, 1, 1)                                   # (d_h,1,1)

    #     # 3) u = (diff - b) / a
    #     u = (diff.unsqueeze(0) - b) / a                            # (d_h,L,L)

    #     # 4) 原始分数 p_{m,n} = (1 - u^2) * exp(-0.5 * u^2)
    #     p = (1 - u**2) * torch.exp(-0.5 * u**2)                   # (d_h,L,L)
    #     p = p / (p.norm(dim=-1, keepdim=True) + 1e-12)
    #     # 5) 严格上三角掩码（True 表示 m < n）
    #     # upper_mask = torch.triu(torch.ones(L, L, device=device, dtype=torch.bool), diagonal=1)  # (L,L)

    #     # 6) 每一行（固定 m）最小值，沿最后一维求最小：(d_h, L, 1)
    #     # row_min = p.amin(dim=-1, keepdim=True)

    #     # 7) 将严格上三角置为对应行的最小值
    #     # p = torch.where(upper_mask, row_min, p)

    #     # 8) 对最后一维做 softmax
    #     # p = 2 * torch.sigmoid(p) ###keep consistent with fox paper

    #     # p = torch.sigmoid(p) ###keep consistent with fox paper
    #     # return p - 1

    #     return p
    @add_start_docstrings(PARALLELIZE_DOCSTRING)
    def parallelize(self, device_map=None):
        # Check validity of device_map
        warnings.warn(
            "`GPT2Model.parallelize` is deprecated and will be removed in v5 of Transformers, you should load your"
            " model with `device_map='balanced'` in the call to `from_pretrained`. You can also provide your own"
            " `device_map` but it needs to be a dictionary module_name to device, so for instance {'h.0': 0, 'h.1': 1,"
            " ...}",
            FutureWarning,
        )
        self.device_map = (
            get_device_map(len(self.h), range(torch.cuda.device_count())) if device_map is None else device_map
        )
        assert_device_map(self.device_map, len(self.h))
        self.model_parallel = True
        self.first_device = "cpu" if "cpu" in self.device_map else "cuda:" + str(min(self.device_map.keys()))
        self.last_device = "cuda:" + str(max(self.device_map.keys()))
        self.wte = self.wte.to(self.first_device)
        self.wpe = self.wpe.to(self.first_device)
        # Load onto devices
        for k, v in self.device_map.items():
            for block in v:
                cuda_device = "cuda:" + str(k)
                self.h[block] = self.h[block].to(cuda_device)
        # ln_f to last
        self.ln_f = self.ln_f.to(self.last_device)

    @add_start_docstrings(DEPARALLELIZE_DOCSTRING)
    def deparallelize(self):
        warnings.warn(
            "Like `parallelize`, `deparallelize` is deprecated and will be removed in v5 of Transformers.",
            FutureWarning,
        )
        self.model_parallel = False
        self.device_map = None
        self.first_device = "cpu"
        self.last_device = "cpu"
        self.wte = self.wte.to("cpu")
        self.wpe = self.wpe.to("cpu")
        for index in range(len(self.h)):
            self.h[index] = self.h[index].to("cpu")
        self.ln_f = self.ln_f.to("cpu")
        torch.cuda.empty_cache()

    def get_input_embeddings(self):
        return self.wte

    def set_input_embeddings(self, new_embeddings):
        self.wte = new_embeddings

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer}
        """
        for layer, heads in heads_to_prune.items():
            self.h[layer].attn.prune_heads(heads)

    @auto_docstring
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        geom_p=0,
        **kwargs,
    ) -> Union[tuple, BaseModelOutputWithPastAndCrossAttentions]:
        r"""
        input_ids (`torch.LongTensor` of shape `(batch_size, input_ids_length)`):
            `input_ids_length` = `sequence_length` if `past_key_values` is `None` else
            `past_key_values.get_seq_length()` (`sequence_length` of input past key value states). Indices of input
            sequence tokens in the vocabulary.

            If `past_key_values` is used, only `input_ids` that do not have their past calculated should be passed as
            `input_ids`.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            self.warn_if_padding_and_no_attention_mask(input_ids, attention_mask)
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
            batch_size = input_ids.shape[0]
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
            batch_size = inputs_embeds.shape[0]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if token_type_ids is not None:
            token_type_ids = token_type_ids.view(-1, input_shape[-1])

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        # based on pattern from src/transformers/models/whisper/modeling_whisper.py::WhisperDecoder
        if use_cache:
            if past_key_values is None:
                past_key_values = DynamicCache(config=self.config)
            elif isinstance(past_key_values, tuple):
                logger.warning_once(
                    "Passing a tuple of `past_key_values` is deprecated and will be removed in Transformers v4.53.0. "
                    "You should pass an instance of `Cache` instead, e.g. "
                    "`past_key_values=DynamicCache.from_legacy_cache(past_key_values)`."
                )
                past_key_values = DynamicCache.from_legacy_cache(past_key_values)

            if self.config.add_cross_attention and not isinstance(past_key_values, EncoderDecoderCache):
                past_key_values = EncoderDecoderCache(past_key_values, DynamicCache(config=self.config))

        if inputs_embeds is None:
            inputs_embeds = self.wte(input_ids)

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )
        rope = None
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)
        attn_impl = getattr(self.config, "attn_implementation", getattr(self.config, "_attn_implementation", "eager"))
        if attn_impl in ['path_attn', 'path_attn_wfreq']:
            hidden_states = inputs_embeds
        elif self.config.pe_method == 'rotary':
            assert attn_impl == "eager", "only in eager mode you can use rotary."
            hidden_states = inputs_embeds
        elif self.config.pe_method in ('no_pe', 'wavelet', 'alibi'):
            hidden_states = inputs_embeds
        else:
            position_embeds = self.wpe(position_ids)
            hidden_states = inputs_embeds + position_embeds.to(inputs_embeds.device)
            
        # Attention mask.
        # ._update_causal_mask() and ._prepare_4d_causal_attention_mask_with_cache_position() copied from LlamaModel
        if attention_mask is not None and attention_mask.ndim < 4:
            attention_mask = attention_mask.view(batch_size, -1)

        causal_mask = create_causal_mask(
            config=self.config,
            input_embeds=inputs_embeds,
            attention_mask=attention_mask,
            cache_position=cache_position,
            past_key_values=past_key_values,
            position_ids=position_ids,
        )

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        _use_sdpa = self._attn_implementation == "sdpa" and output_attentions is False and head_mask is None
        if self.config.add_cross_attention and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
            if _use_sdpa:
                encoder_attention_mask = _prepare_4d_attention_mask_for_sdpa(
                    mask=encoder_attention_mask, dtype=inputs_embeds.dtype, tgt_len=input_shape[-1]
                )
            elif self._attn_implementation != "flash_attention_2":
                encoder_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # head_mask has shape n_layer x batch x n_heads x N x N
        head_mask = self.get_head_mask(head_mask, self.config.n_layer)

        if token_type_ids is not None:
            token_type_embeds = self.wte(token_type_ids)
            hidden_states = hidden_states + token_type_embeds

        hidden_states = self.drop(hidden_states)

        output_shape = (-1,) + input_shape[1:] + (hidden_states.size(-1),)

        all_self_attentions = () if output_attentions else None
        all_cross_attentions = () if output_attentions and self.config.add_cross_attention else None
        all_hidden_states = () if output_hidden_states else None
        dis_loss_total = hidden_states.new_zeros([])
        if getattr(self.config, 'scale_type', 'fixed') == 'learnable' and not self._skip_wavelet_decay_table:
            s_tensor, beta_tensor = self.make_scale_shift_vectors(learnable_switch=True)
            self.d_m = self.make_decay_per_dim_custom(64, self.config.block_size, s_tensor, beta_tensor)
        router1_idx_layers = []
        router2_idx_layers = []
        for i, block in enumerate(self.h):
            # Model parallel
            if self.model_parallel:
                torch.cuda.set_device(hidden_states.device)
                if isinstance(head_mask, torch.Tensor):
                    head_mask = head_mask.to(hidden_states.device)
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            outputs, cur_dis_loss, router1, router2 = block(
                hidden_states,
                past_key_values if not (self.gradient_checkpointing and self.training) else None,
                cache_position,
                causal_mask,
                head_mask[i],
                encoder_hidden_states,  # as a positional argument for gradient checkpointing
                encoder_attention_mask=encoder_attention_mask,
                use_cache=use_cache,
                output_attentions=output_attentions,
                attention_mask_2d=attention_mask,  # ← 传给 PaTH 用的 2D 0/1 mask；其他实现会忽略
                rope=rope,
                wavelet_decay_table=self.d_m,
                geom_p=geom_p,
                analyzer=self.analyzer_tools if self.config.analyzer else None,
                # Always pass input_ids when available so eval-time attention exports
                # can recover token_id/token text even when analyzer is disabled.
                input_ids=input_ids,
                # analyzer=self.analyzer if self.config.analyzer else None,
                # scale_wise_analyzer=self.scale_wise_analyzer if self.config.analyzer else None,
                # router_analyzer=self.router_analyzer if self.config.analyzer else None,
                **kwargs,
            )

            router1_idx_layers.append(router1)
            router2_idx_layers.append(router2)
            if cur_dis_loss is not None:
                dis_loss_total += cur_dis_loss
            hidden_states = outputs[0]

            if output_attentions:
                all_self_attentions = all_self_attentions + (outputs[1],)
                if self.config.add_cross_attention:
                    all_cross_attentions = all_cross_attentions + (outputs[2],)

            # Model Parallel: If it's the last layer for that device, put things on the next device
            if self.model_parallel:
                for k, v in self.device_map.items():
                    if i == v[-1] and "cuda:" + str(k) != self.last_device:
                        hidden_states = hidden_states.to("cuda:" + str(k + 1))
        try:
            router1_idx = torch.stack(router1_idx_layers, dim=0)  # [L,B,T,H]
            router2_idx = torch.stack(router2_idx_layers, dim=0)
            self.last_router1_idx = router1_idx.detach()
            self.last_router2_idx = router2_idx.detach()
        
        except:
            pass
        hidden_states = self.ln_f(hidden_states)

        hidden_states = hidden_states.view(output_shape)
        # Add last hidden state
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        past_key_values = past_key_values if use_cache else None
        if not return_dict:
            return tuple(
                v
                for v in [hidden_states, past_key_values, all_hidden_states, all_self_attentions, all_cross_attentions]
                if v is not None
            )

        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            cross_attentions=all_cross_attentions,
        ), dis_loss_total


@auto_docstring(
    custom_intro="""
    The GPT2 Model transformer with a language modeling head on top (linear layer with weights tied to the input
    embeddings).
    """
)
class GPT2LMHeadModel(GPT2PreTrainedModel, GenerationMixin):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.transformer = GPT2Model(config)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # Model parallel
        self.model_parallel = False
        self.device_map = None

        # Initialize weights and apply final processing
        self.post_init()

    @add_start_docstrings(PARALLELIZE_DOCSTRING)
    def parallelize(self, device_map=None):
        warnings.warn(
            "`GPT2LMHeadModel.parallelize` is deprecated and will be removed in v5 of Transformers, you should load"
            " your model with `device_map='balanced'` in the call to `from_pretrained`. You can also provide your own"
            " `device_map` but it needs to be a dictionary module_name to device, so for instance {'transformer.h.0':"
            " 0, 'transformer.h.1': 1, ...}",
            FutureWarning,
        )
        self.device_map = (
            get_device_map(len(self.transformer.h), range(torch.cuda.device_count()))
            if device_map is None
            else device_map
        )
        assert_device_map(self.device_map, len(self.transformer.h))
        self.transformer.parallelize(self.device_map)
        self.lm_head = self.lm_head.to(self.transformer.first_device)
        self.model_parallel = True

    @add_start_docstrings(DEPARALLELIZE_DOCSTRING)
    def deparallelize(self):
        warnings.warn(
            "Like `parallelize`, `deparallelize` is deprecated and will be removed in v5 of Transformers.",
            FutureWarning,
        )
        self.transformer.deparallelize()
        self.transformer = self.transformer.to("cpu")
        self.lm_head = self.lm_head.to("cpu")
        self.model_parallel = False
        torch.cuda.empty_cache()

    @auto_docstring
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        geom_p = 0,
        **kwargs,
    ) -> Union[tuple, CausalLMOutputWithCrossAttentions]:
        r"""
        input_ids (`torch.LongTensor` of shape `(batch_size, input_ids_length)`):
            `input_ids_length` = `sequence_length` if `past_key_values` is `None` else
            `past_key_values.get_seq_length()` (`sequence_length` of input past key value states). Indices of input
            sequence tokens in the vocabulary.

            If `past_key_values` is used, only `input_ids` that do not have their past calculated should be passed as
            `input_ids`.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)
        labels (`torch.LongTensor` of shape `(batch_size, input_ids_length)`, *optional*):
            Labels for language modeling. Note that the labels **are shifted** inside the model, i.e. you can set
            `labels = input_ids` Indices are selected in `[-100, 0, ..., config.vocab_size]` All labels set to `-100`
            are ignored (masked), the loss is only computed for labels in `[0, ..., config.vocab_size]`
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        transformer_outputs, dis_loss = self.transformer(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            cache_position=cache_position,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            geom_p=geom_p,
        )
        hidden_states = transformer_outputs[0]
        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.transformer.first_device)
            hidden_states = hidden_states.to(self.lm_head.weight.device)

        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            # Flatten the tokens
            loss = self.loss_function(
                logits,
                labels,
                vocab_size=self.config.vocab_size,
                **kwargs,
            )
        # Keep auxiliary dis_loss (including router entropy regularization) explicit in
        # training objective when LM loss exists.
        if loss is not None:
            loss = loss + (float(geom_p) * dis_loss)

        # loss += self.dis_loss_coe * dis_loss
        if not return_dict:
            output = (logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutputWithCrossAttentions(
            loss=loss,
            auxiliary_loss=dis_loss,
            logits=logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
            cross_attentions=transformer_outputs.cross_attentions,
        )


@auto_docstring(
    custom_intro="""
        The GPT2 Model transformer with a language modeling and a multiple-choice classification head on top e.g. for
    RocStories/SWAG tasks. The two heads are two linear layers. The language modeling head has its weights tied to the
    input embeddings, the classification head takes as input the input of a specified classification token index in the
    input sequence).
    """
)
class GPT2DoubleHeadsModel(GPT2PreTrainedModel, GenerationMixin):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        config.num_labels = 1
        self.transformer = GPT2Model(config)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.multiple_choice_head = GPT2SequenceSummary(config)

        # Model parallel
        self.model_parallel = False
        self.device_map = None

        # Initialize weights and apply final processing
        self.post_init()

    @add_start_docstrings(PARALLELIZE_DOCSTRING)
    def parallelize(self, device_map=None):
        warnings.warn(
            "`GPT2DoubleHeadsModel.parallelize` is deprecated and will be removed in v5 of Transformers, you should"
            " load your model with `device_map='balanced'` in the call to `from_pretrained`. You can also provide your"
            " own `device_map` but it needs to be a dictionary module_name to device, so for instance"
            " {'transformer.h.0': 0, 'transformer.h.1': 1, ...}",
            FutureWarning,
        )
        self.device_map = (
            get_device_map(len(self.transformer.h), range(torch.cuda.device_count()))
            if device_map is None
            else device_map
        )
        assert_device_map(self.device_map, len(self.transformer.h))
        self.transformer.parallelize(self.device_map)
        self.lm_head = self.lm_head.to(self.transformer.first_device)
        self.multiple_choice_head = self.multiple_choice_head.to(self.transformer.first_device)
        self.model_parallel = True

    @add_start_docstrings(DEPARALLELIZE_DOCSTRING)
    def deparallelize(self):
        warnings.warn(
            "Like `parallelize`, `deparallelize` is deprecated and will be removed in v5 of Transformers.",
            FutureWarning,
        )
        self.transformer.deparallelize()
        self.transformer = self.transformer.to("cpu")
        self.lm_head = self.lm_head.to("cpu")
        self.multiple_choice_head = self.multiple_choice_head.to("cpu")
        self.model_parallel = False
        torch.cuda.empty_cache()

    @auto_docstring
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        mc_token_ids: Optional[torch.LongTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        mc_labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> Union[tuple, GPT2DoubleHeadsModelOutput]:
        r"""
        input_ids (`torch.LongTensor` of shape `(batch_size, input_ids_length)`):
            `input_ids_length` = `sequence_length` if `past_key_values` is `None` else
            `past_key_values.get_seq_length()` (`sequence_length` of input past key value states). Indices of input
            sequence tokens in the vocabulary.

            If `past_key_values` is used, only `input_ids` that do not have their past calculated should be passed as
            `input_ids`.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)
        mc_token_ids (`torch.LongTensor` of shape `(batch_size, num_choices)`, *optional*, default to index of the last token of the input):
            Index of the classification token in each input sequence. Selected in the range `[0, input_ids.size(-1) -
            1]`.
        labels (`torch.LongTensor` of shape `(batch_size, input_ids_length)`, *optional*):
            Labels for language modeling. Note that the labels **are shifted** inside the model, i.e. you can set
            `labels = input_ids`. Indices are selected in `[-100, 0, ..., config.vocab_size - 1]`. All labels set to
            `-100` are ignored (masked), the loss is only computed for labels in `[0, ..., config.vocab_size - 1]`
        mc_labels (`torch.LongTensor` of shape `(batch_size)`, *optional*):
            Labels for computing the multiple choice classification loss. Indices should be in `[0, ..., num_choices]`
            where *num_choices* is the size of the second dimension of the input tensors. (see *input_ids* above)

        Example:

        ```python
        >>> import torch
        >>> from transformers import AutoTokenizer, GPT2DoubleHeadsModel

        >>> tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
        >>> model = GPT2DoubleHeadsModel.from_pretrained("openai-community/gpt2")

        >>> # Add a [CLS] to the vocabulary (we should train it also!)
        >>> num_added_tokens = tokenizer.add_special_tokens({"cls_token": "[CLS]"})
        >>> # Update the model embeddings with the new vocabulary size
        >>> embedding_layer = model.resize_token_embeddings(len(tokenizer))

        >>> choices = ["Hello, my dog is cute [CLS]", "Hello, my cat is cute [CLS]"]
        >>> encoded_choices = [tokenizer.encode(s) for s in choices]
        >>> cls_token_location = [tokens.index(tokenizer.cls_token_id) for tokens in encoded_choices]

        >>> input_ids = torch.tensor(encoded_choices).unsqueeze(0)  # Batch size: 1, number of choices: 2
        >>> mc_token_ids = torch.tensor([cls_token_location])  # Batch size: 1

        >>> outputs = model(input_ids, mc_token_ids=mc_token_ids)
        >>> lm_logits = outputs.logits
        >>> mc_logits = outputs.mc_logits
        ```"""
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        transformer_outputs = self.transformer(
            input_ids,
            past_key_values=past_key_values,
            cache_position=cache_position,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = transformer_outputs[0]

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.transformer.first_device)
            hidden_states = hidden_states.to(self.lm_head.weight.device)

        lm_logits = self.lm_head(hidden_states)
        mc_logits = self.multiple_choice_head(hidden_states, mc_token_ids).squeeze(-1)

        mc_loss = None
        if mc_labels is not None:
            loss_fct = CrossEntropyLoss()
            mc_loss = loss_fct(mc_logits.view(-1, mc_logits.size(-1)), mc_labels.view(-1))
        lm_loss = None
        if labels is not None:
            labels = labels.to(lm_logits.device)
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = CrossEntropyLoss()
            lm_loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        if not return_dict:
            output = (lm_logits, mc_logits) + transformer_outputs[1:]
            if mc_loss is not None:
                output = (mc_loss,) + output
            return ((lm_loss,) + output) if lm_loss is not None else output

        return GPT2DoubleHeadsModelOutput(
            loss=lm_loss,
            mc_loss=mc_loss,
            logits=lm_logits,
            mc_logits=mc_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )


@auto_docstring(
    custom_intro="""
    The GPT2 Model transformer with a sequence classification head on top (linear layer).

    [`GPT2ForSequenceClassification`] uses the last token in order to do the classification, as other causal models
    (e.g. GPT-1) do.

    Since it does classification on the last token, it requires to know the position of the last token. If a
    `pad_token_id` is defined in the configuration, it finds the last token that is not a padding token in each row. If
    no `pad_token_id` is defined, it simply takes the last value in each row of the batch. Since it cannot guess the
    padding tokens when `inputs_embeds` are passed instead of `input_ids`, it does the same (take the last value in
    each row of the batch).
    """
)
class GPT2ForSequenceClassification(GPT2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.transformer = GPT2Model(config)
        self.score = nn.Linear(config.n_embd, self.num_labels, bias=False)

        # Model parallel
        self.model_parallel = False
        self.device_map = None

        # Initialize weights and apply final processing
        self.post_init()

    @auto_docstring
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[tuple, SequenceClassifierOutputWithPast]:
        r"""
        input_ids (`torch.LongTensor` of shape `(batch_size, input_ids_length)`):
            `input_ids_length` = `sequence_length` if `past_key_values` is `None` else
            `past_key_values.get_seq_length()` (`sequence_length` of input past key value states). Indices of input
            sequence tokens in the vocabulary.

            If `past_key_values` is used, only `input_ids` that do not have their past calculated should be passed as
            `input_ids`.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        transformer_outputs = self.transformer(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = transformer_outputs[0]
        logits = self.score(hidden_states)

        if input_ids is not None:
            batch_size, sequence_length = input_ids.shape[:2]
        else:
            batch_size, sequence_length = inputs_embeds.shape[:2]

        if self.config.pad_token_id is None and batch_size != 1:
            raise ValueError("Cannot handle batch sizes > 1 if no padding token is defined.")
        if self.config.pad_token_id is None:
            last_non_pad_token = -1
        elif input_ids is not None:
            # To handle both left- and right- padding, we take the rightmost token that is not equal to pad_token_id
            non_pad_mask = (input_ids != self.config.pad_token_id).to(logits.device, torch.int32)
            token_indices = torch.arange(input_ids.shape[-1], device=logits.device, dtype=torch.int32)
            last_non_pad_token = (token_indices * non_pad_mask).argmax(-1)
        else:
            last_non_pad_token = -1
            logger.warning_once(
                f"{self.__class__.__name__} will not detect padding tokens in `inputs_embeds`. Results may be "
                "unexpected if using padding tokens in conjunction with `inputs_embeds.`"
            )

        pooled_logits = logits[torch.arange(batch_size, device=logits.device), last_non_pad_token]

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(pooled_logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(pooled_logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(pooled_logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(pooled_logits, labels)
        if not return_dict:
            output = (pooled_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutputWithPast(
            loss=loss,
            logits=pooled_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )


@auto_docstring
class GPT2ForTokenClassification(GPT2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.transformer = GPT2Model(config)
        if hasattr(config, "classifier_dropout") and config.classifier_dropout is not None:
            classifier_dropout = config.classifier_dropout
        elif hasattr(config, "hidden_dropout") and config.hidden_dropout is not None:
            classifier_dropout = config.hidden_dropout
        else:
            classifier_dropout = 0.1
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        # Model parallel
        self.model_parallel = False
        self.device_map = None

        # Initialize weights and apply final processing
        self.post_init()

    @auto_docstring
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[tuple, TokenClassifierOutput]:
        r"""
        input_ids (`torch.LongTensor` of shape `(batch_size, input_ids_length)`):
            `input_ids_length` = `sequence_length` if `past_key_values` is `None` else
            `past_key_values.get_seq_length()` (`sequence_length` of input past key value states). Indices of input
            sequence tokens in the vocabulary.

            If `past_key_values` is used, only `input_ids` that do not have their past calculated should be passed as
            `input_ids`.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        transformer_outputs = self.transformer(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = transformer_outputs[0]
        hidden_states = self.dropout(hidden_states)
        logits = self.classifier(hidden_states)

        loss = None
        if labels is not None:
            labels = labels.to(logits.device)
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + transformer_outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )


@auto_docstring
class GPT2ForQuestionAnswering(GPT2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.transformer = GPT2Model(config)
        self.qa_outputs = nn.Linear(config.hidden_size, 2)

        # Model parallel
        self.model_parallel = False
        self.device_map = None

        # Initialize weights and apply final processing
        self.post_init()

    @auto_docstring
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        start_positions: Optional[torch.LongTensor] = None,
        end_positions: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[tuple, QuestionAnsweringModelOutput]:
        r"""
        input_ids (`torch.LongTensor` of shape `(batch_size, input_ids_length)`):
            `input_ids_length` = `sequence_length` if `past_key_values` is `None` else
            `past_key_values.get_seq_length()` (`sequence_length` of input past key value states). Indices of input
            sequence tokens in the vocabulary.

            If `past_key_values` is used, only `input_ids` that do not have their past calculated should be passed as
            `input_ids`.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.transformer(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]

        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1).contiguous()
        end_logits = end_logits.squeeze(-1).contiguous()

        total_loss = None
        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1).to(start_logits.device)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1).to(end_logits.device)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions = start_positions.clamp(0, ignored_index)
            end_positions = end_positions.clamp(0, ignored_index)

            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2

        if not return_dict:
            output = (start_logits, end_logits) + outputs[2:]
            return ((total_loss,) + output) if total_loss is not None else output

        return QuestionAnsweringModelOutput(
            loss=total_loss,
            start_logits=start_logits,
            end_logits=end_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


__all__ = [
    "GPT2DoubleHeadsModel",
    "GPT2ForQuestionAnswering",
    "GPT2ForSequenceClassification",
    "GPT2ForTokenClassification",
    "GPT2LMHeadModel",
    "GPT2Model",
    "GPT2PreTrainedModel",
]
