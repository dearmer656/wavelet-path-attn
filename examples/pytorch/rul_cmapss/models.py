"""RUL prediction model definitions."""
import sys
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

# Add repo root to path so we can import transformers from src/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'src'))
from transformers.models.gpt2.modeling_gpt2 import GPT2Model
from transformers.models.gpt2.configuration_gpt2 import GPT2Config


class GPT2RULBase(nn.Module):
    """Base class for GPT-2-backed RUL regression models.
    
    Replaces the token embedding with a linear sensor projection and
    adds a scalar regression head on the last-token hidden state.
    """

    def __init__(
        self,
        n_features: int = 24,
        n_layer: int = 2,
        num_heads: int = 4,
        head_dim: int = 16,
        dropout: float = 0.1,
        max_position_embeddings: int = 256,
        attn_implementation: str = 'eager',
        pe_method: str = 'rotary',
    ):
        super().__init__()
        hidden_size = num_heads * head_dim
        self.sensor_proj = nn.Linear(n_features, hidden_size)
        config = GPT2Config(
            n_embd=hidden_size,
            n_head=num_heads,
            n_layer=n_layer,
            n_positions=max_position_embeddings,
            attn_pdrop=dropout,
            resid_pdrop=dropout,
            attn_implementation=attn_implementation,
            pe_method=pe_method,
            use_cache=False,
            vocab_size=50257,
            # safe defaults so GPT2Model.__init__ never accesses missing wavelet/path fields
            wavelet_mode='off',
            scale_type='none',
            scale_range=[0, 16],
            analyzer=False,
            block_size=max_position_embeddings,
        )
        self.backbone = GPT2Model(config)
        self.reg_head = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.FloatTensor, attention_mask: torch.FloatTensor | None = None) -> torch.FloatTensor:
        h = self.sensor_proj(x)                                           # [B, T, hidden]
        outputs = self.backbone(inputs_embeds=h, attention_mask=attention_mask, return_dict=False)
        hidden = outputs[0]                                               # [B, T, hidden]
        if attention_mask is not None:
            # right-padded mask: [1,1,...,1,0,...,0]; last real pos = sum-1
            last_idx = (attention_mask.sum(dim=1).long() - 1).clamp(min=0)  # [B]
            last_h = hidden[torch.arange(hidden.size(0), device=hidden.device), last_idx]
        else:
            last_h = hidden[:, -1, :]                                     # [B, hidden]
        return self.reg_head(last_h).squeeze(-1)                          # [B]


class PaTHRUL(GPT2RULBase):
    """PaTH attention (no PE) RUL regression model."""

    def __init__(self, **kwargs):
        kwargs['attn_implementation'] = 'path_attn'
        kwargs['pe_method'] = 'no_pe'
        super().__init__(**kwargs)


class TransformerRUL(GPT2RULBase):
    """Standard Transformer (eager attention) RUL regression model.
    
    pe_method choices: 'rotary' (RoPE), 'alibi' (ALiBi), 'no_pe' (NoPE).
    """

    def __init__(self, pe_method: str = 'rotary', **kwargs):
        kwargs['attn_implementation'] = 'eager'
        kwargs['pe_method'] = pe_method
        super().__init__(**kwargs)


class LSTM_RUL(nn.Module):
    """Stacked LSTM RUL regression model."""

    def __init__(
        self,
        n_features: int = 24,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.reg_head = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.FloatTensor, attention_mask: torch.FloatTensor | None = None, **kwargs) -> torch.FloatTensor:
        if attention_mask is not None:
            lengths = attention_mask.sum(dim=1).long().cpu()
            packed = nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
            _, (h_n, _) = self.lstm(packed)
        else:
            _, (h_n, _) = self.lstm(x)
        return self.reg_head(h_n[-1]).squeeze(-1)


class _CausalConvBlock(nn.Module):
    """Single dilated causal conv → ReLU → Dropout block."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dilation: int, dropout: float):
        super().__init__()
        self.left_pad = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, dilation=dilation, padding=0)
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        x = F.pad(x, (self.left_pad, 0))
        return self.drop(self.relu(self.conv(x)))


class TCN_RUL(nn.Module):
    """Dilated causal TCN RUL regression model."""

    def __init__(
        self,
        n_features: int = 24,
        channels: tuple = (64, 64, 64),
        kernel_size: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        dilations = (1, 2, 4)
        in_ch = n_features
        blocks = []
        for out_ch, dil in zip(channels, dilations):
            blocks.append(_CausalConvBlock(in_ch, out_ch, kernel_size, dil, dropout))
            in_ch = out_ch
        self.blocks = nn.Sequential(*blocks)
        self.reg_head = nn.Linear(channels[-1], 1)

    def forward(self, x: torch.FloatTensor, attention_mask: torch.FloatTensor | None = None, **kwargs) -> torch.FloatTensor:
        hidden = self.blocks(x.transpose(1, 2))           # [B, channels[-1], T]
        if attention_mask is not None:
            last_idx = (attention_mask.sum(dim=1).long() - 1).clamp(min=0)   # [B]
            last_h = hidden[torch.arange(hidden.size(0), device=hidden.device), :, last_idx]
        else:
            last_h = hidden[:, :, -1]
        return self.reg_head(last_h).squeeze(-1)          # [B]
