"""Sanity test for wavelet relative PE (pe_method='wavelet', relative_type='4')."""
import torch
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../src"))

from transformers import GPT2Config, GPT2Model


def make_config(block_size=64, **kwargs):
    cfg = GPT2Config(
        vocab_size=1000,
        n_positions=block_size,
        n_embd=768,
        n_layer=2,
        n_head=12,         # head_dim = 768/12 = 64
        pe_method="wavelet",
        relative_type="4",
        wavelet_router=False,
        **kwargs,
    )
    cfg._attn_implementation = "eager"
    # extra attrs used in GPT2Model.__init__
    cfg.scale_range = [0, 16]
    cfg.wavelet_mode = "logit_bias_ctxscale_shift_v0"
    cfg.router_band_num = 8
    cfg.analyzer = False
    cfg.scale_type = "fixed"
    cfg.block_size = block_size
    cfg.use_bucketed_decay = False
    return cfg


def test_buffer_shape():
    cfg = make_config(block_size=64)
    model = GPT2Model(cfg)
    attn = model.h[0].attn
    W = attn.wavelet_relative_tensor
    assert W is not None, "wavelet_relative_tensor should not be None"
    head_dim = cfg.n_embd // cfg.n_head  # 64
    assert W.shape == (head_dim, cfg.n_positions, cfg.n_positions), (
        f"Expected ({head_dim}, {cfg.n_positions}, {cfg.n_positions}), got {W.shape}"
    )
    print(f"  buffer shape OK: {W.shape}")


def test_forward_shape():
    cfg = make_config(block_size=32)
    model = GPT2Model(cfg).eval()
    B, T = 2, 20
    input_ids = torch.randint(0, cfg.vocab_size, (B, T))
    with torch.no_grad():
        out = model(input_ids=input_ids)
    # GPT2Model returns (BaseModelOutputWithPastAndCrossAttentions, dis_loss)
    model_out = out[0] if isinstance(out, tuple) else out
    last_hidden = model_out.last_hidden_state
    assert last_hidden.shape == (B, T, cfg.n_embd), (
        f"Unexpected output shape: {last_hidden.shape}"
    )
    print(f"  forward output shape OK: {last_hidden.shape}")


def test_ricker_values():
    """psi(0) should be 1.0 (center of Ricker wavelet)."""
    u = torch.tensor(0.0)
    psi = (1.0 - u**2) * torch.exp(-0.5 * u**2)
    assert abs(psi.item() - 1.0) < 1e-6, f"psi(0) = {psi.item()}, expected 1.0"
    print(f"  psi(0) = {psi.item():.6f} OK")


if __name__ == "__main__":
    print("test_ricker_values ...")
    test_ricker_values()
    print("test_buffer_shape ...")
    test_buffer_shape()
    print("test_forward_shape ...")
    test_forward_shape()
    print("All tests passed.")
