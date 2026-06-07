#!/usr/bin/env python3
"""
dump_path_nmf.py — PAT-186: NMF-based latent motif decomposition of PaTH pre-softmax logits.

Pipeline (L1 pilot, L512 only):
  1. Load GPT-2 small PaTH checkpoint with _nmf_capture=True on PaTHAttention layers.
  2. For each case, run forward pass, collect (E_base_raw * scale) per layer/head.
  3. Preprocess each [T,T] logit map:
       robust ReLU normalization (row-wise, causal-valid positions only)
       → masked sum pooling to MxM
       → optional salient/top-pct filtering
       → map-level L1 normalization
       → flatten to [M*M]
  4. Stack rows into X [N_maps, M*M]; fit sklearn NMF (R=16, init=nndsvda).
  5. Save U.npy, V.npy, meta.json, row_meta.json, basis PNGs, usage CSV.

Capture requires the 2-line _nmf_capture patch in path_attn.py:
    if getattr(self, '_nmf_capture', False):
        self._nmf_last_base_logits = (E_base_raw * scale).detach().to(torch.float32).cpu()
inserted just before `P_base = None` in path_attention_with_wavelet_QH.
"""

import argparse
import csv
import json
import math
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
from sklearn.decomposition import NMF as SklearnNMF
from transformers import AutoModelForCausalLM, AutoTokenizer


# ── preprocessing helpers ─────────────────────────────────────────────────────

def robust_relu_normalize(logit_mat: torch.Tensor) -> torch.Tensor:
    """Row-wise robust ReLU normalization over causal-valid positions (j ≤ i).

    Returns X_raw [T, T] with non-negative values; upper triangle zeroed.
    """
    T = logit_mat.shape[0]
    out = torch.zeros(T, T, dtype=torch.float32)
    for i in range(T):
        vals = logit_mat[i, :i + 1].float()
        if vals.numel() == 0:
            continue
        m_i = float(vals.median().item())
        mad = float((vals - m_i).abs().median().item())
        s_i = 1.4826 * mad + 1e-6
        out[i, :i + 1] = (vals - m_i).div(s_i).clamp_min_(0.0)
    return out


def masked_sum_pool_2d(X_raw: torch.Tensor, M: int = 128) -> torch.Tensor:
    """Pool [T, T] lower-triangular map to [M, M] by masked sum over causal cells."""
    T = X_raw.shape[0]
    row_bin = torch.arange(T) * M // T   # [T], each in [0, M)
    col_bin = torch.arange(T) * M // T

    # Build flat fine-grid indices and causal validity mask
    fi = torch.arange(T).unsqueeze(1).expand(T, T).reshape(-1)   # fine row i
    fj = torch.arange(T).unsqueeze(0).expand(T, T).reshape(-1)   # fine col j
    valid = fj <= fi                                               # causal mask

    ri = row_bin.unsqueeze(1).expand(T, T).reshape(-1)
    ci = col_bin.unsqueeze(0).expand(T, T).reshape(-1)
    flat_idx = (ri * M + ci)[valid]
    flat_vals = X_raw.reshape(-1)[valid]

    out = torch.zeros(M * M, dtype=torch.float32)
    out.scatter_add_(0, flat_idx, flat_vals)
    return out.reshape(M, M)


def salient_mass_filter(pool_map: torch.Tensor, alpha: float = 0.90) -> torch.Tensor:
    """Row-wise cumulative-mass filter: zero cells below the alpha-mass threshold."""
    out = torch.zeros_like(pool_map)
    for r in range(pool_map.shape[0]):
        row = pool_map[r]
        row_sum = row.sum().item()
        if row_sum < 1e-12:
            continue
        sorted_vals, _ = torch.sort(row, descending=True)
        cumsum = torch.cumsum(sorted_vals, dim=0)
        nz = (cumsum >= alpha * row_sum).nonzero(as_tuple=False)
        if nz.numel() == 0:
            out[r] = row
            continue
        k = int(nz[0].item())
        thresh = float(sorted_vals[k].item()) if k < sorted_vals.shape[0] else 0.0
        out[r] = row.clamp_min(thresh) * (row >= thresh).float()
    return out


def top_pct_filter(pool_map: torch.Tensor, keep_frac: float) -> torch.Tensor:
    """Keep the top `keep_frac` fraction of cells by value (per map, not per row)."""
    flat = pool_map.reshape(-1)
    pos = flat[flat > 0]
    if pos.numel() == 0:
        return torch.zeros_like(pool_map)
    thresh = float(torch.quantile(pos, 1.0 - keep_frac).item())
    return pool_map * (pool_map >= thresh).float()


def l1_normalize_map(pool_map: torch.Tensor) -> torch.Tensor:
    s = pool_map.sum().item()
    if s < 1e-12:
        return torch.zeros_like(pool_map)
    return pool_map / s


def preprocess_logit_map(
    logit_mat: torch.Tensor,
    M: int = 128,
    preprocessing: str = "salient",
) -> Optional[torch.Tensor]:
    """Full preprocessing for one [T, T] logit map → flattened [M*M] vector."""
    X_raw = robust_relu_normalize(logit_mat)
    pool = masked_sum_pool_2d(X_raw, M=M)
    if preprocessing == "salient":
        pool = salient_mass_filter(pool, alpha=0.90)
    elif preprocessing == "top10":
        pool = top_pct_filter(pool, keep_frac=0.10)
    elif preprocessing == "top20":
        pool = top_pct_filter(pool, keep_frac=0.20)
    # "dense" = no filtering
    pool = l1_normalize_map(pool)
    if pool.sum().item() < 1e-12:
        return None
    return pool.reshape(-1).float()


# ── text rendering (matches eval_hotpot_long.py) ─────────────────────────────

def render_doc(title: str, sentences: list) -> str:
    return f"Title: {title}\n{' '.join(sentences).strip()}\n\n"


def render_prompt(ex: dict) -> str:
    ctx = "".join(render_doc(t, s) for t, s in ex["context"])
    return f"Question: {ex['question']}\n\nContext:\n{ctx}Answer:"


# ── main ──────────────────────────────────────────────────────────────────────

def run(args):
    checkpoint = str(args.checkpoint)
    seq_len = int(args.seq_len)
    n_case = int(args.n_case)
    rank = int(args.rank)
    M = int(args.pool_size)
    preprocessing = args.preprocessing
    jsonl_path = Path(args.jsonl)
    out_root = Path(args.out_root)
    assert preprocessing in ("salient", "dense", "top10", "top20"), preprocessing

    # Load cases
    cases = []
    with open(jsonl_path) as f:
        for line in f:
            rec = json.loads(line)
            if rec["meta"].get("target_total_tokens") == seq_len:
                cases.append(rec)
                if len(cases) >= n_case:
                    break
    print(f"Loaded {len(cases)} cases for seq_len={seq_len}")

    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading model from {checkpoint} on {device}")
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, use_fast=False)
    model = AutoModelForCausalLM.from_pretrained(
        checkpoint,
        dtype=torch.float32,
        trust_remote_code=True,
        attn_implementation="path_attn",
    )
    model.eval().to(device)

    # Enable NMF logit capture on all PaTHAttention core layers
    # Model structure: transformer.h[i].attn (GPT2PaTHAttention) → .core (PaTHAttention)
    path_attn_layers = []
    for name, module in model.named_modules():
        if type(module).__name__ == "PaTHAttention":
            module._nmf_capture = True
            path_attn_layers.append((name, module))

    n_layers = len(path_attn_layers)
    print(f"Found {n_layers} PaTHAttention layers with _nmf_capture=True")
    if n_layers == 0:
        raise RuntimeError("No PaTHAttention layers found; check model and PYTHONPATH.")

    # Output directory
    ckpt_tag = Path(checkpoint).name  # e.g. checkpoint-15900
    run_tag = f"L{seq_len}_{ckpt_tag}_{preprocessing}_R{rank}"
    out_dir = out_root / run_tag
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output dir: {out_dir}")

    # Accumulate preprocessed rows
    X_rows: List[np.ndarray] = []
    row_meta: List[dict] = []

    for case_idx, rec in enumerate(cases):
        prompt_str = render_prompt(rec)
        answer_str = f" {rec['answer']}"
        prompt_ids = tokenizer(prompt_str, add_special_tokens=False)["input_ids"]
        answer_ids = tokenizer(answer_str, add_special_tokens=False)["input_ids"]
        full_ids = (prompt_ids + answer_ids)[:seq_len]
        actual_len = len(full_ids)
        input_ids = torch.tensor([full_ids], dtype=torch.long, device=device)

        with torch.no_grad():
            _ = model(input_ids)

        for layer_idx, (_, module) in enumerate(path_attn_layers):
            buf = getattr(module, "_nmf_last_base_logits", None)
            if buf is None:
                continue
            # buf: [B, H, T, T] on CPU float32
            attn = buf[0, :, :actual_len, :actual_len]  # [H, T, T]
            n_heads = attn.shape[0]
            for h in range(n_heads):
                row = preprocess_logit_map(attn[h], M=M, preprocessing=preprocessing)
                if row is None:
                    continue
                X_rows.append(row.numpy())
                row_meta.append({"case_id": case_idx, "layer": layer_idx, "head": h})

        if (case_idx + 1) % 5 == 0 or (case_idx + 1) == len(cases):
            print(f"  case {case_idx + 1}/{len(cases)} | rows so far: {len(X_rows)}")

        if device.type == "cuda":
            torch.cuda.empty_cache()

    print(f"Total maps: {len(X_rows)}")
    if not X_rows:
        print("ERROR: no valid rows — check logit capture and preprocessing")
        return

    X = np.stack(X_rows, axis=0).astype(np.float32)  # [N_maps, M*M]
    print(f"X shape: {X.shape}, min={X.min():.4f}, max={X.max():.4f}, "
          f"nnz_frac={float((X > 0).mean()):.3f}")

    # Fit NMF
    print(f"Fitting NMF R={rank}, init=nndsvda ...")
    nmf = SklearnNMF(n_components=rank, init="nndsvda", max_iter=500,
                     random_state=42, verbose=1)
    U = nmf.fit_transform(X)   # [N_maps, R]
    V = nmf.components_        # [R, M*M]
    recon_err = float(nmf.reconstruction_err_)
    print(f"Reconstruction error: {recon_err:.6f}")

    # Save
    np.save(out_dir / "U.npy", U.astype(np.float32))
    np.save(out_dir / "V.npy", V.astype(np.float32))
    with open(out_dir / "row_meta.json", "w") as f:
        json.dump(row_meta, f)
    meta = {
        "checkpoint": checkpoint,
        "seq_len": seq_len,
        "n_case": len(cases),
        "rank": rank,
        "pool_size": M,
        "preprocessing": preprocessing,
        "n_maps": len(X_rows),
        "n_layers": n_layers,
        "reconstruction_err": recon_err,
    }
    with open(out_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    # Basis heatmaps
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        basis_dir = out_dir / "basis_heatmaps"
        basis_dir.mkdir(exist_ok=True)
        for r in range(rank):
            basis = V[r].reshape(M, M)
            vmax = float(np.quantile(basis, 0.995))
            fig, ax = plt.subplots(figsize=(4, 4))
            im = ax.imshow(basis, cmap="viridis", vmin=0, vmax=max(vmax, 1e-8))
            ax.set_title(f"Basis {r} ({preprocessing})", fontsize=9)
            plt.colorbar(im, ax=ax)
            plt.tight_layout()
            plt.savefig(basis_dir / f"basis_{r:02d}.png", dpi=80)
            plt.close(fig)
        print(f"Saved {rank} basis PNGs to {basis_dir}")
    except Exception as e:
        print(f"Warning: heatmap export failed: {e}")

    # Usage aggregation by layer/head
    meta_arr = np.array([[m["case_id"], m["layer"], m["head"]] for m in row_meta])
    usage_rows = []
    for layer in range(n_layers):
        for head in range(12):
            mask = (meta_arr[:, 1] == layer) & (meta_arr[:, 2] == head)
            if mask.sum() == 0:
                continue
            U_sub = U[mask]
            for r in range(rank):
                usage_rows.append({
                    "layer": layer,
                    "head": head,
                    "component": r,
                    "mean_usage": float(U_sub[:, r].mean()),
                    "std_usage": float(U_sub[:, r].std()),
                })
    usage_path = out_dir / "usage_by_layer_head.csv"
    with open(usage_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["layer", "head", "component",
                                               "mean_usage", "std_usage"])
        writer.writeheader()
        writer.writerows(usage_rows)
    print(f"Saved usage table to {usage_path}")
    print(f"Done. Output: {out_dir}")


def main():
    parser = argparse.ArgumentParser(description="PAT-186: NMF motif decomp of PaTH logits")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--jsonl", required=True)
    parser.add_argument("--out_root", required=True)
    parser.add_argument("--seq_len", type=int, default=512)
    parser.add_argument("--n_case", type=int, default=20)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--pool_size", type=int, default=128)
    parser.add_argument("--preprocessing", default="salient",
                        choices=["salient", "dense", "top10", "top20"])
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
