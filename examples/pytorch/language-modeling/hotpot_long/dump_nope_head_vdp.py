#!/usr/bin/env python3
"""
dump_nope_head_vdp.py — PAT-178: per-head V/D/P pattern features for NoPE eager-attention model.

For each case, loads the attention weights via output_attentions=True and computes
vertical_score / diagonal_score / periodic_seq_score per layer per head, under two
query scopes:
  - all_query   : all causal query positions (0 .. actual_total_tokens-1)
  - target_query: positions after the context docs (prompt_fixed_tokens + context_tokens
                  .. actual_total_tokens), i.e. the question-suffix + answer tokens

Output per (layer, case): {out_root}/block_{seq_len}/{model_tag}/layer{L:02d}/case{C:03d}/pattern_features.json
"""

import argparse
import json
import math
from pathlib import Path
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


# ─── standalone V/D/P helpers (extracted from path_attn.py, no self-dependency) ──

def _pattern_support_mask(length: int, support_start: Optional[int], support_end: Optional[int], *, device) -> Optional[torch.Tensor]:
    if support_start is None or support_end is None:
        return None
    start, end = int(support_start), int(support_end)
    if start < 0 or end <= start or length <= 0:
        return None
    mask = torch.zeros(length, dtype=torch.bool, device=device)
    mask[max(0, start):min(length, end)] = True
    return mask if bool(mask.any()) else None


def _row_cos(a: torch.Tensor, b: torch.Tensor) -> float:
    denom = torch.linalg.vector_norm(a, ord=2) * torch.linalg.vector_norm(b, ord=2)
    if float(denom.item()) <= 1e-8:
        return 0.0
    return float((a * b).sum().item() / denom.item())


def _jsd(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.clamp_min(1e-8) / a.clamp_min(1e-8).sum()
    b = b.clamp_min(1e-8) / b.clamp_min(1e-8).sum()
    m = 0.5 * (a + b)
    kl_am = float((a * (a.log() - m.log())).sum().item())
    kl_bm = float((b * (b.log() - m.log())).sum().item())
    return 0.5 * (kl_am + kl_bm)


@torch.no_grad()
def compute_vdp_features(
    mat: torch.Tensor,
    query_stride: int,
    support_start: Optional[int],
    support_end: Optional[int],
    query_row_ids: Optional[list] = None,
) -> dict:
    """Compute V/D/P features from a [Q, K] causal attention probability matrix.

    Args:
        mat: [Q, K] float32 tensor of causal attention probs (upper triangle = 0).
        query_stride: step size when sampling query rows (ignored if query_row_ids given).
        support_start: key-side support start token index (for support_key_mass).
        support_end: key-side support end token index.
        query_row_ids: if given, use exactly these query row indices (overrides stride).

    Returns:
        dict with vertical_score, diagonal_score, periodic_seq_score, and others.
    """
    mat = torch.nan_to_num(mat.detach().to(dtype=torch.float32, device="cpu"), nan=0.0, posinf=0.0, neginf=0.0)
    q_len, k_len = mat.shape
    if q_len <= 0 or k_len <= 0:
        return {}

    # renormalize each row to sum=1 over causal context
    rows = torch.zeros_like(mat)
    for i in range(q_len):
        vals = mat[i, : i + 1].clamp_min(0.0)
        s = vals.sum().clamp_min(1e-8)
        rows[i, : i + 1] = vals / s

    # build list of sampled query row indices
    if query_row_ids is not None:
        rids = [r for r in query_row_ids if 0 <= r < q_len]
    else:
        stride = max(1, int(query_stride))
        rids = list(range(0, q_len, stride))
        if q_len - 1 not in rids:
            rids.append(q_len - 1)

    if len(rids) < 2:
        return {"vertical_score": float("nan"), "diagonal_score": float("nan"),
                "periodic_seq_score": float("nan"), "num_sampled_rows": len(rids)}

    support_mask = _pattern_support_mask(k_len, support_start, support_end, device=rows.device)

    vertical_vals, diagonal_vals, jsd_vals = [], [], []
    aligned = torch.zeros((len(rids), k_len), dtype=torch.float32)

    for ridx, q_idx in enumerate(rids):
        vals = rows[q_idx, : q_idx + 1]
        aligned[ridx, : q_idx + 1] = torch.flip(vals, dims=[0])
        if ridx == 0:
            continue
        prev_q = rids[ridx - 1]
        prev_row = rows[prev_q]
        cur_row = rows[q_idx]
        v = _row_cos(prev_row, cur_row)
        vertical_vals.append(v)
        shift = max(1, q_idx - prev_q)
        d = _row_cos(prev_row[shift:], cur_row[:-shift]) if shift < k_len else 0.0
        diagonal_vals.append(d)
        jsd_vals.append(_jsd(prev_row, cur_row))

    # support_key_mass: average attention from sampled query rows to support key positions
    sampled_rows = rows[rids]  # [n_sampled, k_len]

    # periodic score via FFT over aligned query rows
    if aligned.shape[0] > 2:
        spec = torch.fft.rfft(aligned, dim=0)
        mag = spec.abs()
        nonzero = mag[1:] if mag.shape[0] > 1 else None
        col_mass = aligned.abs().sum(dim=0)
        keep = col_mass > 1e-8
        if nonzero is not None and bool(keep.any()):
            peak = nonzero[:, keep].max(dim=0).values
            dc = mag[0, keep].clamp_min(1e-8)
            periodic_seq = float((peak / dc).mean().item())
        else:
            periodic_seq = 0.0
    else:
        periodic_seq = 0.0

    support_key_mass = float(sampled_rows[:, support_mask].sum(dim=-1).mean().item()) if support_mask is not None else float("nan")

    def _mean(xs): return float(sum(xs) / len(xs)) if xs else float("nan")

    return {
        "vertical_score": _mean(vertical_vals),
        "diagonal_score": _mean(diagonal_vals),
        "unpredictability_score": float(1.0 - _mean(vertical_vals)) if vertical_vals else float("nan"),
        "jsd_unpredictability": _mean(jsd_vals),
        "periodic_seq_score": periodic_seq,
        "support_key_mass": support_key_mass,
        "query_stride": int(query_stride) if query_row_ids is None else -1,
        "num_sampled_rows": len(rids),
    }


# ─── text formatting (matches eval_hotpot_long.py) ────────────────────────────

def render_doc(title: str, sentences: list) -> str:
    return f"Title: {title}\n{' '.join(sentences).strip()}\n\n"


def render_prompt(ex: dict) -> str:
    ctx = "".join(render_doc(t, s) for t, s in ex["context"])
    return f"Question: {ex['question']}\n\nContext:\n{ctx}Answer:"


# ─── main extraction ──────────────────────────────────────────────────────────

def extract_model_tag(checkpoint: str) -> str:
    p = Path(checkpoint)
    parent = p.parent.name  # e.g. finetune_eager_nope_seed42
    step = p.name           # e.g. checkpoint-15900
    seed = "s42" if "seed42" in parent else ("s43" if "seed43" in parent else ("s44" if "seed44" in parent else "sXX"))
    step_num = step.replace("checkpoint-", "ckpt")
    return f"nope_{seed}_{step_num}"


def run(args):
    checkpoint = args.checkpoint
    seq_len = args.seq_len
    case_limit = args.case_limit
    query_stride = args.query_stride
    out_root = Path(args.out_root)
    jsonl_path = Path(args.jsonl)
    model_tag = extract_model_tag(checkpoint)

    # load cases for this seq_len
    cases = []
    with open(jsonl_path) as f:
        for line in f:
            rec = json.loads(line)
            if rec["meta"].get("target_total_tokens") == seq_len:
                cases.append(rec)
                if len(cases) >= case_limit:
                    break
    print(f"Loaded {len(cases)} cases for seq_len={seq_len}")

    # load model + tokenizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading model from {checkpoint} on {device}")
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModelForCausalLM.from_pretrained(checkpoint, dtype=torch.float32, attn_implementation="eager")
    model.eval().to(device)

    feat_dir = out_root / f"block_{seq_len}" / model_tag
    feat_dir.mkdir(parents=True, exist_ok=True)

    for case_idx, rec in enumerate(cases):
        meta = rec["meta"]
        example_id = rec.get("_id", "")
        support_start = meta.get("support_start_tok")
        support_end = meta.get("support_end_tok")
        context_tokens = int(meta.get("context_tokens", 0))
        prompt_fixed_tokens = int(meta.get("prompt_fixed_tokens", 0))
        actual_total_tokens = int(meta.get("actual_total_tokens", seq_len))

        # target-query: positions after the context docs (question-suffix + answer)
        target_query_start = prompt_fixed_tokens + context_tokens
        target_query_end = actual_total_tokens  # exclusive

        # tokenize
        prompt_str = render_prompt(rec)
        answer_str = f" {rec['answer']}"
        prompt_ids = tokenizer(prompt_str, add_special_tokens=False)["input_ids"]
        answer_ids = tokenizer(answer_str, add_special_tokens=False)["input_ids"]
        full_ids = prompt_ids + answer_ids
        # truncate / pad to seq_len
        if len(full_ids) > seq_len:
            full_ids = full_ids[:seq_len]
        actual_len = len(full_ids)
        input_ids = torch.tensor([full_ids], dtype=torch.long, device=device)

        target_query_end = min(target_query_end, actual_len)
        target_query_start = min(target_query_start, target_query_end)
        target_row_ids = list(range(target_query_start, target_query_end))

        with torch.no_grad():
            outputs = model(input_ids, output_attentions=True)
        # outputs.attentions: tuple of n_layers tensors, each [1, n_heads, Q, Q]
        attn_layers = outputs.attentions  # 12 tensors for GPT-2 small

        for layer_idx, attn_layer in enumerate(attn_layers):
            # attn_layer: [1, n_heads, actual_len, actual_len]
            attn = attn_layer[0].cpu()  # [n_heads, Q, Q]
            n_heads = attn.shape[0]

            all_query_feats = []
            target_query_feats = []
            for h in range(n_heads):
                head_mat = attn[h, :actual_len, :actual_len]  # [Q, Q]
                all_query_feats.append(compute_vdp_features(
                    head_mat, query_stride=query_stride,
                    support_start=support_start, support_end=support_end,
                ))
                target_query_feats.append(compute_vdp_features(
                    head_mat, query_stride=query_stride,
                    support_start=support_start, support_end=support_end,
                    query_row_ids=target_row_ids,
                ))

            payload = {
                "meta": {
                    "model_tag": model_tag,
                    "layer": layer_idx,
                    "case_id": case_idx,
                    "q_len": actual_len,
                    "k_len": actual_len,
                    "query_stride": query_stride,
                    "support_start_tok": support_start,
                    "support_end_tok": support_end,
                    "example_id": example_id,
                    "target_query_start": target_query_start,
                    "target_query_end": target_query_end,
                    "n_heads": n_heads,
                },
                "per_head": {
                    "all_query": all_query_feats,
                    "target_query": target_query_feats,
                },
            }

            out_dir = feat_dir / f"layer{layer_idx:02d}" / f"case{case_idx:03d}"
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / "pattern_features.json"
            with open(out_path, "w") as f:
                json.dump(payload, f)

        # free attention tensors
        del attn_layers, outputs
        if device.type == "cuda":
            torch.cuda.empty_cache()

        if (case_idx + 1) % 10 == 0:
            print(f"  case {case_idx + 1}/{len(cases)}")

    print(f"Done. Features written to {feat_dir}")


def main():
    parser = argparse.ArgumentParser(description="PAT-178: dump per-head V/D/P features for NoPE model")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint directory")
    parser.add_argument("--jsonl", required=True, help="Path to hotpot_long_dev_uniform.jsonl")
    parser.add_argument("--out_root", required=True, help="Output root directory")
    parser.add_argument("--seq_len", type=int, required=True, help="Sequence length (filters JSONL cases)")
    parser.add_argument("--case_limit", type=int, default=64, help="Max number of cases to process")
    parser.add_argument("--query_stride", type=int, default=1, help="Query stride for all-query scope (ignored for target-query)")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
