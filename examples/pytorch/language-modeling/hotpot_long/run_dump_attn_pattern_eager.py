"""PAT-169: Attention pattern analysis for standard-attention (eager) models.

Computes vertical_score, diagonal_score, periodic_seq_score, unpredictability_score
from HuggingFace output_attentions=True, using the same formulas as path_attn.py.

Usage:
  python run_dump_attn_pattern_eager.py \\
      --checkpoint <CKPT_DIR> \\
      --model_name <NAME> \\
      --seq_len <N> \\
      --out_root <DIR> \\
      [--jsonl <PATH>] \\
      [--case_limit 64] \\
      [--query_stride 1]
"""

import argparse
import json
import math
import os
import sys
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


# ---------------------------------------------------------------------------
# Pattern score helpers (mirrors path_attn.py static methods exactly)
# ---------------------------------------------------------------------------

def _pattern_row_cos(a: torch.Tensor, b: torch.Tensor) -> float:
    denom = torch.linalg.vector_norm(a, ord=2) * torch.linalg.vector_norm(b, ord=2)
    if float(denom.item()) <= 1e-8:
        return 0.0
    return float((a * b).sum().item() / denom.item())


def _pattern_jsd(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.clamp_min(1e-8)
    b = b.clamp_min(1e-8)
    a = a / a.sum().clamp_min(1e-8)
    b = b / b.sum().clamp_min(1e-8)
    m = 0.5 * (a + b)
    kl_am = float((a * (a.log() - m.log())).sum().item())
    kl_bm = float((b * (b.log() - m.log())).sum().item())
    return 0.5 * (kl_am + kl_bm)


def compute_pattern_features(
    attn_prob: torch.Tensor,  # [q_len, k_len] head-averaged softmax attention
    query_stride: int = 1,
    support_start: int = None,
    support_end: int = None,
) -> dict:
    """Compute pattern scores from a [q_len, k_len] attention probability matrix."""
    q_len, k_len = attn_prob.shape
    rows = attn_prob.float().cpu()

    stride = max(1, int(query_stride))
    row_ids = list(range(0, q_len, stride))
    if (q_len - 1) not in row_ids:
        row_ids.append(q_len - 1)

    vertical_vals = []
    diagonal_vals = []
    jsd_vals = []
    aligned = torch.zeros((len(row_ids), k_len), dtype=torch.float32)

    for ridx, q_idx in enumerate(row_ids):
        vals = rows[q_idx, : q_idx + 1]
        aligned[ridx, : q_idx + 1] = torch.flip(vals, dims=[0])
        if ridx == 0:
            continue
        prev_q = row_ids[ridx - 1]
        cur_q = q_idx
        prev_row = rows[prev_q]
        cur_row = rows[cur_q]
        v = _pattern_row_cos(prev_row, cur_row)
        vertical_vals.append(v)
        shift = max(1, int(cur_q - prev_q))
        if shift < k_len:
            d = _pattern_row_cos(prev_row[shift:], cur_row[:-shift])
        else:
            d = 0.0
        diagonal_vals.append(d)
        jsd_vals.append(_pattern_jsd(prev_row, cur_row))

    if aligned.shape[0] > 2:
        spec = torch.fft.rfft(aligned, dim=0)
        mag = spec.abs()
        nonzero = mag[1:] if int(mag.shape[0]) > 1 else None
        col_mass = aligned.abs().sum(dim=0)
        keep = col_mass > 1e-8
        if nonzero is not None and bool(keep.any()):
            peak = nonzero[:, keep].max(dim=0).values
            dc = mag[0, keep].clamp_min(1e-8)
            periodic_seq = float((peak / dc).mean().item())
        else:
            periodic_seq = 0.0
        fft2 = torch.fft.rfft2(aligned)
        mag2 = fft2.abs()
        mag2[0, 0] = 0.0
        flat = mag2.flatten()
        total = float(flat.sum().item())
        if total > 1e-8:
            topk = min(8, int(flat.numel()))
            seasonal = float(torch.topk(flat, k=topk).values.sum().item() / total)
        else:
            seasonal = 0.0
    else:
        periodic_seq = 0.0
        seasonal = 0.0

    return {
        "vertical_score": float(sum(vertical_vals) / len(vertical_vals)) if vertical_vals else 0.0,
        "diagonal_score": float(sum(diagonal_vals) / len(diagonal_vals)) if diagonal_vals else 0.0,
        "unpredictability_score": float(1.0 - (sum(vertical_vals) / len(vertical_vals))) if vertical_vals else 1.0,
        "jsd_unpredictability": float(sum(jsd_vals) / len(jsd_vals)) if jsd_vals else 0.0,
        "periodic_seq_score": float(periodic_seq),
        "seasonal_2d_score": float(seasonal),
        "query_stride": int(stride),
        "num_sampled_rows": int(len(row_ids)),
    }


# ---------------------------------------------------------------------------
# Data loading: tokenize hotpot_long JSONL into input_ids at target length
# ---------------------------------------------------------------------------

def load_cases(jsonl_path: str, tokenizer, seq_len: int, case_limit: int):
    """Load up to case_limit hotpot records, tokenize context to seq_len tokens."""
    cases = []
    with open(jsonl_path) as f:
        for line in f:
            if len(cases) >= case_limit:
                break
            rec = json.loads(line)
            meta = rec.get("meta", {})
            if int(meta.get("target_total_tokens", 0)) != seq_len:
                continue

            # Build text: "Context:\n" + all doc texts + "\nQuestion: ... \nAnswer:"
            ctx_docs = rec["context"]  # list of [title, sentences]
            text_parts = ["Context:\n"]
            for title, sents in ctx_docs:
                text_parts.append(f"{title}: " + " ".join(sents) + "\n")
            q = rec.get("question", "")
            ans = rec.get("answer", "")
            text_parts.append(f"\nQuestion: {q}\nAnswer: {ans}")
            full_text = "".join(text_parts)

            ids = tokenizer(full_text, add_special_tokens=True, truncation=False)["input_ids"]
            # Pad or truncate to seq_len
            if len(ids) >= seq_len:
                ids = ids[:seq_len]
            else:
                ids = ids + [tokenizer.eos_token_id] * (seq_len - len(ids))

            cases.append({
                "input_ids": ids,
                "meta": meta,
                "_id": rec.get("_id", str(len(cases))),
            })
    return cases


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--seq_len", type=int, required=True)
    parser.add_argument("--out_root", required=True)
    parser.add_argument("--jsonl", default=None)
    parser.add_argument("--case_limit", type=int, default=64)
    parser.add_argument("--query_stride", type=int, default=1)
    parser.add_argument("--model_type", default="eager_std",
                        help="Label for model type in output (e.g. nope_qwab, rotary_qwab)")
    args = parser.parse_args()

    default_jsonl = os.path.join(
        os.path.dirname(__file__), "data", "hotpot_long_dev_uniform.jsonl"
    )
    jsonl_path = args.jsonl or default_jsonl

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Loading tokenizer from {args.checkpoint}")
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    print(f"Loading model from {args.checkpoint}")
    model = AutoModelForCausalLM.from_pretrained(
        args.checkpoint,
        attn_implementation="eager",
        torch_dtype=torch.bfloat16,
        output_attentions=False,  # set per-call below
    ).to(device).eval()

    num_layers = model.config.num_hidden_layers
    run_tag = f"{args.model_type}_pattern_L{args.seq_len}_{args.model_name}"
    feature_root = Path(args.out_root) / f"block_{args.seq_len}" / run_tag
    feature_root.mkdir(parents=True, exist_ok=True)
    print(f"Feature root: {feature_root}")

    print(f"Loading cases from {jsonl_path} (seq_len={args.seq_len}, limit={args.case_limit})")
    cases = load_cases(jsonl_path, tokenizer, args.seq_len, args.case_limit)
    print(f"Loaded {len(cases)} cases")

    for case_id, case in enumerate(cases):
        print(f"Case {case_id}/{len(cases)}  id={case['_id']}")
        input_ids = torch.tensor([case["input_ids"]], dtype=torch.long, device=device)
        meta = case["meta"]
        support_start = meta.get("support_start_tok")
        support_end = meta.get("support_end_tok")

        with torch.no_grad():
            out = model(input_ids=input_ids, output_attentions=True)

        # out.attentions: tuple of [1, num_heads, seq_len, seq_len] per layer
        for layer_idx, attn in enumerate(out.attentions):
            # attn: [1, num_heads, seq_len, seq_len]
            attn_prob = attn[0].float().cpu()  # [num_heads, seq_len, seq_len]
            attn_mean = attn_prob.mean(dim=0)  # [seq_len, seq_len]

            scores = compute_pattern_features(
                attn_mean,
                query_stride=args.query_stride,
                support_start=support_start,
                support_end=support_end,
            )

            layer_dir = feature_root / f"layer{layer_idx:02d}" / f"case{case_id:03d}"
            layer_dir.mkdir(parents=True, exist_ok=True)

            payload = {
                "meta": {
                    "model_type": args.model_type,
                    "layer": layer_idx,
                    "case_id": case_id,
                    "step": -1,
                    "q_len": args.seq_len,
                    "k_len": args.seq_len,
                    "query_stride": args.query_stride,
                    "support_start_tok": int(support_start) if support_start is not None else -1,
                    "support_end_tok": int(support_end) if support_end is not None else -1,
                    "example_id": str(case["_id"]),
                },
                "attention_prob": {
                    "final": scores,
                },
            }
            with (layer_dir / "pattern_features.json").open("w") as fout:
                json.dump(payload, fout, indent=2)

        del out
        torch.cuda.empty_cache()

    print(f"Done. Feature root: {feature_root}")
    print(f"Processed {len(cases)} cases × {num_layers} layers")


if __name__ == "__main__":
    main()
