"""eval_hotpot_long.py — Evaluate a checkpoint on HotpotQA-Long.

Runs model inference on augmented HotpotQA-Long examples (JSONL from
make_hotpot_long.py) and computes F1/EM per length and placement bin,
saving per-example outputs and summary statistics.

Usage:
  python eval_hotpot_long.py \
      --model-path /path/to/checkpoint \
      --hotpot-long-jsonl hotpot_long_dev.jsonl \
      --tokenizer gpt2 \
      --output-dir eval_output/ \
      --batch-size 8 \
      --target-lengths 512 2048 4096 \
      [--max-new-tokens 32] \
      [--device auto]
"""

from __future__ import annotations

import argparse
import ast
import collections
import json
import re
import string
import unicodedata
from pathlib import Path
from typing import Optional

import torch


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
            try:
                v = ast.literal_eval(v)
            except Exception:
                pass
            cfg[k] = v
    return cfg


def add_missing_to_hf_config(config, kv: dict):
    existing = set(config.to_dict().keys())
    added, skipped = [], []
    for k, v in kv.items():
        if k in existing:
            skipped.append(k)
        else:
            setattr(config, k, v)
            added.append(k)
    return added, skipped


# ---------------------------------------------------------------------------
# F1 / EM (standard HotpotQA evaluation)
# ---------------------------------------------------------------------------

def normalize_answer(s: str) -> str:
    s = unicodedata.normalize("NFKD", s).lower()
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = s.translate(str.maketrans("", "", string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def get_tokens(s: str) -> list[str]:
    return normalize_answer(s).split()


def compute_f1(prediction: str, ground_truth: str) -> float:
    pred_toks = get_tokens(prediction)
    gt_toks = get_tokens(ground_truth)
    common = collections.Counter(pred_toks) & collections.Counter(gt_toks)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_toks) if pred_toks else 0.0
    recall = num_same / len(gt_toks) if gt_toks else 0.0
    return (2 * precision * recall) / (precision + recall + 1e-12)


def compute_em(prediction: str, ground_truth: str) -> float:
    return float(normalize_answer(prediction) == normalize_answer(ground_truth))


def best_f1_em(prediction: str, answers: list[str]) -> tuple[float, float]:
    f1 = max(compute_f1(prediction, a) for a in answers) if answers else 0.0
    em = max(compute_em(prediction, a) for a in answers) if answers else 0.0
    return f1, em


# ---------------------------------------------------------------------------
# Context rendering (must match make_hotpot_long.py)
# ---------------------------------------------------------------------------

def render_doc(title: str, sentences: list[str]) -> str:
    body = " ".join(sentences).strip()
    return f"Title: {title}\n{body}\n\n"


def render_input(question: str, context: list[list]) -> str:
    ctx = "".join(render_doc(t, s) for t, s in context)
    return f"Question: {question}\n\nContext:\n{ctx}Answer:"


# ---------------------------------------------------------------------------
# Cluster bootstrap for confidence intervals
# ---------------------------------------------------------------------------

def cluster_bootstrap_ci(
    scores: list[float],
    cluster_ids: list[str],
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Return (mean, lower_ci, upper_ci) using cluster bootstrap by cluster_id."""
    import random
    rng = random.Random(seed)

    cluster_to_scores: dict[str, list[float]] = collections.defaultdict(list)
    for s, cid in zip(scores, cluster_ids):
        cluster_to_scores[cid].append(s)

    clusters = list(cluster_to_scores.keys())
    n_clusters = len(clusters)
    point_mean = sum(scores) / len(scores) if scores else 0.0

    boot_means: list[float] = []
    for _ in range(n_boot):
        sampled_clusters = [rng.choice(clusters) for _ in range(n_clusters)]
        boot_scores = [s for cid in sampled_clusters for s in cluster_to_scores[cid]]
        boot_means.append(sum(boot_scores) / len(boot_scores) if boot_scores else 0.0)

    boot_means.sort()
    lo_idx = int(n_boot * alpha / 2)
    hi_idx = int(n_boot * (1 - alpha / 2))
    return point_mean, boot_means[lo_idx], boot_means[min(hi_idx, n_boot - 1)]


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--hotpot-long-jsonl", required=True)
    parser.add_argument("--tokenizer", default="gpt2")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--target-lengths", nargs="+", type=int, default=None,
                        help="Filter to specific lengths (default: all)")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--n-boot", type=int, default=2000, help="Bootstrap iterations")
    parser.add_argument("--cfg-path", default=None, help="Optional supply_model.cfg-style KEY=value file; keys not already in the model config are set via setattr before model construction (e.g. attn_norm=entmax, entmax_alpha=1.5, entmax_scope=all)")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load model and tokenizer
    from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM

    print(f"Loading tokenizer from {args.tokenizer}...")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    print(f"Loading model from {args.model_path}...")
    config = AutoConfig.from_pretrained(args.model_path)
    # AutoConfig.from_pretrained() does not preserve `attn_implementation` from
    # the checkpoint's config.json (HF treats it as a from_pretrained-time arg,
    # not a persisted field), so PaTH checkpoints would silently fall back to
    # randomly-initialized GPT2Attention. Re-read it directly and re-apply.
    model_config_path = Path(args.model_path) / "config.json"
    if model_config_path.is_file():
        with open(model_config_path, "r", encoding="utf-8") as f:
            raw_config = json.load(f)
        attn_impl = raw_config.get("attn_implementation")
        if attn_impl:
            config._attn_implementation = attn_impl
    if args.cfg_path:
        cfg = read_kv_config(args.cfg_path)
        added, skipped = add_missing_to_hf_config(config, cfg)
        print(f"cfg_path={args.cfg_path}: added={added}, skipped={skipped}")
    if args.device == "auto":
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path, config=config, torch_dtype=torch.float16, device_map="auto"
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(args.model_path, config=config)
        model = model.to(args.device)
    model.eval()

    # Load examples
    print(f"Loading HotpotQA-Long examples from {args.hotpot_long_jsonl}...")
    examples: list[dict] = []
    with open(args.hotpot_long_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            ex = json.loads(line)
            if args.target_lengths and ex["meta"]["target_total_tokens"] not in args.target_lengths:
                continue
            examples.append(ex)
            if args.max_examples and len(examples) >= args.max_examples:
                break

    print(f"Loaded {len(examples):,} examples")

    # Evaluate in batches
    all_results: list[dict] = []

    def run_batch(batch: list[dict]) -> list[str]:
        prompts = [render_input(ex["question"], ex["context"]) for ex in batch]
        enc = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=False,
        )
        input_ids = enc["input_ids"].to(model.device)
        attention_mask = enc["attention_mask"].to(model.device)
        with torch.no_grad():
            out = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        predictions = []
        for i, sample_out in enumerate(out):
            gen_ids = sample_out[input_ids.shape[1]:]
            pred_text = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
            # Take first line / sentence as answer
            pred_text = pred_text.split("\n")[0].strip()
            predictions.append(pred_text)
        return predictions

    for batch_start in range(0, len(examples), args.batch_size):
        batch = examples[batch_start : batch_start + args.batch_size]
        predictions = run_batch(batch)

        for ex, pred in zip(batch, predictions):
            answers = ex.get("answer_aliases", [ex.get("answer", "")])
            f1, em = best_f1_em(pred, answers)
            result = {
                "_id": ex["_id"],
                "base_id": ex["base_id"],
                "target_len": ex["meta"]["target_total_tokens"],
                "actual_len": ex["meta"]["actual_total_tokens"],
                "placement_pct": ex["meta"]["placement_target_pct"],
                "packing_status": ex["meta"]["packing_status"],
                "prediction": pred,
                "answer": ex["answer"],
                "f1": f1,
                "em": em,
            }
            all_results.append(result)

        if (batch_start + len(batch)) % 200 == 0:
            print(f"  {batch_start + len(batch):,}/{len(examples):,} done")

    # Save per-example results
    results_path = out_dir / "per_example_results.jsonl"
    with open(results_path, "w", encoding="utf-8") as f:
        for r in all_results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Per-example results saved to {results_path}")

    # Aggregate by target length
    by_len: dict[int, list[dict]] = collections.defaultdict(list)
    for r in all_results:
        by_len[r["target_len"]].append(r)

    summary_rows: list[dict] = []
    print("\n=== HotpotQA-Long Results ===")
    print(f"{'Length':>8}  {'N':>6}  {'F1':>7}  {'F1_lo':>7}  {'F1_hi':>7}  {'EM':>7}")
    for tlen in sorted(by_len.keys()):
        group = by_len[tlen]
        f1s = [r["f1"] for r in group]
        ems = [r["em"] for r in group]
        base_ids = [r["base_id"] for r in group]
        mean_f1, lo_f1, hi_f1 = cluster_bootstrap_ci(f1s, base_ids, n_boot=args.n_boot)
        mean_em = sum(ems) / len(ems) if ems else 0.0
        n = len(group)
        row = {
            "target_len": tlen,
            "n": n,
            "n_base_ids": len(set(base_ids)),
            "f1_mean": round(mean_f1, 4),
            "f1_ci_lo": round(lo_f1, 4),
            "f1_ci_hi": round(hi_f1, 4),
            "em_mean": round(mean_em, 4),
        }
        summary_rows.append(row)
        print(f"  L{tlen:>5}  {n:>6}  {mean_f1:>7.4f}  {lo_f1:>7.4f}  {hi_f1:>7.4f}  {mean_em:>7.4f}")

    # Save summary
    summary_path = out_dir / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump({"model": args.model_path, "results": summary_rows}, f, indent=2)
    print(f"\nSummary saved to {summary_path}")

    # Also aggregate by placement bin
    by_placement: dict[tuple[int, float], list[float]] = collections.defaultdict(list)
    for r in all_results:
        key = (r["target_len"], r["placement_pct"])
        by_placement[key].append(r["f1"])

    placement_path = out_dir / "by_placement.json"
    placement_rows = [
        {"target_len": k[0], "placement_pct": k[1], "n": len(v),
         "f1_mean": round(sum(v) / len(v), 4)}
        for k, v in sorted(by_placement.items())
    ]
    with open(placement_path, "w") as f:
        json.dump(placement_rows, f, indent=2)
    print(f"Placement breakdown saved to {placement_path}")


if __name__ == "__main__":
    main()
