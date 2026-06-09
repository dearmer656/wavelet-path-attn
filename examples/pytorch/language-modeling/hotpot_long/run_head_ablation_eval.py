#!/usr/bin/env python3
"""
run_head_ablation_eval.py — PAT-187 Stage 1

For each OOD/drifted NMF pattern candidate (from Stage 0), selects the K
heads most cosine-similar to that pattern at ext length, masks them by
zeroing out_wav, and evaluates F1/EM at both L512 and L_ext. Reports
ExtGain = delta(ext) - delta(512) and NetCausalGain vs random controls.
"""

import argparse
import collections
import csv
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))

from analyze_pattern_basis_matching import l2_normalize_rows, resolve_nmf_run_dir
from dump_path_nmf import preprocess_logit_map
from eval_hotpot_long import best_f1_em, render_input


@dataclass(frozen=True)
class HeadRef:
    layer_idx: int
    head_idx: int


@dataclass(frozen=True)
class PatternCandidate:
    pattern_id: int
    pattern_type: str
    max_train_sim: float
    mean_usage_ext: float
    selected_for_mask: bool


@dataclass(frozen=True)
class CasePair:
    base_id: str
    ext_rec: dict
    base_rec: dict


@dataclass(frozen=True)
class ExtRunData:
    checkpoint: str
    pool_size: int
    preprocessing: str
    V_l2: np.ndarray


FIELDNAMES = [
    "model",
    "checkpoint",
    "base_id",
    "ext_record_id",
    "base_record_id",
    "ext_variant_id",
    "base_variant_id",
    "selection_len",
    "eval_len",
    "rank",
    "pattern_id",
    "pattern_type",
    "pattern_max_train_sim",
    "pattern_mean_usage_ext",
    "K",
    "effective_k",
    "selection_rule",
    "replicate_id",
    "n_valid_heads",
    "selected_heads_json",
    "baseline_prediction",
    "masked_prediction",
    "baseline_f1",
    "baseline_em",
    "masked_f1",
    "masked_em",
    "delta_f1",
    "delta_em",
]


def parse_args():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["PaTH-only", "QWAB"], required=True)
    parser.add_argument("--ext_len", type=int, choices=[2048, 4096], required=True)
    parser.add_argument("--rank", type=int, required=True)
    parser.add_argument(
        "--hotpot_jsonl",
        type=Path,
        default=root / "data" / "hotpot_long_dev_uniform.jsonl",
    )
    parser.add_argument(
        "--stage0_csv",
        type=Path,
        default=root / "analysis_outputs" / "pattern_basis_matching" / "ext_ood_pattern_candidates.csv",
    )
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=root / "analysis_outputs" / "head_ablation_eval",
    )
    parser.add_argument("--max_cases", type=int, default=None)
    parser.add_argument("--n_random", type=int, default=20)
    parser.add_argument("--ks", type=int, nargs="+", default=[2, 4, 8])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_new_tokens", type=int, default=32)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def _parse_bool(value):
    if isinstance(value, bool):
        return value
    return str(value).lower() == "true"


def _model_tag(model_label: str) -> str:
    return "path_only" if model_label == "PaTH-only" else "qwab"


def _capture_total_for_model(model_label: str) -> bool:
    return model_label == "QWAB"


def load_pattern_candidates(stage0_csv: Path, model: str, ext_len: int, rank: int) -> list:
    candidates = []
    with open(stage0_csv, newline="") as f:
        for row in csv.DictReader(f):
            if row["model"] != model:
                continue
            if int(row["ext_len"]) != ext_len:
                continue
            if int(row["rank"]) != rank:
                continue
            if not _parse_bool(row["selected_for_mask"]):
                continue
            candidates.append(
                PatternCandidate(
                    pattern_id=int(row["ext_pattern_id"]),
                    pattern_type=row["ext_type"],
                    max_train_sim=float(row["max_train_sim"]),
                    mean_usage_ext=float(row["mean_usage_ext"]),
                    selected_for_mask=True,
                )
            )
    candidates.sort(key=lambda x: x.pattern_id)
    return candidates


def load_ext_run_data(model: str, ext_len: int, rank: int) -> ExtRunData:
    runs_root = Path(__file__).resolve().parent.parent / "runs"
    nmf_dir = resolve_nmf_run_dir(model, ext_len, rank, runs_root)
    V = np.load(nmf_dir / "V.npy").astype(np.float64, copy=False)
    with open(nmf_dir / "meta.json") as f:
        meta = json.load(f)
    return ExtRunData(
        checkpoint=meta["checkpoint"],
        pool_size=int(meta["pool_size"]),
        preprocessing=meta["preprocessing"],
        V_l2=l2_normalize_rows(V),
    )


def pair_cases_by_base_id(jsonl_path: Path, ext_len: int, max_cases: int | None = None) -> list:
    ext_by_base = {}
    base_by_base = {}
    ext_order = []

    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            base_id = rec["base_id"]
            target_len = rec["meta"]["target_total_tokens"]

            if target_len == ext_len and base_id not in ext_by_base:
                ext_by_base[base_id] = rec
                ext_order.append(base_id)
            if target_len == 512 and base_id not in base_by_base:
                base_by_base[base_id] = rec

    pairs = []
    for base_id in ext_order:
        if base_id not in base_by_base:
            continue
        pairs.append(CasePair(base_id=base_id, ext_rec=ext_by_base[base_id], base_rec=base_by_base[base_id]))
        if max_cases is not None and len(pairs) >= max_cases:
            break
    return pairs


def build_capture_input_ids(tokenizer, rec: dict, seq_len: int, device: torch.device) -> torch.Tensor:
    prompt = render_input(rec["question"], rec["context"])
    answer = f" {rec['answer']}"
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    answer_ids = tokenizer(answer, add_special_tokens=False)["input_ids"]
    full_ids = (prompt_ids + answer_ids)[:seq_len]
    return torch.tensor([full_ids], dtype=torch.long, device=device)


def build_generation_inputs(tokenizer, rec: dict, device: torch.device):
    prompt = render_input(rec["question"], rec["context"])
    enc = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
    return enc["input_ids"].to(device), enc["attention_mask"].to(device)


def find_path_attention_modules(model) -> list:
    modules = []
    for _, module in model.named_modules():
        if type(module).__name__ == "PaTHAttention":
            modules.append(module)
    return list(enumerate(modules))


def _clear_capture_flags(path_modules: list) -> None:
    for _, module in path_modules:
        module._nmf_capture = False
        module._nmf_capture_total = False


def _set_capture_flags(path_modules: list, capture_total: bool) -> None:
    _clear_capture_flags(path_modules)
    for _, module in path_modules:
        if capture_total:
            module._nmf_capture_total = True
        else:
            module._nmf_capture = True


def clear_mask_heads(path_modules: list) -> None:
    for _, module in path_modules:
        module._mask_heads = None


def capture_ext_logit_maps(model, path_modules: list, input_ids: torch.Tensor, capture_total: bool) -> dict:
    attr_name = "_nmf_last_total_logits" if capture_total else "_nmf_last_base_logits"
    clear_mask_heads(path_modules)
    _set_capture_flags(path_modules, capture_total)
    try:
        with torch.no_grad():
            model(input_ids)
        out = {}
        for layer_idx, module in path_modules:
            buf = getattr(module, attr_name)
            out[layer_idx] = buf[0].to(torch.float32).cpu().numpy()
        return out
    finally:
        _clear_capture_flags(path_modules)


def preprocess_head_features(logit_maps_by_layer: dict, pool_size: int, preprocessing: str):
    head_features = {}
    valid_heads = []

    for layer_idx, layer_maps in logit_maps_by_layer.items():
        for head_idx in range(layer_maps.shape[0]):
            x = preprocess_logit_map(
                torch.from_numpy(layer_maps[head_idx]),
                M=pool_size,
                preprocessing=preprocessing,
            )
            if x is None:
                continue
            if torch.is_tensor(x):
                x = x.cpu().numpy()
            x = np.asarray(x, dtype=np.float64)
            x_hat = x / (np.linalg.norm(x) + 1e-12)
            head = HeadRef(layer_idx=layer_idx, head_idx=head_idx)
            head_features[head] = x_hat
            valid_heads.append(head)

    return head_features, valid_heads


def compute_pattern_head_similarities(head_features: dict, valid_heads: list, V_l2: np.ndarray, candidate_pattern_ids: list) -> np.ndarray:
    if not valid_heads:
        return np.zeros((len(candidate_pattern_ids), 0), dtype=np.float64)
    head_mat = np.stack([head_features[h] for h in valid_heads], axis=0)
    pattern_mat = V_l2[np.asarray(candidate_pattern_ids)]
    return pattern_mat @ head_mat.T


def select_topk_heads(valid_heads: list, sim_row: np.ndarray, K: int) -> list:
    idx = np.argsort(sim_row)[::-1][:K]
    return [valid_heads[i] for i in idx]


def sample_random_heads(valid_heads: list, K: int, rng: random.Random) -> list:
    return rng.sample(valid_heads, K)


def sample_same_layer_random_heads(top_heads: list, valid_heads: list, rng: random.Random) -> list:
    heads_by_layer = collections.defaultdict(list)
    for head in valid_heads:
        heads_by_layer[head.layer_idx].append(head)

    out = []
    for top_head in top_heads:
        choices = [h for h in heads_by_layer[top_head.layer_idx] if h.head_idx != top_head.head_idx]
        if not choices:
            choices = [top_head]
        out.append(rng.choice(choices))
    return out


def set_mask_heads(path_modules: list, selected_heads: list) -> None:
    by_layer = collections.defaultdict(set)
    for head in selected_heads:
        by_layer[head.layer_idx].add(head.head_idx)
    for layer_idx, module in path_modules:
        module._mask_heads = by_layer[layer_idx] if by_layer[layer_idx] else None


def generate_prediction(model, tokenizer, input_ids: torch.Tensor, attention_mask: torch.Tensor, max_new_tokens: int) -> str:
    with torch.no_grad():
        out = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
    gen_ids = out[0, input_ids.shape[1]:]
    pred = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
    return pred.split("\n")[0].strip()


def evaluate_one_condition(model, tokenizer, path_modules: list, rec: dict, selected_heads: list, device: torch.device, max_new_tokens: int):
    input_ids, attention_mask = build_generation_inputs(tokenizer, rec, device)
    clear_mask_heads(path_modules)
    if selected_heads:
        set_mask_heads(path_modules, selected_heads)
    try:
        prediction = generate_prediction(
            model=model,
            tokenizer=tokenizer,
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
        )
    finally:
        clear_mask_heads(path_modules)
    answers = rec.get("answer_aliases", [rec.get("answer", "")])
    f1, em = best_f1_em(prediction, answers)
    return prediction, f1, em


def _effective_k(selected_heads: list) -> int:
    return len({(h.layer_idx, h.head_idx) for h in selected_heads})


def _selected_heads_json(selected_heads: list) -> str:
    return json.dumps([[h.layer_idx, h.head_idx] for h in selected_heads])


def build_result_row(
    model_label, checkpoint, case_pair, eval_len, ext_len, rank, pattern, K,
    selection_rule, replicate_id, n_valid_heads, selected_heads,
    baseline_prediction, baseline_f1, baseline_em,
    masked_prediction, masked_f1, masked_em,
) -> dict:
    return {
        "model": model_label,
        "checkpoint": checkpoint,
        "base_id": case_pair.base_id,
        "ext_record_id": case_pair.ext_rec.get("_id", ""),
        "base_record_id": case_pair.base_rec.get("_id", ""),
        "ext_variant_id": case_pair.ext_rec.get("variant_id", ""),
        "base_variant_id": case_pair.base_rec.get("variant_id", ""),
        "selection_len": ext_len,
        "eval_len": eval_len,
        "rank": rank,
        "pattern_id": pattern.pattern_id,
        "pattern_type": pattern.pattern_type,
        "pattern_max_train_sim": pattern.max_train_sim,
        "pattern_mean_usage_ext": pattern.mean_usage_ext,
        "K": K,
        "effective_k": _effective_k(selected_heads),
        "selection_rule": selection_rule,
        "replicate_id": replicate_id,
        "n_valid_heads": n_valid_heads,
        "selected_heads_json": _selected_heads_json(selected_heads),
        "baseline_prediction": baseline_prediction,
        "masked_prediction": masked_prediction,
        "baseline_f1": baseline_f1,
        "baseline_em": baseline_em,
        "masked_f1": masked_f1,
        "masked_em": masked_em,
        "delta_f1": masked_f1 - baseline_f1,
        "delta_em": masked_em - baseline_em,
    }


def write_results_csv(out_path: Path, rows: list) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def print_summary(rows: list, ext_len: int) -> None:
    grouped = collections.defaultdict(list)
    for row in rows:
        key = (row["pattern_id"], row["pattern_type"], row["K"], row["selection_rule"], row["eval_len"])
        grouped[key].append(row["delta_f1"])

    ext_gain = {}
    rules = ["pattern_topk", "random_k", "same_layer_random"]

    print()
    print("pattern_id,pattern_type,K,selection_rule,mean_delta_f1_L512,mean_delta_f1_ext,ext_gain_f1")
    for pattern_id, pattern_type, K in sorted({(r["pattern_id"], r["pattern_type"], r["K"]) for r in rows}):
        for rule in rules:
            base_vals = grouped.get((pattern_id, pattern_type, K, rule, 512), [])
            ext_vals = grouped.get((pattern_id, pattern_type, K, rule, ext_len), [])
            if not base_vals or not ext_vals:
                continue
            mean_base = float(np.mean(base_vals))
            mean_ext = float(np.mean(ext_vals))
            gain = mean_ext - mean_base
            ext_gain[(pattern_id, K, rule)] = gain
            print(f"{pattern_id},{pattern_type},{K},{rule},{mean_base:.4f},{mean_ext:.4f},{gain:.4f}")

    print()
    print("pattern_id,pattern_type,K,net_causal_gain_f1")
    for pattern_id, pattern_type, K in sorted({(r["pattern_id"], r["pattern_type"], r["K"]) for r in rows}):
        if (pattern_id, K, "pattern_topk") not in ext_gain:
            continue
        control_vals = [ext_gain[(pattern_id, K, rule)] for rule in ("random_k", "same_layer_random") if (pattern_id, K, rule) in ext_gain]
        if not control_vals:
            continue
        net = ext_gain[(pattern_id, K, "pattern_topk")] - float(np.mean(control_vals))
        print(f"{pattern_id},{pattern_type},{K},{net:.4f}")


def run(args) -> None:
    if any(K > 144 for K in args.ks):
        raise ValueError("K cannot exceed 144")

    candidates = load_pattern_candidates(args.stage0_csv, args.model, args.ext_len, args.rank)
    result_path = args.out_dir / f"head_ablation_results_{_model_tag(args.model)}_L{args.ext_len}_R{args.rank}.csv"

    if not candidates:
        print(f"No selected_for_mask=True candidates for {args.model} L{args.ext_len} R{args.rank}")
        write_results_csv(result_path, [])
        return

    ext_run = load_ext_run_data(args.model, args.ext_len, args.rank)
    pairs = pair_cases_by_base_id(args.hotpot_jsonl, args.ext_len, args.max_cases)
    print(f"Candidates: {len(candidates)} | Case pairs: {len(pairs)}")

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    tokenizer = AutoTokenizer.from_pretrained(ext_run.checkpoint, use_fast=False)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        ext_run.checkpoint,
        torch_dtype=torch.float32,
        trust_remote_code=True,
        attn_implementation="path_attn",
    )
    model.eval().to(device)

    path_modules = find_path_attention_modules(model)
    capture_total = _capture_total_for_model(args.model)
    rng = random.Random(args.seed)
    candidate_pattern_ids = [x.pattern_id for x in candidates]
    rows = []

    for case_idx, case_pair in enumerate(pairs, start=1):
        capture_input_ids = build_capture_input_ids(tokenizer, case_pair.ext_rec, args.ext_len, device)
        logit_maps_by_layer = capture_ext_logit_maps(
            model=model,
            path_modules=path_modules,
            input_ids=capture_input_ids,
            capture_total=capture_total,
        )
        head_features, valid_heads = preprocess_head_features(
            logit_maps_by_layer=logit_maps_by_layer,
            pool_size=ext_run.pool_size,
            preprocessing=ext_run.preprocessing,
        )
        if not valid_heads:
            print(f"  [skip case {case_pair.base_id}] no valid heads after preprocessing")
            continue

        sim_matrix = compute_pattern_head_similarities(
            head_features=head_features,
            valid_heads=valid_heads,
            V_l2=ext_run.V_l2,
            candidate_pattern_ids=candidate_pattern_ids,
        )

        baseline_ext_pred, baseline_ext_f1, baseline_ext_em = evaluate_one_condition(
            model=model, tokenizer=tokenizer, path_modules=path_modules,
            rec=case_pair.ext_rec, selected_heads=[], device=device, max_new_tokens=args.max_new_tokens,
        )
        baseline_base_pred, baseline_base_f1, baseline_base_em = evaluate_one_condition(
            model=model, tokenizer=tokenizer, path_modules=path_modules,
            rec=case_pair.base_rec, selected_heads=[], device=device, max_new_tokens=args.max_new_tokens,
        )

        for pattern_idx, pattern in enumerate(candidates):
            sim_row = sim_matrix[pattern_idx]

            for K in args.ks:
                if K > len(valid_heads):
                    continue

                top_heads = select_topk_heads(valid_heads, sim_row, K)

                for eval_len, rec, bl_pred, bl_f1, bl_em in (
                    (args.ext_len, case_pair.ext_rec, baseline_ext_pred, baseline_ext_f1, baseline_ext_em),
                    (512, case_pair.base_rec, baseline_base_pred, baseline_base_f1, baseline_base_em),
                ):
                    masked_pred, masked_f1, masked_em = evaluate_one_condition(
                        model=model, tokenizer=tokenizer, path_modules=path_modules,
                        rec=rec, selected_heads=top_heads, device=device, max_new_tokens=args.max_new_tokens,
                    )
                    rows.append(build_result_row(
                        model_label=args.model, checkpoint=ext_run.checkpoint,
                        case_pair=case_pair, eval_len=eval_len, ext_len=args.ext_len,
                        rank=args.rank, pattern=pattern, K=K,
                        selection_rule="pattern_topk", replicate_id=0,
                        n_valid_heads=len(valid_heads), selected_heads=top_heads,
                        baseline_prediction=bl_pred, baseline_f1=bl_f1, baseline_em=bl_em,
                        masked_prediction=masked_pred, masked_f1=masked_f1, masked_em=masked_em,
                    ))

                for replicate_id in range(args.n_random):
                    random_heads = sample_random_heads(valid_heads, K, rng)
                    same_layer_heads = sample_same_layer_random_heads(top_heads, valid_heads, rng)

                    for selection_rule, selected_heads in (
                        ("random_k", random_heads),
                        ("same_layer_random", same_layer_heads),
                    ):
                        for eval_len, rec, bl_pred, bl_f1, bl_em in (
                            (args.ext_len, case_pair.ext_rec, baseline_ext_pred, baseline_ext_f1, baseline_ext_em),
                            (512, case_pair.base_rec, baseline_base_pred, baseline_base_f1, baseline_base_em),
                        ):
                            masked_pred, masked_f1, masked_em = evaluate_one_condition(
                                model=model, tokenizer=tokenizer, path_modules=path_modules,
                                rec=rec, selected_heads=selected_heads, device=device, max_new_tokens=args.max_new_tokens,
                            )
                            rows.append(build_result_row(
                                model_label=args.model, checkpoint=ext_run.checkpoint,
                                case_pair=case_pair, eval_len=eval_len, ext_len=args.ext_len,
                                rank=args.rank, pattern=pattern, K=K,
                                selection_rule=selection_rule, replicate_id=replicate_id,
                                n_valid_heads=len(valid_heads), selected_heads=selected_heads,
                                baseline_prediction=bl_pred, baseline_f1=bl_f1, baseline_em=bl_em,
                                masked_prediction=masked_pred, masked_f1=masked_f1, masked_em=masked_em,
                            ))

        if case_idx % 5 == 0 or case_idx == len(pairs):
            print(f"  {case_idx}/{len(pairs)} cases done, {len(rows)} rows so far")

    write_results_csv(result_path, rows)
    print(f"Saved {result_path} ({len(rows)} rows)")
    if rows:
        print_summary(rows, args.ext_len)


def main():
    run(parse_args())


if __name__ == "__main__":
    main()
