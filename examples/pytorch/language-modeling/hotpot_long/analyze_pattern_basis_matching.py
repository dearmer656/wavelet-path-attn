#!/usr/bin/env python3
"""
analyze_pattern_basis_matching.py — PAT-187 Stage 0

Computes train-vs-ext NMF basis similarity (cosine + top-k IoU + Hungarian),
classifies each train pattern as retained/drifted/missing, and flags ext
OOD/new-pattern candidates for head-masking in Stage 1.
"""

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_nmf_basis_alignment import classify_basis


@dataclass(frozen=True)
class RunRef:
    model: str
    seq_len: int
    rank: int
    nmf_dir: Path


@dataclass
class RunData:
    ref: RunRef
    V: np.ndarray
    V_l2: np.ndarray
    basis_types: list
    meta: dict
    pool_size: int
    U_mean: np.ndarray


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=Path(__file__).resolve().parent / "analysis_outputs" / "pattern_basis_matching",
    )
    parser.add_argument("--rank", type=int, nargs="+", default=[16])
    parser.add_argument("--top_p", type=float, default=0.1)
    return parser.parse_args()


def resolve_nmf_run_dir(model: str, seq_len: int, rank: int, runs_root: Path) -> Path:
    if model == "PaTH-only":
        return runs_root / "PA_baseline_multi_seeds" / "nmf_logit_motifs" / (
            f"L{seq_len}_checkpoint-15900_salient_R{rank}"
        )
    if model == "QWAB":
        if seq_len == 4096 and rank == 16:
            name = "L4096_checkpoint-15900_salient_R16_total_QWAB"
        else:
            name = f"L{seq_len}_checkpoint-15900_salient_R{rank}_total"
        return runs_root / "attn_nmf_comparison" / name
    if model == "NoPE":
        return runs_root / "attn_nmf_comparison" / f"L{seq_len}_nope_s42_salient_R{rank}"
    if model == "Rotary":
        return runs_root / "attn_nmf_comparison" / f"L{seq_len}_rotary_s42_salient_R{rank}"
    raise ValueError(f"Unknown model: {model}")


def build_comparison_plan(ranks: list, runs_root: Path) -> list:
    plan = []
    for model in ("PaTH-only", "QWAB", "NoPE", "Rotary"):
        for rank in ranks:
            train_ref = RunRef(
                model=model,
                seq_len=512,
                rank=rank,
                nmf_dir=resolve_nmf_run_dir(model, 512, rank, runs_root),
            )
            for ext_len in (2048, 4096):
                ext_ref = RunRef(
                    model=model,
                    seq_len=ext_len,
                    rank=rank,
                    nmf_dir=resolve_nmf_run_dir(model, ext_len, rank, runs_root),
                )
                plan.append((train_ref, ext_ref))
    return plan


def l2_normalize_rows(V: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norms = np.linalg.norm(V, axis=1, keepdims=True)
    return V / (norms + eps)


def classify_basis_rows(V: np.ndarray, pool_size: int) -> list:
    return [classify_basis(V[r].reshape(pool_size, pool_size)) for r in range(V.shape[0])]


def load_run_data(ref: RunRef) -> RunData:
    V = np.load(ref.nmf_dir / "V.npy").astype(np.float64, copy=False)
    U = np.load(ref.nmf_dir / "U.npy").astype(np.float64, copy=False)
    with open(ref.nmf_dir / "meta.json") as f:
        meta = json.load(f)

    pool_size = int(meta["pool_size"])
    if V.ndim != 2:
        raise ValueError(f"Expected 2D V in {ref.nmf_dir}, got shape {V.shape}")
    if U.ndim != 2:
        raise ValueError(f"Expected 2D U in {ref.nmf_dir}, got shape {U.shape}")
    if U.shape[1] != V.shape[0]:
        raise ValueError(
            f"U/V rank mismatch in {ref.nmf_dir}: U.shape={U.shape}, V.shape={V.shape}"
        )
    if V.shape[1] != pool_size * pool_size:
        raise ValueError(
            f"V feature mismatch in {ref.nmf_dir}: V.shape={V.shape}, pool_size={pool_size}"
        )

    return RunData(
        ref=ref,
        V=V,
        V_l2=l2_normalize_rows(V),
        basis_types=classify_basis_rows(V, pool_size),
        meta=meta,
        pool_size=pool_size,
        U_mean=U.mean(axis=0),
    )


def build_causal_valid_mask(M: int) -> np.ndarray:
    return np.tril(np.ones((M, M), dtype=bool)).reshape(-1)


def topk_support_mask(V: np.ndarray, top_p: float, valid_mask: np.ndarray) -> np.ndarray:
    valid_idx = np.flatnonzero(valid_mask)
    n_valid = valid_idx.shape[0]
    k = max(1, int(math.ceil(top_p * n_valid)))
    support = np.zeros_like(V, dtype=bool)

    for r in range(V.shape[0]):
        row_valid = V[r, valid_idx]
        split = row_valid.shape[0] - k
        top_idx = np.argpartition(row_valid, split)[split:]
        support[r, valid_idx[top_idx]] = True

    return support


def cosine_similarity_matrix(train_V_l2: np.ndarray, ext_V_l2: np.ndarray) -> np.ndarray:
    return train_V_l2 @ ext_V_l2.T


def support_iou_matrix(train_support: np.ndarray, ext_support: np.ndarray) -> np.ndarray:
    intersection = np.logical_and(train_support[:, None, :], ext_support[None, :, :]).sum(axis=2)
    union = np.logical_or(train_support[:, None, :], ext_support[None, :, :]).sum(axis=2)
    return intersection / np.maximum(union, 1)


def hybrid_similarity_matrix(cosine_sim: np.ndarray, iou_sim: np.ndarray) -> np.ndarray:
    return 0.5 * cosine_sim + 0.5 * iou_sim


def hungarian_match(sim: np.ndarray):
    return linear_sum_assignment(-sim)


def status_from_similarity(hybrid_sim: float) -> str:
    if hybrid_sim >= 0.6:
        return "retained"
    if hybrid_sim >= 0.3:
        return "drifted"
    return "missing"


def build_pattern_matching_rows(
    train_run: RunData,
    ext_run: RunData,
    cosine_sim: np.ndarray,
    iou_sim: np.ndarray,
    hybrid_sim: np.ndarray,
    matched_train_idx: np.ndarray,
    matched_ext_idx: np.ndarray,
) -> list:
    match_map = {int(t): int(e) for t, e in zip(matched_train_idx, matched_ext_idx)}
    rows = []

    for train_pattern_id in range(train_run.V.shape[0]):
        ext_pattern_id = match_map.get(train_pattern_id)
        if ext_pattern_id is None:
            rows.append(
                {
                    "model": train_run.ref.model,
                    "train_nmf_dir": str(train_run.ref.nmf_dir),
                    "ext_nmf_dir": str(ext_run.ref.nmf_dir),
                    "train_len": train_run.ref.seq_len,
                    "ext_len": ext_run.ref.seq_len,
                    "rank": train_run.ref.rank,
                    "train_pattern_id": train_pattern_id,
                    "train_type": train_run.basis_types[train_pattern_id],
                    "ext_pattern_id": "",
                    "cosine_vsim": "",
                    "topk_iou": "",
                    "hybrid_sim": "",
                    "status": "missing",
                }
            )
            continue

        hybrid_value = float(hybrid_sim[train_pattern_id, ext_pattern_id])
        rows.append(
            {
                "model": train_run.ref.model,
                "train_nmf_dir": str(train_run.ref.nmf_dir),
                "ext_nmf_dir": str(ext_run.ref.nmf_dir),
                "train_len": train_run.ref.seq_len,
                "ext_len": ext_run.ref.seq_len,
                "rank": train_run.ref.rank,
                "train_pattern_id": train_pattern_id,
                "train_type": train_run.basis_types[train_pattern_id],
                "ext_pattern_id": ext_pattern_id,
                "cosine_vsim": float(cosine_sim[train_pattern_id, ext_pattern_id]),
                "topk_iou": float(iou_sim[train_pattern_id, ext_pattern_id]),
                "hybrid_sim": hybrid_value,
                "status": status_from_similarity(hybrid_value),
            }
        )

    return rows


def build_ext_ood_rows(ext_run: RunData, hybrid_sim: np.ndarray) -> list:
    best_train_pattern_id = hybrid_sim.argmax(axis=0)
    max_train_sim = hybrid_sim.max(axis=0)
    rows = []

    for ext_pattern_id in range(ext_run.V.shape[0]):
        max_sim = float(max_train_sim[ext_pattern_id])
        rows.append(
            {
                "model": ext_run.ref.model,
                "ext_nmf_dir": str(ext_run.ref.nmf_dir),
                "ext_len": ext_run.ref.seq_len,
                "rank": ext_run.ref.rank,
                "ext_pattern_id": ext_pattern_id,
                "ext_type": ext_run.basis_types[ext_pattern_id],
                "best_train_pattern_id": int(best_train_pattern_id[ext_pattern_id]),
                "max_train_sim": max_sim,
                "mean_usage_ext": float(ext_run.U_mean[ext_pattern_id]),
                "selected_for_mask": (max_sim < 0.3) or (0.3 <= max_sim < 0.45),
            }
        )

    return rows


def write_csv_rows(path: Path, fieldnames: list, rows: list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run(args) -> None:
    runs_root = Path(__file__).resolve().parent.parent / "runs"
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    pattern_rows = []
    ext_ood_rows = []

    for train_ref, ext_ref in build_comparison_plan(args.rank, runs_root):
        if not train_ref.nmf_dir.exists():
            print(f"[skip] missing train dir: {train_ref.nmf_dir}")
            continue
        if not ext_ref.nmf_dir.exists():
            print(f"[skip] missing ext dir: {ext_ref.nmf_dir}")
            continue

        print(f"[running] {train_ref.model} R{train_ref.rank} L{train_ref.seq_len}→L{ext_ref.seq_len}")
        train_run = load_run_data(train_ref)
        ext_run = load_run_data(ext_ref)

        if train_run.pool_size != ext_run.pool_size:
            print(
                f"[skip] pool_size mismatch: {train_ref.nmf_dir} ({train_run.pool_size}) "
                f"vs {ext_ref.nmf_dir} ({ext_run.pool_size})"
            )
            continue
        if train_run.V.shape[1] != ext_run.V.shape[1]:
            print(
                f"[skip] feature mismatch: {train_ref.nmf_dir} ({train_run.V.shape[1]}) "
                f"vs {ext_ref.nmf_dir} ({ext_run.V.shape[1]})"
            )
            continue

        valid_mask = build_causal_valid_mask(train_run.pool_size)
        train_support = topk_support_mask(train_run.V, args.top_p, valid_mask)
        ext_support = topk_support_mask(ext_run.V, args.top_p, valid_mask)
        cosine_sim = cosine_similarity_matrix(train_run.V_l2, ext_run.V_l2)
        iou_sim = support_iou_matrix(train_support, ext_support)
        hybrid_sim = hybrid_similarity_matrix(cosine_sim, iou_sim)
        matched_train_idx, matched_ext_idx = hungarian_match(hybrid_sim)

        pattern_rows.extend(
            build_pattern_matching_rows(
                train_run=train_run,
                ext_run=ext_run,
                cosine_sim=cosine_sim,
                iou_sim=iou_sim,
                hybrid_sim=hybrid_sim,
                matched_train_idx=matched_train_idx,
                matched_ext_idx=matched_ext_idx,
            )
        )
        ext_ood_rows.extend(build_ext_ood_rows(ext_run=ext_run, hybrid_sim=hybrid_sim))

    write_csv_rows(
        out_dir / "pattern_basis_matching.csv",
        [
            "model", "train_nmf_dir", "ext_nmf_dir", "train_len", "ext_len", "rank",
            "train_pattern_id", "train_type", "ext_pattern_id",
            "cosine_vsim", "topk_iou", "hybrid_sim", "status",
        ],
        pattern_rows,
    )
    write_csv_rows(
        out_dir / "ext_ood_pattern_candidates.csv",
        [
            "model", "ext_nmf_dir", "ext_len", "rank", "ext_pattern_id", "ext_type",
            "best_train_pattern_id", "max_train_sim", "mean_usage_ext", "selected_for_mask",
        ],
        ext_ood_rows,
    )

    print(f"Saved {out_dir / 'pattern_basis_matching.csv'} ({len(pattern_rows)} rows)")
    print(f"Saved {out_dir / 'ext_ood_pattern_candidates.csv'} ({len(ext_ood_rows)} rows)")


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
