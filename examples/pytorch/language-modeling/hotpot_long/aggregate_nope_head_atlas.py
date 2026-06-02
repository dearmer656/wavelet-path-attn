#!/usr/bin/env python3
"""
aggregate_nope_head_atlas.py — PAT-178: aggregate per-head V/D/P features into atlas CSVs.

Reads all pattern_features.json under feat_root (block_NNN/model_tag/), groups by
(layer, head), computes mean/std over cases for vertical_score, diagonal_score,
periodic_seq_score in both all_query and target_query scopes.

If two block_NNN directories exist under a common parent (e.g., block_512 and block_4096),
also computes length-wise drift: DeltaV, DeltaD, DeltaP, drift_norm per (layer, head).

Usage:
  python aggregate_nope_head_atlas.py \
      --feat_root .../attn_analysis_s42/block_512/nope_s42_ckpt15900 \
      --out_csv ./atlas_nope_s42

Outputs:
  {out_csv}_all_query.csv
  {out_csv}_target_query.csv
  {out_csv}_drift_L{N}.csv  (if drift_against_feat_root is given)
"""

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path


FEAT_KEYS = ["vertical_score", "diagonal_score", "periodic_seq_score"]


def load_feat_root(feat_root: Path) -> dict:
    """Load all pattern_features.json under feat_root.

    Returns: {(layer, head): {"all_query": [floats...] * n_cases, ...}}
    """
    data = defaultdict(lambda: {"all_query": defaultdict(list), "target_query": defaultdict(list)})
    count = 0
    for pf in sorted(feat_root.rglob("pattern_features.json")):
        try:
            d = json.loads(pf.read_text())
        except Exception as e:
            print(f"  warning: failed to read {pf}: {e}")
            continue
        if "per_head" not in d:
            continue  # skip head-averaged files from old format
        layer = int(d["meta"]["layer"])
        for scope in ("all_query", "target_query"):
            head_list = d["per_head"].get(scope, [])
            for head_idx, feats in enumerate(head_list):
                for k in FEAT_KEYS:
                    v = feats.get(k, float("nan"))
                    data[(layer, head_idx)][scope][k].append(v)
        count += 1
    print(f"  Loaded {count} pattern_features.json files from {feat_root}")
    return data


def _mean(xs):
    valid = [x for x in xs if not math.isnan(x)]
    return sum(valid) / len(valid) if valid else float("nan")


def _std(xs):
    valid = [x for x in xs if not math.isnan(x)]
    if len(valid) < 2:
        return float("nan")
    m = sum(valid) / len(valid)
    return math.sqrt(sum((x - m) ** 2 for x in valid) / len(valid))


def write_atlas_csv(data: dict, scope: str, out_path: Path):
    rows = []
    for (layer, head), scopes in sorted(data.items()):
        vals = scopes[scope]
        row = {"layer": layer, "head": head}
        n = len(vals.get("vertical_score", []))
        row["n_cases"] = n
        for k in FEAT_KEYS:
            xs = vals.get(k, [])
            short = k.replace("_score", "").replace("periodic_seq", "P").replace("vertical", "V").replace("diagonal", "D")
            row[f"{short}_mean"] = _mean(xs)
            row[f"{short}_std"] = _std(xs)
        rows.append(row)

    if not rows:
        print(f"  no data for scope={scope}, skipping {out_path}")
        return
    fieldnames = list(rows[0].keys())
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"  Wrote {len(rows)} rows -> {out_path}")


def write_drift_csv(data_ref: dict, data_long: dict, scope: str, out_path: Path):
    rows = []
    keys = sorted(set(data_ref.keys()) & set(data_long.keys()))
    for (layer, head) in keys:
        ref_vals = data_ref[(layer, head)][scope]
        long_vals = data_long[(layer, head)][scope]
        row = {"layer": layer, "head": head, "scope": scope}
        drift_sq = 0.0
        for feat, short in [("vertical_score", "V"), ("diagonal_score", "D"), ("periodic_seq_score", "P")]:
            v_ref = _mean(ref_vals.get(feat, []))
            v_long = _mean(long_vals.get(feat, []))
            delta = float("nan") if (math.isnan(v_ref) or math.isnan(v_long)) else v_long - v_ref
            row[f"Delta{short}"] = delta
            if not math.isnan(delta):
                drift_sq += delta ** 2
        row["drift_norm"] = math.sqrt(drift_sq) if drift_sq > 0 else float("nan")
        rows.append(row)

    if not rows:
        print(f"  no drift data for scope={scope}, skipping {out_path}")
        return
    fieldnames = list(rows[0].keys())
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"  Wrote {len(rows)} rows -> {out_path}")


def main():
    parser = argparse.ArgumentParser(description="PAT-178: aggregate per-head V/D/P atlas")
    parser.add_argument("--feat_root", required=True,
                        help="Path to block_NNN/model_tag/ directory containing layer*/case*/pattern_features.json")
    parser.add_argument("--out_csv", required=True,
                        help="Output CSV prefix (will write {prefix}_all_query.csv, {prefix}_target_query.csv)")
    parser.add_argument("--drift_against", default=None,
                        help="Optional second feat_root (different seq_len) to compute drift against feat_root")
    args = parser.parse_args()

    feat_root = Path(args.feat_root)
    out_prefix = Path(args.out_csv)

    print(f"Loading features from {feat_root}")
    data = load_feat_root(feat_root)

    for scope in ("all_query", "target_query"):
        write_atlas_csv(data, scope, Path(f"{args.out_csv}_{scope}.csv"))

    if args.drift_against:
        drift_root = Path(args.drift_against)
        # infer long seq_len from drift_root name (block_NNN)
        long_tag = drift_root.parent.name if "block_" in drift_root.parent.name else drift_root.name
        print(f"Loading drift reference from {drift_root}")
        data_long = load_feat_root(drift_root)
        for scope in ("all_query", "target_query"):
            write_drift_csv(data, data_long, scope, Path(f"{args.out_csv}_drift_{long_tag}_{scope}.csv"))


if __name__ == "__main__":
    main()
