#!/usr/bin/env python3
"""dump_router_scale_query.py — PAT-200 Phase A.

Query-level dump of the QWAB head-shared scale router for HotpotQA-Long Uniform.

For each (case, layer, sampled query) it records the head-shared router output
(pi_null, pi_s1..pi_sK), derived router-state features, a task-role label, the
absolute/normalized query position, and (optionally) head-aggregated base-attention
features derived from the captured pre-softmax PaTH base logits (QWAB-off attention).

Main row unit = the individual query (NOT (case,layer) averages), per PAT-200.

Outputs (under --out_dir):
  router_scale_query_table.parquet     # main, query-level rows
  router_scale_layer_summary.parquet   # aux, (case,layer) mean profile
  dump_meta.json

Router instrumentation (FLA fla/layers/path_attn.py, already present):
  core._last_ctxscale_router_prob  [B,T,H,K]  (scale probs, head-shared broadcast)
  core._last_ctxscale_null_mass    [B,T,(H,)1]  (pi_null)
Base-attention capture (already present): core._nmf_capture -> core._nmf_last_base_logits [B,H,T,T].
"""

import argparse
import json
import re
import subprocess
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


# ── text rendering (matches dump_path_nmf.py / eval_hotpot_long.py) ───────────

def render_doc(title: str, sentences: list) -> str:
    return f"Title: {title}\n{' '.join(sentences).strip()}\n\n"


def render_prompt(ex: dict) -> str:
    ctx = "".join(render_doc(t, s) for t, s in ex["context"])
    return f"Question: {ex['question']}\n\nContext:\n{ctx}Answer:"


def _git_sha(repo_path: str) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", repo_path, "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return "unknown"


# ── task-role labelling ──────────────────────────────────────────────────────
#
# Roles (single primary, priority order answer_near > evidence > question >
# title_special > irrelevant). Char spans are located in the rendered prompt and
# mapped to token indices via the fast tokenizer's offset mapping, so the labels
# are robust to tokenisation details.

ROLES = ["question", "evidence", "answer_near", "title_special", "irrelevant"]


def _char_spans_of_substr(haystack: str, needle: str) -> List[Tuple[int, int]]:
    spans = []
    if not needle or len(needle) < 3:
        return spans
    start = 0
    while True:
        idx = haystack.find(needle, start)
        if idx < 0:
            break
        spans.append((idx, idx + len(needle)))
        start = idx + len(needle)
    return spans


def build_token_roles(
    ex: dict,
    prompt_str: str,
    offsets: List[Tuple[int, int]],
) -> np.ndarray:
    """Return an int array [n_tokens] of role-id per token (index into ROLES).

    Token role = role of the highest-priority char-span that the token overlaps.
    """
    n = len(offsets)
    # priority masks at char level via per-token assignment
    role_pri = {"answer_near": 4, "evidence": 3, "question": 2, "title_special": 1, "irrelevant": 0}
    pri_to_role = {v: k for k, v in role_pri.items()}

    # Collect char spans per role.
    spans = {r: [] for r in ROLES}

    # question: the question string + the "Question:" marker region
    q = ex.get("question", "")
    spans["question"].extend(_char_spans_of_substr(prompt_str, q))

    # title_special: structural markers and titles
    for marker in ["Question:", "Context:", "Answer:"]:
        spans["title_special"].extend(_char_spans_of_substr(prompt_str, marker))
    for title, _sents in ex.get("context", []):
        spans["title_special"].extend(_char_spans_of_substr(prompt_str, f"Title: {title}"))

    # evidence: supporting-fact sentences
    title_to_sents = {t: s for t, s in ex.get("context", [])}
    for sf in ex.get("supporting_facts", []):
        try:
            sf_title, sf_idx = sf[0], int(sf[1])
        except Exception:
            continue
        sents = title_to_sents.get(sf_title)
        if not sents or sf_idx >= len(sents):
            continue
        sent = sents[sf_idx].strip()
        spans["evidence"].extend(_char_spans_of_substr(prompt_str, sent))

    # answer_near: answer string + aliases occurrences (skip trivial yes/no for span,
    # but still mark; yes/no answers have no informative span so this is best-effort)
    ans_strings = [ex.get("answer", "")] + list(ex.get("answer_aliases", []) or [])
    for a in ans_strings:
        a = (a or "").strip()
        if len(a) >= 3:
            spans["answer_near"].extend(_char_spans_of_substr(prompt_str, a))

    # Build a per-char priority array, then reduce to tokens.
    L = len(prompt_str)
    char_pri = np.zeros(L, dtype=np.int8)
    for r in ROLES:
        if r == "irrelevant":
            continue
        p = role_pri[r]
        for (c0, c1) in spans[r]:
            c0 = max(0, c0); c1 = min(L, c1)
            if c1 > c0:
                seg = char_pri[c0:c1]
                np.maximum(seg, p, out=seg)

    roles = np.zeros(n, dtype=np.int64)  # default irrelevant (0)
    for ti, (a, b) in enumerate(offsets):
        if b <= a:
            roles[ti] = role_pri["irrelevant"]
            continue
        pr = int(char_pri[a:b].max()) if b <= L else 0
        roles[ti] = pr
    # map priority value -> ROLES index
    pri_to_idx = {role_pri[r]: ROLES.index(r) for r in ROLES}
    roles = np.vectorize(pri_to_idx.get)(roles).astype(np.int64)
    return roles


# ── base-attention features (from captured base logits) ──────────────────────

def base_attn_features_for_query(
    base_logits_row: torch.Tensor,  # [H, q+1]  pre-softmax base PaTH logits
    q: int,
    prefix_w: int = 64,
    far_d: int = 512,
    topk: int = 16,
) -> dict:
    """Head-aggregated (mean over heads) features of the QWAB-off base attention row."""
    x = base_logits_row.float()
    x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    H, nk = x.shape
    p = torch.softmax(x, dim=-1)  # [H, nk]
    # self mass = attention to the diagonal (last key == query position)
    self_mass = p[:, -1]
    ent = -(p * (p + 1e-12).log()).sum(dim=-1)
    prefix_mass = p[:, :min(prefix_w, nk)].sum(dim=-1)
    far_cut = q - far_d
    far_mass = p[:, :far_cut].sum(dim=-1) if far_cut > 0 else torch.zeros(H)
    k = min(topk, nk)
    topk_mass = p.topk(k, dim=-1).values.sum(dim=-1)
    sorted_logits = x.sort(dim=-1, descending=True).values
    margin = (sorted_logits[:, 0] - sorted_logits[:, 1]) if nk >= 2 else torch.zeros(H)
    return {
        "selfmass_base": float(self_mass.mean()),
        "entropy_base": float(ent.mean()),
        "prefix_mass_base": float(prefix_mass.mean()),
        "far_mass_base": float(far_mass.mean()),
        "topk_mass_base": float(topk_mass.mean()),
        "base_logit_margin": float(margin.mean()),
    }


# ── query sampling (role-stratified) ─────────────────────────────────────────

def sample_query_indices(roles: np.ndarray, actual_len: int, budget: int,
                         min_pos: int, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    """Role-stratified sample of query positions, shared across layers within a case.

    Returns (positions, sampling_weight) where sampling_weight[i] = n_role_total /
    n_role_sampled so the natural distribution can be recovered by reweighting.
    """
    valid = np.arange(min_pos, actual_len)
    if valid.size == 0:
        return np.array([], dtype=np.int64), np.array([], dtype=np.float64)
    r = roles[valid]
    # Quota: take ALL of the rare informative roles (evidence, answer_near, question)
    # up to a per-role cap, then fill remaining budget from title_special/irrelevant.
    rare = [ROLES.index("answer_near"), ROLES.index("evidence"), ROLES.index("question")]
    common = [ROLES.index("title_special"), ROLES.index("irrelevant")]
    chosen = []
    weights = {}
    per_rare_cap = max(1, budget // 3)
    for rid in rare:
        idx = valid[r == rid]
        if idx.size == 0:
            continue
        take = min(idx.size, per_rare_cap)
        sel = rng.choice(idx, size=take, replace=False)
        chosen.append(sel)
        weights[rid] = idx.size / take
    taken = sum(len(c) for c in chosen)
    remaining = max(0, budget - taken)
    # split remaining across common roles proportional to availability
    common_idx = {rid: valid[r == rid] for rid in common}
    avail_total = sum(ci.size for ci in common_idx.values())
    for rid in common:
        idx = common_idx[rid]
        if idx.size == 0 or remaining <= 0 or avail_total == 0:
            continue
        quota = int(round(remaining * idx.size / avail_total))
        take = min(idx.size, max(0, quota))
        if take == 0:
            continue
        sel = rng.choice(idx, size=take, replace=False)
        chosen.append(sel)
        weights[rid] = idx.size / take
    if not chosen:
        return np.array([], dtype=np.int64), np.array([], dtype=np.float64)
    positions = np.concatenate(chosen)
    order = np.argsort(positions)
    positions = positions[order]
    pos_roles = roles[positions]
    w = np.array([weights.get(int(rr), 1.0) for rr in pos_roles], dtype=np.float64)
    return positions.astype(np.int64), w


def pos_bucket(p: int) -> str:
    if p < 512:
        return "0-512"
    if p < 1024:
        return "512-1024"
    if p < 2048:
        return "1024-2048"
    return "2048-4096"


def layer_bin(layer: int, n_layers: int) -> str:
    third = n_layers / 3.0
    if layer < third:
        return "low"
    if layer < 2 * third:
        return "mid"
    return "high"


# ── main ─────────────────────────────────────────────────────────────────────

def run(args):
    checkpoint = str(args.checkpoint)
    seq_len = int(args.seq_len)
    n_case = int(args.n_case)
    budget = int(args.query_budget)
    min_pos = int(args.min_pos)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(int(args.seed))

    # Load cases (HotpotQA-Long Uniform, target length == seq_len)
    cases = []
    with open(args.jsonl) as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            if int(rec.get("meta", {}).get("target_total_tokens", -1)) == seq_len:
                cases.append(rec)
                if len(cases) >= n_case:
                    break
    print(f"Loaded {len(cases)} cases at seq_len={seq_len}", flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # fast tokenizer for offset mapping (gpt2 vocab)
    try:
        tok = AutoTokenizer.from_pretrained(checkpoint, use_fast=True)
        _ = tok(["x"], return_offsets_mapping=True)
    except Exception:
        tok = AutoTokenizer.from_pretrained("gpt2", use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    config = AutoConfig.from_pretrained(checkpoint)
    config.attn_implementation = "path_attn"
    K = int(getattr(config, "router_band_num", 8))
    print(f"router_band_num K={K}", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        checkpoint, config=config, torch_dtype=torch.float32, trust_remote_code=True,
    )
    model.eval().to(device)

    # Locate PaTHAttention cores; enable router + (optional) base-logit capture.
    cores = []
    for name, module in model.named_modules():
        if type(module).__name__ == "PaTHAttention":
            if args.capture_base_attn:
                module._nmf_capture = True
            cores.append(module)
    n_layers = len(cores)
    print(f"Found {n_layers} PaTHAttention layers (capture_base_attn={args.capture_base_attn})", flush=True)
    if n_layers == 0:
        raise RuntimeError("No PaTHAttention layers found")

    rows: List[dict] = []
    layer_summary: List[dict] = []
    head_shared_checked = False

    for case_idx, rec in enumerate(cases):
        prompt_str = render_prompt(rec)
        answer_str = f" {rec['answer']}"
        enc = tok(prompt_str, add_special_tokens=False, return_offsets_mapping=True)
        prompt_ids = enc["input_ids"]
        offsets = enc["offset_mapping"]
        answer_ids = tok(answer_str, add_special_tokens=False)["input_ids"]
        full_ids = (prompt_ids + answer_ids)[:seq_len]
        # offsets only defined over prompt tokens; pad answer-token offsets as (len,len)
        full_offsets = (offsets + [(len(prompt_str), len(prompt_str))] * len(answer_ids))[:seq_len]
        actual_len = len(full_ids)
        roles = build_token_roles(rec, prompt_str, full_offsets)  # [actual_len]

        input_ids = torch.tensor([full_ids], dtype=torch.long, device=device)
        with torch.no_grad():
            out = model(input_ids, output_hidden_states=True)
        # hidden_states: tuple (n_layers+1) of [1, T, d]; index l is input to layer l
        hs = out.hidden_states

        positions, weights = sample_query_indices(roles, actual_len, budget, min_pos, rng)
        if positions.size == 0:
            print(f"  case {case_idx}: no sampled positions, skipping", flush=True)
            continue

        for layer_idx, core in enumerate(cores):
            prob = getattr(core, "_last_ctxscale_router_prob", None)
            nullm = getattr(core, "_last_ctxscale_null_mass", None)
            if prob is None:
                raise RuntimeError(f"layer {layer_idx}: no _last_ctxscale_router_prob")
            p = prob[0].detach().float().cpu()       # [T,H,K] or [T,K]
            if p.dim() == 3:
                if not head_shared_checked:
                    # verify head-shared: heads identical
                    dev = (p[min_pos:actual_len] - p[min_pos:actual_len, :1]).abs().max().item()
                    print(f"  head-shared check: max head deviation = {dev:.3e}", flush=True)
                    head_shared_checked = True
                p = p.mean(dim=1)                    # [T,K] (head-shared -> mean == any head)
            # null mass per token
            if nullm is not None:
                nm = nullm[0].detach().float().cpu()
                nm = nm.reshape(nm.shape[0], -1).mean(dim=1)  # [T]
            else:
                nm = 1.0 - p.sum(dim=-1)

            # base logits for this layer (optional)
            base_buf = getattr(core, "_nmf_last_base_logits", None) if args.capture_base_attn else None

            # layer summary (mean over sampled positions for join-friendliness)
            pi_s_layer = p[positions]                # [n_sel, K]
            null_layer = nm[positions]               # [n_sel]
            layer_summary.append({
                "case_id": case_idx, "task": "hotpotqa_long", "length": seq_len,
                "layer": layer_idx,
                "mean_pi_null": float(null_layer.mean()),
                **{f"mean_pi_s{j+1}": float(pi_s_layer[:, j].mean()) for j in range(K)},
            })

            hs_layer = hs[layer_idx][0]              # [T,d] input to this layer
            for qi, w in zip(positions.tolist(), weights.tolist()):
                pi_s = p[qi].numpy().astype(np.float64)   # [K]
                pi_null = float(nm[qi])
                wav_mass = float(pi_s.sum())
                denom = wav_mass + 1e-8
                norm_pi = pi_s / denom
                # router-state features
                full9 = np.concatenate([[max(pi_null, 0.0)], np.clip(pi_s, 0, None)])
                full9 = full9 / (full9.sum() + 1e-12)
                router_out_entropy = float(-(full9 * np.log(full9 + 1e-12)).sum())
                srt = np.sort(norm_pi)[::-1]
                top_scale_margin = float(srt[0] - srt[1]) if K >= 2 else float(srt[0])
                router_in_norm = float(hs_layer[qi].norm().item())

                row = {
                    "case_id": case_idx, "task": "hotpotqa_long", "length": seq_len,
                    "layer": layer_idx, "layer_bin": layer_bin(layer_idx, n_layers),
                    "query_idx": int(qi), "norm_pos": float(qi / max(1, actual_len - 1)),
                    "pos_bucket": pos_bucket(qi),
                    "role": ROLES[int(roles[qi])],
                    "sampling_weight": float(w),
                    "pi_null": pi_null, "WavMass": wav_mass,
                    "router_out_entropy": router_out_entropy,
                    "top_scale_margin": top_scale_margin,
                    "router_in_norm": router_in_norm,
                }
                for j in range(K):
                    row[f"pi_s{j+1}"] = float(pi_s[j])
                    row[f"norm_pi_s{j+1}"] = float(norm_pi[j])
                for r in ROLES:
                    row[f"is_{r}"] = int(ROLES[int(roles[qi])] == r)
                if base_buf is not None:
                    blr = base_buf[0, :, qi, :qi + 1]  # [H, qi+1]
                    row.update(base_attn_features_for_query(blr, qi))
                rows.append(row)

        if (case_idx + 1) % 5 == 0 or case_idx + 1 == len(cases):
            print(f"  case {case_idx+1}/{len(cases)} | rows={len(rows)}", flush=True)
        # free per-case buffers
        for core in cores:
            if hasattr(core, "_nmf_last_base_logits"):
                core._nmf_last_base_logits = None
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if not rows:
        print("ERROR: no rows produced", flush=True)
        return
    df = pd.DataFrame(rows)
    df_sum = pd.DataFrame(layer_summary)
    main_path = out_dir / "router_scale_query_table.parquet"
    sum_path = out_dir / "router_scale_layer_summary.parquet"
    df.to_parquet(main_path, index=False)
    df_sum.to_parquet(sum_path, index=False)

    meta = {
        "checkpoint": checkpoint, "seq_len": seq_len, "n_case": len(cases),
        "n_layers": n_layers, "K": K, "query_budget": budget, "min_pos": min_pos,
        "capture_base_attn": bool(args.capture_base_attn),
        "n_rows": len(df), "roles": ROLES, "seed": int(args.seed),
        "role_counts": df["role"].value_counts().to_dict(),
        "git_sha_transformers": _git_sha(str(Path(__file__).parents[4])),
        "git_sha_fla": _git_sha("/cl/work5/hongyu-s/flash-linear-attention"),
    }
    with open(out_dir / "dump_meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Done. rows={len(df)}  -> {main_path}", flush=True)
    print(f"role counts: {meta['role_counts']}", flush=True)


def main():
    ap = argparse.ArgumentParser(description="PAT-200 query-level router scale dump")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--jsonl", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--seq_len", type=int, default=4096)
    ap.add_argument("--n_case", type=int, default=150)
    ap.add_argument("--query_budget", type=int, default=384,
                    help="sampled query positions per case (shared across layers)")
    ap.add_argument("--min_pos", type=int, default=16)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--capture_base_attn", action="store_true", default=True)
    ap.add_argument("--no_base_attn", dest="capture_base_attn", action="store_false")
    args = ap.parse_args()
    run(args)


if __name__ == "__main__":
    main()
