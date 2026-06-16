#!/usr/bin/env python3
"""dump_scale_attention_effects.py — PAT-200 Phase B.

Re-forwards the SAME 150 cases / SAME sampling as Phase A, capturing both the
QWAB-off attention (softmax of PaTH base logits) and the QWAB-on attention
(softmax of logits including the wavelet bias), and computes per-query
attention-effect deltas. Joined to NMF component assignment downstream.

QWAB-off = softmax(core._nmf_last_base_logits[:, :q+1])   (PaTH only)
QWAB-on  = softmax(core._nmf_last_total_logits[:, :q+1])  (+ wavelet bias)

Output: attention_effects_query_table.parquet keyed by (case_id, layer, query_idx),
matching the Phase A router_scale_query_table rows 1:1.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

# reuse Phase A helpers (same dir, run with cwd=hotpot_long)
from dump_router_scale_query import (
    render_prompt, build_token_roles, sample_query_indices, ROLES,
)

EV = ROLES.index("evidence")


def attn_effects(base_row: torch.Tensor, total_row: torch.Tensor, q: int,
                 ev_keys: np.ndarray, ks=(16, 32)) -> dict:
    """Head-aggregated (mean over heads) QWAB on-vs-off effects for one query row.

    base_row/total_row: [H, q+1] pre-mask logits (causal slice already applied).
    ev_keys: boolean array length q+1, True where the key token is evidence.
    """
    b = torch.nan_to_num(base_row.float(), nan=0.0, posinf=0.0, neginf=0.0)
    t = torch.nan_to_num(total_row.float(), nan=0.0, posinf=0.0, neginf=0.0)
    p0 = torch.softmax(b, dim=-1)   # off  [H, nk]
    p1 = torch.softmax(t, dim=-1)   # on   [H, nk]
    H, nk = p0.shape
    out = {}
    ent0 = -(p0 * (p0 + 1e-12).log()).sum(-1)
    ent1 = -(p1 * (p1 + 1e-12).log()).sum(-1)
    out["dEntropy"] = float((ent1 - ent0).mean())
    ess0 = 1.0 / (p0.pow(2).sum(-1) + 1e-12)
    ess1 = 1.0 / (p1.pow(2).sum(-1) + 1e-12)
    out["dESS"] = float((ess1 - ess0).mean())
    out["dSelfMass"] = float((p1[:, -1] - p0[:, -1]).mean())
    for k in ks:
        kk = min(k, nk)
        idx_on = p1.topk(kk, dim=-1).indices    # [H, kk]
        idx_off = p0.topk(kk, dim=-1).indices
        # FlipRate@k: 1 - overlap fraction of top-k sets (per head, mean)
        flip = []
        bmg = []
        for h in range(H):
            son = set(idx_on[h].tolist()); soff = set(idx_off[h].tolist())
            flip.append(1.0 - len(son & soff) / kk)
            # BiasMassGain@k: mass the wavelet bias adds to ON's preferred keys
            on_idx = idx_on[h]
            bmg.append(float(p1[h, on_idx].sum() - p0[h, on_idx].sum()))
        out[f"FlipRate@{k}"] = float(np.mean(flip))
        out[f"BiasMassGain@{k}"] = float(np.mean(bmg))
        # dTopKMass@k: concentration change (each uses its own top-k)
        tkm1 = p1.topk(kk, dim=-1).values.sum(-1)
        tkm0 = p0.topk(kk, dim=-1).values.sum(-1)
        out[f"dTopKMass@{k}"] = float((tkm1 - tkm0).mean())
    # dEvidenceMass: mass on evidence-token keys (on - off)
    if ev_keys.any():
        ek = torch.from_numpy(ev_keys.astype(bool))
        em1 = p1[:, ek].sum(-1)
        em0 = p0[:, ek].sum(-1)
        out["dEvidenceMass"] = float((em1 - em0).mean())
        out["has_evidence_keys"] = 1
    else:
        out["dEvidenceMass"] = float("nan")
        out["has_evidence_keys"] = 0
    return out


def run(args):
    checkpoint = str(args.checkpoint)
    seq_len = int(args.seq_len)
    n_case = int(args.n_case)
    budget = int(args.query_budget)
    min_pos = int(args.min_pos)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(int(args.seed))

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
    try:
        tok = AutoTokenizer.from_pretrained(checkpoint, use_fast=True)
        _ = tok(["x"], return_offsets_mapping=True)
    except Exception:
        tok = AutoTokenizer.from_pretrained("gpt2", use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    config = AutoConfig.from_pretrained(checkpoint)
    config.attn_implementation = "path_attn"
    model = AutoModelForCausalLM.from_pretrained(
        checkpoint, config=config, torch_dtype=torch.float32, trust_remote_code=True)
    model.eval().to(device)

    cores = []
    for name, module in model.named_modules():
        if type(module).__name__ == "PaTHAttention":
            module._nmf_capture = True        # off (base)
            module._nmf_capture_total = True  # on  (total)
            cores.append(module)
    print(f"Found {len(cores)} PaTHAttention layers (on+off capture)", flush=True)

    rows = []
    for case_idx, rec in enumerate(cases):
        prompt_str = render_prompt(rec)
        answer_str = f" {rec['answer']}"
        enc = tok(prompt_str, add_special_tokens=False, return_offsets_mapping=True)
        prompt_ids, offsets = enc["input_ids"], enc["offset_mapping"]
        answer_ids = tok(answer_str, add_special_tokens=False)["input_ids"]
        full_ids = (prompt_ids + answer_ids)[:seq_len]
        full_offsets = (offsets + [(len(prompt_str), len(prompt_str))] * len(answer_ids))[:seq_len]
        actual_len = len(full_ids)
        roles = build_token_roles(rec, prompt_str, full_offsets)
        is_ev = (roles == EV)

        input_ids = torch.tensor([full_ids], dtype=torch.long, device=device)
        with torch.no_grad():
            _ = model(input_ids)

        positions, _w = sample_query_indices(roles, actual_len, budget, min_pos, rng)
        if positions.size == 0:
            continue
        for layer_idx, core in enumerate(cores):
            base_buf = getattr(core, "_nmf_last_base_logits", None)
            tot_buf = getattr(core, "_nmf_last_total_logits", None)
            if base_buf is None or tot_buf is None:
                raise RuntimeError(f"layer {layer_idx}: missing base/total logit buffer")
            for qi in positions.tolist():
                ev_keys = is_ev[:qi + 1]
                eff = attn_effects(base_buf[0, :, qi, :qi + 1],
                                   tot_buf[0, :, qi, :qi + 1], qi, ev_keys)
                rows.append({"case_id": case_idx, "layer": layer_idx,
                             "query_idx": int(qi), "role": ROLES[int(roles[qi])], **eff})
        for core in cores:
            core._nmf_last_base_logits = None
            core._nmf_last_total_logits = None
        if (case_idx + 1) % 5 == 0 or case_idx + 1 == len(cases):
            print(f"  case {case_idx+1}/{len(cases)} | rows={len(rows)}", flush=True)
        if device.type == "cuda":
            torch.cuda.empty_cache()

    df = pd.DataFrame(rows)
    path = out_dir / "attention_effects_query_table.parquet"
    df.to_parquet(path, index=False)
    print(f"Done. rows={len(df)} -> {path}", flush=True)
    print(df[["FlipRate@16", "FlipRate@32", "dEntropy", "dESS",
              "dEvidenceMass"]].describe().to_string(), flush=True)


def main():
    ap = argparse.ArgumentParser(description="PAT-200 Phase B attention-effect dump")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--jsonl", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--seq_len", type=int, default=4096)
    ap.add_argument("--n_case", type=int, default=150)
    ap.add_argument("--query_budget", type=int, default=384)
    ap.add_argument("--min_pos", type=int, default=16)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    run(args)


if __name__ == "__main__":
    main()
