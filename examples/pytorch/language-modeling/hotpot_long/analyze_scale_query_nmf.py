#!/usr/bin/env python3
"""analyze_scale_query_nmf.py — PAT-200 Phase A analysis.

Reads the query-level router-scale table (router_scale_query_table.parquet) and:
  1. Builds X = [norm_pi_s1..K] (wavelet scales only; null kept separate).
  2. Case-level 80:20 held-out split.
  3. NMF sweep R=2..8 over multiple seeds; held-out usage via fixed-V NNLS.
  4. Component stability (Hungarian-matched cosine across seeds).
  5. Component interpretation (dominant scale, fine/mid/coarse mass, entropy).
  6. Query enrichment (ER = P(attr|top-5% of comp) / P(attr)).
  7. Controls: permutation null + layer-only / layer+position predictive baselines.
  8. Decision gates + scale_query_nmf_report.md.

Outputs (under --out_dir, same dir as the table by default):
  scale_nmf_R{R}_components.csv, scale_nmf_R{R}_usage_train.parquet,
  scale_nmf_R{R}_usage_heldout.parquet, scale_nmf_R_sweep_summary.csv,
  component_query_enrichment_R{R}.csv, scale_query_nmf_report.md
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import nnls
from scipy.optimize import linear_sum_assignment
from sklearn.decomposition import NMF
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler


# ── band scale grouping ──────────────────────────────────────────────────────
# QWAB router bands s1..s8 are ordered fine→coarse (near→far wavelet dilation,
# table_direction="near_to_far" in fla path_attn). Grouping is overridable and is
# reported explicitly so it can be re-checked against the scale grid.
def band_groups(K: int):
    third = max(1, K // 3)
    fine = list(range(0, third))
    mid = list(range(third, 2 * third))
    coarse = list(range(2 * third, K))
    return fine, mid, coarse


def component_profile(v: np.ndarray, K: int) -> dict:
    p = v / (v.sum() + 1e-12)
    fine, mid, coarse = band_groups(K)
    ent = float(-(p * np.log(p + 1e-12)).sum())
    return {
        "dominant_scale": int(np.argmax(v)) + 1,
        "fine_mass": float(p[fine].sum()),
        "mid_mass": float(p[mid].sum()),
        "coarse_mass": float(p[coarse].sum()),
        "entropy": ent,
        **{f"v_s{j+1}": float(v[j]) for j in range(K)},
    }


# ── NMF helpers ──────────────────────────────────────────────────────────────

def fit_nmf(X, R, seed, max_iter=600):
    nmf = NMF(n_components=R, init="nndsvdar", random_state=seed,
              max_iter=max_iter, solver="cd")
    U = nmf.fit_transform(X)
    V = nmf.components_  # [R,K]
    return U, V, float(nmf.reconstruction_err_)


def nnls_usage(X, V):
    """Solve U>=0 minimizing ||X - U V||_F row by row (fixed V)."""
    A = V.T  # [K,R]
    U = np.zeros((X.shape[0], V.shape[0]), dtype=np.float64)
    for i in range(X.shape[0]):
        U[i], _ = nnls(A, X[i])
    return U


def rel_recon_err(X, U, V):
    R = X - U @ V
    return float(np.linalg.norm(R) / (np.linalg.norm(X) + 1e-12))


def matched_cosine(Va, Vb):
    """Hungarian-match rows of Va to Vb by cosine; return matched cosine list."""
    na = Va / (np.linalg.norm(Va, axis=1, keepdims=True) + 1e-12)
    nb = Vb / (np.linalg.norm(Vb, axis=1, keepdims=True) + 1e-12)
    C = na @ nb.T  # [R,R] cosine
    ri, ci = linear_sum_assignment(-C)
    return C[ri, ci]


# ── enrichment ───────────────────────────────────────────────────────────────

def build_attributes(df: pd.DataFrame) -> pd.DataFrame:
    A = pd.DataFrame(index=df.index)
    for lb in ["low", "mid", "high"]:
        A[f"layer_{lb}"] = (df["layer_bin"] == lb).astype(int)
    for pb in ["0-512", "512-1024", "1024-2048", "2048-4096"]:
        A[f"pos_{pb}"] = (df["pos_bucket"] == pb).astype(int)
    for r in ["question", "evidence", "answer_near", "title_special", "irrelevant"]:
        col = f"is_{r}"
        if col in df:
            A[f"role_{r}"] = df[col].astype(int)
    # continuous features -> high (above global median) buckets
    cont = [c for c in ["selfmass_base", "entropy_base", "prefix_mass_base",
                        "far_mass_base", "topk_mass_base", "base_logit_margin",
                        "WavMass", "router_out_entropy", "top_scale_margin",
                        "router_in_norm"] if c in df]
    for c in cont:
        med = df[c].median()
        A[f"{c}_high"] = (df[c] > med).astype(int)
    return A


def enrichment(U, attrs: pd.DataFrame, top_frac=0.05):
    """ER(attr,r) = P(attr | top-frac of comp r) / P(attr)."""
    N = U.shape[0]
    base = attrs.mean(axis=0).replace(0, np.nan)
    n_top = max(1, int(round(N * top_frac)))
    rows = []
    for r in range(U.shape[1]):
        order = np.argsort(-U[:, r])[:n_top]
        sub = attrs.iloc[order]
        pa = sub.mean(axis=0)
        er = (pa / base)
        for a in attrs.columns:
            rows.append({"component": r, "attribute": a,
                        "P_attr": float(base[a]) if not np.isnan(base[a]) else 0.0,
                        "P_attr_given_comp": float(pa[a]),
                        "ER": float(er[a]) if np.isfinite(er[a]) else np.nan})
    return pd.DataFrame(rows)


def permutation_null_max_er(U, attrs, top_frac=0.05, n_perm=200, seed=0):
    """Null distribution of per-component max-ER under random top-set selection."""
    rng = np.random.default_rng(seed)
    N = U.shape[0]
    base = attrs.mean(axis=0).values
    base_safe = np.where(base == 0, np.nan, base)
    n_top = max(1, int(round(N * top_frac)))
    Avals = attrs.values
    maxers = []
    for _ in range(n_perm):
        order = rng.choice(N, size=n_top, replace=False)
        pa = Avals[order].mean(axis=0)
        er = pa / base_safe
        maxers.append(np.nanmax(er))
    return float(np.percentile(maxers, 95)), float(np.mean(maxers))


# ── predictive baselines ─────────────────────────────────────────────────────

FEATURE_SETS = {
    "layer_only": ["layer"],
    "layer_pos": ["layer", "query_idx", "norm_pos"],
    "full": ["layer", "query_idx", "norm_pos",
             "is_question", "is_evidence", "is_answer_near", "is_title_special",
             "selfmass_base", "entropy_base", "prefix_mass_base", "far_mass_base",
             "topk_mass_base", "base_logit_margin",
             "WavMass", "router_out_entropy", "top_scale_margin", "router_in_norm"],
}


def predictive_macro_f1(df_tr, U_tr, df_ho, U_ho):
    """Predict component-argmax label from feature sets; macro-F1 on held-out."""
    y_tr = U_tr.argmax(axis=1)
    y_ho = U_ho.argmax(axis=1)
    out = {}
    for name, feats in FEATURE_SETS.items():
        feats = [f for f in feats if f in df_tr.columns and f in df_ho.columns]
        Xtr = df_tr[feats].fillna(0.0).values
        Xho = df_ho[feats].fillna(0.0).values
        sc = StandardScaler().fit(Xtr)
        clf = LogisticRegression(max_iter=400, multi_class="auto", C=1.0)
        try:
            clf.fit(sc.transform(Xtr), y_tr)
            pred = clf.predict(sc.transform(Xho))
            out[name] = float(f1_score(y_ho, pred, average="macro"))
        except Exception as e:
            out[name] = float("nan")
    return out


# ── main ─────────────────────────────────────────────────────────────────────

def run(args):
    table = Path(args.table)
    out_dir = Path(args.out_dir) if args.out_dir else table.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(table)
    K = sum(1 for c in df.columns if c.startswith("norm_pi_s"))
    print(f"Loaded {len(df)} rows, K={K} scales", flush=True)

    Xcols = [f"norm_pi_s{j+1}" for j in range(K)]
    X = df[Xcols].values.astype(np.float64)
    X = np.clip(np.nan_to_num(X), 0, None)

    # case-level 80:20 split
    rng = np.random.default_rng(args.seed)
    cases = np.sort(df["case_id"].unique())
    rng.shuffle(cases)
    n_ho = max(1, int(round(len(cases) * 0.2)))
    ho_cases = set(cases[:n_ho].tolist())
    is_ho = df["case_id"].isin(ho_cases).values
    Xtr, Xho = X[~is_ho], X[is_ho]
    df_tr, df_ho = df[~is_ho].reset_index(drop=True), df[is_ho].reset_index(drop=True)
    print(f"train rows={len(Xtr)} ({len(cases)-n_ho} cases), heldout rows={len(Xho)} ({n_ho} cases)", flush=True)

    attrs_tr = build_attributes(df_tr)
    Rs = list(range(2, 9))
    seeds = list(range(args.n_seeds))
    sweep = []
    chosen = {}

    for R in Rs:
        Us, Vs, train_errs = [], [], []
        for s in seeds:
            U, V, err = fit_nmf(Xtr, R, seed=s)
            Us.append(U); Vs.append(V); train_errs.append(rel_recon_err(Xtr, U, V))
        # stability: matched cosine of seeds 1.. vs seed 0
        stab = []
        for s in range(1, len(seeds)):
            stab.extend(matched_cosine(Vs[0], Vs[s]).tolist())
        stab_mean = float(np.mean(stab)) if stab else 1.0
        stab_frac_085 = float(np.mean(np.array(stab) >= 0.85)) if stab else 1.0
        # primary seed 0
        U0, V0 = Us[0], Vs[0]
        U_ho = nnls_usage(Xho, V0)
        ho_err = rel_recon_err(Xho, U_ho, V0)
        chosen[R] = (U0, V0, U_ho)
        sweep.append({
            "R": R,
            "train_rel_recon_err": float(np.mean(train_errs)),
            "heldout_rel_recon_err": ho_err,
            "stability_mean_cosine": stab_mean,
            "stability_frac_ge_0.85": stab_frac_085,
        })
        print(f"R={R}: train_err={np.mean(train_errs):.4f} ho_err={ho_err:.4f} "
              f"stab={stab_mean:.3f} frac085={stab_frac_085:.2f}", flush=True)

        # save components + usage
        comp_rows = [{"component": r, **component_profile(V0[r], K)} for r in range(R)]
        pd.DataFrame(comp_rows).to_csv(out_dir / f"scale_nmf_R{R}_components.csv", index=False)
        ucols = [f"U{r}" for r in range(R)]
        df_tr_u = df_tr.copy()
        df_tr_u[ucols] = U0
        df_tr_u.to_parquet(out_dir / f"scale_nmf_R{R}_usage_train.parquet", index=False)
        df_ho_u = df_ho.copy()
        df_ho_u[ucols] = U_ho
        df_ho_u.to_parquet(out_dir / f"scale_nmf_R{R}_usage_heldout.parquet", index=False)

        # enrichment (train)
        enr = enrichment(U0, attrs_tr, top_frac=args.top_frac)
        enr.to_csv(out_dir / f"component_query_enrichment_R{R}.csv", index=False)

    sweep_df = pd.DataFrame(sweep)
    sweep_df.to_csv(out_dir / "scale_nmf_R_sweep_summary.csv", index=False)

    # ── detailed analysis for chosen R (best stability among {5,6}, else best overall)
    cand = sweep_df[sweep_df["R"].isin([5, 6])]
    if len(cand):
        R_star = int(cand.sort_values("stability_mean_cosine", ascending=False).iloc[0]["R"])
    else:
        R_star = int(sweep_df.sort_values("stability_mean_cosine", ascending=False).iloc[0]["R"])
    U0, V0, U_ho = chosen[R_star]
    enr = enrichment(U0, attrs_tr, top_frac=args.top_frac)
    enr_ho = enrichment(U_ho, build_attributes(df_ho), top_frac=args.top_frac)
    perm95, perm_mean = permutation_null_max_er(U0, attrs_tr, top_frac=args.top_frac,
                                                n_perm=args.n_perm, seed=args.seed)
    f1 = predictive_macro_f1(df_tr, U0, df_ho, U_ho)

    # gate evaluation
    g1_frac = float(sweep_df[sweep_df.R == R_star]["stability_frac_ge_0.85"].iloc[0])
    gate1 = g1_frac >= 0.5
    # gate2: >=2 comps with a real ER>=1.5 (beyond perm null) OR macro-F1 lift>=0.05
    top_er_per_comp = enr.groupby("component")["ER"].max()
    comps_strong = int((top_er_per_comp >= max(1.5, perm95)).sum())
    f1_lift = (f1.get("full", float("nan")) - f1.get("layer_pos", float("nan")))
    gate2 = (comps_strong >= 2) or (np.isfinite(f1_lift) and f1_lift >= 0.05)

    # write report
    report = []
    report.append(f"# PAT-200 — QWAB Scale-Selection Query-NMF Report\n")
    report.append(f"- Table: `{table}`  | rows={len(df)}  | K={K} scales")
    report.append(f"- Train/heldout split: {len(cases)-n_ho}/{n_ho} cases "
                  f"({len(Xtr)}/{len(Xho)} rows), split by case.")
    report.append(f"- NMF seeds={args.n_seeds}, enrichment top-frac={args.top_frac}, perm n={args.n_perm}\n")
    report.append("## R sweep\n")
    report.append(sweep_df.to_markdown(index=False))
    report.append(f"\n**Chosen R\\* = {R_star}** (best stability among R∈{{5,6}}).\n")
    report.append("## Component profiles (R\\*)\n")
    comp_tab = pd.DataFrame([{"component": r, **component_profile(V0[r], K)} for r in range(R_star)])
    report.append(comp_tab[["component", "dominant_scale", "fine_mass", "mid_mass",
                            "coarse_mass", "entropy"]].to_markdown(index=False))
    report.append("\n> Band grouping fine/mid/coarse = s1..s%d split in thirds (near→far). "
                  "Verify against scale grid before strong claims.\n" % K)
    report.append("## Top enrichment (R\\*, train)\n")
    top_enr = (enr.sort_values("ER", ascending=False)
               .dropna(subset=["ER"]).head(20))
    report.append(top_enr.to_markdown(index=False))
    report.append(f"\nPermutation null max-ER: p95={perm95:.3f}, mean={perm_mean:.3f}.")
    report.append(f"Components with max-ER ≥ max(1.5, p95): **{comps_strong}**.\n")
    report.append("## Held-out enrichment stability (R\\*)\n")
    ho_top = (enr_ho.sort_values("ER", ascending=False).dropna(subset=["ER"]).head(10))
    report.append(ho_top.to_markdown(index=False))
    report.append("\n## Predictive baselines (macro-F1, held-out)\n")
    report.append(f"- layer_only: {f1.get('layer_only'):.3f}")
    report.append(f"- layer_pos:  {f1.get('layer_pos'):.3f}")
    report.append(f"- full:       {f1.get('full'):.3f}")
    report.append(f"- **lift (full − layer_pos): {f1_lift:.3f}**\n")
    report.append("## Decision gates\n")
    report.append(f"- **Gate 1 — stability** (frac matched cosine ≥0.85 = {g1_frac:.2f}): "
                  f"{'PASS' if gate1 else 'FAIL'}")
    report.append(f"- **Gate 2 — enrichment** (strong comps={comps_strong} OR F1 lift={f1_lift:.3f}): "
                  f"{'PASS' if gate2 else 'FAIL'}")
    report.append(f"- Gate 3 — attention-effect separation: deferred to Phase B.\n")
    if gate1 and gate2:
        verdict = ("PROCEED to Phase B. Components are stable and query-enriched → "
                   "evidence for query-conditioned scale routing (pending attention-effect check).")
    elif gate1 and not gate2:
        verdict = ("STOP / reframe as **layer-level scale specialization**: components stable "
                   "but enrichment is weak / layer-driven.")
    else:
        verdict = ("STOP — keep as a **negative diagnostic**: components unstable or enrichment "
                   "does not survive controls.")
    report.append(f"### Verdict\n{verdict}\n")

    rep_path = out_dir / "scale_query_nmf_report.md"
    rep_path.write_text("\n".join(report))
    summary = {"R_star": R_star, "gate1": gate1, "gate2": gate2,
               "stability_frac_ge_085": g1_frac, "comps_strong": comps_strong,
               "f1_lift": f1_lift if np.isfinite(f1_lift) else None,
               "perm95_max_er": perm95, "verdict": verdict}
    (out_dir / "gate_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"Done. report -> {rep_path}", flush=True)
    print(json.dumps(summary, indent=2), flush=True)


def main():
    ap = argparse.ArgumentParser(description="PAT-200 scale-query NMF analysis")
    ap.add_argument("--table", required=True, help="router_scale_query_table.parquet")
    ap.add_argument("--out_dir", default=None)
    ap.add_argument("--n_seeds", type=int, default=5)
    ap.add_argument("--n_perm", type=int, default=200)
    ap.add_argument("--top_frac", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    run(args)


if __name__ == "__main__":
    main()
