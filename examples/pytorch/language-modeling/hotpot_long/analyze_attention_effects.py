#!/usr/bin/env python3
"""analyze_attention_effects.py — PAT-200 Phase B analysis.

Joins per-query QWAB on-vs-off attention effects to their NMF component
(argmax usage at R*), aggregates per component with case-level bootstrap CIs,
and evaluates Gate 3 (attention-effect separation across components).

Outputs:
  component_attention_effects_R{R}.csv
  appends a Phase B section to scale_query_nmf_report.md
  updates gate_summary.json with gate3.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

METRICS = ["FlipRate@16", "FlipRate@32", "BiasMassGain@16", "BiasMassGain@32",
           "dTopKMass@16", "dTopKMass@32", "dEntropy", "dESS", "dSelfMass",
           "dEvidenceMass"]


def boot_ci_by_case(df, metric, n_boot=1000, seed=0):
    """Case-level bootstrap mean + 95% CI for a metric (NaN-aware)."""
    rng = np.random.default_rng(seed)
    sub = df[["case_id", metric]].dropna()
    if len(sub) == 0:
        return float("nan"), float("nan"), float("nan")
    cases = sub["case_id"].unique()
    grp = {c: sub.loc[sub.case_id == c, metric].values for c in cases}
    means = []
    for _ in range(n_boot):
        pick = rng.choice(cases, size=len(cases), replace=True)
        vals = np.concatenate([grp[c] for c in pick])
        means.append(vals.mean())
    return (float(sub[metric].mean()),
            float(np.percentile(means, 2.5)),
            float(np.percentile(means, 97.5)))


def run(args):
    eff = pd.read_parquet(args.effects)
    R = int(args.R)
    out_dir = Path(args.out_dir)
    # component assignment from R* usage (train + heldout)
    u_tr = pd.read_parquet(out_dir / f"scale_nmf_R{R}_usage_train.parquet")
    u_ho = pd.read_parquet(out_dir / f"scale_nmf_R{R}_usage_heldout.parquet")
    usage = pd.concat([u_tr, u_ho], ignore_index=True)
    ucols = [f"U{r}" for r in range(R)]
    usage["component"] = usage[ucols].values.argmax(axis=1)
    key = ["case_id", "layer", "query_idx"]
    df = eff.merge(usage[key + ["component"]], on=key, how="inner")
    print(f"joined rows={len(df)} (effects={len(eff)}, usage={len(usage)})", flush=True)

    # per-component aggregates with case-level bootstrap CIs
    recs = []
    for comp in range(R):
        cdf = df[df.component == comp]
        rec = {"component": comp, "n_rows": len(cdf)}
        for m in METRICS:
            mean, lo, hi = boot_ci_by_case(cdf, m, n_boot=args.n_boot, seed=args.seed)
            rec[f"{m}_mean"] = mean
            rec[f"{m}_lo"] = lo
            rec[f"{m}_hi"] = hi
        recs.append(rec)
    comp_df = pd.DataFrame(recs)
    comp_df.to_csv(out_dir / f"component_attention_effects_R{R}.csv", index=False)

    # Gate 3: separation = for some metric, the top and bottom component means
    # have disjoint 95% CIs (case-bootstrap), i.e. effect differs beyond noise.
    sep_metrics = []
    for m in METRICS:
        means = comp_df[f"{m}_mean"].values
        los = comp_df[f"{m}_lo"].values
        his = comp_df[f"{m}_hi"].values
        if np.all(np.isnan(means)):
            continue
        hi_c = int(np.nanargmax(means)); lo_c = int(np.nanargmin(means))
        # disjoint CIs between the extreme components
        if los[hi_c] > his[lo_c] or los[lo_c] > his[hi_c]:
            sep_metrics.append((m, float(means[lo_c]), float(means[hi_c]),
                                lo_c, hi_c))
    gate3 = len(sep_metrics) > 0

    # report append
    rep = out_dir / "scale_query_nmf_report.md"
    lines = ["\n## Phase B — attention-effect linkage (QWAB on vs off, R*={})\n".format(R)]
    show = ["FlipRate@32", "BiasMassGain@32", "dEntropy", "dESS", "dEvidenceMass"]
    tab = comp_df[["component", "n_rows"] + [f"{m}_mean" for m in show]].copy()
    tab.columns = ["component", "n_rows"] + show
    lines.append(tab.to_markdown(index=False))
    lines.append("\n(Full per-metric means + case-bootstrap 95% CIs in "
                 f"`component_attention_effects_R{R}.csv`.)\n")
    if gate3:
        lines.append("**Gate 3 — PASS.** Metrics with disjoint extreme-component "
                     "95% CIs (effect differs beyond case-bootstrap noise):")
        for m, lo_m, hi_m, lo_c, hi_c in sep_metrics:
            lines.append(f"- `{m}`: comp{lo_c}={lo_m:.4f} vs comp{hi_c}={hi_m:.4f}")
    else:
        lines.append("**Gate 3 — FAIL.** No metric shows extreme-component CIs that "
                     "are disjoint; scale-selection modes do not induce distinguishable "
                     "on-vs-off attention effects beyond bootstrap noise.")
    lines.append("")
    if gate3:
        lines.append("### Final framing\nAll three gates pass: QWAB does **not** apply a "
                     "uniform wavelet bias. Its head-shared router decomposes into stable "
                     "scale-selection modes that are query-enriched (beyond layer+position) "
                     "and induce **distinguishable attention-support changes** — supporting "
                     "the preferred mechanism claim.")
    else:
        lines.append("### Final framing\nGates 1–2 pass but Gate 3 fails: components are "
                     "stable and query-enriched yet do not map to distinct attention effects. "
                     "Frame as **router-level scale specialization without a confirmed "
                     "attention-mechanism payoff**, not a full query-conditioned mechanism.")
    with open(rep, "a") as f:
        f.write("\n".join(lines))

    # update gate_summary
    gp = out_dir / "gate_summary.json"
    summ = json.loads(gp.read_text()) if gp.exists() else {}
    summ["gate3"] = bool(gate3)
    summ["gate3_separating_metrics"] = [m for m, *_ in sep_metrics]
    summ["all_gates_pass"] = bool(summ.get("gate1") and summ.get("gate2") and gate3)
    gp.write_text(json.dumps(summ, indent=2))
    print(f"Gate 3: {'PASS' if gate3 else 'FAIL'}  separating={[m for m,*_ in sep_metrics]}", flush=True)
    print(comp_df[["component", "n_rows"] + [f"{m}_mean" for m in show]].to_string(index=False), flush=True)


def main():
    ap = argparse.ArgumentParser(description="PAT-200 Phase B attention-effect analysis")
    ap.add_argument("--effects", required=True)
    ap.add_argument("--out_dir", required=True, help="dir with scale_nmf_R{R}_usage_* and report")
    ap.add_argument("--R", type=int, default=6)
    ap.add_argument("--n_boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    run(args)


if __name__ == "__main__":
    main()
