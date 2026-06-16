# PAT-200 — QWAB Scale-Selection Query-NMF Report

- Table: `analysis_outputs/qwab_scale_query_nmf/full/router_scale_query_table.parquet`  | rows=691200  | K=8 scales
- Train/heldout split: 120/30 cases (552960/138240 rows), split by case.
- NMF seeds=5, enrichment top-frac=0.05, perm n=200

## R sweep

|   R |   train_rel_recon_err |   heldout_rel_recon_err |   stability_mean_cosine |   stability_frac_ge_0.85 |
|----:|----------------------:|------------------------:|------------------------:|-------------------------:|
|   2 |            0.301219   |               0.301188  |                       1 |                        1 |
|   3 |            0.25974    |               0.259642  |                       1 |                        1 |
|   4 |            0.220806   |               0.220731  |                       1 |                        1 |
|   5 |            0.183159   |               0.182974  |                       1 |                        1 |
|   6 |            0.142631   |               0.142543  |                       1 |                        1 |
|   7 |            0.0987891  |               0.099097  |                       1 |                        1 |
|   8 |            0.00141484 |               0.0013929 |                       1 |                        1 |

**Chosen R\* = 6** (best stability among R∈{5,6}).

## Component profiles (R\*)

|   component |   dominant_scale |   fine_mass |   mid_mass |   coarse_mass |   entropy |
|------------:|-----------------:|------------:|-----------:|--------------:|----------:|
|           0 |                5 |    0.15707  | 0.15386    |     0.689071  |  1.45607  |
|           1 |                8 |    0.324353 | 0.00490771 |     0.67074   |  1.18131  |
|           2 |                3 |    0        | 0.907469   |     0.0925305 |  0.370153 |
|           3 |                2 |    0.919436 | 0          |     0.0805645 |  0.890789 |
|           4 |                4 |    0.109806 | 0.663671   |     0.226523  |  0.965464 |
|           5 |                6 |    0        | 0          |     1         |  1.00514  |

> Band grouping fine/mid/coarse = s1..s8 split in thirds (near→far). Verify against scale grid before strong claims.

## Top enrichment (R\*, train)

|   component | attribute               |    P_attr |   P_attr_given_comp |      ER |
|------------:|:------------------------|----------:|--------------------:|--------:|
|           1 | layer_low               | 0.333333  |           0.966942  | 2.90082 |
|           3 | layer_mid               | 0.333333  |           0.733037  | 2.19911 |
|           5 | layer_low               | 0.333333  |           0.705367  | 2.1161  |
|           2 | top_scale_margin_high   | 0.5       |           0.961625  | 1.92325 |
|           1 | base_logit_margin_high  | 0.5       |           0.900318  | 1.80064 |
|           2 | layer_mid               | 0.333333  |           0.595812  | 1.78743 |
|           0 | top_scale_margin_high   | 0.5       |           0.86867   | 1.73734 |
|           2 | router_in_norm_high     | 0.5       |           0.842701  | 1.6854  |
|           2 | role_question           | 0.0173611 |           0.0286097 | 1.64792 |
|           0 | layer_high              | 0.333333  |           0.536965  | 1.61089 |
|           2 | WavMass_high            | 0.5       |           0.802083  | 1.60417 |
|           2 | router_out_entropy_high | 0.5       |           0.798249  | 1.5965  |
|           4 | layer_mid               | 0.333333  |           0.522895  | 1.56868 |
|           5 | role_title_special      | 0.0356771 |           0.0529152 | 1.48317 |
|           5 | role_question           | 0.0173611 |           0.0252098 | 1.45208 |
|           4 | router_in_norm_high     | 0.5       |           0.696578  | 1.39316 |
|           4 | WavMass_high            | 0.5       |           0.680013  | 1.36003 |
|           4 | router_out_entropy_high | 0.5       |           0.672996  | 1.34599 |
|           4 | prefix_mass_base_high   | 0.5       |           0.666956  | 1.33391 |
|           2 | prefix_mass_base_high   | 0.5       |           0.655237  | 1.31047 |

Permutation null max-ER: p95=1.100, mean=1.043.
Components with max-ER ≥ max(1.5, p95): **6**.

## Held-out enrichment stability (R\*)

|   component | attribute              |    P_attr |   P_attr_given_comp |      ER |
|------------:|:-----------------------|----------:|--------------------:|--------:|
|           1 | layer_low              | 0.333333  |           0.969184  | 2.90755 |
|           3 | layer_mid              | 0.333333  |           0.736979  | 2.21094 |
|           5 | layer_low              | 0.333333  |           0.692419  | 2.07726 |
|           2 | top_scale_margin_high  | 0.5       |           0.966435  | 1.93287 |
|           1 | base_logit_margin_high | 0.5       |           0.896701  | 1.7934  |
|           2 | layer_mid              | 0.333333  |           0.588976  | 1.76693 |
|           0 | top_scale_margin_high  | 0.5       |           0.872106  | 1.74421 |
|           2 | router_in_norm_high    | 0.5       |           0.845631  | 1.69126 |
|           2 | role_question          | 0.0164931 |           0.0273438 | 1.65789 |
|           5 | role_title_special     | 0.0384549 |           0.062066  | 1.614   |

## Predictive baselines (macro-F1, held-out)

- layer_only: 0.225
- layer_pos:  0.223
- full:       0.358
- **lift (full − layer_pos): 0.136**

## Decision gates

- **Gate 1 — stability** (frac matched cosine ≥0.85 = 1.00): PASS
- **Gate 2 — enrichment** (strong comps=6 OR F1 lift=0.136): PASS
- Gate 3 — attention-effect separation: deferred to Phase B.

### Verdict
PROCEED to Phase B. Components are stable and query-enriched → evidence for query-conditioned scale routing (pending attention-effect check).

## Analyst caveats (read before strong claims)

1. **Stability is a weak test here.** Matched-cosine = 1.00 at every R because the
   8-dim NMF uses NNDSVD init, which is near-deterministic regardless of seed. It
   confirms the solution is unique/reproducible but does **not** strongly validate
   "real structure". Treat Gate 1 as passed-but-uninformative.
2. **R selection is a near-tie.** Train/heldout recon error decays smoothly (R=8 ≈ 0
   is dimension-filling for 8-dim simplex data), and stability does not discriminate.
   R*=6 is a tie-break pick; R=5 is equally defensible. Do not over-read the exact R.
3. **Layer is the dominant factor.** The strongest enrichments are layer bins
   (comp1→layer_low ER 2.90, comp3→layer_mid 2.20, comp5→layer_low 2.12,
   comp0→layer_high 1.61). This matches the issue's "layer-level scale specialization"
   caution.
4. **Query-conditioning is real but secondary.** It survives the strongest control:
   held-out macro-F1 rises 0.223 (layer+pos) → 0.358 (full), **+0.136**, well above the
   +0.05 gate; held-out enrichment reproduces train ER. Genuine (non-circular) query
   signals: role_question (comp2/5), role_title_special (comp5), base_logit_margin_high
   (comp1), prefix_mass_base_high (comp2/4), router_in_norm_high.
5. **Some Gate-2 enrichments are partly circular.** top_scale_margin / WavMass /
   router_out_entropy buckets are derived from the same pi NMF decomposed, so their
   ER partly reflects the decomposition, not independent query state. The role /
   base-attention enrichments and the predictive F1 lift are the load-bearing evidence.

**Framing:** Stronger than "uniform wavelet bias" and stronger than pure layer
specialization (query state adds held-out signal beyond layer+position), but the
mechanism claim ("modes induce different attention effects") still requires Phase B.

## Phase B — attention-effect linkage (QWAB on vs off, R*=6)

|   component |   n_rows |   FlipRate@32 |   BiasMassGain@32 |     dEntropy |      dESS |   dEvidenceMass |
|------------:|---------:|--------------:|------------------:|-------------:|----------:|----------------:|
|           0 |   192916 |     0.0241404 |       0.00124947  | -0.00466554  | -2.28074  |    -0.000942401 |
|           1 |   168867 |     0.0102016 |      -0.000580543 |  0.00376948  |  0.398816 |    -0.000402293 |
|           2 |   149063 |     0.0223699 |       0.00116377  | -0.00330606  | -2.61185  |    -0.000974238 |
|           3 |   105362 |     0.0197005 |       0.000341323 |  0.000375608 | -1.7232   |    -0.000746456 |
|           4 |    34781 |     0.0295743 |       0.00199319  | -0.00621804  | -3.95528  |    -0.00133187  |
|           5 |    40211 |     0.0236433 |       0.000948767 | -0.00252849  | -1.60445  |    -0.00109385  |

(Full per-metric means + case-bootstrap 95% CIs in `component_attention_effects_R6.csv`.)

**Gate 3 — PASS.** Metrics with disjoint extreme-component 95% CIs (effect differs beyond case-bootstrap noise):
- `FlipRate@16`: comp1=0.0101 vs comp4=0.0317
- `FlipRate@32`: comp1=0.0102 vs comp4=0.0296
- `BiasMassGain@16`: comp1=-0.0006 vs comp4=0.0015
- `BiasMassGain@32`: comp1=-0.0006 vs comp4=0.0020
- `dTopKMass@16`: comp1=-0.0007 vs comp4=0.0013
- `dTopKMass@32`: comp1=-0.0006 vs comp4=0.0018
- `dEntropy`: comp4=-0.0062 vs comp1=0.0038
- `dESS`: comp4=-3.9553 vs comp1=0.3988
- `dSelfMass`: comp5=-0.0006 vs comp1=-0.0002
- `dEvidenceMass`: comp4=-0.0013 vs comp1=-0.0004

### Final framing
All three gates pass: QWAB does **not** apply a uniform wavelet bias. Its head-shared router decomposes into stable scale-selection modes that are query-enriched (beyond layer+position) and induce **distinguishable attention-support changes** — supporting the preferred mechanism claim.
### Phase B caveats (scope of the claim)
1. **Effects are small in absolute magnitude** (FlipRate@32 ≈ 1–3%, |ΔEntropy| ≲ 0.006,
   |ΔESS| ≲ 4). QWAB's wavelet bias is a *gentle* perturbation; the modes differ
   reliably (disjoint case-bootstrap CIs) but the per-query reshaping is modest.
2. **Direction of the contrast is interpretable:** comp1 (layer_low / coarse-s8) is the
   only mode that *diffuses* attention (+ΔEntropy, +ΔESS, negative BiasMassGain), while
   comp0/2/4 *concentrate* it (−ΔEntropy, −ΔESS, +BiasMassGain), comp4 most strongly.
3. **`ΔEvidenceMass` is uniformly negative across all modes** (−0.0004 … −0.0013): the
   wavelet bias does **not** increase attention mass on supporting-fact tokens for any
   mode — it modestly *reduces* it (least for the diffusing comp1, most for comp4).
   → The **stronger "PaTH makes scale-selection task-aligned"** claim is **NOT supported**
   here; modes differ in attention *shape* (concentration vs diffusion), not in
   evidence-grounding. The supported claim is the preferred (non-task-aligned) one.

**Bottom line:** Gates 1–3 pass → QWAB's head-shared router is a stable, query-conditioned
(beyond layer+position) scale selector whose modes induce distinguishable attention-shape
changes. It is *not* a uniform bias and *not* (on this evidence) a task-aligned evidence booster.
