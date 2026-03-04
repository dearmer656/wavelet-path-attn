# Gate1c Spec Update (Hotfix)

## Scope
- Regime: wavelet ΔE under strong DC/very-low dominance.
- Layers: L10/L11; analysis unit is (layer, query).

## Preconditions
- argmax-only cohorts must be non-empty (`n_min>0` and `n_max>0`) in L10/L11.

## PASS Criteria
1. Criterion A (frequency, optional)
- `centroid_small-large > 0` and bootstrap CI95 excludes 0 in both L10/L11.
- A may fail without blocking PASS if Criterion B passes.
2. Criterion B (shape, primary)
- `profile_rel_l2_small_vs_large > 0.120` and CI95 lower bound > 0.120 in both L10/L11.
- No centroid-sign or frequency-direction consistency requirement is allowed in Criterion B.

## Directionality (Descriptive Only)
- If needed, use spatial-only metrics such as PrefixBias / NearFar.
- Directionality is not a blocking condition when Criterion B is satisfied.

## FAIL Criteria
- Fail only when neither Criterion A nor Criterion B passes.

## Reporting Requirements
- Report L10/L11: profile_L2, CI95, threshold, pass/fail; and final branch.
- Frequency centroid signs may be reported as diagnostics, not as B-gating.
