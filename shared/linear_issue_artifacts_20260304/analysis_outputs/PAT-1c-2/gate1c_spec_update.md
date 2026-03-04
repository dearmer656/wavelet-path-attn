# Gate1c Spec Update

## Scope
- Model regime: wavelet delta logits with strong very-low dominance.
- Layers: L10/L11, state unit = (layer, query).

## Preconditions
- argmax_only min/max samples must be non-empty for L10/L11 (`n_min>0` and `n_max>0`).

## PASS Criteria
1. Frequency branch (A)
- `centroid_small-large > 0` and bootstrap 95% CI excludes 0 in L10 and L11.
2. Shape branch (B)
- `profile_L2(small,large) >= 0.120` and CI low bound >= 0.120 in L10 and L11.
- Directionality must be consistent across layers or explainable by task split.

## FAIL Criteria
- Neither A nor B passes; or layer directions are systematically opposite without valid stratified explanation.

## Reporting Requirements
- Always report: centroid diff + CI, profile_L2 + CI, and task-split sign table.
- Explicitly document whether decision came from A or B branch.
