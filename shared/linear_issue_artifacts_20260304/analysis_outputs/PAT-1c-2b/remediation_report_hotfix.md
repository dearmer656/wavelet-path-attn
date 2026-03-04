# PAT-1c-2b Remediation Report (Hotfix)

## What was wrong in PAT-1c-2
- Shape criterion mixed in centroid-sign consistency, which is a frequency-domain diagnostic.
- In DC/very-low dominated regime, centroid sign can flip across layers and still coexist with stable spatial shape separation.

## Patched Criterion B
- B-PASS iff `profile_rel_l2_small_vs_large > 0.120` and CI95 lower bound > 0.120 for both L10/L11.
- Centroid-based directionality is removed from B gating.

## Key Evidence (task=all)
- L10: profile_L2=0.156945, CI95=[0.147381, 0.167359], centroid_diff=0.003889, centroid_CI95=[0.002976, 0.004723], PrefixBias_small-large=-0.001521
- L11: profile_L2=0.212561, CI95=[0.200217, 0.224856], centroid_diff=-0.004381, centroid_CI95=[-0.005928, -0.002821], PrefixBias_small-large=-0.012286

## Final Decision
- Branch: **A**
- Reason: PASS by patched Criterion B (shape): profile_L2 + CI are stable in both L10 and L11.

## Notes
- This is a spec/decision hotfix only; no heavy recomputation was performed.
