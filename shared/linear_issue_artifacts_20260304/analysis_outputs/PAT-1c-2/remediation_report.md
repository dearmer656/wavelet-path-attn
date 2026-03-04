# PAT-1c-2 Remediation Report

## Decision
- Branch: **B**
- Reason: A/B 均未满足：L10/L11 方向相反且无法由 task split 解释

## Why old Gate1 high-band DoD is not suitable
- 当前 ΔE 频谱由 DC/very-low 主导，高频能量占比很小，`high_band_small > high_band_large` 在该 regime 下不稳定且易受对齐与归一化影响。
- 因此改为 Gate1c：优先考察 low-band 内部重分配（centroid、very_low、low_mid）与空间域形状分离（profile L2）。

## New Gate1c Rule (applied)
- A: `centroid_small-large > 0` 且 `95% CI` 不跨 0（L10/L11）。
- B: `profile_L2 >= 0.120` 且 `CI_low >= 0.120`（L10/L11），并满足方向一致或可由 task split 解释。

## Key Numbers
- L10: centroid_diff=0.003889, CI95=[0.002976, 0.004723], profile_L2=0.156945, CI95=[0.147381, 0.167359]
- L11: centroid_diff=-0.004381, CI95=[-0.005928, -0.002821], profile_L2=0.212561, CI95=[0.200217, 0.224856]

## Next Suggestions
- 减弱 key=0 DC 主导：引入可控 de-DC / key0 downweight。
- 调整对齐策略：由 key=0 固定对齐改为 query-relative 或 learned centering。
- 在不改变语义的前提下校准归一化，避免压平跨尺度差异。
