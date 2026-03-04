# Gate4 Conclusion

- Decision Gate: **A** (任务差异清晰，可用于功能性解释)

## Conclusions
- Layer10 上 XSUM 的 `LARGE` 占比高于 HotpotQA（约 +0.10），而 HotpotQA 的 `SMALL` 占比更高（约 +0.08），说明两任务在 scale 偏好上确实不同。
- `SMALL` 状态的 flip-rate 任务差异明显：HotpotQA 相对 XSUM 更低（L10/L11 均为负 gap），并伴随 prefix_shift gap，提示任务需求对机制路径有方向性约束。
- JS 贡献在 L11 的 state 组成上也存在分层差异（见 `js_contrib_gap` 图），支持“任务条件下机制触发比例不同”的解释。

## Counterexample
- L11 的 `UNIFORM` 比例在 HotpotQA 与 XSUM 几乎持平（gap 很小），说明并非所有 state 都有强任务分化；任务效应更集中在特定 layer/state 组合。

## Artifacts
- `/project/nlp-work5/hongyu-s/analysis_outputs/PAT-8/gate4_task_split_state_stats.json`
- `/project/nlp-work5/hongyu-s/analysis_outputs/PAT-8/gate4_task_split_chain_plots`
