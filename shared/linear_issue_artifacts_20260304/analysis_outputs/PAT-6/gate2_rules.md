# Gate2 Rules (Router Choice)

## Summary
- Decision Gate: **A** (可解释，进入Issue 3)
- Macro-F1: prior=0.1872, logistic=0.3786, tree=0.3933, best=0.3933

## Top Predictors (Tree Importance)
- `self_mass_base`: 0.8193
- `fft_low`: 0.0746
- `prefix_mass_base`: 0.0498
- `entropy_base`: 0.0287
- `fft_centroid`: 0.0149

## Logistic Cues
- `LARGE`: self_mass_base:-0.4002, entropy_base:-0.2168, q_pos_ratio:+0.2030, prefix_mass_base:+0.1125, fft_centroid:+0.0552, margin_base:-0.0510
- `SMALL`: fft_centroid:-0.2048, fft_high:+0.1169, self_mass_base:+0.1038, margin_base:-0.0647, fft_mid:-0.0640, fft_low:-0.0539
- `UNIFORM`: self_mass_base:+0.2964, entropy_base:+0.2015, q_pos_ratio:-0.1838, fft_centroid:+0.1496, margin_base:+0.1157, fft_high:-0.0960

## Shallow Tree Rules
```text
|--- self_mass_base <= 0.0337
|   |--- self_mass_base <= 0.0072
|   |   |--- margin_base <= 0.2486
|   |   |   |--- entropy_base <= 3.5204
|   |   |   |   |--- class: LARGE
|   |   |   |--- entropy_base >  3.5204
|   |   |   |   |--- class: LARGE
|   |   |--- margin_base >  0.2486
|   |   |   |--- class: LARGE
|   |--- self_mass_base >  0.0072
|   |   |--- fft_low <= 0.3892
|   |   |   |--- self_mass_base <= 0.0131
|   |   |   |   |--- class: LARGE
|   |   |   |--- self_mass_base >  0.0131
|   |   |   |   |--- class: LARGE
|   |   |--- fft_low >  0.3892
|   |   |   |--- fft_centroid <= 0.1498
|   |   |   |   |--- class: SMALL
|   |   |   |--- fft_centroid >  0.1498
|   |   |   |   |--- class: LARGE
|--- self_mass_base >  0.0337
|   |--- self_mass_base <= 0.1047
|   |   |--- prefix_mass_base <= 0.2470
|   |   |   |--- fft_low <= 0.5514
|   |   |   |   |--- class: SMALL
|   |   |   |--- fft_low >  0.5514
|   |   |   |   |--- class: SMALL
|   |   |--- prefix_mass_base >  0.2470
|   |   |   |--- class: LARGE
|   |--- self_mass_base >  0.1047
|   |   |--- self_mass_base <= 0.1602
|   |   |   |--- entropy_base <= 2.6105
|   |   |   |   |--- class: UNIFORM
|   |   |   |--- entropy_base >  2.6105
|   |   |   |   |--- class: SMALL
|   |   |--- self_mass_base >  0.1602
|   |   |   |--- prefix_mass_base <= 0.1176
|   |   |   |   |--- class: UNIFORM
|   |   |   |--- prefix_mass_base >  0.1176
|   |   |   |   |--- class: UNIFORM

```
