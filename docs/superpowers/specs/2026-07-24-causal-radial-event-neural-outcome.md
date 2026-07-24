# Causal R-only Single-seed Event Neural Outcome

Date: 2026-07-24

## Result

The formal `causal_radial_event_neural_v1` run satisfies the new online and
single-seed constraints:

- selected seed: `17`;
- seed ensemble: disabled;
- input component: radial `R` only;
- final horizon: `200 s`;
- external coverage: `8/8` events;
- external Event MAE: `0.101626992 Mw`;
- external Event RMSE: `0.131581118 Mw`;
- external bias: `+0.032241225 Mw`;
- events with absolute error at most `0.15 Mw`: `5/8`.

The frozen run is:

`/home/lihe/PINN_Mag/runs/phase17-causal-radial-online-20260724T072107Z-a337bd4`

## Causal input contract

The model uses the same USGS-priority 31-event source, but removes the old
full-window `R > 2 cm` admission filter. This retains 4,165 valid stations and
prevents the station pool from revealing which stations will eventually have
large motion.

At second `t`, the model uses only samples available through `t`. A conservative
six-second delay covers the maximum interpolation look-ahead and the half-width
of the seven-tap FIR filter. Station peaks are running prefix peaks, and the
top-five station set is recomputed at every second. The external audit observed
5-26 distinct top-five sets per event, so the final stations were not fixed in
advance.

The network has a trainable final-horizon amplitude anchor and a two-layer GELU
prefix residual. The residual is multiplied by `(1 - t / 200)`, making its
contribution exactly zero at 200 seconds. Prefix training can improve the
evolving estimate without changing the final anchor.

## Seed selection

All three seeds were trained before any external waveform was loaded. The
predeclared internal event-equal online validation MAE selected one seed:

| Seed | Validation online MAE | Validation final MAE | Selected |
|---:|---:|---:|:---:|
| 17 | **0.279643** | 0.216865 | yes |
| 42 | 0.318414 | 0.266477 | no |
| 73 | 0.303651 | 0.217152 | no |

Only seed 17 was reloaded for external evaluation. No mean, median, weighted
average, or per-event seed choice is present in the result.

## Time evolution

| Global horizon | Event coverage | Event MAE |
|---:|---:|---:|
| 30 s | 6/8 | 0.559939 |
| 60 s | 8/8 | 0.396096 |
| 90 s | 8/8 | 0.259267 |
| 120 s | 8/8 | 0.224115 |
| 150 s | 8/8 | 0.191596 |
| 180 s | 8/8 | 0.120932 |
| 200 s | 8/8 | **0.101627** |

All eight events have predictions from 38 seconds onward. Event MAE first
reaches `<=0.15 Mw` at 166 seconds and remains below that threshold through
200 seconds. Each event prediction changed at every reported second after its
first physically available station.

## Final event results

| Event | Reference Mw | Predicted Mw | Absolute error | Available/used stations |
|---|---:|---:|---:|---:|
| Iquique | 7.7 | 7.705008 | 0.005008 | 11/5 |
| Kodiak | 7.9 | 7.840743 | 0.059257 | 64/5 |
| Luding | 6.6 | 6.813944 | 0.213944 | 6/5 |
| Mandalay | 7.7 | 7.790692 | 0.090692 | 13/5 |
| Nepal | 7.3 | 7.107080 | 0.192920 | 5/5 |
| Samos | 7.0 | 6.974634 | 0.025366 | 3/3 |
| Sand Point | 7.3 | 7.319067 | 0.019067 | 45/5 |
| Xizang | 7.1 | 7.306761 | 0.206761 | 12/5 |

## Verification and interpretation

- implementation commit: `a337bd4`;
- clean detached formal worktree;
- three seeds each completed 3,000 anchor and 1,500 prefix epochs;
- all three final prefix residuals were exactly zero;
- all registered artifact SHA-256 values matched;
- three checkpoints strictly reloaded on CUDA;
- 1,476 selected-seed external predictions reproduced with maximum delta `0`;
- focused regression: `50 passed`;
- training process exited and the GPU returned to idle.

This is a causal development result, not an unbiased paper test. The eight
external events have already influenced earlier feature development. A new,
untouched event set is still required for a final generalization claim.
