# R-only PINN Event Neural Outcome

Date: 2026-07-24

## Result

The formal `radial_pinn_event_neural_v2` run reached the fixed eight-event
development target:

- event coverage: 8/8
- event MAE: `0.107105609 Mw`
- event RMSE: `0.140061299 Mw`
- event bias: `+0.031422444 Mw`
- events with absolute error at most `0.15 Mw`: 5/8

The frozen run is:

`/home/lihe/PINN_Mag/runs/phase16-radial-pinn-event-neural-v2-20260724T042259Z-fd40706`

## Deep-learning path

The result does not call the phase14 ridge predictor. Three frozen phase9
R-only PINNs first produce station-level magnitude predictions. For each event,
the new model combines:

- ten top-five radial amplitude and source-distance features;
- eighteen distribution and disagreement features from PINN seeds 17, 42,
  and 73;
- a trainable amplitude branch and a two-layer GELU nonlinear residual branch.

Three event heads are trained with PyTorch Adam and backpropagation for 3000
epochs using seeds 17, 42, and 73. Their event predictions are averaged.

## Event results

| Event | Stations available/used | Reference Mw | Predicted Mw | Absolute error |
|---|---:|---:|---:|---:|
| Iquique | 11/5 | 7.7 | 7.692 | 0.008 |
| Kodiak | 64/5 | 7.9 | 7.833 | 0.067 |
| Luding | 6/5 | 6.6 | 6.838 | 0.238 |
| Mandalay | 13/5 | 7.7 | 7.792 | 0.092 |
| Nepal | 5/5 | 7.3 | 7.110 | 0.190 |
| Samos | 3/3 | 7.0 | 6.968 | 0.032 |
| Sand Point | 45/5 | 7.3 | 7.294 | 0.006 |
| Xizang | 12/5 | 7.1 | 7.324 | 0.224 |

## Interpretation boundary

This is technically a deep-learning system: it uses the three frozen PINNs and
three trainable nonlinear PyTorch event heads, and the serialized checkpoints
do not contain or invoke the ridge predictor. However, the learned nonlinear
PINN residual contributes only about `0.001 Mw` on the eight events. The
accuracy gain is therefore driven mainly by the new top-five radial
amplitude/distance input path, not by evidence that the PINN features add a
large correction.

The eight events were used as a development validation set in earlier phases.
The `0.107106 Mw` result must not be presented as an unbiased final test result.
A new event set that has never participated in model or feature selection is
required for the paper's final generalization claim.

## Verification

- three event-head checkpoints strictly reloaded on CUDA;
- all 24 head/event predictions exactly reproduced the frozen CSV files;
- three complete 3000-epoch finite training logs;
- all 22 registered artifact hashes matched;
- focused regression: `56 passed`;
- implementation commit: `fd40706`.
