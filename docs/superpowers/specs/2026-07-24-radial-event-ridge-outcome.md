# R-only Radial Event Ridge Outcome

## Objective

Reduce the fixed eight-event magnitude MAE to at most `0.15 Mw` while keeping
the waveform input radial-only and retaining all eight events when possible.

## Method

The magnitude estimator is an event-level head fitted on the 31 accepted
USGS-priority training events and 2483 stations. For each event it:

1. Selects the five stations with the largest radial peak in the existing
   processed waveform window.
2. Computes median and 90th-percentile corrected amplitudes for distance
   exponents `0.5`, `0.75`, `1.0`, and `1.25`.
3. Adds median log-distance and selected-station count.
4. Standardizes the ten event features and fits ridge regression with
   `alpha=10`.

The estimator does not use tangential or vertical waveforms and does not alter
the PINN STF branch. It is deterministic, so it does not require a seed
ensemble. The implementation is commit `a629228`.

## Fixed Eight-event Result

| Method | Events | Event MAE | Event RMSE | Bias |
|---|---:|---:|---:|---:|
| Radial event ridge | 8 | **0.116119** | 0.155502 | +0.044608 |
| Phase-9 R-only ensemble | 8 | 0.206447 | 0.259018 | -0.001114 |
| Phase-13 event-balanced ensemble | 8 | 0.212720 | 0.251184 | -0.018905 |
| PGD-Crowell | 8 | 0.329327 | 0.406931 | +0.207235 |
| PGD-Ruhl | 8 | 0.334524 | 0.366737 | -0.281886 |
| PGD-Melgar | 8 | 0.199427 | 0.227518 | -0.103443 |

The radial event head improves MAE by `0.090328 Mw` relative to the formal
phase-9 R-only ensemble and by `0.083307 Mw` relative to PGD-Melgar. It passes
the requested `<=0.15 Mw` target while retaining all eight events.

The largest remaining absolute errors are Luding (`0.260655`), Xizang
(`0.257404`), and Nepal (`0.202823`). The other five events have absolute
errors no larger than `0.124852 Mw`.

## High-signal Applicability Check

Filtering the existing phase-9 R-only station predictions at `R > 2 cm`
retains 7/8 events and gives ensemble Event MAE `0.116553`. This supports a
high-signal applicability domain, but the result combines station selection
with exclusion of Luding and must always be reported with `7/8` coverage. The
new event head is preferable because it reaches comparable error with 8/8
coverage.

## Audit

- Refit from the frozen config and USGS-priority NPZ reproduced feature means,
  scales, and coefficients exactly.
- Recomputing from all 159 external station rows reproduced all eight event
  predictions exactly.
- Model, internal LOEO, and external prediction artifact hashes match the
  recorded summary.
- Internal 31-event LOEO Event MAE is `0.172906`.
- The detached run worktree is clean; the fitting process exited and the GPU is
  idle.

## Publication Boundary

The fixed eight events were used to compare feature structures and ridge
strengths in Phase 10. They are therefore a development validation set, not an
unbiased final test set. A paper may report this result as validation evidence,
but a new set of events that has not influenced model selection is required for
the final generalization claim.

Formal artifacts are under
`runs/phase14-radial-event-ridge-20260723T170354Z-a629228/`. The `publication/`
subdirectory contains the joined event table, method metrics, and PNG/PDF
comparison figure.
