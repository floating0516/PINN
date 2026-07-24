# Causal Forward-Guided R-only Event Neural Outcome

Date: 2026-07-24

## Result

The formal `causal_forward_guided_event_neural_v2` run preserves the original
four-term loss system while satisfying the causal input and single-seed rules:

- selected seed: `73`;
- seed ensemble: disabled;
- input component: radial `R` only;
- external coverage: `8/8` events;
- external all-second Event MAE: `0.215284407 Mw`;
- external 200-second Event MAE: `0.131989515 Mw`;
- external 200-second Event RMSE: `0.167656470 Mw`;
- external 200-second bias: `+0.090313542 Mw`;
- events with final absolute error at most `0.15 Mw`: `6/8`;
- stable `<=0.15 Mw` external MAE from `89 s` through `200 s`.

The frozen main run is:

`/home/lihe/PINN_Mag/runs/phase19-forward-guided-gated-20260724T101745Z-c7c1736`

The matched no-forward-loss ablation is:

`/home/lihe/PINN_Mag/runs/phase20-forward-guided-no-synth-20260724T102555Z-de2149b`

The method is not called a PINN. It contains no PDE residual. The accurate
description is a causal, forward-guided, multi-task neural network.

## Model and causal input contract

The candidate pool contains all 4,165 valid USGS-priority R-only station
records. It does not use the final `R > 2 cm` filter, so admission does not
reveal which station will eventually have strong motion.

At global second `t`:

1. only the R waveform released through `t-6 s` is visible;
2. the six-second delay covers interpolation and the symmetric FIR half-window;
3. running radial peaks and the top-five stations are recomputed from that
   prefix;
4. causal left-padded TCN blocks encode each selected station;
5. a Transformer uses both an upper-triangular causal mask and an effective
   prefix mask;
6. station features are aggregated into one event representation;
7. the model predicts one shared, origin-aligned event STF and one evolving Mw.

The scalar Mw path combines a stable amplitude-distance anchor with a bounded
deep residual. The residual is multiplied by `(1-t/200)`, so it is active
during online estimation and exactly zero at 200 seconds. This protects the
final unseen-event estimate from the event-specific residual drift observed in
the ungated precursor run.

Focal-mechanism radiation metadata is used only by the training-time forward
loss. It is not an inference input. Consequently, Sand Point's missing
strike/dip/rake does not block prediction.

## Preserved four-term loss

The deep TCN/Transformer representation and shared STF are trained with the
original weights:

```text
L = 1.0 L_MSE + 0.5 L_synth + 1.0 L_mag + 0.1 L_shape
```

- `L_MSE` fits the log-encoded SCARDEC STF;
- `L_shape` fits its amplitude-normalized nonnegative shape;
- `L_mag` fits the USGS-priority catalog Mw;
- `L_synth` compares observed R displacement with displacement synthesized
  differentiably from the shared STF.

The new global clock uses absolute P and S travel times exactly once. The
far-field P/S coefficients retain the signed full-radiation terms and the
original `rho=3400 kg/m3`, `alpha=7900 m/s`, and `beta=4533 m/s` settings.
Intermediate-field terms remain disabled.

The forward equations are associated with Glehman et al. (2026), *Rapid
Earthquake Magnitude Estimation for Local Early Warning Systems Using
Seismogeodesy*, JGR Solid Earth, DOI `10.1029/2025JB033222`.

## Seed selection

All three seeds completed training before external waveforms were loaded. The
predeclared event-equal internal all-second validation MAE selected one seed:

| Seed | Validation online MAE | Validation final MAE | Selected |
|---:|---:|---:|:---:|
| 17 | 0.378165 | 0.221849 | no |
| 42 | 0.406408 | 0.270041 | no |
| 73 | **0.335906** | 0.217152 | yes |

Only seed 73 was reloaded for external evaluation. No seed average, per-event
seed choice, or external-driven seed selection is present.

## Online convergence

The principal neural benefit is earlier convergence. The prior Phase17 causal
amplitude model has a better 200-second anchor, but it converges much later.

| Horizon | Full four-term model | No `L_synth` | Phase17 data baseline |
|---:|---:|---:|---:|
| 60 s | **0.185796** | 0.187856 | 0.396096 |
| 90 s | **0.148414** | 0.150411 | 0.259267 |
| 120 s | **0.139531** | 0.142818 | 0.224115 |
| 150 s | **0.129987** | 0.134587 | 0.191596 |
| 180 s | 0.123281 | 0.125816 | **0.120932** |
| 200 s | 0.131990 | 0.131990 | **0.101627** |
| Stable `<=0.15` from | **89 s** | 91 s | 166 s |

All eight events are covered from 38 seconds. Across every available second,
the full model improves external MAE from Phase17's `0.265576` to `0.215284`.
The selected model's deep residual has mean absolute magnitude `0.069208 Mw`
over all external seconds and maximum absolute magnitude `0.168468 Mw`, so the
deep branch is active before the final horizon.

## Final event results

| Event | Stations available/used | Reference Mw | Predicted Mw | Absolute error |
|---|---:|---:|---:|---:|
| Iquique | 11/5 | 7.7 | 7.718792 | 0.018792 |
| Kodiak | 64/5 | 7.9 | 7.859811 | 0.040189 |
| Luding | 6/5 | 6.6 | 6.914756 | 0.314756 |
| Mandalay | 13/5 | 7.7 | 7.827084 | 0.127084 |
| Nepal | 5/5 | 7.3 | 7.173485 | 0.126515 |
| Samos | 3/3 | 7.0 | 7.089109 | 0.089109 |
| Sand Point | 45/5 | 7.3 | 7.355471 | 0.055471 |
| Xizang | 12/5 | 7.1 | 7.384000 | 0.284000 |

At 200 seconds the deep scalar residual is exactly zero for every event.
Therefore the final `0.131990 Mw` must be attributed to the stable anchor, not
to a direct final-horizon correction from the TCN, Transformer, or forward
loss. Their measurable role is in the earlier trajectory and shared STF.

## Forward-loss ablation

The no-forward run changes only `lambda_synth: 0.5 -> 0.0`. Architecture,
seeds, horizons, optimizer, gate, data, and selection policy are identical.

| Metric | Full model | No `L_synth` | Full-model improvement |
|---|---:|---:|---:|
| Mean validation online MAE over 3 seeds | 0.373493 | 0.374413 | 0.000920 |
| Selected-seed external online MAE | 0.215284 | 0.216126 | 0.000842 |
| Stable `<=0.15` horizon | 89 s | 91 s | 2 s earlier |
| External final MAE | 0.131990 | 0.131990 | 0.000000 |

At each seed's selected checkpoint, the full model also lowers validation
`L_synth` by `0.000561-0.000883`. The magnitude effect is small and not
uniform per seed: seeds 17 and 42 improve internally, while seed 73 is worse by
`0.000654 Mw`. The defensible conclusion is that the forward loss modestly
improves physical waveform consistency and provides a small online benefit. It
does not dominate final magnitude accuracy.

## Verification and interpretation boundary

- main implementation commits: `259d98a`, `c7c1736`;
- ablation configuration commit: `de2149b`;
- clean detached formal worktrees and unique immutable run directories;
- each seed completed 3,000 anchor plus 120 deep epochs;
- selected checkpoint SHA-256: `329ef625c1a822be...`;
- all registered main-run artifact hashes matched;
- selected checkpoint strictly reloaded on CUDA;
- all 1,475 external predictions, anchors, and residuals reproduced with
  maximum delta `0`;
- every selected checkpoint tensor was finite;
- focused affected regression: `40 passed` before formal training and
  `22 passed` after the gate/ablation additions;
- full repository regression: `390 passed, 1 skipped`, plus one known
  fixed-matrix failure caused by two pre-existing USGS experiment configs;
- all training/evaluation processes exited and the GPU returned to idle.

The eight external events are a development validation set. They influenced
the gate decision and earlier feature development, so neither `0.131990` nor
the 89-second convergence point is an unbiased paper test result. A new,
untouched event set is required for the final generalization claim.
