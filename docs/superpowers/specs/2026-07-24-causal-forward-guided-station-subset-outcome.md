# Causal Forward-Guided Station-Subset Outcome

Date: 2026-07-24

## Result

The formal `causal_forward_guided_event_neural_v3` run addresses the high
internal station-random test error without changing the waveform component,
model backbone, causal input protocol, or four-term objective.

- selected seed: `73`;
- seed ensemble: disabled;
- input component: radial `R` only;
- internal test all-second Event MAE: `0.327773601 Mw`;
- internal test 200-second Event MAE: `0.222771251 Mw`;
- improvement over the matched Phase19 selected seed: `0.016831805 Mw` online
  and `0.017226859 Mw` at 200 seconds;
- external development coverage: `8/8` events;
- external all-second Event MAE: `0.220198520 Mw`;
- external 200-second Event MAE: `0.123864031 Mw`.

The frozen formal run is:

`/home/lihe/PINN_Mag/runs/phase22-forward-guided-station-subset-20260724T113401Z-8884db2`

The matched baseline is:

`/home/lihe/PINN_Mag/runs/phase19-forward-guided-gated-20260724T101745Z-c7c1736`

This remains a causal, forward-guided, multi-task neural network. It is not
called a PINN because it contains no PDE residual.

## Single scientific change

Phase19 exposed each training event and horizon to one deterministic top-five
station snapshot. Repeating epochs only shuffled those same snapshots. Phase22
keeps that canonical station pool and adds three deterministic random station
subsets per training event. Each subset retains approximately 25% of the
event's training stations.

The subset membership is fixed across all horizons for that variant. At each
second, the model still recomputes running radial peaks and dynamic top-five
stations from the waveform prefix available at that second. It never uses the
final peak or a future waveform sample to decide which stations matter.

The three formal seeds produced the following augmentation coverage:

| Seed | Station-pool variants | Unique variants | Training snapshots |
|---:|---:|---:|---:|
| 17 | 124 | 114 | 1,576 |
| 42 | 124 | 117 | 1,562 |
| 73 | 124 | 115 | 1,559 |

Validation and test data are not augmented. Their station assignments and
assignment SHA-256 values are identical to Phase19.

## Preserved model and loss

The model continues to use:

1. a conservative six-second causal release delay;
2. prefix-only running radial amplitudes and dynamic top-five selection;
3. causal left-padded TCN blocks;
4. a Transformer with causal and effective-prefix masks;
5. multi-station event aggregation;
6. one origin-aligned shared event STF;
7. absolute P/S travel times and signed radiation coefficients in the forward
   waveform loss.

The original loss weights are unchanged:

```text
L = 1.0 L_MSE + 0.5 L_synth + 1.0 L_mag + 0.1 L_shape
```

The forward equations remain associated with Glehman et al. (2026), *Rapid
Earthquake Magnitude Estimation for Local Early Warning Systems Using
Seismogeodesy*, JGR Solid Earth, DOI `10.1029/2025JB033222`.

## Seed selection

All three seeds completed 3,000 anchor epochs and 120 deep epochs. The seed was
selected only by event-equal internal validation all-second MAE, with
validation final MAE and seed number as deterministic tie breaks.

| Seed | Validation online MAE | Validation final MAE | Test online MAE | Test final MAE | Selected |
|---:|---:|---:|---:|---:|:---:|
| 17 | 0.360821 | 0.216242 | 0.351281 | 0.198948 | no |
| 42 | 0.390380 | 0.240467 | 0.348638 | 0.193354 | no |
| 73 | **0.318842** | **0.194136** | **0.327774** | 0.222771 | yes |

Seed 42 has the lowest observed test final MAE, but selecting it after reading
test would contaminate the test boundary. The predeclared validation rule is
therefore retained, and external evaluation uses only seed 73. There is no
seed averaging or per-event seed choice.

## Internal gate

The external directory remained unopened until the selected seed passed both
predeclared internal requirements:

| Metric | Phase19 baseline | Maximum allowed | Phase22 | Improvement | Passed |
|---|---:|---:|---:|---:|:---:|
| Test all-second MAE | 0.344605 | 0.344605 | 0.327774 | 0.016832 | yes |
| Test 200 s MAE | 0.239998 | 0.229998 | 0.222771 | 0.017227 | yes |

At 200 seconds, 18 of 31 test-event absolute errors improve. The largest
remaining errors are Napa (`0.862951 Mw`), Miyagi2011B (`0.731231 Mw`),
Iquique (`0.563626 Mw`), Melinka (`0.434527 Mw`), Illapel (`0.427115 Mw`),
and Ibaraki (`0.422127 Mw`). These include both sparse and dense station
subsets, so station count alone does not explain the remaining failures.

The internal result is materially better, but `0.222771 Mw` is still above the
desired `0.15 Mw` level. This experiment therefore closes one robustness gap;
it does not establish that internal unseen-station accuracy is solved.

## External development check

Only after the internal gate passed did the runner hash and load the fixed
eight-event external directory.

| Metric | Phase19 | Phase22 | Phase22 minus Phase19 |
|---|---:|---:|---:|
| All-second Event MAE | 0.215284 | 0.220199 | +0.004914 |
| 200-second Event MAE | 0.131990 | 0.123864 | -0.008125 |

The final estimate improves, while all-second external error is slightly worse.
The eight events are a development validation set and are not an unbiased paper
test. Their result was not used to select a seed or tune this candidate.

## Verification

- implementation commit: `8884db2`;
- clean detached formal worktree at that commit;
- source data SHA-256:
  `2e1fa4c12fc1eb03ffc8bf9235491f0886d4b0d360ebcb3486baca4d948cfd6a`;
- selected checkpoint SHA-256:
  `61969f71eff384288de22af0d826fd2f03d181e3743c4894d4ce8dacc8baed6b`;
- all 26 registered run artifact hashes matched;
- all 24 external input hashes matched;
- 270 checkpoint tensors across three seeds were finite;
- each training log contained 3,000 anchor plus 120 deep rows;
- all four deep-loss terms reproduced the recorded weighted total;
- internal and external MAE values were independently recomputed directly from
  prediction CSV files;
- all per-second rows used no more than five currently active stations;
- the scalar neural residual was exactly zero at 200 seconds;
- focused regression: `5 passed`;
- all training and evaluation processes exited and the GPU returned to idle.

The GitHub-facing tables and figures are under
`docs/results/phase22-causal-forward-guided-station-subset/`.
