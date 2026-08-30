# Second-by-Second GNSS Magnitude Estimation with Physics-Guided Forward Synthesis and Counterfactual Seismic-Moment Scaling: Internal Validation of Phase 39

> Manuscript status: complete English draft. The reported evidence is an internal-validation screen, not a complete five-fold OOF campaign, locked-test result, or final evaluation on the eight external events.

## Abstract

Rapid, stable, and physically interpretable earthquake magnitude estimation is a central problem in GNSS-based earthquake early warning. Conventional peak ground displacement (PGD) methods infer magnitude from empirical relationships among peak displacement, source distance, and magnitude. They are efficient, but they compress the waveform into a peak statistic and do not explicitly require an inferred source time function (STF) to reproduce the observed displacement through forward physics. We extend the single-station radial-displacement Phase 39 model with a second-by-second causal-prefix training strategy while preserving its architecture and 1,010,850 parameters. Every observed prefix produces a fixed 200 s nonnegative moment-rate/STF curve, and magnitude is derived only from the integral of that curve. The central physical constraint is the differentiable synthesized-waveform loss, `L_synth`. To expose the constrained model to a broader range of seismic moments, each original sample is paired with a counterfactual moment-scaled sample. We draw `Delta Mw in [-0.75, 0.5]`, set `a = 10^(1.5 Delta Mw)`, multiply the radial waveform, model input, and target STF by `a`, and shift the target magnitude to `Mw + Delta Mw`, while leaving geometry, travel times, and masks unchanged. A soft error-descent loss with 0.03 Mw slack links a random prefix estimate to the 200 s endpoint without requiring strict second-by-second monotonicity.

Across three frozen internal-validation runs, the full method achieved a 200 s equal-event MAE of `0.13979 +/- 0.01930 Mw`, compared with a Phase 39 mean of `0.22993 Mw`. In the representative Fold 0 / Seed 73 run, event MAE decreased from `0.23808` to `0.13784 Mw`, and the post-120 s P95 of event-level one-second changes was `0.01947 Mw`. However, station-weighted MAE worsened from `0.28758` to `0.39058 Mw`. Noto 2024 contributes 375 of 424 validation stations; its event-median residual increased from `+0.229 Mw` under Phase 39 to `+0.423 Mw` under the full method, with a widespread positive station bias. The results support improved event-balanced accuracy and late-stage stability in the current internal validation, but they do not isolate the effect of moment scaling or solve cross-station calibration for dense-network complex ruptures.

**Keywords:** GNSS; earthquake early warning; magnitude estimation; source time function; physics-guided loss; waveform synthesis; seismic-moment scaling; causal prefix

## 1. Introduction

GNSS records permanent and long-period ground displacement without the strong-motion saturation that can affect large-earthquake magnitude estimation. Traditional PGD approaches map peak displacement and source distance to magnitude through empirical scaling laws. This approach is simple and deployable, but it usually reduces the waveform to one peak value, provides no explicit source-release history, and does not test whether the inferred moment release can reproduce the observed displacement through a propagation model.

Deep neural networks can exploit the full time series, yet direct scalar magnitude regression introduces two risks. First, the network may rely on dataset correlations rather than the physical relationship among seismic moment, the STF, and displacement. Second, if different waveform lengths are processed independently at every second, the resulting magnitude trajectory may fluctuate strongly without a direct connection between prefix accuracy and final convergence.

Phase 39 follows a different magnitude pathway. It first predicts a nonnegative STF/moment-rate curve, integrates the curve to obtain scalar moment and Mw, and uses `L_synth` to pass the predicted STF through a differentiable forward operator and compare the resulting radial displacement with the GNSS observation. This study preserves that physical pathway and addresses two training questions:

1. How can the unchanged Phase 39 model learn from genuinely variable 1--200 s causal prefixes?
2. How can it learn the physically expected change in moment when waveform amplitude is scaled, while retaining the required nonzero `L_synth` constraint?

The contributions are:

1. a parameter-neutral causal-prefix objective in which every prefix predicts a complete 200 s STF and obtains Mw from the same STF integral;
2. paired counterfactual seismic-moment scaling that expands the magnitude range over which the physics-constrained model is trained, without treating the scaled pair as an independent earthquake;
3. a slack-aware error-descent term that encourages overall convergence while allowing short-term revisions;
4. joint reporting of equal-event and station-weighted behavior, including a dedicated analysis of the Noto 2024 degradation.

![Method workflow](figures/00_method_workflow.png)

## 2. Methods

### 2.1 Problem definition, inputs, and outputs

For station `i` at observation horizon `h`, the model receives a processed single-component radial GNSS displacement prefix

```text
x_i,1:h in R^(1 x h),    h = 1, 2, ..., 200 s,
```

and five geometric metadata features

```text
g_i = [ln(r_i), sin(theta_i), cos(theta_i), sin(phi_i), cos(phi_i)].
```

Here, `r_i` is source distance and `theta_i`, `phi_i` describe the source-station geometry. Inference uses no handcrafted PGD feature, focal mechanism, recurrent state, or independent Mw regression head.

For any prefix length, the network outputs a fixed 200-step nonnegative moment-rate curve

```text
Mdot_i^(h)(tau),    tau = 1, 2, ..., 200 s.
```

Scalar moment and magnitude are uniquely derived from that output:

```text
M0_i^(h) = integral Mdot_i^(h)(tau) d tau,
Mw_i^(h) = (2/3) [log10 M0_i^(h) - 9.1].
```

Each station produces an independent Mw estimate. The event estimate is the median across validation stations for that event, and the primary metric weights events equally.

### 2.2 Dataset, split, and evidence role

The active dataset snapshot contains 31 earthquakes and 2,558 station records that pass preprocessing and cohort selection. The representative Fold 0 / Seed 73 grouped-event split is:

| Subset | Events | Station records | Scored here |
|---|---:|---:|:---:|
| Training | 19 | 1,744 | Yes |
| Internal validation | 6 | 424 | Yes |
| Locked test | 6 | 390 | No |

The validation events are Anchorage 2018, Maule 2010, Noto 2024, Parkfield 2004, Rat Islands 2014, and Sand Point 2020. Noto alone contributes 375 validation stations. All detailed figures and event analyses use only these six internal-validation events. The locked test fold was not scored, and the eight external events were not loaded.

The screen contains Fold 0 / Seed 73, Fold 0 / Seed 42, and Fold 1 / Seed 73. These runs assess repeatability across two random seeds and a second event fold, but they are not a complete five-fold out-of-fold confirmation.

### 2.3 Waveform and STF preprocessing

Each sample uses radial displacement only, sampled at 1 Hz over 0--200 s after origin. Preprocessing includes:

- median pre-event baseline removal;
- a seven-tap Hamming FIR low-pass filter with a 0.2 Hz cutoff;
- preservation of physical absolute amplitude;
- inclusion in the current cohort when the processed full-record radial peak is at least 2 cm;
- integral-preserving SCARDEC STF processing, with total moment scaled to the catalog Mw.

Because the 2 cm membership rule uses the full 200 s record, this is a causal-prefix experiment on a fixed retrospective station cohort, not an end-to-end real-time station-selection system. The centered filtering and processing contract uses a 6 s delay, so an estimate indexed by observation horizon `h` has an approximate release time of `h + 6` s.

### 2.4 Phase 39 network

The Phase 39 architecture remains unchanged and contains 1,010,850 parameters:

1. a seven-point Conv1D input embedding;
2. six residual dilated TCN blocks;
3. squeeze-excitation channel attention;
4. sinusoidal positional encoding and a three-layer, four-head Transformer encoder;
5. a `moment_shape_factorized` STF output head.

The output head decomposes moment rate into total moment scale and a normalized temporal shape:

```text
p(t) >= 0,    integral p(t) dt = 1,
Mdot0(t) = M0 p(t).
```

Nonnegativity and exact integration to `M0` are therefore enforced by parameterization. The experiment uses no adapter, teacher, recurrent state, EMA, or handcrafted PGD input. All parameters are trained from scratch.

### 2.5 Base science loss and `L_synth`

At either a prefix or the 200 s endpoint, the base science objective is

```text
L_phys = L_STF + L_mag + 0.5 L_synth.
```

The terms are:

- `L_STF`: mean squared error between predicted and reference rates in the `log10(1 + Mdot0/M_ref)` encoding, with `M_ref = 10^18 N m/s`;
- `L_mag`: squared error between catalog Mw and the Mw obtained by integrating the predicted STF;
- `L_synth`: normalized error between the observed radial displacement and the radial displacement synthesized from the predicted STF through a differentiable forward operator.

The forward operator is

```text
u_hat_R = F(Mdot0; r, theta, phi, alpha, beta, rho),
```

with `alpha = 7900 m/s`, `beta = 4533 m/s`, and `rho = 3400 kg/m^3`. It uses absolute P/S delays, far-field P and S terms, and full radiation coefficients; intermediate-field terms are disabled in this configuration. The Glehman scalar radiation-coefficient contract follows the formulation described by [Glehman et al. (2026)](https://doi.org/10.1029/2025JB033222).

For each waveform, `L_synth` is normalized by the maximum observed absolute amplitude and permits one global polarity reversal:

```text
L_synth,i = min over s in {-1,+1}
            sum_t m_it [(s u_hat_it - u_it)/A_i]^2 / sum_t m_it,
A_i = max_t |u_it|.
```

This is the central physical innovation. It requires the predicted STF not only to yield a plausible magnitude but also to generate a radial displacement consistent with the observation under the propagation and radiation model. Because the term is amplitude-normalized, it most directly constrains waveform structure and relative physical consistency; absolute moment scale is jointly constrained by `L_STF`, `L_mag`, and the factorized moment head.

### 2.6 Counterfactual seismic-moment scaling

For each original training sample, we draw

```text
Delta Mw ~ Uniform(-0.75, 0.5),
a = 10^(1.5 Delta Mw).
```

The Mw definition implies that a magnitude shift `Delta Mw` corresponds to multiplication of scalar moment by `a`. We construct

```text
x_tilde       = a x,
u_tilde_R     = a u_R,
Mdot_tilde(t) = a Mdot(t),
Mw_tilde      = Mw + Delta Mw.
```

Source distance, azimuthal geometry, P/S arrival times, sampling interval, validity masks, and event identity remain unchanged. Original and scaled samples receive equal weight in the same batch, and each is evaluated with the complete nonzero-`L_synth` science objective.

The scaled sample is not a newly generated independent earthquake. It is a counterfactual moment variant under fixed geometry and timing. Its purpose is to train an approximate equivariance: if the observed displacement and reference moment rate are multiplied by `a`, the inferred scalar moment should also be multiplied by `a`. Scaling therefore does not replace `L_synth`; it broadens the moment range over which the `L_synth`-constrained model is trained and discourages memorization of a sparse set of catalog magnitude-geometry combinations.

### 2.7 Causal-prefix training and soft error descent

At each optimization step, a randomized prefix `h` is selected through a deterministic cycle over 5--199 s, and the same batch is also processed at the full 200 s endpoint. Both views use the full science objective:

```text
L_phys^h,    L_phys^200.
```

To encourage overall convergence, we define

```text
e_h   = |Mw_hat_h - Mw_true|,
e_200 = |Mw_hat_200 - Mw_true|,
L_descent = [max(0, e_200 - stopgrad(e_h) - 0.03)]^2.
```

The total objective is

```text
L = 0.5 L_phys^h + 1.0 L_phys^200 + 0.5 L_descent.
```

This loss does not impose strict monotonic improvement between adjacent seconds. It only penalizes endpoint error that exceeds the sampled prefix error by more than 0.03 Mw. The model may revise its estimate upward or downward as new evidence arrives, while being encouraged to end with lower error and a stable late-stage trajectory.

### 2.8 Training and evaluation

Training uses AdamW with initial learning rate `1e-4`, weight decay `1e-5`, batch size 64, at most 200 epochs, gradient-norm clipping at 1.0, and cosine warm restarts. Event-balanced sampling prevents high-station-count training events such as Tohoku and Ibaraki from completely dominating optimization. The checkpoint is selected only by internal-validation 200 s event MAE.

At evaluation, each true variable-length prefix `B x 1 x h`, for `h = 1, ..., 200`, is processed again to obtain a new complete STF and Mw. The primary metric is equal-event MAE over the six validation-event medians. Station MAE over all 424 station predictions is reported separately to expose dense-network behavior.

## 3. Results

### 3.1 Three internal-validation runs

![Overall internal-validation results](figures/result_overview.png)

| Fold / Seed | Phase 39 endpoint event MAE | Full-method endpoint event MAE | Improvement | Full-method minimum second-wise MAE |
|---|---:|---:|---:|---:|
| 0 / 73 | 0.23808 | **0.13784** | 0.10024 | 0.13349 @ 195 s |
| 0 / 42 | 0.25944 | **0.16000** | 0.09944 | 0.15983 @ 194 s |
| 1 / 73 | 0.19228 | **0.12154** | 0.07073 | 0.09285 @ 159 s |

The full-method mean is `0.13979 +/- 0.01930 Mw`, compared with a Phase 39 mean of `0.22993 Mw`, for a mean improvement of `0.09014 Mw`. All three runs fall below both the 0.20 Mw screening target and the 0.17 Mw stretch target. Because the evidence covers only two folds and three fold/seed combinations, it remains a configuration screen rather than complete five-fold confirmation.

### 3.2 Second-by-second error and late-stage stability

Representative Fold 0 / Seed 73 equal-event MAE is:

| Observation horizon | Phase 39 | Full method |
|---:|---:|---:|
| 1 s | 1.40645 | 0.66256 |
| 30 s | 0.69278 | 0.32473 |
| 60 s | 0.47773 | 0.29859 |
| 90 s | 0.50009 | 0.29566 |
| 120 s | 0.44818 | 0.31801 |
| 160 s | 0.32534 | 0.24357 |
| 195 s | 0.23582 | **0.13349** |
| 200 s | 0.23808 | **0.13784** |

The full-method trajectory improves overall but is not monotonic. The fraction of adjacent seconds with nonincreasing MAE is 58.8%. Under the aggregate definition that all later MAEs remain below endpoint MAE + 0.05 Mw, the stable horizon is 174 s. The post-120 s P95 of per-event absolute one-second changes is `0.01947 Mw`, compared with approximately `0.04130 Mw` for Phase 39. These results support gradual improvement and late stabilization, but not the claim that every second is more accurate than the previous second.

### 3.3 Equal-event improvement and station-weighted degradation

![True and estimated magnitude scatter](figures/01_prediction_scatter.png)

At 200 s in the representative run, equal-event MAE improves from `0.23808` to `0.13784 Mw`, while station-weighted MAE worsens from `0.28758` to `0.39058 Mw`. The discrepancy follows from extreme station-count imbalance: each event contributes `1/6` of the primary event metric, whereas Noto contributes `375/424 = 88.4%` of the station metric.

| Event | Catalog Mw | Phase 39 estimate | Full-method estimate | Phase 39 absolute error | Full-method absolute error | Stations |
|---|---:|---:|---:|---:|---:|---:|
| Anchorage 2018 | 7.10 | 7.076 | 7.064 | 0.024 | 0.036 | 8 |
| Maule 2010 | 8.80 | 8.830 | 8.823 | 0.030 | 0.023 | 18 |
| Noto 2024 | 7.50 | 7.729 | 7.923 | 0.229 | **0.423** | 375 |
| Parkfield 2004 | 5.97 | 6.926 | 6.293 | 0.956 | **0.323** | 11 |
| Rat Islands 2014 | 7.90 | 7.851 | 7.892 | 0.049 | 0.008 | 3 |
| Sand Point 2020 | 7.60 | 7.460 | 7.586 | 0.140 | 0.014 | 9 |

Most of the gain comes from Parkfield, Sand Point, and Rat Islands. Maule remains accurate, Anchorage degrades slightly, and Noto degrades substantially. The evidence does not support a claim of uniform improvement across events or stations.

### 3.4 Dynamic behavior of representative events

![Selected-event station convergence](figures/02_station_convergence_scatter.png)

Parkfield overestimation decreases from approximately `+0.956 Mw` under Phase 39 to `+0.323 Mw` under the full method, suggesting better low-magnitude calibration, although the event remains outside the `+/-0.20 Mw` band. Maule is underestimated for much of the record and converges rapidly near 200 s. Noto has approximately 0.262 Mw absolute error at 30 s, worsens to 0.557 Mw at 120 s, and ends at 0.423 Mw, indicating persistent positive bias rather than normal convergence.

![Parkfield waveform, PGD, and second-by-second Mw](figures/03_parkfield_pgd_and_mw.png)

The Parkfield example uses station HOGS, approximately 12 km from the epicenter, with a three-component PGD of 11.46 cm. The model input is radial displacement; three-component PGD is shown only for physical context and is not provided to the network. The full method recomputes a complete STF for every prefix. Its event-median Mw adjusts rapidly at early times and then forms a stable plateau near Mw 6.3.

### 3.5 Station geometry and the Noto error

![Epicenters and station distributions](figures/04_selected_event_maps.png)

Stations are colored by the full-method 200 s residual. Noto exhibits widespread positive residuals across its dense network rather than a small set of outliers. Its 375 station residuals are summarized below:

| Metric | Noto 2024 |
|---|---:|
| Median station residual | +0.42283 Mw |
| Mean station residual | +0.38032 Mw |
| Station MAE | 0.41396 Mw |
| 5th percentile | -0.09888 Mw |
| 95th percentile | +0.71913 Mw |
| Fraction of 1--200 s event-median residuals that are positive | 97.5% |

Noto is therefore the main current failure case and the dominant cause of the degraded station-weighted metric.

## 4. Discussion

### 4.1 How the method should be explained

The clearest description is not that Phase 39 received an isolated scaling trick. Rather:

> The proposed system is a second-by-second causal-prefix model with an STF-only magnitude pathway and `L_synth` as its central physical constraint. Counterfactual seismic-moment scaling broadens the moment range over which the physics-constrained model is trained, while the soft descent loss links prefix accuracy to final convergence.

The three components have distinct roles:

- **STF plus `L_synth`:** defines the physical identity of the method. The model must predict an integrable source-release history that is forward-consistent with the waveform;
- **moment scaling:** supplies controlled amplitude/Mw pairs and encourages seismic-moment equivariance under fixed geometry;
- **descent loss:** shapes dynamic behavior so that the endpoint is generally better than a shorter prefix while permitting revisions.

The innovation can therefore be presented in two layers. The core physical innovation remains the nonzero synthesized-waveform constraint. Moment scaling is a complementary training innovation that enforces the complete objective, including `L_synth`, over counterfactual moment states rather than replacing the physical constraint.

### 4.2 Why moment scaling is useful

An observed event usually provides only one catalog Mw. A model can associate geometry, duration, or network-specific patterns with that magnitude without learning that an amplitude change of factor `a` should imply `Delta Mw = (2/3) log10 a`. Paired scaling presents the same geometry at two moment levels and asks whether the model changes its inferred scalar moment according to the physical transformation.

The strategy may:

1. fill the sparse catalog-magnitude axis continuously, especially toward the underrepresented lower-magnitude end;
2. reduce reliance on event identity or fixed geometry-magnitude combinations;
3. apply STF, Mw, and forward-synthesis supervision to both original and counterfactual pairs.

However, the current `L_synth` is normalized by each waveform's peak amplitude and allows a global polarity flip. Its direct absolute-amplitude constraint is therefore limited. Absolute moment supervision under scaling still depends mainly on `L_STF`, `L_mag`, and the factorized moment head. The paper should not claim that scaling directly strengthens the amplitude sensitivity of `L_synth`; the defensible claim is that it broadens the domain of the full physics-guided objective and asks the `L_synth`-constrained model to satisfy moment equivariance.

### 4.3 Conceptual advantages and costs relative to traditional PGD

Relative to empirical PGD scaling, the proposed method has several conceptual advantages:

1. it uses the complete radial waveform prefix rather than one peak statistic;
2. it outputs a complete STF and scalar moment, giving Mw an explicit integral origin;
3. `L_synth` checks whether the inferred source process can reproduce the observed displacement through forward physics;
4. it can update the STF and Mw whenever another second of data becomes available;
5. moment scaling explicitly encodes the physical transformation between amplitude and Mw during training.

The costs are greater model and training complexity, dependence on reference STFs and source-station geometry, and sensitivity to errors in propagation physics, radiation coefficients, and STF labels. This result package does not contain a matched numerical PGD comparison on the same validation events and input conditions, so the current manuscript makes a methodological comparison only and does not claim numerical superiority over PGD.

### 4.4 Why Noto has a large error

The evidence establishes systematic positive bias but does not by itself prove the cause. The following explanations are testable hypotheses:

1. **Amplitude-only scaling ignores magnitude-duration coupling.** The augmentation changes amplitude but not STF duration, rupture propagation, or multi-pulse complexity. Large complex ruptures need not satisfy this amplitude-only equivariance;
2. **A dense heterogeneous network exposes propagation and radiation mismatch.** The 375 stations cover broad distances and azimuths, making simplified velocity, radiation, and field-term assumptions more consequential;
3. **All stations share the same event reference STF.** Stations predict independently, but they are supervised toward one source STF. Unmodeled path effects may be absorbed into the inferred total-moment scale;
4. **Normalized polarity-invariant `L_synth` weakly constrains absolute scale.** A waveform-shape match can coexist with systematic magnitude bias;
5. **Training and station metrics use different weighting.** Event-balanced sampling and the primary event-median metric treat Noto as one event, while station MAE gives it 88.4% of the validation weight;
6. **Noto may lie outside the useful range of fixed-shape moment counterfactuals.** Wide moment scaling under unchanged temporal structure may create unrealistic pairs for complex ruptures.

These hypotheses can be separated through distance- and azimuth-residual analysis, station subsampling, duration-aware scaling, and forward-model sensitivity tests.

### 4.5 A defensible definition of dynamic convergence

The scientific objective should not be strict monotonic decrease of `|error_h|` for every event at every second. A rational estimate may move away from the catalog value when a new waveform phase or station signal arrives. A stronger convergence analysis combines:

- the long-term trend of aggregate MAE;
- quantiles of late one-second changes;
- suffix-stable entry into an error band;
- endpoint error, peak-to-final revision, and plateau width;
- event-specific trajectories rather than only an overall mean.

The full method remains within endpoint MAE + 0.05 Mw after 174 s and has a late-step P95 of 0.01947 Mw, meeting an initial requirement of gradual aggregate improvement and late stabilization. Noto demonstrates that stability alone is not accuracy.

### 4.6 Recommended next steps

Future work should preserve nonzero `L_synth` as the central physical constraint and proceed in the following order:

1. **Strict matched ablation:** under identical fold, seed, initialization, and budget, compare causal-only, `+ moment scaling`, `+ descent`, and matched `lambda_synth = 0` configurations;
2. **Complete five-fold OOF confirmation:** freeze the selected configuration, run all folds once, and do not tune after viewing OOF outcomes;
3. **Noto-specific residual audit:** analyze residuals against distance, azimuth, PGD, subnet, and waveform quality, with fixed-size station subsampling;
4. **Duration-aware scaling:** modify STF duration or time scale together with moment rather than changing amplitude alone;
5. **Hierarchical or robust station loss:** constrain both event medians and within-event residual distributions so equal-event optimization cannot hide dense-network bias;
6. **Multi-station event fusion:** explicitly exploit cross-station consistency while preserving prefix causality, rather than aggregating only at the output median.

## 5. Limitations

1. The evidence comprises three internal-validation runs, not a complete five-fold OOF campaign;
2. causal-prefix retraining, moment scaling, and the descent loss changed together, so a strict component ablation is missing;
3. the locked test fold and eight external events were not scored in this study;
4. the 2 cm station cohort depends on the full 200 s peak and is not an end-to-end real-time system;
5. the TCN uses symmetric padding and the Transformer is not autoregressively masked. The network receives no samples beyond the supplied prefix, but it is not a sample-recursive causal architecture;
6. normalized, global-polarity-invariant `L_synth` provides limited direct constraint on absolute amplitude and local polarity errors;
7. simplified propagation velocities, radiation coefficients, and field terms may generate systematic mismatch for dense-network events such as Noto;
8. the dataset has substantial imbalance in magnitude and station count, with large high-magnitude training events contributing many records.

## 6. Conclusions

We reformulated the unchanged Phase 39 network as a second-by-second causal-prefix model centered on a nonzero synthesized-waveform constraint, and added counterfactual seismic-moment scaling plus a soft error-descent loss. Across three internal-validation runs, the full method reduced mean 200 s equal-event MAE from the Phase 39 value of `0.22993 Mw` to `0.13979 +/- 0.01930 Mw`. The representative run reached approximately 0.02 Mw late-stage one-second variation, supporting progression to a frozen five-fold OOF confirmation.

At the same time, Noto 2024 error increased to 0.423 Mw and its 375 stations showed widespread positive bias, causing substantial degradation in station-weighted MAE. The most defensible conclusion is therefore that the joint causal-prefix, moment-scaling, and `L_synth`-constrained system improves event-balanced internal-validation accuracy and stability, but retains a systematic calibration problem for dense-network complex events, and the isolated contribution of moment scaling remains to be established by matched ablation.

## Data, figures, and code availability

- [Complete Chinese manuscript](MANUSCRIPT_ZH.md)
- [Machine-readable experiment summary](summary.json)
- [English figure manifest](figures/figure_manifest.json)
- [Chinese figure manifest](figures/zh/figure_manifest.json)
- [Reproducible bilingual plotting script](../../../scripts/plotting/plot_phase39_moment_scaling_explainer.py)

## Reference

Glehman, S., et al. (2026). *Journal of Geophysical Research: Solid Earth*. [https://doi.org/10.1029/2025JB033222](https://doi.org/10.1029/2025JB033222)
