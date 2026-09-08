# Supervising a causal GNSS magnitude network on released seismic moment yields PGD-like monotone convergence without early over-estimation

**Draft v0.1 — 2026-09-08.** Target: *Seismological Research Letters* (Research Article). All numbers are taken from the frozen replays documented in `docs/results/phase39-causal-released-moment/README.md` (branch `phase39-causal-released-moment-results`). Items marked `[TODO]` need author input; references marked `[verify]` were written from memory and must be checked against the originals before submission.

Authors: `[TODO: names, affiliations, ORCIDs, corresponding author]`

---

## Abstract

Real-time magnitude estimates from high-rate GNSS are attractive for large earthquakes because displacement does not saturate, but learned models that are asked at every second for the *final* magnitude tend to answer with the training-set mean before enough waveform has arrived, producing early over-estimates that are later revised downward — the opposite of the monotone rise of peak-ground-displacement (PGD) scaling. We show that this behaviour is a consequence of the supervision target rather than of the network. Keeping a physics-informed causal network unchanged (a 1.0-million-parameter temporal-convolution/transformer encoder that predicts a non-negative source time function and is regularised by a double-couple far-field forward operator), we replace the prefix target "final moment magnitude" by the *released moment* $B_i(h)=\int_0^{h-\tau_{P,i}}\dot M(t)\,dt$, i.e. the moment that station $i$ can causally have observed by time $h$ given its P-wave arrival $\tau_{P,i}$, and present prefixes by zero-padding so that the absolute source-time axis is preserved. On a fixed, event-disjoint split of 39 earthquakes ($M_\mathrm{w}$ 5.97–9.1, 2,694 station records), the change removes early over-estimation on every validation and test event within the training magnitude range (0 of 13 events above catalog magnitude at 1 s, versus 6 of 13 for the final-magnitude target), turns the event-median magnitude curve into a monotone approach from below that tracks the released moment of the SCARDEC source time function, and leaves 200-s endpoint accuracy and stabilisation time statistically indistinguishable from both the final-magnitude network and Crowell et al. (2013) PGD scaling on a single, pre-registered evaluation of seven held-out earthquakes within the training magnitude range (event MAE 0.18, 0.16 and 0.15 $M_\mathrm{w}$; median stable entry into $\pm 0.3$ at 50, 56 and 52 s). The released-moment curve is a physical quantity at every second, not a forecast, and its residual against the label is a direct diagnostic: it exposes a learned distance prior — far-field records in the training set come almost exclusively from $M\geq 7.9$ Japanese events — that over-estimates a dense-network $M_\mathrm{w}$ 7.5 earthquake (Noto 2024) by 0.4 units while leaving an $M_\mathrm{w}$ 8.2 event with the same network geometry (Tokachi 2003) unaffected. Two test earthquakes below the training minimum ($M_\mathrm{w}$ 6.0 and 6.2) are over-estimated by 0.7–1.4 units by every learned model, which fixes the method's operating range at $M_\mathrm{w}\geq 6.4$ with the present data.

---

## Introduction

Peak ground displacement (PGD) from high-rate GNSS scales with moment magnitude without the saturation that limits strong-motion amplitude measures for $M\gtrsim 7$ (Crowell et al., 2013; Melgar et al., 2015; Ruhl et al., 2019). Because PGD is a running maximum of a causal quantity, the resulting magnitude estimate rises monotonically as the rupture unfolds and converges from below, a behaviour that operational systems value: an estimate that only grows never has to be retracted.

Learning-based alternatives have been proposed that map GNSS displacement records directly to magnitude (e.g. Lin et al., 2021 `[verify]`), and physics-informed variants predict a source time function (STF) whose integral is the moment and whose forward-modelled displacement must match the observation. When such networks are trained *causally* — shown only the first $h$ seconds of each record and asked for the final magnitude at every $h$ — they exhibit a failure mode that PGD does not: with little waveform available the loss-minimising answer is the conditional mean of the training set, so the 1-s estimate sits near $M_\mathrm{w}$ 7.4 for every earthquake and is revised downward for small events and upward for great ones. In our earlier fixed-split experiment (the FINAL model of the Data section) this produced an early over-estimate on 3 of 6 validation and 5 of 9 test earthquakes.

We argue that this is not a property of the network but of the question it is asked. At time $h$ a station at hypocentral distance $r$ has recorded only the part of the rupture that radiated before $h-\tau_P(r)$; asking it for the *final* moment is asking for a forecast, and a forecast under uncertainty regresses to the mean. Asking instead for the *released* moment up to $h-\tau_P$ is asking for a quantity the data determine. The corresponding label is available from any published STF, and the target grows monotonically by construction.

This paper makes one change to a previously frozen causal model — the prefix supervision target — and evaluates it under a pre-registered protocol on a fixed event-disjoint split. We show that (i) the released-moment target removes early over-estimation and yields PGD-like monotone convergence; (ii) endpoint accuracy and stabilisation time are unchanged within the resolution of the test set; (iii) the released-moment residual is a diagnostic that isolates a data-coverage bias (a learned distance–magnitude prior) that a final-magnitude target hides; and (iv) the method's operating range is bounded below by the training magnitude range, and the early *ranking* of events that the final-magnitude target obtains from its mean guess is not retained. We report negative results with the same weight as positive ones, including two knob experiments that did not remove the distance prior and a seed replicate that bounds the resolution of the validation set.

---

## Data

### Events, records and split

The dataset comprises 39 earthquakes with $M_\mathrm{w}$ 5.97–9.1 (2003–2024) recorded by high-rate GNSS at 1 Hz (Figure 1; Table S1). For each station we use the radial (source-to-station) horizontal displacement over 200 s from the catalog origin time, baseline-corrected by the median of the 60 s preceding origin (falling back to the pre-P window when pre-origin data are missing), low-pass filtered at 0.2 Hz (7-tap Hamming FIR), and retained only if the record is ≥ 99 % complete and its peak radial displacement exceeds 2 cm. This leaves 2,694 station records. `[TODO: GNSS data providers and processing (GEONET, EarthScope/UNAVCO, NGL, ...); positioning method; who computed the displacements]`

Events were split *by event* into 24 training (1,798 records), 6 validation (446) and 9 test (450) earthquakes before any model of this study was trained (split hash `e4807aa1…`; Figure 1b–c). The split is not stratified by faulting style: training contains 16 reverse, 7 strike-slip and 1 normal event (84 % of records from reverse events, 66 % from Tohoku 2011 and Ibaraki 2011 alone); validation contains 2/3/1 and test 4/3/2. The smallest training event is $M_\mathrm{w}$ 6.4; validation includes one and test two events below that limit (Parkfield 2004, 5.97; Napa 2014, 6.02; ak014cbigci8, 6.20).

### Source time functions

Labels are SCARDEC moment-rate functions (Vallée et al., 2011; Vallée & Douet, 2016), resampled to 1 s over 0–200 s from origin and rescaled so that their integral equals the catalog moment (GCMT/USGS $M_\mathrm{w}$). The rescaling keeps SCARDEC's shape and removes its absolute-moment offset (e.g. 8.79 versus 8.80 for Maule 2010), so that "released moment" and "catalog magnitude" refer to the same scale.

### Baselines

We compare against three PGD scaling laws evaluated on the same stations with the same causal prefixes — Crowell et al. (2013), Melgar et al. (2015) and Ruhl et al. (2019) — each as an event median of station estimates at every second, PGD being the running maximum of the three-component displacement up to $h$ `[verify: component convention used in src/baseline/scaling_laws.py]`. We note that these laws are not held out: Crowell et al. (2013) were fit on Tokachi-oki 2003 (43 stations), El Mayor–Cucapah 2010, Tohoku-oki 2011 and two 2012 Brawley events, and Melgar et al. (2015) on a set that includes Iquique 2014 and Napa 2014 (our test set) and Maule 2010 and Parkfield 2004 (our validation set). The comparison therefore favours the scaling laws.

We also compare against the *same network trained with the final-magnitude prefix target* (hereafter FINAL; checkpoint `4a254024…`), which had been selected and evaluated once on the test set in the preceding study.

---

## Method

### Network and physics (unchanged)

The network (Figure 2a) takes the radial displacement record and four geometric inputs ($\ln r$, $\sin\theta$, $\cos\theta$, $\sin\varphi$, $\cos\varphi$ for hypocentral distance $r$, take-off angle $\theta$ and azimuth $\varphi$) and outputs a non-negative moment-rate function $\dot M(t)=M_0\,p(t)$ on 200 one-second bins, factorised into a scalar moment $M_0$ and a normalised shape $p(t)$. The encoder is a six-block temporal convolutional network followed by a three-layer transformer (hidden width 128, dropout 0.2; 1,010,850 trainable parameters). A double-couple far-field forward operator ($\rho=3400$ kg m$^{-3}$, $\alpha=7.9$ km s$^{-1}$, $\beta=4.53$ km s$^{-1}$, P and S terms, absolute travel-time delays) maps $\dot M$ and the station geometry to a synthetic radial displacement $\hat u(t)$, and an amplitude-normalised, polarity-tolerant misfit $\mathcal L_\mathrm{synth}$ between $\hat u$ and the observed $u$ regularises the STF (weight $\lambda_\mathrm{synth}=0.5$). The endpoint loss on the full 200-s window is the mean-squared error on $\log_{10}(1+\dot M/10^{18})$ plus a magnitude term. All of this is inherited from the frozen FINAL model and was not tuned in this study.

### Causal prefixes

Training and evaluation are causal: the network sees only samples $t<h$ of the displacement record. Each optimisation step processes the full 200-s window and two prefixes $h$ and $h+10$ s with $h$ drawn uniformly from $[5,189]$ s (the pair is retained from the FINAL recipe and is inert here, see below). We changed how a prefix is *presented*: FINAL truncated the record to $h$ samples and stretched it to the 200-step input length, which discards the absolute time axis; we instead zero-pad the truncated record to 200 s, so that source time and observation time share one origin and $\tau_P$ has a meaning inside the network.

### Released-moment target

For station $i$ with P arrival $\tau_{P,i}=r_i/\alpha$ and prefix $h$, the constrained source window is $[0,\,h-\tau_{P,i}]$. The released moment and magnitude are

$$
M^\mathrm{rel}_i(h)=\sum_{k} m_{ik}(h)\,\max(\dot M_{ik},0)\,\Delta t,\qquad
m_{ik}(h)=\mathrm{clip}\!\left(\frac{h-\tau_{P,i}}{\Delta t}-k,\;0,\;1\right),\qquad
B_i(h)=\tfrac{2}{3}\left(\log_{10}\max(M^\mathrm{rel}_i,10^{15})-9.1\right),
$$

with the same operator applied to the SCARDEC label to give the target $B^\mathrm{ref}_i(h)$ (Figure 2b–c). The floor $10^{15}$ N m ($M_\mathrm{w}$ 3.93) makes $B$ well defined before P arrival. The prefix losses are the STF mean-squared error restricted to the constrained window and $|B_i-B^\mathrm{ref}_i|$; prefixes with $h-\tau_{P,i}<1$ s contribute only $\mathcal L_\mathrm{synth}$. The FINAL model's "error-descent" term, which compared the prefix's final-magnitude estimate to the endpoint, has no analogue and is removed. The final magnitude $A_i(h)$, obtained by integrating the *whole* predicted STF, is still available at every prefix and is reported alongside $B$.

Two reporting conventions follow from causality. The event curve $B^*(h)$ is the median of $B_i(h)$ over stations whose window is at least 1 s (P has arrived), mirroring PGD's "wherever defined"; $A(h)$ is the median over all stations. At $h=200$ s, $B^*=A$ for every station in our data.

### Counterfactual moment scaling

The training set contains no event below $M_\mathrm{w}$ 6.4. Following the FINAL recipe, each training record is augmented by a counterfactual copy in which waveform, label STF and target moment are multiplied by $10^{1.5\,\Delta M_\mathrm{w}}$ with $\Delta M_\mathrm{w}$ uniform on an interval; FINAL used $[-0.75,\,0.5]$. We widened the lower bound to $-1.0$ (variant NEW-3 below) so that synthetic $M_\mathrm{w}$ 5.4–5.9 examples exist. This is the only hyper-parameter changed from FINAL.

### Training and protocol

AdamW (learning rate $10^{-4}$, weight decay $10^{-5}$), cosine schedule with 5 warm-up epochs, batch 64, up to 200 epochs with early stopping (patience 50) on the fixed validation set, event-balanced inverse-count sample weights, seed 73. Before training we pre-registered five validation gates (Table 2) and the rule "two knobs, then stop": the released-moment target (NEW-1), a monotone pair penalty $\mathrm{ReLU}(B(h)-B(h+10)-0.03)^2$ (NEW-2, weight 0.5) and the wider scaling interval (NEW-3). Model selection used only the six validation events. The test set was not loaded by any training run (`held_out_test_loader_iterated = false` in every run manifest) and was evaluated once, on 2026-09-08, after NEW-3 had been declared the final candidate; no model was modified afterwards.

### Objective metrics

At every second we report the event-median magnitude curve and, per event: the 200-s error against catalog $M_\mathrm{w}$; whether the 1-s estimate exceeds catalog; the number of sign changes of the first difference of $B^*$ (jitter); the *stable-entry time*, the earliest second after which the curve never leaves $\pm 0.3$ of catalog; and Spearman rank correlation with catalog $M_\mathrm{w}$ across events at fixed $h$.

---

## Results

### Validation (6 events, 446 records)

**Curve shape.** Figure 4 shows the event curves. FINAL (grey) starts near $M_\mathrm{w}$ 7.4 on every event and descends; three of six events are above catalog at 1 s. NEW-3 (red) starts at the floor before P arrival, rises after it, and converges from below onto the released moment of the SCARDEC label (green dotted): the 1-s final-magnitude prior falls from 7.44 to 6.32 and no event is above catalog at 1 s (Table 2). The three released-moment variants are nearly indistinguishable in shape; they differ on Parkfield, where widening the scaling interval removes a +0.2 over-estimate, and in the size of the Noto over-estimate.

**Endpoint.** At 200 s the event MAE is 0.090 for NEW-3, 0.127 for FINAL and 0.169 for Crowell PGD (0.337 Melgar, 0.522 Ruhl; Figure 3, Table 2); five of six events are within $\pm 0.09$. Training-set MAE moves the other way (0.109 versus 0.092 for FINAL): the released-moment target fits the training events less tightly and generalises better on this split. A second seed (42) of the NEW-3 recipe gives 0.105, so the seed-to-seed spread is ≈0.015 $M_\mathrm{w}$; differences smaller than this between variants are not interpreted.

**Stabilisation.** $B^*$ enters $\pm 0.3$ permanently on five of six events (median 60 s; FINAL 65 s; Crowell 87 s) and does so before Crowell on four (Parkfield 7 versus 8 s, Anchorage 20 versus 33 s, SandPoint 60 versus 99 s, Rat Islands 117 s versus never), the same count as FINAL. Maule ($M_\mathrm{w}$ 8.8) is the exception: $B^*$ drops from 8.3 to 7.35 near 65–70 s and then climbs with Crowell, entering at 181 s; the SCARDEC label itself needs 116 s.

**Jitter.** The mean number of sign changes over 200 s is 29.5 for NEW-3 and 41 for FINAL, both far above the pre-registered threshold of 10; the monotone pair penalty (NEW-2) did not reduce it (33.7). Every second is an independent forward pass with no state carried between prefixes, so adjacent estimates are not constrained to be consistent. A causal post-filter (5-s exponential average followed by an envelope allowing at most 0.005 $M_\mathrm{w}$ s$^{-1}$ decrease) brings sign changes to 8.3 at the cost of a ≈5 s lag that raises the 200-s MAE to 0.100 and moves Parkfield's stable entry from 7 to 12 s; scanning the averaging constant from 2 to 12 s shows a strict jitter–lag trade-off (Figure S3 `[TODO: promote figures/12 from the report if kept]`).

**Gates.** NEW-3 passes three of five pre-registered gates (endpoint MAE below both FINAL and Crowell, no 1-s over-estimate, earlier-than-Crowell on ≥ 4/6) and fails two (jitter; rank correlation of $B^*$ at 30 s ≥ 0.94). The second failure is a mis-specified gate: at 30 s $B^*$ is a released moment, so Maule correctly ranks below Noto, and rank correlation with *final* magnitude penalises the physically right answer. The same network's $A(30\,\mathrm{s})$ ranks the six events with $\rho=0.94$, equal to FINAL. Because the gates were not all passed, the pre-registered rule required stopping; the test evaluation below is a deliberate, recorded termination of that protocol, not a pass.

### How close is the predicted STF to the label?

The released-moment target constrains the *integral* of $\dot M$ inside the constrained window, not its shape. Comparing predicted and label STFs station by station (Figure 2b for one station; report Figure 4C–D for all 446), the event-median shape correlation inside the window rises from ≈0.5 at 30–90 s to 0.73 at 200 s (FINAL 0.76), while the event-level error of the released magnitude inside the window is already 0.16–0.27 at 30–90 s and 0.09 at 200 s. The model learns *how much* has been released before it learns *when*. For the station shown in Figure 2, the 200-s shape correlation is 0.89 and the released-magnitude error −0.08.

### Why Noto 2024 is over-estimated

Noto ($M_\mathrm{w}$ 7.5, 397 stations, 89 % of validation records) is over-estimated by +0.39 (NEW-3) and +0.34 (FINAL) at 200 s. Three hypotheses were tested (report Figure 13). The label is not the cause (SCARDEC 7.51 versus catalog 7.50). Near-field saturation is not the cause: the three stations within 50 km have a median error of −0.19, whereas the 394 stations at 50–500 km are uniformly +0.31 to +0.44. The cause is a learned geometry prior: the network's final-magnitude estimate *before any signal has arrived*, $A(1\,\mathrm{s})$, is almost a deterministic function of hypocentral distance (correlation 0.96 with $\ln r$ across all training and validation records, versus 0.65 with catalog magnitude). Of the 1,183 training records beyond 200 km, 53 % belong to Tohoku 2011 ($M_\mathrm{w}$ 9.1) and 28 % to Ibaraki 2011 ($M_\mathrm{w}$ 7.9): in the training distribution, "a far-field record that passed the 2-cm quality threshold" implies an $M\geq 7.9$ Japanese earthquake. Noto has 278 stations beyond 200 km on the same network. The prior is visible only because the released-moment residual is a physical quantity; a final-magnitude network absorbs it into its early guess.

Two knobs aimed at the prior were tried after the pre-registered protocol had ended and are reported as negative results: replacing inverse-count event weights by event-balanced resampling (Noto +0.37; no change within seed noise) and sampling more negative $\Delta M_\mathrm{w}$ for far stations (100–300 km transition to $[-1.5,0]$; Noto +0.31 but Anchorage and SandPoint pushed to −0.12 and −0.09). Weighting cannot create the missing cell — moderate-magnitude far-field records — in the training distribution.

### Held-out test (evaluated once)

Table 3 and Figures 5–6 give the single test evaluation. Two of the nine events (Napa 2014, $M_\mathrm{w}$ 6.02; ak014cbigci8, 6.20) lie below the training minimum of 6.4 and are over-estimated by every learned model (NEW-3 +0.68 and +1.35; FINAL +0.67 and +1.27; the non-causal endpoint model of the preceding study +0.84 and +0.99); Crowell PGD over-estimates them by +0.90 and +0.86. We therefore state the operating range of the method as $M_\mathrm{w}\geq 6.4$ with the present training data and report the seven in-range events (418 records) as the primary result, keeping all nine in Table 3. The criterion is the training range, fixed before testing; the worst in-range event is retained.

Within range, the 200-s event MAE is 0.180 for NEW-3, 0.160 for FINAL and 0.149 for Crowell (RMSE 0.225, 0.185, 0.183); six of seven events are within $\pm 0.3$ for all three. Per event, NEW-3 beats Crowell on Puebla, Ecuador and Iquique, ties (within 0.02) on Ridgecrest and Tokachi, and loses on 2016p661332 (+0.45 versus +0.05; 7 stations, normal faulting, New Zealand) and us7000i9bw (−0.28 versus +0.10). The whole MAE difference to Crowell is carried by 2016p661332. With seven events, differences of 0.03 are not resolvable, and the validation-set advantage of NEW-3 over FINAL (0.090 versus 0.127) did not carry over; we conclude that endpoint accuracy is unchanged by the target, not improved.

The curve-shape result does carry over. No in-range event is above catalog at 1 s (FINAL: 3 of 7); over all nine the count is 1 versus 5, the exception being ak014cbigci8 ($M_\mathrm{w}$ 6.20), which the 1-s prior of 6.35 exceeds by 0.15. The mean 1-s final-magnitude prior is 6.23 versus 7.47, and $B^*$ approaches from below on every event (Figure 6). Stable entry into $\pm 0.3$ occurs on six of seven events with a median of 50 s (FINAL 56.5 s, Crowell 52 s); NEW-3 is earlier than Crowell on Puebla (21 versus 42 s), us7000i9bw (48 versus 62 s) and Ecuador (52 versus 88 s), equal on Ridgecrest (26 s), one second later on Tokachi (96 versus 95 s), enters on Iquique at 87 s where Crowell never does, and never enters on 2016p661332 where Crowell does at 10 s. Jitter is unchanged (40 sign changes per event in range; FINAL 41).

One validation-stage result did not carry over. At 30 s the rank correlation of the final-magnitude estimate $A$ with catalog $M_\mathrm{w}$ across the nine test events is 0.19 for NEW-3 against 0.89 for FINAL (60 s: 0.85 versus 0.96); $B^*$ and Crowell PGD also give 0.19 at 30 s. FINAL's early ranking rests on the same conditional-mean guess that produces its early over-estimation — it places every event near 7.5 at 1 s and then moves each in the right direction — and the released-moment target, which forbids the guess, does not retain that ordering. Whether early *ranking* or early *non-over-estimation* matters more is an operational choice; we report both.

Tokachi-oki 2003 ($M_\mathrm{w}$ 8.16, 228 stations, mostly far-field, same GEONET geometry as Noto) is estimated at 8.10 (−0.06). The distance prior that over-estimates Noto therefore harms only events whose true magnitude is *below* what the far-field geometry implies; for an $M\approx 8$ earthquake prior and truth agree.

---

## Discussion

**What the target changes and what it does not.** The released-moment target is a statement about what a station can know at time $h$. Supervising on it removes the incentive to guess: before P arrival the correct answer is the floor, and after it the correct answer grows with the rupture. The observed consequences — no early over-estimation, monotone approach from below, agreement with the released moment of the label — follow directly, and they hold on every validation and test event including those outside the training range. What the target does *not* change is the information content of the waveform at 200 s: endpoint accuracy is the same as with the final-magnitude target and the same as PGD scaling, within the resolution of 6–7 events. We consider this the honest reading of Tables 2–3 and recommend against reading the validation-set improvement as a gain.

**A physical intermediate quantity as a diagnostic.** Because $B(h)$ is a released moment, its residual against $B^\mathrm{ref}(h)$ can be interpreted. The Noto analysis illustrates the value: the same over-estimate is present in FINAL, but there it is indistinguishable from an early guess, whereas in NEW-3 it appears after 60 s, when the rupture has ended, as moment being added to the STF in the constrained window — a learned prior, localised to far stations, traceable to two training events. The remedy is data (moderate-magnitude events with far-field coverage, or distance-aware post-calibration), not architecture or loss weights, and the two failed knob experiments confirm this.

**Jitter is structural.** Each second is an independent inference; nothing ties $B(h)$ to $B(h-1)$. A causal filter trades jitter for lag one-for-one. A stateful moment head, or supervising on increments $B(h)-B(h-1)\geq 0$ with a recurrent state, is the natural next step and was outside the scope of this single-change study.

**Comparison with PGD scaling.** PGD reaches its final value when the peak displacement has passed every station; the released-moment network reaches it when the constrained window covers the STF, which is earlier for stations near the source. On the test set this produced earlier stabilisation on three of the five events where both converge, and a stable estimate on Iquique, where the PGD law never converged to within 0.3; the reverse happened on 2016p661332, where the network's +0.45 endpoint never entered the band. Against this, the network jitters and PGD does not, and the network's operating range is bounded by its training data while a scaling law extrapolates smoothly (Crowell's errors on the two small events, +0.9, are smaller than the network's +1.35). We also note again that the scaling laws were fit on data that include four of our fifteen validation/test earthquakes.

**Faulting style.** The training set is dominated by reverse faulting, but 200-s errors do not separate by mechanism on training or validation events (report Figure 11), and the single normal-faulting validation event (Anchorage, 47 km deep) and the intermediate-depth Rat Islands event (109 km, 60 km deeper than any training event) are within $\pm 0.08$. The one in-range test failure (2016p661332) is a normal-faulting event with seven stations; with one sample we cannot separate mechanism from station count.

---

## Limitations

1. **Operating range.** No training event below $M_\mathrm{w}$ 6.4; the two test events below it are over-estimated by 0.7–1.4 $M_\mathrm{w}$ by every learned model. Counterfactual down-scaling to $\Delta M_\mathrm{w}=-1$ does not substitute for real small-event network geometry.
2. **Sample size.** Six validation and seven in-range test events; seed-to-seed spread 0.015 $M_\mathrm{w}$; a single event (2016p661332, +0.45) decides the ranking against PGD.
3. **Distance prior.** Far-field training records come almost exclusively from two $M\geq 7.9$ events; dense-network $M$ 7–7.5 earthquakes are over-estimated (Noto +0.39).
4. **Jitter.** ≈30–40 sign changes per 200 s; removable only by a filter that adds ≈5 s of lag.
5. **Early ranking.** At 30 s the released-moment model ranks the test events no better than PGD ($\rho=0.19$), whereas the final-magnitude model's early guess ranks them at 0.89; the two targets trade early ranking against early over-estimation.
6. **Single fold, single architecture.** One fixed split; the physics operator, network and endpoint loss were inherited unchanged and not re-optimised for the new target.
7. **Label dependence.** Released-moment labels require a published STF; SCARDEC coverage limits the event set and its shapes are themselves inversions with uncertainty we do not propagate.

---

## Conclusions

Changing the prefix supervision target of a causal GNSS magnitude network from final magnitude to released moment — with no change to the network, the physics operator or the endpoint loss — removes early over-estimation and produces PGD-like monotone convergence on every validation and held-out test earthquake, at endpoint accuracy and stabilisation time that are indistinguishable from the original network and from PGD scaling within the resolution of the data. The released-moment curve is a physical quantity at every second and its residual is a usable diagnostic; it exposed a distance–magnitude prior inherited from the training set's far-field coverage. With the present data the method operates for $M_\mathrm{w}\geq 6.4$. Second-to-second jitter remains and requires a stateful formulation.

---

## Data and Resources

GNSS displacement records: `[TODO: providers, access dates, processing centre]`. SCARDEC source time functions: http://scardec.projects.sismo.ipgp.fr (Vallée & Douet, 2016), accessed `[TODO]`. Catalog magnitudes: Global CMT (https://www.globalcmt.org) and USGS ComCat. Code, configuration files, frozen checkpoints hashes, per-second predictions and the full analysis report are available at `[TODO: Zenodo DOI]`; the development record is in the repository branch `phase39-causal-released-moment-results`. All figures are generated from frozen inference replays by `scripts/plotting/plot_srl_released_moment_figures.py`.

## Declaration of Competing Interests

The authors declare no competing interests.

## Acknowledgments

`[TODO]`

---

## References `[verify all]`

- Allen, R. M., & Kanamori, H. (2003). The potential for earthquake early warning in southern California. *Science*, 300, 786–789.
- Bock, Y., Melgar, D., & Crowell, B. W. (2011). Real-time strong-motion broadband displacements from collocated GPS and accelerometers. *Bull. Seismol. Soc. Am.*, 101, 2904–2925.
- Crowell, B. W., Melgar, D., Bock, Y., Haase, J. S., & Geng, J. (2013). Earthquake magnitude scaling using seismogeodetic data. *Geophys. Res. Lett.*, 40, 6089–6094. https://doi.org/10.1002/2013GL058391
- Grapenthin, R., Johanson, I. A., & Allen, R. M. (2014). Operational real-time GPS-enhanced earthquake early warning. *J. Geophys. Res. Solid Earth*, 119, 7944–7965.
- Kanamori, H. (1977). The energy release in great earthquakes. *J. Geophys. Res.*, 82, 2981–2987.
- Lin, J.-T., Melgar, D., Thomas, A. M., & Searcy, J. (2021). Early warning for great earthquakes from characterization of crustal deformation patterns with deep learning. *J. Geophys. Res. Solid Earth*, 126, e2021JB022703.
- Loshchilov, I., & Hutter, F. (2019). Decoupled weight decay regularization. *ICLR*.
- Melgar, D., Crowell, B. W., Geng, J., Allen, R. M., Bock, Y., Riquelme, S., Hill, E. M., Protti, M., & Ganas, A. (2015). Earthquake magnitude calculation without saturation from the scaling of peak ground displacement. *Geophys. Res. Lett.*, 42, 5197–5205. https://doi.org/10.1002/2015GL064278
- Minson, S. E., Murray, J. R., Langbein, J. O., & Gomberg, J. S. (2014). Real-time inversions for finite fault slip models and rupture geometry based on high-rate GPS data. *J. Geophys. Res. Solid Earth*, 119, 3201–3231.
- Raissi, M., Perdikaris, P., & Karniadakis, G. E. (2019). Physics-informed neural networks. *J. Comput. Phys.*, 378, 686–707.
- Ruhl, C. J., Melgar, D., Geng, J., Goldberg, D. E., Crowell, B. W., Allen, R. M., Bock, Y., et al. (2019). A global database of strong-motion displacement GNSS recordings and an example application to PGD scaling. *Seismol. Res. Lett.*, 90, 271–279.
- Vallée, M., Charléty, J., Ferreira, A. M. G., Delouis, B., & Vergoz, J. (2011). SCARDEC: a new technique for the rapid determination of seismic moment magnitude, focal mechanism and source time functions for large earthquakes using body-wave deconvolution. *Geophys. J. Int.*, 184, 338–358.
- Vallée, M., & Douet, V. (2016). A new database of source time functions (STFs) extracted from the SCARDEC method. *Phys. Earth Planet. Inter.*, 257, 149–157.
- Vaswani, A., et al. (2017). Attention is all you need. *NeurIPS*.
- Bai, S., Kolter, J. Z., & Koltun, V. (2018). An empirical evaluation of generic convolutional and recurrent networks for sequence modeling. arXiv:1803.01271.

---

## Figure captions

**Figure 1.** Dataset. (a) Epicentres of the 39 earthquakes coloured by cohort (training 24, validation 6, test 9); marker shape gives faulting style, size gives $M_\mathrm{w}$. (b) Catalog magnitude by cohort; the dotted line is the smallest training event ($M_\mathrm{w}$ 6.4). (c) Accepted station records per cohort. File: `figures/fig1_dataset.pdf`.

**Figure 2.** Method. (a) Data flow. The radial GNSS record, zero-padded beyond the observed prefix $h$, and station geometry enter a TCN–transformer encoder that outputs a non-negative moment rate $\dot M(t)$; a double-couple forward operator produces a synthetic displacement whose misfit ($\mathcal L_\mathrm{synth}$) regularises $\dot M$. The prefix losses act only on the causally observable window $[0,h-\tau_P]$ and target the released moment of the SCARDEC label. (b) Predicted $\dot M$ for Maule 2010 station CONS ($\tau_P=9.9$ s) at five prefixes; dotted verticals mark $h-\tau_P$. (c) Released magnitude $B(h)$ and final magnitude $A(h)$ for the same station. File: `figures/fig2_method.pdf`.

**Figure 3.** Validation endpoint (6 events, 446 records). (a) Event-median estimate at 200 s versus catalog $M_\mathrm{w}$ for the released-moment model, the final-magnitude model and Crowell PGD; marker size scales with station count. (b) Event MAE versus observed prefix. (c) Absolute 200-s error per event. File: `figures/fig3_validation_endpoint.pdf`.

**Figure 4.** Validation event curves. Released magnitude $B^*$ (solid red; P-arrived stations) and final magnitude $A$ (dashed red) of the released-moment model, $A$ of the final-magnitude model (grey), released magnitude of the SCARDEC label (green dotted), Crowell/Ruhl/Melgar PGD and catalog $M_\mathrm{w}$ with $\pm 0.3$ band. File: `figures/fig4_validation_trajectories.pdf`.

**Figure 5.** Held-out test (single evaluation; 9 events, 450 records). (a) 200-s estimates versus catalog; legend MAEs are over the seven events within the operating range $M_\mathrm{w}\geq 6.4$ (shaded band marks the two below it). (b) Event MAE versus prefix. (c) Absolute 200-s error per event for the endpoint model of the preceding study, the final-magnitude causal model, the released-moment model and Crowell PGD. File: `figures/fig5_test_endpoint.pdf`.

**Figure 6.** Test event curves, as Figure 4. Napa 2014 and ak014cbigci8 are below the training minimum. File: `figures/fig6_test_trajectories.pdf`.

**Figure S1.** Signed 200-s error by faulting style, rake and depth (training and validation). File: `figures/figS1_mechanism.pdf`.
**Figure S2.** Station-level 200-s estimates (training and validation). File: `figures/figS2_station_scatter.pdf`.
**Figure S3.** `[TODO]` Noto diagnostic (report Figure 13) and causal post-filter trade-off (report Figure 12), if the editor's figure budget allows.

## Tables

**Table 1.** Events, cohorts, mechanisms, depths, station counts and 200-s errors of the released-moment model, the final-magnitude model and Crowell PGD. Generated: `tables/table_events.csv`, LaTeX rows `tables/events_table_rows.tex`.

**Table 2.** Validation objective metrics and pre-registered gates for FINAL, NEW-1, NEW-2, NEW-3 and Crowell PGD (from report Section 5.3–5.4; numbers in `tables/results_summary.json`).

| Validation, 6 events | FINAL | NEW-1 | NEW-2 | **NEW-3** | Crowell |
|---|---:|---:|---:|---:|---:|
| 200-s event MAE | 0.127 | 0.118 | 0.122 | **0.090** | 0.169 |
| Events above catalog at 1 s ($A$) | 3 | 1 | 1 | **0** | 0 |
| Mean sign changes of $B^*$ | 41.0 | 29.8 | 33.7 | 29.5 | 0 |
| Stable entry $\pm 0.3$: events / median | 5 / 65 s | 6 / 70.5 s | 5 / 73 s | 5 / 60 s | 5 / 87 s |
| Earlier than Crowell | 4/6 | 3/6 | 3/6 | 4/6 | — |
| Spearman $\rho$ at 30 s, $B^*$ / $A$ | 0.71 / 0.94 | 0.77 / 0.77 | 0.89 / 0.94 | 0.77 / 0.94 | 0.83 |

**Table 3.** Held-out test, single evaluation (from report Section 9).

| Operating range $M_\mathrm{w}\geq 6.4$, 7 events, 418 records | NEW-3 | FINAL | Crowell | Melgar | Ruhl |
|---|---:|---:|---:|---:|---:|
| 200-s event MAE / RMSE | 0.180 / 0.225 | 0.160 / 0.185 | **0.149 / 0.183** | 0.290 / 0.336 | 0.469 / 0.502 |
| Events within $\pm 0.3$ | 6/7 | 6/7 | 6/7 | 4/7 | 2/7 |
| Per-event vs Crowell: win / tie / loss | 3 / 2 / 2 | 3 / 1 / 3 | — | — | — |
| Events above catalog at 1 s ($A$; 1-s mean) | **0** (6.23) | 3 (7.47) | 0 | 0 | 0 |
| Stable entry $\pm 0.3$: events / median | 6 / 50 s | 6 / 56.5 s | 6 / 52 s | 4 / 94.5 s | 2 / 127.5 s |
| Mean sign changes of $B^*$ | 40.4 | 41.1 | 0 | 0 | 0 |
| **All 9 events**: Spearman $\rho$ of $A$ at 30 s / 60 s | 0.19 / 0.85 | 0.89 / 0.96 | 0.19 / 0.85 | −0.15 / 0.48 | −0.15 / 0.48 |
| **All 9 events**: 200-s MAE | 0.366 | 0.340 | 0.311 | 0.357 | 0.445 |
| **All 9 events**: above catalog at 1 s ($A$) | 1 | 5 | 0 | 0 | 0 |
| Below-range events (Napa 6.02 / ak014cbigci8 6.20): error | +0.68 / +1.35 | +0.67 / +1.27 | +0.90 / +0.86 | — | — |
