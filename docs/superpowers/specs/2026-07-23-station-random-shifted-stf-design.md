# Station-Random, Shifted-STF Training Design

**Date:** 2026-07-23
**Status:** Approved in conversation for the simplified constant-velocity stage

## Objective

Restore the original within-event station inference task while retaining the
corrected data, geometry, provenance, and checkpoint reliability work. Training
uses one protocol only: station samples are randomly divided within each event.
The STF training target is shifted into each station's P-arrival time frame.

The eight delivered external events are diagnostic only. They do not select a
checkpoint, rank models, or act as a formal independent test set.

## Preserved Evidence

The completed grouped-event Task 18 campaign and all earlier artifacts remain
immutable. They are historical diagnostics and are not inputs to the revised
model-selection decision.

## Data Split

Use deterministic within-event station splitting with seed-controlled shuffling:

- train: 70 percent of each event's stations;
- validation: 15 percent;
- test: 15 percent;
- at least one station remains in train for every event;
- a station key may occur in exactly one split;
- year is not a balancing variable.

On the accepted 2,483-record dataset, seed 42 currently yields 1,735 train,
374 validation, and 374 test records. All 31 events occur in train. Thirty
events occur in validation and test; Puebla2017 has one station and therefore
remains train-only. Station-weighted catalog-Mw means are 8.0469, 8.0377, and
8.0377 respectively.

The split manifest records per-event station counts, catalog-Mw distributions,
seed, sample keys, and hashes. Split validation fails on duplication, missing
train coverage, a total split fraction more than 0.01 from 0.70/0.15/0.15, or
a station-weighted mean catalog Mw more than 0.05 from the full accepted
dataset mean. Events with at least three accepted stations must contribute at
least one station to each split.

Grouped-event and LOEO protocols are removed from the active training workflow.
Their code may remain readable for historical artifact compatibility, but no
new run may select them.

## Distance and Travel Time

Use three-dimensional source distance throughout the active pipeline. Preserve
epicentral distance only as explicit provenance and for empirical laws that
require it.

The simplified travel-time provider is the only source of P/S delays:

```text
tau_P = source_distance_m / alpha
tau_S = source_distance_m / beta
alpha = 7900 m/s
beta  = 4533 m/s
```

The constants come from the existing physics configuration and replace the
separate legacy `p_velocity_mps=4500` path. The provider records its mode and
velocities in every config snapshot and run manifest.

CRUST1.0 is a deferred provider. When the user supplies its data, its layer
schema, units, interpolation, mantle treatment, and ray/travel-time algorithm
must be reviewed before it can replace constant velocity.

## Station-Aligned STF Target

Retain the canonical 200-second source STF for provenance and moment audits.
Derive the training target independently for each station:

```text
station_rate(t) = source_rate(t - tau_P)
```

Fractional shifts use interpolation on the frozen one-second target grid. The
shifted discrete integral must equal the canonical target moment within the
configured numerical tolerance. No station-dependent amplitude rescaling is
allowed except the single correction required to remove interpolation error.

A 200-second station target is too short: the accepted data have source
distances up to 799.812 km, P delays up to 101.242 seconds, and 35 records fall
below the 99.5-percent retained-moment gate after a 200-second shift. The
station target therefore contains the 200-second canonical source window plus
the rounded maximum P-delay margin. For the current audited dataset this is a
302-step target. Preflight recomputes and freezes the required length before a
run; it fails rather than truncating or silently renormalizing a short window.

## Physics Alignment

The model predicts the station-aligned STF. Propagation delays are not applied
twice:

- the P contribution has zero relative delay because the target is P-aligned;
- the S contribution uses `tau_S - tau_P`;
- the same relative timing applies to far- and intermediate-field terms;
- synthesis loss compares only the observed waveform interval and its mask;
- predicted Mw integrates the complete shifted STF target window.

All shift and synthesis paths consume the same travel-time provider.

## Model Candidates

Model A is the existing TCN+Transformer hybrid `PINNModel` with metadata.

Model B is the user-referenced `Cross1` model. No matching identifier or source
currently exists in the repository, staging tree, configs, or Git history.
Model B remains an explicit input gate until the user supplies its exact name,
configuration, code, or artifact. No existing backbone may be relabeled as
`Cross1` by inference.

Common pipeline work may proceed before Model B is identified. Formal
two-model comparison may not start until the gate is cleared.

## Training Control

For the first revised comparison:

- freeze one split manifest per seed and use the same three manifests for every
  model;
- run seeds 17, 42, and 73;
- train every run for all 200 epochs;
- disable early termination;
- restore explicit `warmup_epochs=5`, `scheduler_T0=15`, and
  `scheduler_T_mult=2`;
- save the checkpoint with the lowest validation station-level catalog-Mw MAE;
- preserve full-state epoch checkpoints and signal-safe resume.

This prevents a run from ending at epoch 51-55 while still allowing the best
validation checkpoint to occur at any epoch.

## Metrics and Selection

Primary internal metrics are station-level catalog-Mw MAE, RMSE, and bias.
Validation station MAE selects each run's checkpoint. Candidate models rank by
their mean validation station MAE across the three required seeds; mean
absolute validation bias and then parameter count break an exact tie. The
locked test splits are evaluated only after the model identity is frozen, and
test metrics cannot change that selection.

Secondary diagnostics include event-median MAE, per-event station IQR/standard
deviation, sample counts, displacement thresholds, STF loss components, and
waveform synthesis error. Secondary metrics do not override the primary
station-level rule.

The eight external events run once after internal selection as a sanity check.
Their results are preserved, but they cannot alter the selected checkpoint,
model, loss weights, or pass/fail decision.

## Verification Gates

Implementation must establish focused failing tests before each behavior
change, then verify:

1. deterministic within-event split counts and no station overlap;
2. per-split station-count and magnitude-distribution audits;
3. one authoritative constant-velocity travel-time provider;
4. fractional P shifting and exact station-target shape;
5. shifted discrete moment conservation and 99.5-percent window retention;
6. P-zero/S-relative delay behavior with no double shift;
7. independent 200-step observation and 302-step station-target lengths;
8. station-level checkpoint selection with no early termination;
9. full checkpoint/resume and signal equivalence;
10. finite CPU and single-GPU smoke results before any formal run.

## Deferred Inputs

- CRUST1.0 data and its required travel-time interpretation;
- exact identity or artifact for the `Cross1` model;
- explicit user approval of the implementation plan derived from this design.
