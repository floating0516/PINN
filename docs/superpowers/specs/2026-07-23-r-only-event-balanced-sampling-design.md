# R-only Event-balanced Sampling Design

## Goal

Reduce domination by high-station-count earthquakes while preserving the verified R-only model, station-random split, and station-level checkpoint selection.

## Controlled Change

- Use the verified USGS-priority NPZ and the existing seed-specific within-event station splits.
- Train R-only seeds 17, 42, and 73 independently for 200 epochs.
- Change only `training.event_balanced_sampling` from `false` to `true`.
- Keep `training.checkpoint_metric: station_mae_catalog`; do not reuse the failed event-checkpoint selector.
- Keep the model, losses, optimizer, scheduler, accepted rows, STF labels, and evaluation aggregation unchanged.
- Do not introduce tempered weights, magnitude-bin weights, architecture changes, or external tuning.

## Rationale And Risk

Tohoku, Ibaraki, and Noto contribute 62.4% of the 1735 training station records but only 3 of 31 equal-weighted event scores. The existing weighted sampler gives each event equal expected total sampling weight while keeping 1735 draws per epoch.

Events with one training station can be sampled roughly 56 times more often than under natural station sampling. Keep station-level checkpoint selection and report low-magnitude, sparse-station, and station-level metrics to detect memorization.

## Decision

- Internal three-seed ensemble Event MAE baseline: `0.1609267`.
- Pass threshold: `<= 0.1509267`, an improvement of at least `0.01 Mw`.
- Also require no material regression in low-magnitude and at-most-three-test-station strata.
- Run the fixed external eight-event evaluation once only after the internal gate passes. External ensemble baseline: `0.2064468`.

## Verification

- Add one focused validator test accepting boolean event-balanced sampling while rejecting non-boolean values.
- Reuse the existing sampler weight and loader tests; do not add a broader test matrix.
- Validate that the formal YAML differs from the phase-9 R-only config only in `training.event_balanced_sampling`.
- Launch only from `pinn-run:train` under `systemd-inhibit --what=sleep --mode=block` in a detached run worktree.

## Outcome

The controlled campaign completed at commit `c9e36c2` in
`runs/phase13-r-event-balanced-20260723T160258Z-c9e36c2`.

| Seed | Station MAE | Event MAE |
|---:|---:|---:|
| 17 | 0.095250 | 0.176327 |
| 42 | 0.096669 | 0.137968 |
| 73 | 0.103685 | 0.175792 |

The internal three-seed ensemble Event MAE improved from `0.160927` to
`0.125224`, exceeding the frozen `0.150927` pass threshold. The `Mw < 7`
stratum improved from `0.299449` to `0.244039`, and events with at most three
test stations improved from `0.221843` to `0.171392`. Mean seed dispersion
also decreased from `0.112731` to `0.098376`. The tradeoff was a mean Station
MAE increase from `0.086590` to `0.098535`.

The one-time external sanity check produced seed Event MAEs of
`0.242825 / 0.196584 / 0.217170` and an ensemble Event MAE of `0.212720`.
This is a slight `0.006273 Mw` regression from the fixed `0.206447` R-only
baseline. The external result was not used for checkpoint selection or further
tuning.

The frozen internal gate passed, but the evidence is mixed across evaluation
roles. Retain event-balanced sampling as an event-prioritized R-only candidate;
do not replace the phase-9 model as the universal R-only baseline. All three
200-epoch logs, checkpoints, split hashes, and internal/external predictions
passed the final audit.
