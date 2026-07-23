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
