# R-only Event Checkpoint Design

## Goal

Align checkpoint selection with the event-level metric used to judge the final three-seed R-only ensemble.

## Controlled Change

- Use the existing USGS-priority NPZ, accepted rows, within-event station splits, R-only input, model, loss, optimizer, scheduler, and 200 epochs.
- Train seeds 17, 42, and 73 independently.
- Change only `training.checkpoint_metric` from `station_mae_catalog` to `event_mae_catalog`.
- Save each seed's checkpoint at its lowest validation Event MAE, then average the three event-median predictions for the internal ensemble.
- Do not enable event-balanced sampling, change the backbone or magnitude head, add waveform components, or tune against the external events.

## Baseline And Decision

- Internal three-seed ensemble Event MAE baseline: `0.1609267`.
- Treat `<= 0.1509267` as a clear internal improvement of at least `0.01 Mw`.
- Report per-seed station/event MAE plus ensemble errors for low-magnitude and sparse-station events.
- Run the fixed eight-event external evaluation once only if the internal gate passes. External R-only ensemble Event MAE baseline is `0.2064468`.

## Verification

- Add one focused configuration test proving that the active station workflow accepts `event_mae_catalog` while still rejecting unrelated checkpoint metrics.
- Run the focused config test, the small config test module, and one existing training checkpoint test.
- Validate that the formal config differs from the prior USGS config only in `training.checkpoint_metric`.
- Launch formal training only in `pinn-run:train` under `systemd-inhibit --what=sleep --mode=block` from a detached worktree at the committed experiment revision.
