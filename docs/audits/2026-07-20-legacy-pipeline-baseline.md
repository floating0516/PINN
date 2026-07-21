# Lightweight Legacy Pipeline Baseline

Audit date: 2026-07-22

## Verifiable Import Evidence

- Lightweight import commit: `42739451a8a45a53bdb0bcc282323734568fa57e`
- Import branch: `import/light-package-20260721`
- Approved archive: `PINN_Mag-train-20260721.tar.gz`
- Archive SHA-256: `65fe554e7ef1db59ada83d685b50ff895f5644beed4bf31eb431895639b55765`
- Imported legacy config: `configs/legacy/config_legacy_2026-07-20.yaml`
- Legacy config SHA-256: `a3fbb47f49008d9e325c923412e6a07ad7b66378865f492c81d9f54bbeac58fb`
- Training NPZ SHA-256: `46408b88d9727e90031988077283a596771102251e7afd75e6136cce60367f75`

The frozen legacy config is a byte-for-byte copy of the package's
`configs/config.yaml`. It intentionally retains the original workstation paths
and must not be used as a server runtime config.

## Historical Artifacts Not Supplied

The uploaded remediation plan names the following historical evidence, but none
of it is present in the lightweight archive or imported repository:

- original Git commit `e9955220a407126956a4e77c60bd064fb3afeb4e`;
- `outputs_experiments/e1_4/models/lm010_noint/best_model.pth`;
- `outputs_experiments/e1_4/models/lm010_noint/config.yaml`;
- `outputs/results/test_set_predictions_far_only.csv`;
- historical output directories and the paper source tree.

Their hashes, exact metrics, and executable relationship to the imported code
cannot be verified. Do not replace them with newly trained weights or generated
placeholder results. Task 17 remains gated on receiving the original checkpoint,
config, and prediction CSV.

## Legacy Result Semantics

The remediation plan describes the unavailable headline metric as a
within-event held-out-station MAE against station-cropped STF-derived magnitude.
It also classifies the eight external events as development-time validation, not
as a blind test. These labels are historical claims from the plan, not metrics
recomputed from the lightweight package.

## Known Invalid Legacy Assumptions

1. Reference STF was shifted by `distance / 4500` and then cropped per station.
2. The same source event could therefore receive station-dependent STF and Mw labels.
3. External-event azimuth could be passed as takeoff angle while geographic azimuth was zeroed.
4. Standard random splits allowed the same event to occur across train, validation, and test.
5. Propagation delay and distance fields were ambiguous across data, forward, and PGD paths.
6. The legacy trainer saved model weights only, without optimizer, scheduler, early-stop, or RNG state.

## Isolation Boundary

All corrected work is performed in the linked development worktree on
`remediation/corrected-pipeline`. The administrative import checkout remains on
the baseline branch. A future formal remote must receive only post-baseline
remediation commits by cherry-pick; unrelated histories must not be merged.
