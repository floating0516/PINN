# USGS-Priority Magnitude Relabeling Design

**Date:** 2026-07-23
**Status:** Approved in conversation for specification and subsequent execution

## Objective

Replace the scalar catalog-magnitude target with a deterministic,
provenance-complete USGS-priority label while preserving scalar supervision for
every currently accepted event. Measure the effect of reference-label changes
against the frozen formal Model A results before retraining, then run one
controlled seed-42 pilot and, if its technical and scientific gates pass, a
formal seeds 17/42/73 campaign.

This work changes scalar labels only in its first experimental stage. It does
not change GNSS waveforms, station metadata, STF arrays, model architecture,
split assignments, loss weights, optimizer settings, or training duration.

## Preserved Evidence And Isolation

The current GCMT NPZ, formal checkpoints, split manifests, external-event
delivery, and all completed run directories remain immutable. New downloads,
manifests, candidate datasets, evaluations, and training runs use unique paths
and hashes. No command may overwrite an existing formal artifact.

The frozen comparison baselines are:

- formal three-seed station MAE mean: `0.0929788`;
- formal three-seed event-median MAE mean: `0.195958`;
- corrected eight-event ensemble MAE: `0.217309`;
- historical grouped-event, 2 cm catalog-scaled-STF MAE: `0.111319`.

Only like-for-like metrics with the same rows, split, aggregation, and
reference label may be compared directly. The historical grouped-event result
remains context, not a pass/fail baseline for the active station-random model.

## Alternatives Considered

Three scalar-label strategies were considered:

1. **Deterministic USGS-priority fallback:** use USGS where a valid moment
   magnitude is available, then fall back through GCMT and SCARDEC while
   recording the selected source. This is the approved approach.
2. **Average USGS and GCMT:** rejected because it creates a synthetic target
   that is not the result of any catalog or source inversion.
3. **Manual per-event best-source selection:** rejected as the default because
   it is subjective, difficult to reproduce, and vulnerable to choosing the
   label that makes model metrics look better.

Manual intervention is limited to resolving event identity. It does not change
the fixed source-priority rule.

## Scalar Magnitude Selection Contract

Every event resolves exactly one `mw_selected` through this ordered chain:

1. a finite USGS ComCat preferred magnitude whose case-insensitive type is one
   of `Mw`, `Mww`, `Mwc`, `Mwr`, or `Mwb`;
2. a finite scalar moment from the highest-preference usable ComCat
   moment-tensor product, converted to Mw and retaining the actual contributor
   network in provenance;
3. a finite GCMT Mw from the frozen catalog row;
4. Mw calculated from the integrated native SCARDEC STF moment.

Body-wave, surface-wave, local, and duration magnitudes such as `mb`, `Ms`,
`ML`, and `Md` are never substituted for Mw. Labels from multiple sources are
never averaged. If all four levels are unavailable, candidate-dataset
publication fails and requires a new cited source; the event is not silently
published without scalar supervision.

Scalar moment in N m is converted using the repository's existing convention:

```text
Mw = (2 / 3) * (log10(M0_Nm) - 9.1)
```

For each event the label manifest records at least:

- `event`, `event_index`, and `usgs_event_id`;
- `mw_selected`, `mw_source`, `mw_source_rank`, and `mw_type`;
- `mw_usgs`, `mw_gcmt`, and `mw_stf_native` when available;
- `product_id`, contributor network, scalar moment, and source update time;
- raw-response SHA-256 and selected-source SHA-256;
- pairwise source differences, warning state, and review state;
- match evidence and any explicit mapping-record identifier.

All stations belonging to one event receive bitwise-identical selected scalar
label and source metadata.

## Event Identity Matching

An existing explicit USGS event ID is authoritative after the returned origin
is checked against the local event. Events without an ID are queried by origin
time, coordinates, and magnitude.

An automatic match requires all of:

- absolute origin-time difference no greater than 30 seconds;
- epicentral separation no greater than 100 km;
- magnitude difference no greater than 0.5;
- exactly one acceptable candidate.

Matches outside these limits or with multiple acceptable candidates enter a
versioned explicit mapping table with recorded evidence. Event-name substring
matching is forbidden. Multiple local names may map to the same physical USGS
event only when the mapping table explicitly identifies the duplicate, as with
the historical Tohoku/Iwate naming case.

## Acquisition, Cache, And Replay

USGS responses are first downloaded to a new snapshot-specific raw cache under
the machine data root. Each response is validated as JSON, hashed, and retained
unchanged. Label resolution operates from the cache, so normal tests and
repeated audits do not depend on live network state.

Network requests use bounded retries for transient failures. A failed or
incomplete request may leave an explicitly incomplete cache entry, but it may
not publish a label manifest or candidate NPZ. Publication is atomic after all
required events pass the audit.

The frozen GCMT CSV and STF files are read-only fallback inputs. Their hashes
are recorded in the new snapshot manifest.

## Conflict And Review Rules

Every available pair among USGS, GCMT, and STF-native Mw is compared:

- absolute difference greater than `0.1 Mw`: record a warning;
- absolute difference at least `0.2 Mw`: set `review_required` and verify event
  identity before publication;
- a numerical disagreement alone never triggers averaging or a lower-priority
  source;
- the priority choice changes only if review proves that the higher-priority
  product belongs to the wrong physical event or is not a valid Mw-family
  product.

All `review_required` events must have an explicit review disposition in the
manifest before candidate-dataset publication.

## Candidate Dataset Contract

The relabeled NPZ is a new artifact beside the frozen GCMT NPZ. It retains the
same 40-event order and all original arrays. For loader compatibility, its
existing `magnitude` array contains the selected values and exactly equals a
new `magnitude_selected` array. It also adds event-aligned
`magnitude_usgs`, `magnitude_gcmt`, `magnitude_stf_native`,
`magnitude_source`, `magnitude_source_rank`, `magnitude_type`,
`usgs_event_id`, and `usgs_product_id` arrays. Missing candidate values use
NaN, while missing text provenance uses an empty string. Only `magnitude` and
these explicitly listed label/provenance arrays may differ from or be absent in
the frozen source NPZ.

Publication requires:

1. all 40 source NPZ events have one deterministic resolution record;
2. all 31 currently accepted formal events have finite scalar labels;
3. every accepted event has nonempty source and source-rank metadata;
4. same-event station labels and source metadata are invariant;
5. every non-label NPZ array is equal to the frozen source array;
6. event ordering, station containers, waveforms, STF inputs, and split keys are
   unchanged;
7. source responses, fallback inputs, mapping table, label manifest, and
   candidate NPZ have recorded SHA-256 values;
8. offline replay from the same cache produces the same selections and
   manifest hash.

## Relationship To STF Supervision

The scalar head trains against `mw_selected`. The STF head continues to consume
the existing station-aligned native STF target during the first experiment.
The native STF moment remains `mw_stf_native`, a secondary reference, and is
not silently rescaled to `mw_selected`.

This isolation is deliberate: the first experiment answers whether scalar
catalog labels caused the observed behavior. Changes to absolute STF-amplitude
or shape-only supervision require a separate approved experiment after the
scalar-label result is understood.

The untraceable Ibaraki and Parkfield STF files remain visible in provenance.
Their scalar labels do not depend on those files when a higher-priority USGS or
GCMT label exists. Noto's STF remains explicitly identified as a USGS
finite-fault moment-rate product.

## Zero-Cost Frozen-Model Comparison

Before retraining, recompute the current formal predictions against
`mw_selected` without changing any checkpoint or prediction. Produce paired
old-versus-new results for exactly the same station and event rows:

- station MAE, RMSE, and bias;
- event-median MAE, RMSE, and bias;
- per-event prediction error and label delta;
- metrics grouped by selected label source and source rank;
- the corrected eight-event metrics under the same resolver.

This separates reference-label sensitivity from training effects. These
results are diagnostics and do not alter the frozen historical registry.

## Seed-42 Pilot

The first retraining run uses seed 42 and the existing seed-42 station split for
all 200 epochs. It keeps the active Model A architecture, native STF targets,
optimizer, scheduler, losses, checkpoint metric, and configuration values
unchanged. The candidate scalar label is the only scientific variable.

The pilot must pass all existing finite-value, full-state checkpoint, strict
reload, signal-safe resume, split-hash, manifest-hash, and provenance checks.
Compare it with the frozen seed-42 model on the same split and the new selected
reference.

If pilot event-level `mw_selected` MAE is worse by more than `0.05 Mw` than the
frozen seed-42 predictions recomputed against the same labels, stop before the
formal three-seed campaign and investigate. This threshold blocks unexplained
degradation; it does not authorize reverting to weaker labels merely to improve
a metric. Station MAE remains secondary because the active within-event split
repeats event labels across stations.

## Formal Retraining And External Comparison

After a passing pilot, train seeds 17, 42, and 73 sequentially for 200 epochs in
a clean detached worktree and unique run directory. Reuse the frozen per-seed
station assignments. Do not tune architecture, loss weights, optimizer,
scheduler, threshold, or checkpoint-selection policy during this campaign.

After all internal artifacts validate, evaluate the same eight external events
once for all three seeds. The external events remain a sanity check and cannot
select checkpoints or trigger hyperparameter changes.

The final comparison presents old and relabeled results side by side using
matched definitions. It reports selected-reference metrics as primary and
USGS, GCMT, and STF-native metrics as source-aware secondary diagnostics.

## Test And Verification Gates

Implementation follows TDD and includes focused cases for:

1. every accepted USGS Mw-family preferred type;
2. non-Mw preferred magnitudes falling through to scalar moment;
3. GCMT fallback and SCARDEC-integral fallback;
4. complete source absence causing a hard publication failure;
5. explicit IDs, unique automatic matches, ambiguous matches, and duplicate
   physical-event mappings;
6. `0.1` warning and `0.2` review thresholds;
7. deterministic source selection independent of display names;
8. raw-cache replay without network access;
9. same-event label invariance and complete provenance fields;
10. equality of all non-label NPZ arrays;
11. deterministic manifest and candidate-artifact hashes;
12. paired zero-cost metric recomputation;
13. finite CPU and CUDA smoke before pilot or formal training.

Fresh focused tests, the complete regression suite, warnings-as-errors
compilation, data audits, and artifact hash verification must all pass before
the next phase starts.

## Failure Behavior

Any non-finite label, unsupported magnitude type, unresolved identity,
unreviewed conflict, unexpected non-label array change, hash mismatch,
incomplete source coverage, or non-finite training result stops the workflow.
Failures are preserved with reason-coded evidence in the unique run or snapshot
directory. No partial candidate dataset is promoted and no failed pilot starts
formal training.

## Approved Execution Boundary

The user approved this design in conversation on 2026-07-23 and authorized the
subsequent data download, old-result comparison, controlled pilot, and formal
retraining if the stated gates pass. The written specification remains subject
to the required user review before the implementation plan and execution begin.
