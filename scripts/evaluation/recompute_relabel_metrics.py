from __future__ import annotations

import argparse
from collections import defaultdict
from collections.abc import Mapping, Sequence
import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any

import numpy as np

SEEDS = (17, 42, 73)


def _finite_float(value: Any, *, field: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{field} is not numeric: {value!r}") from error
    if not math.isfinite(result):
        raise ValueError(f"{field} is not finite: {value!r}")
    return result


def _label_index(
    label_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Mapping[str, Any]]:
    result: dict[str, Mapping[str, Any]] = {}
    for row in label_rows:
        event = str(row.get("event", ""))
        if not event:
            raise ValueError("label row has an empty event")
        if event in result:
            raise ValueError(f"duplicate label event: {event}")
        result[event] = row
    return result


def pair_prediction_rows(
    prediction_rows: Sequence[Mapping[str, Any]],
    label_rows: Sequence[Mapping[str, Any]],
    *,
    prediction_key: str,
    old_reference_key: str,
) -> list[dict[str, Any]]:
    labels = _label_index(label_rows)
    missing = sorted(
        {
            str(row.get("event", ""))
            for row in prediction_rows
            if str(row.get("event", "")) not in labels
        }
    )
    if missing:
        raise ValueError(f"missing selected labels for prediction events: {missing}")

    paired: list[dict[str, Any]] = []
    for source_row in prediction_rows:
        event = str(source_row["event"])
        label = labels[event]
        prediction = _finite_float(
            source_row.get(prediction_key),
            field=prediction_key,
        )
        old_reference = _finite_float(
            source_row.get(old_reference_key),
            field=old_reference_key,
        )
        selected_reference = _finite_float(
            label.get("mw_selected"),
            field="mw_selected",
        )
        row = dict(source_row)
        row.update(
            {
                "mw_old": old_reference,
                "mw_selected": selected_reference,
                "label_delta": selected_reference - old_reference,
                "error_old": prediction - old_reference,
                "error_selected": prediction - selected_reference,
                "mw_source": str(label.get("mw_source", "")),
                "mw_source_rank": int(label.get("mw_source_rank", 0)),
                "mw_type": str(label.get("mw_type", "")),
                "usgs_event_id": str(label.get("usgs_event_id", "")),
            }
        )
        if not row["mw_source"] or row["mw_source_rank"] not in {1, 2, 3, 4}:
            raise ValueError(f"label provenance is incomplete for {event}")
        paired.append(row)
    return paired


def _metrics(
    rows: Sequence[Mapping[str, Any]],
    *,
    prediction_key: str,
    reference_key: str,
) -> dict[str, Any]:
    errors = np.asarray(
        [
            _finite_float(row[prediction_key], field=prediction_key)
            - _finite_float(row[reference_key], field=reference_key)
            for row in rows
        ],
        dtype=np.float64,
    )
    if errors.size == 0:
        raise ValueError("cannot summarize an empty prediction table")
    return {
        "count": int(errors.size),
        "mae": float(np.mean(np.abs(errors))),
        "rmse": float(np.sqrt(np.mean(errors**2))),
        "bias": float(np.mean(errors)),
    }


def summarize_paired_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    prediction_key: str,
) -> dict[str, Any]:
    def summarize(group: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        return {
            "count": len(group),
            "old": _metrics(
                group,
                prediction_key=prediction_key,
                reference_key="mw_old",
            ),
            "selected": _metrics(
                group,
                prediction_key=prediction_key,
                reference_key="mw_selected",
            ),
        }

    by_source: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    by_source_rank: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        by_source[str(row["mw_source"])].append(row)
        by_source_rank[str(int(row["mw_source_rank"]))].append(row)
    overall = summarize(rows)
    return {
        "old": overall["old"],
        "selected": overall["selected"],
        "by_source": {
            key: summarize(group) for key, group in sorted(by_source.items())
        },
        "by_source_rank": {
            key: summarize(group)
            for key, group in sorted(by_source_rank.items())
        },
    }


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def _read_label_manifest(path: Path) -> list[dict[str, Any]]:
    if path.suffix.casefold() == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            raise ValueError(f"label manifest must contain a list: {path}")
        return [dict(row) for row in payload]
    return [dict(row) for row in _read_csv(path)]


def _single_path(paths: Sequence[Path], *, description: str) -> Path:
    if len(paths) != 1:
        raise ValueError(
            f"expected one {description}, found {len(paths)}: "
            f"{[str(path) for path in paths]}"
        )
    return paths[0]


def _formal_paths(run_root: Path, seed: int) -> tuple[Path, Path]:
    base = run_root / "campaign"
    station = _single_path(
        sorted(base.glob(f"*/seed_{seed}/results/*/station_predictions.csv")),
        description=f"formal station prediction table for seed {seed}",
    )
    event = _single_path(
        sorted(base.glob(f"*/seed_{seed}/results/*/event_summary.csv")),
        description=f"formal event prediction table for seed {seed}",
    )
    return station, event


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json_bytes(payload: Any) -> bytes:
    return (
        json.dumps(
            payload,
            ensure_ascii=False,
            allow_nan=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def _csv_bytes(rows: Sequence[Mapping[str, Any]]) -> bytes:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(
        stream,
        fieldnames=fields,
        lineterminator="\n",
    )
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode("utf-8")


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite comparison output: {path}")
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent,
            prefix=f".{path.name}.tmp-",
            delete=False,
        ) as temporary:
            temporary.write(payload)
            temporary.flush()
            os.fsync(temporary.fileno())
            temporary_path = Path(temporary.name)
        os.replace(temporary_path, path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _mean_seed_metrics(
    seed_summaries: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for level in ("station", "event"):
        result[level] = {}
        for reference in ("old", "selected"):
            result[level][reference] = {
                metric: float(
                    np.mean(
                        [
                            float(summary[level][reference][metric])
                            for summary in seed_summaries.values()
                        ]
                    )
                )
                for metric in ("mae", "rmse", "bias")
            }
    return result


def _ensemble_rows(
    event_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in event_rows:
        grouped[str(row["event"])].append(row)
    result: list[dict[str, Any]] = []
    for event in sorted(grouped):
        rows = grouped[event]
        seeds = {int(row["seed"]) for row in rows}
        if seeds != set(SEEDS) or len(rows) != len(SEEDS):
            raise ValueError(f"external event lacks exactly three seeds: {event}")
        old_values = {_finite_float(row["mw_old"], field="mw_old") for row in rows}
        selected_values = {
            _finite_float(row["mw_selected"], field="mw_selected")
            for row in rows
        }
        sources = {str(row["mw_source"]) for row in rows}
        ranks = {int(row["mw_source_rank"]) for row in rows}
        if (
            len(old_values) != 1
            or len(selected_values) != 1
            or len(sources) != 1
            or len(ranks) != 1
        ):
            raise ValueError(f"external label is not seed-invariant: {event}")
        result.append(
            {
                "event": event,
                "mw_pred_ensemble": float(
                    np.mean(
                        [
                            _finite_float(
                                row["mw_pred_median"],
                                field="mw_pred_median",
                            )
                            for row in rows
                        ]
                    )
                ),
                "mw_old": old_values.pop(),
                "mw_selected": selected_values.pop(),
                "mw_source": sources.pop(),
                "mw_source_rank": ranks.pop(),
                "seed_count": len(rows),
            }
        )
    return result


def recompute_comparison(
    *,
    label_manifest: str | Path,
    external_label_manifest: str | Path,
    formal_run: str | Path,
    external_run: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    label_path = Path(label_manifest).resolve()
    external_label_path = Path(external_label_manifest).resolve()
    formal_root = Path(formal_run).resolve()
    external_root = Path(external_run).resolve()
    output = Path(output_dir).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"comparison output is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)

    labels = _read_label_manifest(label_path)
    external_labels = _read_label_manifest(external_label_path)
    formal_station_rows: list[dict[str, Any]] = []
    formal_event_rows: list[dict[str, Any]] = []
    seed_summaries: dict[str, Any] = {}
    input_hashes = {
        str(label_path): _sha256(label_path),
        str(external_label_path): _sha256(external_label_path),
    }
    for seed in SEEDS:
        station_path, event_path = _formal_paths(formal_root, seed)
        input_hashes[str(station_path)] = _sha256(station_path)
        input_hashes[str(event_path)] = _sha256(event_path)
        stations = pair_prediction_rows(
            _read_csv(station_path),
            labels,
            prediction_key="mw_pred",
            old_reference_key="mw_catalog",
        )
        events = pair_prediction_rows(
            _read_csv(event_path),
            labels,
            prediction_key="mw_pred_median",
            old_reference_key="mw_catalog",
        )
        station_with_seed = [{"seed": seed, **row} for row in stations]
        event_with_seed = [{"seed": seed, **row} for row in events]
        formal_station_rows.extend(station_with_seed)
        formal_event_rows.extend(event_with_seed)
        seed_summaries[str(seed)] = {
            "station": summarize_paired_rows(
                station_with_seed,
                prediction_key="mw_pred",
            ),
            "event": summarize_paired_rows(
                event_with_seed,
                prediction_key="mw_pred_median",
            ),
            "station_count": len(station_with_seed),
            "event_count": len(event_with_seed),
        }

    external_station_path = external_root / "station_predictions_all_seeds.csv"
    external_event_path = external_root / "event_predictions_all_seeds.csv"
    input_hashes[str(external_station_path)] = _sha256(external_station_path)
    input_hashes[str(external_event_path)] = _sha256(external_event_path)
    external_station_rows = pair_prediction_rows(
        _read_csv(external_station_path),
        external_labels,
        prediction_key="mw_pred",
        old_reference_key="mw_catalog",
    )
    external_event_rows = pair_prediction_rows(
        _read_csv(external_event_path),
        external_labels,
        prediction_key="mw_pred_median",
        old_reference_key="mw_catalog",
    )
    external_seed_summaries: dict[str, Any] = {}
    for seed in SEEDS:
        station_seed = [
            row for row in external_station_rows if int(row["seed"]) == seed
        ]
        event_seed = [
            row for row in external_event_rows if int(row["seed"]) == seed
        ]
        external_seed_summaries[str(seed)] = {
            "station": summarize_paired_rows(
                station_seed,
                prediction_key="mw_pred",
            ),
            "event": summarize_paired_rows(
                event_seed,
                prediction_key="mw_pred_median",
            ),
            "station_count": len(station_seed),
            "event_count": len(event_seed),
        }
    ensemble_rows = _ensemble_rows(external_event_rows)
    ensemble_summary = summarize_paired_rows(
        ensemble_rows,
        prediction_key="mw_pred_ensemble",
    )

    summary = {
        "formal": {
            "seeds": seed_summaries,
            "three_seed_mean": _mean_seed_metrics(seed_summaries),
            "seed_42_selected_event_mae": seed_summaries["42"]["event"][
                "selected"
            ]["mae"],
        },
        "external": {
            "seeds": external_seed_summaries,
            "three_seed_mean": _mean_seed_metrics(external_seed_summaries),
            "ensemble_event": ensemble_summary,
        },
        "input_sha256": dict(sorted(input_hashes.items())),
    }
    outputs = {
        "formal_station_paired.csv": _csv_bytes(formal_station_rows),
        "formal_event_paired.csv": _csv_bytes(formal_event_rows),
        "external_station_paired.csv": _csv_bytes(external_station_rows),
        "external_event_paired.csv": _csv_bytes(external_event_rows),
        "external_ensemble_paired.csv": _csv_bytes(ensemble_rows),
        "summary.json": _json_bytes(summary),
    }
    for name, payload in outputs.items():
        _atomic_write(output / name, payload)
    return summary


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Recompute frozen predictions against selected magnitudes",
    )
    parser.add_argument("--label-manifest", required=True, type=Path)
    parser.add_argument("--external-label-manifest", required=True, type=Path)
    parser.add_argument("--formal-run", required=True, type=Path)
    parser.add_argument("--external-run", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    summary = recompute_comparison(
        label_manifest=args.label_manifest,
        external_label_manifest=args.external_label_manifest,
        formal_run=args.formal_run,
        external_run=args.external_run,
        output_dir=args.output_dir,
    )
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
