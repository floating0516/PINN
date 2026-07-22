import json
from pathlib import Path

from src.utils.provenance import (
    RUN_MANIFEST_FIELDS,
    current_git_commit,
    sha256_file,
    split_protocol_from_manifest,
    write_json,
)
from scripts.experiments.run_experiment import update_result_registry


def test_sha256_changes_when_file_changes(tmp_path: Path) -> None:
    path = tmp_path / "x.txt"
    path.write_text("a", encoding="utf-8")
    first = sha256_file(path)

    path.write_text("b", encoding="utf-8")

    assert sha256_file(path) != first


def test_run_manifest_field_contract_is_fixed() -> None:
    assert RUN_MANIFEST_FIELDS == (
        "pipeline_version",
        "git_commit",
        "git_dirty",
        "config_sha256",
        "dataset_manifest_sha256",
        "split_sha256",
        "checkpoint_sha256",
        "python_version",
        "torch_version",
        "numpy_version",
        "random_seed",
        "started_at_utc",
        "completed_at_utc",
    )


def test_current_git_commit_returns_head() -> None:
    root = Path(__file__).resolve().parents[1]

    commit = current_git_commit(root)

    assert len(commit) == 40
    assert all(character in "0123456789abcdef" for character in commit)


def test_write_json_is_stable_and_newline_terminated(tmp_path: Path) -> None:
    path = tmp_path / "result.json"

    write_json(path, {"b": 2, "a": 1})

    assert json.loads(path.read_text(encoding="utf-8")) == {"a": 1, "b": 2}
    assert path.read_bytes().endswith(b"\n")


def test_experiment_registry_collects_external_artifacts(tmp_path: Path) -> None:
    registry_path = tmp_path / "result_registry.json"
    station_csv = tmp_path / "station_predictions.csv"
    figure = tmp_path / "event_summary.png"
    station_csv.write_text("event,mw_pred\nA,7.0\n", encoding="utf-8")
    figure.write_bytes(b"png")
    write_json(registry_path, {"artifacts": {"csv": [], "figures": []}})

    result = update_result_registry(
        experiment_name="smoke",
        overrides=["training.epochs=1"],
        eval_result={"result_registry_path": registry_path},
        unseen_result={
            "metrics": {"event_mae": 0.1},
            "station_csv": station_csv,
            "event_summary_figure": figure,
        },
    )

    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    assert result == registry_path
    assert registry["experiment_name"] == "smoke"
    assert registry["external_evaluation"]["metrics"]["event_mae"] == 0.1
    assert str(station_csv) in registry["artifacts"]["csv"]
    assert str(figure) in registry["artifacts"]["figures"]


def test_saved_split_protocol_overrides_configured_protocol(
    tmp_path: Path,
) -> None:
    split_path = tmp_path / "split.json"
    write_json(split_path, {"protocol": "loeo"})

    protocol = split_protocol_from_manifest(
        split_path,
        configured_protocol="grouped_event",
    )

    assert protocol == "loeo"
