from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
from typing import Any, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.plotting.plot_phase39_expanded_fixed_split import (  # noqa: E402
    write_manifest,
)
from src.utils.provenance import sha256_file  # noqa: E402


DEFAULT_CAUSAL_RUN_ROOT = Path(
    "/home/lihe/PINN_Mag/runs/"
    "phase39-causal-expanded-fixed-test-20260831-v1"
)
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT / "docs/results/phase39-expanded-fixed-split"
)
EXPECTED_SPLIT_SHA256 = (
    "e4807aa1e6b5b389caf23974f62ff9da6b8add7f7887ec23a59d5d35a455eba7"
)
EXPECTED_CHECKPOINT_SHA256 = (
    "4a2540241fcbd9182edd05fb5fdb24641b1c836fa1ccd2ad12b101f91fa3e1ab"
)
SELECTED_EVENTS = (
    "Napa2014",
    "ak014cbigci8",
    "Ecuador2016",
    "Ridgecrest2019",
    "Tokachi2003",
    "us7000i9bw",
)
METHOD_ORDER = ("direct", "crowell", "ruhl", "melgar")
METHOD_LABELS = {
    "direct": "Causal Phase 39",
    "crowell": "Crowell 2013",
    "ruhl": "Ruhl 2019",
    "melgar": "Melgar 2015",
}
METHOD_COLORS = {
    "direct": "#D1495B",
    "crowell": "#4C78A8",
    "ruhl": "#59A14F",
    "melgar": "#B279A2",
}
METHOD_STYLES = {
    "direct": "-",
    "crowell": "-",
    "ruhl": "--",
    "melgar": "-.",
}
README_START = "<!-- causal-event-trajectories:start -->"
README_END = "<!-- causal-event-trajectories:end -->"
TRAJECTORY_FIGURE_PREFIX = (
    "figures/en/11_causal_event_magnitude_trajectories."
)


def configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.unicode_minus": False,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "savefig.facecolor": "white",
            "figure.facecolor": "white",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"JSON document must be an object: {path}")
    return value


def _hash_bytes(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_trajectories(
    causal_run_root: Path,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    summary = load_json(causal_run_root / "summary.json")
    if summary.get("status") != "complete":
        raise ValueError("causal test replay is incomplete")
    if summary.get("split_assignment_sha256") != EXPECTED_SPLIT_SHA256:
        raise ValueError("causal test split assignment changed")
    if summary.get("checkpoint_sha256") != EXPECTED_CHECKPOINT_SHA256:
        raise ValueError("causal checkpoint changed")
    event_path = Path(
        str(summary["artifacts"]["event_predictions_path"])
    ).resolve()
    if sha256_file(event_path) != summary["artifacts"][
        "event_predictions_sha256"
    ]:
        raise ValueError("causal event trajectory evidence hash changed")
    frame = pd.read_csv(event_path)
    selected = frame[frame["event"].isin(SELECTED_EVENTS)].copy()
    if set(selected["event"]) != set(SELECTED_EVENTS):
        raise ValueError("selected causal event coverage changed")
    if set(selected["method"]) != set(METHOD_ORDER):
        raise ValueError("selected causal method coverage changed")
    for event in SELECTED_EVENTS:
        event_rows = selected[selected["event"] == event]
        direct_horizons = set(
            event_rows.loc[
                event_rows["method"] == "direct",
                "observation_horizon_sec",
            ].astype(int)
        )
        if direct_horizons != set(range(1, 201)):
            raise ValueError(f"direct horizon coverage changed: {event}")
        for method in METHOD_ORDER[1:]:
            horizons = event_rows.loc[
                event_rows["method"] == method,
                "observation_horizon_sec",
            ].astype(int)
            if horizons.empty or int(horizons.max()) != 200:
                raise ValueError(f"PGD endpoint coverage changed: {event}/{method}")
    return selected, summary


def _shared_magnitude_limits(frame: pd.DataFrame) -> tuple[float, float]:
    values = np.concatenate(
        [
            frame["mw_pred_median"].to_numpy(dtype=float),
            frame["mw_catalog"].to_numpy(dtype=float),
        ]
    )
    minimum = float(np.nanmin(values))
    maximum = float(np.nanmax(values))
    padding = max(0.20, 0.05 * (maximum - minimum))
    return minimum - padding, maximum + padding


def plot_trajectories(
    frame: pd.DataFrame,
    output_stem: Path,
) -> list[Path]:
    configure_matplotlib()
    figure, axes = plt.subplots(
        3,
        2,
        figsize=(15.8, 13.2),
        sharex=True,
        sharey=True,
        constrained_layout=False,
    )
    figure.subplots_adjust(
        left=0.075,
        right=0.985,
        bottom=0.065,
        top=0.885,
        hspace=0.17,
        wspace=0.045,
    )
    y_min, y_max = _shared_magnitude_limits(frame)
    panel_letters = "ABCDEF"
    legend_handles = []
    legend_labels = []
    for panel_index, (axis, event) in enumerate(
        zip(axes.flat, SELECTED_EVENTS)
    ):
        event_rows = frame[frame["event"] == event]
        catalog = float(event_rows["mw_catalog"].median())
        endpoint_direct = event_rows[
            (event_rows["method"] == "direct")
            & (event_rows["observation_horizon_sec"] == 200)
        ]
        if len(endpoint_direct) != 1:
            raise ValueError(f"direct endpoint row changed: {event}")
        station_count = int(endpoint_direct.iloc[0]["n_stations"])
        for method in METHOD_ORDER:
            method_rows = event_rows[event_rows["method"] == method].sort_values(
                "observation_horizon_sec"
            )
            line = axis.plot(
                method_rows["observation_horizon_sec"],
                method_rows["mw_pred_median"],
                color=METHOD_COLORS[method],
                linestyle=METHOD_STYLES[method],
                linewidth=2.25 if method == "direct" else 1.55,
                alpha=1.0 if method == "direct" else 0.90,
                label=METHOD_LABELS[method],
                zorder=4 if method == "direct" else 3,
            )[0]
            endpoint = method_rows[
                method_rows["observation_horizon_sec"] == 200
            ]
            if len(endpoint) == 1:
                axis.scatter(
                    [200],
                    [float(endpoint.iloc[0]["mw_pred_median"])],
                    color=METHOD_COLORS[method],
                    s=22 if method == "direct" else 15,
                    zorder=5,
                )
            if panel_index == 0:
                legend_handles.append(line)
                legend_labels.append(METHOD_LABELS[method])
        catalog_line = axis.axhline(
            catalog,
            color="#202124",
            linewidth=1.2,
            linestyle=":",
            label="Catalog Mw",
            zorder=2,
        )
        if panel_index == 0:
            legend_handles.append(catalog_line)
            legend_labels.append("Catalog Mw")
        axis.set_xlim(1, 200)
        axis.set_ylim(y_min, y_max)
        axis.set_xticks([1, 50, 100, 150, 200])
        axis.grid(True, color="#D7DCE2", linewidth=0.55, alpha=0.75)
        axis.set_title(
            f"{panel_letters[panel_index]}. {event} | catalog Mw {catalog:.2f} | "
            f"{station_count} stations",
            loc="left",
            fontweight="bold",
            fontsize=10.5,
        )
        if panel_index // 2 == 2:
            axis.set_xlabel("Observed causal prefix (s)")
        if panel_index % 2 == 0:
            axis.set_ylabel("Estimated magnitude (Mw)")
    figure.legend(
        legend_handles,
        legend_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.955),
        ncol=5,
        frameon=False,
        fontsize=9.5,
    )
    figure.suptitle(
        "Event-wise causal magnitude evolution: Phase 39 and empirical PGD methods",
        fontsize=14,
        fontweight="bold",
        y=0.985,
    )
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    generated = []
    for suffix, dpi in ((".png", 220), (".pdf", 300)):
        path = output_stem.with_suffix(suffix)
        metadata = (
            {"CreationDate": None, "ModDate": None}
            if suffix == ".pdf"
            else None
        )
        figure.savefig(path, dpi=dpi, bbox_inches="tight", metadata=metadata)
        generated.append(path)
    plt.close(figure)
    return generated


def write_selected_evidence(
    frame: pd.DataFrame,
    causal_summary: dict[str, Any],
    output_dir: Path,
) -> list[Path]:
    analysis_dir = output_dir / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    trajectory_path = analysis_dir / "causal_selected_event_trajectories.csv"
    frame.sort_values(
        ["event", "method", "observation_horizon_sec"]
    ).to_csv(trajectory_path, index=False, lineterminator="\n")
    summary_path = analysis_dir / "causal_test_summary.json"
    summary_path.write_text(
        json.dumps(
            causal_summary,
            ensure_ascii=False,
            allow_nan=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return [trajectory_path, summary_path]


def update_readme(output_dir: Path) -> None:
    readme_path = output_dir / "README.md"
    content = readme_path.read_text(encoding="utf-8")
    block = f"""{README_START}
## Causal Event Magnitude Trajectories

![Causal event magnitude trajectories](figures/en/11_causal_event_magnitude_trajectories.png)

[PDF](figures/en/11_causal_event_magnitude_trajectories.pdf) |
[Selected event trajectory data](analysis/causal_selected_event_trajectories.csv) |
[Causal test summary](analysis/causal_test_summary.json)

This separate follow-up experiment uses a validation-frozen causal Phase 39
checkpoint and reports event-median estimates at every observed prefix from
1 to 200 seconds for Phase 39, with empirical PGD estimates shown wherever PGD
is defined. Each panel compares causal Phase 39 with the cumulative-PGD Crowell
2013, Ruhl 2019, and Melgar 2015 relations; the dotted horizontal line is the
catalog magnitude. The selected events include the two dominant failures,
sparse and dense station networks, and both moderate and large earthquakes.
Figures 1-10 above remain the original endpoint experiment.

该图为单独的逐秒因果前缀实验。每个子图同时绘制因果 Phase 39、Crowell、
Ruhl 和 Melgar 四条震级估计曲线；经验 PGD 在存在有效值的时刻绘制，水平
虚线为目录震级。前面的图 1-10 仍然对应原始 200 秒终点实验，数值和图件
均未改变。
{README_END}

"""
    if README_START in content:
        start = content.index(README_START)
        end = content.index(README_END, start) + len(README_END)
        content = content[:start] + block.rstrip() + content[end:]
    else:
        marker = "## Supplementary Protocol Diagnostics"
        if marker not in content:
            raise ValueError("README insertion marker changed")
        content = content.replace(marker, block + marker, 1)
    readme_path.write_text(content, encoding="utf-8")


def generate(
    *,
    causal_run_root: Path,
    output_dir: Path,
) -> dict[str, Any]:
    old_manifest = load_json(output_dir / "publication_manifest.json")
    preserved_figure_hashes = {
        str(row["path"]): str(row["sha256"])
        for row in old_manifest["files"]
        if str(row["path"]).startswith("figures/")
        and not str(row["path"]).startswith(TRAJECTORY_FIGURE_PREFIX)
    }
    frame, causal_summary = load_trajectories(causal_run_root)
    figure_paths = plot_trajectories(
        frame,
        output_dir
        / "figures/en/11_causal_event_magnitude_trajectories",
    )
    evidence_paths = write_selected_evidence(
        frame,
        causal_summary,
        output_dir,
    )
    update_readme(output_dir)
    for relative, expected_hash in preserved_figure_hashes.items():
        path = output_dir / relative
        if not path.is_file() or _hash_bytes(path) != expected_hash:
            raise ValueError(f"existing figure changed: {relative}")
    publication_summary = load_json(output_dir / "summary.json")
    manifest = write_manifest(output_dir, publication_summary)
    return {
        "status": "complete",
        "selected_events": list(SELECTED_EVENTS),
        "figure_paths": [str(path) for path in figure_paths],
        "evidence_paths": [str(path) for path in evidence_paths],
        "preserved_figure_count": len(preserved_figure_hashes),
        "publication_file_count": int(manifest["file_count"]),
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Add one event-wise causal magnitude trajectory figure without "
            "regenerating existing Phase 39 publication figures."
        )
    )
    parser.add_argument(
        "--causal-run-root",
        type=Path,
        default=DEFAULT_CAUSAL_RUN_ROOT,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    result = generate(
        causal_run_root=args.causal_run_root.resolve(),
        output_dir=args.output_dir.resolve(),
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
