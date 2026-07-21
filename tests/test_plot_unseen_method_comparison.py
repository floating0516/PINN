import csv
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.plot_unseen_method_comparison import (
    default_output_path,
    load_event_summary_rows,
    plot_method_comparison,
)


def test_load_event_summary_rows_reads_pinn_and_pgd_columns(tmp_path: Path):
    csv_path = tmp_path / "event_summary.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "event",
                "mw_true",
                "mw_pred_median",
                "pgd_crowell_mw_pred_median",
                "pgd_ruhl_mw_pred_median",
                "pgd_melgar_mw_pred_median",
            ],
        )
        writer.writeheader()
        writer.writerows(
            [
                {
                    "event": "EventA",
                    "mw_true": 7.1,
                    "mw_pred_median": 7.0,
                    "pgd_crowell_mw_pred_median": 7.3,
                    "pgd_ruhl_mw_pred_median": 6.9,
                    "pgd_melgar_mw_pred_median": 7.2,
                },
                {
                    "event": "EventB",
                    "mw_true": 7.7,
                    "mw_pred_median": 7.5,
                    "pgd_crowell_mw_pred_median": 7.9,
                    "pgd_ruhl_mw_pred_median": 7.4,
                    "pgd_melgar_mw_pred_median": 7.6,
                },
            ]
        )

    rows = load_event_summary_rows(csv_path)

    assert [row["event"] for row in rows] == ["EventA", "EventB"]
    assert rows[0]["mw_true"] == pytest.approx(7.1)
    assert rows[0]["pinn"] == pytest.approx(7.0)
    assert rows[0]["crowell"] == pytest.approx(7.3)
    assert rows[0]["ruhl"] == pytest.approx(6.9)
    assert rows[0]["melgar"] == pytest.approx(7.2)
    assert rows[0]["pinn_error"] == pytest.approx(-0.1)
    assert rows[0]["crowell_error"] == pytest.approx(0.2)
    assert rows[0]["ruhl_error"] == pytest.approx(-0.2)
    assert rows[0]["melgar_error"] == pytest.approx(0.1)


def test_default_output_path_uses_same_directory_as_input_csv(tmp_path: Path):
    csv_path = tmp_path / "event_summary.csv"

    output_path = default_output_path(csv_path)

    assert output_path == tmp_path / "unseen_event_method_comparison.png"


def test_plot_method_comparison_creates_figure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    csv_path = tmp_path / "event_summary.csv"
    output_path = tmp_path / "comparison.png"
    legend_calls: list[dict[str, object]] = []
    title_calls: list[str] = []
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "event",
                "mw_true",
                "mw_pred_median",
                "pgd_crowell_mw_pred_median",
                "pgd_ruhl_mw_pred_median",
                "pgd_melgar_mw_pred_median",
            ],
        )
        writer.writeheader()
        writer.writerows(
            [
                {
                    "event": "EventA",
                    "mw_true": 7.1,
                    "mw_pred_median": 7.0,
                    "pgd_crowell_mw_pred_median": 7.3,
                    "pgd_ruhl_mw_pred_median": 6.9,
                    "pgd_melgar_mw_pred_median": 7.2,
                },
                {
                    "event": "EventB",
                    "mw_true": 7.7,
                    "mw_pred_median": 7.5,
                    "pgd_crowell_mw_pred_median": 7.9,
                    "pgd_ruhl_mw_pred_median": 7.4,
                    "pgd_melgar_mw_pred_median": 7.6,
                },
            ]
        )

    import matplotlib.axes

    original_legend = matplotlib.axes.Axes.legend
    original_set_title = matplotlib.axes.Axes.set_title

    def capture_legend(self, *args, **kwargs):
        legend_calls.append(kwargs)
        return original_legend(self, *args, **kwargs)

    def capture_set_title(self, label, *args, **kwargs):
        title_calls.append(label)
        return original_set_title(self, label, *args, **kwargs)

    monkeypatch.setattr(matplotlib.axes.Axes, "legend", capture_legend)
    monkeypatch.setattr(matplotlib.axes.Axes, "set_title", capture_set_title)

    saved = plot_method_comparison(csv_path=csv_path, output_path=output_path)

    assert saved == output_path
    assert output_path.exists()
    assert not output_path.with_suffix(".pdf").exists()
    assert legend_calls
    assert legend_calls[0]["loc"] == "upper right"
    assert not title_calls
