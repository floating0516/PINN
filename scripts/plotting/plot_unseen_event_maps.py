from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.gnss_dataset_loader import plot_unseen_event_map_gmt
from src.evaluation.evaluate_unseen import _format_event_display_name, load_event_bundle

CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yaml"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "figures" / "unseen_event_maps"


def load_config(config_path: Path = CONFIG_PATH) -> dict:
    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_output_dir(config: dict | None = None) -> Path:
    if config is None:
        return DEFAULT_OUTPUT_DIR
    figures_dir = (config.get("paths", {}) or {}).get("figures_dir")
    if figures_dir:
        return Path(figures_dir).expanduser().resolve() / "unseen_event_maps"
    return DEFAULT_OUTPUT_DIR


def _safe_file_stem(label: str) -> str:
    return label.lower().replace(" ", "_").replace("/", "_")


def plot_single_event_map(
    *,
    event_dir: Path,
    output_dir: Path,
    relief_resolution: str | None,
    label_stations: bool,
) -> Path:
    bundle = load_event_bundle(event_dir)
    event_label = _format_event_display_name(
        event_name=bundle.event_name,
        event_dir_name=bundle.event_dir_name,
        magnitude=bundle.magnitude,
    )
    station_names = [station.station for station in bundle.stations]
    station_lons = [float(station.longitude) for station in bundle.stations]
    station_lats = [float(station.latitude) for station in bundle.stations]
    output_path = output_dir / f"{_safe_file_stem(event_label)}_map.png"
    return plot_unseen_event_map_gmt(
        event_name=event_label,
        output_path=output_path,
        event_lon=float(bundle.longitude),
        event_lat=float(bundle.latitude),
        depth_km=float(bundle.depth_km),
        magnitude=float(bundle.magnitude),
        station_lons=station_lons,
        station_lats=station_lats,
        station_names=station_names,
        strike=float(bundle.strike),
        dip=float(bundle.dip),
        rake=float(bundle.rake),
        mechanism_label=bundle.mechanism.strip() or None,
        label_stations=label_stations,
        relief_resolution=None,
        map_scale_km=100.0,
        inset_enabled=True,
        inset_position="jBR+w3.0c/2.1c+o0.18c/0.18c",
        station_name_font="7p,Helvetica-Bold,black",
        title_font="16p,Helvetica-Bold,black",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="使用 PyGMT 为未见地震事件绘制台站-震源分布图")
    parser.add_argument("--event-dir", action="append", required=True, help="单个未见事件目录，可重复传入多次")
    parser.add_argument("--output-dir", default=None, help="输出目录，默认 outputs/figures/unseen_event_maps")
    parser.add_argument("--relief-resolution", default="15s", help="PyGMT 地形分辨率，如 15s/30s；传 none 关闭地形")
    parser.add_argument("--no-label-stations", action="store_true", help="不标注台站名称")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    config = load_config()
    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else resolve_output_dir(config)
    output_dir.mkdir(parents=True, exist_ok=True)
    relief_resolution = None if str(args.relief_resolution).strip().lower() == "none" else args.relief_resolution

    saved_paths: list[Path] = []
    for event_dir in args.event_dir:
        out = plot_single_event_map(
            event_dir=Path(event_dir).expanduser().resolve(),
            output_dir=output_dir,
            relief_resolution=relief_resolution,
            label_stations=not args.no_label_stations,
        )
        saved_paths.append(out)
        print(f"已保存: {out}")

    print(f"共生成 {len(saved_paths)} 张事件地图，输出目录: {output_dir}")


if __name__ == "__main__":
    main()
