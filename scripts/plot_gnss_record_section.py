from pathlib import Path
import sys

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from scripts.build_gnss_event_npz import plot_gnss_event_examples
except ImportError:
    from build_gnss_event_npz import plot_gnss_event_examples


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yaml"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "figures"


def load_config(config_path: Path = CONFIG_PATH) -> dict:
    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)



def resolve_npz_path(config: dict) -> Path:
    data_path = config.get("paths", {}).get("data_path")
    if not data_path:
        raise ValueError("configs/config.yaml 中缺少 paths.data_path")
    return Path(data_path).expanduser().resolve()



def plot_record_section_waveforms(
    npz_path: Path | str,
    output_dir: Path | str,
    time_min_sec: float = 0.0,
    time_max_sec: float = 600.0,
    km_per_cm: float | None = None,
    scale_cm: float | None = None,
    file_suffix: str = "png",
) -> None:
    plot_gnss_event_examples(
        npz_path=npz_path,
        output_dir=output_dir,
        time_min_sec=time_min_sec,
        time_max_sec=time_max_sec,
        km_per_cm=km_per_cm,
        scale_cm=scale_cm,
        file_suffix=file_suffix,
    )



def main() -> None:
    config = load_config()
    npz_path = resolve_npz_path(config)
    output_dir = DEFAULT_OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    plot_record_section_waveforms(
        npz_path=npz_path,
        output_dir=output_dir,
        time_min_sec=0.0,
        time_max_sec=600.0,
        file_suffix="png",
    )

    print(f"已保存每个事件的三分量波形图到：{output_dir}")


if __name__ == "__main__":
    main()
