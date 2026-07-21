from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.evaluate_unseen import evaluate_unseen_events
from scripts.plotting.plot_unseen_method_comparison import default_output_path, plot_method_comparison

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_DIR = PROJECT_ROOT / 'outputs' / 'models' / '20260326_212942'
DEFAULT_EVENT_ROOT = Path('/Users/lihe/Library/Mobile Documents/com~apple~CloudDocs/GNSS_EQDATA')
DEFAULT_RESULTS_ROOT = PROJECT_ROOT / 'outputs' / 'results'


def resolve_event_dirs(event_root: str | Path) -> list[Path]:
    root = Path(event_root)
    return [
        root / 'iquique-aftershock-2014-chile',
        root / 'nepal-aftershock-2015',
        root / 'kodiak-2018-alaska',
        root / 'samos-2020-greece',
        root / 'luding-2022-china',
        root / 'xizang-2025-southern-tibetan-plateau',
        root / 'myanmar-2025-mandalay',
        root / 'sand-point-2025-alaska',
    ]


def build_experiment_specs(results_root: str | Path) -> list[dict[str, object]]:
    root = Path(results_root)
    return [
        {
            'name': 'unseen_events_8_all_stations',
            'radial_peak_min_cm': 0.0,
            'output_dir': root / 'unseen_events_8_all_stations',
        },
        {
            'name': 'unseen_events_8_1cm',
            'radial_peak_min_cm': 1.0,
            'output_dir': root / 'unseen_events_8_1cm',
        },
        {
            'name': 'unseen_events_8_2cm',
            'radial_peak_min_cm': 2.0,
            'output_dir': root / 'unseen_events_8_2cm',
        },
    ]


def run_experiments(*, model_dir: str | Path, event_root: str | Path, results_root: str | Path) -> None:
    event_dirs = [str(path) for path in resolve_event_dirs(event_root)]
    for spec in build_experiment_specs(results_root):
        output_dir = Path(spec['output_dir'])
        result = evaluate_unseen_events(
            event_dirs=event_dirs,
            model_dir=str(model_dir),
            output_dir=str(output_dir),
            radial_peak_min_cm_override=float(spec['radial_peak_min_cm']),
        )
        plot_method_comparison(
            csv_path=result['event_csv'],
            output_path=default_output_path(result['event_csv']),
        )
        print(f"{spec['name']}: {len(result['station_rows'])} stations, {len(result['event_rows'])} events")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='批量运行 8 个未见事件的三套阈值实验并生成对比图')
    parser.add_argument('--model-dir', default=str(DEFAULT_MODEL_DIR), help='模型目录，需包含 best_model.pth 和 config.yaml')
    parser.add_argument('--event-root', default=str(DEFAULT_EVENT_ROOT), help='8 个未见事件目录的根路径')
    parser.add_argument('--results-root', default=str(DEFAULT_RESULTS_ROOT), help='结果输出根目录')
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run_experiments(
        model_dir=args.model_dir,
        event_root=args.event_root,
        results_root=args.results_root,
    )


if __name__ == '__main__':
    main()
