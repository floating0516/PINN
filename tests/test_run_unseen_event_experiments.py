import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.experiments.run_unseen_event_experiments import build_experiment_specs, resolve_event_dirs


ROOT = Path('/data/gnss-events')


def test_resolve_event_dirs_returns_eight_fixed_events():
    event_dirs = resolve_event_dirs(ROOT)

    assert len(event_dirs) == 8
    assert event_dirs[0] == ROOT / 'iquique-aftershock-2014-chile'
    assert event_dirs[1] == ROOT / 'nepal-aftershock-2015'
    assert event_dirs[2] == ROOT / 'kodiak-2018-alaska'
    assert event_dirs[3] == ROOT / 'samos-2020-greece'
    assert event_dirs[4] == ROOT / 'luding-2022-china'
    assert event_dirs[5] == ROOT / 'xizang-2025-southern-tibetan-plateau'
    assert event_dirs[6] == ROOT / 'myanmar-2025-mandalay'
    assert event_dirs[7] == ROOT / 'sand-point-2025-alaska'


def test_build_experiment_specs_returns_three_named_runs(tmp_path: Path):
    specs = build_experiment_specs(tmp_path)

    assert [spec['name'] for spec in specs] == [
        'unseen_events_8_all_stations',
        'unseen_events_8_1cm',
        'unseen_events_8_2cm',
    ]
    assert [spec['radial_peak_min_cm'] for spec in specs] == [0.0, 1.0, 2.0]
    assert specs[0]['output_dir'] == tmp_path / 'unseen_events_8_all_stations'
    assert specs[1]['output_dir'] == tmp_path / 'unseen_events_8_1cm'
    assert specs[2]['output_dir'] == tmp_path / 'unseen_events_8_2cm'


def test_script_runs_from_repo_root_without_module_import_errors():
    completed = subprocess.run(
        [sys.executable, 'scripts/experiments/run_unseen_event_experiments.py', '--help'],
        cwd=Path(__file__).resolve().parent.parent,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0
    assert '批量运行 8 个未见事件的三套阈值实验并生成对比图' in completed.stdout
