"""批量对多组实验的模型目录运行未见事件评估。"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.experiments.run_unseen_event_experiments import run_experiments, resolve_event_dirs

DEFAULT_EVENT_ROOT = Path('/Users/lihe/Library/Mobile Documents/com~apple~CloudDocs/GNSS_EQDATA')

# E1.4 实验（模型目录使用 lm* 命名，参数与实验名对应）
E1_4_EXPERIMENTS = [
    ('full_lp150',   'outputs_experiments/e1_4/models/lm015_int'),
    ('full_lp200',   'outputs_experiments/e1_4/models/lm020_int'),
    ('full_lp300',   'outputs_experiments/e1_4/models/lm030_int'),
    ('farOnly_lp100','outputs_experiments/e1_4/models/lm010_noint'),
    ('farOnly_lp150','outputs_experiments/e1_4/models/lm015_noint'),
]

# E1.5 实验（输出到 outputs_experiments/e1_5/unseen/<exp_name>/）
E1_5_EXPERIMENTS = [
    ('ls010_simplified', 'outputs_experiments/e1_5/models/20260416_142216'),
    ('ls030_simplified', 'outputs_experiments/e1_5/models/20260416_144200'),
    ('ls050_simplified', 'outputs_experiments/e1_5/models/20260416_150233'),
    ('ls070_simplified', 'outputs_experiments/e1_5/models/20260416_151630'),
    ('ls100_simplified', 'outputs_experiments/e1_5/models/20260416_153015'),
    ('ls010_full',       'outputs_experiments/e1_5/models/20260416_155458'),
    ('ls030_full',       'outputs_experiments/e1_5/models/20260416_162150'),
    ('ls050_full',       'outputs_experiments/e1_5/models/20260416_164717'),
    ('ls070_full',       'outputs_experiments/e1_5/models/20260416_171247'),
    ('ls100_full',       'outputs_experiments/e1_5/models/20260416_173747'),
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--event-root', default=str(DEFAULT_EVENT_ROOT))
    parser.add_argument('--series', choices=['e1_4', 'e1_5', 'all'], default='all')
    args = parser.parse_args()

    batches: list[tuple[str, list]] = []
    if args.series in ('e1_4', 'all'):
        batches.append(('e1_4', E1_4_EXPERIMENTS))
    if args.series in ('e1_5', 'all'):
        batches.append(('e1_5', E1_5_EXPERIMENTS))

    total = sum(len(exps) for _, exps in batches)
    done = 0
    for series, experiments in batches:
        for exp_name, rel_model_dir in experiments:
            model_dir = PROJECT_ROOT / rel_model_dir
            results_root = PROJECT_ROOT / f'outputs_experiments/{series}/unseen/{exp_name}'
            print(f'\n[{done+1}/{total}] {series}/{exp_name}  model={model_dir.name}')
            run_experiments(
                model_dir=model_dir,
                event_root=args.event_root,
                results_root=results_root,
            )
            done += 1
    print(f'\n全部完成，共 {done} 组实验。')


if __name__ == '__main__':
    main()
