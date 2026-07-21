import copy
import csv
import json
import sys
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from src.evaluation.evaluate import evaluate
from src.training.train import train
from src.utils.run_dirs import make_run_id

CONFIG_PATH = ROOT / 'configs' / 'config.yaml'
RESULTS_ROOT = ROOT / 'outputs' / 'results'

LAMBDA_MAG_VALUES = [0.3, 0.4, 0.5, 0.6]


def load_config() -> dict[str, Any]:
    with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def build_trial_config(base_config: dict[str, Any], lambda_mag: float, seed: int) -> dict[str, Any]:
    config = copy.deepcopy(base_config)
    training_cfg = config.setdefault('training', {})
    stf_rate_cfg = training_cfg.setdefault('stf_rate_loss', {})
    training_cfg['loss_name'] = 'stf_rate'
    training_cfg['random_seed'] = int(seed)
    stf_rate_cfg['lambda_mag'] = float(lambda_mag)
    return config


def write_csv_rows(csv_path: Path, rows: list[dict[str, Any]]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        'trial_id',
        'lambda_mag',
        'random_seed',
        'run_id',
        'model_dir',
        'best_model_path',
        'best_model_swa_path',
        'config_snapshot_path',
        'training_log_path',
        'eval_results_dir',
        'eval_results_run_id',
        'test_mae',
        'test_rmse',
        'baseline_mae',
        'baseline_rmse',
        'sample_count',
        'status',
    ]
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def normalize_record(record: dict[str, Any]) -> dict[str, Any]:
    normalized: dict[str, Any] = {}
    for key, value in record.items():
        if isinstance(value, Path):
            normalized[key] = str(value)
        else:
            normalized[key] = value
    return normalized


def main() -> None:
    base_config = load_config()
    search_id = f'hparam_search_{make_run_id()}'
    search_dir = RESULTS_ROOT / search_id
    search_dir.mkdir(parents=True, exist_ok=True)

    trial_plan = []
    for idx, lambda_mag in enumerate(LAMBDA_MAG_VALUES, start=1):
        trial_plan.append(
            {
                'trial_id': f'physics_{idx:02d}',
                'lambda_mag': float(lambda_mag),
                'random_seed': int(base_config.get('training', {}).get('random_seed', 42)) + idx - 1,
            }
        )

    search_space = {
        'search_id': search_id,
        'config_path': str(CONFIG_PATH),
        'parameter': 'training.stf_rate_loss.lambda_mag',
        'values': LAMBDA_MAG_VALUES,
        'trial_count': len(trial_plan),
    }
    with open(search_dir / 'search_space.json', 'w', encoding='utf-8') as f:
        json.dump(search_space, f, ensure_ascii=False, indent=2)

    trial_records: list[dict[str, Any]] = []

    print(f'调参任务目录: {search_dir}')
    for trial in trial_plan:
        trial_id = str(trial['trial_id'])
        lambda_mag = float(trial['lambda_mag'])
        random_seed = int(trial['random_seed'])
        print(f'[{trial_id}] 开始 | lambda_mag={lambda_mag}')

        trial_config = build_trial_config(base_config, lambda_mag=lambda_mag, seed=random_seed)
        train_result = train(config=trial_config)
        best_model_path = train_result.get('best_model_path')

        record: dict[str, Any] = {
            'trial_id': trial_id,
            'lambda_mag': lambda_mag,
            'random_seed': random_seed,
            'run_id': train_result.get('run_id'),
            'model_dir': train_result.get('models_dir'),
            'best_model_path': best_model_path,
            'best_model_swa_path': train_result.get('best_model_swa_path'),
            'config_snapshot_path': train_result.get('config_snapshot_path'),
            'training_log_path': train_result.get('log_file'),
            'eval_results_dir': None,
            'eval_results_run_id': None,
            'test_mae': float('nan'),
            'test_rmse': float('nan'),
            'baseline_mae': float('nan'),
            'baseline_rmse': float('nan'),
            'sample_count': 0,
            'status': 'train_only',
        }

        if best_model_path is not None:
            eval_result = evaluate(
                model_path=best_model_path,
                results_run_id=f'{search_id}_{trial_id}',
                config=trial_config,
            )
            record.update(
                {
                    'eval_results_dir': eval_result.get('results_dir'),
                    'eval_results_run_id': eval_result.get('results_run_id'),
                    'test_mae': float(eval_result.get('mae', float('nan'))),
                    'test_rmse': float(eval_result.get('rmse', float('nan'))),
                    'baseline_mae': float(eval_result.get('baseline_mae', float('nan'))),
                    'baseline_rmse': float(eval_result.get('baseline_rmse', float('nan'))),
                    'sample_count': int(eval_result.get('sample_count', 0)),
                    'status': 'completed',
                }
            )
        else:
            print(f'[{trial_id}] 未生成 best_model.pth，跳过评估。')

        trial_records.append(normalize_record(record))
        write_csv_rows(search_dir / 'trial_summary.csv', trial_records)

        ranked = sorted(
            [r for r in trial_records if r.get('status') == 'completed'],
            key=lambda item: float(item.get('test_mae', float('inf'))),
        )
        write_csv_rows(search_dir / 'best_trials.csv', ranked)

        if record['status'] == 'completed':
            print(
                f'[{trial_id}] 完成 | test_MAE={record["test_mae"]:.4f} '
                f'| test_RMSE={record["test_rmse"]:.4f} | run_id={record["run_id"]}'
            )

    print('调参完成。')
    print(f'汇总文件: {search_dir / "trial_summary.csv"}')
    print(f'最优排序: {search_dir / "best_trials.csv"}')


if __name__ == '__main__':
    main()
