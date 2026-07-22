from pathlib import Path


CALL_SITE_FILES = (
    "src/training/train.py",
    "src/evaluation/evaluate.py",
    "src/evaluation/evaluate_no_stf.py",
    "src/evaluation/evaluate_unseen.py",
    "scripts/experiments/loeo_event_level_eval.py",
    "scripts/robustness/latency_analysis.py",
)


def test_production_metadata_uses_the_shared_builder() -> None:
    for name in CALL_SITE_FILES:
        text = Path(name).read_text(encoding="utf-8")
        assert "from src.data.metadata import build_metadata_tensor" in text, name
        assert "torch.stack(" not in text, name
        assert "dist_log =" not in text, name
        assert "phi_deg = ds_helper.default_phi_deg" not in text, name
        assert "phi_deg_val = ds_helper.default_phi_deg" not in text, name


def test_external_geometry_uses_the_shared_v2_sample_builder() -> None:
    for name in (
        "src/evaluation/evaluate_unseen.py",
        "scripts/robustness/latency_analysis.py",
    ):
        text = Path(name).read_text(encoding="utf-8")
        assert "build_station_sample(" in text, name
        assert "_calculate_geodetics(" not in text, name
        assert "_build_unseen_dataset_helper" not in text, name
