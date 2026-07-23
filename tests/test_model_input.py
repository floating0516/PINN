from __future__ import annotations

import pytest
import torch

from src.data.model_input import assemble_model_input


def test_assemble_model_input_defaults_to_radial() -> None:
    radial = torch.arange(8, dtype=torch.float32).reshape(2, 1, 4)

    result = assemble_model_input({"radial": radial}, {})

    assert result.shape == (2, 1, 4)
    torch.testing.assert_close(result, radial)


def test_assemble_model_input_stacks_rt_in_declared_order() -> None:
    config = {
        "model": {"input_components": ["radial", "tangential"]}
    }
    batch = {
        "radial": torch.ones(2, 1, 4),
        "tangential": torch.full((2, 1, 4), 2.0),
    }

    result = assemble_model_input(batch, config)

    assert result.shape == (2, 2, 4)
    torch.testing.assert_close(result[:, 0], batch["radial"][:, 0])
    torch.testing.assert_close(result[:, 1], batch["tangential"][:, 0])


def test_assemble_model_input_accepts_batch_time_tensors() -> None:
    config = {
        "model": {"input_components": ["radial", "tangential"]}
    }
    batch = {
        "radial": torch.ones(2, 4),
        "tangential": torch.full((2, 4), 2.0),
    }

    result = assemble_model_input(batch, config)

    assert result.shape == (2, 2, 4)


def test_assemble_model_input_rejects_component_shape_mismatch() -> None:
    config = {
        "model": {"input_components": ["radial", "tangential"]}
    }
    batch = {
        "radial": torch.ones(2, 1, 4),
        "tangential": torch.ones(2, 1, 5),
    }

    with pytest.raises(ValueError, match="shapes differ"):
        assemble_model_input(batch, config)


def test_assemble_model_input_rejects_nonfinite_values() -> None:
    radial = torch.tensor([[[1.0, float("nan")]]])

    with pytest.raises(FloatingPointError, match="non-finite"):
        assemble_model_input({"radial": radial}, {})
