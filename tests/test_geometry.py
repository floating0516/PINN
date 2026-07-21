import math

import pytest
import torch

from src.data.geometry import compute_source_station_geometry
from src.data.metadata import build_metadata_tensor


def test_collocated_station_uses_depth_as_source_distance() -> None:
    geometry = compute_source_station_geometry(35.0, 140.0, 20.0, 35.0, 140.0)

    assert geometry.epicentral_distance_m == 0.0
    assert geometry.source_distance_m == 20_000.0
    assert geometry.takeoff_angle_deg == 0.0


def test_due_north_station_has_northward_azimuth() -> None:
    geometry = compute_source_station_geometry(35.0, 140.0, 10.0, 36.0, 140.0)

    assert 110_000.0 < geometry.epicentral_distance_m < 112_000.0
    assert abs(geometry.azimuth_deg - 0.0) < 1.0e-6
    assert abs(geometry.back_azimuth_deg - 180.0) < 1.0e-6
    assert geometry.source_distance_m == math.hypot(
        geometry.epicentral_distance_m, 10_000.0
    )
    expected = math.degrees(
        math.atan2(geometry.epicentral_distance_m, 10_000.0)
    )
    assert abs(geometry.takeoff_angle_deg - expected) < 1.0e-6


def test_oblique_route_uses_reverse_initial_bearing_as_back_azimuth() -> None:
    geometry = compute_source_station_geometry(60.0, -150.0, 10.0, 10.0, 80.0)

    assert abs(geometry.back_azimuth_deg - 22.856196912016) < 1.0e-6


def test_metadata_order_is_log_r_theta_azimuth() -> None:
    source_distance = torch.tensor([100_000.0])
    theta = torch.tensor([30.0])
    azimuth = torch.tensor([90.0])

    metadata = build_metadata_tensor(source_distance, theta, azimuth)

    assert metadata.shape == (1, 5)
    assert torch.allclose(metadata[0, 0], torch.log(source_distance)[0])
    assert torch.allclose(metadata[0, 1], torch.tensor(0.5), atol=1.0e-6)
    assert torch.allclose(
        metadata[0, 2], torch.tensor(math.sqrt(3.0) / 2.0), atol=1.0e-6
    )
    assert torch.allclose(metadata[0, 3], torch.tensor(1.0), atol=1.0e-6)
    assert torch.allclose(metadata[0, 4], torch.tensor(0.0), atol=1.0e-6)


def test_metadata_clamps_distances_up_to_one_meter() -> None:
    source_distance = torch.tensor([0.0, 0.5, 1.0])
    angles = torch.zeros(3)

    metadata = build_metadata_tensor(source_distance, angles, angles)

    assert torch.isfinite(metadata[:, 0]).all()
    assert torch.equal(metadata[:, 0], torch.zeros(3))


@pytest.mark.parametrize(
    "values",
    [
        (math.nan, 140.0, 20.0, 35.0, 140.0),
        (35.0, math.inf, 20.0, 35.0, 140.0),
        (35.0, 140.0, -math.inf, 35.0, 140.0),
        (35.0, 140.0, 20.0, math.nan, 140.0),
        (35.0, 140.0, 20.0, 35.0, math.inf),
    ],
)
def test_nonfinite_geometry_inputs_are_rejected(
    values: tuple[float, float, float, float, float],
) -> None:
    with pytest.raises(ValueError, match="有限值"):
        compute_source_station_geometry(*values)


def test_negative_depth_is_rejected() -> None:
    with pytest.raises(ValueError, match="depth_km"):
        compute_source_station_geometry(35.0, 140.0, -1.0, 36.0, 140.0)
