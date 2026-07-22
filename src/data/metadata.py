from __future__ import annotations

import torch


def metadata_distance_from_config(
    config: dict,
    *,
    source_distance_m: torch.Tensor,
    epicentral_distance_m: torch.Tensor,
) -> torch.Tensor:
    mode = str(
        (config.get("geometry", {}) or {}).get(
            "network_distance",
            "hypocentral",
        )
    ).lower()
    if mode == "hypocentral":
        return source_distance_m
    if mode == "epicentral":
        return epicentral_distance_m
    raise ValueError(f"unsupported geometry.network_distance: {mode}")


def build_metadata_tensor(
    source_distance_m: torch.Tensor,
    takeoff_angle_deg: torch.Tensor,
    azimuth_deg: torch.Tensor,
) -> torch.Tensor:
    r = source_distance_m.reshape(-1).clamp_min(1.0)
    theta = torch.deg2rad(takeoff_angle_deg.reshape(-1))
    azimuth = torch.deg2rad(azimuth_deg.reshape(-1))
    return torch.stack(
        [
            torch.log(r),
            torch.sin(theta),
            torch.cos(theta),
            torch.sin(azimuth),
            torch.cos(azimuth),
        ],
        dim=1,
    )
