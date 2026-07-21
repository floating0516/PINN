from __future__ import annotations

import torch


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
