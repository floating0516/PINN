import math
from dataclasses import dataclass
from typing import Any

import torch

from src.baseline.scaling_laws import AVAILABLE_SCALING_LAWS, predict_mw


@dataclass(frozen=True)
class Baseline:
    """
    EEW_0012：Rapid Earthquake Magnitude Estimation for Local Early Warning Systems using Seismogeodesy

    实现目标:
        - 基于论文式(6)-(10)从水平径向位移 u_hr 估计地震矩 M0 与矩震级 Mwg。

    参数:
        rho: 介质密度 (kg/m^3)
        alpha: P 波速度 (m/s)
        beta: S 波速度 (m/s)
    """

    rho: float
    alpha: float
    beta: float

    @staticmethod
    def from_config(config: dict[str, Any]) -> "Baseline":
        physics_cfg = config.get("physics", {}) or {}
        rho = float(physics_cfg.get("rho", 3400.0))
        alpha = float(physics_cfg.get("alpha", 7900.0))
        beta = float(physics_cfg.get("beta", 4533.0))
        return Baseline(rho=rho, alpha=alpha, beta=beta)

    @staticmethod
    def calculate_moment_magnitude(moment_nm: torch.Tensor) -> torch.Tensor:
        moment_nm = torch.clamp(moment_nm, min=1.0e-10)
        return (2.0 / 3.0) * (torch.log10(moment_nm) - 9.1)

    @staticmethod
    def _radiation_coeffs_uhr(
        theta_rad: torch.Tensor,
        phi_rad: torch.Tensor,
        apply_radiation_pattern: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        计算用于 u_hr 的辐射项系数 (EEW_0012 Eq. 4)。

        参数:
            theta_rad: (B,) 天顶角（弧度）
            phi_rad: (B,) 水平夹角（弧度）
            apply_radiation_pattern: False 时返回全 1（模拟实时不可用）
        返回:
            A_ip, A_is, A_fp, A_fs: (B,)
        """
        if not apply_radiation_pattern:
            ones = torch.ones_like(theta_rad)
            return ones, ones, ones, ones

        sin_theta = torch.sin(theta_rad)
        cos_theta = torch.cos(theta_rad)
        sin2 = torch.sin(2.0 * theta_rad)
        cos2 = torch.cos(2.0 * theta_rad)
        cos_phi = torch.cos(phi_rad)

        r_proj = sin_theta
        theta_proj = cos_theta

        a_ip = cos_phi * (4.0 * sin2 * r_proj - 2.0 * cos2 * theta_proj)
        a_is = cos_phi * (-3.0 * sin2 * r_proj + 3.0 * cos2 * theta_proj)
        a_fp = cos_phi * (sin2 * r_proj)
        a_fs = cos_phi * (cos2 * theta_proj)
        return a_ip, a_is, a_fp, a_fs

    def _compute_coefficients(
        self,
        r_m: torch.Tensor,
        theta_deg: torch.Tensor,
        phi_deg: torch.Tensor,
        apply_radiation_pattern: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        计算 C_far 和 C_int 系数 (EEW_0012 Eq. 7-8)。

        全部使用 float64 精度以避免数值下溢。

        返回:
            c_far: (B,) 远场系数 (float64)
            c_int: (B,) 中场系数 (float64)
        """
        theta_rad = torch.deg2rad(theta_deg.double())
        phi_rad = torch.deg2rad(phi_deg.double())
        r_safe = torch.clamp(r_m.double(), min=1.0)

        a_ip, a_is, a_fp, a_fs = self._radiation_coeffs_uhr(
            theta_rad, phi_rad, apply_radiation_pattern
        )

        four_pi_rho = 4.0 * math.pi * self.rho
        c_far = (
            a_fp / (four_pi_rho * (self.alpha ** 3) * r_safe)
            + a_fs / (four_pi_rho * (self.beta ** 3) * r_safe)
        )
        c_int = (
            a_ip / (four_pi_rho * (self.alpha ** 2) * r_safe ** 2)
            + a_is / (four_pi_rho * (self.beta ** 2) * r_safe ** 2)
        )
        return c_far, c_int

    def _recursive_m0(
        self,
        u_hr: torch.Tensor,
        c_far: torch.Tensor,
        c_int: torch.Tensor,
        dt: float,
        t_max: int,
    ) -> torch.Tensor:
        """
        执行 EEW_0012 Eq. 6 递推并返回 max|M0(t)|。

        全部在 float64 精度下运算。

        参数:
            u_hr: (B, T) 位移序列 (任意 dtype，内部转 float64)
            c_far: (B,) 远场系数 (float64)
            c_int: (B,) 中场系数 (float64)
            dt: 采样间隔 (s)
            t_max: 使用的最大时间步数
        返回:
            m0_peak: (B,) float64
        """
        gamma = c_far / dt
        denom = c_int + gamma
        u64 = u_hr[:, :t_max].double()

        prev_m0 = torch.zeros_like(denom)
        m0_peak = torch.zeros_like(denom)

        for t in range(t_max):
            numer = u64[:, t] + gamma * prev_m0
            m0_t = torch.abs(numer / denom)
            prev_m0 = m0_t
            m0_peak = torch.maximum(m0_peak, m0_t)

        return m0_peak

    def calculate_seismic_moment(
        self,
        u_hr: torch.Tensor,
        r_m: torch.Tensor,
        theta_deg: torch.Tensor,
        phi_deg: torch.Tensor,
        dt: float,
        apply_radiation_pattern: bool = False,
        window_end_steps: int | None = None,
        include_intermediate_field: bool = True,
    ) -> torch.Tensor:
        """
        根据 EEW_0012 式(6)-(9)从 u_hr 估计地震矩 M0（N·m）。

        全部使用 float64 精度计算，避免系数极小值引起的精度问题。

        参数:
            u_hr: (B, T) 水平径向位移（m）
            r_m: (B,) 震源-台站距离（m）
            theta_deg: (B,) 天顶角（度）
            phi_deg: (B,) 水平夹角（度）
            dt: 采样间隔（s）
            apply_radiation_pattern: 是否启用辐射修正
            window_end_steps: 可选，限制使用的时间步数
            include_intermediate_field: 是否包含中场项(C_int)，论文Section 3.3发现逆冲事件设为False(C_int=0)效果更好
        返回:
            M0: (B,) float32
        """
        if u_hr.dim() != 2:
            raise ValueError(f"u_hr 期望形状为 (B,T)，实际为 {tuple(u_hr.shape)}")
        if dt <= 0.0:
            raise ValueError(f"dt 必须为正数，实际为 {dt}")

        c_far, c_int = self._compute_coefficients(
            r_m, theta_deg, phi_deg, apply_radiation_pattern
        )
        if not include_intermediate_field:
            c_int = torch.zeros_like(c_int)

        t_max = int(u_hr.size(1))
        if window_end_steps is not None:
            t_max = int(max(0, min(t_max, int(window_end_steps))))

        if t_max == 0:
            return torch.zeros(u_hr.size(0), device=u_hr.device)

        m0_peak = self._recursive_m0(u_hr, c_far, c_int, dt, t_max)
        return torch.clamp(m0_peak, min=0.0).float()

    def calculate_mwg(
        self,
        u_hr: torch.Tensor,
        r_m: torch.Tensor,
        theta_deg: torch.Tensor,
        phi_deg: torch.Tensor,
        dt: float,
        apply_radiation_pattern: bool = False,
        window_end_steps: int | None = None,
        include_intermediate_field: bool = True,
    ) -> torch.Tensor:
        """
        根据 EEW_0012 从 u_hr 估计矩震级（Mwg）。

        当 apply_radiation_pattern=False 时，尝试 u_hr 和 -u_hr 两种符号，
        取产生更大 M0 的结果（论文实时场景下位移正负号未知）。

        返回:
            (B,) Mwg
        """
        m0 = self.calculate_seismic_moment(
            u_hr=u_hr,
            r_m=r_m,
            theta_deg=theta_deg,
            phi_deg=phi_deg,
            dt=dt,
            apply_radiation_pattern=apply_radiation_pattern,
            window_end_steps=window_end_steps,
            include_intermediate_field=include_intermediate_field,
        )

        if not apply_radiation_pattern:
            m0_flip = self.calculate_seismic_moment(
                u_hr=-u_hr,
                r_m=r_m,
                theta_deg=theta_deg,
                phi_deg=phi_deg,
                dt=dt,
                apply_radiation_pattern=False,
                window_end_steps=window_end_steps,
                include_intermediate_field=include_intermediate_field,
            )
            m0 = torch.maximum(m0, m0_flip)

        return self.calculate_moment_magnitude(m0)
