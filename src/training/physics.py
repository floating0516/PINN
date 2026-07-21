import torch
import torch.nn as nn
import numpy as np
from typing import Tuple

class PhysicsUtils:
    """
    物理工具集合：实现从垂直位移到地震矩与矩震级的物理转换。
    
    设计动机（Why）
    - PINN 需要通过物理项约束预测值，使网络不偏离基本地球物理规律。
    - 依据 EEW_0011 的公式，以垂直分量在远场近似为矩率，时间积分获得地震矩，
      再通过经验转换得到矩震级。
    """

    def __init__(self, config: dict):
        self.rho = float(config['physics']['rho'])
        self.alpha = float(config['physics']['alpha'])
        self.beta = float(config['physics'].get('beta', 3500.0))
        self.attenuation = float(config['physics']['attenuation'])
        self.geom = float(config['physics'].get('geometrical_spreading_factor', 1.0))
        self.free_surface = float(config['physics'].get('free_surface_factor', 1.0))
        self.radiation_scale = float(config['physics'].get('radiation_pattern', 1.0))
        
    @staticmethod
    def calculate_moment_magnitude(moment_Nm: torch.Tensor) -> torch.Tensor:
        """
        根据地震矩 M0（单位 N·m）计算矩震级 Mw。
        
        原因（Why）
        - Mw 与 M0 的对数线性关系是地震学标准做法，可稳定对不同尺度事件进行统一刻度。
        """
        # 避免对 0 或负值取对数，保证数值稳定
        moment_Nm = torch.clamp(moment_Nm, min=1e10)
        return (2.0 / 3.0) * (torch.log10(moment_Nm) - 9.1)

    def calculate_seismic_moment(self, u_z: torch.Tensor, r_m: torch.Tensor, dt: float) -> torch.Tensor:
        """
        依据垂直位移 u_z（Batch, Time），震中距 r（米）与采样间隔 dt（秒）估算地震矩 M0。
        
        原因（Why）
        - 远场近似下，垂直分量主要体现 P 波的矩率，时间积分可回收地震矩；
        - 取积分的绝对峰值可在复杂震源时序下保证鲁棒性（峰值可能晚于首波）。
        """
        # 1）时间积分：将矩率近似的垂直位移进行积分以恢复地震矩尺度
        integrated_disp = torch.cumsum(u_z, dim=1) * dt

        # 2）峰值取绝对值：抵抗相位与符号差异，聚焦最大能量释放时刻
        max_abs_integral = torch.max(torch.abs(integrated_disp), dim=1).values

        # 3）应用常数项：依据 Eq.4 计算 M0
        term1 = 4 * np.pi * self.rho * (self.alpha ** 3)
        M0 = term1 * r_m * max_abs_integral

        return M0

    def _radiation_coeff_p(self, theta_rad: torch.Tensor, phi_rad: torch.Tensor) -> torch.Tensor:
        # 近似：P 波辐射系数 ~ |cos(theta)|
        return self.radiation_scale * torch.abs(torch.cos(theta_rad))

    def _radiation_coeff_s(self, theta_rad: torch.Tensor, phi_rad: torch.Tensor) -> torch.Tensor:
        # 近似：S 波辐射系数 ~ |sin(theta)|
        return self.radiation_scale * torch.abs(torch.sin(theta_rad))

    def predict_radial_from_rate(
        self,
        dot_M0: torch.Tensor,
        r_m: torch.Tensor,
        theta_deg: torch.Tensor,
        phi_deg: torch.Tensor,
        dt: float,
    ) -> torch.Tensor:
        # 输入：dot_M0 (B, T); r (B,), 角度 (B,)
        B, T = dot_M0.shape
        theta_rad = torch.deg2rad(theta_deg)
        phi_rad = torch.deg2rad(phi_deg)

        Ap = self._radiation_coeff_p(theta_rad, phi_rad)  # (B,)
        As = self._radiation_coeff_s(theta_rad, phi_rad)  # (B,)

        # 传播延迟（步数）
        delay_p_steps = torch.clamp((r_m / self.alpha / dt).round().long(), min=0)
        delay_s_steps = torch.clamp((r_m / self.beta / dt).round().long(), min=0)

        # 系数（远场近似，包含几何散射与自由面项）
        Cp = (self.geom * self.free_surface * Ap) / (4.0 * np.pi * self.rho * (self.alpha ** 3) * torch.clamp(r_m, min=1.0))
        Cs = (self.geom * self.free_surface * As) / (4.0 * np.pi * self.rho * (self.beta ** 3) * torch.clamp(r_m, min=1.0))

        # 按样本逐一应用延迟与积分
        u_pred = dot_M0.new_zeros((B, T))
        for i in range(B):
            # 延迟
            dp = int(delay_p_steps[i].item())
            ds = int(delay_s_steps[i].item())
            seq_p = dot_M0[i]
            seq_s = dot_M0[i]
            if dp > 0:
                seq_p = torch.cat([torch.zeros(dp, device=dot_M0.device, dtype=dot_M0.dtype), seq_p[:-dp]])
            if ds > 0:
                seq_s = torch.cat([torch.zeros(ds, device=dot_M0.device, dtype=dot_M0.dtype), seq_s[:-ds]])
            # 时间积分（恢复位移量纲）
            up = torch.cumsum(seq_p, dim=0) * dt
            us = torch.cumsum(seq_s, dim=0) * dt
            u_pred[i] = Cp[i] * up + Cs[i] * us
        # 简单衰减（整体缩放）
        u_pred = self.attenuation * u_pred
        return u_pred

    def magnitude_from_rate(self, dot_M0: torch.Tensor, dt: float) -> torch.Tensor:
        # 将矩率积分得到 M0 峰值，并转换为 Mw
        M0_seq = torch.cumsum(torch.clamp(dot_M0, min=0.0), dim=1) * dt
        M0_peak = torch.max(M0_seq, dim=1).values
        return self.calculate_moment_magnitude(M0_peak)

class PhysicsLoss(nn.Module):
    """
    物理约束损失：将数据拟合项与物理一致性项组合，推动网络学习对物理规律的遵循。
    
    原因（Why）
    - 仅拟合数据可能受噪声与分布偏移影响；加入物理项可提升泛化与可解释性。
    """

    def __init__(self, config: dict):
        super(PhysicsLoss, self).__init__()
        self.utils = PhysicsUtils(config)
        training_cfg = (config.get('training', {}) or {})
        ds_cfg = (config.get('dataset', {}) or {})
        self.weight_phys = float(training_cfg.get('physics_loss_weight', 0.0))
        self.weight_stf = float(training_cfg.get('stf_loss_weight', 0.0))
        self.weight_stf_smooth = float(training_cfg.get('stf_smooth_loss_weight', 0.0))
        self.stf_m_ref = float(ds_cfg.get('stf_m_ref', 1.0e18))
        self.rate_representation = str(training_cfg.get('rate_representation', 'auto')).lower()
        if self.rate_representation == 'auto':
            self.rate_representation = 'log1p' if 'stf_m_ref' in ds_cfg else 'linear'
        self.dt = 1.0

    def _rate_to_moment_rate(self, rate: torch.Tensor) -> torch.Tensor:
        if self.rate_representation == 'log1p':
            denom = max(float(self.stf_m_ref), 1.0e-30)
            rate_safe = torch.nan_to_num(rate, nan=0.0, posinf=0.0, neginf=0.0)
            rate_safe = torch.clamp(rate_safe, min=-20.0, max=6.0)
            dot_m0 = denom * (torch.pow(10.0, rate_safe) - 1.0)
            return torch.clamp(dot_m0, min=0.0)
        rate_safe = torch.nan_to_num(rate, nan=0.0, posinf=0.0, neginf=0.0)
        return torch.clamp(rate_safe, min=0.0)
    
    @staticmethod
    def _shape_loss(pred_dot_m0: torch.Tensor, true_dot_m0: torch.Tensor) -> torch.Tensor:
        eps = 1.0e-12
        pred = torch.clamp(pred_dot_m0, min=0.0)
        true = torch.clamp(true_dot_m0, min=0.0)
        pred_sum = torch.sum(pred, dim=1, keepdim=True).clamp(min=eps)
        true_sum = torch.sum(true, dim=1, keepdim=True).clamp(min=eps)
        pred_n = pred / pred_sum
        true_n = true / true_sum
        return nn.MSELoss()(pred_n, true_n)
    
    @staticmethod
    def _smooth_loss(dot_m0: torch.Tensor) -> torch.Tensor:
        if dot_m0.size(1) < 2:
            return dot_m0.new_tensor(0.0)
        diff = dot_m0[:, 1:] - dot_m0[:, :-1]
        return torch.mean(torch.abs(diff))
        
    def forward(
        self,
        pred_rate: torch.Tensor,
        stf_log_obs: torch.Tensor,
        true_mag: torch.Tensor,
        r_m: torch.Tensor,
        theta_deg: torch.Tensor,
        phi_deg: torch.Tensor,
        dt: float = 1.0,
        stf_true: torch.Tensor | None = None,
        has_stf: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if stf_log_obs.dim() == 3:
            stf_log_obs_t = stf_log_obs.squeeze(1)
        else:
            stf_log_obs_t = stf_log_obs
        stf_log_obs_t = torch.nan_to_num(stf_log_obs_t, nan=0.0, posinf=0.0, neginf=0.0)
        pred_rate = torch.nan_to_num(pred_rate, nan=0.0, posinf=0.0, neginf=0.0)

        if has_stf is None:
            data_loss = nn.MSELoss()(pred_rate, stf_log_obs_t)
        else:
            mask = has_stf.bool().view(-1)
            if torch.any(mask):
                data_loss = nn.MSELoss()(pred_rate[mask], stf_log_obs_t[mask])
            else:
                data_loss = pred_rate.new_tensor(0.0)

        pred_dot_m0 = self._rate_to_moment_rate(pred_rate)

        # 物理项：积分后的矩震级与目录震级一致
        Mw_pred = self.utils.magnitude_from_rate(pred_dot_m0, dt)
        physics_loss_mag = nn.MSELoss()(Mw_pred.unsqueeze(1), true_mag.unsqueeze(1))
        physics_loss = physics_loss_mag

        stf_loss = pred_rate.new_tensor(0.0)
        if stf_true is not None:
            if has_stf is None:
                stf_loss = self._shape_loss(pred_dot_m0, stf_true)
            else:
                mask = has_stf.bool().view(-1)
                if torch.any(mask):
                    stf_loss = self._shape_loss(pred_dot_m0[mask], stf_true[mask])



        total_loss = (
            data_loss
            + self.weight_phys * physics_loss
            + self.weight_stf * stf_loss
        )
        return total_loss, data_loss, physics_loss, stf_loss
