from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# 辅助函数
# =============================================================================

def _as_batch_vector(x: float | torch.Tensor, batch_size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """将标量或张量转换为形状 (B,) 的批次向量"""
    x_t = torch.as_tensor(x, device=device, dtype=dtype)
    if x_t.dim() == 0:
        return x_t.expand(batch_size)
    if x_t.shape == (batch_size,):
        return x_t
    raise ValueError(f"期望标量或形状为 (B,) 的张量，但得到 shape={tuple(x_t.shape)}")


def _as_batch_coef(x: float | torch.Tensor, batch_size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """将标量或张量转换为形状 (B, 1) 的系数张量"""
    x_t = torch.as_tensor(x, device=device, dtype=dtype)
    if x_t.dim() == 0:
        return x_t.view(1, 1)
    if x_t.shape == (batch_size,):
        return x_t.view(batch_size, 1)
    if x_t.shape == (batch_size, 1):
        return x_t
    raise ValueError(f"期望标量或形状为 (B,) / (B,1) 的张量，但得到 shape={tuple(x_t.shape)}")


def _shift_with_zeros(x: torch.Tensor, shifts: torch.Tensor) -> torch.Tensor:
    """按样本不同延迟移位张量，前面填零"""
    if x.dim() != 2:
        raise ValueError("x 维度应为 [B, T]")
    if shifts.dim() != 1 or shifts.shape[0] != x.shape[0]:
        raise ValueError("shifts 维度应为 [B]")
    if x.numel() == 0:
        return x

    B, T = x.shape
    device = x.device
    idx_t = torch.arange(T, device=device, dtype=torch.long).view(1, T).expand(B, T)
    shifts_t = shifts.to(device=device, dtype=torch.long).view(B, 1).expand(B, T)
    src_idx = idx_t - shifts_t
    valid = src_idx >= 0
    src_idx = torch.clamp(src_idx, min=0)
    shifted = x.gather(1, src_idx)
    return shifted * valid.to(dtype=x.dtype)


# =============================================================================
# 物理约束函数 (来自 physics.py 和 EEW_0012)
# =============================================================================

def compute_moment_magnitude(M0_Nm: torch.Tensor) -> torch.Tensor:
    """
    根据地震矩 M0（单位 N·m）计算矩震级 Mw (EEW_0012 Equation 10)
    
    Mw = (2/3)(log10(M0) - 9.1)
    
    原因（Why）：
    - Mw 与 M0 的对数线性关系是地震学标准做法
    - 通过约束预测的 Mw 与目录震级一致，提升物理可信度
    """
    M0_safe = torch.clamp(M0_Nm, min=1e10)
    return (2.0 / 3.0) * (torch.log10(M0_safe) - 9.1)


def compute_shape_loss(pred: torch.Tensor, true: torch.Tensor) -> torch.Tensor:
    """
    归一化形状损失 - 只比较 STF 形状，不受幅度影响
    
    原因（Why）：
    - 即使幅度有偏，STF 的时间演化形状应与真实接近
    - 归一化后只比较"形状"，避免被幅度差异主导
    """
    eps = 1e-12
    pred_pos = torch.clamp(pred, min=0.0)
    true_pos = torch.clamp(true, min=0.0)
    pred_sum = pred_pos.sum(dim=1, keepdim=True).clamp(min=eps)
    true_sum = true_pos.sum(dim=1, keepdim=True).clamp(min=eps)
    pred_n = pred_pos / pred_sum
    true_n = true_pos / true_sum
    return F.mse_loss(pred_n, true_n)


# =============================================================================
# 辐射花型系数计算 (EEW_0012 Equation 4)
# =============================================================================

def compute_radiation_coefficients(
    theta_deg: torch.Tensor,
    phi_deg: torch.Tensor,
    mode: str = "simplified",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    计算辐射花型系数 (仅径向分量)
    
    参数:
        theta_deg: (B,) 天顶角 (从垂直线到台站的角度, 度)
        phi_deg: (B,) 水平角 (滑动方向与台站之间的角度, 度)
        mode: "simplified" | "full" | "none"
            - simplified: 所有系数设为 1 (适用于实时且无源参数)
            - full: 使用完整公式 (需要准确的 theta, phi)
            - none: 所有系数设为 0 (禁用辐射校正)
    
    返回:
        A_IP, A_IS, A_FP, A_FS: (B,) 形状的辐射系数
    """
    device = theta_deg.device
    dtype = theta_deg.dtype
    B = theta_deg.shape[0]
    
    if mode == "none":
        zeros = torch.zeros(B, device=device, dtype=dtype)
        return zeros, zeros, zeros, zeros
    
    if mode == "simplified":
        # 简化模式: 所有系数设为 1 (文章建议走滑断层可忽略辐射校正)
        ones = torch.ones(B, device=device, dtype=dtype)
        return ones, ones, ones, ones
    
    # 完整模式: 使用 EEW_0012 公式 (4)
    # 论文中辐射系数是向量形式，需要投影到水平径向分量
    theta_rad = torch.deg2rad(theta_deg)
    phi_rad = torch.deg2rad(phi_deg)
    
    # 球坐标到水平径向的投影系数
    sin_theta = torch.sin(theta_rad)  # r̂ 投影到水平径向
    cos_theta = torch.cos(theta_rad)  # θ̂ 投影到水平径向
    
    sin_2theta = torch.sin(2 * theta_rad)
    cos_2theta = torch.cos(2 * theta_rad)
    cos_phi = torch.cos(phi_rad)
    
    # 中场P波: A^IP = 4*sin(2θ)*cos(φ)*r̂ - 2*cos(2θ)*cos(φ)*θ̂
    # 投影到水平径向: r̂ -> sin(θ), θ̂ -> cos(θ)
    A_IP = cos_phi * (4 * sin_2theta * sin_theta - 2 * cos_2theta * cos_theta)
    
    # 中场S波: A^IS = -3*sin(2θ)*cos(φ)*r̂ + 3*cos(2θ)*cos(φ)*θ̂
    A_IS = cos_phi * (-3 * sin_2theta * sin_theta + 3 * cos_2theta * cos_theta)
    
    # 远场P波: A^FP = sin(2θ)*cos(φ)*r̂
    A_FP = cos_phi * sin_2theta * sin_theta
    
    # 远场S波: A^FS = cos(2θ)*cos(φ)*θ̂
    A_FS = cos_phi * cos_2theta * cos_theta
    
    # 保留辐射系数的符号 (论文 EEW_0012 Eq.4: 系数可为负, 表示反向运动方向)
    return A_IP, A_IS, A_FP, A_FS


# =============================================================================
# 物理系数计算 (EEW_0012 Equations 7-8)
# =============================================================================

def compute_physical_coefficients(
    r_m: torch.Tensor,
    rho: float,
    alpha: float,
    beta: float,
    A_IP: torch.Tensor,
    A_IS: torch.Tensor,
    A_FP: torch.Tensor,
    A_FS: torch.Tensor,
    geom: float = 1.0,
    free_surface: float = 1.0,
    attenuation: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    计算中场和远场物理系数 (EEW_0012 Equations 7-8)
    
    参数:
        r_m: (B,) 震中距 (米)
        rho: 密度 (kg/m³)
        alpha: P波速度 (m/s)
        beta: S波速度 (m/s)
        A_IP, A_IS, A_FP, A_FS: (B,) 辐射花型系数
        geom: 几何扩散因子
        free_surface: 自由表面放大因子
        attenuation: 衰减因子
    
    返回:
        C_int_P, C_int_S, C_far_P, C_far_S: (B,) 形状的物理系数
    """
    device = r_m.device
    dtype = r_m.dtype
    
    PI_4 = 4.0 * math.pi
    r_safe = torch.clamp(r_m, min=1.0)  # 避免除零
    
    # 综合因子
    scale = geom * free_surface * attenuation
    
    # 中场系数 (Equation 8): C_int = A/(4πρv²r²)
    C_int_P = (scale * A_IP) / (PI_4 * rho * (alpha ** 2) * (r_safe ** 2))
    C_int_S = (scale * A_IS) / (PI_4 * rho * (beta ** 2) * (r_safe ** 2))
    
    # 远场系数 (Equation 7): C_far = A/(4πρv³r)
    C_far_P = (scale * A_FP) / (PI_4 * rho * (alpha ** 3) * r_safe)
    C_far_S = (scale * A_FS) / (PI_4 * rho * (beta ** 3) * r_safe)
    
    return C_int_P, C_int_S, C_far_P, C_far_S


# =============================================================================
# 正演算子: 从矩率合成位移 (EEW_0012 Equation 3)
# =============================================================================

def forward_displacement_from_rate(
    rate_hat: torch.Tensor,
    dt: torch.Tensor,
    r_m: torch.Tensor,
    alpha: float,
    beta: float,
    C_int_P: torch.Tensor,
    C_int_S: torch.Tensor,
    C_far_P: torch.Tensor,
    C_far_S: torch.Tensor,
    include_intermediate: bool = True,
    skip_delays: bool = False,
) -> torch.Tensor:
    """
    从矩率正演合成位移 (EEW_0012 Equation 3)
    
    物理公式:
        u_hr(r,t) = C_int_P * M0(t-tP) + C_int_S * M0(t-tS) 
                  + C_far_P * dot_M0(t-tP) + C_far_S * dot_M0(t-tS)
    
    由于输入是矩率 dot_M0，远场项直接使用矩率，中场项需要积分得到 M0
    
    参数:
        rate_hat: (B, T) 预测的矩率
        dt: (B,) 或标量，采样间隔
        r_m: (B,) 震中距
        alpha, beta: P/S波速度
        C_int_P, C_int_S: (B,) 中场系数
        C_far_P, C_far_S: (B,) 远场系数
        include_intermediate: 是否包含中场项
    
    返回:
        u_hat: (B, T) 合成位移
    """
    B, T = rate_hat.shape
    device = rate_hat.device
    dtype = rate_hat.dtype
    
    # 确保 dt 是批次向量
    if isinstance(dt, (int, float)):
        dt_b = torch.full((B,), float(dt), device=device, dtype=dtype)
    else:
        dt_b = dt.view(B)
    
    dt_bt = dt_b.view(B, 1)
    
    # 计算延迟步数
    alpha_t = torch.as_tensor(alpha, device=device, dtype=dtype)
    beta_t = torch.as_tensor(beta, device=device, dtype=dtype)
    
    nP = torch.floor((r_m / torch.clamp(alpha_t, min=1e-12)) / torch.clamp(dt_b, min=1e-12)).to(torch.long)
    nS = torch.floor((r_m / torch.clamp(beta_t, min=1e-12)) / torch.clamp(dt_b, min=1e-12)).to(torch.long)
    nP = torch.clamp(nP, min=0)
    nS = torch.clamp(nS, min=0)
    
    # 积分得到矩历史 M0(t) = ∫ dot_M0 dt
    M0_hat = torch.cumsum(rate_hat * dt_bt, dim=1)
    
    # 应用延迟
    # 当 skip_delays=True 时，数据加载器已将 STF 对齐到 P 波到时，
    # 因此 P 波延迟跳过（模型输出已在 P 波时间坐标），
    # 但 S 波仍需施加 **相对延迟** Δn = nS - nP = r*(1/β - 1/α)/dt
    if skip_delays:
        M0_p = M0_hat
        rate_p = rate_hat
        # S 波相对延迟
        nS_rel = torch.clamp(nS - nP, min=0)
        if nS_rel.max().item() > 0:
            M0_s = _shift_with_zeros(M0_hat, nS_rel)
            rate_s = _shift_with_zeros(rate_hat, nS_rel)
        else:
            M0_s = M0_hat
            rate_s = rate_hat
    else:
        M0_p = _shift_with_zeros(M0_hat, nP)
        M0_s = _shift_with_zeros(M0_hat, nS)
        rate_p = _shift_with_zeros(rate_hat, nP)
        rate_s = _shift_with_zeros(rate_hat, nS)
    
    # 系数扩展为 (B, 1) 以便广播
    C_int_P_t = C_int_P.view(B, 1)
    C_int_S_t = C_int_S.view(B, 1)
    C_far_P_t = C_far_P.view(B, 1)
    C_far_S_t = C_far_S.view(B, 1)
    
    # 远场贡献: 使用矩率 dot_M0 (EEW_0012 Eq.3)
    # u_far = C_far_P * dot_M0(t-tP) + C_far_S * dot_M0(t-tS)
    u_far = C_far_P_t * rate_p + C_far_S_t * rate_s
    
    # 中场贡献: 使用矩历史 M0 (EEW_0012 Eq.3)
    # u_int = C_int_P * M0(t-tP) + C_int_S * M0(t-tS)
    if include_intermediate:
        u_int = C_int_P_t * M0_p + C_int_S_t * M0_s
    else:
        u_int = torch.zeros_like(u_far)
    
    u_hat = u_int + u_far
    return u_hat


# =============================================================================
# 主损失函数
# =============================================================================

def pinn_loss_stf_rate(
    rate_hat: torch.Tensor,
    rate_ref: torch.Tensor | None,
    u_obs: torch.Tensor,
    dt: float | torch.Tensor,
    r: float | torch.Tensor,
    alpha: float,
    beta: float,
    rho: float,
    theta_deg: torch.Tensor,
    phi_deg: torch.Tensor,
    geom: float = 1.0,
    free_surface: float = 1.0,
    attenuation: float = 1.0,
    lambda_MSE: float = 1.0,
    lambda_wave: float = 0.5,
    lambda_nonneg: float = 1.0,
    lambda_smooth: float = 0.1,
    lambda_physics: float = 0.1,
    lambda_shape: float = 0.1,
    has_ref: torch.Tensor | None = None,
    include_intermediate: bool = True,
    radiation_mode: str = "simplified",
    pred_rate_log: torch.Tensor | None = None,
    true_mag: torch.Tensor | None = None,
    stf_m_ref: float = 1.0e18,
    skip_delays: bool = False,
) -> tuple[torch.Tensor, dict[str, float]]:
    """
    基于物理的 PINN 损失函数 (EEW_0012 实现)
    
    参数:
        rate_hat: (B, T) 网络预测的矩率 dot_M0(t)
        rate_ref: (B, T) 参考矩率（可为 None）
        u_obs: (B, T) 观测的水平径向分量
        dt: 采样间隔（标量或 (B,)）
        r: 震中距（标量或 (B,)）
        alpha, beta: P、S 波速度
        rho: 密度
        theta_deg, phi_deg: (B,) 辐射角度
        geom, free_surface, attenuation: 物理因子
        lambda_*: 损失权重
        has_ref: (B,) 是否有参考 STF 的掩码
        include_intermediate: 是否包含中场项
        radiation_mode: 辐射花型模式 ("simplified"|"full"|"none")
    
    返回:
        total_loss: 标量 Tensor
        loss_dict: 各项损失数值
    """
    if rate_hat.dim() != 2:
        raise ValueError("rate_hat 维度应为 [B, T]")
    if u_obs.dim() != 2:
        raise ValueError("u_obs 维度应为 [B, T]")
    
    B, T = rate_hat.shape
    if u_obs.shape != (B, T):
        raise ValueError("u_obs 与 rate_hat 维度应一致")

    device = rate_hat.device
    dtype = rate_hat.dtype

    # 转换为批次向量
    dt_b = _as_batch_vector(dt, B, device, dtype)
    r_b = _as_batch_vector(r, B, device, dtype)
    
    # 确保角度是批次张量
    if theta_deg.dim() == 0:
        theta_deg = theta_deg.expand(B)
    if phi_deg.dim() == 0:
        phi_deg = phi_deg.expand(B)

    # 计算辐射花型系数
    A_IP, A_IS, A_FP, A_FS = compute_radiation_coefficients(
        theta_deg, phi_deg, mode=radiation_mode
    )
    
    # 计算物理系数
    C_int_P, C_int_S, C_far_P, C_far_S = compute_physical_coefficients(
        r_b, rho, alpha, beta, A_IP, A_IS, A_FP, A_FS,
        geom=geom, free_surface=free_surface, attenuation=attenuation
    )
    
    # 正演合成位移
    u_hat = forward_displacement_from_rate(
        rate_hat, dt_b, r_b, alpha, beta,
        C_int_P, C_int_S, C_far_P, C_far_S,
        include_intermediate=include_intermediate,
        skip_delays=skip_delays,
    )

    # 波形损失 (仅用观测值归一化，保证梯度稳定性)
    # 用 u_obs 的最大值作为尺度因子（不依赖预测值，避免训练初期尺度波动）
    u_scale = u_obs.abs().amax(dim=1, keepdim=True) + 1e-12  # (B, 1)
    L_wave = F.mse_loss(u_hat / u_scale, u_obs / u_scale)

    # MSE 损失（在 log1p 空间比较，避免尺度问题）
    if rate_ref is not None and pred_rate_log is not None:
        # rate_ref 和 pred_rate_log 都应该是 log1p 编码 (范围 0~3)
        if rate_ref.shape != (B, T):
            raise ValueError("rate_ref 维度应为 [B, T]")
        if has_ref is None:
            L_MSE = F.mse_loss(pred_rate_log, rate_ref)
        else:
            mask = has_ref.to(device=device).bool().view(-1)
            if torch.any(mask):
                L_MSE = F.mse_loss(pred_rate_log[mask], rate_ref[mask])
            else:
                L_MSE = rate_hat.new_tensor(0.0)
    else:
        L_MSE = rate_hat.new_tensor(0.0)

    # 非负约束
    neg_part = torch.clamp(-rate_hat, min=0.0)
    L_nonneg = torch.mean(neg_part ** 2)

    # 平滑约束 (使用归一化率值避免大数值溢出)
    if T > 1:
        # 归一化: 除以最大值 + epsilon，使差分在合理范围
        rate_max = rate_hat.abs().max() + 1e-12
        rate_normalized = rate_hat / rate_max
        diff = rate_normalized[:, 1:] - rate_normalized[:, :-1]
        L_smooth = torch.mean(diff ** 2)
    else:
        L_smooth = rate_hat.new_tensor(0.0)

    # 物理约束: 震级一致性 (EEW_0012 Equation 10)
    L_physics = rate_hat.new_tensor(0.0)
    if true_mag is not None:
        dt_scalar = float(dt_b[0].item()) if dt_b.dim() > 0 else float(dt)
        M0_seq = torch.cumsum(torch.clamp(rate_hat, min=0.0), dim=1) * dt_scalar
        M0_peak = torch.max(M0_seq, dim=1).values
        Mw_pred = compute_moment_magnitude(M0_peak)
        L_physics = F.mse_loss(Mw_pred, true_mag.view(-1))

    # 形状损失: 归一化后比较 STF 形状
    L_shape = rate_hat.new_tensor(0.0)
    if rate_ref is not None:
        # 解码 rate_ref 到原始空间进行形状比较
        rate_ref_safe = torch.nan_to_num(rate_ref, nan=0.0, posinf=0.0, neginf=0.0)
        rate_ref_safe = torch.clamp(rate_ref_safe, min=-20.0, max=6.0)
        rate_ref_decoded = stf_m_ref * (torch.pow(10.0, rate_ref_safe) - 1.0)
        rate_ref_decoded = torch.clamp(rate_ref_decoded, min=0.0)
        if has_ref is not None:
            mask = has_ref.to(device=device).bool().view(-1)
            if torch.any(mask):
                L_shape = compute_shape_loss(rate_hat[mask], rate_ref_decoded[mask])
        else:
            L_shape = compute_shape_loss(rate_hat, rate_ref_decoded)

    # 总损失
    total_loss = (
        float(lambda_MSE) * L_MSE
        + float(lambda_wave) * L_wave
        + float(lambda_nonneg) * L_nonneg
        + float(lambda_smooth) * L_smooth
        + float(lambda_physics) * L_physics
        + float(lambda_shape) * L_shape
    )

    loss_dict = {
        "L_total": float(total_loss.detach().cpu()),
        "L_MSE": float(L_MSE.detach().cpu()),
        "L_wave": float(L_wave.detach().cpu()),
        "L_nonneg": float(L_nonneg.detach().cpu()),
        "L_smooth": float(L_smooth.detach().cpu()),
        "L_physics": float(L_physics.detach().cpu()),
        "L_shape": float(L_shape.detach().cpu()),
    }
    return total_loss, loss_dict


# =============================================================================
# 封装类: STFRateWaveformLoss
# =============================================================================

class STFRateWaveformLoss(nn.Module):
    """
    基于物理的 STF 矩率损失函数 (EEW_0012 实现)
    
    特性:
        - 支持中场项 (Intermediate-field)
        - 支持简化/完整辐射花型
        - 保留积分方式
    """
    
    def __init__(self, config: dict[str, Any]):
        super().__init__()
        self.config = config
        train_cfg = (config.get("training", {}) or {})
        ds_cfg = (config.get("dataset", {}) or {})
        phys_cfg = (config.get("physics", {}) or {})
        stf_cfg = (train_cfg.get("stf_rate_loss", {}) or {})

        # 损失权重
        self.lambda_MSE = float(stf_cfg.get("lambda_MSE", 1.0))
        self.lambda_wave = float(stf_cfg.get("lambda_wave", 0.5))
        self.lambda_nonneg = float(stf_cfg.get("lambda_nonneg", 1.0))
        self.lambda_smooth = float(stf_cfg.get("lambda_smooth", 0.1))
        self.lambda_physics = float(stf_cfg.get("lambda_physics", 0.1))
        self.lambda_shape = float(stf_cfg.get("lambda_shape", 0.1))

        # 矩率表示方式
        self.rate_representation = str(train_cfg.get("rate_representation", "auto")).lower()
        self.stf_m_ref = float(ds_cfg.get("stf_m_ref", 1.0e18))
        if self.rate_representation == "auto":
            self.rate_representation = "log1p" if "stf_m_ref" in ds_cfg else "linear"

        # 物理参数
        self.rho = float(phys_cfg.get("rho", 3400.0))
        self.alpha = float(phys_cfg.get("alpha", 7900.0))
        self.beta = float(phys_cfg.get("beta", 4500.0))
        self.geom = float(phys_cfg.get("geometrical_spreading_factor", 1.0))
        self.free_surface = float(phys_cfg.get("free_surface_factor", 1.0))
        self.attenuation = float(phys_cfg.get("attenuation", 1.0))

        # 新增配置: 中场项和辐射模式
        self.include_intermediate = bool(stf_cfg.get("include_intermediate_field", True))
        self.radiation_mode = str(stf_cfg.get("radiation_pattern_mode", "simplified")).lower()
        self.skip_travel_delays = bool(stf_cfg.get("skip_travel_delays", True))

    def _decode_rate(self, pred_rate: torch.Tensor) -> torch.Tensor:
        """将网络输出解码为真实矩率值"""
        if self.rate_representation == "log1p":
            denom = max(float(self.stf_m_ref), 1.0e-30)
            pred_rate_safe = torch.nan_to_num(pred_rate, nan=0.0, posinf=0.0, neginf=0.0)
            pred_rate_safe = torch.clamp(pred_rate_safe, min=-20.0, max=6.0)
            dot_m0 = denom * (torch.pow(10.0, pred_rate_safe) - 1.0)
            return torch.clamp(dot_m0, min=0.0)
        pred_rate_safe = torch.nan_to_num(pred_rate, nan=0.0, posinf=0.0, neginf=0.0)
        return torch.clamp(pred_rate_safe, min=0.0)

    def forward(
        self,
        pred_rate: torch.Tensor,
        radial_obs: torch.Tensor,
        r_m: torch.Tensor,
        theta_deg: torch.Tensor,
        phi_deg: torch.Tensor,
        dt: float,
        stf_true: torch.Tensor | None = None,
        has_stf: torch.Tensor | None = None,
        true_mag: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """
        前向计算损失
        
        参数:
            pred_rate: (B, T) 网络预测的矩率（可能是 log1p 编码）
            radial_obs: (B, T) 或 (B, 1, T) 观测径向位移
            r_m: (B,) 震中距 (米)
            theta_deg, phi_deg: (B,) 辐射角度
            dt: 采样间隔
            stf_true: (B, T) 参考 STF（可选）
            has_stf: (B,) 是否有 STF 的标记
        """
        # 处理输入维度
        if radial_obs.dim() == 3:
            u_obs = radial_obs.squeeze(1)
        else:
            u_obs = radial_obs
        if u_obs.dim() != 2:
            raise ValueError("radial_obs 维度应为 [B,T] 或 [B,1,T]")

        # 解码矩率 (用于物理正演)
        rate_hat = self._decode_rate(pred_rate)

        # 准备参考 STF (转换为 log1p 空间以匹配网络输出)
        rate_ref_log = None
        has_ref = None
        if stf_true is not None:
            if has_stf is None:
                # 将 stf_true 转换为 log1p 编码
                stf_nonneg = torch.clamp(stf_true, min=0.0)
                denom = max(float(self.stf_m_ref), 1.0e-30)
                rate_ref_log = torch.log10(1.0 + stf_nonneg / denom)
            else:
                has_ref = has_stf.view(-1).bool()
                if torch.any(has_ref):
                    stf_nonneg = torch.clamp(stf_true, min=0.0)
                    denom = max(float(self.stf_m_ref), 1.0e-30)
                    rate_ref_log = torch.log10(1.0 + stf_nonneg / denom)

        # 调用核心损失函数
        # 注意: 传入 pred_rate (log1p 空间) 和 rate_ref_log (log1p 空间) 用于 MSE
        #       传入 rate_hat (原始空间) 用于物理正演计算 L_wave
        total_loss, loss_dict = pinn_loss_stf_rate(
            rate_hat=rate_hat,              # 原始尺度，用于物理正演
            rate_ref=rate_ref_log,          # log1p 尺度，用于 MSE
            u_obs=u_obs,
            dt=float(dt),
            r=r_m,
            alpha=self.alpha,
            beta=self.beta,
            rho=self.rho,
            theta_deg=theta_deg,
            phi_deg=phi_deg,
            geom=self.geom,
            free_surface=self.free_surface,
            attenuation=self.attenuation,
            lambda_MSE=self.lambda_MSE,
            lambda_wave=self.lambda_wave,
            lambda_nonneg=self.lambda_nonneg,
            lambda_smooth=self.lambda_smooth,
            lambda_physics=self.lambda_physics,
            lambda_shape=self.lambda_shape,
            has_ref=has_ref,
            include_intermediate=self.include_intermediate,
            radiation_mode=self.radiation_mode,
            pred_rate_log=pred_rate,
            true_mag=true_mag,
            stf_m_ref=self.stf_m_ref,
            skip_delays=self.skip_travel_delays,
        )
        return total_loss, loss_dict

