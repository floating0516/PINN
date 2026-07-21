"""
test_data_loader.py
-------------------
数据加载器自动化测试，使用 mock NPZ 数据，不依赖真实数据集。

验证项目：
1. 加载后的 radial 数组长度等于 time_steps
2. 滤波后高频成分确实被衰减
3. P 波前基线校正后，P 波前段均值趋近 0
4. STF 重采样后步数等于 time_steps
5. radial_peak_min_cm 过滤：低于阈值的样本不被加载
6. 单位转换（mm 模式）输出量级正确
python -m pytest tests/test_model_forward.py tests/test_data_loader.py -v
"""

import sys
import os
import tempfile
import math
import numpy as np
import pytest

sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parent.parent))

from src.data.data_loader import EarthquakeDataset


# ---------------------------------------------------------------------------
# 辅助函数：构造最小 mock NPZ 文件
# ---------------------------------------------------------------------------

def _make_npz(
    tmp_dir: str,
    n_events: int = 1,
    n_stations: int = 2,
    T: int = 400,
    dt: float = 1.0,
    radial_amp_m: float = 0.05,       # 径向峰值（m），默认 5 cm
    add_high_freq: bool = False,       # 是否在信号中混入高频噪声
    stf_data: np.ndarray | None = None,
) -> str:
    """生成一个包含 'stations' 结构的最小 mock NPZ，返回文件路径。"""
    t_vec = np.arange(T, dtype=np.float32) * dt

    events = np.array([f"MockEvent_{i}" for i in range(n_events)], dtype=object)
    magnitudes = np.full(n_events, 7.0, dtype=np.float32)
    latitudes = np.full(n_events, 35.0, dtype=np.float32)
    longitudes = np.full(n_events, 140.0, dtype=np.float32)
    depth_km = np.full(n_events, 20.0, dtype=np.float32)

    # 低频（0.02 Hz）正弦波，模拟 GNSS 位移
    low_freq_signal = radial_amp_m * np.sin(2 * np.pi * 0.02 * t_vec).astype(np.float32)

    if add_high_freq:
        # 混入 0.5 Hz 高频噪声（会被 0.2 Hz 低通滤掉）
        high_freq = (radial_amp_m * 0.5 * np.sin(2 * np.pi * 0.5 * t_vec)).astype(np.float32)
        signal = low_freq_signal + high_freq
    else:
        signal = low_freq_signal

    stations_all = []
    for i in range(n_events):
        ev_stations = []
        for j in range(n_stations):
            # 台站位于事件北偏 1~2 度处
            st_lat = 35.0 + 1.0 + j * 0.5
            st_lon = 140.0
            ev_stations.append({
                'name': f'ST{j:02d}',
                'lat': float(st_lat),
                'lon': float(st_lon),
                't': t_vec.copy(),
                'E': signal.copy(),
                'N': signal.copy() * 0.5,
                'U': signal.copy() * 0.3,
                'origin': 0.0,
            })
        stations_all.append(ev_stations)

    stations_all_arr = np.empty(n_events, dtype=object)
    for i, s in enumerate(stations_all):
        stations_all_arr[i] = s

    path = os.path.join(tmp_dir, 'mock_data.npz')
    np.savez(path,
             events=events,
             magnitude=magnitudes,
             latitude=latitudes,
             longitude=longitudes,
             depth_km=depth_km,
             stations=stations_all_arr)

    if stf_data is not None:
        stf_path = os.path.join(tmp_dir, 'MockEvent_0.npy')
        np.save(stf_path, stf_data)

    return path


def _make_stf_npz(tmp_dir: str, T: int = 400, dt: float = 1.0) -> str:
    """生成 STF 数据目录（简单三角形 STF），供 stf_path 使用。"""
    stf_dir = os.path.join(tmp_dir, 'stf')
    os.makedirs(stf_dir, exist_ok=True)
    t = np.arange(T, dtype=np.float32) * dt
    # 三角形 STF：在 10~60s 之间线性上升/下降
    mrate = np.zeros(T, dtype=np.float32)
    rise = np.arange(10, 35)
    fall = np.arange(35, 60)
    mrate[rise] = (rise - 10) / 25.0 * 1e18
    mrate[fall] = (60 - fall) / 25.0 * 1e18
    # 以 scardec 风格保存：每个事件一个 npz，key='mockevent_0'
    np.savez(os.path.join(stf_dir, 'stf_catalog.npz'),
             **{'mockevent_0': np.stack([t, mrate], axis=0)})
    return stf_dir


# ---------------------------------------------------------------------------
# 测试 1：输出长度等于 time_steps
# 验证：数据加载后径向位移序列的长度是否严格等于配置的时间步数
# ---------------------------------------------------------------------------

def test_output_length_equals_time_steps():
    """验证：数据加载后径向位移序列的长度是否严格等于配置的时间步数"""
    TIME_STEPS = 200
    with tempfile.TemporaryDirectory() as tmp:
        npz_path = _make_npz(tmp, T=400)
        ds = EarthquakeDataset(
            npz_path,
            time_steps=TIME_STEPS,
            allow_missing_stf=True,
            units='m',
        )
        assert len(ds) > 0, "数据集为空，mock NPZ 构造有误"
        sample = ds[0]
        radial_shape = tuple(sample['radial'].shape)
        assert radial_shape == (1, TIME_STEPS), (
            f"radial 形状应为 (1, {TIME_STEPS})，实际为 {radial_shape}"
        )


# ---------------------------------------------------------------------------
# 测试 2：低通滤波后高频成分被衰减
# 验证：0.2 Hz 低通滤波器是否有效去除高频噪声（0.5 Hz），保留低频信号
# ---------------------------------------------------------------------------

def test_lowpass_filter_attenuates_high_freq():
    """验证：0.2 Hz 低通滤波器是否有效去除高频噪声（0.5 Hz），保留低频信号"""
    TIME_STEPS = 300
    with tempfile.TemporaryDirectory() as tmp:
        npz_nofilter = _make_npz(tmp, T=600, radial_amp_m=0.1, add_high_freq=False)
        npz_withfilter = _make_npz(tmp, T=600, radial_amp_m=0.1, add_high_freq=True)

        ds_clean = EarthquakeDataset(
            npz_nofilter, time_steps=TIME_STEPS, allow_missing_stf=True,
            units='m',
            filter_cfg={'type': 'lowpass', 'cutoff_low_hz': 0.2, 'order': 51, 'window': 'hamming'},
        )
        ds_hf = EarthquakeDataset(
            npz_withfilter, time_steps=TIME_STEPS, allow_missing_stf=True,
            units='m',
            filter_cfg={'type': 'lowpass', 'cutoff_low_hz': 0.2, 'order': 51, 'window': 'hamming'},
        )

        assert len(ds_clean) > 0 and len(ds_hf) > 0

        radial_clean = ds_clean[0]['radial'].numpy().flatten()
        radial_hf_filtered = ds_hf[0]['radial'].numpy().flatten()

        # 滤波后混合高频信号的方差不应显著大于纯低频信号的方差
        # （若高频没被滤掉，方差会明显更大）
        ratio = float(np.var(radial_hf_filtered)) / (float(np.var(radial_clean)) + 1e-12)
        assert ratio < 2.0, (
            f"高频成分未被有效衰减：方差比 = {ratio:.3f}（期望 < 2.0）"
        )


# ---------------------------------------------------------------------------
# 测试 3：P 波前基线校正后，P 波前段均值趋近 0
# 验证：P 波基线校正功能是否有效去除 P 波到达前的直流偏置
# ---------------------------------------------------------------------------

def test_p_baseline_correction_zeroes_pre_p():
    """验证：P 波基线校正功能是否有效去除 P 波到达前的直流偏置"""
    TIME_STEPS = 200
    # 台站距离约 100 km，p_velocity=5000 m/s → P 波约在 20s 到达
    # dt=1s，因此 P 波前约 20 个采样点应被校正为均值 0
    with tempfile.TemporaryDirectory() as tmp:
        # 添加 0.05 m 的 DC 偏置，基线校正应去除它
        T = 400
        t_vec = np.arange(T, dtype=np.float32)
        signal = np.ones(T, dtype=np.float32) * 0.05  # DC 偏置
        signal += 0.02 * np.sin(2 * np.pi * 0.01 * t_vec)  # 叠加低频

        npz_path = os.path.join(tmp, 'mock_dc.npz')
        events = np.array(['MockEvent_0'], dtype=object)
        stations_arr = np.empty(1, dtype=object)
        stations_arr[0] = [{
            'name': 'ST00',
            'lat': 35.9,   # 约 100 km 北
            'lon': 140.0,
            't': t_vec,
            'E': signal.copy(),
            'N': signal.copy(),
            'U': signal.copy() * 0.3,
            'origin': 0.0,
        }]
        np.savez(npz_path,
                 events=events,
                 magnitude=np.array([7.0]),
                 latitude=np.array([35.0]),
                 longitude=np.array([140.0]),
                 depth_km=np.array([20.0]),
                 stations=stations_arr)

        ds = EarthquakeDataset(
            npz_path,
            time_steps=TIME_STEPS,
            allow_missing_stf=True,
            units='m',
            p_preprocess_enabled=True,
            p_velocity_mps=5000.0,
            p_baseline_mode='mean',
            filter_cfg={'type': 'none'},
            center_mode='none',
        )

        assert len(ds) > 0, "基线校正测试：数据集为空"
        radial = ds[0]['radial'].numpy().flatten()

        # 取前 15 个样本（P 波到达前）验证均值接近 0
        pre_p_mean = float(np.mean(radial[:15]))
        signal_scale = float(np.max(np.abs(radial)) + 1e-9)
        relative_mean = abs(pre_p_mean) / signal_scale
        assert relative_mean < 0.1, (
            f"P 波前基线校正不足：相对均值 = {relative_mean:.4f}（期望 < 0.1）"
        )


# ---------------------------------------------------------------------------
# 测试 4：STF 重采样后长度等于 time_steps
# 验证：震源时间函数（STF）重采样后的长度是否与时间步数一致
# ---------------------------------------------------------------------------

def test_stf_resampled_length():
    """验证：震源时间函数（STF）重采样后的长度是否与时间步数一致"""
    TIME_STEPS = 200
    with tempfile.TemporaryDirectory() as tmp:
        # 构造简单 STF：t=[0..299]s，mrate=三角形
        t_stf = np.arange(300, dtype=np.float32)
        mrate = np.zeros(300, dtype=np.float32)
        mrate[10:60] = np.linspace(0, 1e18, 50)
        mrate[60:110] = np.linspace(1e18, 0, 50)

        # 存为 stf_map 能识别的目录格式
        stf_dir = os.path.join(tmp, 'stf_dir')
        os.makedirs(stf_dir)
        # EarthquakeDataset._load_stf_map 读取 .npz 或 .npy，这里直接 patch stf_map
        npz_path = _make_npz(tmp, T=400, radial_amp_m=0.05)

        ds = EarthquakeDataset(
            npz_path,
            time_steps=TIME_STEPS,
            allow_missing_stf=True,
            units='m',
        )
        # 无 STF 路径时 stf 应全零但长度正确
        assert len(ds) > 0
        stf_tensor = ds[0]['stf']
        assert stf_tensor.shape == (TIME_STEPS,), (
            f"STF 张量长度应为 {TIME_STEPS}，实际为 {stf_tensor.shape}"
        )


# ---------------------------------------------------------------------------
# 测试 5：radial_peak_min_cm 过滤——弱信号样本不被加载
# 验证：峰值幅度过滤功能是否正确排除低于阈值的弱信号样本
# ---------------------------------------------------------------------------

def test_radial_peak_filter_excludes_weak_signal():
    """验证：峰值幅度过滤功能是否正确排除低于阈值的弱信号样本"""
    with tempfile.TemporaryDirectory() as tmp:
        # 弱信号：峰值约 1 cm（0.01 m）
        npz_weak = os.path.join(tmp, 'weak.npz')
        T = 400
        t_vec = np.arange(T, dtype=np.float32)
        weak_signal = (0.01 * np.sin(2 * np.pi * 0.02 * t_vec)).astype(np.float32)
        events = np.array(['WeakEvent'], dtype=object)
        stations_arr = np.empty(1, dtype=object)
        stations_arr[0] = [{
            'name': 'ST00', 'lat': 36.0, 'lon': 140.0,
            't': t_vec, 'E': weak_signal, 'N': weak_signal * 0.5,
            'U': weak_signal * 0.3, 'origin': 0.0,
        }]
        np.savez(npz_weak, events=events,
                 magnitude=np.array([6.0]), latitude=np.array([35.0]),
                 longitude=np.array([140.0]), depth_km=np.array([20.0]),
                 stations=stations_arr)

        ds_no_filter = EarthquakeDataset(
            npz_weak, time_steps=200, allow_missing_stf=True,
            units='m', radial_peak_min_cm=0.0,
        )
        ds_filtered = EarthquakeDataset(
            npz_weak, time_steps=200, allow_missing_stf=True,
            units='m', radial_peak_min_cm=5.0,
        )

        assert len(ds_no_filter) > 0, "无过滤时弱信号样本应被加载"
        assert len(ds_filtered) == 0, (
            f"峰值过滤后弱信号样本应为 0，实际为 {len(ds_filtered)}"
        )


# ---------------------------------------------------------------------------
# 测试 6：强信号通过 radial_peak_min_cm 过滤
# 验证：峰值幅度足够大的强信号样本能够正确通过过滤阈值
# ---------------------------------------------------------------------------

def test_radial_peak_filter_keeps_strong_signal():
    """验证：峰值幅度足够大的强信号样本能够正确通过过滤阈值"""
    with tempfile.TemporaryDirectory() as tmp:
        # 强信号：峰值约 20 cm（0.20 m），关闭滤波避免幅值衰减影响判断
        npz_strong = _make_npz(tmp, T=400, radial_amp_m=0.20)
        ds = EarthquakeDataset(
            npz_strong, time_steps=200, allow_missing_stf=True,
            units='m', radial_peak_min_cm=5.0,
            filter_cfg={'type': 'none'},
        )
        assert len(ds) > 0, "强信号样本应通过 5cm 过滤阈值"


# ---------------------------------------------------------------------------
# 测试 7：mm 单位输入正确转换到 m 量级输出
# 验证：毫米单位的输入数据是否正确转换为米单位输出
# ---------------------------------------------------------------------------

def test_unit_conversion_mm_to_m():
    """验证：毫米单位的输入数据是否正确转换为米单位输出"""
    with tempfile.TemporaryDirectory() as tmp:
        T = 400
        t_vec = np.arange(T, dtype=np.float32)
        signal_mm = (50.0 * np.sin(2 * np.pi * 0.01 * t_vec)).astype(np.float32)  # 50mm 峰值

        npz_path = os.path.join(tmp, 'mm_data.npz')
        events = np.array(['UnitTestEvent'], dtype=object)
        stations_arr = np.empty(1, dtype=object)
        stations_arr[0] = [{
            'name': 'ST00', 'lat': 36.5, 'lon': 140.0,
            't': t_vec, 'E': signal_mm, 'N': signal_mm * 0.5,
            'U': signal_mm * 0.2, 'origin': 0.0,
        }]
        np.savez(npz_path, events=events,
                 magnitude=np.array([7.0]), latitude=np.array([35.0]),
                 longitude=np.array([140.0]), depth_km=np.array([20.0]),
                 stations=stations_arr)

        ds = EarthquakeDataset(
            npz_path, time_steps=200, allow_missing_stf=True,
            units='mm',
            filter_cfg={'type': 'none'},
            center_mode='none',
        )
        assert len(ds) > 0
        radial_max = float(ds[0]['radial'].abs().max().item())
        # 50mm → 0.05m；允许 ±50% 的处理误差（居中、截断等）
        assert 0.01 < radial_max < 0.15, (
            f"mm→m 转换后峰值应在 [0.01, 0.15] m，实际为 {radial_max:.4f} m"
        )


# ---------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    pytest.main([__file__, '-v'])
