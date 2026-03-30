import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from pathlib import Path
import yaml
import math
from typing import Tuple, Dict, Any, List

class EarthquakeDataset(Dataset):
    """
    事件级 GNSS 位移数据集，负责：
    - 读取 NPZ 中的 E/N/U 三分量；
    - 将 E/N 投影为径向分量（主输入）；
    - 提供垂直分量与震中距给物理约束项；
    - 序列统一长度（填充/截断）。
    
    设计动机（Why）
    - 震级与位移幅值相关，但网络训练更稳定于统一长度与合理数值尺度。
    """

    def __init__(self, npz_path: str | Path, time_steps: int = 256, blacklist: List[str] | None = None, units: str = "auto", center_mode: str = "median", window_min_sec: float = 0.0, window_max_sec: float = 600.0, stf_path: str | None = None, stf_m_ref: float = 1.0e18, default_theta_deg: float = 45.0, default_phi_deg: float = 0.0, filter_cfg: Dict[str, Any] | None = None, p_preprocess_enabled: bool = False, p_velocity_mps: float = 7900.0, p_arrival_offset_sec: float = 0.0, p_baseline_mode: str = "mean", allow_missing_stf: bool = False, radial_peak_min_cm: float = 0.0):
        self.time_steps = time_steps
        self.samples = []
        self.blacklist = set([str(x) for x in (blacklist or [])])
        self.units = units
        self.center_mode = center_mode
        self.window_min_sec = float(window_min_sec)
        self.window_max_sec = float(window_max_sec)
        self.stf_path = stf_path
        self.stf_m_ref = float(stf_m_ref) if float(stf_m_ref) > 0.0 else 1.0e18
        self.default_theta_deg = float(default_theta_deg)
        self.default_phi_deg = float(default_phi_deg)
        self.stf_map = self._load_stf_map(stf_path) if stf_path else {}
        fcfg = filter_cfg or {}
        self.filter_type = str(fcfg.get('type', 'none')).lower()
        self.cutoff_low_hz = float(fcfg.get('cutoff_low_hz', 0.5))
        self.cutoff_high_hz = float(fcfg.get('cutoff_high_hz', 2.0))
        self.filter_order = int(fcfg.get('order', 51))
        self.filter_window = str(fcfg.get('window', 'hamming')).lower()
        self.p_preprocess_enabled = bool(p_preprocess_enabled)
        self.p_velocity_mps = float(p_velocity_mps)
        self.p_arrival_offset_sec = float(p_arrival_offset_sec)
        self.p_baseline_mode = str(p_baseline_mode).lower()
        self.allow_missing_stf = bool(allow_missing_stf)
        self.radial_peak_min_cm = float(radial_peak_min_cm)
        self._load_data(npz_path)
        
    def _load_data(self, npz_path: str | Path) -> None:
        print(f"Loading data from {npz_path}...")
        try:
            data = np.load(npz_path, allow_pickle=True)
        except FileNotFoundError:
            print(f"Error: File {npz_path} not found.")
            return

        events = data['events']
        magnitudes = data['magnitude']
        event_lats = data['latitude']
        event_lons = data['longitude']
        mechanism_values = None
        for k in ['mechanism', 'fault_type', 'fm_type', 'source_mechanism', 'focal_mechanism', 'mech']:
            if k in data:
                mechanism_values = data[k]
                break
        rake_values = None
        for k in ['rake', 'rake_deg']:
            if k in data:
                rake_values = data[k]
                break

        # 兼容不同数据结构：优先使用旧版 'enu'/'station_info'，否则尝试新版 'stations'
        has_enu = 'enu' in data and 'station_info' in data
        has_stations = 'stations' in data
        depths_km = data['depth_km'] if 'depth_km' in data else np.full_like(magnitudes, np.nan)

        loaded_events = 0
        loaded_samples = 0
        skipped_nan = 0
        skipped_radial_peak = 0

        def _normalize_station_info(info_obj):
            """将 station_info 正规化为 {name: {lat, lon}} 的映射"""
            if isinstance(info_obj, dict):
                return info_obj
            if isinstance(info_obj, list):
                mapping = {}
                for idx, st in enumerate(info_obj):
                    name = st.get('name') or st.get('station') or st.get('id') or f'st_{idx}'
                    lat = st.get('lat') if 'lat' in st else st.get('latitude')
                    lon = st.get('lon') if 'lon' in st else st.get('longitude')
                    mapping[name] = {'lat': lat, 'lon': lon}
                return mapping
            return {}

        def _iter_stations_container(container):
            """按统一接口返回 (name, payload) 可迭代序列，兼容 dict 或 list"""
            if isinstance(container, dict):
                return container.items()
            if isinstance(container, list):
                return [(st.get('name', f'st_{i}'), st) for i, st in enumerate(container)]
            return []

        def _get_field(payload, keys):
            """从 payload 中按候选键集合取值"""
            for k in keys:
                if isinstance(payload, dict) and k in payload:
                    return payload[k]
            return None
        
        def _normalize_name(s: str) -> str:
            s = str(s)
            s = s.lower()
            keep = []
            for ch in s:
                if ch.isalnum() or ch in [' ', '-', '_']:
                    keep.append(ch)
            return "".join(keep).replace("  ", " ").strip()

        def _mechanism_to_code(val) -> int:
            if val is None:
                return -1
            try:
                if isinstance(val, (np.integer, int)):
                    iv = int(val)
                    if iv in [0, 1, 2]:
                        return iv
                    if iv in [1, 2, 3]:
                        return iv - 1
            except Exception:
                pass
            try:
                if isinstance(val, (bytes, np.bytes_)):
                    val = val.decode('utf-8', errors='ignore')
            except Exception:
                pass
            s = str(val).strip().lower()
            s = s.replace('_', '').replace('-', '').replace(' ', '')
            if ('normal' in s) or ('nf' == s) or ('正断' in s) or ('正斷' in s):
                return 0
            if ('strike' in s) or ('strikeslip' in s) or ('ss' == s) or ('走滑' in s):
                return 1
            if ('reverse' in s) or ('thrust' in s) or ('rv' == s) or ('逆冲' in s) or ('逆衝' in s) or ('冲断' in s) or ('衝斷' in s):
                return 2
            return -1

        def _rake_to_code(val) -> int:
            try:
                rake = float(val)
            except Exception:
                return -1
            if not np.isfinite(rake):
                return -1
            rake = ((rake + 180.0) % 360.0) - 180.0
            if (abs(rake) <= 30.0) or (abs(rake) >= 150.0):
                return 1
            if rake >= 30.0 and rake <= 150.0:
                return 2
            if rake <= -30.0 and rake >= -150.0:
                return 0
            return -1
        
        def _match_stf_for_event(event_name: str) -> tuple[np.ndarray, np.ndarray] | None:
            if not self.stf_map:
                return None
            key = _normalize_name(event_name)
            if key in self.stf_map:
                return self.stf_map[key]
            for k, val in self.stf_map.items():
                if key in k or k in key:
                    return val
            return None
        
        def _resample_stf_to(dt: float, T: int, stf_pair: tuple[np.ndarray, np.ndarray] | None, p_shift_sec: float = 0.0) -> np.ndarray | None:
            if stf_pair is None:
                return None
            t_src, mrate_src = stf_pair
            if t_src is None or mrate_src is None or len(t_src) == 0:
                return None
            # 对 STF：统一降采样到 1Hz，且仅使用震后（t>=0）的数据
            mask_src = np.isfinite(t_src) & np.isfinite(mrate_src)
            t_src = t_src[mask_src]
            mrate_src = mrate_src[mask_src]
            if len(t_src) == 0:
                return None
            # 丢弃 t<0，与数据窗起点对齐为 0s
            keep = t_src >= 0.0
            t_src = t_src[keep]
            mrate_src = mrate_src[keep]
            if len(t_src) == 0:
                return None
            # 排序并去重以便插值
            idx = np.argsort(t_src)
            t_src = t_src[idx]
            mrate_src = mrate_src[idx]
            # 目标为 1Hz 网格
            t_dst = np.arange(T, dtype=np.float32) * 1.0
            t_min = float(np.nanmin(t_src))
            t_max = float(np.nanmax(t_src))
            mask = (t_dst >= t_min) & (t_dst <= t_max)
            out = np.zeros(T, dtype=np.float32)
            if np.any(mask):
                out[mask] = np.interp(t_dst[mask], t_src, mrate_src)
            if p_shift_sec and p_shift_sec > 0.0:
                ip = int(round(p_shift_sec))
                if ip >= T:
                    out = np.zeros(T, dtype=np.float32)
                elif ip > 0:
                    shifted = np.zeros(T, dtype=np.float32)
                    shifted[ip:] = out[:T - ip]
                    out = shifted
            return out
        
        def _slice_after_origin(t_vec: np.ndarray, series_map: Dict[str, np.ndarray], origin_val: float | None) -> tuple[np.ndarray, Dict[str, np.ndarray]]:
            """
            根据发震时刻截取时间窗（t>=origin），若未提供 origin 则尝试依据 t 是否跨越 0 进行判定。
            返回截取后的 t 与各分量。
            """
            if t_vec is None or len(t_vec) == 0:
                return t_vec, series_map
            if origin_val is not None:
                mask = t_vec >= origin_val
            else:
                t_min = float(np.nanmin(t_vec))
                t_max = float(np.nanmax(t_vec))
                if t_min < 0.0 and t_max > 0.0:
                    mask = t_vec >= 0.0
                else:
                    mask = np.ones_like(t_vec, dtype=bool)
            t_cut = t_vec[mask]
            cut_map = {}
            for k, v in series_map.items():
                try:
                    cut_map[k] = v[mask]
                except Exception:
                    cut_map[k] = v
            return t_cut, cut_map
        
        def _apply_time_window(t_vec: np.ndarray, series_map: Dict[str, np.ndarray], tmin: float, tmax: float) -> tuple[np.ndarray, Dict[str, np.ndarray]]:
            if t_vec is None or len(t_vec) == 0:
                return t_vec, series_map
            mask = (t_vec >= tmin) & (t_vec <= tmax)
            if not np.any(mask):
                return t_vec[:0], {k: v[:0] for k, v in series_map.items()}
            t_cut = t_vec[mask]
            cut_map = {}
            for k, v in series_map.items():
                try:
                    cut_map[k] = v[mask]
                except Exception:
                    cut_map[k] = v
            return t_cut, cut_map
        
        def _all_finite_same_length(t_arr: np.ndarray, arrs: List[np.ndarray]) -> bool:
            if not (isinstance(t_arr, np.ndarray) and np.isfinite(t_arr).all()):
                return False
            L = len(t_arr)
            for a in arrs:
                if not (isinstance(a, np.ndarray) and len(a) == L and np.isfinite(a).all()):
                    return False
            return True
        
        def _passes_radial_peak_threshold(radial_processed: np.ndarray) -> bool:
            if not (self.radial_peak_min_cm and self.radial_peak_min_cm > 0.0):
                return True
            if radial_processed is None:
                return False
            try:
                radial_peak_val = float(np.nanmax(np.abs(radial_processed)))
            except Exception:
                return False
            if not np.isfinite(radial_peak_val):
                return False
            return radial_peak_val > (self.radial_peak_min_cm / 100.0)

        if has_enu:
            enu_data = data['enu']
            station_infos = data['station_info']
            for i, event_name in enumerate(events):
                if str(event_name) in self.blacklist:
                    continue
                ev_mag = magnitudes[i]
                ev_lat = event_lats[i]
                ev_lon = event_lons[i]
                if mechanism_values is not None:
                    ev_mechanism = _mechanism_to_code(mechanism_values[i])
                elif rake_values is not None:
                    ev_mechanism = _rake_to_code(rake_values[i])
                else:
                    ev_mechanism = -1
                stations_data = enu_data[i]
                stations_info = _normalize_station_info(station_infos[i])
                iter_stations = _iter_stations_container(stations_data)
                if len(list(iter_stations)) == 0:
                    continue
                loaded_events += 1
                for station_name, waveforms in iter_stations:
                    st_meta = stations_info.get(station_name, {})
                    st_lat = st_meta.get('lat', st_meta.get('latitude', np.nan))
                    st_lon = st_meta.get('lon', st_meta.get('longitude', np.nan))
                    if (np.isnan(st_lat) or np.isnan(st_lon)) and isinstance(waveforms, dict):
                        st_lat = waveforms.get('lat', waveforms.get('latitude', st_lat))
                        st_lon = waveforms.get('lon', waveforms.get('longitude', st_lon))
                    if np.isnan(st_lat) or np.isnan(st_lon):
                        continue
                    dist_m, azimuth, _ = self._calculate_geodetics(ev_lat, ev_lon, st_lat, st_lon)
                    t = _get_field(waveforms, ['t', 'time'])
                    e = _get_field(waveforms, ['E', 'east'])
                    n = _get_field(waveforms, ['N', 'north'])
                    u = _get_field(waveforms, ['U', 'up', 'vertical'])
                    origin_val = _get_field(waveforms, ['origin', 'origin_s', 'origin_time', 'origin_epoch', 'origin_ts', 't0', 'origin_sec'])
                    if t is None or e is None or n is None or u is None:
                        skipped_nan += 1
                        continue
                    if len(e) == 0 or len(n) == 0 or len(u) == 0:
                        skipped_nan += 1
                        continue
                    if not _all_finite_same_length(t, [e, n, u]):
                        skipped_nan += 1
                        continue
                    # 截取发震后时间窗
                    t, parts = _slice_after_origin(t, {'E': e, 'N': n, 'U': u}, origin_val)
                    t, parts = _apply_time_window(t, parts, self.window_min_sec, self.window_max_sec)
                    e, n, u = parts['E'], parts['N'], parts['U']
                    az_rad = math.radians(azimuth)
                    radial = e * math.sin(az_rad) + n * math.cos(az_rad)
                    radial_processed, dt = self._preprocess_waveform(t, radial)
                    vertical_processed, _ = self._preprocess_waveform(t, u)
                    if radial_processed is None or vertical_processed is None:
                        continue
                    if self.p_preprocess_enabled and np.isfinite(dist_m) and dt > 0.0 and self.p_velocity_mps > 0.0:
                        radial_processed, vertical_processed = self._apply_p_baseline(radial_processed, vertical_processed, dist_m, dt)
                    if not _passes_radial_peak_threshold(radial_processed):
                        skipped_radial_peak += 1
                        continue
                    # 角度近似：以震源深度与震中距估计入射角（度）
                    ev_depth_km = depths_km[i] if isinstance(depths_km, np.ndarray) else np.nan
                    if np.isfinite(ev_depth_km):
                        theta_deg = float(np.degrees(np.arctan2(dist_m, max(ev_depth_km * 1000.0, 1.0))))
                    else:
                        theta_deg = self.default_theta_deg
                    phi_deg = float(azimuth) if np.isfinite(azimuth) else self.default_phi_deg
                    # STF 标签重采样
                    stf_pair = _match_stf_for_event(str(events[i]))
                    p_shift = (dist_m / self.p_velocity_mps + self.p_arrival_offset_sec) if (np.isfinite(dist_m) and self.p_velocity_mps > 0.0) else 0.0
                    stf_resampled = _resample_stf_to(dt, self.time_steps, stf_pair, p_shift_sec=float(p_shift))
                    if stf_resampled is None and not self.allow_missing_stf:
                        continue
                    if stf_resampled is None:
                        stf_log = None
                    else:
                        stf_nonneg = np.maximum(stf_resampled, 0.0)
                        denom = max(float(self.stf_m_ref), 1.0e-30)
                        stf_log = np.log10(1.0 + stf_nonneg / denom).astype(np.float32)
                    self.samples.append({
                        'event': str(events[i]),
                        'event_index': int(i),
                        'station': str(station_name),
                        'radial': radial_processed,
                        'vertical': vertical_processed,
                        'distance': dist_m,
                        'magnitude': ev_mag,
                        'mechanism': ev_mechanism,
                        'dt': dt,
                        'theta_deg': theta_deg,
                        'phi_deg': phi_deg,
                        'stf': stf_resampled,
                        'stf_log': stf_log,
                        'has_stf': stf_resampled is not None,
                    })
                    loaded_samples += 1
        elif has_stations:
            stations_all = data['stations']
            for i, event_name in enumerate(events):
                if str(event_name) in self.blacklist:
                    continue
                ev_mag = magnitudes[i]
                ev_lat = event_lats[i]
                ev_lon = event_lons[i]
                if mechanism_values is not None:
                    ev_mechanism = _mechanism_to_code(mechanism_values[i])
                elif rake_values is not None:
                    ev_mechanism = _rake_to_code(rake_values[i])
                else:
                    ev_mechanism = -1
                event_stations = stations_all[i]
                # 支持两种结构：dict[station_name] -> {t,E,N,U,lat,lon} 或 list[ {name,lat,lon,t,E,N,U} ]
                iterable = _iter_stations_container(event_stations)
                if len(list(iterable)) == 0:
                    continue
                loaded_events += 1
                for station_name, st in iterable:
                    st_lat = st.get('lat', np.nan)
                    st_lon = st.get('lon', np.nan)
                    if np.isnan(st_lat) or np.isnan(st_lon):
                        st_lat = st.get('latitude', st_lat)
                        st_lon = st.get('longitude', st_lon)
                    if np.isnan(st_lat) or np.isnan(st_lon):
                        continue
                    dist_m, azimuth, _ = self._calculate_geodetics(ev_lat, ev_lon, st_lat, st_lon)
                    t = _get_field(st, ['t', 'time'])
                    e = _get_field(st, ['E', 'east'])
                    n = _get_field(st, ['N', 'north'])
                    u = _get_field(st, ['U', 'up', 'vertical'])
                    origin_val = _get_field(st, ['origin', 'origin_s', 'origin_time', 'origin_epoch', 'origin_ts', 't0', 'origin_sec'])
                    if t is None or e is None or n is None or u is None:
                        skipped_nan += 1
                        continue
                    if len(e) == 0 or len(n) == 0 or len(u) == 0:
                        skipped_nan += 1
                        continue
                    if not _all_finite_same_length(t, [e, n, u]):
                        skipped_nan += 1
                        continue
                    # 截取发震后时间窗
                    t, parts = _slice_after_origin(t, {'E': e, 'N': n, 'U': u}, origin_val)
                    t, parts = _apply_time_window(t, parts, self.window_min_sec, self.window_max_sec)
                    e, n, u = parts['E'], parts['N'], parts['U']
                    az_rad = math.radians(azimuth)
                    radial = e * math.sin(az_rad) + n * math.cos(az_rad)
                    radial_processed, dt = self._preprocess_waveform(t, radial)
                    vertical_processed, _ = self._preprocess_waveform(t, u)
                    if radial_processed is None or vertical_processed is None:
                        continue
                    if self.p_preprocess_enabled and np.isfinite(dist_m) and dt > 0.0 and self.p_velocity_mps > 0.0:
                        radial_processed, vertical_processed = self._apply_p_baseline(radial_processed, vertical_processed, dist_m, dt)
                    if not _passes_radial_peak_threshold(radial_processed):
                        skipped_radial_peak += 1
                        continue
                    ev_depth_km = depths_km[i] if isinstance(depths_km, np.ndarray) else np.nan
                    if np.isfinite(ev_depth_km):
                        theta_deg = float(np.degrees(np.arctan2(dist_m, max(ev_depth_km * 1000.0, 1.0))))
                    else:
                        theta_deg = self.default_theta_deg
                    phi_deg = float(azimuth) if np.isfinite(azimuth) else self.default_phi_deg
                    stf_pair = _match_stf_for_event(str(events[i]))
                    p_shift = (dist_m / self.p_velocity_mps + self.p_arrival_offset_sec) if (np.isfinite(dist_m) and self.p_velocity_mps > 0.0) else 0.0
                    stf_resampled = _resample_stf_to(dt, self.time_steps, stf_pair, p_shift_sec=float(p_shift))
                    if stf_resampled is None and not self.allow_missing_stf:
                        continue
                    if stf_resampled is None:
                        stf_log = None
                    else:
                        stf_nonneg = np.maximum(stf_resampled, 0.0)
                        denom = max(float(self.stf_m_ref), 1.0e-30)
                        stf_log = np.log10(1.0 + stf_nonneg / denom).astype(np.float32)
                    self.samples.append({
                        'event': str(events[i]),
                        'event_index': int(i),
                        'station': str(station_name),
                        'radial': radial_processed,
                        'vertical': vertical_processed,
                        'distance': dist_m,
                        'magnitude': ev_mag,
                        'mechanism': ev_mechanism,
                        'dt': dt,
                        'theta_deg': theta_deg,
                        'phi_deg': phi_deg,
                        'stf': stf_resampled,
                        'stf_log': stf_log,
                        'has_stf': stf_resampled is not None,
                    })
                    loaded_samples += 1
        else:
            print("Unsupported dataset structure: expected keys 'enu'/'station_info' or 'stations'.")

        print(f"Loaded {len(self.samples)} samples from {loaded_events} events. Skipped {skipped_nan} invalid samples (NaN/Inf/length mismatch). Skipped {skipped_radial_peak} samples by radial peak threshold.")

    def _load_stf_map(self, stf_path: str | None) -> Dict[str, tuple[np.ndarray, np.ndarray]]:
        if not stf_path:
            return {}
        p = Path(stf_path)
        if not p.exists() or not p.is_dir():
            return {}
        mapping: Dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for f in p.glob("*.stf"):
            try:
                t_list: List[float] = []
                m_list: List[float] = []
                with open(str(f), 'r', encoding='utf-8', errors='ignore') as fh:
                    for line in fh:
                        s = line.strip()
                        if not s:
                            continue
                        parts = s.replace('D', 'E').split()
                        vals: List[float] = []
                        for tok in parts:
                            try:
                                vals.append(float(tok))
                            except Exception:
                                vals = []
                                break
                        if len(vals) == 2:
                            t_list.append(vals[0])
                            m_list.append(vals[1])
                        else:
                            continue
                if len(t_list) == 0:
                    continue
                t = np.array(t_list, dtype=np.float32)
                mrate = np.array(m_list, dtype=np.float32)
                key = f.stem.lower()
                mapping[key] = (t, mrate)
            except Exception:
                continue
        return mapping

    def _calculate_geodetics(self, lat1: float, lon1: float, lat2: float, lon2: float) -> Tuple[float, float, float]:
        """
        计算震中距与方位角。
        
        原因（Why）
        - 径向投影与远场近似依赖源-台站几何关系；
        - 使用 Haversine 与大圆方位角具备足够工程精度。
        """
        
        R = 6371000.0 # Earth radius in meters
        phi1, phi2 = math.radians(lat1), math.radians(lat2)
        dphi = math.radians(lat2 - lat1)
        dlambda = math.radians(lon2 - lon1)
        
        # Distance
        a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2)**2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
        dist_m = R * c
        
        # Azimuth (Event -> Station)
        y = math.sin(dlambda) * math.cos(phi2)
        x = math.cos(phi1) * math.sin(phi2) - math.sin(phi1) * math.cos(phi2) * math.cos(dlambda)
        theta = math.atan2(y, x)
        azimuth = (math.degrees(theta) + 360) % 360
        
        # Back Azimuth (Station -> Event)
        back_azimuth = (azimuth + 180) % 360
        
        return dist_m, azimuth, back_azimuth

    def _apply_filter(self, data_mm: np.ndarray, dt: float) -> np.ndarray:
        if self.filter_type == 'none':
            return data_mm
        fs = 1.0 / max(dt, 1e-6)
        order = max(int(self.filter_order), 3)
        if order % 2 == 0:
            order += 1
        win = self.filter_window
        if win == 'hamming':
            w = np.hamming(order)
        elif win == 'hann' or win == 'hanning':
            w = np.hanning(order)
        else:
            w = np.ones(order)
        M = order - 1
        n = np.arange(order)
        def lp(cut_hz: float) -> np.ndarray:
            fc = max(min(cut_hz, fs * 0.49), 1e-6) / fs
            h = 2.0 * fc * np.sinc(2.0 * fc * (n - M / 2.0))
            h = h * w
            h = h / np.sum(h)
            return h
        if self.filter_type == 'lowpass':
            h = lp(self.cutoff_low_hz)
        elif self.filter_type == 'highpass':
            h_low = lp(self.cutoff_low_hz)
            h = -h_low
            h[M // 2] = h[M // 2] + 1.0
        elif self.filter_type == 'bandpass':
            h_high = lp(self.cutoff_high_hz)
            h_low = lp(self.cutoff_low_hz)
            h = h_high - h_low
        else:
            return data_mm
        y = np.convolve(data_mm, h.astype(np.float64), mode='same')
        return y.astype(np.float32)

    def _apply_p_baseline(self, radial_processed: np.ndarray, vertical_processed: np.ndarray, dist_m: float, dt: float) -> Tuple[np.ndarray, np.ndarray]:
        ip = int(round((dist_m / self.p_velocity_mps + self.p_arrival_offset_sec) / dt))
        ip = max(0, min(self.time_steps, ip))
        if ip > 0:
            src_r = radial_processed[:ip]
            src_v = vertical_processed[:ip]
            if self.p_baseline_mode == "median":
                b_r = float(np.nanmedian(src_r)) if len(src_r) > 0 else float(np.nanmedian(radial_processed))
                b_v = float(np.nanmedian(src_v)) if len(src_v) > 0 else float(np.nanmedian(vertical_processed))
            else:
                b_r = float(np.nanmean(src_r)) if len(src_r) > 0 else float(np.nanmean(radial_processed))
                b_v = float(np.nanmean(src_v)) if len(src_v) > 0 else float(np.nanmean(vertical_processed))
            radial_processed[:ip] = b_r
            vertical_processed[:ip] = b_v
        else:
            if self.p_baseline_mode == "median":
                b_r = float(np.nanmedian(radial_processed))
                b_v = float(np.nanmedian(vertical_processed))
            else:
                b_r = float(np.nanmean(radial_processed))
                b_v = float(np.nanmean(vertical_processed))
        radial_processed = radial_processed - b_r
        vertical_processed = vertical_processed - b_v
        return radial_processed, vertical_processed

    def _preprocess_waveform(self, t: np.ndarray, data: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        序列预处理：估计 dt、数值尺度调整与长度统一。
        
        原因（Why）
        - 幅值携带震级信息，避免逐样本归一导致幅值丢失；
        - 统一时间步利于批训练。
        """
        # 1）估计采样间隔
        if len(t) > 1:
            t_finite = t[np.isfinite(t)]
            if len(t_finite) > 1:
                dt = np.mean(np.diff(t_finite))
            else:
                dt = 1.0
        else:
            dt = 1.0 # Fallback
            
        # 2）单位转换到厘米
        amp = float(np.nanmax(np.abs(data))) if len(data) > 0 else 0.0
        if self.units == "m":
            data_mm = data 
        elif self.units == "cm":
            data_mm = data * 100.0
        elif self.units == "mm":
            data_mm = data * 0.001
        else:
            # auto：按幅值启发式
            if amp < 2.0:
                data_mm = data    # 典型米级（<2m）  
            elif amp < 200.0:
                data_mm = data / 100.0     # 典型厘米级（<200cm）
            else:
                data_mm = data / 1000.0           # 认为已是毫米
        
        # 2.5）居中（消除大偏移）
        if self.center_mode == "median":
            center = float(np.nanmedian(data_mm))
            data_mm = data_mm - center
        elif self.center_mode == "initial" and len(data_mm) > 0:
            data_mm = data_mm - float(data_mm[0])
        # 非有限值置零
        if isinstance(data_mm, np.ndarray):
            data_mm = np.where(np.isfinite(data_mm), data_mm, 0.0)
        
        # 3）滤波
        if isinstance(data_mm, np.ndarray) and len(data_mm) > 0:
            data_mm = self._apply_filter(data_mm, float(dt))
        
        # 4）长度统一：填充/截断到固定时间步
        L = len(data_mm)
        if L < self.time_steps:
            padded = np.zeros(self.time_steps)
            padded[:L] = data_mm
            final_data = padded
        else:
            final_data = data_mm[:self.time_steps]
            
        return final_data.astype(np.float32), float(dt)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.samples[idx]
        return {
            'event': str(sample.get('event', '')),
            'station': str(sample.get('station', '')),
            'event_index': torch.tensor(int(sample.get('event_index', -1)), dtype=torch.long),
            'radial': torch.tensor(sample['radial'], dtype=torch.float32).unsqueeze(0), # (1, Time)
            'vertical': torch.tensor(sample['vertical'], dtype=torch.float32), # (Time,) 单位 mm（Why：与径向一致便于共同学习）
            'distance': torch.tensor(sample['distance'], dtype=torch.float32),
            'magnitude': torch.tensor(sample['magnitude'], dtype=torch.float32),
            'mechanism': torch.tensor(int(sample.get('mechanism', -1)), dtype=torch.long),
            'dt': torch.tensor(sample['dt'], dtype=torch.float32),
            'theta_deg': torch.tensor(sample.get('theta_deg', self.default_theta_deg), dtype=torch.float32),
            'phi_deg': torch.tensor(sample.get('phi_deg', self.default_phi_deg), dtype=torch.float32),
            'stf': torch.tensor(sample['stf'], dtype=torch.float32) if sample.get('stf') is not None else torch.zeros(self.time_steps, dtype=torch.float32),
            'stf_log': torch.tensor(sample['stf_log'], dtype=torch.float32) if sample.get('stf_log') is not None else torch.zeros(self.time_steps, dtype=torch.float32),
            'has_stf': torch.tensor(bool(sample.get('has_stf', False))),
        }

def get_data_loaders(config: dict) -> Tuple[DataLoader, DataLoader, DataLoader]:
    data_path = config['paths']['data_path']
    time_steps = config['training']['time_steps']
    batch_size = config['training']['batch_size']
    val_split = config['training']['validation_split']
    test_split = config['training']['test_split']
    ds_cfg = (config.get('dataset', {}) or {})
    blacklist = ds_cfg.get('blacklist_events', [])
    units = ds_cfg.get('units', 'auto')
    center_mode = ds_cfg.get('center_mode', 'median')
    window_min_sec = float(ds_cfg.get('window_min_sec', 0.0))
    window_max_sec = float(ds_cfg.get('window_max_sec', 600.0))
    stf_path = ds_cfg.get('stf_path')
    stf_m_ref = float(ds_cfg.get('stf_m_ref', 1.0e18))
    default_theta_deg = float(ds_cfg.get('default_theta_deg', 45.0))
    default_phi_deg = float(ds_cfg.get('default_phi_deg', 0.0))
    filter_cfg = ds_cfg.get('filter', {})
    p_preprocess_enabled = bool(ds_cfg.get('p_preprocess_enabled', False))
    p_velocity_mps = float(ds_cfg.get('p_velocity_mps', (config.get('physics', {}) or {}).get('alpha', 7900.0)))
    p_arrival_offset_sec = float(ds_cfg.get('p_arrival_offset_sec', 0.0))
    p_baseline_mode = str(ds_cfg.get('p_baseline_mode', 'mean'))
    radial_peak_min_cm = float(ds_cfg.get('radial_peak_min_cm', ds_cfg.get('pgd_min_cm', 0.0)))
    
    allow_missing_stf = bool(ds_cfg.get('allow_missing_stf', False))
    dataset = EarthquakeDataset(
        data_path,
        time_steps,
        blacklist,
        units,
        center_mode,
        window_min_sec,
        window_max_sec,
        stf_path,
        stf_m_ref,
        default_theta_deg,
        default_phi_deg,
        filter_cfg,
        p_preprocess_enabled,
        p_velocity_mps,
        p_arrival_offset_sec,
        p_baseline_mode,
        allow_missing_stf=allow_missing_stf,
        radial_peak_min_cm=radial_peak_min_cm,
    )
    
    # Split
    total_size = len(dataset)
    test_size = int(total_size * test_split)
    val_size = int(total_size * val_split)
    train_size = total_size - test_size - val_size
    
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(config['training']['random_seed'])
    )
    
    seed = int((config.get('training', {}) or {}).get('random_seed', 42))

    def _worker_init_fn(worker_id: int) -> None:
        import random as _random
        worker_seed = seed + worker_id
        _random.seed(worker_seed)
        np.random.seed(worker_seed)
        torch.manual_seed(worker_seed)

    g = torch.Generator()
    g.manual_seed(seed)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              worker_init_fn=_worker_init_fn, generator=g)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader
