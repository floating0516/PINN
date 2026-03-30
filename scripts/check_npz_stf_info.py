from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np


NPZ_PATH = Path(r"F:\dataset_catalog\gnss_events_matched.gcmt.npz")
STF_DIR = Path(r"F:\dataset_catalog\STF_SCARDEC")
PREVIEW_EVENT_COUNT = 5
PREVIEW_STF_COUNT = 5


@dataclass
class StfStats:
    file_name: str
    sample_count: int
    t_min: float
    t_max: float
    dt_median: float
    mrate_min: float
    mrate_max: float
    m0_integral: float


def normalize_name(raw_name: str) -> str:
    text = str(raw_name).lower()
    keep_chars: list[str] = []
    for ch in text:
        if ch.isalnum() or ch in [" ", "-", "_"]:
            keep_chars.append(ch)
    return "".join(keep_chars).replace("  ", " ").strip()


def as_object_list(payload: Any) -> list[Any]:
    if payload is None:
        return []
    if isinstance(payload, np.ndarray):
        return list(payload.tolist())
    if isinstance(payload, list):
        return payload
    return [payload]


def summarize_npz(npz_path: Path) -> dict[str, Any]:
    if not npz_path.exists():
        raise FileNotFoundError(f"未找到 NPZ 文件: {npz_path}")

    summary: dict[str, Any] = {}
    with np.load(npz_path, allow_pickle=True) as data:
        keys = sorted(list(data.keys()))
        summary["keys"] = keys
        summary["events"] = as_object_list(data["events"]) if "events" in data else []
        summary["magnitude"] = np.asarray(data["magnitude"], dtype=np.float64) if "magnitude" in data else np.array([])
        summary["latitude"] = np.asarray(data["latitude"], dtype=np.float64) if "latitude" in data else np.array([])
        summary["longitude"] = np.asarray(data["longitude"], dtype=np.float64) if "longitude" in data else np.array([])
        summary["depth_km"] = np.asarray(data["depth_km"], dtype=np.float64) if "depth_km" in data else np.array([])
        summary["enu"] = as_object_list(data["enu"]) if "enu" in data else []
        summary["station_info"] = as_object_list(data["station_info"]) if "station_info" in data else []
        summary["shape_dtype"] = {k: {"shape": tuple(data[k].shape), "dtype": str(data[k].dtype)} for k in keys}
    return summary


def iter_station_items(event_container: Any) -> Iterable[tuple[str, Any]]:
    if isinstance(event_container, dict):
        for station_name, station_payload in event_container.items():
            yield str(station_name), station_payload
    elif isinstance(event_container, list):
        for idx, station_payload in enumerate(event_container):
            if isinstance(station_payload, dict) and "name" in station_payload:
                station_name = str(station_payload["name"])
            else:
                station_name = f"station_{idx}"
            yield station_name, station_payload


def first_station_series(enu_event: Any) -> tuple[str, np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    for station_name, station_payload in iter_station_items(enu_event):
        if not isinstance(station_payload, dict):
            continue
        if not {"t", "E", "N", "U"}.issubset(station_payload.keys()):
            continue
        t = np.asarray(station_payload["t"], dtype=np.float64)
        e = np.asarray(station_payload["E"], dtype=np.float64)
        n = np.asarray(station_payload["N"], dtype=np.float64)
        u = np.asarray(station_payload["U"], dtype=np.float64)
        if len(t) == 0:
            continue
        return station_name, t, e, n, u
    return None


def station_count_for_event(enu_event: Any, station_info_event: Any) -> int:
    enu_count = sum(1 for _ in iter_station_items(enu_event))
    info_count = sum(1 for _ in iter_station_items(station_info_event))
    return max(enu_count, info_count)


def parse_single_stf(stf_file: Path) -> StfStats | None:
    t_values: list[float] = []
    mrate_values: list[float] = []

    with stf_file.open("r", encoding="utf-8", errors="ignore") as file_obj:
        for line in file_obj:
            text = line.strip()
            if not text:
                continue
            parts = text.replace("D", "E").split()
            values: list[float] = []
            for token in parts:
                try:
                    values.append(float(token))
                except Exception:
                    values = []
                    break
            if len(values) != 2:
                continue
            t_values.append(values[0])
            mrate_values.append(values[1])

    if not t_values:
        return None

    t_array = np.asarray(t_values, dtype=np.float64)
    mrate_array = np.asarray(mrate_values, dtype=np.float64)
    finite_mask = np.isfinite(t_array) & np.isfinite(mrate_array)
    t_array = t_array[finite_mask]
    mrate_array = mrate_array[finite_mask]
    if len(t_array) == 0:
        return None

    order = np.argsort(t_array)
    t_array = t_array[order]
    mrate_array = mrate_array[order]

    dt_median = float(np.median(np.diff(t_array))) if len(t_array) > 1 else float("nan")
    m0_integral = float(np.trapz(mrate_array, t_array))

    return StfStats(
        file_name=stf_file.name,
        sample_count=int(len(t_array)),
        t_min=float(np.min(t_array)),
        t_max=float(np.max(t_array)),
        dt_median=dt_median,
        mrate_min=float(np.min(mrate_array)),
        mrate_max=float(np.max(mrate_array)),
        m0_integral=m0_integral,
    )


def summarize_stf_dir(stf_dir: Path) -> list[StfStats]:
    if not stf_dir.exists():
        raise FileNotFoundError(f"未找到 STF 目录: {stf_dir}")
    if not stf_dir.is_dir():
        raise NotADirectoryError(f"STF 路径不是目录: {stf_dir}")

    result: list[StfStats] = []
    for stf_file in sorted(stf_dir.glob("*.stf")):
        stats = parse_single_stf(stf_file)
        if stats is not None:
            result.append(stats)
    return result


def extract_year_token(raw_name: str) -> str:
    digits = "".join(ch for ch in str(raw_name) if ch.isdigit())
    return digits[:4] if len(digits) >= 4 else ""


def print_table(headers: list[str], rows: list[list[str]]) -> None:
    if not headers:
        return
    col_count = len(headers)
    col_widths = [len(h) for h in headers]
    for row in rows:
        for i in range(min(col_count, len(row))):
            col_widths[i] = max(col_widths[i], len(row[i]))

    def _format_row(values: list[str]) -> str:
        parts: list[str] = []
        for idx in range(col_count):
            text = values[idx] if idx < len(values) else ""
            parts.append(text.ljust(col_widths[idx]))
        return "| " + " | ".join(parts) + " |"

    split_line = "+-" + "-+-".join("-" * w for w in col_widths) + "-+"
    print(split_line)
    print(_format_row(headers))
    print(split_line)
    for row in rows:
        print(_format_row(row))
    print(split_line)


def print_event_station_table(npz_summary: dict[str, Any]) -> None:
    events = [str(v) for v in npz_summary.get("events", [])]
    enu_list = npz_summary.get("enu", [])
    station_info_list = npz_summary.get("station_info", [])
    magnitudes = npz_summary.get("magnitude", np.array([]))

    rows: list[list[str]] = []
    for idx, event_name in enumerate(events):
        enu_event = enu_list[idx] if idx < len(enu_list) else {}
        station_info_event = station_info_list[idx] if idx < len(station_info_list) else {}
        station_count = station_count_for_event(enu_event, station_info_event)
        mw = float(magnitudes[idx]) if idx < len(magnitudes) else float("nan")
        rows.append([str(idx), event_name, f"{mw:.2f}", str(station_count)])

    print("=" * 80)
    print("事件-台站数量表")
    print("=" * 80)
    print(f"事件总数: {len(rows)}")
    print_table(
        headers=["序号", "事件名", "Mw", "台站数"],
        rows=rows,
    )


def print_npz_report(npz_summary: dict[str, Any]) -> None:
    events = [str(v) for v in npz_summary.get("events", [])]
    magnitudes = npz_summary.get("magnitude", np.array([]))
    latitudes = npz_summary.get("latitude", np.array([]))
    longitudes = npz_summary.get("longitude", np.array([]))
    depths = npz_summary.get("depth_km", np.array([]))
    enu_list = npz_summary.get("enu", [])
    station_info_list = npz_summary.get("station_info", [])

    print("=" * 80)
    print("NPZ 文件概览")
    print("=" * 80)
    print(f"字段数量: {len(npz_summary.get('keys', []))}")
    print(f"字段列表: {npz_summary.get('keys', [])}")
    print(f"事件数量: {len(events)}")
    if len(magnitudes) > 0:
        print(
            f"Mw 统计: min={np.nanmin(magnitudes):.3f}, "
            f"max={np.nanmax(magnitudes):.3f}, mean={np.nanmean(magnitudes):.3f}"
        )
    if len(depths) > 0:
        print(
            f"深度统计(km): min={np.nanmin(depths):.3f}, "
            f"max={np.nanmax(depths):.3f}, mean={np.nanmean(depths):.3f}"
        )

    station_counts: list[int] = []
    for idx in range(min(len(enu_list), len(station_info_list))):
        station_counts.append(station_count_for_event(enu_list[idx], station_info_list[idx]))
    if station_counts:
        station_count_array = np.asarray(station_counts, dtype=np.int32)
        print(
            f"台站数统计: min={int(np.min(station_count_array))}, "
            f"max={int(np.max(station_count_array))}, mean={float(np.mean(station_count_array)):.2f}"
        )

    print("-" * 80)
    print("字段 shape/dtype")
    for key, meta in npz_summary.get("shape_dtype", {}).items():
        print(f"{key:>16s}: shape={meta['shape']}, dtype={meta['dtype']}")

    print("-" * 80)
    print(f"事件抽样预览(前 {PREVIEW_EVENT_COUNT} 个)")
    max_preview = min(PREVIEW_EVENT_COUNT, len(events), len(enu_list))
    for idx in range(max_preview):
        event_name = events[idx]
        mw = float(magnitudes[idx]) if idx < len(magnitudes) else float("nan")
        lat = float(latitudes[idx]) if idx < len(latitudes) else float("nan")
        lon = float(longitudes[idx]) if idx < len(longitudes) else float("nan")
        dep = float(depths[idx]) if idx < len(depths) else float("nan")
        station_count = station_count_for_event(
            enu_list[idx],
            station_info_list[idx] if idx < len(station_info_list) else {},
        )
        print(
            f"[{idx:03d}] event={event_name}, Mw={mw:.2f}, "
            f"lat={lat:.3f}, lon={lon:.3f}, depth_km={dep:.2f}, stations={station_count}"
        )

        series = first_station_series(enu_list[idx])
        if series is None:
            print("      首台站序列: 无有效 t/E/N/U 数据")
            continue
        station_name, t, e, n, u = series
        dt_median = float(np.median(np.diff(t))) if len(t) > 1 else float("nan")
        print(
            f"      台站={station_name}, n={len(t)}, t=[{float(np.min(t)):.2f}, {float(np.max(t)):.2f}], "
            f"dt_med={dt_median:.3f}s, E=[{float(np.min(e)):.3f}, {float(np.max(e)):.3f}] mm, "
            f"N=[{float(np.min(n)):.3f}, {float(np.max(n)):.3f}] mm, "
            f"U=[{float(np.min(u)):.3f}, {float(np.max(u)):.3f}] mm"
        )


def print_stf_report(stf_stats: list[StfStats]) -> None:
    print("=" * 80)
    print("STF 目录概览")
    print("=" * 80)
    print(f"有效 STF 文件数量: {len(stf_stats)}")
    if not stf_stats:
        return

    sample_counts = np.asarray([s.sample_count for s in stf_stats], dtype=np.int32)
    durations = np.asarray([s.t_max - s.t_min for s in stf_stats], dtype=np.float64)
    m0_values = np.asarray([s.m0_integral for s in stf_stats], dtype=np.float64)
    peaks = np.asarray([s.mrate_max for s in stf_stats], dtype=np.float64)
    print(
        f"采样点统计: min={int(np.min(sample_counts))}, "
        f"max={int(np.max(sample_counts))}, mean={float(np.mean(sample_counts)):.1f}"
    )
    print(
        f"持续时长统计(s): min={np.min(durations):.2f}, "
        f"max={np.max(durations):.2f}, mean={np.mean(durations):.2f}"
    )
    print(
        f"峰值矩率统计(N·m/s): min={np.min(peaks):.3e}, "
        f"max={np.max(peaks):.3e}, mean={np.mean(peaks):.3e}"
    )
    print(
        f"M0 积分统计(N·m): min={np.min(m0_values):.3e}, "
        f"max={np.max(m0_values):.3e}, mean={np.mean(m0_values):.3e}"
    )

    print("-" * 80)
    print(f"STF 抽样预览(前 {PREVIEW_STF_COUNT} 个)")
    for item in stf_stats[:PREVIEW_STF_COUNT]:
        print(
            f"{item.file_name}: n={item.sample_count}, "
            f"t=[{item.t_min:.2f}, {item.t_max:.2f}]s, dt_med={item.dt_median:.3f}s, "
            f"mrate=[{item.mrate_min:.3e}, {item.mrate_max:.3e}] N·m/s, "
            f"M0={item.m0_integral:.3e} N·m"
        )


def print_npz_stf_match(npz_summary: dict[str, Any], stf_stats: list[StfStats]) -> None:
    events = [str(v) for v in npz_summary.get("events", [])]
    event_norm_map = {normalize_name(event): event for event in events}
    stf_norm_map = {normalize_name(Path(item.file_name).stem): item.file_name for item in stf_stats}
    stf_names = [Path(item.file_name).stem for item in stf_stats]

    direct_match = sorted(set(event_norm_map.keys()) & set(stf_norm_map.keys()))
    fuzzy_match_count = 0
    fuzzy_pairs: list[tuple[str, str]] = []
    unmatched_events: list[str] = []
    for event_name in events:
        event_key = normalize_name(event_name)
        if event_key in stf_norm_map:
            continue
        found = False
        for stf_key in stf_norm_map.keys():
            if event_key in stf_key or stf_key in event_key:
                found = True
                fuzzy_match_count += 1
                fuzzy_pairs.append((event_name, stf_norm_map[stf_key]))
                break
        if not found:
            unmatched_events.append(event_name)

    unmatched_stf_files: list[str] = []
    for stf_name in stf_names:
        stf_key = normalize_name(stf_name)
        if stf_key not in event_norm_map:
            unmatched_stf_files.append(stf_name)

    explain_rows: list[list[str]] = []
    for event_name in unmatched_events:
        event_year = extract_year_token(event_name)
        candidate = ""
        if event_year:
            same_year_candidates = [name for name in unmatched_stf_files if extract_year_token(name) == event_year]
            if same_year_candidates:
                candidate = ", ".join(same_year_candidates[:3])
        explain_rows.append([event_name, event_year or "-", candidate or "-"])

    print("=" * 80)
    print("NPZ 与 STF 匹配检查")
    print("=" * 80)
    print(f"事件总数: {len(events)}")
    print(f"STF 总数: {len(stf_stats)}")
    print(f"精确匹配数: {len(direct_match)}")
    print(f"模糊匹配数: {fuzzy_match_count}")
    print(f"未匹配事件数: {len(unmatched_events)}")
    print(f"未匹配 STF 数: {len(unmatched_stf_files)}")
    print("说明: 两边数量都为 40 只代表元素个数相同，不代表事件名一一对应。")
    if unmatched_events:
        print("未匹配事件清单")
        print_table(
            headers=["NPZ事件名", "年份", "同年份未匹配STF候选"],
            rows=explain_rows,
        )
    if unmatched_stf_files:
        stf_rows = [[name, extract_year_token(name) or "-"] for name in unmatched_stf_files]
        print("未匹配 STF 清单")
        print_table(
            headers=["STF文件名(去后缀)", "年份"],
            rows=stf_rows,
        )
    if fuzzy_pairs:
        fuzzy_rows = [[left, right] for left, right in fuzzy_pairs]
        print("模糊匹配详情")
        print_table(
            headers=["NPZ事件名", "匹配STF文件"],
            rows=fuzzy_rows,
        )


def main() -> None:
    print(f"NPZ 路径: {NPZ_PATH}")
    print(f"STF 路径: {STF_DIR}")
    npz_summary = summarize_npz(NPZ_PATH)
    stf_stats = summarize_stf_dir(STF_DIR)
    print_npz_report(npz_summary)
    print_event_station_table(npz_summary)
    print_stf_report(stf_stats)
    print_npz_stf_match(npz_summary, stf_stats)


if __name__ == "__main__":
    main()
