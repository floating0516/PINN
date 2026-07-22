from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from src.data.manifest import make_manifest_row
from src.data.records_v2 import (
    NormalizedStationRecord,
    _iter_normalized_station_records,
)
from src.data.sample_builder import SampleRejected, build_station_sample
from src.data.stf import ProcessedSTF, STFWindowTooShort, resample_source_stf
from src.data.waveform import WaveformConfig
from src.utils.config_v2 import (
    stf_m_ref_from_config,
    stf_output_steps_from_config,
    validate_config_v2,
)


STFMatch = tuple[Path, np.ndarray, np.ndarray]
ProcessedSTFMatch = tuple[ProcessedSTF, str]


@dataclass(frozen=True)
class _CachedSTFWindowFailure:
    message: str


STFCacheEntry = ProcessedSTFMatch | _CachedSTFWindowFailure | None


def _normalize_event_name(value: str) -> str:
    return "".join(character for character in str(value).lower() if character.isalnum())


def _load_stf_files(stf_dir: str | Path) -> dict[str, STFMatch]:
    root = Path(stf_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"STF directory not found: {root}")
    mapping: dict[str, STFMatch] = {}
    for path in sorted(root.glob("*.stf")):
        rows: list[tuple[float, float]] = []
        with path.open("r", encoding="utf-8", errors="ignore") as stream:
            for line in stream:
                tokens = line.replace("D", "E").split()
                if len(tokens) != 2:
                    continue
                try:
                    rows.append((float(tokens[0]), float(tokens[1])))
                except ValueError:
                    continue
        if not rows:
            continue
        key = _normalize_event_name(path.stem)
        if not key:
            raise ValueError(f"empty normalized STF key: {path}")
        if key in mapping:
            raise ValueError(f"duplicate normalized STF key: {key}")
        values = np.asarray(rows, dtype=np.float64)
        mapping[key] = (path, values[:, 0], values[:, 1])
    return mapping


def _match_stf(
    event_name: str,
    mapping: dict[str, STFMatch],
) -> STFMatch | None:
    key = _normalize_event_name(event_name)
    if not key:
        raise ValueError("event name has an empty normalized STF key")
    if key in mapping:
        return mapping[key]
    candidates = [
        item
        for candidate, item in mapping.items()
        if key in candidate or candidate in key
    ]
    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) > 1:
        paths = ", ".join(str(item[0]) for item in candidates)
        raise ValueError(f"ambiguous STF match for {event_name}: {paths}")
    return None


def _waveform_config(config: dict[str, Any]) -> WaveformConfig:
    dataset = config["dataset"]
    waveform = dataset["waveform"]
    baseline = dataset["baseline"]
    filter_config = dataset["filter"]
    return WaveformConfig(
        sample_rate_hz=float(dataset["sample_rate_hz"]),
        start_sec=float(waveform["start_sec"]),
        duration_sec=float(waveform["duration_sec"]),
        min_valid_fraction=float(waveform["min_valid_fraction"]),
        max_interpolation_gap_sec=float(
            waveform["max_interpolation_gap_sec"]
        ),
        baseline_method=str(baseline["method"]),
        pre_event_start_sec=float(baseline["pre_event_start_sec"]),
        pre_event_end_sec=float(baseline["pre_event_end_sec"]),
        baseline_fallback=str(baseline["fallback"]),
        baseline_fallback_max_sec=float(baseline["fallback_max_sec"]),
        baseline_min_samples=int(baseline["min_samples"]),
        filter_type=str(filter_config["type"]),
        cutoff_hz=float(filter_config["cutoff_hz"]),
        num_taps=int(filter_config["num_taps"]),
        filter_window=str(filter_config["window"]),
    )


class CorrectedEarthquakeDataset(Dataset):
    def __init__(self, config: dict[str, Any]) -> None:
        validate_config_v2(config)
        self.config = config
        self.samples: list[dict[str, Any]] = []
        self.rejections: list[dict[str, Any]] = []
        self.manifest_rows: list[dict[str, Any]] = []
        dataset_config = config["dataset"]
        stf_config = dataset_config["stf"]
        self.stf_m_ref = stf_m_ref_from_config(config)
        self.stf_output_steps = stf_output_steps_from_config(config)
        self.waveform_config = _waveform_config(config)
        self.blacklist = {
            str(value)
            for value in dataset_config.get("blacklist_events", [])
        }
        self.allow_missing_stf = bool(
            dataset_config.get("allow_missing_stf", False)
        )
        self.stf_files = _load_stf_files(stf_config["path"])
        self.event_stf_cache: dict[str, STFCacheEntry] = {}

        with np.load(config["paths"]["data_path"], allow_pickle=True) as data:
            for record in _iter_normalized_station_records(data):
                self._consume_record(record)

        accepted_events = {sample["event"] for sample in self.samples}
        print(
            f"Loaded {len(self.samples)} corrected samples from "
            f"{len(accepted_events)} accepted events"
        )

    def _event_stf(self, event_name: str) -> ProcessedSTFMatch | None:
        cache_key = _normalize_event_name(event_name)
        if cache_key in self.event_stf_cache:
            cached = self.event_stf_cache[cache_key]
            if isinstance(cached, _CachedSTFWindowFailure):
                raise STFWindowTooShort(cached.message)
            return cached

        match = _match_stf(event_name, self.stf_files)
        if match is None:
            self.event_stf_cache[cache_key] = None
            return None
        path, source_time, source_rate = match
        stf_config = self.config["dataset"]["stf"]
        try:
            processed = resample_source_stf(
                source_time,
                source_rate,
                start_sec=float(stf_config["start_sec"]),
                duration_sec=float(stf_config["duration_sec"]),
                sample_rate_hz=float(
                    self.config["dataset"]["sample_rate_hz"]
                ),
                min_retained_moment_fraction=float(
                    stf_config["min_retained_moment_fraction"]
                ),
                preserve_integral=bool(stf_config["preserve_integral"]),
            )
        except STFWindowTooShort as exc:
            self.event_stf_cache[cache_key] = _CachedSTFWindowFailure(
                str(exc)
            )
            raise
        result = (processed, str(path))
        self.event_stf_cache[cache_key] = result
        return result

    def _reject(
        self,
        record: NormalizedStationRecord,
        reason: str,
        detail: str = "",
        *,
        sample: dict[str, Any] | None = None,
        event_stf: ProcessedSTFMatch | None = None,
        has_stf: bool | None = None,
    ) -> None:
        rejection = {
            "event": record.event,
            "station": record.station,
            "reason": reason,
        }
        if detail:
            rejection["detail"] = detail
        self.rejections.append(rejection)
        processed_stf = None if event_stf is None else event_stf[0]
        self.manifest_rows.append(
            make_manifest_row(
                record,
                accepted=False,
                rejection_reason=reason,
                sample=sample,
                processed_stf=processed_stf,
                has_stf=has_stf,
            )
        )

    def _consume_record(self, record: NormalizedStationRecord) -> None:
        if record.event in self.blacklist:
            self._reject(record, "blacklisted_event")
            return
        try:
            event_stf = self._event_stf(record.event)
        except STFWindowTooShort as exc:
            self._reject(
                record,
                "stf_window_too_short",
                str(exc),
                has_stf=True,
            )
            return
        if event_stf is None and not self.allow_missing_stf:
            self._reject(record, "missing_stf", has_stf=False)
            return

        try:
            sample = build_station_sample(
                record,
                units=str(self.config["dataset"]["units"]),
                waveform_config=self.waveform_config,
                alpha_m_per_s=float(self.config["physics"]["alpha"]),
                radial_peak_min_cm=float(
                    self.config["dataset"]["radial_peak_min_cm"]
                ),
            )
        except SampleRejected as exc:
            self._reject(
                record,
                exc.reason,
                exc.detail,
                sample=exc.sample,
                event_stf=event_stf,
                has_stf=event_stf is not None,
            )
            return

        if event_stf is None:
            stf = np.zeros(self.stf_output_steps, dtype=np.float32)
            stf_log = np.zeros(self.stf_output_steps, dtype=np.float32)
            stf_dt_sec = 1.0 / float(
                self.config["dataset"]["sample_rate_hz"]
            )
            mw_stf_native = float("nan")
            retained_fraction = float("nan")
            stf_path = ""
            has_stf = False
        else:
            processed_stf, stf_path = event_stf
            stf = processed_stf.rate_nm_per_s.astype(np.float32)
            stf_log = np.log10(
                1.0 + processed_stf.rate_nm_per_s / self.stf_m_ref
            ).astype(np.float32)
            stf_dt_sec = processed_stf.dt_sec
            mw_stf_native = processed_stf.mw_native
            retained_fraction = processed_stf.retained_moment_fraction
            has_stf = True

        sample.update(
            {
                "stf": stf,
                "stf_log": stf_log,
                "stf_dt_sec": float(stf_dt_sec),
                "mw_stf_native": float(mw_stf_native),
                "stf_retained_moment_fraction": float(retained_fraction),
                "stf_path": stf_path,
                "has_stf": has_stf,
            }
        )
        self.samples.append(sample)
        self.manifest_rows.append(
            make_manifest_row(
                record,
                accepted=True,
                sample=sample,
                processed_stf=(None if event_stf is None else event_stf[0]),
                has_stf=has_stf,
            )
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.samples[index]
        return {
            "event": sample["event"],
            "station": sample["station"],
            "event_index": torch.tensor(
                sample["event_index"], dtype=torch.long
            ),
            "mechanism": torch.tensor(
                sample["mechanism"], dtype=torch.long
            ),
            "radial": torch.from_numpy(sample["radial"]).float().unsqueeze(0),
            "vertical": torch.from_numpy(sample["vertical"]).float(),
            "waveform_valid_mask": torch.from_numpy(
                sample["waveform_valid_mask"]
            ).bool(),
            "waveform_dt_sec": torch.tensor(
                sample["waveform_dt_sec"], dtype=torch.float32
            ),
            "raw_dt_sec": torch.tensor(
                sample["raw_dt_sec"], dtype=torch.float32
            ),
            "epicentral_distance_m": torch.tensor(
                sample["epicentral_distance_m"], dtype=torch.float32
            ),
            "source_distance_m": torch.tensor(
                sample["source_distance_m"], dtype=torch.float32
            ),
            "theta_deg": torch.tensor(
                sample["theta_deg"], dtype=torch.float32
            ),
            "azimuth_deg": torch.tensor(
                sample["azimuth_deg"], dtype=torch.float32
            ),
            "phi_slip_deg": torch.tensor(
                sample["phi_slip_deg"], dtype=torch.float32
            ),
            "magnitude_catalog": torch.tensor(
                sample["magnitude_catalog"], dtype=torch.float32
            ),
            "stf": torch.from_numpy(sample["stf"]).float(),
            "stf_log": torch.from_numpy(sample["stf_log"]).float(),
            "stf_dt_sec": torch.tensor(
                sample["stf_dt_sec"], dtype=torch.float32
            ),
            "mw_stf_native": torch.tensor(
                sample["mw_stf_native"], dtype=torch.float32
            ),
            "stf_retained_moment_fraction": torch.tensor(
                sample["stf_retained_moment_fraction"],
                dtype=torch.float32,
            ),
            "has_stf": torch.tensor(sample["has_stf"], dtype=torch.bool),
            "valid_fraction": torch.tensor(
                sample["valid_fraction"], dtype=torch.float32
            ),
            "radial_peak_cm": torch.tensor(
                sample["radial_peak_cm"], dtype=torch.float32
            ),
            "baseline_source": sample["baseline_source"],
            "stf_path": sample["stf_path"],
        }
