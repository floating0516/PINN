from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class ScalingLawSpec:
    name: str
    citation_key: str
    pgd_unit: str
    distance_kind: str
    a: float
    b: float
    c: float


CROWELL_2013 = ScalingLawSpec(
    name="crowell",
    citation_key="crowell_2013",
    pgd_unit="cm",
    distance_kind="hypocentral",
    a=-6.687,
    b=1.500,
    c=-0.214,
)

RUHL_2019 = ScalingLawSpec(
    name="ruhl",
    citation_key="ruhl_2019",
    pgd_unit="m",
    distance_kind="hypocentral",
    a=-5.919,
    b=1.009,
    c=-0.145,
)

MELGAR_2015 = ScalingLawSpec(
    name="melgar",
    citation_key="melgar_2015",
    pgd_unit="cm",
    distance_kind="hypocentral",
    a=-4.434,
    b=1.047,
    c=-0.138,
)

AVAILABLE_SCALING_LAWS: dict[str, ScalingLawSpec] = {
    CROWELL_2013.name: CROWELL_2013,
    RUHL_2019.name: RUHL_2019,
    MELGAR_2015.name: MELGAR_2015,
}


def _validate_inputs(pgd_m: float, source_distance_km: float) -> None:
    if not math.isfinite(pgd_m) or pgd_m <= 0.0:
        raise ValueError(f"pgd_m 必须为正且有限，实际为 {pgd_m}")
    if not math.isfinite(source_distance_km) or source_distance_km <= 0.0:
        raise ValueError(f"source_distance_km 必须为正且有限，实际为 {source_distance_km}")


def _pgd_in_spec_unit(pgd_m: float, pgd_unit: str) -> float:
    if pgd_unit == "m":
        return pgd_m
    if pgd_unit == "cm":
        return pgd_m * 100.0
    raise ValueError(f"不支持的 PGD 单位: {pgd_unit}")


def predict_mw_from_coefficients(*, pgd_m: float, source_distance_km: float, spec: ScalingLawSpec) -> float:
    _validate_inputs(pgd_m, source_distance_km)
    pgd_value = _pgd_in_spec_unit(pgd_m, spec.pgd_unit)
    log_r = math.log10(source_distance_km)
    denominator = spec.b + spec.c * log_r
    if abs(denominator) < 1.0e-12:
        raise ValueError("标度律分母接近 0，无法稳定反解 Mw")
    return (math.log10(pgd_value) - spec.a) / denominator


def predict_mw_melgar(*, pgd_m: float, source_distance_km: float) -> float:
    return predict_mw_from_coefficients(pgd_m=pgd_m, source_distance_km=source_distance_km, spec=MELGAR_2015)


def predict_mw_ruhl(*, pgd_m: float, source_distance_km: float) -> float:
    return predict_mw_from_coefficients(pgd_m=pgd_m, source_distance_km=source_distance_km, spec=RUHL_2019)


def predict_mw_crowell(*, pgd_m: float, source_distance_km: float) -> float:
    return predict_mw_from_coefficients(pgd_m=pgd_m, source_distance_km=source_distance_km, spec=CROWELL_2013)


def predict_mw(*, law_name: str, pgd_m: float, source_distance_km: float) -> float:
    try:
        spec = AVAILABLE_SCALING_LAWS[law_name.lower()]
    except KeyError as exc:
        raise ValueError(f"未知标度律: {law_name}") from exc
    return predict_mw_from_coefficients(pgd_m=pgd_m, source_distance_km=source_distance_km, spec=spec)
