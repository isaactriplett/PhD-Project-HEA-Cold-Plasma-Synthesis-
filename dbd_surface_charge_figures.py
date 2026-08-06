# -*- coding: utf-8 -*-
"""Supplementary summaries and figures for the DBD surface-charge analysis.

The functions in this module deliberately operate on sequences of dictionaries
so that the main analysis can write the same rows to CSV before plotting them.
They have no pandas or SciPy dependency.  Matplotlib is imported only when a
plotting function is called.

Two aggregation rules are important here:

* lobe-level plots are capture-balanced: observations are first reduced to one
  median per capture and bin, then captures are aggregated; and
* uncertainty intervals resample those capture-level values with a circular
  moving-block bootstrap, preserving short-range acquisition-order structure.

The dose clock is an electrical charge-equivalent clock.  It is not an
electron-only measurement, a chemical conversion prediction, or a claim of
100 percent Faradaic utilization.
"""

from __future__ import annotations

import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Callable, Mapping, Sequence

import numpy as np


AVOGADRO_PER_MOL = 6.02214076e23

# Okabe-Ito-derived colours, with a darker blue for negative charge so the two
# polarity traces remain distinguishable in print and under common CVD modes.
POLARITY_COLORS = {
    "negative": "#0072B2",
    "positive": "#D55E00",
}
MATERIAL_COLORS = {
    "argon_only": "#0072B2",
    "pure_water": "#56B4E9",
    "BMIM_nitrate": "#E69F00",
    "5mM_Mn_nitrate_in_water": "#009E73",
}
FREQUENCY_COLORS = {
    4: "#0072B2",
    10: "#E69F00",
    20: "#009E73",
}
MATERIAL_ORDER = {
    "argon_only": 0,
    "pure_water": 1,
    "BMIM_nitrate": 2,
    "5mM_Mn_nitrate_in_water": 3,
}
MATERIAL_LABELS = {
    "argon_only": "Argon / no liquid",
    "pure_water": "Pure water",
    "BMIM_nitrate": "BMIM nitrate",
    "5mM_Mn_nitrate_in_water": "5 mM Mn nitrate",
}


def _float(value: object) -> float | None:
    """Return a finite float, accepting CSV strings and rejecting booleans."""
    if value is None or isinstance(value, (bool, np.bool_)):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if np.isfinite(result) else None


def _pick_float(row: Mapping[str, object], *keys: str) -> float | None:
    for key in keys:
        value = _float(row.get(key))
        if value is not None:
            return value
    return None


def _float_or_nan(value: object) -> float:
    result = _float(value)
    return math.nan if result is None else result


def _material(row: Mapping[str, object]) -> str:
    direct = str(row.get("material") or "").strip()
    if direct:
        return direct
    condition = str(row.get("condition") or "").strip()
    match = re.match(r"^(.*)_([0-9]+(?:\.[0-9]+)?)kHz$", condition)
    return match.group(1) if match else condition or "unknown"


def _frequency_khz(row: Mapping[str, object]) -> float | None:
    direct = _pick_float(row, "burst_kHz", "frequency_kHz", "frequency_khz")
    if direct is not None:
        return direct
    condition = str(row.get("condition") or "")
    match = re.search(r"_([0-9]+(?:\.[0-9]+)?)kHz$", condition)
    if match:
        return float(match.group(1))
    detected_hz = _pick_float(row, "detected_burst_Hz", "detected_burst_frequency_Hz")
    return None if detected_hz is None else detected_hz / 1000.0


def _frequency_label(frequency_khz: float | None) -> str:
    if frequency_khz is None:
        return "frequency unknown"
    rounded = round(frequency_khz)
    value = str(int(rounded)) if abs(frequency_khz - rounded) < 1e-8 else f"{frequency_khz:g}"
    return f"{value} kHz"


def _condition(row: Mapping[str, object]) -> str:
    direct = str(row.get("condition") or "").strip()
    if direct:
        return direct
    material = _material(row)
    frequency = _frequency_khz(row)
    if frequency is None:
        return material
    rounded = round(frequency)
    token = str(int(rounded)) if abs(frequency - rounded) < 1e-8 else f"{frequency:g}"
    return f"{material}_{token}kHz"


def _condition_label(row: Mapping[str, object]) -> str:
    direct = str(row.get("condition_label") or "").strip()
    if direct:
        return direct
    return f"{MATERIAL_LABELS.get(_material(row), _material(row))}, {_frequency_label(_frequency_khz(row))}"


def _polarity(row: Mapping[str, object]) -> str:
    value = str(
        row.get("target_charge_polarity")
        or row.get("charge_polarity")
        or row.get("polarity")
        or "unknown"
    ).strip().lower()
    if value.startswith("neg") or value == "-1":
        return "negative"
    if value.startswith("pos") or value == "1":
        return "positive"
    return value


def _capture_key(row: Mapping[str, object]) -> tuple[str, str]:
    member = str(row.get("member") or "")
    capture = str(row.get("capture_index") or "")
    if member:
        return member, capture
    return capture or "capture_unknown", ""


def _capture_sort_value(rows: Sequence[Mapping[str, object]]) -> tuple[float, str]:
    indices = [_float(row.get("capture_index")) for row in rows]
    finite = [value for value in indices if value is not None]
    index = min(finite) if finite else math.inf
    member = min((str(row.get("member") or "") for row in rows), default="")
    return index, member


def _material_sort(material: str) -> tuple[int, str]:
    return MATERIAL_ORDER.get(material, 99), material


def _frequency_sort(frequency: float | None) -> tuple[int, float]:
    return (1, math.inf) if frequency is None else (0, float(frequency))


def _moving_block_indices(
    length: int,
    replicates: int,
    block_length: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Circular moving-block capture indices, one bootstrap draw per row."""
    replicates = max(1, int(replicates))
    if length < 1:
        return np.empty((replicates, 0), dtype=int)
    block = max(1, min(int(block_length), length))
    if block == 1:
        return rng.integers(0, length, size=(replicates, length))
    count = int(math.ceil(length / block))
    starts = rng.integers(0, length, size=(replicates, count))
    offsets = np.arange(block, dtype=int)
    return ((starts[:, :, None] + offsets) % length).reshape(replicates, -1)[:, :length]


def _interval(draws: np.ndarray) -> tuple[float | None, float | None]:
    values = np.asarray(draws, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return None, None
    low, high = np.percentile(values, [2.5, 97.5])
    return float(low), float(high)


def _base_group(row: Mapping[str, object]) -> tuple[str, str, float | None]:
    return _condition(row), _material(row), _frequency_khz(row)


def build_capture_balanced_binned(
    rows: Sequence[Mapping[str, object]],
    x_key: str,
    y_key: str = "model_dependent_charge_nC",
    bins: int = 12,
    replicates: int = 1000,
    block_length: int = 4,
    seed: int = 0,
) -> list[dict]:
    """Build capture-balanced binned dose-response or stationarity summaries.

    Equal-width bin boundaries are shared by the two polarities within each
    material/frequency condition.  Within a bin, all lobes from one capture are
    first reduced to a median.  Point estimates and bootstrap intervals then
    describe the distribution across captures, preventing a capture with more
    detected duty periods from receiving more statistical weight.
    """
    bins = max(1, int(bins))
    rng = np.random.default_rng(seed)
    usable: list[Mapping[str, object]] = []
    for row in rows:
        if _float(row.get(x_key)) is None or _float(row.get(y_key)) is None:
            continue
        polarity = _polarity(row)
        if polarity not in POLARITY_COLORS:
            continue
        usable.append(row)

    base_rows: dict[tuple[str, str, float | None], list[Mapping[str, object]]] = defaultdict(list)
    for row in usable:
        base_rows[_base_group(row)].append(row)

    output: list[dict] = []
    base_order = sorted(base_rows, key=lambda item: (_material_sort(item[1]), _frequency_sort(item[2]), item[0]))
    for condition, material, frequency in base_order:
        condition_rows = base_rows[(condition, material, frequency)]
        x_values = np.asarray([float(row[x_key]) for row in condition_rows], dtype=float)
        x_min = float(np.min(x_values))
        x_max = float(np.max(x_values))
        if np.isclose(x_min, x_max, rtol=0.0, atol=max(1e-15, abs(x_min) * 1e-12)):
            pad = max(abs(x_min) * 1e-6, 1e-12)
            edges = np.asarray([x_min - pad, x_max + pad])
        else:
            edges = np.linspace(x_min, x_max, bins + 1)

        for polarity in ("negative", "positive"):
            polarity_rows = [row for row in condition_rows if _polarity(row) == polarity]
            if not polarity_rows:
                continue
            bin_rows: dict[int, list[Mapping[str, object]]] = defaultdict(list)
            for row in polarity_rows:
                x = float(row[x_key])
                index = int(np.searchsorted(edges, x, side="right") - 1)
                index = max(0, min(index, len(edges) - 2))
                bin_rows[index].append(row)

            for index in sorted(bin_rows):
                current = bin_rows[index]
                by_capture: dict[tuple[str, str], list[Mapping[str, object]]] = defaultdict(list)
                for row in current:
                    by_capture[_capture_key(row)].append(row)
                ordered_captures = sorted(by_capture.values(), key=_capture_sort_value)
                capture_x = np.asarray(
                    [np.median([float(row[x_key]) for row in capture]) for capture in ordered_captures],
                    dtype=float,
                )
                capture_y = np.asarray(
                    [np.median([float(row[y_key]) for row in capture]) for capture in ordered_captures],
                    dtype=float,
                )
                indices = _moving_block_indices(
                    len(capture_y), replicates, block_length, rng
                )
                y_draws = np.median(capture_y[indices], axis=1)
                x_draws = np.median(capture_x[indices], axis=1)
                y_low, y_high = _interval(y_draws)
                x_low, x_high = _interval(x_draws)
                x_estimate = float(np.median(capture_x))
                y_estimate = float(np.median(capture_y))
                label_source = current[0]
                result = {
                    "condition": condition,
                    "condition_label": _condition_label(label_source),
                    "material": material,
                    "burst_kHz": frequency,
                    "polarity": polarity,
                    "bin_index": index,
                    "bin_left": float(edges[index]),
                    "bin_right": float(edges[index + 1]),
                    "source_x_key": x_key,
                    "source_y_key": y_key,
                    "x_median": x_estimate,
                    "x_ci_low": x_low,
                    "x_ci_high": x_high,
                    "estimate": y_estimate,
                    "ci_low": y_low,
                    "ci_high": y_high,
                    "n_captures": len(capture_y),
                    "n_observations": len(current),
                    "within_capture_statistic": "median",
                    "across_capture_statistic": "median",
                    "uncertainty": "95_percent_circular_moving_block_bootstrap_over_captures",
                    x_key: x_estimate,
                    y_key: y_estimate,
                    f"{y_key}_ci_low": y_low,
                    f"{y_key}_ci_high": y_high,
                }
                output.append(result)
    return output


def build_stationarity_metrics(
    rows: Sequence[Mapping[str, object]],
    replicates: int = 1000,
    block_length: int = 4,
    seed: int = 0,
) -> list[dict]:
    """Compare the first and last record-time quintiles without lobe pooling.

    The first/last values, their difference, relative difference, and an OLS
    time slope are computed separately for each capture.  Reported estimates
    are medians of those capture-level diagnostics.
    """
    rng = np.random.default_rng(seed)
    grouped: dict[tuple[str, str, float | None, str], list[Mapping[str, object]]] = defaultdict(list)
    for row in rows:
        time = _float(row.get("midpoint_s"))
        charge = _float(row.get("model_dependent_charge_nC"))
        polarity = _polarity(row)
        if time is None or charge is None or polarity not in POLARITY_COLORS:
            continue
        condition, material, frequency = _base_group(row)
        grouped[(condition, material, frequency, polarity)].append(row)

    output: list[dict] = []
    group_order = sorted(
        grouped,
        key=lambda item: (_material_sort(item[1]), _frequency_sort(item[2]), item[0], item[3]),
    )
    for condition, material, frequency, polarity in group_order:
        observations = grouped[(condition, material, frequency, polarity)]
        by_capture: dict[tuple[str, str], list[Mapping[str, object]]] = defaultdict(list)
        for row in observations:
            by_capture[_capture_key(row)].append(row)
        capture_diagnostics: list[tuple[tuple[float, str], dict[str, float]]] = []
        for capture in by_capture.values():
            points = sorted(
                (
                    (float(row["midpoint_s"]), float(row["model_dependent_charge_nC"]))
                    for row in capture
                    if _float(row.get("midpoint_s")) is not None
                    and _float(row.get("model_dependent_charge_nC")) is not None
                ),
                key=lambda item: item[0],
            )
            if len(points) < 4:
                continue
            time = np.asarray([point[0] for point in points], dtype=float)
            charge = np.asarray([point[1] for point in points], dtype=float)
            span = float(time[-1] - time[0])
            if span <= 0:
                continue
            phase = (time - time[0]) / span
            early = charge[phase <= 0.20]
            late = charge[phase >= 0.80]
            if not len(early) or not len(late):
                continue
            early_value = float(np.median(early))
            late_value = float(np.median(late))
            drift = late_value - early_value
            scale = float(np.median(np.abs(charge)))
            relative = drift / scale if scale > 1e-15 else math.nan
            centered_time = time - float(np.mean(time))
            denominator = float(np.dot(centered_time, centered_time))
            slope = (
                float(np.dot(centered_time, charge - float(np.mean(charge))) / denominator)
                if denominator > 0
                else math.nan
            )
            capture_diagnostics.append(
                (
                    _capture_sort_value(capture),
                    {
                        "first_quintile_nC": early_value,
                        "last_quintile_nC": late_value,
                        "absolute_drift_nC": drift,
                        "relative_drift_fraction": relative,
                        "slope_nC_per_s": slope,
                    },
                )
            )
        capture_diagnostics.sort(key=lambda item: item[0])
        if not capture_diagnostics:
            continue
        metrics = [item[1] for item in capture_diagnostics]
        indices = _moving_block_indices(len(metrics), replicates, block_length, rng)
        result: dict[str, object] = {
            "condition": condition,
            "condition_label": _condition_label(observations[0]),
            "material": material,
            "burst_kHz": frequency,
            "polarity": polarity,
            "n_captures": len(metrics),
            "n_observations": len(observations),
            "early_window": "first_20_percent_of_each_capture_time_span",
            "late_window": "last_20_percent_of_each_capture_time_span",
            "within_capture_statistic": "median",
            "across_capture_statistic": "median",
            "uncertainty": "95_percent_circular_moving_block_bootstrap_over_captures",
        }
        for key in (
            "first_quintile_nC",
            "last_quintile_nC",
            "absolute_drift_nC",
            "relative_drift_fraction",
            "slope_nC_per_s",
        ):
            values = np.asarray([metric[key] for metric in metrics], dtype=float)
            result[key] = float(np.nanmedian(values))
            draws = np.nanmedian(values[indices], axis=1)
            low, high = _interval(draws)
            result[f"{key}_ci_low"] = low
            result[f"{key}_ci_high"] = high
        relative = _float(result.get("relative_drift_fraction"))
        result["relative_drift_percent"] = None if relative is None else 100.0 * relative
        low = _float(result.get("relative_drift_fraction_ci_low"))
        high = _float(result.get("relative_drift_fraction_ci_high"))
        result["relative_drift_percent_ci_low"] = None if low is None else 100.0 * low
        result["relative_drift_percent_ci_high"] = None if high is None else 100.0 * high
        drift_low = _float(result.get("absolute_drift_nC_ci_low"))
        drift_high = _float(result.get("absolute_drift_nC_ci_high"))
        if drift_low is not None and drift_low > 0:
            result["stationarity_status"] = "increase_detected_first_to_last_quintile"
        elif drift_high is not None and drift_high < 0:
            result["stationarity_status"] = "decrease_detected_first_to_last_quintile"
        else:
            result["stationarity_status"] = "no_first_to_last_change_resolved_at_95_percent_CI"
        output.append(result)
    return output


def _is_whole_record_rate(row: Mapping[str, object]) -> bool:
    metric = str(row.get("metric") or "").strip()
    return metric in {
        "whole_record_average_flow_per_s",
        "whole_record_average_equivalent_flow_per_s",
        "record_average_equivalent_flow_per_s",
    }


def _rate_from_long_row(
    row: Mapping[str, object],
) -> tuple[float | None, float | None, float | None]:
    if _is_whole_record_rate(row):
        estimate = _pick_float(row, "estimate", "diagnostic_signed_estimate")
        low = _pick_float(
            row,
            "analysis_ci_low",
            "diagnostic_analysis_ci_low",
            "repeat_ci_low",
        )
        high = _pick_float(
            row,
            "analysis_ci_high",
            "diagnostic_analysis_ci_high",
            "repeat_ci_high",
        )
        return estimate, low, high
    # Also accept the one-row-per-polarity headline format.
    key = "whole_record_average_flow_per_s"
    estimate = _pick_float(
        row,
        key,
        "whole_record_average_equivalent_flow_per_s",
        "record_average_equivalent_flow_per_s",
    )
    low = _pick_float(
        row,
        f"{key}_analysis_ci_low",
        f"{key}_repeat_ci_low",
        "whole_record_average_equivalent_flow_per_s_analysis_ci_low",
    )
    high = _pick_float(
        row,
        f"{key}_analysis_ci_high",
        f"{key}_repeat_ci_high",
        "whole_record_average_equivalent_flow_per_s_analysis_ci_high",
    )
    return estimate, low, high


def build_dose_clock_rows(
    long_results: Sequence[Mapping[str, object]],
    volume_ml: float = 2.5,
    concentration_mM: float = 5.0,
    equivalents_per_ion: float = 1.0,
) -> list[dict]:
    """Build 0--5 mL electrical-dose curves from negative whole-record rates.

    ``minutes = c_mM * V_mL * N_A * 1e-6 * equivalents / (60 * rate)``.
    Rate confidence limits are inverted when converted to time: the upper rate
    produces the lower dose time and vice versa.
    """
    if volume_ml < 0:
        raise ValueError("volume_ml must be non-negative")
    if concentration_mM <= 0:
        raise ValueError("concentration_mM must be positive")
    if equivalents_per_ion <= 0:
        raise ValueError("equivalents_per_ion must be positive")

    volume_grid = list(np.linspace(0.0, 5.0, 101))
    if not any(np.isclose(value, volume_ml, rtol=0.0, atol=1e-12) for value in volume_grid):
        volume_grid.append(float(volume_ml))
        volume_grid.sort()

    rate_rows: dict[str, Mapping[str, object]] = {}
    for row in long_results:
        if _polarity(row) != "negative":
            continue
        if row.get("metric") and not _is_whole_record_rate(row):
            continue
        material = _material(row)
        # BMIM is retained only as the requested hypothetical rate-transfer
        # comparator; the Mn-nitrate liquid is the composition-matched case.
        if material not in {"BMIM_nitrate", "5mM_Mn_nitrate_in_water"}:
            continue
        estimate, _, _ = _rate_from_long_row(row)
        if estimate is None or estimate <= 0:
            continue
        rate_rows.setdefault(_condition(row), row)

    output: list[dict] = []
    ordered = sorted(
        rate_rows.items(),
        key=lambda item: (_material_sort(_material(item[1])), _frequency_sort(_frequency_khz(item[1])), item[0]),
    )
    for condition, source in ordered:
        rate, rate_low, rate_high = _rate_from_long_row(source)
        if rate is None or rate <= 0:
            continue
        if rate_low is not None and rate_high is not None and rate_low > rate_high:
            rate_low, rate_high = rate_high, rate_low
        material = _material(source)
        reportable_rate = _pick_float(source, "estimate") is not None
        if material == "BMIM_nitrate":
            interpretation = (
                "hypothetical electrical-rate transfer to a 5 mM metal-ion inventory; "
                "BMIM nitrate itself contains no metal"
            )
        else:
            interpretation = (
                "composition-matched 5 mM Mn-ion inventory; electrical charge-equivalent "
                "clock, not chemical conversion"
            )
        if not reportable_rate:
            interpretation += (
                "; diagnostic scenario only because the source rate did not pass its "
                "passive-background reporting gate"
            )
        coefficient = concentration_mM * AVOGADRO_PER_MOL * 1e-6 * equivalents_per_ion / 60.0
        for volume in volume_grid:
            central = coefficient * volume / rate
            # Inversion is intentional: a high delivery rate gives a low time.
            time_low = (
                coefficient * volume / rate_high
                if rate_high is not None and rate_high > 0
                else None
            )
            time_high = (
                coefficient * volume / rate_low
                if rate_low is not None and rate_low > 0
                else None
            )
            output.append(
                {
                    "condition": condition,
                    "condition_label": _condition_label(source),
                    "material": material,
                    "burst_kHz": _frequency_khz(source),
                    "volume_ml": float(volume),
                    "is_reactor_volume": bool(np.isclose(volume, volume_ml, rtol=0.0, atol=1e-12)),
                    "reactor_volume_ml": float(volume_ml),
                    "concentration_mM": float(concentration_mM),
                    "equivalents_per_ion": float(equivalents_per_ion),
                    "negative_charge_equivalent_rate_per_s": rate,
                    "negative_charge_equivalent_rate_ci_low_per_s": rate_low,
                    "negative_charge_equivalent_rate_ci_high_per_s": rate_high,
                    "minutes_per_ion_equivalent": float(central),
                    "minutes_per_ion_equivalent_ci_low": None if time_low is None else float(time_low),
                    "minutes_per_ion_equivalent_ci_high": None if time_high is None else float(time_high),
                    "interpretation": interpretation,
                    "source_rate_reportable": reportable_rate,
                    "source_rate_evidence_tier": source.get("evidence_tier"),
                    "assumptions": (
                        "total_negative_electrical_charge_equivalents; 100_percent_delivery_and_"
                        "utilization; not_electron_specific; not_a_chemical_yield_prediction"
                    ),
                }
            )
    return output


def _save(
    fig: object,
    output: Path | str,
    save_figure: Callable[[object, Path, object], None],
    args: object,
) -> None:
    """Invoke the caller's common PNG/PDF save-and-close policy."""
    save_figure(fig, Path(output), args)


def _summary_order(summaries: Sequence[Mapping[str, object]]) -> list[Mapping[str, object]]:
    return sorted(
        summaries,
        key=lambda row: (_material_sort(_material(row)), _frequency_sort(_frequency_khz(row)), _condition(row)),
    )


def _condition_colors(rows: Sequence[Mapping[str, object]]) -> list[str]:
    return [MATERIAL_COLORS.get(_material(row), "#777777") for row in rows]


def _errorbar(
    estimate: float | None, low: float | None, high: float | None
) -> np.ndarray | None:
    if estimate is None or low is None or high is None:
        return None
    return np.asarray([[max(0.0, estimate - low)], [max(0.0, high - estimate)]])


def _raw_condition_values(
    per_file_rows: Sequence[Mapping[str, object]],
    keys: Sequence[str],
) -> dict[str, list[float]]:
    output: dict[str, list[float]] = defaultdict(list)
    for row in per_file_rows:
        value = _pick_float(row, *keys)
        if value is not None:
            output[_condition(row)].append(value)
    return output


def plot_power_audit(
    summaries: Sequence[Mapping[str, object]],
    per_file_rows: Sequence[Mapping[str, object]],
    output: Path | str,
    save_figure: Callable[[object, Path, object], None],
    args: object,
) -> None:
    """Plot capture distributions and bootstrap CIs for loop energy and power."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    ordered = _summary_order(summaries)
    if not ordered:
        return
    x = np.arange(len(ordered), dtype=float)
    colors = _condition_colors(ordered)
    labels = [_condition_label(row).replace(", ", "\n") for row in ordered]
    raw_energy = _raw_condition_values(
        per_file_rows, ("burst_energy_median_uJ", "burst_energy_uJ", "energy_per_burst_uJ")
    )
    raw_power = _raw_condition_values(
        per_file_rows, ("apparent_lissajous_power_mW", "apparent_power_mW", "discharge_power_mW")
    )

    panels = (
        (
            "Median energy per duty-burst period (µJ)",
            ("burst_energy_median_uJ", "burst_energy_uJ", "energy_per_burst_uJ"),
            ("burst_energy_ci_low_uJ", "burst_energy_median_uJ_ci_low", "burst_energy_median_uJ_repeat_ci_low"),
            ("burst_energy_ci_high_uJ", "burst_energy_median_uJ_ci_high", "burst_energy_median_uJ_repeat_ci_high"),
            raw_energy,
            1.0,
        ),
        (
            "Apparent raw Lissajous input power (W)",
            ("apparent_lissajous_power_mW", "apparent_power_mW", "discharge_power_mW"),
            ("apparent_lissajous_power_ci_low_mW", "apparent_lissajous_power_mW_ci_low", "apparent_lissajous_power_mW_repeat_ci_low"),
            ("apparent_lissajous_power_ci_high_mW", "apparent_lissajous_power_mW_ci_high", "apparent_lissajous_power_mW_repeat_ci_high"),
            raw_power,
            1.0e-3,
        ),
    )
    fig, axes = plt.subplots(2, 1, figsize=(max(10.0, 0.85 * len(ordered)), 8.4), sharex=True)
    for axis, (ylabel, value_keys, low_keys, high_keys, raw, scale) in zip(axes, panels):
        values = [
            None if _pick_float(row, *value_keys) is None
            else scale * float(_pick_float(row, *value_keys))
            for row in ordered
        ]
        for index, (row, value, color) in enumerate(zip(ordered, values, colors)):
            if value is None:
                continue
            axis.bar(index, value, width=0.68, color=color, alpha=0.70, edgecolor="#333333", linewidth=0.5)
            low = _pick_float(row, *low_keys)
            high = _pick_float(row, *high_keys)
            error = _errorbar(
                value,
                None if low is None else scale * low,
                None if high is None else scale * high,
            )
            if error is not None:
                axis.errorbar(index, value, yerr=error, fmt="none", color="#202020", capsize=3, linewidth=1.0)
            capture_values = [scale * item for item in raw.get(_condition(row), [])]
            if capture_values:
                offsets = np.linspace(-0.22, 0.22, len(capture_values)) if len(capture_values) > 1 else np.asarray([0.0])
                axis.scatter(index + offsets, capture_values, s=9, color="#202020", alpha=0.20, linewidths=0, zorder=3)
        axis.set_ylabel(ylabel)
        axis.set_ylim(bottom=0)
        axis.grid(axis="y", alpha=0.22)
        axis.spines[["top", "right"]].set_visible(False)
    axes[-1].set_xticks(x, labels, rotation=35, ha="right")
    materials = sorted({_material(row) for row in ordered}, key=_material_sort)
    handles = [
        Patch(facecolor=MATERIAL_COLORS.get(material, "#777777"), label=MATERIAL_LABELS.get(material, material), alpha=0.75)
        for material in materials
    ]
    if handles:
        axes[0].legend(handles=handles, frameon=False, ncol=min(4, len(handles)), loc="upper left")
    fig.suptitle(
        "Q–V loop energy and apparent discharge-circuit power\n"
        "Bars: median across captures; whiskers: 95% capture-block bootstrap",
        fontsize=12,
    )
    fig.text(
        0.5,
        0.005,
        "Raw Qm–V loop area includes plasma, dielectric, liquid, and phase-skew contributions; it is not plasma-only power.",
        ha="center",
        fontsize=9,
        color="#444444",
    )
    fig.tight_layout(rect=(0, 0.045, 1, 0.93))
    _save(fig, output, save_figure, args)


def plot_duty_audit(
    summaries: Sequence[Mapping[str, object]],
    per_file_rows: Sequence[Mapping[str, object]],
    output: Path | str,
    save_figure: Callable[[object, Path, object], None],
    args: object,
) -> None:
    """Plot the envelope-active fraction used by the duty-period segmentation."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    from matplotlib.ticker import PercentFormatter

    ordered = _summary_order(summaries)
    if not ordered:
        return
    x = np.arange(len(ordered), dtype=float)
    colors = _condition_colors(ordered)
    labels = [_condition_label(row).replace(", ", "\n") for row in ordered]
    raw = _raw_condition_values(
        per_file_rows, ("duty_on_fraction", "duty_on_fraction_median", "duty_fraction")
    )
    fig, axis = plt.subplots(figsize=(max(10.0, 0.85 * len(ordered)), 5.8))
    for index, (row, color) in enumerate(zip(ordered, colors)):
        value = _pick_float(row, "duty_on_fraction", "duty_on_fraction_median", "duty_fraction")
        if value is None:
            continue
        value = min(1.0, max(0.0, value))
        axis.bar(index, value, width=0.68, color=color, alpha=0.72, edgecolor="#333333", linewidth=0.5)
        low = _pick_float(row, "duty_on_fraction_ci_low", "duty_on_fraction_repeat_ci_low", "duty_fraction_ci_low")
        high = _pick_float(row, "duty_on_fraction_ci_high", "duty_on_fraction_repeat_ci_high", "duty_fraction_ci_high")
        error = _errorbar(value, low, high)
        if error is not None:
            axis.errorbar(index, value, yerr=error, fmt="none", color="#202020", capsize=3, linewidth=1.0)
        capture_values = [min(1.0, max(0.0, item)) for item in raw.get(_condition(row), [])]
        if capture_values:
            offsets = np.linspace(-0.22, 0.22, len(capture_values)) if len(capture_values) > 1 else np.asarray([0.0])
            axis.scatter(index + offsets, capture_values, s=9, color="#202020", alpha=0.20, linewidths=0, zorder=3)
        axis.text(index, min(0.985, value + 0.025), f"{100.0 * value:.1f}%", ha="center", va="bottom", fontsize=8)
    axis.set_xticks(x, labels, rotation=35, ha="right")
    axis.set_ylim(0, 1.08)
    axis.yaxis.set_major_formatter(PercentFormatter(1.0))
    axis.set_ylabel("Fraction of each duty-burst period envelope-active")
    axis.grid(axis="y", alpha=0.22)
    axis.spines[["top", "right"]].set_visible(False)
    materials = sorted({_material(row) for row in ordered}, key=_material_sort)
    handles = [
        Patch(facecolor=MATERIAL_COLORS.get(material, "#777777"), label=MATERIAL_LABELS.get(material, material), alpha=0.75)
        for material in materials
    ]
    if handles:
        axis.legend(handles=handles, frameon=False, ncol=min(4, len(handles)), loc="upper left")
    axis.set_title(
        "Duty-on fraction audit\n"
        "Same carrier-envelope threshold and duty-period windows used for lobe and Q–V energy extraction"
    )
    fig.tight_layout()
    _save(fig, output, save_figure, args)


def _arg_int(args: object, name: str, default: int) -> int:
    try:
        return int(getattr(args, name))
    except (AttributeError, TypeError, ValueError):
        return default


def plot_binned_facet(
    rows: Sequence[Mapping[str, object]],
    output: Path | str,
    save_figure: Callable[[object, Path, object], None],
    args: object,
    kind: str,
) -> None:
    """Plot capture-balanced dose-response or within-record stationarity facets.

    Raw peak-lobe rows may be supplied directly; the function then builds the
    binned summary and overlays a deterministic, faint subsample.  Pre-binned
    rows returned by :func:`build_capture_balanced_binned` are also accepted.
    """
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    normalized = kind.strip().lower().replace("-", "_")
    if normalized in {"dose", "dose_response", "per_lobe_dose_response"}:
        x_key = "amplitude_kV"
        x_label = "Carrier half-cycle amplitude (kV)"
        title = "Per-lobe electrical dose response"
        x_scale = 1.0
    elif normalized in {"stationarity", "within_record_stationarity"}:
        x_key = "midpoint_s"
        x_label = "Midpoint within record (ms)"
        title = "Within-record stationarity of peak-lobe charge"
        x_scale = 1000.0
    else:
        raise ValueError("kind must be 'dose_response' or 'stationarity'")

    is_binned = bool(rows) and all("bin_index" in row and "estimate" in row for row in rows)
    if is_binned:
        summaries = list(rows)
        raw_rows: list[Mapping[str, object]] = []
    else:
        raw_rows = list(rows)
        summaries = build_capture_balanced_binned(
            raw_rows,
            x_key=x_key,
            bins=_arg_int(args, "supplementary_bins", 12),
            replicates=_arg_int(args, "bootstrap_replicates", 1000),
            block_length=_arg_int(
                args,
                "bootstrap_block_files",
                _arg_int(args, "bootstrap_block_length", 4),
            ),
            seed=_arg_int(
                args,
                "random_seed",
                _arg_int(args, "bootstrap_seed", 0),
            ) + (41 if normalized.startswith("dose") else 43),
        )
    if not summaries:
        return
    materials = sorted({_material(row) for row in summaries}, key=_material_sort)
    frequencies = sorted({_frequency_khz(row) for row in summaries}, key=_frequency_sort)
    fig, axes = plt.subplots(
        len(materials),
        len(frequencies),
        figsize=(max(4.0, 3.7 * len(frequencies)), max(3.1, 2.75 * len(materials))),
        squeeze=False,
        sharex=False,
        sharey=False,
    )
    raw_grouped: dict[tuple[str, float | None, str], list[Mapping[str, object]]] = defaultdict(list)
    for row in raw_rows:
        if _float(row.get(x_key)) is not None and _float(row.get("model_dependent_charge_nC")) is not None:
            raw_grouped[(_material(row), _frequency_khz(row), _polarity(row))].append(row)

    for row_index, material in enumerate(materials):
        for column_index, frequency in enumerate(frequencies):
            axis = axes[row_index, column_index]
            present = False
            for polarity in ("negative", "positive"):
                current = [
                    row
                    for row in summaries
                    if _material(row) == material
                    and _frequency_khz(row) == frequency
                    and _polarity(row) == polarity
                ]
                current.sort(key=lambda row: float(row.get("x_median", row.get(x_key, 0.0))))
                if not current:
                    continue
                present = True
                x = np.asarray(
                    [x_scale * float(row.get("x_median", row[x_key])) for row in current],
                    dtype=float,
                )
                y = np.asarray([float(row["estimate"]) for row in current], dtype=float)
                low = np.asarray([_float_or_nan(row.get("ci_low")) for row in current], dtype=float)
                high = np.asarray([_float_or_nan(row.get("ci_high")) for row in current], dtype=float)
                color = POLARITY_COLORS[polarity]
                raw = raw_grouped.get((material, frequency, polarity), [])
                if raw:
                    raw = sorted(
                        raw,
                        key=lambda row: (
                            _float(row.get("capture_index")) or math.inf,
                            _float(row.get(x_key)) or math.inf,
                        ),
                    )
                    sample_count = min(320, len(raw))
                    sample_indices = np.linspace(0, len(raw) - 1, sample_count, dtype=int)
                    axis.scatter(
                        [x_scale * float(raw[index][x_key]) for index in sample_indices],
                        [float(raw[index]["model_dependent_charge_nC"]) for index in sample_indices],
                        s=5,
                        color=color,
                        alpha=0.055,
                        linewidths=0,
                        rasterized=True,
                    )
                valid_band = np.isfinite(low) & np.isfinite(high)
                if normalized.startswith("dose"):
                    errors = np.vstack(
                        (np.maximum(0.0, y - low), np.maximum(0.0, high - y))
                    )
                    errors[:, ~valid_band] = 0.0
                    axis.errorbar(
                        x,
                        y,
                        yerr=errors,
                        fmt="o",
                        linestyle="none",
                        markersize=3.4,
                        capsize=1.8,
                        linewidth=0.8,
                        color=color,
                        label=polarity.capitalize(),
                    )
                else:
                    bin_indices = np.asarray(
                        [int(row.get("bin_index", index)) for index, row in enumerate(current)],
                        dtype=int,
                    )
                    segment_starts = np.r_[0, np.flatnonzero(np.diff(bin_indices) > 1) + 1]
                    segment_stops = np.r_[segment_starts[1:], len(current)]
                    for segment_index, (segment_start, segment_stop) in enumerate(
                        zip(segment_starts, segment_stops)
                    ):
                        segment = slice(int(segment_start), int(segment_stop))
                        segment_valid = valid_band[segment]
                        if np.count_nonzero(segment_valid) >= 2:
                            sx = x[segment][segment_valid]
                            axis.fill_between(
                                sx,
                                low[segment][segment_valid],
                                high[segment][segment_valid],
                                color=color,
                                alpha=0.15,
                                linewidth=0,
                            )
                        axis.plot(
                            x[segment],
                            y[segment],
                            marker="o",
                            markersize=3.2,
                            linewidth=1.35,
                            color=color,
                            label=polarity.capitalize() if segment_index == 0 else None,
                        )
            if not present:
                axis.text(0.5, 0.5, "No data", ha="center", va="center", transform=axis.transAxes, color="#777777")
            axis.axhline(0.0, color="#777777", linewidth=0.7, alpha=0.45)
            axis.grid(alpha=0.18)
            axis.spines[["top", "right"]].set_visible(False)
            if row_index == 0:
                axis.set_title(_frequency_label(frequency))
            if row_index == len(materials) - 1:
                axis.set_xlabel(x_label)
            if column_index == 0:
                axis.set_ylabel(f"{MATERIAL_LABELS.get(material, material)}\nCharge (nC)")
    handles = [
        Line2D([0], [0], color=POLARITY_COLORS[polarity], marker="o", linewidth=1.5, label=f"{polarity.capitalize()} charge")
        for polarity in ("negative", "positive")
    ]
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False)
    fig.suptitle(
        f"{title}\n"
        "Points: median of capture-level bin medians; bands: 95% moving-block capture bootstrap",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0.055, 1, 0.93))
    _save(fig, output, save_figure, args)


def plot_dose_clock(
    rows: Sequence[Mapping[str, object]],
    output: Path | str,
    save_figure: Callable[[object, Path, object], None],
    args: object,
) -> None:
    """Plot minutes per ion-equivalent against liquid volume for BMIM and Mn."""
    import matplotlib.pyplot as plt

    if not rows:
        return
    materials = [
        material
        for material in ("BMIM_nitrate", "5mM_Mn_nitrate_in_water")
        if any(_material(row) == material for row in rows)
    ]
    if not materials:
        return
    fig, axes = plt.subplots(1, len(materials), figsize=(6.0 * len(materials), 5.1), squeeze=False)
    axes_flat = axes[0]
    for axis, material in zip(axes_flat, materials):
        conditions = sorted(
            {_condition(row) for row in rows if _material(row) == material},
            key=lambda condition: _frequency_sort(
                _frequency_khz(next(row for row in rows if _condition(row) == condition))
            ),
        )
        for condition in conditions:
            current = sorted(
                [row for row in rows if _condition(row) == condition],
                key=lambda row: float(row["volume_ml"]),
            )
            frequency = _frequency_khz(current[0])
            rounded_frequency = int(round(frequency)) if frequency is not None else -1
            color = FREQUENCY_COLORS.get(rounded_frequency, "#777777")
            x = np.asarray([float(row["volume_ml"]) for row in current], dtype=float)
            y = np.asarray([float(row["minutes_per_ion_equivalent"]) for row in current], dtype=float)
            low = np.asarray(
                [_float_or_nan(row.get("minutes_per_ion_equivalent_ci_low")) for row in current],
                dtype=float,
            )
            high = np.asarray(
                [_float_or_nan(row.get("minutes_per_ion_equivalent_ci_high")) for row in current],
                dtype=float,
            )
            reportable = bool(current[0].get("source_rate_reportable"))
            linestyle = (
                "-" if reportable and material != "BMIM_nitrate"
                else "--" if reportable
                else ":"
            )
            axis.plot(x, y, color=color, linewidth=1.8, linestyle=linestyle, label=_frequency_label(frequency))
            valid = np.isfinite(low) & np.isfinite(high)
            if np.count_nonzero(valid) >= 2:
                axis.fill_between(x[valid], low[valid], high[valid], color=color, alpha=0.13, linewidth=0)
            reactor = [row for row in current if bool(row.get("is_reactor_volume"))]
            if reactor:
                point = reactor[0]
                axis.scatter(
                    [float(point["volume_ml"])],
                    [float(point["minutes_per_ion_equivalent"])],
                    s=40,
                    color=color,
                    edgecolor="white",
                    linewidth=0.8,
                    zorder=4,
                )
        reactor_volumes = [_float(row.get("reactor_volume_ml")) for row in rows if _material(row) == material]
        reactor_volumes = [value for value in reactor_volumes if value is not None]
        if reactor_volumes:
            axis.axvline(float(np.median(reactor_volumes)), color="#555555", linestyle=":", linewidth=1.0, label="Reactor volume")
        axis.set_xlim(0, max(5.0, *(reactor_volumes or [5.0])))
        axis.set_ylim(bottom=0)
        axis.set_xlabel("Liquid volume (mL)")
        axis.set_ylabel("Minutes to 1 negative-charge equivalent per ion")
        axis.grid(alpha=0.20)
        axis.spines[["top", "right"]].set_visible(False)
        if material == "BMIM_nitrate":
            axis.set_title("BMIM rate — hypothetical 5 mM metal-ion inventory")
        else:
            axis.set_title("5 mM Mn nitrate — composition-matched inventory")
        axis.legend(frameon=False)
    concentration = _pick_float(rows[0], "concentration_mM") or 5.0
    equivalents = _pick_float(rows[0], "equivalents_per_ion") or 1.0
    fig.suptitle(
        f"Electrical dose clock: {concentration:g} mM inventory, {equivalents:g} e⁻-equivalent per ion\n"
        "Curves use measured total negative charge-equivalent rates",
        fontsize=12,
    )
    fig.text(
        0.5,
        0.005,
        "Assumes 100% charge delivery and utilization; negative charge includes electrons and negative ions. "
        "Mn²⁺ full reduction requires 2 equivalents per ion. "
        + (
            "Dotted curves use non-reportable diagnostic rates."
            if any(not bool(row.get("source_rate_reportable")) for row in rows)
            else ""
        ),
        ha="center",
        fontsize=8.8,
        color="#444444",
    )
    fig.tight_layout(rect=(0, 0.055, 1, 0.90))
    _save(fig, output, save_figure, args)


__all__ = [
    "build_capture_balanced_binned",
    "build_stationarity_metrics",
    "build_dose_clock_rows",
    "plot_power_audit",
    "plot_duty_audit",
    "plot_binned_facet",
    "plot_dose_clock",
]
