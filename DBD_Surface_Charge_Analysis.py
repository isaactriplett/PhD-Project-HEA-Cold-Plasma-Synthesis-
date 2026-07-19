# -*- coding: utf-8 -*-
"""Supervisor-ready charge-transfer analysis for pulsed AC DBD waveforms.

This script is deliberately separate from :mod:`Lissajous_Figures`.  It reads
the voltage-scan ZIP archive directly, locks Channel-D polarity from the
below-breakdown ensemble, extracts one observation per carrier half-cycle, and
keeps three evidence tiers separate:

``direct_nominal``
    The terminal charge ``Qm = Cref * Vmonitor`` using the nominal monitor
    capacitance.
``background_subtracted_terminal_excess``
    Terminal charge after subtracting the empirical 40/60/75 %-breakdown
    response.  This is the strongest generally available charge-transfer
    proxy, but it is not automatically the plasma-to-surface charge.
``exploratory_model_dependent`` or ``validated_model_dependent``
    The background-subtracted charge multiplied by
    ``F = Cd / (Cd - Ccell)``.  A geometry-derived ``Cd`` is always labeled
    exploratory.  A scan-derived value is validated only after all predefined
    physical and statistical gates pass.

The requested "peak flux" is reported as a *peak half-cycle-average total
charge-equivalent flow*.  The existing sampling does not resolve a true
nanosecond microdischarge peak, and no area-normalized flux is reported unless
``--active-area-mm2`` is supplied.

Example
-------
python DBD_Surface_Charge_Analysis.py "C:\\path\\waveforms.zip" \
    --output-dir dbd_surface_charge_report
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import zipfile
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np

from Lissajous_Figures import (
    AnalysisError as WaveformAnalysisError,
    select_two_duty_cycles,
)
from Lissajous_Scan_Analysis import (
    ACTIVE_FIT_LEVELS,
    ELEMENTARY_CHARGE_C,
    NC_PER_C,
    PASSIVE_FIT_LEVELS,
    Condition,
    MemberRecord,
    burst_extrema_features,
    burst_windows,
    choose_current_sign,
    choose_q_sign,
    constrained_carrier_frequency,
    detect_clipping,
    evenly_spaced,
    inventory_scan_archive,
    moving_average,
    phasor_ratio,
    read_waveform_member,
)


ANALYSIS_VERSION = "1.0-surface-charge"
LEVELS = (40, 60, 75, 90, 100, 105, 115)
SCAN_FIT_LEVELS = (40, 60, 75)
TRANSITION_LEVELS = (90, 100)
HIGH_FIELD_LEVELS = (105, 115)
COLORS = {
    "negative": "#2864b7",
    "positive": "#d15b35",
    "neutral": "#555555",
    "invalid": "#999999",
    "passive": "#278c77",
}
CONDUCTIVE_LIQUIDS = {"BMIM_nitrate", "5mM_Mn_nitrate_in_water"}


class SurfaceChargeError(RuntimeError):
    """Raised when the archive cannot support the requested analysis."""


@dataclass(frozen=True)
class LobeObservation:
    burst_index: int
    voltage_polarity: int
    amplitude_kV: float
    duration_s: float
    midpoint_s: float
    raw_directed_charge_nC: float


@dataclass
class FileObservation:
    record: MemberRecord
    level_label: str
    carrier_Hz: float
    detected_burst_Hz: float
    burst_detection_method: str
    burst_frequency_relative_error: float
    duration_s: float
    voltage_pp_kV: float
    cstar_raw_F: complex | None
    sign_vote: int | None
    lobes: list[LobeObservation]
    raw_extrema_x_kV: float | None
    raw_extrema_y_nC: float | None
    charge_lsb_nC: float | None
    clipping_flag: bool
    skipped_rows: int
    quiet_edge_change_raw_nC: float | None
    quiet_edge_status: str
    qv_voltage_kV: np.ndarray = field(repr=False)
    qv_charge_raw_nC: np.ndarray = field(repr=False)


@dataclass
class CalibrationModel:
    condition: Condition
    sign: int
    sign_agreement: float
    sign_votes: int
    sign_status: str
    passive_complete: bool
    passive_status: str
    cprime_pF: float | None
    cprime_ci_low_pF: float | None
    cprime_ci_high_pF: float | None
    closs_pF: float | None
    closs_ci_low_pF: float | None
    closs_ci_high_pF: float | None
    tan_delta: float | None
    cprime_relative_span: float | None
    passive_slopes_nC_per_kV: dict[int, float | None]
    passive_slope_mad_nC_per_kV: dict[int, float | None]
    passive_threshold_nC: dict[int, float]
    passive_file_slopes: dict[int, list[float]]
    charge_lsb_nC: float | None
    scan_cd_pF: float | None
    scan_cd_ci_low_pF: float | None
    scan_cd_ci_high_pF: float | None
    scan_cd_physical_fraction: float
    scan_cd_status: str
    geometry_cd_pF: float | None
    geometry_factor: float | None
    geometry_factor_low: float | None
    geometry_factor_high: float | None
    factor_source: str
    evidence_tier: str
    failed_gates: list[str]


def parse_arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Estimate polarity-resolved DBD charge transfer with capture-level confidence intervals.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("archive_zip", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("dbd_surface_charge_report"))
    parser.add_argument("--files-per-scan-level", type=int, default=16)
    parser.add_argument(
        "--files-per-maximum",
        type=int,
        default=0,
        help="Maximum-voltage files per condition; 0 uses all captures.",
    )
    parser.add_argument("--bootstrap-replicates", type=int, default=3000)
    parser.add_argument("--bootstrap-block-files", type=int, default=4)
    parser.add_argument("--random-seed", type=int, default=20260718)
    parser.add_argument("--reference-capacitance-uf", type=float, default=0.1)
    parser.add_argument(
        "--reference-capacitance-relative-uncertainty",
        type=float,
        default=0.10,
        help="Bounded scale sensitivity, not a statistical standard deviation.",
    )
    parser.add_argument("--monitor-gain-relative-uncertainty", type=float, default=0.03)
    parser.add_argument("--voltage-scale", type=float, default=1000.0)
    parser.add_argument("--monitor-voltage-scale", type=float, default=1.0)
    parser.add_argument("--carrier-min-khz", type=float, default=100.0)
    parser.add_argument("--carrier-max-khz", type=float, default=160.0)
    parser.add_argument(
        "--charge-polarity",
        choices=("auto", "1", "-1"),
        default="auto",
        help="Locked Channel-D polarity; auto uses each condition's passive ensemble.",
    )
    parser.add_argument("--minimum-sign-agreement", type=float, default=0.90)
    parser.add_argument(
        "--target-negative-on-pin-negative",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Map negative pin-voltage lobes to negative-carrier delivery at the liquid.",
    )
    parser.add_argument("--active-area-mm2", type=float, default=None)
    parser.add_argument("--beaker-diameter-cm", type=float, default=4.0)
    parser.add_argument("--beaker-diameter-relative-range", type=float, default=0.05)
    parser.add_argument("--glass-thickness-mm", type=float, default=1.0)
    parser.add_argument("--glass-thickness-relative-range", type=float, default=0.10)
    parser.add_argument("--pyrex-epsilon-min", type=float, default=4.0)
    parser.add_argument("--pyrex-epsilon-max", type=float, default=5.0)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--no-pdf", action="store_true")
    parser.add_argument("--no-plots", action="store_true")
    return parser.parse_args(argv)


def finite(values: Iterable[float | None]) -> np.ndarray:
    array = np.asarray([value for value in values if value is not None], dtype=float)
    return array[np.isfinite(array)]


def median_or_none(values: Iterable[float | None]) -> float | None:
    array = finite(values)
    return float(np.median(array)) if len(array) else None


def robust_mad(values: Iterable[float | None]) -> float | None:
    array = finite(values)
    if not len(array):
        return None
    center = np.median(array)
    return float(1.4826 * np.median(np.abs(array - center)))


def percentile_interval(values: Iterable[float | None], low: float = 2.5, high: float = 97.5) -> tuple[float | None, float | None]:
    array = finite(values)
    if not len(array):
        return None, None
    return float(np.percentile(array, low)), float(np.percentile(array, high))


def relative_span(values: Iterable[float | None]) -> float | None:
    array = finite(values)
    if len(array) < 2:
        return None
    center = abs(float(np.median(array)))
    if center <= np.finfo(float).eps:
        return None
    return float((np.percentile(array, 90) - np.percentile(array, 10)) / center)


def linear_fit(x: Iterable[float], y: Iterable[float]) -> tuple[float | None, float | None, float | None]:
    xa, ya = finite(x), finite(y)
    if len(xa) != len(ya) or len(xa) < 2 or np.ptp(xa) <= np.finfo(float).eps:
        return None, None, None
    slope, intercept = np.polyfit(xa, ya, 1)
    predicted = slope * xa + intercept
    total = float(np.sum((ya - np.mean(ya)) ** 2))
    r2 = None if total <= 0 else 1.0 - float(np.sum((ya - predicted) ** 2)) / total
    return float(slope), float(intercept), float(r2) if r2 is not None else None


def estimate_charge_lsb_nC(monitor_voltage_V: np.ndarray, capacitance_F: float) -> float | None:
    unique = np.unique(np.asarray(monitor_voltage_V, dtype=float))
    differences = np.diff(unique)
    differences = differences[differences > max(np.finfo(float).eps, 1.0e-12)]
    if not len(differences):
        return None
    # The lower half avoids mistaking skipped ADC codes for one code.
    subset = np.sort(differences)[: max(1, len(differences) // 2)]
    return float(np.median(subset) * capacitance_F * NC_PER_C)


def _crossing_lobes(
    time_s: np.ndarray,
    voltage_V: np.ndarray,
    charge_C: np.ndarray,
    carrier_Hz: float,
    burst_index: int,
) -> list[LobeObservation]:
    if len(time_s) < 10:
        return []
    dt = float(np.median(np.diff(time_s)))
    period_samples = max(5, int(round(1.0 / (carrier_Hz * dt))))
    smooth_window = min(11, max(3, int(round(0.025 * period_samples)) | 1))
    smoothed = moving_average(voltage_V, smooth_window)
    center = 0.5 * (float(np.percentile(smoothed, 2.5)) + float(np.percentile(smoothed, 97.5)))
    centered = smoothed - center
    left, right = centered[:-1], centered[1:]
    indices = np.flatnonzero(((left <= 0) & (right > 0)) | ((left >= 0) & (right < 0)))
    if len(indices) < 3:
        return []
    crossing_times: list[float] = []
    for index in indices:
        denominator = centered[index + 1] - centered[index]
        fraction = 0.5 if denominator == 0 else float(np.clip(-centered[index] / denominator, 0.0, 1.0))
        crossing_times.append(float(time_s[index] + fraction * (time_s[index + 1] - time_s[index])))
    crossing_times_a = np.asarray(crossing_times)
    q_at = np.interp(crossing_times_a, time_s, charge_C)
    half_period = 0.5 / carrier_Hz
    robust_peak = max(abs(float(np.percentile(centered, 1))), abs(float(np.percentile(centered, 99))))
    if robust_peak <= 0:
        return []
    result: list[LobeObservation] = []
    for j, (start, stop) in enumerate(zip(indices[:-1], indices[1:])):
        duration = crossing_times_a[j + 1] - crossing_times_a[j]
        segment = centered[start : stop + 2]
        if len(segment) < 3:
            continue
        amplitude = float(np.max(np.abs(segment)))
        if not (0.60 * half_period <= duration <= 1.40 * half_period):
            continue
        if amplitude < 0.08 * robust_peak:
            continue
        polarity = 1 if float(np.mean(segment)) >= 0 else -1
        directed = polarity * float(q_at[j + 1] - q_at[j]) * NC_PER_C
        result.append(
            LobeObservation(
                burst_index=burst_index,
                voltage_polarity=polarity,
                amplitude_kV=amplitude * 1.0e-3,
                duration_s=float(duration),
                midpoint_s=0.5 * float(crossing_times_a[j + 1] + crossing_times_a[j]),
                raw_directed_charge_nC=directed,
            )
        )
    return result


def extract_lobes(
    time_s: np.ndarray,
    voltage_V: np.ndarray,
    charge_C: np.ndarray,
    carrier_Hz: float,
    burst_kHz: int,
) -> list[LobeObservation]:
    windows = burst_windows(time_s, voltage_V, burst_kHz, carrier_Hz)
    if not windows:
        windows = [slice(0, len(time_s))]
    result: list[LobeObservation] = []
    for burst_index, window in enumerate(windows):
        result.extend(
            _crossing_lobes(
                time_s[window], voltage_V[window], charge_C[window], carrier_Hz, burst_index
            )
        )
    return result


def quiet_edge_charge(
    time_s: np.ndarray,
    voltage_V: np.ndarray,
    charge_C: np.ndarray,
    carrier_Hz: float,
) -> tuple[float | None, str]:
    if len(time_s) < 40:
        return None, "record_too_short"
    dt = float(np.median(np.diff(time_s)))
    count = min(len(time_s) // 4, max(20, int(round(2.0 / (carrier_Hz * dt)))))
    centered = voltage_V - np.median(voltage_V)
    peak = max(abs(float(np.percentile(centered, 1))), abs(float(np.percentile(centered, 99))))
    if peak <= 0:
        return None, "no_voltage_signal"
    start_quiet = float(np.percentile(np.abs(centered[:count]), 95)) < 0.05 * peak
    stop_quiet = float(np.percentile(np.abs(centered[-count:]), 95)) < 0.05 * peak
    if not (start_quiet and stop_quiet):
        return None, "not_measured_no_quiet_record_edges"
    charge_span = max(
        float(np.ptp(charge_C)),
        np.finfo(float).eps,
    )
    for edge_time, edge_charge in (
        (time_s[:count], charge_C[:count]),
        (time_s[-count:], charge_C[-count:]),
    ):
        centered_time = edge_time - edge_time[0]
        slope = float(np.polyfit(centered_time, edge_charge, 1)[0])
        drift = abs(slope * float(centered_time[-1] - centered_time[0]))
        noise = robust_mad(edge_charge)
        tolerance = max(3.0 * float(noise or 0.0), 0.01 * charge_span)
        if drift > tolerance:
            return None, "quiet_voltage_but_charge_not_stationary"
    pre = float(np.median(charge_C[:count]))
    post = float(np.median(charge_C[-count:]))
    return (
        (post - pre) * NC_PER_C,
        "external_terminal_change_measured_dc_coupling_unverified",
    )


def representative_qv(
    time_s: np.ndarray,
    voltage_V: np.ndarray,
    charge_C: np.ndarray,
    burst_kHz: int,
    carrier_Hz: float,
    points: int = 1200,
) -> tuple[np.ndarray, np.ndarray]:
    windows = burst_windows(time_s, voltage_V, burst_kHz, carrier_Hz)
    if windows:
        window = max(windows, key=lambda item: float(np.ptp(voltage_V[item])))
    else:
        window = slice(0, len(time_s))
    u = voltage_V[window]
    q = charge_C[window]
    if len(u) > points:
        indices = np.unique(np.rint(np.linspace(0, len(u) - 1, points)).astype(int))
        u, q = u[indices], q[indices]
    return u * 1.0e-3, q * NC_PER_C


def sign_vote(cstar_raw: complex | None) -> int | None:
    if cstar_raw is None or not np.isfinite(cstar_raw.real) or not np.isfinite(cstar_raw.imag):
        return None
    return choose_q_sign(cstar_raw)


def analyze_file(archive: zipfile.ZipFile, record: MemberRecord, args: argparse.Namespace) -> FileObservation:
    waveform = read_waveform_member(
        archive,
        record,
        args.voltage_scale,
        args.monitor_voltage_scale,
        False,  # use source-to-ground here; the locked monitor sign is applied later
        1,
    )
    time_s = waveform.time_s
    voltage = waveform.source_voltage_V
    capacitance_F = args.reference_capacitance_uf * 1.0e-6
    charge = capacitance_F * waveform.monitor_voltage_V
    carrier = constrained_carrier_frequency(
        time_s,
        voltage,
        args.carrier_min_khz * 1000.0,
        args.carrier_max_khz * 1000.0,
    )
    try:
        duty = select_two_duty_cycles(
            time_s,
            voltage,
            waveform.current_input_A,
        )
        detected_burst = float(duty.frequency_Hz)
        detection_method = duty.method
        if not np.isfinite(detected_burst) or detected_burst >= 0.5 * carrier:
            detected_burst = float(record.condition.burst_kHz * 1000.0)
            detection_method = "nominal_folder_frequency_after_carrier_fallback"
    except WaveformAnalysisError:
        detected_burst = float(record.condition.burst_kHz * 1000.0)
        detection_method = "nominal_folder_frequency_after_detection_failure"
    nominal_burst = float(record.condition.burst_kHz * 1000.0)
    burst_error = abs(detected_burst - nominal_burst) / nominal_burst
    cstar = phasor_ratio(time_s, charge, voltage, carrier)
    used_burst_kHz = detected_burst * 1.0e-3
    lobes = extract_lobes(time_s, voltage, charge, carrier, used_burst_kHz)
    windows = burst_windows(time_s, voltage, used_burst_kHz, carrier)
    raw_x, raw_y = burst_extrema_features(time_s, voltage, charge, windows)
    quiet_change, quiet_status = quiet_edge_charge(time_s, voltage, charge, carrier)
    qv_u, qv_q = representative_qv(
        time_s, voltage, charge, used_burst_kHz, carrier
    )
    return FileObservation(
        record=record,
        level_label="MAX" if record.is_maximum else str(record.level_percent),
        carrier_Hz=carrier,
        detected_burst_Hz=detected_burst,
        burst_detection_method=detection_method,
        burst_frequency_relative_error=burst_error,
        duration_s=float(time_s[-1] - time_s[0]),
        voltage_pp_kV=float((np.percentile(voltage, 99.9) - np.percentile(voltage, 0.1)) * 1.0e-3),
        cstar_raw_F=cstar,
        sign_vote=sign_vote(cstar),
        lobes=lobes,
        raw_extrema_x_kV=raw_x,
        raw_extrema_y_nC=raw_y,
        charge_lsb_nC=estimate_charge_lsb_nC(waveform.monitor_voltage_V, capacitance_F),
        clipping_flag=detect_clipping(
            waveform.source_voltage_V,
            waveform.current_input_A,
            waveform.monitor_voltage_V,
        ),
        skipped_rows=waveform.skipped_rows,
        quiet_edge_change_raw_nC=quiet_change,
        quiet_edge_status=quiet_status,
        qv_voltage_kV=qv_u,
        qv_charge_raw_nC=qv_q,
    )


def bootstrap_median_draws(
    values: Sequence[float],
    replicates: int,
    rng: np.random.Generator,
    block_length: int = 1,
) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    if not len(array):
        return np.asarray([], dtype=float)
    if len(array) == 1:
        return np.full(replicates, array[0])
    block_length = max(1, min(int(block_length), len(array)))
    draws = np.empty(replicates, dtype=float)
    for iteration in range(replicates):
        if block_length == 1:
            indices = rng.integers(0, len(array), size=len(array))
        else:
            starts = rng.integers(0, len(array), size=int(math.ceil(len(array) / block_length)))
            indices = np.concatenate(
                [(start + np.arange(block_length)) % len(array) for start in starts]
            )[: len(array)]
        draws[iteration] = float(np.median(array[indices]))
    return draws


def locked_condition_sign(
    observations: Sequence[FileObservation], requested: str, minimum_agreement: float
) -> tuple[int, float, int, str]:
    if requested in {"1", "-1"}:
        return int(requested), 1.0, 0, "manual_locked"
    passive = [
        row for row in observations
        if row.level_label in {str(level) for level in PASSIVE_FIT_LEVELS}
        and row.sign_vote in {-1, 1}
    ]
    votes = np.asarray([int(row.sign_vote) for row in passive], dtype=int)
    if not len(votes):
        return 1, 0.0, 0, "ambiguous_no_passive_votes"
    sign = 1 if int(np.sum(votes)) >= 0 else -1
    agreement = float(np.mean(votes == sign))
    status = "auto_locked" if agreement >= minimum_agreement else "ambiguous_majority_locked_for_diagnostic"
    return sign, agreement, int(len(votes)), status


def file_passive_slope(row: FileObservation, sign: int, polarity: int) -> float | None:
    values = [
        sign * lobe.raw_directed_charge_nC / lobe.amplitude_kV
        for lobe in row.lobes
        if lobe.voltage_polarity == polarity and lobe.amplitude_kV > 0.02
    ]
    return float(np.median(values)) if values else None


def _level_cstar_values(
    observations: Sequence[FileObservation], sign: int, level: int
) -> list[complex]:
    return [
        sign * row.cstar_raw_F
        for row in observations
        if row.level_label == str(level) and row.cstar_raw_F is not None
    ]


def scan_secant_bootstrap(
    observations: Sequence[FileObservation],
    sign: int,
    replicates: int,
    rng: np.random.Generator,
    ccell_pF: float | None = None,
) -> tuple[float | None, float | None, float | None, float]:
    groups: dict[int, list[tuple[float, float]]] = defaultdict(list)
    for row in observations:
        if row.level_label not in {str(level) for level in HIGH_FIELD_LEVELS}:
            continue
        if row.raw_extrema_x_kV is None or row.raw_extrema_y_nC is None:
            continue
        groups[int(row.level_label)].append(
            (sign * float(row.raw_extrema_x_kV), float(row.raw_extrema_y_nC))
        )
    if any(level not in groups or len(groups[level]) < 2 for level in HIGH_FIELD_LEVELS):
        return None, None, None, 0.0
    central: dict[int, tuple[float, float]] = {}
    for level in HIGH_FIELD_LEVELS:
        values = np.asarray(groups[level], dtype=float)
        central[level] = (float(np.median(values[:, 0])), float(np.median(values[:, 1])))
    denominator = central[115][0] - central[105][0]
    slope = None if abs(denominator) <= 0.02 else (central[115][1] - central[105][1]) / denominator
    draws: list[float] = []
    for _ in range(replicates):
        sampled: dict[int, tuple[float, float]] = {}
        for level in HIGH_FIELD_LEVELS:
            values = np.asarray(groups[level], dtype=float)
            indices = rng.integers(0, len(values), size=len(values))
            selected = values[indices]
            sampled[level] = (float(np.median(selected[:, 0])), float(np.median(selected[:, 1])))
        dx = sampled[115][0] - sampled[105][0]
        if abs(dx) > 0.02:
            draws.append((sampled[115][1] - sampled[105][1]) / dx)
    low, high = percentile_interval(draws)
    physical_limit = 1.05 * ccell_pF if ccell_pF is not None else 0.0
    physical = float(np.mean(np.asarray(draws) > physical_limit)) if draws else 0.0
    return float(slope) if slope is not None else None, low, high, physical


def geometry_cd_pF(diameter_cm: float, thickness_mm: float, epsilon_r: float) -> float:
    epsilon_0 = 8.8541878128e-12
    area_m2 = math.pi * (diameter_cm * 0.01 / 2.0) ** 2
    return epsilon_0 * epsilon_r * area_m2 / (thickness_mm * 1.0e-3) * 1.0e12


def geometry_factor_samples(
    cprime_pF: float,
    args: argparse.Namespace,
    count: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    diameter = args.beaker_diameter_cm * rng.uniform(
        1.0 - args.beaker_diameter_relative_range,
        1.0 + args.beaker_diameter_relative_range,
        size=count,
    )
    thickness = args.glass_thickness_mm * rng.uniform(
        1.0 - args.glass_thickness_relative_range,
        1.0 + args.glass_thickness_relative_range,
        size=count,
    )
    epsilon_r = rng.uniform(args.pyrex_epsilon_min, args.pyrex_epsilon_max, size=count)
    epsilon_0 = 8.8541878128e-12
    area = math.pi * (diameter * 0.01 / 2.0) ** 2
    cd = epsilon_0 * epsilon_r * area / (thickness * 1.0e-3) * 1.0e12
    valid = cd > 1.05 * cprime_pF
    factor = np.full(count, np.nan)
    factor[valid] = cd[valid] / (cd[valid] - cprime_pF)
    return cd, factor


def build_calibration(
    condition: Condition,
    observations: Sequence[FileObservation],
    sign: int,
    sign_agreement: float,
    sign_votes: int,
    sign_status: str,
    args: argparse.Namespace,
    rng: np.random.Generator,
) -> CalibrationModel:
    passive_rows = [row for row in observations if row.level_label in {"40", "60", "75"}]
    passive_complete = all(any(row.level_label == str(level) for row in passive_rows) for level in PASSIVE_FIT_LEVELS)
    cprime_by_level: list[float] = []
    cprime_files: list[float] = []
    closs_files: list[float] = []
    for level in PASSIVE_FIT_LEVELS:
        values = _level_cstar_values(observations, sign, level)
        cprime_by_level.append(float(np.median([value.real * 1.0e12 for value in values])) if values else np.nan)
        cprime_files.extend(value.real * 1.0e12 for value in values)
        closs_files.extend(-value.imag * 1.0e12 for value in values)
    cprime_pF = median_or_none(cprime_files)
    closs_pF = median_or_none(closs_files)
    cprime_parts: list[np.ndarray] = []
    closs_parts: list[np.ndarray] = []
    for level in PASSIVE_FIT_LEVELS:
        level_values = _level_cstar_values(observations, sign, level)
        if not level_values:
            continue
        indices = moving_block_index_matrix(
            len(level_values),
            args.bootstrap_replicates,
            args.bootstrap_block_files,
            rng,
        )
        cprime_parts.append(
            np.asarray([value.real * 1.0e12 for value in level_values])[indices]
        )
        closs_parts.append(
            np.asarray([-value.imag * 1.0e12 for value in level_values])[indices]
        )
    cprime_draws = (
        np.median(np.concatenate(cprime_parts, axis=1), axis=1)
        if cprime_parts else np.asarray([], dtype=float)
    )
    closs_draws = (
        np.median(np.concatenate(closs_parts, axis=1), axis=1)
        if closs_parts else np.asarray([], dtype=float)
    )
    cprime_low, cprime_high = percentile_interval(cprime_draws)
    closs_low, closs_high = percentile_interval(closs_draws)
    tan_delta = None
    if cprime_pF is not None and closs_pF is not None and abs(cprime_pF) > np.finfo(float).eps:
        tan_delta = closs_pF / cprime_pF
    span = relative_span(cprime_by_level)

    slopes: dict[int, float | None] = {}
    slope_mad: dict[int, float | None] = {}
    file_slopes: dict[int, list[float]] = {-1: [], 1: []}
    thresholds: dict[int, float] = {}
    charge_lsb = median_or_none(row.charge_lsb_nC for row in passive_rows)
    for polarity in (-1, 1):
        for row in passive_rows:
            value = file_passive_slope(row, sign, polarity)
            if value is not None:
                file_slopes[polarity].append(value)
        slopes[polarity] = median_or_none(file_slopes[polarity])
        slope_mad[polarity] = robust_mad(file_slopes[polarity])
        residuals: list[float] = []
        if slopes[polarity] is not None:
            for row in passive_rows:
                for lobe in row.lobes:
                    if lobe.voltage_polarity == polarity:
                        residuals.append(
                            sign * lobe.raw_directed_charge_nC
                            - float(slopes[polarity]) * lobe.amplitude_kV
                        )
        residual_array = finite(residuals)
        threshold = float(np.percentile(residual_array, 95)) if len(residual_array) else 0.0
        if charge_lsb is not None:
            threshold = max(threshold, charge_lsb)
        thresholds[polarity] = max(0.0, threshold)

    failed: list[str] = []
    if not passive_complete:
        failed.append("missing_40_60_or_75_percent_level")
    if sign_agreement < args.minimum_sign_agreement:
        failed.append("charge_polarity_vote_ambiguous")
    if cprime_pF is None or cprime_pF <= 0:
        failed.append("nonpositive_Cprime")
    if closs_pF is None or closs_pF < 0:
        failed.append("nonpassive_loss_orientation")
    if (
        cprime_pF is not None
        and closs_pF is not None
        and math.degrees(math.atan2(max(0.0, closs_pF), max(np.finfo(float).eps, cprime_pF))) > 45.0
    ):
        failed.append("passive_phase_exceeds_45_degrees")
    if span is None or span > 0.30:
        failed.append("Cprime_not_stable_across_passive_levels")
    burst_error = median_or_none(row.burst_frequency_relative_error for row in passive_rows)
    if burst_error is None or burst_error > 0.15:
        failed.append("detected_burst_frequency_disagrees_with_condition_label")
    for polarity in (-1, 1):
        slope = slopes.get(polarity)
        spread = slope_mad.get(polarity)
        name = "negative_voltage" if polarity < 0 else "positive_voltage"
        if slope is None or not np.isfinite(slope) or slope <= 0:
            failed.append(f"missing_or_nonpositive_passive_slope_{name}")
        elif spread is None or not np.isfinite(spread) or spread / slope > 0.50:
            failed.append(f"unstable_passive_slope_{name}")
    passive_status = "supported_effective_complex_at_carrier" if not failed else "failed_passive_model_validation"

    scan_cd, scan_low, scan_high, physical_fraction = scan_secant_bootstrap(
        observations, sign, args.bootstrap_replicates, rng, cprime_pF
    )
    scan_gates: list[str] = []
    if scan_cd is None:
        scan_gates.append("missing_or_degenerate_105_115_secant")
    if cprime_pF is None or scan_cd is None or scan_cd <= 1.05 * cprime_pF:
        scan_gates.append("Cd_not_greater_than_Ccell")
    if physical_fraction < 0.80:
        scan_gates.append("bootstrap_physical_fraction_below_0.80")
    # Two levels cannot establish a publication-quality active regression.
    scan_gates.append("only_two_independent_active_amplitudes")
    scan_status = "rejected_diagnostic_two_level_secant" if scan_gates else "validated_effective_Cd"

    geometry_cd = None
    geometry_factor = None
    geometry_low = None
    geometry_high = None
    factor_source = "none"
    evidence = "background_subtracted_terminal_excess"
    if (
        passive_status.startswith("supported")
        and condition.material in CONDUCTIVE_LIQUIDS
        and cprime_pF is not None
        and cprime_pF > 0
    ):
        geometry_cd = geometry_cd_pF(
            args.beaker_diameter_cm,
            args.glass_thickness_mm,
            0.5 * (args.pyrex_epsilon_min + args.pyrex_epsilon_max),
        )
        if geometry_cd > 1.05 * cprime_pF:
            geometry_factor = geometry_cd / (geometry_cd - cprime_pF)
            _, factor_samples = geometry_factor_samples(
                cprime_pF, args, args.bootstrap_replicates, rng
            )
            geometry_low, geometry_high = percentile_interval(factor_samples)
            factor_source = "full_base_pyrex_geometry_scenario"
            evidence = "exploratory_model_dependent"
    if scan_status == "validated_effective_Cd" and scan_cd is not None and cprime_pF is not None:
        geometry_factor = scan_cd / (scan_cd - cprime_pF)
        factor_source = "validated_voltage_scan"
        evidence = "validated_model_dependent"
    if not passive_status.startswith("supported"):
        evidence = "diagnostic_passive_background_model_rejected"
    failed.extend(scan_gates)
    return CalibrationModel(
        condition=condition,
        sign=sign,
        sign_agreement=sign_agreement,
        sign_votes=sign_votes,
        sign_status=sign_status,
        passive_complete=passive_complete,
        passive_status=passive_status,
        cprime_pF=cprime_pF,
        cprime_ci_low_pF=cprime_low,
        cprime_ci_high_pF=cprime_high,
        closs_pF=closs_pF,
        closs_ci_low_pF=closs_low,
        closs_ci_high_pF=closs_high,
        tan_delta=tan_delta,
        cprime_relative_span=span,
        passive_slopes_nC_per_kV=slopes,
        passive_slope_mad_nC_per_kV=slope_mad,
        passive_threshold_nC=thresholds,
        passive_file_slopes=file_slopes,
        charge_lsb_nC=charge_lsb,
        scan_cd_pF=scan_cd,
        scan_cd_ci_low_pF=scan_low,
        scan_cd_ci_high_pF=scan_high,
        scan_cd_physical_fraction=physical_fraction,
        scan_cd_status=scan_status,
        geometry_cd_pF=geometry_cd,
        geometry_factor=geometry_factor,
        geometry_factor_low=geometry_low,
        geometry_factor_high=geometry_high,
        factor_source=factor_source,
        evidence_tier=evidence,
        failed_gates=failed,
    )


def target_name(voltage_polarity: int, negative_on_pin_negative: bool) -> str:
    negative = voltage_polarity < 0 if negative_on_pin_negative else voltage_polarity > 0
    return "negative" if negative else "positive"


def per_file_metrics(
    row: FileObservation,
    calibration: CalibrationModel,
    factor: float,
    negative_on_pin_negative: bool,
) -> dict[str, float]:
    by_target: dict[str, list[tuple[LobeObservation, float]]] = {"negative": [], "positive": []}
    for lobe in row.lobes:
        slope = calibration.passive_slopes_nC_per_kV.get(lobe.voltage_polarity)
        if slope is None:
            continue
        excess = (
            calibration.sign * lobe.raw_directed_charge_nC
            - float(slope) * lobe.amplitude_kV
        )
        corrected = factor * excess
        by_target[target_name(lobe.voltage_polarity, negative_on_pin_negative)].append(
            (lobe, corrected)
        )
    output: dict[str, float] = {}
    totals: dict[str, float] = {}
    for target, pairs in by_target.items():
        if not pairs:
            continue
        charges = np.asarray([value for _, value in pairs], dtype=float)
        durations = np.asarray([lobe.duration_s for lobe, _ in pairs], dtype=float)
        # Select the maximum-amplitude lobe of this target polarity per duty burst.
        peak_envelope: list[tuple[float, float]] = []
        for burst_id in sorted({lobe.burst_index for lobe, _ in pairs}):
            candidates = [(lobe, value) for lobe, value in pairs if lobe.burst_index == burst_id]
            if candidates:
                lobe, value = max(candidates, key=lambda item: item[0].amplitude_kV)
                peak_envelope.append((value, lobe.duration_s))
        peak_values = np.asarray([value for value, _ in peak_envelope], dtype=float)
        peak_durations = np.asarray([duration for _, duration in peak_envelope], dtype=float)
        total_nC = float(np.sum(charges))
        totals[target] = total_nC
        output[f"{target}_peak_envelope_halfcycle_median_nC"] = float(np.median(peak_values))
        output[f"{target}_peak_envelope_halfcycle_p95_nC"] = float(np.percentile(peak_values, 95))
        output[f"{target}_all_halfcycle_p95_nC"] = float(np.percentile(charges, 95))
        output[f"{target}_record_total_nC"] = total_nC
        output[f"{target}_record_average_equivalent_flow_per_s"] = (
            total_nC * 1.0e-9 / (ELEMENTARY_CHARGE_C * row.duration_s)
        )
        rates = peak_values * 1.0e-9 / (ELEMENTARY_CHARGE_C * peak_durations)
        output[f"{target}_peak_halfcycle_average_equivalent_flow_p95_per_s"] = float(
            np.percentile(rates, 95)
        )
        voltage_polarity = -1 if target_name(-1, negative_on_pin_negative) == target else 1
        threshold = factor * calibration.passive_threshold_nC.get(voltage_polarity, 0.0)
        output[f"{target}_resolved_halfcycles"] = float(np.sum(charges > threshold))
        output[f"{target}_total_halfcycles"] = float(len(charges))
        output[f"{target}_resolved_halfcycle_fraction"] = float(np.mean(charges > threshold))
        output[f"{target}_peak_charge_resolution_threshold_nC"] = float(threshold)
        output[f"{target}_peak_rate_resolution_threshold_per_s"] = float(
            np.percentile(
                threshold * 1.0e-9 / (ELEMENTARY_CHARGE_C * peak_durations), 95
            )
        )
    if "positive" in totals and "negative" in totals:
        output["transport_imbalance_nC"] = totals["positive"] - totals["negative"]
    return output


def passive_holdout_limits(
    observations: Sequence[FileObservation],
    calibration: CalibrationModel,
    factor: float,
    negative_on_pin_negative: bool,
) -> dict[str, float]:
    """95th-percentile null limits from the unused 90%-breakdown captures."""
    rows = [row for row in observations if row.level_label == "90"]
    metrics = [
        per_file_metrics(row, calibration, factor, negative_on_pin_negative)
        for row in rows
    ]
    keys = {
        f"{target}_{suffix}"
        for target in ("negative", "positive")
        for suffix in (
            "peak_envelope_halfcycle_p95_nC",
            "peak_halfcycle_average_equivalent_flow_p95_per_s",
            "record_total_nC",
            "record_average_equivalent_flow_per_s",
        )
    }
    output: dict[str, float] = {}
    for key in keys:
        values = finite(row.get(key) for row in metrics)
        if len(values):
            output[key] = max(0.0, float(np.percentile(values, 95)))
    return output


def peak_lobe_audit_rows(
    row: FileObservation,
    calibration: CalibrationModel,
    factor: float,
    negative_on_pin_negative: bool,
) -> list[dict]:
    by_target_burst: dict[tuple[str, int], list[LobeObservation]] = defaultdict(list)
    for lobe in row.lobes:
        target = target_name(lobe.voltage_polarity, negative_on_pin_negative)
        by_target_burst[(target, lobe.burst_index)].append(lobe)
    output: list[dict] = []
    for (target, burst_index), lobes in sorted(by_target_burst.items()):
        lobe = max(lobes, key=lambda item: item.amplitude_kV)
        slope = calibration.passive_slopes_nC_per_kV.get(lobe.voltage_polarity)
        if slope is None:
            continue
        directed = calibration.sign * lobe.raw_directed_charge_nC
        background = float(slope) * lobe.amplitude_kV
        excess = directed - background
        corrected = factor * excess
        output.append({
            "condition": row.record.condition.label,
            "member": row.record.member,
            "capture_index": row.record.capture_index,
            "burst_index": burst_index,
            "target_charge_polarity": target,
            "pin_voltage_polarity": lobe.voltage_polarity,
            "amplitude_kV": lobe.amplitude_kV,
            "duration_s": lobe.duration_s,
            "midpoint_s": lobe.midpoint_s,
            "nominal_directed_terminal_charge_nC": directed,
            "passive_background_nC": background,
            "terminal_excess_nC": excess,
            "model_factor": factor,
            "model_dependent_charge_nC": corrected,
            "halfcycle_average_equivalent_flow_per_s": corrected * 1.0e-9 / (ELEMENTARY_CHARGE_C * lobe.duration_s),
            "above_passive_resolution_threshold": excess > calibration.passive_threshold_nC.get(lobe.voltage_polarity, 0.0),
        })
    return output


def aggregate_with_ci(
    rows: Sequence[dict[str, float]],
    replicates: int,
    block_length: int,
    rng: np.random.Generator,
) -> tuple[dict[str, float], dict[str, tuple[float | None, float | None]], dict[str, np.ndarray]]:
    keys = sorted(set().union(*(row.keys() for row in rows))) if rows else []
    point: dict[str, float] = {}
    intervals: dict[str, tuple[float | None, float | None]] = {}
    draws_by_key: dict[str, np.ndarray] = {}
    for key in keys:
        values = [row[key] for row in rows if key in row and np.isfinite(row[key])]
        if not values:
            continue
        point[key] = float(np.median(values))
        draws = bootstrap_median_draws(values, replicates, rng, block_length)
        intervals[key] = percentile_interval(draws)
        draws_by_key[key] = draws
    return point, intervals, draws_by_key


def moving_block_index_matrix(
    length: int,
    replicates: int,
    block_length: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Circular moving-block indices, one capture-resample per row."""
    if length < 1:
        return np.empty((replicates, 0), dtype=int)
    block_length = max(1, min(int(block_length), length))
    if block_length == 1:
        return rng.integers(0, length, size=(replicates, length))
    n_blocks = int(math.ceil(length / block_length))
    starts = rng.integers(0, length, size=(replicates, n_blocks))
    offsets = np.arange(block_length, dtype=int)
    return ((starts[:, :, None] + offsets) % length).reshape(replicates, -1)[:, :length]


def passive_calibration_draws(
    observations: Sequence[FileObservation],
    calibration: CalibrationModel,
    replicates: int,
    block_length: int,
    rng: np.random.Generator,
) -> tuple[dict[int, np.ndarray], np.ndarray]:
    """Stratified capture bootstrap for lobe slopes and reactive Ccell."""
    slope_parts: dict[int, list[np.ndarray]] = {-1: [], 1: []}
    cprime_parts: list[np.ndarray] = []
    for level in PASSIVE_FIT_LEVELS:
        rows = sorted(
            [row for row in observations if row.level_label == str(level)],
            key=lambda item: item.record.capture_index,
        )
        if not rows:
            continue
        indices = moving_block_index_matrix(len(rows), replicates, block_length, rng)
        cprime_values = np.asarray(
            [
                calibration.sign * row.cstar_raw_F.real * 1.0e12
                if row.cstar_raw_F is not None else np.nan
                for row in rows
            ],
            dtype=float,
        )
        cprime_parts.append(cprime_values[indices])
        for polarity in (-1, 1):
            values = np.asarray(
                [
                    file_passive_slope(row, calibration.sign, polarity)
                    if file_passive_slope(row, calibration.sign, polarity) is not None
                    else np.nan
                    for row in rows
                ],
                dtype=float,
            )
            slope_parts[polarity].append(values[indices])

    slopes: dict[int, np.ndarray] = {}
    for polarity in (-1, 1):
        if slope_parts[polarity]:
            slopes[polarity] = np.nanmedian(
                np.concatenate(slope_parts[polarity], axis=1), axis=1
            )
        else:
            slopes[polarity] = np.full(
                replicates,
                calibration.passive_slopes_nC_per_kV.get(polarity, np.nan),
            )
    if cprime_parts:
        cprime = np.nanmedian(np.concatenate(cprime_parts, axis=1), axis=1)
    else:
        cprime = np.full(replicates, calibration.cprime_pF or np.nan)
    return slopes, cprime


def _file_metric_draws(
    row: FileObservation,
    calibration: CalibrationModel,
    target: str,
    slope_draws: np.ndarray,
    factor_draws: np.ndarray,
    negative_on_pin_negative: bool,
) -> dict[str, np.ndarray]:
    lobes = [
        lobe for lobe in row.lobes
        if target_name(lobe.voltage_polarity, negative_on_pin_negative) == target
    ]
    if not lobes:
        return {}
    raw = calibration.sign * np.asarray(
        [lobe.raw_directed_charge_nC for lobe in lobes], dtype=float
    )
    amplitude = np.asarray([lobe.amplitude_kV for lobe in lobes], dtype=float)
    duration = np.asarray([lobe.duration_s for lobe in lobes], dtype=float)
    selected_indices: list[int] = []
    for burst_index in sorted({lobe.burst_index for lobe in lobes}):
        candidates = [
            index for index, lobe in enumerate(lobes)
            if lobe.burst_index == burst_index
        ]
        selected_indices.append(max(candidates, key=lambda index: amplitude[index]))
    selected = np.asarray(selected_indices, dtype=int)
    peak_charge = factor_draws[:, None] * (
        raw[selected][None, :] - slope_draws[:, None] * amplitude[selected][None, :]
    )
    peak_rate = peak_charge * 1.0e-9 / (
        ELEMENTARY_CHARGE_C * duration[selected][None, :]
    )
    total = factor_draws * (
        float(np.sum(raw)) - slope_draws * float(np.sum(amplitude))
    )
    return {
        f"{target}_peak_envelope_halfcycle_p95_nC": np.percentile(
            peak_charge, 95, axis=1
        ),
        f"{target}_peak_halfcycle_average_equivalent_flow_p95_per_s": np.percentile(
            peak_rate, 95, axis=1
        ),
        f"{target}_record_total_nC": total,
        f"{target}_record_average_equivalent_flow_per_s": (
            total * 1.0e-9 / (ELEMENTARY_CHARGE_C * row.duration_s)
        ),
    }


def hierarchical_analysis_draws(
    max_rows: Sequence[FileObservation],
    calibration_rows: Sequence[FileObservation],
    calibration: CalibrationModel,
    args: argparse.Namespace,
    rng: np.random.Generator,
) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:
    """Jointly resample passive calibration captures and legacy MAX captures."""
    replicates = args.bootstrap_replicates
    slope_draws, cprime_draws = passive_calibration_draws(
        calibration_rows,
        calibration,
        replicates,
        args.bootstrap_block_files,
        rng,
    )
    if (
        calibration.factor_source == "full_base_pyrex_geometry_scenario"
        and calibration.geometry_cd_pF is not None
    ):
        denominator = calibration.geometry_cd_pF - cprime_draws
        factor_draws = np.where(
            denominator > 0.05 * calibration.geometry_cd_pF,
            calibration.geometry_cd_pF / denominator,
            np.nan,
        )
    elif calibration.geometry_factor is not None:
        factor_draws = np.full(replicates, calibration.geometry_factor)
    else:
        factor_draws = np.ones(replicates)

    ordered = sorted(max_rows, key=lambda item: item.record.capture_index)
    if not ordered:
        return {}, factor_draws, cprime_draws
    max_indices = moving_block_index_matrix(
        len(ordered), replicates, args.bootstrap_block_files, rng
    )
    matrices: dict[str, list[np.ndarray]] = defaultdict(list)
    for row in ordered:
        for target in ("negative", "positive"):
            polarity = -1 if target_name(-1, args.target_negative_on_pin_negative) == target else 1
            metrics = _file_metric_draws(
                row,
                calibration,
                target,
                slope_draws[polarity],
                factor_draws,
                args.target_negative_on_pin_negative,
            )
            for key, values in metrics.items():
                matrices[key].append(values)
    output: dict[str, np.ndarray] = {}
    replicate_rows = np.arange(replicates)[:, None]
    for key, columns in matrices.items():
        matrix = np.column_stack(columns)
        sampled = matrix[replicate_rows, max_indices]
        output[key] = np.nanmedian(sampled, axis=1)
    return output, factor_draws, cprime_draws


def factor_sensitivity_draws(
    calibration: CalibrationModel,
    args: argparse.Namespace,
    rng: np.random.Generator,
    count: int,
    cprime_draws: np.ndarray | None = None,
) -> np.ndarray:
    cap_scale = rng.uniform(
        1.0 - args.reference_capacitance_relative_uncertainty,
        1.0 + args.reference_capacitance_relative_uncertainty,
        size=count,
    )
    gain_scale = rng.uniform(
        1.0 - args.monitor_gain_relative_uncertainty,
        1.0 + args.monitor_gain_relative_uncertainty,
        size=count,
    )
    charge_scale = cap_scale * gain_scale
    if (
        calibration.factor_source == "full_base_pyrex_geometry_scenario"
        and calibration.cprime_pF is not None
    ):
        # The same Channel-D scale changes both the terminal excess and the
        # inferred Ccell.  Recompute F for every geometry/scale draw rather
        # than multiplying a nominal-F distribution after the fact.
        cd, _ = geometry_factor_samples(calibration.cprime_pF, args, count, rng)
        cprime = (
            np.asarray(cprime_draws, dtype=float)
            if cprime_draws is not None and len(cprime_draws) == count
            else np.full(count, calibration.cprime_pF)
        )
        magnitude_ratio = 1.0
        if (
            calibration.closs_pF is not None
            and calibration.cprime_pF > 0
            and calibration.closs_pF >= 0
        ):
            magnitude_ratio = math.sqrt(
                1.0 + (calibration.closs_pF / calibration.cprime_pF) ** 2
            )
        # A bounded model-form scenario spans C' to |C*|.  It is not treated
        # as statistical confidence; it exposes the consequence of reducing a
        # lossy complex cell to the scalar classical correction.
        model_form_multiplier = rng.uniform(1.0, magnitude_ratio, size=count)
        scaled_cprime = charge_scale * cprime * model_form_multiplier
        valid = cd > 1.05 * scaled_cprime
        factor = np.full(count, np.nan)
        factor[valid] = cd[valid] / (cd[valid] - scaled_cprime[valid])
    elif calibration.geometry_factor is not None:
        # A scan-derived ratio Cd/(Cd-Ccell) is invariant to a common monitor
        # scale because both fitted capacitances scale together.
        factor = np.full(count, calibration.geometry_factor)
    else:
        factor = np.ones(count)
    return factor * charge_scale


def select_archive_records(
    manifest: Sequence[MemberRecord], args: argparse.Namespace
) -> tuple[dict[tuple[Condition, str], list[MemberRecord]], set[str]]:
    groups: dict[tuple[Condition, str], list[MemberRecord]] = defaultdict(list)
    for record in manifest:
        level = "MAX" if record.is_maximum else str(record.level_percent)
        groups[(record.condition, level)].append(record)
    selected: dict[tuple[Condition, str], list[MemberRecord]] = {}
    members: set[str] = set()
    for key, rows in groups.items():
        count = args.files_per_maximum if key[1] == "MAX" else args.files_per_scan_level
        chosen = evenly_spaced(rows, count)
        selected[key] = chosen
        members.update(row.member for row in chosen)
    return selected, members


def scan_source_map(
    conditions: Sequence[Condition], observations: dict[Condition, list[FileObservation]]
) -> dict[Condition, Condition | None]:
    complete: dict[str, list[Condition]] = defaultdict(list)
    for condition in conditions:
        labels = {row.level_label for row in observations.get(condition, [])}
        if all(str(level) in labels for level in (*PASSIVE_FIT_LEVELS, *HIGH_FIELD_LEVELS)):
            complete[condition.material].append(condition)
    result: dict[Condition, Condition | None] = {}
    for condition in conditions:
        candidates = complete.get(condition.material, [])
        if condition in candidates:
            result[condition] = condition
        elif not candidates:
            result[condition] = None
        elif condition.material == "pure_water":
            result[condition] = min(candidates, key=lambda item: abs(item.burst_kHz - 4))
        else:
            result[condition] = min(candidates, key=lambda item: abs(item.burst_kHz - 20))
    return result


def write_csv(path: Path, rows: Sequence[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def headline_rows(
    summaries: Sequence[dict], long_results: Sequence[dict]
) -> list[dict]:
    """Return a compact, supervisor-facing table with one row per polarity."""
    lookup = {
        (str(row["condition"]), str(row["polarity"]), str(row["metric"])): row
        for row in long_results
    }
    output: list[dict] = []
    metric_columns = {
        "peak_halfcycle_charge_nC": "peak_halfcycle_charge_nC",
        "peak_halfcycle_average_flow_per_s": "peak_halfcycle_average_flow_per_s",
        "whole_record_average_flow_per_s": "whole_record_average_flow_per_s",
        "record_total_charge_nC": "record_total_charge_nC",
    }
    for summary in summaries:
        condition = str(summary.get("condition", ""))
        for polarity in ("negative", "positive"):
            result = {
                "condition": condition,
                "condition_label": summary.get("condition_label"),
                "material": summary.get("material"),
                "burst_kHz": summary.get("burst_kHz"),
                "charge_polarity": polarity,
                "n_max_files": summary.get("n_max_files"),
                "scan_source": summary.get("scan_source"),
                "calibration_transferred": summary.get("scan_transferred"),
                "evidence_tier": summary.get("evidence_tier"),
                "carrier_polarity_assignment": summary.get("carrier_polarity_assignment"),
                "carrier_polarity_assignment_status": summary.get(
                    "carrier_polarity_assignment_status"
                ),
                "Ccell_reactive_pF": summary.get("Ccell_reactive_pF"),
                "Ccell_reactive_ci_low_pF": summary.get("Ccell_reactive_ci_low_pF"),
                "Ccell_reactive_ci_high_pF": summary.get("Ccell_reactive_ci_high_pF"),
                "Cd_scan_status": summary.get("Cd_scan_status"),
                "Cd_geometry_scenario_pF": summary.get("Cd_geometry_scenario_pF"),
                "charge_correction_factor": summary.get("charge_correction_factor"),
                "model_sensitivity_draw_valid_fraction": summary.get(
                    "model_sensitivity_draw_valid_fraction"
                ),
                "retained_charge_nC": summary.get("retained_terminal_charge_nC"),
                "retained_charge_ci_low_nC": summary.get("retained_terminal_charge_ci_low_nC"),
                "retained_charge_ci_high_nC": summary.get("retained_terminal_charge_ci_high_nC"),
                "retained_charge_status": summary.get("retained_charge_status"),
            }
            statuses: list[str] = []
            for metric_name, prefix in metric_columns.items():
                row = lookup.get((condition, polarity, metric_name), {})
                result[prefix] = row.get("estimate")
                result[f"{prefix}_repeat_ci_low"] = row.get("repeat_ci_low")
                result[f"{prefix}_repeat_ci_high"] = row.get("repeat_ci_high")
                result[f"{prefix}_analysis_ci_low"] = row.get("analysis_ci_low")
                result[f"{prefix}_analysis_ci_high"] = row.get("analysis_ci_high")
                result[f"{prefix}_model_sensitivity_low"] = row.get(
                    "model_scale_sensitivity_low"
                )
                result[f"{prefix}_model_sensitivity_high"] = row.get(
                    "model_scale_sensitivity_high"
                )
                status = row.get("evidence_tier")
                result[f"{prefix}_status"] = status
                if status:
                    statuses.append(str(status))
            result["reportability"] = (
                "reportable_with_stated_evidence_tier"
                if statuses and all(not status.startswith(("not_", "consistent_")) for status in statuses)
                else "one_or_more_metrics_not_resolved"
            )
            output.append(result)
    return output


def results_overview_text(rows: Sequence[dict]) -> str:
    def number(value: object, scientific: bool = False) -> str:
        try:
            value_f = float(value)
        except (TypeError, ValueError):
            return "—"
        if not np.isfinite(value_f):
            return "—"
        return f"{value_f:.2e}" if scientific else f"{value_f:.3g}"

    def estimate_with_ci(row: dict, prefix: str, scientific: bool = False) -> str:
        estimate = number(row.get(prefix), scientific)
        if estimate == "—":
            return estimate
        low = number(row.get(f"{prefix}_analysis_ci_low"), scientific)
        high = number(row.get(f"{prefix}_analysis_ci_high"), scientific)
        model_low = number(row.get(f"{prefix}_model_sensitivity_low"), scientific)
        model_high = number(row.get(f"{prefix}_model_sensitivity_high"), scientific)
        return f"{estimate} [{low}, {high}]; model {model_low}–{model_high}"

    lines = [
        "# Results overview",
        "",
        "These are model-qualified electrical charge-equivalent results. Negative carriers mean the net electrical equivalent of electrons plus negative ions; the waveforms do not separate species. `—` means the metric did not clear its reporting gate.",
        "",
        "Each result cell is `estimate [95% joint technical/calibration interval]; model sensitivity range`. The model range is usually the dominant uncertainty.",
        "",
        "| Condition | Polarity | Peak half-cycle (nC) | Peak half-cycle-average rate (e s⁻¹) | Whole-record rate (e s⁻¹) | Evidence/status |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            "| {condition} | {polarity} | {charge} | {peak} | {record} | {status} |".format(
                condition=row.get("condition_label") or row.get("condition"),
                polarity=row.get("charge_polarity"),
                charge=estimate_with_ci(row, "peak_halfcycle_charge_nC"),
                peak=estimate_with_ci(row, "peak_halfcycle_average_flow_per_s", True),
                record=estimate_with_ci(row, "whole_record_average_flow_per_s", True),
                status=f"{row.get('evidence_tier')}; {row.get('reportability')}",
            )
        )
    lines.extend(
        [
            "",
            "Bracketed intervals are 95% joint passive-calibration/MAX-capture moving-block bootstrap intervals. The explicitly displayed model-sensitivity ranges separately include the declared monitor scale, its shared effect on Ccell, a C′-to-|C*| scalar-model bracket, and the full-base Pyrex geometry scenario where used.",
            "",
            "A true instantaneous microdischarge peak is not resolved by the existing sampling. Retained surface charge is not inferred from polarity imbalance; it requires quiet pre/post-discharge plateaus.",
            "",
        ]
    )
    return "\n".join(lines)


def save_figure(fig: plt.Figure, base: Path, dpi: int, pdf: bool) -> None:
    fig.savefig(base.with_suffix(".png"), dpi=dpi, bbox_inches="tight")
    if pdf:
        fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def display_label(condition: Condition) -> str:
    material = {
        "argon_only": "Argon / no liquid",
        "pure_water": "Pure water",
        "BMIM_nitrate": "BMIM nitrate",
        "5mM_Mn_nitrate_in_water": "5 mM Mn nitrate",
    }.get(condition.material, condition.material)
    return f"{material}, {condition.burst_kHz} kHz"


def plot_qv_grid(
    condition: Condition,
    observations: Sequence[FileObservation],
    calibration: CalibrationModel,
    output: Path,
    args: argparse.Namespace,
) -> None:
    levels = [str(level) for level in LEVELS] + ["MAX"]
    fig, axes = plt.subplots(2, 4, figsize=(14, 7.6), sharex=False, sharey=False)
    for axis, level in zip(axes.flat, levels):
        rows = [row for row in observations if row.level_label == level]
        if not rows:
            axis.text(0.5, 0.5, "No data", ha="center", va="center", transform=axis.transAxes)
            axis.set_title(f"{level}{'%' if level != 'MAX' else ''}")
            continue
        for row in rows:
            axis.plot(
                row.qv_voltage_kV,
                calibration.sign * row.qv_charge_raw_nC,
                color="#4d7399",
                alpha=0.10,
                linewidth=0.65,
            )
        representative = min(rows, key=lambda item: abs(item.voltage_pp_kV - np.median([r.voltage_pp_kV for r in rows])))
        axis.plot(
            representative.qv_voltage_kV,
            calibration.sign * representative.qv_charge_raw_nC,
            color="#194d78",
            linewidth=1.2,
            label="median-amplitude capture",
        )
        if calibration.cprime_pF is not None:
            x = np.asarray(axis.get_xlim())
            axis.plot(x, calibration.cprime_pF * x, "--", color=COLORS["passive"], linewidth=1.0, label="$C'_{cell}$ slope guide")
        axis.set_title(f"{level}{'%' if level != 'MAX' else ' (legacy)'}")
        axis.grid(alpha=0.20)
        axis.tick_params(labelsize=8)
    for axis in axes[:, 0]:
        axis.set_ylabel("Nominal monitor charge, Qm (nC)")
    for axis in axes[-1, :]:
        axis.set_xlabel("Applied DBD voltage (kV)")
    title_status = "SUPPORTED passive C*" if calibration.passive_status.startswith("supported") else "PASSIVE MODEL REJECTED"
    fig.suptitle(f"Q–V trajectories: {display_label(condition)}\nLocked Channel-D sign {calibration.sign:+d}; {title_status}", fontsize=12)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False)
    fig.tight_layout(rect=(0, 0.04, 1, 0.94))
    save_figure(fig, output, args.dpi, not args.no_pdf)


def plot_maximum_qv(
    condition: Condition,
    observations: Sequence[FileObservation],
    calibration: CalibrationModel,
    output: Path,
    args: argparse.Namespace,
) -> None:
    rows = [row for row in observations if row.level_label == "MAX"]
    if not rows:
        return
    fig, axis = plt.subplots(figsize=(6.6, 5.4))
    for row in rows:
        axis.plot(
            row.qv_voltage_kV,
            calibration.sign * row.qv_charge_raw_nC,
            color="#517ca4",
            alpha=0.08,
            linewidth=0.6,
        )
    representative = min(
        rows,
        key=lambda item: abs(
            item.voltage_pp_kV - np.median([candidate.voltage_pp_kV for candidate in rows])
        ),
    )
    axis.plot(
        representative.qv_voltage_kV,
        calibration.sign * representative.qv_charge_raw_nC,
        color="#143f65",
        linewidth=1.4,
        label="median-amplitude capture",
    )
    if calibration.cprime_pF is not None:
        x = np.asarray(axis.get_xlim())
        axis.plot(
            x,
            calibration.cprime_pF * x,
            "--",
            color=COLORS["passive"],
            linewidth=1.1,
            label="$C'_{cell}$ slope guide",
        )
    axis.set_xlabel("Applied DBD voltage (kV)")
    axis.set_ylabel("Nominal monitor charge, Qm (nC)")
    axis.set_title(
        f"Legacy maximum-voltage Q–V trajectories\n{display_label(condition)}; "
        f"locked Channel-D sign {calibration.sign:+d}"
    )
    axis.grid(alpha=0.22)
    axis.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    save_figure(fig, output, args.dpi, not args.no_pdf)


def plot_processing_example(
    archive: zipfile.ZipFile,
    row: FileObservation,
    calibration: CalibrationModel,
    output: Path,
    args: argparse.Namespace,
) -> None:
    waveform = read_waveform_member(
        archive,
        row.record,
        args.voltage_scale,
        args.monitor_voltage_scale,
        False,
        1,
    )
    t_ms = waveform.time_s * 1.0e3
    voltage_kV = waveform.source_voltage_V * 1.0e-3
    charge_nC = calibration.sign * args.reference_capacitance_uf * 1.0e3 * waveform.monitor_voltage_V
    y_raw = phasor_ratio(
        waveform.time_s,
        waveform.current_input_A,
        waveform.source_voltage_V,
        row.carrier_Hz,
    )
    current_sign = choose_current_sign(y_raw)
    current_A = current_sign * waveform.current_input_A
    peak_index = int(np.argmax(np.abs(voltage_kV)))
    half_width = max(5, int(round(1.0 / (row.carrier_Hz * np.median(np.diff(waveform.time_s))))))
    start, stop = max(0, peak_index - half_width), min(len(t_ms), peak_index + half_width + 1)
    fig, axes = plt.subplots(3, 2, figsize=(13, 7.4), sharex="col")
    signals = (
        (voltage_kV, "Applied DBD voltage (kV)", "#2e5687"),
        (
            current_A,
            "Polarity-adjusted Pearson signal (A)\n(upstream diagnostic)",
            "#9b5142",
        ),
        (charge_nC, "Nominal monitor charge (nC)", "#287a68"),
    )
    stride = max(1, len(t_ms) // 30000)
    for axis_row, (values, label, color) in zip(axes, signals):
        axis_row[0].plot(t_ms[::stride], values[::stride], color=color, linewidth=0.8)
        axis_row[1].plot(t_ms[start:stop], values[start:stop], color=color, linewidth=1.0)
        axis_row[0].set_ylabel(label)
        axis_row[1].grid(alpha=0.20)
        axis_row[0].grid(alpha=0.20)
    center = 0.5 * (
        float(np.percentile(voltage_kV[start:stop], 2.5))
        + float(np.percentile(voltage_kV[start:stop], 97.5))
    )
    centered = voltage_kV[start:stop] - center
    crossings = np.flatnonzero(np.signbit(centered[:-1]) != np.signbit(centered[1:]))
    for j in range(len(crossings) - 1):
        left = t_ms[start + crossings[j]]
        right = t_ms[start + crossings[j + 1]]
        polarity = np.mean(centered[crossings[j] : crossings[j + 1] + 1]) >= 0
        axes[0, 1].axvspan(
            left,
            right,
            color=COLORS["positive" if polarity else "negative"],
            alpha=0.09,
        )
    zoom_start_s = waveform.time_s[start]
    zoom_stop_s = waveform.time_s[stop - 1]
    zoom_lobes = [
        lobe for lobe in row.lobes
        if zoom_start_s <= lobe.midpoint_s <= zoom_stop_s
    ]
    boundaries_ms = sorted(
        {
            boundary * 1.0e3
            for lobe in zoom_lobes
            for boundary in (
                lobe.midpoint_s - 0.5 * lobe.duration_s,
                lobe.midpoint_s + 0.5 * lobe.duration_s,
            )
        }
    )
    for boundary in boundaries_ms:
        for axis in axes[:, 1]:
            axis.axvline(boundary, color="#555555", linewidth=0.55, alpha=0.28)
    if zoom_lobes:
        example_lobe = max(zoom_lobes, key=lambda lobe: lobe.amplitude_kV)
        t0 = example_lobe.midpoint_s - 0.5 * example_lobe.duration_s
        t1 = example_lobe.midpoint_s + 0.5 * example_lobe.duration_s
        endpoints_q = np.interp([t0, t1], waveform.time_s, charge_nC)
        axes[2, 1].scatter(
            [t0 * 1.0e3, t1 * 1.0e3],
            endpoints_q,
            color="#111111",
            s=22,
            zorder=5,
        )
        slope = calibration.passive_slopes_nC_per_kV.get(
            example_lobe.voltage_polarity
        )
        if slope is not None:
            directed = calibration.sign * example_lobe.raw_directed_charge_nC
            background = float(slope) * example_lobe.amplitude_kV
            excess = directed - background
            factor = calibration.geometry_factor or 1.0
            axes[2, 1].text(
                0.02,
                0.97,
                (
                    f"Example lobe: Dₕ={directed:.1f} nC\n"
                    f"Bₛ(Aₕ)={background:.1f} nC\n"
                    f"Xₕ=Dₕ−Bₛ={excess:.1f} nC\n"
                    f"F={factor:.2f}; F·Xₕ={factor*excess:.1f} nC"
                ),
                transform=axes[2, 1].transAxes,
                ha="left",
                va="top",
                fontsize=7.5,
                bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.82, "edgecolor": "#bbbbbb"},
            )
    axes[-1, 0].set_xlabel("Time (ms)")
    axes[-1, 1].set_xlabel("Time (ms)")
    axes[0, 0].set_title("Full 10 ms acquisition")
    axes[0, 1].set_title(
        "Two-carrier-cycle illustrative zoom; example lobe is not the headline p95"
    )
    fig.suptitle(
        f"Signal-processing example: {display_label(row.record.condition)}, legacy MAX\n"
        "Pearson is an upstream diagnostic; charge and reported totals use Channel D",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    save_figure(fig, output, args.dpi, not args.no_pdf)


def _level_point_stats(
    rows: Sequence[FileObservation], sign: int, level: str
) -> dict[str, float | None]:
    selected = [row for row in rows if row.level_label == level]
    return {
        "vpp": median_or_none(row.voltage_pp_kV for row in selected),
        "vpp_low": percentile_interval(row.voltage_pp_kV for row in selected)[0],
        "vpp_high": percentile_interval(row.voltage_pp_kV for row in selected)[1],
        "x": median_or_none(sign * row.raw_extrema_x_kV if row.raw_extrema_x_kV is not None else None for row in selected),
        "x_low": percentile_interval(sign * row.raw_extrema_x_kV if row.raw_extrema_x_kV is not None else None for row in selected)[0],
        "x_high": percentile_interval(sign * row.raw_extrema_x_kV if row.raw_extrema_x_kV is not None else None for row in selected)[1],
        "y": median_or_none(row.raw_extrema_y_nC for row in selected),
        "y_low": percentile_interval(row.raw_extrema_y_nC for row in selected)[0],
        "y_high": percentile_interval(row.raw_extrema_y_nC for row in selected)[1],
    }


def plot_scan_fit(
    condition: Condition,
    observations: Sequence[FileObservation],
    calibration: CalibrationModel,
    output: Path,
    args: argparse.Namespace,
) -> list[dict]:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    stats_rows: list[dict] = []
    level_colors = plt.cm.viridis(np.linspace(0.08, 0.92, len(LEVELS)))
    for color, level in zip(level_colors, LEVELS):
        selected = [row for row in observations if row.level_label == str(level)]
        for row in selected:
            axes[0].scatter(level, row.voltage_pp_kV, color=color, alpha=0.20, s=13)
            if row.raw_extrema_x_kV is not None and row.raw_extrema_y_nC is not None:
                axes[1].scatter(
                    calibration.sign * row.raw_extrema_x_kV,
                    row.raw_extrema_y_nC,
                    color=color,
                    alpha=0.20,
                    s=13,
                )
        stats = _level_point_stats(observations, calibration.sign, str(level))
        stats_rows.append({"condition": condition.label, "level": level, **stats})
        if stats["vpp"] is not None:
            axes[0].errorbar(
                level,
                stats["vpp"],
                yerr=[[stats["vpp"] - stats["vpp_low"]], [stats["vpp_high"] - stats["vpp"]]],
                fmt="o",
                color=color,
                capsize=3,
                markersize=6,
            )
        if stats["x"] is not None and stats["y"] is not None:
            axes[1].errorbar(
                stats["x"],
                stats["y"],
                xerr=[[stats["x"] - stats["x_low"]], [stats["x_high"] - stats["x"]]],
                yerr=[[stats["y"] - stats["y_low"]], [stats["y_high"] - stats["y"]]],
                fmt="o",
                color=color,
                capsize=3,
                markersize=6,
                label=f"{level}%",
            )
    max_rows = [row for row in observations if row.level_label == "MAX"]
    for row in max_rows:
        if row.raw_extrema_x_kV is not None and row.raw_extrema_y_nC is not None:
            axes[1].scatter(
                calibration.sign * row.raw_extrema_x_kV,
                row.raw_extrema_y_nC,
                marker="*",
                facecolors="none",
                edgecolors="#222222",
                alpha=0.25,
                s=70,
            )
    scan_levels = [level for level in LEVELS if level <= 115]
    vpp_points = [_level_point_stats(observations, calibration.sign, str(level))["vpp"] for level in scan_levels]
    valid = [(x, y) for x, y in zip(scan_levels, vpp_points) if y is not None]
    if len(valid) >= 2:
        slope, intercept, r2 = linear_fit([x for x, _ in valid], [y for _, y in valid])
        grid = np.linspace(min(x for x, _ in valid), max(x for x, _ in valid), 100)
        axes[0].plot(grid, slope * grid + intercept, color="#333333", linewidth=1.2)
        axes[0].text(0.03, 0.96, f"Measured Vpp fit: $R^2$={r2:.3f}", va="top", transform=axes[0].transAxes, fontsize=8)

    passive_points = [
        _level_point_stats(observations, calibration.sign, str(level)) for level in PASSIVE_FIT_LEVELS
    ]
    px = [row["x"] for row in passive_points if row["x"] is not None and row["y"] is not None]
    py = [row["y"] for row in passive_points if row["x"] is not None and row["y"] is not None]
    pslope, pintercept, pr2 = linear_fit(px, py)
    if pslope is not None:
        grid = np.linspace(min(px), max(px), 100)
        style = "-" if calibration.passive_status.startswith("supported") else "--"
        axes[1].plot(grid, pslope * grid + pintercept, style, color=COLORS["passive"], linewidth=1.5, label=f"40–75% fit ({'accepted' if style == '-' else 'rejected'})")
    high_points = [
        _level_point_stats(observations, calibration.sign, str(level)) for level in HIGH_FIELD_LEVELS
    ]
    hx = [row["x"] for row in high_points if row["x"] is not None and row["y"] is not None]
    hy = [row["y"] for row in high_points if row["x"] is not None and row["y"] is not None]
    if len(hx) == 2 and abs(hx[1] - hx[0]) > 0.02:
        slope = (hy[1] - hy[0]) / (hx[1] - hx[0])
        intercept = hy[0] - slope * hx[0]
        grid = np.linspace(min(hx), max(hx), 100)
        axes[1].plot(grid, slope * grid + intercept, "--", color="#b23a48", linewidth=1.5, label="105–115% secant (diagnostic)")
    axes[0].set_xlabel("Commanded breakdown voltage (%)")
    axes[0].set_ylabel("Measured voltage, Vpp (kV)")
    axes[1].set_xlabel(r"$U(Q_+) - U(Q_-)$ (kV)")
    axes[1].set_ylabel(r"$Q_+ - Q_-$ (nC)")
    axes[0].grid(alpha=0.22)
    axes[1].grid(alpha=0.22)
    axes[1].legend(fontsize=7, frameon=False, ncol=2)
    fig.suptitle(
        f"Breakdown-percentage fits: {display_label(condition)}\n"
        "Small points: captures; bars: 2.5–97.5% capture interval; legacy MAX is out-of-sample",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    save_figure(fig, output, args.dpi, not args.no_pdf)
    return stats_rows


def plot_capacitance_summary(
    calibrations: dict[Condition, CalibrationModel],
    observations: dict[Condition, list[FileObservation]],
    output: Path,
    args: argparse.Namespace,
) -> None:
    supported = [condition for condition, model in calibrations.items() if model.passive_status.startswith("supported")]
    if not supported:
        return
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.5), sharex=True)
    condition_colors = plt.cm.tab10(np.linspace(0.0, 0.8, len(supported)))
    offsets = np.linspace(-1.8, 1.8, len(supported)) if len(supported) > 1 else np.asarray([0.0])
    for offset, color, condition in zip(offsets, condition_colors, supported):
        model = calibrations[condition]
        for level in PASSIVE_FIT_LEVELS:
            values = _level_cstar_values(observations[condition], model.sign, level)
            cprime = np.asarray([value.real * 1.0e12 for value in values])
            closs = np.asarray([-value.imag * 1.0e12 for value in values])
            for axis, data, label in ((axes[0], cprime, "C'"), (axes[1], closs, "C''")):
                axis.scatter(
                    np.full(len(data), level + offset),
                    data,
                    s=14,
                    alpha=0.25,
                    color=color,
                )
                if len(data):
                    low, high = np.percentile(data, [2.5, 97.5])
                    axis.errorbar(
                        level + offset,
                        np.median(data),
                        yerr=[[np.median(data) - low], [high - np.median(data)]],
                        fmt="o",
                        color=color,
                        capsize=3,
                        label=display_label(condition) if level == 40 else None,
                    )
        axes[0].axhline(model.cprime_pF, linestyle="--", color=color, linewidth=1.0, alpha=0.7)
        axes[1].axhline(model.closs_pF, linestyle="--", color=color, linewidth=1.0, alpha=0.7)
    axes[0].set_ylabel("Reactive capacitance, C' (pF)")
    axes[1].set_ylabel("Loss capacitance, C'' (pF)")
    for axis in axes:
        axis.set_xlabel("Breakdown voltage (%)")
        axis.set_xticks(PASSIVE_FIT_LEVELS)
        axis.grid(alpha=0.22)
        axis.legend(fontsize=8, frameon=False)
    fig.suptitle("Supported carrier-frequency complex cell capacitance\nPoints are captures; bars are 2.5–97.5% capture intervals")
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    save_figure(fig, output, args.dpi, not args.no_pdf)


def summary_metric_reportable(row: dict, target: str, metric: str) -> bool:
    key = f"{target}_{metric}"
    point = row.get(key)
    low = row.get(f"{key}_analysis_ci_low")
    high = row.get(f"{key}_analysis_ci_high")
    if not str(row.get("Ccell_status", "")).startswith("supported"):
        return False
    if float(row.get("model_sensitivity_draw_valid_fraction") or 0.0) < 0.80:
        return False
    if not (
        point is not None and low is not None and high is not None
        and np.isfinite(point) and np.isfinite(low) and np.isfinite(high)
        and point > 0 and low <= point <= high
    ):
        return False
    resolution = 0.0
    if metric == "peak_envelope_halfcycle_p95_nC":
        resolution = max(
            float(row.get(f"{target}_peak_charge_resolution_threshold_nC") or 0.0),
            float(row.get(f"{key}_passive_90_holdout_p95") or 0.0),
        )
    elif metric == "peak_halfcycle_average_equivalent_flow_p95_per_s":
        resolution = max(
            float(row.get(f"{target}_peak_rate_resolution_threshold_per_s") or 0.0),
            float(row.get(f"{key}_passive_90_holdout_p95") or 0.0),
        )
    elif not bool(row.get("scan_transferred")):
        resolution = max(
            0.0, float(row.get(f"{key}_passive_90_holdout_p95") or 0.0)
        )
    resolved_fraction = float(row.get(f"{target}_resolved_halfcycle_fraction") or 0.0)
    return low > resolution and (
        metric.startswith("peak_") or resolved_fraction >= 0.05
    )


def plot_result_forest(
    summaries: Sequence[dict],
    metric: str,
    title: str,
    xlabel: str,
    output: Path,
    args: argparse.Namespace,
    log_scale: bool = False,
) -> None:
    labels = [row["condition_label"] for row in summaries]
    y = np.arange(len(labels))
    fig, axis = plt.subplots(figsize=(11, max(5.5, 0.46 * len(labels))))
    for offset, target in ((-0.12, "negative"), (0.12, "positive")):
        for index, row in enumerate(summaries):
            key = f"{target}_{metric}"
            point = row.get(key)
            low = row.get(f"{key}_analysis_ci_low")
            high = row.get(f"{key}_analysis_ci_high")
            supported = str(row.get("Ccell_status", "")).startswith("supported")
            reportable = summary_metric_reportable(row, target, metric)
            point_valid = point is not None and np.isfinite(point) and point > 0
            interval_valid = (
                point_valid
                and low is not None
                and high is not None
                and np.isfinite(low)
                and np.isfinite(high)
                and low <= point <= high
                and (not log_scale or low > 0)
            )
            if reportable and interval_valid:
                transferred = bool(row.get("scan_transferred"))
                axis.errorbar(
                    point,
                    index + offset,
                    xerr=[[point - low], [high - point]],
                    fmt="o",
                    markerfacecolor="none" if transferred else COLORS[target],
                    markeredgecolor=COLORS[target],
                    color=COLORS[target],
                    capsize=3,
                )
            elif point_valid:
                axis.scatter(
                    point,
                    index + offset,
                    marker="x",
                    color=COLORS["invalid"],
                    s=28,
                    zorder=3,
                )
            else:
                axis.text(
                    0.985,
                    index + offset,
                    "NR",
                    transform=axis.get_yaxis_transform(),
                    ha="right",
                    va="center",
                    color=COLORS["invalid"],
                    fontsize=7,
                )
            slo = row.get(f"{key}_sensitivity_low")
            shi = row.get(f"{key}_sensitivity_high")
            if (
                reportable
                and point_valid
                and slo is not None
                and shi is not None
                and np.isfinite(slo)
                and np.isfinite(shi)
                and shi > slo
                and (not log_scale or slo > 0)
            ):
                axis.plot([slo, shi], [index + offset, index + offset], color=COLORS[target], alpha=0.22, linewidth=7, solid_capstyle="butt")
    axis.set_yticks(y)
    axis.set_yticklabels(labels, fontsize=8)
    axis.set_ylim(len(labels) - 0.5, -0.5)
    axis.set_xlabel(xlabel)
    axis.set_title(
        title
        + "\nThin bars: 95% joint capture/calibration bootstrap; broad bars: model/scale sensitivity"
        + "\nNegative pin-voltage lobes are assigned to negative-carrier delivery (explicit assumption)"
        + "\nSupported liquid estimates use an exploratory full-base-Pyrex Cd scenario"
    )
    if log_scale:
        axis.set_xscale("log")
    axis.grid(axis="x", alpha=0.22)
    axis.scatter([], [], marker="o", color=COLORS["negative"], label="Negative-carrier assignment")
    axis.scatter([], [], marker="o", color=COLORS["positive"], label="Positive-carrier assignment")
    axis.scatter([], [], marker="o", facecolors="none", edgecolors="#333333", label="Open: calibration transferred")
    axis.scatter([], [], marker="x", color=COLORS["invalid"], label="× / NR: not reportable")
    axis.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    save_figure(fig, output, args.dpi, not args.no_pdf)


def plot_negative_rate_frequency(
    summaries: Sequence[dict], output: Path, args: argparse.Namespace
) -> None:
    """Direct comparison requested most often: negative rate versus burst frequency."""
    material_colors = {
        "argon_only": "#777777",
        "pure_water": "#2f7ebc",
        "BMIM_nitrate": "#7a4fa3",
        "5mM_Mn_nitrate_in_water": "#b26a2b",
    }
    material_labels = {
        "argon_only": "Ar / no liquid",
        "pure_water": "Pure water",
        "BMIM_nitrate": "BMIM nitrate",
        "5mM_Mn_nitrate_in_water": "5 mM Mn nitrate",
    }
    key = "negative_record_average_equivalent_flow_per_s"
    fig, axis = plt.subplots(figsize=(8.4, 5.6))
    for material in sorted({str(row.get("material")) for row in summaries}):
        rows = sorted(
            [row for row in summaries if row.get("material") == material],
            key=lambda row: float(row.get("burst_kHz") or 0),
        )
        color = material_colors.get(material, "#444444")
        valid_line = [
            row for row in rows
            if summary_metric_reportable(
                row, "negative", "record_average_equivalent_flow_per_s"
            )
            and row.get(key) is not None
            and np.isfinite(row[key])
            and row[key] > 0
        ]
        if valid_line:
            axis.plot(
                [row["burst_kHz"] for row in valid_line],
                [row[key] for row in valid_line],
                color=color,
                linewidth=1.0,
                linestyle="--" if any(row.get("scan_transferred") for row in valid_line) else "-",
                alpha=0.65,
                label=material_labels.get(material, material),
            )
        else:
            continue
        for row in rows:
            point = row.get(key)
            if point is None or not np.isfinite(point) or point <= 0:
                continue
            supported = str(row.get("Ccell_status", "")).startswith("supported")
            reportable = summary_metric_reportable(
                row, "negative", "record_average_equivalent_flow_per_s"
            )
            low = row.get(f"{key}_analysis_ci_low")
            high = row.get(f"{key}_analysis_ci_high")
            if (
                reportable and low is not None and high is not None
                and np.isfinite(low) and np.isfinite(high) and low > 0
                and low <= point <= high
            ):
                axis.errorbar(
                    row["burst_kHz"],
                    point,
                    yerr=[[point - low], [high - point]],
                    fmt="o",
                    color=color,
                    markerfacecolor="none" if row.get("scan_transferred") else color,
                    markeredgecolor=color,
                    capsize=3,
                    zorder=3,
                )
                slo, shi = row.get(f"{key}_sensitivity_low"), row.get(f"{key}_sensitivity_high")
                if (
                    slo is not None and shi is not None
                    and np.isfinite(slo) and np.isfinite(shi) and slo > 0 and shi > slo
                ):
                    axis.plot(
                        [row["burst_kHz"], row["burst_kHz"]],
                        [slo, shi],
                        color=color,
                        alpha=0.18,
                        linewidth=7,
                        solid_capstyle="butt",
                        zorder=1,
                    )
            elif reportable:
                axis.scatter(
                    row["burst_kHz"],
                    point,
                    marker="o",
                    facecolors="none" if row.get("scan_transferred") else color,
                    edgecolors=color,
                    s=38,
                    zorder=3,
                )
            else:
                axis.scatter(
                    row["burst_kHz"], point, marker="x", color=color, alpha=0.55, s=38
                )
    axis.set_yscale("log")
    axis.set_xticks([4, 10, 20])
    axis.set_xlabel("Duty-burst frequency (kHz)")
    axis.set_ylabel("Net negative charge-equivalent delivery rate (e s⁻¹)")
    axis.set_title(
        "Negative whole-record delivery-rate comparison\n"
        "Filled: independently scanned; open: transferred calibration; ×: rejected diagnostic\n"
        "Exploratory full-base Pyrex model; dashed connections are guides to the eye"
    )
    axis.text(
        0.99,
        0.02,
        "Ar/no-liquid and water omitted: passive background model rejected\n"
        "Negative pin-voltage assignment assumed",
        transform=axis.transAxes,
        ha="right",
        va="bottom",
        fontsize=8,
        color="#555555",
    )
    axis.grid(alpha=0.22, which="both")
    axis.legend(frameon=False, fontsize=8, ncol=2)
    fig.tight_layout()
    save_figure(fig, output, args.dpi, not args.no_pdf)


def plot_retained_status(summaries: Sequence[dict], output: Path, args: argparse.Namespace) -> None:
    labels = [row["condition_label"] for row in summaries]
    counts = [row.get("quiet_edge_files", 0) for row in summaries]
    totals = [row.get("n_max_files", 0) for row in summaries]
    fig, axis = plt.subplots(figsize=(10, max(4.5, 0.42 * len(labels))))
    y = np.arange(len(labels))
    axis.barh(y, totals, color="#dddddd", label="MAX captures")
    axis.barh(y, counts, color=COLORS["passive"], label="Quiet at both record edges")
    axis.set_yticks(y)
    axis.set_yticklabels(labels, fontsize=8)
    axis.invert_yaxis()
    axis.set_xlabel("Number of captures")
    axis.set_title("Persistent retained charge requires quiet pre- and post-record plateaus\nTransport imbalance is not substituted for retained surface charge")
    axis.legend(frameon=False)
    axis.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    save_figure(fig, output, args.dpi, not args.no_pdf)


def plot_coverage(
    manifest: Sequence[MemberRecord], selected_members: set[str], output: Path, args: argparse.Namespace
) -> None:
    conditions = sorted({row.condition for row in manifest})
    level_labels = [str(level) for level in LEVELS] + ["MAX"]
    matrix = np.zeros((len(conditions), len(level_labels)))
    for i, condition in enumerate(conditions):
        for j, level in enumerate(level_labels):
            matrix[i, j] = sum(
                row.condition == condition
                and ("MAX" if row.is_maximum else str(row.level_percent)) == level
                and row.member in selected_members
                for row in manifest
            )
    fig, axis = plt.subplots(figsize=(10.5, max(4.5, 0.42 * len(conditions))))
    image = axis.imshow(matrix, aspect="auto", cmap="Blues", vmin=0, vmax=max(1, np.max(matrix)))
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            text = "—" if matrix[i, j] == 0 else str(int(matrix[i, j]))
            axis.text(j, i, text, ha="center", va="center", fontsize=8, color="white" if matrix[i, j] > 0.55 * np.max(matrix) else "#222222")
    axis.set_xticks(np.arange(len(level_labels)))
    axis.set_xticklabels([f"{level}%" if level != "MAX" else "Legacy MAX" for level in level_labels], rotation=35, ha="right")
    axis.set_yticks(np.arange(len(conditions)))
    axis.set_yticklabels([display_label(condition) for condition in conditions], fontsize=8)
    axis.set_title("Waveform coverage used in the reporting analysis\nCells are separate waveform captures (sequential technical repeats)")
    fig.colorbar(image, ax=axis, label="Selected captures")
    fig.tight_layout()
    save_figure(fig, output, args.dpi, not args.no_pdf)


def flatten_calibration(model: CalibrationModel) -> dict:
    row = asdict(model)
    row["condition"] = model.condition.label
    row["material"] = model.condition.material
    row["burst_kHz"] = model.condition.burst_kHz
    row["failed_gates"] = ";".join(model.failed_gates)
    for name in ("passive_slopes_nC_per_kV", "passive_slope_mad_nC_per_kV", "passive_threshold_nC"):
        values = row.pop(name)
        for polarity, value in values.items():
            row[f"{name}_{'negative_voltage' if int(polarity) < 0 else 'positive_voltage'}"] = value
    row.pop("passive_file_slopes", None)
    return row


def methodology_text(args: argparse.Namespace) -> str:
    return f"""# DBD surface-charge reporting analysis

Analysis version: `{ANALYSIS_VERSION}`

## Definitions

The nominal monitor charge is `Qm(t) = Cref Vmonitor(t)`, with
`Cref = {args.reference_capacitance_uf:g} µF`. Channel-D polarity is inferred
once from the 40/60/75 %-breakdown ensemble for each independently scanned
condition and then locked for every percentage and legacy MAX capture.
This sign lock orients the electrical monitor; it does not establish which
carrier species reaches the liquid. The default reporting assignment treats a
negative pin-voltage lobe as negative-carrier delivery, and every table/figure
records that as an explicit, unvalidated physical assumption.

The duty-burst frequency is measured from the waveform activity envelope for
each capture. Folder frequency is used only if envelope detection falls back to
the carrier or fails; a >15 % mismatch fails passive-model validation.

Adjacent interpolated carrier-voltage zero crossings define a half-cycle. For
voltage polarity `s = ±1`, the directed terminal charge is

`D_h = s [Qm(t1) - Qm(t0)]`.

Separate passive functions for positive and negative voltage lobes are fitted
from 40/60/75 % data. The operational terminal excess is

`X_h = D_h - B_s(A_h)`.

The 90 % captures are excluded from that fit and used as an out-of-sample
passive null. Peak metrics must exceed the larger of the training-residual
limit and the 90 %-holdout p95 at the lower endpoint of the joint 95 % analysis
interval. Same-frequency whole-record metrics are compared with the same
90 %-holdout statistic. Transferred-frequency whole-record results have no
same-frequency holdout and are labeled accordingly.

When a valid dielectric capacitance is available, the classical model gives

`q_surface,h = [Cd/(Cd-Ccell)] X_h`.

Geometry-derived `Cd` results are explicitly exploratory. The active 105/115 %
secant is diagnostic because only two independent active amplitudes are
available; legacy MAX is out-of-sample and is never used to fit `Cd`.

## Requested outputs

- **Charge per peak half-cycle:** p95 across the maximum-amplitude carrier lobe
  of each polarity in each duty burst.
- **Peak rate:** p95 of half-cycle-average `q/(e Δt)` across the same
  maximum-amplitude lobe selected once per duty burst. It is not an
  instantaneous nanosecond particle flux.
- **Whole-record rate:** signed sum of background-subtracted lobe charge divided
  by elementary charge and full record duration, including duty-off time.
- **Retained charge:** only measured when stable quiet plateaus exist at both
  record edges. Polarity imbalance is reported separately and never relabeled
  retained surface charge.

Negative rate means a **net external-terminal electrical equivalent** assigned
to electrons plus negative ions under the configured pin-polarity mapping. It
is not a species-resolved gross particle count, and memory voltage can shift
actual gas conduction relative to a source-voltage zero crossing.
Area-normalized flux is blank unless an active area is supplied.

## Uncertainty

The independent sampling unit is a waveform capture, not a carrier half-cycle.
`repeat_ci` is a conditional MAX-capture repeatability interval with the passive
calibration fixed. `analysis_ci` jointly resamples 40/60/75 % calibration
captures within level and legacy MAX captures using a
{args.bootstrap_replicates}-replicate moving-block bootstrap with block length
{args.bootstrap_block_files}. These are technical-repeat intervals; the 64
sequential captures are not independent biological or experimental repeats.

Broad model-sensitivity intervals additionally sample the declared monitor
scale and approximate full-base Pyrex geometry, propagate the common Channel-D
scale through both terminal charge and Ccell in `Cd/(Cd-Ccell)`, and span the
scalar-cell choice from `C'` to `|C*|`. These are bounded scenario ranges, not
frequentist confidence intervals, and they still do not cover unknown active
area or every possible circuit-model error. Invalid correction-factor draws
are counted and the result is not reportable if more than 20 % are unphysical.

The nominal monitor-capacitance scale sensitivity is ±
{100*args.reference_capacitance_relative_uncertainty:.1f} %, and Channel-D gain
sensitivity is ±{100*args.monitor_gain_relative_uncertainty:.1f} %.
"""


def analyze_archive(args: argparse.Namespace) -> dict:
    if args.files_per_scan_level < 0 or args.files_per_maximum < 0:
        raise SurfaceChargeError("File counts must be zero or positive.")
    if args.bootstrap_replicates < 200:
        raise SurfaceChargeError("Use at least 200 bootstrap replicates.")
    archive_path = args.archive_zip.expanduser().resolve()
    if not archive_path.is_file():
        raise SurfaceChargeError(f"Archive does not exist: {archive_path}")
    output = args.output_dir.expanduser().resolve()
    figures = output / "figures"
    qv_dir = figures / "qv_scans"
    max_qv_dir = figures / "maximum_qv"
    scan_dir = figures / "breakdown_fits"
    output.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    qv_dir.mkdir(parents=True, exist_ok=True)
    max_qv_dir.mkdir(parents=True, exist_ok=True)
    scan_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.random_seed)

    with zipfile.ZipFile(archive_path) as archive:
        manifest = inventory_scan_archive(archive)
        selected, selected_members = select_archive_records(manifest, args)
        observations_by_condition: dict[Condition, list[FileObservation]] = defaultdict(list)
        failures: list[dict] = []
        for key in sorted(selected, key=lambda item: (item[0], item[1])):
            for record in selected[key]:
                try:
                    observations_by_condition[record.condition].append(analyze_file(archive, record, args))
                except (ValueError, OSError, SurfaceChargeError) as error:
                    failures.append({"member": record.member, "error": str(error)})

        conditions = sorted(observations_by_condition)
        source_map = scan_source_map(conditions, observations_by_condition)
        calibrations: dict[Condition, CalibrationModel] = {}
        for condition in conditions:
            source = source_map[condition]
            if source is None or source in calibrations:
                continue
            source_rows = observations_by_condition[source]
            sign, agreement, votes, status = locked_condition_sign(
                source_rows, args.charge_polarity, args.minimum_sign_agreement
            )
            calibrations[source] = build_calibration(
                source, source_rows, sign, agreement, votes, status, args, rng
            )

        summaries: list[dict] = []
        long_results: list[dict] = []
        per_file_rows: list[dict] = []
        peak_lobe_rows: list[dict] = []
        level_fit_rows: list[dict] = []
        for condition in conditions:
            source = source_map[condition]
            max_rows = sorted(
                [row for row in observations_by_condition[condition] if row.level_label == "MAX"],
                key=lambda item: item.record.capture_index,
            )
            if source is None or source not in calibrations:
                summaries.append({
                    "condition": condition.label,
                    "condition_label": display_label(condition),
                    "material": condition.material,
                    "burst_kHz": condition.burst_kHz,
                    "evidence_tier": "not_identifiable",
                    "reason": "No same-material complete voltage scan.",
                    "n_max_files": len(max_rows),
                    "quiet_edge_files": sum(row.quiet_edge_change_raw_nC is not None for row in max_rows),
                })
                continue
            calibration = calibrations[source]
            transferred = source != condition
            central_factor = calibration.geometry_factor if calibration.geometry_factor is not None else 1.0
            holdout_limits = passive_holdout_limits(
                observations_by_condition[source],
                calibration,
                central_factor,
                args.target_negative_on_pin_negative,
            )
            file_metric_pairs = [
                (
                    row,
                    per_file_metrics(
                        row,
                        calibration,
                        central_factor,
                        args.target_negative_on_pin_negative,
                    ),
                )
                for row in max_rows
            ]
            file_metric_pairs = [(row, metrics) for row, metrics in file_metric_pairs if metrics]
            file_metrics = [metrics for _, metrics in file_metric_pairs]
            point, intervals, bootstrap_draws = aggregate_with_ci(
                file_metrics,
                args.bootstrap_replicates,
                args.bootstrap_block_files,
                rng,
            )
            analysis_draws, calibration_factor_draws, cprime_draws = hierarchical_analysis_draws(
                max_rows,
                observations_by_condition[source],
                calibration,
                args,
                rng,
            )
            evidence = calibration.evidence_tier
            reasons = list(calibration.failed_gates)
            if transferred:
                evidence = (
                    "exploratory_transferred_model"
                    if calibration.passive_status.startswith("supported") and central_factor != 1.0
                    else "diagnostic_transferred_background"
                )
                reasons.append("calibration_transferred_across_burst_frequency")
            sensitivity_factor = factor_sensitivity_draws(
                calibration,
                args,
                rng,
                args.bootstrap_replicates,
                cprime_draws,
            )
            if central_factor <= 0:
                central_factor = 1.0
            summary: dict = {
                "condition": condition.label,
                "condition_label": display_label(condition),
                "material": condition.material,
                "burst_kHz": condition.burst_kHz,
                "scan_source": source.label,
                "scan_transferred": transferred,
                "n_max_files": len(max_rows),
                "n_halfcycles": sum(len(row.lobes) for row in max_rows),
                "detected_burst_frequency_Hz": median_or_none(
                    row.detected_burst_Hz for row in max_rows
                ),
                "detected_burst_frequency_relative_error": median_or_none(
                    row.burst_frequency_relative_error for row in max_rows
                ),
                "evidence_tier": evidence,
                "failed_gates": ";".join(reasons),
                "charge_polarity_sign": calibration.sign,
                "charge_polarity_agreement": calibration.sign_agreement,
                "carrier_polarity_assignment": (
                    "negative_pin_voltage_assumed_negative_carrier_delivery"
                    if args.target_negative_on_pin_negative
                    else "positive_pin_voltage_assumed_negative_carrier_delivery"
                ),
                "carrier_polarity_assignment_status": "explicit_physical_assumption_not_validated_by_channel_D_sign_lock",
                "Ccell_status": calibration.passive_status,
                "Ccell_reactive_pF": calibration.cprime_pF,
                "Ccell_reactive_ci_low_pF": calibration.cprime_ci_low_pF,
                "Ccell_reactive_ci_high_pF": calibration.cprime_ci_high_pF,
                "Ccell_loss_pF": calibration.closs_pF,
                "Ccell_loss_ci_low_pF": calibration.closs_ci_low_pF,
                "Ccell_loss_ci_high_pF": calibration.closs_ci_high_pF,
                "Cd_scan_status": calibration.scan_cd_status,
                "Cd_scan_pF": calibration.scan_cd_pF,
                "Cd_geometry_scenario_pF": calibration.geometry_cd_pF,
                "Cd_geometry_scenario_low_pF": (
                    geometry_cd_pF(
                        args.beaker_diameter_cm * (1.0 - args.beaker_diameter_relative_range),
                        args.glass_thickness_mm * (1.0 + args.glass_thickness_relative_range),
                        args.pyrex_epsilon_min,
                    )
                    if calibration.geometry_cd_pF is not None else None
                ),
                "Cd_geometry_scenario_high_pF": (
                    geometry_cd_pF(
                        args.beaker_diameter_cm * (1.0 + args.beaker_diameter_relative_range),
                        args.glass_thickness_mm * (1.0 - args.glass_thickness_relative_range),
                        args.pyrex_epsilon_max,
                    )
                    if calibration.geometry_cd_pF is not None else None
                ),
                "charge_correction_factor": central_factor,
                "charge_correction_factor_low": calibration.geometry_factor_low,
                "charge_correction_factor_high": calibration.geometry_factor_high,
                "hierarchical_factor_draw_valid_fraction": float(
                    np.mean(np.isfinite(calibration_factor_draws))
                ),
                "model_sensitivity_draw_valid_fraction": float(
                    np.mean(np.isfinite(sensitivity_factor))
                ),
                "factor_source": calibration.factor_source,
                "instantaneous_peak_status": "not_time_resolved_existing_104ns_sampling",
                "retained_charge_status": "not_measured_no_quiet_record_edges",
                "quiet_edge_files": sum(row.quiet_edge_change_raw_nC is not None for row in max_rows),
                "transport_imbalance_is_not_retained_charge": True,
            }
            for key, value in holdout_limits.items():
                summary[f"{key}_passive_90_holdout_p95"] = value
            for key, value in point.items():
                summary[key] = value
                low, high = intervals.get(key, (None, None))
                summary[f"{key}_repeat_ci_low"] = low
                summary[f"{key}_repeat_ci_high"] = high
                draws = analysis_draws.get(key)
                if draws is not None and len(draws):
                    alow, ahigh = percentile_interval(draws)
                    summary[f"{key}_analysis_ci_low"] = alow
                    summary[f"{key}_analysis_ci_high"] = ahigh
                    valid_factor = (
                        np.isfinite(calibration_factor_draws)
                        & (calibration_factor_draws > 0)
                        & np.isfinite(sensitivity_factor)
                    )
                    combined = np.full(len(draws), np.nan)
                    combined[valid_factor] = (
                        draws[valid_factor]
                        * sensitivity_factor[valid_factor]
                        / calibration_factor_draws[valid_factor]
                    )
                    slo, shi = percentile_interval(combined)
                    summary[f"{key}_sensitivity_low"] = slo
                    summary[f"{key}_sensitivity_high"] = shi
            quiet_values = [calibration.sign * row.quiet_edge_change_raw_nC for row in max_rows if row.quiet_edge_change_raw_nC is not None]
            if len(quiet_values) >= 3:
                qdraws = bootstrap_median_draws(
                    quiet_values, args.bootstrap_replicates, rng, args.bootstrap_block_files
                )
                qlow, qhigh = percentile_interval(qdraws)
                summary["retained_terminal_change_nC"] = float(np.median(quiet_values))
                summary["retained_terminal_change_ci_low_nC"] = qlow
                summary["retained_terminal_change_ci_high_nC"] = qhigh
                summary["retained_charge_status"] = (
                    "external_terminal_change_measured_not_local_surface_charge_"
                    "and_dc_coupling_unverified"
                )
            for target in ("negative", "positive"):
                metrics = {
                    "peak_halfcycle_charge_nC": f"{target}_peak_envelope_halfcycle_p95_nC",
                    "peak_halfcycle_average_flow_per_s": f"{target}_peak_halfcycle_average_equivalent_flow_p95_per_s",
                    "whole_record_average_flow_per_s": f"{target}_record_average_equivalent_flow_per_s",
                    "record_total_charge_nC": f"{target}_record_total_nC",
                }
                for metric_name, key in metrics.items():
                    diagnostic_estimate = summary.get(key)
                    central_positive = (
                        diagnostic_estimate is not None
                        and np.isfinite(diagnostic_estimate)
                        and diagnostic_estimate > 0
                    )
                    repeat_low = summary.get(f"{key}_repeat_ci_low")
                    analysis_low = summary.get(f"{key}_analysis_ci_low")
                    analysis_high = summary.get(f"{key}_analysis_ci_high")
                    passive_supported = calibration.passive_status.startswith("supported")
                    resolution = 0.0
                    if metric_name == "peak_halfcycle_charge_nC":
                        resolution = max(
                            0.0,
                            float(summary.get(f"{target}_peak_charge_resolution_threshold_nC") or 0.0),
                            float(summary.get(f"{key}_passive_90_holdout_p95") or 0.0),
                        )
                    elif metric_name == "peak_halfcycle_average_flow_per_s":
                        resolution = max(
                            0.0,
                            float(summary.get(f"{target}_peak_rate_resolution_threshold_per_s") or 0.0),
                            float(summary.get(f"{key}_passive_90_holdout_p95") or 0.0),
                        )
                    elif not transferred:
                        resolution = max(
                            0.0,
                            float(summary.get(f"{key}_passive_90_holdout_p95") or 0.0),
                        )
                    resolved_fraction = float(
                        summary.get(f"{target}_resolved_halfcycle_fraction") or 0.0
                    )
                    factor_stable = float(
                        summary.get("model_sensitivity_draw_valid_fraction") or 0.0
                    ) >= 0.80
                    reportable = (
                        passive_supported
                        and factor_stable
                        and central_positive
                        and analysis_low is not None
                        and np.isfinite(analysis_low)
                        and analysis_low > resolution
                        and (
                            metric_name.startswith("peak_")
                            or resolved_fraction >= 0.05
                        )
                    )
                    metric_status = evidence
                    if not passive_supported:
                        metric_status = "not_reportable_passive_background_model_rejected"
                    elif not factor_stable:
                        metric_status = "not_reportable_model_factor_unbounded_in_more_than_20_percent_sensitivity_draws"
                    elif not central_positive:
                        metric_status = "not_resolved_above_passive_background"
                    elif analysis_low is None or analysis_low <= resolution:
                        metric_status = "not_resolved_above_passive_background_at_95_percent_analysis_CI"
                    elif not metric_name.startswith("peak_") and resolved_fraction < 0.05:
                        metric_status = "not_reportable_fewer_than_5_percent_halfcycles_resolved"
                    long_results.append({
                        "condition": condition.label,
                        "polarity": target,
                        "metric": metric_name,
                        "estimate": diagnostic_estimate if reportable else None,
                        "diagnostic_signed_estimate": diagnostic_estimate,
                        "repeat_ci_low": repeat_low if reportable else None,
                        "repeat_ci_high": summary.get(f"{key}_repeat_ci_high") if reportable else None,
                        "analysis_ci_low": analysis_low if reportable else None,
                        "analysis_ci_high": analysis_high if reportable else None,
                        "passive_resolution_threshold": resolution,
                        "passive_resolution_basis": (
                            "maximum_of_training_residual_and_out_of_sample_90_percent_statistic"
                            if metric_name.startswith("peak_")
                            else "out_of_sample_90_percent_same_statistic"
                            if not transferred
                            else "resolved_fraction_only_no_same_frequency_holdout"
                        ),
                        "resolved_halfcycle_fraction": resolved_fraction,
                        "model_scale_sensitivity_low": summary.get(f"{key}_sensitivity_low") if reportable else None,
                        "model_scale_sensitivity_high": summary.get(f"{key}_sensitivity_high") if reportable else None,
                        "unit": "nC" if "charge" in metric_name else "elementary_charge_equivalents_per_s",
                        "evidence_tier": metric_status,
                        "definition": (
                            "p95 across maximum-amplitude carrier half-cycles selected once per duty burst"
                            if metric_name == "peak_halfcycle_charge_nC" else
                            "p95 half-cycle-average net charge-equivalent rate across the same maximum-amplitude lobe selected once per duty burst; not instantaneous flux"
                            if metric_name == "peak_halfcycle_average_flow_per_s" else
                            "signed background-subtracted sum divided by full record duration"
                            if metric_name == "whole_record_average_flow_per_s" else
                            "signed background-subtracted charge summed over the full record"
                        ),
                        "n_files": len(file_metrics),
                        "scan_source": source.label,
                    })
            if args.active_area_mm2 is not None and args.active_area_mm2 > 0:
                area_m2 = args.active_area_mm2 * 1.0e-6
                for target in ("negative", "positive"):
                    for base in ("peak_halfcycle_average_equivalent_flow_p95_per_s", "record_average_equivalent_flow_per_s"):
                        key = f"{target}_{base}"
                        if key in summary:
                            summary[f"{key}_per_m2"] = summary[key] / area_m2
            for file_row, metrics in file_metric_pairs:
                per_file_rows.append({
                    "condition": condition.label,
                    "member": file_row.record.member,
                    "capture_index": file_row.record.capture_index,
                    "carrier_Hz": file_row.carrier_Hz,
                    "detected_burst_Hz": file_row.detected_burst_Hz,
                    "burst_detection_method": file_row.burst_detection_method,
                    "burst_frequency_relative_error": file_row.burst_frequency_relative_error,
                    "voltage_pp_kV": file_row.voltage_pp_kV,
                    "locked_charge_sign": calibration.sign,
                    "file_sign_vote": file_row.sign_vote,
                    "sign_vote_disagrees": file_row.sign_vote is not None and file_row.sign_vote != calibration.sign,
                    "charge_lsb_nC": file_row.charge_lsb_nC,
                    **metrics,
                })
                peak_lobe_rows.extend(
                    peak_lobe_audit_rows(
                        file_row,
                        calibration,
                        central_factor,
                        args.target_negative_on_pin_negative,
                    )
                )
            summaries.append(summary)

        if not args.no_plots:
            plot_coverage(manifest, selected_members, figures / "01_data_coverage", args)
            for source, calibration in calibrations.items():
                source_rows = observations_by_condition[source]
                plot_qv_grid(source, source_rows, calibration, qv_dir / f"{source.label}_qv_traces", args)
                level_fit_rows.extend(
                    plot_scan_fit(source, source_rows, calibration, scan_dir / f"{source.label}_breakdown_fit", args)
                )
            for condition in conditions:
                source = source_map[condition]
                if source is None or source not in calibrations:
                    continue
                plot_maximum_qv(
                    condition,
                    observations_by_condition[condition],
                    calibrations[source],
                    max_qv_dir / f"{condition.label}_maximum_qv",
                    args,
                )
            example_condition = next(
                (
                    condition for condition in conditions
                    if condition.material == "BMIM_nitrate" and condition.burst_kHz == 20
                ),
                None,
            )
            if example_condition is not None and example_condition in calibrations:
                example_rows = [
                    row for row in observations_by_condition[example_condition]
                    if row.level_label == "MAX"
                ]
                if example_rows:
                    example = min(
                        example_rows,
                        key=lambda item: abs(
                            item.voltage_pp_kV
                            - np.median([candidate.voltage_pp_kV for candidate in example_rows])
                        ),
                    )
                    plot_processing_example(
                        archive,
                        example,
                        calibrations[example_condition],
                        figures / "00_signal_processing_example",
                        args,
                    )
            plot_capacitance_summary(calibrations, observations_by_condition, figures / "02_validated_complex_Ccell", args)
            plot_result_forest(
                summaries,
                "peak_envelope_halfcycle_p95_nC",
                "Positive and negative charge in the peak-envelope carrier half-cycle",
                "Charge per peak half-cycle (nC)",
                figures / "03_peak_halfcycle_charge",
                args,
            )
            plot_result_forest(
                summaries,
                "peak_halfcycle_average_equivalent_flow_p95_per_s",
                "Peak half-cycle-average total charge-equivalent flow",
                "Elementary-charge equivalents s⁻¹",
                figures / "04_peak_halfcycle_average_flow",
                args,
                log_scale=True,
            )
            plot_result_forest(
                summaries,
                "record_average_equivalent_flow_per_s",
                "Whole-record average net charge-equivalent delivery rate",
                "Elementary-charge equivalents s⁻¹",
                figures / "05_whole_record_flow",
                args,
                log_scale=True,
            )
            plot_negative_rate_frequency(
                summaries,
                figures / "05b_negative_rate_vs_burst_frequency",
                args,
            )
            plot_retained_status(summaries, figures / "06_retained_charge_availability", args)

        manifest_rows = [
            {
                "member": record.member,
                "material": record.condition.material,
                "burst_kHz": record.condition.burst_kHz,
                "level": "MAX" if record.is_maximum else record.level_percent,
                "capture_index": record.capture_index,
                "selected": record.member in selected_members,
            }
            for record in manifest
        ]
        headline = headline_rows(summaries, long_results)
        write_csv(output / "headline_results.csv", headline)
        write_csv(output / "supervisor_summary.csv", summaries)
        write_csv(output / "long_form_results.csv", long_results)
        write_csv(output / "per_file_maximum_metrics.csv", per_file_rows)
        write_csv(output / "peak_halfcycle_observations.csv", peak_lobe_rows)
        write_csv(output / "capacitance_and_fit_results.csv", [flatten_calibration(model) for model in calibrations.values()])
        write_csv(output / "breakdown_level_fit_points.csv", level_fit_rows)
        write_csv(output / "archive_manifest.csv", manifest_rows)
        (output / "RESULTS_OVERVIEW.md").write_text(
            results_overview_text(headline), encoding="utf-8"
        )
        (output / "METHODS_AND_LIMITATIONS.md").write_text(methodology_text(args), encoding="utf-8")
        captions = """# Figure captions

1. **Data coverage.** Number of separate waveform captures selected at each commanded percentage of breakdown voltage. Captures are sequential technical repeats, not independently rebuilt experiments. Legacy MAX data were acquired in a separate session.
2. **Complex cell capacitance.** Capture-level reactive and loss capacitances from 40/60/75 % data. Only conditions passing passive orientation and amplitude-stability gates are shown.
3. **Peak half-cycle charge.** Peak-envelope p95 net charge. Thin bars jointly resample passive-calibration and MAX captures; broad bars are declared geometry/monitor-scale/model-form sensitivity, not statistical confidence intervals. Open markers transfer a same-material calibration across duty-burst frequency; gray crosses have a rejected passive background and are diagnostic only.
4. **Peak rate.** P95 half-cycle-average net charge-equivalent rate across the same peak-envelope lobes. This is not a nanosecond-resolved particle-flux maximum.
5. **Whole-record rate.** Signed background-subtracted charge divided by the full 10 ms record and elementary charge. The frequency panel distinguishes independently scanned, transferred, and rejected conditions.
6. **Retained-charge availability.** Persistent retained charge requires stationary quiet monitor plateaus at both record edges; DC coupling also remains to be verified. Charge imbalance is not substituted when this criterion fails.
7. **Q–V scan supplements.** Faint lines are independent captures; the bold line is the median-amplitude capture. Slope guides are descriptive. Rejected models remain visibly labeled.
8. **Breakdown fits.** Small points are captures; large points and bars show the capture distribution. Passive fits use 40/60/75 %. Transition levels 90/100 % are excluded. The 105/115 % line is a diagnostic two-level secant. Legacy MAX is an open out-of-sample star and is never fitted.

All carrier labels use an explicit physical assignment: negative pin-voltage lobes are treated as negative-carrier delivery. Channel-D sign locking does not independently validate this species/polarity mapping.
"""
        (output / "FIGURE_CAPTIONS.md").write_text(captions, encoding="utf-8")
        audit = {
            "analysis_version": ANALYSIS_VERSION,
            "archive": str(archive_path),
            "selected_files": len(selected_members),
            "successfully_analyzed_files": sum(len(rows) for rows in observations_by_condition.values()),
            "failures": failures,
            "bootstrap_replicates": args.bootstrap_replicates,
            "bootstrap_block_files": args.bootstrap_block_files,
            "random_seed": args.random_seed,
            "calibrations": {condition.label: flatten_calibration(model) for condition, model in calibrations.items()},
        }
        (output / "analysis_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    return {
        "output_dir": str(output),
        "selected_files": len(selected_members),
        "failures": len(failures),
        "conditions": len(summaries),
    }


def main(argv: Sequence[str] | None = None) -> int:
    try:
        args = parse_arguments(argv)
        result = analyze_archive(args)
    except (SurfaceChargeError, OSError, ValueError, zipfile.BadZipFile) as error:
        print(f"ERROR: {error}", file=__import__("sys").stderr)
        return 2
    print(
        f"DBD charge report complete: {result['conditions']} conditions, "
        f"{result['selected_files']} selected files, {result['failures']} failures.\n"
        f"Output: {result['output_dir']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
