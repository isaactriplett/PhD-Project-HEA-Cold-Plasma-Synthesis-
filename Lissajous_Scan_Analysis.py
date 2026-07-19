# -*- coding: utf-8 -*-
"""Batch analysis for voltage-scan AC-DBD waveform archives.

This companion to :mod:`Lissajous_Figures` analyzes a ZIP archive containing
below-breakdown, near-breakdown, above-breakdown, and legacy maximum-voltage
PicoScope CSV exports.  It deliberately keeps three questions separate:

1. Does Channel D behave like the voltage across the stated monitor capacitor?
2. What capacitances are supported by the multi-amplitude scan?
3. Only if the calibration and equivalent-circuit checks pass, what gas-gap
   charge-equivalent transfer is supported at the old maximum voltage?

The script never interprets negative charge as electrons alone; it is an
electron-plus-negative-ion charge equivalent.  Likewise, a positive result is
a positive-ion charge equivalent.  A point-to-plane or partially covered DBD
generally yields an *effective* active dielectric capacitance, not necessarily
the full geometric Pyrex capacitance.

Example
-------
python Lissajous_Scan_Analysis.py "C:\\path\\waveforms.zip" \
    --output-dir scan_analysis --files-per-level 0

``--files-per-level 0`` uses every waveform.  Positive values select that many
evenly spaced captures from each acquisition, never the first consecutive N.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import re
import zipfile
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np


ANALYSIS_VERSION = "2.0-scan"
ELEMENTARY_CHARGE_C = 1.602176634e-19
PF_PER_F = 1.0e12
NC_PER_C = 1.0e9
RAW_LEVELS = (40, 60, 75, 90, 100, 105, 115)
PASSIVE_FIT_LEVELS = (40, 60, 75)
ACTIVE_FIT_LEVELS = (105, 115)
MATERIAL_LABELS = {
    "argon": "argon_only",
    "pure water": "pure_water",
    "ionic liquid": "BMIM_nitrate",
    "manganese nitrate in water": "5mM_Mn_nitrate_in_water",
}


class ScanAnalysisError(RuntimeError):
    """Raised when an archive or waveform cannot support an analysis step."""


@dataclass(frozen=True, order=True)
class Condition:
    material: str
    burst_kHz: int

    @property
    def label(self) -> str:
        return f"{self.material}_{self.burst_kHz}kHz"


@dataclass(frozen=True)
class MemberRecord:
    member: str
    condition: Condition
    level_percent: int | None
    is_maximum: bool
    capture_index: int


@dataclass
class Waveform:
    time_s: np.ndarray
    source_voltage_V: np.ndarray
    voltage_DBD_V: np.ndarray
    current_input_A: np.ndarray
    monitor_voltage_V: np.ndarray
    skipped_rows: int


@dataclass
class FileFeatures:
    member: str
    level_label: str
    carrier_Hz: float
    record_duration_s: float
    voltage_pp_kV: float
    q_sign: int
    current_sign: int
    monitor_Cprime_pF: float | None
    monitor_Closs_pF: float | None
    monitor_tan_delta: float | None
    monitor_phase_deg: float | None
    current_Creactive_pF: float | None
    current_G_uS: float | None
    current_monitor_gain_ratio: float | None
    current_monitor_phase_error_deg: float | None
    monitor_x_Uqpp_kV: float | None
    monitor_y_Qpp_nC: float | None
    current_x_Uqpp_kV: float | None
    current_y_Qpp_nC: float | None
    skipped_rows: int
    clipping_flag: bool
    quiet_both_edges: bool


@dataclass
class LinearFit:
    slope: float | None
    intercept: float | None
    r_squared: float | None
    n: int


def parse_arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze multi-amplitude DBD Lissajous scans stored in a ZIP archive.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("archive_zip", type=Path)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--files-per-level",
        type=int,
        default=16,
        help="Evenly spaced captures per level; 0 uses every capture",
    )
    parser.add_argument("--reference-capacitance-uf", type=float, default=0.1)
    parser.add_argument("--voltage-scale", type=float, default=1000.0)
    parser.add_argument("--monitor-voltage-scale", type=float, default=1.0)
    parser.add_argument(
        "--source-to-ground",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Subtract Channel D from the scaled source voltage to form V_DBD",
    )
    parser.add_argument("--reference-polarity", choices=(-1, 1), type=int, default=1)
    parser.add_argument("--carrier-min-khz", type=float, default=100.0)
    parser.add_argument("--carrier-max-khz", type=float, default=160.0)
    parser.add_argument("--dpi", type=int, default=180)
    parser.add_argument("--no-plots", action="store_true")
    return parser.parse_args(argv)


def _capture_index(member: str) -> int:
    match = re.search(r"_(\d+)\.csv$", member, flags=re.IGNORECASE)
    return int(match.group(1)) if match else 0


def inventory_scan_archive(archive: zipfile.ZipFile) -> list[MemberRecord]:
    """Return raw waveform members while excluding duplicates/derived exports."""
    records: list[MemberRecord] = []
    for member in archive.namelist():
        normalized = member.replace("\\", "/")
        lowered = normalized.lower()
        if not lowered.endswith(".csv") or "_analysis/" in lowered:
            continue
        parts = normalized.split("/")
        if len(parts) < 4:
            continue
        material_part = next((part.lower() for part in parts if part.lower() in MATERIAL_LABELS), None)
        if material_part is None:
            continue
        material = MATERIAL_LABELS[material_part]
        # A byte-identical convenience copy of one old Ar waveform sits at the
        # material root and must not become a thirteenth MAX condition.
        if lowered.endswith("/argon/argon 20khz.csv"):
            continue

        is_maximum = any(part.lower().startswith("lissajousfigure") for part in parts)
        burst_kHz: int | None = None
        if is_maximum:
            folder = next(part for part in parts if part.lower().startswith("lissajousfigure"))
            match = re.search(r"(4|10|20)\s*k\s*hz", folder, flags=re.IGNORECASE)
            if match:
                burst_kHz = int(match.group(1))
            elif material_part == "argon" and "figure1" in folder.lower():
                burst_kHz = 20
        else:
            for part in parts:
                match = re.fullmatch(r"\s*(4|10|20)\s*k\s*hz\s*", part, flags=re.IGNORECASE)
                if match:
                    burst_kHz = int(match.group(1))
                    break
        if burst_kHz is None:
            continue

        level_percent: int | None = None
        if not is_maximum:
            joined = "/".join(parts).lower()
            for level in (40, 60, 75, 90, 105, 115):
                if f"{level}% breakdown" in joined:
                    level_percent = level
                    break
            if level_percent is None and "breakdown" in joined:
                level_percent = 100
        records.append(
            MemberRecord(
                member=normalized,
                condition=Condition(material, burst_kHz),
                level_percent=level_percent,
                is_maximum=is_maximum,
                capture_index=_capture_index(normalized),
            )
        )
    return sorted(records, key=lambda row: (row.condition, row.is_maximum, row.level_percent or 999, row.capture_index, row.member))


def evenly_spaced(records: Sequence[MemberRecord], count: int) -> list[MemberRecord]:
    ordered = sorted(records, key=lambda row: (row.capture_index, row.member))
    if count == 0 or count >= len(ordered):
        return ordered
    if count < 1:
        raise ScanAnalysisError("--files-per-level must be 0 or a positive integer.")
    indices = np.unique(np.rint(np.linspace(0, len(ordered) - 1, count)).astype(int))
    return [ordered[int(index)] for index in indices]


def _fast_numeric_csv(payload: bytes) -> tuple[np.ndarray, int]:
    """Parse four columns quickly, with a row-skipping fallback for corrupt bytes."""
    try:
        data = np.loadtxt(io.BytesIO(payload), delimiter=",", skiprows=3, usecols=(0, 1, 2, 3))
        if data.ndim == 1:
            data = data.reshape(1, -1)
        return np.asarray(data, dtype=float), 0
    except (ValueError, UnicodeError):
        rows: list[tuple[float, float, float, float]] = []
        skipped = 0
        text = payload.decode("utf-8-sig", errors="replace")
        for row in csv.reader(io.StringIO(text)):
            if len(row) < 4:
                continue
            try:
                values = tuple(float(cell.strip()) for cell in row[:4])
            except ValueError:
                skipped += 1
                continue
            if all(math.isfinite(value) for value in values):
                rows.append(values)  # type: ignore[arg-type]
            else:
                skipped += 1
        if len(rows) < 40:
            raise ScanAnalysisError("Fewer than 40 numeric rows remained after CSV validation.")
        return np.asarray(rows, dtype=float), skipped


def read_waveform_member(
    archive: zipfile.ZipFile,
    record: MemberRecord,
    voltage_scale: float,
    monitor_voltage_scale: float,
    source_to_ground: bool,
    reference_polarity: int,
) -> Waveform:
    data, skipped = _fast_numeric_csv(archive.read(record.member))
    time_s = data[:, 0] * 1.0e-3
    order = np.argsort(time_s, kind="stable")
    data = data[order]
    time_s = time_s[order]
    keep = np.r_[True, np.diff(time_s) > 0]
    data = data[keep]
    time_s = time_s[keep]
    if len(time_s) < 40 or not np.all(np.isfinite(data)):
        raise ScanAnalysisError("Waveform is too short or contains non-finite values.")
    monitor = data[:, 3] * monitor_voltage_scale
    source = data[:, 1] * voltage_scale
    voltage = source - reference_polarity * monitor if source_to_ground else source.copy()
    return Waveform(time_s, source, voltage, data[:, 2], monitor, skipped)


def moving_average(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return values.copy()
    window = int(window) | 1
    window = min(window, len(values) - (1 - len(values) % 2))
    if window <= 1:
        return values.copy()
    pad = window // 2
    return np.convolve(np.pad(values, pad, mode="edge"), np.ones(window) / window, mode="valid")


def constrained_carrier_frequency(time_s: np.ndarray, voltage_V: np.ndarray, low_Hz: float, high_Hz: float) -> float:
    dt = float(np.median(np.diff(time_s)))
    centered = voltage_V - np.mean(voltage_V)
    spectrum = np.abs(np.fft.rfft(centered * np.hanning(len(centered))))
    frequencies = np.fft.rfftfreq(len(centered), dt)
    mask = (frequencies >= low_Hz) & (frequencies <= high_Hz)
    if not np.any(mask):
        raise ScanAnalysisError("No FFT bins fall inside the requested carrier band.")
    candidates = np.flatnonzero(mask)
    index = int(candidates[np.argmax(spectrum[candidates])])
    return float(frequencies[index])


def complex_amplitude(time_s: np.ndarray, values: np.ndarray, frequency_Hz: float) -> complex:
    centered = values - np.mean(values)
    window = np.hanning(len(values))
    basis = np.exp(-2j * np.pi * frequency_Hz * time_s)
    return complex(np.sum(centered * window * basis) / np.sum(window))


def phasor_ratio(time_s: np.ndarray, numerator: np.ndarray, denominator: np.ndarray, frequency_Hz: float) -> complex | None:
    top = complex_amplitude(time_s, numerator, frequency_Hz)
    bottom = complex_amplitude(time_s, denominator, frequency_Hz)
    if abs(bottom) <= np.finfo(float).eps:
        return None
    return top / bottom


def wrap_degrees(angle_deg: float) -> float:
    return (angle_deg + 180.0) % 360.0 - 180.0


def choose_q_sign(cstar: complex | None) -> int:
    if cstar is None:
        return 1
    if abs(cstar.real) > 0.10 * abs(cstar):
        return 1 if cstar.real >= 0 else -1
    # If loss dominates, retain the sign producing the smaller absolute phase.
    return 1 if abs(wrap_degrees(np.degrees(np.angle(cstar)))) <= 90 else -1


def choose_current_sign(y: complex | None) -> int:
    if y is None:
        return -1
    # A passive capacitive current leads voltage.  This is a sign convention,
    # not proof that the Pearson current is confined to the DBD branch.
    return 1 if y.imag >= 0 else -1


def burst_windows(time_s: np.ndarray, voltage_V: np.ndarray, burst_kHz: int, carrier_Hz: float) -> list[slice]:
    period_s = 1.0 / (burst_kHz * 1000.0)
    dt = float(np.median(np.diff(time_s)))
    smooth_samples = max(3, int(round(0.75 / (carrier_Hz * dt))) | 1)
    activity = moving_average(np.abs(voltage_V - np.median(voltage_V)), smooth_samples)
    phase = np.mod(time_s, period_s) / period_s
    bins = 128
    indices = np.minimum((phase * bins).astype(int), bins - 1)
    profile = np.full(bins, np.nan)
    for index in range(bins):
        selected = activity[indices == index]
        if len(selected):
            profile[index] = np.median(selected)
    if np.all(~np.isfinite(profile)):
        return []
    boundary_phase = (int(np.nanargmin(profile)) + 0.5) / bins
    origin = boundary_phase * period_s
    k_min = math.ceil((time_s[0] - origin) / period_s)
    k_max = math.floor((time_s[-1] - origin) / period_s) - 1
    windows: list[slice] = []
    for k in range(k_min, k_max + 1):
        start_t = origin + k * period_s
        stop_t = start_t + period_s
        start = int(np.searchsorted(time_s, start_t, side="left"))
        stop = int(np.searchsorted(time_s, stop_t, side="left"))
        if stop - start >= 20:
            windows.append(slice(start, stop))
    return windows


def integrate_current_per_burst(time_s: np.ndarray, voltage_V: np.ndarray, current_A: np.ndarray, windows: Sequence[slice]) -> np.ndarray:
    charge = np.full(len(time_s), np.nan)
    for window in windows:
        t = time_s[window]
        u = voltage_V[window]
        i = current_A[window]
        if len(t) < 3:
            continue
        quiet = np.abs(u - np.median(u)) <= np.percentile(np.abs(u - np.median(u)), 20)
        baseline = float(np.median(i[quiet])) if np.any(quiet) else float(np.median(i))
        corrected = i - baseline
        increments = 0.5 * (corrected[1:] + corrected[:-1]) * np.diff(t)
        q = np.r_[0.0, np.cumsum(increments)]
        # Pearson transformers cannot establish DC charge.  Removing only the
        # linear integration drift preserves carrier-scale extrema while making
        # that limitation explicit.
        q -= np.linspace(q[0], q[-1], len(q))
        charge[window] = q
    return charge


def burst_extrema_features(
    time_s: np.ndarray,
    voltage_V: np.ndarray,
    charge_C: np.ndarray,
    windows: Sequence[slice],
) -> tuple[float | None, float | None]:
    pairs: list[tuple[float, float]] = []
    dt = float(np.median(np.diff(time_s)))
    for window in windows:
        u = voltage_V[window]
        q = charge_C[window]
        finite = np.isfinite(u) & np.isfinite(q)
        if finite.sum() < 20:
            continue
        u = u[finite]
        q = q[finite]
        smooth = max(3, min(11, int(round(0.5e-6 / dt)) | 1))
        us = moving_average(u, smooth)
        qs = moving_average(q, smooth)
        hi = int(np.argmax(qs))
        lo = int(np.argmin(qs))
        x = (us[hi] - us[lo]) * 1.0e-3
        y = (qs[hi] - qs[lo]) * NC_PER_C
        if np.isfinite(x) and np.isfinite(y) and abs(x) > 0.02 and y > 0:
            pairs.append((x, y))
    if not pairs:
        return None, None
    values = np.asarray(pairs)
    return float(np.median(values[:, 0])), float(np.median(values[:, 1]))


def edge_quiet(time_s: np.ndarray, voltage_V: np.ndarray, carrier_Hz: float) -> bool:
    dt = float(np.median(np.diff(time_s)))
    count = min(len(time_s) // 4, max(20, int(round(2.0 / (carrier_Hz * dt)))))
    centered = voltage_V - np.median(voltage_V)
    peak = max(abs(float(np.percentile(centered, 1))), abs(float(np.percentile(centered, 99))))
    if peak <= 0:
        return False
    start = float(np.percentile(np.abs(centered[:count]), 95)) < 0.05 * peak
    stop = float(np.percentile(np.abs(centered[-count:]), 95)) < 0.05 * peak
    return bool(start and stop)


def detect_clipping(*channels: np.ndarray) -> bool:
    for values in channels:
        unique, counts = np.unique(values, return_counts=True)
        if len(unique) < 3:
            return True
        if counts[0] > 0.01 * len(values) or counts[-1] > 0.01 * len(values):
            return True
    return False


def extract_file_features(
    waveform: Waveform,
    record: MemberRecord,
    reference_capacitance_F: float,
    carrier_low_Hz: float,
    carrier_high_Hz: float,
) -> FileFeatures:
    t = waveform.time_s
    u = waveform.voltage_DBD_V
    d = waveform.monitor_voltage_V
    i_raw = waveform.current_input_A
    carrier = constrained_carrier_frequency(t, waveform.source_voltage_V, carrier_low_Hz, carrier_high_Hz)
    q_unsigned = reference_capacitance_F * d
    cstar_raw = phasor_ratio(t, q_unsigned, u, carrier)
    q_sign = choose_q_sign(cstar_raw)
    cstar = None if cstar_raw is None else q_sign * cstar_raw
    y_raw = phasor_ratio(t, i_raw, u, carrier)
    current_sign = choose_current_sign(y_raw)
    current = current_sign * i_raw
    y = None if y_raw is None else current_sign * y_raw
    i_over_d = phasor_ratio(t, current, d, carrier)

    expected_gain = 2.0 * np.pi * carrier * reference_capacitance_F
    gain_ratio = abs(i_over_d) / expected_gain if i_over_d is not None and expected_gain > 0 else None
    phase_error = None
    if i_over_d is not None:
        phase = np.degrees(np.angle(i_over_d))
        phase_error = min(abs(wrap_degrees(phase - 90.0)), abs(wrap_degrees(phase + 90.0)))

    windows = burst_windows(t, u, record.condition.burst_kHz, carrier)
    monitor_x, monitor_y = burst_extrema_features(t, u, q_sign * q_unsigned, windows)
    integrated_q = integrate_current_per_burst(t, u, current, windows)
    current_x, current_y = burst_extrema_features(t, u, integrated_q, windows)
    level_label = "MAX" if record.is_maximum else str(record.level_percent)

    cprime = cstar.real * PF_PER_F if cstar is not None else None
    closs = -cstar.imag * PF_PER_F if cstar is not None else None
    tan_delta = -cstar.imag / cstar.real if cstar is not None and abs(cstar.real) > np.finfo(float).eps else None
    return FileFeatures(
        member=record.member,
        level_label=level_label,
        carrier_Hz=carrier,
        record_duration_s=float(t[-1] - t[0]),
        voltage_pp_kV=float((np.percentile(u, 99.9) - np.percentile(u, 0.1)) * 1.0e-3),
        q_sign=q_sign,
        current_sign=current_sign,
        monitor_Cprime_pF=float(cprime) if cprime is not None else None,
        monitor_Closs_pF=float(closs) if closs is not None else None,
        monitor_tan_delta=float(tan_delta) if tan_delta is not None else None,
        monitor_phase_deg=float(np.degrees(np.angle(cstar))) if cstar is not None else None,
        current_Creactive_pF=float(y.imag / (2.0 * np.pi * carrier) * PF_PER_F) if y is not None else None,
        current_G_uS=float(y.real * 1.0e6) if y is not None else None,
        current_monitor_gain_ratio=float(gain_ratio) if gain_ratio is not None else None,
        current_monitor_phase_error_deg=float(phase_error) if phase_error is not None else None,
        monitor_x_Uqpp_kV=monitor_x,
        monitor_y_Qpp_nC=monitor_y,
        current_x_Uqpp_kV=current_x,
        current_y_Qpp_nC=current_y,
        skipped_rows=waveform.skipped_rows,
        clipping_flag=detect_clipping(waveform.source_voltage_V, waveform.current_input_A, waveform.monitor_voltage_V),
        quiet_both_edges=edge_quiet(t, u, carrier),
    )


def median_or_none(values: Iterable[float | None]) -> float | None:
    finite = np.asarray([value for value in values if value is not None and np.isfinite(value)], dtype=float)
    return float(np.median(finite)) if len(finite) else None


def robust_relative_span(values: Iterable[float | None]) -> float | None:
    finite = np.asarray([value for value in values if value is not None and np.isfinite(value)], dtype=float)
    if len(finite) < 2 or abs(float(np.median(finite))) <= np.finfo(float).eps:
        return None
    return float((np.percentile(finite, 90) - np.percentile(finite, 10)) / abs(np.median(finite)))


def linear_fit(x: Sequence[float], y: Sequence[float]) -> LinearFit:
    xa = np.asarray(x, dtype=float)
    ya = np.asarray(y, dtype=float)
    finite = np.isfinite(xa) & np.isfinite(ya)
    xa, ya = xa[finite], ya[finite]
    if len(xa) < 2 or np.ptp(xa) <= np.finfo(float).eps:
        return LinearFit(None, None, None, int(len(xa)))
    slope, intercept = np.polyfit(xa, ya, 1)
    predicted = slope * xa + intercept
    total = float(np.sum((ya - np.mean(ya)) ** 2))
    residual = float(np.sum((ya - predicted) ** 2))
    r2 = 1.0 - residual / total if total > 0 else None
    return LinearFit(float(slope), float(intercept), float(r2) if r2 is not None else None, len(xa))


def per_level_medians(features: Sequence[FileFeatures], prefix: str) -> dict[str, dict[str, float | None]]:
    groups: dict[str, list[FileFeatures]] = defaultdict(list)
    for row in features:
        groups[row.level_label].append(row)
    result: dict[str, dict[str, float | None]] = {}
    for level, rows in groups.items():
        result[level] = {
            "x": median_or_none(getattr(row, f"{prefix}_x_Uqpp_kV") for row in rows),
            "y": median_or_none(getattr(row, f"{prefix}_y_Qpp_nC") for row in rows),
            "voltage_pp_kV": median_or_none(row.voltage_pp_kV for row in rows),
            "n_files": float(len(rows)),
        }
    return result


def scan_fit_summary(condition: Condition, features: Sequence[FileFeatures]) -> dict:
    by_level: dict[str, list[FileFeatures]] = defaultdict(list)
    for row in features:
        by_level[row.level_label].append(row)
    passive = [row for level in PASSIVE_FIT_LEVELS for row in by_level.get(str(level), [])]
    cprime_levels = [median_or_none(row.monitor_Cprime_pF for row in by_level.get(str(level), [])) for level in PASSIVE_FIT_LEVELS]
    cprime_levels = [value for value in cprime_levels if value is not None]
    ccell_monitor = float(np.median(cprime_levels)) if cprime_levels else None
    phase_levels = [median_or_none(abs(row.monitor_phase_deg) for row in by_level.get(str(level), [])) for level in PASSIVE_FIT_LEVELS]
    phase_levels = [value for value in phase_levels if value is not None]
    phase_median = float(np.median(phase_levels)) if phase_levels else None
    tan_levels = [median_or_none(row.monitor_tan_delta for row in by_level.get(str(level), [])) for level in PASSIVE_FIT_LEVELS]
    tan_levels = [value for value in tan_levels if value is not None]
    tan_delta = float(np.median(tan_levels)) if tan_levels else None
    closs_levels = [median_or_none(row.monitor_Closs_pF for row in by_level.get(str(level), [])) for level in PASSIVE_FIT_LEVELS]
    closs_levels = [value for value in closs_levels if value is not None]
    closs_median = float(np.median(closs_levels)) if closs_levels else None
    ccell_stability = robust_relative_span(cprime_levels)

    gain_ratio = median_or_none(row.current_monitor_gain_ratio for row in passive)
    phase_error = median_or_none(row.current_monitor_phase_error_deg for row in passive)
    monitor_kcl_valid = bool(
        gain_ratio is not None
        and phase_error is not None
        and 0.75 <= gain_ratio <= 1.25
        and phase_error <= 20.0
    )
    monitor_qv_valid = bool(
        ccell_monitor is not None
        and ccell_monitor > 0
        and phase_median is not None
        and phase_median <= 45.0
        and closs_median is not None
        and closs_median >= 0.0
        and ccell_stability is not None
        and ccell_stability <= 0.30
    )

    route_results: dict[str, dict] = {}
    for route in ("monitor", "current"):
        medians = per_level_medians(features, route)
        off_x, off_y = [], []
        for level in PASSIVE_FIT_LEVELS:
            row = medians.get(str(level))
            if row and row["x"] is not None and row["y"] is not None:
                off_x.append(float(row["x"]))
                off_y.append(float(row["y"]))
        off_fit = linear_fit(off_x, off_y)

        on_x, on_y = [], []
        for level in (*ACTIVE_FIT_LEVELS, "MAX"):
            row = medians.get(str(level))
            if row and row["x"] is not None and row["y"] is not None:
                on_x.append(float(row["x"]))
                on_y.append(float(row["y"]))
        on_fit = linear_fit(on_x, on_y)
        high_secant = None
        if "115" in medians and "MAX" in medians:
            x1, y1 = medians["115"]["x"], medians["115"]["y"]
            x2, y2 = medians["MAX"]["x"], medians["MAX"]["y"]
            if None not in (x1, y1, x2, y2) and abs(float(x2) - float(x1)) > 0.05:
                high_secant = (float(y2) - float(y1)) / (float(x2) - float(x1))

        if route == "monitor":
            ccell = ccell_monitor if ccell_monitor is not None else off_fit.slope
        else:
            current_levels = [
                median_or_none(row.current_Creactive_pF for row in by_level.get(str(level), []))
                for level in PASSIVE_FIT_LEVELS
            ]
            current_levels = [value for value in current_levels if value is not None and value > 0]
            ccell = float(np.median(current_levels)) if current_levels else off_fit.slope
        cd = high_secant if high_secant is not None else on_fit.slope
        factor = None
        if ccell is not None and cd is not None and cd > ccell * 1.05:
            factor = cd / (cd - ccell)
        route_results[route] = {
            "level_medians": medians,
            "off_qpp_fit": asdict(off_fit),
            "on_qmax_fit": asdict(on_fit),
            "Ccell_pF": ccell,
            "Cd_effective_high_field_pF": cd,
            "Cd_global_pF": on_fit.slope,
            "Cd_high_field_secant_pF": high_secant,
            "gas_charge_correction_factor": factor,
            "model_identifiable": bool(
                factor is not None
                and on_fit.r_squared is not None
                and on_fit.r_squared >= 0.90
                and factor <= 10.0
            ),
        }

    return {
        "condition": asdict(condition),
        "complete_scan": all(str(level) in by_level for level in (*PASSIVE_FIT_LEVELS, *ACTIVE_FIT_LEVELS)) and "MAX" in by_level,
        "files_analyzed": len(features),
        "monitor_qv_passive_valid": monitor_qv_valid,
        "monitor_kcl_with_pearson_valid": monitor_kcl_valid,
        "monitor_Cprime_pF": ccell_monitor,
        "monitor_Closs_pF": closs_median,
        "monitor_tan_delta": tan_delta,
        "monitor_phase_abs_median_deg": phase_median,
        "monitor_Cprime_relative_span": ccell_stability,
        "pearson_to_monitor_gain_ratio": gain_ratio,
        "pearson_to_monitor_phase_error_deg": phase_error,
        "routes": route_results,
        "interpretation": (
            "The monitor route is absolute only if Channel D wiring/value is independently known and the passive Q-V check passes. "
            "Pearson KCL failure means Channel C cannot validate Channel D; it may include upstream parasitic current or incorrect termination/scaling. "
            "The current route is therefore a provisional fallback, not an independent absolute calibration."
        ),
    }


def carrier_half_cycles(
    waveform: Waveform,
    carrier_Hz: float,
    q_sign: int,
    current_sign: int,
    reference_capacitance_F: float,
) -> list[dict[str, float]]:
    t, u = waveform.time_s, waveform.voltage_DBD_V
    dt = float(np.median(np.diff(t)))
    period_samples = max(5, int(round(1.0 / (carrier_Hz * dt))))
    us = moving_average(u, min(11, max(3, int(round(0.025 * period_samples)) | 1)))
    center = 0.5 * (float(np.percentile(us, 2.5)) + float(np.percentile(us, 97.5)))
    centered = us - center
    left, right = centered[:-1], centered[1:]
    indices = np.flatnonzero(((left <= 0) & (right > 0)) | ((left >= 0) & (right < 0)))
    if len(indices) < 3:
        return []
    crossing_times = []
    for index in indices:
        denominator = centered[index + 1] - centered[index]
        fraction = 0.5 if denominator == 0 else float(np.clip(-centered[index] / denominator, 0, 1))
        crossing_times.append(float(t[index] + fraction * (t[index + 1] - t[index])))
    crossing_times = np.asarray(crossing_times)
    q_monitor = q_sign * reference_capacitance_F * waveform.monitor_voltage_V
    q_at = np.interp(crossing_times, t, q_monitor)
    current = current_sign * waveform.current_input_A
    baseline = float(np.median(current[np.abs(centered) <= np.percentile(np.abs(centered), 10)]))
    cumulative = np.r_[0.0, np.cumsum(0.5 * ((current[1:] - baseline) + (current[:-1] - baseline)) * np.diff(t))]
    qi_at = np.interp(crossing_times, t, cumulative)
    half_period = 0.5 / carrier_Hz
    robust_peak = max(abs(float(np.percentile(centered, 1))), abs(float(np.percentile(centered, 99))))
    rows: list[dict[str, float]] = []
    for j, (start, stop) in enumerate(zip(indices[:-1], indices[1:])):
        duration = crossing_times[j + 1] - crossing_times[j]
        segment = centered[start : stop + 2]
        amplitude = float(np.max(np.abs(segment)))
        if not (0.60 * half_period <= duration <= 1.40 * half_period and amplitude >= 0.08 * robust_peak):
            continue
        polarity = 1.0 if float(np.mean(segment)) >= 0 else -1.0
        rows.append(
            {
                "polarity": polarity,
                "amplitude_kV": amplitude * 1.0e-3,
                "duration_s": duration,
                "monitor_directed_C": polarity * float(q_at[j + 1] - q_at[j]),
                "current_directed_C": polarity * float(qi_at[j + 1] - qi_at[j]),
            }
        )
    return rows


def fit_passive_lobe_slopes(
    archive: zipfile.ZipFile,
    selected: dict[tuple[Condition, str], list[MemberRecord]],
    file_features: dict[str, FileFeatures],
    condition: Condition,
    args: argparse.Namespace,
) -> dict[str, dict[str, float | None]]:
    ratios: dict[str, dict[int, list[float]]] = {
        "monitor": {-1: [], 1: []},
        "current": {-1: [], 1: []},
    }
    for level in PASSIVE_FIT_LEVELS:
        for record in selected.get((condition, str(level)), []):
            feature = file_features.get(record.member)
            if feature is None:
                continue
            waveform = read_waveform_member(
                archive, record, args.voltage_scale, args.monitor_voltage_scale,
                args.source_to_ground, args.reference_polarity,
            )
            lobes = carrier_half_cycles(
                waveform, feature.carrier_Hz, feature.q_sign, feature.current_sign,
                args.reference_capacitance_uf * 1.0e-6,
            )
            for polarity in (-1, 1):
                for route in ("monitor", "current"):
                    values = [
                        row[f"{route}_directed_C"] * NC_PER_C / row["amplitude_kV"]
                        for row in lobes
                        if int(row["polarity"]) == polarity and row["amplitude_kV"] > 0.02
                    ]
                    if values:
                        ratios[route][polarity].append(float(np.median(values)))
    result: dict[str, dict[str, float | None]] = {}
    for route in ("monitor", "current"):
        result[route] = {}
        for polarity in (-1, 1):
            values = np.asarray(ratios[route][polarity], dtype=float)
            key = "negative" if polarity < 0 else "positive"
            result[route][f"{key}_nC_per_kV"] = float(np.median(values)) if len(values) else None
            result[route][f"{key}_MAD_nC_per_kV"] = (
                float(1.4826 * np.median(np.abs(values - np.median(values)))) if len(values) else None
            )
    return result


def summarize_max_rates(
    archive: zipfile.ZipFile,
    records: Sequence[MemberRecord],
    file_features: dict[str, FileFeatures],
    route: str,
    passive_model: dict[str, float | None],
    correction_factor: float | None,
    args: argparse.Namespace,
) -> dict:
    if correction_factor is None:
        return {"status": "not_identifiable", "reason": "Cd is not safely greater than Ccell."}
    file_rows: list[dict[str, float]] = []
    for record in records:
        feature = file_features.get(record.member)
        if feature is None:
            continue
        waveform = read_waveform_member(
            archive, record, args.voltage_scale, args.monitor_voltage_scale,
            args.source_to_ground, args.reference_polarity,
        )
        lobes = carrier_half_cycles(
            waveform, feature.carrier_Hz, feature.q_sign, feature.current_sign,
            args.reference_capacitance_uf * 1.0e-6,
        )
        record_duration = float(waveform.time_s[-1] - waveform.time_s[0])
        result: dict[str, float] = {}
        totals: dict[int, float] = {}
        for polarity in (-1, 1):
            name = "negative" if polarity < 0 else "positive"
            slope = passive_model.get(f"{name}_nC_per_kV")
            if slope is None:
                continue
            corrected: list[float] = []
            rates: list[float] = []
            for row in lobes:
                if int(row["polarity"]) != polarity:
                    continue
                measured_nC = row[f"{route}_directed_C"] * NC_PER_C
                excess_nC = correction_factor * (measured_nC - float(slope) * row["amplitude_kV"])
                corrected.append(excess_nC)
                rates.append(excess_nC * 1.0e-9 / (ELEMENTARY_CHARGE_C * row["duration_s"]))
            if not corrected:
                continue
            values = np.asarray(corrected)
            # Signed summation avoids the positive-noise bias that would result
            # from clipping every background-subtracted lobe at zero.
            total_C = float(np.sum(values) * 1.0e-9)
            totals[polarity] = total_C
            active = values[values > max(0.0, np.percentile(values, 25))]
            if not len(active):
                active = values
            result[f"{name}_charge_per_active_half_cycle_median_nC"] = float(np.median(active))
            result[f"{name}_charge_per_half_cycle_p95_nC"] = float(np.percentile(values, 95))
            result[f"{name}_record_average_equivalent_rate_per_s"] = total_C / (ELEMENTARY_CHARGE_C * record_duration)
            result[f"{name}_half_cycle_average_equivalent_rate_p95_per_s"] = float(np.percentile(rates, 95))
            result[f"{name}_corrected_charge_total_nC"] = total_C * NC_PER_C
            result[f"{name}_half_cycles"] = float(len(values))
        if -1 in totals and 1 in totals:
            result["polarity_charge_imbalance_nC"] = (totals[1] - totals[-1]) * NC_PER_C
        if result:
            file_rows.append(result)
    if not file_rows:
        return {"status": "not_available", "reason": "No usable maximum-voltage half-cycles."}
    keys = sorted(set().union(*(row.keys() for row in file_rows)))
    summary = {key: median_or_none(row.get(key) for row in file_rows) for key in keys}
    return {
        "status": "provisional",
        "files_analyzed": len(file_rows),
        "results": summary,
        "retained_charge_status": (
            "Unavailable: records do not contain quiet pre- and post-burst windows. "
            "Polarity imbalance is not the same as retained surface charge."
        ),
    }


def write_csv(path: Path, rows: Sequence[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(rows[0].keys())
    for row in rows[1:]:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def flatten(prefix: str, value, output: dict) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            flatten(f"{prefix}_{key}" if prefix else str(key), child, output)
    elif not isinstance(value, (list, tuple)):
        output[prefix] = value


def plot_scan(condition: Condition, summary: dict, output: Path, dpi: int) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    for axis, route, color in zip(axes, ("monitor", "current"), ("#1f77b4", "#d95f02")):
        route_result = summary["routes"][route]
        medians = route_result["level_medians"]
        for level, marker in (("40", "o"), ("60", "o"), ("75", "o"), ("90", "s"), ("100", "D"), ("105", "^"), ("115", "^"), ("MAX", "*")):
            row = medians.get(level)
            if row and row["x"] is not None and row["y"] is not None:
                axis.scatter(row["x"], row["y"], marker=marker, s=80 if level == "MAX" else 42, color=color)
                axis.annotate(level, (row["x"], row["y"]), xytext=(4, 3), textcoords="offset points", fontsize=8)
        on = route_result["on_qmax_fit"]
        if on["slope"] is not None:
            x_values = [row["x"] for row in medians.values() if row["x"] is not None]
            grid = np.linspace(min(x_values), max(x_values), 100)
            axis.plot(grid, on["slope"] * grid + on["intercept"], "--", color=color, alpha=0.8)
        axis.set_xlabel(r"$U(Q_+) - U(Q_-)$ (kV)")
        axis.set_ylabel(r"$Q_+ - Q_-$ (nC)")
        title = "Channel-D nominal Q" if route == "monitor" else "Pearson-integrated Q (fallback)"
        axis.set_title(title)
        axis.grid(alpha=0.25)
    fig.suptitle(f"{condition.material}, {condition.burst_kHz} kHz scan-level extrema")
    fig.tight_layout()
    fig.savefig(output, dpi=dpi)
    plt.close(fig)


def analyze_archive(args: argparse.Namespace) -> tuple[dict, str]:
    archive_path = args.archive_zip.expanduser().resolve()
    if not archive_path.is_file():
        raise ScanAnalysisError(f"Archive does not exist: {archive_path}")
    if args.files_per_level < 0:
        raise ScanAnalysisError("--files-per-level must be 0 or positive.")
    if args.reference_capacitance_uf <= 0:
        raise ScanAnalysisError("--reference-capacitance-uf must be positive.")
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else archive_path.with_name(f"{archive_path.stem}_analysis")
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = output_dir / "figures"
    if not args.no_plots:
        figures_dir.mkdir(exist_ok=True)

    with zipfile.ZipFile(archive_path) as archive:
        manifest = inventory_scan_archive(archive)
        grouped: dict[tuple[Condition, str], list[MemberRecord]] = defaultdict(list)
        for record in manifest:
            label = "MAX" if record.is_maximum else str(record.level_percent)
            grouped[(record.condition, label)].append(record)
        selected = {key: evenly_spaced(rows, args.files_per_level) for key, rows in grouped.items()}

        file_features: dict[str, FileFeatures] = {}
        failures: list[dict] = []
        total = sum(len(rows) for rows in selected.values())
        completed = 0
        for key in sorted(selected, key=lambda item: (item[0], item[1])):
            for record in selected[key]:
                try:
                    waveform = read_waveform_member(
                        archive, record, args.voltage_scale, args.monitor_voltage_scale,
                        args.source_to_ground, args.reference_polarity,
                    )
                    file_features[record.member] = extract_file_features(
                        waveform,
                        record,
                        args.reference_capacitance_uf * 1.0e-6,
                        args.carrier_min_khz * 1000.0,
                        args.carrier_max_khz * 1000.0,
                    )
                except Exception as error:  # keep a complete, auditable batch log
                    failures.append({"member": record.member, "error": str(error)})
                completed += 1
                if completed % 50 == 0 or completed == total:
                    print(f"Processed {completed}/{total} selected waveform files...", flush=True)

        conditions = sorted({record.condition for record in manifest})
        scan_summaries: dict[str, dict] = {}
        for condition in conditions:
            rows = [
                feature for member, feature in file_features.items()
                if next((record.condition for record in manifest if record.member == member), None) == condition
            ]
            if not rows:
                continue
            summary = scan_fit_summary(condition, rows)
            scan_summaries[condition.label] = summary
            if not args.no_plots and summary["complete_scan"]:
                plot_scan(condition, summary, figures_dir / f"{condition.label}_scan_fit.png", args.dpi)

        # Same-material transfer map for unscanned duty frequencies.  It is
        # explicitly labeled and never treated as independently measured.
        sources: dict[str, Condition] = {}
        for condition in conditions:
            summary = scan_summaries.get(condition.label)
            if summary and summary["complete_scan"]:
                current = sources.get(condition.material)
                if current is None or condition.burst_kHz == 20 or (condition.material == "pure_water" and condition.burst_kHz == 4):
                    sources[condition.material] = condition

        rate_rows: list[dict] = []
        passive_models: dict[str, dict] = {}
        for condition in conditions:
            source_condition = condition if scan_summaries.get(condition.label, {}).get("complete_scan") else sources.get(condition.material)
            if source_condition is None:
                rate_rows.append({
                    "material": condition.material,
                    "burst_kHz": condition.burst_kHz,
                    "route": "none",
                    "status": "not_identifiable",
                    "reason": "No same-material complete voltage scan exists.",
                })
                continue
            source_summary = scan_summaries[source_condition.label]
            if source_condition.label not in passive_models:
                passive_models[source_condition.label] = fit_passive_lobe_slopes(
                    archive, selected, file_features, source_condition, args,
                )
            model = passive_models[source_condition.label]
            max_records = selected.get((condition, "MAX"), [])
            for route in ("monitor", "current"):
                route_fit = source_summary["routes"][route]
                apparent_factor = route_fit["gas_charge_correction_factor"]
                if route == "current" and apparent_factor is not None:
                    # The Pearson channel contains an unresolved parallel
                    # displacement contribution.  If Cp >= 0, the physical
                    # gas-charge factor is bounded by 1 < Ftrue < Fapp, where
                    # Fapp uses the two Pearson-derived slopes.  Preserve that
                    # sensitivity range instead of presenting Fapp as Cd.
                    rates = {
                        "status": "sensitivity_range",
                        "factor_lower": 1.0,
                        "factor_upper_apparent": apparent_factor,
                        "lower_F1": summarize_max_rates(
                            archive, max_records, file_features, route,
                            model[route], 1.0, args,
                        ),
                        "upper_Fapp": summarize_max_rates(
                            archive, max_records, file_features, route,
                            model[route], apparent_factor, args,
                        ),
                        "reason": (
                            "Pearson passive/active slopes include an unmeasured parallel capacitance; "
                            "the physical correction factor cannot be isolated."
                        ),
                    }
                else:
                    rates = summarize_max_rates(
                        archive,
                        max_records,
                        file_features,
                        route,
                        model[route],
                        apparent_factor,
                        args,
                    )
                calibration_valid = (
                    source_summary["monitor_qv_passive_valid"]
                    if route == "monitor"
                    else source_summary["monitor_kcl_with_pearson_valid"]
                )
                flat: dict = {
                    "material": condition.material,
                    "burst_kHz": condition.burst_kHz,
                    "route": route,
                    "scan_source": source_condition.label,
                    "extrapolated_across_burst_frequency": source_condition != condition,
                    "calibration_valid": calibration_valid,
                    "model_identifiable": route_fit["model_identifiable"],
                    "reported_status": (
                        "validated" if calibration_valid and route_fit["model_identifiable"]
                        else "provisional_not_for_absolute_reporting"
                    ),
                }
                flatten("rate", rates, flat)
                rate_rows.append(flat)

        manifest_rows = [
            {
                "member": record.member,
                "material": record.condition.material,
                "burst_kHz": record.condition.burst_kHz,
                "level": "MAX" if record.is_maximum else record.level_percent,
                "capture_index": record.capture_index,
                "selected": any(record in rows for rows in selected.values()),
            }
            for record in manifest
        ]
        feature_rows = [asdict(feature) for feature in file_features.values()]
        cap_rows: list[dict] = []
        validated_rows: list[dict] = []
        for condition_label, summary in scan_summaries.items():
            flat = {"condition": condition_label}
            flatten("", {key: value for key, value in summary.items() if key != "routes"}, flat)
            for route, route_data in summary["routes"].items():
                route_flat = dict(flat)
                route_flat["route"] = route
                flatten("", {key: value for key, value in route_data.items() if key != "level_medians"}, route_flat)
                cap_rows.append(route_flat)
            monitor_route = summary["routes"]["monitor"]
            independent = bool(summary["complete_scan"])
            ccell_supported = independent and bool(summary["monitor_qv_passive_valid"])
            cd_supported = ccell_supported and bool(monitor_route["model_identifiable"])
            validated_rows.append({
                "condition": condition_label,
                "independent_complete_scan": independent,
                "Ccell_status": (
                    "supported_effective_complex_at_carrier" if ccell_supported
                    else ("no_independent_scan" if not independent else "failed_passive_QV_validation")
                ),
                "Ccell_reactive_pF": summary["monitor_Cprime_pF"] if ccell_supported else None,
                "Ccell_loss_pF": summary["monitor_Closs_pF"] if ccell_supported else None,
                "loss_tangent": summary["monitor_tan_delta"] if ccell_supported else None,
                "Cd_status": "supported_effective_high_field" if cd_supported else "not_identifiable",
                "Cd_effective_pF": monitor_route["Cd_effective_high_field_pF"] if cd_supported else None,
                "surface_charge_status": "supported_model_dependent" if cd_supported else "not_identifiable",
                "retained_charge_status": "not_measured_no_quiet_record_edges",
                "note": (
                    "Ccell is a carrier-frequency complex effective value, not a classical burst-wide Q-V slope."
                    if ccell_supported else
                    "No scalar capacitance passed the passive orientation/amplitude-stability checks."
                ),
            })

        write_csv(output_dir / "archive_manifest.csv", manifest_rows)
        write_csv(output_dir / "file_level_qc_features.csv", feature_rows)
        write_csv(output_dir / "scan_capacitance_results.csv", cap_rows)
        write_csv(output_dir / "validated_results.csv", validated_rows)
        write_csv(output_dir / "max_voltage_charge_results.csv", rate_rows)
        audit = {
            "analysis_version": ANALYSIS_VERSION,
            "archive": str(archive_path),
            "files_per_level_requested": args.files_per_level,
            "raw_members_found": len(manifest),
            "selected_files": total,
            "successfully_analyzed_files": len(file_features),
            "failures": failures,
            "conditions": scan_summaries,
            "passive_lobe_models": passive_models,
            "method_limits": [
                "Channel-D charge scales linearly with the assumed reference capacitance; its tolerance was not supplied.",
                "Pearson current is not a valid independent charge calibration unless its termination, scale, and current path are established.",
                "Negative charge is electrons plus negative ions; species are not electrically separable here.",
                "The p95 half-cycle-average rate is not an instantaneous nanosecond microdischarge peak.",
                "Missing-frequency results transfer a same-material effective model and are explicitly flagged.",
                "No retained surface charge is reported without quiet pre- and post-burst windows.",
            ],
        }
        (output_dir / "analysis_audit.json").write_text(json.dumps(audit, indent=2, allow_nan=False), encoding="utf-8")

    report_lines = [
        "DBD voltage-scan analysis",
        f"Analysis version: {ANALYSIS_VERSION}",
        f"Archive: {archive_path}",
        f"Selected/analyzed files: {total}/{len(file_features)} successful; {len(failures)} failed",
        "",
        "Absolute-reporting rule:",
        "  A monitor result is absolute only when passive Channel-D Q-V behavior is plausible and stable.",
        "  Pearson-current results are provisional unless I = Cref*dVmonitor/dt passes gain and phase checks.",
        "  A failed route remains in the CSV as a diagnostic but must not be reported as plasma-to-surface charge.",
        "",
    ]
    for label, summary in scan_summaries.items():
        monitor = summary["routes"]["monitor"]
        ccell_text = (
            f"{summary['monitor_Cprime_pF']:.6g} - i {summary['monitor_Closs_pF']:.6g} pF"
            if summary["complete_scan"] and summary["monitor_qv_passive_valid"]
            else "not supported by the passive-QV validation"
        )
        cd_text = (
            f"{monitor['Cd_effective_high_field_pF']:.6g} pF"
            if summary["monitor_qv_passive_valid"] and monitor["model_identifiable"]
            else "not identifiable from these scan extrema"
        )
        report_lines.extend([
            f"{label}:",
            f"  Monitor passive-QV valid: {summary['monitor_qv_passive_valid']}; Pearson/monitor KCL valid: {summary['monitor_kcl_with_pearson_valid']}",
            f"  Ccell*: {ccell_text}; tan(delta): {summary['monitor_tan_delta'] if summary['monitor_qv_passive_valid'] else 'not reportable'}",
            f"  Cd,eff: {cd_text}",
            "  Absolute plasma-to-surface charge: not identifiable" if not (
                summary["monitor_qv_passive_valid"] and monitor["model_identifiable"]
            ) else "  Absolute plasma-to-surface charge: model-dependent result available in CSV",
        ])
    report = "\n".join(report_lines)
    (output_dir / "analysis_report.txt").write_text(report, encoding="utf-8")
    return {
        "output_dir": str(output_dir),
        "conditions": scan_summaries,
        "failures": failures,
    }, report


def main(argv: Sequence[str] | None = None) -> int:
    try:
        args = parse_arguments(argv)
        _, report = analyze_archive(args)
    except (ScanAnalysisError, OSError, zipfile.BadZipFile, ValueError) as error:
        print(f"ERROR: {error}", file=__import__("sys").stderr)
        return 2
    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
