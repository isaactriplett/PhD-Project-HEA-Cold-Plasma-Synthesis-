# -*- coding: utf-8 -*-
"""Analyze four-channel AC-DBD oscilloscope exports with a Lissajous capacitor.

The expected columns are time (ms), applied voltage, current, and voltage across
the monitor/Lissajous capacitor.  The input may contain arbitrary non-numeric
header and units rows, such as the CSV exports from the oscilloscope used here.
Alongside the complete waveform figure, the script writes a second waveform
figure automatically zoomed to two repeated duty cycles/bursts.

Example
-------
python Lissajous_Figures.py "C:\\path\\to\\waveform.csv" --no-show

For a folder/archive containing below- and above-breakdown voltage scans, use
``Lissajous_Scan_Analysis.py``.  A single non-parallelogram Q-V trace cannot
independently establish both C_cell and C_d; the single-file two-slope result is
therefore a classical-model diagnostic, not a substitute for the scan-level
calibration and identifiability checks in the batch script.

By default, the applied-voltage values are interpreted as kV (and are converted
to V internally), and the monitor capacitance is 0.1 uF.  Change
--voltage-scale to 1 when the voltage column is already in volts.

Current polarity is detected for each file by comparing Pearson-current charge
integrals with monitor-capacitor charge changes over equal-voltage carrier
half-cycles.  Use --current-polarity 1 or -1 only when the automatic diagnostic
is ambiguous or a separately calibrated sign must be enforced.

The capacitance fit uses the conventional, lumped, two-capacitor DBD model:

* plasma-off sides of Q(V) give C_cell (the displacement capacitance);
* plasma-on sides give the effective dielectric/discharge capacitance.

For a fully bridged gap, the latter is the dielectric capacitance.  For partial,
surface, or packed-bed discharges it is an effective capacitance; the reported
dielectric-reaching charge is then an apparent, model-dependent value unless a
physical dielectric capacitance is supplied with --dielectric-capacitance-pf.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np


ANALYSIS_VERSION = "1.4"
PF_PER_F = 1.0e12
NC_PER_C = 1.0e9
KV_PER_V = 1.0e-3
ELEMENTARY_CHARGE_C = 1.602176634e-19


@dataclass
class LineFit:
    """A linear Q(V) fit in plotting units: Q in nC and V in kV."""

    slope_nC_per_kV: float
    intercept_nC: float
    r_squared: float
    rms_residual_nC: float
    n_points: int
    x_min_kV: float
    x_max_kV: float

    @property
    def capacitance_F(self) -> float:
        # 1 nC/kV = 1 pF.
        return self.slope_nC_per_kV / PF_PER_F


@dataclass
class HalfCycleFit:
    """Piecewise-linear fit of one monotonic voltage half-cycle."""

    start: int
    breakpoint: int
    stop: int
    direction: str
    first: LineFit
    second: LineFit
    off_is_first: bool

    @property
    def off(self) -> LineFit:
        return self.first if self.off_is_first else self.second

    @property
    def on(self) -> LineFit:
        return self.second if self.off_is_first else self.first


@dataclass
class DutyCycleSelection:
    """The automatically selected window used for the two-duty-cycle plot."""

    start: int
    stop: int
    period_samples: int
    period_s: float
    frequency_Hz: float
    method: str
    activity_channel: str
    cycles_displayed: float


@dataclass
class CurrentPolarityDecision:
    """Auditable choice of the multiplier applied to the current column."""

    sign: int
    requested: str
    method: str
    half_cycle_charge_correlation_raw: float | None
    usable_half_cycles: int
    direction_sign_agreement: float | None
    delay_sign_agreement: float | None
    confidence: str


class AnalysisError(RuntimeError):
    """Raised when the data do not support the requested DBD analysis."""


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create voltage/current/charge waveforms and Q-V Lissajous "
            "capacitance fits from a four-column AC-DBD CSV export."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("input_csv", type=Path, help="CSV containing time, voltage, current, and monitor-capacitor voltage")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for plots and data outputs; default is <input-stem>_analysis beside the CSV",
    )
    parser.add_argument(
        "--reference-capacitance-uf",
        type=float,
        default=0.1,
        help="Monitor/Lissajous capacitor capacitance in microfarads",
    )
    parser.add_argument(
        "--voltage-scale",
        type=float,
        default=1000.0,
        help="Volts per value in column 2; use 1 for a voltage column already in V",
    )
    parser.add_argument(
        "--reference-voltage-scale",
        type=float,
        default=1.0,
        help="Volts per value in column 4",
    )
    parser.add_argument(
        "--current-polarity",
        choices=("auto", "1", "-1"),
        default="auto",
        help=(
            "Multiplier applied to the current column. Auto compares current "
            "integrated over carrier half-cycles with the corresponding change "
            "in monitor-capacitor voltage; use 1 or -1 to override an ambiguous trace"
        ),
    )
    parser.add_argument(
        "--charge-polarity",
        choices=("auto", "1", "-1"),
        default="auto",
        help="Sign of charge inferred from column 4; auto selects the sign with a positive capacitive correlation",
    )
    parser.add_argument(
        "--source-to-ground",
        action="store_true",
        help=(
            "Treat column 2 as source-to-ground voltage rather than the voltage "
            "across the DBD; subtract the signed monitor-capacitor voltage to obtain V_DBD"
        ),
    )
    parser.add_argument(
        "--reference-polarity",
        type=int,
        choices=(-1, 1),
        default=1,
        help="Sign of monitor voltage in V_DBD = V_source - sign*V_monitor when --source-to-ground is used",
    )
    parser.add_argument(
        "--dielectric-capacitance-pf",
        type=float,
        default=None,
        help=(
            "Known physical dielectric capacitance in pF.  If supplied, it is used "
            "for the dielectric-charge calculation instead of assuming a fully bridged gap."
        ),
    )
    parser.add_argument(
        "--frequency-hz",
        type=float,
        default=None,
        help="Analysis frequency used to delimit voltage half-cycles; by default the dominant waveform frequency is estimated",
    )
    parser.add_argument(
        "--smooth-window",
        type=int,
        default=0,
        help="Odd moving-average window in samples for locating voltage extrema; 0 chooses one automatically",
    )
    parser.add_argument(
        "--max-plot-points",
        type=int,
        default=30000,
        help="Maximum raw samples drawn in the Lissajous trace",
    )
    parser.add_argument("--dpi", type=int, default=180, help="PNG output resolution")
    parser.add_argument("--no-show", action="store_true", help="Save plots without opening interactive windows")
    return parser.parse_args()


def _detect_delimiter(path: Path) -> str:
    """Return a likely delimiter while tolerating short / unusual CSV headers."""
    with path.open("r", newline="", encoding="utf-8-sig", errors="replace") as handle:
        sample = handle.read(16384)
    try:
        return csv.Sniffer().sniff(sample, delimiters=",;\t").delimiter
    except csv.Error:
        return ","


def read_waveform_csv(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Read the first four numeric columns, ignoring metadata/unit/blank rows."""
    if not path.is_file():
        raise AnalysisError(f"Input file does not exist: {path}")

    delimiter = _detect_delimiter(path)
    numeric_rows: list[tuple[float, float, float, float]] = []
    with path.open("r", newline="", encoding="utf-8-sig", errors="replace") as handle:
        for row in csv.reader(handle, delimiter=delimiter):
            if len(row) < 4:
                continue
            try:
                values = tuple(float(cell.strip()) for cell in row[:4])
            except ValueError:
                continue
            if all(math.isfinite(value) for value in values):
                numeric_rows.append(values)

    if len(numeric_rows) < 40:
        raise AnalysisError(
            "Fewer than 40 valid four-column samples were found. "
            "Check the delimiter and ensure columns are time, voltage, current, monitor voltage."
        )

    data = np.asarray(numeric_rows, dtype=float)
    time_ms, voltage_raw, current_A, reference_voltage_raw = data.T

    order = np.argsort(time_ms, kind="stable")
    time_ms = time_ms[order]
    voltage_raw = voltage_raw[order]
    current_A = current_A[order]
    reference_voltage_raw = reference_voltage_raw[order]

    # np.gradient requires strictly increasing coordinates.  Keep the first value
    # at a duplicated timestamp; it preserves the waveform ordering otherwise.
    keep = np.r_[True, np.diff(time_ms) > 0]
    time_ms = time_ms[keep]
    voltage_raw = voltage_raw[keep]
    current_A = current_A[keep]
    reference_voltage_raw = reference_voltage_raw[keep]

    if len(time_ms) < 40 or np.ptp(time_ms) <= 0:
        raise AnalysisError("Time values must contain at least 40 strictly increasing samples.")
    return time_ms, voltage_raw, current_A, reference_voltage_raw


def moving_average(values: np.ndarray, window: int) -> np.ndarray:
    """Centered moving average with edge padding and an odd positive window."""
    if window <= 1:
        return values.copy()
    if window % 2 == 0:
        window += 1
    window = min(window, len(values) if len(values) % 2 else len(values) - 1)
    if window <= 1:
        return values.copy()
    pad = window // 2
    padded = np.pad(values, pad, mode="edge")
    kernel = np.full(window, 1.0 / window)
    return np.convolve(padded, kernel, mode="valid")


def estimate_fundamental_frequency(time_s: np.ndarray, voltage_V: np.ndarray) -> float | None:
    """Estimate the dominant AC frequency from an FFT of an evenly sampled trace."""
    spacing = np.diff(time_s)
    dt = float(np.median(spacing))
    if dt <= 0 or not np.isfinite(dt):
        return None
    # Tolerate a sparse missing/corrupt oscilloscope row by interpolating back to
    # the median time grid.  Broadly nonuniform data are still rejected because
    # an FFT carrier estimate would then be misleading.
    nonuniform = np.abs(spacing - dt) > 0.05 * dt
    if np.any(nonuniform):
        if float(np.mean(nonuniform)) > 0.01 or float(np.max(spacing)) > 10.0 * dt:
            return None
        sample_count = int(round((time_s[-1] - time_s[0]) / dt)) + 1
        if sample_count < 3 or sample_count > 2 * len(time_s):
            return None
        uniform_time = time_s[0] + np.arange(sample_count) * dt
        voltage_for_fft = np.interp(uniform_time, time_s, voltage_V)
    else:
        voltage_for_fft = voltage_V
    centered = voltage_for_fft - np.mean(voltage_for_fft)
    if np.ptp(centered) == 0:
        return None
    spectrum = np.abs(np.fft.rfft(centered * np.hanning(len(centered))))
    frequencies = np.fft.rfftfreq(len(centered), dt)
    if len(frequencies) < 3:
        return None
    spectrum[0] = 0.0
    index = int(np.argmax(spectrum))
    frequency = float(frequencies[index])
    return frequency if frequency > 0 else None


def normalized_autocorrelation(values: np.ndarray) -> np.ndarray | None:
    """Return the biased-corrected autocorrelation, normalized at zero lag."""
    centered = values - np.mean(values)
    energy = float(np.dot(centered, centered))
    if energy <= np.finfo(float).eps:
        return None
    count = len(centered)
    fft_length = 1 << (2 * count - 1).bit_length()
    spectrum = np.fft.rfft(centered, fft_length)
    correlation = np.fft.irfft(spectrum * np.conjugate(spectrum), fft_length)[:count]
    correlation /= np.arange(count, 0, -1)
    return correlation / correlation[0]


def local_maxima(values: np.ndarray) -> np.ndarray:
    """Indices of simple local maxima, excluding the two endpoints."""
    if len(values) < 3:
        return np.empty(0, dtype=int)
    return np.flatnonzero((values[1:-1] >= values[:-2]) & (values[1:-1] > values[2:])) + 1


def well_spaced_peaks(values: np.ndarray, minimum_spacing: int) -> np.ndarray:
    """Greedily retain the strongest local maxima, one per candidate cycle."""
    candidates = local_maxima(values)
    if not len(candidates):
        return candidates
    selected: list[int] = []
    for candidate in candidates[np.argsort(values[candidates])[::-1]]:
        if all(abs(int(candidate) - existing) >= minimum_spacing for existing in selected):
            selected.append(int(candidate))
    return np.asarray(sorted(selected), dtype=int)


def normalize_activity(values: np.ndarray) -> np.ndarray:
    """Robustly scale absolute activity so voltage and current can be combined."""
    centered = values - np.median(values)
    activity = np.abs(centered)
    scale = float(np.percentile(activity, 95))
    return activity / scale if scale > np.finfo(float).eps else np.zeros_like(activity)


def select_two_duty_cycles(
    time_s: np.ndarray,
    voltage_V: np.ndarray,
    current_A: np.ndarray,
) -> DutyCycleSelection:
    """Find two adjacent excitation periods directly from the measured waveform.

    A conventional continuous AC trace falls back to the dominant voltage
    frequency.  For burst-mode sources, voltage and current envelopes suppress
    the fast carrier; repeated envelope rising edges identify the slower
    repetition/duty-cycle period.  This makes the zoom useful for both steady
    sinusoidal DBDs and damped high-frequency bursts.
    """
    spacing = np.diff(time_s)
    dt_s = float(np.median(spacing))
    if dt_s <= 0 or not np.isfinite(dt_s):
        raise AnalysisError("Cannot select duty cycles because time spacing is invalid.")
    count = len(time_s)
    if count < 8:
        raise AnalysisError("Cannot select duty cycles from fewer than eight samples.")

    carrier_frequency = estimate_fundamental_frequency(time_s, voltage_V)
    if carrier_frequency is not None:
        carrier_samples = max(3, int(round(1.0 / (carrier_frequency * dt_s))))
    else:
        carrier_samples = max(3, count // 16)

    # Smooth over roughly one and a half carrier cycles.  This intentionally
    # removes a continuous sine wave's |V| ripple while retaining any slower
    # pulse/burst envelope.  The cap prevents a short record from being almost
    # entirely edge-padded.
    envelope_window = max(5, int(round(1.5 * carrier_samples)) | 1)
    envelope_window = min(envelope_window, max(5, (count // 5) | 1))
    activity_signals = {
        "voltage": normalize_activity(voltage_V),
        "current": normalize_activity(current_A),
    }
    envelopes = {name: moving_average(activity, envelope_window) for name, activity in activity_signals.items()}

    def contrast(values: np.ndarray) -> float:
        low, high = np.percentile(values, (10, 90))
        return float((high - low) / high) if high > np.finfo(float).eps else 0.0

    contrasts = {name: contrast(values) for name, values in envelopes.items()}

    def rising_edge_candidate(values: np.ndarray) -> tuple[int, tuple[int, int]] | None:
        low, high = np.percentile(values, (10, 90))
        active = values >= low + 0.30 * (high - low)
        starts = np.flatnonzero(active & ~np.r_[False, active[:-1]])
        stops = np.flatnonzero(active & ~np.r_[active[1:], False]) + 1
        run_lengths = stops - starts
        if len(run_lengths):
            minimum_length = max(3, int(round(0.20 * np.median(run_lengths))))
            starts = starts[run_lengths >= minimum_length]
        if len(starts) < 3:
            return None
        intervals = np.diff(starts)
        initial_period = float(np.median(intervals))
        inliers = intervals[(intervals >= 0.60 * initial_period) & (intervals <= 1.40 * initial_period)]
        if not len(inliers):
            return None
        candidate_period = float(np.median(inliers))
        triples = [
            (int(starts[index]), int(starts[index + 2]))
            for index in range(len(starts) - 2)
            if abs((starts[index + 1] - starts[index]) - candidate_period) <= 0.15 * candidate_period
            and abs((starts[index + 2] - starts[index + 1]) - candidate_period) <= 0.15 * candidate_period
        ]
        if not triples:
            return None
        midpoint = 0.5 * (count - 1)
        return max(3, int(round(candidate_period))), min(
            triples, key=lambda pair: abs(0.5 * (pair[0] + pair[1]) - midpoint)
        )

    edge_candidates = {name: rising_edge_candidate(values) for name, values in envelopes.items()}
    period_samples: int | None = None
    method = "dominant voltage frequency fallback"
    activity_channel = "voltage"
    envelope = envelopes[activity_channel]
    aligned_window: tuple[int, int] | None = None

    # Prefer a visibly gated voltage envelope.  A continuous AC DBD can have
    # current microdischarges every carrier cycle, which must not be mistaken
    # for a slower duty-cycle envelope.  Current-only detection is accepted
    # when it is clearly much slower than the voltage carrier.
    voltage_candidate = edge_candidates["voltage"]
    current_candidate = edge_candidates["current"]
    if voltage_candidate is not None and contrasts["voltage"] >= 0.12:
        period_samples, aligned_window = voltage_candidate
        method = "voltage activity-envelope rising edges"
    elif current_candidate is not None and contrasts["current"] >= 0.12:
        candidate_period, candidate_window = current_candidate
        if candidate_period >= 4 * carrier_samples:
            period_samples, aligned_window = candidate_period, candidate_window
            activity_channel = "current"
            envelope = envelopes[activity_channel]
            method = "current activity-envelope rising edges"

    # If an envelope is visibly modulated but thresholding cannot identify clean
    # rising edges, use its autocorrelation as a less phase-specific fallback.
    edge = min(envelope_window, count // 10)
    core = envelope[edge : count - edge] if count - 2 * edge >= 16 else envelope
    raw_core = activity_signals[activity_channel][edge : count - edge] if count - 2 * edge >= 16 else activity_signals[activity_channel]
    raw_variation = float(np.std(raw_core))
    modulation_ratio = float(np.std(core)) / raw_variation if raw_variation > np.finfo(float).eps else 0.0
    if period_samples is None:
        activity_channel = "voltage" if contrasts["voltage"] >= contrasts["current"] else "current"
        envelope = envelopes[activity_channel]
        core = envelope[edge : count - edge] if count - 2 * edge >= 16 else envelope
        raw_core = activity_signals[activity_channel][edge : count - edge] if count - 2 * edge >= 16 else activity_signals[activity_channel]
        raw_variation = float(np.std(raw_core))
        modulation_ratio = float(np.std(core)) / raw_variation if raw_variation > np.finfo(float).eps else 0.0
    if period_samples is None and modulation_ratio >= 0.12 and len(core) >= 16:
        autocorrelation = normalized_autocorrelation(core)
        if autocorrelation is not None:
            min_lag = max(3, int(round(1.5 * carrier_samples)))
            max_lag = len(core) // 3
            candidates = local_maxima(autocorrelation)
            candidates = candidates[(candidates >= min_lag) & (candidates <= max_lag)]
            if len(candidates):
                best_correlation = float(np.max(autocorrelation[candidates]))
                credible = candidates[autocorrelation[candidates] >= max(0.15, 0.60 * best_correlation)]
                if len(credible):
                    candidate_period = int(np.min(credible))
                    if activity_channel == "voltage" or candidate_period >= 4 * carrier_samples:
                        period_samples = candidate_period
                        method = f"{activity_channel} activity-envelope autocorrelation"

    if period_samples is None:
        if carrier_frequency is not None:
            period_samples = max(3, int(round(1.0 / (carrier_frequency * dt_s))))
        else:
            # Last-resort duration split: still generate a useful zoom rather
            # than failing the complete analysis for a non-periodic capture.
            period_samples = max(3, count // 4)
            method = "record-duration fallback"

    # Align burst-mode plots to two adjacent activity rises where possible.
    full_window = 2 * period_samples
    start = max(0, (count - full_window) // 2)
    if aligned_window is not None:
        start, stop = aligned_window
    elif method.endswith("autocorrelation") and full_window <= count:
        peaks = well_spaced_peaks(envelope, max(1, int(round(0.60 * period_samples))))
        pairs = [
            (int(left), int(right))
            for left, right in zip(peaks[:-1], peaks[1:])
            if abs((right - left) - period_samples) <= 0.20 * period_samples
        ]
        if pairs:
            midpoint = 0.5 * (count - 1)
            left_peak, _ = min(pairs, key=lambda pair: abs(0.5 * (pair[0] + pair[1]) - midpoint))
            start = int(round(left_peak - 0.5 * period_samples))
            start = max(0, min(start, count - full_window))
        stop = min(count, start + full_window)
    else:
        stop = min(count, start + full_window)
    displayed = 2.0 if aligned_window is not None else (stop - start) / period_samples
    return DutyCycleSelection(
        start=start,
        stop=stop,
        period_samples=period_samples,
        period_s=period_samples * dt_s,
        frequency_Hz=1.0 / (period_samples * dt_s),
        method=method,
        activity_channel=activity_channel,
        cycles_displayed=float(displayed),
    )


def choose_smoothing_window(requested: int, samples_per_half_cycle: int) -> int:
    if requested < 0:
        raise AnalysisError("--smooth-window must be zero or a positive integer.")
    if requested:
        return requested if requested % 2 else requested + 1
    # About 3 percent of a half-cycle: enough to suppress ADC stair-steps while
    # retaining the voltage extrema used to delimit a half-cycle.
    return max(5, int(round(0.03 * samples_per_half_cycle)) | 1)


def robust_correlation(x: np.ndarray, y: np.ndarray) -> float | None:
    """Return a Pearson correlation after conservative paired MAD clipping."""
    mask = np.isfinite(x) & np.isfinite(y)
    x = np.asarray(x[mask], dtype=float)
    y = np.asarray(y[mask], dtype=float)
    if len(x) < 8:
        return None

    keep = np.ones(len(x), dtype=bool)
    for values in (x, y):
        median = float(np.median(values))
        scale = 1.4826 * float(np.median(np.abs(values - median)))
        if scale > np.finfo(float).eps:
            keep &= np.abs(values - median) <= 8.0 * scale
    if keep.sum() >= max(8, int(0.60 * len(x))):
        x = x[keep]
        y = y[keep]

    x = x - np.mean(x)
    y = y - np.mean(y)
    denominator = float(np.sqrt(np.dot(x, x) * np.dot(y, y)))
    if denominator <= np.finfo(float).eps:
        return None
    return float(np.dot(x, y) / denominator)


def resolve_current_polarity(
    requested_polarity: str,
    time_s: np.ndarray,
    voltage_DBD_V: np.ndarray,
    current_input_A: np.ndarray,
    monitor_voltage_V: np.ndarray,
    frequency_Hz: float | None = None,
) -> CurrentPolarityDecision:
    """Choose current sign from half-cycle charge conservation.

    Consecutive carrier-voltage zero crossings have nearly equal applied
    voltage.  Integrating the upstream Pearson current between them therefore
    cancels most probe/cable displacement charge.  The resulting charge should
    have the same sign as the simultaneous change in monitor-capacitor voltage.
    Correlations are checked separately for both crossing directions and over a
    small, physically plausible deskew range.  An ambiguous automatic result is
    rejected rather than silently flipped.
    """
    if requested_polarity not in {"auto", "1", "-1"}:
        raise AnalysisError("--current-polarity must be auto, 1, or -1.")
    manual_sign = None if requested_polarity == "auto" else int(requested_polarity)

    if frequency_Hz is None:
        frequency_Hz = estimate_fundamental_frequency(time_s, voltage_DBD_V)
    if frequency_Hz is None or frequency_Hz <= 0:
        if manual_sign is not None:
            return CurrentPolarityDecision(
                sign=manual_sign,
                requested=requested_polarity,
                method="manual override; automatic half-cycle diagnostic unavailable",
                half_cycle_charge_correlation_raw=None,
                usable_half_cycles=0,
                direction_sign_agreement=None,
                delay_sign_agreement=None,
                confidence="not evaluated",
            )
        raise AnalysisError(
            "Automatic current-polarity detection needs a resolvable carrier frequency; "
            "supply --frequency-hz or override with --current-polarity 1 or -1."
        )

    dt_s = float(np.median(np.diff(time_s)))
    if not np.isfinite(dt_s) or dt_s <= 0:
        raise AnalysisError("Automatic current-polarity detection requires increasing time values.")
    period_samples = max(4, int(round(1.0 / (frequency_Hz * dt_s))))
    half_period_s = 0.5 / frequency_Hz

    # Light, centered smoothing prevents ADC stair-steps from generating extra
    # zero crossings while preserving carrier timing.
    crossing_window = max(3, int(round(0.025 * period_samples)) | 1)
    crossing_window = min(11, crossing_window)
    voltage_smoothed = moving_average(voltage_DBD_V, crossing_window)
    voltage_center = 0.5 * (
        float(np.percentile(voltage_smoothed, 2.5))
        + float(np.percentile(voltage_smoothed, 97.5))
    )
    centered_voltage = voltage_smoothed - voltage_center
    left = centered_voltage[:-1]
    right = centered_voltage[1:]
    crossing_indices = np.flatnonzero(
        ((left <= 0.0) & (right > 0.0)) | ((left >= 0.0) & (right < 0.0))
    )

    if len(crossing_indices) < 12:
        diagnostic_error = "fewer than twelve carrier zero crossings were found"
        if manual_sign is not None:
            return CurrentPolarityDecision(
                sign=manual_sign,
                requested=requested_polarity,
                method=f"manual override; automatic diagnostic unavailable ({diagnostic_error})",
                half_cycle_charge_correlation_raw=None,
                usable_half_cycles=0,
                direction_sign_agreement=None,
                delay_sign_agreement=None,
                confidence="not evaluated",
            )
        raise AnalysisError(
            f"Automatic current-polarity detection is unavailable because {diagnostic_error}; "
            "use --current-polarity 1 or -1 after verifying the probe direction."
        )

    crossing_times: list[float] = []
    crossing_directions: list[int] = []
    for index in crossing_indices:
        denominator = centered_voltage[index + 1] - centered_voltage[index]
        fraction = 0.5 if denominator == 0 else -centered_voltage[index] / denominator
        fraction = float(np.clip(fraction, 0.0, 1.0))
        crossing_times.append(float(time_s[index] + fraction * (time_s[index + 1] - time_s[index])))
        crossing_directions.append(1 if denominator > 0 else -1)
    crossing_times_array = np.asarray(crossing_times, dtype=float)
    crossing_directions_array = np.asarray(crossing_directions, dtype=int)

    robust_voltage_peak = max(
        abs(float(np.percentile(centered_voltage, 1.0))),
        abs(float(np.percentile(centered_voltage, 99.0))),
    )
    minimum_peak = 0.15 * robust_voltage_peak
    usable_interval = np.zeros(len(crossing_times_array) - 1, dtype=bool)
    for interval, (start_index, stop_index) in enumerate(zip(crossing_indices[:-1], crossing_indices[1:])):
        duration = crossing_times_array[interval + 1] - crossing_times_array[interval]
        if not (0.60 * half_period_s <= duration <= 1.40 * half_period_s):
            continue
        segment = centered_voltage[start_index : stop_index + 2]
        if len(segment) >= 3 and float(np.max(np.abs(segment))) >= minimum_peak:
            usable_interval[interval] = True

    monitor_at_crossing = np.interp(crossing_times_array, time_s, monitor_voltage_V)
    monitor_charge_steps = np.diff(monitor_at_crossing)
    current_centered = current_input_A - float(np.mean(current_input_A))
    cumulative_current_charge = np.r_[
        0.0,
        np.cumsum(
            0.5
            * (current_centered[:-1] + current_centered[1:])
            * np.diff(time_s)
        ),
    ]

    max_lag_samples = min(5, max(1, int(round(0.05 * period_samples))))
    lag_samples = np.arange(-max_lag_samples, max_lag_samples + 1, dtype=int)
    lag_scores: list[float] = []
    central_score: float | None = None
    central_charge_steps: np.ndarray | None = None
    central_monitor_steps: np.ndarray | None = None
    central_directions: np.ndarray | None = None
    for lag in lag_samples:
        shifted_times = crossing_times_array + lag * dt_s
        in_range = (shifted_times[:-1] >= time_s[0]) & (shifted_times[1:] <= time_s[-1])
        selected = usable_interval & in_range
        if selected.sum() < 10:
            continue
        current_at_crossing = np.interp(shifted_times, time_s, cumulative_current_charge)
        current_charge_steps = np.diff(current_at_crossing)
        score = robust_correlation(current_charge_steps[selected], monitor_charge_steps[selected])
        if score is not None:
            lag_scores.append(score)
        if lag == 0:
            central_score = score
            central_charge_steps = current_charge_steps[selected]
            central_monitor_steps = monitor_charge_steps[selected]
            central_directions = crossing_directions_array[:-1][selected]

    if not lag_scores or central_score is None or central_charge_steps is None or central_directions is None:
        diagnostic_error = "too few active, complete carrier half-cycles remained"
        if manual_sign is not None:
            return CurrentPolarityDecision(
                sign=manual_sign,
                requested=requested_polarity,
                method=f"manual override; automatic diagnostic unavailable ({diagnostic_error})",
                half_cycle_charge_correlation_raw=None,
                usable_half_cycles=0,
                direction_sign_agreement=None,
                delay_sign_agreement=None,
                confidence="not evaluated",
            )
        raise AnalysisError(
            f"Automatic current-polarity detection is unavailable because {diagnostic_error}; "
            "use --current-polarity 1 or -1 after verifying the probe direction."
        )

    # PicoScope channels are sampled synchronously, so the acquired, zero-lag
    # relationship is the sign criterion.  The bounded lag scan is retained as
    # a confidence diagnostic rather than optimized: upstream displacement
    # current can otherwise manufacture either sign after even a small shift.
    correlation = float(central_score)
    inferred_sign = 1 if correlation >= 0 else -1
    delay_agreement = float(np.mean(np.sign(lag_scores) == inferred_sign))
    direction_scores: list[float] = []
    for direction in (-1, 1):
        selected_direction = central_directions == direction
        if selected_direction.sum() >= 8:
            score = robust_correlation(
                central_charge_steps[selected_direction],
                central_monitor_steps[selected_direction],
            )
            if score is not None:
                direction_scores.append(score)
    direction_agreement = (
        float(np.mean(np.sign(direction_scores) == inferred_sign))
        if direction_scores
        else None
    )
    usable_half_cycles = int(len(central_charge_steps))

    if (
        abs(correlation) >= 0.75
        and usable_half_cycles >= 20
        and delay_agreement >= 0.90
        and direction_agreement == 1.0
    ):
        confidence = "high"
    elif (
        abs(correlation) >= 0.40
        and usable_half_cycles >= 10
        and direction_agreement is not None
        and direction_agreement == 1.0
    ):
        confidence = "medium"
    else:
        confidence = "ambiguous"

    if manual_sign is not None:
        return CurrentPolarityDecision(
            sign=manual_sign,
            requested=requested_polarity,
            method="manual override with half-cycle charge diagnostic",
            half_cycle_charge_correlation_raw=correlation,
            usable_half_cycles=usable_half_cycles,
            direction_sign_agreement=direction_agreement,
            delay_sign_agreement=delay_agreement,
            confidence=confidence,
        )
    if confidence == "ambiguous":
        raise AnalysisError(
            "Automatic current-polarity detection is ambiguous "
            f"(raw half-cycle charge correlation {correlation:+.3f}, "
            f"{usable_half_cycles} usable half-cycles). Verify the Channel D polarity "
            "and rerun with --current-polarity 1 or -1."
        )
    return CurrentPolarityDecision(
        sign=inferred_sign,
        requested=requested_polarity,
        method="automatic half-cycle Pearson/monitor-capacitor charge consistency",
        half_cycle_charge_correlation_raw=correlation,
        usable_half_cycles=usable_half_cycles,
        direction_sign_agreement=direction_agreement,
        delay_sign_agreement=delay_agreement,
        confidence=confidence,
    )


def apparent_charge_equivalent_rates(
    time_s: np.ndarray,
    voltage_DBD_V: np.ndarray,
    monitor_voltage_V: np.ndarray,
    reference_capacitance_F: float,
    frequency_Hz: float | None = None,
    requested_charge_polarity: str = "auto",
) -> dict:
    """Estimate polarity-resolved terminal charge transfer at equal voltage.

    Each active carrier lobe is bounded by interpolated equal-voltage crossings.
    The ideal capacitive contribution therefore returns to its initial value,
    while hysteretic charge transfer appears as ``C_ref * delta(V_monitor)``.
    This is an area-integrated *apparent terminal* measurement: without a
    below-breakdown reference it can include dielectric loss, liquid conduction,
    polarization, and negative ions as well as plasma-delivered electrons.
    """
    if reference_capacitance_F <= 0:
        raise AnalysisError("Reference capacitance must be positive for charge-rate analysis.")
    if requested_charge_polarity not in {"auto", "1", "-1"}:
        raise AnalysisError("Charge polarity must be auto, 1, or -1.")
    if frequency_Hz is None:
        frequency_Hz = estimate_fundamental_frequency(time_s, voltage_DBD_V)
    if frequency_Hz is None or frequency_Hz <= 0:
        raise AnalysisError("Charge-rate analysis needs a resolvable carrier frequency.")

    dt_s = float(np.median(np.diff(time_s)))
    period_samples = max(4, int(round(1.0 / (frequency_Hz * dt_s))))
    half_period_s = 0.5 / frequency_Hz
    crossing_window = min(11, max(3, int(round(0.025 * period_samples)) | 1))
    voltage_smoothed = moving_average(voltage_DBD_V, crossing_window)
    voltage_center = 0.5 * (
        float(np.percentile(voltage_smoothed, 2.5))
        + float(np.percentile(voltage_smoothed, 97.5))
    )
    centered_voltage = voltage_smoothed - voltage_center
    left = centered_voltage[:-1]
    right = centered_voltage[1:]
    crossing_indices = np.flatnonzero(
        ((left <= 0.0) & (right > 0.0)) | ((left >= 0.0) & (right < 0.0))
    )
    if len(crossing_indices) < 12:
        raise AnalysisError("Too few carrier crossings were found for charge-rate analysis.")

    crossing_times: list[float] = []
    for index in crossing_indices:
        denominator = centered_voltage[index + 1] - centered_voltage[index]
        fraction = 0.5 if denominator == 0 else -centered_voltage[index] / denominator
        fraction = float(np.clip(fraction, 0.0, 1.0))
        crossing_times.append(float(time_s[index] + fraction * (time_s[index + 1] - time_s[index])))
    crossing_times_array = np.asarray(crossing_times, dtype=float)

    robust_voltage_peak = max(
        abs(float(np.percentile(centered_voltage, 1.0))),
        abs(float(np.percentile(centered_voltage, 99.0))),
    )
    if robust_voltage_peak <= np.finfo(float).eps:
        raise AnalysisError("Applied voltage has no resolvable carrier amplitude.")
    minimum_peak = 0.15 * robust_voltage_peak
    usable: list[bool] = []
    voltage_polarities: list[int] = []
    for interval, (start_index, stop_index) in enumerate(zip(crossing_indices[:-1], crossing_indices[1:])):
        duration = crossing_times_array[interval + 1] - crossing_times_array[interval]
        segment = centered_voltage[start_index : stop_index + 2]
        valid = (
            0.60 * half_period_s <= duration <= 1.40 * half_period_s
            and len(segment) >= 3
            and float(np.max(np.abs(segment))) >= minimum_peak
        )
        usable.append(valid)
        voltage_polarities.append(1 if float(np.mean(segment)) >= 0 else -1)

    usable_mask = np.asarray(usable, dtype=bool)
    if usable_mask.sum() < 10:
        raise AnalysisError("Too few active carrier half-cycles remained for charge-rate analysis.")
    voltage_polarity = np.asarray(voltage_polarities, dtype=int)[usable_mask]
    half_cycle_durations_s = np.diff(crossing_times_array)[usable_mask]
    monitor_at_crossing = np.interp(crossing_times_array, time_s, monitor_voltage_V)
    raw_charge_steps_C = reference_capacitance_F * np.diff(monitor_at_crossing)[usable_mask]

    # For a passive stack, the median V-polarity times delta(Q) is positive.
    # This is the lobe-integrated counterpart of requiring positive absorbed
    # Q-V loop energy and is more stable here than differentiating quantized Q.
    orientation_score_C = float(np.median(voltage_polarity * raw_charge_steps_C))
    if abs(orientation_score_C) <= np.finfo(float).eps:
        raise AnalysisError("Monitor-charge orientation is ambiguous in the active half-cycles.")
    automatically_inferred_polarity = 1 if orientation_score_C > 0 else -1
    monitor_charge_polarity = (
        automatically_inferred_polarity
        if requested_charge_polarity == "auto"
        else int(requested_charge_polarity)
    )
    charge_steps_C = monitor_charge_polarity * raw_charge_steps_C
    record_duration_s = float(time_s[-1] - time_s[0])

    def summarize_polarity(polarity: int) -> dict:
        selected = voltage_polarity == polarity
        directed_charge_C = polarity * charge_steps_C[selected]
        durations_s = half_cycle_durations_s[selected]
        matched = directed_charge_C > 0
        if matched.sum() < 4:
            raise AnalysisError(
                f"Too few charge-consistent {'positive' if polarity > 0 else 'negative'} half-cycles remained."
            )
        total_C = float(np.sum(directed_charge_C))
        if total_C <= 0:
            raise AnalysisError(
                f"Net {'positive' if polarity > 0 else 'negative'}-polarity charge is not resolved above zero."
            )
        half_cycle_rates_per_s = directed_charge_C / (ELEMENTARY_CHARGE_C * durations_s)
        return {
            "active_half_cycles": int(selected.sum()),
            "charge_direction_matched_half_cycles": int(matched.sum()),
            "charge_direction_match_fraction": float(np.mean(matched)),
            "gross_forward_direction_charge_C": float(np.sum(directed_charge_C[matched])),
            "reverse_direction_charge_magnitude_C": float(-np.sum(directed_charge_C[~matched])),
            "apparent_transferred_charge_total_C": total_C,
            "apparent_transferred_charge_total_nC": total_C * NC_PER_C,
            "charge_per_half_cycle_mean_C": float(np.mean(directed_charge_C)),
            "charge_per_half_cycle_mean_nC": float(np.mean(directed_charge_C) * NC_PER_C),
            "charge_per_half_cycle_median_C": float(np.median(directed_charge_C)),
            "charge_per_half_cycle_median_nC": float(np.median(directed_charge_C) * NC_PER_C),
            "charge_per_half_cycle_iqr_nC": float(
                (np.percentile(directed_charge_C, 75) - np.percentile(directed_charge_C, 25)) * NC_PER_C
            ),
            "charge_per_half_cycle_p95_C": float(np.percentile(directed_charge_C, 95)),
            "charge_per_half_cycle_p95_nC": float(np.percentile(directed_charge_C, 95) * NC_PER_C),
            "singly_charged_equivalents_per_half_cycle_median": float(
                np.median(directed_charge_C) / ELEMENTARY_CHARGE_C
            ),
            "singly_charged_equivalents_per_half_cycle_p95": float(
                np.percentile(directed_charge_C, 95) / ELEMENTARY_CHARGE_C
            ),
            "record_average_charge_rate_C_per_s": total_C / record_duration_s,
            "record_average_singly_charged_equivalent_rate_per_s": (
                total_C / (ELEMENTARY_CHARGE_C * record_duration_s)
            ),
            "half_cycle_average_equivalent_rate_p95_per_s": float(
                np.percentile(half_cycle_rates_per_s, 95)
            ),
        }

    negative = summarize_polarity(-1)
    positive = summarize_polarity(1)

    unique_monitor_values = np.unique(monitor_voltage_V)
    estimated_charge_code_C: float | None = None
    if 2 <= len(unique_monitor_values) <= 4096:
        positive_steps = np.diff(unique_monitor_values)
        positive_steps = positive_steps[positive_steps > np.finfo(float).eps]
        if len(positive_steps):
            estimated_charge_code_C = reference_capacitance_F * float(np.min(positive_steps))

    edge_samples = min(len(time_s) // 4, max(10, 2 * period_samples))
    quiet_limit = 0.05 * robust_voltage_peak
    start_quiet = float(np.percentile(np.abs(centered_voltage[:edge_samples]), 95)) <= quiet_limit
    end_quiet = float(np.percentile(np.abs(centered_voltage[-edge_samples:]), 95)) <= quiet_limit
    retained_charge_C: float | None = None
    retention_status: str
    if start_quiet and end_quiet:
        signed_monitor_charge_C = monitor_charge_polarity * reference_capacitance_F * monitor_voltage_V
        retained_charge_C = float(
            np.median(signed_monitor_charge_C[-edge_samples:])
            - np.median(signed_monitor_charge_C[:edge_samples])
        )
        if estimated_charge_code_C is not None and abs(retained_charge_C) < 2.0 * estimated_charge_code_C:
            retention_status = "Not resolved above two estimated monitor-ADC charge codes."
        else:
            retention_status = "Apparent terminal charge change between quiet near-zero-voltage edge windows."
    else:
        retention_status = (
            "Unavailable: the record does not begin and end with quiet near-zero-voltage windows; "
            "a post-burst decay capture is required."
        )

    return {
        "method": "equal-voltage carrier-half-cycle monitor-capacitor charge transfer",
        "record_duration_s": record_duration_s,
        "carrier_frequency_Hz": float(frequency_Hz),
        "monitor_charge_polarity_requested": requested_charge_polarity,
        "monitor_charge_polarity_applied": monitor_charge_polarity,
        "monitor_charge_polarity_automatically_inferred": automatically_inferred_polarity,
        "monitor_charge_orientation_score_C": orientation_score_C,
        "estimated_monitor_ADC_charge_code_C": estimated_charge_code_C,
        "estimated_monitor_ADC_charge_code_nC": (
            estimated_charge_code_C * NC_PER_C if estimated_charge_code_C is not None else None
        ),
        "negative_applied_voltage": negative,
        "positive_applied_voltage": positive,
        "apparent_polarity_charge_imbalance_C": (
            positive["apparent_transferred_charge_total_C"]
            - negative["apparent_transferred_charge_total_C"]
        ),
        "apparent_retained_terminal_charge_C": retained_charge_C,
        "apparent_retained_terminal_charge_nC": (
            retained_charge_C * NC_PER_C if retained_charge_C is not None else None
        ),
        "retention_status": retention_status,
        "equivalent_circuit_gas_gap_charge_correction_applied": False,
        "interpretation": (
            "Negative-polarity output is an apparent negative-charge-equivalent external-terminal rate "
            "(electrons plus any negative ions), not a species-resolved electron flux. "
            "Positive-polarity output is a positive-charge-equivalent terminal rate. "
            "Without a matched below-breakdown trace both include dielectric/liquid loss and polarization. "
            "No equivalent-circuit gas-gap charge correction has been applied, so these values must not "
            "be reported as plasma-to-surface particle rates."
        ),
    }


def resolve_charge_polarity(
    requested_polarity: str,
    time_s: np.ndarray,
    voltage_DBD_V: np.ndarray,
    unsigned_charge_C: np.ndarray,
) -> int:
    """Choose the monitor-probe sign without silently changing an explicit user choice.

    In a conventional capacitive measurement dQ/dt and dV/dt are positively
    correlated.  This criterion is intentionally only a sign convention; it is
    not used to decide whether a trace satisfies the classical DBD model.
    """
    if requested_polarity != "auto":
        return int(requested_polarity)
    window = max(5, min(101, (len(voltage_DBD_V) // 200) | 1))
    voltage_smoothed = moving_average(voltage_DBD_V, window)
    charge_smoothed = moving_average(unsigned_charge_C, window)
    d_voltage = np.gradient(voltage_smoothed, time_s)
    d_charge = np.gradient(charge_smoothed, time_s)
    correlation = float(np.dot(d_voltage, d_charge))
    return 1 if correlation >= 0 else -1


def find_turning_points(voltage_smoothed: np.ndarray, samples_per_half_cycle: int) -> list[int]:
    """Locate alternating maxima/minima with a voltage-amplitude hysteresis.

    This avoids treating digitizer stair-steps around an extremum as many short
    half-cycles.  The first and final partial cycles are intentionally omitted by
    the caller because only intervals between two turning points are used.
    """
    amplitude = float(np.percentile(voltage_smoothed, 97.5) - np.percentile(voltage_smoothed, 2.5))
    if amplitude <= 0:
        raise AnalysisError("Applied voltage has no measurable amplitude.")
    reversal = 0.04 * amplitude
    min_spacing = max(5, int(0.30 * samples_per_half_cycle))
    lookahead = min(len(voltage_smoothed) - 1, max(5, samples_per_half_cycle // 6))
    initial_change = voltage_smoothed[lookahead] - voltage_smoothed[0]
    rising = initial_change >= 0

    extrema: list[int] = []
    candidate_index = 0
    candidate_value = float(voltage_smoothed[0])

    for index in range(1, len(voltage_smoothed)):
        value = float(voltage_smoothed[index])
        if rising:
            if value >= candidate_value:
                candidate_index, candidate_value = index, value
            elif candidate_value - value >= reversal and index - candidate_index >= min_spacing:
                extrema.append(candidate_index)
                rising = False
                candidate_index, candidate_value = index, value
        else:
            if value <= candidate_value:
                candidate_index, candidate_value = index, value
            elif value - candidate_value >= reversal and index - candidate_index >= min_spacing:
                extrema.append(candidate_index)
                rising = True
                candidate_index, candidate_value = index, value

    # Enforce a physically plausible alternation.  This is primarily a safeguard
    # for files with a short acquisition or a non-sinusoidal source waveform.
    filtered: list[int] = []
    for index in extrema:
        if not filtered or index - filtered[-1] >= min_spacing:
            filtered.append(index)
    return filtered


def linear_fit(x_kV: np.ndarray, y_nC: np.ndarray, robust: bool = True) -> LineFit:
    """Fit y = mx + b, with conservative iterative MAD clipping of outliers."""
    mask = np.isfinite(x_kV) & np.isfinite(y_nC)
    x = x_kV[mask]
    y = y_nC[mask]
    if len(x) < 3 or np.ptp(x) <= np.finfo(float).eps:
        raise AnalysisError("A Q-V segment has too little voltage span for a capacitance fit.")

    included = np.ones(len(x), dtype=bool)
    for _ in range(5 if robust else 1):
        slope, intercept = np.polyfit(x[included], y[included], 1)
        residuals = y - (slope * x + intercept)
        if not robust:
            break
        residuals_included = residuals[included]
        median = float(np.median(residuals_included))
        scale = 1.4826 * float(np.median(np.abs(residuals_included - median)))
        if scale <= np.finfo(float).eps:
            break
        updated = np.abs(residuals - median) <= 4.5 * scale
        if updated.sum() < max(3, int(0.55 * len(x))):
            break
        if np.array_equal(updated, included):
            break
        included = updated

    slope, intercept = np.polyfit(x[included], y[included], 1)
    residuals = y[included] - (slope * x[included] + intercept)
    rms = float(np.sqrt(np.mean(residuals**2)))
    total = float(np.sum((y[included] - np.mean(y[included])) ** 2))
    r_squared = 1.0 if total <= np.finfo(float).eps else 1.0 - float(np.sum(residuals**2)) / total
    return LineFit(
        slope_nC_per_kV=float(slope),
        intercept_nC=float(intercept),
        r_squared=r_squared,
        rms_residual_nC=rms,
        n_points=int(included.sum()),
        x_min_kV=float(np.min(x[included])),
        x_max_kV=float(np.max(x[included])),
    )


def fit_two_segments(x_kV: np.ndarray, y_nC: np.ndarray) -> tuple[int, LineFit, LineFit]:
    """Find the least-squares breakpoint for two independent Q(V) lines.

    The waveform is monotonic in voltage over the supplied half-cycle.  Searching
    a time-ordered breakpoint avoids trying to infer a capacitance from noisy
    pointwise derivatives and preserves separate offsets for each Q-V side.
    """
    n = len(x_kV)
    min_points = max(20, int(round(0.12 * n)))
    if n < 2 * min_points + 2:
        raise AnalysisError("A detected half-cycle is too short for a two-segment fit.")

    prefix_n = np.arange(n + 1, dtype=float)
    prefix_x = np.r_[0.0, np.cumsum(x_kV)]
    prefix_y = np.r_[0.0, np.cumsum(y_nC)]
    prefix_xx = np.r_[0.0, np.cumsum(x_kV * x_kV)]
    prefix_yy = np.r_[0.0, np.cumsum(y_nC * y_nC)]
    prefix_xy = np.r_[0.0, np.cumsum(x_kV * y_nC)]

    def coefficients(left: np.ndarray, right: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        count = prefix_n[right] - prefix_n[left]
        sum_x = prefix_x[right] - prefix_x[left]
        sum_y = prefix_y[right] - prefix_y[left]
        sum_xx = prefix_xx[right] - prefix_xx[left]
        sum_yy = prefix_yy[right] - prefix_yy[left]
        sum_xy = prefix_xy[right] - prefix_xy[left]
        denominator = count * sum_xx - sum_x * sum_x
        valid = denominator > np.finfo(float).eps
        slope = np.full(len(count), np.nan)
        intercept = np.full(len(count), np.nan)
        slope[valid] = (count[valid] * sum_xy[valid] - sum_x[valid] * sum_y[valid]) / denominator[valid]
        intercept[valid] = (sum_y[valid] - slope[valid] * sum_x[valid]) / count[valid]
        sse = (
            sum_yy
            + slope * slope * sum_xx
            + count * intercept * intercept
            + 2.0 * slope * intercept * sum_x
            - 2.0 * slope * sum_xy
            - 2.0 * intercept * sum_y
        )
        return slope, intercept, sse

    candidates = np.arange(min_points, n - min_points + 1)
    _, _, sse_left = coefficients(np.zeros_like(candidates), candidates)
    _, _, sse_right = coefficients(candidates, np.full_like(candidates, n))
    objective = sse_left + sse_right
    if not np.isfinite(objective).any():
        raise AnalysisError("Could not find a valid piecewise Q-V fit for a half-cycle.")
    breakpoint = int(candidates[int(np.nanargmin(objective))])

    first = linear_fit(x_kV[:breakpoint], y_nC[:breakpoint])
    second = linear_fit(x_kV[breakpoint:], y_nC[breakpoint:])
    return breakpoint, first, second


def collect_half_cycle_fits(
    voltage_V: np.ndarray,
    charge_C: np.ndarray,
    extrema: Iterable[int],
) -> list[HalfCycleFit]:
    """Fit each complete monotonic half-cycle and identify the lower/off slope."""
    x = voltage_V * KV_PER_V
    y = charge_C * NC_PER_C
    extrema = list(extrema)
    lengths = np.diff(extrema)
    if len(lengths) < 4:
        raise AnalysisError("Fewer than four complete voltage half-cycles were detected.")
    median_length = float(np.median(lengths))
    fits: list[HalfCycleFit] = []

    for start, stop in zip(extrema[:-1], extrema[1:]):
        length = stop - start
        if length < 0.65 * median_length or length > 1.35 * median_length:
            continue
        # Include the endpoint to preserve the full voltage excursion.
        segment_x = x[start : stop + 1]
        segment_y = y[start : stop + 1]
        try:
            breakpoint_relative, first, second = fit_two_segments(segment_x, segment_y)
        except AnalysisError:
            continue
        if not (np.isfinite(first.slope_nC_per_kV) and np.isfinite(second.slope_nC_per_kV)):
            continue
        direction = "rising" if voltage_V[stop] > voltage_V[start] else "falling"
        off_is_first = first.slope_nC_per_kV <= second.slope_nC_per_kV
        fits.append(
            HalfCycleFit(
                start=start,
                breakpoint=start + breakpoint_relative,
                stop=stop,
                direction=direction,
                first=first,
                second=second,
                off_is_first=off_is_first,
            )
        )
    if len(fits) < 4:
        raise AnalysisError("Too few usable half-cycles remained after Q-V segmentation.")
    return fits


def segment_indices(fits: Iterable[HalfCycleFit], state: str, direction: str) -> np.ndarray:
    """Return raw waveform indices assigned to an off or on branch direction."""
    pieces: list[np.ndarray] = []
    for fit in fits:
        if fit.direction != direction:
            continue
        if state == "off":
            start, stop = (fit.start, fit.breakpoint) if fit.off_is_first else (fit.breakpoint, fit.stop)
        elif state == "on":
            start, stop = (fit.breakpoint, fit.stop) if fit.off_is_first else (fit.start, fit.breakpoint)
        else:
            raise ValueError(f"Unknown state: {state}")
        pieces.append(np.arange(start, stop + 1))
    return np.concatenate(pieces) if pieces else np.empty(0, dtype=int)


def fixed_slope_fit(x_kV: np.ndarray, y_nC: np.ndarray, slope_nC_per_kV: float) -> LineFit:
    """Robustly estimate a branch intercept with its capacitance slope fixed."""
    residual_offsets = y_nC - slope_nC_per_kV * x_kV
    intercept = float(np.median(residual_offsets))
    residuals = y_nC - (slope_nC_per_kV * x_kV + intercept)
    rms = float(np.sqrt(np.mean(residuals**2)))
    total = float(np.sum((y_nC - np.mean(y_nC)) ** 2))
    r_squared = 1.0 if total <= np.finfo(float).eps else 1.0 - float(np.sum(residuals**2)) / total
    return LineFit(
        slope_nC_per_kV=float(slope_nC_per_kV),
        intercept_nC=intercept,
        r_squared=r_squared,
        rms_residual_nC=rms,
        n_points=len(x_kV),
        x_min_kV=float(np.min(x_kV)),
        x_max_kV=float(np.max(x_kV)),
    )


def engineering(value: float | None, unit: str, scale: float) -> str:
    if value is None or not np.isfinite(value):
        return "not available"
    return f"{value / scale:.5g} {unit}"


def save_processed_csv(
    path: Path,
    time_ms: np.ndarray,
    voltage_input: np.ndarray,
    voltage_DBD_V: np.ndarray,
    current_input_A: np.ndarray,
    current_corrected_A: np.ndarray,
    monitor_voltage_V: np.ndarray,
    charge_C: np.ndarray,
) -> None:
    data = np.column_stack(
        (
            time_ms,
            voltage_input,
            voltage_DBD_V,
            current_input_A,
            current_corrected_A,
            monitor_voltage_V,
            charge_C,
            charge_C * NC_PER_C,
        )
    )
    header = (
        "time_ms,voltage_input_column2,voltage_DBD_V,current_input_A,current_corrected_A,"
        "monitor_capacitor_voltage_V,charge_C,charge_nC"
    )
    np.savetxt(path, data, delimiter=",", header=header, comments="", fmt="%.12g")


def plot_waveforms(
    path: Path,
    time_ms: np.ndarray,
    voltage_DBD_V: np.ndarray,
    current_A: np.ndarray,
    charge_C: np.ndarray,
    dpi: int,
    close: bool,
    title: str = "AC-DBD voltage, current, and monitor-capacitor charge",
) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(12, 8.5), sharex=True, constrained_layout=True)
    axes[0].plot(time_ms, voltage_DBD_V * KV_PER_V, color="#1f77b4", linewidth=0.85)
    axes[0].set_ylabel("Voltage (kV)")
    axes[0].set_title(title)

    axes[1].plot(time_ms, current_A, color="#d62728", linewidth=0.7)
    axes[1].set_ylabel("Current (A)")

    axes[2].plot(time_ms, charge_C * NC_PER_C, color="#2ca02c", linewidth=0.8)
    axes[2].set_xlabel("Time (ms)")
    axes[2].set_ylabel("Charge (nC)")

    for axis in axes:
        axis.grid(True, alpha=0.25)
        axis.margins(x=0)
    fig.savefig(path, dpi=dpi)
    if close:
        plt.close(fig)


def plot_lissajous(
    path: Path,
    voltage_DBD_V: np.ndarray,
    charge_C: np.ndarray,
    branch_fits: dict[str, dict[str, LineFit]],
    displacement_F: float | None,
    discharge_F: float | None,
    off_line_separation_C: float | None,
    charge_to_dielectric_C: float | None,
    model_fit_available: bool,
    max_points: int,
    dpi: int,
    close: bool,
) -> None:
    x_kV = voltage_DBD_V * KV_PER_V
    y_nC = charge_C * NC_PER_C
    count = min(len(x_kV), max(100, max_points))
    indices = np.linspace(0, len(x_kV) - 1, count, dtype=int)

    fig, axis = plt.subplots(figsize=(9.5, 7.2), constrained_layout=True)
    axis.plot(x_kV[indices], y_nC[indices], color="#34495e", linewidth=0.65, alpha=0.8, label="Measured Q-V trace")

    styles = {
        "off": ("#1b9e77", "--", "Plasma-off / displacement fit"),
        "on": ("#d95f02", "-.", "Plasma-on / discharge fit"),
    }
    for state, (color, linestyle, label) in styles.items():
        added_label = False
        for direction in ("rising", "falling"):
            fit = branch_fits.get(state, {}).get(direction)
            if fit is None:
                continue
            x_line = np.linspace(fit.x_min_kV, fit.x_max_kV, 200)
            y_line = fit.slope_nC_per_kV * x_line + fit.intercept_nC
            axis.plot(
                x_line,
                y_line,
                color=color,
                linestyle=linestyle,
                linewidth=2.0,
                label=label if not added_label else "_nolegend_",
            )
            added_label = True

    axis.set_title("DBD Q-V Lissajous figure with piecewise capacitance fits")
    axis.set_xlabel("DBD voltage (kV)")
    axis.set_ylabel("Monitor-capacitor charge (nC)")
    axis.grid(True, alpha=0.25)
    axis.legend(loc="best")

    annotation: list[str] = []
    if displacement_F is not None:
        annotation.append(f"C displacement = {displacement_F * PF_PER_F:.4g} pF")
    if discharge_F is not None:
        annotation.append(f"C discharge (effective) = {discharge_F * PF_PER_F:.4g} pF")
    if off_line_separation_C is not None:
        annotation.append(f"Off-line separation at 0 V = {off_line_separation_C * NC_PER_C:.4g} nC")
    if charge_to_dielectric_C is not None:
        annotation.append(f"Estimated dielectric charge / half-cycle = {charge_to_dielectric_C * NC_PER_C:.4g} nC")
    if not model_fit_available:
        annotation = ["Classical two-slope capacitance fit unavailable", "See the summary report before assigning capacitances."]
    if annotation:
        axis.text(
            0.02,
            0.98,
            "\n".join(annotation),
            transform=axis.transAxes,
            va="top",
            ha="left",
            fontsize=9,
            bbox={"facecolor": "white", "edgecolor": "0.7", "alpha": 0.9, "boxstyle": "round,pad=0.35"},
        )
    fig.savefig(path, dpi=dpi)
    if close:
        plt.close(fig)


def safe_float(value: float | None) -> float | None:
    return None if value is None or not np.isfinite(value) else float(value)


def run_analysis(args: argparse.Namespace) -> tuple[dict, str]:
    if args.reference_capacitance_uf <= 0:
        raise AnalysisError("--reference-capacitance-uf must be positive.")
    if args.voltage_scale == 0 or args.reference_voltage_scale == 0:
        raise AnalysisError("Voltage scale factors must be non-zero.")
    if args.max_plot_points < 100:
        raise AnalysisError("--max-plot-points must be at least 100.")
    if args.dielectric_capacitance_pf is not None and args.dielectric_capacitance_pf <= 0:
        raise AnalysisError("--dielectric-capacitance-pf must be positive.")

    input_path = args.input_csv.expanduser().resolve()
    time_ms, voltage_input, current_input_A, monitor_input = read_waveform_csv(input_path)
    time_s = time_ms * 1.0e-3
    monitor_voltage_V = monitor_input * args.reference_voltage_scale
    source_voltage_V = voltage_input * args.voltage_scale
    voltage_DBD_V = source_voltage_V.copy()
    if args.source_to_ground:
        voltage_DBD_V -= args.reference_polarity * monitor_voltage_V

    if args.frequency_hz is not None and args.frequency_hz <= 0:
        raise AnalysisError("--frequency-hz must be positive.")
    frequency_Hz = args.frequency_hz if args.frequency_hz is not None else estimate_fundamental_frequency(time_s, voltage_DBD_V)
    frequency_source = "user-supplied" if args.frequency_hz is not None else "dominant FFT component"

    reference_capacitance_F = args.reference_capacitance_uf * 1.0e-6
    unsigned_charge_C = reference_capacitance_F * monitor_voltage_V
    apparent_charge_rates: dict | None = None
    apparent_charge_rate_failure: str | None = None
    try:
        apparent_charge_rates = apparent_charge_equivalent_rates(
            time_s,
            voltage_DBD_V,
            monitor_voltage_V,
            reference_capacitance_F,
            frequency_Hz,
            args.charge_polarity,
        )
    except AnalysisError as error:
        apparent_charge_rate_failure = str(error)

    if apparent_charge_rates is not None:
        charge_polarity = int(apparent_charge_rates["monitor_charge_polarity_applied"])
        charge_polarity_method = "equal-voltage half-cycle passive charge transfer"
    else:
        charge_polarity = resolve_charge_polarity(args.charge_polarity, time_s, voltage_DBD_V, unsigned_charge_C)
        charge_polarity_method = "fallback smoothed capacitive derivative correlation"
    charge_C = charge_polarity * unsigned_charge_C

    # Resolve current against the physically signed monitor charge.  Treating
    # raw Channel D and current signs independently can make corrected I and Q
    # point in opposite directions when either probe channel was inverted.
    current_decision = resolve_current_polarity(
        args.current_polarity,
        time_s,
        voltage_DBD_V,
        current_input_A,
        charge_polarity * monitor_voltage_V,
        frequency_Hz,
    )
    current_A = current_decision.sign * current_input_A

    dt_s = float(np.median(np.diff(time_s)))
    if frequency_Hz is None:
        # The extrema finder only needs an approximate scale.  This fallback
        # remains usable for a few manually captured cycles.
        samples_per_half_cycle = max(30, len(time_s) // 12)
    else:
        samples_per_half_cycle = max(30, int(round(0.5 / (frequency_Hz * dt_s))))
    smoothing_window = choose_smoothing_window(args.smooth_window, samples_per_half_cycle)

    warnings: list[str] = []
    if apparent_charge_rate_failure is not None:
        warnings.append(f"Apparent polarity-resolved charge rates unavailable: {apparent_charge_rate_failure}")
    else:
        warnings.append(
            "Polarity-resolved carrier rates are apparent external-terminal charge-equivalent values, "
            "not electron/ion species measurements; matched below-breakdown traces and an "
            "equivalent-circuit capacitance correction are required for plasma-to-surface rates."
        )
    polarity_score = current_decision.half_cycle_charge_correlation_raw
    if (
        args.current_polarity != "auto"
        and polarity_score is not None
        and abs(polarity_score) >= 0.40
        and current_decision.sign * polarity_score < 0
    ):
        warnings.append(
            "The manual current-polarity override conflicts with the half-cycle "
            f"charge diagnostic (raw correlation {polarity_score:+.3f})."
        )
    if args.current_polarity == "auto" and current_decision.confidence == "medium":
        warnings.append(
            "Automatic current polarity has medium confidence; verify it with a "
            "known capacitive load or a below-breakdown calibration trace."
        )
    try:
        duty_cycle_selection = select_two_duty_cycles(time_s, voltage_DBD_V, current_A)
    except AnalysisError as error:
        # The normal CSV validation already assures this is very rare, but a
        # plot should still be produced if automatic cycle selection cannot run.
        fallback_period = max(3, len(time_s) // 4)
        fallback_stop = min(len(time_s), 2 * fallback_period)
        duty_cycle_selection = DutyCycleSelection(
            start=0,
            stop=fallback_stop,
            period_samples=fallback_period,
            period_s=fallback_period * dt_s,
            frequency_Hz=1.0 / (fallback_period * dt_s),
            method="record-duration emergency fallback",
            activity_channel="none",
            cycles_displayed=fallback_stop / fallback_period,
        )
        warnings.append(f"Automatic duty-cycle detection fell back to the record duration: {error}")

    # Always create the requested waveform and Q-V outputs.  The capacitance
    # model is deliberately allowed to fail independently: pulsed/ring-down,
    # inductive, or strongly non-stationary waveforms do not have the four
    # quasi-linear Q-V sides required by the conventional Lissajous method.
    model_fit_status = "not_available"
    model_fit_failure: str | None = None
    half_cycle_fits: list[HalfCycleFit] = []
    off_slopes = np.empty(0, dtype=float)
    on_slopes = np.empty(0, dtype=float)
    branch_fits: dict[str, dict[str, LineFit]] = {"off": {}, "on": {}}
    displacement_F: float | None = None
    discharge_F: float | None = None
    gap_capacitance_F: float | None = None
    relative_separation: float | None = None
    off_line_separation_C: float | None = None
    charge_to_dielectric_C: float | None = None
    charge_model_note = "Not available: a classical two-slope Lissajous fit was not established."

    try:
        voltage_smoothed = moving_average(voltage_DBD_V, smoothing_window)
        extrema = find_turning_points(voltage_smoothed, samples_per_half_cycle)
        half_cycle_fits = collect_half_cycle_fits(voltage_DBD_V, charge_C, extrema)
        off_slopes = np.asarray([fit.off.slope_nC_per_kV for fit in half_cycle_fits], dtype=float)
        on_slopes = np.asarray([fit.on.slope_nC_per_kV for fit in half_cycle_fits], dtype=float)
        if np.median(off_slopes) <= 0 or np.median(on_slopes) <= 0:
            raise AnalysisError(
                "The waveform does not produce two positive, quasi-linear Q-V slopes after monitor-polarity correction."
            )

        displacement_F = float(np.median(off_slopes) / PF_PER_F)
        discharge_F = float(np.median(on_slopes) / PF_PER_F)
        relative_separation = (discharge_F - displacement_F) / displacement_F
        if discharge_F <= displacement_F:
            raise AnalysisError(
                "The fitted discharge capacitance is not greater than the displacement capacitance."
            )
        if relative_separation < 0.05:
            warnings.append("The two fitted slopes differ by less than 5%; capacitances are weakly resolved.")

        x_kV = voltage_DBD_V * KV_PER_V
        y_nC = charge_C * NC_PER_C
        for state, common_slope in (("off", displacement_F * PF_PER_F), ("on", discharge_F * PF_PER_F)):
            for direction in ("rising", "falling"):
                indices = segment_indices(half_cycle_fits, state, direction)
                if len(indices) >= 10:
                    branch_fits[state][direction] = fixed_slope_fit(x_kV[indices], y_nC[indices], common_slope)

        off_rising = branch_fits["off"].get("rising")
        off_falling = branch_fits["off"].get("falling")
        if off_rising is None or off_falling is None:
            warnings.append("Could not obtain both plasma-off branch intercepts; dielectric charge is unavailable.")
            charge_model_note = "Not available because both plasma-off branch fits were not found."
        else:
            off_line_separation_C = abs(off_rising.intercept_nC - off_falling.intercept_nC) / NC_PER_C
            dielectric_model_F = (
                args.dielectric_capacitance_pf / PF_PER_F
                if args.dielectric_capacitance_pf is not None
                else discharge_F
            )
            denominator = 1.0 - displacement_F / dielectric_model_F
            if denominator <= 0:
                warnings.append("Dielectric-charge formula is undefined because C_diel is not greater than C_displacement.")
                charge_model_note = "Not available: C_diel must be greater than C_displacement."
            else:
                charge_to_dielectric_C = off_line_separation_C / denominator
                if args.dielectric_capacitance_pf is None:
                    charge_model_note = (
                        "Apparent value assuming the fitted discharge slope equals the physical dielectric capacitance "
                        "(fully bridged gap)."
                    )
                else:
                    charge_model_note = "Calculated with the supplied physical dielectric capacitance."

        gap_capacitance_F = displacement_F * discharge_F / (discharge_F - displacement_F)
        if len(half_cycle_fits) < 8:
            warnings.append("Fewer than eight usable half-cycles were fitted; use more steady-state cycles for stronger statistics.")
        off_r2 = [fit.r_squared for fit in branch_fits["off"].values()]
        on_r2 = [fit.r_squared for fit in branch_fits["on"].values()]
        if off_r2 and min(off_r2) < 0.90:
            warnings.append("At least one plasma-off fit has R^2 < 0.90; C_displacement is an effective fit.")
        if on_r2 and min(on_r2) < 0.90:
            warnings.append("At least one plasma-on fit has R^2 < 0.90; C_discharge is an effective fit.")
        model_fit_status = "available"
    except AnalysisError as error:
        model_fit_failure = str(error)
        warnings.append(
            "Classical Lissajous capacitance fit unavailable: "
            f"{error} Use a steady conventional AC waveform, or inspect this non-ideal trace before assigning capacitances."
        )
        half_cycle_fits = []
        off_slopes = np.empty(0, dtype=float)
        on_slopes = np.empty(0, dtype=float)
        branch_fits = {"off": {}, "on": {}}
        displacement_F = None
        discharge_F = None
        gap_capacitance_F = None
        relative_separation = None
        off_line_separation_C = None
        charge_to_dielectric_C = None

    default_output = input_path.parent / f"{input_path.stem}_analysis"
    output_dir = (args.output_dir.expanduser() if args.output_dir else default_output).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = input_path.stem
    processed_csv = output_dir / f"{stem}_processed.csv"
    waveform_png = output_dir / f"{stem}_waveforms.png"
    duty_cycle_waveform_png = output_dir / f"{stem}_waveforms_two_duty_cycles.png"
    lissajous_png = output_dir / f"{stem}_lissajous.png"
    summary_json = output_dir / f"{stem}_summary.json"
    summary_txt = output_dir / f"{stem}_summary.txt"

    save_processed_csv(
        processed_csv,
        time_ms,
        voltage_input,
        voltage_DBD_V,
        current_input_A,
        current_A,
        monitor_voltage_V,
        charge_C,
    )
    plot_waveforms(waveform_png, time_ms, voltage_DBD_V, current_A, charge_C, args.dpi, args.no_show)
    zoom = duty_cycle_selection
    if "fallback" in zoom.method:
        zoom_title = f"AC-DBD waveforms — {zoom.cycles_displayed:.3g} carrier-cycle zoom ({zoom.frequency_Hz:.5g} Hz)"
    else:
        zoom_title = f"AC-DBD waveforms — {zoom.cycles_displayed:.3g} detected duty cycles ({zoom.frequency_Hz:.5g} Hz)"
    plot_waveforms(
        duty_cycle_waveform_png,
        time_ms[zoom.start : zoom.stop],
        voltage_DBD_V[zoom.start : zoom.stop],
        current_A[zoom.start : zoom.stop],
        charge_C[zoom.start : zoom.stop],
        args.dpi,
        args.no_show,
        title=zoom_title,
    )
    plot_lissajous(
        lissajous_png,
        voltage_DBD_V,
        charge_C,
        branch_fits,
        displacement_F,
        discharge_F,
        off_line_separation_C,
        charge_to_dielectric_C,
        model_fit_status == "available",
        args.max_plot_points,
        args.dpi,
        args.no_show,
    )

    def serialize_branch_fit(fit: LineFit | None) -> dict | None:
        if fit is None:
            return None
        return {
            "capacitance_pF": safe_float(fit.capacitance_F * PF_PER_F),
            "intercept_nC": safe_float(fit.intercept_nC),
            "r_squared": safe_float(fit.r_squared),
            "rms_residual_nC": safe_float(fit.rms_residual_nC),
            "n_points": fit.n_points,
            "voltage_fit_range_kV": [safe_float(fit.x_min_kV), safe_float(fit.x_max_kV)],
        }

    summary = {
        "analysis_version": ANALYSIS_VERSION,
        "input_file": str(input_path),
        "output_directory": str(output_dir),
        "input_samples": int(len(time_ms)),
        "time_range_ms": [safe_float(float(time_ms[0])), safe_float(float(time_ms[-1]))],
        "analysis_frequency_Hz": safe_float(frequency_Hz),
        "analysis_frequency_source": frequency_source,
        "classical_lissajous_model_status": model_fit_status,
        "classical_lissajous_model_failure": model_fit_failure,
        "usable_half_cycles": int(len(half_cycle_fits)),
        "duty_cycle_zoom": {
            "detection_method": duty_cycle_selection.method,
            "activity_channel": duty_cycle_selection.activity_channel,
            "duty_cycle_frequency_Hz": safe_float(duty_cycle_selection.frequency_Hz),
            "duty_cycle_period_ms": safe_float(duty_cycle_selection.period_s * 1.0e3),
            "period_samples": duty_cycle_selection.period_samples,
            "window_time_ms": [
                safe_float(float(time_ms[duty_cycle_selection.start])),
                safe_float(float(time_ms[duty_cycle_selection.stop - 1])),
            ],
            "cycles_displayed": safe_float(duty_cycle_selection.cycles_displayed),
        },
        "assumptions": {
            "reference_capacitance_uF": safe_float(args.reference_capacitance_uf),
            "voltage_scale_V_per_input_unit": safe_float(args.voltage_scale),
            "reference_voltage_scale_V_per_input_unit": safe_float(args.reference_voltage_scale),
            "current_polarity_requested": args.current_polarity,
            "current_polarity_applied": current_decision.sign,
            "current_polarity_method": current_decision.method,
            "current_polarity_confidence": current_decision.confidence,
            "current_polarity_half_cycle_charge_correlation_raw": safe_float(
                current_decision.half_cycle_charge_correlation_raw
            ),
            "current_polarity_half_cycle_charge_correlation_corrected": safe_float(
                current_decision.sign * current_decision.half_cycle_charge_correlation_raw
                if current_decision.half_cycle_charge_correlation_raw is not None
                else None
            ),
            "current_polarity_usable_half_cycles": current_decision.usable_half_cycles,
            "current_polarity_direction_sign_agreement": safe_float(
                current_decision.direction_sign_agreement
            ),
            "current_polarity_delay_sign_agreement": safe_float(
                current_decision.delay_sign_agreement
            ),
            "charge_polarity_requested": args.charge_polarity,
            "charge_polarity_applied": charge_polarity,
            "charge_polarity_method": charge_polarity_method,
            "source_to_ground_voltage_column": bool(args.source_to_ground),
            "smoothing_window_samples": int(smoothing_window),
        },
        "surface_charge_transfer_apparent": {
            "status": "available" if apparent_charge_rates is not None else "not_available",
            "failure": apparent_charge_rate_failure,
            "results": apparent_charge_rates,
        },
        "capacitances": {
            "displacement_capacitance_F": safe_float(displacement_F),
            "displacement_capacitance_pF": safe_float(displacement_F * PF_PER_F if displacement_F is not None else None),
            "discharge_capacitance_effective_F": safe_float(discharge_F),
            "discharge_capacitance_effective_pF": safe_float(discharge_F * PF_PER_F if discharge_F is not None else None),
            "derived_gap_capacitance_F": safe_float(gap_capacitance_F),
            "derived_gap_capacitance_pF": safe_float(gap_capacitance_F * PF_PER_F if gap_capacitance_F is not None else None),
            "relative_slope_separation": safe_float(relative_separation),
            "cycle_to_cycle_displacement_std_pF": safe_float(float(np.std(off_slopes)) if len(off_slopes) else None),
            "cycle_to_cycle_discharge_std_pF": safe_float(float(np.std(on_slopes)) if len(on_slopes) else None),
        },
        "dielectric_charge": {
            "off_line_separation_C": safe_float(off_line_separation_C),
            "off_line_separation_nC": safe_float(off_line_separation_C * NC_PER_C if off_line_separation_C is not None else None),
            "charge_reaching_dielectric_per_half_cycle_C": safe_float(charge_to_dielectric_C),
            "charge_reaching_dielectric_per_half_cycle_nC": safe_float(charge_to_dielectric_C * NC_PER_C if charge_to_dielectric_C is not None else None),
            "absolute_charge_reaching_dielectric_per_cycle_C": safe_float(2.0 * charge_to_dielectric_C if charge_to_dielectric_C is not None else None),
            "absolute_charge_reaching_dielectric_per_cycle_nC": safe_float(2.0 * charge_to_dielectric_C * NC_PER_C if charge_to_dielectric_C is not None else None),
            "model_note": charge_model_note,
            "physical_dielectric_capacitance_supplied_pF": safe_float(args.dielectric_capacitance_pf),
        },
        "branch_fits": {
            state: {direction: serialize_branch_fit(fit) for direction, fit in directions.items()}
            for state, directions in branch_fits.items()
        },
        "outputs": {
            "processed_csv": str(processed_csv),
            "waveforms_png": str(waveform_png),
            "waveforms_two_duty_cycles_png": str(duty_cycle_waveform_png),
            "lissajous_png": str(lissajous_png),
            "summary_json": str(summary_json),
            "summary_text": str(summary_txt),
        },
        "warnings": warnings,
    }
    with summary_json.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, allow_nan=False)
        handle.write("\n")

    polarity_score_text = (
        f"{current_decision.half_cycle_charge_correlation_raw:+.4f}"
        if current_decision.half_cycle_charge_correlation_raw is not None
        else "not available"
    )
    polarity_agreement_text = (
        f"{100.0 * current_decision.direction_sign_agreement:.1f}%"
        if current_decision.direction_sign_agreement is not None
        else "not available"
    )
    if apparent_charge_rates is None:
        charge_rate_report_lines = [
            f"Apparent polarity-resolved charge rates: not available ({apparent_charge_rate_failure})"
        ]
    else:
        negative_rates = apparent_charge_rates["negative_applied_voltage"]
        positive_rates = apparent_charge_rates["positive_applied_voltage"]
        retained_nC = apparent_charge_rates["apparent_retained_terminal_charge_nC"]
        retained_text = f"{retained_nC:.5g} nC" if retained_nC is not None else "not available"
        charge_rate_report_lines = [
            "Apparent external-terminal equal-voltage half-cycle charge transfer "
            "(no below-breakdown subtraction or gas-gap correction):",
            (
                "  Negative pin, electron/negative-ion-equivalent: median "
                f"{negative_rates['charge_per_half_cycle_median_nC']:.5g} nC/half-cycle; "
                f"robust peak (p95) {negative_rates['charge_per_half_cycle_p95_nC']:.5g} nC/half-cycle "
                f"({negative_rates['singly_charged_equivalents_per_half_cycle_p95']:.5e} charge equivalents); "
                f"record-average {negative_rates['record_average_singly_charged_equivalent_rate_per_s']:.5e} s^-1; "
                f"robust peak half-cycle-average "
                f"{negative_rates['half_cycle_average_equivalent_rate_p95_per_s']:.5e} s^-1"
            ),
            (
                "  Positive pin, positive-charge-equivalent: median "
                f"{positive_rates['charge_per_half_cycle_median_nC']:.5g} nC/half-cycle; "
                f"robust peak (p95) {positive_rates['charge_per_half_cycle_p95_nC']:.5g} nC/half-cycle "
                f"({positive_rates['singly_charged_equivalents_per_half_cycle_p95']:.5e} charge equivalents); "
                f"record-average {positive_rates['record_average_singly_charged_equivalent_rate_per_s']:.5e} s^-1; "
                f"robust peak half-cycle-average "
                f"{positive_rates['half_cycle_average_equivalent_rate_p95_per_s']:.5e} s^-1"
            ),
            f"  Apparent retained terminal charge: {retained_text}; {apparent_charge_rates['retention_status']}",
        ]
    report_lines = [
        "AC-DBD Lissajous analysis",
        f"Input: {input_path}",
        f"Samples: {len(time_ms):,}; usable half-cycles: {len(half_cycle_fits)}",
        f"Analysis frequency ({frequency_source}): {engineering(frequency_Hz, 'Hz', 1.0)}",
        (
            f"Current polarity: requested {args.current_polarity}; applied multiplier "
            f"{current_decision.sign:+d}; {current_decision.confidence} confidence; "
            f"raw half-cycle charge correlation {polarity_score_text}; "
            f"direction agreement {polarity_agreement_text} over "
            f"{current_decision.usable_half_cycles} half-cycles"
        ),
        (
            f"Duty-cycle zoom ({duty_cycle_selection.method}, {duty_cycle_selection.activity_channel}): "
            f"{engineering(duty_cycle_selection.frequency_Hz, 'Hz', 1.0)}; "
            f"{duty_cycle_selection.cycles_displayed:.3g} cycles from "
            f"{time_ms[duty_cycle_selection.start]:.6g} to {time_ms[duty_cycle_selection.stop - 1]:.6g} ms"
        ),
        "",
        *charge_rate_report_lines,
        "",
        f"Displacement capacitance, C_cell (plasma off): {engineering(displacement_F, 'pF', 1.0 / PF_PER_F)}",
        f"Discharge capacitance, effective C_diel (plasma on): {engineering(discharge_F, 'pF', 1.0 / PF_PER_F)}",
        f"Derived gap capacitance (full-bridge model): {engineering(gap_capacitance_F, 'pF', 1.0 / PF_PER_F)}",
        f"Off-line separation at V=0: {engineering(off_line_separation_C, 'nC', 1.0 / NC_PER_C)}",
        f"Charge reaching dielectric per half-cycle: {engineering(charge_to_dielectric_C, 'nC', 1.0 / NC_PER_C)}",
        f"Absolute charge reaching dielectric per cycle: {engineering(2.0 * charge_to_dielectric_C if charge_to_dielectric_C is not None else None, 'nC', 1.0 / NC_PER_C)}",
        f"Charge model: {charge_model_note}",
        "",
        "Outputs:",
        f"  {processed_csv}",
        f"  {waveform_png}",
        f"  {duty_cycle_waveform_png}",
        f"  {lissajous_png}",
        f"  {summary_json}",
    ]
    if warnings:
        report_lines.extend(["", "Warnings:"] + [f"  - {warning}" for warning in warnings])
    report = "\n".join(report_lines) + "\n"
    summary_txt.write_text(report, encoding="utf-8")
    return summary, report


def main() -> int:
    args = parse_arguments()
    try:
        _, report = run_analysis(args)
    except AnalysisError as error:
        print(f"Analysis error: {error}", file=sys.stderr)
        return 2
    except OSError as error:
        print(f"File error: {error}", file=sys.stderr)
        return 2
    print(report)
    if not args.no_show:
        plt.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
