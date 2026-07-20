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

from dbd_surface_charge_figures import (
    build_capture_balanced_binned,
    build_dose_clock_rows,
    build_stationarity_metrics,
    plot_binned_facet,
    plot_dose_clock,
    plot_duty_audit,
    plot_power_audit,
)

from Lissajous_Figures import (
    AnalysisError as WaveformAnalysisError,
    duty_activity_mask,
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


ANALYSIS_VERSION = "2.0-surface-charge-power"
LEVELS = (40, 60, 75, 90, 100, 105, 115)
SCAN_FIT_LEVELS = (40, 60, 75)
TRANSITION_LEVELS = (90,)
HIGH_FIELD_LEVELS = (105, 115)
ACTIVE_CD_LEVELS = (100, 105, 115)
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
    background_cprime_basis_nC_per_pF: float = 0.0
    background_closs_basis_nC_per_pF: float = 0.0


@dataclass(frozen=True)
class BurstPeriodObservation:
    burst_index: int
    start_s: float
    stop_s: float
    midpoint_s: float
    signed_energy_uJ: float
    energy_uJ: float
    duty_on_fraction: float | None
    closure_delta_V_kV: float
    closure_delta_Q_nC: float
    closure_contribution_fraction: float | None


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
    complex_fit_residual_rms_nC: float | None = None
    complex_fit_signal_codes: float | None = None
    voltage_clipping_flag: bool = False
    current_clipping_flag: bool = False
    monitor_clipping_flag: bool = False
    duty_on_fraction: float | None = None
    duty_envelope_contrast: float | None = None
    burst_energy_uJ: list[float] = field(default_factory=list, repr=False)
    burst_periods: list[BurstPeriodObservation] = field(default_factory=list, repr=False)
    burst_energy_median_uJ: float | None = None
    apparent_power_mW: float | None = None


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
    background_status: str = "failed_empirical_background_validation"
    background_failed_gates: list[str] = field(default_factory=list)
    scan_cd_intercept_nC: float | None = None
    scan_cd_r_squared: float | None = None
    scan_cd_levels: str = ""
    scan_cd_pairwise_relative_span: float | None = None
    scan_cd_breakdown_active_fraction: float | None = None
    scan_cd_clean_counts: str = ""
    scan_cd_bootstrap_pF: np.ndarray = field(
        default_factory=lambda: np.asarray([], dtype=float), repr=False
    )
    geometry_only_factor: float | None = None
    geometry_only_factor_low: float | None = None
    geometry_only_factor_high: float | None = None
    scan_cd_charge_factor_status: str = "not_available"
    scan_based_factor_diagnostic: float | None = None
    scan_based_factor_diagnostic_low: float | None = None
    scan_based_factor_diagnostic_high: float | None = None


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
    parser.add_argument("--active-cd-min-r-squared", type=float, default=0.98)
    parser.add_argument("--active-cd-min-clean-captures", type=int, default=4)
    parser.add_argument("--active-cd-min-breakdown-active-fraction", type=float, default=0.50)
    parser.add_argument("--active-cd-max-pairwise-relative-span", type=float, default=0.35)
    parser.add_argument("--liquid-volume-ml", type=float, default=2.5)
    parser.add_argument("--metal-ion-concentration-mM", type=float, default=5.0)
    parser.add_argument("--dose-electrons-per-metal-ion", type=float, default=1.0)
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


def carrier_analytic_voltage(
    time_s: np.ndarray,
    voltage_V: np.ndarray,
    carrier_Hz: float,
    low_ratio: float = 0.90,
    high_ratio: float = 1.10,
) -> np.ndarray:
    """Return a narrow-band analytic carrier centred on the carrier line.

    The band is deliberately narrow: the ground-return transfer function has a
    series resonance just above the carrier (measured: C' swings to +61 pF at
    140 kHz and negative beyond 160 kHz), so any wide-band fit averages across
    a sign flip and collapses C'.  A narrow unmasked projection equals the
    carrier-line ratio Q(fc)/V(fc), with ~1e3 carrier cycles of processing
    gain against ADC quantization."""
    count = len(time_s)
    if count < 8 or carrier_Hz <= 0:
        return np.zeros(count, dtype=complex)
    dt = float(np.median(np.diff(time_s)))
    if not np.isfinite(dt) or dt <= 0:
        return np.zeros(count, dtype=complex)
    centered = np.asarray(voltage_V, dtype=float) - float(np.median(voltage_V))
    spectrum = np.fft.fft(centered)
    frequencies = np.fft.fftfreq(count, dt)
    keep = (frequencies >= low_ratio * carrier_Hz) & (
        frequencies <= high_ratio * carrier_Hz
    )
    analytic_spectrum = np.zeros(count, dtype=complex)
    analytic_spectrum[keep] = 2.0 * spectrum[keep]
    return np.fft.ifft(analytic_spectrum)


def complex_capacitance_least_squares(
    time_s: np.ndarray,
    voltage_V: np.ndarray,
    charge_C: np.ndarray,
    carrier_Hz: float,
) -> tuple[complex | None, np.ndarray, float | None, float | None]:
    """Fit ``q = C' v_carrier + C'' v90_carrier + offset + drift``.

    The complete passive waveform contributes to the fit, avoiding the
    zero-crossing endpoint ratios that can lock to one ADC code.  The returned
    convention is ``C* = C' - i C''``.
    """
    analytic = carrier_analytic_voltage(time_s, voltage_V, carrier_Hz)
    amplitude = np.abs(analytic)
    if len(amplitude) < 8 or not np.any(np.isfinite(amplitude)):
        return None, analytic, None, None
    scale = float(np.percentile(amplitude[np.isfinite(amplitude)], 90))
    if scale <= np.finfo(float).eps:
        return None, analytic, None, None
    mask = (
        np.isfinite(charge_C)
        & np.isfinite(analytic.real)
        & np.isfinite(analytic.imag)
        & (amplitude >= 0.10 * scale)
    )
    if int(np.sum(mask)) < 20:
        return None, analytic, None, None
    centered_time = time_s - float(np.mean(time_s[mask]))
    time_scale = max(float(np.ptp(time_s[mask])), np.finfo(float).eps)
    design = np.column_stack(
        (
            analytic.real[mask],
            analytic.imag[mask],
            np.ones(int(np.sum(mask))),
            centered_time[mask] / time_scale,
        )
    )
    coefficients, _, rank, _ = np.linalg.lstsq(design, charge_C[mask], rcond=None)
    if rank < 4 or not np.all(np.isfinite(coefficients[:2])):
        return None, analytic, None, None
    prediction = design @ coefficients
    residual_rms_nC = float(
        np.sqrt(np.mean((charge_C[mask] - prediction) ** 2)) * NC_PER_C
    )
    # Signal metric: raw monitor-charge swing, not the narrow-band prediction.
    # The narrow carrier-line estimator has large processing gain, so the
    # meaningful "is there signal" measure is the raw swing against the ADC
    # step, evaluated on the same samples the fit used.
    signal_pp_nC = float(
        (np.percentile(charge_C[mask], 99) - np.percentile(charge_C[mask], 1))
        * NC_PER_C
    )
    return (
        complex(float(coefficients[0]), -float(coefficients[1])),
        analytic,
        residual_rms_nC,
        signal_pp_nC,
    )


def passive_background_nC(
    lobe: LobeObservation,
    cprime_pF: float | np.ndarray,
    closs_pF: float | np.ndarray,
) -> float | np.ndarray:
    """Predict directed passive charge at one lobe from the complex model."""
    return (
        cprime_pF * lobe.background_cprime_basis_nC_per_pF
        + closs_pF * lobe.background_closs_basis_nC_per_pF
    )


def lissajous_burst_periods(
    time_s: np.ndarray,
    voltage_V: np.ndarray,
    charge_C: np.ndarray,
    windows: Sequence[slice],
    activity_mask: np.ndarray | None,
) -> list[BurstPeriodObservation]:
    """Calculate raw closed Q-V area and envelope on-fraction per burst period."""
    output: list[BurstPeriodObservation] = []
    for burst_index, window in enumerate(windows):
        t = np.asarray(time_s[window], dtype=float)
        v = np.asarray(voltage_V[window], dtype=float) * 1.0e-3
        q = np.asarray(charge_C[window], dtype=float) * NC_PER_C
        finite_mask = np.isfinite(t) & np.isfinite(v) & np.isfinite(q)
        if int(np.sum(finite_mask)) < 20:
            continue
        t, v, q = t[finite_mask], v[finite_mask], q[finite_mask]
        signed = 0.5 * float(np.sum(v * (np.roll(q, -1) - np.roll(q, 1))))
        energy = abs(signed)
        closure = 0.5 * float(v[-1] * q[0] - q[-1] * v[0])
        closure_fraction = abs(closure) / energy if energy > np.finfo(float).eps else None
        on_fraction = None
        if activity_mask is not None:
            active = np.asarray(activity_mask[window], dtype=float)[finite_mask]
            if len(active) >= 2 and t[-1] > t[0]:
                weights = np.diff(t)
                on_fraction = float(
                    np.sum(0.5 * (active[:-1] + active[1:]) * weights)
                    / (t[-1] - t[0])
                )
        output.append(
            BurstPeriodObservation(
                burst_index=burst_index,
                start_s=float(t[0]),
                stop_s=float(t[-1]),
                midpoint_s=0.5 * float(t[0] + t[-1]),
                signed_energy_uJ=signed,
                energy_uJ=energy,
                duty_on_fraction=on_fraction,
                closure_delta_V_kV=float(v[-1] - v[0]),
                closure_delta_Q_nC=float(q[-1] - q[0]),
                closure_contribution_fraction=closure_fraction,
            )
        )
    return output


def apparent_power_from_burst_energy_mW(
    energy_uJ: float, burst_frequency_Hz: float
) -> float:
    """Convert energy per burst period to average power in milliwatts."""
    return float(energy_uJ) * float(burst_frequency_Hz) / 1000.0


def _crossing_lobes(
    time_s: np.ndarray,
    voltage_V: np.ndarray,
    charge_C: np.ndarray,
    carrier_voltage: np.ndarray,
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
    carrier_inphase_at = np.interp(
        crossing_times_a, time_s, np.asarray(carrier_voltage).real
    )
    carrier_quadrature_at = np.interp(
        crossing_times_a, time_s, np.asarray(carrier_voltage).imag
    )
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
                background_cprime_basis_nC_per_pF=(
                    polarity
                    * float(carrier_inphase_at[j + 1] - carrier_inphase_at[j])
                    * 1.0e-3
                ),
                background_closs_basis_nC_per_pF=(
                    polarity
                    * float(carrier_quadrature_at[j + 1] - carrier_quadrature_at[j])
                    * 1.0e-3
                ),
            )
        )
    return result


def extract_lobes(
    time_s: np.ndarray,
    voltage_V: np.ndarray,
    charge_C: np.ndarray,
    carrier_Hz: float,
    burst_kHz: int,
    carrier_voltage: np.ndarray | None = None,
    windows: Sequence[slice] | None = None,
) -> list[LobeObservation]:
    if carrier_voltage is None:
        carrier_voltage = carrier_analytic_voltage(time_s, voltage_V, carrier_Hz)
    windows = list(windows) if windows is not None else burst_windows(
        time_s, voltage_V, burst_kHz, carrier_Hz
    )
    if not windows:
        windows = [slice(0, len(time_s))]
    result: list[LobeObservation] = []
    for burst_index, window in enumerate(windows):
        result.extend(
            _crossing_lobes(
                time_s[window],
                voltage_V[window],
                charge_C[window],
                carrier_voltage[window],
                carrier_Hz,
                burst_index,
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
    activity_mask: np.ndarray | None = None
    duty_fraction_hint: float | None = None
    duty_contrast: float | None = None
    try:
        duty = select_two_duty_cycles(
            time_s,
            voltage,
            waveform.current_input_A,
        )
        detected_burst = float(duty.frequency_Hz)
        detection_method = duty.method
        if (
            not np.isfinite(detected_burst)
            or detected_burst >= 0.5 * carrier
            or "fallback" in detection_method
        ):
            detected_burst = float(record.condition.burst_kHz * 1000.0)
            detection_method = "nominal_folder_frequency_after_carrier_fallback"
        else:
            activity_mask = duty_activity_mask(
                time_s, voltage, waveform.current_input_A, duty
            )
            duty_fraction_hint = duty.active_fraction
            duty_contrast = duty.envelope_contrast
    except WaveformAnalysisError:
        detected_burst = float(record.condition.burst_kHz * 1000.0)
        detection_method = "nominal_folder_frequency_after_detection_failure"
    nominal_burst = float(record.condition.burst_kHz * 1000.0)
    burst_error = abs(detected_burst - nominal_burst) / nominal_burst
    cstar, analytic_voltage, fit_residual_nC, fit_signal_pp_nC = (
        complex_capacitance_least_squares(time_s, voltage, charge, carrier)
    )
    if cstar is None:
        cstar = phasor_ratio(time_s, charge, voltage, carrier)
    used_burst_kHz = detected_burst * 1.0e-3
    windows = burst_windows(time_s, voltage, used_burst_kHz, carrier)
    lobes = extract_lobes(
        time_s,
        voltage,
        charge,
        carrier,
        used_burst_kHz,
        analytic_voltage,
        windows,
    )
    raw_x, raw_y = burst_extrema_features(time_s, voltage, charge, windows)
    periods = lissajous_burst_periods(
        time_s, voltage, charge, windows, activity_mask
    )
    period_energies = [period.energy_uJ for period in periods]
    period_duties = [
        period.duty_on_fraction
        for period in periods
        if period.duty_on_fraction is not None
    ]
    energy_median = median_or_none(period_energies)
    duty_on_fraction = median_or_none(period_duties)
    if duty_on_fraction is None:
        duty_on_fraction = duty_fraction_hint
    quiet_change, quiet_status = quiet_edge_charge(time_s, voltage, charge, carrier)
    qv_u, qv_q = representative_qv(
        time_s, voltage, charge, used_burst_kHz, carrier
    )
    charge_lsb_nC = estimate_charge_lsb_nC(
        waveform.monitor_voltage_V, capacitance_F
    )
    voltage_clipped = detect_clipping(waveform.source_voltage_V)
    current_clipped = detect_clipping(waveform.current_input_A)
    monitor_clipped = detect_clipping(waveform.monitor_voltage_V)
    apparent_power_mW = (
        apparent_power_from_burst_energy_mW(energy_median, detected_burst)
        if energy_median is not None and not voltage_clipped and not monitor_clipped
        else None
    )
    signal_codes = (
        fit_signal_pp_nC / charge_lsb_nC
        if fit_signal_pp_nC is not None
        and charge_lsb_nC is not None
        and charge_lsb_nC > 0
        else None
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
        charge_lsb_nC=charge_lsb_nC,
        clipping_flag=voltage_clipped or current_clipped or monitor_clipped,
        skipped_rows=waveform.skipped_rows,
        quiet_edge_change_raw_nC=quiet_change,
        quiet_edge_status=quiet_status,
        qv_voltage_kV=qv_u,
        qv_charge_raw_nC=qv_q,
        complex_fit_residual_rms_nC=fit_residual_nC,
        complex_fit_signal_codes=signal_codes,
        voltage_clipping_flag=voltage_clipped,
        current_clipping_flag=current_clipped,
        monitor_clipping_flag=monitor_clipped,
        duty_on_fraction=duty_on_fraction,
        duty_envelope_contrast=duty_contrast,
        burst_energy_uJ=period_energies,
        burst_periods=periods,
        burst_energy_median_uJ=energy_median,
        apparent_power_mW=apparent_power_mW,
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


def active_cd_three_level_fit(
    observations: Sequence[FileObservation],
    sign: int,
    cprime_pF: float | None,
    cprime_draws: np.ndarray,
    closs_pF: float | None,
    passive_threshold_nC: dict[int, float],
    args: argparse.Namespace,
    rng: np.random.Generator,
) -> dict:
    """Fit an effective active-branch ``Cd`` from 100/105/115 % medians.

    The three commanded levels are independent amplitude settings, while the
    many files at each setting are technical repeats.  Consequently the
    regression is performed on three capture medians and the bootstrap
    resamples captures *within* level.  The result remains provisional: three
    levels can test curvature and consistency, but do not replace a denser
    active-amplitude scan or a direct dielectric-capacitance measurement.
    """
    groups: dict[int, list[tuple[FileObservation, float, float]]] = defaultdict(list)
    raw_counts: dict[int, int] = {}
    for level in ACTIVE_CD_LEVELS:
        candidates: list[tuple[FileObservation, float, float]] = []
        for row in observations:
            if row.level_label != str(level):
                continue
            if row.raw_extrema_x_kV is None or row.raw_extrema_y_nC is None:
                continue
            if row.voltage_clipping_flag or row.monitor_clipping_flag:
                continue
            x = sign * float(row.raw_extrema_x_kV)
            y = float(row.raw_extrema_y_nC)
            if np.isfinite(x) and np.isfinite(y) and x > 0.02 and y > 0:
                candidates.append((row, x, y))
        raw_counts[level] = len(candidates)
        if candidates:
            x_center = float(np.median([item[1] for item in candidates]))
            y_center = float(np.median([item[2] for item in candidates]))
            # Remove collapsed/pathological captures without trimming the
            # ordinary repeat distribution.  This catches the known BMIM 115
            # % capture whose extrema are close to zero on both axes.
            candidates = [
                item for item in candidates
                if 0.25 * x_center <= item[1] <= 4.0 * x_center
                and 0.25 * y_center <= item[2] <= 4.0 * y_center
            ]
        groups[level] = candidates

    clean_counts = {level: len(groups[level]) for level in ACTIVE_CD_LEVELS}
    centers: dict[int, tuple[float, float]] = {}
    for level in ACTIVE_CD_LEVELS:
        if groups[level]:
            centers[level] = (
                float(np.median([item[1] for item in groups[level]])),
                float(np.median([item[2] for item in groups[level]])),
            )

    slope = intercept = r_squared = None
    pairwise: list[float] = []
    pairwise_span = None
    monotonic_x = monotonic_y = False
    if len(centers) == len(ACTIVE_CD_LEVELS):
        x = np.asarray([centers[level][0] for level in ACTIVE_CD_LEVELS])
        y = np.asarray([centers[level][1] for level in ACTIVE_CD_LEVELS])
        monotonic_x = bool(np.all(np.diff(x) > 0.02))
        monotonic_y = bool(np.all(np.diff(y) > 0.0))
        slope, intercept, r_squared = linear_fit(x, y)
        for left, right in zip(ACTIVE_CD_LEVELS[:-1], ACTIVE_CD_LEVELS[1:]):
            dx = centers[right][0] - centers[left][0]
            if abs(dx) > 0.02:
                pairwise.append((centers[right][1] - centers[left][1]) / dx)
        dx = centers[115][0] - centers[100][0]
        if abs(dx) > 0.02:
            pairwise.append((centers[115][1] - centers[100][1]) / dx)
        if len(pairwise) >= 2:
            denominator = abs(float(np.median(pairwise)))
            if denominator > np.finfo(float).eps:
                pairwise_span = float((max(pairwise) - min(pairwise)) / denominator)

    breakdown_active: list[bool] = []
    if cprime_pF is not None and closs_pF is not None:
        for row, _, _ in groups[100]:
            resolved = False
            for lobe in row.lobes:
                background = passive_background_nC(lobe, cprime_pF, closs_pF)
                excess = sign * lobe.raw_directed_charge_nC - float(background)
                if excess > passive_threshold_nC.get(lobe.voltage_polarity, 0.0):
                    resolved = True
                    break
            breakdown_active.append(resolved)
    breakdown_active_fraction = (
        float(np.mean(breakdown_active)) if breakdown_active else None
    )

    bootstrap_slopes = np.full(args.bootstrap_replicates, np.nan)
    if all(clean_counts[level] >= 2 for level in ACTIVE_CD_LEVELS):
        sampled_centers: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        for level in ACTIVE_CD_LEVELS:
            values = np.asarray([(item[1], item[2]) for item in groups[level]], dtype=float)
            indices = moving_block_index_matrix(
                len(values),
                args.bootstrap_replicates,
                args.bootstrap_block_files,
                rng,
            )
            sampled_centers[level] = (
                np.median(values[:, 0][indices], axis=1),
                np.median(values[:, 1][indices], axis=1),
            )
        for iteration in range(args.bootstrap_replicates):
            bx = np.asarray(
                [sampled_centers[level][0][iteration] for level in ACTIVE_CD_LEVELS]
            )
            by = np.asarray(
                [sampled_centers[level][1][iteration] for level in ACTIVE_CD_LEVELS]
            )
            if np.all(np.diff(bx) > 0.02):
                fitted, _, _ = linear_fit(bx, by)
                if fitted is not None:
                    bootstrap_slopes[iteration] = fitted
    finite_slopes = bootstrap_slopes[np.isfinite(bootstrap_slopes)]
    low, high = percentile_interval(finite_slopes)
    if cprime_pF is None:
        physical_fraction = 0.0
    elif len(finite_slopes):
        if len(cprime_draws) == len(bootstrap_slopes):
            valid = np.isfinite(bootstrap_slopes) & np.isfinite(cprime_draws)
            physical_fraction = float(
                np.mean(bootstrap_slopes[valid] > 1.05 * cprime_draws[valid])
            ) if np.any(valid) else 0.0
        else:
            physical_fraction = float(np.mean(finite_slopes > 1.05 * cprime_pF))
    else:
        physical_fraction = 0.0

    gates: list[str] = []
    for level in ACTIVE_CD_LEVELS:
        if clean_counts[level] < args.active_cd_min_clean_captures:
            gates.append(f"fewer_than_{args.active_cd_min_clean_captures}_clean_captures_at_{level}_percent")
    if not monotonic_x:
        gates.append("active_voltage_not_monotonic")
    if not monotonic_y:
        gates.append("active_charge_not_monotonic")
    if slope is None or not np.isfinite(slope) or slope <= 0:
        gates.append("missing_or_nonpositive_three_level_slope")
    if r_squared is None or r_squared < args.active_cd_min_r_squared:
        gates.append("three_level_r_squared_below_threshold")
    if pairwise_span is None or pairwise_span > args.active_cd_max_pairwise_relative_span:
        gates.append("pairwise_active_slopes_inconsistent")
    if (
        breakdown_active_fraction is None
        or breakdown_active_fraction < args.active_cd_min_breakdown_active_fraction
    ):
        gates.append("breakdown_level_not_resolved_as_active")
    if physical_fraction < 0.95:
        gates.append("Cd_bootstrap_physical_fraction_below_0.95")

    status = (
        "supported_provisional_three_level_effective_Cd"
        if not gates
        else "rejected_three_level_effective_Cd"
    )
    return {
        "slope_pF": slope,
        "intercept_nC": intercept,
        "r_squared": r_squared,
        "ci_low_pF": low,
        "ci_high_pF": high,
        "physical_fraction": physical_fraction,
        "pairwise_relative_span": pairwise_span,
        "breakdown_active_fraction": breakdown_active_fraction,
        "levels": ";".join(
            f"{level}:{centers[level][0]:.9g}kV,{centers[level][1]:.9g}nC"
            for level in ACTIVE_CD_LEVELS if level in centers
        ),
        "clean_counts": ";".join(
            f"{level}:{clean_counts[level]}/{raw_counts[level]}" for level in ACTIVE_CD_LEVELS
        ),
        "bootstrap_pF": bootstrap_slopes,
        "failed_gates": gates,
        "status": status,
    }


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
    passive_all = [
        row for row in observations
        if row.level_label in {str(level) for level in PASSIVE_FIT_LEVELS}
    ]
    passive_rows = [
        row for row in passive_all
        if not row.voltage_clipping_flag and not row.monitor_clipping_flag
    ]
    passive_complete = all(
        any(row.level_label == str(level) for row in passive_rows)
        for level in PASSIVE_FIT_LEVELS
    )

    cprime_by_level: list[float] = []
    closs_by_level: list[float] = []
    cprime_files: list[float] = []
    closs_files: list[float] = []
    cprime_parts: list[np.ndarray] = []
    closs_parts: list[np.ndarray] = []
    for level in PASSIVE_FIT_LEVELS:
        level_rows = [
            row for row in passive_rows
            if row.level_label == str(level) and row.cstar_raw_F is not None
        ]
        qualified = [
            row for row in level_rows
            if (row.complex_fit_signal_codes or 0.0) >= 8.0
        ]
        rows_for_cstar = qualified if qualified else level_rows
        values = [sign * row.cstar_raw_F for row in rows_for_cstar]
        level_cprime = [value.real * 1.0e12 for value in values]
        level_closs = [-value.imag * 1.0e12 for value in values]
        cprime_by_level.append(median_or_none(level_cprime) or np.nan)
        closs_by_level.append(median_or_none(level_closs) or np.nan)
        cprime_files.extend(level_cprime)
        closs_files.extend(level_closs)
        if values:
            indices = moving_block_index_matrix(
                len(values),
                args.bootstrap_replicates,
                args.bootstrap_block_files,
                rng,
            )
            cprime_parts.append(np.asarray(level_cprime, dtype=float)[indices])
            closs_parts.append(np.asarray(level_closs, dtype=float)[indices])

    cprime_pF = median_or_none(cprime_files)
    closs_pF = median_or_none(closs_files)
    cprime_draws = (
        np.nanmedian(np.concatenate(cprime_parts, axis=1), axis=1)
        if cprime_parts else np.asarray([], dtype=float)
    )
    closs_draws = (
        np.nanmedian(np.concatenate(closs_parts, axis=1), axis=1)
        if closs_parts else np.asarray([], dtype=float)
    )
    cprime_low, cprime_high = percentile_interval(cprime_draws)
    closs_low, closs_high = percentile_interval(closs_draws)
    tan_delta = (
        closs_pF / cprime_pF
        if cprime_pF is not None and closs_pF is not None
        and abs(cprime_pF) > np.finfo(float).eps
        else None
    )
    span = relative_span(cprime_by_level)

    # Retain the old amplitude-ratio slopes as explicitly deprecated
    # diagnostics.  They are no longer used for subtraction or uncertainty;
    # this prevents few-code controls from collapsing onto an ADC grid ratio.
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
        if cprime_pF is not None and closs_pF is not None:
            for row in passive_rows:
                for lobe in row.lobes:
                    if lobe.voltage_polarity == polarity:
                        residuals.append(
                            sign * lobe.raw_directed_charge_nC
                            - float(passive_background_nC(lobe, cprime_pF, closs_pF))
                        )
        residual_array = finite(residuals)
        threshold = (
            float(np.percentile(residual_array, 95)) if len(residual_array) else 0.0
        )
        if charge_lsb is not None:
            threshold = max(threshold, charge_lsb)
        thresholds[polarity] = max(0.0, threshold)

    background_failed: list[str] = []
    if not passive_complete:
        background_failed.append("missing_unclipped_40_60_or_75_percent_level")
    if sign_agreement < args.minimum_sign_agreement:
        background_failed.append("charge_polarity_vote_ambiguous")
    if cprime_pF is None or closs_pF is None:
        background_failed.append("complex_background_coefficients_unavailable")
    available_fraction = (
        len(cprime_files) / len(passive_rows) if passive_rows else 0.0
    )
    if available_fraction < 0.75:
        background_failed.append("complex_fit_available_for_fewer_than_75_percent_passive_captures")
    signal_codes = median_or_none(row.complex_fit_signal_codes for row in passive_rows)
    qualifying = [
        row for row in passive_rows
        if (row.complex_fit_signal_codes or 0.0) >= 8.0
    ]
    if len(qualifying) < 8:
        background_failed.append(
            "fewer_than_8_passive_captures_with_carrier_signal_at_or_above_8_ADC_codes"
        )
    burst_error = median_or_none(row.burst_frequency_relative_error for row in passive_rows)
    if burst_error is None or burst_error > 0.15:
        background_failed.append("detected_burst_frequency_disagrees_with_condition_label")

    # The unused 90 % data are an empirical prediction audit, not another fit
    # level.  A grossly larger residual than both the training null and three
    # ADC steps rejects transfer of the complex background to breakdown.
    holdout_rows = [
        row for row in observations
        if row.level_label == "90"
        and not row.voltage_clipping_flag and not row.monitor_clipping_flag
    ]
    if holdout_rows and cprime_pF is not None and closs_pF is not None:
        for polarity in (-1, 1):
            per_capture: list[float] = []
            for row in holdout_rows:
                residuals = [
                    sign * lobe.raw_directed_charge_nC
                    - float(passive_background_nC(lobe, cprime_pF, closs_pF))
                    for lobe in row.lobes if lobe.voltage_polarity == polarity
                ]
                if residuals:
                    per_capture.append(float(np.percentile(residuals, 95)))
            if per_capture:
                allowed = max(
                    3.0 * thresholds[polarity],
                    thresholds[polarity] + 3.0 * float(charge_lsb or 0.0),
                )
                if float(np.median(per_capture)) > allowed:
                    name = "negative_voltage" if polarity < 0 else "positive_voltage"
                    background_failed.append(f"90_percent_holdout_background_failure_{name}")
    background_status = (
        "supported_whole_waveform_complex_background"
        if not background_failed
        else "failed_empirical_complex_background_validation"
    )

    physical_failed = list(background_failed)
    if cprime_pF is None or cprime_pF <= 0:
        physical_failed.append("nonpositive_Cprime")
    if closs_pF is None or closs_pF < 0:
        physical_failed.append("nonpassive_loss_orientation")
    if (
        cprime_pF is not None and closs_pF is not None
        and math.degrees(
            math.atan2(max(0.0, closs_pF), max(np.finfo(float).eps, cprime_pF))
        ) > 45.0
    ):
        physical_failed.append("passive_phase_exceeds_45_degrees")
    if span is None or span > 0.30:
        physical_failed.append("Cprime_not_stable_across_passive_levels")
    passive_status = (
        "supported_effective_complex_Ccell_at_carrier"
        if not physical_failed else "failed_physical_Ccell_validation"
    )

    active_fit = active_cd_three_level_fit(
        observations,
        sign,
        cprime_pF,
        cprime_draws,
        closs_pF,
        thresholds,
        args,
        rng,
    )
    scan_gates = list(active_fit["failed_gates"])
    # The active Q–V slope can support a provisional effective Cd even when
    # the separate passive Ccell measurement is too quantized to support the
    # correction factor.  Keep those two decisions explicit.
    scan_status = active_fit["status"]
    scan_factor_status = (
        "supported_provisional_Cd_over_Cd_minus_Ccell_factor"
        if scan_status.startswith("supported") and passive_status.startswith("supported")
        else "not_usable_physical_Ccell_not_supported"
        if scan_status.startswith("supported")
        else "not_usable_active_Cd_fit_rejected"
    )

    geometry_cd = None
    geometry_only_factor = None
    geometry_only_low = None
    geometry_only_high = None
    central_factor = None
    central_low = None
    central_high = None
    scan_factor_diagnostic = None
    scan_factor_diagnostic_low = None
    scan_factor_diagnostic_high = None
    factor_source = "none"
    evidence = "background_subtracted_terminal_excess"
    if (
        passive_status.startswith("supported")
        and condition.material in CONDUCTIVE_LIQUIDS
        and cprime_pF is not None and cprime_pF > 0
    ):
        geometry_cd = geometry_cd_pF(
            args.beaker_diameter_cm,
            args.glass_thickness_mm,
            0.5 * (args.pyrex_epsilon_min + args.pyrex_epsilon_max),
        )
        if geometry_cd > 1.05 * cprime_pF:
            geometry_only_factor = geometry_cd / (geometry_cd - cprime_pF)
            _, geometry_samples = geometry_factor_samples(
                cprime_pF, args, args.bootstrap_replicates, rng
            )
            geometry_only_low, geometry_only_high = percentile_interval(geometry_samples)

    scan_cd = active_fit["slope_pF"]
    if (
        scan_status.startswith("supported")
        and scan_cd is not None and cprime_pF is not None
        and scan_cd > 1.05 * cprime_pF
    ):
        scan_factor_diagnostic = scan_cd / (scan_cd - cprime_pF)
        scan_draws = np.asarray(active_fit["bootstrap_pF"], dtype=float)
        if len(scan_draws) == len(cprime_draws):
            denominator = scan_draws - cprime_draws
            factor_draws = np.where(
                denominator > 0.05 * scan_draws,
                scan_draws / denominator,
                np.nan,
            )
            scan_factor_diagnostic_low, scan_factor_diagnostic_high = percentile_interval(
                factor_draws
            )
        if passive_status.startswith("supported"):
            central_factor = scan_factor_diagnostic
            central_low = scan_factor_diagnostic_low
            central_high = scan_factor_diagnostic_high
            factor_source = "provisional_three_level_active_scan"
            evidence = "supported_provisional_model_dependent"
    if central_factor is None and geometry_only_factor is not None:
        central_factor = geometry_only_factor
        central_low, central_high = geometry_only_low, geometry_only_high
        factor_source = "full_base_pyrex_geometry_scenario"
        evidence = "exploratory_model_dependent"
    if not background_status.startswith("supported"):
        evidence = "diagnostic_complex_background_model_rejected"

    factor_gates = (
        ["charge_factor:physical_Ccell_not_supported"]
        if scan_status.startswith("supported") and not passive_status.startswith("supported")
        else []
    )
    failed = list(dict.fromkeys(
        physical_failed
        + [f"active_Cd:{gate}" for gate in scan_gates]
        + factor_gates
    ))
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
        scan_cd_ci_low_pF=active_fit["ci_low_pF"],
        scan_cd_ci_high_pF=active_fit["ci_high_pF"],
        scan_cd_physical_fraction=active_fit["physical_fraction"],
        scan_cd_status=scan_status,
        geometry_cd_pF=geometry_cd,
        geometry_factor=central_factor,
        geometry_factor_low=central_low,
        geometry_factor_high=central_high,
        factor_source=factor_source,
        evidence_tier=evidence,
        failed_gates=failed,
        background_status=background_status,
        background_failed_gates=background_failed,
        scan_cd_intercept_nC=active_fit["intercept_nC"],
        scan_cd_r_squared=active_fit["r_squared"],
        scan_cd_levels=active_fit["levels"],
        scan_cd_pairwise_relative_span=active_fit["pairwise_relative_span"],
        scan_cd_breakdown_active_fraction=active_fit["breakdown_active_fraction"],
        scan_cd_clean_counts=active_fit["clean_counts"],
        scan_cd_bootstrap_pF=np.asarray(active_fit["bootstrap_pF"], dtype=float),
        geometry_only_factor=geometry_only_factor,
        geometry_only_factor_low=geometry_only_low,
        geometry_only_factor_high=geometry_only_high,
        scan_cd_charge_factor_status=scan_factor_status,
        scan_based_factor_diagnostic=scan_factor_diagnostic,
        scan_based_factor_diagnostic_low=scan_factor_diagnostic_low,
        scan_based_factor_diagnostic_high=scan_factor_diagnostic_high,
    )


def target_name(voltage_polarity: int, negative_on_pin_negative: bool) -> str:
    negative = voltage_polarity < 0 if negative_on_pin_negative else voltage_polarity > 0
    return "negative" if negative else "positive"


def lobe_terminal_excess_nC(
    lobe: LobeObservation,
    calibration: CalibrationModel,
    cprime_pF: float | np.ndarray | None = None,
    closs_pF: float | np.ndarray | None = None,
) -> float | np.ndarray | None:
    """Return directed terminal charge minus the complex passive prediction."""
    reactive = calibration.cprime_pF if cprime_pF is None else cprime_pF
    loss = calibration.closs_pF if closs_pF is None else closs_pF
    if reactive is None or loss is None:
        return None
    return (
        calibration.sign * lobe.raw_directed_charge_nC
        - passive_background_nC(lobe, reactive, loss)
    )


def per_file_metrics(
    row: FileObservation,
    calibration: CalibrationModel,
    factor: float,
    negative_on_pin_negative: bool,
) -> dict[str, float]:
    by_target: dict[str, list[tuple[LobeObservation, float]]] = {"negative": [], "positive": []}
    for lobe in row.lobes:
        excess = lobe_terminal_excess_nC(lobe, calibration)
        if excess is None:
            continue
        corrected = factor * float(excess)
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
        excess_value = lobe_terminal_excess_nC(lobe, calibration)
        if excess_value is None:
            continue
        directed = calibration.sign * lobe.raw_directed_charge_nC
        background = float(
            passive_background_nC(
                lobe,
                float(calibration.cprime_pF),
                float(calibration.closs_pF),
            )
        )
        excess = float(excess_value)
        corrected = factor * excess
        output.append({
            "condition": row.record.condition.label,
            "material": row.record.condition.material,
            "burst_kHz": row.record.condition.burst_kHz,
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
            "background_model": "whole_waveform_complex_Cstar_endpoint_projection",
            "evidence_tier": calibration.evidence_tier,
            "factor_source": calibration.factor_source,
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


def electrical_condition_metrics(
    rows: Sequence[FileObservation],
    replicates: int,
    block_length: int,
    rng: np.random.Generator,
) -> dict:
    """Aggregate calibration-independent raw Q-V energy and envelope duty."""
    output: dict = {}
    definitions = {
        "raw_qv_energy_per_burst_uJ": [
            row.burst_energy_median_uJ for row in rows
            if row.burst_energy_median_uJ is not None
            and not row.voltage_clipping_flag and not row.monitor_clipping_flag
        ],
        "apparent_lissajous_power_mW": [
            row.apparent_power_mW for row in rows
            if row.apparent_power_mW is not None
        ],
        "detected_activity_on_fraction": [
            row.duty_on_fraction for row in rows
            if row.duty_on_fraction is not None
        ],
    }
    for key, values in definitions.items():
        clean = finite(values)
        output[key] = float(np.median(clean)) if len(clean) else None
        draws = bootstrap_median_draws(clean, replicates, rng, block_length)
        low, high = percentile_interval(draws)
        output[f"{key}_ci_low"] = low
        output[f"{key}_ci_high"] = high
        output[f"{key}_n_captures"] = int(len(clean))
    # Stable aliases used by the plotting/report layer.
    output["burst_energy_median_uJ"] = output["raw_qv_energy_per_burst_uJ"]
    output["burst_energy_median_uJ_ci_low"] = output["raw_qv_energy_per_burst_uJ_ci_low"]
    output["burst_energy_median_uJ_ci_high"] = output["raw_qv_energy_per_burst_uJ_ci_high"]
    output["duty_on_fraction"] = output["detected_activity_on_fraction"]
    output["duty_on_fraction_ci_low"] = output["detected_activity_on_fraction_ci_low"]
    output["duty_on_fraction_ci_high"] = output["detected_activity_on_fraction_ci_high"]
    output["raw_qv_energy_definition"] = (
        "median_cyclic_shoelace_area_per_complete_duty_burst_within_capture_"
        "then_median_across_captures"
    )
    output["apparent_power_scope"] = (
        "raw_Qm_V_apparent_reactor_input_loss_not_plasma_only"
    )
    output["duty_fraction_definition"] = (
        "fraction_of_detected_activity_envelope_above_P10_plus_0.30_times_P90_minus_P10"
    )
    methods = [row.burst_detection_method for row in rows]
    channels = [
        "current" if "current" in method.lower()
        else "voltage" if "voltage" in method.lower()
        else "nominal_fallback"
        for method in methods
    ]
    if channels:
        output["duty_detector_channel_mode"] = max(
            sorted(set(channels)), key=channels.count
        )
        output["duty_detector_channel_agreement"] = float(
            np.mean(np.asarray(channels) == output["duty_detector_channel_mode"])
        )
    else:
        output["duty_detector_channel_mode"] = None
        output["duty_detector_channel_agreement"] = None
    closure = [
        period.closure_contribution_fraction
        for row in rows for period in row.burst_periods
        if period.closure_contribution_fraction is not None
    ]
    output["raw_qv_closure_contribution_fraction_median"] = median_or_none(closure)
    output["raw_qv_closure_contribution_fraction_p95"] = (
        float(np.percentile(finite(closure), 95)) if len(finite(closure)) else None
    )
    return output


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
) -> tuple[np.ndarray, np.ndarray]:
    """Joint stratified bootstrap for the complex passive coefficients."""
    cprime_parts: list[np.ndarray] = []
    closs_parts: list[np.ndarray] = []
    for level in PASSIVE_FIT_LEVELS:
        rows = sorted(
            [
                row for row in observations
                if row.level_label == str(level)
                and row.cstar_raw_F is not None
                and not row.voltage_clipping_flag
                and not row.monitor_clipping_flag
            ],
            key=lambda item: item.record.capture_index,
        )
        if not rows:
            continue
        indices = moving_block_index_matrix(len(rows), replicates, block_length, rng)
        cprime_values = np.asarray(
            [
                (calibration.sign * row.cstar_raw_F).real * 1.0e12
                for row in rows
            ],
            dtype=float,
        )
        closs_values = np.asarray(
            [
                -(calibration.sign * row.cstar_raw_F).imag * 1.0e12
                for row in rows
            ],
            dtype=float,
        )
        cprime_parts.append(cprime_values[indices])
        closs_parts.append(closs_values[indices])
    if cprime_parts:
        cprime = np.nanmedian(np.concatenate(cprime_parts, axis=1), axis=1)
        closs = np.nanmedian(np.concatenate(closs_parts, axis=1), axis=1)
    else:
        cprime = np.full(replicates, calibration.cprime_pF or np.nan)
        closs = np.full(replicates, calibration.closs_pF or np.nan)
    return cprime, closs


def _file_metric_draws(
    row: FileObservation,
    calibration: CalibrationModel,
    target: str,
    cprime_draws: np.ndarray,
    closs_draws: np.ndarray,
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
    reactive_basis = np.asarray(
        [lobe.background_cprime_basis_nC_per_pF for lobe in lobes], dtype=float
    )
    loss_basis = np.asarray(
        [lobe.background_closs_basis_nC_per_pF for lobe in lobes], dtype=float
    )
    duration = np.asarray([lobe.duration_s for lobe in lobes], dtype=float)
    selected_indices: list[int] = []
    for burst_index in sorted({lobe.burst_index for lobe in lobes}):
        candidates = [
            index for index, lobe in enumerate(lobes)
            if lobe.burst_index == burst_index
        ]
        selected_indices.append(max(candidates, key=lambda index: amplitude[index]))
    selected = np.asarray(selected_indices, dtype=int)
    background = (
        cprime_draws[:, None] * reactive_basis[None, :]
        + closs_draws[:, None] * loss_basis[None, :]
    )
    peak_charge = factor_draws[:, None] * (
        raw[selected][None, :] - background[:, selected]
    )
    peak_rate = peak_charge * 1.0e-9 / (
        ELEMENTARY_CHARGE_C * duration[selected][None, :]
    )
    total = factor_draws * (
        float(np.sum(raw)) - np.sum(background, axis=1)
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
    cprime_draws, closs_draws = passive_calibration_draws(
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
    elif (
        calibration.factor_source == "provisional_three_level_active_scan"
        and len(calibration.scan_cd_bootstrap_pF) == replicates
    ):
        cd_draws = calibration.scan_cd_bootstrap_pF
        denominator = cd_draws - cprime_draws
        factor_draws = np.where(
            denominator > 0.05 * cd_draws,
            cd_draws / denominator,
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
                cprime_draws,
                closs_draws,
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
    cprime = (
        np.asarray(cprime_draws, dtype=float)
        if cprime_draws is not None and len(cprime_draws) == count
        else np.full(count, calibration.cprime_pF or np.nan)
    )
    if calibration.factor_source == "provisional_three_level_active_scan":
        cd_scan = (
            calibration.scan_cd_bootstrap_pF
            if len(calibration.scan_cd_bootstrap_pF) == count
            else np.full(count, calibration.scan_cd_pF or np.nan)
        )
        active_denominator = cd_scan - cprime
        active_factor = np.where(
            active_denominator > 0.05 * cd_scan,
            cd_scan / active_denominator,
            np.nan,
        )
        factor = active_factor.copy()
        # Keep the full-base Pyrex estimate as a visibly separate model-form
        # scenario while ensuring the sensitivity envelope includes both the
        # measured active secant and geometry assumptions.
        if calibration.geometry_only_factor is not None and calibration.cprime_pF is not None:
            cd_geometry, _ = geometry_factor_samples(
                calibration.cprime_pF, args, count, rng
            )
            scaled_cprime = charge_scale * cprime
            geometry_denominator = cd_geometry - scaled_cprime
            geometry_factor = np.where(
                geometry_denominator > 0.05 * cd_geometry,
                cd_geometry / geometry_denominator,
                np.nan,
            )
            use_geometry = np.arange(count) % 2 == 1
            factor[use_geometry] = geometry_factor[use_geometry]
    elif (
        calibration.factor_source == "full_base_pyrex_geometry_scenario"
        and calibration.cprime_pF is not None
    ):
        # The same Channel-D scale changes both the terminal excess and the
        # inferred Ccell.  Recompute F for every geometry/scale draw rather
        # than multiplying a nominal-F distribution after the fact.
        cd, _ = geometry_factor_samples(calibration.cprime_pF, args, count, rng)
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
        if all(str(level) in labels for level in PASSIVE_FIT_LEVELS):
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
                "passive_background_status": summary.get("passive_background_status"),
                "carrier_polarity_assignment": summary.get("carrier_polarity_assignment"),
                "carrier_polarity_assignment_status": summary.get(
                    "carrier_polarity_assignment_status"
                ),
                "Ccell_reactive_pF": summary.get("Ccell_reactive_pF"),
                "Ccell_reactive_ci_low_pF": summary.get("Ccell_reactive_ci_low_pF"),
                "Ccell_reactive_ci_high_pF": summary.get("Ccell_reactive_ci_high_pF"),
                "Cd_scan_status": summary.get("Cd_scan_status"),
                "Cd_scan_pF": summary.get("Cd_scan_pF"),
                "Cd_scan_ci_low_pF": summary.get("Cd_scan_ci_low_pF"),
                "Cd_scan_ci_high_pF": summary.get("Cd_scan_ci_high_pF"),
                "Cd_scan_charge_factor_status": summary.get("Cd_scan_charge_factor_status"),
                "Cd_scan_based_factor_diagnostic": summary.get("Cd_scan_based_factor_diagnostic"),
                "Cd_geometry_scenario_pF": summary.get("Cd_geometry_scenario_pF"),
                "charge_correction_factor": summary.get("charge_correction_factor"),
                "model_sensitivity_draw_valid_fraction": summary.get(
                    "model_sensitivity_draw_valid_fraction"
                ),
                "retained_charge_nC": summary.get("retained_terminal_charge_nC"),
                "retained_charge_ci_low_nC": summary.get("retained_terminal_charge_ci_low_nC"),
                "retained_charge_ci_high_nC": summary.get("retained_terminal_charge_ci_high_nC"),
                "retained_charge_status": summary.get("retained_charge_status"),
                "raw_qv_energy_per_burst_uJ": summary.get("raw_qv_energy_per_burst_uJ"),
                "raw_qv_energy_per_burst_uJ_ci_low": summary.get("raw_qv_energy_per_burst_uJ_ci_low"),
                "raw_qv_energy_per_burst_uJ_ci_high": summary.get("raw_qv_energy_per_burst_uJ_ci_high"),
                "apparent_lissajous_power_mW": summary.get("apparent_lissajous_power_mW"),
                "apparent_lissajous_power_mW_ci_low": summary.get("apparent_lissajous_power_mW_ci_low"),
                "apparent_lissajous_power_mW_ci_high": summary.get("apparent_lissajous_power_mW_ci_high"),
                "detected_activity_on_fraction": summary.get("detected_activity_on_fraction"),
                "detected_activity_on_fraction_ci_low": summary.get("detected_activity_on_fraction_ci_low"),
                "detected_activity_on_fraction_ci_high": summary.get("detected_activity_on_fraction_ci_high"),
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
        "## Raw Q–V energy, apparent reactor power, and detected duty fraction",
        "",
        "These electrical quantities do not require the passive-background or Cd model. Power is raw Qm–V loop power for the complete reactor circuit, not plasma-only power.",
        "",
        "| Condition | Energy per duty period (µJ) | Apparent power (W) | Detected activity-on fraction |",
        "|---|---:|---:|---:|",
    ]
    seen_conditions: set[str] = set()
    for row in rows:
        condition = str(row.get("condition"))
        if condition in seen_conditions:
            continue
        seen_conditions.add(condition)
        energy = number(row.get("raw_qv_energy_per_burst_uJ"))
        power_mW = row.get("apparent_lissajous_power_mW")
        power_W = (
            float(power_mW) / 1000.0
            if power_mW is not None and np.isfinite(float(power_mW)) else None
        )
        duty = row.get("detected_activity_on_fraction")
        duty_text = (
            f"{100.0 * float(duty):.1f}%"
            if duty is not None and np.isfinite(float(duty)) else "—"
        )
        lines.append(
            f"| {row.get('condition_label') or condition} | {energy} | "
            f"{number(power_W)} | {duty_text} |"
        )
    lines.extend([
        "",
        "## Polarity-resolved charge-transfer estimates",
        "",
        "| Condition | Polarity | Peak half-cycle (nC) | Peak half-cycle-average rate (e s⁻¹) | Whole-record rate (e s⁻¹) | Evidence/status |",
        "|---|---:|---:|---:|---:|---|",
    ])
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
            "Peak aggregation is nested: within each capture and polarity, the maximum-amplitude carrier lobe is selected once per duty burst and its p95 is calculated; the condition estimate is then the median of those capture-level p95 values. Lobes are never pooled across the 64 captures to calculate a single p95.",
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
        excess_value = lobe_terminal_excess_nC(example_lobe, calibration)
        if excess_value is not None:
            directed = calibration.sign * example_lobe.raw_directed_charge_nC
            background = float(
                passive_background_nC(
                    example_lobe,
                    float(calibration.cprime_pF),
                    float(calibration.closs_pF),
                )
            )
            excess = float(excess_value)
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
        _level_point_stats(observations, calibration.sign, str(level))
        for level in ACTIVE_CD_LEVELS
    ]
    hx = [row["x"] for row in high_points if row["x"] is not None and row["y"] is not None]
    hy = [row["y"] for row in high_points if row["x"] is not None and row["y"] is not None]
    if (
        len(hx) == 3
        and calibration.scan_cd_pF is not None
        and calibration.scan_cd_intercept_nC is not None
    ):
        slope = calibration.scan_cd_pF
        intercept = calibration.scan_cd_intercept_nC
        grid = np.linspace(min(hx), max(hx), 100)
        accepted = calibration.scan_cd_status.startswith("supported")
        r2_text = (
            f"{calibration.scan_cd_r_squared:.3f}"
            if calibration.scan_cd_r_squared is not None else "n/a"
        )
        axes[1].plot(
            grid,
            slope * grid + intercept,
            "-" if accepted else "--",
            color="#b23a48",
            linewidth=1.5,
            label=(
                "100/105/115% Cd fit "
                f"({'provisional accepted' if accepted else 'rejected'}; R²={r2_text})"
            ),
        )
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
    conditions = sorted(calibrations)
    if not conditions:
        return
    fig, axes = plt.subplots(1, 3, figsize=(14.2, 4.8), sharex=True)
    condition_colors = plt.cm.tab10(np.linspace(0.0, 0.8, len(conditions)))
    offsets = (
        np.linspace(-1.8, 1.8, len(conditions))
        if len(conditions) > 1 else np.asarray([0.0])
    )
    for offset, color, condition in zip(offsets, condition_colors, conditions):
        model = calibrations[condition]
        accepted = model.passive_status.startswith("supported")
        marker = "o" if accepted else "x"
        for level in PASSIVE_FIT_LEVELS:
            level_rows = [
                row for row in observations[condition]
                if row.level_label == str(level) and row.cstar_raw_F is not None
            ]
            values = [model.sign * row.cstar_raw_F for row in level_rows]
            cprime = np.asarray([value.real * 1.0e12 for value in values])
            closs = np.asarray([-value.imag * 1.0e12 for value in values])
            codes = finite(row.complex_fit_signal_codes for row in level_rows)
            for axis, data in ((axes[0], cprime), (axes[1], closs), (axes[2], codes)):
                axis.scatter(
                    np.full(len(data), level + offset),
                    data,
                    s=14,
                    alpha=0.25,
                    color=color,
                    marker=marker,
                )
                if len(data):
                    low, high = np.percentile(data, [2.5, 97.5])
                    axis.errorbar(
                        level + offset,
                        np.median(data),
                        yerr=[[np.median(data) - low], [high - np.median(data)]],
                        fmt=marker,
                        color=color,
                        capsize=3,
                        label=(
                            f"{display_label(condition)} ({'accepted' if accepted else 'rejected'})"
                            if level == 40 and axis is axes[0] else None
                        ),
                    )
        if model.cprime_pF is not None:
            axes[0].axhline(model.cprime_pF, linestyle="--", color=color, linewidth=0.8, alpha=0.55)
        if model.closs_pF is not None:
            axes[1].axhline(model.closs_pF, linestyle="--", color=color, linewidth=0.8, alpha=0.55)
    axes[0].set_ylabel("Reactive capacitance, C' (pF)")
    axes[1].set_ylabel("Loss capacitance, C'' (pF)")
    axes[2].set_ylabel("Passive Channel-D carrier signal (ADC codes p-p)")
    axes[2].axhline(8.0, color="#b23a48", linestyle="--", linewidth=1.1, label="8-code gate")
    for axis in axes:
        axis.set_xlabel("Breakdown voltage (%)")
        axis.set_xticks(PASSIVE_FIT_LEVELS)
        axis.grid(alpha=0.22)
    axes[0].legend(fontsize=7, frameon=False)
    axes[2].legend(fontsize=8, frameon=False)
    fig.suptitle(
        "Whole-waveform complex C* fit and passive-signal quantization audit\n"
        "Points are captures; crosses are rejected conditions; bars are 2.5–97.5% capture intervals"
    )
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    save_figure(fig, output, args.dpi, not args.no_pdf)


def summary_metric_reportable(row: dict, target: str, metric: str) -> bool:
    key = f"{target}_{metric}"
    point = row.get(key)
    low = row.get(f"{key}_analysis_ci_low")
    high = row.get(f"{key}_analysis_ci_high")
    if not str(row.get("passive_background_status", row.get("Ccell_status", ""))).startswith("supported"):
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
            supported = str(
                row.get("passive_background_status", row.get("Ccell_status", ""))
            ).startswith("supported")
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
            supported = str(
                row.get("passive_background_status", row.get("Ccell_status", ""))
            ).startswith("supported")
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
    row["background_failed_gates"] = ";".join(model.background_failed_gates)
    bootstrap = np.asarray(row.pop("scan_cd_bootstrap_pF", []), dtype=float)
    row["scan_cd_bootstrap_valid_draws"] = int(np.sum(np.isfinite(bootstrap)))
    for name in ("passive_slopes_nC_per_kV", "passive_slope_mad_nC_per_kV", "passive_threshold_nC"):
        values = row.pop(name)
        for polarity, value in values.items():
            prefix = "deprecated_lobe_ratio_" if name != "passive_threshold_nC" else ""
            row[f"{prefix}{name}_{'negative_voltage' if int(polarity) < 0 else 'positive_voltage'}"] = value
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

The passive response is estimated by projecting every usable sample in each
40/60/75 %-breakdown waveform onto in-phase and quadrature carrier bases. The
least-squares model is

`Qm(t) = C' V_I(t) + C'' V_Q(t) + offset + linear drift`.

This whole-waveform complex-C* regression replaces the deprecated lobe-amplitude
ratio fit, which could lock onto one ADC-code ratio. A passive capture with less
than eight peak-to-peak Channel-D codes remains explicitly quantization-limited;
the regression cannot create information absent from the acquisition.

Adjacent interpolated carrier-voltage zero crossings define a half-cycle. For
voltage polarity `s = ±1`, the directed terminal charge is

`D_h = s [Qm(t1) - Qm(t0)]`.

The fitted in-phase and quadrature carrier values are interpolated at both lobe
endpoints. Their exact endpoint changes give the complex passive prediction
`B_h(C', C'')`; the operational terminal excess is

`X_h = D_h - B_h(C', C'')`.

The 90 % captures are excluded from that fit and used as an out-of-sample
passive null. Peak metrics must exceed the larger of the training-residual
limit and the 90 %-holdout p95 at the lower endpoint of the joint 95 % analysis
interval. Same-frequency whole-record metrics are compared with the same
90 %-holdout statistic. Transferred-frequency whole-record results have no
same-frequency holdout and are labeled accordingly.

When a valid dielectric capacitance is available, the classical model gives

`q_surface,h = [Cd/(Cd-Ccell)] X_h`.

The effective active-branch `Cd` fit uses the capture medians at 100, 105, and
115 % breakdown, with an intercept. Captures are resampled within commanded
level. Use requires unclipped records, at least
{args.active_cd_min_clean_captures} clean captures per level, resolved activity
at 100 %, monotonic voltage and charge, `R² ≥ {args.active_cd_min_r_squared:g}`,
consistent pairwise slopes, and at least 95 % physical bootstrap draws with
`Cd > 1.05 Ccell`. A passing three-level result is labeled **provisional**, not
fully validated; a denser active scan or direct dielectric measurement remains
preferable. Legacy MAX is out-of-sample and is never fitted. The full-base
Pyrex geometry result is retained as a separate model scenario rather than
being blended into the statistical confidence interval.

## Requested outputs

- **Charge per peak half-cycle:** within each capture and polarity, select the
  maximum-amplitude carrier lobe once per duty burst and calculate p95; report
  the median of those capture-level p95 values. Lobes from 64 captures are not
  pooled into one p95.
- **Peak rate:** p95 of half-cycle-average `q/(e Δt)` across the same
  maximum-amplitude lobe selected once per duty burst. It is not an
  instantaneous nanosecond particle flux.
- **Whole-record rate:** signed sum of background-subtracted lobe charge divided
  by elementary charge and full record duration, including duty-off time.
- **Retained charge:** only measured when stable quiet plateaus exist at both
  record edges. Polarity imbalance is reported separately and never relabeled
  retained surface charge.
- **Raw Q–V energy and apparent power:** for each complete duty-burst period,
  `E = 0.5 |Σ V_i (Q_{{i+1}} - Q_{{i-1}})|`, with cyclic indices, kV, and nC,
  so `E` is in µJ. A capture is represented by its median period energy and
  `P = f_burst E`; condition intervals resample captures. This is total raw
  Lissajous reactor input loss (plasma + dielectric + liquid + phase-skew), not
  plasma-only power.
- **Detected duty-on fraction:** time fraction above the same envelope threshold
  `P10 + 0.30(P90-P10)` within the same duty-period windows. The detector channel
  is audited because current- and voltage-envelope fractions are not identical
  physical quantities.
- **Dose clock:** ideal minutes to one negative-charge equivalent per ion are
  `c_mM V_mL N_A 10^-6 / (60 R_-)`, using a default volume of
  {args.liquid_volume_ml:g} mL and concentration of
  {args.metal_ion_concentration_mM:g} mM. It assumes 100 % delivery/utilization,
  includes electrons plus negative ions, and is not a chemical conversion time.
  BMIM curves are hypothetical rate-transfer comparisons because BMIM nitrate
  contains no metal; Mn²⁺ reduction requires at least two equivalents.

Negative rate means a **net external-terminal electrical equivalent** assigned
to electrons plus negative ions under the configured pin-polarity mapping. It
is not a species-resolved gross particle count, and memory voltage can shift
actual gas conduction relative to a source-voltage zero crossing.
Area-normalized flux is blank unless an active area is supplied.

## Uncertainty

The independent sampling unit is a waveform capture, not a carrier half-cycle.
`repeat_ci` is a conditional MAX-capture repeatability interval with the passive
calibration fixed. `analysis_ci` jointly resamples 40/60/75 % complex-C*
calibration captures within level, 100/105/115 % active-Cd captures when that
model passes, and legacy MAX captures using a
{args.bootstrap_replicates}-replicate moving-block bootstrap with block length
{args.bootstrap_block_files}. These are technical-repeat intervals; the 64
sequential captures are not independent biological or experimental repeats.

Broad model-sensitivity intervals additionally sample the declared monitor
scale and keep the measured active-Cd and approximate full-base Pyrex geometry
as separate sampled scenarios. The common Channel-D scale is propagated through
both terminal charge and Ccell in `Cd/(Cd-Ccell)`, and the geometry-only branch
spans the scalar-cell choice from `C'` to `|C*|`. These are bounded scenario ranges, not
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
        burst_period_rows: list[dict] = []
        level_fit_rows: list[dict] = []
        for condition in conditions:
            source = source_map[condition]
            max_rows = sorted(
                [row for row in observations_by_condition[condition] if row.level_label == "MAX"],
                key=lambda item: item.record.capture_index,
            )
            electrical = electrical_condition_metrics(
                max_rows,
                args.bootstrap_replicates,
                args.bootstrap_block_files,
                rng,
            )
            for row in max_rows:
                for period in row.burst_periods:
                    burst_period_rows.append({
                        "condition": condition.label,
                        "material": condition.material,
                        "burst_kHz": condition.burst_kHz,
                        "member": row.record.member,
                        "capture_index": row.record.capture_index,
                        "period_index": period.burst_index,
                        "start_s": period.start_s,
                        "stop_s": period.stop_s,
                        "midpoint_s": period.midpoint_s,
                        "signed_raw_qv_energy_uJ": period.signed_energy_uJ,
                        "raw_qv_energy_uJ": period.energy_uJ,
                        "apparent_power_using_capture_frequency_mW": (
                            period.energy_uJ * row.detected_burst_Hz / 1000.0
                        ),
                        "detected_activity_on_fraction": period.duty_on_fraction,
                        "closure_delta_V_kV": period.closure_delta_V_kV,
                        "closure_delta_Q_nC": period.closure_delta_Q_nC,
                        "closure_contribution_fraction": period.closure_contribution_fraction,
                        "detected_burst_Hz": row.detected_burst_Hz,
                        "detector_method": row.burst_detection_method,
                        "source_voltage_clipped": row.voltage_clipping_flag,
                        "monitor_voltage_clipped": row.monitor_clipping_flag,
                        "power_usable": not row.voltage_clipping_flag and not row.monitor_clipping_flag,
                    })
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
                    **electrical,
                })
                for file_row in max_rows:
                    per_file_rows.append({
                        "condition": condition.label,
                        "material": condition.material,
                        "burst_kHz": condition.burst_kHz,
                        "member": file_row.record.member,
                        "capture_index": file_row.record.capture_index,
                        "carrier_Hz": file_row.carrier_Hz,
                        "detected_burst_Hz": file_row.detected_burst_Hz,
                        "burst_detection_method": file_row.burst_detection_method,
                        "burst_frequency_relative_error": file_row.burst_frequency_relative_error,
                        "voltage_pp_kV": file_row.voltage_pp_kV,
                        "raw_qv_energy_per_burst_uJ": file_row.burst_energy_median_uJ,
                        "burst_energy_median_uJ": file_row.burst_energy_median_uJ,
                        "apparent_lissajous_power_mW": file_row.apparent_power_mW,
                        "detected_activity_on_fraction": file_row.duty_on_fraction,
                        "duty_on_fraction": file_row.duty_on_fraction,
                        "duty_envelope_contrast": file_row.duty_envelope_contrast,
                        "source_voltage_clipped": file_row.voltage_clipping_flag,
                        "current_clipped": file_row.current_clipping_flag,
                        "monitor_voltage_clipped": file_row.monitor_clipping_flag,
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
            file_metrics = [metrics for _, metrics in file_metric_pairs if metrics]
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
                if calibration.background_status.startswith("supported"):
                    evidence = (
                        "exploratory_transferred_model"
                        if central_factor != 1.0
                        else "background_subtracted_transferred_across_frequency"
                    )
                else:
                    evidence = "diagnostic_transferred_background"
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
                "passive_background_status": calibration.background_status,
                "passive_background_failed_gates": ";".join(calibration.background_failed_gates),
                "Ccell_status": calibration.passive_status,
                "Ccell_reactive_pF": calibration.cprime_pF,
                "Ccell_reactive_ci_low_pF": calibration.cprime_ci_low_pF,
                "Ccell_reactive_ci_high_pF": calibration.cprime_ci_high_pF,
                "Ccell_loss_pF": calibration.closs_pF,
                "Ccell_loss_ci_low_pF": calibration.closs_ci_low_pF,
                "Ccell_loss_ci_high_pF": calibration.closs_ci_high_pF,
                "Cd_scan_status": calibration.scan_cd_status,
                "Cd_scan_pF": calibration.scan_cd_pF,
                "Cd_scan_ci_low_pF": calibration.scan_cd_ci_low_pF,
                "Cd_scan_ci_high_pF": calibration.scan_cd_ci_high_pF,
                "Cd_scan_intercept_nC": calibration.scan_cd_intercept_nC,
                "Cd_scan_r_squared": calibration.scan_cd_r_squared,
                "Cd_scan_pairwise_relative_span": calibration.scan_cd_pairwise_relative_span,
                "Cd_scan_breakdown_active_fraction": calibration.scan_cd_breakdown_active_fraction,
                "Cd_scan_clean_counts": calibration.scan_cd_clean_counts,
                "Cd_scan_charge_factor_status": calibration.scan_cd_charge_factor_status,
                "Cd_scan_based_factor_diagnostic": calibration.scan_based_factor_diagnostic,
                "Cd_scan_based_factor_diagnostic_low": calibration.scan_based_factor_diagnostic_low,
                "Cd_scan_based_factor_diagnostic_high": calibration.scan_based_factor_diagnostic_high,
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
                "geometry_only_charge_correction_factor": calibration.geometry_only_factor,
                "geometry_only_charge_correction_factor_low": calibration.geometry_only_factor_low,
                "geometry_only_charge_correction_factor_high": calibration.geometry_only_factor_high,
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
                **electrical,
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
                    passive_supported = calibration.background_status.startswith("supported")
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
                        "diagnostic_analysis_ci_low": analysis_low,
                        "diagnostic_analysis_ci_high": analysis_high,
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
                            "within-capture p95 across maximum-amplitude carrier half-cycles selected once per duty burst, then median of capture-level p95 values across captures"
                            if metric_name == "peak_halfcycle_charge_nC" else
                            "within-capture p95 half-cycle-average net charge-equivalent rate across the same maximum-amplitude lobes, then median of capture-level p95 values; not instantaneous flux"
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
                    "material": condition.material,
                    "burst_kHz": condition.burst_kHz,
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
                    "complex_fit_residual_rms_nC": file_row.complex_fit_residual_rms_nC,
                    "complex_fit_signal_codes": file_row.complex_fit_signal_codes,
                    "raw_qv_energy_per_burst_uJ": file_row.burst_energy_median_uJ,
                    "burst_energy_median_uJ": file_row.burst_energy_median_uJ,
                    "apparent_lissajous_power_mW": file_row.apparent_power_mW,
                    "detected_activity_on_fraction": file_row.duty_on_fraction,
                    "duty_on_fraction": file_row.duty_on_fraction,
                    "duty_envelope_contrast": file_row.duty_envelope_contrast,
                    "source_voltage_clipped": file_row.voltage_clipping_flag,
                    "current_clipped": file_row.current_clipping_flag,
                    "monitor_voltage_clipped": file_row.monitor_clipping_flag,
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

        level_fit_rows = [
            {
                "condition": source.label,
                "material": source.material,
                "burst_kHz": source.burst_kHz,
                "level": level,
                **_level_point_stats(
                    observations_by_condition[source],
                    calibration.sign,
                    str(level),
                ),
            }
            for source, calibration in calibrations.items()
            for level in LEVELS
        ]
        dose_response_binned = build_capture_balanced_binned(
            peak_lobe_rows,
            x_key="amplitude_kV",
            bins=12,
            replicates=args.bootstrap_replicates,
            block_length=args.bootstrap_block_files,
            seed=args.random_seed + 41,
        )
        stationarity_binned = build_capture_balanced_binned(
            peak_lobe_rows,
            x_key="midpoint_s",
            bins=12,
            replicates=args.bootstrap_replicates,
            block_length=args.bootstrap_block_files,
            seed=args.random_seed + 43,
        )
        stationarity_rows = build_stationarity_metrics(
            peak_lobe_rows,
            replicates=args.bootstrap_replicates,
            block_length=args.bootstrap_block_files,
            seed=args.random_seed + 47,
        )
        dose_clock_rows = build_dose_clock_rows(
            long_results,
            volume_ml=args.liquid_volume_ml,
            concentration_mM=args.metal_ion_concentration_mM,
            equivalents_per_ion=args.dose_electrons_per_metal_ion,
        )

        if not args.no_plots:
            plot_coverage(manifest, selected_members, figures / "01_data_coverage", args)
            for source, calibration in calibrations.items():
                source_rows = observations_by_condition[source]
                plot_qv_grid(source, source_rows, calibration, qv_dir / f"{source.label}_qv_traces", args)
                plot_scan_fit(
                    source,
                    source_rows,
                    calibration,
                    scan_dir / f"{source.label}_breakdown_fit",
                    args,
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
            supplementary_save = lambda fig, base, _args: save_figure(
                fig, base, args.dpi, not args.no_pdf
            )
            plot_power_audit(
                summaries,
                per_file_rows,
                figures / "07_apparent_discharge_power",
                supplementary_save,
                args,
            )
            plot_duty_audit(
                summaries,
                per_file_rows,
                figures / "08_duty_on_fraction_audit",
                supplementary_save,
                args,
            )
            plot_binned_facet(
                peak_lobe_rows,
                figures / "09_per_lobe_dose_response",
                supplementary_save,
                args,
                "dose_response",
            )
            plot_binned_facet(
                peak_lobe_rows,
                figures / "10_within_record_stationarity",
                supplementary_save,
                args,
                "stationarity",
            )
            plot_dose_clock(
                dose_clock_rows,
                figures / "11_negative_charge_equivalent_dose_clock",
                supplementary_save,
                args,
            )

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
        electrical_condition_rows = [
            {
                key: value for key, value in summary.items()
                if key in {"condition", "condition_label", "material", "burst_kHz", "n_max_files"}
                or key.startswith((
                    "raw_qv_",
                    "apparent_lissajous_power_",
                    "detected_activity_",
                    "duty_",
                    "burst_energy_",
                ))
            }
            for summary in summaries
        ]
        write_csv(output / "headline_results.csv", headline)
        write_csv(output / "supervisor_summary.csv", summaries)
        write_csv(output / "electrical_condition_metrics.csv", electrical_condition_rows)
        write_csv(output / "long_form_results.csv", long_results)
        write_csv(output / "per_file_maximum_metrics.csv", per_file_rows)
        write_csv(output / "peak_halfcycle_observations.csv", peak_lobe_rows)
        write_csv(output / "burst_period_observations.csv", burst_period_rows)
        write_csv(output / "dose_response_binned.csv", dose_response_binned)
        write_csv(output / "stationarity_binned.csv", stationarity_binned)
        write_csv(output / "stationarity_metrics.csv", stationarity_rows)
        write_csv(output / "dose_clock_results.csv", dose_clock_rows)
        write_csv(output / "capacitance_and_fit_results.csv", [flatten_calibration(model) for model in calibrations.values()])
        write_csv(output / "breakdown_level_fit_points.csv", level_fit_rows)
        write_csv(output / "archive_manifest.csv", manifest_rows)
        (output / "RESULTS_OVERVIEW.md").write_text(
            results_overview_text(headline), encoding="utf-8"
        )
        (output / "METHODS_AND_LIMITATIONS.md").write_text(methodology_text(args), encoding="utf-8")
        captions = """# Figure captions

1. **Signal-processing example.** Full acquisition and two-carrier-cycle zoom showing the source voltage, sign-audited Pearson diagnostic, Channel-D charge, detected lobe boundaries, complex-C* passive endpoint prediction, and model factor.
2. **Data coverage.** Number of separate waveform captures selected at each commanded percentage of breakdown voltage. Captures are sequential technical repeats, not independently rebuilt experiments. Legacy MAX data were acquired in a separate session.
3. **Complex cell capacitance.** Capture-level reactive and loss capacitances from whole-waveform in-phase/quadrature fits to 40/60/75 % data. Only conditions passing the empirical background, quantization, orientation, and stability gates are shown.
4. **Peak half-cycle charge.** Within each capture, p95 is calculated across the maximum-amplitude lobe selected once per duty burst; markers are medians of capture-level p95 values. Thin intervals jointly resample passive calibration, active Cd where supported, and MAX captures. Broad intervals are model scenarios, not confidence intervals.
5. **Peak rate.** The same nested aggregation applied to half-cycle-average net charge-equivalent rate. This is not a nanosecond-resolved particle-flux maximum.
6. **Whole-record rate and frequency audit.** Signed background-subtracted charge divided by the full record duration and elementary charge. Independently scanned, transferred, and rejected conditions remain visibly distinct.
7. **Retained-charge availability.** Persistent retained charge requires stationary quiet monitor plateaus at both record edges; DC coupling also remains to be verified. Charge imbalance is not substituted.
8. **Q–V scan supplements.** Faint lines are independent captures; the bold line is the median-amplitude capture. Slope guides are descriptive. Rejected models remain labeled.
9. **Breakdown fits.** Small points are captures; large points and bars show capture distributions. The effective Cd regression uses capture medians at 100/105/115 %, an intercept, and predefined activity/monotonicity/R²/pairwise/physical gates. A passing result is provisional. Legacy MAX is out-of-sample and never fitted.
10. **Apparent discharge power.** Per-period raw Qm–V cyclic shoelace area is reduced to a capture median, then to a condition median with capture-block bootstrap intervals. Power is `f_burst E`. It includes plasma, dielectric, liquid, and phase-skew losses and is not plasma-only power.
11. **Duty-on audit.** Fraction of each detected duty-burst period above `P10 + 0.30(P90-P10)` using the same activity envelope and period windows as lobe and power segmentation. Bars are condition medians; points are captures.
12. **Per-lobe dose response and stationarity.** Charge is first reduced to a median within amplitude/time bin and capture, then aggregated across captures. Faint points are a deterministic raw-lobe subsample. Empty amplitude bins are not connected. The stationarity table additionally reports first-to-last-quintile drift.
13. **Dose clock.** Ideal minutes to one negative-charge equivalent per ion versus volume at 5 mM; the 2.5 mL reactor is marked. BMIM is a hypothetical electrical-rate transfer, negative charge includes electrons and negative ions, dotted curves use non-reportable diagnostic rates, and Mn²⁺ full reduction requires at least two equivalents.

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
            "liquid_volume_ml": args.liquid_volume_ml,
            "metal_ion_concentration_mM": args.metal_ion_concentration_mM,
            "dose_electrons_per_metal_ion": args.dose_electrons_per_metal_ion,
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
