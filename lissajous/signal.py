"""Measured-frequency signal, loop, charge, and QC estimators."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .io import Waveform


@dataclass
class FrequencyEstimate:
    carrier_Hz: float
    zero_cross_Hz: float | None
    relative_disagreement: float | None
    carrier_status: str
    carrier_method: str
    harmonic_decision: str
    crosscheck_diagnostic: str
    autocorr_carrier_Hz: float | None
    autocorr_relative_disagreement: float | None
    offset_one_sidedness: float
    burst_Hz: float | None
    burst_zero_cross_Hz: float | None
    burst_relative_disagreement: float | None
    burst_status: str
    duty_on_fraction: float
    burst_on_cycles: float | None
    duty_status: str
    envelope_contrast: float
    envelope_threshold: float | None
    envelope: np.ndarray
    active_mask: np.ndarray


@dataclass
class LoopMetrics:
    energy_signed_uJ: np.ndarray
    dQ_intercept_nC: np.ndarray
    positive_half_dQ_nC: np.ndarray
    negative_half_dQ_nC: np.ndarray
    cycle_midpoint_s: np.ndarray
    active_cycle_fraction: float
    loop_phase: np.ndarray
    loop_voltage_kV: np.ndarray
    loop_charge_nC: np.ndarray
    Ccell_branch_pF: float | None
    Cd_branch_pF: float | None


def _parabolic_peak(magnitude: np.ndarray, index: int) -> float:
    if index <= 0 or index >= magnitude.size - 1:
        return float(index)
    left = float(magnitude[index - 1])
    center = float(magnitude[index])
    right = float(magnitude[index + 1])
    denom = left - 2.0 * center + right
    if abs(denom) <= np.finfo(float).eps:
        return float(index)
    return float(index + 0.5 * (left - right) / denom)


def _block_mean(values: np.ndarray, factor: int) -> np.ndarray:
    if factor <= 1:
        return values
    usable = (values.size // factor) * factor
    if usable < factor:
        return values
    return values[:usable].reshape((-1, factor)).mean(axis=1)


def dominant_frequency(
    time_s: np.ndarray,
    values: np.ndarray,
    minimum_Hz: float,
    maximum_Hz: float,
) -> float:
    """Find the strongest measured line without substituting a burst-folder label."""

    dt = float(np.median(np.diff(time_s)))
    fs = 1.0 / dt
    target_fs = max(8.0 * maximum_Hz, 4.0 * minimum_Hz)
    factor = max(1, int(np.floor(fs / target_fs)))
    signal = _block_mean(values - float(np.mean(values)), factor)
    dt_eff = dt * factor
    if signal.size < 32:
        raise ValueError("record is too short for frequency estimation")
    window = np.hanning(signal.size)
    spectrum = np.abs(np.fft.rfft(signal * window))
    frequency = np.fft.rfftfreq(signal.size, dt_eff)
    band = (frequency >= minimum_Hz) & (frequency <= min(maximum_Hz, 0.49 / dt_eff))
    indices = np.flatnonzero(band)
    if not indices.size:
        raise ValueError("frequency search band contains no FFT bins")
    peak = int(indices[np.argmax(spectrum[indices])])
    refined = _parabolic_peak(spectrum, peak)
    return float(refined / (signal.size * dt_eff))


def zero_cross_mode_frequency(
    time_s: np.ndarray,
    values: np.ndarray,
    minimum_Hz: float,
    maximum_Hz: float,
    active_mask: np.ndarray | None = None,
) -> float | None:
    """Estimate the within-burst carrier from the modal rising-crossing interval."""

    centered = values - float(np.median(values))
    crossing = (centered[:-1] <= 0) & (centered[1:] > 0)
    if active_mask is not None:
        active = np.asarray(active_mask, dtype=bool)
        if active.shape != centered.shape:
            raise ValueError("active_mask must match the waveform shape")
        crossing &= active[:-1] & active[1:]
    indices = np.flatnonzero(crossing)
    if indices.size < 4:
        return None
    slopes = centered[indices + 1] - centered[indices]
    slope_floor = 0.15 * float(np.percentile(np.abs(slopes), 90))
    indices = indices[np.abs(slopes) >= slope_floor]
    slopes = slopes[np.abs(slopes) >= slope_floor]
    if indices.size < 4:
        return None
    fraction = np.divide(
        -centered[indices],
        slopes,
        out=np.zeros_like(slopes, dtype=float),
        where=np.abs(slopes) > np.finfo(float).eps,
    )
    crossing_t = time_s[indices] + fraction * (time_s[indices + 1] - time_s[indices])
    periods = np.diff(crossing_t)
    if active_mask is not None and periods.size:
        # Do not turn an off-window gap into a spurious low-frequency period.
        consecutive_active = np.asarray(
            [
                bool(np.all(active_mask[left : right + 2]))
                for left, right in zip(indices[:-1], indices[1:])
            ],
            dtype=bool,
        )
        periods = periods[consecutive_active]
    valid = (
        np.isfinite(periods)
        & (periods >= 1.0 / maximum_Hz)
        & (periods <= 1.0 / minimum_Hz)
    )
    periods = periods[valid]
    if periods.size < 3:
        return None
    log_period = np.log10(periods)
    if float(np.ptp(log_period)) < 0.02:
        return float(1.0 / np.median(periods))
    bins = min(80, max(12, int(np.sqrt(periods.size) * 2)))
    count, edges = np.histogram(log_period, bins=bins)
    peak = int(np.argmax(count))
    in_peak = (log_period >= edges[peak]) & (log_period <= edges[peak + 1])
    if np.count_nonzero(in_peak) < 3:
        return None
    return float(1.0 / np.median(periods[in_peak]))


def zero_cross_frequency(
    time_s: np.ndarray,
    values: np.ndarray,
    reference_Hz: float,
    tolerance: float = 0.35,
    active_mask: np.ndarray | None = None,
) -> float | None:
    """Cross-check frequency using rising zero crossings near the FFT period."""

    centered = values - float(np.median(values))
    crossing = (centered[:-1] <= 0) & (centered[1:] > 0)
    if active_mask is not None:
        active = np.asarray(active_mask, dtype=bool)
        if active.shape != centered.shape:
            raise ValueError("active_mask must match the waveform shape")
        crossing &= active[:-1] & active[1:]
    indices = np.flatnonzero(crossing)
    if indices.size < 3:
        return None
    dv = centered[indices + 1] - centered[indices]
    fraction = np.divide(
        -centered[indices],
        dv,
        out=np.zeros_like(dv, dtype=float),
        where=np.abs(dv) > np.finfo(float).eps,
    )
    crossing_t = time_s[indices] + fraction * (time_s[indices + 1] - time_s[indices])
    periods = np.diff(crossing_t)
    if active_mask is not None and periods.size:
        consecutive_active = np.asarray(
            [
                bool(np.all(active_mask[left : right + 2]))
                for left, right in zip(indices[:-1], indices[1:])
            ],
            dtype=bool,
        )
        periods = periods[consecutive_active]
    reference_period = 1.0 / reference_Hz
    valid = (
        np.isfinite(periods)
        & (periods >= (1.0 - tolerance) * reference_period)
        & (periods <= (1.0 + tolerance) * reference_period)
    )
    if np.count_nonzero(valid) < 2:
        return None
    return float(1.0 / np.median(periods[valid]))


def moving_mean(values: np.ndarray, width: int) -> np.ndarray:
    """Centered moving mean in O(N), with edge values padded."""

    width = int(max(1, min(width, values.size)))
    if width <= 1:
        return values.astype(float, copy=True)
    left = width // 2
    right = width - 1 - left
    padded = np.pad(values, (left, right), mode="edge")
    cumsum = np.cumsum(np.insert(padded, 0, 0.0))
    return (cumsum[width:] - cumsum[:-width]) / width


def _line_amplitude(
    time_s: np.ndarray,
    values: np.ndarray,
    frequency_Hz: float,
    active_mask: np.ndarray | None = None,
) -> float:
    """Return a windowed line amplitude, optionally restricted to drive-on data."""

    centered = values - float(np.mean(values))
    weights = np.hanning(values.size)
    if active_mask is not None:
        active = np.asarray(active_mask, dtype=bool)
        if active.shape != values.shape:
            raise ValueError("active_mask must match the waveform shape")
        weights = weights * active
    normalizer = float(np.sum(weights))
    if normalizer <= np.finfo(float).eps:
        return 0.0
    basis = np.exp(-2j * np.pi * float(frequency_Hz) * time_s)
    return float(abs(2.0 * np.sum(weights * centered * basis) / normalizer))


def harmonic_aware_carrier_frequency(
    time_s: np.ndarray,
    values: np.ndarray,
    minimum_Hz: float,
    maximum_Hz: float,
    *,
    active_mask: np.ndarray | None = None,
    harmonic_energy_ratio: float = 0.20,
) -> tuple[float, str, float | None]:
    """Estimate carrier f0 and explicitly arbitrate f0 versus f0/2 locking.

    The FFT maximum is evaluated together with the modal rising-zero-crossing
    rate.  A candidate at half the crossing rate is promoted only when the line
    at the crossing rate retains a configured fraction of the candidate's
    amplitude.  This prevents burst/ring-down subharmonics from becoming f0
    while retaining a documented decision rather than a hidden correction.
    """

    working = np.asarray(values, dtype=float)
    if active_mask is not None:
        active = np.asarray(active_mask, dtype=bool)
        if np.count_nonzero(active) >= 32:
            center = float(np.median(working[active]))
            working = (working - center) * active
    candidate = dominant_frequency(time_s, working, minimum_Hz, maximum_Hz)
    crossing = zero_cross_mode_frequency(
        time_s,
        values,
        minimum_Hz,
        maximum_Hz,
        active_mask=active_mask,
    )
    if crossing is None:
        return candidate, "fft_peak_no_modal_crossing", None

    ratio = crossing / max(candidate, np.finfo(float).eps)
    candidate_amplitude = _line_amplitude(
        time_s, values, candidate, active_mask=active_mask
    )
    crossing_amplitude = _line_amplitude(
        time_s, values, crossing, active_mask=active_mask
    )
    line_ratio = crossing_amplitude / max(
        candidate_amplitude, np.finfo(float).eps
    )

    if 1.75 <= ratio <= 2.25 and line_ratio >= harmonic_energy_ratio:
        refined = dominant_frequency(
            time_s,
            working,
            max(minimum_Hz, 0.88 * crossing),
            min(maximum_Hz, 1.12 * crossing),
        )
        return (
            refined,
            f"promoted_f0_over_f0_over_2_line_ratio_{line_ratio:.3f}",
            crossing,
        )
    if ratio >= 3.5 and line_ratio >= harmonic_energy_ratio:
        # Preliminary whole-record spectra can be dominated by the burst clock.
        refined = dominant_frequency(
            time_s,
            working,
            max(minimum_Hz, 0.88 * crossing),
            min(maximum_Hz, 1.12 * crossing),
        )
        return (
            refined,
            f"promoted_carrier_over_burst_line_ratio_{line_ratio:.3f}",
            crossing,
        )
    if 0.44 <= ratio <= 0.58:
        half_amplitude = _line_amplitude(
            time_s, values, crossing, active_mask=active_mask
        )
        if half_amplitude >= harmonic_energy_ratio * candidate_amplitude:
            refined = dominant_frequency(
                time_s,
                working,
                max(minimum_Hz, 0.88 * crossing),
                min(maximum_Hz, 1.12 * crossing),
            )
            return (
                refined,
                f"demoted_second_harmonic_line_ratio_{line_ratio:.3f}",
                crossing,
            )
    resolution = _effective_resolution_fraction(time_s, candidate, active_mask)
    if (
        abs(ratio - 1.0) > max(0.03, resolution)
        and line_ratio >= harmonic_energy_ratio
    ):
        # The gated FFT line is broad; a well-populated modal crossing is the
        # sharper estimator. Refine the peak inside the crossing's neighbourhood
        # rather than silently retaining a sidelobe.
        refined = dominant_frequency(
            time_s,
            working,
            max(minimum_Hz, 0.88 * crossing),
            min(maximum_Hz, 1.12 * crossing),
        )
        return (
            refined,
            f"refined_to_modal_crossing_line_ratio_{line_ratio:.3f}",
            crossing,
        )
    return candidate, f"retained_fft_peak_crossing_ratio_{ratio:.3f}", crossing


def _effective_resolution_fraction(
    time_s: np.ndarray,
    frequency_Hz: float,
    active_mask: np.ndarray | None,
) -> float:
    """Fractional frequency resolution of the record actually transformed.

    Gating to the burst on-period shortens the effective observation time,
    so the spectral line broadens to roughly 1/T_on. Holding such an
    estimate to a tolerance tighter than its own resolution guarantees a
    failed cross-check, which says nothing about the data.
    """

    dt = float(np.median(np.diff(time_s)))
    if active_mask is not None:
        samples = int(np.count_nonzero(np.asarray(active_mask, dtype=bool)))
    else:
        samples = int(np.asarray(time_s).size)
    observation = max(samples * dt, dt)
    if frequency_Hz <= 0.0:
        return 1.0
    return float(1.0 / (observation * frequency_Hz))


def _crossing_spread(
    time_s: np.ndarray,
    values: np.ndarray,
    active_mask: np.ndarray | None,
) -> float | None:
    """Interquartile spread of rising-crossing intervals, as a fraction.

    A sustained carrier gives a tight interval distribution. A short damped
    ring gives a very broad one, and any single 'frequency' quoted for it is
    a summary statistic of a quantity that is not well defined.
    """

    centered = np.asarray(values, dtype=float) - float(np.median(values))
    crossing = (centered[:-1] <= 0) & (centered[1:] > 0)
    if active_mask is not None:
        active = np.asarray(active_mask, dtype=bool)
        crossing &= active[:-1] & active[1:]
    indices = np.flatnonzero(crossing)
    if indices.size < 6:
        return None
    periods = np.diff(np.asarray(time_s)[indices])
    periods = periods[periods > 0]
    if periods.size < 4:
        return None
    frequencies = 1.0 / periods
    median = float(np.median(frequencies))
    if median <= 0.0:
        return None
    spread = float(
        np.percentile(frequencies, 75) - np.percentile(frequencies, 25)
    )
    return spread / median


def _active_run_lengths(active: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    padded = np.pad(np.asarray(active, dtype=np.int8), (1, 1))
    transitions = np.diff(padded)
    starts = np.flatnonzero(transitions == 1)
    stops = np.flatnonzero(transitions == -1)
    return starts, stops, stops - starts


def burst_and_duty(
    time_s: np.ndarray,
    voltage_V: np.ndarray,
    carrier_Hz: float,
    detector: dict[str, Any] | None = None,
) -> tuple[
    float | None,
    float | None,
    float,
    float,
    np.ndarray,
    np.ndarray,
    float | None,
    float | None,
    str,
]:
    """Separate a carrier envelope's repetition rate and on fraction."""

    detector = detector or {}
    dt = float(np.median(np.diff(time_s)))
    smoothing_cycles = float(detector.get("smoothing_cycles", 1.5))
    width = max(3, int(round(smoothing_cycles / (carrier_Hz * dt))))
    envelope = moving_mean(np.abs(voltage_V - np.median(voltage_V)), width)
    low, high = np.percentile(envelope, [10, 90])
    full = max(float(np.percentile(envelope, 99)), np.finfo(float).eps)
    contrast = float((high - low) / full)
    minimum_contrast = float(detector.get("minimum_contrast", 0.12))
    if contrast < minimum_contrast:
        active = np.ones_like(envelope, dtype=bool)
        return (
            None,
            None,
            1.0,
            contrast,
            envelope,
            active,
            None,
            None,
            "continuous_no_burst_detected",
        )
    threshold_fraction = float(detector.get("threshold_fraction", 0.30))
    threshold = float(low + threshold_fraction * (high - low))
    active = envelope >= threshold
    duty = float(np.mean(active))

    centered = envelope - float(np.mean(envelope))
    fs = 1.0 / dt
    maximum = min(0.45 * carrier_Hz, 50000.0, 0.45 * fs)
    minimum = max(100.0, 2.0 / max(float(np.ptp(time_s)), dt))
    if maximum <= minimum:
        return (
            None,
            None,
            duty,
            contrast,
            envelope,
            active,
            threshold,
            None,
            "burst_spectrum_unavailable",
        )
    factor = max(1, int(np.floor(fs / max(8.0 * maximum, 1.0))))
    reduced = _block_mean(centered, factor)
    window = np.hanning(reduced.size)
    spectrum = np.abs(np.fft.rfft(reduced * window))
    frequency = np.fft.rfftfreq(reduced.size, dt * factor)
    band = (frequency >= minimum) & (frequency <= maximum)
    indices = np.flatnonzero(band)
    burst_Hz: float | None = None
    if indices.size:
        local = indices[
            (indices > 0)
            & (indices < spectrum.size - 1)
            & (spectrum[indices] >= spectrum[indices - 1])
            & (spectrum[indices] >= spectrum[indices + 1])
        ]
        if local.size:
            strongest = float(np.max(spectrum[local]))
            significant = local[spectrum[local] >= 0.15 * strongest]
            bin_width = 1.0 / (reduced.size * dt * factor)
            chosen = int(local[np.argmax(spectrum[local])])
            # A clipped/square envelope can put more power at 2f or 3f. Prefer
            # the lowest significant line only when another significant peak
            # corroborates it as a harmonic family.
            for candidate in sorted(significant.tolist()):
                candidate_Hz = candidate * bin_width
                harmonic = any(
                    abs(other * bin_width / candidate_Hz - order) <= 0.08
                    for other in significant
                    for order in (2, 3, 4)
                    if other > candidate
                )
                if harmonic or candidate == chosen:
                    chosen = candidate
                    break
            burst_Hz = float(
                _parabolic_peak(spectrum, chosen) / (reduced.size * dt * factor)
            )

    transitions = np.flatnonzero((~active[:-1]) & active[1:])
    burst_zc: float | None = None
    if transitions.size >= 3 and burst_Hz:
        periods = np.diff(time_s[transitions])
        target = 1.0 / burst_Hz
        valid = (periods > 0.5 * target) & (periods < 1.5 * target)
        if np.count_nonzero(valid) >= 2:
            burst_zc = float(1.0 / np.median(periods[valid]))
    starts, stops, lengths = _active_run_lengths(active)
    complete = (starts > 0) & (stops < active.size)
    on_cycles = (
        float(np.median(lengths[complete]) * dt * carrier_Hz)
        if np.any(complete)
        else None
    )
    minimum_cycles = float(detector.get("minimum_on_cycles", 2.0))
    if burst_Hz is None:
        duty_status = "burst_frequency_not_identified"
    elif on_cycles is None:
        duty_status = "duty_detector_suspect_no_complete_on_window"
    elif on_cycles < minimum_cycles:
        duty_status = "duty_detector_suspect"
    else:
        duty_status = "duty_detector_passed"
    return (
        burst_Hz,
        burst_zc,
        duty,
        contrast,
        envelope,
        active,
        threshold,
        on_cycles,
        duty_status,
    )


def _one_sidedness(values: np.ndarray, active_mask: np.ndarray | None) -> float:
    """Fraction of the signal's excursion that sits on one side of its median.

    0.5 is symmetric. Values far from 0.5 mean a DC offset (or a genuinely
    one-sided waveform) large enough that zero-crossings track the envelope,
    not the carrier — the failure mode where a carrier stops crossing zero.
    """

    series = np.asarray(values, dtype=float)
    if active_mask is not None and np.any(active_mask):
        series = series[np.asarray(active_mask, dtype=bool)]
    if series.size == 0:
        return 0.5
    centered = series - np.median(series)
    span = np.percentile(centered, 99) - np.percentile(centered, 1)
    if span <= 0:
        return 0.5
    return float(np.mean(centered > 0.0))


def autocorrelation_carrier_frequency(
    time_s: np.ndarray,
    values: np.ndarray,
    minimum_Hz: float,
    maximum_Hz: float,
    active_mask: np.ndarray | None = None,
) -> tuple[float | None, str]:
    """Carrier period from the first autocorrelation peak.

    Autocorrelation measures the lag at which the whole waveform best repeats.
    It is immune to the two failures that mislead the other estimators: a DC
    offset (which makes a carrier stop crossing zero, collapsing the zero-cross
    estimate onto the burst envelope) and isolated transient spikes (which
    inflate a naive peak count). The mean is removed first, so any offset is
    irrelevant by construction.

    Returns (frequency_Hz, method_tag). ``None`` when no admissible periodic
    peak stands clearly above the surrounding correlation floor.
    """

    series = np.asarray(values, dtype=float)
    t = np.asarray(time_s, dtype=float)
    if active_mask is not None and np.any(active_mask):
        # Analyse the longest contiguous active run so burst gaps do not inject
        # the burst period into the autocorrelation.
        mask = np.asarray(active_mask, dtype=bool)
        starts, stops, lengths = _active_run_lengths(mask)
        if lengths.size:
            longest = int(np.argmax(lengths))
            lo, hi = int(starts[longest]), int(stops[longest])
            if hi - lo > 16:
                series = series[lo:hi]
                t = t[lo:hi]
    if series.size < 32:
        return None, "autocorr_insufficient_samples"

    dt = float(np.median(np.diff(t)))
    if dt <= 0:
        return None, "autocorr_bad_timebase"

    # Remove slow baseline (DC + burst envelope) with a moving average that is
    # slow relative to the fastest admissible carrier but fast relative to the
    # burst. Without this, the autocorrelation of a one-sided record locks onto
    # the envelope period instead of the carrier.
    fastest_period_samples = max(1.0, 1.0 / (maximum_Hz * dt))
    smooth = int(round(3.0 * fastest_period_samples)) | 1
    if 3 <= smooth < series.size // 2:
        kernel = np.ones(smooth) / smooth
        baseline = np.convolve(series, kernel, mode="same")
        series = series - baseline
    else:
        series = series - series.mean()
    norm = float(np.dot(series, series))
    if norm <= 0:
        return None, "autocorr_zero_power"

    ac = np.correlate(series, series, mode="full")[series.size - 1 :]
    ac = ac / ac[0]

    lag_min = max(1, int(np.floor(1.0 / (maximum_Hz * dt))))
    lag_max = int(np.ceil(1.0 / (minimum_Hz * dt)))
    lag_max = min(lag_max, ac.size - 2)
    if lag_max <= lag_min + 1:
        return None, "autocorr_band_too_narrow"

    window = ac[lag_min : lag_max + 1]
    # First strong local maximum: rises above its neighbours and clears the
    # local correlation floor, so a slow envelope roll-off is not mistaken for
    # a period.
    interior = np.flatnonzero(
        (window[1:-1] >= window[:-2]) & (window[1:-1] > window[2:])
    )
    best_lag = None
    best_corr = 0.0
    minimum_corr = 0.10  # a real periodicity repeats at well above noise level
    for rel in interior:
        idx = rel + 1
        corr = float(window[idx])
        if corr < minimum_corr:
            continue
        local_floor = float(np.median(window[max(0, idx - 20) : idx + 20]))
        if corr <= local_floor + 0.05:
            continue
        # Take the FIRST qualifying peak, not the globally tallest: the carrier
        # fundamental is the shortest repeat lag, while harmonics or the burst
        # period can correlate higher at longer lags.
        best_lag = lag_min + idx
        best_corr = corr
        break
    if best_lag is None:
        return None, "autocorr_no_admissible_peak"

    refined = _parabolic_peak(ac, best_lag)
    if refined <= 0:
        return None, "autocorr_bad_refinement"
    frequency = 1.0 / (refined * dt)
    if not (minimum_Hz <= frequency <= maximum_Hz):
        return None, f"autocorr_out_of_band_{frequency / 1e3:.0f}kHz"
    return frequency, f"autocorr_peak_corr_{best_corr:.2f}"


def estimate_frequencies(waveform: Waveform, config: dict[str, Any]) -> FrequencyEstimate:
    analysis = config["analysis"]
    minimum = float(analysis.get("minimum_carrier_hz", 1000.0))
    maximum = float(analysis.get("maximum_carrier_hz", 300000.0))
    estimator = analysis.get("carrier_estimator", {})
    preliminary, preliminary_decision, _ = harmonic_aware_carrier_frequency(
        waveform.time_s,
        waveform.applied_voltage_V,
        minimum,
        maximum,
        harmonic_energy_ratio=float(
            estimator.get("f0_vs_half_line_amplitude_ratio", 0.20)
        ),
    )
    (
        _,
        _,
        _,
        _,
        preliminary_envelope,
        preliminary_active,
        _,
        _,
        _,
    ) = burst_and_duty(
        waveform.time_s,
        waveform.applied_voltage_V,
        preliminary,
        analysis.get("duty_detector", {}),
    )
    use_gate = bool(
        estimator.get("gate_on_period", True)
        and np.any(preliminary_active)
        and np.mean(preliminary_active) < 0.98
    )
    carrier, harmonic_decision, _ = harmonic_aware_carrier_frequency(
        waveform.time_s,
        waveform.applied_voltage_V,
        minimum,
        maximum,
        active_mask=preliminary_active if use_gate else None,
        harmonic_energy_ratio=float(
            estimator.get("f0_vs_half_line_amplitude_ratio", 0.20)
        ),
    )
    (
        burst,
        burst_zc,
        duty,
        contrast,
        envelope,
        active,
        envelope_threshold,
        burst_on_cycles,
        duty_status,
    ) = burst_and_duty(
        waveform.time_s,
        waveform.applied_voltage_V,
        carrier,
        analysis.get("duty_detector", {}),
    )
    crossing = zero_cross_frequency(
        waveform.time_s,
        waveform.applied_voltage_V,
        carrier,
        active_mask=active if burst is not None else None,
    )
    disagreement = None if crossing is None else abs(crossing - carrier) / carrier
    configured_tolerance = float(
        analysis.get("frequency_crosscheck_tolerance", 0.03)
    )
    resolution_fraction = _effective_resolution_fraction(
        waveform.time_s, carrier, active if use_gate else None
    )
    effective_tolerance = max(configured_tolerance, resolution_fraction)
    status = "crosscheck_reported"
    crossing_spread = _crossing_spread(
        waveform.time_s, waveform.applied_voltage_V, active if use_gate else None
    )
    minimum_cycles = float(
        estimator.get("carrier_minimum_cycles_for_definition", 3.0)
    )
    spread_limit = float(estimator.get("carrier_crossing_spread_limit", 0.5))
    autocorr_carrier, autocorr_method = autocorrelation_carrier_frequency(
        waveform.time_s,
        waveform.applied_voltage_V,
        minimum,
        maximum,
        active_mask=active if burst is not None else None,
    )
    autocorr_disagreement = (
        abs(autocorr_carrier - carrier) / carrier
        if autocorr_carrier is not None and carrier > 0
        else None
    )
    one_sidedness = _one_sidedness(
        waveform.applied_voltage_V, active if burst is not None else None
    )
    offset_limit = float(estimator.get("offset_one_sidedness_limit", 0.35))
    offset_suspected = abs(one_sidedness - 0.5) > offset_limit
    autocorr_tolerance = max(
        configured_tolerance,
        float(estimator.get("autocorr_crosscheck_tolerance", 0.06)),
    )
    autocorr_agrees = (
        autocorr_disagreement is not None
        and autocorr_disagreement <= autocorr_tolerance
    )
    ill_defined_reasons: list[str] = []
    if burst_on_cycles is not None and burst_on_cycles < minimum_cycles:
        ill_defined_reasons.append(f"on_cycles_{burst_on_cycles:.1f}")
    if crossing_spread is not None and crossing_spread > spread_limit:
        ill_defined_reasons.append(f"crossing_iqr_{crossing_spread:.2f}")
    if crossing is None and autocorr_carrier is None:
        status = "carrier_estimate_only_no_crosscheck"
    elif offset_suspected and autocorr_agrees:
        # DC offset makes zero-crossings track the envelope, not the carrier.
        # Autocorrelation is offset-immune, so when it corroborates the FFT the
        # carrier is trustworthy even though the zero-cross check would fail.
        status = "verified_fft_and_autocorrelation_offset_signal"
    elif disagreement is not None and disagreement <= configured_tolerance:
        status = "verified_fft_and_zero_cross"
    elif disagreement is not None and disagreement <= effective_tolerance:
        status = "verified_within_resolution_limit"
    elif autocorr_agrees:
        # Autocorrelation is offset- and spike-immune. When it corroborates the
        # FFT the carrier is trustworthy even if the zero-cross check disagrees
        # (short ring, transient-rich, or one-sided records all break zero-cross
        # while leaving the true period intact in the autocorrelation).
        status = "verified_fft_and_autocorrelation"
    elif ill_defined_reasons and not autocorr_agrees:
        # Independent estimators disagree *and* the record is a short damped
        # ring whose gated spectrum is a comb at the burst rate. The carrier
        # is not a well-defined quantity here; agreement above would have
        # settled it, so report the regime rather than arbitrating.
        status = "carrier_ill_defined_short_ring"
    else:
        status = "crosscheck_offset_exceeds_resolution"
    ratio = (
        crossing / carrier
        if crossing is not None and carrier > 0
        else None
    )
    if ratio is None:
        diagnostic = "zero_cross_unavailable"
    elif status == "verified_fft_and_autocorrelation_offset_signal":
        diagnostic = (
            f"offset_one_sided_{100.0 * one_sidedness:.0f}pct_"
            f"zero_cross_tracks_envelope_autocorr_{autocorr_method}"
        )
    elif status.startswith("verified_"):
        diagnostic = "crosscheck_within_tolerance"
    elif ill_defined_reasons:
        diagnostic = "carrier_ill_defined_" + ",".join(ill_defined_reasons)
    elif 1.85 <= ratio <= 2.15:
        diagnostic = "zero_cross_near_2xf0_ringdown_or_crossing_doubling"
    elif 0.45 <= ratio <= 0.55:
        diagnostic = "zero_cross_near_f0_over_2_subharmonic_lock"
    elif disagreement is not None and disagreement > effective_tolerance:
        diagnostic = (
            f"crosscheck_offset_{100.0 * disagreement:.2f}_percent"
            f"_vs_resolution_{100.0 * resolution_fraction:.2f}_percent"
        )
    elif disagreement is not None and disagreement > configured_tolerance:
        diagnostic = (
            f"within_resolution_limit_offset_{100.0 * disagreement:.2f}_percent"
            f"_resolution_{100.0 * resolution_fraction:.2f}_percent"
        )
    else:
        diagnostic = "crosscheck_within_tolerance"
    burst_disagreement = (
        abs(burst_zc - burst) / burst
        if burst is not None and burst_zc is not None and burst > 0
        else None
    )
    if burst is None:
        burst_status = "continuous_no_burst_detected"
    elif burst_zc is None:
        burst_status = "spectral_burst_only"
    elif burst_disagreement is not None and burst_disagreement > 0.10:
        burst_status = "spectral_burst_retained_crosscheck_failed"
    else:
        burst_status = "verified_spectral_and_envelope_edges"
    return FrequencyEstimate(
        carrier_Hz=carrier,
        zero_cross_Hz=crossing,
        relative_disagreement=disagreement,
        carrier_status=status,
        carrier_method=(
            "gated_on_period_harmonic_aware_fft"
            if use_gate
            else "whole_record_harmonic_aware_fft"
        ),
        harmonic_decision=(
            harmonic_decision
            if harmonic_decision
            else preliminary_decision
        ),
        crosscheck_diagnostic=diagnostic,
        autocorr_carrier_Hz=autocorr_carrier,
        autocorr_relative_disagreement=autocorr_disagreement,
        offset_one_sidedness=one_sidedness,
        burst_Hz=burst,
        burst_zero_cross_Hz=burst_zc,
        burst_relative_disagreement=burst_disagreement,
        burst_status=burst_status,
        duty_on_fraction=duty,
        burst_on_cycles=burst_on_cycles,
        duty_status=duty_status,
        envelope_contrast=contrast,
        envelope_threshold=envelope_threshold,
        envelope=envelope,
        active_mask=active,
    )


def complex_amplitude(
    time_s: np.ndarray,
    values: np.ndarray,
    frequency_Hz: float,
    *,
    window: bool = True,
) -> complex:
    centered_t = time_s - float(np.mean(time_s))
    centered_y = values - float(np.mean(values))
    denom = float(np.dot(centered_t, centered_t))
    if denom:
        centered_y = centered_y - float(np.dot(centered_t, centered_y) / denom) * centered_t
    weights = np.hanning(values.size) if window else np.ones(values.size)
    basis = np.exp(-2j * np.pi * frequency_Hz * time_s)
    normalizer = float(np.sum(weights))
    return complex(2.0 * np.sum(weights * centered_y * basis) / normalizer)


def complex_capacitance(
    time_s: np.ndarray,
    voltage_kV: np.ndarray,
    charge_nC: np.ndarray,
    frequency_Hz: float,
) -> complex:
    vhat = complex_amplitude(time_s, voltage_kV, frequency_Hz)
    qhat = complex_amplitude(time_s, charge_nC, frequency_Hz)
    if abs(vhat) <= np.finfo(float).eps:
        return complex(np.nan, np.nan)
    # nC/kV is numerically pF.
    return qhat / vhat


def time_domain_slope_pF(voltage_kV: np.ndarray, charge_nC: np.ndarray) -> float | None:
    voltage = voltage_kV - float(np.mean(voltage_kV))
    charge = charge_nC - float(np.mean(charge_nC))
    denom = float(np.dot(voltage, voltage))
    if denom <= np.finfo(float).eps:
        return None
    return float(np.dot(voltage, charge) / denom)


def harmonic_ratio(
    time_s: np.ndarray,
    values: np.ndarray,
    frequency_Hz: float,
    maximum_harmonic: int = 5,
) -> float | None:
    fundamental = abs(complex_amplitude(time_s, values, frequency_Hz))
    if fundamental <= np.finfo(float).eps:
        return None
    harmonics = [
        abs(complex_amplitude(time_s, values, order * frequency_Hz))
        for order in range(2, maximum_harmonic + 1)
        if order * frequency_Hz < 0.48 / float(np.median(np.diff(time_s)))
    ]
    return float(np.sqrt(np.sum(np.square(harmonics))) / fundamental) if harmonics else 0.0


def _cyclic_energy_uJ(voltage_kV: np.ndarray, charge_nC: np.ndarray) -> float:
    return float(
        0.5
        * np.sum(
            voltage_kV
            * (np.roll(charge_nC, -1) - np.roll(charge_nC, 1))
        )
    )


def _linear_intercept(x0: float, x1: float, y0: float, y1: float) -> float:
    denom = x1 - x0
    if abs(denom) <= np.finfo(float).eps:
        return float(0.5 * (y0 + y1))
    fraction = -x0 / denom
    return float(y0 + fraction * (y1 - y0))


def _smooth_circular(values: np.ndarray, width: int = 7) -> np.ndarray:
    width = max(1, min(int(width), values.size))
    if width <= 1:
        return values
    left = width // 2
    padded = np.concatenate([values[-left:], values, values[: width - left - 1]])
    kernel = np.ones(width) / width
    return np.convolve(padded, kernel, mode="valid")


def branch_slopes_pF(
    voltage_kV: np.ndarray,
    charge_nC: np.ndarray,
) -> tuple[float | None, float | None]:
    """Split local Q–V slopes into low/high branch populations."""

    if voltage_kV.size < 40:
        return None, None
    voltage = _smooth_circular(voltage_kV, 9)
    charge = _smooth_circular(charge_nC, 9)
    dV = np.gradient(voltage)
    dQ = np.gradient(charge)
    threshold = float(np.percentile(np.abs(dV), 35))
    valid = np.isfinite(dV) & np.isfinite(dQ) & (np.abs(dV) > max(threshold, 1e-9))
    slopes = dQ[valid] / dV[valid]
    finite = slopes[np.isfinite(slopes)]
    if finite.size < 20:
        return None, None
    lower, upper = np.percentile(finite, [20, 80])
    centers = np.array([lower, upper], dtype=float)
    for _ in range(30):
        distance = np.abs(finite[:, None] - centers[None, :])
        label = np.argmin(distance, axis=1)
        updated = np.array(
            [
                np.median(finite[label == group]) if np.any(label == group) else centers[group]
                for group in range(2)
            ]
        )
        if np.allclose(updated, centers, rtol=1e-5, atol=1e-6):
            centers = updated
            break
        centers = updated
    low, high = sorted(float(value) for value in centers)
    if low <= 0 or high <= low * 1.05:
        return None, None
    return low, high


def cycle_loop_metrics(
    time_s: np.ndarray,
    voltage_kV: np.ndarray,
    charge_nC: np.ndarray,
    carrier_Hz: float,
) -> LoopMetrics:
    """Compute per-carrier-cycle signed energy and zero-voltage intercept width."""

    centered = voltage_kV - float(np.median(voltage_kV))
    rising = np.flatnonzero((centered[:-1] <= 0) & (centered[1:] > 0))
    if rising.size < 2:
        return LoopMetrics(
            *(np.array([], dtype=float) for _ in range(5)),
            0.0,
            np.array([], dtype=float),
            np.array([], dtype=float),
            np.array([], dtype=float),
            None,
            None,
        )
    period = 1.0 / carrier_Hz
    peak_span = max(float(np.percentile(voltage_kV, 99) - np.percentile(voltage_kV, 1)), 1e-12)
    cycle_rows: list[tuple[float, float, float, float, float, np.ndarray, np.ndarray]] = []
    for left, right in zip(rising[:-1], rising[1:]):
        duration = float(time_s[right] - time_s[left])
        if duration < 0.70 * period or duration > 1.30 * period or right - left < 8:
            continue
        v = voltage_kV[left : right + 1]
        q = charge_nC[left : right + 1]
        if float(np.ptp(v)) < 0.25 * peak_span:
            continue
        local_centered = v - float(np.median(voltage_kV))
        falling = np.flatnonzero((local_centered[:-1] >= 0) & (local_centered[1:] < 0))
        if not falling.size:
            continue
        fall = int(falling[np.argmin(np.abs(falling - 0.5 * (v.size - 1)))])
        q_start = _linear_intercept(
            float(local_centered[0]),
            float(local_centered[min(1, v.size - 1)]),
            float(q[0]),
            float(q[min(1, q.size - 1)]),
        )
        q_fall = _linear_intercept(
            float(local_centered[fall]),
            float(local_centered[fall + 1]),
            float(q[fall]),
            float(q[fall + 1]),
        )
        q_end = _linear_intercept(
            float(local_centered[-2]),
            float(local_centered[-1]),
            float(q[-2]),
            float(q[-1]),
        )
        q_rising = 0.5 * (q_start + q_end)
        energy = _cyclic_energy_uJ(v, q)
        intercept = abs(q_fall - q_rising)
        positive_half = q_fall - q_start
        negative_half = q_end - q_fall
        midpoint = 0.5 * float(time_s[left] + time_s[right])
        phase = np.linspace(0.0, 1.0, v.size)
        grid = np.linspace(0.0, 1.0, 256, endpoint=False)
        cycle_rows.append(
            (
                energy,
                intercept,
                positive_half,
                negative_half,
                midpoint,
                np.interp(grid, phase, v),
                np.interp(grid, phase, q),
            )
        )
    if not cycle_rows:
        return LoopMetrics(
            *(np.array([], dtype=float) for _ in range(5)),
            0.0,
            np.array([], dtype=float),
            np.array([], dtype=float),
            np.array([], dtype=float),
            None,
            None,
        )
    rows = cycle_rows
    loop_v = np.median(np.vstack([row[5] for row in rows]), axis=0)
    loop_q = np.median(np.vstack([row[6] for row in rows]), axis=0)
    ccell, cd = branch_slopes_pF(loop_v, loop_q)
    return LoopMetrics(
        energy_signed_uJ=np.asarray([row[0] for row in rows]),
        dQ_intercept_nC=np.asarray([row[1] for row in rows]),
        positive_half_dQ_nC=np.asarray([row[2] for row in rows]),
        negative_half_dQ_nC=np.asarray([row[3] for row in rows]),
        cycle_midpoint_s=np.asarray([row[4] for row in rows]),
        active_cycle_fraction=float(len(rows) / max(rising.size - 1, 1)),
        loop_phase=np.linspace(0.0, 1.0, 256, endpoint=False),
        loop_voltage_kV=loop_v,
        loop_charge_nC=loop_q,
        Ccell_branch_pF=ccell,
        Cd_branch_pF=cd,
    )


def burst_period_metrics(
    time_s: np.ndarray,
    voltage_kV: np.ndarray,
    charge_nC: np.ndarray,
    active_mask: np.ndarray,
    burst_Hz: float | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Signed Q–V energy and duty using the detector's exact active mask."""

    if not burst_Hz or burst_Hz <= 0:
        return np.array([], dtype=float), np.array([], dtype=float)
    active = np.asarray(active_mask, dtype=bool)
    if active.shape != charge_nC.shape:
        raise ValueError("active_mask must match the waveform shape")
    starts = np.flatnonzero((~active[:-1]) & active[1:]) + 1
    if starts.size < 2:
        return np.array([], dtype=float), np.array([], dtype=float)
    expected = 1.0 / burst_Hz
    energies: list[float] = []
    duties: list[float] = []
    for left, right in zip(starts[:-1], starts[1:]):
        duration = float(time_s[right] - time_s[left])
        if duration < 0.65 * expected or duration > 1.35 * expected or right - left < 8:
            continue
        energies.append(_cyclic_energy_uJ(voltage_kV[left : right + 1], charge_nC[left : right + 1]))
        duties.append(float(np.mean(active[left : right + 1])))
    return np.asarray(energies), np.asarray(duties)


def band_limited_charge(
    time_s: np.ndarray,
    charge_nC: np.ndarray,
    maximum_Hz: float,
) -> np.ndarray:
    """Zero-phase FFT low-pass used by derivative-based cross-checks."""

    if charge_nC.size < 16:
        return np.asarray(charge_nC, dtype=float).copy()
    dt = float(np.median(np.diff(time_s)))
    nyquist = 0.5 / dt
    cutoff = min(max(float(maximum_Hz), 0.0), 0.95 * nyquist)
    if cutoff <= 0:
        raise ValueError("maximum_Hz must be positive")
    pad = min(charge_nC.size - 1, max(16, charge_nC.size // 20))
    median = float(np.median(charge_nC))
    centered = np.asarray(charge_nC, dtype=float) - median
    padded = np.pad(centered, pad, mode="reflect")
    spectrum = np.fft.rfft(padded)
    frequency = np.fft.rfftfreq(padded.size, dt)
    lower = 0.85 * cutoff
    response = np.ones_like(frequency)
    response[frequency >= cutoff] = 0.0
    transition = (frequency > lower) & (frequency < cutoff)
    response[transition] = 0.5 * (
        1.0
        + np.cos(
            np.pi * (frequency[transition] - lower) / max(cutoff - lower, 1e-30)
        )
    )
    filtered = np.fft.irfft(spectrum * response, n=padded.size)
    return filtered[pad : pad + charge_nC.size] + median


def gross_charge_rate_tv_C_s(
    time_s: np.ndarray,
    charge_nC: np.ndarray,
    carrier_Hz: float,
    *,
    maximum_harmonic: float = 3.0,
    correction_factor: float | None = None,
) -> float | None:
    """Segmentation-free, band-limited total variation of measured charge."""

    duration = float(np.ptp(time_s))
    if duration <= 0 or charge_nC.size < 2:
        return None
    filtered = band_limited_charge(
        time_s,
        charge_nC,
        maximum_Hz=float(maximum_harmonic) * float(carrier_Hz),
    )
    total_nC = float(np.sum(np.abs(np.diff(filtered))))
    if correction_factor is not None:
        total_nC *= float(correction_factor)
    return total_nC * 1.0e-9 / duration


def chain_dissipation_estimate(
    time_s: np.ndarray,
    charge_nC: np.ndarray,
    carrier_Hz: float,
    resistance_ohm: float,
    *,
    maximum_harmonic: float = 3.0,
) -> tuple[float | None, float | None]:
    """Estimate I_rms from band-limited dQ/dt and return I²R chain power."""

    if charge_nC.size < 3:
        return None, None
    filtered = band_limited_charge(
        time_s,
        charge_nC,
        maximum_Hz=float(maximum_harmonic) * float(carrier_Hz),
    )
    current_A = np.gradient(filtered * 1.0e-9, time_s)
    finite = current_A[np.isfinite(current_A)]
    if not finite.size:
        return None, None
    rms = float(np.sqrt(np.mean(np.square(finite))))
    return rms, float(rms * rms * float(resistance_ohm))


def retained_charge(
    charge_nC: np.ndarray,
    envelope: np.ndarray,
) -> tuple[float | None, str]:
    """Measure terminal retained charge only when both record edges are quiet."""

    n = charge_nC.size
    width = max(8, int(0.05 * n))
    low, high = np.percentile(envelope, [10, 90])
    threshold = float(low + 0.20 * (high - low))
    first_quiet = float(np.mean(envelope[:width] < threshold))
    last_quiet = float(np.mean(envelope[-width:] < threshold))
    middle_active = float(np.mean(envelope[width:-width] >= threshold)) if n > 2 * width else 0.0
    if first_quiet < 0.8 or last_quiet < 0.8 or middle_active < 0.05:
        return None, "not_measured_no_quiet_record_edges"
    value = float(np.mean(charge_nC[-width:]) - np.mean(charge_nC[:width]))
    return value, "measured_quiet_edges"


def multiline_capacitance_points(
    time_s: np.ndarray,
    voltage_kV: np.ndarray,
    charge_nC: np.ndarray,
    *,
    minimum_Hz: float = 4000.0,
    maximum_Hz: float = 175000.0,
    relative_threshold: float = 0.02,
    max_points: int = 24,
) -> list[dict[str, float]]:
    """Extract significant same-record spectral transfer lines for 7_20."""

    dt = float(np.median(np.diff(time_s)))
    window = np.hanning(time_s.size)
    v = np.fft.rfft((voltage_kV - np.mean(voltage_kV)) * window)
    q = np.fft.rfft((charge_nC - np.mean(charge_nC)) * window)
    frequency = np.fft.rfftfreq(time_s.size, dt)
    magnitude = np.abs(v)
    band = (frequency >= minimum_Hz) & (frequency <= maximum_Hz)
    candidates = np.flatnonzero(
        band
        & (magnitude >= relative_threshold * max(float(np.max(magnitude[band])), 1e-15))
    )
    if candidates.size:
        local = candidates[
            (candidates > 0)
            & (candidates < magnitude.size - 1)
            & (magnitude[candidates] >= magnitude[candidates - 1])
            & (magnitude[candidates] >= magnitude[candidates + 1])
        ]
    else:
        local = candidates
    ranked = sorted(local.tolist(), key=lambda index: float(magnitude[index]), reverse=True)
    selected: list[int] = []
    for index in ranked:
        if all(abs(index - prior) >= 2 for prior in selected):
            selected.append(index)
        if len(selected) >= max_points:
            break
    rows: list[dict[str, float]] = []
    for index in sorted(selected):
        if abs(v[index]) <= np.finfo(float).eps:
            continue
        cstar = q[index] / v[index]
        rows.append(
            {
                "frequency_kHz": float(frequency[index] / 1000.0),
                "Cmag_pF": float(abs(cstar)),
                "Creal_pF": float(cstar.real),
                "Cimag_pF": float(-cstar.imag),
                "phase_deg": float(np.degrees(np.angle(cstar))),
                "relative_voltage_line": float(magnitude[index] / np.max(magnitude[band])),
            }
        )
    return rows


def current_sign(
    time_s: np.ndarray,
    current_A: np.ndarray | None,
    charge_nC: np.ndarray,
    frequency_Hz: float,
) -> tuple[int | None, float | None]:
    """Choose the Pearson sign closest to I = dQ/dt and report phase error."""

    if current_A is None:
        return None, None
    ihat = complex_amplitude(time_s, current_A, frequency_Hz)
    qhat_C = complex_amplitude(time_s, charge_nC * 1.0e-9, frequency_Hz)
    expected = 1j * 2.0 * np.pi * frequency_Hz * qhat_C
    if abs(ihat) <= np.finfo(float).eps or abs(expected) <= np.finfo(float).eps:
        return None, None
    raw_error = float(np.degrees(np.angle(ihat / expected)))
    flipped_error = float(np.degrees(np.angle(-ihat / expected)))
    wrap = lambda value: (value + 180.0) % 360.0 - 180.0
    raw_error = wrap(raw_error)
    flipped_error = wrap(flipped_error)
    if abs(flipped_error) < abs(raw_error):
        return -1, flipped_error
    return 1, raw_error
