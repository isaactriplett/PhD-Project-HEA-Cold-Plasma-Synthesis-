"""End-to-end measured-frequency Lissajous v2 quantification.

Example:

    python -m lissajous.quantify --config config.yaml \
        --data-root "C:/.../waveforms/July 2026" \
        --out outputs/lissajous_v2
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .config import load_config, output_config, resolved_data_root, resolved_output
from .io import (
    SourceCatalog,
    SourceRef,
    Waveform,
    discover_sources,
    parse_path_metadata,
    read_waveform,
)
from .report import (
    assert_no_retired_numbers,
    condition_summary_rows,
    detect_onsets,
    discharge_summary_rows,
    factor_frequency_rows,
    frequency_consistency_rows,
    presentation_numbers,
    results_markdown,
    stats,
    synthesis_summary_rows,
    write_csv,
    write_json,
)
from .signal import (
    burst_period_metrics,
    chain_dissipation_estimate,
    complex_capacitance,
    current_sign,
    cycle_loop_metrics,
    estimate_frequencies,
    gross_charge_rate_tv_C_s,
    harmonic_ratio,
    multiline_capacitance_points,
    retained_charge,
    time_domain_slope_pF,
)

ELEMENTARY_CHARGE_C = 1.602176634e-19


PER_CAPTURE_STABLE_COLUMNS = [
    "cond",
    "capture",
    "f0_kHz",
    "Vamp_kV",
    "Cline_pF",
    "Creal_pF",
    "Cimag_pF",
    "phase_deg",
    "Clobe_pF",
    "codesD",
    "codesA",
    "Ipk_mA",
    "medium",
    "freq_label",
    "level_pct",
    "seg",
    "band_tag",
    "clipA",
    "clipD",
    "dc_offset",
    "dQ_cycle_nC",
    "dQ_gap_nC",
    "U_cycle_uJ",
    "P_W",
    "Cd_pF",
    "Ccell_pF",
    "syst_frac",
]

PER_CAPTURE_EXTRA_COLUMNS = [
    "path",
    "source_uri",
    "dataset_type",
    "source_type",
    "run_key",
    "display_label",
    "medium_display",
    "burst_frequency_label_kHz",
    "save_idx",
    "f0_zero_cross_kHz",
    "f0_relative_disagreement",
    "f0_autocorr_kHz",
    "f0_autocorr_disagreement",
    "offset_one_sidedness",
    "frequency_status",
    "carrier_estimator_method",
    "harmonic_decision",
    "frequency_crosscheck_diagnostic",
    "lf_tag_evidence",
    "burst_Hz",
    "burst_zero_cross_Hz",
    "burst_relative_disagreement",
    "burst_status",
    "duty_on_fraction",
    "burst_on_cycles",
    "duty_status",
    "envelope_contrast",
    "envelope_threshold_V",
    "q_sign",
    "condition_q_sign",
    "polarity_lock_status",
    "polarity_lock_agreement",
    "raw_phase_deg",
    "raw_U_cycle_signed_uJ",
    "raw_U_burst_signed_uJ",
    "polarity_flipped_by_audit",
    "polarity_audit_action",
    "current_sign",
    "current_phase_error_deg",
    "charge_lsb_nC",
    "Cline_quantization_uncertainty_pF",
    "harmonic_ratio",
    "active_cycle_fraction",
    "U_cycle_signed_uJ",
    "U_burst_uJ",
    "U_burst_signed_uJ",
    "P_method",
    "I_charge_rms_A",
    "P_chain_est_W",
    "P_chain_est_low_W",
    "P_chain_est_high_W",
    "dQ_positive_nC",
    "dQ_negative_nC",
    "positive_rate_C_s",
    "negative_rate_C_s",
    "gross_rate_C_s",
    "gross_rate_C_s_tv",
    "gross_rate_tv_ratio",
    "net_rate_C_s",
    "gross_rate_C_min",
    "positive_average_equivalent_flow_per_s",
    "negative_average_equivalent_flow_per_s",
    "positive_peak_halfcycle_average_equivalent_flow_per_s",
    "negative_peak_halfcycle_average_equivalent_flow_per_s",
    "capture_gross_dose_C",
    "dose_20min_C",
    "retained_charge_nC",
    "retained_charge_status",
    "F_same_band",
    "charge_correction_factor",
    "orientation_status",
    "phase_status",
    "quantization_status",
    "capacitance_valid",
    "excluded",
    "exclusion_flags",
]

MANIFEST_COLUMNS = [
    "path",
    "source_uri",
    "storage",
    "size_bytes",
    "signature",
    "dataset_type",
    "source_type",
    "medium",
    "medium_display",
    "run_key",
    "freq_label",
    "nominal_frequency_kHz",
    "burst_frequency_label_kHz",
    "level_pct",
    "commanded_kV",
    "save_idx",
    "seg_idx",
    "n_samples",
    "dt_s",
    "duration_s",
    "channel_labels",
    "channel_units",
    "channel_mapping",
    "clipA",
    "clipC",
    "clipD",
    "codesA",
    "codesC",
    "codesD",
    "lsbA_raw",
    "lsbD_raw",
    "f0_Hz",
    "f0_zero_cross_Hz",
    "f0_relative_disagreement",
    "f0_autocorr_kHz",
    "f0_autocorr_disagreement",
    "offset_one_sidedness",
    "frequency_status",
    "carrier_estimator_method",
    "harmonic_decision",
    "frequency_crosscheck_diagnostic",
    "lf_tag_evidence",
    "burst_Hz",
    "burst_zero_cross_Hz",
    "burst_relative_disagreement",
    "burst_status",
    "duty_on_fraction",
    "burst_on_cycles",
    "duty_status",
    "band_tag",
    "phase_deg",
    "dc_offset",
    "drift_V_per_s",
    "detrended",
    "contaminated",
    "excluded",
    "exclusion_flags",
    "parse_status",
    "parse_error",
]


def parse_arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Quantify charge, transfer capacitance, power, duty, and QC from July waveforms."
    )
    parser.add_argument("--config", default="config.yaml", help="JSON-compatible YAML config")
    parser.add_argument("--data-root", help="Override paths.data_root")
    parser.add_argument("--out", help="Override paths.default_output")
    parser.add_argument(
        "--no-archives",
        action="store_true",
        help="Read extracted CSVs even when a matching top-level ZIP mirror exists",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        help="Development/diagnostic cap after deterministic path sorting",
    )
    parser.add_argument(
        "--path-regex",
        help="Analyze only logical paths matching this regular expression (diagnostic)",
    )
    parser.add_argument(
        "--no-multiline",
        action="store_true",
        help="Skip 7_20 spectral-line extraction (fundamental metrics still run)",
    )
    parser.add_argument("--quiet", action="store_true", help="Suppress progress messages")
    return parser.parse_args(argv)


def band_tag(
    frequency_Hz: float,
    config: dict[str, Any],
    *,
    carrier_evidence: str | None = None,
) -> str:
    """Tag from the measured electrical carrier, never the folder label.

    An ``LF-geometric`` tag is a physical claim: it licenses the word
    "geometric" downstream. It is therefore awarded only when a
    sub-threshold carrier estimate is corroborated. ``carrier_evidence``
    carries that verdict from :func:`lf_tag_evidence`; anything other than
    ``"verified"`` demotes the capture to ``carrier-transfer`` under the
    configured ``transition_band_policy``. Callers without evidence (unit
    tests, spectral-line rows) keep the plain frequency rule.
    """

    maximum = float(config["analysis"]["lf_geometric_max_hz"])
    if frequency_Hz > maximum:
        return "carrier-transfer"
    if carrier_evidence is None or carrier_evidence == "verified":
        return "LF-geometric"
    return "carrier-transfer"


def lf_tag_evidence(
    freq: FrequencyEstimate,
    quantization_status: str,
    config: dict[str, Any],
) -> str:
    """Corroboration verdict for a sub-threshold carrier estimate.

    Returns ``"verified"`` or ``lf_tag_unverified:<reasons>``. A single
    degenerate record must not be able to mint a low-frequency anchor that
    the supply is not known to produce. The criteria are resolution
    (enough ADC codes to trust the estimate) and corroboration (an
    independent zero-crossing estimate that agrees). A continuous,
    un-gated record is *not* disqualified: a genuine LF anchor
    acquisition would be exactly that.
    """

    reasons: list[str] = []
    if quantization_status == "quantization_limited":
        reasons.append("quantization_limited")
    if freq.zero_cross_Hz is None:
        reasons.append("no_zero_cross_corroboration")
    else:
        tolerance = float(
            config["analysis"].get("frequency_crosscheck_tolerance", 0.03)
        )
        if (
            freq.relative_disagreement is not None
            and freq.relative_disagreement > tolerance
        ):
            reasons.append(
                f"crosscheck_offset_{100.0 * freq.relative_disagreement:.1f}pct"
            )
    if not reasons:
        return "verified"
    return "lf_tag_unverified:" + ",".join(reasons)


def _median(values: np.ndarray) -> float | None:
    finite = values[np.isfinite(values)]
    return float(np.median(finite)) if finite.size else None


def _phase_status(phase_deg: float) -> str:
    wrapped = (phase_deg + 180.0) % 360.0 - 180.0
    if abs(wrapped) <= 45.0:
        return "polarity_consistent_near_zero"
    if abs(abs(wrapped) - 180.0) <= 45.0:
        return "polarity_inverted_near_180"
    return "reactive_or_transfer_phase"


def _orientation(
    signed_energy: float | None,
    *,
    active: bool,
) -> tuple[float | None, str]:
    if signed_energy is None:
        return None, "not_available_no_complete_cycles"
    if signed_energy < 0:
        if active:
            return None, "failed_negative_median_energy"
        return None, "sub_onset_reactive_loop_energy_ineligible"
    return signed_energy, "passed_nonnegative_energy"


def _power_from_energies(
    *,
    cycle_energy_uJ: float | None,
    burst_energy_uJ: float | None,
    carrier_Hz: float | None,
    burst_Hz: float | None,
) -> tuple[float | None, str]:
    """Select one explicit power estimator without a burst-to-cycle fallback."""

    if burst_Hz is not None and float(burst_Hz) > 0:
        if burst_energy_uJ is None:
            return None, "burst_power_withheld_no_valid_burst_energy"
        return (
            float(burst_energy_uJ) * float(burst_Hz) * 1.0e-6,
            "burst_shoelace_times_measured_burst_Hz",
        )
    if cycle_energy_uJ is None or carrier_Hz is None:
        return None, "cycle_power_withheld_no_valid_cycle_energy_or_carrier_rate"
    return (
        float(cycle_energy_uJ) * float(carrier_Hz) * 1.0e-6,
        "cycle_shoelace_times_measured_carrier_Hz_continuous",
    )


def _charge_lsb_nC(waveform: Waveform, config: dict[str, Any]) -> float | None:
    raw = waveform.lsb.get("charge_monitor")
    if raw is None:
        return None
    return (
        float(raw)
        * float(config["calibration"]["measuring_capacitor_F"]["value"])
        * 1.0e9
    )


def _capture_is_active(meta: dict[str, Any]) -> bool:
    if meta["dataset_type"] == "synthesis":
        return True
    level = meta.get("level_pct")
    return level is None or float(level) >= 100.0


def analyze_capture(
    source: SourceRef,
    waveform: Waveform,
    meta: dict[str, Any],
    config: dict[str, Any],
    *,
    extract_multiline: bool,
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]], dict[str, np.ndarray]]:
    freq = estimate_frequencies(waveform, config)
    codes_a = waveform.code_counts.get("applied_voltage", 0)
    codes_d = waveform.code_counts.get("charge_monitor", 0)
    minimum_codes = int(config["analysis"]["minimum_codes"])
    quantization_status = (
        "quantization_limited"
        if min(codes_a, codes_d) < minimum_codes
        else "codes_gate_passed"
    )
    lf_evidence = (
        lf_tag_evidence(freq, quantization_status, config)
        if freq.carrier_Hz <= float(config["analysis"]["lf_geometric_max_hz"])
        else "not_applicable_above_lf_threshold"
    )
    tag = band_tag(freq.carrier_Hz, config, carrier_evidence=lf_evidence)
    voltage_kV = waveform.applied_voltage_V / 1000.0
    raw_cstar = complex_capacitance(
        waveform.time_s, voltage_kV, waveform.charge_nC, freq.carrier_Hz
    )
    raw_phase = float(np.degrees(np.angle(raw_cstar)))
    q_sign = 1 if not np.isfinite(raw_cstar.real) or raw_cstar.real >= 0 else -1
    charge_nC = q_sign * waveform.charge_nC
    cstar = complex_capacitance(waveform.time_s, voltage_kV, charge_nC, freq.carrier_Hz)
    phase = float(np.degrees(np.angle(cstar)))
    cline = float(abs(cstar))
    time_slope = time_domain_slope_pF(voltage_kV, charge_nC)
    loop = cycle_loop_metrics(waveform.time_s, voltage_kV, charge_nC, freq.carrier_Hz)
    burst_energy, burst_duty = burst_period_metrics(
        waveform.time_s,
        voltage_kV,
        charge_nC,
        freq.active_mask,
        freq.burst_Hz,
    )
    active = _capture_is_active(meta)
    signed_cycle_energy = _median(loop.energy_signed_uJ)
    raw_signed_cycle_energy = (
        signed_cycle_energy * q_sign
        if signed_cycle_energy is not None
        else None
    )
    reported_cycle_energy, orientation_status = _orientation(
        signed_cycle_energy, active=active
    )
    signed_burst_energy = _median(burst_energy)
    raw_signed_burst_energy = (
        signed_burst_energy * q_sign
        if signed_burst_energy is not None
        else None
    )
    reported_burst_energy, burst_orientation = _orientation(
        signed_burst_energy, active=active
    )
    power_W, power_method = _power_from_energies(
        cycle_energy_uJ=reported_cycle_energy,
        burst_energy_uJ=reported_burst_energy,
        carrier_Hz=freq.carrier_Hz,
        burst_Hz=freq.burst_Hz,
    )
    if freq.burst_Hz is not None and signed_burst_energy is not None:
        orientation_status = burst_orientation

    dq_cycle = _median(loop.dQ_intercept_nC)
    half = np.concatenate(
        [loop.positive_half_dQ_nC, loop.negative_half_dQ_nC]
    )
    positive = half[half > 0]
    negative = -half[half < 0]
    positive_per_half = _median(positive)
    negative_per_half = _median(negative)

    f_same: float | None = None
    if (
        loop.Cd_branch_pF is not None
        and loop.Ccell_branch_pF is not None
        and loop.Cd_branch_pF > loop.Ccell_branch_pF
    ):
        f_same = float(
            loop.Cd_branch_pF / (loop.Cd_branch_pF - loop.Ccell_branch_pF)
        )
    correction: float | None
    if tag == "carrier-transfer":
        correction = float(config["calibration"]["same_band_charge_factor"]["value"])
    else:
        correction = f_same
    dq_gap = dq_cycle * correction if dq_cycle is not None and correction is not None else None

    duration = float(np.ptp(waveform.time_s))
    positive_total_nC = float(np.sum(positive)) if positive.size else 0.0
    negative_total_nC = float(np.sum(negative)) if negative.size else 0.0
    if correction is not None:
        positive_total_nC *= correction
        negative_total_nC *= correction
    positive_rate = positive_total_nC * 1e-9 / duration if duration > 0 else None
    negative_rate = negative_total_nC * 1e-9 / duration if duration > 0 else None
    gross_rate = (
        positive_rate + negative_rate
        if positive_rate is not None and negative_rate is not None
        else None
    )
    tv_rate = gross_charge_rate_tv_C_s(
        waveform.time_s,
        charge_nC,
        freq.carrier_Hz,
        maximum_harmonic=float(
            config["analysis"].get("tv_lowpass_maximum_harmonic", 3.0)
        ),
        correction_factor=correction,
    )
    tv_ratio = (
        tv_rate / gross_rate
        if tv_rate is not None and gross_rate is not None and gross_rate > 0
        else None
    )
    net_rate = (
        (positive_total_nC - negative_total_nC) * 1e-9 / duration
        if duration > 0
        else None
    )
    corrected_positive_half = (
        positive_per_half * correction
        if positive_per_half is not None and correction is not None
        else positive_per_half
    )
    corrected_negative_half = (
        negative_per_half * correction
        if negative_per_half is not None and correction is not None
        else negative_per_half
    )
    positive_peak_flow = (
        corrected_positive_half
        * 1e-9
        * 2.0
        * freq.carrier_Hz
        / ELEMENTARY_CHARGE_C
        if corrected_positive_half is not None
        else None
    )
    negative_peak_flow = (
        corrected_negative_half
        * 1e-9
        * 2.0
        * freq.carrier_Hz
        / ELEMENTARY_CHARGE_C
        if corrected_negative_half is not None
        else None
    )
    retained, retained_status = retained_charge(charge_nC, freq.envelope)
    if waveform.detrended:
        retained = None
        retained_status = "not_measured_signal_detrended"
    i_sign, i_phase_error = current_sign(
        waveform.time_s,
        waveform.current_A,
        charge_nC,
        freq.carrier_Hz,
    )

    charge_lsb = _charge_lsb_nC(waveform, config)
    vamp = 0.5 * float(np.percentile(voltage_kV, 99) - np.percentile(voltage_kV, 1))
    quant_unc = (
        0.5 * charge_lsb / vamp
        if charge_lsb is not None and vamp > np.finfo(float).eps
        else None
    )
    clip_a = waveform.clip_counts.get("applied_voltage", 0)
    clip_d = waveform.clip_counts.get("charge_monitor", 0)
    flags = list(meta["exclusion_flags"])
    if clip_a:
        flags.append("channel_A_overrange")
    if clip_d:
        flags.append("channel_D_overrange")
    if quantization_status == "quantization_limited":
        flags.append("fewer_than_30_codes")
    if lf_evidence != "verified" and freq.carrier_Hz <= float(
        config["analysis"]["lf_geometric_max_hz"]
    ):
        flags.append(f"lf_tag_demoted_{lf_evidence.split(':', 1)[1]}")
    frequency_tolerance = float(
        config["analysis"].get("frequency_crosscheck_tolerance", 0.03)
    )
    if (
        freq.relative_disagreement is not None
        and freq.relative_disagreement > frequency_tolerance
    ):
        flags.append(
            f"frequency_crosscheck_offset_{100.0 * freq.relative_disagreement:.2f}_percent"
        )
    if freq.burst_status == "spectral_burst_retained_crosscheck_failed":
        flags.append("burst_edge_crosscheck_failed_spectral_value_retained")
    if orientation_status.startswith("failed") or ";failed_" in orientation_status:
        flags.append("energy_orientation_failed")
    if freq.duty_status.startswith("duty_detector_suspect"):
        flags.append("duty_detector_suspect")
    capacitance_valid = not (
        meta["contaminated"]
        or clip_a
        or clip_d
        or quantization_status == "quantization_limited"
    )
    excluded = bool(meta["contaminated"] or clip_a or clip_d)

    divider_unc = float(
        config["calibration"]["channel_a_divider"].get("relative_uncertainty", 0.0)
    )
    capacitor_unc = float(
        config["calibration"]["measuring_capacitor_F"].get("relative_uncertainty", 0.0)
    )
    syst_frac = float(math.hypot(divider_unc, capacitor_unc))
    current_peak = (
        float(np.percentile(np.abs(i_sign * waveform.current_A), 99) * 1000.0)
        if waveform.current_A is not None and i_sign is not None
        else None
    )
    harmonic = harmonic_ratio(
        waveform.time_s, charge_nC, freq.carrier_Hz
    )
    chain = config["chain_model"]
    current_rms, chain_power = chain_dissipation_estimate(
        waveform.time_s,
        charge_nC,
        freq.carrier_Hz,
        float(chain["R_ohm"]),
        maximum_harmonic=float(
            config["analysis"].get("tv_lowpass_maximum_harmonic", 3.0)
        ),
    )
    resistance_range = chain.get(
        "R_range_ohm", [chain["R_ohm"], chain["R_ohm"]]
    )
    chain_power_low = (
        float(current_rms**2 * float(resistance_range[0]))
        if current_rms is not None
        else None
    )
    chain_power_high = (
        float(current_rms**2 * float(resistance_range[1]))
        if current_rms is not None
        else None
    )

    metric: dict[str, Any] = {
        "cond": meta["cond"],
        "capture": meta.get("seg_idx"),
        "f0_kHz": freq.carrier_Hz / 1000.0,
        "Vamp_kV": vamp,
        "Cline_pF": cline,
        "Creal_pF": float(cstar.real),
        "Cimag_pF": float(-cstar.imag),
        "phase_deg": phase,
        "Clobe_pF": time_slope,
        "codesD": codes_d,
        "codesA": codes_a,
        "Ipk_mA": current_peak,
        "medium": meta["medium"],
        "medium_display": meta["medium_display"],
        "freq_label": meta["freq_label"],
        "level_pct": meta["level_pct"],
        "seg": meta.get("seg_idx"),
        "band_tag": tag,
        "clipA": clip_a,
        "clipD": clip_d,
        "dc_offset": waveform.dc_offset_V,
        "dQ_cycle_nC": dq_cycle,
        "dQ_gap_nC": dq_gap,
        "U_cycle_uJ": reported_cycle_energy,
        "P_W": power_W,
        "Cd_pF": loop.Cd_branch_pF,
        "Ccell_pF": loop.Ccell_branch_pF,
        "syst_frac": syst_frac,
        "path": source.relative_path,
        "source_uri": source.source_uri,
        "dataset_type": meta["dataset_type"],
        "source_type": meta["source_type"],
        "run_key": meta["run_key"],
        "display_label": meta["display_label"],
        "burst_frequency_label_kHz": meta["burst_frequency_label_kHz"],
        "save_idx": meta.get("save_idx"),
        "f0_zero_cross_kHz": (
            freq.zero_cross_Hz / 1000.0 if freq.zero_cross_Hz is not None else None
        ),
        "f0_relative_disagreement": freq.relative_disagreement,
        "f0_autocorr_kHz": (
            freq.autocorr_carrier_Hz / 1000.0
            if freq.autocorr_carrier_Hz is not None
            else None
        ),
        "f0_autocorr_disagreement": freq.autocorr_relative_disagreement,
        "offset_one_sidedness": freq.offset_one_sidedness,
        "frequency_status": freq.carrier_status,
        "carrier_estimator_method": freq.carrier_method,
        "harmonic_decision": freq.harmonic_decision,
        "frequency_crosscheck_diagnostic": freq.crosscheck_diagnostic,
        "lf_tag_evidence": lf_evidence,
        "burst_Hz": freq.burst_Hz,
        "burst_zero_cross_Hz": freq.burst_zero_cross_Hz,
        "burst_relative_disagreement": freq.burst_relative_disagreement,
        "burst_status": freq.burst_status,
        "duty_on_fraction": (
            _median(burst_duty)
            if burst_duty.size
            else freq.duty_on_fraction
        ),
        "burst_on_cycles": freq.burst_on_cycles,
        "duty_status": freq.duty_status,
        "envelope_contrast": freq.envelope_contrast,
        "envelope_threshold_V": freq.envelope_threshold,
        "q_sign": q_sign,
        "raw_phase_deg": raw_phase,
        "raw_U_cycle_signed_uJ": raw_signed_cycle_energy,
        "raw_U_burst_signed_uJ": raw_signed_burst_energy,
        "polarity_flipped_by_audit": False,
        "polarity_audit_action": "pending_condition_audit",
        "current_sign": i_sign,
        "current_phase_error_deg": i_phase_error,
        "charge_lsb_nC": charge_lsb,
        "Cline_quantization_uncertainty_pF": quant_unc,
        "harmonic_ratio": harmonic,
        "active_cycle_fraction": loop.active_cycle_fraction,
        "U_cycle_signed_uJ": signed_cycle_energy,
        "U_burst_uJ": reported_burst_energy,
        "U_burst_signed_uJ": signed_burst_energy,
        "P_method": power_method,
        "I_charge_rms_A": current_rms,
        "P_chain_est_W": chain_power,
        "P_chain_est_low_W": chain_power_low,
        "P_chain_est_high_W": chain_power_high,
        "dQ_positive_nC": positive_per_half,
        "dQ_negative_nC": negative_per_half,
        "positive_rate_C_s": positive_rate,
        "negative_rate_C_s": negative_rate,
        "gross_rate_C_s": gross_rate,
        "gross_rate_C_s_tv": tv_rate,
        "gross_rate_tv_ratio": tv_ratio,
        "net_rate_C_s": net_rate,
        "gross_rate_C_min": gross_rate * 60.0 if gross_rate is not None else None,
        "positive_average_equivalent_flow_per_s": (
            positive_rate / ELEMENTARY_CHARGE_C
            if positive_rate is not None
            else None
        ),
        "negative_average_equivalent_flow_per_s": (
            negative_rate / ELEMENTARY_CHARGE_C
            if negative_rate is not None
            else None
        ),
        "positive_peak_halfcycle_average_equivalent_flow_per_s": positive_peak_flow,
        "negative_peak_halfcycle_average_equivalent_flow_per_s": negative_peak_flow,
        "capture_gross_dose_C": gross_rate * duration if gross_rate is not None else None,
        "dose_20min_C": gross_rate * 1200.0 if gross_rate is not None else None,
        "retained_charge_nC": retained,
        "retained_charge_status": retained_status,
        "F_same_band": f_same,
        "charge_correction_factor": correction,
        "orientation_status": orientation_status,
        "phase_status": _phase_status(phase),
        "quantization_status": quantization_status,
        "capacitance_valid": capacitance_valid,
        "excluded": excluded,
        "exclusion_flags": ";".join(sorted(set(flags))),
    }
    manifest: dict[str, Any] = {
        "path": source.relative_path,
        "source_uri": source.source_uri,
        "storage": "zip" if source.archive_path is not None else "file",
        "size_bytes": source.size,
        "signature": source.signature,
        "dataset_type": meta["dataset_type"],
        "source_type": meta["source_type"],
        "medium": meta["medium"],
        "medium_display": meta["medium_display"],
        "run_key": meta["run_key"],
        "freq_label": meta["freq_label"],
        "nominal_frequency_kHz": meta["nominal_frequency_kHz"],
        "burst_frequency_label_kHz": meta["burst_frequency_label_kHz"],
        "level_pct": meta["level_pct"],
        "commanded_kV": meta["commanded_kV"],
        "save_idx": meta["save_idx"],
        "seg_idx": meta["seg_idx"],
        "n_samples": waveform.time_s.size,
        "dt_s": float(np.median(np.diff(waveform.time_s))),
        "duration_s": duration,
        "channel_labels": "|".join(waveform.headers),
        "channel_units": "|".join(waveform.units),
        "channel_mapping": json.dumps(waveform.role_indices, sort_keys=True),
        "clipA": clip_a,
        "clipC": waveform.clip_counts.get("legacy_current", 0),
        "clipD": clip_d,
        "codesA": codes_a,
        "codesC": waveform.code_counts.get("legacy_current", 0),
        "codesD": codes_d,
        "lsbA_raw": waveform.lsb.get("applied_voltage"),
        "lsbD_raw": waveform.lsb.get("charge_monitor"),
        "f0_Hz": freq.carrier_Hz,
        "f0_zero_cross_Hz": freq.zero_cross_Hz,
        "f0_relative_disagreement": freq.relative_disagreement,
        "f0_autocorr_kHz": (
            freq.autocorr_carrier_Hz / 1000.0
            if freq.autocorr_carrier_Hz is not None
            else None
        ),
        "f0_autocorr_disagreement": freq.autocorr_relative_disagreement,
        "offset_one_sidedness": freq.offset_one_sidedness,
        "frequency_status": freq.carrier_status,
        "carrier_estimator_method": freq.carrier_method,
        "harmonic_decision": freq.harmonic_decision,
        "frequency_crosscheck_diagnostic": freq.crosscheck_diagnostic,
        "lf_tag_evidence": lf_evidence,
        "burst_Hz": freq.burst_Hz,
        "burst_zero_cross_Hz": freq.burst_zero_cross_Hz,
        "burst_relative_disagreement": freq.burst_relative_disagreement,
        "burst_status": freq.burst_status,
        "duty_on_fraction": metric["duty_on_fraction"],
        "burst_on_cycles": freq.burst_on_cycles,
        "duty_status": freq.duty_status,
        "band_tag": tag,
        "phase_deg": phase,
        "dc_offset": waveform.dc_offset_V,
        "drift_V_per_s": waveform.drift_V_per_s,
        "detrended": waveform.detrended,
        "contaminated": meta["contaminated"],
        "excluded": excluded,
        "exclusion_flags": metric["exclusion_flags"],
        "parse_status": "ok",
        "parse_error": "",
    }

    spectral: list[dict[str, Any]] = []
    if (
        extract_multiline
        and meta["dataset_type"] == "dispersion_7_20"
        and not meta["contaminated"]
        and not clip_a
        and not clip_d
    ):
        for row in multiline_capacitance_points(
            waveform.time_s,
            voltage_kV,
            charge_nC,
            minimum_Hz=float(config["analysis"]["multiline_minimum_hz"]),
            maximum_Hz=float(config["analysis"]["multiline_maximum_hz"]),
            relative_threshold=float(
                config["analysis"]["multiline_relative_voltage_threshold"]
            ),
            max_points=int(config["analysis"]["multiline_max_points_per_capture"]),
        ):
            spectral.append(
                {
                    **row,
                    "cond": meta["cond"],
                    "capture": meta["seg_idx"],
                    "medium": meta["medium"],
                    "band_tag": band_tag(row["frequency_kHz"] * 1000.0, config),
                    "source_type": "7_20_multiline",
                    "provenance": source.relative_path,
                    "flags": (
                        "quantization_limited"
                        if quantization_status == "quantization_limited"
                        else ""
                    ),
                }
            )

    representative: dict[str, np.ndarray] = {}
    if (
        meta["dataset_type"] == "voltage_ladder"
        and meta["medium"] == "argon_only"
        and meta["freq_label"] == "4 kHz"
        and meta["level_pct"] is not None
    ):
        level = f"{float(meta['level_pct']):g}"
        stride = max(1, waveform.time_s.size // 6000)
        prefix = f"argon4k_{level}"
        representative = {
            f"{prefix}_time_s": waveform.time_s[::stride] - waveform.time_s[0],
            f"{prefix}_V_kV": voltage_kV[::stride],
            f"{prefix}_Q_nC": charge_nC[::stride],
            f"{prefix}_loop_V_kV": loop.loop_voltage_kV,
            f"{prefix}_loop_Q_nC": loop.loop_charge_nC,
            f"{prefix}_f0_Hz": np.asarray(freq.carrier_Hz),
            f"{prefix}_burst_Hz": np.asarray(
                freq.burst_Hz if freq.burst_Hz is not None else np.nan
            ),
            f"{prefix}_dc_offset_V": np.asarray(waveform.dc_offset_V),
            f"{prefix}_clipA": np.asarray(clip_a),
            f"{prefix}_clipD": np.asarray(clip_d),
            f"{prefix}_seg": np.asarray(meta.get("seg_idx") or -1),
        }
    return manifest, metric, spectral, representative


def failure_manifest(
    source: SourceRef,
    meta: dict[str, Any],
    error: Exception,
) -> dict[str, Any]:
    row = {column: None for column in MANIFEST_COLUMNS}
    row.update(
        {
            "path": source.relative_path,
            "source_uri": source.source_uri,
            "storage": "zip" if source.archive_path is not None else "file",
            "size_bytes": source.size,
            "signature": source.signature,
            "dataset_type": meta["dataset_type"],
            "source_type": meta["source_type"],
            "medium": meta["medium"],
            "medium_display": meta["medium_display"],
            "run_key": meta["run_key"],
            "freq_label": meta["freq_label"],
            "nominal_frequency_kHz": meta["nominal_frequency_kHz"],
            "burst_frequency_label_kHz": meta["burst_frequency_label_kHz"],
            "level_pct": meta["level_pct"],
            "commanded_kV": meta["commanded_kV"],
            "save_idx": meta["save_idx"],
            "seg_idx": meta["seg_idx"],
            "contaminated": meta["contaminated"],
            "excluded": True,
            "exclusion_flags": ";".join(meta["exclusion_flags"] + ["parse_failed"]),
            "parse_status": "failed",
            "parse_error": f"{type(error).__name__}: {error}",
        }
    )
    return row


def _aggregate_dispersion(
    captures: Sequence[dict[str, Any]],
    spectral: Sequence[dict[str, Any]],
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    groups: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for capture in captures:
        level = capture.get("level_pct")
        passive_ladder = (
            capture.get("dataset_type") == "voltage_ladder"
            and level not in (None, "")
            and float(level) <= 90.0
        )
        include = passive_ladder or capture.get("dataset_type") in {
            "july7_8_operational",
            "dispersion_7_20",
        }
        if not include or not capture.get("capacitance_valid"):
            continue
        key = (
            str(capture.get("medium")),
            str(capture.get("dataset_type")),
            str(capture.get("freq_label")),
            str(capture.get("band_tag")),
        )
        groups[key].append(capture)
    for key in sorted(groups):
        values = groups[key]
        f = stats(row.get("f0_kHz") for row in values)
        cmag = stats(row.get("Cline_pF") for row in values)
        creal = stats(row.get("Creal_pF") for row in values)
        cimag = stats(row.get("Cimag_pF") for row in values)
        phase = stats(row.get("phase_deg") for row in values)
        rows.append(
            {
                "frequency_kHz": f["median"],
                "frequency_p2_5_kHz": f["p2_5"],
                "frequency_p97_5_kHz": f["p97_5"],
                "Cmag_pF": cmag["median"],
                "Cmag_p2_5_pF": cmag["p2_5"],
                "Cmag_p97_5_pF": cmag["p97_5"],
                "Creal_pF": creal["median"],
                "Cimag_pF": cimag["median"],
                "phase_deg": phase["median"],
                "band_tag": key[3],
                "source_type": f"{key[1]}_fundamental",
                "medium": key[0],
                "provenance": f"capture median, folder label {key[2]}",
                "flags": "",
                "N": cmag["N"],
                "plot_eligible": True,
            }
        )

    line_groups: dict[tuple[str, str, float], list[dict[str, Any]]] = defaultdict(list)
    for row in spectral:
        rounded = round(float(row["frequency_kHz"]) * 4.0) / 4.0
        line_groups[(str(row["cond"]), str(row["band_tag"]), rounded)].append(row)
    for (cond, tag, rounded), values in sorted(line_groups.items()):
        cmag = stats(row.get("Cmag_pF") for row in values)
        creal = stats(row.get("Creal_pF") for row in values)
        cimag = stats(row.get("Cimag_pF") for row in values)
        phase = stats(row.get("phase_deg") for row in values)
        minimum_plot_n = int(
            config["analysis"].get("multiline_plot_min_N", 3)
        )
        plot_eligible = int(cmag["N"]) >= minimum_plot_n
        rows.append(
            {
                "frequency_kHz": rounded,
                "frequency_p2_5_kHz": None,
                "frequency_p97_5_kHz": None,
                "Cmag_pF": cmag["median"],
                "Cmag_p2_5_pF": cmag["p2_5"],
                "Cmag_p97_5_pF": cmag["p97_5"],
                "Creal_pF": creal["median"],
                "Cimag_pF": cimag["median"],
                "phase_deg": phase["median"],
                "band_tag": tag,
                "source_type": "7_20_multiline",
                "medium": "dry_fixture",
                "provenance": cond,
                "flags": ";".join(
                    sorted(
                        {
                            *{
                                flag
                                for row in values
                                for flag in str(row.get("flags") or "").split(";")
                                if flag
                            },
                            *(
                                set()
                                if plot_eligible
                                else {f"plot_excluded_N_lt_{minimum_plot_n}"}
                            ),
                        }
                    )
                ),
                "N": cmag["N"],
                "plot_eligible": plot_eligible,
            }
        )

    for legacy in config.get("legacy_dispersion_points", []):
        rows.append(
            {
                "frequency_kHz": legacy["frequency_kHz"],
                "frequency_p2_5_kHz": None,
                "frequency_p97_5_kHz": None,
                "Cmag_pF": legacy["Cmag_pF"],
                "Cmag_p2_5_pF": None,
                "Cmag_p97_5_pF": None,
                "Creal_pF": None,
                "Cimag_pF": None,
                "phase_deg": None,
                "band_tag": legacy["band_tag"],
                "source_type": legacy["source_type"],
                "medium": legacy["medium"],
                "provenance": legacy["provenance"],
                "flags": "historical_transfer_context",
                "N": None,
                "plot_eligible": True,
            }
        )

    chain = config["chain_model"]
    C = float(chain["C_true_F"])
    L = float(chain["L_H"])
    R = float(chain["R_ohm"])
    model_range = chain.get("plot_frequency_range_Hz", [4000.0, 170000.0])
    grid = np.geomspace(float(model_range[0]), float(model_range[1]), 240)
    omega = 2.0 * np.pi * grid
    cstar = C / (1.0 - omega**2 * L * C + 1j * omega * R * C)
    for frequency, value in zip(grid, cstar):
        rows.append(
            {
                "frequency_kHz": float(frequency / 1000.0),
                "frequency_p2_5_kHz": None,
                "frequency_p97_5_kHz": None,
                "Cmag_pF": float(abs(value) * 1e12),
                "Cmag_p2_5_pF": None,
                "Cmag_p97_5_pF": None,
                "Creal_pF": float(value.real * 1e12),
                "Cimag_pF": float(-value.imag * 1e12),
                "phase_deg": float(np.degrees(np.angle(value))),
                "band_tag": "model",
                "source_type": "series_RLC_model",
                "medium": "measurement_chain",
                "provenance": chain["provenance"],
                "flags": "order-of-magnitude\u2020",
                "N": None,
                "plot_eligible": True,
            }
        )
    return sorted(
        rows,
        key=lambda row: (
            float(row["frequency_kHz"]),
            str(row["source_type"]),
            str(row["medium"]),
        ),
    )


def _polarity_group_key(row: dict[str, Any]) -> tuple[str, str, str]:
    dataset = str(row.get("dataset_type"))
    if dataset == "synthesis":
        return dataset, str(row.get("run_key")), ""
    if dataset == "dispersion_7_20":
        return dataset, str(row.get("cond")), ""
    return dataset, str(row.get("medium")), str(row.get("freq_label"))


def _refresh_orientation_and_power(row: dict[str, Any]) -> None:
    active = _capture_is_active(row)
    signed_cycle = row.get("U_cycle_signed_uJ")
    reported_cycle, cycle_status = _orientation(
        float(signed_cycle) if signed_cycle not in (None, "") else None,
        active=active,
    )
    row["U_cycle_uJ"] = reported_cycle
    signed_burst = row.get("U_burst_signed_uJ")
    reported_burst, burst_status = _orientation(
        float(signed_burst) if signed_burst not in (None, "") else None,
        active=active,
    )
    row["U_burst_uJ"] = reported_burst
    row["orientation_status"] = (
        burst_status
        if row.get("burst_Hz") not in (None, "") and signed_burst not in (None, "")
        else cycle_status
    )
    row["P_W"], row["P_method"] = _power_from_energies(
        cycle_energy_uJ=reported_cycle,
        burst_energy_uJ=reported_burst,
        carrier_Hz=(
            float(row["f0_kHz"]) * 1e3
            if row.get("f0_kHz") not in (None, "")
            else None
        ),
        burst_Hz=(
            float(row["burst_Hz"])
            if row.get("burst_Hz") not in (None, "")
            else None
        ),
    )


def _flip_capture_orientation(
    row: dict[str, Any],
    manifest: dict[str, Any] | None,
    *,
    provenance_flag: str,
) -> None:
    """Apply a global Q→−Q orientation change and refresh signed outputs."""

    for field in (
        "Creal_pF",
        "Cimag_pF",
        "Clobe_pF",
        "U_cycle_signed_uJ",
        "U_burst_signed_uJ",
        "retained_charge_nC",
        "net_rate_C_s",
    ):
        if row.get(field) not in (None, ""):
            row[field] = -float(row[field])
    if row.get("phase_deg") not in (None, ""):
        row["phase_deg"] = (
            float(row["phase_deg"]) + 180.0 + 180.0
        ) % 360.0 - 180.0
        row["phase_status"] = _phase_status(float(row["phase_deg"]))
    if row.get("q_sign") in (-1, 1):
        row["q_sign"] = -int(row["q_sign"])
    if row.get("current_sign") in (-1, 1):
        row["current_sign"] = -int(row["current_sign"])
    row["dQ_positive_nC"], row["dQ_negative_nC"] = (
        row.get("dQ_negative_nC"),
        row.get("dQ_positive_nC"),
    )
    row["positive_rate_C_s"], row["negative_rate_C_s"] = (
        row.get("negative_rate_C_s"),
        row.get("positive_rate_C_s"),
    )
    for positive_field, negative_field in (
        (
            "positive_average_equivalent_flow_per_s",
            "negative_average_equivalent_flow_per_s",
        ),
        (
            "positive_peak_halfcycle_average_equivalent_flow_per_s",
            "negative_peak_halfcycle_average_equivalent_flow_per_s",
        ),
    ):
        row[positive_field], row[negative_field] = (
            row.get(negative_field),
            row.get(positive_field),
        )
    _refresh_orientation_and_power(row)
    # Branch slopes reverse with Q and cannot remain physical capacitances.
    row["Cd_pF"] = None
    row["Ccell_pF"] = None
    row["F_same_band"] = None
    capture_flags = set(
        filter(None, str(row.get("exclusion_flags") or "").split(";"))
    )
    capture_flags.discard("energy_orientation_failed")
    capture_flags.add(provenance_flag)
    if str(row.get("orientation_status", "")).startswith("failed"):
        capture_flags.add("energy_orientation_failed")
    row["exclusion_flags"] = ";".join(sorted(capture_flags))
    if manifest is not None:
        manifest["phase_deg"] = row.get("phase_deg")
        manifest_flags = set(
            filter(None, str(manifest.get("exclusion_flags") or "").split(";"))
        )
        manifest_flags.discard("energy_orientation_failed")
        manifest_flags.add(provenance_flag)
        if str(row.get("orientation_status", "")).startswith("failed"):
            manifest_flags.add("energy_orientation_failed")
        manifest["exclusion_flags"] = ";".join(sorted(manifest_flags))


def _apply_condition_polarity_lock(
    captures: list[dict[str, Any]],
    manifests: list[dict[str, Any]],
) -> None:
    """Lock charge-monitor polarity across each physical acquisition configuration."""

    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in captures:
        groups[_polarity_group_key(row)].append(row)
    manifest_by_uri = {str(row.get("source_uri")): row for row in manifests}
    for rows in groups.values():
        preferred = [
            row
            for row in rows
            if row.get("q_sign") in (-1, 1)
            and not row.get("excluded")
            and row.get("quantization_status") == "codes_gate_passed"
            and (
                row.get("level_pct") in (None, "")
                or float(row.get("level_pct")) <= 90.0
            )
        ]
        voters = preferred or [
            row
            for row in rows
            if row.get("q_sign") in (-1, 1) and not row.get("excluded")
        ]
        if not voters:
            voters = [row for row in rows if row.get("q_sign") in (-1, 1)]
        vote = sum(int(row["q_sign"]) for row in voters)
        agreement: float
        if vote == 0:
            raw_active_energy = [
                float(
                    row.get("raw_U_burst_signed_uJ")
                    if row.get("raw_U_burst_signed_uJ") not in (None, "")
                    else row.get("raw_U_cycle_signed_uJ")
                )
                for row in rows
                if _capture_is_active(row)
                and (
                    row.get("raw_U_burst_signed_uJ") not in (None, "")
                    or row.get("raw_U_cycle_signed_uJ") not in (None, "")
                )
            ]
            raw_median = _median(np.asarray(raw_active_energy, dtype=float))
            locked = 1 if raw_median is None or raw_median >= 0 else -1
            agreement = 0.5
            status = (
                "condition_locked_by_energy_audit_equal_votes_"
                f"agreement_{agreement:.3f}"
            )
        else:
            locked = 1 if vote > 0 else -1
            agreement = sum(
                int(row["q_sign"]) == locked for row in voters
            ) / max(len(voters), 1)
            status = (
                f"condition_locked_agreement_{agreement:.3f}"
                if agreement >= 0.75
                else f"majority_locked_low_agreement_{agreement:.3f}"
            )
        for row in rows:
            original = int(row.get("q_sign") or locked)
            row["condition_q_sign"] = locked
            row["polarity_lock_status"] = status
            row["polarity_lock_agreement"] = agreement
            if original != locked:
                _flip_capture_orientation(
                    row,
                    manifest_by_uri.get(str(row.get("source_uri"))),
                    provenance_flag="capture_polarity_overridden_by_condition_lock",
                )
            row["q_sign"] = locked


def _apply_condition_polarity_audit(
    captures: list[dict[str, Any]],
    manifests: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Recover condition-level monitor inversions using signed loop energy."""

    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in captures:
        groups[_polarity_group_key(row)].append(row)
    manifest_by_uri = {str(row.get("source_uri")): row for row in manifests}
    pre_audit_by_condition: dict[str, tuple[float | None, float | None]] = {}
    condition_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in captures:
        condition_rows[str(row.get("cond"))].append(row)
    for cond, rows in condition_rows.items():
        phase_values = np.asarray(
            [
                float(row["phase_deg"])
                for row in rows
                if row.get("phase_deg") not in (None, "")
            ],
            dtype=float,
        )
        energy_values = np.asarray(
            [
                float(
                    row.get("U_burst_signed_uJ")
                    if row.get("U_burst_signed_uJ") not in (None, "")
                    else row.get("U_cycle_signed_uJ")
                )
                for row in rows
                if (
                    row.get("U_burst_signed_uJ") not in (None, "")
                    or row.get("U_cycle_signed_uJ") not in (None, "")
                )
            ],
            dtype=float,
        )
        pre_audit_by_condition[cond] = (
            _median(phase_values),
            _median(energy_values),
        )
    group_action: dict[tuple[str, str, str], tuple[bool, str]] = {}
    for key, rows in groups.items():
        active_rows = [
            row for row in rows if _capture_is_active(row) and not row.get("excluded")
        ]
        finite_energy = np.asarray(
            [
                float(
                    row.get("U_burst_signed_uJ")
                    if row.get("U_burst_signed_uJ") not in (None, "")
                    else row.get("U_cycle_signed_uJ")
                )
                for row in active_rows
                if (
                    row.get("U_burst_signed_uJ") not in (None, "")
                    or row.get("U_cycle_signed_uJ") not in (None, "")
                )
            ],
            dtype=float,
        )
        median_energy = _median(finite_energy)
        phase = _median(
            np.asarray(
                [
                    float(row["raw_phase_deg"])
                    for row in rows
                    if row.get("raw_phase_deg") not in (None, "")
                ],
                dtype=float,
            )
        )
        phase_distance_180 = (
            abs(abs((phase + 180.0) % 360.0 - 180.0) - 180.0)
            if phase is not None
            else None
        )
        transfer_band = any(
            row.get("band_tag") == "carrier-transfer" for row in rows
        )
        should_flip = bool(
            median_energy is not None
            and median_energy < 0
            and (
                phase_distance_180 is not None
                and phase_distance_180 <= 45.0
                or transfer_band
            )
        )
        if should_flip:
            basis = (
                "raw_phase_near_180_and_trial_energy_nonnegative"
                if phase_distance_180 is not None and phase_distance_180 <= 45.0
                else "carrier_transfer_phase_non_geometric_trial_energy_nonnegative"
            )
            for row in rows:
                _flip_capture_orientation(
                    row,
                    manifest_by_uri.get(str(row.get("source_uri"))),
                    provenance_flag="polarity_flipped_by_audit",
                )
                row["polarity_flipped_by_audit"] = True
                row["polarity_audit_action"] = basis
                if row.get("condition_q_sign") in (-1, 1):
                    row["condition_q_sign"] = -int(row["condition_q_sign"])
            group_action[key] = True, basis
        else:
            basis = (
                "no_active_energy"
                if median_energy is None
                else "orientation_retained_nonnegative_energy"
                if median_energy >= 0
                else "withheld_reactive_loop_not_monitor_inversion"
            )
            for row in rows:
                row["polarity_flipped_by_audit"] = False
                row["polarity_audit_action"] = basis
                _refresh_orientation_and_power(row)
            group_action[key] = False, basis

    audit_rows: list[dict[str, Any]] = []
    for cond, rows in sorted(condition_rows.items()):
        key = _polarity_group_key(rows[0])
        flipped, action = group_action[key]
        pre_energy_values: list[float] = []
        for row in rows:
            value = (
                row.get("raw_U_burst_signed_uJ")
                if row.get("raw_U_burst_signed_uJ") not in (None, "")
                else row.get("raw_U_cycle_signed_uJ")
            )
            if value not in (None, ""):
                # Raw energy is before both the lock and audit.
                pre_energy_values.append(float(value))
        raw_median_energy = _median(np.asarray(pre_energy_values, dtype=float))
        final_energy = _median(
            np.asarray(
                [
                    float(
                        row.get("U_burst_signed_uJ")
                        if row.get("U_burst_signed_uJ") not in (None, "")
                        else row.get("U_cycle_signed_uJ")
                    )
                    for row in rows
                    if (
                        row.get("U_burst_signed_uJ") not in (None, "")
                        or row.get("U_cycle_signed_uJ") not in (None, "")
                    )
                ],
                dtype=float,
            )
        )
        audit_rows.append(
            {
                "cond": cond,
                "dataset_type": rows[0].get("dataset_type"),
                "medium": rows[0].get("medium"),
                "freq_label": rows[0].get("freq_label"),
                "level_pct": rows[0].get("level_pct"),
                "median_raw_phase_f0_deg": _median(
                    np.asarray(
                        [
                            float(row["raw_phase_deg"])
                            for row in rows
                            if row.get("raw_phase_deg") not in (None, "")
                        ],
                        dtype=float,
                    )
                ),
                "median_phase_f0_deg": _median(
                    np.asarray(
                        [
                            float(row["phase_deg"])
                            for row in rows
                            if row.get("phase_deg") not in (None, "")
                        ],
                        dtype=float,
                    )
                ),
                "polarity_lock_status": rows[0].get("polarity_lock_status"),
                "polarity_lock_agreement": rows[0].get("polarity_lock_agreement"),
                "median_pre_audit_phase_f0_deg": pre_audit_by_condition[cond][0],
                "median_pre_audit_U_signed_uJ": pre_audit_by_condition[cond][1],
                "trial_global_channel_D_flip_median_U_signed_uJ": (
                    -pre_audit_by_condition[cond][1]
                    if pre_audit_by_condition[cond][1] is not None
                    else None
                ),
                "median_raw_U_signed_uJ": raw_median_energy,
                "trial_raw_channel_D_flip_median_U_signed_uJ": (
                    -raw_median_energy if raw_median_energy is not None else None
                ),
                "median_final_U_signed_uJ": final_energy,
                "polarity_flipped_by_audit": flipped,
                "audit_action": action,
                "N_capture": len(rows),
            }
        )
    return audit_rows


def _duplicate_flags(manifests: list[dict[str, Any]]) -> None:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in manifests:
        signature = str(row.get("signature") or "")
        if signature:
            groups[signature].append(row)
    for rows in groups.values():
        if len(rows) < 2:
            continue
        for row in rows:
            flags = set(filter(None, str(row.get("exclusion_flags") or "").split(";")))
            flags.add("duplicate_storage_signature")
            row["exclusion_flags"] = ";".join(sorted(flags))


def _apply_known_duplicate_exclusions(
    manifests: list[dict[str, Any]],
    captures: list[dict[str, Any]],
    config: dict[str, Any],
) -> None:
    known = config.get("known_duplicate_paths", {})
    if not isinstance(known, dict):
        return
    manifest_by_path = {str(row.get("path")): row for row in manifests}
    capture_by_path = {str(row.get("path")): row for row in captures}
    for duplicate_path, details in known.items():
        canonical = (
            details.get("canonical_path")
            if isinstance(details, dict)
            else str(details)
        )
        duplicate = manifest_by_path.get(str(duplicate_path))
        if duplicate is None:
            continue
        flags = set(
            filter(None, str(duplicate.get("exclusion_flags") or "").split(";"))
        )
        flags.add(f"known_duplicate_of:{canonical}")
        duplicate["exclusion_flags"] = ";".join(sorted(flags))
        duplicate["excluded"] = True
        capture = capture_by_path.get(str(duplicate_path))
        if capture is not None:
            capture_flags = set(
                filter(None, str(capture.get("exclusion_flags") or "").split(";"))
            )
            capture_flags.add(f"known_duplicate_of:{canonical}")
            capture["exclusion_flags"] = ";".join(sorted(capture_flags))
            capture["excluded"] = True
            capture["capacitance_valid"] = False


def run(args: argparse.Namespace) -> dict[str, Any]:
    config = load_config(args.config)
    data_root = resolved_data_root(config, args.data_root)
    output = resolved_output(config, args.out)
    output.mkdir(parents=True, exist_ok=True)
    prefer_archives = bool(config["analysis"].get("prefer_archives", True)) and not args.no_archives
    sources, discovery_errors = discover_sources(
        data_root, prefer_archives=prefer_archives
    )
    if args.path_regex:
        pattern = re.compile(args.path_regex, flags=re.IGNORECASE)
        sources = [source for source in sources if pattern.search(source.relative_path)]
    if args.max_files is not None:
        sources = sources[: max(0, args.max_files)]
    if not sources:
        raise RuntimeError(f"No waveform CSV files were discovered under {data_root}")

    manifests: list[dict[str, Any]] = []
    captures: list[dict[str, Any]] = []
    spectral: list[dict[str, Any]] = []
    representative: dict[str, np.ndarray] = {}
    representative_score: dict[str, float] = {}
    start = time.monotonic()
    with SourceCatalog(sources) as catalog:
        for index, source in enumerate(sources, start=1):
            meta = parse_path_metadata(source.relative_path, config)
            try:
                waveform = read_waveform(catalog.read_bytes(source), config)
                manifest, metric, lines, rep = analyze_capture(
                    source,
                    waveform,
                    meta,
                    config,
                    extract_multiline=not args.no_multiline,
                )
                manifests.append(manifest)
                captures.append(metric)
                spectral.extend(lines)
                if rep:
                    level = f"{float(meta['level_pct']):g}"
                    score = abs(float(meta.get("seg_idx") or 1) - 32.0)
                    if level not in representative_score or score < representative_score[level]:
                        prefix = f"argon4k_{level}_"
                        for key in [key for key in representative if key.startswith(prefix)]:
                            representative.pop(key)
                        representative.update(rep)
                        representative_score[level] = score
            except Exception as exc:  # each failure remains visible in manifest/RESULTS
                manifests.append(failure_manifest(source, meta, exc))
            if not args.quiet and (index == 1 or index % 100 == 0 or index == len(sources)):
                elapsed = time.monotonic() - start
                rate = index / elapsed if elapsed > 0 else 0.0
                remaining = (len(sources) - index) / rate if rate > 0 else float("nan")
                print(
                    f"[{index:4d}/{len(sources)}] parsed={len(captures)} "
                    f"failed={len(manifests)-len(captures)} "
                    f"rate={rate:.2f}/s eta={remaining/60.0:.1f} min",
                    flush=True,
                )

    _duplicate_flags(manifests)
    _apply_known_duplicate_exclusions(manifests, captures, config)
    _apply_condition_polarity_lock(captures, manifests)
    polarity_audit = _apply_condition_polarity_audit(captures, manifests)
    summaries = condition_summary_rows(captures)
    onsets = detect_onsets(captures, config)
    discharges = discharge_summary_rows(captures, onsets)
    synthesis = synthesis_summary_rows(captures, config)
    consistency = frequency_consistency_rows(
        captures, float(config["analysis"]["r3_relative_tolerance"])
    )
    factors = factor_frequency_rows(captures, config)
    dispersion = _aggregate_dispersion(captures, spectral, config)
    numbers = presentation_numbers(
        captures, synthesis, onsets, consistency, factors, config
    )

    write_csv(output / "manifest.csv", manifests, MANIFEST_COLUMNS)
    write_csv(
        output / "per_capture_metrics.csv",
        captures,
        PER_CAPTURE_STABLE_COLUMNS + PER_CAPTURE_EXTRA_COLUMNS,
    )
    write_csv(output / "condition_summary.csv", summaries)
    write_csv(output / "dispersion_master.csv", dispersion)
    write_csv(output / "discharge_metrics.csv", discharges)
    write_csv(output / "synthesis_charge.csv", synthesis)
    write_csv(output / "frequency_consistency.csv", consistency)
    write_csv(output / "factor_frequency.csv", factors)
    write_csv(output / "discharge_onset.csv", onsets)
    write_csv(output / "polarity_audit.csv", polarity_audit)
    if representative:
        np.savez_compressed(output / "figure_data.npz", **representative)
    write_json(output / "presentation_numbers.json", numbers)
    write_json(output / "config.yaml", output_config(config))
    (output / "RESULTS.md").write_text(
        results_markdown(
            manifests,
            captures,
            synthesis,
            consistency,
            config,
            discovery_errors,
        ),
        encoding="utf-8",
    )
    assert_no_retired_numbers(output)

    orientation_failures = [
        row
        for row in captures
        if "failed_negative_median_energy" in str(row.get("orientation_status"))
    ]
    if config["analysis"].get("strict_orientation") and orientation_failures:
        conditions = sorted({str(row["cond"]) for row in orientation_failures})
        raise RuntimeError(
            "energy_orientation_failed: negative median signed energy in "
            + ", ".join(conditions)
        )
    return {
        "data_root": str(data_root),
        "output": str(output),
        "logical_csvs": len(sources),
        "parsed": len(captures),
        "failed": len(manifests) - len(captures),
        "orientation_failures": len(orientation_failures),
        "LF_geometric_captures": sum(
            row.get("band_tag") == "LF-geometric" for row in captures
        ),
        "carrier_transfer_captures": sum(
            row.get("band_tag") == "carrier-transfer" for row in captures
        ),
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_arguments(argv)
    try:
        result = run(args)
    except Exception as exc:
        print(f"Lissajous v2 failed: {exc}", file=sys.stderr)
        return 1
    if not args.quiet:
        print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
