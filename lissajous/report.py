"""Deterministic tables, acceptance gates, numbers pack, and text report."""

from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np


FORBIDDEN_OUTPUT_TOKENS = ("50.07 pF", "F = 2.254", "458 nC")


def finite(values: Iterable[Any]) -> np.ndarray:
    parsed: list[float] = []
    for value in values:
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(number):
            parsed.append(number)
    return np.asarray(parsed, dtype=float)


def stats(values: Iterable[Any]) -> dict[str, float | int | None]:
    array = finite(values)
    if not array.size:
        return {"median": None, "p2_5": None, "p97_5": None, "N": 0}
    return {
        "median": float(np.median(array)),
        "p2_5": float(np.percentile(array, 2.5)),
        "p97_5": float(np.percentile(array, 97.5)),
        "N": int(array.size),
    }


def _serializable(value: Any) -> Any:
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return ""
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    return value


def write_csv(
    path: Path,
    rows: Sequence[dict[str, Any]],
    fieldnames: Sequence[str] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        observed: list[str] = []
        seen: set[str] = set()
        for row in rows:
            for key in row:
                if key not in seen:
                    observed.append(key)
                    seen.add(key)
        fieldnames = observed
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _serializable(row.get(key)) for key in fieldnames})


SUMMARY_METRICS = (
    "f0_kHz",
    "Vamp_kV",
    "Cline_pF",
    "Creal_pF",
    "Cimag_pF",
    "phase_deg",
    "Clobe_pF",
    "dQ_cycle_nC",
    "dQ_gap_nC",
    "U_cycle_uJ",
    "U_burst_uJ",
    "P_W",
    "P_chain_est_W",
    "Cd_pF",
    "Ccell_pF",
    "charge_correction_factor",
    "duty_on_fraction",
    "burst_on_cycles",
    "burst_Hz",
    "dQ_positive_nC",
    "dQ_negative_nC",
)


def _orientation_subset_status(rows: Sequence[dict[str, Any]]) -> str:
    counts: dict[str, int] = defaultdict(int)
    for row in rows:
        counts[str(row.get("orientation_status") or "not_available")] += 1
    if not counts:
        return ""
    if len(counts) == 1:
        return next(iter(counts))
    detail = ",".join(
        f"{status}:n={count}" for status, count in sorted(counts.items())
    )
    return f"mixed_capture_subsets({detail})"


def condition_summary_rows(captures: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in captures:
        groups[str(row.get("cond", ""))].append(row)
    output: list[dict[str, Any]] = []
    for cond in sorted(groups, key=str.casefold):
        rows = groups[cond]
        tags = {str(row.get("band_tag", "")) for row in rows if row.get("band_tag")}
        if len(tags) > 1:
            raise RuntimeError(f"cross-band aggregation blocked for condition {cond}: {sorted(tags)}")
        identity = rows[0]
        condition_qc = {
            "polarity_lock_status": _join_flags(
                row.get("polarity_lock_status") for row in rows
            ),
            "orientation_status": _orientation_subset_status(rows),
            "quantization_status": _join_flags(
                row.get("quantization_status") for row in rows
            ),
            "exclusion_flags": _join_flags(
                row.get("exclusion_flags") for row in rows
            ),
            "P_method": _join_flags(row.get("P_method") for row in rows),
        }
        for metric in SUMMARY_METRICS:
            result = stats(row.get(metric) for row in rows)
            output.append(
                {
                    "cond": cond,
                    "medium": identity.get("medium"),
                    "medium_display": identity.get("medium_display"),
                    "dataset_type": identity.get("dataset_type"),
                    "freq_label": identity.get("freq_label"),
                    "level_pct": identity.get("level_pct"),
                    "band_tag": next(iter(tags), ""),
                    **condition_qc,
                    "metric": metric,
                    **result,
                }
            )
    return output


def detect_onsets(
    captures: Sequence[dict[str, Any]],
    config: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Require two of ΔQ, loop-energy, and harmonic diagnostics above a passive floor."""

    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in captures:
        if row.get("dataset_type") != "voltage_ladder" or row.get("excluded"):
            continue
        groups[(str(row.get("medium")), str(row.get("freq_label")))].append(row)
    output: list[dict[str, Any]] = []
    quiet_limit = float(
        (config or {}).get("analysis", {}).get(
            "baseline_quiet_harmonic_ratio_max", 0.25
        )
    )
    for (medium, freq_label), rows in sorted(groups.items()):
        levels = sorted(
            {
                float(row["level_pct"])
                for row in rows
                if row.get("level_pct") not in (None, "")
            }
        )
        baseline = [
            row
            for row in rows
            if row.get("level_pct") not in (None, "")
            and float(row["level_pct"]) < 100.0
        ]
        forty_rows = [
            row
            for row in rows
            if row.get("level_pct") not in (None, "")
            and abs(float(row["level_pct"]) - 40.0) < 1e-9
        ]
        forty_harmonic = stats(
            row.get("harmonic_ratio") for row in forty_rows
        )["median"]
        if not baseline:
            baseline_status = "no_baseline"
        elif forty_harmonic is None:
            baseline_status = "baseline_unverified_no_40pct_level"
        elif float(forty_harmonic) <= quiet_limit:
            baseline_status = "baseline_quiet_40pct_harmonic_gate_passed"
        else:
            baseline_status = "baseline_not_quiet_40pct_harmonic_gate_failed"
        baseline_dq = finite(row.get("dQ_cycle_nC") for row in baseline)
        baseline_u = finite(abs(float(row.get("U_cycle_signed_uJ", 0) or 0)) for row in baseline)
        baseline_h = finite(row.get("harmonic_ratio") for row in baseline)

        def threshold(values: np.ndarray, floor: float = 0.0) -> float:
            if not values.size:
                return floor
            median = float(np.median(values))
            mad = float(np.median(np.abs(values - median)))
            return max(median + 5.0 * 1.4826 * mad, floor)

        baseline_lsb = finite(row.get("charge_lsb_nC") for row in baseline)
        lsb_floor = float(np.median(baseline_lsb)) if baseline_lsb.size else 0.0
        dq_floor = max(threshold(baseline_dq), 3.0 * lsb_floor)
        u_floor = threshold(baseline_u)
        h_floor = threshold(baseline_h)
        onset: float | None = None
        evidence = ""
        per_level: list[str] = []
        evaluable = baseline_status != "no_baseline"
        for level in levels:
            level_rows = [row for row in rows if float(row.get("level_pct")) == level]
            med_dq = stats(row.get("dQ_cycle_nC") for row in level_rows)["median"]
            med_u = stats(abs(float(row.get("U_cycle_signed_uJ", 0) or 0)) for row in level_rows)["median"]
            med_h = stats(row.get("harmonic_ratio") for row in level_rows)["median"]
            gates = [
                med_dq is not None and float(med_dq) > dq_floor,
                med_u is not None and float(med_u) > u_floor,
                med_h is not None and float(med_h) > h_floor,
            ]
            per_level.append(f"{level:g}%:{sum(gates)}/3")
            if (
                evaluable
                and onset is None
                and level >= 75.0
                and sum(gates) >= 2
            ):
                onset = level
                evidence = ",".join(
                    name for name, passed in zip(("dQ", "loop_area", "harmonics"), gates) if passed
                )
        if evidence:
            disposition = evidence
            if baseline_status.startswith("baseline_not_quiet"):
                disposition += ";baseline_not_quiet_thresholds_may_be_inflated"
            elif baseline_status.startswith("baseline_unverified"):
                disposition += ";baseline_unverified"
        elif baseline_status == "no_baseline":
            disposition = "not_evaluable_no_baseline"
        elif baseline_status.startswith("baseline_not_quiet"):
            disposition = "not_detected_baseline_not_quiet_thresholds_may_be_inflated"
        elif baseline_status.startswith("baseline_unverified"):
            disposition = "not_detected_baseline_unverified"
        else:
            disposition = "not_detected"
        output.append(
            {
                "medium": medium,
                "medium_display": rows[0].get("medium_display"),
                "freq_label": freq_label,
                "onset_level_pct": onset,
                "evidence": disposition,
                "baseline_status": baseline_status,
                "baseline_40pct_harmonic_ratio": forty_harmonic,
                "baseline_quiet_harmonic_ratio_limit": quiet_limit,
                "level_gate_counts": ";".join(per_level),
                "dQ_threshold_nC": dq_floor,
                "U_threshold_uJ": u_floor,
                "harmonic_threshold": h_floor,
                "N": len(rows),
            }
        )
    return output


def discharge_summary_rows(
    captures: Sequence[dict[str, Any]],
    onset_rows: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    onset = {
        (str(row["medium"]), str(row["freq_label"])): row.get("onset_level_pct")
        for row in onset_rows
    }
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in captures:
        level = row.get("level_pct")
        key = (str(row.get("medium")), str(row.get("freq_label")))
        detected = onset.get(key)
        active = row.get("dataset_type") in {
            "synthesis",
            "july7_8_operational",
        }
        if level not in (None, ""):
            active = active or (
                detected not in (None, "") and float(level) >= float(detected)
            )
        if not active or row.get("excluded"):
            continue
        groups[str(row.get("cond"))].append(row)

    metrics = (
        "f0_kHz",
        "burst_Hz",
        "duty_on_fraction",
        "burst_on_cycles",
        "Vamp_kV",
        "dQ_cycle_nC",
        "dQ_gap_nC",
        "U_cycle_uJ",
        "U_burst_uJ",
        "P_W",
        "P_chain_est_W",
        "P_chain_est_low_W",
        "P_chain_est_high_W",
        "Cd_pF",
        "Ccell_pF",
        "F_same_band",
        "charge_correction_factor",
        "retained_charge_nC",
    )
    output: list[dict[str, Any]] = []
    for cond in sorted(groups, key=str.casefold):
        rows = groups[cond]
        identity = rows[0]
        result: dict[str, Any] = {
            "cond": cond,
            "medium": identity.get("medium"),
            "medium_display": identity.get("medium_display"),
            "dataset_type": identity.get("dataset_type"),
            "freq_label": identity.get("freq_label"),
            "level_pct": identity.get("level_pct"),
            "band_tag": identity.get("band_tag"),
            "orientation_status": _orientation_subset_status(rows),
            "polarity_lock_status": _join_flags(
                row.get("polarity_lock_status") for row in rows
            ),
            "quantization_status": _join_flags(
                row.get("quantization_status") for row in rows
            ),
            "exclusion_flags": _join_flags(
                row.get("exclusion_flags") for row in rows
            ),
            "P_method": _join_flags(row.get("P_method") for row in rows),
            "N_capture": len(rows),
        }
        for metric in metrics:
            summary = stats(row.get(metric) for row in rows)
            result[f"{metric}_median"] = summary["median"]
            result[f"{metric}_p2_5"] = summary["p2_5"]
            result[f"{metric}_p97_5"] = summary["p97_5"]
            result[f"{metric}_N"] = summary["N"]
        output.append(result)
    return output


def _join_flags(values: Iterable[Any]) -> str:
    flags: set[str] = set()
    for value in values:
        for flag in str(value or "").split(";"):
            if flag:
                flags.add(flag)
    return ";".join(sorted(flags))


def synthesis_summary_rows(
    captures: Sequence[dict[str, Any]],
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in captures:
        if row.get("dataset_type") == "synthesis":
            groups[str(row.get("run_key"))].append(row)
    output: list[dict[str, Any]] = []
    for run_key in sorted(groups, key=str.casefold):
        all_rows = groups[run_key]
        rows = [row for row in all_rows if not row.get("excluded")]
        run_cfg = config.get("synthesis_runs", {}).get(run_key, {})
        qc_flags = [
            (
                "unclipped_captures_available"
                if rows
                else "no_unclipped_channel_D_captures"
            )
        ]
        frequency_tolerance = float(
            config["analysis"].get("frequency_crosscheck_tolerance", 0.03)
        )
        frequency_failures = sum(
            row.get("f0_relative_disagreement") not in (None, "")
            and float(row["f0_relative_disagreement"]) > frequency_tolerance
            for row in rows
        )
        if frequency_failures:
            qc_flags.append(
                f"frequency_crosscheck_offset_gt_{100*frequency_tolerance:.0f}pct_n={frequency_failures}"
            )
        duty_suspect = sum(
            str(row.get("duty_status") or "").startswith(
                "duty_detector_suspect"
            )
            for row in rows
        )
        if duty_suspect:
            qc_flags.append(f"duty_detector_suspect_n={duty_suspect}")
        polarity_states = _join_flags(
            row.get("polarity_lock_status") for row in rows
        )
        if "ambiguous" in polarity_states:
            qc_flags.append("polarity_lock_ambiguous")
        result: dict[str, Any] = {
            "run_key": run_key,
            "label": run_cfg.get("label", all_rows[0].get("display_label", run_key)),
            "medium": all_rows[0].get("medium"),
            "medium_display": all_rows[0].get("medium_display"),
            "band_tag": all_rows[0].get("band_tag"),
            "inventory_status": "confirmed" if run_cfg.get("confirmed") else "unknown*",
            "metal_inventory_mol": run_cfg.get("metal_inventory_mol"),
            "electron_stoichiometry_z": run_cfg.get("electron_stoichiometry_z"),
            "N_capture_total": len(all_rows),
            "N_capture": len(rows),
            "qc_status": ";".join(qc_flags),
            "frequency_status": _join_flags(
                row.get("frequency_status") for row in rows
            ),
            "frequency_crosscheck_diagnostic": _join_flags(
                row.get("frequency_crosscheck_diagnostic") for row in rows
            ),
            "duty_status": _join_flags(
                row.get("duty_status") for row in rows
            ),
        }
        fields = (
            "f0_kHz",
            "burst_Hz",
            "duty_on_fraction",
            "burst_on_cycles",
            "Vamp_kV",
            "dQ_cycle_nC",
            "dQ_gap_nC",
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
            "dose_20min_C",
            "capture_gross_dose_C",
            "retained_charge_nC",
            "U_burst_uJ",
            "P_W",
            "P_chain_est_W",
            "P_chain_est_low_W",
            "P_chain_est_high_W",
        )
        for field in fields:
            field_rows = rows
            if field == "duty_on_fraction":
                field_rows = [
                    row
                    for row in rows
                    if not str(row.get("duty_status") or "").startswith(
                        "duty_detector_suspect"
                    )
                ]
            summary = stats(row.get(field) for row in field_rows)
            result[f"{field}_median"] = summary["median"]
            result[f"{field}_p2_5"] = summary["p2_5"]
            result[f"{field}_p97_5"] = summary["p97_5"]
            result[f"{field}_N"] = summary["N"]
        result["P_method"] = _join_flags(row.get("P_method") for row in rows)
        result["polarity_lock_status"] = _join_flags(
            row.get("polarity_lock_status") for row in rows
        )
        result["orientation_status"] = _orientation_subset_status(rows)
        ratio = result.get("gross_rate_tv_ratio_median")
        result["gross_rate_tv_crosscheck_status"] = (
            "not_available"
            if ratio in (None, "")
            else "agrees_within_factor_2"
            if 0.5 <= float(ratio) <= 2.0
            else "outside_factor_2"
        )
        inventory = run_cfg.get("metal_inventory_mol")
        z = run_cfg.get("electron_stoichiometry_z")
        if inventory not in (None, "") and z not in (None, ""):
            required_C = float(inventory) * float(z) * 96485.33212
            rate = result.get("gross_rate_C_s_median")
            result["minutes_per_inventory_electron_equivalent"] = (
                required_C / (float(rate) * 60.0) if rate and float(rate) > 0 else None
            )
        else:
            result["minutes_per_inventory_electron_equivalent"] = None
        output.append(result)
    return output


def frequency_consistency_rows(
    captures: Sequence[dict[str, Any]],
    relative_tolerance: float,
) -> list[dict[str, Any]]:
    """R3: require three measured LF carrier anchors in the same medium/configuration."""

    grouped: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for row in captures:
        level = row.get("level_pct")
        if (
            row.get("dataset_type") == "voltage_ladder"
            and row.get("band_tag") == "LF-geometric"
            and level not in (None, "")
            and float(level) <= 90.0
            and not row.get("excluded")
        ):
            grouped[str(row.get("medium"))][str(row.get("freq_label"))].append(row)
    output: list[dict[str, Any]] = []
    media = sorted(
        {
            str(row.get("medium"))
            for row in captures
            if row.get("dataset_type") == "voltage_ladder"
        }
    )
    for medium in media:
        values: dict[str, float] = {}
        counts: dict[str, int] = {}
        for label, rows in grouped.get(medium, {}).items():
            summary = stats(row.get("Cline_pF") for row in rows)
            if summary["median"] is not None:
                values[label] = float(summary["median"])
                counts[label] = int(summary["N"])
        expected = ("4 kHz", "10 kHz", "20 kHz")
        present = [label for label in expected if label in values]
        status = "insufficient_genuine_LF_anchors"
        span = None
        if len(present) == 3:
            array = np.asarray([values[label] for label in expected])
            span = float((np.max(array) - np.min(array)) / np.median(array))
            status = "consistent" if span <= relative_tolerance else "inconsistent"
        output.append(
            {
                "medium": medium,
                "C_4kHz_pF": values.get("4 kHz"),
                "N_4kHz": counts.get("4 kHz", 0),
                "C_10kHz_pF": values.get("10 kHz"),
                "N_10kHz": counts.get("10 kHz", 0),
                "C_20kHz_pF": values.get("20 kHz"),
                "N_20kHz": counts.get("20 kHz", 0),
                "relative_span": span,
                "tolerance": relative_tolerance,
                "R3_status": status,
                "band_tag": "LF-geometric",
            }
        )
    return output


def factor_frequency_rows(
    captures: Sequence[dict[str, Any]],
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    """Recompute F only from passive/active observations in one band/configuration."""

    groups: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in captures:
        if row.get("dataset_type") != "voltage_ladder" or row.get("excluded"):
            continue
        key = (
            str(row.get("medium")),
            str(row.get("freq_label")),
            str(row.get("band_tag")),
            str(row.get("dataset_type")),
        )
        groups[key].append(row)
    output: list[dict[str, Any]] = []
    for (medium, freq_label, tag, dataset), rows in sorted(groups.items()):
        passive = [
            row
            for row in rows
            if row.get("level_pct") not in (None, "")
            and float(row["level_pct"]) <= 90.0
            and row.get("capacitance_valid")
        ]
        active = [
            row
            for row in rows
            if row.get("level_pct") not in (None, "")
            and float(row["level_pct"]) >= 100.0
            and row.get("Cd_pF") not in (None, "")
        ]
        ccell = stats(row.get("Cline_pF") for row in passive)
        cd = stats(row.get("Cd_pF") for row in active)
        f_values: list[float] = []
        if ccell["median"] is not None:
            ccell_value = float(ccell["median"])
            for row in active:
                cd_value = float(row["Cd_pF"])
                if cd_value > ccell_value:
                    f_values.append(cd_value / (cd_value - ccell_value))
        factor = stats(f_values)
        status = "supported_same_band" if factor["N"] else "not_identifiable_same_band"
        output.append(
            {
                "medium": medium,
                "freq_label": freq_label,
                "measured_f0_kHz": stats(row.get("f0_kHz") for row in rows)["median"],
                "band_tag": tag,
                "Ccell_pF": ccell["median"],
                "Ccell_N": ccell["N"],
                "Cd_pF": cd["median"],
                "Cd_N": cd["N"],
                "F_median": factor["median"],
                "F_p2_5": factor["p2_5"],
                "F_p97_5": factor["p97_5"],
                "F_N": factor["N"],
                "status": status,
                "provenance": "same measured-frequency band and acquisition configuration",
            }
        )
    carrier_factor = config["calibration"]["same_band_charge_factor"]
    output.append(
        {
            "medium": "operational_carrier_reference",
            "freq_label": "measured carrier",
            "measured_f0_kHz": None,
            "band_tag": "carrier-transfer",
            "Ccell_pF": None,
            "Ccell_N": None,
            "Cd_pF": None,
            "Cd_N": None,
            "F_median": carrier_factor["value"],
            "F_p2_5": None,
            "F_p97_5": None,
            "F_N": None,
            "status": "v1.1_same_band_reference",
            "provenance": carrier_factor["provenance"],
        }
    )
    return output


def presentation_numbers(
    captures: Sequence[dict[str, Any]],
    synthesis: Sequence[dict[str, Any]],
    onsets: Sequence[dict[str, Any]],
    consistency: Sequence[dict[str, Any]],
    factors: Sequence[dict[str, Any]],
    config: dict[str, Any],
) -> dict[str, Any]:
    geometric: dict[str, Any] = {}
    for row in consistency:
        medium = str(row["medium"])
        geometric[medium] = {
            "value": None,
            "ci": None,
            "band": "LF-geometric",
            "provenance": "Measured-frequency R3 consistency gate",
            "flags": [row["R3_status"]],
        }
    onset_pack = {
        f"{row['medium']}|{row['freq_label']}": {
            "value": row.get("onset_level_pct"),
            "ci": None,
            "band": _modal_band(
                capture.get("band_tag")
                for capture in captures
                if capture.get("medium") == row.get("medium")
                and capture.get("freq_label") == row.get("freq_label")
            ),
            "provenance": "Two-of-three ΔQ/loop-area/harmonic onset gate",
            "flags": [row.get("evidence")],
        }
        for row in onsets
    }
    dose_pack: dict[str, Any] = {}
    for row in synthesis:
        dose_pack[str(row["run_key"])] = {
            "value": row.get("gross_rate_C_s_median"),
            "ci": [row.get("gross_rate_C_s_p2_5"), row.get("gross_rate_C_s_p97_5")],
            "band": row.get("band_tag"),
            "provenance": "Channel-D zero-crossing charge throughput across captures",
            "flags": [
                value
                for value in (
                    row.get("inventory_status"),
                    row.get("qc_status"),
                    row.get("frequency_status"),
                    row.get("polarity_lock_status"),
                    row.get("orientation_status"),
                )
                if value
            ],
        }
    factor_pack = {
        f"{row['medium']}|{row['freq_label']}": {
            "value": row.get("F_median"),
            "ci": [row.get("F_p2_5"), row.get("F_p97_5")],
            "band": row.get("band_tag"),
            "provenance": row.get("provenance"),
            "flags": [row.get("status")],
        }
        for row in factors
    }
    chain = config["chain_model"]
    return {
        "analysis_version": config["analysis"]["version"],
        "geometric_capacitance": geometric,
        "onset_levels": onset_pack,
        "synthesis_gross_charge_rate_C_s": dose_pack,
        "same_band_F_by_frequency": factor_pack,
        "same_band_charge_factor": {
            "value": config["calibration"]["same_band_charge_factor"]["value"],
            "ci": None,
            "band": "carrier-transfer",
            "provenance": config["calibration"]["same_band_charge_factor"]["provenance"],
            "flags": ["same-band only"],
        },
        "chain_model": {
            "C_true_pF": {
                "value": float(chain["C_true_F"]) * 1e12,
                "ci": [float(value) * 1e12 for value in chain["C_true_range_F"]],
                "band": "carrier-transfer",
                "provenance": chain["provenance"],
                "flags": ["order-of-magnitude\u2020"],
            },
            "L_H": {
                "value": chain["L_H"],
                "ci": chain["L_range_H"],
                "band": "carrier-transfer",
                "provenance": chain["provenance"],
                "flags": ["order-of-magnitude\u2020"],
            },
            "R_ohm": {
                "value": chain["R_ohm"],
                "ci": chain["R_range_ohm"],
                "band": "carrier-transfer",
                "provenance": chain["provenance"],
                "flags": ["order-of-magnitude\u2020"],
            },
            "f_res_kHz": {
                "value": float(chain["f_res_Hz"]) / 1000.0,
                "ci": [float(value) / 1000.0 for value in chain["f_res_range_Hz"]],
                "band": "carrier-transfer",
                "provenance": chain["provenance"],
                "flags": ["order-of-magnitude\u2020"],
            },
        },
        "open_items": config.get("open_items", []),
    }


def _modal_band(values: Iterable[Any]) -> str:
    counts: dict[str, int] = defaultdict(int)
    for value in values:
        if value:
            counts[str(value)] += 1
    return max(counts, key=counts.get) if counts else ""


def results_markdown(
    manifests: Sequence[dict[str, Any]],
    captures: Sequence[dict[str, Any]],
    synthesis: Sequence[dict[str, Any]],
    consistency: Sequence[dict[str, Any]],
    config: dict[str, Any],
    discovery_errors: Sequence[str],
) -> str:
    unknowns = config.get("open_items", [])
    parsed = sum(row.get("parse_status") == "ok" for row in manifests)
    failed = [row for row in manifests if row.get("parse_status") != "ok"]
    bands = defaultdict(int)
    for row in captures:
        bands[str(row.get("band_tag"))] += 1
    f0_summary = stats(row.get("f0_kHz") for row in captures)
    frequency_crosscheck_failures = sum(
        "disagreement" in str(row.get("frequency_status") or "").lower()
        for row in captures
    )
    orientation_failures = sum(
        not row.get("excluded")
        and "failed" in str(row.get("orientation_status") or "").lower()
        for row in captures
    )
    polarity_ambiguities = sum(
        not row.get("excluded")
        and "ambiguous" in str(row.get("polarity_lock_status") or "").lower()
        for row in captures
    )
    quantization_limited = sum(
        str(row.get("quantization_status") or "").lower()
        == "quantization_limited"
        for row in captures
    )
    retained_identified = sum(
        row.get("retained_charge_nC") not in (None, "")
        for row in captures
        if row.get("dataset_type") == "synthesis" and not row.get("excluded")
    )
    lines = [
        "# Lissajous v2 results",
        "",
        "## Open starred items",
        "",
    ]
    lines.extend(f"- {item}" for item in unknowns)
    lines.extend(
        [
            "",
            "## Measured-frequency conclusion",
            "",
            (
                "The folder labels `4 kHz`, `10 kHz`, and `20 kHz` describe burst "
                "repetition in the inspected operational records, not the electrical "
                "carrier. Across all captures, measured Channel-A f0 has median "
                f"{_fmt(f0_summary['median'])} kHz and a 2.5th–97.5th percentile "
                f"span of {_fmt(f0_summary['p2_5'])}–{_fmt(f0_summary['p97_5'])} "
                "kHz. These capacitances are therefore tagged `carrier-transfer`; "
                "they are not promoted to geometric capacitances."
            ),
            "",
            (
                f"Parsed {parsed} of {len(manifests)} logical CSV captures. "
                f"Capacitance rows: {bands.get('LF-geometric', 0)} LF-geometric and "
                f"{bands.get('carrier-transfer', 0)} carrier-transfer."
            ),
            "",
            "## R3 frequency-consistency gate",
            "",
            "| Medium | 4 kHz | 10 kHz | 20 kHz | Relative span | Status |",
            "|---|---:|---:|---:|---:|---|",
        ]
    )
    for row in consistency:
        lines.append(
            "| {medium} | {c4} | {c10} | {c20} | {span} | {status} |".format(
                medium=row["medium"],
                c4=_fmt(row.get("C_4kHz_pF")),
                c10=_fmt(row.get("C_10kHz_pF")),
                c20=_fmt(row.get("C_20kHz_pF")),
                span=_fmt(row.get("relative_span")),
                status=row["R3_status"],
            )
        )
    lines.extend(
        [
            "",
            (
                "Until three same-configuration, genuinely low-frequency carrier "
                "measurements pass this gate, the word “geometric” is unsupported "
                "for the four operational media."
            ),
            "",
            "## Synthesis charge delivery",
            "",
            (
                "Channel-D charge throughput, duty, and dose remain reportable at the "
                "carrier band. Positive and negative quantities are electrical charge "
                "equivalents; the waveform does not separate electrons from negative ions. "
                "Peak flow fields are half-cycle-average charge-equivalent rates, not "
                "nanosecond microdischarge peaks."
            ),
            "",
            "| Run | f0 (kHz) | Burst (Hz) | Duty | Gross rate (C/min) | Power (W) | 20 min dose (C) | N | QC |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for row in synthesis:
        rate_min = (
            float(row["gross_rate_C_s_median"]) * 60.0
            if row.get("gross_rate_C_s_median") not in (None, "")
            else None
        )
        lines.append(
            "| {label} | {f0} | {burst} | {duty} | {rate} | {power} | {dose} | {n} | {qc} |".format(
                label=row.get("label", row["run_key"]),
                f0=_fmt(row.get("f0_kHz_median")),
                burst=_fmt(row.get("burst_Hz_median")),
                duty=_fmt(row.get("duty_on_fraction_median")),
                rate=_fmt(rate_min),
                power=_fmt(row.get("P_W_median")),
                dose=_fmt(row.get("dose_20min_C_median")),
                n=row.get("N_capture", 0),
                qc=row.get("qc_status", ""),
            )
        )
    lines.extend(
        [
            "",
            (
                "Power uses burst-period shoelace energy multiplied by the measured "
                "burst rate whenever a burst is resolved. It is withheld—not replaced "
                "by a carrier-cycle estimate—when no valid nonnegative burst energy "
                "is available, including orientation rejection or an incomplete "
                "burst-period window. "
                "Cycle-energy × measured carrier f0 is used only for records classified "
                "as continuous."
            ),
            "",
            (
                "Net retained charge is not identifiable from the synthesis records "
                "because quiet pre- and post-burst edges were unavailable."
                if retained_identified == 0
                else (
                    f"Net retained charge passed the quiet-edge gate in "
                    f"{retained_identified} synthesis captures."
                )
            ),
            "",
            "## Measurement-chain model",
            "",
            (
                "The series-chain element values and 159–164 kHz resonance are retained "
                "as order-of-magnitude estimates†. The fit describes the inflation/sign-"
                "flip trend; it is not a replacement for a true LF cell measurement."
            ),
            "",
            "## Parsing and QC exceptions",
            "",
        ]
    )
    if discovery_errors:
        lines.extend(f"- Discovery: {error}" for error in discovery_errors)
    if failed:
        lines.extend(
            f"- `{row.get('path')}`: {row.get('parse_status')} — {row.get('parse_error', '')}"
            for row in failed
        )
    if not discovery_errors and not failed:
        lines.append("- None.")
    lines.extend(
        [
            (
                f"- Frequency FFT/zero-cross cross-check failures: "
                f"{frequency_crosscheck_failures}/{len(captures)} captures; "
                "the spectral f0 is retained with a named QC flag."
            ),
            (
                f"- Quantization-limited (< configured code threshold): "
                f"{quantization_limited}/{len(captures)} captures."
            ),
            (
                f"- Non-excluded captures with failed energy orientation: "
                f"{orientation_failures}; affected energy/power values are withheld."
            ),
            (
                f"- Non-excluded captures with an ambiguous condition-level polarity "
                f"lock: {polarity_ambiguities}; sign-sensitive quantities require review."
            ),
        ]
    )
    lines.extend(
        [
            "",
            "Statistical intervals are capture-level 2.5th–97.5th percentiles. "
            "The separate `syst_frac` field carries the configured correlated calibration uncertainty.",
            "",
        ]
    )
    return "\n".join(lines)


def _fmt(value: Any) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "—"
    if not math.isfinite(number):
        return "—"
    return f"{number:.4g}"


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def assert_no_retired_numbers(root: Path) -> None:
    """Fail before hand-off if a retired headline leaks into a v2 text output."""

    offenders: list[str] = []
    for path in root.rglob("*"):
        if path.suffix.casefold() not in {".csv", ".json", ".md", ".yaml"} or not path.is_file():
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        for token in FORBIDDEN_OUTPUT_TOKENS:
            if token in text:
                offenders.append(f"{path}: {token}")
    if offenders:
        raise RuntimeError("retired-number guard failed:\n" + "\n".join(offenders))
