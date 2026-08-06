import unittest
from types import SimpleNamespace

import numpy as np

from DBD_Surface_Charge_Analysis import (
    CalibrationModel,
    FileObservation,
    LobeObservation,
    active_cd_three_level_fit,
    apparent_power_from_burst_energy_mW,
    bootstrap_median_draws,
    complex_capacitance_least_squares,
    extract_lobes,
    geometry_cd_pF,
    lissajous_burst_periods,
    locked_condition_sign,
    per_file_metrics,
    quiet_edge_charge,
    scan_secant_bootstrap,
    summary_metric_reportable,
    target_name,
)
from Lissajous_Scan_Analysis import Condition, MemberRecord


def observation(
    level="40",
    vote=1,
    lobes=None,
    capture=1,
    x=None,
    y=None,
):
    condition = Condition("BMIM_nitrate", 20)
    return FileObservation(
        record=MemberRecord(
            member=f"capture_{capture:02d}.csv",
            condition=condition,
            level_percent=None if level == "MAX" else int(level),
            is_maximum=level == "MAX",
            capture_index=capture,
        ),
        level_label=level,
        carrier_Hz=120_000.0,
        detected_burst_Hz=20_000.0,
        burst_detection_method="synthetic",
        burst_frequency_relative_error=0.0,
        duration_s=1.0e-3,
        voltage_pp_kV=4.0,
        cstar_raw_F=complex(25e-12, -5e-12),
        sign_vote=vote,
        lobes=[] if lobes is None else lobes,
        raw_extrema_x_kV=x,
        raw_extrema_y_nC=y,
        charge_lsb_nC=0.1,
        clipping_flag=False,
        skipped_rows=0,
        quiet_edge_change_raw_nC=None,
        quiet_edge_status="not_measured_no_quiet_record_edges",
        qv_voltage_kV=np.asarray([0.0, 1.0]),
        qv_charge_raw_nC=np.asarray([0.0, 1.0]),
    )


def calibration():
    return CalibrationModel(
        condition=Condition("BMIM_nitrate", 20),
        sign=1,
        sign_agreement=1.0,
        sign_votes=12,
        sign_status="auto_locked",
        passive_complete=True,
        passive_status="supported",
        cprime_pF=25.0,
        cprime_ci_low_pF=24.0,
        cprime_ci_high_pF=26.0,
        closs_pF=5.0,
        closs_ci_low_pF=4.0,
        closs_ci_high_pF=6.0,
        tan_delta=0.2,
        cprime_relative_span=0.02,
        passive_slopes_nC_per_kV={-1: 2.0, 1: 2.0},
        passive_slope_mad_nC_per_kV={-1: 0.1, 1: 0.1},
        passive_threshold_nC={-1: 0.5, 1: 0.5},
        passive_file_slopes={-1: [2.0], 1: [2.0]},
        charge_lsb_nC=0.1,
        scan_cd_pF=None,
        scan_cd_ci_low_pF=None,
        scan_cd_ci_high_pF=None,
        scan_cd_physical_fraction=0.0,
        scan_cd_status="not_identifiable_only_two_active_amplitudes",
        geometry_cd_pF=50.0,
        geometry_factor=2.0,
        geometry_factor_low=1.8,
        geometry_factor_high=2.2,
        factor_source="full_base_pyrex_geometry_scenario",
        evidence_tier="exploratory_model_dependent",
        failed_gates=[],
    )


class SignAndExtractionTests(unittest.TestCase):
    def test_condition_sign_is_locked_from_passive_majority(self):
        rows = [observation(str(level), -1, capture=i) for i, level in enumerate([40, 60, 75] * 4)]
        rows.append(observation("MAX", 1, capture=99))
        sign, agreement, count, status = locked_condition_sign(rows, "auto", 0.90)
        self.assertEqual(sign, -1)
        self.assertEqual(count, 12)
        self.assertEqual(agreement, 1.0)
        self.assertEqual(status, "auto_locked")

    def test_carrier_halfcycles_are_found(self):
        carrier = 120_000.0
        time_s = np.arange(0.0, 1.0e-3, 0.1e-6)
        phase = 2.0 * np.pi * carrier * time_s
        voltage = 2_000.0 * np.sin(phase)
        charge = 1.0e-9 * np.sin(phase - 0.25)
        lobes = extract_lobes(time_s, voltage, charge, carrier, 4)
        self.assertGreater(len(lobes), 150)
        durations = np.asarray([row.duration_s for row in lobes])
        self.assertAlmostEqual(float(np.median(durations)), 0.5 / carrier, delta=0.03 / carrier)
        self.assertEqual({row.voltage_polarity for row in lobes}, {-1, 1})

    def test_target_mapping_is_explicit(self):
        self.assertEqual(target_name(-1, True), "negative")
        self.assertEqual(target_name(1, True), "positive")
        self.assertEqual(target_name(-1, False), "positive")


class MetricTests(unittest.TestCase):
    def test_burst_energy_to_power_conversion(self):
        self.assertAlmostEqual(
            apparent_power_from_burst_energy_mW(10.0, 20_000.0),
            200.0,
        )

    def test_complex_capacitance_fit_recovers_quantized_passive_waveform(self):
        carrier_hz = 100_000.0
        sample_hz = 10_000_000.0
        time_s = np.arange(0.0, 2.0e-3, 1.0 / sample_hz)
        phase = 2.0 * np.pi * carrier_hz * time_s
        voltage_amplitude_v = 2_000.0
        expected_cprime_f = 25.0e-12
        expected_closs_f = 6.0e-12

        voltage_v = voltage_amplitude_v * np.sin(phase)
        charge_c = (
            expected_cprime_f * voltage_v
            - expected_closs_f * voltage_amplitude_v * np.cos(phase)
            + 2.0e-9
            + 4.0e-9 * (time_s - np.mean(time_s)) / np.ptp(time_s)
        )

        # Deliberately coarsen both channels to reproduce the ADC-code grid
        # that makes endpoint/lobe slopes quantization-locked.
        voltage_v = np.round(voltage_v / 157.0) * 157.0
        charge_c = np.round(charge_c / 7.874e-9) * 7.874e-9

        capacitance, _, residual_rms_n_c, signal_pp_n_c = (
            complex_capacitance_least_squares(
                time_s, voltage_v, charge_c, carrier_hz
            )
        )

        self.assertIsNotNone(capacitance)
        self.assertAlmostEqual(capacitance.real * 1.0e12, 25.0, delta=0.75)
        self.assertAlmostEqual(-capacitance.imag * 1.0e12, 6.0, delta=0.50)
        self.assertGreater(signal_pp_n_c, 90.0)
        self.assertLess(residual_rms_n_c, 0.5 * 7.874)

    def test_lissajous_energy_matches_ellipse_and_is_orientation_invariant(self):
        count = 8_192
        period_s = 50.0e-6
        phase = np.linspace(0.0, 2.0 * np.pi, count, endpoint=False)
        time_s = phase * period_s / (2.0 * np.pi)
        voltage_amplitude_kv = 2.0
        charge_amplitude_nc = 10.0
        phase_lag = 0.35
        voltage_v = 1.0e3 * voltage_amplitude_kv * np.cos(phase)
        charge_c = (
            charge_amplitude_nc
            * 1.0e-9
            * np.cos(phase - phase_lag)
        )
        expected_u_j = (
            np.pi
            * voltage_amplitude_kv
            * charge_amplitude_nc
            * abs(np.sin(phase_lag))
        )

        forward = lissajous_burst_periods(
            time_s, voltage_v, charge_c, [slice(None)], None
        )[0]
        reversed_path = lissajous_burst_periods(
            time_s, voltage_v[::-1], charge_c[::-1], [slice(None)], None
        )[0]
        opposite_charge = lissajous_burst_periods(
            time_s, voltage_v, -charge_c, [slice(None)], None
        )[0]

        self.assertAlmostEqual(forward.energy_uJ, expected_u_j, delta=0.01)
        self.assertAlmostEqual(reversed_path.energy_uJ, forward.energy_uJ, places=10)
        self.assertAlmostEqual(opposite_charge.energy_uJ, forward.energy_uJ, places=10)
        self.assertAlmostEqual(
            reversed_path.signed_energy_uJ, -forward.signed_energy_uJ, places=10
        )
        self.assertAlmostEqual(
            opposite_charge.signed_energy_uJ, -forward.signed_energy_uJ, places=10
        )

    def test_in_phase_qv_trace_has_negligible_enclosed_energy(self):
        count = 4_096
        phase = np.linspace(0.0, 2.0 * np.pi, count, endpoint=False)
        time_s = phase * 50.0e-6 / (2.0 * np.pi)
        voltage_v = 2_000.0 * np.cos(phase)
        charge_c = 10.0e-9 * np.cos(phase)

        result = lissajous_burst_periods(
            time_s, voltage_v, charge_c, [slice(None)], None
        )[0]

        self.assertLess(result.energy_uJ, 1.0e-10)

    def test_per_file_metrics_subtract_passive_charge_before_model_factor(self):
        lobes = [
            LobeObservation(
                0,
                -1,
                1.0,
                4.0e-6,
                1e-5,
                12.0,
                background_cprime_basis_nC_per_pF=2.0 / 25.0,
            ),
            LobeObservation(
                0,
                1,
                1.0,
                4.0e-6,
                2e-5,
                7.0,
                background_cprime_basis_nC_per_pF=2.0 / 25.0,
            ),
            LobeObservation(
                1,
                -1,
                2.0,
                4.0e-6,
                3e-5,
                24.0,
                background_cprime_basis_nC_per_pF=4.0 / 25.0,
            ),
            LobeObservation(
                1,
                1,
                2.0,
                4.0e-6,
                4e-5,
                14.0,
                background_cprime_basis_nC_per_pF=4.0 / 25.0,
            ),
        ]
        metrics = per_file_metrics(observation("MAX", lobes=lobes), calibration(), 2.0, True)
        self.assertAlmostEqual(metrics["negative_record_total_nC"], 60.0)
        self.assertAlmostEqual(metrics["positive_record_total_nC"], 30.0)
        self.assertAlmostEqual(metrics["negative_peak_envelope_halfcycle_p95_nC"], 39.0)
        self.assertAlmostEqual(metrics["positive_peak_envelope_halfcycle_p95_nC"], 19.5)
        self.assertAlmostEqual(metrics["transport_imbalance_nC"], -30.0)
        self.assertGreater(metrics["negative_peak_halfcycle_average_equivalent_flow_p95_per_s"], 0.0)

    def test_quiet_edges_measure_terminal_offset_only_when_both_are_quiet(self):
        carrier = 120_000.0
        time_s = np.arange(0.0, 2.0e-3, 0.1e-6)
        voltage = 2_000.0 * np.sin(2.0 * np.pi * carrier * time_s)
        voltage[:300] = 0.0
        voltage[-300:] = 0.0
        charge = np.zeros_like(time_s)
        charge[-300:] = 1.0e-9
        value, status = quiet_edge_charge(time_s, voltage, charge, carrier)
        self.assertAlmostEqual(value, 1.0, places=6)
        self.assertEqual(status, "external_terminal_change_measured_dc_coupling_unverified")

    def test_two_level_active_slope_is_diagnostic_and_positive(self):
        rows = [
            observation("105", capture=1, x=2.50, y=80.0),
            observation("105", capture=2, x=2.52, y=80.5),
            observation("115", capture=3, x=3.00, y=100.0),
            observation("115", capture=4, x=3.02, y=100.5),
        ]
        slope, low, high, physical = scan_secant_bootstrap(
            rows, 1, 300, np.random.default_rng(10)
        )
        self.assertGreater(slope, 0.0)
        self.assertGreater(low, 0.0)
        self.assertGreater(high, low)
        self.assertEqual(physical, 1.0)

    def test_three_level_active_cd_fit_passes_consistent_provisional_scan(self):
        rows = []
        centers = {100: 2.54, 105: 3.22, 115: 4.00}
        capture = 0
        for level, center in centers.items():
            for offset in (-0.02, -0.01, 0.0, 0.01, 0.02):
                capture += 1
                x = center + offset
                lobe = LobeObservation(
                    0,
                    -1,
                    x / 2.0,
                    4.0e-6,
                    1.0e-5,
                    20.0,
                    0.0,
                    0.0,
                )
                rows.append(
                    observation(
                        str(level),
                        lobes=[lobe],
                        capture=capture,
                        x=x,
                        y=214.0 * x - 392.0,
                    )
                )
        args = SimpleNamespace(
            bootstrap_replicates=200,
            bootstrap_block_files=1,
            active_cd_min_clean_captures=4,
            active_cd_min_r_squared=0.98,
            active_cd_max_pairwise_relative_span=0.35,
            active_cd_min_breakdown_active_fraction=0.50,
        )
        result = active_cd_three_level_fit(
            rows,
            sign=1,
            cprime_pF=28.0,
            cprime_draws=np.full(200, 28.0),
            closs_pF=5.0,
            passive_threshold_nC={-1: 1.0, 1: 1.0},
            args=args,
            rng=np.random.default_rng(11),
        )
        self.assertEqual(
            result["status"],
            "supported_provisional_three_level_effective_Cd",
        )
        self.assertAlmostEqual(result["slope_pF"], 214.0, delta=0.5)
        self.assertGreater(result["r_squared"], 0.999)
        self.assertEqual(result["physical_fraction"], 1.0)


class UncertaintyTests(unittest.TestCase):
    def test_moving_block_bootstrap_is_reproducible(self):
        first = bootstrap_median_draws([1, 2, 3, 4, 5], 200, np.random.default_rng(4), 2)
        second = bootstrap_median_draws([1, 2, 3, 4, 5], 200, np.random.default_rng(4), 2)
        np.testing.assert_array_equal(first, second)

    def test_full_base_pyrex_geometry_has_expected_scale(self):
        value = geometry_cd_pF(4.0, 1.0, 4.5)
        self.assertGreater(value, 45.0)
        self.assertLess(value, 55.0)

    def test_plot_reportability_matches_passive_resolution_gate(self):
        key = "negative_peak_envelope_halfcycle_p95_nC"
        row = {
            "Ccell_status": "supported_effective_complex_at_carrier",
            "model_sensitivity_draw_valid_fraction": 1.0,
            key: 10.0,
            f"{key}_analysis_ci_low": 8.0,
            f"{key}_analysis_ci_high": 12.0,
            "negative_peak_charge_resolution_threshold_nC": 2.0,
            f"{key}_passive_90_holdout_p95": 3.0,
            "negative_resolved_halfcycle_fraction": 0.8,
            "scan_transferred": False,
        }
        self.assertTrue(
            summary_metric_reportable(
                row, "negative", "peak_envelope_halfcycle_p95_nC"
            )
        )
        row[f"{key}_analysis_ci_low"] = 2.5
        self.assertFalse(
            summary_metric_reportable(
                row, "negative", "peak_envelope_halfcycle_p95_nC"
            )
        )


if __name__ == "__main__":
    unittest.main()
