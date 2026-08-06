import os
import tempfile
import unittest
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np

from Lissajous_Figures import (
    AnalysisError,
    apparent_charge_equivalent_rates,
    duty_activity_mask,
    estimate_fundamental_frequency,
    resolve_current_polarity,
    save_processed_csv,
    select_two_duty_cycles,
)


class CurrentPolarityTests(unittest.TestCase):
    @staticmethod
    def synthetic_trace(sign: int = 1, noisy: bool = False):
        frequency_hz = 20_000.0
        time_s = np.arange(0.0, 80.0 / frequency_hz, 1.0 / 2_000_000.0)
        omega = 2.0 * np.pi * frequency_hz
        voltage_v = 5_000.0 * np.sin(omega * time_s)

        # A slowly varying hysteretic monitor charge gives non-zero charge
        # transfer between equal-voltage half-cycle boundaries.  The upstream
        # current also includes a much larger probe/cable displacement term,
        # which should cancel when integrated between those boundaries.
        envelope = 1.0 + 0.30 * np.sin(2.0 * np.pi * 1_370.0 * time_s)
        monitor_charge_c = -40.0e-9 * envelope * np.cos(omega * time_s)
        monitor_voltage_v = monitor_charge_c / 0.1e-6
        transferred_current_a = np.gradient(monitor_charge_c, time_s)
        parasitic_current_a = 18.0e-12 * np.gradient(voltage_v, time_s)
        current_a = transferred_current_a + parasitic_current_a

        if noisy:
            rng = np.random.default_rng(4815)
            current_a += rng.normal(0.0, 0.03 * np.std(current_a), len(current_a))
            spike_indices = rng.choice(len(current_a), size=50, replace=False)
            current_a[spike_indices] += rng.normal(0.0, 8.0 * np.std(current_a), len(spike_indices))
            current_a += 0.017
        return time_s, voltage_v, sign * current_a, monitor_voltage_v, frequency_hz

    def test_auto_keeps_correct_current(self):
        trace = self.synthetic_trace(sign=1)
        decision = resolve_current_polarity("auto", *trace)
        self.assertEqual(decision.sign, 1)
        self.assertGreater(decision.half_cycle_charge_correlation_raw, 0.4)

    def test_auto_flips_inverted_current(self):
        trace = self.synthetic_trace(sign=-1)
        decision = resolve_current_polarity("auto", *trace)
        self.assertEqual(decision.sign, -1)
        self.assertLess(decision.half_cycle_charge_correlation_raw, -0.4)

    def test_noise_offsets_and_spikes_do_not_change_sign(self):
        trace = self.synthetic_trace(sign=-1, noisy=True)
        decision = resolve_current_polarity("auto", *trace)
        self.assertEqual(decision.sign, -1)
        self.assertIn(decision.confidence, {"medium", "high"})

    def test_no_monitor_charge_is_ambiguous(self):
        time_s, voltage_v, current_a, _, frequency_hz = self.synthetic_trace(sign=1)
        monitor_voltage_v = 0.2 * voltage_v / np.max(np.abs(voltage_v))
        with self.assertRaisesRegex(AnalysisError, "current-polarity detection"):
            resolve_current_polarity(
                "auto",
                time_s,
                voltage_v,
                current_a,
                monitor_voltage_v,
                frequency_hz,
            )

    def test_manual_override_always_wins(self):
        trace = self.synthetic_trace(sign=-1)
        decision = resolve_current_polarity("1", *trace)
        self.assertEqual(decision.sign, 1)
        self.assertIn("manual override", decision.method)

    def test_apparent_charge_rates_are_invariant_to_monitor_inversion(self):
        time_s, voltage_v, _, monitor_voltage_v, frequency_hz = self.synthetic_trace(sign=1)
        normal = apparent_charge_equivalent_rates(
            time_s,
            voltage_v,
            monitor_voltage_v,
            0.1e-6,
            frequency_hz,
        )
        inverted = apparent_charge_equivalent_rates(
            time_s,
            voltage_v,
            -monitor_voltage_v,
            0.1e-6,
            frequency_hz,
        )
        self.assertEqual(normal["monitor_charge_polarity_applied"], 1)
        self.assertEqual(inverted["monitor_charge_polarity_applied"], -1)
        for polarity in ("negative_applied_voltage", "positive_applied_voltage"):
            self.assertAlmostEqual(
                normal[polarity]["record_average_singly_charged_equivalent_rate_per_s"],
                inverted[polarity]["record_average_singly_charged_equivalent_rate_per_s"],
            )
            self.assertGreater(
                normal[polarity]["singly_charged_equivalents_per_half_cycle_p95"],
                normal[polarity]["singly_charged_equivalents_per_half_cycle_median"],
            )
            self.assertGreater(normal[polarity]["charge_direction_match_fraction"], 0.95)
        self.assertIsNone(normal["apparent_retained_terminal_charge_C"])

    def test_current_sign_uses_physically_signed_monitor_charge(self):
        trace = self.synthetic_trace(sign=1)
        time_s, voltage_v, current_a, monitor_voltage_v, frequency_hz = trace
        charge_result = apparent_charge_equivalent_rates(
            time_s,
            voltage_v,
            -monitor_voltage_v,
            0.1e-6,
            frequency_hz,
        )
        signed_monitor_v = charge_result["monitor_charge_polarity_applied"] * -monitor_voltage_v
        decision = resolve_current_polarity(
            "auto",
            time_s,
            voltage_v,
            current_a,
            signed_monitor_v,
            frequency_hz,
        )
        self.assertEqual(decision.sign, 1)

    def test_frequency_estimator_tolerates_isolated_missing_rows(self):
        frequency_hz = 127_000.0
        time_s = np.arange(0.0, 0.01, 104.0e-9)
        voltage_v = np.sin(2.0 * np.pi * frequency_hz * time_s)
        keep = np.ones(len(time_s), dtype=bool)
        keep[[12_345, 56_789, 80_000]] = False
        estimated = estimate_fundamental_frequency(time_s[keep], voltage_v[keep])
        self.assertIsNotNone(estimated)
        self.assertLess(abs(estimated - frequency_hz), 150.0)

    def test_processed_csv_preserves_raw_and_corrected_current(self):
        time_ms = np.array([0.0, 0.1, 0.2])
        voltage_input = np.array([0.0, 1.0, 0.0])
        voltage_dbd = 1_000.0 * voltage_input
        current_input = np.array([0.1, -0.2, 0.3])
        current_corrected = -current_input
        monitor_voltage = np.array([0.0, 0.1, 0.0])
        charge = 0.1e-6 * monitor_voltage
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "processed.csv"
            save_processed_csv(
                path,
                time_ms,
                voltage_input,
                voltage_dbd,
                current_input,
                current_corrected,
                monitor_voltage,
                charge,
            )
            header = path.read_text(encoding="utf-8").splitlines()[0]
            data = np.loadtxt(path, delimiter=",", skiprows=1)
        self.assertIn("current_input_A,current_corrected_A", header)
        np.testing.assert_allclose(data[:, 3], current_input)
        np.testing.assert_allclose(data[:, 4], current_corrected)


class DutyCycleAuditTests(unittest.TestCase):
    @staticmethod
    def synthetic_burst_trace(duty_fraction: float = 0.25):
        sample_rate_hz = 1_000_000.0
        carrier_hz = 20_000.0
        burst_hz = 1_000.0
        time_s = np.arange(0.0, 0.020, 1.0 / sample_rate_hz)
        phase = np.mod(time_s * burst_hz, 1.0)
        expected_active = phase < duty_fraction
        envelope = expected_active.astype(float)
        voltage_v = 5_000.0 * envelope * np.sin(2.0 * np.pi * carrier_hz * time_s)
        current_a = 0.020 * envelope * np.cos(2.0 * np.pi * carrier_hz * time_s)
        return time_s, voltage_v, current_a, expected_active, carrier_hz, burst_hz

    def test_detects_approximately_quarter_duty_fraction(self):
        time_s, voltage_v, current_a, _, _, burst_hz = self.synthetic_burst_trace()
        selection = select_two_duty_cycles(time_s, voltage_v, current_a)
        self.assertNotIn("fallback", selection.method)
        self.assertAlmostEqual(selection.frequency_Hz, burst_hz, delta=0.01 * burst_hz)
        self.assertIsNotNone(selection.active_fraction)
        self.assertAlmostEqual(selection.active_fraction, 0.25, delta=0.04)
        self.assertGreater(selection.envelope_contrast, 0.90)
        self.assertIsNotNone(selection.activity_threshold)

    def test_continuous_wave_fallback_is_fully_active(self):
        time_s, _, _, _, carrier_hz, _ = self.synthetic_burst_trace()
        voltage_v = 5_000.0 * np.sin(2.0 * np.pi * carrier_hz * time_s)
        current_a = 0.020 * np.cos(2.0 * np.pi * carrier_hz * time_s)
        selection = select_two_duty_cycles(time_s, voltage_v, current_a)
        mask = duty_activity_mask(time_s, voltage_v, current_a, selection)
        self.assertEqual(selection.method, "dominant voltage frequency fallback")
        self.assertEqual(selection.active_fraction, 1.0)
        self.assertIsNotNone(mask)
        self.assertTrue(np.all(mask))

    def test_activity_mask_aligns_with_synthetic_burst(self):
        time_s, voltage_v, current_a, expected_active, _, _ = self.synthetic_burst_trace()
        selection = select_two_duty_cycles(time_s, voltage_v, current_a)
        mask = duty_activity_mask(time_s, voltage_v, current_a, selection)
        self.assertIsNotNone(mask)
        self.assertEqual(mask.dtype, np.bool_)
        self.assertEqual(mask.shape, time_s.shape)
        self.assertGreater(float(np.mean(mask == expected_active)), 0.95)
        self.assertAlmostEqual(float(np.mean(mask)), selection.active_fraction, delta=0.01)


if __name__ == "__main__":
    unittest.main()
