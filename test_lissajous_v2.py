import csv
import json
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np

from lissajous.io import Waveform, parse_path_metadata, read_waveform
from lissajous.quantify import (
    PER_CAPTURE_STABLE_COLUMNS,
    _power_from_energies,
    band_tag,
    parse_arguments,
    run,
)
from lissajous.report import (
    FORBIDDEN_OUTPUT_TOKENS,
    assert_no_retired_numbers,
    condition_summary_rows,
)
from lissajous.signal import (
    _cyclic_energy_uJ,
    complex_capacitance,
    cycle_loop_metrics,
    estimate_frequencies,
    time_domain_slope_pF,
)


STABLE_PREFIX = [
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


def minimal_config(
    data_root: str = "unused-data",
    output: str = "unused-output",
) -> dict:
    return {
        "analysis": {
            "version": "test-v2",
            "r3_relative_tolerance": 0.10,
            "minimum_codes": 30,
            "minimum_carrier_hz": 1_000.0,
            "maximum_carrier_hz": 300_000.0,
            "frequency_crosscheck_tolerance": 0.03,
            "lf_geometric_max_hz": 20_000.0,
            "carrier_transfer_min_hz": 60_000.0,
            "transition_band_policy": "conservatively_tag_as_carrier-transfer",
            "detrend_fraction_threshold": 100.0,
            "strict_orientation": False,
            "prefer_archives": False,
            "multiline_max_points_per_capture": 8,
            "multiline_relative_voltage_threshold": 0.02,
        },
        "paths": {
            "data_root": data_root,
            "default_output": output,
        },
        "calibration": {
            "channel_a_divider": {
                "value": 2.0,
                "relative_uncertainty": 0.03,
                "provenance": "synthetic divider",
            },
            "measuring_capacitor_F": {
                "value": 1.0e-7,
                "relative_uncertainty": 0.05,
                "provenance": "synthetic measuring capacitor",
            },
            "same_band_charge_factor": {
                "value": 1.15,
                "relative_uncertainty": 0.0,
                "provenance": "synthetic same-band factor",
            },
            "channel_roles": {
                "time": ["Time"],
                "applied_voltage": ["Channel A"],
                "legacy_current": ["Channel C"],
                "charge_monitor": ["Channel D"],
            },
        },
        "chain_model": {
            "C_true_F": 2.7e-12,
            "C_true_range_F": [2.0e-12, 3.4e-12],
            "L_H": 0.38,
            "L_range_H": [0.30, 0.50],
            "R_ohm": 10_000.0,
            "R_range_ohm": [7_000.0, 15_000.0],
            "f_res_Hz": 161_500.0,
            "f_res_range_Hz": [159_000.0, 164_000.0],
            "provenance": "synthetic chain model",
        },
        "legacy_dispersion_points": [],
        "synthesis_runs": {},
        "open_items": ["Synthetic test configuration only*"],
    }


def frequency_waveform(time_s: np.ndarray, voltage_V: np.ndarray) -> Waveform:
    zeros = np.zeros_like(time_s)
    return Waveform(
        time_s=time_s,
        applied_voltage_V=voltage_V,
        current_A=None,
        monitor_voltage_V=zeros,
        charge_nC=zeros,
        headers=["Time", "Channel A", "Channel D"],
        units=["s", "V", "V"],
        role_indices={"time": 0, "applied_voltage": 1, "charge_monitor": 2},
        clip_counts={},
        code_counts={},
        lsb={},
        n_samples_raw=time_s.size,
        skipped_rows=0,
        dc_offset_V=0.0,
        drift_V_per_s=0.0,
        detrended=False,
    )


def write_synthetic_capture(path: Path) -> None:
    sample_rate_Hz = 512_000.0
    frequency_Hz = 4_000.0
    time_s = np.arange(5_120, dtype=float) / sample_rate_Hz
    phase = 2.0 * np.pi * frequency_Hz * time_s

    # The configured divider is two, so this becomes a 1 kV applied waveform.
    channel_a_V = 500.0 * np.sin(phase)
    charge_nC = 45.0 * np.sin(phase) - 8.0 * np.cos(phase)
    channel_d_V = 0.2 + charge_nC / 100.0
    channel_c_A = 2.0 * np.pi * frequency_Hz * (
        45.0e-9 * np.cos(phase) + 8.0e-9 * np.sin(phase)
    )

    lines = ["Time,Channel A,Channel C,Channel D", "(s),(V),(A),(V)"]
    lines.extend(
        ",".join(f"{value:.12g}" for value in row)
        for row in zip(time_s, channel_a_V, channel_c_A, channel_d_V)
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


class WaveformParsingTests(unittest.TestCase):
    def test_header_role_permutation_units_and_infinity_interpolation(self):
        rows = []
        for index in range(64):
            channel_d = f"{100.0 + 2.0 * index:.8g}"
            channel_c = f"{2.0 + 0.1 * index:.8g}"
            channel_a = f"{0.1 + 0.01 * index:.8g}"
            if index == 10:
                channel_a = "\N{INFINITY}"
            if index == 20:
                channel_d = "-Infinity"
            if index == 30:
                channel_c = "inf"
            rows.append(
                f"{channel_d},{channel_c},{2.5 * index:.8g},{channel_a}"
            )
        raw = (
            "Channel D,Channel C,Time,Channel A\n"
            "(mV),(mA),(\N{MICRO SIGN}s),(kV)\n"
            + "\n".join(rows)
            + "\n"
        ).encode("utf-8")

        waveform = read_waveform(raw, minimal_config())

        self.assertEqual(
            waveform.role_indices,
            {
                "time": 2,
                "applied_voltage": 3,
                "legacy_current": 1,
                "charge_monitor": 0,
            },
        )
        self.assertEqual(waveform.clip_counts["applied_voltage"], 1)
        self.assertEqual(waveform.clip_counts["legacy_current"], 1)
        self.assertEqual(waveform.clip_counts["charge_monitor"], 1)
        self.assertEqual(waveform.clip_counts["time"], 0)
        self.assertEqual(waveform.n_samples_raw, 64)
        self.assertEqual(waveform.skipped_rows, 0)
        self.assertFalse(waveform.detrended)

        self.assertAlmostEqual(waveform.time_s[-1], 157.5e-6, places=13)
        self.assertAlmostEqual(waveform.applied_voltage_V[10], 400.0, places=10)
        self.assertAlmostEqual(waveform.dc_offset_V, 0.163, places=12)
        self.assertAlmostEqual(waveform.monitor_voltage_V[20], -0.023, places=12)
        self.assertAlmostEqual(waveform.current_A[30], -0.00015, places=12)
        self.assertTrue(np.all(np.isfinite(waveform.applied_voltage_V)))
        self.assertTrue(np.all(np.isfinite(waveform.monitor_voltage_V)))
        self.assertTrue(np.all(np.isfinite(waveform.current_A)))


class FrequencyAndBandTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = minimal_config()
        cls.sample_rate_Hz = 1_024_000.0
        cls.time_s = (
            np.arange(int(0.020 * cls.sample_rate_Hz), dtype=float)
            / cls.sample_rate_Hz
        )

    def test_measured_continuous_four_khz_is_not_a_burst(self):
        voltage = 1_000.0 * np.sin(2.0 * np.pi * 4_000.0 * self.time_s)
        estimate = estimate_frequencies(
            frequency_waveform(self.time_s, voltage), self.config
        )

        self.assertAlmostEqual(estimate.carrier_Hz, 4_000.0, delta=5.0)
        self.assertIsNone(estimate.burst_Hz)
        self.assertAlmostEqual(estimate.duty_on_fraction, 1.0, places=12)
        self.assertEqual(band_tag(estimate.carrier_Hz, self.config), "LF-geometric")

    def test_measured_128_khz_carrier_is_separate_from_four_khz_burst(self):
        gate = ((self.time_s * 4_000.0) % 1.0) < 0.32
        voltage = (
            1_000.0
            * np.sin(2.0 * np.pi * 128_000.0 * self.time_s)
            * gate
        )
        estimate = estimate_frequencies(
            frequency_waveform(self.time_s, voltage), self.config
        )

        self.assertAlmostEqual(estimate.carrier_Hz, 128_000.0, delta=100.0)
        self.assertIsNotNone(estimate.burst_Hz)
        self.assertAlmostEqual(estimate.burst_Hz, 4_000.0, delta=100.0)
        self.assertGreater(estimate.duty_on_fraction, 0.25)
        self.assertLess(estimate.duty_on_fraction, 0.42)
        self.assertEqual(
            band_tag(estimate.carrier_Hz, self.config), "carrier-transfer"
        )

    def test_band_boundary_uses_measured_frequency(self):
        self.assertEqual(band_tag(20_000.0, self.config), "LF-geometric")
        self.assertEqual(
            band_tag(np.nextafter(20_000.0, math.inf), self.config),
            "carrier-transfer",
        )
        self.assertEqual(band_tag(128_000.0, self.config), "carrier-transfer")


class AnalyticEstimatorTests(unittest.TestCase):
    def test_complex_capacitance_and_time_slope(self):
        frequency_Hz = 5_000.0
        time_s = np.arange(12_000, dtype=float) / 600_000.0
        phase = 2.0 * np.pi * frequency_Hz * time_s + 0.17
        expected = complex(73.0, -19.0)
        voltage_kV = 2.4 * np.cos(phase)
        charge_nC = 2.4 * (
            expected.real * np.cos(phase) - expected.imag * np.sin(phase)
        )

        observed = complex_capacitance(
            time_s, voltage_kV, charge_nC, frequency_Hz
        )

        self.assertAlmostEqual(observed.real, expected.real, delta=2.0e-5)
        self.assertAlmostEqual(observed.imag, expected.imag, delta=2.0e-5)
        self.assertAlmostEqual(
            time_domain_slope_pF(voltage_kV, charge_nC),
            expected.real,
            places=10,
        )

    def test_signed_shoelace_and_cycle_energies_preserve_orientation(self):
        carrier_Hz = 4_000.0
        samples_per_cycle = 128
        phase = np.linspace(0.0, 2.0 * np.pi, samples_per_cycle, endpoint=False)
        voltage_one = 1.5 * np.cos(phase)
        positive_charge_one = 45.0 * np.cos(phase) + 8.0 * np.sin(phase)
        negative_charge_one = 45.0 * np.cos(phase) - 8.0 * np.sin(phase)

        positive_area = _cyclic_energy_uJ(voltage_one, positive_charge_one)
        negative_area = _cyclic_energy_uJ(voltage_one, negative_charge_one)
        self.assertAlmostEqual(positive_area, math.pi * 1.5 * 8.0, delta=0.03)
        self.assertAlmostEqual(positive_area, -negative_area, places=10)

        time_s = (
            np.arange(10 * samples_per_cycle, dtype=float)
            / (carrier_Hz * samples_per_cycle)
        )
        full_phase = 2.0 * np.pi * carrier_Hz * time_s
        voltage = 1.5 * np.cos(full_phase)
        positive_charge = 45.0 * np.cos(full_phase) + 8.0 * np.sin(full_phase)
        negative_charge = 45.0 * np.cos(full_phase) - 8.0 * np.sin(full_phase)
        positive = cycle_loop_metrics(
            time_s, voltage, positive_charge, carrier_Hz
        )
        negative = cycle_loop_metrics(
            time_s, voltage, negative_charge, carrier_Hz
        )

        self.assertGreater(positive.energy_signed_uJ.size, 5)
        self.assertTrue(np.all(positive.energy_signed_uJ > 0.0))
        self.assertTrue(np.all(negative.energy_signed_uJ < 0.0))
        np.testing.assert_allclose(
            positive.energy_signed_uJ,
            -negative.energy_signed_uJ,
            rtol=0.0,
            atol=1.0e-10,
        )

    def test_burst_power_never_falls_back_to_cycle_energy(self):
        power, method = _power_from_energies(
            cycle_energy_uJ=20.0,
            burst_energy_uJ=None,
            carrier_Hz=100_000.0,
            burst_Hz=4_000.0,
        )
        self.assertIsNone(power)
        self.assertEqual(method, "burst_power_withheld_no_valid_burst_energy")

        power, method = _power_from_energies(
            cycle_energy_uJ=20.0,
            burst_energy_uJ=500.0,
            carrier_Hz=100_000.0,
            burst_Hz=4_000.0,
        )
        self.assertAlmostEqual(power, 2.0)
        self.assertEqual(
            method,
            "burst_shoelace_times_measured_burst_Hz",
        )

    def test_continuous_power_uses_cycle_energy(self):
        power, method = _power_from_energies(
            cycle_energy_uJ=20.0,
            burst_energy_uJ=None,
            carrier_Hz=100_000.0,
            burst_Hz=None,
        )
        self.assertAlmostEqual(power, 2.0)
        self.assertEqual(
            method,
            "cycle_shoelace_times_measured_carrier_Hz_continuous",
        )


class MetadataAndAggregationTests(unittest.TestCase):
    def setUp(self):
        self.config = minimal_config()

    def test_path_metadata_and_7_20_contamination(self):
        ladder = parse_path_metadata(
            "Lissajouswaveformsdifferentmediums/argon/4 kHz/breakdown/"
            "60% breakdown/0_6breakdown-0015/0_6breakdown-0015_32.csv",
            self.config,
        )
        self.assertEqual(ladder["dataset_type"], "voltage_ladder")
        self.assertEqual(ladder["source_type"], "ladder_capture")
        self.assertEqual(ladder["medium"], "argon_only")
        self.assertEqual(ladder["nominal_frequency_kHz"], 4.0)
        self.assertEqual(ladder["level_pct"], 60.0)
        self.assertEqual(ladder["save_idx"], 15)
        self.assertEqual(ladder["seg_idx"], 32)
        self.assertEqual(ladder["cond"], "argon_only_4_khz_60")

        contaminated = parse_path_metadata(
            "7_20/128 kHz 2 kV/fixture_7.csv", self.config
        )
        self.assertEqual(contaminated["dataset_type"], "dispersion_7_20")
        self.assertEqual(contaminated["source_type"], "multiline_7_20")
        self.assertEqual(contaminated["medium"], "dry_fixture")
        self.assertEqual(contaminated["commanded_kV"], 2.0)
        self.assertTrue(contaminated["contaminated"])
        self.assertIn(
            "surface_discharge_contaminated_not_displacement_only",
            contaminated["exclusion_flags"],
        )

        passive = parse_path_metadata(
            "7_20/128 kHz 1.5 kV/fixture_8.csv", self.config
        )
        self.assertEqual(passive["commanded_kV"], 1.5)
        self.assertFalse(passive["contaminated"])
        self.assertEqual(passive["exclusion_flags"], [])

    def test_cross_band_condition_aggregation_is_blocked(self):
        captures = [
            {"cond": "same-condition", "band_tag": "LF-geometric"},
            {"cond": "same-condition", "band_tag": "carrier-transfer"},
        ]
        with self.assertRaisesRegex(RuntimeError, "cross-band aggregation blocked"):
            condition_summary_rows(captures)


class OutputContractTests(unittest.TestCase):
    def test_per_capture_stable_columns_are_the_exact_prefix(self):
        self.assertEqual(PER_CAPTURE_STABLE_COLUMNS, STABLE_PREFIX)

    def test_tiny_data_root_run_writes_contract_tables_without_retired_tokens(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data_root = root / "data"
            output = root / "output"
            capture = (
                data_root
                / "Lissajouswaveformsdifferentmediums"
                / "argon"
                / "4 kHz"
                / "breakdown"
                / "105% breakdown"
                / "synthetic-0001"
                / "synthetic-0001_1.csv"
            )
            write_synthetic_capture(capture)

            config_path = root / "config.yaml"
            config_path.write_text(
                json.dumps(
                    minimal_config(str(data_root), str(output)),
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
            args = parse_arguments(
                [
                    "--config",
                    str(config_path),
                    "--data-root",
                    str(data_root),
                    "--out",
                    str(output),
                    "--no-archives",
                    "--no-multiline",
                    "--quiet",
                ]
            )
            result = run(args)

            self.assertEqual(result["logical_csvs"], 1)
            self.assertEqual(result["parsed"], 1)
            self.assertEqual(result["failed"], 0)
            self.assertEqual(result["LF_geometric_captures"], 1)
            self.assertEqual(result["carrier_transfer_captures"], 0)

            contract_files = {
                "manifest.csv",
                "per_capture_metrics.csv",
                "condition_summary.csv",
                "dispersion_master.csv",
                "discharge_metrics.csv",
                "synthesis_charge.csv",
                "frequency_consistency.csv",
                "factor_frequency.csv",
                "discharge_onset.csv",
                "presentation_numbers.json",
                "config.yaml",
                "RESULTS.md",
            }
            self.assertTrue(
                contract_files.issubset(
                    {path.name for path in output.iterdir() if path.is_file()}
                )
            )

            with (output / "manifest.csv").open(
                newline="", encoding="utf-8"
            ) as handle:
                manifest_rows = list(csv.DictReader(handle))
            self.assertEqual(len(manifest_rows), 1)
            self.assertEqual(manifest_rows[0]["parse_status"], "ok")
            self.assertEqual(manifest_rows[0]["band_tag"], "LF-geometric")
            self.assertAlmostEqual(
                float(manifest_rows[0]["f0_Hz"]), 4_000.0, delta=5.0
            )

            with (output / "per_capture_metrics.csv").open(
                newline="", encoding="utf-8"
            ) as handle:
                reader = csv.DictReader(handle)
                metric_rows = list(reader)
                fieldnames = reader.fieldnames
            self.assertEqual(fieldnames[: len(STABLE_PREFIX)], STABLE_PREFIX)
            self.assertEqual(len(metric_rows), 1)
            self.assertEqual(metric_rows[0]["band_tag"], "LF-geometric")

            assert_no_retired_numbers(output)
            text_outputs = "\n".join(
                path.read_text(encoding="utf-8", errors="replace")
                for path in output.iterdir()
                if path.suffix.casefold() in {".csv", ".json", ".md", ".yaml"}
            )
            for token in FORBIDDEN_OUTPUT_TOKENS:
                self.assertNotIn(token, text_outputs)


if __name__ == "__main__":
    unittest.main()
