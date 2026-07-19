import io
import unittest
import zipfile

import numpy as np

from Lissajous_Scan_Analysis import (
    Condition,
    FileFeatures,
    MemberRecord,
    _fast_numeric_csv,
    choose_current_sign,
    choose_q_sign,
    evenly_spaced,
    inventory_scan_archive,
    scan_fit_summary,
)


def feature(level, cprime=25.0, closs=5.0, phase=-11.3, gain=1.0, phase_error=2.0, x=1.0, y=25.0):
    return FileFeatures(
        member=f"{level}.csv",
        level_label=str(level),
        carrier_Hz=120_000.0,
        record_duration_s=0.01,
        voltage_pp_kV=2.0 * x,
        q_sign=1,
        current_sign=-1,
        monitor_Cprime_pF=cprime,
        monitor_Closs_pF=closs,
        monitor_tan_delta=closs / cprime,
        monitor_phase_deg=phase,
        current_Creactive_pF=cprime,
        current_G_uS=1.0,
        current_monitor_gain_ratio=gain,
        current_monitor_phase_error_deg=phase_error,
        monitor_x_Uqpp_kV=x,
        monitor_y_Qpp_nC=y,
        current_x_Uqpp_kV=x,
        current_y_Qpp_nC=y,
        skipped_rows=0,
        clipping_flag=False,
        quiet_both_edges=False,
    )


class ArchiveInventoryTests(unittest.TestCase):
    def test_nested_argon_four_khz_and_legacy_max_are_found(self):
        names = [
            "root/argon/4 kHz/breakdown/40% breakdown/run/run_01.csv",
            "root/argon/4 kHz/breakdown/105% breakdown/run/run_01.csv",
            "root/argon/Lissajousfigure1 20 kHz/Lissajousfigure1_03.csv",
            "root/argon/argon 20kHz.csv",
            "root/argon/Lissajousfigure4kHz-0004_03_analysis/processed.csv",
        ]
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w") as archive:
            for name in names:
                archive.writestr(name, "")
        buffer.seek(0)
        with zipfile.ZipFile(buffer) as archive:
            records = inventory_scan_archive(archive)
        self.assertEqual(len(records), 3)
        self.assertIn((4, 40, False), [(r.condition.burst_kHz, r.level_percent, r.is_maximum) for r in records])
        self.assertIn((20, None, True), [(r.condition.burst_kHz, r.level_percent, r.is_maximum) for r in records])

    def test_even_selection_spans_the_acquisition(self):
        condition = Condition("argon_only", 4)
        records = [MemberRecord(f"x_{i:02d}.csv", condition, 40, False, i) for i in range(1, 65)]
        selected = evenly_spaced(records, 5)
        self.assertEqual([row.capture_index for row in selected], [1, 17, 33, 48, 64])


class ParsingAndSignTests(unittest.TestCase):
    def test_bad_numeric_row_is_skipped(self):
        rows = [f"{index / 10},1,2,3" for index in range(50)]
        rows[20] = "2.0,2,3,bad"
        payload = ("Time,A,C,D\n(ms),(V),(A),(V)\n\n" + "\n".join(rows) + "\n").encode()
        data, skipped = _fast_numeric_csv(payload)
        self.assertEqual(data.shape, (49, 4))
        self.assertGreaterEqual(skipped, 1)

    def test_phasor_sign_conventions(self):
        self.assertEqual(choose_q_sign(complex(-20e-12, 5e-12)), -1)
        self.assertEqual(choose_current_sign(complex(0.0, -1e-6)), -1)


class ModelGateTests(unittest.TestCase):
    def test_passive_stable_monitor_route_passes(self):
        rows = [
            feature(40, x=1.0, y=25.0),
            feature(60, x=1.5, y=37.5),
            feature(75, x=2.0, y=50.0),
            feature(105, x=2.5, y=85.0),
            feature(115, x=3.0, y=110.0),
            feature("MAX", x=5.0, y=210.0),
        ]
        summary = scan_fit_summary(Condition("BMIM_nitrate", 20), rows)
        self.assertTrue(summary["monitor_qv_passive_valid"])
        self.assertTrue(summary["monitor_kcl_with_pearson_valid"])
        self.assertGreater(summary["routes"]["monitor"]["Cd_effective_high_field_pF"], 25.0)

    def test_negative_loss_orientation_is_rejected(self):
        rows = [
            feature(40, closs=-4.0, phase=11.0, x=1.0, y=20.0),
            feature(60, closs=-4.0, phase=11.0, x=1.5, y=30.0),
            feature(75, closs=-4.0, phase=11.0, x=2.0, y=40.0),
            feature(105, closs=-4.0, phase=11.0, x=2.5, y=70.0),
            feature(115, closs=-4.0, phase=11.0, x=3.0, y=90.0),
            feature("MAX", closs=-4.0, phase=11.0, x=5.0, y=170.0),
        ]
        summary = scan_fit_summary(Condition("pure_water", 4), rows)
        self.assertFalse(summary["monitor_qv_passive_valid"])


if __name__ == "__main__":
    unittest.main()
