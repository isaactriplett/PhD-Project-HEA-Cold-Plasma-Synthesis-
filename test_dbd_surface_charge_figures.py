import unittest

from dbd_surface_charge_figures import AVOGADRO_PER_MOL, build_dose_clock_rows


class DoseClockTests(unittest.TestCase):
    def test_reactor_volume_time_uses_inventory_and_inverts_rate_interval(self):
        rate_per_s = 1.00e18
        rate_low_per_s = 0.80e18
        rate_high_per_s = 1.25e18
        volume_ml = 2.5
        concentration_mm = 5.0
        rows = build_dose_clock_rows(
            [
                {
                    "condition": "5mM_Mn_nitrate_in_water_20kHz",
                    "material": "5mM_Mn_nitrate_in_water",
                    "burst_kHz": 20,
                    "target_charge_polarity": "negative",
                    "metric": "whole_record_average_flow_per_s",
                    "estimate": rate_per_s,
                    "analysis_ci_low": rate_low_per_s,
                    "analysis_ci_high": rate_high_per_s,
                }
            ],
            volume_ml=volume_ml,
            concentration_mM=concentration_mm,
            equivalents_per_ion=1.0,
        )

        reactor_rows = [row for row in rows if row["is_reactor_volume"]]
        self.assertEqual(len(reactor_rows), 1)
        result = reactor_rows[0]
        inventory_equivalents = (
            concentration_mm * volume_ml * AVOGADRO_PER_MOL * 1.0e-6
        )
        expected_minutes = inventory_equivalents / (60.0 * rate_per_s)
        expected_low = inventory_equivalents / (60.0 * rate_high_per_s)
        expected_high = inventory_equivalents / (60.0 * rate_low_per_s)

        self.assertAlmostEqual(
            result["minutes_per_ion_equivalent"], expected_minutes
        )
        self.assertAlmostEqual(
            result["minutes_per_ion_equivalent_ci_low"], expected_low
        )
        self.assertAlmostEqual(
            result["minutes_per_ion_equivalent_ci_high"], expected_high
        )
        self.assertLess(
            result["minutes_per_ion_equivalent_ci_low"],
            result["minutes_per_ion_equivalent"],
        )
        self.assertGreater(
            result["minutes_per_ion_equivalent_ci_high"],
            result["minutes_per_ion_equivalent"],
        )

    def test_two_electron_equivalents_double_dose_time(self):
        source = {
            "condition": "5mM_Mn_nitrate_in_water_4kHz",
            "material": "5mM_Mn_nitrate_in_water",
            "burst_kHz": 4,
            "target_charge_polarity": "negative",
            "metric": "whole_record_average_flow_per_s",
            "estimate": 2.0e18,
        }
        one_equivalent = build_dose_clock_rows(
            [source], volume_ml=2.5, equivalents_per_ion=1.0
        )
        two_equivalents = build_dose_clock_rows(
            [source], volume_ml=2.5, equivalents_per_ion=2.0
        )
        one = next(row for row in one_equivalent if row["is_reactor_volume"])
        two = next(row for row in two_equivalents if row["is_reactor_volume"])

        self.assertAlmostEqual(
            two["minutes_per_ion_equivalent"],
            2.0 * one["minutes_per_ion_equivalent"],
        )


if __name__ == "__main__":
    unittest.main()
