import csv
import unittest
from pathlib import Path
from unittest.mock import patch

from lissajous.presentation_all import _parser, build_all
from lissajous.presentation_figures import FIGURE_STEMS as CORE_STEMS
from lissajous.presentation_supplement import FIGURE_STEMS as SUPPLEMENT_STEMS


ROOT = Path(__file__).resolve().parent
OUTPUT_ROOT = ROOT / "outputs" / "lissajous_v2"


class PresentationOrchestrationTests(unittest.TestCase):
    def test_combined_builder_only_composes_the_two_saved_output_renderers(self):
        output_root = Path("saved-analysis")
        legacy_dir = Path("legacy-tables")
        core_outputs = [output_root / "presentation_figures" / "fig1.png"]
        supplement_outputs = [
            output_root / "presentation_figures" / "fig09.png"
        ]

        with (
            patch(
                "lissajous.presentation_all.build_presentation_figures",
                return_value=core_outputs,
            ) as core,
            patch(
                "lissajous.presentation_all.build_presentation_supplement",
                return_value=supplement_outputs,
            ) as supplement,
        ):
            observed = build_all(output_root, legacy_dir)

        self.assertEqual(observed, core_outputs + supplement_outputs)
        core.assert_called_once_with(output_root)
        supplement.assert_called_once_with(output_root, legacy_dir)

    def test_combined_cli_defaults_to_shipped_summary_locations(self):
        args = _parser().parse_args([])
        self.assertEqual(args.out, Path("outputs/lissajous_v2"))
        self.assertEqual(args.legacy_dir, Path("dbd_surface_charge_report"))


class PresentationOutputContractTests(unittest.TestCase):
    def test_figure_numbering_and_legacy_stems_are_explicit(self):
        expected_core_ids = (
            "fig1",
            "fig2a",
            "fig2b",
            "fig3",
            "fig4",
            "fig5",
            "fig6",
            "fig7",
            "fig8",
        )
        self.assertEqual(
            tuple(stem.split("_", 1)[0] for stem in CORE_STEMS),
            expected_core_ids,
        )
        self.assertEqual(set(SUPPLEMENT_STEMS), set(range(9, 16)))
        for number in (13, 14, 15):
            self.assertIn("legacy", SUPPLEMENT_STEMS[number])

    def test_shipped_core_manifest_is_complete_and_nonlegacy(self):
        manifest_path = OUTPUT_ROOT / "presentation_core_manifest.csv"
        with manifest_path.open(encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))

        expected_ids = {
            "fig1",
            "fig2a",
            "fig2b",
            "fig3",
            "fig4",
            "fig5",
            "fig6",
            "fig7",
            "fig8",
        }
        self.assertEqual({row["figure_id"] for row in rows}, expected_ids)
        for row in rows:
            self.assertEqual(row["legacy_model_dependent"], "false")
            for column in ("png", "pdf", "caption"):
                self.assertTrue(
                    (OUTPUT_ROOT / row[column]).is_file(),
                    f"Missing {column} for {row['figure_id']}",
                )

    def test_shipped_supplement_manifest_is_complete_and_truthfully_labeled(self):
        manifest_path = OUTPUT_ROOT / "presentation_supplement_manifest.csv"
        with manifest_path.open(encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))

        expected_ids = {
            "fig09",
            "fig09b",
            "fig10",
            "fig11a",
            "fig11b",
            "fig12",
            "fig13a",
            "fig13b",
            "fig13c",
            "fig13d",
            "fig14a",
            "fig14b",
            "fig14c",
            "fig14d",
            "fig15",
        }
        self.assertEqual({row["figure_id"] for row in rows}, expected_ids)

        for row in rows:
            for column in ("png", "pdf", "caption"):
                self.assertTrue(
                    (OUTPUT_ROOT / row[column]).is_file(),
                    f"Missing {column} for {row['figure_id']}",
                )
            should_be_legacy = row["figure_id"].startswith(
                ("fig13", "fig14", "fig15")
            )
            self.assertEqual(
                row["legacy_model_dependent"],
                str(should_be_legacy).lower(),
            )

    def test_index_lists_every_figure_and_preserves_key_limitations(self):
        index_path = OUTPUT_ROOT / "PRESENTATION_FIGURE_INDEX.md"
        text = index_path.read_text(encoding="utf-8")

        for stem in CORE_STEMS:
            self.assertIn(f"presentation_figures/{stem}.png", text)

        with (
            OUTPUT_ROOT / "presentation_supplement_manifest.csv"
        ).open(encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                self.assertIn(row["png"], text)

        limitation_phrases = (
            "geometric",
            "withheld",
            "negative polarity combines electrons and negative ions",
            "not electron-specific",
            "not establish retained dielectric charge",
            "legacy/model-dependent appendix",
        )
        for phrase in limitation_phrases:
            self.assertIn(phrase, text.lower())

    def test_readme_separates_preserved_audit_and_clear_presentation_sets(self):
        text = (ROOT / "LISSAJOUS_V2_README.md").read_text(encoding="utf-8")
        self.assertIn("original eight-figure audit set", text)
        self.assertIn("python -m lissajous.figures", text)
        self.assertIn("python -m lissajous.presentation_all", text)
        self.assertIn("PRESENTATION_FIGURE_INDEX.md", text)
        self.assertIn("does not rerun waveform quantification", text)

    def test_v21_presentation_contract_is_encoded_on_canvas(self):
        core_source = (
            ROOT / "lissajous" / "presentation_figures.py"
        ).read_text(encoding="utf-8")
        supplement_source = (
            ROOT / "lissajous" / "presentation_supplement.py"
        ).read_text(encoding="utf-8")
        source = core_source + supplement_source

        required_phrases = (
            "series-RLC chain fit (v1.2)†",
            "poor real-admittance residuals",
            "Quiet-edge gate",
            "retained_charge_nC N = 0",
            "electrical charge, both polarities summed",
            "not liquid heating",
            "within-burst charge return not excluded",
            "half-cycle with maximum |ΔQ| per capture",
            "negative − positive",
            "Record set:",
            "LEGACY_MODEL_CANVAS",
        )
        for phrase in required_phrases:
            self.assertIn(phrase, source)

        for retired_wording in (
            "illustrative model",
            "nominal 4 kHz",
            "nominal burst-frequency",
        ):
            self.assertNotIn(retired_wording, source)

    def test_legacy_captions_use_vendored_forward_slash_provenance(self):
        caption_dir = OUTPUT_ROOT / "presentation_captions"
        for figure_id in (
            "fig13a",
            "fig13b",
            "fig13c",
            "fig13d",
            "fig14a",
            "fig14b",
            "fig14c",
            "fig14d",
            "fig15",
        ):
            text = (caption_dir / f"{figure_id}_caption.md").read_text(
                encoding="utf-8"
            )
            self.assertIn("`legacy/", text)
            self.assertNotIn("dbd_surface_charge_report\\", text)


if __name__ == "__main__":
    unittest.main()
