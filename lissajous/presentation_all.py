"""Regenerate the complete presentation figure set from saved analysis outputs.

This entry point is intentionally presentation-only: it composes the core and
supplementary renderers and never invokes waveform quantification.

Run with::

    python -m lissajous.presentation_all --out outputs/lissajous_v2
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from .presentation_figures import build_presentation_figures
from .presentation_supplement import generate_all as build_presentation_supplement


def build_all(
    output_root: Path,
    legacy_dir: Path,
) -> list[Path]:
    """Render every presentation figure and return all generated paths."""

    root = Path(output_root)
    legacy = Path(legacy_dir)
    outputs = list(build_presentation_figures(root))
    outputs.extend(build_presentation_supplement(root, legacy))
    return outputs


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Regenerate all presentation figures from existing Lissajous-v2 "
            "summary products; waveform quantification is not rerun."
        )
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("outputs/lissajous_v2"),
        help="Existing Lissajous-v2 output directory and figure destination.",
    )
    parser.add_argument(
        "--legacy-dir",
        type=Path,
        default=Path("dbd_surface_charge_report"),
        help="Directory containing the legacy binned tables for Figures 13–15.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    outputs = build_all(args.out, args.legacy_dir)
    print(
        f"Generated {len(outputs)} presentation files under "
        f"{args.out.resolve()}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
