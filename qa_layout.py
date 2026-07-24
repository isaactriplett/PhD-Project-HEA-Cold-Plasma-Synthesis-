"""Detect overlapping and out-of-frame text in the presentation figure set.

Hooks the single save choke point in each figure module, renders every
figure, and measures real text bounding boxes against each other and
against the canvas. Reports offenders; changes nothing.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

from matplotlib.text import Text  # noqa: E402

from lissajous import presentation_figures as pf  # noqa: E402
from lissajous import presentation_supplement as ps  # noqa: E402

REPORT: list[tuple[str, str, str]] = []
MIN_OVERLAP_PX = 12.0


def _offview_ticklabels(fig):
    """Tick labels matplotlib creates for ticks outside the view limits.

    These artists exist and report positions far off-canvas, but are never
    rendered. Counting them as layout faults would bury the real ones.
    """

    skip = set()
    for ax in fig.get_axes():
        for axis, limits in (
            (ax.xaxis, ax.get_xlim()),
            (ax.yaxis, ax.get_ylim()),
        ):
            low, high = min(limits), max(limits)
            for tick in list(axis.get_major_ticks()) + list(axis.get_minor_ticks()):
                location = tick.get_loc()
                if location is None:
                    continue
                if location < low - 1e-9 or location > high + 1e-9:
                    for label in (tick.label1, tick.label2):
                        if label is not None:
                            skip.add(id(label))
    return skip


def _boxes(fig):
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    skip = _offview_ticklabels(fig)
    out = []
    for artist in fig.findobj(Text):
        if id(artist) in skip:
            continue
        label = (artist.get_text() or "").strip()
        if not label:
            continue
        if not artist.get_visible():
            continue
        try:
            # Annotation.get_window_extent() unions the text with its leader
            # arrow, which reports huge phantom boxes. Call the Text base
            # implementation to measure only the glyphs.
            bbox = Text.get_window_extent(artist, renderer=renderer)
        except Exception:
            continue
        if bbox.width <= 0 or bbox.height <= 0:
            continue
        out.append((label, bbox, artist))
    return out


def _inspect(fig, stem: str) -> None:
    width, height = fig.canvas.get_width_height()
    entries = _boxes(fig)

    for label, bbox, _artist in entries:
        over = []
        if bbox.x0 < -1.0:
            over.append(f"left by {abs(bbox.x0):.0f}px")
        if bbox.y0 < -1.0:
            over.append(f"bottom by {abs(bbox.y0):.0f}px")
        if bbox.x1 > width + 1.0:
            over.append(f"right by {bbox.x1 - width:.0f}px")
        if bbox.y1 > height + 1.0:
            over.append(f"top by {bbox.y1 - height:.0f}px")
        if over:
            REPORT.append((stem, "OUT-OF-FRAME", f"{label[:48]!r} runs off {', '.join(over)}"))

    for i in range(len(entries)):
        label_a, box_a, artist_a = entries[i]
        for j in range(i + 1, len(entries)):
            label_b, box_b, artist_b = entries[j]
            if artist_a.axes is not None and artist_a.axes is not artist_b.axes:
                if artist_b.axes is not None:
                    continue
            x_overlap = min(box_a.x1, box_b.x1) - max(box_a.x0, box_b.x0)
            y_overlap = min(box_a.y1, box_b.y1) - max(box_a.y0, box_b.y0)
            if x_overlap <= MIN_OVERLAP_PX or y_overlap <= 2.0:
                continue
            area = x_overlap * y_overlap
            smaller = min(box_a.width * box_a.height, box_b.width * box_b.height)
            if smaller <= 0:
                continue
            if area / smaller < 0.14:
                continue
            REPORT.append(
                (
                    stem,
                    "OVERLAP",
                    f"{label_a[:34]!r} x {label_b[:34]!r} "
                    f"({area / smaller:.0%} of smaller, {x_overlap:.0f}x{y_overlap:.0f}px)",
                )
            )


def _wrap(module):
    original = module._save

    def patched(fig, *args, **kwargs):
        stem = kwargs.get("stem")
        if not stem:
            for candidate in args:
                if isinstance(candidate, str):
                    stem = candidate
        if not stem:
            for candidate in args:
                if isinstance(candidate, int):
                    stem = getattr(module, "FIGURE_STEMS", {}).get(candidate) or f"figure_{candidate}"
        if not stem:
            stem = "unknown"
        try:
            _inspect(fig, stem)
        except Exception as error:  # pragma: no cover - diagnostics only
            REPORT.append((stem, "INSPECT-FAILED", repr(error)))
        return original(fig, *args, **kwargs)

    module._save = patched
    return original


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="outputs/lissajous_v2")
    parser.add_argument("--legacy-dir", default="dbd_surface_charge_report")
    args = parser.parse_args()

    _wrap(pf)
    _wrap(ps)

    root = Path(args.out)
    pf.build_presentation_figures(root)
    ps.generate_all(root, Path(args.legacy_dir))

    by_figure: dict[str, list[tuple[str, str]]] = {}
    for stem, kind, detail in REPORT:
        by_figure.setdefault(stem, []).append((kind, detail))

    if not by_figure:
        print("No overlapping or out-of-frame text detected.")
        return 0

    print(f"=== {len(REPORT)} issue(s) across {len(by_figure)} figure(s) ===\n")
    for stem in sorted(by_figure, key=lambda s: -len(by_figure[s])):
        issues = by_figure[stem]
        print(f"{stem}  ({len(issues)})")
        for kind, detail in issues[:8]:
            print(f"    {kind:<14} {detail}")
        if len(issues) > 8:
            print(f"    ... and {len(issues) - 8} more")
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
