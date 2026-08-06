"""EDS linescan plotter for Velox CSV exports.

Renders Velox line-profile exports as publication figures. Element colours are
taken from the Velox EDS maps themselves (sampled from the ``-int.png`` files)
so the plots match the corresponding STEM figures:

    Cu  #FF0000 red      Ni  #00FF00 green      Ag  #00FFFF cyan

Colours are slightly darkened here for legibility as thin lines on white.

Velox CSV format
----------------
row 0   channel names, first column blank      e.g. "","HAADF","Cu","Ag"
row 1   quantity labels                        e.g. "Position","Intensity",...
row 2   units                                  e.g. "m","Counts","Counts"
row 3+  data; column 0 is position in METRES, remaining columns are counts

Each EDS channel is normalised to its own maximum so weak channels stay
visible; HAADF is drawn as a filled grey backdrop for spatial context.

Smoothing is a presentation choice: at typical Velox point spacing (<1 nm)
the raw trace is dominated by counting noise well below the EDS spatial
resolution. Quote the window in the figure caption.

Usage
-----
    python EDS_Linescan_Plotter.py scan.csv -o fig.png --smooth 21
    python EDS_Linescan_Plotter.py scan.csv --title "DES | NGD" --note "r=0.82"
    python EDS_Linescan_Plotter.py --find FOLDER --match "linescan.csv" -o out/
"""
from __future__ import annotations

import argparse
import csv
import glob
import os
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

__all__ = ["load_linescan", "smooth", "plot_linescan"]

NAVY = "#102A4F"
GREY = "#9AA5B1"

# Sampled from Velox -int.png element maps (mean of brightest 2% of pixels).
ELEMENT_COLOURS = {
    "Cu": "#E00000",
    "Ni": "#00A000",
    "Ag": "#00A8B5",
    "Fe": "#C86400",
    "Au": "#B8860B",
    "Cr": "#7B3FA0",
    "Mn": "#C2185B",
}
DEFAULT_COLOUR = "#444444"


def load_linescan(path: str) -> tuple[list[str], np.ndarray]:
    """Return (channel_names, array) from a Velox linescan CSV.

    Column 0 of the array is position in metres; the rest are counts.
    Non-numeric rows are skipped rather than raising, because Velox
    occasionally emits trailing blank or annotation rows.
    """
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        rows = list(csv.reader(fh))
    if len(rows) < 4:
        raise ValueError(f"{path}: fewer than 4 rows, not a Velox export")

    names = [c.strip() for c in rows[0]]
    data = []
    for row in rows[3:]:
        if len(row) < len(names):
            continue
        try:
            data.append([float(v) for v in row[: len(names)]])
        except ValueError:
            continue
    if not data:
        raise ValueError(f"{path}: no numeric data rows")
    return names, np.asarray(data)


def smooth(x: np.ndarray, k: int) -> np.ndarray:
    """Edge-padded boxcar mean. k <= 1 returns the input unchanged."""
    if k <= 1:
        return x
    pad = k // 2
    return np.convolve(np.pad(x, pad, mode="edge"), np.ones(k) / k, mode="valid")


def plot_linescan(
    path: str,
    out_png: str,
    smooth_k: int = 21,
    title: str | None = None,
    note: str | None = None,
    dpi: int = 300,
) -> str:
    """Render one linescan CSV to a PNG. Returns the output path."""
    names, arr = load_linescan(path)
    pos_nm = arr[:, 0] * 1e9
    channels = {names[j]: arr[:, j] for j in range(1, len(names))}

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax2 = ax.twinx()

    if "HAADF" in channels:
        h = smooth(channels["HAADF"], smooth_k)
        span = max(h.max() - h.min(), 1e-12)
        hn = (h - h.min()) / span
        ax2.fill_between(pos_nm, hn, color=GREY, alpha=0.30, lw=0, zorder=1)
        ax2.plot(pos_nm, hn, color=GREY, lw=1.2, zorder=2)
        ax2.set_ylabel("HAADF (normalised)", color="#6B7280", fontsize=12)
        ax2.tick_params(axis="y", labelcolor="#6B7280", labelsize=10)
        ax2.set_ylim(0, 1.35)
        ax2.spines["top"].set_visible(False)

    for element, signal in channels.items():
        if element == "HAADF":
            continue
        s = smooth(signal, smooth_k)
        peak = s.max()
        if peak <= 0:
            print(f"  {element}: all zero, skipped", file=sys.stderr)
            continue
        ax.plot(
            pos_nm,
            s / peak,
            lw=2.8,
            color=ELEMENT_COLOURS.get(element, DEFAULT_COLOUR),
            label=f"{element}   (max {signal.max():.2f} cts)",
            zorder=3,
        )

    ax.set_xlabel("Position / nm", fontsize=13)
    ax.set_ylabel("EDS counts (each normalised)", fontsize=13)
    ax.set_xlim(pos_nm.min(), pos_nm.max())
    ax.set_ylim(0, 1.35)
    ax.tick_params(labelsize=11)
    ax.legend(loc="upper left", fontsize=11, frameon=False)
    ax.spines["top"].set_visible(False)

    if title:
        ax.set_title(title, fontsize=14, color=NAVY, fontweight="bold", pad=10)
    if note:
        ax.text(
            0.99, 0.99, note, transform=ax.transAxes,
            ha="right", va="top", fontsize=10.5, color="#374151",
        )

    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(out_png)) or ".", exist_ok=True)
    fig.savefig(out_png, dpi=dpi)
    plt.close(fig)
    return out_png


def _find(folder: str, match: str) -> list[str]:
    """Recursive search, skipping Velox's duplicate '.csv.csv' exports."""
    return sorted(
        p
        for p in glob.glob(os.path.join(folder, "**", "*.csv"), recursive=True)
        if os.path.basename(p).endswith(match) and not p.endswith(".csv.csv")
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("csv", nargs="?", help="Velox linescan CSV")
    ap.add_argument("-o", "--out", default=None,
                    help="output PNG, or a directory in --find mode")
    ap.add_argument("--find", metavar="FOLDER",
                    help="batch mode: search FOLDER recursively")
    ap.add_argument("--match", default="linescan.csv",
                    help="filename suffix to match in --find mode")
    ap.add_argument("--smooth", type=int, default=21,
                    help="boxcar window in points (default 21; 0 or 1 = none)")
    ap.add_argument("--title", default=None)
    ap.add_argument("--note", default=None)
    ap.add_argument("--dpi", type=int, default=300)
    args = ap.parse_args(argv)

    if args.find:
        hits = _find(args.find, args.match)
        if not hits:
            print(f"no files ending '{args.match}' under {args.find}", file=sys.stderr)
            return 1
        outdir = args.out or "."
        for p in hits:
            stem = os.path.splitext(os.path.basename(p))[0].replace(" ", "_")
            dst = plot_linescan(
                p, os.path.join(outdir, stem + ".png"),
                smooth_k=args.smooth, title=args.title,
                note=args.note, dpi=args.dpi,
            )
            print("wrote", dst)
        return 0

    if not args.csv:
        ap.error("give a CSV path, or use --find FOLDER")
    dst = plot_linescan(
        args.csv, args.out or "linescan.png",
        smooth_k=args.smooth, title=args.title,
        note=args.note, dpi=args.dpi,
    )
    print("wrote", dst)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
