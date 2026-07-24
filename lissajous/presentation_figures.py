"""Render the compact presentation figure set for the Lissajous v2 analysis.

This is a read-only presentation layer.  It consumes the CSV and NPZ products
already written below ``outputs/lissajous_v2`` and never invokes the
quantification pipeline.

Run with::

    python -m lissajous.presentation_figures --out outputs/lissajous_v2
"""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import matplotlib as mpl

mpl.use("Agg", force=True)
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Rectangle
import numpy as np

from .config import load_config


FIGURE_SIZE_IN = (12.0, 6.75)
PNG_DPI = 200

BLACK = "#111111"
GRAY = "#6B6B6B"
LIGHT_GRAY = "#E6E6E6"
BLUE = "#0072B2"
SKY = "#56B4E9"
GREEN = "#009E73"
ORANGE = "#E69F00"
RED = "#D55E00"
PURPLE = "#CC79A7"
YELLOW = "#F0E442"

MEDIUM_LABELS = {
    "manganese": "5 mM Mn nitrate",
    "water": "Pure water",
    "argon": "Argon",
    "bmim": "BMIM nitrate",
    "dry_fixture": "Dry fixture",
    "measurement_chain": "Series-chain model",
}

CONFIG_MEDIUM_FOLDERS = {
    "argon": "argon",
    "water": "pure water",
    "bmim": "ionic liquid",
    "manganese": "manganese nitrate in water",
}

SERIES_RLC_FIT_BOX = (
    "series-RLC chain fit (v1.2)†\n"
    "C = 2.7 pF [2.0–3.4]\n"
    "L = 0.36 H [0.30–0.50]\n"
    "R = 10 kΩ [7–15 kΩ]\n"
    r"$f_{\mathrm{res}}$ = 161.5 kHz [159–164]"
    "\n† order-of-magnitude; poor real-admittance residuals"
)

FIGURE_STEMS = (
    "fig1_frequency_identity",
    "fig2a_chain_dispersion_magnitude",
    "fig2b_chain_dispersion_real",
    "fig3_onset_and_charge",
    "fig4_synthesis_dose",
    "fig5_waveform_method",
    "fig6_qv_quantization",
    "fig7_energy_eligibility",
    "fig8_qc_gate_summary",
)

DECLARED_SYNTHESIS_RUNS = (
    "7_18/AgPd 5percent hydrogen",
    "7_19/1_3betaineEGAgCuNi-0003",
    "7_19/1_3betaineEGCuNi-0004",
    "7_19/1_3betaineEGPdHydrogen",
    "7_19/1_3betaineEGPdHydrogen-0002",
    "7_21/AgCuNi1_6betaine_12PD",
    "7_9/BMIMNTf2Pt20kHz",
)

RUN_SHORT_LABELS = {
    "7_18/AgPd 5percent hydrogen": "Ag–Pd · Ar/5% H₂",
    "7_19/1_3betaineEGAgCuNi-0003": "1:3 betaine:EG · Ag–Cu–Ni",
    "7_19/1_3betaineEGCuNi-0004": "1:3 betaine:EG · Cu–Ni",
    "7_19/1_3betaineEGPdHydrogen": "1:3 betaine:EG · Pd/H₂ · run 1",
    "7_19/1_3betaineEGPdHydrogen-0002": "1:3 betaine:EG · Pd/H₂ · run 2",
    "7_21/AgCuNi1_6betaine_12PD": "1:6 betaine:1,2-PD · Ag–Cu–Ni",
    "7_9/BMIMNTf2Pt20kHz": "BMIM–NTf₂ · Pt",
}

RUN_COLORS = {
    "7_18/AgPd 5percent hydrogen": BLUE,
    "7_19/1_3betaineEGAgCuNi-0003": GREEN,
    "7_19/1_3betaineEGCuNi-0004": ORANGE,
    "7_19/1_3betaineEGPdHydrogen": PURPLE,
    "7_19/1_3betaineEGPdHydrogen-0002": RED,
    "7_21/AgCuNi1_6betaine_12PD": BLUE,
    "7_9/BMIMNTf2Pt20kHz": RED,
}


@dataclass
class Inputs:
    root: Path
    manifest: list[dict[str, str]]
    per_capture: list[dict[str, str]]
    dispersion: list[dict[str, str]]
    discharge: list[dict[str, str]]
    onset: list[dict[str, str]]
    synthesis: list[dict[str, str]]
    frequency_consistency: list[dict[str, str]]
    figure_data: dict[str, np.ndarray]


@dataclass(frozen=True)
class Stat:
    value: float
    low: float | None
    high: float | None
    n: int | None


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as handle:
        reader = csv.DictReader(handle)
        return [
            {
                str(key).strip(): ("" if value is None else str(value).strip())
                for key, value in row.items()
                if key is not None
            }
            for row in reader
            if any(value not in (None, "") for value in row.values())
        ]


def _load_inputs(root: Path) -> Inputs:
    figure_data: dict[str, np.ndarray] = {}
    npz_path = root / "figure_data.npz"
    if npz_path.exists():
        with np.load(npz_path, allow_pickle=False) as archive:
            figure_data = {key: np.asarray(archive[key]) for key in archive.files}
    return Inputs(
        root=root,
        manifest=_read_csv(root / "manifest.csv"),
        per_capture=_read_csv(root / "per_capture_metrics.csv"),
        dispersion=_read_csv(root / "dispersion_master.csv"),
        discharge=_read_csv(root / "discharge_metrics.csv"),
        onset=_read_csv(root / "discharge_onset.csv"),
        synthesis=_read_csv(root / "synthesis_charge.csv"),
        frequency_consistency=_read_csv(root / "frequency_consistency.csv"),
        figure_data=figure_data,
    )


def _number(value: object) -> float | None:
    if value is None or isinstance(value, (bool, np.bool_)):
        return None
    if isinstance(value, np.ndarray):
        if value.size != 1:
            return None
        value = value.reshape(-1)[0]
    if isinstance(value, (int, float, np.integer, np.floating)):
        result = float(value)
        return result if math.isfinite(result) else None
    text = str(value).strip().replace(",", "").replace("−", "-")
    if not text or text.lower() in {"nan", "none", "null", "n/a", "na", "inf", "-inf"}:
        return None
    try:
        result = float(text)
    except ValueError:
        return None
    return result if math.isfinite(result) else None


def _integer(value: object) -> int | None:
    parsed = _number(value)
    return int(round(parsed)) if parsed is not None else None


def _truthy(value: object) -> bool:
    return str(value).strip().casefold() in {"1", "true", "yes", "y"}


def _finite(values: Iterable[object]) -> np.ndarray:
    parsed = [_number(value) for value in values]
    return np.asarray([value for value in parsed if value is not None], dtype=float)


def _summary(values: Iterable[object]) -> Stat | None:
    array = _finite(values)
    if not array.size:
        return None
    return Stat(
        value=float(np.median(array)),
        low=float(np.percentile(array, 2.5)),
        high=float(np.percentile(array, 97.5)),
        n=int(array.size),
    )


def _row_stat(row: Mapping[str, object], prefix: str) -> Stat | None:
    value = _number(row.get(f"{prefix}_median"))
    if value is None:
        return None
    return Stat(
        value=value,
        low=_number(row.get(f"{prefix}_p2_5")),
        high=_number(row.get(f"{prefix}_p97_5")),
        n=_integer(row.get(f"{prefix}_N")),
    )


def _canonical_medium(value: object) -> str:
    text = str(value).strip().casefold().replace("-", "_").replace(" ", "_")
    if "manganese" in text or "mn_nitrate" in text:
        return "manganese"
    if "pure_water" in text:
        return "water"
    if "argon" in text:
        return "argon"
    if "bmim" in text:
        return "bmim"
    if "dry_fixture" in text:
        return "dry_fixture"
    if "measurement_chain" in text:
        return "measurement_chain"
    return text or "unknown"


def _medium_label(value: object) -> str:
    return MEDIUM_LABELS.get(
        _canonical_medium(value),
        str(value).replace("_", " "),
    )


def _configure_medium_labels(root: Path) -> None:
    config_path = root / "config.yaml"
    if not config_path.is_file():
        return
    try:
        config = load_config(config_path)
    except (OSError, RuntimeError, ValueError):
        return
    entries = config.get("medium_labels", {})
    if not isinstance(entries, Mapping):
        return
    for canonical, folder in CONFIG_MEDIUM_FOLDERS.items():
        entry = entries.get(folder)
        if not isinstance(entry, Mapping):
            continue
        display = str(entry.get("display", "")).strip()
        if display:
            MEDIUM_LABELS[canonical] = display


def _medium_color(value: object) -> str:
    return {
        "manganese": GREEN,
        "water": SKY,
        "argon": BLUE,
        "bmim": ORANGE,
        "dry_fixture": BLACK,
        "measurement_chain": GRAY,
    }.get(_canonical_medium(value), PURPLE)


def _freq_number(label: object) -> float:
    text = str(label)
    for token in text.replace("kHz", "").replace("khz", "").split():
        value = _number(token)
        if value is not None:
            return value
    return math.inf


def _fmt(value: float | None, digits: int = 1) -> str:
    if value is None or not math.isfinite(value):
        return "unavailable"
    return f"{value:.{digits}f}"


def _register_font() -> str:
    for path in font_manager.findSystemFonts(fontext="ttf"):
        name = Path(path).name.casefold()
        if "calibri" in name or "carlito" in name:
            try:
                font_manager.fontManager.addfont(path)
            except (OSError, RuntimeError, ValueError):
                pass
    available = {entry.name.casefold(): entry.name for entry in font_manager.fontManager.ttflist}
    for desired in ("Calibri", "Carlito"):
        if desired.casefold() in available:
            return available[desired.casefold()]
    return "DejaVu Sans"


def _configure_style() -> None:
    family = _register_font()
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": [family, "Calibri", "Carlito", "DejaVu Sans"],
            "font.size": 11.0,
            "axes.labelsize": 12.0,
            "axes.titlesize": 12.0,
            "axes.linewidth": 0.9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": "#D8D8D8",
            "grid.linewidth": 0.65,
            "grid.alpha": 0.75,
            "xtick.labelsize": 11.0,
            "ytick.labelsize": 11.0,
            "lines.linewidth": 1.8,
            "lines.markersize": 6.0,
            "figure.dpi": PNG_DPI,
            "savefig.dpi": PNG_DPI,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def _panel_label(ax: plt.Axes, text: str) -> None:
    ax.text(
        0.012,
        0.985,
        text,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontweight="bold",
        fontsize=12,
        color=BLACK,
        zorder=20,
    )


def _empty_panel(ax: plt.Axes, message: str) -> None:
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.text(
        0.5,
        0.5,
        message,
        transform=ax.transAxes,
        ha="center",
        va="center",
        color=GRAY,
        fontsize=12,
    )


def _spread_annotations(
    fig: plt.Figure,
    ax: plt.Axes,
    annotations: list,
    *,
    min_gap_px: float = 7.0,
) -> None:
    """Push a stack of leader-line labels apart until none overlap.

    Label positions fixed in data coordinates only work for one particular
    axes size, so any layout change re-collides them. This measures the real
    rendered boxes and separates them, leaving the leader lines to follow.
    """

    live = [a for a in annotations if a is not None]
    if len(live) < 2:
        return
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    origin = ax.transData.transform((0.0, 0.0))
    unit = ax.transData.transform((0.0, 1.0))
    px_per_unit = unit[1] - origin[1]
    if abs(px_per_unit) < 1e-9:
        return

    for _ in range(4):
        try:
            ordered = sorted(
                live, key=lambda a: a.get_window_extent(renderer=renderer).y0
            )
        except Exception:  # pragma: no cover - degenerate renderer state
            return
        moved = False
        for index in range(1, len(ordered)):
            below = ordered[index - 1].get_window_extent(renderer=renderer)
            current = ordered[index].get_window_extent(renderer=renderer)
            gap = current.y0 - below.y1
            if gap >= min_gap_px:
                continue
            shift_px = min_gap_px - gap
            x_pos, y_pos = ordered[index].get_position()
            ordered[index].set_position((x_pos, y_pos + shift_px / px_per_unit))
            moved = True
        if not moved:
            return
        fig.canvas.draw()


def _scope_tag(fig: plt.Figure, scope: str) -> None:
    """Print the source record set as a self-contained on-canvas subtitle."""

    # Reserve the strip the tag occupies so panel titles cannot be laid out
    # underneath it. Without this the layout engine treats the top edge as
    # free space and centred titles collide with the tag.
    engine = fig.get_layout_engine()
    if engine is not None and hasattr(engine, "set"):
        try:
            rect = list(engine.get().get("rect", (0.0, 0.0, 1.0, 1.0)))
            top = rect[1] + rect[3]
            rect[3] = max(0.05, min(top, 0.962) - rect[1])
            engine.set(rect=tuple(rect))
        except Exception:  # pragma: no cover - layout engines without rect
            pass

    fig.text(
        0.5,
        0.995,
        f"Record set: {scope}",
        ha="center",
        va="top",
        color=GRAY,
        fontsize=10.5,
    )


def _series_rlc_fit_box(
    ax: plt.Axes,
    *,
    x: float = 0.985,
    y: float = 0.965,
) -> None:
    ax.text(
        x,
        y,
        SERIES_RLC_FIT_BOX,
        transform=ax.transAxes,
        ha="right",
        va="top",
        color=GRAY,
        fontsize=10,
        linespacing=1.18,
        bbox={
            "boxstyle": "round,pad=0.45",
            "facecolor": "white",
            "edgecolor": GRAY,
            "alpha": 0.94,
        },
        zorder=30,
    )


def _x_error(stat: Stat) -> np.ndarray | None:
    if stat.low is None or stat.high is None:
        return None
    return np.asarray(
        [[max(0.0, stat.value - stat.low)], [max(0.0, stat.high - stat.value)]],
        dtype=float,
    )


def _y_error(stat: Stat) -> np.ndarray | None:
    return _x_error(stat)


def _save(fig: plt.Figure, root: Path, stem: str) -> tuple[Path, Path]:
    directory = root / "presentation_figures"
    directory.mkdir(parents=True, exist_ok=True)
    png = directory / f"{stem}.png"
    pdf = directory / f"{stem}.pdf"
    fig.set_size_inches(*FIGURE_SIZE_IN, forward=True)
    fig.savefig(
        png,
        dpi=PNG_DPI,
        facecolor="white",
        metadata={"Software": "lissajous.presentation_figures"},
    )
    fig.savefig(
        pdf,
        facecolor="white",
        metadata={
            "Creator": "lissajous.presentation_figures",
            "Subject": "Lissajous v2 presentation figure",
            "CreationDate": None,
            "ModDate": None,
        },
    )
    plt.close(fig)
    return png, pdf


def _write_caption(
    root: Path,
    stem: str,
    title: str,
    takeaway: str,
    bullets: Sequence[str],
) -> Path:
    directory = root / "presentation_captions"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{stem}.md"
    lines = [f"# {title}", "", f"**Takeaway.** {takeaway}", ""]
    lines.extend(f"- {bullet}" for bullet in bullets)
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def _condition_label(medium: object, freq_label: object, level: object | None = None) -> str:
    base = f"{_medium_label(medium)} · {str(freq_label).strip()}"
    parsed_level = _number(level)
    if parsed_level is not None:
        base += f" · {parsed_level:g}%"
    return base


def _figure1_frequency_identity(data: Inputs) -> tuple[Path, Path, Path]:
    rows = [
        row
        for row in data.manifest
        if row.get("dataset_type") == "voltage_ladder"
        and _number(row.get("f0_Hz")) is not None
    ]
    grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[(_canonical_medium(row.get("medium")), row.get("freq_label", ""))].append(row)

    summaries: list[dict[str, object]] = []
    for (medium, freq_label), group in grouped.items():
        burst_label = _summary(row.get("nominal_frequency_kHz") for row in group)
        burst = _summary(
            (_number(row.get("burst_Hz")) or math.nan) / 1000.0 for row in group
        )
        carrier = _summary(
            (_number(row.get("f0_Hz")) or math.nan) / 1000.0 for row in group
        )
        if burst_label and carrier:
            summaries.append(
                {
                    "medium": medium,
                    "freq_label": freq_label,
                    "burst_label": burst_label,
                    "burst": burst,
                    "carrier": carrier,
                }
            )
    summaries.sort(
        key=lambda item: (
            float(item["burst_label"].value),  # type: ignore[union-attr]
            _medium_label(item["medium"]),
        )
    )

    fig, ax = plt.subplots(figsize=FIGURE_SIZE_IN, layout="constrained")
    _scope_tag(fig, "voltage-ladder")
    if not summaries:
        _empty_panel(ax, "No voltage-ladder carrier frequencies were available.")
    else:
        ax.axvspan(3.0, 20.0, color=SKY, alpha=0.12, zorder=0)
        ax.axvspan(60.0, 180.0, color=ORANGE, alpha=0.065, zorder=0)
        ax.axvspan(120.0, 130.0, color=ORANGE, alpha=0.18, zorder=0)
        y = np.arange(len(summaries), dtype=float)
        labels: list[str] = []
        for idx, item in enumerate(summaries):
            burst_label = item["burst_label"]
            carrier = item["carrier"]
            assert isinstance(burst_label, Stat) and isinstance(carrier, Stat)
            color = _medium_color(item["medium"])
            ax.plot(
                [burst_label.value, carrier.value],
                [idx, idx],
                color="#A5A5A5",
                linewidth=2.0,
                zorder=2,
            )
            ax.scatter(
                burst_label.value,
                idx,
                s=62,
                facecolor="white",
                edgecolor=color,
                linewidth=1.8,
                zorder=4,
            )
            ax.errorbar(
                carrier.value,
                idx,
                xerr=_x_error(carrier),
                fmt="o",
                color=color,
                ecolor=color,
                elinewidth=1.5,
                capsize=3,
                markersize=7,
                zorder=5,
            )
            low = carrier.low if carrier.low is not None else carrier.value
            high = carrier.high if carrier.high is not None else carrier.value
            interval = (
                f"{carrier.value:.1f} [{low:.1f}–{high:.1f}] kHz"
                if abs(high - low) >= 0.1
                else f"{carrier.value:.1f} kHz"
            )
            ax.text(
                max(carrier.value * 1.025, carrier.value + 1.8),
                idx,
                interval,
                ha="left",
                va="center",
                color=color,
                fontsize=11,
            )
            labels.append(
                f"{_medium_label(item['medium'])} · {burst_label.value:g} kHz burst"
            )
            if idx == 0:
                ax.annotate(
                    "burst frequency",
                    (burst_label.value, idx),
                    xytext=(0, -28),
                    textcoords="offset points",
                    color=GRAY,
                    ha="center",
                    va="top",
                    arrowprops={"arrowstyle": "-", "color": GRAY, "lw": 0.9},
                )
                ax.annotate(
                    "measured f0",
                    (carrier.value, idx),
                    xytext=(0, -28),
                    textcoords="offset points",
                    color=GRAY,
                    ha="center",
                    va="top",
                    arrowprops={"arrowstyle": "-", "color": GRAY, "lw": 0.9},
                )
        ax.set_xscale("log")
        ax.set_xlim(3.0, 225.0)
        ax.set_xticks([4, 10, 20, 60, 120, 160, 220])
        ax.set_xticklabels(["4", "10", "20", "60", "120", "160", "220"])
        ax.set_yticks(y, labels)
        ax.invert_yaxis()
        ax.set_xlabel("Frequency (kHz, logarithmic scale)")
        ax.set_ylabel("")
        ax.grid(axis="y", visible=False)
        ax.text(
            0.02,
            0.985,
            "LF target ≤20 kHz",
            transform=ax.transAxes,
            color=BLUE,
            ha="left",
            va="top",
        )
        ax.text(
            0.61,
            0.985,
            "carrier-transfer region",
            transform=ax.transAxes,
            color=ORANGE,
            ha="center",
            va="top",
        )

    all_f0 = _finite(row.get("f0_Hz") for row in data.manifest) / 1000.0
    low_frequency_count = int(np.sum(all_f0 <= 20.0)) if all_f0.size else 0
    media_rows = {
        _canonical_medium(row.get("medium")) for row in data.frequency_consistency
    }
    r3_failed = sum(
        "insufficient" in row.get("R3_status", "").casefold()
        for row in data.frequency_consistency
    )
    r3_total = len(media_rows) or len(data.frequency_consistency)
    ax.text(
        0.985,
        1.055,
        f"{low_frequency_count:,}/{all_f0.size:,} carriers ≤20 kHz  ·  "
        f"R3 not testable in {r3_failed}/{r3_total} media",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        color=RED,
        fontweight="bold",
    )

    stem = FIGURE_STEMS[0]
    png, pdf = _save(fig, data.root, stem)
    burst_text = "; ".join(
        f"{item['burst_label'].value:g}→{item['carrier'].value:.1f} kHz"
        for item in summaries
        if isinstance(item["burst_label"], Stat) and isinstance(item["carrier"], Stat)
    )
    caption = _write_caption(
        data.root,
        stem,
        "Figure 1 — Burst-frequency labels do not provide a low-frequency carrier anchor",
        (
            f"None of the {all_f0.size:,} parsed captures had a measured carrier at or "
            "below 20 kHz, so the low-frequency R3 hypothesis is not testable."
        ),
        (
            f"Voltage-ladder condition medians (burst→carrier): {burst_text}.",
            (
                "Horizontal intervals are capture-level 2.5–97.5 percentiles; the open "
                "symbol is the burst-frequency label and the filled symbol is measured f0."
            ),
            (
                f"The saved frequency-consistency table reports insufficient genuine "
                f"low-frequency anchors for {r3_failed}/{r3_total} media."
            ),
            "Source: manifest.csv and frequency_consistency.csv; no new quantification.",
        ),
    )
    return png, pdf, caption


def _dispersion_rows(
    rows: Sequence[Mapping[str, str]],
    source: str,
) -> list[Mapping[str, str]]:
    return [
        row
        for row in rows
        if row.get("source_type") == source
        and _number(row.get("frequency_kHz")) is not None
    ]


def _figure2_chain_dispersion_composite(data: Inputs) -> tuple[Path, Path, Path]:
    model = _dispersion_rows(data.dispersion, "series_RLC_model")
    comb = [
        row
        for row in _dispersion_rows(data.dispersion, "7_20_multiline")
        if (_number(row.get("N")) or 0) >= 60
        and (_number(row.get("frequency_kHz")) or 0) >= 60
    ]
    dry = _dispersion_rows(data.dispersion, "v1.2_dry_fixture")
    operational = [
        row
        for row in data.dispersion
        if row.get("source_type")
        in {"july7_8_operational_fundamental", "v1.1_carrier"}
        and _number(row.get("frequency_kHz")) is not None
    ]

    fig, axes = plt.subplots(
        2,
        1,
        figsize=FIGURE_SIZE_IN,
        sharex=True,
        layout="constrained",
        gridspec_kw={"height_ratios": [1.0, 1.0]},
    )
    ax_mag, ax_real = axes
    _scope_tag(
        fig,
        "voltage-ladder + operational (Lissajousfigure) + chain calibration",
    )
    _panel_label(ax_mag, "A  Apparent magnitude")
    _panel_label(ax_real, "B  Real component")

    for ax in axes:
        ax.axvspan(120.0, 130.0, color=ORANGE, alpha=0.15, zorder=0)
        ax.set_xscale("log")
        ax.set_xlim(4.0, 220.0)
        ax.set_xticks([4, 10, 20, 60, 100, 120, 160, 200])
        ax.set_xticklabels(["4", "10", "20", "60", "100", "120", "160", "200"])
        ax.grid(axis="x", which="minor", visible=False)

    model_f = _finite(row.get("frequency_kHz") for row in model)
    model_mag = _finite(row.get("Cmag_pF") for row in model)
    model_real = _finite(row.get("Creal_pF") for row in model)
    model_peak_f: float | None = None
    model_peak_c: float | None = None
    if model_f.size and model_mag.size == model_f.size:
        order = np.argsort(model_f)
        model_f = model_f[order]
        model_mag = model_mag[order]
        ax_mag.plot(model_f, model_mag, color=GRAY, linestyle="--", linewidth=2.0)
        peak_idx = int(np.argmax(model_mag))
        model_peak_f = float(model_f[peak_idx])
        model_peak_c = float(model_mag[peak_idx])
        ax_mag.scatter(model_peak_f, model_peak_c, color=GRAY, marker="D", s=42, zorder=5)
        ax_mag.annotate(
            f"series-RLC chain fit (v1.2)†\npeak {model_peak_f:.0f} kHz",
            (model_peak_f, model_peak_c),
            xytext=(18, -5),
            textcoords="offset points",
            color=GRAY,
            ha="left",
            va="center",
        )
    if model_f.size and model_real.size == model_f.size:
        ax_real.plot(model_f, model_real, color=GRAY, linestyle="--", linewidth=2.0)
        if model_peak_f is not None:
            ax_real.axvline(model_peak_f, color=GRAY, linestyle=":", linewidth=1.2)
            ax_real.text(
                model_peak_f * 1.025,
                0.05,
                "model peak†",
                transform=ax_real.get_xaxis_transform(),
                rotation=90,
                color=GRAY,
                ha="left",
                va="bottom",
            )
        _series_rlc_fit_box(ax_mag, x=0.985, y=0.965)

    source_style = {
        "4khz1kv_0003": (BLUE, "7_20 · 4-kHz drive"),
        "10khz1kv_0002": (ORANGE, "7_20 · 10-kHz drive"),
        "20khz1kv": (GREEN, "7_20 · 20-kHz drive"),
    }
    comb_groups: dict[str, list[Mapping[str, str]]] = defaultdict(list)
    for row in comb:
        comb_groups[row.get("provenance", "")].append(row)
    endpoint_offsets = {
        "4khz1kv_0003": (8, 4),
        "10khz1kv_0002": (8, -14),
        "20khz1kv": (8, 12),
    }
    for provenance, group in comb_groups.items():
        color, direct_label = source_style.get(provenance, (PURPLE, provenance))
        ordered = sorted(group, key=lambda row: _number(row.get("frequency_kHz")) or 0)
        x = np.asarray([_number(row.get("frequency_kHz")) for row in ordered], dtype=float)
        mag = np.asarray([_number(row.get("Cmag_pF")) for row in ordered], dtype=float)
        real = np.asarray([_number(row.get("Creal_pF")) for row in ordered], dtype=float)
        ax_mag.plot(x, mag, color=color, alpha=0.62, linewidth=1.45)
        ax_mag.scatter(
            x,
            mag,
            s=29,
            facecolor="white",
            edgecolor=color,
            linewidth=1.2,
            alpha=0.95,
            zorder=4,
        )
        ax_real.plot(x, real, color=color, alpha=0.62, linewidth=1.45)
        ax_real.scatter(
            x,
            real,
            s=29,
            facecolor="white",
            edgecolor=color,
            linewidth=1.2,
            alpha=0.95,
            zorder=4,
        )
        offset = endpoint_offsets.get(provenance, (8, 0))
        ax_mag.annotate(
            direct_label,
            (x[-1], mag[-1]),
            xytext=offset,
            textcoords="offset points",
            color=color,
            ha="left",
            va="center",
        )
        ax_real.annotate(
            direct_label,
            (x[-1], real[-1]),
            xytext=offset,
            textcoords="offset points",
            color=color,
            ha="left",
            va="center",
        )

    operational_groups: dict[str, list[Mapping[str, str]]] = defaultdict(list)
    for row in operational:
        operational_groups[_canonical_medium(row.get("medium"))].append(row)
    op_offsets = {
        "argon": (-18, -17),
        "water": (8, -10),
        "manganese": (8, 8),
        "bmim": (8, 18),
    }
    for medium, group in operational_groups.items():
        frequency = _summary(row.get("frequency_kHz") for row in group)
        magnitude = _summary(row.get("Cmag_pF") for row in group)
        real = _summary(row.get("Creal_pF") for row in group)
        if not frequency or not magnitude:
            continue
        color = _medium_color(medium)
        ax_mag.scatter(
            frequency.value,
            magnitude.value,
            marker="D",
            s=58,
            facecolor=color,
            edgecolor="white",
            linewidth=0.8,
            zorder=7,
        )
        ax_mag.annotate(
            f"{_medium_label(medium)} operational",
            (frequency.value, magnitude.value),
            xytext=op_offsets.get(medium, (7, 7)),
            textcoords="offset points",
            color=color,
            ha="left",
            va="center",
        )
        if real:
            ax_real.scatter(
                frequency.value,
                real.value,
                marker="D",
                s=58,
                facecolor=color,
                edgecolor="white",
                linewidth=0.8,
                zorder=7,
            )
            ax_real.annotate(
                _medium_label(medium),
                (frequency.value, real.value),
                xytext=op_offsets.get(medium, (7, 7)),
                textcoords="offset points",
                color=color,
                ha="left",
                va="center",
            )

    dry_frequency: float | None = None
    dry_magnitude: float | None = None
    if dry:
        dry_frequency = _number(dry[0].get("frequency_kHz"))
        dry_magnitude = _number(dry[0].get("Cmag_pF"))
        if dry_frequency is not None and dry_magnitude is not None:
            ax_mag.scatter(
                dry_frequency,
                dry_magnitude,
                marker="*",
                s=130,
                facecolor=BLACK,
                edgecolor="white",
                linewidth=0.8,
                zorder=8,
            )
            ax_mag.annotate(
                f"dry fixture · {dry_magnitude:.1f} pF",
                (dry_frequency, dry_magnitude),
                xytext=(10, -18),
                textcoords="offset points",
                color=BLACK,
                ha="left",
                va="center",
            )

    ax_mag.set_ylabel(r"$|C_{\mathrm{app}}|$ (pF)")
    ax_real.set_ylabel(r"$\mathrm{Re}(C_{\mathrm{app}})$ (pF)")
    ax_real.set_xlabel("Frequency (kHz, logarithmic scale)")
    ax_mag.set_ylim(bottom=-3)
    ax_real.axhline(0, color=BLACK, linewidth=0.9)
    ax_real.set_ylim(-75, 155)
    ax_mag.text(
        124.8,
        0.96,
        "measured carrier band",
        transform=ax_mag.get_xaxis_transform(),
        color=ORANGE,
        ha="center",
        va="top",
        rotation=90,
    )
    ax_mag.text(
        0.015,
        0.07,
        "Hollow 7_20 points: N≥60, quantization-limited",
        transform=ax_mag.transAxes,
        color=GRAY,
        ha="left",
        va="bottom",
    )

    stem = FIGURE_STEMS[1]
    png, pdf = _save(fig, data.root, stem)
    caption = _write_caption(
        data.root,
        stem,
        "Figure 2 — The measurement chain is strongly frequency dispersive near the carrier band",
        (
            "The apparent capacitance rises sharply and its real component changes sign "
            "near the 120–170 kHz operating region, so it cannot be treated as a static "
            "geometric dielectric capacitance."
        ),
        (
            (
                f"The series-RLC chain fit (v1.2) peaks at "
                f"{_fmt(model_peak_f, 1)} kHz and {_fmt(model_peak_c, 1)} pF†."
            ),
            (
                f"The dry fixture is {_fmt(dry_magnitude, 1)} pF at "
                f"{_fmt(dry_frequency, 0)} kHz; operational diamonds are medium-level "
                "medians of saved carrier-line values."
            ),
            (
                "Only grouped 7_20 spectral rows with N≥60 and f≥60 kHz are shown; their "
                "hollow symbols retain the quantization-limited status."
            ),
            (
                "†The series-chain fit has poor real-admittance residuals and is shown "
                "only as order-of-magnitude transfer context."
            ),
            "Source: dispersion_master.csv; active-secant and active-Cd scenarios omitted.",
        ),
    )
    return png, pdf, caption


def _figure2_chain_dispersion(
    data: Inputs,
) -> tuple[Path, Path, Path, Path, Path, Path]:
    """Split magnitude and in-phase chain evidence into two readable slides."""

    model = _dispersion_rows(data.dispersion, "series_RLC_model")
    dry = _dispersion_rows(data.dispersion, "v1.2_dry_fixture")
    comb_rows = [
        row
        for row in _dispersion_rows(data.dispersion, "7_20_multiline")
        if (_number(row.get("N")) or 0) >= 60
        and (_number(row.get("frequency_kHz")) or 0) >= 60
    ]
    operational = [
        row
        for row in data.dispersion
        if row.get("source_type")
        in {"july7_8_operational_fundamental", "v1.1_carrier"}
        and _number(row.get("frequency_kHz")) is not None
    ]

    model_points = [
        (
            _number(row.get("frequency_kHz")),
            _number(row.get("Cmag_pF")),
            _number(row.get("Creal_pF")),
        )
        for row in model
    ]
    model_points = [
        (frequency, magnitude, real)
        for frequency, magnitude, real in model_points
        if frequency is not None and magnitude is not None and real is not None
    ]
    model_f = np.asarray([point[0] for point in model_points], dtype=float)
    model_mag = np.asarray([point[1] for point in model_points], dtype=float)
    model_real = np.asarray([point[2] for point in model_points], dtype=float)
    model_peak_f: float | None = None
    model_peak_mag: float | None = None
    if model_f.size:
        model_order = np.argsort(model_f)
        model_f = model_f[model_order]
        model_mag = model_mag[model_order]
        model_real = model_real[model_order]
        peak_index = int(np.argmax(model_mag))
        model_peak_f = float(model_f[peak_index])
        model_peak_mag = float(model_mag[peak_index])

    operational_groups: dict[str, list[Mapping[str, str]]] = defaultdict(list)
    for row in operational:
        medium = _canonical_medium(row.get("medium"))
        if medium in {"argon", "water", "manganese", "bmim"}:
            operational_groups[medium].append(row)
    operational_stats: dict[str, tuple[Stat, Stat, Stat | None]] = {}
    for medium, rows in operational_groups.items():
        frequency = _summary(row.get("frequency_kHz") for row in rows)
        magnitude = _summary(row.get("Cmag_pF") for row in rows)
        real = _summary(row.get("Creal_pF") for row in rows)
        if frequency and magnitude:
            operational_stats[medium] = (frequency, magnitude, real)

    def configure_frequency_axis(ax: plt.Axes) -> None:
        ax.axvspan(120.0, 130.0, color=ORANGE, alpha=0.16, zorder=0)
        ax.set_xscale("log")
        ax.set_xlim(55.0, 280.0)
        ax.set_xticks([60, 80, 100, 120, 160, 200, 260])
        ax.set_xticklabels(["60", "80", "100", "120", "160", "200", "260"])
        ax.grid(axis="x", which="minor", visible=False)
        ax.set_xlabel("Frequency (kHz, logarithmic scale)")
        ax.text(
            124.8,
            0.86,
            "measured\ncarrier band",
            transform=ax.get_xaxis_transform(),
            color=ORANGE,
            ha="center",
            va="top",
            rotation=90,
        )

    # Figure 2a: one well-populated comb, dry fixture, and operational medians.
    fig_mag, ax_mag = plt.subplots(figsize=FIGURE_SIZE_IN, layout="constrained")
    _scope_tag(
        fig_mag,
        "voltage-ladder + operational (Lissajousfigure) + chain calibration",
    )
    _panel_label(ax_mag, "2A  Apparent magnitude")
    configure_frequency_axis(ax_mag)
    if model_f.size:
        ax_mag.plot(model_f, model_mag, color=GRAY, linestyle="--", linewidth=2.2)
        if model_peak_f is not None and model_peak_mag is not None:
            ax_mag.scatter(
                model_peak_f,
                model_peak_mag,
                marker="D",
                s=58,
                color=GRAY,
                zorder=5,
            )
        _series_rlc_fit_box(ax_mag, x=0.985, y=0.965)

    comb20 = sorted(
        [row for row in comb_rows if row.get("provenance") == "20khz1kv"],
        key=lambda row: _number(row.get("frequency_kHz")) or 0,
    )
    comb20_x = np.asarray(
        [_number(row.get("frequency_kHz")) for row in comb20],
        dtype=float,
    )
    comb20_mag = np.asarray([_number(row.get("Cmag_pF")) for row in comb20], dtype=float)
    if comb20_x.size:
        ax_mag.plot(comb20_x, comb20_mag, color=GREEN, linewidth=2.4, alpha=0.8)
        ax_mag.scatter(
            comb20_x,
            comb20_mag,
            s=62,
            facecolor="white",
            edgecolor=GREEN,
            linewidth=1.8,
            zorder=6,
        )
        selected_progression = [
            row
            for row in comb20
            if round(_number(row.get("frequency_kHz")) or -1) in {80, 120, 140, 160}
        ]
        progression_text = " → ".join(
            f"{_number(row.get('Cmag_pF')):.1f}"
            for row in selected_progression
            if _number(row.get("Cmag_pF")) is not None
        )
        frequency_text = " → ".join(
            f"{_number(row.get('frequency_kHz')):.0f}"
            for row in selected_progression
            if _number(row.get("frequency_kHz")) is not None
        )
        ax_mag.annotate(
            f"7_20 · 20-kHz comb\n{progression_text} pF\nat {frequency_text} kHz",
            (comb20_x[-1], comb20_mag[-1]),
            xytext=(68, 112),
            textcoords="data",
            ha="left",
            va="center",
            color=GREEN,
            fontweight="bold",
            arrowprops={"arrowstyle": "-", "color": GREEN, "lw": 1.2},
            bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": GREEN},
        )

    operational_mag_positions = {
        "argon": (184, 20),
        "water": (184, 36),
        "manganese": (184, 53),
        "bmim": (184, 70),
    }
    mag_annotations: list = []
    for medium in ("argon", "water", "manganese", "bmim"):
        stats = operational_stats.get(medium)
        if not stats:
            continue
        frequency, magnitude, _ = stats
        color = _medium_color(medium)
        ax_mag.scatter(
            frequency.value,
            magnitude.value,
            marker="D",
            s=76,
            facecolor=color,
            edgecolor="white",
            linewidth=1.0,
            zorder=8,
        )
        mag_annotations.append(
            ax_mag.annotate(
                f"{_medium_label(medium)} operational · {magnitude.value:.1f} pF",
                (frequency.value, magnitude.value),
                xytext=operational_mag_positions[medium],
                textcoords="data",
                color=color,
                ha="left",
                va="center",
                arrowprops={"arrowstyle": "-", "color": color, "lw": 0.9},
            )
        )

    dry_frequency = _number(dry[0].get("frequency_kHz")) if dry else None
    dry_magnitude = _number(dry[0].get("Cmag_pF")) if dry else None
    if dry_frequency is not None and dry_magnitude is not None:
        ax_mag.scatter(
            dry_frequency,
            dry_magnitude,
            marker="*",
            s=180,
            facecolor=BLACK,
            edgecolor="white",
            linewidth=0.9,
            zorder=9,
        )
        mag_annotations.append(
            ax_mag.annotate(
                f"dry fixture · {dry_magnitude:.1f} pF at {dry_frequency:.0f} kHz",
                (dry_frequency, dry_magnitude),
                xytext=(184, 88),
                textcoords="data",
                color=BLACK,
                ha="left",
                va="center",
                arrowprops={"arrowstyle": "-", "color": BLACK, "lw": 0.9},
            )
        )
    ax_mag.set_ylabel(r"$|C_{\mathrm{app}}|$ (pF)")
    ax_mag.set_ylim(-3, 150)
    ax_mag.text(
        0.985,
        0.025,
        "Hollow comb points: N≥60 · quantization-limited",
        transform=ax_mag.transAxes,
        color=GRAY,
        ha="right",
        va="bottom",
    )

    stem_mag = FIGURE_STEMS[1]
    png_mag, pdf_mag = _save(fig_mag, data.root, stem_mag)
    caption_mag = _write_caption(
        data.root,
        stem_mag,
        "Figure 2a — Apparent magnitude rises sharply through the carrier band",
        (
            "In the best-populated 20-kHz comb, |C_app| rises from 4.9 to 97.8 pF "
            "between 80 and 160 kHz, so one static capacitance cannot calibrate every condition."
        ),
        (
            "20-kHz comb progression: 4.9, 27.0, 61.7, and 97.8 pF at 80, 120, 140, and 160 kHz.",
            (
                f"Dry fixture: {_fmt(dry_magnitude, 1)} pF at {_fmt(dry_frequency, 0)} kHz; "
                "diamonds are medium-level medians of saved operational carrier points."
            ),
            (
                f"The series-RLC chain fit (v1.2) peaks at {_fmt(model_peak_mag, 1)} pF "
                f"and {_fmt(model_peak_f, 1)} kHz†."
            ),
            (
                "†The comb is quantization-limited and the model is order-of-magnitude "
                "transfer-chain context, not physical Cd or Ccell."
            ),
            "Source: dispersion_master.csv; active-secant and active-Cd scenarios omitted.",
        ),
    )

    # Figure 2b: isolate the in-phase sign reversal near resonance.
    fig_real, ax_real = plt.subplots(figsize=FIGURE_SIZE_IN, layout="constrained")
    _scope_tag(
        fig_real,
        "voltage-ladder + operational (Lissajousfigure) + chain calibration",
    )
    _panel_label(ax_real, "2B  Real component")
    configure_frequency_axis(ax_real)
    if model_f.size:
        ax_real.plot(model_f, model_real, color=GRAY, linestyle="--", linewidth=2.2)
        if model_peak_f is not None:
            ax_real.axvline(model_peak_f, color=GRAY, linestyle=":", linewidth=1.2)
        _series_rlc_fit_box(ax_real, x=0.985, y=0.965)

    comb4 = sorted(
        [row for row in comb_rows if row.get("provenance") == "4khz1kv_0003"],
        key=lambda row: _number(row.get("frequency_kHz")) or 0,
    )
    comb4_x = np.asarray([_number(row.get("frequency_kHz")) for row in comb4], dtype=float)
    comb4_real = np.asarray([_number(row.get("Creal_pF")) for row in comb4], dtype=float)
    if comb4_x.size:
        ax_real.plot(comb4_x, comb4_real, color=BLUE, linewidth=2.5, alpha=0.82)
        ax_real.scatter(
            comb4_x,
            comb4_real,
            s=54,
            facecolor="white",
            edgecolor=BLUE,
            linewidth=1.7,
            zorder=6,
        )
        ax_real.annotate(
            "7_20 · 4-kHz comb",
            (comb4_x[-1], comb4_real[-1]),
            xytext=(188, -61),
            textcoords="data",
            color=BLUE,
            ha="left",
            va="center",
            arrowprops={"arrowstyle": "-", "color": BLUE, "lw": 1.0},
        )

    def closest_comb4(frequency: float) -> tuple[float | None, float | None]:
        if not comb4_x.size:
            return None, None
        index = int(np.argmin(np.abs(comb4_x - frequency)))
        return float(comb4_x[index]), float(comb4_real[index])

    f152, real152 = closest_comb4(152.0)
    f156, real156 = closest_comb4(156.0)
    if None not in (f152, real152, f156, real156):
        ax_real.scatter(
            [f152, f156],
            [real152, real156],
            s=92,
            facecolor=[GREEN, RED],
            edgecolor="white",
            linewidth=1.0,
            zorder=9,
        )
        ax_real.annotate(
            f"sign reversal\n+{real152:.2f} pF at {f152:.0f} kHz\n"
            f"{real156:.2f} pF at {f156:.0f} kHz",
            ((f152 + f156) / 2.0, (real152 + real156) / 2.0),
            xytext=(68, -38),
            textcoords="data",
            color=RED,
            ha="left",
            va="center",
            fontweight="bold",
            arrowprops={"arrowstyle": "->", "color": RED, "lw": 1.3},
            bbox={"boxstyle": "round,pad=0.4", "facecolor": "white", "edgecolor": RED},
        )

    # 12-unit spacing renders as ~31 px here, exactly one line height, so the
    # labels touch. 16 units buys roughly a 9 px gap at this axes size.
    operational_real_positions = {
        "argon": (68, -3),
        "water": (68, 13),
        "manganese": (68, 29),
        "bmim": (68, 45),
    }
    real_annotations: list = []
    for medium in ("argon", "water", "manganese", "bmim"):
        stats = operational_stats.get(medium)
        if not stats or stats[2] is None:
            continue
        frequency, _, real = stats
        assert real is not None
        color = _medium_color(medium)
        ax_real.scatter(
            frequency.value,
            real.value,
            marker="D",
            s=72,
            facecolor=color,
            edgecolor="white",
            linewidth=1.0,
            zorder=8,
        )
        real_annotations.append(
            ax_real.annotate(
                f"{_medium_label(medium)} operational · {real.value:.1f} pF",
                (frequency.value, real.value),
                xytext=operational_real_positions[medium],
                textcoords="data",
                color=color,
                ha="left",
                va="center",
                arrowprops={"arrowstyle": "-", "color": color, "lw": 0.9},
            )
        )
    ax_real.axhline(0, color=BLACK, linewidth=0.9)
    ax_real.set_ylim(-75, 60)
    ax_real.set_ylabel(r"$\mathrm{Re}(C_{\mathrm{app}})$ (pF)")
    ax_real.text(
        0.985,
        0.035,
        "carrier-transfer / QC evidence · not a negative physical capacitance",
        transform=ax_real.transAxes,
        color=RED,
        ha="right",
        va="bottom",
        fontweight="bold",
    )

    stem_real = FIGURE_STEMS[2]
    png_real, pdf_real = _save(fig_real, data.root, stem_real)
    caption_real = _write_caption(
        data.root,
        stem_real,
        "Figure 2b — The in-phase response reverses sign near resonance",
        (
            f"Re(C_app) changes from +{_fmt(real152, 2)} pF at {_fmt(f152, 0)} kHz "
            f"to {_fmt(real156, 2)} pF at {_fmt(f156, 0)} kHz, directly exposing "
            "the monitoring chain's phase sensitivity."
        ),
        (
            "The highlighted values come from the N≥60 7_20 4-kHz-drive comb.",
            "Operational diamonds are medium-level medians at the measured carrier band.",
            (
                "The sign reversal is carrier-transfer/QC context, not negative physical "
                "dielectric capacitance or a geometric dielectric inference."
            ),
            (
                "†The dashed series-RLC chain fit (v1.2) is order-of-magnitude "
                "context because its real-admittance residuals are poor."
            ),
            "Source: dispersion_master.csv; active-secant and active-Cd scenarios omitted.",
        ),
    )
    return png_mag, pdf_mag, caption_mag, png_real, pdf_real, caption_real


def _figure3_onset_and_charge(data: Inputs) -> tuple[Path, Path, Path]:
    onset_rows = sorted(
        data.onset,
        key=lambda row: (
            _freq_number(row.get("freq_label")),
            _medium_label(row.get("medium")),
        ),
    )
    onset_keys = {
        (_canonical_medium(row.get("medium")), row.get("freq_label", "")): row
        for row in onset_rows
        if _number(row.get("onset_level_pct")) is not None
    }
    charge_groups: dict[tuple[str, str], list[Mapping[str, str]]] = defaultdict(list)
    for row in data.discharge:
        key = (_canonical_medium(row.get("medium")), row.get("freq_label", ""))
        if (
            row.get("dataset_type") == "voltage_ladder"
            and key in onset_keys
            and _number(row.get("level_pct")) is not None
            and _row_stat(row, "dQ_cycle_nC") is not None
        ):
            charge_groups[key].append(row)

    fig, axes = plt.subplots(
        1,
        2,
        figsize=FIGURE_SIZE_IN,
        layout="constrained",
        gridspec_kw={"width_ratios": [0.92, 1.25]},
    )
    ax_onset, ax_charge = axes
    _scope_tag(fig, "voltage-ladder")
    _panel_label(ax_onset, "A  Objective onset gate")
    _panel_label(ax_charge, "B  Raw charge after onset")

    if onset_rows:
        y = np.arange(len(onset_rows), dtype=float)
        labels: list[str] = []
        for idx, row in enumerate(onset_rows):
            medium = _canonical_medium(row.get("medium"))
            color = _medium_color(medium)
            onset = _number(row.get("onset_level_pct"))
            if onset is not None:
                ax_onset.scatter(onset, idx, color=color, s=72, zorder=4)
                evidence_terms = {
                    "dq": "charge rise",
                    "loop_area": "loop area",
                    "harmonics": "harmonic rise",
                }
                evidence = " + ".join(
                    evidence_terms.get(term.strip().casefold(), term.replace("_", " "))
                    for term in row.get("evidence", "").split(",")
                    if term.strip()
                )
                ax_onset.text(
                    onset + 1.1,
                    idx,
                    f"{onset:g}% · {evidence}",
                    color=color,
                    ha="left",
                    va="center",
                )
            else:
                ax_onset.annotate(
                    "",
                    xy=(120.0, idx),
                    xytext=(112.0, idx),
                    arrowprops={"arrowstyle": "-|>", "color": RED, "lw": 1.8},
                )
                ax_onset.text(
                    121.0,
                    idx,
                    "not detected through 115%",
                    color=RED,
                    ha="left",
                    va="center",
                )
            labels.append(_condition_label(medium, row.get("freq_label")))
        ax_onset.axvline(100.0, color=GRAY, linestyle=":", linewidth=1.2)
        ax_onset.set_yticks(y, labels)
        ax_onset.set_ylim(len(onset_rows) - 0.45, -0.75)
        ax_onset.set_xlim(75, 143)
        # 100 and 105 collide at this width; drop to a readable spacing.
        ax_onset.set_xticks([75, 90, 105, 120, 135])
        ax_onset.set_xlabel("Commanded voltage level (%)")
        ax_onset.grid(axis="y", visible=False)
    else:
        _empty_panel(ax_onset, "No saved onset table was available.")

    direct_offsets = {
        ("argon", "4 kHz"): (8, 7),
        ("water", "4 kHz"): (8, -15),
        ("water", "10 kHz"): (8, 8),
        ("manganese", "20 kHz"): (-10, 10),
    }
    for (medium, freq_label), rows in sorted(
        charge_groups.items(),
        key=lambda item: (_freq_number(item[0][1]), _medium_label(item[0][0])),
    ):
        ordered = sorted(rows, key=lambda row: _number(row.get("level_pct")) or 0)
        x = np.asarray([_number(row.get("Vamp_kV_median")) for row in ordered], dtype=float)
        stats = [_row_stat(row, "dQ_cycle_nC") for row in ordered]
        valid_stats = [stat for stat in stats if stat is not None]
        if not valid_stats or len(valid_stats) != len(x):
            continue
        y = np.asarray([stat.value for stat in valid_stats], dtype=float)
        lower = np.asarray(
            [
                max(0.0, stat.value - (stat.low if stat.low is not None else stat.value))
                for stat in valid_stats
            ]
        )
        upper = np.asarray(
            [
                max(0.0, (stat.high if stat.high is not None else stat.value) - stat.value)
                for stat in valid_stats
            ]
        )
        color = _medium_color(medium)
        marker = "D" if freq_label == "10 kHz" else "o"
        ax_charge.plot(x, y, color=color, linewidth=1.8, alpha=0.82)
        ax_charge.errorbar(
            x,
            y,
            yerr=np.vstack([lower, upper]),
            fmt=marker,
            color=color,
            ecolor=color,
            capsize=3,
            markersize=6.5,
            zorder=5,
        )
        offset = direct_offsets.get((medium, freq_label), (8, 0))
        ax_charge.annotate(
            _condition_label(medium, freq_label),
            (x[-1], y[-1]),
            xytext=offset,
            textcoords="offset points",
            color=color,
            ha="right" if offset[0] < 0 else "left",
            va="center",
        )
    ax_charge.set_xlabel("Measured voltage amplitude (kV)")
    ax_charge.set_ylabel("Raw ΔQ per carrier cycle (nC)")
    ax_charge.set_xlim(1.2, 4.9)
    ax_charge.set_ylim(bottom=0)
    ax_charge.margins(y=0.16)
    ax_charge.text(
        0.98,
        0.97,
        "carrier-transfer values\nno F correction",
        transform=ax_charge.transAxes,
        color=RED,
        ha="right",
        va="top",
        fontweight="bold",
    )

    stem = FIGURE_STEMS[3]
    png, pdf = _save(fig, data.root, stem)
    onset_positive = [
        _condition_label(row.get("medium"), row.get("freq_label"))
        for row in onset_rows
        if _number(row.get("onset_level_pct")) is not None
    ]
    not_detected = len(onset_rows) - len(onset_positive)
    mn_rows = charge_groups.get(("manganese", "20 kHz"), [])
    mn_values = ", ".join(
        f"{_number(row.get('level_pct')):g}%: "
        f"{_number(row.get('dQ_cycle_nC_median')):.1f} nC"
        for row in sorted(mn_rows, key=lambda row: _number(row.get("level_pct")) or 0)
        if _number(row.get("level_pct")) is not None
        and _number(row.get("dQ_cycle_nC_median")) is not None
    )
    caption = _write_caption(
        data.root,
        stem,
        "Figure 3 — Objective onset separates detected ladders from carrier-transfer charge trends",
        (
            f"Four ladder conditions first pass the objective onset gate at 100%, while "
            f"{not_detected} conditions remain undetected through 115%."
        ),
        (
            f"Onset-positive conditions: {', '.join(onset_positive)}.",
            f"Mn-nitrate raw ΔQ progression: {mn_values}.",
            (
                "Panel B shows raw medians and capture-level 2.5–97.5 percentile "
                "intervals only; no same-band F correction is applied."
            ),
            "Source: discharge_onset.csv and voltage-ladder rows in discharge_metrics.csv.",
        ),
    )
    return png, pdf, caption


def _figure4_synthesis_dose(data: Inputs) -> tuple[Path, Path, Path]:
    by_run = {row.get("run_key", ""): row for row in data.synthesis}
    selected = [by_run[key] for key in DECLARED_SYNTHESIS_RUNS if key in by_run]
    valid = [
        row
        for row in selected
        if _row_stat(row, "dose_20min_C") is not None
        and (_integer(row.get("N_capture")) or 0) > 0
    ]
    invalid = [row for row in selected if row not in valid]
    valid.sort(
        key=lambda row: (_row_stat(row, "dose_20min_C") or Stat(0, None, None, None)).value,
        reverse=True,
    )
    invalid.sort(key=lambda row: RUN_SHORT_LABELS.get(row.get("run_key", ""), ""))
    ordered = valid + invalid

    fig, ax = plt.subplots(figsize=FIGURE_SIZE_IN, layout="constrained")
    _scope_tag(fig, "synthesis")
    if not ordered:
        _empty_panel(ax, "No declared synthesis rows were available.")
    else:
        y = np.arange(len(ordered), dtype=float)
        labels = [RUN_SHORT_LABELS.get(row.get("run_key", ""), row.get("label", "")) for row in ordered]
        xmax = 65.0
        for idx, row in enumerate(ordered):
            run = row.get("run_key", "")
            dose = _row_stat(row, "dose_20min_C")
            rate = _row_stat(row, "gross_rate_C_min")
            n = _integer(row.get("N_capture")) or 0
            if dose is not None and n > 0:
                color = RUN_COLORS.get(run, BLUE)
                frequency_warning = "frequency_crosscheck_failed" in row.get("qc_status", "")
                ax.errorbar(
                    dose.value,
                    idx,
                    xerr=_x_error(dose),
                    fmt="o",
                    color=color,
                    ecolor=color,
                    capsize=4,
                    markersize=8,
                    elinewidth=1.8,
                    markerfacecolor="white" if frequency_warning else color,
                    markeredgecolor=color,
                    markeredgewidth=1.8,
                    zorder=5,
                )
                warning = "†" if frequency_warning else ""
                rate_text = f"{rate.value:.2f} C/min" if rate else "rate unavailable"
                ax.text(
                    dose.value + 1.5,
                    idx,
                    f"{dose.value:.1f} C · {rate_text} · N={n}{warning}",
                    color=color,
                    ha="left",
                    va="center",
                )
                xmax = max(xmax, (dose.high or dose.value) + 12)
            else:
                ax.scatter(1.0, idx, marker="x", color=RED, s=85, linewidth=2.4, zorder=5)
                ax.text(
                    3.0,
                    idx,
                    "withheld — Channel D clipped",
                    color=RED,
                    ha="left",
                    va="center",
                    fontweight="bold",
                )
        ax.set_yticks(y, labels)
        ax.set_ylim(len(ordered) - 0.45, -1.55)
        ax.set_xlim(0, xmax)
        ax.set_xlabel("Delivered charge over 20 min (C)")
        ax.grid(axis="y", visible=False)
        ax.text(
            0.985,
            0.16,
            "† hollow marker = frequency cross-check warning retained",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            color=RED,
        )
        ax.text(
            0.985,
            0.08,
            "All seven synthesis runs in the saved handoff tables",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            color=GRAY,
        )
        ax.text(
            0.985,
            0.975,
            (
                "electrical charge, both polarities summed;\n"
                "not electron-specific, not Faradaic"
            ),
            transform=ax.transAxes,
            ha="right",
            va="top",
            color=RED,
            fontweight="bold",
        )

    stem = FIGURE_STEMS[4]
    png, pdf = _save(fig, data.root, stem)
    value_text = "; ".join(
        (
            f"{RUN_SHORT_LABELS.get(row.get('run_key', ''), row.get('label', ''))}: "
            f"{_row_stat(row, 'dose_20min_C').value:.2f} C "
            f"({_row_stat(row, 'gross_rate_C_min').value:.3f} C/min)"
        )
        for row in valid
        if _row_stat(row, "dose_20min_C") and _row_stat(row, "gross_rate_C_min")
    )
    withheld_text = ", ".join(
        RUN_SHORT_LABELS.get(row.get("run_key", ""), row.get("label", "")) for row in invalid
    )
    caption = _write_caption(
        data.root,
        stem,
        "Figure 4 — Electrical dose ranks the usable synthesis runs",
        (
            "Among the seven saved synthesis runs, five have reportable 20-min "
            "charge dose and two are withheld because every Channel-D capture clipped."
        ),
        (
            f"Reportable dose (rate): {value_text}.",
            f"Withheld: {withheld_text}.",
            (
                "†Cu–Ni has 2/64 carrier cross-check disagreements; Pd/H₂ run 1 and "
                "the 7_21 1:6 betaine:1,2-PD run each have 64/64. Their doses remain "
                "shown with hollow markers and the warning retained."
            ),
            "Source: synthesis_charge.csv; intervals are 2.5–97.5 capture percentiles.",
        ),
    )
    return png, pdf, caption


def _representative_manifest_row(
    data: Inputs,
    *,
    level: float,
    segment: int | None,
) -> Mapping[str, str] | None:
    candidates = [
        row
        for row in data.manifest
        if row.get("dataset_type") == "voltage_ladder"
        and _canonical_medium(row.get("medium")) == "argon"
        and row.get("freq_label") == "4 kHz"
        and _number(row.get("level_pct")) == level
    ]
    if segment is not None:
        exact = [row for row in candidates if _integer(row.get("seg_idx")) == segment]
        if exact:
            return exact[0]
    return candidates[0] if candidates else None


def _figure5_waveform_method(data: Inputs) -> tuple[Path, Path, Path]:
    prefix = "argon4k_115"
    required = (f"{prefix}_time_s", f"{prefix}_V_kV", f"{prefix}_Q_nC")
    fig, axes = plt.subplots(
        2,
        1,
        figsize=FIGURE_SIZE_IN,
        layout="constrained",
        gridspec_kw={"height_ratios": [1.0, 1.05]},
    )
    ax_overview, ax_zoom = axes
    _scope_tag(fig, "voltage-ladder")
    _panel_label(ax_overview, "A  Burst-scale overview")
    _panel_label(ax_zoom, "B  Three carrier cycles")

    path_text = "representative trace unavailable"
    f0 = _number(data.figure_data.get(f"{prefix}_f0_Hz"))
    burst = _number(data.figure_data.get(f"{prefix}_burst_Hz"))
    offset = _number(data.figure_data.get(f"{prefix}_dc_offset_V"))
    segment = _integer(data.figure_data.get(f"{prefix}_seg"))
    manifest_row = _representative_manifest_row(data, level=115.0, segment=segment)
    if manifest_row:
        path_text = manifest_row.get("path", path_text)
    codes_a = _integer(manifest_row.get("codesA")) if manifest_row else None
    codes_d = _integer(manifest_row.get("codesD")) if manifest_row else None
    segment_name = str(path_text).replace("\\", "/").rsplit("/", 1)[-1]
    short_path = f".../{segment_name}"

    if all(key in data.figure_data for key in required) and f0 and burst:
        t = np.asarray(data.figure_data[f"{prefix}_time_s"], dtype=float)
        voltage = np.asarray(data.figure_data[f"{prefix}_V_kV"], dtype=float)
        charge = np.asarray(data.figure_data[f"{prefix}_Q_nC"], dtype=float)
        finite = np.isfinite(t) & np.isfinite(voltage) & np.isfinite(charge)
        t, voltage, charge = t[finite], voltage[finite], charge[finite]
        center = float(t[int(np.argmax(np.abs(voltage)))])
        overview_half_width = max(2.5 / burst, 0.00055)
        overview_mask = (t >= center - overview_half_width) & (t <= center + overview_half_width)
        if not np.any(overview_mask):
            overview_mask = np.ones_like(t, dtype=bool)
        overview_start = float(t[overview_mask][0])
        overview_t_ms = (t[overview_mask] - overview_start) * 1000.0
        ax_overview.plot(overview_t_ms, voltage[overview_mask], color=BLUE, linewidth=1.2)
        ax_overview.set_ylabel("Channel A voltage (kV)", color=BLUE)
        ax_overview.tick_params(axis="y", colors=BLUE)
        ax_overview.set_xlabel("Time within representative window (ms)")
        overview_charge = ax_overview.twinx()
        overview_charge.spines["right"].set_visible(True)
        overview_charge.plot(
            overview_t_ms,
            charge[overview_mask],
            color=RED,
            linewidth=1.0,
            alpha=0.7,
        )
        overview_charge.set_ylabel("Channel D charge (nC)", color=RED)
        overview_charge.tick_params(axis="y", colors=RED)
        overview_charge.grid(False)
        ax_overview.text(
            0.985,
            0.92,
            "voltage",
            transform=ax_overview.transAxes,
            color=BLUE,
            ha="right",
            va="top",
        )
        ax_overview.text(
            0.985,
            0.82,
            "charge",
            transform=ax_overview.transAxes,
            color=RED,
            ha="right",
            va="top",
        )

        zoom_half_width = 1.5 / f0
        zoom_mask = (t >= center - zoom_half_width) & (t <= center + zoom_half_width)
        zoom_start = float(t[zoom_mask][0]) if np.any(zoom_mask) else center
        zoom_t_us = (t[zoom_mask] - zoom_start) * 1e6
        ax_zoom.plot(
            zoom_t_us,
            voltage[zoom_mask],
            color=BLUE,
            marker="o",
            markersize=3.5,
            linewidth=1.8,
        )
        ax_zoom.set_xlabel("Time within three-cycle window (µs)")
        ax_zoom.set_ylabel("Channel A voltage (kV)", color=BLUE)
        ax_zoom.tick_params(axis="y", colors=BLUE)
        zoom_charge = ax_zoom.twinx()
        zoom_charge.spines["right"].set_visible(True)
        zoom_charge.plot(
            zoom_t_us,
            charge[zoom_mask],
            color=RED,
            marker="o",
            markersize=3.2,
            linewidth=1.6,
        )
        zoom_charge.set_ylabel("Channel D charge (nC)", color=RED)
        zoom_charge.tick_params(axis="y", colors=RED)
        zoom_charge.grid(False)
        ax_zoom.text(
            0.99,
            0.93,
            "voltage",
            transform=ax_zoom.transAxes,
            color=BLUE,
            ha="right",
            va="top",
        )
        ax_zoom.text(
            0.99,
            0.82,
            "charge",
            transform=ax_zoom.transAxes,
            color=RED,
            ha="right",
            va="top",
        )
        ax_zoom.text(
            0.018,
            0.08,
            (
                f"segment: {short_path}\n"
                f"f0 = {f0 / 1000.0:.2f} kHz  ·  burst = {burst / 1000.0:.3f} kHz  ·  "
                f"ADC codes A/D = {codes_a}/{codes_d}\n"
                f"Channel-D offset = {_fmt(offset, 3)} V"
            ),
            transform=ax_zoom.transAxes,
            ha="left",
            va="bottom",
            color=BLACK,
            bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": LIGHT_GRAY},
        )
    else:
        _empty_panel(ax_overview, "The saved 115% representative waveform is unavailable.")
        _empty_panel(ax_zoom, "The saved 115% representative waveform is unavailable.")

    stem = FIGURE_STEMS[5]
    png, pdf = _save(fig, data.root, stem)
    caption = _write_caption(
        data.root,
        stem,
        "Figure 5 — Representative waveform used for the carrier-cycle method",
        (
            "A 4 kHz burst argon record contains a measured ≈128-kHz carrier inside "
            "the burst envelope; the zoom exposes the samples used for cycle-wise V–Q work."
        ),
        (
            (
                f"Representative values: f0={_fmt((f0 or math.nan) / 1000.0, 2)} kHz, "
                f"burst={_fmt((burst or math.nan) / 1000.0, 3)} kHz, "
                f"DC offset={_fmt(offset, 3)} V, ADC codes A/D={codes_a}/{codes_d}."
            ),
            f"Saved representative segment: `{path_text}`.",
            "Source: figure_data.npz plus the matching manifest.csv row; no reprocessing.",
        ),
    )
    return png, pdf, caption


def _figure6_qv_quantization(data: Inputs) -> tuple[Path, Path, Path]:
    fig, axes = plt.subplots(
        1,
        2,
        figsize=FIGURE_SIZE_IN,
        sharex=True,
        sharey=True,
        layout="constrained",
    )
    levels = (60, 115)
    _scope_tag(fig, "voltage-ladder")
    manifests: dict[int, Mapping[str, str] | None] = {}
    for ax, level, panel in zip(axes, levels, ("A", "B")):
        prefix = f"argon4k_{level}"
        segment = _integer(data.figure_data.get(f"{prefix}_seg"))
        manifest_row = _representative_manifest_row(
            data,
            level=float(level),
            segment=segment,
        )
        manifests[level] = manifest_row
        v_key, q_key = f"{prefix}_loop_V_kV", f"{prefix}_loop_Q_nC"
        _panel_label(ax, f"{panel}  Argon · 4 kHz burst · {level}%")
        if v_key not in data.figure_data or q_key not in data.figure_data:
            _empty_panel(ax, "Representative Q–V loop unavailable.")
            continue
        voltage = np.asarray(data.figure_data[v_key], dtype=float)
        charge = np.asarray(data.figure_data[q_key], dtype=float)
        valid = np.isfinite(voltage) & np.isfinite(charge)
        voltage, charge = voltage[valid], charge[valid]
        ax.plot(voltage, charge, color=BLUE, linewidth=2.2, alpha=0.9)
        ax.scatter(
            voltage[::4],
            charge[::4],
            s=13,
            facecolor="white",
            edgecolor=BLUE,
            linewidth=0.8,
            alpha=0.75,
            zorder=4,
        )
        ax.axhline(0, color=GRAY, linewidth=0.8)
        ax.axvline(0, color=GRAY, linewidth=0.8)
        codes_d = _integer(manifest_row.get("codesD")) if manifest_row else None
        if level == 60:
            annotation = (
                f"Channel D = {codes_d} ADC codes\n"
                "geometric slope withheld\n"
                "quantization-limited"
            )
            annotation_color = RED
        else:
            row = next(
                (
                    item
                    for item in data.discharge
                    if item.get("dataset_type") == "voltage_ladder"
                    and _canonical_medium(item.get("medium")) == "argon"
                    and item.get("freq_label") == "4 kHz"
                    and _number(item.get("level_pct")) == 115.0
                ),
                None,
            )
            dq = _row_stat(row, "dQ_cycle_nC") if row else None
            annotation = (
                f"Channel D = {codes_d} ADC codes\n"
                f"raw ΔQ = {_fmt(dq.value if dq else None, 1)} nC/cycle\n"
                "[carrier-transfer]"
            )
            annotation_color = BLUE
        ax.text(
            0.035,
            0.95,
            annotation,
            transform=ax.transAxes,
            ha="left",
            va="top",
            color=annotation_color,
            fontweight="bold",
            bbox={"boxstyle": "round,pad=0.4", "facecolor": "white", "edgecolor": annotation_color},
        )
        ax.set_xlabel("Channel A voltage (kV)")
    axes[0].set_ylabel("Channel D charge (nC)")
    axes[0].set_xlim(-0.9, 0.9)
    axes[0].set_ylim(-22, 22)

    stem = FIGURE_STEMS[6]
    png, pdf = _save(fig, data.root, stem)
    paths = {
        level: (manifests[level].get("path", "unavailable") if manifests[level] else "unavailable")
        for level in levels
    }
    caption = _write_caption(
        data.root,
        stem,
        "Figure 6 — Quantization controls which Q–V claims are eligible",
        (
            "The 60% representative loop spans only seven Channel-D codes and cannot "
            "support a geometric slope, whereas the 115% record passes the code gate "
            "for raw carrier-cycle charge."
        ),
        (
            "At 115%, raw ΔQ is 25.6 nC per carrier cycle [carrier-transfer]; no F correction is shown.",
            f"60% representative segment: `{paths[60]}`.",
            f"115% representative segment: `{paths[115]}`.",
            "Source: figure_data.npz, manifest.csv, and discharge_metrics.csv.",
        ),
    )
    return png, pdf, caption


def _energy_sort_key(row: Mapping[str, str]) -> tuple[int, float, float]:
    medium = _canonical_medium(row.get("medium"))
    priority = {"manganese": 0, "water": 1, "argon": 2}.get(medium, 3)
    return priority, _freq_number(row.get("freq_label")), _number(row.get("level_pct")) or 0


def _figure7_energy_eligibility(data: Inputs) -> tuple[Path, Path, Path]:
    rows = sorted(
        [
            row
            for row in data.discharge
            if row.get("dataset_type") == "voltage_ladder"
            and _number(row.get("level_pct")) is not None
        ],
        key=_energy_sort_key,
    )
    fig, ax = plt.subplots(figsize=FIGURE_SIZE_IN, layout="constrained")
    _scope_tag(fig, "voltage-ladder")
    if not rows:
        _empty_panel(ax, "No voltage-ladder energy eligibility rows were available.")
    else:
        y = np.arange(len(rows), dtype=float)
        labels = [
            _condition_label(row.get("medium"), row.get("freq_label"), row.get("level_pct"))
            for row in rows
        ]
        valid_count = 0
        limited_count = 0
        withheld_count = 0
        for idx, row in enumerate(rows):
            power = _row_stat(row, "P_W")
            energy = _row_stat(row, "U_burst_uJ")
            orientation = row.get("orientation_status", "")
            if power is not None and energy is not None and (power.n or 0) > 0:
                valid_count += 1
                limited = (power.n or 0) <= 1 or (
                    "failed_negative" in orientation and "passed_nonnegative" in orientation
                )
                if limited:
                    limited_count += 1
                color = ORANGE if limited else GREEN
                marker = "D" if limited else "o"
                ax.errorbar(
                    power.value,
                    idx,
                    xerr=_x_error(power),
                    fmt=marker,
                    color=color,
                    ecolor=color,
                    capsize=3,
                    markersize=7,
                    elinewidth=1.6,
                    zorder=5,
                )
                qualifier = " · limited" if limited else ""
                ax.text(
                    max(3.0, power.value + 1.8),
                    idx,
                    f"P={power.value:.1f} W · U={energy.value / 1000.0:.2f} mJ · "
                    f"N={power.n}{qualifier}",
                    color=color,
                    ha="left",
                    va="center",
                )
            else:
                withheld_count += 1
                ax.scatter(0.0, idx, marker="x", color=RED, s=76, linewidth=2.2, zorder=5)
                ax.text(
                    2.4,
                    idx,
                    "withheld — negative median energy",
                    color=RED,
                    ha="left",
                    va="center",
                )
        ax.set_yticks(y, labels)
        ax.set_ylim(len(rows) - 0.45, -0.65)
        ax.set_xlim(-4, 115)
        ax.set_xlabel("Eligible burst-average power (W)")
        ax.grid(axis="y", visible=False)
        ax.text(
            0.985,
            0.96,
            f"{valid_count - limited_count} robust · {limited_count} N=1/mixed · "
            f"{withheld_count} withheld",
            transform=ax.transAxes,
            ha="right",
            va="top",
            color=BLACK,
            fontweight="bold",
        )

    valid_rows = [
        row
        for row in rows
        if _row_stat(row, "P_W") is not None and (_row_stat(row, "P_W").n or 0) > 0
    ]
    invalid_rows = [row for row in rows if row not in valid_rows]
    valid_text = "; ".join(
        (
            f"{_condition_label(row.get('medium'), row.get('freq_label'), row.get('level_pct'))}: "
            f"{_row_stat(row, 'U_burst_uJ').value / 1000.0:.2f} mJ, "
            f"{_row_stat(row, 'P_W').value:.1f} W, N={_row_stat(row, 'P_W').n}"
        )
        for row in valid_rows
        if _row_stat(row, "U_burst_uJ") and _row_stat(row, "P_W")
    )
    stem = FIGURE_STEMS[7]
    png, pdf = _save(fig, data.root, stem)
    caption = _write_caption(
        data.root,
        stem,
        "Figure 7 — Energy and power are reported only after the orientation gate",
        (
            f"{len(valid_rows)} onset-positive ladder conditions have at least one "
            f"nonnegative-energy capture; {len(invalid_rows)} are withheld."
        ),
        (
            f"Eligible condition summaries: {valid_text}.",
            (
                "Pure water · 4 kHz · 115% is marked limited because only one capture "
                "passes the energy-orientation gate."
            ),
            (
                "Red crosses are not zero energy: their burst energy and power are "
                "withheld after a negative median orientation result."
            ),
            "Source: voltage-ladder rows in discharge_metrics.csv.",
        ),
    )
    return png, pdf, caption


def _fraction_color(fraction: float, *, positive: bool = True) -> str:
    if positive:
        if fraction >= 0.95:
            return "#BDE4D4"
        if fraction > 0:
            return "#FBE4AF"
        return "#F4C5B6"
    if fraction <= 0:
        return "#BDE4D4"
    if fraction < 0.1:
        return "#FBE4AF"
    return "#F4C5B6"


def _figure8_qc_gate_summary(data: Inputs) -> tuple[Path, Path, Path]:
    manifest_total = len(data.manifest)
    unexcluded = [row for row in data.manifest if not _truthy(row.get("excluded"))]
    unexcluded_total = len(unexcluded)
    verified = sum(
        str(row.get("frequency_status") or "").startswith("verified_") for row in unexcluded
    )
    quant_limited = sum(
        row.get("quantization_status") == "quantization_limited" for row in data.per_capture
    )
    clipped_d = sum((_number(row.get("clipD")) or 0) > 0 for row in data.manifest)
    excluded = sum(_truthy(row.get("excluded")) for row in data.manifest)
    energy_failed = sum(
        "failed_negative_median_energy" in row.get("orientation_status", "")
        for row in data.per_capture
        if not _truthy(row.get("excluded"))
    )
    retained_charge_eligible = sum(
        _number(row.get("retained_charge_nC")) is not None for row in data.per_capture
    )
    ill_defined = sum(
        str(row.get("frequency_status") or "") == "carrier_ill_defined_short_ring"
        for row in unexcluded
    )
    metrics = (
        ("Frequency cross-check verified", verified, unexcluded_total, GREEN),
        (
            "Carrier ill-defined (short ring)\n"
            "physical regime, not a QC failure",
            ill_defined,
            unexcluded_total,
            ORANGE,
        ),
        ("Charge resolution limited", quant_limited, manifest_total, ORANGE),
        ("Channel-D clipping", clipped_d, manifest_total, RED),
        ("Explicit exclusion", excluded, manifest_total, RED),
        ("Energy orientation failed", energy_failed, unexcluded_total, RED),
        (
            "Quiet-edge gate\n"
            f"(net/retained charge identifiable): {retained_charge_eligible:,}/"
            f"{manifest_total:,}",
            retained_charge_eligible,
            manifest_total,
            RED,
        ),
    )

    fig, axes = plt.subplots(
        1,
        2,
        figsize=FIGURE_SIZE_IN,
        layout="constrained",
        gridspec_kw={"width_ratios": [0.9, 1.35]},
    )
    ax_gate, ax_matrix = axes
    _scope_tag(
        fig,
        "voltage-ladder + operational (Lissajousfigure) + synthesis",
    )
    _panel_label(ax_gate, "A  Independent QC gates")
    ax_matrix.set_title(
        "B  Synthesis-run usability",
        loc="left",
        fontweight="bold",
        pad=22,
    )

    y = np.arange(len(metrics), dtype=float)
    values = [100.0 * count / denom if denom else 0.0 for _, count, denom, _ in metrics]
    colors = [color for _, _, _, color in metrics]
    ax_gate.barh(y, values, color=colors, alpha=0.82, height=0.58)
    for idx, ((_, count, denom, color), value) in enumerate(zip(metrics, values)):
        ax_gate.text(
            min(98.0, max(value + 1.5, 8.5)),
            idx,
            f"{count:,}/{denom:,} · {value:.1f}%",
            color=color,
            ha="left",
            va="center",
            fontweight="bold",
        )
    ax_gate.set_yticks(y, [name for name, _, _, _ in metrics])
    ax_gate.invert_yaxis()
    ax_gate.set_xlim(0, 112)
    ax_gate.set_xlabel("Share of stated denominator (%)")
    ax_gate.grid(axis="y", visible=False)
    # Footer notes live in figure coordinates. Hanging them below the axes in
    # axes coordinates makes constrained_layout shrink the axes to nothing and
    # abandon the layout, which unplaces every artist in both panels.
    fig.get_layout_engine().set(rect=(0.0, 0.11, 1.0, 0.852))
    fig.text(
        0.012,
        0.082,
        "Categories overlap; bar lengths are not additive.",
        ha="left",
        va="bottom",
        color=GRAY,
    )
    fig.text(
        0.012,
        0.046,
        (
            f"Denominators: {manifest_total:,} parsed captures (resolution, clipping, "
            f"exclusion, quiet-edge); {unexcluded_total:,} non-excluded "
            "(frequency, energy orientation)."
        ),
        ha="left",
        va="bottom",
        color=GRAY,
        fontsize=10,
    )
    fig.text(
        0.012,
        0.012,
        (
            "No record contains both quiet pre- and post-burst edges; "
            "retained_charge_nC N = 0."
        ),
        ha="left",
        va="bottom",
        color=RED,
        fontsize=10,
        fontweight="bold",
    )

    manifest_by_run: dict[str, list[Mapping[str, str]]] = defaultdict(list)
    for row in data.manifest:
        if row.get("run_key") in DECLARED_SYNTHESIS_RUNS:
            manifest_by_run[row.get("run_key", "")].append(row)
    synthesis_by_run = {row.get("run_key", ""): row for row in data.synthesis}
    run_rows = list(DECLARED_SYNTHESIS_RUNS)
    columns = ("Channel D\nunclipped", "f0\ncross-check", "20-min\ndose")
    ax_matrix.set_xlim(0, len(columns))
    ax_matrix.set_ylim(0, len(run_rows))
    for row_idx, run in enumerate(run_rows):
        captures = manifest_by_run.get(run, [])
        total = len(captures)
        unclipped = sum((_number(row.get("clipD")) or 0) == 0 for row in captures)
        freq_ok = sum(
            str(row.get("frequency_status") or "").startswith("verified_") for row in captures
        )
        synthesis_row = synthesis_by_run.get(run, {})
        dose_reported = _row_stat(synthesis_row, "dose_20min_C") is not None
        cells = (
            (unclipped / total if total else 0.0, f"{unclipped}/{total}", True),
            (freq_ok / total if total else 0.0, f"{freq_ok}/{total}", True),
            (1.0 if dose_reported else 0.0, "reported" if dose_reported else "withheld", True),
        )
        plot_y = len(run_rows) - row_idx - 1
        for col_idx, (fraction, text, positive) in enumerate(cells):
            ax_matrix.add_patch(
                Rectangle(
                    (col_idx + 0.04, plot_y + 0.08),
                    0.92,
                    0.84,
                    facecolor=_fraction_color(fraction, positive=positive),
                    edgecolor="white",
                    linewidth=2.0,
                )
            )
            ax_matrix.text(
                col_idx + 0.5,
                plot_y + 0.5,
                text,
                ha="center",
                va="center",
                color=BLACK if fraction > 0 else RED,
                fontweight="bold",
            )
    ax_matrix.set_xticks(
        np.arange(len(columns), dtype=float) + 0.5,
        columns,
        rotation=0,
    )
    ax_matrix.xaxis.tick_top()
    ax_matrix.set_yticks(
        np.arange(len(run_rows), dtype=float) + 0.5,
        [RUN_SHORT_LABELS[run] for run in reversed(run_rows)],
    )
    ax_matrix.tick_params(axis="both", length=0)
    ax_matrix.tick_params(axis="x", pad=5)
    ax_matrix.grid(False)
    for spine in ax_matrix.spines.values():
        spine.set_visible(False)
    ax_matrix.text(
        1.0,
        -0.12,
        "Counts are capture-level; dose is a run-level output gate.",
        transform=ax_matrix.transAxes,
        ha="right",
        va="top",
        color=GRAY,
    )

    stem = FIGURE_STEMS[8]
    png, pdf = _save(fig, data.root, stem)
    caption = _write_caption(
        data.root,
        stem,
        "Figure 8 — Independent QC gates determine which electrical claims survive",
        (
            "Frequency verification, charge resolution, clipping, exclusion, energy "
            "orientation, and quiet-edge eligibility remove different subsets; their "
            "percentages must not be added."
        ),
        (
            (
                f"Frequency verified: {verified:,}/{unexcluded_total:,}; charge-resolution "
                f"limited: {quant_limited:,}/{manifest_total:,}; Channel-D clipped: "
                f"{clipped_d:,}/{manifest_total:,}."
            ),
            (
                f"Explicitly excluded: {excluded:,}/{manifest_total:,}; energy orientation "
                f"failed: {energy_failed:,}/{unexcluded_total:,} non-excluded captures."
            ),
            (
                f"Retained charge is withheld for all captures: {retained_charge_eligible:,}/"
                f"{manifest_total:,} records have quiet edges, and retained_charge_nC has N=0."
            ),
            (
                "The seven-run matrix shows why a dose can remain reportable with a retained "
                "frequency warning, while all-clipped Channel-D runs are withheld."
            ),
            "Source: manifest.csv, per_capture_metrics.csv, and synthesis_charge.csv.",
        ),
    )
    return png, pdf, caption


def _write_core_manifest(root: Path) -> Path:
    """Write a compact index for the nine core presentation figures."""

    path = root / "presentation_core_manifest.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("figure_id", "png", "pdf", "caption", "legacy_model_dependent"),
        )
        writer.writeheader()
        for stem in FIGURE_STEMS:
            writer.writerow(
                {
                    "figure_id": stem.split("_", 1)[0],
                    "png": f"presentation_figures/{stem}.png",
                    "pdf": f"presentation_figures/{stem}.pdf",
                    "caption": f"presentation_captions/{stem}.md",
                    "legacy_model_dependent": "false",
                }
            )
    return path


def build_presentation_figures(root: Path) -> list[Path]:
    """Build all nine PNG/PDF figure pairs and their caption markdown files."""

    root = Path(root)
    _configure_style()
    _configure_medium_labels(root)
    data = _load_inputs(root)
    required = (
        root / "manifest.csv",
        root / "dispersion_master.csv",
        root / "discharge_metrics.csv",
        root / "synthesis_charge.csv",
    )
    missing = [path.name for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Presentation figures require existing quantified outputs; missing: "
            + ", ".join(missing)
        )
    for stale in (
        root / "presentation_figures" / "fig2_chain_dispersion.png",
        root / "presentation_figures" / "fig2_chain_dispersion.pdf",
        root / "presentation_captions" / "fig2_chain_dispersion.md",
    ):
        stale.unlink(missing_ok=True)
    outputs: list[Path] = []
    for builder in (
        _figure1_frequency_identity,
        _figure2_chain_dispersion,
        _figure3_onset_and_charge,
        _figure4_synthesis_dose,
        _figure5_waveform_method,
        _figure6_qv_quantization,
        _figure7_energy_eligibility,
        _figure8_qc_gate_summary,
    ):
        outputs.extend(builder(data))
    outputs.append(_write_core_manifest(root))
    return outputs


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Render the nine compact Lissajous presentation figures."
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("outputs/lissajous_v2"),
        help="Directory containing the existing Lissajous v2 CSV/NPZ outputs.",
    )
    args = parser.parse_args(argv)
    outputs = build_presentation_figures(args.out)
    for path in outputs:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
