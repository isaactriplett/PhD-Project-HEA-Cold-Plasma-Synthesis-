"""Generate supplementary presentation figures from existing analysis outputs.

This module is deliberately read-only with respect to the quantified tables.  It
does not re-run waveform analysis; it turns the current Lissajous-v2 and legacy
report CSVs into presentation-oriented comparison plots.

Example
-------
python -m lissajous.presentation_supplement --out outputs/lissajous_v2
"""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from matplotlib.lines import Line2D

from .config import load_config


FIGSIZE = (12.0, 6.75)
DPI = 200

LEGACY_MODEL_CANVAS = (
    "Legacy model: terminal half-cycle ΔQ − whole-waveform complex-C* endpoint "
    "background, scaled by Cd/(Cd − C′)."
)

OKABE_ITO = {
    "orange": "#E69F00",
    "sky": "#56B4E9",
    "green": "#009E73",
    "yellow": "#F0E442",
    "blue": "#0072B2",
    "vermillion": "#D55E00",
    "purple": "#CC79A7",
    "gray": "#6B6B6B",
    "light_gray": "#B8B8B8",
}

MATERIAL_ORDER = (
    "argon_only",
    "pure_water",
    "BMIM_nitrate",
    "5mM_Mn_nitrate_in_water",
)

MATERIAL_LABELS = {
    "argon_only": "Argon / no liquid",
    "pure_water": "Pure water",
    "BMIM_nitrate": "BMIM nitrate",
    "5mM_Mn_nitrate_in_water": "5 mM Mn nitrate",
}

CONFIG_MEDIUM_FOLDERS = {
    "argon_only": "argon",
    "pure_water": "pure water",
    "BMIM_nitrate": "ionic liquid",
    "5mM_Mn_nitrate_in_water": "manganese nitrate in water",
}

MATERIAL_COLORS = {
    "argon_only": OKABE_ITO["gray"],
    "pure_water": OKABE_ITO["blue"],
    "BMIM_nitrate": OKABE_ITO["purple"],
    "5mM_Mn_nitrate_in_water": OKABE_ITO["green"],
}

LEGACY_PANEL_IDS = {
    "argon_only": ("a", "argon_no_liquid"),
    "pure_water": ("b", "pure_water"),
    "BMIM_nitrate": ("c", "bmim_nitrate"),
    "5mM_Mn_nitrate_in_water": ("d", "mn_nitrate"),
}

FREQUENCY_COLORS = {
    4: OKABE_ITO["blue"],
    10: OKABE_ITO["green"],
    20: OKABE_ITO["vermillion"],
}

SYNTHESIS_ORDER = (
    "7_18/AgPd 5percent hydrogen",
    "7_19/1_3betaineEGAgCuNi-0003",
    "7_19/1_3betaineEGCuNi-0004",
    "7_19/1_3betaineEGPdHydrogen",
    "7_19/1_3betaineEGPdHydrogen-0002",
    "7_21/AgCuNi1_6betaine_12PD",
    "7_9/BMIMNTf2Pt20kHz",
)

SYNTHESIS_LABELS = {
    "7_18/AgPd 5percent hydrogen": "Ag-Pd, Ar/5% H₂",
    "7_19/1_3betaineEGAgCuNi-0003": "DES Ag-Cu-Ni",
    "7_19/1_3betaineEGCuNi-0004": "DES Cu-Ni",
    "7_19/1_3betaineEGPdHydrogen": "DES Pd/H₂, run 1",
    "7_19/1_3betaineEGPdHydrogen-0002": "DES Pd/H₂, run 2",
    "7_21/AgCuNi1_6betaine_12PD": "1:6 betaine:1,2-PD Ag-Cu-Ni",
    "7_9/BMIMNTf2Pt20kHz": "BMIM-NTf₂ + Pt",
}

SYNTHESIS_COLORS = {
    "7_18/AgPd 5percent hydrogen": OKABE_ITO["gray"],
    "7_19/1_3betaineEGAgCuNi-0003": OKABE_ITO["sky"],
    "7_19/1_3betaineEGCuNi-0004": OKABE_ITO["blue"],
    "7_19/1_3betaineEGPdHydrogen": OKABE_ITO["vermillion"],
    "7_19/1_3betaineEGPdHydrogen-0002": OKABE_ITO["orange"],
    "7_21/AgCuNi1_6betaine_12PD": OKABE_ITO["green"],
    "7_9/BMIMNTf2Pt20kHz": OKABE_ITO["purple"],
}

FIGURE_STEMS = {
    9: "fig09_postbreakdown_maximum_power",
    10: "fig10_duty_on_fraction",
    11: "fig11_synthesis_signed_halfcycle_flow",
    12: "fig12_synthesis_gross_charge_rate",
    13: "fig13_legacy_per_lobe_dose_response",
    14: "fig14_legacy_within_record_stationarity",
    15: "fig15_legacy_dose_clock",
}


def _configure_style() -> None:
    available = {font.name for font in font_manager.fontManager.ttflist}
    family = next(
        (
            candidate
            for candidate in ("Calibri", "Carlito", "DejaVu Sans")
            if candidate in available
        ),
        "DejaVu Sans",
    )
    matplotlib.rcParams.update(
        {
            "font.family": family,
            "font.size": 11,
            "axes.labelsize": 11,
            "axes.titlesize": 11,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "legend.fontsize": 11,
            "axes.linewidth": 0.8,
            "lines.linewidth": 1.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.dpi": DPI,
        }
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
            MATERIAL_LABELS[canonical] = display


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"Required input table is missing: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _number(row: Mapping[str, object], key: str) -> float | None:
    value = row.get(key)
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        parsed = float(text)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _integer(row: Mapping[str, object], key: str) -> int:
    value = _number(row, key)
    return int(round(value)) if value is not None else 0


def _truthy(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _stat(
    row: Mapping[str, object], base: str
) -> tuple[float, float | None, float | None, int] | None:
    median = _number(row, f"{base}_median")
    if median is None:
        median = _number(row, base)
    if median is None:
        return None
    low = _number(row, f"{base}_p2_5")
    high = _number(row, f"{base}_p97_5")
    n = _integer(row, f"{base}_N")
    return median, low, high, n


def _legacy_stat(
    row: Mapping[str, object],
) -> tuple[float, float | None, float | None]:
    value = _number(row, "estimate")
    if value is None:
        value = _number(row, "model_dependent_charge_nC")
    if value is None:
        raise ValueError("Legacy binned row has no charge estimate")
    low = _number(row, "ci_low")
    high = _number(row, "ci_high")
    return value, low, high


def _xerr(
    median: float, low: float | None, high: float | None
) -> np.ndarray | None:
    if low is None or high is None:
        return None
    return np.asarray([[max(0.0, median - low)], [max(0.0, high - median)]])


def _yerr(
    median: float, low: float | None, high: float | None
) -> np.ndarray | None:
    return _xerr(median, low, high)


def _finish_axis(ax: plt.Axes, grid_axis: str = "x") -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(
        True,
        axis=grid_axis,
        color="#D9D9D9",
        linewidth=0.7,
        alpha=0.75,
        zorder=0,
    )
    ax.set_axisbelow(True)


def _scope_tag(fig: plt.Figure, scope: str) -> None:
    """Print the source record set as a self-contained on-canvas subtitle."""

    # Reserve the strip the tag occupies so titles and secondary-axis labels
    # cannot be laid out underneath it.
    engine = fig.get_layout_engine()
    if engine is not None and hasattr(engine, "set"):
        try:
            rect = list(engine.get().get("rect", (0.0, 0.0, 1.0, 1.0)))
            top = rect[1] + rect[3]
            rect[3] = max(0.05, min(top, 0.955) - rect[1])
            engine.set(rect=tuple(rect))
        except Exception:  # pragma: no cover - layout engines without rect
            pass

    fig.text(
        0.5,
        0.995,
        f"Record set: {scope}",
        ha="center",
        va="top",
        color=OKABE_ITO["gray"],
        fontsize=10.5,
    )


def _legacy_model_tag(fig: plt.Figure, *, y: float = 0.895) -> None:
    fig.text(
        0.5,
        y,
        LEGACY_MODEL_CANVAS,
        ha="center",
        va="center",
        color="#333333",
        fontsize=9.8,
        bbox={
            "boxstyle": "round,pad=0.3",
            "facecolor": "white",
            "edgecolor": OKABE_ITO["light_gray"],
            "alpha": 0.96,
        },
    )


def _provenance_path(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def _reserve_bottom(fig: plt.Figure, height: float = 0.06) -> None:
    """Free a strip at the figure bottom for a full-width caveat line."""

    engine = fig.get_layout_engine()
    if engine is None or not hasattr(engine, "set"):
        return
    try:
        rect = list(engine.get().get("rect", (0.0, 0.0, 1.0, 1.0)))
        rect[1] = max(rect[1], height)
        rect[3] = max(0.05, rect[3] - height)
        engine.set(rect=tuple(rect))
    except Exception:  # pragma: no cover - layout engines without rect
        pass


def _save(
    fig: plt.Figure,
    root: Path,
    number: int,
    *,
    suffix: str = "",
    stem: str | None = None,
) -> list[Path]:
    figure_dir = root / "presentation_figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    output_stem = stem or FIGURE_STEMS[number]
    if suffix and stem is None:
        output_stem = output_stem.replace(
            f"fig{number:02d}",
            f"fig{number:02d}{suffix}",
            1,
        )
    png = figure_dir / f"{output_stem}.png"
    pdf = figure_dir / f"{output_stem}.pdf"
    fig.savefig(png, dpi=DPI, facecolor="white")
    fig.savefig(
        pdf,
        facecolor="white",
        metadata={"CreationDate": None, "ModDate": None},
    )
    plt.close(fig)
    return [png, pdf]


def _caption(
    root: Path,
    number: int,
    body: str,
    values: Iterable[str] = (),
    notes: Iterable[str] = (),
    provenance: Iterable[str] = (),
    *,
    suffix: str = "",
) -> Path:
    caption_dir = root / "presentation_captions"
    caption_dir.mkdir(parents=True, exist_ok=True)
    label = f"{number:02d}{suffix}"
    path = caption_dir / f"fig{label}_caption.md"
    lines = [f"# Figure {label} caption", "", body.strip()]
    values = [item.strip() for item in values if item.strip()]
    notes = [item.strip() for item in notes if item.strip()]
    provenance = [item.strip() for item in provenance if item.strip()]
    if values:
        lines.extend(["", "Values:", *[f"- {item}" for item in values]])
    if notes:
        lines.extend(["", "Notes:", *[f"- {item}" for item in notes]])
    if provenance:
        lines.extend(
            ["", "Provenance:", *[f"- `{item}`" for item in provenance]]
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _frequency_from_label(label: str) -> int | None:
    text = label.lower().replace(" ", "")
    for value in (4, 10, 20):
        if f"{value}khz" in text:
            return value
    return None


def _ordered_synthesis(
    rows: Sequence[Mapping[str, object]],
) -> list[Mapping[str, object]]:
    by_key = {str(row.get("run_key", "")): row for row in rows}
    ordered = [by_key[key] for key in SYNTHESIS_ORDER if key in by_key]
    ordered.extend(
        row
        for key, row in sorted(by_key.items())
        if key not in SYNTHESIS_ORDER
    )
    return ordered


def _synthesis_label(row: Mapping[str, object]) -> str:
    key = str(row.get("run_key", ""))
    return SYNTHESIS_LABELS.get(key, str(row.get("label", key)))


def _synthesis_color(row: Mapping[str, object]) -> str:
    key = str(row.get("run_key", ""))
    return SYNTHESIS_COLORS.get(key, OKABE_ITO["gray"])


def _condition_rows(
    discharge: Sequence[Mapping[str, object]],
) -> list[Mapping[str, object]]:
    rows = [
        row
        for row in discharge
        if str(row.get("dataset_type", "")) == "july7_8_operational"
        and not str(row.get("level_pct", "")).strip()
    ]

    def sort_key(row: Mapping[str, object]) -> tuple[int, int, str]:
        medium = str(row.get("medium", ""))
        try:
            medium_index = MATERIAL_ORDER.index(medium)
        except ValueError:
            medium_index = len(MATERIAL_ORDER)
        frequency = _frequency_from_label(str(row.get("freq_label", ""))) or 999
        return medium_index, frequency, str(row.get("cond", ""))

    return sorted(rows, key=sort_key)


def _condition_label(row: Mapping[str, object]) -> str:
    medium = str(row.get("medium", ""))
    material = MATERIAL_LABELS.get(medium, medium.replace("_", " "))
    frequency = str(row.get("freq_label", "")).strip() or "unlabelled"
    return f"{material} — {frequency} burst"


def make_fig09(
    root: Path,
    discharge: Sequence[Mapping[str, object]],
) -> tuple[list[Path], Path]:
    rows = _condition_rows(discharge)
    fig, ax = plt.subplots(figsize=FIGSIZE, constrained_layout=True)
    _scope_tag(fig, "operational (Lissajousfigure)")
    y_positions = np.arange(len(rows))[::-1]
    values: list[str] = []

    for y, row in zip(y_positions, rows):
        medium = str(row.get("medium", ""))
        color = MATERIAL_COLORS.get(medium, OKABE_ITO["gray"])
        power = _stat(row, "P_W")
        orientation = str(row.get("orientation_status", "")).strip()
        amplitude = _number(row, "Vamp_kV_median")
        label = _condition_label(row)
        if power is not None and orientation == "passed_nonnegative_energy":
            median, low, high, n = power
            ax.errorbar(
                median,
                y,
                xerr=_xerr(median, low, high),
                fmt="o",
                markersize=7,
                color=color,
                markeredgecolor=color,
                capsize=3,
                zorder=3,
            )
            annotation = f"{median:.3g} W"
            if amplitude is not None:
                annotation += f" · {amplitude:.3g} kV"
            annotation += f" · N={n}"
            ax.text(
                max(median * 1.10, 0.48),
                y,
                annotation,
                va="center",
                ha="left",
                fontsize=11,
            )
            values.append(f"{label}: {annotation}.")
        elif power is not None:
            median, low, high, n = power
            ax.errorbar(
                median,
                y,
                xerr=_xerr(median, low, high),
                fmt="D",
                markersize=7,
                markerfacecolor="white",
                markeredgecolor=OKABE_ITO["vermillion"],
                color=OKABE_ITO["vermillion"],
                capsize=3,
                zorder=3,
            )
            ax.text(
                max(median * 1.10, 0.48),
                y,
                f"{median:.3g} W · QC only · N={n}",
                va="center",
                ha="left",
                fontsize=11,
                color=OKABE_ITO["vermillion"],
            )
            values.append(
                f"{label}: {median:.3g} W, N={n}; shown only as QC because "
                f"orientation status is `{orientation}`."
            )
        else:
            ax.scatter(
                [0.015],
                [y],
                marker="x",
                s=65,
                linewidths=2,
                color=OKABE_ITO["vermillion"],
                transform=ax.get_yaxis_transform(),
                clip_on=False,
                zorder=3,
            )
            reason = orientation or "no valid burst-energy power"
            ax.text(
                0.038,
                y,
                f"withheld — {reason.replace('_', ' ')}",
                transform=ax.get_yaxis_transform(),
                va="center",
                ha="left",
                fontsize=11,
                color=OKABE_ITO["vermillion"],
            )
            values.append(f"{label}: power withheld ({reason}).")

    ax.set_yticks(y_positions, [_condition_label(row) for row in rows])
    ax.set_ylim(-0.55, len(rows) + 1.45)
    ax.set_xscale("log")
    ax.set_xlim(0.3, 260)
    ax.set_xlabel("Discharge power (W; log scale)")
    ax.set_ylabel("Selected operational condition")
    _finish_axis(ax, "x")
    ax.text(
        0.995,
        0.985,
        "Selection: highest-amplitude post-onset record per medium × burst",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=10.5,
        fontweight="bold",
        color="#333333",
    )
    _reserve_bottom(fig)
    fig.text(
        0.985,
        0.015,
        (
            "electrical burst-loop power at the measurement plane · "
            "includes series-chain† and gas-phase dissipation; not liquid heating · "
            "† order-of-magnitude chain context"
        ),
        ha="right",
        va="bottom",
        fontsize=9.5,
        color=OKABE_ITO["vermillion"],
    )

    for boundary in (2.5, 5.5, 8.5):
        if boundary < len(rows):
            ax.axhline(
                len(rows) - 1 - boundary,
                color="#BDBDBD",
                linewidth=0.8,
                zorder=1,
            )

    paths = _save(fig, root, 9)
    caption = _caption(
        root,
        9,
        (
            "For each medium × burst frequency, the highest-amplitude post-onset "
            "operational record is selected. Its discharge power is shown as a "
            "capture median with a 2.5–97.5% interval. Conditions with mixed "
            "energy orientation are explicitly marked QC-only; conditions with "
            "no non-negative burst-energy estimate are withheld."
        ),
        values=values,
        notes=[
            (
                "Power is electrical burst-loop power at the measurement plane; "
                "it includes series-chain† and gas-phase dissipation and is not "
                "liquid heating, Faradaic power, or plasma-only power."
            ),
            (
                "P_chain_est_W† = I_rms²R is tabulated with I from band-limited "
                "dQ/dt and R = 10 kΩ [7–15 kΩ]; † denotes an order-of-magnitude "
                "series-chain estimate."
            ),
            "Measured amplitudes are printed because conditions are not amplitude-matched.",
        ],
        provenance=["outputs/lissajous_v2/discharge_metrics.csv"],
    )
    return paths, caption


def make_fig09b(
    root: Path,
    synthesis: Sequence[Mapping[str, object]],
) -> tuple[list[Path], Path]:
    rows = _ordered_synthesis(synthesis)
    fig, ax = plt.subplots(figsize=FIGSIZE, constrained_layout=True)
    _scope_tag(fig, "synthesis")
    y_positions = np.arange(len(rows))[::-1]
    labels: list[str] = []
    values: list[str] = []

    for y, row in zip(y_positions, rows):
        label = _synthesis_label(row)
        qc = str(row.get("qc_status", ""))
        frequency_qc = "frequency_crosscheck_failed" in qc
        if frequency_qc:
            label += "†"
        labels.append(label)
        color = _synthesis_color(row)
        power = _stat(row, "P_W")
        if power is None or _integer(row, "N_capture") <= 0:
            ax.scatter(
                [0.008],
                [y],
                marker="x",
                s=70,
                linewidths=2,
                color=OKABE_ITO["vermillion"],
                transform=ax.get_yaxis_transform(),
                clip_on=False,
                zorder=3,
            )
            ax.text(
                0.025,
                y,
                "withheld — Channel D clipped",
                transform=ax.get_yaxis_transform(),
                va="center",
                ha="left",
                color=OKABE_ITO["vermillion"],
                fontsize=11,
            )
            values.append(
                f"{_synthesis_label(row)}: power withheld because Channel D was clipped."
            )
            continue

        median, low, high, n = power
        ax.errorbar(
            median,
            y,
            xerr=_xerr(median, low, high),
            fmt="o",
            markersize=7,
            color=color,
            markerfacecolor="white" if frequency_qc else color,
            markeredgecolor=color,
            capsize=3,
            zorder=3,
        )
        text_x = max(median, high or median) + 1.7
        ax.text(
            text_x,
            y,
            f"{median:.2f} W · N={n}",
            va="center",
            ha="left",
            fontsize=11,
        )
        values.append(
            f"{_synthesis_label(row)}: {median:.2f} W "
            f"[{low:.2f}, {high:.2f}] W, N={n}."
        )

    ax.set_xlim(0.0, 145.0)
    ax.set_ylim(-0.55, len(rows) + 1.35)
    ax.set_yticks(y_positions, labels)
    ax.set_xlabel("Electrical discharge power (W)")
    ax.set_ylabel("Synthesis run")
    _finish_axis(ax, "x")
    ax.text(
        0.995,
        0.86,
        "† Carrier cross-check failed (hollow marker)",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=11,
        color="#555555",
    )
    ax.text(
        0.995,
        0.985,
        (
            "electrical burst-loop power at the measurement plane\n"
            "includes series-chain† and gas-phase dissipation; not liquid heating\n"
            "† order-of-magnitude chain context"
        ),
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=10,
        color=OKABE_ITO["vermillion"],
    )

    paths = _save(
        fig,
        root,
        9,
        suffix="b",
        stem="fig09b_synthesis_discharge_power",
    )
    caption = _caption(
        root,
        9,
        (
            "Synthesis-run electrical discharge power is shown as the "
            "capture median with a 2.5–97.5% interval. Two runs with no "
            "unclipped Channel-D capture are explicitly withheld."
        ),
        values=values,
        notes=[
            (
                "Power is electrical burst-loop power at the measurement plane; "
                "it includes series-chain† and gas-phase dissipation and is not "
                "liquid heating, plasma-only power, or Faradaic power."
            ),
            (
                "P_chain_est_W† = I_rms²R is tabulated with I from band-limited "
                "dQ/dt and R = 10 kΩ [7–15 kΩ]; † denotes an order-of-magnitude "
                "series-chain estimate."
            ),
            "A dagger and hollow marker flag runs whose FFT and zero-cross carrier estimates failed the frequency cross-check.",
            "N is the number of captures contributing a valid power estimate and can be smaller than the run's unclipped-capture count.",
        ],
        provenance=["outputs/lissajous_v2/synthesis_charge.csv"],
        suffix="b",
    )
    return paths, caption


def make_fig10(
    root: Path,
    discharge: Sequence[Mapping[str, object]],
    condition_summary: Sequence[Mapping[str, object]],
    synthesis: Sequence[Mapping[str, object]],
) -> tuple[list[Path], Path]:
    conditions = _condition_rows(discharge)
    duty_by_cond = {
        str(row.get("cond", "")): row
        for row in condition_summary
        if str(row.get("metric", "")) == "duty_on_fraction"
    }
    synth_rows = _ordered_synthesis(synthesis)
    fig, (left, right) = plt.subplots(
        1,
        2,
        figsize=FIGSIZE,
        gridspec_kw={"width_ratios": (1.28, 1.0)},
        constrained_layout=True,
    )
    _scope_tag(fig, "operational (Lissajousfigure) + synthesis")
    values: list[str] = []

    y_left = np.arange(len(conditions))[::-1]
    for y, row in zip(y_left, conditions):
        cond = str(row.get("cond", ""))
        summary = duty_by_cond.get(cond)
        medium = str(row.get("medium", ""))
        color = MATERIAL_COLORS.get(medium, OKABE_ITO["gray"])
        stat = None
        if summary is not None:
            median = _number(summary, "median")
            if median is not None:
                summary_n = _integer(summary, "N")
                discharge_n = _integer(row, "N_capture")
                n = (
                    min(summary_n, discharge_n)
                    if summary_n > 0 and discharge_n > 0
                    else max(summary_n, discharge_n)
                )
                stat = (
                    median,
                    _number(summary, "p2_5"),
                    _number(summary, "p97_5"),
                    n,
                )
        if stat is None:
            left.scatter(0.015, y, marker="x", s=60, color=OKABE_ITO["vermillion"])
            left.text(
                0.04,
                y,
                "withheld",
                va="center",
                color=OKABE_ITO["vermillion"],
            )
            continue
        median, low, high, n = stat
        left.errorbar(
            median,
            y,
            xerr=_xerr(median, low, high),
            fmt="o",
            markersize=6.5,
            color=color,
            capsize=3,
        )
        text_x = min(0.71, max(median, high or median) + 0.018)
        left.text(text_x, y, f"{median:.3f} · N={n}", va="center", fontsize=11)
        values.append(
            f"{_condition_label(row)}: duty-on fraction {median:.3f}, N={n}."
        )

    left.set_yticks(y_left, [_condition_label(row) for row in conditions])
    left.set_xlim(0.0, 0.76)
    left.set_xlabel("Duty-on fraction of burst period")
    left.set_ylabel("Operational maximum condition")
    left.text(
        0.01,
        0.985,
        "(a) Operational records",
        transform=left.transAxes,
        ha="left",
        va="top",
        fontsize=11,
    )
    _finish_axis(left, "x")

    y_right = np.arange(len(synth_rows))[::-1]
    right_labels: list[str] = []
    for y, row in zip(y_right, synth_rows):
        label = _synthesis_label(row)
        right_labels.append(label)
        color = _synthesis_color(row)
        stat = _stat(row, "duty_on_fraction")
        if stat is None or _integer(row, "N_capture") <= 0:
            right.scatter(
                0.015,
                y,
                marker="x",
                s=65,
                linewidths=2,
                color=OKABE_ITO["vermillion"],
            )
            right.text(
                0.04,
                y,
                "withheld — Channel D clipped",
                va="center",
                color=OKABE_ITO["vermillion"],
                fontsize=11,
            )
            values.append(f"{label}: duty withheld because Channel D was clipped.")
            continue
        median, low, high, n = stat
        right.errorbar(
            median,
            y,
            xerr=_xerr(median, low, high),
            fmt="o",
            markersize=6.5,
            color=color,
            capsize=3,
        )
        text_x = min(0.71, max(median, high or median) + 0.018)
        right.text(text_x, y, f"{median:.3f} · N={n}", va="center", fontsize=11)
        values.append(f"{label}: duty-on fraction {median:.3f}, N={n}.")

    right.set_yticks(y_right, right_labels)
    right.set_xlim(0.0, 0.76)
    right.set_xlabel("Duty-on fraction of burst period")
    right.set_ylabel("Synthesis run")
    right.text(
        0.01,
        0.985,
        "(b) Synthesis records",
        transform=right.transAxes,
        ha="left",
        va="top",
        fontsize=11,
    )
    _finish_axis(right, "x")

    paths = _save(fig, root, 10)
    caption = _caption(
        root,
        10,
        (
            "The measured envelope-active fraction of each burst period is "
            "reported separately for operational maximum records and synthesis "
            "records. Whiskers are capture-level 2.5–97.5% intervals."
        ),
        values=values,
        notes=[
            "Duty-on fraction describes the burst envelope; it is not the electrical carrier frequency.",
            "Crosses identify runs for which clipping prevents a reportable synthesis summary.",
        ],
        provenance=[
            "outputs/lissajous_v2/condition_summary.csv",
            "outputs/lissajous_v2/synthesis_charge.csv",
        ],
    )
    return paths, caption


def make_fig11a(
    root: Path,
    synthesis: Sequence[Mapping[str, object]],
) -> tuple[list[Path], Path]:
    rows = _ordered_synthesis(synthesis)
    fig, ax = plt.subplots(figsize=FIGSIZE, constrained_layout=True)
    _scope_tag(fig, "synthesis")
    y_positions = np.arange(len(rows))[::-1]
    labels: list[str] = []
    values: list[str] = []

    for y, row in zip(y_positions, rows):
        label = _synthesis_label(row)
        qc = str(row.get("qc_status", ""))
        frequency_qc = "frequency_crosscheck_failed" in qc
        if frequency_qc:
            label += "†"
        labels.append(label)
        color = _synthesis_color(row)
        positive = _stat(row, "dQ_positive_nC")
        negative = _stat(row, "dQ_negative_nC")
        if (
            positive is None
            or negative is None
            or _integer(row, "N_capture") <= 0
        ):
            ax.scatter(
                0.0,
                y,
                marker="x",
                s=70,
                linewidths=2,
                color=OKABE_ITO["vermillion"],
                zorder=3,
            )
            ax.text(
                8.0,
                y,
                "withheld — Channel D clipped",
                va="center",
                ha="left",
                color=OKABE_ITO["vermillion"],
                fontsize=11,
            )
            values.append(
                f"{_synthesis_label(row)}: peak-half-cycle charge withheld "
                "because Channel D was clipped."
            )
            continue

        p, p_low, p_high, p_n = positive
        n, n_low, n_high, n_n = negative
        n_signed = -n
        n_left = -n_high if n_high is not None else None
        n_right = -n_low if n_low is not None else None
        asymmetry = n - p

        ax.errorbar(
            p,
            y,
            xerr=_xerr(p, p_low, p_high),
            fmt="o",
            markersize=7,
            color=color,
            markerfacecolor="white" if frequency_qc else color,
            markeredgecolor=color,
            capsize=3,
            zorder=3,
        )
        ax.scatter(
            asymmetry,
            y,
            marker="^",
            s=66,
            facecolor=OKABE_ITO["yellow"],
            edgecolor="#333333",
            linewidth=1.0,
            zorder=4,
        )
        ax.errorbar(
            n_signed,
            y,
            xerr=_xerr(n_signed, n_left, n_right),
            fmt="s",
            markersize=7,
            color=color,
            markerfacecolor="white",
            markeredgecolor=color,
            capsize=3,
            zorder=3,
        )
        ax.text(
            max(p, p_high or p) + 6.0,
            y,
            f"+{p:.1f}",
            va="center",
            ha="left",
            fontsize=11,
        )
        ax.text(
            n_signed - 6.0,
            y,
            f"−{n:.1f}",
            va="center",
            ha="right",
            fontsize=11,
            bbox={
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.82,
                "pad": 0.4,
            },
            zorder=4,
        )
        ax.text(
            asymmetry,
            y + 0.22,
            f"Δ{asymmetry:+.1f}",
            va="bottom",
            ha="center",
            fontsize=10,
            color="#333333",
            zorder=5,
        )
        if (
            str(row.get("run_key", "")) == "7_21/AgCuNi1_6betaine_12PD"
            and n_low is not None
            and n_high is not None
        ):
            interval_midpoint = -0.5 * (n_low + n_high)
            ax.annotate(
                (
                    f"unstable negative interval: [{n_low:.1f}, {n_high:.1f}] nC\n"
                    "8× span; peak-half-cycle selector fragile"
                ),
                (interval_midpoint, y),
                xytext=(-360.0, y + 0.72),
                textcoords="data",
                ha="left",
                va="center",
                color=OKABE_ITO["vermillion"],
                fontsize=10,
                fontweight="bold",
                arrowprops={
                    "arrowstyle": "->",
                    "color": OKABE_ITO["vermillion"],
                    "lw": 1.2,
                },
                bbox={
                    "boxstyle": "round,pad=0.32",
                    "facecolor": "white",
                    "edgecolor": OKABE_ITO["vermillion"],
                    "alpha": 0.94,
                },
                zorder=8,
            )
        values.append(
            f"{_synthesis_label(row)}: positive {p:.1f} nC "
            f"[{p_low:.1f}, {p_high:.1f}] and negative {n:.1f} nC "
            f"[{n_low:.1f}, {n_high:.1f}], negative − positive "
            f"{asymmetry:+.1f} nC, N={min(p_n, n_n)}."
        )

    ax.axvline(0.0, color="#555555", linewidth=1.0)
    ax.set_xlim(-370.0, 340.0)
    ax.set_ylim(-0.55, len(rows) + 1.45)
    ax.set_yticks(y_positions, labels)
    ax.set_xlabel(
        "Peak-half-cycle charge (nC; negative polarity plotted left)"
    )
    ax.set_ylabel("Synthesis run")
    _finish_axis(ax, "x")
    ax.text(
        0.995,
        0.965,
        "Selection: half-cycle with maximum |ΔQ| per capture",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=10.5,
        color="#333333",
        fontweight="bold",
    )
    ax.legend(
        handles=[
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="none",
                markerfacecolor=OKABE_ITO["gray"],
                markeredgecolor=OKABE_ITO["gray"],
                label="Positive charge",
            ),
            Line2D(
                [0],
                [0],
                marker="s",
                linestyle="none",
                markerfacecolor="white",
                markeredgecolor=OKABE_ITO["gray"],
                label="Negative charge (plotted left)",
            ),
            Line2D(
                [0],
                [0],
                marker="^",
                linestyle="none",
                markerfacecolor=OKABE_ITO["yellow"],
                markeredgecolor="#333333",
                label="Asymmetry: negative − positive",
            ),
        ],
        loc="upper right",
        bbox_to_anchor=(1.0, 0.925),
        frameon=False,
        ncol=3,
        columnspacing=1.1,
        handletextpad=0.4,
    )
    _reserve_bottom(fig)
    fig.text(
        0.985,
        0.015,
        (
            "negative polarity = electrons + negative ions; not species-resolved · "
            "within-burst charge return not excluded; not net delivered charge · "
            "† Carrier cross-check failed (hollow positive marker)"
        ),
        ha="right",
        va="bottom",
        fontsize=9.5,
        color="#555555",
    )

    paths = _save(
        fig,
        root,
        11,
        suffix="a",
        stem="fig11a_synthesis_signed_peak_halfcycle_charge",
    )
    caption = _caption(
        root,
        11,
        (
            "Positive- and negative-polarity electrical charge per peak "
            "half-cycle are shown on a diverging axis. For every capture, the "
            "selected half-cycle is the one with maximum |ΔQ|. Triangles show "
            "the per-run polarity asymmetry (negative − positive). Whiskers are "
            "capture-level 2.5–97.5% intervals."
        ),
        values=values,
        notes=[
            "This is charge per selected peak half-cycle, not charge per second, whole-waveform throughput, or retained dielectric charge.",
            "The electrical waveform is not species-resolved; negative polarity cannot separate electrons from negative ions.",
            "Within-burst charge return is not excluded, so this is not net delivered charge.",
            "The 1:6 betaine:1,2-PD negative interval [39.5, 332.6] nC spans about 8×; its peak-half-cycle selector is flagged as unstable.",
            "A dagger and hollow positive marker flag runs whose FFT and zero-cross carrier estimates failed the frequency cross-check.",
        ],
        provenance=["outputs/lissajous_v2/synthesis_charge.csv"],
        suffix="a",
    )
    return paths, caption


def make_fig11b(
    root: Path,
    synthesis: Sequence[Mapping[str, object]],
) -> tuple[list[Path], Path]:
    rows = _ordered_synthesis(synthesis)
    fig, ax = plt.subplots(figsize=FIGSIZE, constrained_layout=True)
    _scope_tag(fig, "synthesis")
    y_positions = np.arange(len(rows))[::-1]
    labels: list[str] = []
    values: list[str] = []
    scale = 1.0e17

    for y, row in zip(y_positions, rows):
        label = _synthesis_label(row)
        qc = str(row.get("qc_status", ""))
        frequency_qc = "frequency_crosscheck_failed" in qc
        if frequency_qc:
            label += "†"
        labels.append(label)
        color = _synthesis_color(row)
        positive = _stat(
            row, "positive_peak_halfcycle_average_equivalent_flow_per_s"
        )
        negative = _stat(
            row, "negative_peak_halfcycle_average_equivalent_flow_per_s"
        )
        if (
            positive is None
            or negative is None
            or _integer(row, "N_capture") <= 0
        ):
            ax.scatter(
                0.0,
                y,
                marker="x",
                s=70,
                linewidths=2,
                color=OKABE_ITO["vermillion"],
            )
            ax.text(
                0.18,
                y,
                "withheld — Channel D clipped",
                va="center",
                ha="left",
                color=OKABE_ITO["vermillion"],
                fontsize=11,
            )
            values.append(f"{_synthesis_label(row)}: flow withheld because Channel D was clipped.")
            continue

        p, p_low, p_high, p_n = positive
        n, n_low, n_high, n_n = negative
        p_scaled = p / scale
        p_low_scaled = p_low / scale if p_low is not None else None
        p_high_scaled = p_high / scale if p_high is not None else None
        n_scaled = -n / scale
        n_left = -(n_high / scale) if n_high is not None else None
        n_right = -(n_low / scale) if n_low is not None else None
        marker_face = "white" if frequency_qc else color

        ax.errorbar(
            p_scaled,
            y,
            xerr=_xerr(p_scaled, p_low_scaled, p_high_scaled),
            fmt="o",
            markersize=7,
            color=color,
            markerfacecolor=marker_face,
            markeredgecolor=color,
            capsize=3,
            zorder=3,
        )
        ax.errorbar(
            n_scaled,
            y,
            xerr=_xerr(n_scaled, n_left, n_right),
            fmt="s",
            markersize=7,
            color=color,
            markerfacecolor="white",
            markeredgecolor=color,
            capsize=3,
            zorder=3,
        )
        ax.text(
            max(p_scaled, p_high_scaled or p_scaled) + 0.16,
            y,
            f"+{p_scaled:.2f}",
            va="center",
            ha="left",
            fontsize=11,
        )
        ax.text(
            n_scaled - 0.16,
            y,
            f"−{abs(n_scaled):.2f}",
            va="center",
            ha="right",
            fontsize=11,
            bbox={
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.82,
                "pad": 0.4,
            },
            zorder=4,
        )
        values.append(
            f"{_synthesis_label(row)}: positive {p_scaled:.2f} and negative "
            f"{abs(n_scaled):.2f} ×10^17 charge equivalents s⁻¹ "
            f"(N={min(p_n, n_n)})."
        )

    ax.axvline(0.0, color="#555555", linewidth=1.0)
    ax.set_xlim(-6.4, 6.4)
    ax.set_ylim(-0.55, len(rows) + 1.15)
    ax.set_yticks(y_positions, labels)
    ax.set_xlabel(
        "Half-cycle-average charge-equivalent flow (10¹⁷ elementary charges s⁻¹)"
    )
    ax.set_ylabel("Synthesis run")
    _finish_axis(ax, "x")
    ax.text(
        0.995,
        0.985,
        (
            "Negative = e⁻ + negative ions; not species-resolved\n"
            "within-burst charge return not excluded; not net delivered charge\n"
            "† Carrier cross-check failed (hollow positive marker)"
        ),
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=11,
        color="#555555",
    )

    paths = _save(
        fig,
        root,
        11,
        suffix="b",
        stem="fig11b_synthesis_signed_halfcycle_flow",
    )
    caption = _caption(
        root,
        11,
        (
            "Positive- and negative-polarity half-cycle-average electrical "
            "charge-equivalent flow rates are shown on a diverging axis. "
            "Whiskers are capture-level 2.5–97.5% intervals."
        ),
        values=values,
        notes=[
            "These are half-cycle-average rates, not nanosecond microdischarge peak rates.",
            "Negative polarity combines electrons and negative ions; the waveform is not species-resolved.",
            "Within-burst charge return is not excluded, so this is not net delivered charge.",
            "The rates are electrical delivery equivalents and are not Faradaic-utilization measurements.",
            "A dagger and hollow positive marker flag runs whose FFT and zero-cross carrier estimates failed the frequency cross-check.",
        ],
        provenance=["outputs/lissajous_v2/synthesis_charge.csv"],
        suffix="b",
    )
    return paths, caption


def make_fig12(
    root: Path,
    synthesis: Sequence[Mapping[str, object]],
) -> tuple[list[Path], Path]:
    rows = _ordered_synthesis(synthesis)
    fig, ax = plt.subplots(figsize=FIGSIZE, constrained_layout=True)
    _scope_tag(fig, "synthesis")
    y_positions = np.arange(len(rows))[::-1]
    labels: list[str] = []
    values: list[str] = []

    for y, row in zip(y_positions, rows):
        label = _synthesis_label(row)
        labels.append(label)
        color = _synthesis_color(row)
        rate = _stat(row, "gross_rate_C_min")
        dose = _stat(row, "dose_20min_C")
        if rate is None or _integer(row, "N_capture") <= 0:
            ax.scatter(
                0.02,
                y,
                marker="x",
                s=70,
                linewidths=2,
                color=OKABE_ITO["vermillion"],
            )
            ax.text(
                0.08,
                y,
                "withheld — Channel D clipped",
                va="center",
                ha="left",
                color=OKABE_ITO["vermillion"],
                fontsize=11,
            )
            values.append(f"{label}: gross rate withheld because Channel D was clipped.")
            continue
        median, low, high, n = rate
        ax.errorbar(
            median,
            y,
            xerr=_xerr(median, low, high),
            fmt="o",
            markersize=7,
            color=color,
            capsize=3,
            zorder=3,
        )
        dose_value = dose[0] if dose is not None else median * 20.0
        ax.text(
            min(3.30, max(median, high or median) + 0.07),
            y,
            f"{median:.3g} C min⁻¹ · {dose_value:.3g} C/20 min · N={n}",
            va="center",
            ha="left",
            fontsize=11,
        )
        values.append(
            f"{label}: {median:.3g} C min⁻¹ and {dose_value:.3g} C in 20 min, N={n}."
        )

    ax.set_xlim(0.0, 4.1)
    ax.set_yticks(y_positions, labels)
    ax.set_xlabel("Gross electrical charge-delivery rate (C min⁻¹)")
    ax.set_ylabel("Synthesis run")
    _finish_axis(ax, "x")
    top = ax.secondary_xaxis(
        "top",
        functions=(lambda rate: rate * 20.0, lambda dose: dose / 20.0),
    )
    top.set_xlabel("Equivalent electrical charge delivered in 20 min (C)")
    # Pin the secondary ticks: letting the locator choose puts a tick beyond
    # the transformed limit, which then renders off-canvas.
    top.set_xticks([0, 20, 40, 60, 80])

    paths = _save(fig, root, 12)
    caption = _caption(
        root,
        12,
        (
            "Gross electrical charge-delivery rate is shown for each synthesis "
            "run, with a directly linked top scale for the electrical charge "
            "delivered in 20 minutes. Whiskers are capture-level 2.5–97.5% intervals."
        ),
        values=values,
        notes=[
            "The upper scale is exactly rate × 20 min; it is not an independent measurement.",
            "Electrical charge dose is not a metal-ion conversion, chemical yield, or Faradaic-efficiency claim.",
        ],
        provenance=["outputs/lissajous_v2/synthesis_charge.csv"],
    )
    return paths, caption


def _legacy_groups(
    rows: Sequence[Mapping[str, object]],
) -> dict[tuple[str, int], list[Mapping[str, object]]]:
    groups: dict[tuple[str, int], list[Mapping[str, object]]] = defaultdict(list)
    for row in rows:
        material = str(row.get("material", ""))
        burst = _number(row, "burst_kHz")
        if material and burst is not None:
            groups[(material, int(round(burst)))].append(row)
    return groups


def _legacy_material_axes(
    material: str,
    *,
    include_low_n_key: bool,
) -> tuple[plt.Figure, np.ndarray]:
    fig, axes = plt.subplots(
        1,
        3,
        figsize=FIGSIZE,
        sharex=True,
        sharey=True,
    )
    fig.subplots_adjust(
        left=0.085,
        right=0.985,
        bottom=0.15,
        top=0.82,
        wspace=0.10,
    )
    _scope_tag(fig, "legacy report")
    _legacy_model_tag(fig)
    for col, frequency in enumerate((4, 10, 20)):
        axes[col].set_title(f"Burst frequency: {frequency} kHz", fontsize=11)

    handles = [
        Line2D(
            [0],
            [0],
            color=OKABE_ITO["blue"],
            marker="o",
            markersize=5,
            label="Positive polarity",
        ),
        Line2D(
            [0],
            [0],
            color=OKABE_ITO["vermillion"],
            marker="s",
            markersize=5,
            label="Negative polarity",
        ),
    ]
    if include_low_n_key:
        handles.append(
            Line2D(
                [0],
                [0],
                color=OKABE_ITO["gray"],
                marker="x",
                linestyle="none",
                markersize=6,
                label="QC only: N < 16",
            )
        )
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.53, 0.973),
        frameon=False,
        ncol=len(handles),
        handlelength=1.6,
        columnspacing=1.3,
    )
    fig.text(
        0.085,
        0.965,
        MATERIAL_LABELS[material],
        ha="left",
        va="top",
        fontsize=12,
        color="#333333",
    )
    fig.text(
        0.995,
        0.965,
        "LEGACY / MODEL-DEPENDENT",
        ha="right",
        va="top",
        fontsize=11,
        color=OKABE_ITO["vermillion"],
    )
    return fig, axes


def _legacy_charge_limits(
    rows: Sequence[Mapping[str, object]],
) -> tuple[float, float]:
    values = [0.0]
    for row in rows:
        estimate, low, high = _legacy_stat(row)
        values.extend(
            value
            for value in (estimate, low, high)
            if value is not None and math.isfinite(value)
        )
    minimum = min(values)
    maximum = max(values)
    span = max(50.0, maximum - minimum)
    padded_low = minimum - 0.08 * span
    padded_high = maximum + 0.12 * span
    step = 50.0 if span <= 600.0 else 100.0
    return (
        step * math.floor(padded_low / step),
        step * math.ceil(padded_high / step),
    )


def _plot_legacy_dose_panel(
    ax: plt.Axes,
    rows: Sequence[Mapping[str, object]],
) -> None:
    for polarity, color, marker in (
        ("positive", OKABE_ITO["blue"], "o"),
        ("negative", OKABE_ITO["vermillion"], "s"),
    ):
        subset = sorted(
            (
                row
                for row in rows
                if str(row.get("polarity", "")).lower() == polarity
            ),
            key=lambda row: _number(row, "x_median") or math.inf,
        )
        high_n = [row for row in subset if _integer(row, "n_captures") >= 16]
        low_n = [row for row in subset if _integer(row, "n_captures") < 16]
        if high_n:
            x = np.asarray([_number(row, "x_median") for row in high_n], dtype=float)
            stats = [_legacy_stat(row) for row in high_n]
            y = np.asarray([stat[0] for stat in stats], dtype=float)
            low = np.asarray(
                [stat[1] if stat[1] is not None else stat[0] for stat in stats]
            )
            high = np.asarray(
                [stat[2] if stat[2] is not None else stat[0] for stat in stats]
            )
            ax.plot(x, y, color=color, marker=marker, markersize=4, linewidth=1.3)
            ax.fill_between(x, low, high, color=color, alpha=0.11, linewidth=0)
            ax.text(
                x[-1],
                y[-1],
                "+" if polarity == "positive" else "−",
                color=color,
                fontsize=11,
                ha="left",
                va="center",
                clip_on=True,
            )
        if low_n:
            x = [_number(row, "x_median") for row in low_n]
            y = [_legacy_stat(row)[0] for row in low_n]
            ax.scatter(
                x,
                y,
                marker="x",
                s=26,
                linewidths=1.2,
                color=color,
                alpha=0.55,
                zorder=3,
            )
    ax.axhline(0.0, color="#777777", linewidth=0.7)
    _finish_axis(ax, "both")


def make_fig13(
    root: Path,
    legacy_rows: Sequence[Mapping[str, object]],
    legacy_dir: Path,
) -> tuple[list[Path], list[Path]]:
    groups = _legacy_groups(legacy_rows)
    paths: list[Path] = []
    captions: list[Path] = []
    for material in MATERIAL_ORDER:
        suffix, slug = LEGACY_PANEL_IDS[material]
        fig, axes = _legacy_material_axes(
            material,
            include_low_n_key=True,
        )
        material_rows = [
            row for row in legacy_rows if str(row.get("material", "")) == material
        ]
        y_limits = _legacy_charge_limits(material_rows)
        for col_index, frequency in enumerate((4, 10, 20)):
            ax = axes[col_index]
            _plot_legacy_dose_panel(
                ax,
                groups.get((material, frequency), ()),
            )
            ax.set_xlim(0.0, 6.15)
            ax.set_ylim(*y_limits)
        fig.supxlabel("Measured amplitude (kV)", fontsize=11, y=0.055)
        fig.supylabel(
            "Legacy model-dependent per-lobe charge (nC)",
            fontsize=11,
            x=0.018,
        )

        paths.extend(
            _save(
                fig,
                root,
                13,
                suffix=suffix,
                stem=f"fig13{suffix}_legacy_dose_response_{slug}",
            )
        )
        captions.append(
            _caption(
                root,
                13,
                (
                    "Legacy model-dependent per-lobe charge for "
                    f"{MATERIAL_LABELS[material]} is shown against measured "
                    "voltage amplitude in three burst-frequency panels. "
                    "Positive and negative polarities are kept separate."
                ),
                notes=[
                    "Filled line-connected points require at least 16 contributing captures; crosses retain lower-N bins as QC only.",
                    "Bands are the legacy 95% circular moving-block bootstrap intervals over captures.",
                    "These values depend on the legacy displacement/background model and are not direct geometric or surface-charge measurements.",
                    "Axes are identical across the three frequency panels in this material-specific figure.",
                ],
                provenance=[
                    _provenance_path(
                        legacy_dir / "dose_response_binned.csv",
                        root,
                    )
                ],
                suffix=suffix,
            )
        )
    return paths, captions


def _plot_stationarity_panel(
    ax: plt.Axes,
    rows: Sequence[Mapping[str, object]],
) -> None:
    for polarity, color, marker in (
        ("positive", OKABE_ITO["blue"], "o"),
        ("negative", OKABE_ITO["vermillion"], "s"),
    ):
        subset = sorted(
            (
                row
                for row in rows
                if str(row.get("polarity", "")).lower() == polarity
            ),
            key=lambda row: _number(row, "x_median") or math.inf,
        )
        if not subset:
            continue
        x = np.asarray(
            [1000.0 * (_number(row, "x_median") or 0.0) for row in subset],
            dtype=float,
        )
        stats = [_legacy_stat(row) for row in subset]
        y = np.asarray([stat[0] for stat in stats], dtype=float)
        low = np.asarray(
            [stat[1] if stat[1] is not None else stat[0] for stat in stats]
        )
        high = np.asarray(
            [stat[2] if stat[2] is not None else stat[0] for stat in stats]
        )
        ax.plot(x, y, color=color, marker=marker, markersize=3.6, linewidth=1.3)
        ax.fill_between(x, low, high, color=color, alpha=0.11, linewidth=0)
        ax.annotate(
            "+" if polarity == "positive" else "−",
            (x[-1], y[-1]),
            xytext=(3, 7 if polarity == "positive" else -7),
            textcoords="offset points",
            color=color,
            fontsize=11,
            ha="left",
            va="center",
            clip_on=True,
        )
    ax.axhline(0.0, color="#777777", linewidth=0.7)
    _finish_axis(ax, "both")


def make_fig14(
    root: Path,
    legacy_rows: Sequence[Mapping[str, object]],
    legacy_dir: Path,
) -> tuple[list[Path], list[Path]]:
    groups = _legacy_groups(legacy_rows)
    paths: list[Path] = []
    captions: list[Path] = []
    for material in MATERIAL_ORDER:
        suffix, slug = LEGACY_PANEL_IDS[material]
        fig, axes = _legacy_material_axes(
            material,
            include_low_n_key=False,
        )
        material_rows = [
            row for row in legacy_rows if str(row.get("material", "")) == material
        ]
        y_limits = _legacy_charge_limits(material_rows)
        for col_index, frequency in enumerate((4, 10, 20)):
            ax = axes[col_index]
            _plot_stationarity_panel(
                ax,
                groups.get((material, frequency), ()),
            )
            ax.set_xlim(0.0, 10.35)
            ax.set_ylim(*y_limits)
        fig.supxlabel("Midpoint in 10 ms record (ms)", fontsize=11, y=0.055)
        fig.supylabel(
            "Legacy model-dependent per-lobe charge (nC)",
            fontsize=11,
            x=0.018,
        )

        paths.extend(
            _save(
                fig,
                root,
                14,
                suffix=suffix,
                stem=f"fig14{suffix}_legacy_stationarity_{slug}",
            )
        )
        captions.append(
            _caption(
                root,
                14,
                (
                    "Within-record stationarity of the legacy model-dependent "
                    f"per-lobe charge for {MATERIAL_LABELS[material]} is shown "
                    "over each 10 ms record in three burst-frequency panels."
                ),
                notes=[
                    "Bands are legacy 95% circular moving-block bootstrap intervals over captures.",
                    "Time trends diagnose memory/drift within the acquisition record; they do not establish retained dielectric charge.",
                    "The charge scale remains dependent on the legacy displacement/background model.",
                    "Axes are identical across the three frequency panels in this material-specific figure.",
                ],
                provenance=[
                    _provenance_path(
                        legacy_dir / "stationarity_binned.csv",
                        root,
                    )
                ],
                suffix=suffix,
            )
        )
    return paths, captions


def make_fig15(
    root: Path,
    legacy_rows: Sequence[Mapping[str, object]],
    legacy_dir: Path,
) -> tuple[list[Path], Path]:
    groups = _legacy_groups(legacy_rows)
    panel_materials = ("BMIM_nitrate", "5mM_Mn_nitrate_in_water")
    panel_labels = {
        "BMIM_nitrate": (
            f"(a) Hypothetical {MATERIAL_LABELS['BMIM_nitrate']} rate transfer"
        ),
        "5mM_Mn_nitrate_in_water": (
            f"(b) {MATERIAL_LABELS['5mM_Mn_nitrate_in_water']}"
        ),
    }
    fig, axes = plt.subplots(
        1,
        2,
        figsize=FIGSIZE,
        sharex=True,
        sharey=True,
        constrained_layout=False,
    )
    fig.subplots_adjust(
        left=0.075,
        right=0.99,
        bottom=0.13,
        top=0.82,
        wspace=0.065,
    )
    _scope_tag(fig, "legacy report")
    _legacy_model_tag(fig, y=0.885)
    values: list[str] = []

    for ax, material in zip(axes, panel_materials):
        for frequency in (4, 10, 20):
            rows = sorted(
                groups.get((material, frequency), ()),
                key=lambda row: _number(row, "volume_ml") or -math.inf,
            )
            if not rows:
                continue
            color = FREQUENCY_COLORS[frequency]
            x = np.asarray([_number(row, "volume_ml") for row in rows], dtype=float)
            y = np.asarray(
                [_number(row, "minutes_per_ion_equivalent") for row in rows],
                dtype=float,
            )
            low = np.asarray(
                [
                    _number(row, "minutes_per_ion_equivalent_ci_low")
                    if _number(row, "minutes_per_ion_equivalent_ci_low") is not None
                    else value
                    for row, value in zip(rows, y)
                ]
            )
            high = np.asarray(
                [
                    _number(row, "minutes_per_ion_equivalent_ci_high")
                    if _number(row, "minutes_per_ion_equivalent_ci_high") is not None
                    else value
                    for row, value in zip(rows, y)
                ]
            )
            ax.plot(x, y, color=color, linewidth=2.0)
            ax.fill_between(x, low, high, color=color, alpha=0.12, linewidth=0)
            reactor = next(
                (row for row in rows if _truthy(row.get("is_reactor_volume"))),
                min(rows, key=lambda row: abs((_number(row, "volume_ml") or 0.0) - 2.5)),
            )
            reactor_x = _number(reactor, "volume_ml") or 2.5
            reactor_y = _number(reactor, "minutes_per_ion_equivalent") or 0.0
            ax.scatter(
                reactor_x,
                reactor_y,
                s=48,
                color=color,
                edgecolor="white",
                linewidth=0.8,
                zorder=4,
            )
            offset = {4: 0.26, 10: 0.05, 20: -0.27}[frequency]
            ax.text(
                reactor_x + 0.08,
                reactor_y + offset,
                f"{reactor_y:.2f} min",
                color=color,
                fontsize=11,
                va="center",
            )
            ax.text(
                4.92,
                y[-1],
                f"{frequency} kHz",
                color=color,
                fontsize=11,
                ha="right",
                va="center",
            )
            values.append(
                f"{MATERIAL_LABELS[material]}, {frequency} kHz, 2.5 mL: "
                f"{reactor_y:.3g} min per ion equivalent."
            )

        ax.axvline(
            2.5,
            color="#666666",
            linewidth=1.0,
            linestyle="--",
            zorder=1,
        )
        ax.text(
            2.5,
            8.15,
            "2.5 mL reactor",
            ha="center",
            va="top",
            fontsize=11,
            color="#555555",
        )
        ax.text(
            0.02,
            0.98,
            panel_labels[material],
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=11,
        )
        ax.set_xlim(0.0, 5.0)
        ax.set_ylim(0.0, 10.5)
        ax.set_xlabel("Liquid volume (mL)")
        _finish_axis(ax, "both")

    axes[0].set_ylabel("Minutes per 5 mM monovalent-ion equivalent")
    fig.text(
        0.995,
        0.995,
        "LEGACY / MODEL-DEPENDENT",
        ha="right",
        va="top",
        fontsize=11,
        color=OKABE_ITO["vermillion"],
    )
    axes[1].text(
        0.975,
        0.975,
        (
            "Assumptions\n"
            "100% delivery and utilization of negative equivalents\n"
            "upper-bound clock\n"
            "ionic liquid (BMIM*) panel hypothetical; no metal present"
        ),
        transform=axes[1].transAxes,
        ha="right",
        va="top",
        fontsize=10.5,
        color=OKABE_ITO["vermillion"],
        fontweight="bold",
        bbox={
            "boxstyle": "round,pad=0.4",
            "facecolor": "white",
            "edgecolor": OKABE_ITO["vermillion"],
            "alpha": 0.94,
        },
        zorder=10,
    )

    paths = _save(fig, root, 15)
    caption = _caption(
        root,
        15,
        (
            "The legacy/model-dependent dose clock converts negative electrical "
            "charge-equivalent rate into the time required to supply one "
            "monovalent charge equivalent per ion in a hypothetical 5 mM inventory. "
            "The confirmed 2.5 mL reactor volume is marked."
        ),
        values=values,
        notes=[
            "The calculation assumes 100% delivery and utilization of total negative electrical charge equivalents.",
            "It is not electron-specific, not a Faradaic-efficiency measurement, and not a chemical-yield prediction.",
            "The ionic-liquid comparator contains no metal; its BMIM* panel is a hypothetical transfer of the measured electrical rate to a 5 mM metal-ion inventory.",
        ],
        provenance=[
            _provenance_path(
                legacy_dir / "dose_clock_results.csv",
                root,
            )
        ],
    )
    return paths, caption


def _remove_stale_composites(root: Path) -> None:
    stale_paths = [
        root / "presentation_figures" / f"{FIGURE_STEMS[number]}.{extension}"
        for number in (11, 13, 14)
        for extension in ("png", "pdf")
    ]
    stale_paths.extend(
        root / "presentation_captions" / f"fig{number:02d}_caption.md"
        for number in (11, 13, 14)
    )
    for path in stale_paths:
        path.unlink(missing_ok=True)


def _write_supplement_manifest(root: Path, outputs: Sequence[Path]) -> Path:
    caption_dir = root / "presentation_captions"
    png_paths = sorted(
        path for path in outputs if path.suffix.lower() == ".png"
    )
    rows: list[dict[str, str]] = []
    for png in png_paths:
        figure_id = png.stem.split("_", 1)[0]
        pdf = png.with_suffix(".pdf")
        caption = caption_dir / f"{figure_id}_caption.md"
        rows.append(
            {
                "figure_id": figure_id,
                "png": png.relative_to(root).as_posix(),
                "pdf": pdf.relative_to(root).as_posix(),
                "caption": caption.relative_to(root).as_posix(),
                "legacy_model_dependent": str(
                    figure_id.startswith(("fig13", "fig14", "fig15"))
                ).lower(),
            }
        )

    path = root / "presentation_supplement_manifest.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "figure_id",
                "png",
                "pdf",
                "caption",
                "legacy_model_dependent",
            ],
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)
    return path


def generate_all(root: Path, legacy_dir: Path) -> list[Path]:
    _configure_style()
    _configure_medium_labels(root)
    _remove_stale_composites(root)
    vendored_legacy_dir = root / "legacy"
    legacy_filenames = (
        "dose_response_binned.csv",
        "stationarity_binned.csv",
        "dose_clock_results.csv",
    )
    resolved_legacy_dir = (
        vendored_legacy_dir
        if all((vendored_legacy_dir / name).is_file() for name in legacy_filenames)
        else legacy_dir
    )
    discharge = _read_csv(root / "discharge_metrics.csv")
    condition_summary = _read_csv(root / "condition_summary.csv")
    synthesis = _read_csv(root / "synthesis_charge.csv")
    dose_response = _read_csv(resolved_legacy_dir / "dose_response_binned.csv")
    stationarity = _read_csv(resolved_legacy_dir / "stationarity_binned.csv")
    dose_clock = _read_csv(resolved_legacy_dir / "dose_clock_results.csv")

    outputs: list[Path] = []
    for maker, args in (
        (make_fig09, (root, discharge)),
        (make_fig09b, (root, synthesis)),
        (make_fig10, (root, discharge, condition_summary, synthesis)),
        (make_fig11a, (root, synthesis)),
        (make_fig11b, (root, synthesis)),
        (make_fig12, (root, synthesis)),
        (make_fig13, (root, dose_response, resolved_legacy_dir)),
        (make_fig14, (root, stationarity, resolved_legacy_dir)),
        (make_fig15, (root, dose_clock, resolved_legacy_dir)),
    ):
        figure_paths, caption_paths = maker(*args)
        outputs.extend(figure_paths)
        if isinstance(caption_paths, Path):
            outputs.append(caption_paths)
        else:
            outputs.extend(caption_paths)
    outputs.append(_write_supplement_manifest(root, outputs))
    return outputs


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate supplementary presentation figures from existing "
            "Lissajous-v2 and legacy report tables."
        )
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("outputs/lissajous_v2"),
        help="Existing Lissajous-v2 output directory and supplement destination.",
    )
    parser.add_argument(
        "--legacy-dir",
        type=Path,
        default=Path("dbd_surface_charge_report"),
        help="Directory containing the legacy binned presentation tables.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    outputs = generate_all(args.out, args.legacy_dir)
    print(
        f"Generated {len(outputs)} supplementary presentation files under "
        f"{args.out.resolve()}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
