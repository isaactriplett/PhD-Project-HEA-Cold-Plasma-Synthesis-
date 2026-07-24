"""Build the presentation figures for the Lissajous v2 analysis.

This module is intentionally a small, read-only presentation layer.  It reads
the CSV products written by :mod:`lissajous.quantify`, plus the optional
``figure_data.npz`` trace bundle, and writes the eight deterministic figure
pairs and their captions.  It has no pandas or SciPy dependency.

The plotting code is deliberately defensive.  A missing column, an empty
selection, or an unavailable representative trace produces a labelled
empty-state panel and a caption note; it never fabricates a zero-valued
measurement.

Run with::

    python -m lissajous.figures --out outputs/lissajous_v2
"""

from __future__ import annotations

import argparse
import csv
import math
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import matplotlib as mpl

# The figures are batch artifacts.  Selecting Agg before importing pyplot also
# makes the CLI reliable on headless machines.
mpl.use("Agg", force=True)
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D
import numpy as np

from .report import assert_no_retired_numbers


FIGURE_SIZE_IN = (12.0, 6.75)
PNG_DPI = 200

# Okabe-Ito palette.  Known media receive a fixed colour in every figure; the
# remaining colours are used for sources, channels, and unknown media.
OKABE_ITO = {
    "black": "#000000",
    "orange": "#E69F00",
    "sky_blue": "#56B4E9",
    "bluish_green": "#009E73",
    "yellow": "#F0E442",
    "blue": "#0072B2",
    "vermillion": "#D55E00",
    "reddish_purple": "#CC79A7",
}

MEDIUM_COLORS = {
    "argon": OKABE_ITO["blue"],
    "pure_water": OKABE_ITO["sky_blue"],
    "ionic_liquid": OKABE_ITO["orange"],
    "manganese_nitrate": OKABE_ITO["bluish_green"],
    "bmim_ntf2_pt": OKABE_ITO["orange"],
    "agpd_hydrogen": OKABE_ITO["vermillion"],
    "betaine_eg": OKABE_ITO["reddish_purple"],
}

MEDIUM_LABELS = {
    "argon": "Argon",
    "pure_water": "Pure water",
    "ionic_liquid": "Ionic liquid (BMIM*)",
    "manganese_nitrate": "5 mM Mn nitrate",
    "bmim_ntf2_pt": "BMIM-NTf2 + Pt",
    "agpd_hydrogen": "Ag-Pd, Ar/5% H2",
    "betaine_eg": "1:3 betaine:EG",
    "unknown": "Unspecified medium",
}

TABLE_NAMES = (
    "manifest.csv",
    "per_capture_metrics.csv",
    "condition_summary.csv",
    "dispersion_master.csv",
    "discharge_metrics.csv",
    "synthesis_charge.csv",
)

FIGURE_STEMS = (
    "fig1_waveforms_ladder",
    "fig2_loop_evolution_argon4k",
    "fig3_geometric_anchor",
    "fig4_dispersion_master",
    "fig5_mediums_comparison",
    "fig6_discharge_quantities",
    "fig7_synthesis_dose",
    "fig8_qc_appendix",
)

_NUMBER_RE = re.compile(
    r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
)


@dataclass
class FigureInputs:
    """All table and optional trace inputs used by the figure set."""

    root: Path
    manifest: list[dict[str, str]] = field(default_factory=list)
    per_capture: list[dict[str, str]] = field(default_factory=list)
    condition_summary: list[dict[str, str]] = field(default_factory=list)
    dispersion: list[dict[str, str]] = field(default_factory=list)
    discharge: list[dict[str, str]] = field(default_factory=list)
    synthesis: list[dict[str, str]] = field(default_factory=list)
    figure_data: dict[str, np.ndarray] = field(default_factory=dict)
    input_notes: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class Stat:
    """A median, central percentile interval, and sample count."""

    value: float
    low: float | None = None
    high: float | None = None
    n: int | None = None


@dataclass
class AnchorPoint:
    medium: str
    frequency_khz: float
    level_pct: float | None
    amplitude_kv: float
    capacitance: Stat
    syst_frac: float | None
    provenance: str


@dataclass
class DischargePoint:
    medium: str
    condition: str
    freq_label: str
    frequency_khz: float | None
    band_tag: str
    level_pct: float | None
    amplitude_kv: float
    raw_dq: Stat | None
    corrected_dq: Stat | None
    energy: Stat | None
    energy_basis: str | None
    power: Stat | None
    power_method: str | None
    correction_factor: float | None
    provenance: str
    orientation_flag: str | None = None


@dataclass
class SynthesisPoint:
    run: str
    medium: str
    dq: Stat | None
    rate_c_min: Stat | None
    dose_20min_c: Stat | None
    f0_khz: float | None
    burst_hz: float | None
    duty: float | None
    provenance: str
    dq_derivation: str | None = None
    qc_status: str | None = None


def _finite_number(value: object) -> float | None:
    """Parse one finite number from a CSV cell or numpy scalar."""

    if value is None or isinstance(value, (bool, np.bool_)):
        return None
    if isinstance(value, np.ndarray):
        if value.size != 1:
            return None
        value = value.reshape(-1)[0]
    if isinstance(value, (int, float, np.integer, np.floating)):
        result = float(value)
        return result if math.isfinite(result) else None
    text = str(value).strip()
    if not text or text.lower() in {
        "na",
        "n/a",
        "nan",
        "none",
        "null",
        "missing",
        "inf",
        "-inf",
    }:
        return None
    text = text.replace(",", "").replace("\N{MINUS SIGN}", "-")
    match = _NUMBER_RE.search(text)
    if not match:
        return None
    try:
        result = float(match.group(0))
    except ValueError:
        return None
    return result if math.isfinite(result) else None


def _pick_number(row: Mapping[str, object], *keys: str) -> float | None:
    for key in keys:
        value = _finite_number(row.get(key))
        if value is not None:
            return value
    return None


def _pick_text(row: Mapping[str, object], *keys: str) -> str:
    for key in keys:
        value = row.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    return ""


def _boolish(value: object) -> bool | None:
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on", "detected", "discharge"}:
        return True
    if text in {"0", "false", "no", "n", "off", "none", "closed"}:
        return False
    return None


def _format_number(value: float | None, digits: int = 3) -> str:
    if value is None or not math.isfinite(value):
        return "not available"
    magnitude = abs(value)
    if magnitude and (magnitude >= 10_000 or magnitude < 0.01):
        return f"{value:.{max(1, digits - 1)}e}"
    return f"{value:.{digits}g}"


def _format_stat(stat: Stat | None, unit: str = "") -> str:
    if stat is None:
        return "not available"
    suffix = f" {unit}" if unit else ""
    result = f"{_format_number(stat.value)}{suffix}"
    if stat.low is not None and stat.high is not None:
        result += (
            f" [{_format_number(stat.low)}, {_format_number(stat.high)}]"
        )
    if stat.n is not None:
        result += f", N={stat.n}"
    return result


def _as_int(value: object) -> int | None:
    number = _finite_number(value)
    if number is None:
        return None
    return int(round(number))


def _normalize_key(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.strip().lower()).strip("_")


def _canonical_medium(value: object) -> str:
    text = _normalize_key(str(value or ""))
    if not text:
        return "unknown"
    if "betaine" in text or "ethylene_glycol" in text:
        return "betaine_eg"
    if ("ag" in text and "pd" in text and "hydrogen" in text) or "agpd" in text:
        return "agpd_hydrogen"
    if "bmim" in text and ("ntf" in text or "pt" in text):
        return "bmim_ntf2_pt"
    if "manganese" in text or re.search(r"(^|_)mn(_|$)", text):
        return "manganese_nitrate"
    if "pure_water" in text or text in {"water", "h2o"}:
        return "pure_water"
    if "ionic" in text or "bmim" in text:
        return "ionic_liquid"
    if "argon" in text or text in {"ar", "gas_only", "empty_cell"}:
        return "argon"
    return text


def _medium(row: Mapping[str, object]) -> str:
    value = _pick_text(
        row,
        "medium",
        "material",
        "cell_medium",
        "sample",
        "run_medium",
    )
    if not value:
        value = _pick_text(row, "cond", "condition", "run", "path")
    return _canonical_medium(value)


def _medium_label(medium: str) -> str:
    if medium in MEDIUM_LABELS:
        return MEDIUM_LABELS[medium]
    return medium.replace("_", " ").strip().title() or MEDIUM_LABELS["unknown"]


def _medium_color(medium: str) -> str:
    if medium in MEDIUM_COLORS:
        return MEDIUM_COLORS[medium]
    alternatives = (
        OKABE_ITO["yellow"],
        OKABE_ITO["black"],
        OKABE_ITO["vermillion"],
        OKABE_ITO["reddish_purple"],
    )
    # A stable string hash independent of Python's randomized hash seed.
    checksum = sum((index + 1) * ord(char) for index, char in enumerate(medium))
    return alternatives[checksum % len(alternatives)]


def _frequency_khz(row: Mapping[str, object]) -> float | None:
    direct = _pick_number(
        row,
        "frequency_kHz",
        "frequency_khz",
        "f0_kHz",
        "f0_khz",
        "f0_kHz_median",
        "f0_khz_median",
        "freq_kHz",
        "freq_khz",
        "measured_frequency_kHz",
    )
    if direct is not None:
        return direct
    hz = _pick_number(
        row,
        "f0_Hz",
        "frequency_Hz",
        "carrier_Hz",
        "measured_f0_Hz",
    )
    if hz is not None:
        return hz / 1000.0
    label = _pick_text(row, "freq_label", "frequency_label", "cond", "condition")
    match = re.search(r"(\d+(?:\.\d+)?)\s*k\s*hz", label, flags=re.IGNORECASE)
    return float(match.group(1)) if match else None


def _level_pct(row: Mapping[str, object]) -> float | None:
    direct = _pick_number(
        row,
        "level_pct",
        "breakdown_pct",
        "level_percent",
        "nominal_level_pct",
    )
    if direct is not None:
        return direct
    text = _pick_text(row, "level", "cond", "condition", "path")
    match = re.search(r"(\d+(?:\.\d+)?)\s*%", text)
    if match:
        return float(match.group(1))
    normalized = _normalize_key(text)
    match = re.search(r"(?:^|_)(\d+)_(\d+)breakdown(?:_|$)", normalized)
    if match:
        return float(f"{match.group(1)}.{match.group(2)}") * 100.0
    return None


def _band(row: Mapping[str, object]) -> str:
    text = _pick_text(row, "band_tag", "band").strip()
    normalized = _normalize_key(text)
    if normalized in {"lf_geometric", "low_frequency_geometric", "geometric"}:
        return "LF-geometric"
    if normalized in {
        "carrier_transfer",
        "transfer",
        "carrier",
        "measurement_chain_transfer",
    }:
        return "carrier-transfer"
    return text or "band-unassigned"


def _is_lf(row: Mapping[str, object]) -> bool:
    return _band(row) == "LF-geometric"


def _is_transfer(row: Mapping[str, object]) -> bool:
    return _band(row) == "carrier-transfer"


def _flags(row: Mapping[str, object]) -> str:
    return " ".join(
        _pick_text(
            row,
            "flags",
            "exclusion_flags",
            "qc_flags",
            "flag",
            "status_flags",
        )
        .lower()
        .split()
    )


def _is_excluded(row: Mapping[str, object]) -> bool:
    explicit = _boolish(
        row.get("exclude_from_displacement")
        if "exclude_from_displacement" in row
        else row.get("excluded")
    )
    if explicit is True:
        return True
    flags = _flags(row)
    return any(
        token in flags
        for token in (
            "contaminat",
            "exclude",
            "surface_discharge",
            "sub_lsb",
            "quantization_limited",
        )
    )


def _is_hard_excluded(row: Mapping[str, object]) -> bool:
    """Return exclusions that must not be displayed even as QC context.

    Quantization-limited rows remain scientifically unusable for fitted
    geometric values, but Figure 4 needs to show the historical 7_20
    multiline observations that motivated the transfer-chain
    reinterpretation.  This narrower predicate permits those rows to be
    displayed with an explicit QC warning while still removing contaminated,
    deliberately excluded, surface-discharge, and sub-LSB observations.
    """

    explicit = _boolish(
        row.get("exclude_from_displacement")
        if "exclude_from_displacement" in row
        else row.get("excluded")
    )
    if explicit is True:
        return True
    flags = _flags(row)
    return any(
        token in flags
        for token in (
            "contaminat",
            "exclude",
            "surface_discharge",
            "sub_lsb",
        )
    )


def _is_subbreakdown(row: Mapping[str, object]) -> bool:
    for key in (
        "discharge_detected",
        "is_discharge",
        "discharge_on",
        "loop_open",
    ):
        state = _boolish(row.get(key))
        if state is not None:
            return not state
    regime = _pick_text(row, "regime", "classification").lower()
    if regime:
        if any(token in regime for token in ("sub", "closed", "off")):
            return True
        if any(token in regime for token in ("discharge", "open", "on")):
            return False
    level = _level_pct(row)
    # This is only a presentation fallback for older summary tables lacking the
    # objective-onset column.  The quantified objective flag always wins.
    return level is not None and level < 100.0


def _provenance(row: Mapping[str, object]) -> str:
    return (
        _pick_text(
            row,
            "provenance",
            "path",
            "source_path",
            "capture",
            "cond",
            "condition",
            "run",
            "run_id",
            "run_key",
            "label",
        )
        or "source row without provenance label"
    )


def _looks_retired(row: Mapping[str, object]) -> bool:
    """Suppress rows explicitly marked as retired by the analysis contract."""

    text = " ".join(str(value) for value in row.values()).lower()
    if "retired" in text or "deprecated" in text:
        return True
    cd = _pick_number(row, "Cd_pF", "Cd_pF_median", "cd_pf")
    if cd is not None and abs(cd - 50.07) < 0.005:
        return True
    f_value = _pick_number(row, "F", "F_same_band", "F_median")
    if f_value is not None and abs(f_value - 2.254) < 0.0005:
        return True
    medium = _medium(row)
    dq = _pick_number(
        row,
        "dQ_cycle_nC",
        "dQ_cycle_nC_median",
        "dQ_nC",
    )
    return medium == "manganese_nitrate" and dq is not None and abs(dq - 458.0) < 0.05


def _read_csv(path: Path) -> tuple[list[dict[str, str]], str | None]:
    if not path.exists():
        return [], f"{path.name} is missing"
    try:
        with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as handle:
            reader = csv.DictReader(handle)
            if not reader.fieldnames:
                return [], f"{path.name} has no header"
            rows: list[dict[str, str]] = []
            for raw in reader:
                row = {
                    str(key).strip(): ("" if value is None else str(value).strip())
                    for key, value in raw.items()
                    if key is not None
                }
                if any(value for value in row.values()) and not _looks_retired(row):
                    rows.append(row)
            if not rows:
                return [], f"{path.name} contains no usable rows"
            return rows, None
    except (OSError, csv.Error) as exc:
        return [], f"{path.name} could not be read: {exc}"


def _read_figure_data(path: Path) -> tuple[dict[str, np.ndarray], str | None]:
    if not path.exists():
        return {}, "figure_data.npz is absent; trace-dependent panels may be empty"
    try:
        with np.load(path, allow_pickle=False) as archive:
            return {key: np.asarray(archive[key]) for key in archive.files}, None
    except (OSError, ValueError, KeyError) as exc:
        return {}, f"figure_data.npz could not be read: {exc}"


def load_inputs(root: Path) -> FigureInputs:
    """Load all supported table products below ``root``."""

    root = Path(root)
    result = FigureInputs(root=root)
    destinations = {
        "manifest.csv": "manifest",
        "per_capture_metrics.csv": "per_capture",
        "condition_summary.csv": "condition_summary",
        "dispersion_master.csv": "dispersion",
        "discharge_metrics.csv": "discharge",
        "synthesis_charge.csv": "synthesis",
    }
    for filename in TABLE_NAMES:
        rows, note = _read_csv(root / filename)
        setattr(result, destinations[filename], rows)
        if note:
            result.input_notes.append(note)
    result.figure_data, note = _read_figure_data(root / "figure_data.npz")
    if note:
        result.input_notes.append(note)
    return result


def _register_presentation_font() -> str:
    """Register and select Calibri, with Carlito as the metric fallback."""

    desired = ("Calibri", "Carlito")
    try:
        for path in font_manager.findSystemFonts(fontext="ttf"):
            filename = Path(path).name.lower()
            if "calibri" in filename or "carlito" in filename:
                try:
                    font_manager.fontManager.addfont(path)
                except (OSError, RuntimeError, ValueError):
                    pass
    except (OSError, RuntimeError):
        pass
    available = {entry.name.casefold(): entry.name for entry in font_manager.fontManager.ttflist}
    for family in desired:
        if family.casefold() in available:
            return available[family.casefold()]
    return "DejaVu Sans"


def configure_style() -> str:
    """Apply the shared, slide-friendly matplotlib style."""

    family = _register_presentation_font()
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": [family, "Calibri", "Carlito", "DejaVu Sans"],
            "font.size": 9.5,
            "axes.labelsize": 10.5,
            "axes.titlesize": 10.5,
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": "#D9D9D9",
            "grid.linewidth": 0.6,
            "grid.alpha": 0.75,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "legend.fontsize": 8.0,
            "legend.frameon": False,
            "lines.linewidth": 1.6,
            "lines.markersize": 5.0,
            "figure.dpi": PNG_DPI,
            "savefig.dpi": PNG_DPI,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    return family


def _new_figure(
    nrows: int = 1,
    ncols: int = 1,
    *,
    sharex: bool = False,
    sharey: bool = False,
    width_ratios: Sequence[float] | None = None,
) -> tuple[plt.Figure, np.ndarray]:
    gridspec_kw = {"width_ratios": width_ratios} if width_ratios else None
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=FIGURE_SIZE_IN,
        sharex=sharex,
        sharey=sharey,
        constrained_layout=True,
        squeeze=False,
        gridspec_kw=gridspec_kw,
    )
    return fig, axes


def _panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        0.012,
        0.985,
        label,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontweight="bold",
        color=OKABE_ITO["black"],
    )


def _empty_panel(
    ax: plt.Axes,
    detail: str,
    *,
    xlabel: str = "",
    ylabel: str = "",
) -> None:
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    ax.text(
        0.5,
        0.5,
        f"No qualifying data\n{detail}",
        transform=ax.transAxes,
        ha="center",
        va="center",
        color="#555555",
        linespacing=1.35,
    )


def _safe_errorbar(
    ax: plt.Axes,
    x: Sequence[float],
    stats: Sequence[Stat],
    **kwargs: object,
):
    values = np.asarray([stat.value for stat in stats], dtype=float)
    lower = np.asarray(
        [
            stat.value - stat.low
            if stat.low is not None and stat.low <= stat.value
            else 0.0
            for stat in stats
        ],
        dtype=float,
    )
    upper = np.asarray(
        [
            stat.high - stat.value
            if stat.high is not None and stat.high >= stat.value
            else 0.0
            for stat in stats
        ],
        dtype=float,
    )
    yerr = np.vstack([lower, upper]) if np.any(lower > 0) or np.any(upper > 0) else None
    return ax.errorbar(x, values, yerr=yerr, capsize=2.5, **kwargs)


def _save_figure(fig: plt.Figure, root: Path, stem: str) -> list[Path]:
    figures = root / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    png = figures / f"{stem}.png"
    pdf = figures / f"{stem}.pdf"
    fig.set_size_inches(*FIGURE_SIZE_IN, forward=True)
    fig.savefig(png, dpi=PNG_DPI, facecolor="white")
    fig.savefig(
        pdf,
        facecolor="white",
        metadata={
            "Creator": "lissajous.figures",
            "Subject": "Lissajous v2 quantified presentation figure",
            "CreationDate": None,
            "ModDate": None,
        },
    )
    plt.close(fig)
    return [png, pdf]


def _write_caption(
    root: Path,
    number: int,
    description: str,
    *,
    values: Sequence[str] = (),
    sources: Sequence[str] = (),
    notes: Sequence[str] = (),
) -> Path:
    captions = root / "captions"
    captions.mkdir(parents=True, exist_ok=True)
    path = captions / f"fig{number}_caption.md"
    lines = [f"# Figure {number} caption", "", description.strip()]
    if values:
        lines.extend(["", "Values and annotations:"])
        lines.extend(f"- {value}" for value in values)
    if sources:
        lines.extend(["", "Provenance:"])
        lines.extend(f"- {source}" for source in dict.fromkeys(sources))
    if notes:
        lines.extend(["", "Notes:"])
        lines.extend(f"- {note}" for note in notes)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    return path


def _stat(row: Mapping[str, object], candidates: Sequence[str]) -> Stat | None:
    """Read a metric from either wide or long-form summary columns."""

    metric_name = _pick_text(row, "metric", "measure", "quantity")
    metric_normalized = _normalize_key(metric_name)
    for candidate in candidates:
        normalized = _normalize_key(candidate)
        unit_match = re.match(
            r"^(.*)_(pF|nC|uJ|kV|kHz|Hz|W|C)$",
            candidate,
            flags=re.IGNORECASE,
        )
        low_between = (
            f"{unit_match.group(1)}_p2_5_{unit_match.group(2)}"
            if unit_match
            else ""
        )
        high_between = (
            f"{unit_match.group(1)}_p97_5_{unit_match.group(2)}"
            if unit_match
            else ""
        )
        direct_keys = (
            candidate,
            normalized,
            f"{candidate}_median",
            f"{normalized}_median",
            f"median_{candidate}",
            f"median_{normalized}",
        )
        value = _pick_number(row, *direct_keys)
        long_match = metric_normalized == normalized
        if value is None and long_match:
            value = _pick_number(row, "median", "value", "estimate")
        if value is None:
            continue
        low = _pick_number(
            row,
            f"{candidate}_p2_5",
            f"{normalized}_p2_5",
            f"{candidate}_p2.5",
            low_between,
            f"{candidate}_ci_low",
            f"{normalized}_ci_low",
            f"{candidate}_low",
        )
        high = _pick_number(
            row,
            f"{candidate}_p97_5",
            f"{normalized}_p97_5",
            f"{candidate}_p97.5",
            high_between,
            f"{candidate}_ci_high",
            f"{normalized}_ci_high",
            f"{candidate}_high",
        )
        n = None
        for key in (
            f"{candidate}_N",
            f"{normalized}_N",
            f"{candidate}_n",
            f"{normalized}_n",
        ):
            n = _as_int(row.get(key))
            if n is not None:
                break
        if long_match:
            low = low if low is not None else _pick_number(row, "p2_5", "ci_low", "low")
            high = high if high is not None else _pick_number(row, "p97_5", "ci_high", "high")
            n = n if n is not None else _as_int(row.get("N") or row.get("n"))
        else:
            low = low if low is not None else _pick_number(row, "p2_5", "ci_low")
            high = high if high is not None else _pick_number(row, "p97_5", "ci_high")
            n = n if n is not None else _as_int(row.get("N") or row.get("n"))
        return Stat(value=value, low=low, high=high, n=n)
    return None


def _summary_records(rows: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    """Pivot long-form ``metric, median, p2_5, p97_5, N`` rows."""

    long_rows = [
        row
        for row in rows
        if _pick_text(row, "metric", "measure", "quantity")
    ]
    if not long_rows:
        return [dict(row) for row in rows]

    identity_fields = (
        "cond",
        "condition",
        "medium",
        "material",
        "freq_label",
        "frequency_kHz",
        "frequency_khz",
        "f0_kHz",
        "level_pct",
        "band_tag",
        "dataset_type",
        "source_type",
        "provenance",
        "path",
        "discharge_detected",
        "is_discharge",
        "regime",
    )
    groups: dict[tuple[str, ...], dict[str, object]] = {}
    for row in long_rows:
        key = tuple(str(row.get(field, "") or "") for field in identity_fields)
        record = groups.setdefault(
            key,
            {
                field: row.get(field, "")
                for field in identity_fields
                if row.get(field, "") not in (None, "")
            },
        )
        metric = _pick_text(row, "metric", "measure", "quantity")
        value = _pick_number(row, "median", "value", "estimate")
        if value is None:
            continue
        record[metric] = value
        low = _pick_number(row, "p2_5", "ci_low", "low")
        high = _pick_number(row, "p97_5", "ci_high", "high")
        n = _as_int(row.get("N") or row.get("n"))
        if low is not None:
            record[f"{metric}_p2_5"] = low
        if high is not None:
            record[f"{metric}_p97_5"] = high
        if n is not None:
            record[f"{metric}_N"] = n
        syst = _pick_number(row, "syst_frac", "systematic_fraction")
        if syst is not None:
            record["syst_frac"] = syst

    # Preserve any already-wide rows in a mixed table.
    wide_rows = [
        dict(row)
        for row in rows
        if not _pick_text(row, "metric", "measure", "quantity")
    ]
    return list(groups.values()) + wide_rows


def _aggregate(values: Iterable[float]) -> Stat | None:
    array = np.asarray(
        [value for value in values if value is not None and math.isfinite(value)],
        dtype=float,
    )
    if not array.size:
        return None
    low, high = np.percentile(array, [2.5, 97.5])
    return Stat(
        value=float(np.median(array)),
        low=float(low),
        high=float(high),
        n=int(array.size),
    )


def _stat_across_rows(
    rows: Sequence[Mapping[str, object]], candidates: Sequence[str]
) -> Stat | None:
    direct = [_stat(row, candidates) for row in rows]
    direct = [stat for stat in direct if stat is not None]
    if not direct:
        return None
    if len(direct) == 1 and (
        direct[0].low is not None
        or direct[0].high is not None
        or direct[0].n is not None
    ):
        return direct[0]
    return _aggregate(stat.value for stat in direct)


def _npz_array(data: Mapping[str, np.ndarray], *keys: str) -> np.ndarray | None:
    for key in keys:
        if key in data:
            try:
                result = np.asarray(data[key], dtype=float).reshape(-1)
            except (TypeError, ValueError):
                continue
            if result.size:
                return result
    return None


def _npz_scalar(data: Mapping[str, np.ndarray], *keys: str) -> float | None:
    for key in keys:
        if key in data:
            value = _finite_number(data[key])
            if value is not None:
                return value
    return None


def _paired_finite(*arrays: np.ndarray) -> tuple[np.ndarray, ...]:
    length = min((array.size for array in arrays), default=0)
    if length == 0:
        return tuple(np.asarray([], dtype=float) for _ in arrays)
    trimmed = tuple(np.asarray(array[:length], dtype=float) for array in arrays)
    mask = np.logical_and.reduce([np.isfinite(array) for array in trimmed])
    return tuple(array[mask] for array in trimmed)


def _downsample(arrays: Sequence[np.ndarray], maximum: int) -> tuple[np.ndarray, ...]:
    if not arrays:
        return ()
    length = min(array.size for array in arrays)
    if length <= maximum:
        return tuple(array[:length] for array in arrays)
    indices = np.linspace(0, length - 1, maximum, dtype=int)
    return tuple(array[indices] for array in arrays)


def _frequency_matches(row: Mapping[str, object], target_khz: float) -> bool:
    frequency = _frequency_khz(row)
    if frequency is not None:
        return abs(frequency - target_khz) <= max(0.25, 0.08 * target_khz)
    label = _pick_text(row, "freq_label", "frequency_label", "path")
    return bool(
        re.search(
            rf"(?<!\d){re.escape(f'{target_khz:g}')}\s*k\s*hz",
            label,
            flags=re.IGNORECASE,
        )
    )


def _nominal_frequency_matches(
    row: Mapping[str, object], target_khz: float
) -> bool:
    """Match an acquisition label without mistaking it for measured f0."""

    label = _pick_text(row, "freq_label", "frequency_label")
    if not label:
        label = _pick_text(row, "path", "source_path", "cond", "condition")
    return bool(
        re.search(
            rf"(?<!\d){re.escape(f'{target_khz:g}')}\s*k\s*hz",
            label,
            flags=re.IGNORECASE,
        )
    )


def _matching_rows(
    rows: Sequence[Mapping[str, object]],
    *,
    medium: str | None = None,
    frequency_khz: float | None = None,
    level_pct: float | None = None,
) -> list[Mapping[str, object]]:
    output: list[Mapping[str, object]] = []
    for row in rows:
        if medium is not None and _medium(row) != medium:
            continue
        if frequency_khz is not None and not _frequency_matches(row, frequency_khz):
            continue
        if level_pct is not None:
            level = _level_pct(row)
            if level is None or abs(level - level_pct) > 0.51:
                continue
        output.append(row)
    return output


def _manifest_representative(
    inputs: FigureInputs, level_pct: float
) -> Mapping[str, object] | None:
    candidates = [
        row
        for row in inputs.manifest
        if _medium(row) == "argon"
        and _nominal_frequency_matches(row, 4.0)
        and _level_pct(row) is not None
        and abs(float(_level_pct(row)) - level_pct) <= 0.51
    ]
    target_seg = _npz_scalar(
        inputs.figure_data,
        f"argon4k_{level_pct:g}_seg",
    )
    if target_seg is not None:
        segment_matches = [
            row
            for row in candidates
            if _pick_number(row, "seg_idx", "seg", "segment") is not None
            and abs(
                float(_pick_number(row, "seg_idx", "seg", "segment"))
                - target_seg
            )
            < 0.5
        ]
        if segment_matches:
            candidates = segment_matches
    if not candidates:
        return None

    def rank(row: Mapping[str, object]) -> tuple[int, int, int, str]:
        parse_status = _pick_text(row, "parse_status", "status").lower()
        parse_bad = int(bool(parse_status) and parse_status not in {"ok", "parsed", "success"})
        excluded = int(_is_excluded(row))
        clip_total = sum(
            _as_int(row.get(key)) or 0
            for key in row
            if _normalize_key(key).startswith("clip")
        )
        return parse_bad, excluded, clip_total, _provenance(row)

    return sorted(candidates, key=rank)[0]


def make_fig1(inputs: FigureInputs) -> tuple[list[Path], Path]:
    fig, axes_array = _new_figure(1, 2)
    axes = axes_array[0]
    values: list[str] = []
    sources: list[str] = []
    notes: list[str] = []

    for index, (ax, level) in enumerate(zip(axes, (60, 115))):
        prefix = f"argon4k_{level}"
        time_s = _npz_array(
            inputs.figure_data,
            f"{prefix}_time_s",
            f"{prefix}_t_s",
        )
        voltage_kv = _npz_array(
            inputs.figure_data,
            f"{prefix}_V_kV",
            f"{prefix}_voltage_kV",
        )
        charge_nc = _npz_array(
            inputs.figure_data,
            f"{prefix}_Q_nC",
            f"{prefix}_charge_nC",
        )
        representative = _manifest_representative(inputs, level)
        if representative is not None:
            sources.append(
                f"{level}% representative capture: `{_provenance(representative)}`."
            )

        if time_s is None or voltage_kv is None or charge_nc is None:
            _panel_label(ax, f"({chr(97 + index)}) {level}% nominal level")
            _empty_panel(
                ax,
                f"{prefix}_time_s, _V_kV, or _Q_nC is absent from figure_data.npz",
                xlabel="Time (ms)",
                ylabel="Applied voltage / charge (see caption)",
            )
            notes.append(
                f"The {level}% waveform panel is an empty state because its "
                "representative trace arrays were unavailable."
            )
            continue

        time_s, voltage_kv, charge_nc = _paired_finite(
            time_s, voltage_kv, charge_nc
        )
        if time_s.size < 2:
            _panel_label(ax, f"({chr(97 + index)}) {level}% nominal level")
            _empty_panel(
                ax,
                "representative arrays contain fewer than two finite samples",
                xlabel="Time (ms)",
                ylabel="Applied voltage / charge (see caption)",
            )
            notes.append(
                f"The {level}% waveform panel had fewer than two aligned finite samples."
            )
            continue

        full_time = time_s
        duration_ms = float(np.ptp(full_time) * 1000.0)
        burst_hz = _npz_scalar(
            inputs.figure_data,
            f"{prefix}_burst_Hz",
            f"{prefix}_burst_hz",
        )
        if representative is not None and burst_hz is None:
            burst_hz = _pick_number(
                representative,
                "burst_Hz",
                "burst_rate_Hz",
            )
        displayed_bursts: float | None = None
        if burst_hz is not None and burst_hz > 0:
            window_s = 2.0 / burst_hz
            if duration_ms / 1000.0 > window_s:
                keep = time_s <= time_s[0] + window_s
                if np.count_nonzero(keep) >= 8:
                    time_s = time_s[keep]
                    voltage_kv = voltage_kv[keep]
                    charge_nc = charge_nc[keep]
                    displayed_bursts = 2.0
        time_ms = (time_s - time_s[0]) * 1000.0
        time_ms, voltage_kv, charge_nc = _downsample(
            (time_ms, voltage_kv, charge_nc), 12_000
        )
        voltage_line = ax.plot(
            time_ms,
            voltage_kv,
            color=OKABE_ITO["blue"],
            label="Applied voltage",
            rasterized=False,
        )[0]
        ax.set_xlabel("Time (ms)")
        ax.set_ylabel("Applied voltage (kV)", color=OKABE_ITO["blue"])
        ax.tick_params(axis="y", colors=OKABE_ITO["blue"])
        twin = ax.twinx()
        twin.spines["right"].set_visible(True)
        twin.grid(False)
        charge_line = twin.plot(
            time_ms,
            charge_nc,
            color=OKABE_ITO["vermillion"],
            label="Monitor charge",
            alpha=0.9,
            rasterized=False,
        )[0]
        twin.set_ylabel("Charge, Q (nC)", color=OKABE_ITO["vermillion"])
        twin.tick_params(axis="y", colors=OKABE_ITO["vermillion"])
        _panel_label(ax, f"({chr(97 + index)}) {level}% nominal level")

        f0_hz = _npz_scalar(
            inputs.figure_data,
            f"{prefix}_f0_Hz",
            f"{prefix}_f0_hz",
        )
        dc_offset_v = _npz_scalar(
            inputs.figure_data,
            f"{prefix}_dc_offset_V",
            f"{prefix}_dc_offset_v",
        )
        clip_a = _npz_scalar(inputs.figure_data, f"{prefix}_clipA")
        clip_d = _npz_scalar(inputs.figure_data, f"{prefix}_clipD")
        codes_a = None
        codes_d = None
        quantization_status = ""
        qc_flags = ""
        if representative is not None:
            f0_hz = f0_hz or _pick_number(representative, "f0_Hz")
            dc_offset_v = (
                dc_offset_v
                if dc_offset_v is not None
                else _pick_number(representative, "dc_offset", "dc_offset_V")
            )
            clip_a = (
                clip_a
                if clip_a is not None
                else _pick_number(representative, "clipA")
            )
            clip_d = (
                clip_d
                if clip_d is not None
                else _pick_number(representative, "clipD")
            )
            codes_a = _pick_number(representative, "codesA")
            codes_d = _pick_number(representative, "codesD")
            if (
                codes_a is not None
                and codes_d is not None
                and min(codes_a, codes_d) < 30
            ):
                quantization_status = "quantization-limited"
            else:
                quantization_status = "codes gate passed"
            qc_flags = _pick_text(
                representative,
                "exclusion_flags",
                "qc_flags",
                "flags",
            )

        annotation_lines = [
            (
                f"measured f0 = {_format_number(f0_hz / 1000.0)} kHz"
                if f0_hz is not None
                else "measured f0 not recorded"
            ),
            (
                f"measured burst = {_format_number(burst_hz / 1000.0)} kHz; "
                f"displayed = {_format_number(displayed_bursts, 2)} periods"
                if burst_hz is not None and displayed_bursts is not None
                else (
                    f"measured burst = {_format_number(burst_hz / 1000.0)} kHz"
                    if burst_hz is not None
                    else "burst rate not resolved"
                )
            ),
            f"full segment duration = {_format_number(duration_ms)} ms",
            (
                f"DC offset removed = {_format_number(dc_offset_v * 1000.0)} mV"
                if dc_offset_v is not None
                else "DC offset removed; value not recorded"
            ),
            (
                "interpolated over-range samples: "
                f"A={_format_number(clip_a, 4)}, D={_format_number(clip_d, 4)}"
                if clip_a is not None or clip_d is not None
                else "interpolated over-range count not recorded"
            ),
            (
                f"ADC codes: A={_format_number(codes_a, 4)}, "
                f"D={_format_number(codes_d, 4)} ({quantization_status})"
                if codes_a is not None or codes_d is not None
                else "ADC-code occupancy not recorded"
            ),
        ]
        if qc_flags:
            annotation_lines.append(f"QC flags: {qc_flags}")
        ax.text(
            0.02,
            0.03,
            "\n".join(annotation_lines),
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=8,
            bbox={
                "boxstyle": "square,pad=0.28",
                "facecolor": "white",
                "edgecolor": "#B5B5B5",
                "alpha": 0.88,
            },
        )
        ax.legend(
            [voltage_line, charge_line],
            ["Applied voltage", "Monitor charge"],
            loc="upper right",
        )
        values.append(
            f"{level}% nominal argon ladder segment: "
            f"measured f0={_format_number(f0_hz / 1000.0 if f0_hz is not None else None)} kHz; "
            f"measured burst={_format_number(burst_hz / 1000.0 if burst_hz is not None else None)} kHz; "
            f"full duration={_format_number(duration_ms)} ms"
            + (
                f"; displayed window={_format_number(1000.0 * np.ptp(time_s))} ms "
                f"({_format_number(displayed_bursts, 2)} measured burst periods)"
                if displayed_bursts is not None
                else ""
            )
            + "; "
            f"removed Channel-D offset={_format_number(dc_offset_v * 1000.0 if dc_offset_v is not None else None)} mV; "
            f"interpolated samples A={_format_number(clip_a, 4)}, "
            f"D={_format_number(clip_d, 4)}; "
            f"ADC codes A={_format_number(codes_a, 4)}, "
            f"D={_format_number(codes_d, 4)} "
            f"({quantization_status or 'not recorded'}); "
            f"QC flags={qc_flags or 'none'}."
        )

    paths = _save_figure(fig, inputs.root, FIGURE_STEMS[0])
    if not sources:
        sources.append(
            "Representative arrays are read only from `figure_data.npz`; "
            "`manifest.csv` supplied no matching path."
        )
    caption = _write_caption(
        inputs.root,
        1,
        (
            "Representative records from the 4 kHz-burst argon "
            "ladder show the applied voltage and offset-corrected "
            "measuring-capacitor charge at a sub-breakdown and an "
            "above-breakdown level. Channel A measures a much faster carrier, "
            "so these records are carrier-transfer data—not low-frequency "
            "geometric anchors. The plotted charge is Q = C_m V_D; it is not a "
            "capacitance estimate."
        ),
        values=values,
        sources=sources,
        notes=notes,
    )
    return paths, caption


def _argon4k_subbreakdown_stat(inputs: FigureInputs) -> Stat | None:
    rows = [
        row
        for row in inputs.per_capture
        if _medium(row) == "argon"
        and _nominal_frequency_matches(row, 4.0)
        and _is_lf(row)
        and _is_subbreakdown(row)
        and not _is_excluded(row)
    ]
    stat = _stat_across_rows(rows, ("Cline_pF", "Ccell_pF", "C_cell_geom_pF"))
    if stat is not None:
        return stat
    summary = [
        row
        for row in _summary_records(inputs.condition_summary)
        if _medium(row) == "argon"
        and _nominal_frequency_matches(row, 4.0)
        and _is_lf(row)
        and _is_subbreakdown(row)
        and not _is_excluded(row)
    ]
    return _stat_across_rows(
        summary, ("Cline_pF", "Ccell_pF", "C_cell_geom_pF")
    )


def _argon4k_115_rows(inputs: FigureInputs) -> list[Mapping[str, object]]:
    rows: list[Mapping[str, object]] = []
    for source in (
        inputs.discharge,
        inputs.per_capture,
        _summary_records(inputs.condition_summary),
    ):
        matches = [
            row
            for row in source
            if _medium(row) == "argon"
            and _nominal_frequency_matches(row, 4.0)
            and _level_pct(row) is not None
            and abs(float(_level_pct(row)) - 115.0) <= 0.51
        ]
        if matches:
            rows.extend(row for row in matches if not _is_excluded(row))
            if rows:
                break
    return rows


def make_fig2(inputs: FigureInputs) -> tuple[list[Path], Path]:
    fig, axes_array = _new_figure()
    ax = axes_array[0, 0]
    levels = (40, 60, 75, 90, 100, 105, 115)
    styles = ("-", "--", "-.", ":", "-", "--", "-.")
    markers = ("o", "s", "^", "D", "v", "P", "X")
    plotted_levels: list[int] = []
    sources: list[str] = []
    notes: list[str] = []
    values: list[str] = []

    for index, level in enumerate(levels):
        prefix = f"argon4k_{level}"
        voltage = _npz_array(
            inputs.figure_data,
            f"{prefix}_loop_V_kV",
            f"{prefix}_loop_voltage_kV",
        )
        charge = _npz_array(
            inputs.figure_data,
            f"{prefix}_loop_Q_nC",
            f"{prefix}_loop_charge_nC",
        )
        if voltage is None or charge is None:
            continue
        voltage, charge = _paired_finite(voltage, charge)
        if voltage.size < 3:
            continue
        voltage, charge = _downsample((voltage, charge), 4_000)
        alpha = 0.48 + 0.07 * index
        ax.plot(
            voltage,
            charge,
            color=_medium_color("argon"),
            linestyle=styles[index],
            marker=markers[index],
            markevery=max(1, len(voltage) // 18),
            markerfacecolor="white" if level < 100 else _medium_color("argon"),
            markeredgewidth=0.8,
            alpha=min(alpha, 1.0),
            label=f"{level}% nominal",
        )
        plotted_levels.append(level)
        representative = _manifest_representative(inputs, float(level))
        if representative is not None:
            sources.append(f"{level}% loop: `{_provenance(representative)}`.")

    ax.set_xlabel("Applied voltage (kV)")
    ax.set_ylabel("Charge, Q (nC)")
    _panel_label(ax, "(a) Cycle-averaged argon ladder")
    context_rows = [
        row
        for row in inputs.per_capture
        if _medium(row) == "argon"
        and _nominal_frequency_matches(row, 4.0)
        and not _is_excluded(row)
    ]
    context_f0 = _aggregate(
        value
        for row in context_rows
        for value in [_frequency_khz(row)]
        if value is not None
    )
    context_burst = _aggregate(
        value
        for row in context_rows
        for value in [_pick_number(row, "burst_Hz", "burst_rate_Hz")]
        if value is not None
    )
    context_bands = sorted({_band(row) for row in context_rows})
    if plotted_levels:
        ax.legend(loc="best", ncol=2)
        context_lines = ["burst-frequency folder label: 4 kHz"]
        if context_f0 is not None:
            context_lines.append(
                f"measured carrier f0 ≈ {_format_number(context_f0.value)} kHz"
            )
        if context_burst is not None:
            context_lines.append(
                f"measured burst ≈ {_format_number(context_burst.value / 1000.0)} kHz"
            )
        if context_bands:
            context_lines.append("band: " + ", ".join(context_bands))
        ax.text(
            0.99,
            0.98,
            "\n".join(context_lines),
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=8,
            bbox={
                "boxstyle": "square,pad=0.28",
                "facecolor": "white",
                "edgecolor": "#B5B5B5",
                "alpha": 0.9,
            },
        )
    else:
        _empty_panel(
            ax,
            "no argon4k_*_loop_V_kV / _loop_Q_nC arrays in figure_data.npz",
            xlabel="Applied voltage (kV)",
            ylabel="Charge, Q (nC)",
        )
        notes.append(
            "No cycle-averaged argon 4 kHz loop arrays were available; "
            "the panel is intentionally empty."
        )

    annotation: list[str] = []
    sub_stat = _argon4k_subbreakdown_stat(inputs)
    if sub_stat is not None:
        annotation.append(
            f"sub-breakdown Ccell = {_format_number(sub_stat.value)} pF "
            f"[LF-geometric], N={sub_stat.n if sub_stat.n is not None else 'not recorded'}"
        )
        values.append(
            "Pooled sub-breakdown line slope: "
            f"{_format_stat(sub_stat, 'pF')} [LF-geometric], from "
            "`per_capture_metrics.csv` or `condition_summary.csv`."
        )

    above_rows = _argon4k_115_rows(inputs)
    above_lf_rows = [row for row in above_rows if _is_lf(row)]
    cd = _stat_across_rows(
        above_lf_rows, ("Cd_pF", "C_d_pF", "branch_Cd_pF")
    )
    ccell = _stat_across_rows(
        above_lf_rows, ("Ccell_pF", "C_cell_pF", "branch_Ccell_pF")
    )
    dq = _stat_across_rows(
        above_rows, ("dQ_cycle_nC", "dQ_raw_nC", "deltaQ_nC")
    )
    if cd is not None:
        annotation.append(
            f"115% Cd branch = {_format_number(cd.value)} pF [LF-geometric]"
        )
        values.append(
            f"115% dielectric/on-branch slope: {_format_stat(cd, 'pF')} "
            "[LF-geometric], from the above-breakdown argon 4 kHz summary."
        )
    if ccell is not None:
        annotation.append(
            f"115% Ccell branch = {_format_number(ccell.value)} pF [LF-geometric]"
        )
        values.append(
            f"115% cell/off-branch slope: {_format_stat(ccell, 'pF')} "
            "[LF-geometric], from the above-breakdown argon 4 kHz summary."
        )
    if dq is not None:
        dq_band = next(
            (_band(row) for row in above_rows if _band(row) != "band-unassigned"),
            "band-unassigned",
        )
        annotation.append(
            f"115% \N{GREEK CAPITAL LETTER DELTA}Q = "
            f"{_format_number(dq.value)} nC/cycle [{dq_band} acquisition]"
        )
        values.append(
            f"115% raw transferred charge: {_format_stat(dq, 'nC/cycle')} "
            f"[{dq_band} acquisition]."
        )
    if annotation and plotted_levels:
        ax.text(
            0.018,
            0.03,
            "\n".join(annotation),
            transform=ax.transAxes,
            va="bottom",
            ha="left",
            fontsize=8,
            bbox={
                "boxstyle": "square,pad=0.3",
                "facecolor": "white",
                "edgecolor": "#B5B5B5",
                "alpha": 0.9,
            },
        )
    elif plotted_levels:
        notes.append(
            "Loop arrays were plotted, but no qualifying LF-geometric slope was "
            "available for annotation."
        )

    if plotted_levels:
        values.insert(
            0,
            "Nominal ladder levels plotted: "
            + ", ".join(f"{level}%" for level in plotted_levels)
            + ". The 4 kHz text is the burst/folder label; measured carrier "
            "frequencies remain capture-derived.",
        )
    if context_f0 is not None:
        values.append(
            "4 kHz-burst argon ladder carrier: "
            f"{_format_stat(context_f0, 'kHz')} "
            f"[{', '.join(context_bands) if context_bands else 'band-unassigned'}]."
        )
    notes.append(
        "The stair-stepped polygons expose PicoScope quantization and are shown "
        "as QC/context only. They are not treated as textbook low-frequency "
        "parallelograms or as geometric-capacitance evidence."
    )
    paths = _save_figure(fig, inputs.root, FIGURE_STEMS[1])
    caption = _write_caption(
        inputs.root,
        2,
        (
            "Cycle-averaged Q-V trajectories across the 4 kHz-burst "
            "burst-labeled argon voltage ladder show how the measured "
            "carrier-transfer loop changes with amplitude. The measured "
            "electrical carrier is much faster than 4 kHz, and the quantized "
            "trajectories are not promoted to low-frequency geometric evidence. "
            "Capacitance annotations remain restricted to rows explicitly tagged "
            "LF-geometric."
        ),
        values=values,
        sources=sources or ["Trace provenance was unavailable in `manifest.csv`."],
        notes=notes,
    )
    return paths, caption


def _anchor_points(inputs: FigureInputs) -> list[AnchorPoint]:
    """Return one sub-breakdown LF summary per medium/frequency/level."""

    summary_rows = _summary_records(inputs.condition_summary)
    capture_groups: dict[
        tuple[str, float, float | None, str], list[Mapping[str, object]]
    ] = defaultdict(list)
    for row in inputs.per_capture:
        frequency = _frequency_khz(row)
        if (
            frequency is None
            or not _is_lf(row)
            or not _is_subbreakdown(row)
            or _is_excluded(row)
        ):
            continue
        key = (
            _medium(row),
            round(frequency, 6),
            _level_pct(row),
            _pick_text(row, "cond", "condition"),
        )
        capture_groups[key].append(row)

    points: dict[
        tuple[str, float, float | None, str], AnchorPoint
    ] = {}
    for row in summary_rows:
        frequency = _frequency_khz(row)
        if (
            frequency is None
            or not _is_lf(row)
            or not _is_subbreakdown(row)
            or _is_excluded(row)
        ):
            continue
        capacitance = _stat(
            row, ("Cline_pF", "C_cell_geom_pF", "Ccell_pF")
        )
        amplitude = _stat(row, ("Vamp_kV", "V_amp_kV", "amplitude_kV"))
        if capacitance is None:
            continue
        condition = _pick_text(row, "cond", "condition")
        key = (
            _medium(row),
            round(frequency, 6),
            _level_pct(row),
            condition,
        )
        matching_capture_rows = capture_groups.get(key, [])
        if amplitude is None and matching_capture_rows:
            amplitude = _stat_across_rows(
                matching_capture_rows,
                ("Vamp_kV", "V_amp_kV", "amplitude_kV"),
            )
        if amplitude is None:
            continue
        syst = _pick_number(row, "syst_frac", "systematic_fraction")
        if syst is None and matching_capture_rows:
            syst_values = [
                value
                for value in (
                    _pick_number(item, "syst_frac", "systematic_fraction")
                    for item in matching_capture_rows
                )
                if value is not None
            ]
            syst = float(np.median(syst_values)) if syst_values else None
        points[key] = AnchorPoint(
            medium=_medium(row),
            frequency_khz=float(frequency),
            level_pct=_level_pct(row),
            amplitude_kv=amplitude.value,
            capacitance=capacitance,
            syst_frac=syst,
            provenance=_provenance(row),
        )

    for key, rows in capture_groups.items():
        if key in points:
            continue
        capacitance = _stat_across_rows(
            rows, ("Cline_pF", "C_cell_geom_pF", "Ccell_pF")
        )
        amplitude = _stat_across_rows(
            rows, ("Vamp_kV", "V_amp_kV", "amplitude_kV")
        )
        if capacitance is None or amplitude is None:
            continue
        syst_values = [
            value
            for value in (
                _pick_number(row, "syst_frac", "systematic_fraction")
                for row in rows
            )
            if value is not None
        ]
        points[key] = AnchorPoint(
            medium=key[0],
            frequency_khz=key[1],
            level_pct=key[2],
            amplitude_kv=amplitude.value,
            capacitance=capacitance,
            syst_frac=float(np.median(syst_values)) if syst_values else None,
            provenance="; ".join(dict.fromkeys(_provenance(row) for row in rows)),
        )
    return sorted(
        points.values(),
        key=lambda point: (
            point.frequency_khz,
            _medium_label(point.medium),
            math.inf if point.level_pct is None else point.level_pct,
            point.amplitude_kv,
        ),
    )


def _frequency_groups(
    points: Sequence[AnchorPoint],
) -> list[tuple[float, list[AnchorPoint]]]:
    groups: dict[float, list[AnchorPoint]] = defaultdict(list)
    for point in points:
        groups[round(point.frequency_khz, 3)].append(point)
    return [(frequency, groups[frequency]) for frequency in sorted(groups)]


def _annotation_position(
    ax: plt.Axes, x: float, y: float, label: str, color: str
) -> None:
    ax.annotate(
        label,
        xy=(x, y),
        xytext=(3, 4),
        textcoords="offset points",
        fontsize=7.3,
        color=color,
    )


def make_fig3(inputs: FigureInputs) -> tuple[list[Path], Path]:
    points = _anchor_points(inputs)
    frequency_groups = _frequency_groups(points)
    values: list[str] = []
    sources: list[str] = []
    notes: list[str] = []

    if not frequency_groups:
        fig, axes_array = _new_figure()
        ax = axes_array[0, 0]
        _panel_label(ax, "(a) LF-geometric anchor")
        _empty_panel(
            ax,
            "no sub-breakdown rows tagged LF-geometric with Cline and measured amplitude",
            xlabel="Measured amplitude (kV)",
            ylabel="Line capacitance, Cline (pF; LF-geometric)",
        )
        notes.append(
            "No operational LF-geometric anchor was available. No carrier-transfer "
            "capacitance was substituted."
        )
    else:
        count = len(frequency_groups)
        ncols = min(3, count)
        nrows = int(math.ceil(count / ncols))
        fig, axes_array = _new_figure(nrows, ncols, sharey=True)
        flat_axes = list(axes_array.reshape(-1))
        legend_handles: dict[str, Line2D] = {}
        for panel_index, ((frequency, group), ax) in enumerate(
            zip(frequency_groups, flat_axes)
        ):
            _panel_label(
                ax,
                f"({chr(97 + panel_index)}) measured f0 = {_format_number(frequency)} kHz",
            )
            ax.set_xlabel("Measured amplitude (kV)")
            if panel_index % ncols == 0:
                ax.set_ylabel("Line capacitance, Cline (pF; LF-geometric)")

            by_medium: dict[str, list[AnchorPoint]] = defaultdict(list)
            for point in group:
                by_medium[point.medium].append(point)
            for medium in sorted(by_medium, key=_medium_label):
                medium_points = sorted(
                    by_medium[medium], key=lambda point: point.amplitude_kv
                )
                x = [point.amplitude_kv for point in medium_points]
                stats = [point.capacitance for point in medium_points]
                color = _medium_color(medium)

                # Systematic calibration bands are drawn independently of the
                # percentile interval so the two uncertainties are not combined.
                for point in medium_points:
                    if point.syst_frac is not None and point.syst_frac >= 0:
                        half = abs(point.capacitance.value * point.syst_frac)
                        ax.vlines(
                            point.amplitude_kv,
                            point.capacitance.value - half,
                            point.capacitance.value + half,
                            color=color,
                            linewidth=6,
                            alpha=0.14,
                            zorder=1,
                        )
                _safe_errorbar(
                    ax,
                    x,
                    stats,
                    color=color,
                    marker="o",
                    markerfacecolor="white",
                    markeredgecolor=color,
                    linestyle="-",
                    zorder=3,
                )
                label = _medium_label(medium)
                legend_handles[label] = Line2D(
                    [0],
                    [0],
                    color=color,
                    marker="o",
                    markerfacecolor="white",
                    label=label,
                )
                for point in medium_points:
                    n_label = (
                        str(point.capacitance.n)
                        if point.capacitance.n is not None
                        else "?"
                    )
                    _annotation_position(
                        ax,
                        point.amplitude_kv,
                        point.capacitance.value,
                        f"N={n_label}",
                        color,
                    )
                    syst_text = (
                        f"; separate systematic ±{100.0 * point.syst_frac:.2g}%"
                        if point.syst_frac is not None
                        else "; systematic fraction not recorded"
                    )
                    values.append(
                        f"{_medium_label(medium)}, measured f0={_format_number(point.frequency_khz)} kHz, "
                        f"amplitude={_format_number(point.amplitude_kv)} kV"
                        + (
                            f", nominal level={_format_number(point.level_pct)}%"
                            if point.level_pct is not None
                            else ""
                        )
                        + f": Cline={_format_stat(point.capacitance, 'pF')} "
                        f"[LF-geometric]{syst_text}."
                    )
                    sources.append(
                        f"{_medium_label(medium)}, {_format_number(point.frequency_khz)} kHz: "
                        f"`{point.provenance}`."
                    )

                # Print the pooled value per medium/frequency as requested.  Use
                # capture-level values when available; otherwise pool the plotted
                # condition medians and state that limitation in the caption.
                raw_values = [
                    stat.value
                    for row in inputs.per_capture
                    if _medium(row) == medium
                    and _is_lf(row)
                    and _is_subbreakdown(row)
                    and not _is_excluded(row)
                    and _frequency_khz(row) is not None
                    and abs(_frequency_khz(row) - frequency)
                    <= max(0.05, 0.02 * frequency)
                    for stat in [
                        _stat(
                            row,
                            ("Cline_pF", "C_cell_geom_pF", "Ccell_pF"),
                        )
                    ]
                    if stat is not None
                ]
                pooled = (
                    _aggregate(raw_values)
                    if raw_values
                    else _aggregate(
                        point.capacitance.value for point in medium_points
                    )
                )
                if pooled is not None:
                    position = 0.96 - 0.075 * list(
                        sorted(by_medium, key=_medium_label)
                    ).index(medium)
                    ax.text(
                        0.98,
                        position,
                        f"{label}: {_format_number(pooled.value)} pF "
                        f"[LF-geometric], N={pooled.n}",
                        transform=ax.transAxes,
                        ha="right",
                        va="top",
                        color=color,
                        fontsize=7.6,
                    )
                    values.append(
                        f"Pooled {label}, measured f0≈{_format_number(frequency)} kHz: "
                        f"{_format_stat(pooled, 'pF')} [LF-geometric]."
                    )
            ax.margins(x=0.08, y=0.15)

        for unused in flat_axes[len(frequency_groups) :]:
            unused.set_visible(False)
        if legend_handles:
            fig.legend(
                list(legend_handles.values()),
                list(legend_handles),
                loc="outside lower center",
                ncol=min(4, len(legend_handles)),
            )
        notes.append(
            "Vertical percentile bars are statistical [2.5, 97.5] intervals. "
            "Broad translucent bars, where present, are the separate multiplicative "
            "systematic calibration uncertainty."
        )

    paths = _save_figure(fig, inputs.root, FIGURE_STEMS[2])
    caption = _write_caption(
        inputs.root,
        3,
        (
            "Sub-breakdown line capacitance is plotted against measured voltage "
            "amplitude separately at each observed low-frequency anchor. Only "
            "rows explicitly tagged LF-geometric are included; carrier-transfer "
            "values are excluded."
        ),
        values=values,
        sources=sources or ["`per_capture_metrics.csv` and `condition_summary.csv`."],
        notes=notes,
    )
    return paths, caption


def _dispersion_source(row: Mapping[str, object]) -> str:
    value = _normalize_key(
        _pick_text(row, "source_type", "dataset_type", "source", "series")
    )
    if not value:
        return "unspecified_source"
    return value


def _source_label(source: str) -> str:
    mapping = {
        "lf_anchor": "New LF anchor",
        "lf_geometric": "New LF anchor",
        "july_7_8": "July 7/8",
        "july7_8": "July 7/8",
        "lissajousfigure": "July 7/8",
        "july7_8_operational_fundamental": "July 7–8 operational",
        "multiline": "7_20 multiline H(f)",
        "7_20_multiline": "7_20 multiline H(f)",
        "carrier_v1_1": "v1.1 operational",
        "v1_1": "v1.1 operational",
        "v1_1_carrier": "v1.1 carrier",
        "v1_1_active_cd": "v1.1 active Cd",
        "v1_1_active_secant": "v1.1 active secant",
        "v1_2_dry_fixture": "v1.2 dry fixture",
        "dry_fixture": "Dry fixture",
        "rlc_model": "Series-RLC model†",
        "series_rlc_model": "Series-RLC model†",
        "model": "Series-RLC model†",
        "unspecified_source": "Unspecified source",
    }
    return mapping.get(source, source.replace("_", " ").title())


def _source_marker(source: str) -> str:
    if "dry" in source:
        return "P"
    if "carrier" in source or "v1_1" in source:
        return "^"
    if "multiline" in source or "7_20" in source:
        return "D"
    if "july" in source or "lissajousfigure" in source:
        return "s"
    return "o"


def _dispersion_marker(source: str, series: str) -> str:
    if source in {"7_20_multiline", "multiline"}:
        normalized = _normalize_key(series)
        if normalized.startswith("4khz"):
            return "D"
        if normalized.startswith("10khz"):
            return "v"
        if normalized.startswith("20khz"):
            return "^"
    return _source_marker(source)


def _compact_dispersion_identity(
    source: str,
    medium: str,
    series: str,
) -> str:
    if source in {"7_20_multiline", "multiline"}:
        match = re.match(r"(\d+)\s*k?hz", series, flags=re.IGNORECASE)
        drive = f"{match.group(1)} kHz drive" if match else series
        return f"7_20 {drive}".strip()
    if source == "july7_8_operational_fundamental":
        return f"{_medium_label(medium)} · Jul 7/8"
    if source in {"series_rlc_model", "rlc_model", "model"}:
        return "Series-RLC model†"
    if source == "v1_2_dry_fixture":
        return "Dry fixture"
    if source == "v1_1_active_cd":
        return "v1.1 active Cd"
    if source == "v1_1_active_secant":
        return "v1.1 active secant"
    if source == "v1_1_carrier":
        return f"v1.1 · {_medium_label(medium)}"
    return _source_label(source)


def _is_model_row(row: Mapping[str, object]) -> bool:
    source = _dispersion_source(row)
    return "model" in source or "rlc" in source


def _dispersion_plot_eligible(row: Mapping[str, object]) -> bool:
    """Keep sparse multiline diagnostics in the table but off plotted summaries."""

    if _dispersion_source(row) not in {"7_20_multiline", "multiline"}:
        return True
    count = _pick_number(row, "N", "n", "capture_count")
    return count is not None and count >= 3


def _dispersion_group_color(
    source: str, medium: str, band: str
) -> str:
    if "model" in source or "rlc" in source:
        return OKABE_ITO["orange"]
    if medium != "unknown":
        return _medium_color(medium)
    if "dry" in source:
        return OKABE_ITO["black"]
    if band == "LF-geometric":
        return OKABE_ITO["bluish_green"]
    if band == "carrier-transfer":
        return OKABE_ITO["vermillion"]
    return OKABE_ITO["orange"]


def _model_resonance_khz(rows: Sequence[Mapping[str, object]]) -> float | None:
    for row in rows:
        value = _pick_number(
            row,
            "f_res_kHz",
            "f_res_khz",
            "resonance_kHz",
            "resonance_frequency_kHz",
        )
        if value is not None:
            return value
        hz = _pick_number(row, "f_res_Hz", "resonance_Hz")
        if hz is not None:
            return hz / 1000.0
    model_points = [
        (_frequency_khz(row), _pick_number(row, "Cmag_pF", "C_apparent_pF"))
        for row in rows
        if _is_model_row(row)
    ]
    model_points = [
        (frequency, capacitance)
        for frequency, capacitance in model_points
        if frequency is not None and capacitance is not None
    ]
    if len(model_points) >= 3:
        return max(model_points, key=lambda pair: pair[1])[0]
    return None


def _carrier_span(
    rows: Sequence[Mapping[str, object]],
    supplemental: Sequence[Mapping[str, object]] = (),
) -> tuple[float, float] | None:
    for row in rows:
        low = _pick_number(
            row,
            "carrier_band_low_kHz",
            "carrier_low_kHz",
            "band_low_kHz",
        )
        high = _pick_number(
            row,
            "carrier_band_high_kHz",
            "carrier_high_kHz",
            "band_high_kHz",
        )
        if low is not None and high is not None and high >= low:
            return low, high
    frequencies = sorted(
        {
            round(frequency, 6)
            for row in rows
            if _is_transfer(row)
            and (
                "carrier" in _dispersion_source(row)
                or "v1_1" in _dispersion_source(row)
                or "operational" in _dispersion_source(row)
            )
            for frequency in [_frequency_khz(row)]
            if frequency is not None and frequency > 0
        }
    )
    if len(frequencies) >= 2:
        return frequencies[0], frequencies[-1]
    supplemental_frequencies = sorted(
        {
            round(frequency, 6)
            for row in supplemental
            if _is_transfer(row)
            and "synthesis"
            in _normalize_key(_pick_text(row, "dataset_type", "source_type"))
            for frequency in [_frequency_khz(row)]
            if frequency is not None and frequency > 0
        }
    )
    combined = sorted(set(frequencies + supplemental_frequencies))
    if len(combined) >= 2:
        # The band remains data-derived: historical transfer points and measured
        # synthesis carriers supply the displayed lower/upper bounds.
        return combined[0], combined[-1]
    return None


def make_fig4(inputs: FigureInputs) -> tuple[list[Path], Path]:
    fig, axes_array = _new_figure(2, 1, sharex=True)
    top, bottom = axes_array[:, 0]
    _panel_label(top, "(a) Apparent-capacitance magnitude")
    _panel_label(bottom, "(b) Complex response / phase convention")

    rows = [
        row
        for row in inputs.dispersion
        if not _is_hard_excluded(row)
        and _dispersion_plot_eligible(row)
        and _frequency_khz(row) is not None
        and _frequency_khz(row) > 0
    ]
    data_rows = [
        row
        for row in rows
        if _is_model_row(row)
        or _band(row) in {"LF-geometric", "carrier-transfer"}
    ]
    groups: dict[
        tuple[str, str, str, str], list[Mapping[str, object]]
    ] = defaultdict(list)
    for row in data_rows:
        source = _dispersion_source(row)
        series = (
            _provenance(row)
            if source in {"7_20_multiline", "multiline"}
            else ""
        )
        groups[
            (
                source,
                "chain-model" if _is_model_row(row) else _band(row),
                _medium(row),
                series,
            )
        ].append(row)

    values: list[str] = []
    sources: list[str] = []
    notes: list[str] = []
    plotted_top = False
    plotted_bottom = False

    for group_key in sorted(groups):
        source, band, medium, series = group_key
        group_rows = sorted(
            groups[group_key],
            key=lambda row: _frequency_khz(row) or math.inf,
        )
        color = _dispersion_group_color(source, medium, band)
        source_label = _source_label(source)
        compact_identity = _compact_dispersion_identity(
            source,
            medium,
            series,
        )
        is_model = any(_is_model_row(row) for row in group_rows)
        qc_limited = any(
            "quantization_limited" in _flags(row) for row in group_rows
        )
        band_label = "transfer model†" if is_model else band
        if qc_limited:
            band_label += "; QC-limited"
        label = f"{compact_identity} [{band_label}]"
        frequency = []
        magnitude_stats: list[Stat] = []
        magnitude_rows: list[Mapping[str, object]] = []
        for row in group_rows:
            stat = _stat(
                row,
                (
                    "Cmag_pF",
                    "C_apparent_pF",
                    "Cline_pF",
                    "magnitude_pF",
                ),
            )
            f_khz = _frequency_khz(row)
            if stat is None or stat.value <= 0 or f_khz is None:
                continue
            frequency.append(f_khz)
            magnitude_stats.append(stat)
            magnitude_rows.append(row)
        if frequency:
            if is_model:
                top.plot(
                    frequency,
                    [stat.value for stat in magnitude_stats],
                    color=color,
                    linestyle="-",
                    linewidth=2.0,
                    label=label,
                    zorder=1,
                )
            else:
                _safe_errorbar(
                    top,
                    frequency,
                    magnitude_stats,
                    color=color,
                    marker=_dispersion_marker(source, series),
                    markerfacecolor=(
                        "white" if band == "LF-geometric" else color
                    ),
                    markeredgecolor=color,
                    linestyle=":" if qc_limited else "-",
                    label=label,
                    zorder=3,
                    alpha=0.58 if qc_limited else 1.0,
                )
                if source in {"7_20_multiline", "multiline"}:
                    values.append(
                        f"{source_label}, {series or _medium_label(medium)}: "
                        f"{len(frequency)} plotted spectral lines from "
                        f"{_format_number(min(frequency))} to "
                        f"{_format_number(max(frequency))} kHz; "
                        f"|C_app| range "
                        f"{_format_number(min(stat.value for stat in magnitude_stats))}"
                        "–"
                        f"{_format_number(max(stat.value for stat in magnitude_stats))} "
                        f"pF [{band}; QC-limited—not accepted as an LF anchor]."
                    )
                    sources.append(
                        f"{source_label}: `{series or _provenance(group_rows[0])}`."
                    )
                else:
                    for f_khz, stat, row in zip(
                        frequency, magnitude_stats, magnitude_rows
                    ):
                        n_text = f", N={stat.n}" if stat.n is not None else ""
                        phase = _pick_number(row, "phase_deg", "phase_degrees")
                        values.append(
                            f"{source_label}, {_medium_label(medium)}, "
                            f"f={_format_number(f_khz)} kHz: "
                            f"|C_app|={_format_number(stat.value)} pF [{band}]"
                            + (
                                f", phase={_format_number(phase)}°"
                                if phase is not None
                                else ""
                            )
                            + n_text
                            + "."
                        )
                        sources.append(
                            f"{source_label}: `{_provenance(row)}`."
                        )
            plotted_top = True

        real_frequency: list[float] = []
        real_stats: list[Stat] = []
        imag_frequency: list[float] = []
        imag_stats: list[Stat] = []
        phase_frequency: list[float] = []
        phase_stats: list[Stat] = []
        for row in group_rows:
            f_khz = _frequency_khz(row)
            if f_khz is None:
                continue
            real = _stat(row, ("Creal_pF", "C_real_pF", "real_pF"))
            imag = _stat(row, ("Cimag_pF", "C_imag_pF", "imag_pF"))
            phase = _stat(row, ("phase_deg", "phase_degrees"))
            if real is not None:
                real_frequency.append(f_khz)
                real_stats.append(real)
            if imag is not None:
                imag_frequency.append(f_khz)
                imag_stats.append(imag)
            if phase is not None:
                phase_frequency.append(f_khz)
                phase_stats.append(phase)

        if real_frequency or imag_frequency:
            if real_frequency:
                bottom.plot(
                    real_frequency,
                    [stat.value for stat in real_stats],
                    color=color,
                    marker=(
                        _dispersion_marker(source, series)
                        if not is_model
                        else None
                    ),
                    markerfacecolor="white",
                    linestyle="-",
                    label=f"{compact_identity} [{band_label}]",
                    alpha=0.58 if qc_limited else 1.0,
                )
            if imag_frequency:
                bottom.plot(
                    imag_frequency,
                    [stat.value for stat in imag_stats],
                    color=color,
                    marker=(
                        _dispersion_marker(source, series)
                        if not is_model
                        else None
                    ),
                    markerfacecolor="white",
                    linestyle="--",
                    label="_nolegend_",
                    alpha=0.5 if qc_limited else 0.85,
                )
            plotted_bottom = True
        elif phase_frequency:
            bottom.plot(
                phase_frequency,
                [stat.value for stat in phase_stats],
                color=color,
                marker=_source_marker(source) if not is_model else None,
                markerfacecolor="white",
                linestyle="-",
                label=f"{compact_identity} [{band_label}]",
            )
            plotted_bottom = True

        if is_model and frequency:
            values.append(
                f"{source_label}: {len(frequency)} evaluated grid points from "
                f"{_format_number(min(frequency))} to {_format_number(max(frequency))} kHz; "
                "the curve is an order-of-magnitude chain-transfer model†."
            )
            sources.extend(
                f"{source_label}: `{_provenance(row)}`."
                for row in group_rows[:1]
            )

    top.set_xscale("log")
    bottom.set_xscale("log")
    top.set_ylabel("|C_app| (pF)")
    bottom.set_xlabel("Measured frequency (kHz)")
    if any(
        _stat(row, ("Creal_pF", "C_real_pF", "real_pF")) is not None
        or _stat(row, ("Cimag_pF", "C_imag_pF", "imag_pF")) is not None
        for row in data_rows
    ):
        bottom.set_ylabel("Complex C_app component (pF)")
        bottom.axhline(0.0, color="#666666", linewidth=0.8, zorder=0)
    else:
        bottom.set_ylabel("Phase (degrees)")
        bottom.axhline(0.0, color="#666666", linewidth=0.8, zorder=0)

    carrier_span = _carrier_span(data_rows, inputs.per_capture)
    if carrier_span is not None:
        low, high = carrier_span
        for ax in (top, bottom):
            ax.axvspan(
                low,
                high,
                color=OKABE_ITO["yellow"],
                alpha=0.22,
                zorder=0,
            )
        top.text(
            math.sqrt(low * high),
            0.97,
            f"carrier band\n{_format_number(low)}–{_format_number(high)} kHz",
            transform=top.get_xaxis_transform(),
            ha="center",
            va="top",
            fontsize=7.5,
        )
        values.append(
            f"Carrier-band shading spans {_format_number(low)}–{_format_number(high)} kHz, "
            "derived from the carrier-transfer rows in `dispersion_master.csv`."
        )
    else:
        notes.append(
            "No carrier-band bounds could be derived from the dispersion table, "
            "so no carrier-band shading was added."
        )

    resonance = _model_resonance_khz(data_rows)
    if resonance is not None:
        for ax in (top, bottom):
            ax.axvline(
                resonance,
                color=OKABE_ITO["vermillion"],
                linestyle=":",
                linewidth=1.2,
            )
        top.text(
            resonance,
            0.76,
            f"f_res={_format_number(resonance)} kHz†",
            transform=top.get_xaxis_transform(),
            rotation=90,
            ha="right",
            va="top",
            color=OKABE_ITO["vermillion"],
            fontsize=8,
        )
        values.append(
            f"Series-resonance marker: f_res={_format_number(resonance)} kHz†, "
            "taken from the table or the maximum of its evaluated RLC-model grid."
        )
    else:
        notes.append(
            "No fitted resonance field or evaluable model peak was present; "
            "no resonance marker was fabricated."
        )

    if plotted_top:
        top.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, 0.91),
            ncol=4,
            fontsize=6.4,
        )
        positive_frequencies = [
            _frequency_khz(row)
            for row in data_rows
            if _frequency_khz(row) is not None and _frequency_khz(row) > 0
        ]
        if positive_frequencies:
            minimum = min(positive_frequencies)
            maximum = max(positive_frequencies)
            if math.isclose(minimum, maximum):
                top.set_xlim(minimum / 1.6, maximum * 1.6)
            else:
                top.set_xlim(minimum / 1.12, maximum * 1.12)
    else:
        _empty_panel(
            top,
            "dispersion_master.csv has no valid, band-tagged capacitance magnitudes",
            xlabel="Measured frequency (kHz)",
            ylabel="|C_app| (pF)",
        )
    if plotted_bottom:
        handles, labels = bottom.get_legend_handles_labels()
        handles.extend(
            [
                Line2D(
                    [0],
                    [0],
                    color=OKABE_ITO["black"],
                    linestyle="-",
                    label="solid = Re(C_app)",
                ),
                Line2D(
                    [0],
                    [0],
                    color=OKABE_ITO["black"],
                    linestyle="--",
                    label="dashed = Im(C_app)",
                ),
            ]
        )
        labels.extend(("solid = Re(C_app)", "dashed = Im(C_app)"))
        bottom.legend(handles, labels, loc="best", ncol=4, fontsize=6.4)
    else:
        _empty_panel(
            bottom,
            "no phase or complex response columns were available",
            xlabel="Measured frequency (kHz)",
            ylabel="Complex response / phase",
        )
    notes.append(
        "LF-geometric and carrier-transfer observations are not pooled. "
        "Contaminated, sub-LSB, and explicitly excluded rows are omitted. "
        "The 7_20 multiline context additionally requires N >= 3, so isolated "
        "fixture lines remain auditable in the CSV without appearing on the plot. "
        "Quantization-limited 7_20 multiline rows are retained as translucent "
        "open-marker context because they document the resonance trend, but "
        "they are not used as geometric anchors or fitted observations."
    )

    paths = _save_figure(fig, inputs.root, FIGURE_STEMS[3])
    caption = _write_caption(
        inputs.root,
        4,
        (
            "The master dispersion view places low-frequency geometric anchors, "
            "higher-frequency chain-transfer observations, fixture data, and the "
            "series-RLC transfer model on one measured-frequency axis. The lower "
            "panel exposes the real-part sign reversal (or phase response when "
            "complex components are unavailable), demonstrating why carrier-band "
            "Q/V slopes are transfer values rather than geometric capacitances."
        ),
        values=values,
        sources=sources or ["`dispersion_master.csv` contained no plotted rows."],
        notes=notes,
    )
    return paths, caption


def _lf_group_stats(
    inputs: FigureInputs,
) -> list[tuple[str, float, Stat, float | None, str]]:
    """Pooled LF-geometric Ccell/Cline by medium and measured frequency."""

    capture_groups: dict[tuple[str, float], list[Mapping[str, object]]] = defaultdict(list)
    for row in inputs.per_capture:
        frequency = _frequency_khz(row)
        if (
            frequency is None
            or not _is_lf(row)
            or not _is_subbreakdown(row)
            or _is_excluded(row)
        ):
            continue
        if _stat(row, ("Ccell_pF", "Cline_pF", "C_cell_geom_pF")) is None:
            continue
        capture_groups[(_medium(row), round(frequency, 3))].append(row)

    output: list[tuple[str, float, Stat, float | None, str]] = []
    for (medium, frequency), rows in sorted(capture_groups.items()):
        stat = _stat_across_rows(
            rows, ("Ccell_pF", "Cline_pF", "C_cell_geom_pF")
        )
        if stat is None:
            continue
        syst_values = [
            value
            for value in (
                _pick_number(row, "syst_frac", "systematic_fraction")
                for row in rows
            )
            if value is not None
        ]
        output.append(
            (
                medium,
                frequency,
                stat,
                float(np.median(syst_values)) if syst_values else None,
                "; ".join(dict.fromkeys(_provenance(row) for row in rows)),
            )
        )
    if output:
        return output

    # Fallback for a numbers-only handoff where per-capture rows are absent.
    summary_groups: dict[
        tuple[str, float], list[tuple[Stat, Mapping[str, object]]]
    ] = defaultdict(list)
    for row in _summary_records(inputs.condition_summary):
        frequency = _frequency_khz(row)
        if (
            frequency is None
            or not _is_lf(row)
            or not _is_subbreakdown(row)
            or _is_excluded(row)
        ):
            continue
        stat = _stat(row, ("Ccell_pF", "Cline_pF", "C_cell_geom_pF"))
        if stat is not None:
            summary_groups[(_medium(row), round(frequency, 3))].append((stat, row))
    for (medium, frequency), entries in sorted(summary_groups.items()):
        if len(entries) == 1:
            pooled = entries[0][0]
        else:
            pooled = _aggregate(stat.value for stat, _ in entries)
        if pooled is None:
            continue
        syst_values = [
            value
            for value in (
                _pick_number(row, "syst_frac", "systematic_fraction")
                for _, row in entries
            )
            if value is not None
        ]
        output.append(
            (
                medium,
                frequency,
                pooled,
                float(np.median(syst_values)) if syst_values else None,
                "; ".join(
                    dict.fromkeys(_provenance(row) for _, row in entries)
                ),
            )
        )
    return output


def make_fig5(inputs: FigureInputs) -> tuple[list[Path], Path]:
    fig, axes_array = _new_figure()
    ax = axes_array[0, 0]
    _panel_label(ax, "(a) LF-geometric cell comparison")
    groups = _lf_group_stats(inputs)
    values: list[str] = []
    sources: list[str] = []
    notes: list[str] = []

    if not groups:
        _empty_panel(
            ax,
            "no pooled sub-breakdown Ccell/Cline rows tagged LF-geometric",
            xlabel="Cell medium",
            ylabel="Cell capacitance (pF; LF-geometric)",
        )
        notes.append(
            "No operational LF-geometric anchor was available. "
            "Carrier-transfer values were not substituted."
        )
    else:
        media = sorted({medium for medium, _, _, _, _ in groups}, key=_medium_label)
        frequencies = sorted({frequency for _, frequency, _, _, _ in groups})
        x_base = np.arange(len(media), dtype=float)
        spread = 0.54
        offsets = (
            np.linspace(-spread / 2, spread / 2, len(frequencies))
            if len(frequencies) > 1
            else np.asarray([0.0])
        )
        markers = ("o", "s", "^", "D", "P", "X")
        for frequency_index, frequency in enumerate(frequencies):
            entries = {
                medium: (stat, syst, provenance)
                for medium, f_value, stat, syst, provenance in groups
                if math.isclose(f_value, frequency, rel_tol=0.0, abs_tol=1e-6)
            }
            handle_added = False
            for medium_index, medium in enumerate(media):
                if medium not in entries:
                    continue
                stat, syst, provenance = entries[medium]
                x = x_base[medium_index] + offsets[frequency_index]
                color = _medium_color(medium)
                if syst is not None and syst >= 0:
                    half = abs(stat.value * syst)
                    ax.vlines(
                        x,
                        stat.value - half,
                        stat.value + half,
                        color=color,
                        linewidth=7,
                        alpha=0.14,
                        zorder=1,
                    )
                _safe_errorbar(
                    ax,
                    [x],
                    [stat],
                    color=color,
                    marker=markers[frequency_index % len(markers)],
                    markerfacecolor="white",
                    markeredgecolor=color,
                    linestyle="none",
                    label=(
                        f"{_format_number(frequency)} kHz [LF-geometric]"
                        if not handle_added
                        else None
                    ),
                    zorder=3,
                )
                handle_added = True
                ax.annotate(
                    f"N={stat.n if stat.n is not None else '?'}",
                    (x, stat.value),
                    xytext=(0, 6),
                    textcoords="offset points",
                    ha="center",
                    fontsize=7.2,
                    color=color,
                )
                syst_text = (
                    f", separate systematic ±{100.0 * syst:.2g}%"
                    if syst is not None
                    else ", systematic fraction not recorded"
                )
                values.append(
                    f"{_medium_label(medium)}, measured f0≈{_format_number(frequency)} kHz: "
                    f"Ccell={_format_stat(stat, 'pF')} [LF-geometric]{syst_text}."
                )
                sources.append(
                    f"{_medium_label(medium)}, {_format_number(frequency)} kHz: "
                    f"`{provenance}`."
                )

        ax.set_xticks(x_base, [_medium_label(medium) for medium in media])
        for tick in ax.get_xticklabels():
            tick.set_rotation(12)
            tick.set_ha("right")
        ax.set_xlabel("Cell medium")
        ax.set_ylabel("Cell capacitance (pF; LF-geometric)")
        ax.legend(loc="best", ncol=min(3, len(frequencies)))
        ax.margins(x=0.08, y=0.18)
        ax.text(
            0.99,
            0.02,
            "Series gap / barrier / liquid decomposition is discussed on the slide.",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=8,
            color="#555555",
        )
        notes.append(
            "Points are medians and vertical whiskers are [2.5, 97.5] statistical "
            "intervals. Translucent bars, when present, show systematic calibration "
            "uncertainty separately."
        )

    paths = _save_figure(fig, inputs.root, FIGURE_STEMS[4])
    caption = _write_caption(
        inputs.root,
        5,
        (
            "Pooled low-frequency cell capacitance is compared across media at "
            "each measured anchor frequency. All plotted capacitances are tagged "
            "LF-geometric; no carrier-transfer value is included."
        ),
        values=values,
        sources=sources or ["`per_capture_metrics.csv` and `condition_summary.csv`."],
        notes=notes,
    )
    return paths, caption


def _discharge_points(inputs: FigureInputs) -> list[DischargePoint]:
    rows: list[Mapping[str, object]]
    if inputs.discharge:
        rows = list(inputs.discharge)
    else:
        rows = _summary_records(inputs.condition_summary)
    captures_by_condition: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    for capture in inputs.per_capture:
        captures_by_condition[
            _pick_text(capture, "cond", "condition")
        ].append(capture)
    output: list[DischargePoint] = []
    for row in rows:
        if _is_excluded(row):
            continue
        dataset_type = _normalize_key(
            _pick_text(row, "dataset_type", "source_type")
        )
        # Figure 6 is an amplitude-response figure. Keep the actual voltage
        # ladders here; one-off operational maxima and synthesis runs remain in
        # discharge_metrics.csv and synthesis_charge.csv.
        if dataset_type != "voltage_ladder":
            continue
        band = _band(row)
        if band not in {
            "LF-geometric",
            "carrier-transfer",
            "band-unassigned",
        }:
            continue
        condition = _pick_text(row, "cond", "condition")
        condition_captures = captures_by_condition.get(condition, [])
        measured_frequency = _pick_number(
            row,
            "f0_kHz_median",
            "f0_khz_median",
            "measured_f0_kHz",
        )
        if measured_frequency is None:
            frequency_values = [
                value
                for capture in condition_captures
                for value in [_pick_number(capture, "f0_kHz", "f0_khz")]
                if value is not None
            ]
            if frequency_values:
                measured_frequency = float(np.median(frequency_values))
        measured_burst = _pick_number(
            row,
            "burst_Hz_median",
            "burst_rate_Hz_median",
        )
        if measured_burst is None:
            burst_values = [
                value
                for capture in condition_captures
                for value in [
                    _pick_number(capture, "burst_Hz", "burst_rate_Hz")
                ]
                if value is not None
            ]
            if burst_values:
                measured_burst = float(np.median(burst_values))
        amplitude = _stat(row, ("Vamp_kV", "V_amp_kV", "amplitude_kV"))
        raw = _stat(
            row,
            (
                "dQ_cycle_nC",
                "dQ_raw_nC",
                "deltaQ_raw_nC",
                "deltaQ_nC",
            ),
        )
        corrected = _stat(
            row,
            (
                "dQ_gap_nC",
                "dQ_corrected_nC",
                "dQ_F_corrected_nC",
                "deltaQ_corrected_nC",
            ),
        )
        if measured_burst is not None and measured_burst > 0:
            energy = _stat(
                row,
                (
                    "U_burst_uJ",
                    "energy_burst_uJ",
                ),
            )
            energy_basis = "burst"
        else:
            energy = _stat(
                row,
                (
                    "U_cycle_uJ",
                    "U_cycle_abs_uJ",
                    "energy_cycle_uJ",
                    "U_uJ",
                ),
            )
            energy_basis = "carrier cycle"
        power = _stat(row, ("P_W", "power_W", "P_discharge_W"))
        has_metric = any(item is not None for item in (raw, corrected, energy, power))
        if amplitude is None or not has_metric:
            continue
        explicit_discharge = None
        for key in ("discharge_detected", "is_discharge", "discharge_on", "loop_open"):
            state = _boolish(row.get(key))
            if state is not None:
                explicit_discharge = state
                break
        level = _level_pct(row)
        if explicit_discharge is False:
            continue
        if explicit_discharge is None and level is not None and level < 100.0:
            # A row carrying actual discharge metrics is retained even below a
            # nominal threshold because objective onset may precede 100%.
            if raw is None and energy is None:
                continue
        orientation_text = _pick_text(row, "orientation_status")
        polarity_text = _pick_text(
            row,
            "polarity_lock_status",
            "polarity_status",
        )
        orientation_flags: list[str] = []
        if energy is not None and energy.value < 0:
            orientation_flags.append(
                "POLARITY/ORIENTATION ERROR: negative median U; energy and power omitted"
            )
            energy = None
            power = None
        elif "failed" in orientation_text.lower():
            orientation_flags.append(orientation_text)
        if "ambiguous" in polarity_text.lower():
            orientation_flags.append(polarity_text)
        correction_factor = _pick_number(
            row,
            "charge_correction_factor_median",
            "charge_correction_factor",
            "gap_correction_F",
        )
        if (
            correction_factor is None
            and raw is not None
            and corrected is not None
            and abs(raw.value) > np.finfo(float).eps
        ):
            correction_factor = corrected.value / raw.value
        output.append(
            DischargePoint(
                medium=_medium(row),
                condition=condition,
                freq_label=_pick_text(row, "freq_label", "frequency_label"),
                frequency_khz=measured_frequency,
                band_tag=band,
                level_pct=level,
                amplitude_kv=amplitude.value,
                raw_dq=raw,
                corrected_dq=corrected,
                energy=energy,
                energy_basis=energy_basis,
                power=power,
                power_method=_pick_text(row, "P_method"),
                correction_factor=correction_factor,
                provenance=_provenance(row),
                orientation_flag=(
                    "; ".join(dict.fromkeys(orientation_flags))
                    if orientation_flags
                    else None
                ),
            )
        )
    return sorted(
        output,
        key=lambda point: (
            _medium_label(point.medium),
            point.freq_label,
            math.inf if point.frequency_khz is None else point.frequency_khz,
            point.amplitude_kv,
        ),
    )


def _discharge_group_label(
    medium: str,
    freq_label: str,
    frequency_khz: float | None,
    band_tag: str,
) -> str:
    short_medium = {
        "argon": "Ar",
        "manganese_nitrate": "Mn-water",
        "pure_water": "Water",
        "ionic_liquid": "ionic liquid",
    }.get(medium, _medium_label(medium))
    nominal_match = re.search(r"(\d+(?:\.\d+)?)", freq_label)
    nominal = f"b{nominal_match.group(1)}" if nominal_match else "burst"
    # Keep the in-panel key compact. The measured carrier frequency and the
    # transfer-band provenance remain explicit in the caption and values block.
    return f"{short_medium} ({nominal})"


def make_fig6(inputs: FigureInputs) -> tuple[list[Path], Path]:
    fig, axes_array = _new_figure(1, 3)
    axes = axes_array[0]
    for index, label in enumerate(
        ("(a) Transferred charge", "(b) Dissipated energy", "(c) Discharge power")
    ):
        _panel_label(axes[index], label)
        axes[index].set_xlabel("Measured amplitude (kV)")
    axes[0].set_ylabel("\N{GREEK CAPITAL LETTER DELTA}Q (nC/cycle)")
    axes[1].set_ylabel("Loop energy (µJ; basis below)")
    axes[2].set_ylabel("P (W)")

    points = _discharge_points(inputs)
    by_group: dict[
        tuple[str, str, float | None, str], list[DischargePoint]
    ] = defaultdict(list)
    for point in points:
        frequency = (
            None
            if point.frequency_khz is None
            else round(point.frequency_khz, 3)
        )
        by_group[
            (
                point.medium,
                point.freq_label,
                frequency,
                point.band_tag,
            )
        ].append(point)

    values: list[str] = []
    sources: list[str] = []
    notes: list[str] = []
    plotted = [False, False, False]

    for (medium, freq_label, frequency, band_tag), group in sorted(
        by_group.items(),
        key=lambda item: (
            _medium_label(item[0][0]),
            item[0][1],
            math.inf if item[0][2] is None else item[0][2],
        ),
    ):
        group = sorted(group, key=lambda point: point.amplitude_kv)
        color = _medium_color(medium)
        base_label = _discharge_group_label(
            medium,
            freq_label,
            frequency,
            band_tag,
        )

        raw_points = [point for point in group if point.raw_dq is not None]
        if raw_points:
            _safe_errorbar(
                axes[0],
                [point.amplitude_kv for point in raw_points],
                [point.raw_dq for point in raw_points if point.raw_dq is not None],
                color=color,
                marker="o",
                markerfacecolor="white",
                linestyle="-",
                label=base_label,
            )
            plotted[0] = True
        corrected_points = [
            point for point in group if point.corrected_dq is not None
        ]
        if corrected_points:
            _safe_errorbar(
                axes[0],
                [point.amplitude_kv for point in corrected_points],
                [
                    point.corrected_dq
                    for point in corrected_points
                    if point.corrected_dq is not None
                ],
                color=color,
                marker="s",
                markerfacecolor=color,
                linestyle="--",
                label="_nolegend_",
            )
            plotted[0] = True

        energy_points = [point for point in group if point.energy is not None]
        if energy_points:
            _safe_errorbar(
                axes[1],
                [point.amplitude_kv for point in energy_points],
                [point.energy for point in energy_points if point.energy is not None],
                color=color,
                marker="o",
                markerfacecolor="white",
                linestyle="-",
                label=base_label,
            )
            plotted[1] = True
        power_points = [point for point in group if point.power is not None]
        if power_points:
            _safe_errorbar(
                axes[2],
                [point.amplitude_kv for point in power_points],
                [point.power for point in power_points if point.power is not None],
                color=color,
                marker="o",
                markerfacecolor="white",
                linestyle="-",
                label=base_label,
            )
            plotted[2] = True

        for point in group:
            available_stats = [
                stat
                for stat in (
                    point.raw_dq,
                    point.corrected_dq,
                    point.energy,
                    point.power,
                )
                if stat is not None
            ]
            n_value = next(
                (stat.n for stat in available_stats if stat.n is not None), None
            )
            if n_value is not None:
                if point.raw_dq is not None:
                    _annotation_position(
                        axes[0],
                        point.amplitude_kv,
                        point.raw_dq.value,
                        f"N={n_value}",
                        color,
                    )
                if point.energy is not None:
                    _annotation_position(
                        axes[1],
                        point.amplitude_kv,
                        point.energy.value,
                        f"N={n_value}",
                        color,
                    )
                if point.power is not None:
                    _annotation_position(
                        axes[2],
                        point.amplitude_kv,
                        point.power.value,
                        f"N={n_value}",
                        color,
                    )
            frequency_text = (
                f"measured f0={_format_number(point.frequency_khz)} kHz"
                if point.frequency_khz is not None
                else "measured f0 not recorded"
            )
            nominal_text = (
                f", burst-frequency label={point.freq_label}"
                if point.freq_label
                else ""
            )
            level_text = (
                f", nominal level={_format_number(point.level_pct)}%"
                if point.level_pct is not None
                else ""
            )
            f_text = (
                f"; applied charge factor={_format_number(point.correction_factor)}"
                if point.correction_factor is not None
                else "; applied charge factor not recorded"
            )
            values.append(
                f"{_medium_label(point.medium)}, {frequency_text}, "
                f"amplitude={_format_number(point.amplitude_kv)} kV"
                f"{nominal_text}{level_text} [{point.band_tag}]: "
                f"raw ΔQ={_format_stat(point.raw_dq, 'nC/cycle')}; "
                f"factor-corrected ΔQ={_format_stat(point.corrected_dq, 'nC/cycle')}"
                f"{f_text}; U_{point.energy_basis or 'unknown basis'}="
                f"{_format_stat(point.energy, 'µJ')}; "
                f"P={_format_stat(point.power, 'W')} "
                f"(method={point.power_method or 'not recorded'})."
            )
            sources.append(
                f"{_medium_label(point.medium)}, {frequency_text}: "
                f"`{point.provenance}`."
            )
            if point.orientation_flag:
                notes.append(
                    f"{_medium_label(point.medium)}, {frequency_text}, "
                    f"{_format_number(point.amplitude_kv)} kV: {point.orientation_flag}."
                )

    empty_details = (
        "no raw or same-band-F-corrected discharge-charge summaries",
        "no non-negative oriented burst/cycle loop-energy summaries",
        "no discharge-power summaries",
    )
    for ax, was_plotted, detail in zip(axes, plotted, empty_details):
        if was_plotted:
            handles, labels = ax.get_legend_handles_labels()
            ax.legend(
                handles,
                labels,
                loc="lower right",
                ncol=1,
                fontsize=8.0,
                frameon=True,
                facecolor="white",
                framealpha=0.92,
                edgecolor="#CCCCCC",
            )
            ax.margins(x=0.08, y=0.16)
            if ax is axes[0]:
                ax.text(
                    0.99,
                    0.98,
                    "o  raw charge\ns  carrier-reference F correction",
                    transform=ax.transAxes,
                    ha="right",
                    va="top",
                    fontsize=8.0,
                    color="#333333",
                    linespacing=1.25,
                    bbox={
                        "boxstyle": "round,pad=0.2",
                        "facecolor": "white",
                        "edgecolor": "#DDDDDD",
                        "alpha": 0.92,
                    },
                )
            else:
                ax.text(
                    0.99,
                    0.98,
                    "carrier-transfer band",
                    transform=ax.transAxes,
                    ha="right",
                    va="top",
                    fontsize=6.7,
                    color="#555555",
                )
        else:
            xlabel = ax.get_xlabel()
            ylabel = ax.get_ylabel()
            _empty_panel(ax, detail, xlabel=xlabel, ylabel=ylabel)
    if not points:
        notes.append(
            "No objective-onset/above-breakdown rows with measured amplitude "
            "were available in `discharge_metrics.csv` or the condition summary."
        )
    notes.append(
        "Raw and corrected charge are both shown when available. For the "
        "carrier-transfer ladders, the correction uses the explicitly configured "
        "same-band carrier-reference factor F=1.15; per-row branch-derived F "
        "values are not substituted."
    )
    energy_bases = {
        point.energy_basis
        for point in points
        if point.energy is not None and point.energy_basis
    }
    if energy_bases == {"burst"}:
        axes[1].set_ylabel("U_burst (µJ/burst)")
    elif energy_bases == {"carrier cycle"}:
        axes[1].set_ylabel("U_cycle (µJ/carrier cycle)")
    notes.append(
        "For burst-modulated records, the energy panel shows burst-period "
        "shoelace energy and power is P = U_burst × measured burst rate. "
        "No U_cycle × f0 fallback is used when a valid nonnegative burst-period "
        "energy is unavailable. "
        "Cycle-energy power is reserved for records classified as continuous."
    )
    if any(point.band_tag == "carrier-transfer" for point in points):
        notes.append(
            "The plotted ladder records are carrier-transfer acquisitions. "
            "Charge remains a Channel-D observable; energy and power are "
            "conditional on the starred Channel-A node/topology assumption and "
            "are not geometric-capacitance claims."
        )

    paths = _save_figure(fig, inputs.root, FIGURE_STEMS[5])
    caption = _write_caption(
        inputs.root,
        6,
        (
            "At and above objectively detected discharge onset, the measured "
            "charge transferred per cycle, same-band-F-corrected charge, "
            "orientation-screened loop energy, and method-matched power are "
            "shown against measured voltage amplitude for the voltage ladders. "
            "Compact legends identify medium and burst-frequency label; measured "
            "carrier f0, acquisition band, applied correction factor, and full "
            "provenance are reported in this caption."
        ),
        values=values,
        sources=sources or ["`discharge_metrics.csv` and `condition_summary.csv`."],
        notes=notes,
    )
    return paths, caption


def _run_name(row: Mapping[str, object]) -> str:
    return (
        _pick_text(
            row,
            "run",
            "run_id",
            "label",
            "run_key",
            "condition",
            "cond",
            "dataset",
            "dataset_name",
            "path",
        )
        or "unnamed run"
    )


def _run_sort_key(run: str) -> tuple[int, int, str]:
    normalized = run.replace("\\", "/").lower()
    match = re.search(r"(?:^|/|_)(7)[_-](\d{1,2})(?:/|_|$)", normalized)
    if match:
        return int(match.group(1)), int(match.group(2)), normalized
    return 99, 99, normalized


def _synthesis_dq_for_row(
    row: Mapping[str, object],
) -> tuple[Stat | None, str | None]:
    direct = _stat(
        row,
        (
            "dQ_cycle_nC",
            "dQ_raw_nC",
            "deltaQ_cycle_nC",
            "charge_per_cycle_nC",
        ),
    )
    if direct is not None:
        return direct, None
    positive = _stat(row, ("dQ_positive_nC", "positive_dQ_nC"))
    negative = _stat(row, ("dQ_negative_nC", "negative_dQ_nC"))
    if positive is None and negative is None:
        return None, None
    if positive is not None and negative is not None:
        return (
            Stat(
                value=abs(positive.value) + abs(negative.value),
                n=positive.n if positive.n is not None else negative.n,
            ),
            "computed as |ΔQ_positive| + |ΔQ_negative| because no full-cycle field was supplied",
        )
    only = positive if positive is not None else negative
    assert only is not None
    return (
        Stat(
            value=abs(only.value),
            low=(abs(only.low) if only.low is not None else None),
            high=(abs(only.high) if only.high is not None else None),
            n=only.n,
        ),
        "absolute available half-cycle ΔQ used because no full-cycle field was supplied",
    )


def _rate_c_min_for_row(row: Mapping[str, object]) -> Stat | None:
    direct = _stat(
        row,
        (
            "charge_rate_C_per_min",
            "delivery_rate_C_per_min",
            "charge_throughput_C_min",
            "gross_rate_C_min",
            "rate_C_min",
            "C_per_min",
        ),
    )
    if direct is not None:
        return direct
    per_second = _stat(
        row,
        (
            "charge_rate_C_per_s",
            "delivery_rate_C_per_s",
            "charge_throughput_C_s",
            "rate_C_s",
        ),
    )
    if per_second is None:
        return None
    return Stat(
        value=60.0 * per_second.value,
        low=(60.0 * per_second.low if per_second.low is not None else None),
        high=(60.0 * per_second.high if per_second.high is not None else None),
        n=per_second.n,
    )


def _dose_20_for_row(
    row: Mapping[str, object], rate_c_min: Stat | None
) -> Stat | None:
    direct = _stat(
        row,
        (
            "dose_20min_C",
            "dose_clock_20min_C",
            "C_per_20_min",
            "charge_20min_C",
        ),
    )
    if direct is not None:
        return direct
    if rate_c_min is None:
        return None
    return Stat(
        value=20.0 * rate_c_min.value,
        low=(20.0 * rate_c_min.low if rate_c_min.low is not None else None),
        high=(20.0 * rate_c_min.high if rate_c_min.high is not None else None),
        n=rate_c_min.n,
    )


def _combine_stats(stats: Sequence[Stat]) -> Stat | None:
    if not stats:
        return None
    if len(stats) == 1 and (
        stats[0].low is not None
        or stats[0].high is not None
        or stats[0].n is not None
    ):
        return stats[0]
    return _aggregate(stat.value for stat in stats)


def _synthesis_points(inputs: FigureInputs) -> list[SynthesisPoint]:
    groups: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    for row in inputs.synthesis:
        if not _is_excluded(row):
            run = _run_name(row)
            normalized_run = _normalize_key(
                _pick_text(row, "run_key", "run", "run_id") or run
            )
            requested_date = re.search(
                r"(?:^|_)7_(9|18|19)(?:_|$)",
                normalized_run,
            )
            # Figure 7 follows the handoff's declared synthesis scope. Other
            # dated runs remain in synthesis_charge.csv but are not silently
            # added to this six-run comparison.
            if requested_date is None:
                continue
            groups[run].append(row)

    output: list[SynthesisPoint] = []
    for run, rows in sorted(groups.items(), key=lambda item: _run_sort_key(item[0])):
        dqs: list[Stat] = []
        dq_derivations: list[str] = []
        rates: list[Stat] = []
        doses: list[Stat] = []
        frequencies: list[float] = []
        bursts: list[float] = []
        duties: list[float] = []
        for row in rows:
            dq, derivation = _synthesis_dq_for_row(row)
            if dq is not None:
                dqs.append(dq)
            if derivation:
                dq_derivations.append(derivation)
            rate = _rate_c_min_for_row(row)
            if rate is not None:
                rates.append(rate)
            dose = _dose_20_for_row(row, rate)
            if dose is not None:
                doses.append(dose)
            frequency = _frequency_khz(row)
            if frequency is not None:
                frequencies.append(frequency)
            burst = _pick_number(row, "burst_Hz", "burst_rate_Hz")
            if burst is None:
                burst = _pick_number(
                    row,
                    "burst_Hz_median",
                    "burst_rate_Hz_median",
                )
            if burst is not None:
                bursts.append(burst)
            duty = _pick_number(
                row,
                "duty_on_fraction",
                "duty_on_fraction_median",
                "duty_fraction",
                "active_fraction",
            )
            if duty is not None:
                duties.append(duty)
        output.append(
            SynthesisPoint(
                run=run,
                medium=_medium(rows[0]),
                dq=_combine_stats(dqs),
                rate_c_min=_combine_stats(rates),
                dose_20min_c=_combine_stats(doses),
                f0_khz=(
                    float(np.median(frequencies)) if frequencies else None
                ),
                burst_hz=float(np.median(bursts)) if bursts else None,
                duty=float(np.median(duties)) if duties else None,
                provenance="; ".join(
                    dict.fromkeys(_provenance(row) for row in rows)
                ),
                dq_derivation=(
                    "; ".join(dict.fromkeys(dq_derivations))
                    if dq_derivations
                    else None
                ),
                qc_status=_pick_text(rows[0], "qc_status"),
            )
        )
    return output


def _short_run_label(run: str) -> str:
    normalized = _normalize_key(run)
    date_match = re.search(r"(?:^|_)7_(9|18|19)(?:_|$)", normalized)
    date = f"7_{date_match.group(1)} " if date_match else ""
    if "bmim" in normalized and ("ntf" in normalized or "pt" in normalized):
        return f"{date}BMIM-NTf2 + Pt".strip()
    if "agpd" in normalized or ("ag" in normalized and "pd" in normalized and "hydrogen" in normalized):
        return f"{date}Ag-Pd / H2".strip()
    if "agcuni" in normalized or all(token in normalized for token in ("ag", "cu", "ni")):
        if "1_6" in normalized or "1_2_pd" in normalized:
            return f"{date}Ag-Cu-Ni\n1:6 betaine:1,2-PD".strip()
        if "1_3" in normalized or "betaine_eg" in normalized:
            return f"{date}Ag-Cu-Ni\n1:3 betaine:EG".strip()
        return f"{date}Ag-Cu-Ni".strip()
    if "cuni" in normalized or ("cu" in normalized and "ni" in normalized):
        return f"{date}Cu-Ni".strip()
    if "pdhydrogen" in normalized or ("pd" in normalized and "hydrogen" in normalized):
        suffix = " #2" if "0002" in normalized else ""
        return f"{date}Pd-H2{suffix}".strip()
    text = run.replace("\\", "/").rstrip("/").split("/")[-1]
    text = re.sub(r"\.(csv|psdata)$", "", text, flags=re.IGNORECASE)
    text = text.replace("_", " ")
    if len(text) <= 22:
        return text
    parts = text.split()
    if len(parts) > 1:
        midpoint = max(1, len(parts) // 2)
        return " ".join(parts[:midpoint]) + "\n" + " ".join(parts[midpoint:])
    return text[:20] + "…"


def make_fig7(inputs: FigureInputs) -> tuple[list[Path], Path]:
    fig, axes_array = _new_figure(1, 2, width_ratios=(1.35, 1.0))
    charge_ax, dose_ax = axes_array[0]
    _panel_label(charge_ax, "(a) Per-cycle charge and delivery rate")
    _panel_label(dose_ax, "(b) 20 min electrical dose clock")
    points = _synthesis_points(inputs)
    values: list[str] = []
    sources: list[str] = []
    notes: list[str] = []

    if not points:
        _empty_panel(
            charge_ax,
            "synthesis_charge.csv has no usable run summaries",
            xlabel="Synthesis run",
            ylabel="\N{GREEK CAPITAL LETTER DELTA}Q (nC/cycle) / rate (C/min)",
        )
        _empty_panel(
            dose_ax,
            "no delivery rate or 20 min dose-clock value",
            xlabel="Synthesis run",
            ylabel="Electrical charge in 20 min (C)",
        )
        notes.append(
            "No synthesis-run charge summaries were available; no carrier "
            "capacitance or folder-derived burst frequency was substituted."
        )
    else:
        x = np.arange(len(points), dtype=float)
        labels = [_short_run_label(point.run) for point in points]
        dq_points = [
            (index, point)
            for index, point in enumerate(points)
            if point.dq is not None
        ]
        if dq_points:
            for index, point in dq_points:
                color = _medium_color(point.medium)
                _safe_errorbar(
                    charge_ax,
                    [float(index)],
                    [point.dq] if point.dq is not None else [],
                    color=color,
                    marker="o",
                    markerfacecolor="white",
                    markeredgecolor=color,
                    linestyle="none",
                    zorder=3,
                )
                if point.dq is not None:
                    charge_ax.annotate(
                        f"N={point.dq.n if point.dq.n is not None else '?'}",
                        (float(index), point.dq.value),
                        xytext=(7, 6),
                        textcoords="offset points",
                        ha="left",
                        fontsize=8.1,
                        color=color,
                    )
            charge_ax.set_ylabel("\N{GREEK CAPITAL LETTER DELTA}Q (nC/cycle)")
        else:
            charge_ax.text(
                0.5,
                0.72,
                "ΔQ/cycle unavailable",
                transform=charge_ax.transAxes,
                ha="center",
                va="center",
                color="#555555",
            )
            charge_ax.set_ylabel("\N{GREEK CAPITAL LETTER DELTA}Q (nC/cycle)")

        rate_ax = charge_ax.twinx()
        rate_ax.spines["right"].set_visible(True)
        rate_ax.grid(False)
        rate_points = [
            (index, point)
            for index, point in enumerate(points)
            if point.rate_c_min is not None
        ]
        if rate_points:
            for index, point in rate_points:
                color = _medium_color(point.medium)
                _safe_errorbar(
                    rate_ax,
                    [float(index)],
                    [point.rate_c_min] if point.rate_c_min is not None else [],
                    color=color,
                    marker="D",
                    markerfacecolor=color,
                    markeredgecolor=color,
                    linestyle="none",
                    zorder=2,
                )
            rate_ax.set_ylabel("Delivery rate (C/min)")
        else:
            rate_ax.set_yticks([])
            rate_ax.set_ylabel("Delivery rate unavailable")

        charge_ax.set_xticks(x, labels)
        charge_ax.set_xlabel("Synthesis run")
        for tick in charge_ax.get_xticklabels():
            tick.set_rotation(18)
            tick.set_ha("right")
        charge_ax.margins(x=0.08, y=0.18)
        charge_ax.legend(
            handles=[
                Line2D(
                    [0],
                    [0],
                    marker="o",
                    markerfacecolor="white",
                    markeredgecolor=OKABE_ITO["black"],
                    linestyle="none",
                    color=OKABE_ITO["black"],
                    label="Median ΔQ; [2.5, 97.5]",
                ),
                Line2D(
                    [0],
                    [0],
                    marker="D",
                    markerfacecolor=OKABE_ITO["black"],
                    linestyle="none",
                    color=OKABE_ITO["black"],
                    label="Delivery rate",
                ),
            ],
            loc="best",
        )
        for index, point in enumerate(points):
            if point.dq is None and point.rate_c_min is None:
                status_label = (
                    "withheld:\nclipped Channel D"
                    if point.qc_status
                    and "no_unclipped_channel_d" in point.qc_status.lower()
                    else "not available"
                )
                charge_ax.text(
                    float(index),
                    0.10,
                    status_label,
                    transform=charge_ax.get_xaxis_transform(),
                    ha="center",
                    va="bottom",
                    fontsize=8.1,
                    color=OKABE_ITO["vermillion"],
                    bbox={
                        "boxstyle": "round,pad=0.18",
                        "facecolor": "white",
                        "edgecolor": "none",
                        "alpha": 0.85,
                    },
                )

        dose_values = [
            point.dose_20min_c.value
            if point.dose_20min_c is not None
            else math.nan
            for point in points
        ]
        if any(math.isfinite(value) for value in dose_values):
            bars = dose_ax.bar(
                x,
                [value if math.isfinite(value) else 0.0 for value in dose_values],
                color=[_medium_color(point.medium) for point in points],
                edgecolor="white",
                linewidth=0.8,
            )
            for index, (bar, point) in enumerate(zip(bars, points)):
                stat = point.dose_20min_c
                if stat is None:
                    bar.set_alpha(0.0)
                    status_label = (
                        "withheld:\nclipped Channel D"
                        if point.qc_status
                        and "no_unclipped_channel_d" in point.qc_status.lower()
                        else "not available"
                    )
                    dose_ax.text(
                        float(index),
                        0.10,
                        status_label,
                        transform=dose_ax.get_xaxis_transform(),
                        ha="center",
                        va="bottom",
                        fontsize=8.1,
                        color=OKABE_ITO["vermillion"],
                        bbox={
                            "boxstyle": "round,pad=0.18",
                            "facecolor": "white",
                            "edgecolor": "none",
                            "alpha": 0.85,
                        },
                    )
                    continue
                if stat.low is not None and stat.high is not None:
                    lower = max(0.0, stat.value - stat.low)
                    upper = max(0.0, stat.high - stat.value)
                    dose_ax.errorbar(
                        [index],
                        [stat.value],
                        yerr=np.asarray([[lower], [upper]]),
                        color=OKABE_ITO["black"],
                        capsize=2.5,
                        linestyle="none",
                    )
                dose_ax.annotate(
                    f"N={stat.n if stat.n is not None else '?'}",
                    (bar.get_x() + bar.get_width() / 2.0, stat.value),
                    xytext=(0, 5),
                    textcoords="offset points",
                    ha="center",
                    fontsize=8.1,
                )
            dose_ax.set_xticks(x, labels)
            for tick in dose_ax.get_xticklabels():
                tick.set_rotation(18)
                tick.set_ha("right")
            dose_ax.set_xlabel("Synthesis run")
            dose_ax.set_ylabel("Electrical charge in 20 min (C)")
            dose_ax.margins(y=0.18)
        else:
            _empty_panel(
                dose_ax,
                "no dose_20min_C field and no delivery rate from which to derive it",
                xlabel="Synthesis run",
                ylabel="Electrical charge in 20 min (C)",
            )

        for point in points:
            metadata = []
            if point.f0_khz is not None:
                metadata.append(f"measured carrier f0={_format_number(point.f0_khz)} kHz")
            if point.burst_hz is not None:
                metadata.append(f"burst rate={_format_number(point.burst_hz)} Hz")
            if point.duty is not None:
                metadata.append(f"duty-on fraction={_format_number(point.duty)}")
            values.append(
                f"{point.run}: ΔQ/cycle={_format_stat(point.dq, 'nC/cycle')}; "
                f"delivery rate={_format_stat(point.rate_c_min, 'C/min')}; "
                f"20 min electrical dose={_format_stat(point.dose_20min_c, 'C')}"
                + (f"; {', '.join(metadata)}" if metadata else "")
                + (
                    f"; QC={point.qc_status}"
                    if point.qc_status
                    else ""
                )
                + "."
            )
            sources.append(f"{point.run}: `{point.provenance}`.")
            if point.dq_derivation:
                notes.append(f"{point.run}: {point.dq_derivation}.")

    notes.append(
        "The 20 min bar is an electrical charge dose clock, not an electron-only "
        "or Faradaic-utilization claim. Rates use the measured carrier/burst/duty "
        "values already quantified for each run."
    )
    notes.append(
        "The presentation comparison is intentionally limited to the handoff's "
        "declared 7_9, 7_18, and four 7_19 synthesis runs. Any additional dated "
        "run remains available in synthesis_charge.csv but is not added here."
    )
    paths = _save_figure(fig, inputs.root, FIGURE_STEMS[6])
    caption = _write_caption(
        inputs.root,
        7,
        (
            "Actual synthesis runs are compared by transferred charge per carrier "
            "cycle, measured charge-delivery rate, and the implied electrical "
            "charge delivered in 20 minutes. These Channel-D charge quantities "
            "survive at carrier frequency; no geometric capacitance is inferred."
        ),
        values=values,
        sources=sources or ["`synthesis_charge.csv`."],
        notes=notes,
    )
    return paths, caption


def _dynamic_numeric_fields(
    rows: Sequence[Mapping[str, object]], prefix: str
) -> list[str]:
    fields = {
        key
        for row in rows
        for key in row
        if _normalize_key(key).startswith(_normalize_key(prefix))
        and any(_finite_number(item.get(key)) is not None for item in rows)
    }
    return sorted(fields, key=_normalize_key)


def _channel_label(field_name: str, prefix: str) -> str:
    normalized = _normalize_key(field_name)
    normalized_prefix = _normalize_key(prefix)
    suffix = normalized[len(normalized_prefix) :].strip("_")
    return suffix.upper() if suffix else "unspecified channel"


def _wrapped_phase(value: float) -> float:
    return ((value + 180.0) % 360.0) - 180.0


def make_fig8(inputs: FigureInputs) -> tuple[list[Path], Path]:
    fig, axes_array = _new_figure(2, 2)
    codes_ax, clips_ax, frequency_ax, phase_ax = axes_array.reshape(-1)
    _panel_label(codes_ax, "(a) ADC-code occupancy")
    _panel_label(clips_ax, "(b) Interpolated over-range samples")
    _panel_label(frequency_ax, "(c) Segment-to-segment f0 stability")
    _panel_label(phase_ax, "(d) Polarity/orientation audit")
    values: list[str] = []
    sources: list[str] = []
    notes: list[str] = []

    manifest_rows = inputs.manifest
    code_fields = _dynamic_numeric_fields(manifest_rows, "codes")
    code_colors = (
        OKABE_ITO["blue"],
        OKABE_ITO["orange"],
        OKABE_ITO["bluish_green"],
        OKABE_ITO["reddish_purple"],
        OKABE_ITO["vermillion"],
    )
    plotted_codes = False
    for index, field_name in enumerate(code_fields):
        values_for_field = [
            value
            for value in (
                _finite_number(row.get(field_name)) for row in manifest_rows
            )
            if value is not None and value >= 0
        ]
        if not values_for_field:
            continue
        maximum = max(values_for_field)
        bins = min(24, max(5, int(math.sqrt(len(values_for_field))) + 1))
        codes_ax.hist(
            values_for_field,
            bins=bins,
            histtype="step",
            linewidth=1.7,
            color=code_colors[index % len(code_colors)],
            label=f"Channel {_channel_label(field_name, 'codes')}, N={len(values_for_field)}",
        )
        limited = sum(value < 30.0 for value in values_for_field)
        values.append(
            f"Channel {_channel_label(field_name, 'codes')}: "
            f"N={len(values_for_field)}, median={_format_number(float(np.median(values_for_field)))} "
            f"distinct codes, range={_format_number(min(values_for_field))}–"
            f"{_format_number(maximum)}, quantization-limited (<30 codes)={limited}."
        )
        plotted_codes = True
    if plotted_codes:
        codes_ax.axvline(
            30.0,
            color=OKABE_ITO["vermillion"],
            linestyle=":",
            linewidth=1.2,
            label="~30-code QC threshold",
        )
        codes_ax.set_xlabel("Distinct ADC codes per segment")
        codes_ax.set_ylabel("Segment count")
        codes_ax.legend(loc="best")
    else:
        _empty_panel(
            codes_ax,
            "manifest.csv has no numeric codes* fields",
            xlabel="Distinct ADC codes per segment",
            ylabel="Segment count",
        )

    clip_fields = _dynamic_numeric_fields(manifest_rows, "clip")
    clip_totals: list[float] = []
    clip_labels: list[str] = []
    clip_nonzero: list[int] = []
    clip_n: list[int] = []
    for field_name in clip_fields:
        channel_values = [
            value
            for value in (
                _finite_number(row.get(field_name)) for row in manifest_rows
            )
            if value is not None and value >= 0
        ]
        if not channel_values:
            continue
        clip_labels.append(f"Channel {_channel_label(field_name, 'clip')}")
        clip_totals.append(float(np.sum(channel_values)))
        clip_nonzero.append(sum(value > 0 for value in channel_values))
        clip_n.append(len(channel_values))
    if clip_totals:
        positions = np.arange(len(clip_totals))
        bars = clips_ax.bar(
            positions,
            clip_totals,
            color=[
                code_colors[index % len(code_colors)]
                for index in range(len(clip_totals))
            ],
        )
        clips_ax.set_xticks(positions, clip_labels)
        clips_ax.set_xlabel("Header-derived channel field")
        clips_ax.set_ylabel("Total interpolated samples")
        for bar, total, nonzero, count, label in zip(
            bars, clip_totals, clip_nonzero, clip_n, clip_labels
        ):
            clips_ax.annotate(
                f"{nonzero}/{count} captures >0",
                (bar.get_x() + bar.get_width() / 2.0, total),
                xytext=(0, 5),
                textcoords="offset points",
                ha="center",
                fontsize=7.2,
            )
            values.append(
                f"{label}: {_format_number(total, 5)} interpolated samples in total; "
                f"{nonzero} of {count} captures had at least one over-range sample."
            )
        clips_ax.margins(y=0.2)
    else:
        _empty_panel(
            clips_ax,
            "manifest.csv has no numeric clip* fields",
            xlabel="Header-derived channel field",
            ylabel="Total interpolated samples",
        )

    stability_candidates = [
        row
        for row in manifest_rows
        if _frequency_khz(row) is not None
        and _frequency_khz(row) > 0
        and not _is_excluded(row)
    ]
    stability_rows = [
        row
        for row in stability_candidates
        if _normalize_key(_pick_text(row, "frequency_status"))
        != "fft_zero_cross_disagreement"
    ]
    stability_rejected = len(stability_candidates) - len(stability_rows)
    if stability_rows:
        stability_groups: dict[
            tuple[str, str], list[Mapping[str, object]]
        ] = defaultdict(list)
        for row in stability_rows:
            path = _pick_text(row, "path", "source_path")
            parent = str(Path(path).parent) if path else _pick_text(row, "cond", "condition")
            condition = _pick_text(row, "cond", "condition") or parent
            stability_groups[(_medium(row), condition)].append(row)
        medium_seen: set[str] = set()
        all_frequencies: list[float] = []
        all_relative_deviations: list[float] = []
        for (medium, condition), group in sorted(
            stability_groups.items(),
            key=lambda item: (
                _medium_label(item[0][0]),
                item[0][1],
            ),
        ):
            ordered = sorted(
                group,
                key=lambda row: (
                    _pick_number(row, "seg_idx", "seg", "segment")
                    if _pick_number(row, "seg_idx", "seg", "segment") is not None
                    else math.inf,
                    _provenance(row),
                ),
            )
            segment = [
                (
                    _pick_number(row, "seg_idx", "seg", "segment")
                    if _pick_number(row, "seg_idx", "seg", "segment") is not None
                    else float(index)
                )
                for index, row in enumerate(ordered)
            ]
            frequency = [
                _frequency_khz(row)
                for row in ordered
                if _frequency_khz(row) is not None
            ]
            if len(frequency) != len(segment):
                continue
            all_frequencies.extend(float(value) for value in frequency)
            condition_median = float(np.median(frequency))
            if condition_median <= 0:
                continue
            relative_deviation = [
                100.0 * (float(value) / condition_median - 1.0)
                for value in frequency
            ]
            all_relative_deviations.extend(relative_deviation)
            frequency_ax.plot(
                segment,
                relative_deviation,
                color=_medium_color(medium),
                marker=".",
                linestyle="-",
                linewidth=0.8,
                alpha=0.32,
                label=(
                    f"{_medium_label(medium)}"
                    if medium not in medium_seen
                    else None
                ),
            )
            medium_seen.add(medium)
        frequency_ax.axhline(
            0.0,
            color="#666666",
            linewidth=0.8,
            zorder=0,
        )
        frequency_ax.set_xlabel("Segment index")
        frequency_ax.set_ylabel("f0 deviation from condition median (%)")
        if medium_seen:
            frequency_ax.legend(loc="upper right", ncol=2)
        if all_frequencies:
            low, high = np.percentile(all_frequencies, [2.5, 97.5])
            values.append(
                f"Measured f0 across manifest segments: median="
                f"{_format_number(float(np.median(all_frequencies)))} kHz, "
                f"[2.5, 97.5]=[{_format_number(float(low))}, "
                f"{_format_number(float(high))}] kHz, N={len(all_frequencies)}."
            )
        if all_relative_deviations:
            relative_low, relative_high = np.percentile(
                all_relative_deviations,
                [2.5, 97.5],
            )
            values.append(
                "Within-condition f0 deviation: median="
                f"{_format_number(float(np.median(all_relative_deviations)))}%, "
                f"[2.5, 97.5]=[{_format_number(float(relative_low))}, "
                f"{_format_number(float(relative_high))}]%, across "
                f"{len(stability_groups)} independently drawn conditions; "
                f"N={len(all_relative_deviations)} verified captures. "
                f"FFT/zero-cross disagreement withheld from this panel="
                f"{stability_rejected} captures."
            )
        frequency_ax.text(
            0.99,
            0.02,
            f"verified FFT + zero-cross only\nwithheld disagreement: N={stability_rejected}",
            transform=frequency_ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=8.0,
            color="#555555",
            bbox={
                "boxstyle": "round,pad=0.2",
                "facecolor": "white",
                "edgecolor": "#DDDDDD",
                "alpha": 0.92,
            },
        )
    else:
        _empty_panel(
            frequency_ax,
            "manifest.csv has no finite measured f0",
            xlabel="Segment index",
            ylabel="Measured f0 (kHz)",
        )

    audit_rows = [
        row for row in inputs.per_capture if not _is_excluded(row)
    ]
    if audit_rows:
        polarity_locked = 0
        polarity_ambiguous = 0
        polarity_unavailable = 0
        orientation_pass = 0
        orientation_fail = 0
        orientation_unavailable = 0
        for row in audit_rows:
            polarity = _normalize_key(
                _pick_text(row, "polarity_lock_status", "polarity_status")
            )
            if "ambiguous" in polarity:
                polarity_ambiguous += 1
            elif "locked" in polarity:
                polarity_locked += 1
            else:
                polarity_unavailable += 1

            orientation = _normalize_key(
                _pick_text(row, "orientation_status")
            )
            if "failed" in orientation:
                orientation_fail += 1
            elif "passed" in orientation:
                orientation_pass += 1
            else:
                orientation_unavailable += 1

        audit_labels = (
            "Polarity\nlocked",
            "Polarity\nambiguous",
            "Energy sign\npassed",
            "Energy sign\nfailed",
            "Energy sign\nN/A",
        )
        audit_counts = (
            polarity_locked,
            polarity_ambiguous,
            orientation_pass,
            orientation_fail,
            orientation_unavailable,
        )
        bars = phase_ax.bar(
            np.arange(len(audit_counts)),
            audit_counts,
            color=(
                OKABE_ITO["bluish_green"],
                OKABE_ITO["orange"],
                OKABE_ITO["bluish_green"],
                OKABE_ITO["vermillion"],
                "#999999",
            ),
        )
        phase_ax.set_xticks(np.arange(len(audit_counts)), audit_labels)
        phase_ax.set_xlabel("Capture-level polarity/orientation QC")
        phase_ax.set_ylabel("Capture count")
        for bar, count in zip(bars, audit_counts):
            phase_ax.annotate(
                f"N={count}",
                (bar.get_x() + bar.get_width() / 2.0, count),
                xytext=(0, 5),
                textcoords="offset points",
                ha="center",
                fontsize=8.0,
            )
        phase_ax.text(
            0.5,
            0.98,
            "No LF-geometric phase anchor:\ncarrier Q/V phase is chain-dependent",
            transform=phase_ax.transAxes,
            ha="center",
            va="top",
            fontsize=8.0,
            color="#555555",
            bbox={
                "boxstyle": "round,pad=0.2",
                "facecolor": "white",
                "edgecolor": "#DDDDDD",
                "alpha": 0.92,
            },
        )
        phase_ax.margins(y=0.24)
        values.append(
            "Polarity/orientation audit after explicit exclusions: "
            f"condition polarity locked={polarity_locked}, "
            f"ambiguous={polarity_ambiguous}, unavailable={polarity_unavailable}; "
            f"non-negative energy passed={orientation_pass}, "
            f"failed={orientation_fail}, unavailable={orientation_unavailable}."
        )

        phase_values = [
            _wrapped_phase(value)
            for row in audit_rows
            for value in [_pick_number(row, "phase_deg", "phase_degrees")]
            if value is not None
        ]
        if phase_values:
            near_zero = sum(abs(phase) <= 45.0 for phase in phase_values)
            near_inverted = sum(
                abs(abs(phase) - 180.0) <= 45.0 for phase in phase_values
            )
            other = len(phase_values) - near_zero - near_inverted
            values.append(
                "Carrier-transfer wrapped Q/V phase (reported only as a "
                "chain-response diagnostic, not a polarity verdict): median="
                f"{_format_number(float(np.median(phase_values)))}°, "
                f"N={len(phase_values)}; near 0°={near_zero}, "
                f"near ±180°={near_inverted}, other={other}."
            )
    else:
        _empty_panel(
            phase_ax,
            "per_capture_metrics.csv has no non-excluded polarity/orientation QC",
            xlabel="Capture-level polarity/orientation QC",
            ylabel="Capture count",
        )

    parse_failures = [
        row
        for row in manifest_rows
        if _pick_text(row, "parse_status", "status").lower()
        not in {"", "ok", "parsed", "success"}
    ]
    explicit_exclusions = sum(
        _boolish(row.get("excluded")) is True for row in manifest_rows
    )
    contaminated_count = sum(
        _boolish(row.get("contaminated")) is True for row in manifest_rows
    )
    quantization_count = sum(
        _normalize_key(_pick_text(row, "quantization_status"))
        == "quantization_limited"
        for row in inputs.per_capture
    )
    values.append(
        f"Manifest QC population: N={len(manifest_rows)} rows; "
        f"parse-status failures={len(parse_failures)}; explicitly excluded="
        f"{explicit_exclusions}; contaminated={contaminated_count}; "
        f"quantization-limited={quantization_count}."
    )
    sources.extend(
        f"Parse/QC flag: `{_provenance(row)}` "
        f"({_pick_text(row, 'parse_status', 'status', 'exclusion_flags', 'flags')})."
        for row in parse_failures[:50]
    )
    if len(parse_failures) > 50:
        notes.append(
            f"{len(parse_failures) - 50} additional parse-status failures are "
            "retained in manifest.csv but omitted from this caption for length."
        )
    notes.append(
        "The ~30-code line is a quantization warning threshold, not a hard "
        "replacement for the per-capture QC decision. Channel fields come from "
        "the manifest rather than folder-based channel assumptions."
    )
    notes.append(
        "Because no supplied operational capture qualifies as LF-geometric, "
        "carrier-transfer Q/V phase is not used as a polarity pass/fail test. "
        "Panel (d) therefore reports the condition-level polarity lock and "
        "non-negative-energy orientation checks that actually gate the outputs."
    )
    paths = _save_figure(fig, inputs.root, FIGURE_STEMS[7])
    caption = _write_caption(
        inputs.root,
        8,
        (
            "The appendix QC view audits ADC-code occupancy, interpolated "
            "over-range samples, verified within-condition measured-fundamental "
            "stability, and the polarity/orientation gates across captured "
            "segments. Each stability trace is drawn independently and "
            "normalized to its own condition median; unrelated conditions are "
            "never joined, and FFT/zero-cross disagreements are withheld."
        ),
        values=values,
        sources=sources or ["`manifest.csv` and `per_capture_metrics.csv`."],
        notes=notes,
    )
    return paths, caption


def generate_all(out: Path) -> list[Path]:
    """Generate every required figure and caption beneath ``out``."""

    configure_style()
    inputs = load_inputs(Path(out))
    outputs: list[Path] = []
    builders = (
        make_fig1,
        make_fig2,
        make_fig3,
        make_fig4,
        make_fig5,
        make_fig6,
        make_fig7,
        make_fig8,
    )
    for builder in builders:
        figure_paths, caption_path = builder(inputs)
        outputs.extend(figure_paths)
        outputs.append(caption_path)

    # Input notes are appended to each caption so missing/malformed sources are
    # never silently dropped, even when another table was sufficient to plot.
    if inputs.input_notes:
        note_block = (
            "\nInput availability notes:\n"
            + "\n".join(f"- {note}" for note in inputs.input_notes)
            + "\n"
        )
        for number in range(1, 9):
            caption_path = inputs.root / "captions" / f"fig{number}_caption.md"
            with caption_path.open("a", encoding="utf-8") as handle:
                handle.write(note_block)
    assert_no_retired_numbers(inputs.root)
    return outputs


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate the eight Lissajous v2 presentation figures and captions "
            "from quantified CSV outputs."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Directory containing the quantified CSV outputs",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    outputs = generate_all(args.out)
    print(
        f"Wrote {len(outputs)} Lissajous figure/caption artifacts under "
        f"{args.out.resolve()}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
