"""Fast, header-aware access to PicoScope CSV trees and ZIP mirrors."""

from __future__ import annotations

import csv
import io
import re
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Iterable

import numpy as np

from .config import medium_display


class WaveformParseError(RuntimeError):
    """Raised when a waveform does not satisfy the PicoScope parsing contract."""


@dataclass(frozen=True)
class SourceRef:
    """One logical CSV, stored directly or as a ZIP member."""

    relative_path: str
    local_path: Path | None
    archive_path: Path | None
    member: str | None
    size: int
    crc: int | None

    @property
    def source_uri(self) -> str:
        if self.archive_path is not None:
            return f"{self.archive_path}::{self.member}"
        return str(self.local_path)

    @property
    def signature(self) -> str:
        if self.archive_path is not None:
            return f"zip:{self.archive_path.stat().st_size}:{self.crc}:{self.size}"
        if self.local_path is None:
            return "missing"
        stat = self.local_path.stat()
        return f"file:{stat.st_size}:{stat.st_mtime_ns}"


@dataclass
class Waveform:
    """Calibrated arrays plus raw acquisition/QC metadata."""

    time_s: np.ndarray
    applied_voltage_V: np.ndarray
    current_A: np.ndarray | None
    monitor_voltage_V: np.ndarray
    charge_nC: np.ndarray
    headers: list[str]
    units: list[str]
    role_indices: dict[str, int]
    clip_counts: dict[str, int]
    code_counts: dict[str, int]
    lsb: dict[str, float | None]
    n_samples_raw: int
    skipped_rows: int
    dc_offset_V: float
    drift_V_per_s: float
    detrended: bool


class SourceCatalog:
    """Keep ZIP files open while many members are read sequentially."""

    def __init__(self, sources: Iterable[SourceRef]):
        self.sources = list(sources)
        self._archives: dict[Path, zipfile.ZipFile] = {}

    def __enter__(self) -> "SourceCatalog":
        for path in sorted(
            {source.archive_path for source in self.sources if source.archive_path is not None},
            key=lambda item: str(item).casefold(),
        ):
            assert path is not None
            self._archives[path] = zipfile.ZipFile(path)
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        for archive in self._archives.values():
            archive.close()
        self._archives.clear()

    def read_bytes(self, source: SourceRef) -> bytes:
        if source.archive_path is not None and source.member is not None:
            return self._archives[source.archive_path].read(source.member)
        if source.local_path is None:
            raise FileNotFoundError(source.source_uri)
        return source.local_path.read_bytes()


def _normal_rel(path: str) -> str:
    return PurePosixPath(path.replace("\\", "/")).as_posix().lstrip("./")


def _ancillary(relative_path: str) -> bool:
    parts = PurePosixPath(relative_path).parts
    return any(part.casefold().endswith("_analysis") for part in parts)


def discover_sources(
    data_root: Path,
    *,
    prefer_archives: bool = True,
) -> tuple[list[SourceRef], list[str]]:
    """Discover each logical waveform once, preferring ZIP mirrors when configured."""

    if not data_root.exists():
        raise FileNotFoundError(f"Data root does not exist: {data_root}")
    sources: dict[str, SourceRef] = {}
    errors: list[str] = []

    # Direct files establish the authoritative logical paths. ZIP entries replace
    # duplicates only after successful archive inspection.
    for path in data_root.rglob("*.csv"):
        rel = _normal_rel(str(path.relative_to(data_root)))
        if _ancillary(rel):
            continue
        try:
            size = path.stat().st_size
        except OSError as exc:
            errors.append(f"{rel}: stat failed: {exc}")
            continue
        sources[rel.casefold()] = SourceRef(rel, path, None, None, size, None)

    for psdata in data_root.rglob("*.psdata"):
        expected_dir = psdata.with_suffix("")
        has_matching_csv = expected_dir.is_dir() and any(expected_dir.glob("*.csv"))
        if not has_matching_csv:
            prefix = psdata.stem.casefold()
            has_matching_csv = any(
                candidate.is_dir()
                and candidate.name.casefold().startswith(prefix)
                and any(candidate.glob("*.csv"))
                for candidate in psdata.parent.iterdir()
            )
        if not has_matching_csv:
            rel = _normal_rel(str(psdata.relative_to(data_root)))
            errors.append(f"{rel}: psdata_without_matching_csv_set")

    for archive_path in sorted(data_root.glob("*.zip"), key=lambda item: item.name.casefold()):
        try:
            with zipfile.ZipFile(archive_path) as archive:
                for info in archive.infolist():
                    if info.is_dir() or not info.filename.casefold().endswith(".csv"):
                        continue
                    rel = _normal_rel(info.filename)
                    if _ancillary(rel):
                        continue
                    key = rel.casefold()
                    candidate = SourceRef(
                        rel,
                        None,
                        archive_path,
                        info.filename,
                        info.file_size,
                        info.CRC,
                    )
                    direct = sources.get(key)
                    if (
                        prefer_archives
                        and direct is not None
                        and direct.local_path is not None
                        and direct.size != info.file_size
                    ):
                        errors.append(
                            f"{rel}: extracted/ZIP size mismatch "
                            f"({direct.size} vs {info.file_size}); extracted file retained"
                        )
                        continue
                    if prefer_archives or key not in sources:
                        sources[key] = candidate
        except (OSError, zipfile.BadZipFile) as exc:
            errors.append(f"{archive_path.name}: archive inspection failed: {exc}")

    ordered = sorted(sources.values(), key=lambda source: source.relative_path.casefold())
    return ordered, errors


def _read_leading_lines(raw: bytes, count: int = 12) -> list[tuple[int, int, bytes]]:
    lines: list[tuple[int, int, bytes]] = []
    start = 0
    for _ in range(count):
        end = raw.find(b"\n", start)
        if end < 0:
            end = len(raw)
        lines.append((start, min(end + 1, len(raw)), raw[start:end].rstrip(b"\r")))
        start = end + 1
        if start >= len(raw):
            break
    return lines


def _csv_fields(line: bytes) -> list[str]:
    text = line.decode("utf-8-sig", errors="replace")
    return next(csv.reader([text]))


def _looks_numeric(line: bytes) -> bool:
    first = line.split(b",", 1)[0].strip()
    if not first:
        return False
    try:
        float(first)
        return True
    except ValueError:
        return first in {
            b"inf",
            b"-inf",
            b"Infinity",
            b"-Infinity",
            b"\xe2\x88\x9e",
            b"-\xe2\x88\x9e",
        }


def _numeric_matrix(raw: bytes, ncols: int, data_start: int) -> tuple[np.ndarray, int]:
    payload = raw[data_start:]
    replacements = (
        (b"-\xe2\x88\x9e", b" nan"),
        (b"\xe2\x88\x9e", b" nan"),
        (b"-Infinity", b" nan"),
        (b"Infinity", b" nan"),
        (b"-inf", b" nan"),
        (b"+inf", b" nan"),
    )
    for old, new in replacements:
        payload = payload.replace(old, new)
    values = np.fromstring(payload.replace(b",", b" "), sep=" ", dtype=np.float64)
    remainder = values.size % ncols
    if values.size and remainder == 0:
        return values.reshape((-1, ncols)), 0

    # Rare malformed rows take the slower but shape-preserving route.
    matrix = np.genfromtxt(
        io.BytesIO(raw[data_start:]),
        delimiter=",",
        dtype=np.float64,
        invalid_raise=False,
        filling_values=np.nan,
        encoding="utf-8",
    )
    if matrix.ndim == 1:
        matrix = matrix.reshape((1, -1))
    if matrix.shape[1] != ncols:
        raise WaveformParseError(
            f"numeric data has {matrix.shape[1]} columns but the header has {ncols}"
        )
    return matrix, int(remainder)


def _normalize_label(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip().casefold())


def _role_indices(headers: list[str], aliases: dict[str, list[str]]) -> dict[str, int]:
    normalized = [_normalize_label(header) for header in headers]
    indices: dict[str, int] = {}
    for role, candidates in aliases.items():
        wanted = {_normalize_label(candidate) for candidate in candidates}
        matches = [index for index, label in enumerate(normalized) if label in wanted]
        if len(matches) == 1:
            indices[role] = matches[0]
        elif len(matches) > 1:
            raise WaveformParseError(f"multiple columns match configured role {role!r}")
    missing = [role for role in ("time", "applied_voltage", "charge_monitor") if role not in indices]
    if missing:
        raise WaveformParseError(
            "required channel labels were not found: "
            + ", ".join(missing)
            + f"; observed headers={headers!r}"
        )
    return indices


def _unit_scale(unit: str, kind: str) -> float:
    normalized = unit.strip().strip("()").replace("\u00b5", "u").casefold()
    tables = {
        "time": {"s": 1.0, "ms": 1.0e-3, "us": 1.0e-6, "ns": 1.0e-9},
        "voltage": {"v": 1.0, "mv": 1.0e-3, "kv": 1.0e3},
        "current": {"a": 1.0, "ma": 1.0e-3, "ua": 1.0e-6},
    }
    try:
        return tables[kind][normalized]
    except KeyError as exc:
        raise WaveformParseError(f"unsupported {kind} unit {unit!r}") from exc


def _interpolate(values: np.ndarray) -> np.ndarray:
    finite = np.isfinite(values)
    if not np.any(finite):
        raise WaveformParseError("channel contains no finite samples")
    if np.all(finite):
        return values.astype(np.float64, copy=True)
    index = np.arange(values.size, dtype=np.float64)
    return np.interp(index, index[finite], values[finite])


def _code_count(values: np.ndarray) -> int:
    finite = values[np.isfinite(values)]
    return int(np.unique(finite).size)


def _lsb(values: np.ndarray) -> float | None:
    unique = np.unique(values[np.isfinite(values)])
    if unique.size < 2:
        return None
    steps = np.diff(unique)
    steps = steps[steps > np.finfo(np.float64).eps]
    return float(np.median(steps)) if steps.size else None


def read_waveform(
    raw: bytes,
    config: dict[str, Any],
) -> Waveform:
    """Parse, calibrate, interpolate and offset-correct one PicoScope CSV."""

    leading = _read_leading_lines(raw)
    if len(leading) < 3:
        raise WaveformParseError("file has fewer than the two required header rows")
    headers = [field.strip() for field in _csv_fields(leading[0][2])]
    units = [field.strip() for field in _csv_fields(leading[1][2])]
    if len(headers) != len(units):
        raise WaveformParseError(
            f"header/unit column mismatch ({len(headers)} vs {len(units)})"
        )
    numeric_line = next((item for item in leading[2:] if _looks_numeric(item[2])), None)
    if numeric_line is None:
        raise WaveformParseError("numeric data did not begin within the first 12 rows")
    matrix, malformed_remainder = _numeric_matrix(raw, len(headers), numeric_line[0])
    role_indices = _role_indices(
        headers,
        config.get("calibration", {}).get("channel_roles", {}),
    )

    time_raw = matrix[:, role_indices["time"]]
    keep = np.isfinite(time_raw)
    skipped_rows = int(np.count_nonzero(~keep)) + malformed_remainder
    matrix = matrix[keep]
    if matrix.shape[0] < 32:
        raise WaveformParseError("fewer than 32 finite-time samples remain")

    role_to_kind = {
        "time": "time",
        "applied_voltage": "voltage",
        "legacy_current": "current",
        "charge_monitor": "voltage",
    }
    clip_counts: dict[str, int] = {}
    code_counts: dict[str, int] = {}
    lsb: dict[str, float | None] = {}
    interpolated: dict[str, np.ndarray] = {}
    for role, index in role_indices.items():
        raw_values = matrix[:, index]
        clip_counts[role] = int(np.count_nonzero(~np.isfinite(raw_values)))
        code_counts[role] = _code_count(raw_values)
        lsb[role] = _lsb(raw_values)
        values = _interpolate(raw_values)
        values *= _unit_scale(units[index], role_to_kind[role])
        interpolated[role] = values

    time_s = interpolated["time"]
    if np.any(np.diff(time_s) <= 0):
        order = np.argsort(time_s, kind="stable")
        time_s = time_s[order]
        for role in interpolated:
            if role != "time":
                interpolated[role] = interpolated[role][order]
    dt = np.diff(time_s)
    if not np.all(np.isfinite(dt)) or float(np.median(dt)) <= 0:
        raise WaveformParseError("time base is not finite and increasing")

    divider = float(config["calibration"]["channel_a_divider"]["value"])
    applied_voltage_V = interpolated["applied_voltage"] * divider
    monitor_voltage_V = interpolated["charge_monitor"]
    raw_monitor = monitor_voltage_V.copy()
    dc_offset_V = float(np.mean(raw_monitor))

    centered_time = time_s - float(np.mean(time_s))
    denom = float(np.dot(centered_time, centered_time))
    drift = float(np.dot(centered_time, raw_monitor - dc_offset_V) / denom) if denom else 0.0
    excursion = abs(drift) * float(np.ptp(time_s))
    robust_swing = float(np.percentile(raw_monitor, 95) - np.percentile(raw_monitor, 5))
    threshold = float(config["analysis"].get("detrend_fraction_threshold", 0.10))
    detrended = bool(robust_swing > 0 and excursion > threshold * robust_swing)
    monitor_voltage_V = raw_monitor - dc_offset_V
    if detrended:
        monitor_voltage_V = monitor_voltage_V - drift * centered_time

    capacitance_F = float(config["calibration"]["measuring_capacitor_F"]["value"])
    charge_nC = capacitance_F * monitor_voltage_V * 1.0e9
    current = interpolated.get("legacy_current")
    if current is not None:
        current = current - float(np.mean(current))

    return Waveform(
        time_s=time_s,
        applied_voltage_V=applied_voltage_V,
        current_A=current,
        monitor_voltage_V=monitor_voltage_V,
        charge_nC=charge_nC,
        headers=headers,
        units=units,
        role_indices=role_indices,
        clip_counts=clip_counts,
        code_counts=code_counts,
        lsb=lsb,
        n_samples_raw=int(matrix.shape[0] + skipped_rows),
        skipped_rows=skipped_rows,
        dc_offset_V=dc_offset_V,
        drift_V_per_s=drift,
        detrended=detrended,
    )


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.casefold()).strip("_")


def parse_path_metadata(relative_path: str, config: dict[str, Any]) -> dict[str, Any]:
    """Extract folder/drive labels while keeping measured carrier frequency separate."""

    posix = PurePosixPath(_normal_rel(relative_path))
    parts = list(posix.parts)
    lower_parts = [part.casefold() for part in parts]
    text = "/".join(parts)
    lower = text.casefold()
    top = parts[0] if parts else ""

    medium = "unknown"
    dataset_type = "unclassified"
    source_type = "raw_capture"
    run_key = ""
    if top.casefold() == "lissajouswaveformsdifferentmediums" and len(parts) >= 2:
        medium_map = {
            "argon": "argon_only",
            "pure water": "pure_water",
            "ionic liquid": "BMIM_nitrate",
            "manganese nitrate in water": "5mM_Mn_nitrate_in_water",
        }
        medium = medium_map.get(parts[1].casefold(), _slug(parts[1]))
        if "breakdown" in lower:
            dataset_type = "voltage_ladder"
            source_type = "ladder_capture"
        else:
            dataset_type = "july7_8_operational"
            source_type = "operational_capture"
    elif top.casefold() == "7_20":
        medium = "dry_fixture"
        dataset_type = "dispersion_7_20"
        source_type = "multiline_7_20"
    elif re.fullmatch(r"7_\d+", top.casefold()) and len(parts) >= 2:
        run_key = f"{parts[0]}/{parts[1]}"
        medium = _slug(parts[1])
        dataset_type = "synthesis"
        source_type = "synthesis_capture"

    frequency_match = re.search(r"(?<!\d)(\d+(?:\.\d+)?)\s*k\s*hz", lower)
    freq_value = float(frequency_match.group(1)) if frequency_match else None
    freq_label = f"{freq_value:g} kHz" if freq_value is not None else ""

    level_pct: float | None = None
    percent_match = re.search(r"(\d+(?:\.\d+)?)%\s*breakdown", lower)
    if percent_match:
        level_pct = float(percent_match.group(1))
    else:
        encoded_match = re.search(r"(?:^|[/_])0_(4|6|75|9)breakdown", lower)
        if encoded_match:
            encoded = {"4": 40.0, "6": 60.0, "75": 75.0, "9": 90.0}
            level_pct = encoded[encoded_match.group(1)]
        elif re.search(r"(?:^|/)105breakdown", lower):
            level_pct = 105.0
        elif re.search(r"(?:^|/)115breakdown", lower):
            level_pct = 115.0
        elif any(part == "breakdown" or re.match(r"breakdown-\d+$", part) for part in lower_parts):
            level_pct = 100.0

    parent = parts[-2] if len(parts) >= 2 else ""
    command_match = re.search(
        r"(?P<freq>\d+(?:\.\d+)?)\s*k\s*hz\s*(?P<kv>\d+(?:\.\d+)?)\s*kv",
        parent.casefold(),
    )
    commanded_kV = float(command_match.group("kv")) if command_match else None

    acquisition_idx = None
    for part in reversed(parts[:-1]):
        match = re.search(r"-(\d{4})$", part)
        if match:
            acquisition_idx = int(match.group(1))
            break
    segment_match = re.search(r"_(\d+)\.csv$", parts[-1], flags=re.IGNORECASE)
    segment_idx = int(segment_match.group(1)) if segment_match else None

    if dataset_type == "synthesis":
        cond = _slug(run_key)
    elif dataset_type == "dispersion_7_20":
        cond = _slug(parent)
    else:
        level_label = "MAX" if level_pct is None else f"{level_pct:g}"
        cond = _slug(f"{medium}_{freq_label}_{level_label}")

    exclusion_flags: list[str] = []
    contaminated = bool(dataset_type == "dispersion_7_20" and commanded_kV is not None and commanded_kV >= 2)
    if contaminated:
        exclusion_flags.append("surface_discharge_contaminated_not_displacement_only")

    synthesis_cfg = config.get("synthesis_runs", {}).get(run_key, {})
    display_label = synthesis_cfg.get("label", run_key or cond)
    return {
        "path": posix.as_posix(),
        "top_folder": top,
        "dataset_type": dataset_type,
        "source_type": source_type,
        "medium": medium,
        "medium_display": medium_display(config, medium),
        "run_key": run_key,
        "display_label": display_label,
        "freq_label": freq_label,
        "burst_frequency_label_kHz": freq_value,
        # Backward-compatible source-label alias retained for v2.0 readers.
        "nominal_frequency_kHz": freq_value,
        "level_pct": level_pct,
        "commanded_kV": commanded_kV,
        "save_idx": acquisition_idx,
        "seg_idx": segment_idx,
        "cond": cond,
        "contaminated": contaminated,
        "exclusion_flags": exclusion_flags,
    }
