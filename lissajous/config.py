"""Configuration loading and provenance helpers for Lissajous v2."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any


class ConfigError(RuntimeError):
    """Raised when the analysis configuration is incomplete or invalid."""


_CANONICAL_MEDIUM_TO_FOLDER = {
    "argon_only": "argon",
    "pure_water": "pure water",
    "BMIM_nitrate": "ionic liquid",
    "5mM_Mn_nitrate_in_water": "manganese nitrate in water",
}


def load_config(path: str | Path) -> dict[str, Any]:
    """Load JSON-compatible YAML, with optional PyYAML support for regular YAML."""

    config_path = Path(path).expanduser().resolve()
    text = config_path.read_text(encoding="utf-8-sig")
    try:
        config = json.loads(text)
    except json.JSONDecodeError:
        try:
            import yaml  # type: ignore
        except ImportError as exc:  # pragma: no cover - optional dependency path
            raise ConfigError(
                f"{config_path} is not JSON-compatible YAML and PyYAML is not installed"
            ) from exc
        config = yaml.safe_load(text)
    if not isinstance(config, dict):
        raise ConfigError(f"{config_path} must contain a mapping at its top level")
    required = ("analysis", "paths", "calibration", "chain_model")
    missing = [key for key in required if key not in config]
    if missing:
        raise ConfigError(f"{config_path} is missing required sections: {', '.join(missing)}")
    config = copy.deepcopy(config)
    config["_config_path"] = str(config_path)
    return config


def resolved_data_root(config: dict[str, Any], override: str | Path | None = None) -> Path:
    """Resolve a configured or command-line data root without hardcoded fallbacks."""

    raw = override if override is not None else config.get("paths", {}).get("data_root")
    if not raw:
        raise ConfigError("No data root was supplied and paths.data_root is empty")
    path = Path(str(raw)).expanduser()
    if not path.is_absolute():
        path = Path(config["_config_path"]).parent / path
    return path.resolve()


def resolved_output(config: dict[str, Any], override: str | Path | None = None) -> Path:
    """Resolve the output directory relative to the configuration file."""

    raw = override if override is not None else config.get("paths", {}).get("default_output")
    if not raw:
        raise ConfigError("No output path was supplied and paths.default_output is empty")
    path = Path(str(raw)).expanduser()
    if not path.is_absolute():
        path = Path(config["_config_path"]).parent / path
    return path.resolve()


def calibration_value(config: dict[str, Any], key: str) -> float:
    """Return one numeric calibration value with a useful validation error."""

    entry = config.get("calibration", {}).get(key)
    if not isinstance(entry, dict) or "value" not in entry:
        raise ConfigError(f"calibration.{key}.value is required")
    try:
        return float(entry["value"])
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"calibration.{key}.value must be numeric") from exc


def marker(entry: dict[str, Any]) -> str:
    """Return the epistemic marker (`*` or `†`) associated with an entry."""

    value = str(entry.get("marker", ""))
    return value if value in {"", "*", "\u2020"} else ""


def medium_metadata(config: dict[str, Any], medium: str) -> dict[str, Any]:
    """Resolve one canonical analysis key through the folder-label provenance map."""

    labels = config.get("medium_labels", {})
    if not isinstance(labels, dict):
        return {}
    source_key = _CANONICAL_MEDIUM_TO_FOLDER.get(str(medium), str(medium))
    entry = labels.get(source_key)
    return dict(entry) if isinstance(entry, dict) else {}


def medium_display(config: dict[str, Any], medium: str) -> str:
    """Return the configured display label while preserving unknown identifiers."""

    entry = medium_metadata(config, medium)
    display = entry.get("display")
    return str(display) if display not in (None, "") else str(medium).replace("_", " ")


def output_config(config: dict[str, Any]) -> dict[str, Any]:
    """Return the deterministic, serializable configuration written to outputs."""

    result = copy.deepcopy(config)
    result.pop("_config_path", None)
    return result
