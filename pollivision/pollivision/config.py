"""Configuration loading.

Configuration is plain YAML with a two-level merge: a base file plus optional
overlays (a species profile, a camera profile, CLI overrides). Keeping it as
nested dicts rather than a rigid schema means field technicians can retune
thresholds without touching Python.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Iterable, Optional

import yaml

CONFIG_ROOT = Path(__file__).resolve().parent.parent / "configs"


def _deep_merge(base: dict, overlay: dict) -> dict:
    """Recursively merge ``overlay`` into ``base``, returning a new dict."""
    out = copy.deepcopy(base)
    for key, value in overlay.items():
        if key in out and isinstance(out[key], dict) and isinstance(value, dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


class Config:
    """Nested-dict config with dotted-path access.

    ``cfg.get("perception.sex.weights.vlm", 1.0)`` reads a nested key with a
    default, which keeps call sites free of defensive dict chaining.
    """

    def __init__(self, data: Optional[dict] = None) -> None:
        self._data: dict[str, Any] = data or {}

    # -- access ------------------------------------------------------------ #

    def get(self, path: str, default: Any = None) -> Any:
        node: Any = self._data
        for part in path.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node

    def section(self, path: str) -> "Config":
        value = self.get(path, {})
        return Config(value if isinstance(value, dict) else {})

    def set(self, path: str, value: Any) -> None:
        parts = path.split(".")
        node = self._data
        for part in parts[:-1]:
            node = node.setdefault(part, {})
            if not isinstance(node, dict):
                raise TypeError(f"Cannot set '{path}': '{part}' is not a mapping")
        node[parts[-1]] = value

    def __contains__(self, path: str) -> bool:
        sentinel = object()
        return self.get(path, sentinel) is not sentinel

    def __repr__(self) -> str:
        return f"Config({json.dumps(self._data, default=str)[:200]}...)"

    @property
    def data(self) -> dict:
        return self._data

    def merged(self, overlay: dict | "Config") -> "Config":
        other = overlay.data if isinstance(overlay, Config) else overlay
        return Config(_deep_merge(self._data, other))

    def to_yaml(self) -> str:
        return yaml.safe_dump(self._data, sort_keys=False)


def _read_yaml(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def load_config(
    base: str | Path | None = None,
    overlays: Optional[Iterable[str | Path]] = None,
    overrides: Optional[dict] = None,
) -> Config:
    """Load ``base`` then apply overlays and a final dict of overrides.

    Names without a path separator resolve against the bundled ``configs/``
    directory, so ``load_config("default", ["species/pumpkin"])`` works.
    """
    base = base or "default"
    cfg = Config(_read_yaml(_resolve(base)))
    for overlay in overlays or []:
        cfg = cfg.merged(_read_yaml(_resolve(overlay)))
    if overrides:
        cfg = cfg.merged(overrides)
    return cfg


def _resolve(name: str | Path) -> Path:
    path = Path(name)
    if path.suffix in {".yaml", ".yml"} and path.exists():
        return path
    for candidate in (
        CONFIG_ROOT / f"{name}.yaml",
        CONFIG_ROOT / f"{name}.yml",
        CONFIG_ROOT / str(name),
        path,
    ):
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"No config named '{name}' (looked under {CONFIG_ROOT})")


def parse_overrides(items: Iterable[str]) -> dict:
    """Turn ``["a.b=3", "c=hello"]`` into a nested override dict.

    Values are parsed as YAML scalars so numbers, booleans and lists work.
    """
    out: dict = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"Override '{item}' is not of the form key.path=value")
        key, _, raw = item.partition("=")
        value = yaml.safe_load(raw)
        node = out
        parts = key.strip().split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = value
    return out
