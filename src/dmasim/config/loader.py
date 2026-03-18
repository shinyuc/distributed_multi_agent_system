from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

from dmasim.config.schema import RootConfig, from_dict


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _resolve_extends(path: Path, visited: set[Path] | None = None) -> dict[str, Any]:
    path = path.resolve()
    visited = visited or set()
    if path in visited:
        raise ValueError(f"Cyclic config extends detected at: {path}")
    visited.add(path)

    raw = _load_yaml(path)
    if "extends" not in raw:
        return raw

    parent_path = Path(raw["extends"])
    if not parent_path.is_absolute():
        parent_path = (path.parent / parent_path).resolve()
    parent = _resolve_extends(parent_path, visited=visited)

    child = {k: v for k, v in raw.items() if k != "extends"}
    return _deep_merge(parent, child)


def load_config(path: str | Path) -> RootConfig:
    cfg_path = Path(path).resolve()
    raw = _resolve_extends(cfg_path)
    return from_dict(raw)
