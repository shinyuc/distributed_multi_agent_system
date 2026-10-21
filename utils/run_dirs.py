"""Collision-resistant output directories for simulation runs.

Every run gets its own directory, created with an exclusive ``mkdir``:

    <root>/<YYYYmmdd_HHMMSS_ffffff>_<config-stem>_p<pid>_<4 hex>

The microsecond timestamp, the process id and a random suffix make two runs started
at the same instant pick different names; ``mkdir(exist_ok=False)`` makes the choice
atomic, so even an identical name can never be shared (the loser retries).

An explicitly requested directory is used only if it does not exist yet or is empty.
A non-empty directory is an error unless ``overwrite=True``, in which case the old
directory is renamed to ``<name>.replaced_<timestamp>`` (never deleted) and a fresh
empty directory is created in its place.
"""

from __future__ import annotations

import os
import re
import secrets
from datetime import datetime
from pathlib import Path


class OutputDirExistsError(FileExistsError):
    """Raised when an explicit output directory already contains files."""


def _timestamp(now: datetime | None = None) -> str:
    return (now or datetime.now()).strftime("%Y%m%d_%H%M%S_%f")


def _slug(label: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", label).strip("-") or "run"


def _suffix() -> str:
    return secrets.token_hex(2)


def make_unique_run_dir(
    root: str | Path,
    label: str,
    *,
    now: datetime | None = None,
    max_attempts: int = 100,
) -> Path:
    """Create and return a new, empty, uniquely named run directory under ``root``."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    stamp = _timestamp(now)
    base = f"{stamp}_{_slug(label)}_p{os.getpid()}"
    for _ in range(max_attempts):
        candidate = root / f"{base}_{_suffix()}"
        try:
            candidate.mkdir(exist_ok=False)
        except FileExistsError:
            continue
        return candidate
    raise RuntimeError(f"Could not create a unique run directory under {root} after {max_attempts} attempts.")


def prepare_explicit_dir(path: str | Path, *, overwrite: bool = False) -> tuple[Path, Path | None]:
    """Use ``path`` as the run directory without silently overwriting results.

    Returns ``(path, moved_to)``; ``moved_to`` is where previous contents were moved
    when ``overwrite`` is set (``None`` otherwise).
    """
    path = Path(path)
    if path.exists() and not path.is_dir():
        raise OutputDirExistsError(f"Output path {path} exists and is not a directory.")
    if path.is_dir() and any(path.iterdir()):
        if not overwrite:
            raise OutputDirExistsError(
                f"Output directory {path} already contains results. Choose a new directory, omit the "
                f"explicit directory to get a unique timestamped one, or pass --overwrite to move the "
                f"existing contents aside (they are renamed, never deleted)."
            )
        moved_to = path.with_name(f"{path.name}.replaced_{_timestamp()}_{_suffix()}")
        path.rename(moved_to)
        path.mkdir(parents=True, exist_ok=False)
        return path, moved_to
    path.mkdir(parents=True, exist_ok=True)
    return path, None
