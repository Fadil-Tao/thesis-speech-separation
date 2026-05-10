# -*- coding: utf-8 -*-
"""Portable path resolution for dataset generators and training scripts.

Resolution priority (highest first):
    1. Explicit argument passed by caller (CLI flag in scripts).
    2. Environment variable.
    3. Default derived from project root.

Project root is auto-detected by walking up from this file (``utils/paths.py``
sits one level below the repo root). Override with ``TSS_PROJECT_ROOT`` if
the repo is mounted at an unusual location.

Environment variables:
    TSS_PROJECT_ROOT    Override repo root.
    TSS_RAW_DIR         Path to raw TITML-IDN corpus.
    TSS_SYNTHETIC_DIR   Path to synthetic mixtures parent dir.
    TSS_CHECKPOINT_DIR  Path to checkpoints parent dir.

Typical cloud workflow:
    export TSS_RAW_DIR=/workspace/data/TTML-IDN
    export TSS_SYNTHETIC_DIR=/workspace/data/synthetic
    export TSS_CHECKPOINT_DIR=/workspace/checkpoints
"""

from __future__ import annotations

import os
from pathlib import Path


_THIS_FILE = Path(__file__).resolve()
_DEFAULT_PROJECT_ROOT = _THIS_FILE.parent.parent


def get_project_root() -> Path:
    env = os.environ.get("TSS_PROJECT_ROOT")
    return Path(env).expanduser().resolve() if env else _DEFAULT_PROJECT_ROOT


def get_raw_dir(override: str | os.PathLike | None = None) -> Path:
    if override:
        return Path(override).expanduser().resolve()
    env = os.environ.get("TSS_RAW_DIR")
    if env:
        return Path(env).expanduser().resolve()
    return get_project_root() / "dataset" / "raw" / "TTML-IDN"


def get_synthetic_dir(
    name: str | None = None,
    override: str | os.PathLike | None = None,
) -> Path:
    """Return synthetic dataset dir.

    If ``override`` is given, it is treated as the *final* path and returned
    as-is (caller already pointed at the intended dataset). Otherwise the
    base comes from ``$TSS_SYNTHETIC_DIR`` or the project default, and
    ``name`` (e.g. ``TITML-2spk``) is appended when provided.
    """
    if override:
        return Path(override).expanduser().resolve()
    env = os.environ.get("TSS_SYNTHETIC_DIR")
    base = Path(env).expanduser().resolve() if env else get_project_root() / "dataset" / "synthetic"
    return base / name if name else base


def get_checkpoint_dir(
    *parts: str,
    override: str | os.PathLike | None = None,
) -> Path:
    """Return checkpoint dir, optionally joined with subpath parts."""
    if override:
        base = Path(override).expanduser().resolve()
    else:
        env = os.environ.get("TSS_CHECKPOINT_DIR")
        if env:
            base = Path(env).expanduser().resolve()
        else:
            base = get_project_root() / "checkpoints"
    for p in parts:
        base = base / p
    return base
