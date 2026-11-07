"""Explicit user-local data/output roots for non-production research probes."""

from __future__ import annotations

import os
from pathlib import Path


def research_roots() -> tuple[Path, str]:
    """Return data root and case key supplied by the caller, never guessed."""

    raw_root = os.environ.get("HEART_REGISTRY_DATA_ROOT")
    case_id = os.environ.get("HEART_REGISTRY_CASE_ID")
    if not raw_root or not case_id:
        raise RuntimeError(
            "Set HEART_REGISTRY_DATA_ROOT and HEART_REGISTRY_CASE_ID for this research probe"
        )
    if not case_id.isdigit():
        raise ValueError("HEART_REGISTRY_CASE_ID must match the case-folder numeric key")
    root = Path(raw_root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Research data root does not exist: {root}")
    return root, case_id
