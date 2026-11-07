"""Transactional publication for generated directory trees."""

from __future__ import annotations

import shutil
import time
import uuid
from pathlib import Path
from typing import Callable, TypeVar


T = TypeVar("T")


def _remove_private_tree(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)


def _replace_directory(source: Path, target: Path) -> Path:
    for attempt in range(5):
        try:
            return source.replace(target)
        except PermissionError:
            if attempt == 4:
                raise
            time.sleep(0.05 * (attempt + 1))
    raise AssertionError("unreachable")


def build_then_publish_directory(
    output_dir: Path,
    build: Callable[[Path], T],
    *,
    seed_from_existing: bool,
) -> T:
    """Build beside ``output_dir`` and swap it in only after full success."""
    output_dir = Path(output_dir)
    parent = output_dir.parent
    parent.mkdir(parents=True, exist_ok=True)
    transaction_id = uuid.uuid4().hex
    staging_dir = parent / f".{output_dir.name}.stage-{transaction_id}"
    backup_dir = parent / f".{output_dir.name}.backup-{transaction_id}"
    if output_dir.exists():
        if not output_dir.is_dir():
            raise ValueError(f"Output path is not a directory: {output_dir}")
        if seed_from_existing:
            shutil.copytree(output_dir, staging_dir)
        else:
            staging_dir.mkdir()
    else:
        staging_dir.mkdir()

    try:
        result = build(staging_dir)
    except BaseException:
        try:
            _remove_private_tree(staging_dir)
        except OSError:
            pass
        raise

    previous_moved = False
    try:
        if output_dir.exists():
            _replace_directory(output_dir, backup_dir)
            previous_moved = True
        _replace_directory(staging_dir, output_dir)
    except BaseException as publish_error:
        try:
            _remove_private_tree(staging_dir)
            if previous_moved:
                _replace_directory(backup_dir, output_dir)
        except OSError as rollback_error:
            raise RuntimeError(
                f"Failed to publish {output_dir} and restore its previous tree"
            ) from rollback_error
        raise publish_error

    if previous_moved:
        _remove_private_tree(backup_dir)
    return result
