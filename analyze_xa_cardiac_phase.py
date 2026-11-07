"""Generate the research-only image-derived XA cardiac phase report."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "tools"))

from xa_cardiac_phase import build_phase_alignment


def atomic_write_json(output: Path, report: dict) -> None:
    """Write a completed report atomically, preserving an existing target on failure."""
    target = Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=target.parent,
            prefix=f".{target.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            json.dump(report, temporary, ensure_ascii=False, indent=2, allow_nan=False)
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_path, target)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--series-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    report = build_phase_alignment(args.series_root)
    atomic_write_json(args.output, report)
    print(json.dumps({"output": str(args.output), "status": report["status"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
