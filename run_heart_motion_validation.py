"""CLI for isolated, research-only CT/XA motion validation foundations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ctdr_pipeline.motion_validation.runner import run_validation


def main() -> None:
    parser = argparse.ArgumentParser(description="Research-only CT/XA motion observability and synthetic validation; no patient 3D reconstruction.")
    parser.add_argument("--case-dir", type=Path, required=True)
    parser.add_argument("--assets-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True, help="A new directory outside all input directories; never overwritten.")
    args = parser.parse_args()
    report = run_validation(args.case_dir, args.assets_root, args.output_dir)
    print(json.dumps({"output_dir": report["output_dir"], "gates": report["gates"], "source_integrity": report["source_integrity"], "patient_3d_optimization_enabled": False}, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
