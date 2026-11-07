"""Audit every XA cine in a study and emit a machine-readable manifest."""

from __future__ import annotations

import argparse
from pathlib import Path

import pydicom

try:
    from .xa_cine import audit_record, write_json
except ImportError:  # direct script execution
    from xa_cine import audit_record, write_json


def find_xa_dicoms(root: Path) -> list[Path]:
    results: list[Path] = []
    for path in sorted(root.rglob("*.DCM")):
        try:
            header = pydicom.dcmread(path, stop_before_pixels=True, specific_tags=["Modality"])
        except Exception:
            continue
        if str(getattr(header, "Modality", "")) == "XA":
            results.append(path)
    return results


def build_manifest(xa_dir: Path) -> dict[str, object]:
    records = [audit_record(path, xa_dir) for path in find_xa_dicoms(xa_dir)]
    return {
        "schema_version": 1,
        "source_root": str(xa_dir),
        "series_count": len(records),
        "total_frame_count": sum(int(item["frame_count"]) for item in records),
        "records": records,
        "interpretation": {
            "coordinate_policy": "DICOM row/column order; no geometric flip",
            "absolute_pose_warning": "Angles and distances describe C-arm geometry, but missing patient-frame tags prevent an independent absolute CT pose.",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--xa-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = build_manifest(args.xa_dir)
    write_json(args.output, manifest)
    print(f"Wrote {args.output} ({manifest['series_count']} series, {manifest['total_frame_count']} frames)")


if __name__ == "__main__":
    main()
