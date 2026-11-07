"""Export one XA multi-frame DICOM into frames, raw stack, and preview."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

try:
    from .xa_cine import audit_record, read_cine, write_frames, write_json, write_mp4, write_raw_stack
except ImportError:  # direct script execution
    from xa_cine import audit_record, read_cine, write_frames, write_json, write_mp4, write_raw_stack


def extract_cine(dicom_path: Path, output_dir: Path) -> dict[str, object]:
    dataset, frames, frame_times = read_cine(dicom_path)
    frame_paths = write_frames(frames, output_dir / "frames")
    raw_path = output_dir / "cine_u8.raw"
    raw_bytes = write_raw_stack(raw_path, frames)
    preview_path = output_dir / "raw_preview.mp4"
    write_mp4(preview_path, frames, frame_times)
    record = audit_record(dicom_path)
    manifest = {
        "schema_version": 1,
        "source_dicom": str(dicom_path),
        "source_record": record,
        "shape_frame_row_column": list(map(int, frames.shape)),
        "dtype": str(frames.dtype),
        "frame_times_ms": [float(value) for value in frame_times],
        "median_frame_time_ms": float(np.median(frame_times)),
        "raw_stack": {"path": raw_path.name, "byte_length": raw_bytes, "order": "frame,row,column"},
        "frame_png_pattern": "frames/frame_%04d.png",
        "frame_png_count": len(frame_paths),
        "preview_mp4": preview_path.name,
        "orientation_policy": "No horizontal/vertical flip; DICOM row 0 remains image top.",
    }
    write_json(output_dir / "series_manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dicom", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    manifest = extract_cine(args.dicom, args.output_dir)
    print(f"Extracted {manifest['frame_png_count']} frames to {args.output_dir}")


if __name__ == "__main__":
    main()
