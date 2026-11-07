"""Create an explicitly diagnostic-only CT pose for one XA run.

The run has valid C-arm angles, SOD/SDD, and detector spacing, but it lacks a
FrameOfReferenceUID and absolute table/patient pose.  We therefore transfer
only the same-exam Series 7 isocenter and device angle convention.  This is a
useful visual initialization, not an independently solved 3D-2D registration.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pydicom


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def distance_to_box(point: np.ndarray, minimum: np.ndarray, maximum: np.ndarray) -> float:
    delta = np.maximum(np.maximum(minimum - point, 0.0), point - maximum)
    return float(np.linalg.norm(delta))


def calibrate_series(dicom_path: Path, reference_report: Path, scene_path: Path) -> dict[str, Any]:
    # Importing from the existing viewer generator keeps the convention exactly
    # aligned with the already reviewed Series 7 scene.
    import sys
    sys.path.insert(0, str(PROJECT_ROOT))
    try:
        from build_ct_dr_3d_viewer import OrientationCandidate, detector_basis, detector_plane_geometry
    finally:
        sys.path.pop(0)

    dataset = pydicom.dcmread(dicom_path, stop_before_pixels=True)
    series_number = int(dataset.SeriesNumber)
    run_id = f"series_{series_number:02d}"
    reference = json.loads(reference_report.read_text(encoding="utf-8"))["selected"]
    scene = json.loads(scene_path.read_text(encoding="utf-8"))
    candidate = OrientationCandidate(**reference["candidate"])
    delta_primary = float(reference["delta_primary_deg"])
    delta_secondary = float(reference["delta_secondary_deg"])
    primary = float(dataset.PositionerPrimaryAngle)
    secondary = float(dataset.PositionerSecondaryAngle)
    basis = detector_basis(primary, secondary, candidate, delta_primary, delta_secondary)

    isocenter = np.asarray(reference["isocenter_lps_mm"], dtype=np.float64)
    sdd = float(dataset.DistanceSourceToDetector)
    sod = float(dataset.DistanceSourceToPatient)
    source = isocenter - sod * basis.n_lps
    detector_center = isocenter + (sdd - sod) * basis.n_lps
    rows = int(dataset.Rows)
    columns = int(dataset.Columns)
    principal = np.asarray([(columns - 1) / 2.0, (rows - 1) / 2.0], dtype=np.float64)

    calibration = {
        "candidate": asdict(candidate),
        "candidate_name": candidate.name,
        "isocenter_lps_mm": isocenter.tolist(),
        "delta_primary_deg": delta_primary,
        "delta_secondary_deg": delta_secondary,
        "principal_offset_uv_px": [0.0, 0.0],
        "principal_uv_px": principal.tolist(),
        "effective_primary_deg": float(basis.effective_primary_deg),
        "effective_secondary_deg": float(basis.effective_secondary_deg),
        "u_lps": basis.u_lps.tolist(),
        "v_lps": basis.v_lps.tolist(),
        "n_lps": basis.n_lps.tolist(),
        "source_lps_mm": source.tolist(),
        "detector_center_lps_mm": detector_center.tolist(),
        "sdd_mm": sdd,
        "sod_mm": sod,
        "dicom_primary_angle_deg": primary,
        "dicom_secondary_angle_deg": secondary,
        "pixel_spacing_row_col_mm": [float(value) for value in dataset.ImagerPixelSpacing],
    }
    detector = detector_plane_geometry(calibration, dataset)
    ct_min = np.asarray(scene["ct"]["bounds_min_lps_mm"], dtype=np.float64)
    ct_max = np.asarray(scene["ct"]["bounds_max_lps_mm"], dtype=np.float64)
    orthogonality = max(
        abs(float(np.dot(basis.u_lps, basis.v_lps))),
        abs(float(np.dot(basis.u_lps, basis.n_lps))),
        abs(float(np.dot(basis.v_lps, basis.n_lps))),
    )
    consistency = {
        "basis_max_abs_dot": orthogonality,
        "source_to_detector_distance_mm": float(np.linalg.norm(detector_center - source)),
        "source_to_isocenter_distance_mm": float(np.linalg.norm(isocenter - source)),
        "detector_to_isocenter_distance_mm": float(np.linalg.norm(detector_center - isocenter)),
        "isocenter_distance_to_ct_bbox_mm": distance_to_box(isocenter, ct_min, ct_max),
        "expected_source_to_detector_mm": sdd,
        "expected_source_to_isocenter_mm": sod,
    }
    return {
        "schema_version": 1,
        "series": f"XA Series {series_number}",
        "series_number": series_number,
        "run_id": run_id,
        "source_dicom": str(dicom_path),
        "quality_status": "diagnostic-only",
        "quality_gate_passed": False,
        "quality_reasons": [
            "Uses the Series 7 landmark-fit isocenter as a same-exam initialization.",
            f"Uses Series 7 device sign convention and angle correction; no Series {series_number}-specific landmarks were fitted.",
            f"Series {series_number} lacks FrameOfReferenceUID, ImagePositionPatient, ImageOrientationPatient, and absolute table position.",
            "The detector plane is suitable for visual exploration but not for navigation or quantitative clinical use.",
        ],
        "transferred_from": {
            "reference_report": str(reference_report),
            "reference_series": "XA Series 7",
            "reference_rms_error_px": float(reference["rms_error_px"]),
            "transferred_parameters": ["isocenter_lps_mm", "candidate", "delta_primary_deg", "delta_secondary_deg"],
            "not_transferred": ["principal_offset_uv_px"],
        },
        "calibration": calibration,
        "detector": detector,
        "geometry_consistency": consistency,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dicom", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = calibrate_series(args.dicom, args.reference, args.scene)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"quality_status": result["quality_status"], **result["geometry_consistency"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
