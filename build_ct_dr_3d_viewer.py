"""Build an offline interactive CT/DR research viewer from explicit inputs.

The generator performs four tasks:

1. Reconstruct the target XA detector pose from DICOM geometry and the four
   physician landmarks using a constrained physical projection model.
2. Export a downsampled 16-bit CT volume for interactive orthogonal slicing.
3. Convert the labelled vertebral-body masks to a compact GLB surface model.
4. Package all assets with a pinned local Three.js runtime.

All scene coordinates are DICOM patient LPS millimetres.  The 2D affine from
the annotation workflow is used only to recover corresponding XA pixels; it is
never promoted to a 3D camera matrix.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pydicom
import SimpleITK as sitk
from PIL import Image, ImageDraw
from scipy.optimize import least_squares
from skimage.measure import marching_cubes
import trimesh


PROJECT_ROOT = Path(__file__).resolve().parent
CASE_ID = "CASE-NOT-SET"


@dataclass(frozen=True)
class OrientationCandidate:
    primary_sign: int
    secondary_sign: int
    secondary_mode: str
    inplane_rotation_deg: int

    @property
    def name(self) -> str:
        return (
            f"p{self.primary_sign:+d}_s{self.secondary_sign:+d}_"
            f"{self.secondary_mode}_r{self.inplane_rotation_deg}"
        )


@dataclass(frozen=True)
class DetectorBasis:
    u_lps: np.ndarray
    v_lps: np.ndarray
    n_lps: np.ndarray
    effective_primary_deg: float
    effective_secondary_deg: float


def normalize(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float64)
    length = float(np.linalg.norm(vector))
    if length <= 1e-12:
        raise ValueError("Cannot normalize a zero vector")
    return vector / length


def axis_angle_matrix(axis: np.ndarray, angle_rad: float) -> np.ndarray:
    x, y, z = normalize(axis)
    c = math.cos(angle_rad)
    s = math.sin(angle_rad)
    one_minus_c = 1.0 - c
    return np.asarray(
        [
            [c + x * x * one_minus_c, x * y * one_minus_c - z * s, x * z * one_minus_c + y * s],
            [y * x * one_minus_c + z * s, c + y * y * one_minus_c, y * z * one_minus_c - x * s],
            [z * x * one_minus_c - y * s, z * y * one_minus_c + x * s, c + z * z * one_minus_c],
        ],
        dtype=np.float64,
    )


def detector_basis(
    primary_dicom_deg: float,
    secondary_dicom_deg: float,
    candidate: OrientationCandidate,
    delta_primary_deg: float = 0.0,
    delta_secondary_deg: float = 0.0,
) -> DetectorBasis:
    primary = candidate.primary_sign * primary_dicom_deg + delta_primary_deg
    secondary = candidate.secondary_sign * secondary_dicom_deg + delta_secondary_deg

    # AP baseline: image columns increase toward +LPS X; image rows increase
    # caudally (-LPS Z); rays travel from anterior to posterior (+LPS Y).
    u = np.asarray([1.0, 0.0, 0.0])
    v = np.asarray([0.0, 0.0, -1.0])
    n = np.asarray([0.0, 1.0, 0.0])

    primary_rotation = axis_angle_matrix([0.0, 0.0, 1.0], math.radians(primary))
    u = primary_rotation @ u
    v = primary_rotation @ v
    n = primary_rotation @ n

    if candidate.secondary_mode == "rotated_u":
        secondary_axis = u
    elif candidate.secondary_mode == "patient_x":
        secondary_axis = np.asarray([1.0, 0.0, 0.0])
    else:
        raise ValueError(f"Unknown secondary mode: {candidate.secondary_mode}")
    secondary_rotation = axis_angle_matrix(secondary_axis, math.radians(secondary))
    u = secondary_rotation @ u
    v = secondary_rotation @ v
    n = secondary_rotation @ n

    if candidate.inplane_rotation_deg == 180:
        u = -u
        v = -v
    elif candidate.inplane_rotation_deg != 0:
        inplane = axis_angle_matrix(n, math.radians(candidate.inplane_rotation_deg))
        u = inplane @ u
        v = inplane @ v

    u = normalize(u)
    v = normalize(v - np.dot(v, u) * u)
    n = normalize(np.cross(u, v))
    return DetectorBasis(u, v, n, primary, secondary)


def project_points(
    points_lps: np.ndarray,
    isocenter_lps: np.ndarray,
    basis: DetectorBasis,
    sdd_mm: float,
    sod_mm: float,
    pixel_spacing_row_col_mm: np.ndarray,
    principal_uv_px: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    points = np.asarray(points_lps, dtype=np.float64)
    isocenter = np.asarray(isocenter_lps, dtype=np.float64)
    row_spacing, column_spacing = np.asarray(pixel_spacing_row_col_mm, dtype=np.float64)
    source = isocenter - sod_mm * basis.n_lps
    detector_center = isocenter + (sdd_mm - sod_mm) * basis.n_lps
    point_vectors = points - source[None, :]
    ray_depth = point_vectors @ basis.n_lps
    scales = sdd_mm / ray_depth
    intersections = source[None, :] + scales[:, None] * point_vectors
    offsets = intersections - detector_center[None, :]
    pixels = np.column_stack(
        [
            principal_uv_px[0] + (offsets @ basis.u_lps) / column_spacing,
            principal_uv_px[1] + (offsets @ basis.v_lps) / row_spacing,
        ]
    )
    return pixels, intersections, ray_depth, source, detector_center


def transform_points(points_xy: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    points = np.atleast_2d(np.asarray(points_xy, dtype=np.float64))
    homogeneous = np.column_stack([points, np.ones(len(points), dtype=np.float64)])
    transformed = (np.asarray(matrix, dtype=np.float64) @ homogeneous.T).T
    return transformed[:, :2] / transformed[:, 2:3]


def load_xa_and_correspondences(
    registration_summary_path: Path,
) -> tuple[object, np.ndarray, list[str], dict[str, object]]:
    summary = json.loads(registration_summary_path.read_text(encoding="utf-8"))
    candidate = next(
        row
        for row in summary["dicom_search"]["xa_top_10"]
        if str(row["series_number"]) == "7" and int(row["frame_index"]) == 0
    )
    dicom_to_slide = np.asarray(candidate["homography_dicom_to_slide"], dtype=np.float64)
    slide_points = np.asarray(
        [[row["target_x_px"], row["target_y_px"]] for row in summary["landmarks"]],
        dtype=np.float64,
    )
    native_points = transform_points(slide_points, np.linalg.inv(dicom_to_slide))
    labels = [str(row["landmark_id"]) for row in summary["landmarks"]]
    dataset = pydicom.dcmread(candidate["path"], force=True)
    return dataset, native_points, labels, candidate


def candidate_orientations() -> list[OrientationCandidate]:
    return [
        OrientationCandidate(primary_sign, secondary_sign, secondary_mode, inplane)
        for secondary_mode in ("rotated_u", "patient_x")
        for primary_sign in (1, -1)
        for secondary_sign in (1, -1)
        for inplane in (0, 180)
    ]


def bbox_distance(point: np.ndarray, minimum: np.ndarray, maximum: np.ndarray) -> float:
    below = np.maximum(minimum - point, 0.0)
    above = np.maximum(point - maximum, 0.0)
    return float(np.linalg.norm(below + above))


def fit_candidate(
    candidate: OrientationCandidate,
    points_lps: np.ndarray,
    observed_uv: np.ndarray,
    weights: np.ndarray,
    primary_deg: float,
    secondary_deg: float,
    sdd_mm: float,
    sod_mm: float,
    spacing_row_col: np.ndarray,
    image_center_uv: np.ndarray,
    ct_bounds_min: np.ndarray,
    ct_bounds_max: np.ndarray,
) -> dict[str, object]:
    point_center = np.mean(points_lps, axis=0)
    lower_isocenter = ct_bounds_min - np.asarray([100.0, 100.0, 100.0])
    upper_isocenter = ct_bounds_max + np.asarray([100.0, 100.0, 100.0])
    initial_isocenter = np.clip(point_center, lower_isocenter, upper_isocenter)

    fixed_basis = detector_basis(primary_deg, secondary_deg, candidate)

    def fixed_residual(parameters: np.ndarray) -> np.ndarray:
        predicted, _, _, _, _ = project_points(
            points_lps,
            parameters,
            fixed_basis,
            sdd_mm,
            sod_mm,
            spacing_row_col,
            image_center_uv,
        )
        return ((predicted - observed_uv) * np.sqrt(weights)[:, None]).reshape(-1)

    fixed = least_squares(
        fixed_residual,
        initial_isocenter,
        bounds=(lower_isocenter, upper_isocenter),
        loss="soft_l1",
        f_scale=4.0,
        max_nfev=1000,
    )

    initial = np.concatenate([fixed.x, np.zeros(4, dtype=np.float64)])
    lower = np.concatenate([lower_isocenter, [-3.0, -3.0, -25.0, -25.0]])
    upper = np.concatenate([upper_isocenter, [3.0, 3.0, 25.0, 25.0]])

    def refined_residual(parameters: np.ndarray) -> np.ndarray:
        basis = detector_basis(
            primary_deg,
            secondary_deg,
            candidate,
            delta_primary_deg=float(parameters[3]),
            delta_secondary_deg=float(parameters[4]),
        )
        principal = image_center_uv + parameters[5:7]
        predicted, _, _, _, _ = project_points(
            points_lps,
            parameters[:3],
            basis,
            sdd_mm,
            sod_mm,
            spacing_row_col,
            principal,
        )
        data = ((predicted - observed_uv) * np.sqrt(weights)[:, None]).reshape(-1)
        # Keep the solution physically close to DICOM geometry.  The regularizer
        # is expressed in pixel-equivalent residuals and prevents four points
        # from freely defining an arbitrary camera.
        regularization = np.asarray(
            [
                parameters[3] / 1.5,
                parameters[4] / 1.5,
                parameters[5] / 12.0,
                parameters[6] / 12.0,
            ],
            dtype=np.float64,
        )
        return np.concatenate([data, regularization])

    refined = least_squares(
        refined_residual,
        initial,
        bounds=(lower, upper),
        loss="soft_l1",
        f_scale=3.0,
        max_nfev=2000,
    )
    parameters = refined.x
    basis = detector_basis(
        primary_deg,
        secondary_deg,
        candidate,
        delta_primary_deg=float(parameters[3]),
        delta_secondary_deg=float(parameters[4]),
    )
    principal = image_center_uv + parameters[5:7]
    predicted, intersections, ray_depth, source, detector_center = project_points(
        points_lps,
        parameters[:3],
        basis,
        sdd_mm,
        sod_mm,
        spacing_row_col,
        principal,
    )
    errors = np.linalg.norm(predicted - observed_uv, axis=1)
    rms = float(np.sqrt(np.mean(errors**2)))
    max_error = float(np.max(errors))
    order_consistent = bool(np.all(np.diff(predicted[:, 1]) > 0.0))
    points_between = bool(np.all((ray_depth > 0.0) & (ray_depth < sdd_mm)))
    orthogonality = np.asarray(
        [
            np.dot(basis.u_lps, basis.v_lps),
            np.dot(basis.u_lps, basis.n_lps),
            np.dot(basis.v_lps, basis.n_lps),
        ]
    )
    isocenter_bbox_distance = bbox_distance(parameters[:3], ct_bounds_min, ct_bounds_max)
    source_side_expected = bool(np.dot(basis.n_lps, [0.0, 1.0, 0.0]) > 0.5)
    score = rms
    if not order_consistent:
        score += 1000.0
    if not points_between:
        score += 1000.0
    if isocenter_bbox_distance > 120.0:
        score += isocenter_bbox_distance
    if not source_side_expected:
        score += 250.0

    return {
        "candidate": asdict(candidate),
        "candidate_name": candidate.name,
        "score": float(score),
        "success": bool(refined.success),
        "message": refined.message,
        "isocenter_lps_mm": parameters[:3].tolist(),
        "delta_primary_deg": float(parameters[3]),
        "delta_secondary_deg": float(parameters[4]),
        "principal_offset_uv_px": parameters[5:7].tolist(),
        "principal_uv_px": principal.tolist(),
        "effective_primary_deg": basis.effective_primary_deg,
        "effective_secondary_deg": basis.effective_secondary_deg,
        "u_lps": basis.u_lps.tolist(),
        "v_lps": basis.v_lps.tolist(),
        "n_lps": basis.n_lps.tolist(),
        "source_lps_mm": source.tolist(),
        "detector_center_lps_mm": detector_center.tolist(),
        "predicted_uv_px": predicted.tolist(),
        "detector_intersections_lps_mm": intersections.tolist(),
        "ray_depth_mm": ray_depth.tolist(),
        "errors_px": errors.tolist(),
        "rms_error_px": rms,
        "max_error_px": max_error,
        "order_consistent": order_consistent,
        "points_between_source_and_detector": points_between,
        "source_side_expected": source_side_expected,
        "basis_max_abs_dot": float(np.max(np.abs(orthogonality))),
        "isocenter_distance_to_ct_bbox_mm": isocenter_bbox_distance,
    }


def calibration_quality(result: dict[str, object]) -> tuple[str, list[str]]:
    reasons: list[str] = []
    if not result["order_consistent"]:
        reasons.append("landmark vertical order is inconsistent")
    if not result["points_between_source_and_detector"]:
        reasons.append("one or more landmarks are outside source-detector segment")
    if not result["source_side_expected"]:
        reasons.append("source/detector side is inconsistent with AP baseline")
    if float(result["basis_max_abs_dot"]) > 1e-5:
        reasons.append("detector basis is not orthogonal")
    if float(result["isocenter_distance_to_ct_bbox_mm"]) > 120.0:
        reasons.append("isocenter is implausibly far from CT volume")

    rms = float(result["rms_error_px"])
    maximum = float(result["max_error_px"])
    if reasons or rms > 10.0 or maximum > 15.0:
        if rms > 10.0:
            reasons.append("reprojection RMS exceeds 10 px")
        if maximum > 15.0:
            reasons.append("maximum reprojection error exceeds 15 px")
        return "diagnostic-only", reasons
    if rms >= 5.0 or maximum >= 10.0:
        reasons.append("reprojection is above accepted threshold")
        return "warning", reasons
    return "accepted", ["all physical and reprojection gates passed"]


def fit_calibration(
    points_lps: np.ndarray,
    observed_uv: np.ndarray,
    labels: list[str],
    xa_dataset: object,
    ct_bounds_min: np.ndarray,
    ct_bounds_max: np.ndarray,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    primary = float(xa_dataset.PositionerPrimaryAngle)
    secondary = float(xa_dataset.PositionerSecondaryAngle)
    sdd = float(xa_dataset.DistanceSourceToDetector)
    sod = float(xa_dataset.DistanceSourceToPatient)
    spacing = np.asarray(xa_dataset.ImagerPixelSpacing, dtype=np.float64)
    center = np.asarray([(int(xa_dataset.Columns) - 1) / 2.0, (int(xa_dataset.Rows) - 1) / 2.0])
    weights = np.asarray([1.0 if label in {"T1", "T2"} else 0.65 for label in labels])
    results = [
        fit_candidate(
            candidate,
            points_lps,
            observed_uv,
            weights,
            primary,
            secondary,
            sdd,
            sod,
            spacing,
            center,
            ct_bounds_min,
            ct_bounds_max,
        )
        for candidate in candidate_orientations()
    ]
    results.sort(key=lambda row: float(row["score"]))
    best = dict(results[0])
    status, reasons = calibration_quality(best)
    best["quality_status"] = status
    best["quality_reasons"] = reasons
    best["observed_uv_px"] = observed_uv.tolist()
    best["landmark_labels"] = labels
    return best, results


def sample_depth_sensitivity(
    depth_rows: list[dict[str, object]],
    observed_uv: np.ndarray,
    labels: list[str],
    xa_dataset: object,
    ct_bounds_min: np.ndarray,
    ct_bounds_max: np.ndarray,
    selected_candidate: OrientationCandidate,
    samples: int = 48,
) -> dict[str, object]:
    rng = np.random.default_rng(1234567)
    sources: list[np.ndarray] = []
    detectors: list[np.ndarray] = []
    isocenters: list[np.ndarray] = []
    rms_values: list[float] = []
    weights = np.asarray([1.0 if label in {"T1", "T2"} else 0.65 for label in labels])
    spacing = np.asarray(xa_dataset.ImagerPixelSpacing, dtype=np.float64)
    center = np.asarray([(int(xa_dataset.Columns) - 1) / 2.0, (int(xa_dataset.Rows) - 1) / 2.0])
    for _ in range(samples):
        sampled_points: list[np.ndarray] = []
        for row in depth_rows:
            reference = np.asarray(row["reference_plane_lps_mm"], dtype=np.float64)
            estimate = np.asarray(row["estimated_lps_mm"], dtype=np.float64)
            estimated_depth = float(row["estimated_depth_from_localizer_plane_mm"])
            normal = normalize((estimate - reference) / max(abs(estimated_depth), 1e-6))
            low, high = map(float, row["depth_interval_p10_p90_mm"])
            sampled_depth = rng.uniform(low, high)
            sampled_points.append(reference + sampled_depth * normal)
        result = fit_candidate(
            selected_candidate,
            np.asarray(sampled_points),
            observed_uv,
            weights,
            float(xa_dataset.PositionerPrimaryAngle),
            float(xa_dataset.PositionerSecondaryAngle),
            float(xa_dataset.DistanceSourceToDetector),
            float(xa_dataset.DistanceSourceToPatient),
            spacing,
            center,
            ct_bounds_min,
            ct_bounds_max,
        )
        sources.append(np.asarray(result["source_lps_mm"], dtype=np.float64))
        detectors.append(np.asarray(result["detector_center_lps_mm"], dtype=np.float64))
        isocenters.append(np.asarray(result["isocenter_lps_mm"], dtype=np.float64))
        rms_values.append(float(result["rms_error_px"]))

    def summarize(vectors: Iterable[np.ndarray]) -> dict[str, object]:
        array = np.asarray(list(vectors), dtype=np.float64)
        return {
            "mean_lps_mm": np.mean(array, axis=0).tolist(),
            "std_lps_mm": np.std(array, axis=0).tolist(),
            "p10_lps_mm": np.percentile(array, 10, axis=0).tolist(),
            "p90_lps_mm": np.percentile(array, 90, axis=0).tolist(),
        }

    return {
        "sample_count": samples,
        "sampling_range": "independent uniform sampling within each landmark P10-P90 depth interval",
        "source": summarize(sources),
        "detector_center": summarize(detectors),
        "isocenter": summarize(isocenters),
        "rms_error_px": {
            "mean": float(np.mean(rms_values)),
            "p10": float(np.percentile(rms_values, 10)),
            "p50": float(np.percentile(rms_values, 50)),
            "p90": float(np.percentile(rms_values, 90)),
            "max": float(np.max(rms_values)),
        },
    }


def image_edges_lps(image: sitk.Image) -> np.ndarray:
    size = np.asarray(image.GetSize(), dtype=np.float64)
    spacing = np.asarray(image.GetSpacing(), dtype=np.float64)
    origin = np.asarray(image.GetOrigin(), dtype=np.float64)
    direction = np.asarray(image.GetDirection(), dtype=np.float64).reshape(3, 3)
    corners: list[np.ndarray] = []
    for x in (-0.5, size[0] - 0.5):
        for y in (-0.5, size[1] - 0.5):
            for z in (-0.5, size[2] - 0.5):
                corners.append(origin + direction @ (np.asarray([x, y, z]) * spacing))
    return np.asarray(corners)


def validate_ct_geometry(ct_image: sitk.Image, ct_dir: Path, expected_series: str = "4") -> dict[str, object]:
    rows: list[tuple[int, np.ndarray, str, Path]] = []
    for path in ct_dir.glob("*.DCM"):
        header = pydicom.dcmread(str(path), stop_before_pixels=True, force=True)
        if str(getattr(header, "SeriesNumber", "")) != expected_series:
            continue
        rows.append(
            (
                int(header.InstanceNumber),
                np.asarray(header.ImagePositionPatient, dtype=np.float64),
                str(header.FrameOfReferenceUID),
                path,
            )
        )
    rows.sort(key=lambda row: row[0])
    if len(rows) != int(ct_image.GetSize()[2]):
        raise RuntimeError(f"CT slice count differs: DICOM={len(rows)}, NIfTI={ct_image.GetSize()[2]}")
    actual_first = np.asarray(ct_image.TransformIndexToPhysicalPoint((0, 0, 0)))
    actual_last = np.asarray(
        ct_image.TransformIndexToPhysicalPoint((0, 0, ct_image.GetSize()[2] - 1))
    )
    if not np.allclose(actual_first, rows[0][1], atol=0.05):
        raise RuntimeError(f"CT first-slice geometry mismatch: {actual_first} vs {rows[0][1]}")
    if not np.allclose(actual_last, rows[-1][1], atol=0.05):
        raise RuntimeError(f"CT last-slice geometry mismatch: {actual_last} vs {rows[-1][1]}")
    frame_uids = {row[2] for row in rows}
    if len(frame_uids) != 1:
        raise RuntimeError(f"CT series has multiple FrameOfReferenceUID values: {frame_uids}")
    return {
        "slice_count": len(rows),
        "first_instance": rows[0][0],
        "last_instance": rows[-1][0],
        "first_ipp_lps_mm": rows[0][1].tolist(),
        "last_ipp_lps_mm": rows[-1][1].tolist(),
        "frame_of_reference_uid": next(iter(frame_uids)),
        "first_path": str(rows[0][3].resolve()),
        "last_path": str(rows[-1][3].resolve()),
    }


def normalize_xa_frame(dataset: object, frame_index: int = 0) -> np.ndarray:
    pixels = np.asarray(dataset.pixel_array)
    if pixels.ndim == 2:
        image = pixels
    else:
        image = pixels[frame_index]
    image = image.astype(np.float32)
    low, high = np.percentile(image, [0.5, 99.5])
    image = np.clip((image - low) / max(high - low, 1e-6), 0.0, 1.0)
    if str(getattr(dataset, "PhotometricInterpretation", "")) == "MONOCHROME1":
        image = 1.0 - image
    return np.rint(image * 255.0).astype(np.uint8)


def window_ct(array: np.ndarray, center: float, width: float) -> np.ndarray:
    low = center - width / 2.0
    normalized = np.clip((array.astype(np.float32) - low) / width, 0.0, 1.0)
    return np.rint(normalized * 255.0).astype(np.uint8)


def save_initial_slice_pngs(
    ct_image: sitk.Image,
    landmark_points: np.ndarray,
    assets_dir: Path,
) -> dict[str, object]:
    ct = np.asarray(sitk.GetArrayViewFromImage(ct_image))
    center_point = np.median(landmark_points, axis=0)
    x_float, y_float, z_float = ct_image.TransformPhysicalPointToContinuousIndex(
        tuple(map(float, center_point))
    )
    x_index = int(np.clip(round(x_float), 0, ct_image.GetSize()[0] - 1))
    y_index = int(np.clip(round(y_float), 0, ct_image.GetSize()[1] - 1))
    z_index = int(np.clip(round(z_float), 0, ct_image.GetSize()[2] - 1))
    slices = {
        "axial": (ct[z_index], z_index),
        "coronal": (ct[:, y_index, :], y_index),
        "sagittal": (ct[:, :, x_index], x_index),
    }
    result: dict[str, object] = {}
    for name, (array, index) in slices.items():
        path = assets_dir / f"ct_{name}.png"
        Image.fromarray(window_ct(array, center=300.0, width=1800.0)).save(path)
        result[name] = {"index": index, "path": f"assets/{path.name}", "image_size": list(array.shape[::-1])}
    result["index_xyz"] = [x_index, y_index, z_index]
    return result


def export_ct_volume(ct_image: sitk.Image, assets_dir: Path, stride: int = 2) -> dict[str, object]:
    array = np.asarray(sitk.GetArrayViewFromImage(ct_image))
    sampled = array[::stride, ::stride, ::stride]
    clipped = np.clip(sampled.astype(np.float32), -1024.0, 3071.0)
    encoded = np.rint((clipped + 1024.0) / 4095.0 * 65535.0).astype("<u2")
    output_path = assets_dir / "ct_volume_u16.raw"
    encoded.tofile(output_path)
    original_spacing = np.asarray(ct_image.GetSpacing(), dtype=np.float64)
    return {
        "path": f"assets/{output_path.name}",
        "encoding": "little-endian uint16; normalized HU range -1024..3071",
        "hu_min": -1024.0,
        "hu_max": 3071.0,
        "dimensions_xyz": [int(encoded.shape[2]), int(encoded.shape[1]), int(encoded.shape[0])],
        "origin_lps_mm": list(map(float, ct_image.GetOrigin())),
        "spacing_xyz_mm": (original_spacing * stride).tolist(),
        "direction_lps": np.asarray(ct_image.GetDirection(), dtype=np.float64).reshape(3, 3).tolist(),
        "stride_zyx": [stride, stride, stride],
        "byte_length": int(encoded.nbytes),
    }


def rgba_hex(rgb: np.ndarray) -> str:
    values = np.clip(np.rint(np.asarray(rgb)[:3] * 255.0), 0, 255).astype(int)
    return "#" + "".join(f"{value:02x}" for value in values)


def export_vertebrae_glb(
    masks_dir: Path,
    output_path: Path,
    step_size: int = 2,
) -> list[dict[str, object]]:
    mask_paths = sorted(masks_dir.glob("vertebrae_*.nii.gz"))
    nonempty: list[tuple[Path, sitk.Image, np.ndarray]] = []
    for path in mask_paths:
        image = sitk.ReadImage(str(path))
        array = np.asarray(sitk.GetArrayViewFromImage(image))
        if np.any(array > 0):
            nonempty.append((path, image, array))
    colors = plt.get_cmap("turbo")(np.linspace(0.08, 0.92, len(nonempty)))
    scene = trimesh.Scene()
    records: list[dict[str, object]] = []
    for index, (path, image, array) in enumerate(nonempty):
        vertices_zyx, faces, _, _ = marching_cubes(
            array.astype(np.uint8), level=0.5, step_size=step_size, allow_degenerate=False
        )
        indices_xyz = vertices_zyx[:, ::-1]
        spacing = np.asarray(image.GetSpacing(), dtype=np.float64)
        origin = np.asarray(image.GetOrigin(), dtype=np.float64)
        direction = np.asarray(image.GetDirection(), dtype=np.float64).reshape(3, 3)
        vertices_lps = origin[None, :] + (indices_xyz * spacing[None, :]) @ direction.T
        color = np.rint(colors[index, :3] * 255.0).astype(np.uint8)
        mesh = trimesh.Trimesh(vertices=vertices_lps, faces=faces, process=False)
        mesh.visual.face_colors = np.tile(np.append(color, 210), (len(faces), 1))
        label = path.name.replace(".nii.gz", "")
        scene.add_geometry(mesh, node_name=label, geom_name=label)
        records.append(
            {
                "label": label,
                "color": rgba_hex(colors[index]),
                "vertex_count": int(len(vertices_lps)),
                "face_count": int(len(faces)),
                "centroid_lps_mm": np.mean(vertices_lps, axis=0).tolist(),
                "bbox_min_lps_mm": np.min(vertices_lps, axis=0).tolist(),
                "bbox_max_lps_mm": np.max(vertices_lps, axis=0).tolist(),
            }
        )
    output_path.write_bytes(trimesh.exchange.gltf.export_glb(scene))
    return records


def detector_plane_geometry(
    calibration: dict[str, object],
    xa_dataset: object,
) -> dict[str, object]:
    u_axis = np.asarray(calibration["u_lps"], dtype=np.float64)
    v_axis = np.asarray(calibration["v_lps"], dtype=np.float64)
    n_axis = np.asarray(calibration["n_lps"], dtype=np.float64)
    center = np.asarray(calibration["detector_center_lps_mm"], dtype=np.float64)
    principal = np.asarray(calibration["principal_uv_px"], dtype=np.float64)
    row_spacing, col_spacing = map(float, xa_dataset.ImagerPixelSpacing)
    rows = int(xa_dataset.Rows)
    columns = int(xa_dataset.Columns)
    pixel_edges = np.asarray(
        [
            [-0.5, -0.5],
            [columns - 0.5, -0.5],
            [columns - 0.5, rows - 0.5],
            [-0.5, rows - 0.5],
        ],
        dtype=np.float64,
    )
    corners = np.asarray(
        [
            center
            + (uv[0] - principal[0]) * col_spacing * u_axis
            + (uv[1] - principal[1]) * row_spacing * v_axis
            for uv in pixel_edges
        ]
    )
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, 0] = u_axis
    matrix[:3, 1] = v_axis
    matrix[:3, 2] = n_axis
    matrix[:3, 3] = center
    return {
        "rows": rows,
        "columns": columns,
        "pixel_spacing_row_col_mm": [row_spacing, col_spacing],
        "physical_width_mm": columns * col_spacing,
        "physical_height_mm": rows * row_spacing,
        "pixel_edge_corners_uv": pixel_edges.tolist(),
        "corners_lps_mm": corners.tolist(),
        "detector_local_to_patient_lps_row_major": matrix.tolist(),
    }


def save_reprojection_figure(
    xa_image: np.ndarray,
    labels: list[str],
    observed: np.ndarray,
    predicted: np.ndarray,
    errors: np.ndarray,
    status: str,
    output_path: Path,
) -> None:
    fig, axis = plt.subplots(figsize=(7, 7), dpi=160)
    axis.imshow(xa_image, cmap="gray", vmin=0, vmax=255)
    axis.scatter(observed[:, 0], observed[:, 1], s=55, facecolors="none", edgecolors="lime", linewidths=1.8, label="Observed")
    axis.scatter(predicted[:, 0], predicted[:, 1], s=38, marker="x", color="magenta", linewidths=1.8, label="Reprojected")
    for label, obs, pred, error in zip(labels, observed, predicted, errors):
        axis.plot([obs[0], pred[0]], [obs[1], pred[1]], color="yellow", linewidth=1.0)
        axis.text(obs[0] + 5, obs[1] - 5, f"{label} {error:.2f}px", color="yellow", fontsize=8)
    axis.set_title(f"Physical XA calibration reprojection — {status}")
    axis.set_xlim(0, xa_image.shape[1])
    axis.set_ylim(xa_image.shape[0], 0)
    axis.set_axis_off()
    axis.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def save_thumbnail(
    assets_dir: Path,
    output_path: Path,
    status: str,
    rms_error: float,
) -> None:
    names = ["ct_axial.png", "ct_coronal.png", "ct_sagittal.png", "calibration_reprojection.png"]
    images = [Image.open(assets_dir / name).convert("RGB") for name in names]
    tile_width, tile_height = 480, 360
    canvas = Image.new("RGB", (tile_width * 2, tile_height * 2 + 48), (12, 18, 28))
    for index, image in enumerate(images):
        image.thumbnail((tile_width, tile_height), Image.Resampling.LANCZOS)
        x = (index % 2) * tile_width + (tile_width - image.width) // 2
        y = (index // 2) * tile_height + (tile_height - image.height) // 2 + 48
        canvas.paste(image, (x, y))
    draw = ImageDraw.Draw(canvas)
    draw.text((18, 15), f"CT–DR 3D viewer | {status} | reprojection RMS {rms_error:.2f}px", fill=(230, 238, 248))
    canvas.save(output_path)


def copy_web_sources(web_source: Path, output_dir: Path) -> None:
    for name in (
        "index.html",
        "app.js",
        "styles.css",
        "rendering_mode.mjs",
        "rendering_scene.mjs",
        "thorax_variants.mjs",
    ):
        source = web_source / name
        if not source.exists():
            raise FileNotFoundError(f"Missing viewer source file: {source}")
        shutil.copy2(source, output_dir / name)


def copy_three_vendor(three_source: Path, output_dir: Path) -> None:
    mappings = {
        three_source / "build" / "three.module.min.js": output_dir / "vendor" / "three.module.min.js",
        three_source / "build" / "three.core.min.js": output_dir / "vendor" / "three.core.min.js",
        three_source / "examples" / "jsm" / "controls" / "OrbitControls.js": output_dir / "vendor" / "addons" / "controls" / "OrbitControls.js",
        three_source / "examples" / "jsm" / "loaders" / "GLTFLoader.js": output_dir / "vendor" / "addons" / "loaders" / "GLTFLoader.js",
        three_source / "examples" / "jsm" / "utils" / "BufferGeometryUtils.js": output_dir / "vendor" / "addons" / "utils" / "BufferGeometryUtils.js",
        three_source / "examples" / "jsm" / "utils" / "SkeletonUtils.js": output_dir / "vendor" / "addons" / "utils" / "SkeletonUtils.js",
        three_source / "examples" / "jsm" / "postprocessing" / "EffectComposer.js": output_dir / "vendor" / "addons" / "postprocessing" / "EffectComposer.js",
        three_source / "examples" / "jsm" / "postprocessing" / "RenderPass.js": output_dir / "vendor" / "addons" / "postprocessing" / "RenderPass.js",
        three_source / "examples" / "jsm" / "postprocessing" / "GTAOPass.js": output_dir / "vendor" / "addons" / "postprocessing" / "GTAOPass.js",
        three_source / "examples" / "jsm" / "postprocessing" / "OutputPass.js": output_dir / "vendor" / "addons" / "postprocessing" / "OutputPass.js",
        three_source / "examples" / "jsm" / "postprocessing" / "Pass.js": output_dir / "vendor" / "addons" / "postprocessing" / "Pass.js",
        three_source / "examples" / "jsm" / "postprocessing" / "ShaderPass.js": output_dir / "vendor" / "addons" / "postprocessing" / "ShaderPass.js",
        three_source / "examples" / "jsm" / "postprocessing" / "MaskPass.js": output_dir / "vendor" / "addons" / "postprocessing" / "MaskPass.js",
        three_source / "examples" / "jsm" / "shaders" / "CopyShader.js": output_dir / "vendor" / "addons" / "shaders" / "CopyShader.js",
        three_source / "examples" / "jsm" / "shaders" / "OutputShader.js": output_dir / "vendor" / "addons" / "shaders" / "OutputShader.js",
        three_source / "examples" / "jsm" / "shaders" / "GTAOShader.js": output_dir / "vendor" / "addons" / "shaders" / "GTAOShader.js",
        three_source / "examples" / "jsm" / "shaders" / "PoissonDenoiseShader.js": output_dir / "vendor" / "addons" / "shaders" / "PoissonDenoiseShader.js",
        three_source / "examples" / "jsm" / "math" / "SimplexNoise.js": output_dir / "vendor" / "addons" / "math" / "SimplexNoise.js",
        three_source / "LICENSE": output_dir / "vendor" / "THREE_LICENSE.txt",
    }
    for source, target in mappings.items():
        if not source.exists():
            raise FileNotFoundError(
                f"Missing pinned Three.js file: {source}. Download three@0.185.1 into {three_source}."
            )
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)


def candidate_from_result(result: dict[str, object]) -> OrientationCandidate:
    row = result["candidate"]
    return OrientationCandidate(
        primary_sign=int(row["primary_sign"]),
        secondary_sign=int(row["secondary_sign"]),
        secondary_mode=str(row["secondary_mode"]),
        inplane_rotation_deg=int(row["inplane_rotation_deg"]),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-id", required=True, help="Case key from the caller's approved dataset")
    parser.add_argument("--ct-dir", type=Path, required=True)
    parser.add_argument("--ct-nifti", type=Path, required=True)
    parser.add_argument("--masks-dir", type=Path, required=True)
    parser.add_argument("--depths", type=Path, required=True)
    parser.add_argument("--registration-summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--web-source", type=Path, required=True)
    parser.add_argument("--three-source", type=Path, required=True)
    parser.add_argument("--xa-frame-index", type=int, default=0)
    parser.add_argument("--mesh-step-size", type=int, default=2)
    parser.add_argument("--depth-samples", type=int, default=48)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    global CASE_ID
    CASE_ID = args.case_id
    output_dir = args.output_dir.resolve()
    assets_dir = output_dir / "assets"
    assets_dir.mkdir(parents=True, exist_ok=True)

    depth_document = json.loads(args.depths.resolve().read_text(encoding="utf-8"))
    depth_rows = depth_document["landmarks"]
    labels = [str(row["landmark_id"]) for row in depth_rows]
    points_lps = np.asarray([row["estimated_lps_mm"] for row in depth_rows], dtype=np.float64)

    ct_image = sitk.ReadImage(str(args.ct_nifti.resolve()))
    ct_validation = validate_ct_geometry(ct_image, args.ct_dir.resolve())
    if ct_validation["frame_of_reference_uid"] != depth_document["localizer"]["frame_of_reference_uid"]:
        raise RuntimeError("CT and physical localizer FrameOfReferenceUID differ")
    ct_corners = image_edges_lps(ct_image)
    ct_min = np.min(ct_corners, axis=0)
    ct_max = np.max(ct_corners, axis=0)

    xa_dataset, observed_uv, summary_labels, xa_linkage = load_xa_and_correspondences(
        args.registration_summary.resolve()
    )
    if summary_labels != labels:
        raise RuntimeError(f"Landmark order differs between depth and registration files: {labels} vs {summary_labels}")

    calibration, candidates = fit_calibration(
        points_lps,
        observed_uv,
        labels,
        xa_dataset,
        ct_min,
        ct_max,
    )
    selected_candidate = candidate_from_result(calibration)
    sensitivity = sample_depth_sensitivity(
        depth_rows,
        observed_uv,
        labels,
        xa_dataset,
        ct_min,
        ct_max,
        selected_candidate,
        samples=args.depth_samples,
    )
    calibration["depth_sensitivity"] = sensitivity
    calibration["sdd_mm"] = float(xa_dataset.DistanceSourceToDetector)
    calibration["sod_mm"] = float(xa_dataset.DistanceSourceToPatient)
    calibration["dicom_primary_angle_deg"] = float(xa_dataset.PositionerPrimaryAngle)
    calibration["dicom_secondary_angle_deg"] = float(xa_dataset.PositionerSecondaryAngle)
    calibration["pixel_spacing_row_col_mm"] = list(map(float, xa_dataset.ImagerPixelSpacing))

    xa_image = normalize_xa_frame(xa_dataset, args.xa_frame_index)
    Image.fromarray(xa_image).save(assets_dir / "target_xa_frame0.png")
    xa_image.astype(np.uint8, copy=False).tofile(assets_dir / "target_xa_frame0_u8.raw")
    save_reprojection_figure(
        xa_image,
        labels,
        observed_uv,
        np.asarray(calibration["predicted_uv_px"]),
        np.asarray(calibration["errors_px"]),
        str(calibration["quality_status"]),
        assets_dir / "calibration_reprojection.png",
    )

    initial_slices = save_initial_slice_pngs(ct_image, points_lps, assets_dir)
    ct_volume = export_ct_volume(ct_image, assets_dir, stride=2)
    vertebrae = export_vertebrae_glb(
        args.masks_dir.resolve(), assets_dir / "vertebrae.glb", step_size=args.mesh_step_size
    )
    detector = detector_plane_geometry(calibration, xa_dataset)

    source = np.asarray(calibration["source_lps_mm"], dtype=np.float64)
    detector_center = np.asarray(calibration["detector_center_lps_mm"], dtype=np.float64)
    u_axis = np.asarray(calibration["u_lps"], dtype=np.float64)
    v_axis = np.asarray(calibration["v_lps"], dtype=np.float64)
    principal = np.asarray(calibration["principal_uv_px"], dtype=np.float64)
    row_spacing, col_spacing = map(float, xa_dataset.ImagerPixelSpacing)
    predicted_intersections = np.asarray(calibration["detector_intersections_lps_mm"])
    observed_intersections = np.asarray(
        [
            detector_center
            + (uv[0] - principal[0]) * col_spacing * u_axis
            + (uv[1] - principal[1]) * row_spacing * v_axis
            for uv in observed_uv
        ]
    )

    scene_landmarks: list[dict[str, object]] = []
    for index, (label, point, depth_row) in enumerate(zip(labels, points_lps, depth_rows)):
        scene_landmarks.append(
            {
                "id": label,
                "vertebra": depth_row["selected_vertebra"],
                "quality": depth_row["quality"],
                "point_lps_mm": point.tolist(),
                "reference_plane_lps_mm": depth_row["reference_plane_lps_mm"],
                "depth_p10_p90_mm": depth_row["depth_interval_p10_p90_mm"],
                "observed_uv_px": observed_uv[index].tolist(),
                "reprojected_uv_px": calibration["predicted_uv_px"][index],
                "reprojection_error_px": calibration["errors_px"][index],
                "predicted_detector_intersection_lps_mm": predicted_intersections[index].tolist(),
                "observed_detector_point_lps_mm": observed_intersections[index].tolist(),
            }
        )

    calibration_report = {
        "metadata": {
            "case_id": CASE_ID,
            "status": calibration["quality_status"],
            "research_only": True,
            "model": "constrained pinhole C-arm model with fixed DICOM SDD/SOD/pixel spacing",
        },
        "selected": calibration,
        "candidates": candidates,
        "ct_validation": ct_validation,
        "xa_linkage": xa_linkage,
    }
    (output_dir / "calibration_report.json").write_text(
        json.dumps(calibration_report, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    scene_document = {
        "metadata": {
            "case_id": CASE_ID,
            "coordinate_system": "DICOM patient LPS",
            "unit": "mm",
            "research_only": True,
            "clinical_status": "not clinically validated",
            "quality_status": calibration["quality_status"],
            "quality_reasons": calibration["quality_reasons"],
            "three_version": "0.185.1",
        },
        "ct": {
            "nifti_path": str(args.ct_nifti.resolve()),
            "size_xyz": list(map(int, ct_image.GetSize())),
            "spacing_xyz_mm": list(map(float, ct_image.GetSpacing())),
            "origin_lps_mm": list(map(float, ct_image.GetOrigin())),
            "direction_lps": np.asarray(ct_image.GetDirection()).reshape(3, 3).tolist(),
            "bounds_min_lps_mm": ct_min.tolist(),
            "bounds_max_lps_mm": ct_max.tolist(),
            "corners_lps_mm": ct_corners.tolist(),
            "validation": ct_validation,
            "volume": ct_volume,
            "initial_slices": initial_slices,
            "window_presets": {
                "bone": {"center": 300, "width": 1800},
                "lung": {"center": -600, "width": 1500},
                "mediastinal": {"center": 40, "width": 400},
            },
        },
        "vertebrae": {"glb_path": "assets/vertebrae.glb", "labels": vertebrae},
        "xa": {
            "dicom_path": str(Path(xa_linkage["path"]).resolve()),
            "series_number": int(getattr(xa_dataset, "SeriesNumber", -1)),
            "frame_index": args.xa_frame_index,
            "texture_path": "assets/target_xa_frame0.png",
            "texture_raw_path": "assets/target_xa_frame0_u8.raw",
            "texture_raw_encoding": "row-major uint8; first byte is native top-left pixel",
            "photometric_interpretation": str(xa_dataset.PhotometricInterpretation),
            "dicom_primary_angle_deg": float(xa_dataset.PositionerPrimaryAngle),
            "dicom_secondary_angle_deg": float(xa_dataset.PositionerSecondaryAngle),
            "sdd_mm": float(xa_dataset.DistanceSourceToDetector),
            "sod_mm": float(xa_dataset.DistanceSourceToPatient),
            "source_lps_mm": source.tolist(),
            "isocenter_lps_mm": calibration["isocenter_lps_mm"],
            "detector_center_lps_mm": detector_center.tolist(),
            "u_lps": calibration["u_lps"],
            "v_lps": calibration["v_lps"],
            "n_lps": calibration["n_lps"],
            "effective_primary_deg": calibration["effective_primary_deg"],
            "effective_secondary_deg": calibration["effective_secondary_deg"],
            "principal_uv_px": calibration["principal_uv_px"],
            "detector": detector,
            "calibration": {
                "candidate_name": calibration["candidate_name"],
                "rms_error_px": calibration["rms_error_px"],
                "max_error_px": calibration["max_error_px"],
                "quality_status": calibration["quality_status"],
                "quality_reasons": calibration["quality_reasons"],
                "depth_sensitivity": sensitivity,
            },
        },
        "landmarks": scene_landmarks,
    }
    (output_dir / "scene_geometry.json").write_text(
        json.dumps(scene_document, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    copy_web_sources(args.web_source.resolve(), output_dir)
    copy_three_vendor(args.three_source.resolve(), output_dir)
    save_thumbnail(
        assets_dir,
        assets_dir / "thumbnail.png",
        str(calibration["quality_status"]),
        float(calibration["rms_error_px"]),
    )

    print(f"quality_status={calibration['quality_status']}")
    print(f"candidate={calibration['candidate_name']}")
    print(f"reprojection_rms_px={calibration['rms_error_px']:.4f}")
    print(f"reprojection_max_px={calibration['max_error_px']:.4f}")
    print(f"isocenter_lps_mm={np.round(calibration['isocenter_lps_mm'], 3).tolist()}")
    print(f"source_lps_mm={np.round(calibration['source_lps_mm'], 3).tolist()}")
    print(f"detector_center_lps_mm={np.round(calibration['detector_center_lps_mm'], 3).tolist()}")
    print(f"vertebrae_glb_mb={(assets_dir / 'vertebrae.glb').stat().st_size / 1024**2:.3f}")
    print(f"ct_volume_mb={(assets_dir / 'ct_volume_u16.raw').stat().st_size / 1024**2:.3f}")
    print(f"viewer={output_dir / 'index.html'}")


if __name__ == "__main__":
    main()
