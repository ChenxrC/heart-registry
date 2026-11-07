from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage


@dataclass(frozen=True)
class MetricInputs:
    source_points_2d: np.ndarray | None = None
    target_points_2d: np.ndarray | None = None
    affine_2d: np.ndarray | None = None
    image_diagonal: float | None = None
    drr: np.ndarray | None = None
    target: np.ndarray | None = None
    roi: np.ndarray | None = None
    drr_fov_mask: np.ndarray | None = None
    target_fov_mask: np.ndarray | None = None
    candidate_poses: np.ndarray | None = None
    candidate_scores: np.ndarray | None = None
    objective_initial: float | None = None
    objective_final: float | None = None
    pptx_inlier_ratio: float | None = None
    affine_condition_number: float | None = None
    dicom_confidence: float | None = None
    dicom_candidate_margin: float | None = None
    independent_3d_truth_mm: np.ndarray | None = None
    predicted_3d_points_mm: np.ndarray | None = None
    mirror_detected: bool = False
    vertical_flip_detected: bool = False
    parameter_boundary: bool = False
    runtime_failure: str | None = None
    geometry_failure: str | None = None
    upstream_statuses: tuple[str, ...] = ()


@dataclass(frozen=True)
class CaseMetrics:
    point_rms_px: float | None = None
    point_mean_px: float | None = None
    point_median_px: float | None = None
    point_max_px: float | None = None
    point_p95_px: float | None = None
    pptx_rms_diagonal: float | None = None
    pptx_p95_diagonal: float | None = None
    pptx_inlier_ratio: float | None = None
    affine_condition_number: float | None = None
    dicom_confidence: float | None = None
    dicom_candidate_margin: float | None = None
    drr_roi_ncc: float | None = None
    drr_gradient_ncc: float | None = None
    drr_mutual_information: float | None = None
    contour_chamfer_px: float | None = None
    drr_objective_improvement: float | None = None
    pose_translation_spread_mm: float | None = None
    pose_rotation_spread_deg: float | None = None
    fov_iou: float | None = None
    reprojection_rms_px: float | None = None
    tre_mm: float | None = None
    tre_reason: str | None = "independent_3d_ground_truth_unavailable"
    mirror_detected: bool = False
    vertical_flip_detected: bool = False
    parameter_boundary: bool = False
    runtime_failure: str | None = None
    geometry_failure: str | None = None
    upstream_statuses: tuple[str, ...] = ()


def _points(value: np.ndarray | None, dimensions: int, name: str) -> np.ndarray | None:
    if value is None:
        return None
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != dimensions:
        raise ValueError(f"{name} must have shape Nx{dimensions}")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain finite values")
    return array


def _image(value: np.ndarray | None, name: str) -> np.ndarray | None:
    if value is None:
        return None
    array = np.asarray(value, dtype=np.float64).squeeze()
    if array.ndim != 2:
        raise ValueError(f"{name} must reduce to a 2D image")
    return array


def _point_statistics(residuals: np.ndarray) -> dict[str, float]:
    distances = np.linalg.norm(residuals, axis=1)
    return {
        "rms": float(np.sqrt(np.mean(np.square(distances)))),
        "mean": float(np.mean(distances)),
        "median": float(np.median(distances)),
        "max": float(np.max(distances)),
        "p95": float(np.percentile(distances, 95)),
    }


def _weighted_ncc(first: np.ndarray, second: np.ndarray, weight: np.ndarray) -> float:
    valid = np.isfinite(first) & np.isfinite(second) & np.isfinite(weight) & (weight > 0)
    if not np.any(valid):
        return 0.0
    weights = np.clip(weight[valid], 0.0, None)
    denominator = max(float(weights.sum()), 1e-12)
    first_values, second_values = first[valid], second[valid]
    first_mean = float(np.sum(weights * first_values) / denominator)
    second_mean = float(np.sum(weights * second_values) / denominator)
    first_centered = first_values - first_mean
    second_centered = second_values - second_mean
    numerator = float(np.sum(weights * first_centered * second_centered))
    energy = float(
        np.sqrt(
            np.sum(weights * np.square(first_centered))
            * np.sum(weights * np.square(second_centered))
        )
    )
    return numerator / max(energy, 1e-12)


def _gradient(image: np.ndarray) -> np.ndarray:
    grad_x = ndimage.sobel(image, axis=1, mode="nearest")
    grad_y = ndimage.sobel(image, axis=0, mode="nearest")
    return np.hypot(grad_x, grad_y)


def _mutual_information(first: np.ndarray, second: np.ndarray, weight: np.ndarray) -> float:
    valid = np.isfinite(first) & np.isfinite(second) & (weight > 0)
    if int(valid.sum()) < 2:
        return 0.0
    histogram, _, _ = np.histogram2d(first[valid], second[valid], bins=32)
    probability = histogram / max(float(histogram.sum()), 1.0)
    first_probability = probability.sum(axis=1, keepdims=True)
    second_probability = probability.sum(axis=0, keepdims=True)
    expected = first_probability @ second_probability
    occupied = probability > 0
    return float(np.sum(probability[occupied] * np.log(probability[occupied] / expected[occupied])))


def _contour_mask(gradient: np.ndarray, roi: np.ndarray) -> np.ndarray:
    values = gradient[roi > 0]
    if values.size == 0 or float(np.max(values)) <= 0:
        return np.zeros_like(gradient, dtype=bool)
    threshold = float(np.percentile(values, 75))
    return (gradient >= threshold) & (roi > 0)


def _symmetric_chamfer(first: np.ndarray, second: np.ndarray) -> float:
    if not np.any(first) or not np.any(second):
        return float("inf")
    distance_to_second = ndimage.distance_transform_edt(~second)
    distance_to_first = ndimage.distance_transform_edt(~first)
    return float((distance_to_second[first].mean() + distance_to_first[second].mean()) / 2.0)


def _fov_iou(first: np.ndarray, second: np.ndarray) -> float:
    first = np.asarray(first, dtype=bool)
    second = np.asarray(second, dtype=bool)
    if first.shape != second.shape:
        raise ValueError("FOV masks must have identical shapes")
    union = np.logical_or(first, second).sum()
    if union == 0:
        return 1.0
    return float(np.logical_and(first, second).sum() / union)


def _pose_spread(poses: np.ndarray | None) -> tuple[float | None, float | None]:
    if poses is None:
        return None, None
    array = np.asarray(poses, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != 6 or len(array) == 0:
        raise ValueError("candidate_poses must have shape Nx6")
    rotation_center = np.median(array[:, :3], axis=0)
    translation_center = np.median(array[:, 3:], axis=0)
    rotation = float(np.max(np.linalg.norm(array[:, :3] - rotation_center, axis=1)))
    translation = float(np.max(np.linalg.norm(array[:, 3:] - translation_center, axis=1)))
    return translation, rotation


def calculate_case_metrics(inputs: MetricInputs) -> CaseMetrics:
    source = _points(inputs.source_points_2d, 2, "source_points_2d")
    target_points = _points(inputs.target_points_2d, 2, "target_points_2d")
    point_statistics: dict[str, float] = {}
    if source is not None or target_points is not None or inputs.affine_2d is not None:
        if source is None or target_points is None or inputs.affine_2d is None:
            raise ValueError("2D source, target, and affine must be supplied together")
        if source.shape != target_points.shape or len(source) == 0:
            raise ValueError("2D source and target points must have equal non-zero shape")
        affine = np.asarray(inputs.affine_2d, dtype=np.float64)
        if affine.shape != (3, 3):
            raise ValueError("affine_2d must have shape 3x3")
        homogeneous = np.column_stack([source, np.ones(len(source))])
        transformed = (affine @ homogeneous.T).T
        transformed = transformed[:, :2] / transformed[:, 2, None]
        point_statistics = _point_statistics(transformed - target_points)

    drr = _image(inputs.drr, "drr")
    target = _image(inputs.target, "target")
    roi = _image(inputs.roi, "roi")
    roi_ncc = gradient_ncc = mutual_information = chamfer = None
    if drr is not None or target is not None:
        if drr is None or target is None or drr.shape != target.shape:
            raise ValueError("DRR and target must be supplied with identical shapes")
        if roi is None:
            roi = np.ones_like(drr)
        if roi.shape != drr.shape:
            raise ValueError("ROI must have the same shape as DRR")
        roi = np.clip(roi, 0.0, 1.0)
        roi_ncc = _weighted_ncc(drr, target, roi)
        drr_gradient, target_gradient = _gradient(drr), _gradient(target)
        gradient_ncc = _weighted_ncc(drr_gradient, target_gradient, roi)
        mutual_information = _mutual_information(drr, target, roi)
        chamfer = _symmetric_chamfer(
            _contour_mask(drr_gradient, roi), _contour_mask(target_gradient, roi)
        )

    fov_iou = None
    if inputs.drr_fov_mask is not None or inputs.target_fov_mask is not None:
        if inputs.drr_fov_mask is None or inputs.target_fov_mask is None:
            raise ValueError("both FOV masks are required")
        fov_iou = _fov_iou(inputs.drr_fov_mask, inputs.target_fov_mask)

    translation_spread, rotation_spread = _pose_spread(inputs.candidate_poses)
    candidate_margin = inputs.dicom_candidate_margin
    if inputs.candidate_scores is not None:
        scores = np.asarray(inputs.candidate_scores, dtype=np.float64).reshape(-1)
        finite = np.sort(scores[np.isfinite(scores)])[::-1]
        if finite.size >= 2 and candidate_margin is None:
            candidate_margin = float(finite[0] - finite[1])

    tre = None
    tre_reason = "independent_3d_ground_truth_unavailable"
    truth = _points(inputs.independent_3d_truth_mm, 3, "independent_3d_truth_mm")
    predicted = _points(inputs.predicted_3d_points_mm, 3, "predicted_3d_points_mm")
    if truth is not None or predicted is not None:
        if truth is None or predicted is None or truth.shape != predicted.shape or len(truth) == 0:
            raise ValueError("independent 3D truth and predictions require equal non-zero shapes")
        tre = float(np.sqrt(np.mean(np.sum(np.square(predicted - truth), axis=1))))
        tre_reason = None

    diagonal = inputs.image_diagonal
    rms_normalized = None
    p95_normalized = None
    if point_statistics and diagonal is not None:
        if diagonal <= 0:
            raise ValueError("image_diagonal must be positive")
        rms_normalized = point_statistics["rms"] / diagonal
        p95_normalized = point_statistics["p95"] / diagonal

    improvement = None
    if inputs.objective_initial is not None and inputs.objective_final is not None:
        improvement = float(inputs.objective_final - inputs.objective_initial)

    return CaseMetrics(
        point_rms_px=point_statistics.get("rms"),
        point_mean_px=point_statistics.get("mean"),
        point_median_px=point_statistics.get("median"),
        point_max_px=point_statistics.get("max"),
        point_p95_px=point_statistics.get("p95"),
        pptx_rms_diagonal=rms_normalized,
        pptx_p95_diagonal=p95_normalized,
        pptx_inlier_ratio=inputs.pptx_inlier_ratio,
        affine_condition_number=inputs.affine_condition_number,
        dicom_confidence=inputs.dicom_confidence,
        dicom_candidate_margin=candidate_margin,
        drr_roi_ncc=roi_ncc,
        drr_gradient_ncc=gradient_ncc,
        drr_mutual_information=mutual_information,
        contour_chamfer_px=chamfer,
        drr_objective_improvement=improvement,
        pose_translation_spread_mm=translation_spread,
        pose_rotation_spread_deg=rotation_spread,
        fov_iou=fov_iou,
        reprojection_rms_px=point_statistics.get("rms"),
        tre_mm=tre,
        tre_reason=tre_reason,
        mirror_detected=inputs.mirror_detected,
        vertical_flip_detected=inputs.vertical_flip_detected,
        parameter_boundary=inputs.parameter_boundary,
        runtime_failure=inputs.runtime_failure,
        geometry_failure=inputs.geometry_failure,
        upstream_statuses=inputs.upstream_statuses,
    )
