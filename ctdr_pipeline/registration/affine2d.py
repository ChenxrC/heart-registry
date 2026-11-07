from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np

from ctdr_pipeline.pptx.annotations import AnnotationSet


@dataclass(frozen=True)
class LandmarkPair:
    landmark_id: str
    source_xy: tuple[float, float]
    target_xy: tuple[float, float]


@dataclass(frozen=True)
class Registration2DMetrics:
    rms_px: float
    mean_px: float
    median_px: float
    max_px: float
    p95_px: float
    normalized_rms: float
    normalized_p95: float
    inlier_ratio: float
    determinant: float
    condition_number: float
    rotation_deg: float
    scale_x: float
    scale_y: float
    shear: float
    translation_x: float
    translation_y: float
    order_consistent: bool


@dataclass(frozen=True)
class Registration2DResult:
    matrix: np.ndarray
    inverse_matrix: np.ndarray
    inliers: list[bool]
    residuals_px: list[float]
    metrics: Registration2DMetrics
    accepted: bool
    reasons: list[str]
    landmarks: list[LandmarkPair]


def transform_points(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    values = np.asarray(points, dtype=np.float64)
    transform = np.asarray(matrix, dtype=np.float64)
    homogeneous = np.column_stack([values, np.ones(len(values), dtype=np.float64)])
    mapped = homogeneous @ transform.T
    return mapped[:, :2] / mapped[:, 2:3]


def _least_squares_affine(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    design = np.column_stack([source, np.ones(len(source), dtype=np.float64)])
    coefficients, _, rank, _ = np.linalg.lstsq(design, target, rcond=None)
    if rank < 3:
        raise ValueError("landmarks are collinear; affine transform is undefined")
    matrix = np.eye(3, dtype=np.float64)
    matrix[:2, :] = coefficients.T
    return matrix


def _order_consistent(
    source: np.ndarray, target: np.ndarray, matrix: np.ndarray
) -> bool:
    if len(source) < 2:
        return True
    source_centered = source - source.mean(axis=0, keepdims=True)
    _, _, vectors = np.linalg.svd(source_centered, full_matrices=False)
    source_axis = vectors[0]
    if source_axis[1] < 0 or (abs(source_axis[1]) < 1e-8 and source_axis[0] < 0):
        source_axis = -source_axis
    target_axis = np.linalg.inv(matrix[:2, :2]).T @ source_axis
    target_axis /= max(float(np.linalg.norm(target_axis)), 1e-12)
    target_centered = target - target.mean(axis=0, keepdims=True)
    source_order = np.argsort(source_centered @ source_axis)
    target_order = np.argsort(target_centered @ target_axis)
    return bool(np.array_equal(source_order, target_order))


def _decompose(matrix: np.ndarray) -> tuple[float, float, float, float]:
    linear = matrix[:2, :2]
    scale_x = float(np.linalg.norm(linear[:, 0]))
    unit_x = linear[:, 0] / max(scale_x, 1e-12)
    shear_projection = float(np.dot(unit_x, linear[:, 1]))
    orthogonal_y = linear[:, 1] - shear_projection * unit_x
    scale_y = float(np.linalg.norm(orthogonal_y))
    shear = shear_projection / max(scale_y, 1e-12)
    rotation_deg = math.degrees(math.atan2(unit_x[1], unit_x[0]))
    return rotation_deg, scale_x, scale_y, shear


def estimate_affine_ransac(
    source: np.ndarray,
    target: np.ndarray,
    *,
    image_diagonal: float,
    ransac_threshold_px: float = 3.0,
    minimum_inlier_ratio: float = 0.75,
    maximum_normalized_rms: float = 0.02,
    maximum_normalized_p95: float = 0.03,
    maximum_condition_number: float = 100.0,
    landmark_ids: list[str] | None = None,
) -> Registration2DResult:
    source_values = np.asarray(source, dtype=np.float64)
    target_values = np.asarray(target, dtype=np.float64)
    if source_values.shape != target_values.shape or source_values.ndim != 2 or source_values.shape[1] != 2:
        raise ValueError("source and target landmarks must have matching Nx2 shapes")
    if len(source_values) < 3:
        raise ValueError("at least three landmarks are required")
    if image_diagonal <= 0:
        raise ValueError("image_diagonal must be positive")

    affine, inlier_mask = cv2.estimateAffine2D(
        source_values,
        target_values,
        method=cv2.RANSAC,
        ransacReprojThreshold=ransac_threshold_px,
        maxIters=5000,
        confidence=0.999,
        refineIters=10,
    )
    if affine is None or inlier_mask is None:
        raise ValueError("RANSAC could not estimate an affine transform")
    inliers_array = inlier_mask.reshape(-1).astype(bool)
    if int(inliers_array.sum()) < 3:
        raise ValueError("RANSAC found fewer than three affine inliers")
    matrix = _least_squares_affine(
        source_values[inliers_array], target_values[inliers_array]
    )
    predicted = transform_points(source_values, matrix)
    residuals = np.linalg.norm(predicted - target_values, axis=1)
    gated_residuals = residuals[inliers_array]
    rms = float(np.sqrt(np.mean(np.square(gated_residuals))))
    p95 = float(np.percentile(gated_residuals, 95))
    determinant = float(np.linalg.det(matrix[:2, :2]))
    condition_number = float(np.linalg.cond(matrix[:2, :2]))
    rotation, scale_x, scale_y, shear = _decompose(matrix)
    order_consistent = _order_consistent(
        source_values[inliers_array], target_values[inliers_array], matrix
    )
    metrics = Registration2DMetrics(
        rms_px=rms,
        mean_px=float(np.mean(gated_residuals)),
        median_px=float(np.median(gated_residuals)),
        max_px=float(np.max(gated_residuals)),
        p95_px=p95,
        normalized_rms=rms / image_diagonal,
        normalized_p95=p95 / image_diagonal,
        inlier_ratio=float(inliers_array.mean()),
        determinant=determinant,
        condition_number=condition_number,
        rotation_deg=rotation,
        scale_x=scale_x,
        scale_y=scale_y,
        shear=shear,
        translation_x=float(matrix[0, 2]),
        translation_y=float(matrix[1, 2]),
        order_consistent=order_consistent,
    )
    reasons: list[str] = []
    if metrics.inlier_ratio < minimum_inlier_ratio:
        reasons.append("insufficient_inlier_ratio")
    if metrics.normalized_rms > maximum_normalized_rms:
        reasons.append("rms_above_threshold")
    if metrics.normalized_p95 > maximum_normalized_p95:
        reasons.append("p95_above_threshold")
    if determinant <= 0:
        reasons.append("mirror_or_flip")
    if condition_number > maximum_condition_number:
        reasons.append("ill_conditioned_affine")
    if not order_consistent:
        reasons.append("landmark_order_inconsistent")
    ids = landmark_ids or [f"P{index + 1}" for index in range(len(source_values))]
    landmarks = [
        LandmarkPair(
            landmark_id=ids[index],
            source_xy=(float(source_values[index, 0]), float(source_values[index, 1])),
            target_xy=(float(target_values[index, 0]), float(target_values[index, 1])),
        )
        for index in range(len(source_values))
    ]
    return Registration2DResult(
        matrix=matrix,
        inverse_matrix=np.linalg.inv(matrix),
        inliers=inliers_array.tolist(),
        residuals_px=residuals.tolist(),
        metrics=metrics,
        accepted=not reasons,
        reasons=reasons,
        landmarks=landmarks,
    )


def register_landmarks(
    source: AnnotationSet,
    target: AnnotationSet,
    *,
    image_diagonal: float,
) -> Registration2DResult:
    if len(source.landmarks) != len(target.landmarks):
        raise ValueError("ordered landmark counts differ; manual review is required")
    return estimate_affine_ransac(
        source.points,
        target.points,
        image_diagonal=image_diagonal,
    )
