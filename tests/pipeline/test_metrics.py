from __future__ import annotations

import numpy as np

from ctdr_pipeline.evaluation.metrics import MetricInputs, calculate_case_metrics


def test_calculate_case_metrics_for_identical_images_and_known_points() -> None:
    image = np.zeros((32, 32), dtype=np.float32)
    image[8:24, 12:20] = 1.0
    points = np.array([[4.0, 5.0], [20.0, 6.0], [18.0, 25.0]])
    transform = np.array([[1.0, 0.0, 3.0], [0.0, 1.0, -2.0], [0.0, 0.0, 1.0]])
    target_points = points + np.array([3.0, -2.0])
    metrics = calculate_case_metrics(
        MetricInputs(
            source_points_2d=points,
            target_points_2d=target_points,
            affine_2d=transform,
            image_diagonal=float(np.hypot(32, 32)),
            drr=image,
            target=image,
            roi=np.ones_like(image),
            drr_fov_mask=image > 0,
            target_fov_mask=image > 0,
            candidate_poses=np.array(
                [[0, 0, 0, 0, 700, 0], [2, 0, 0, 4, 700, 3]], dtype=float
            ),
            candidate_scores=np.array([0.8, 0.7]),
            objective_initial=0.4,
            objective_final=0.8,
            pptx_inlier_ratio=1.0,
            affine_condition_number=1.0,
            dicom_confidence=0.9,
        )
    )
    assert metrics.point_rms_px < 1e-8
    assert metrics.pptx_rms_diagonal < 1e-8
    assert metrics.drr_roi_ncc > 0.999
    assert metrics.drr_gradient_ncc > 0.999
    assert metrics.fov_iou == 1.0
    assert abs(metrics.dicom_candidate_margin - 0.1) < 1e-8
    assert metrics.tre_mm is None
    assert metrics.tre_reason == "independent_3d_ground_truth_unavailable"


def test_independent_3d_landmarks_enable_tre() -> None:
    truth = np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]])
    predicted = truth + np.array([3.0, 4.0, 0.0])
    metrics = calculate_case_metrics(
        MetricInputs(
            independent_3d_truth_mm=truth,
            predicted_3d_points_mm=predicted,
        )
    )
    assert metrics.tre_mm == 5.0
    assert metrics.tre_reason is None


def test_missing_truth_never_substitutes_pose_self_consistency() -> None:
    metrics = calculate_case_metrics(
        MetricInputs(
            candidate_poses=np.array(
                [[0, 0, 0, 0, 700, 0], [20, 0, 0, 100, 700, 0]], dtype=float
            )
        )
    )
    assert metrics.tre_mm is None
    assert metrics.tre_reason == "independent_3d_ground_truth_unavailable"
