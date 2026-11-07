from __future__ import annotations

from dataclasses import replace

import pytest

from ctdr_pipeline.config import QualityThresholds
from ctdr_pipeline.evaluation.metrics import CaseMetrics
from ctdr_pipeline.evaluation.quality import decide_quality
from ctdr_pipeline.schemas import CaseStatus


def passing_metrics() -> CaseMetrics:
    return CaseMetrics(
        pptx_rms_diagonal=0.02,
        pptx_p95_diagonal=0.03,
        pptx_inlier_ratio=0.75,
        affine_condition_number=100.0,
        dicom_confidence=0.60,
        dicom_candidate_margin=0.05,
        drr_roi_ncc=0.35,
        drr_gradient_ncc=0.35,
        drr_mutual_information=0.2,
        contour_chamfer_px=2.0,
        drr_objective_improvement=0.03,
        pose_translation_spread_mm=15.0,
        pose_rotation_spread_deg=7.0,
        fov_iou=0.50,
        tre_mm=None,
        tre_reason="independent_3d_ground_truth_unavailable",
    )


def test_exact_quality_thresholds_pass() -> None:
    decision = decide_quality(passing_metrics(), QualityThresholds())
    assert decision.status == CaseStatus.ACCEPTED_RESEARCH_CANDIDATE
    assert decision.accepted
    assert not decision.failed_rules


@pytest.mark.parametrize(
    ("field", "value", "expected_rule"),
    [
        ("pptx_rms_diagonal", 0.020001, "pptx_rms_diagonal"),
        ("pptx_p95_diagonal", 0.030001, "pptx_p95_diagonal"),
        ("pptx_inlier_ratio", 0.749999, "pptx_inlier_ratio"),
        ("affine_condition_number", 100.001, "affine_condition_number"),
        ("dicom_confidence", 0.599999, "dicom_confidence"),
        ("dicom_candidate_margin", 0.049999, "dicom_candidate_margin"),
        ("drr_roi_ncc", 0.349999, "drr_roi_ncc"),
        ("drr_gradient_ncc", 0.349999, "drr_gradient_ncc"),
        ("drr_objective_improvement", 0.029999, "drr_objective_improvement"),
        ("pose_translation_spread_mm", 15.001, "pose_translation_spread_mm"),
        ("pose_rotation_spread_deg", 7.001, "pose_rotation_spread_deg"),
        ("fov_iou", 0.499999, "fov_iou"),
    ],
)
def test_any_worse_threshold_is_not_accepted(
    field: str, value: float, expected_rule: str
) -> None:
    decision = decide_quality(
        replace(passing_metrics(), **{field: value}), QualityThresholds()
    )
    assert not decision.accepted
    assert any(expected_rule in rule for rule in decision.failed_rules)


@pytest.mark.parametrize("field", ["mirror_detected", "vertical_flip_detected", "parameter_boundary"])
def test_orientation_or_boundary_evidence_rejects_3d(field: str) -> None:
    decision = decide_quality(
        replace(passing_metrics(), **{field: True}), QualityThresholds()
    )
    assert decision.status == CaseStatus.REJECTED_3D
    assert not decision.accepted


def test_runtime_failure_has_highest_precedence_but_all_rules_are_returned() -> None:
    metrics = replace(
        passing_metrics(),
        runtime_failure="CUDA OOM",
        mirror_detected=True,
        dicom_confidence=0.1,
    )
    decision = decide_quality(metrics, QualityThresholds())
    assert decision.status == CaseStatus.FAILED_RUNTIME
    assert any("runtime_failure" in rule for rule in decision.failed_rules)
    assert any("mirror_detected" in rule for rule in decision.failed_rules)
    assert any("dicom_confidence" in rule for rule in decision.failed_rules)

