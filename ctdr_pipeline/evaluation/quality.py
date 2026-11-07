from __future__ import annotations

from dataclasses import dataclass

from ctdr_pipeline.config import QualityThresholds
from ctdr_pipeline.evaluation.metrics import CaseMetrics
from ctdr_pipeline.schemas import CaseStatus


@dataclass(frozen=True)
class QualityDecision:
    status: CaseStatus
    accepted: bool
    failed_rules: tuple[str, ...]


def _minimum(failed: list[str], name: str, value: float | None, threshold: float) -> None:
    if value is None:
        failed.append(f"{name}:missing")
    elif value < threshold:
        failed.append(f"{name}:{value:.8g}<{threshold:.8g}")


def _maximum(failed: list[str], name: str, value: float | None, threshold: float) -> None:
    if value is None:
        failed.append(f"{name}:missing")
    elif value > threshold:
        failed.append(f"{name}:{value:.8g}>{threshold:.8g}")


def decide_quality(metrics: CaseMetrics, config: QualityThresholds) -> QualityDecision:
    runtime_rules: list[str] = []
    incomplete_rules: list[str] = []
    two_d_rules: list[str] = []
    missing_rules: list[str] = []
    dicom_rules: list[str] = []
    geometry_rules: list[str] = []
    rejection_rules: list[str] = []
    review_rules: list[str] = []

    if metrics.runtime_failure:
        runtime_rules.append(f"runtime_failure:{metrics.runtime_failure}")
    if metrics.geometry_failure:
        geometry_rules.append(f"geometry_failure:{metrics.geometry_failure}")

    status_values = {status.value: status for status in CaseStatus}
    for value in sorted({str(value) for value in metrics.upstream_statuses}):
        status = status_values.get(value)
        rule = f"upstream_status:{value}"
        if status == CaseStatus.SKIPPED_INCOMPLETE_PPTX_PAIR:
            incomplete_rules.append(rule)
        elif status in {
            CaseStatus.SKIPPED_CASE_DIRECTORY_MISSING,
            CaseStatus.SKIPPED_CT_MISSING,
            CaseStatus.SKIPPED_XA_MISSING,
        }:
            missing_rules.append(rule)
        elif status == CaseStatus.REVIEW_REQUIRED_2D:
            two_d_rules.append(rule)
        elif status == CaseStatus.REVIEW_REQUIRED_DICOM_AMBIGUITY:
            dicom_rules.append(rule)
        elif status == CaseStatus.REJECTED_GEOMETRY:
            geometry_rules.append(rule)
        elif status == CaseStatus.REJECTED_3D:
            rejection_rules.append(rule)
        elif status == CaseStatus.REVIEW_REQUIRED_3D:
            review_rules.append(rule)
        elif status == CaseStatus.FAILED_RUNTIME:
            runtime_rules.append(rule)

    _maximum(two_d_rules, "pptx_rms_diagonal", metrics.pptx_rms_diagonal, config.pptx_rms_diagonal)
    _maximum(two_d_rules, "pptx_p95_diagonal", metrics.pptx_p95_diagonal, config.pptx_p95_diagonal)
    _minimum(two_d_rules, "pptx_inlier_ratio", metrics.pptx_inlier_ratio, config.pptx_inlier_ratio)
    _maximum(
        two_d_rules,
        "affine_condition_number",
        metrics.affine_condition_number,
        config.affine_condition_number,
    )
    _minimum(dicom_rules, "dicom_confidence", metrics.dicom_confidence, config.dicom_confidence)
    _minimum(
        dicom_rules,
        "dicom_candidate_margin",
        metrics.dicom_candidate_margin,
        config.dicom_candidate_margin,
    )

    if metrics.mirror_detected:
        rejection_rules.append("mirror_detected:true")
    if metrics.vertical_flip_detected:
        rejection_rules.append("vertical_flip_detected:true")
    if metrics.parameter_boundary:
        rejection_rules.append("parameter_boundary:true")
    _minimum(rejection_rules, "fov_iou", metrics.fov_iou, config.fov_iou)

    _minimum(review_rules, "drr_roi_ncc", metrics.drr_roi_ncc, config.drr_roi_ncc)
    _minimum(
        review_rules,
        "drr_gradient_ncc",
        metrics.drr_gradient_ncc,
        config.drr_gradient_ncc,
    )
    _minimum(
        review_rules,
        "drr_objective_improvement",
        metrics.drr_objective_improvement,
        config.drr_objective_improvement,
    )
    _maximum(
        review_rules,
        "pose_translation_spread_mm",
        metrics.pose_translation_spread_mm,
        config.pose_translation_spread_mm,
    )
    _maximum(
        review_rules,
        "pose_rotation_spread_deg",
        metrics.pose_rotation_spread_deg,
        config.pose_rotation_spread_deg,
    )

    failed = (
        runtime_rules
        + incomplete_rules
        + two_d_rules
        + missing_rules
        + dicom_rules
        + geometry_rules
        + rejection_rules
        + review_rules
    )
    if runtime_rules:
        status = CaseStatus.FAILED_RUNTIME
    elif incomplete_rules:
        status = CaseStatus.SKIPPED_INCOMPLETE_PPTX_PAIR
    elif two_d_rules:
        status = CaseStatus.REVIEW_REQUIRED_2D
    elif missing_rules:
        missing_status = next(
            (
                status_values.get(rule.split(":", 1)[1])
                for rule in missing_rules
                if rule.startswith("upstream_status:")
            ),
            None,
        )
        status = missing_status or CaseStatus.SKIPPED_CASE_DIRECTORY_MISSING
    elif dicom_rules:
        status = CaseStatus.REVIEW_REQUIRED_DICOM_AMBIGUITY
    elif geometry_rules:
        status = CaseStatus.REJECTED_GEOMETRY
    elif rejection_rules:
        status = CaseStatus.REJECTED_3D
    elif review_rules:
        status = CaseStatus.REVIEW_REQUIRED_3D
    else:
        status = CaseStatus.ACCEPTED_RESEARCH_CANDIDATE
    return QualityDecision(
        status=status,
        accepted=status == CaseStatus.ACCEPTED_RESEARCH_CANDIDATE,
        failed_rules=tuple(failed),
    )
