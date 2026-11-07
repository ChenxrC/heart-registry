from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import Field, ValidationError

from ctdr_pipeline.schemas import StrictModel


class QualityThresholds(StrictModel):
    pptx_rms_diagonal: float = 0.02
    pptx_p95_diagonal: float = 0.03
    pptx_inlier_ratio: float = 0.75
    affine_condition_number: float = 100.0
    dicom_confidence: float = 0.60
    dicom_candidate_margin: float = 0.05
    drr_roi_ncc: float = 0.35
    drr_gradient_ncc: float = 0.35
    drr_objective_improvement: float = 0.03
    pose_translation_spread_mm: float = 15.0
    pose_rotation_spread_deg: float = 7.0
    fov_iou: float = 0.50


class RegistrationSettings(StrictModel):
    random_seed: int = 20260809
    resolutions: list[int] = Field(default_factory=lambda: [256, 512])
    coarse_iterations: int = 30
    refine_iterations: int = 80
    multistart_count: int = 12
    refine_top_k: int = 3
    roi_method: str = "baseline"
    rotation_learning_rate_deg: float = 1.5
    translation_learning_rate_mm: float = 3.0
    rotation_bounds_deg: list[list[float]] = Field(
        default_factory=lambda: [[-180.0, 180.0]] * 3
    )
    translation_bounds_mm: list[list[float]] = Field(
        default_factory=lambda: [[-500.0, 500.0], [300.0, 1500.0], [-500.0, 500.0]]
    )


class PipelineConfig(StrictModel):
    data_root: Path
    output_root: Path = Path("runs")
    gpus: list[str] = Field(default_factory=lambda: ["cuda:0"])
    gpu_workers: dict[str, int] = Field(default_factory=lambda: {"cuda:0": 1})
    cpu_workers: int = 4
    quality: QualityThresholds = Field(default_factory=QualityThresholds)
    registration: RegistrationSettings = Field(default_factory=RegistrationSettings)
    dicom_overrides: Path | None = None
    anonymize_reports: bool = True
    fail_on_review: bool = False


def load_config(path: Path) -> PipelineConfig:
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        if not isinstance(raw, dict):
            raise ValueError("configuration root must be a mapping")
        return PipelineConfig.model_validate(raw)
    except ValidationError as exc:
        raise ValueError(str(exc)) from exc
