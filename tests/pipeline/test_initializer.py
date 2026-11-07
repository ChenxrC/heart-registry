from pathlib import Path

import numpy as np
import pytest

from ctdr_pipeline.dicom.geometry import ProjectionGeometry
from ctdr_pipeline.dicom.linker import (
    DicomLinkageResult,
    ImageMatchCandidate,
    ImageMatchDecision,
)
from ctdr_pipeline.registration.affine2d import estimate_affine_ransac
from ctdr_pipeline.registration.initializer import generate_pose_seeds


def linkage() -> DicomLinkageResult:
    candidate = ImageMatchCandidate(
        path=str(Path("image.dcm")),
        series_uid="series",
        frame_index=0,
        orientation="identity",
        good_matches=20,
        inliers=20,
        inlier_ratio=1.0,
        fov_iou=1.0,
        confidence=1.0,
        homography_candidate_to_slide=np.eye(3).reshape(-1).tolist(),
    )
    decision = ImageMatchDecision(
        selected=candidate,
        candidates=[candidate],
        status="accepted",
        confidence=1.0,
        candidate_margin=1.0,
    )
    return DicomLinkageResult(
        case_id="1234567",
        source=decision,
        target=decision,
        status="accepted",
    )


def geometry() -> ProjectionGeometry:
    return ProjectionGeometry(
        rows=512,
        columns=512,
        pixel_spacing_row_mm=0.3,
        pixel_spacing_col_mm=0.3,
        detector_origin_lps=np.zeros(3),
        row_direction_lps=np.array([1.0, 0.0, 0.0]),
        column_direction_lps=np.array([0.0, 1.0, 0.0]),
        normal_lps=np.array([0.0, 0.0, 1.0]),
        source_lps=np.array([0.0, 0.0, -1000.0]),
        sid_mm=1000.0,
        sod_mm=700.0,
        primary_angle_deg=15.0,
        secondary_angle_deg=-5.0,
        provenance={"sid_mm": "dicom"},
        missing_fields=[],
    )


def affine():
    source = np.array([[10, 10], [80, 12], [14, 70], [70, 80]], dtype=float)
    target = source + np.array([12.0, -8.0])
    return estimate_affine_ransac(source, target, image_diagonal=200.0)


def test_pose_seeds_are_deterministic_and_include_physical_base() -> None:
    first = generate_pose_seeds(linkage(), affine(), geometry(), seed=42, count=8)
    second = generate_pose_seeds(linkage(), affine(), geometry(), seed=42, count=8)

    assert np.array_equal(
        np.asarray([seed.as_vector for seed in first]),
        np.asarray([seed.as_vector for seed in second]),
    )
    assert first[0].name == "dicom_affine_base"
    assert first[0].rotation_deg[0] == pytest.approx(15.0, abs=1e-10)
    assert first[0].translation_mm[1] == pytest.approx(700.0, abs=1e-10)
    assert sum(not seed.diagnostic_only for seed in first) == 8
    assert any(seed.diagnostic_only for seed in first)


def test_pose_seed_uses_detector_spacing_for_affine_translation() -> None:
    seeds = generate_pose_seeds(linkage(), affine(), geometry(), seed=7, count=3)

    assert seeds[0].translation_mm[0] == pytest.approx(12.0 * 0.3, abs=1e-10)
    assert seeds[0].translation_mm[2] == pytest.approx(-8.0 * 0.3, abs=1e-10)
