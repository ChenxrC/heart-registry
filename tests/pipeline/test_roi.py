from pathlib import Path

import numpy as np
import SimpleITK as sitk

from ctdr_pipeline.dicom.geometry import ProjectionGeometry
from ctdr_pipeline.registration.affine2d import LandmarkPair
from ctdr_pipeline.registration.roi import build_roi
from ctdr_pipeline.registration.volume import CtVolume


def make_volume() -> CtVolume:
    values = np.full((32, 32, 32), -1000, dtype=np.float32)
    values[10:20, 12:20, 12:20] = 500
    values[10:20, 24:32, 24:32] = 700
    image = sitk.GetImageFromArray(values)
    image.SetSpacing((1.0, 1.0, 1.0))
    affine = np.eye(4, dtype=float)
    return CtVolume(
        series_uid="ct-series",
        data_hu=values,
        voxel_to_lps=affine,
        spacing_xyz=(1.0, 1.0, 1.0),
        origin_lps=(0.0, 0.0, 0.0),
        direction=np.eye(3),
        sitk_image=image,
        source_paths=[],
    )


def make_geometry() -> ProjectionGeometry:
    return ProjectionGeometry(
        rows=32,
        columns=32,
        pixel_spacing_row_mm=1.0,
        pixel_spacing_col_mm=1.0,
        detector_origin_lps=np.array([0.0, 0.0, 0.0]),
        row_direction_lps=np.array([1.0, 0.0, 0.0]),
        column_direction_lps=np.array([0.0, 1.0, 0.0]),
        normal_lps=np.array([0.0, 0.0, 1.0]),
        source_lps=np.array([16.0, 16.0, -100.0]),
        sid_mm=1000.0,
        sod_mm=700.0,
        primary_angle_deg=0.0,
        secondary_angle_deg=0.0,
        provenance={"sid_mm": "dicom"},
        missing_fields=[],
    )


def landmarks() -> list[LandmarkPair]:
    return [
        LandmarkPair(
            landmark_id="T1",
            source_xy=(16.0, 16.0),
            target_xy=(16.0, 16.0),
        )
    ]


def test_baseline_roi_keeps_bone_near_landmark_ray_and_removes_distant_bone() -> None:
    result = build_roi(
        make_volume(),
        landmarks(),
        make_geometry(),
        method="baseline",
        corridor_radius_mm=10.0,
        minimum_component_voxels=100,
    )

    assert result.volume_mask[15, 16, 16]
    assert not result.volume_mask[15, 28, 28]
    assert result.projection_mask.shape == (32, 32)
    assert result.method == "baseline"


def test_totalsegmentator_failure_falls_back_to_baseline(monkeypatch) -> None:
    def fail(*args, **kwargs):
        raise RuntimeError("model unavailable")

    monkeypatch.setattr("ctdr_pipeline.registration.roi._run_totalsegmentator", fail)

    result = build_roi(
        make_volume(),
        landmarks(),
        make_geometry(),
        method="totalsegmentator",
        minimum_component_voxels=100,
    )

    assert result.method == "baseline_fallback"
    assert any("RuntimeError" in warning for warning in result.warnings)
    assert result.volume_mask.any()
