import numpy as np
import pytest
from pydicom.dataset import Dataset

from ctdr_pipeline.dicom.geometry import pixel_to_lps_ray, projection_geometry


def test_projection_geometry_builds_ray_through_detector_pixel() -> None:
    dataset = Dataset()
    dataset.Rows = 100
    dataset.Columns = 120
    dataset.PixelSpacing = [1.0, 0.5]
    dataset.ImageOrientationPatient = [1, 0, 0, 0, 1, 0]
    dataset.ImagePositionPatient = [10.0, 20.0, 30.0]
    dataset.DistanceSourceToDetector = 1000.0
    dataset.DistanceSourceToPatient = 700.0
    dataset.PositionerPrimaryAngle = 15.0
    dataset.PositionerSecondaryAngle = -5.0

    geometry = projection_geometry(dataset)
    ray = pixel_to_lps_ray(np.array([20.0, 10.0]), geometry)

    expected_detector_point = np.array([20.0, 30.0, 30.0])
    assert geometry.complete
    assert np.linalg.norm(ray.direction) == pytest.approx(1.0)
    assert np.allclose(
        ray.origin + ray.direction * np.dot(expected_detector_point - ray.origin, ray.direction),
        expected_detector_point,
    )


def test_missing_projection_fields_are_explicit() -> None:
    dataset = Dataset()
    dataset.Rows = 100
    dataset.Columns = 120

    geometry = projection_geometry(dataset)

    assert not geometry.complete
    assert geometry.provenance["sid_mm"] == "unknown"
    assert "pixel_spacing" in geometry.missing_fields
