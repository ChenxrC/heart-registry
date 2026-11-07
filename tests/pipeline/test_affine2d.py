import numpy as np

from ctdr_pipeline.registration.affine2d import (
    estimate_affine_ransac,
    transform_points,
)


def test_affine_recovery_has_subpixel_rms_and_positive_determinant() -> None:
    source = np.array([[10, 10], [80, 12], [14, 70], [70, 80]], dtype=float)
    truth = np.array(
        [[1.1, 0.15, 8.0], [-0.08, 0.95, 12.0], [0.0, 0.0, 1.0]],
        dtype=float,
    )
    target = transform_points(source, truth)

    result = estimate_affine_ransac(source, target, image_diagonal=150.0)

    assert result.metrics.rms_px < 1e-6
    assert result.metrics.determinant > 0
    assert result.accepted


def test_ransac_ignores_one_large_outlier() -> None:
    source = np.array(
        [[10, 10], [80, 12], [14, 70], [70, 80], [45, 42]], dtype=float
    )
    truth = np.array(
        [[1.0, 0.1, 7.0], [-0.05, 1.05, 4.0], [0.0, 0.0, 1.0]], dtype=float
    )
    target = transform_points(source, truth)
    target[-1] += np.array([80.0, -60.0])

    result = estimate_affine_ransac(source, target, image_diagonal=150.0)

    assert sum(result.inliers) == 4
    assert result.accepted
    assert np.allclose(result.matrix, truth, atol=1e-5)


def test_vertical_flip_can_never_be_accepted() -> None:
    source = np.array([[10, 10], [80, 12], [14, 70], [70, 80]], dtype=float)
    target = source.copy()
    target[:, 1] = 99.0 - target[:, 1]

    result = estimate_affine_ransac(source, target, image_diagonal=140.0)

    assert not result.accepted
    assert result.metrics.determinant < 0
    assert "mirror_or_flip" in result.reasons
