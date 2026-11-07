from __future__ import annotations

import numpy as np

import build_ct_dr_3d_viewer as viewer


def test_ap_detector_basis_matches_lps_convention() -> None:
    candidate = viewer.OrientationCandidate(1, 1, "patient_x", 0)
    basis = viewer.detector_basis(0.0, 0.0, candidate)
    np.testing.assert_allclose(basis.u_lps, [1.0, 0.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(basis.v_lps, [0.0, 0.0, -1.0], atol=1e-12)
    np.testing.assert_allclose(basis.n_lps, [0.0, 1.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(np.cross(basis.u_lps, basis.v_lps), basis.n_lps, atol=1e-12)


def test_every_orientation_candidate_is_orthonormal() -> None:
    candidates = viewer.candidate_orientations()
    assert len(candidates) == 16
    for candidate in candidates:
        basis = viewer.detector_basis(-1.1, -0.2, candidate)
        matrix = np.column_stack([basis.u_lps, basis.v_lps, basis.n_lps])
        np.testing.assert_allclose(matrix.T @ matrix, np.eye(3), atol=1e-12)
        assert np.linalg.det(matrix) > 0.999999


def test_projection_uses_sdd_sod_and_native_detector_row_direction() -> None:
    basis = viewer.detector_basis(0.0, 0.0, viewer.OrientationCandidate(1, 1, "patient_x", 0))
    pixels, _, depths, source, detector = viewer.project_points(
        points_lps=np.asarray([[10.0, 0.0, -20.0]]),
        isocenter_lps=np.zeros(3),
        basis=basis,
        sdd_mm=1000.0,
        sod_mm=800.0,
        pixel_spacing_row_col_mm=np.asarray([0.5, 0.5]),
        principal_uv_px=np.asarray([255.5, 255.5]),
    )
    np.testing.assert_allclose(source, [0.0, -800.0, 0.0])
    np.testing.assert_allclose(detector, [0.0, 200.0, 0.0])
    np.testing.assert_allclose(depths, [800.0])
    np.testing.assert_allclose(pixels[0], [280.5, 305.5])


def test_quality_gate_does_not_promote_warning_to_accepted() -> None:
    base = {
        "order_consistent": True,
        "points_between_source_and_detector": True,
        "source_side_expected": True,
        "basis_max_abs_dot": 0.0,
        "isocenter_distance_to_ct_bbox_mm": 0.0,
        "rms_error_px": 4.9,
        "max_error_px": 9.9,
    }
    assert viewer.calibration_quality(base)[0] == "accepted"
    assert viewer.calibration_quality(dict(base, rms_error_px=5.0))[0] == "warning"
    assert viewer.calibration_quality(dict(base, order_consistent=False))[0] == "diagnostic-only"
