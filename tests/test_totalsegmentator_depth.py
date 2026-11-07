from __future__ import annotations

import numpy as np

from run_totalsegmentator_depth import (
    LocalizerGeometry,
    localizer_pixel_to_lps,
    lps_to_localizer_pixel,
    recover_landmark_depths,
    transform_points,
    validate_frame_of_reference,
)


def geometry() -> LocalizerGeometry:
    return LocalizerGeometry(
        origin_lps_mm=np.array([-200.0, 153.0, -500.0]),
        row_direction_lps=np.array([1.0, 0.0, 0.0]),
        column_direction_lps=np.array([0.0, 0.0, -1.0]),
        normal_direction_lps=np.array([0.0, 1.0, 0.0]),
        row_spacing_mm=0.8,
        column_spacing_mm=0.8,
        rows=512,
        columns=512,
    )


def test_localizer_pixel_lps_round_trip_including_depth() -> None:
    localizer = geometry()
    uv = np.array([[125.0, 80.0], [250.5, 130.25]])
    depth = np.array([15.0, 42.0])

    points = localizer_pixel_to_lps(uv, localizer, depth)
    recovered_uv, recovered_depth = lps_to_localizer_pixel(points, localizer)

    assert np.allclose(recovered_uv, uv)
    assert np.allclose(recovered_depth, depth)


def test_homogeneous_point_transform() -> None:
    matrix = np.array(
        [
            [2.0, 0.0, 3.0],
            [0.0, 4.0, -5.0],
            [0.0, 0.0, 1.0],
        ]
    )
    result = transform_points(np.array([[1.0, 2.0], [3.0, 4.0]]), matrix)
    assert np.allclose(result, [[5.0, 3.0], [9.0, 11.0]])


def test_frame_of_reference_validation_rejects_mismatch() -> None:
    validate_frame_of_reference("same", "same")
    try:
        validate_frame_of_reference("ct", "localizer")
    except RuntimeError as error:
        assert "do not share FrameOfReferenceUID" in str(error)
    else:
        raise AssertionError("Expected mismatched FrameOfReferenceUID to be rejected")


def test_depth_recovery_uses_semantic_label_and_reports_interval() -> None:
    localizer = geometry()
    uv = np.array([100.0, 100.0])
    reference = localizer_pixel_to_lps(uv, localizer)[0]
    t1_points = np.array(
        [reference + np.array([dx, depth, dz]) for dx in (-1.0, 0.0, 1.0) for dz in (-1.0, 0.0, 1.0) for depth in (10.0, 14.0, 18.0)]
    )
    c7_points = t1_points + np.array([0.0, -2.0, 0.0])
    rows = recover_landmark_depths(
        [
            {
                "landmark_id": "T1",
                "kind": "vertebra_region_center",
                "slide_xy_px": [0.0, 0.0],
                "localizer_uv_px": uv.tolist(),
            }
        ],
        {"vertebrae_C7": c7_points, "vertebrae_T1": t1_points},
        localizer,
        neighborhood_mm=4.0,
    )

    row = rows[0]
    assert row["selected_vertebra"] == "vertebrae_T1"
    assert row["selection_reason"] == "physician_anatomical_label"
    assert np.isclose(row["estimated_depth_from_localizer_plane_mm"], 14.0)
    assert row["depth_interval_p10_p90_mm"][0] < 14.0
    assert row["depth_interval_p10_p90_mm"][1] > 14.0
    assert row["label_ambiguity"] == "resolved_by_physician_anatomical_label"


def test_depth_recovery_selects_nearest_projected_vertebra_for_generic_point() -> None:
    localizer = geometry()
    uv = np.array([80.0, 120.0])
    reference = localizer_pixel_to_lps(uv, localizer)[0]
    t4_points = np.array(
        [reference + np.array([dx, 30.0, dz]) for dx in (-1.0, 0.0, 1.0) for dz in (-1.0, 0.0, 1.0)]
    )
    t5_points = t4_points + np.array([0.0, 10.0, -20.0])
    rows = recover_landmark_depths(
        [
            {
                "landmark_id": "Z",
                "kind": "circle_center",
                "slide_xy_px": [0.0, 0.0],
                "localizer_uv_px": uv.tolist(),
            }
        ],
        {"vertebrae_T4": t4_points, "vertebrae_T5": t5_points},
        localizer,
        neighborhood_mm=4.0,
    )

    assert rows[0]["selected_vertebra"] == "vertebrae_T4"
    assert rows[0]["selection_reason"] == "nearest_projected_vertebra"
    assert np.isclose(rows[0]["estimated_depth_from_localizer_plane_mm"], 30.0)
