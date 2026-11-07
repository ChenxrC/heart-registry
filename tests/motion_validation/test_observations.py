"""Behavioral tests with hand-placed image texture, not fitted-motion or mocks."""

from __future__ import annotations

import importlib
import json
from pathlib import Path

import numpy as np
import pytest


def _api(name: str):
    # Keep the initial RED a feature assertion, not a collection/import error.
    try:
        module = importlib.import_module("ctdr_pipeline.motion_validation.observations")
    except ModuleNotFoundError as error:
        if error.name != "ctdr_pipeline.motion_validation.observations":
            raise
        pytest.fail("Task 2 observations implementation is missing")
    return getattr(module, name)


def _texture() -> np.ndarray:
    # An asymmetric, manually constructed corner-rich target on a flat field.
    image = np.full((80, 96), 60, dtype=np.uint8)
    image[24:49, 28:55] = 110
    image[29:43, 34:48] = 180
    image[35:40, 39:44] = 80
    image[27:32, 49:53] = 150
    return image


def _shift(image: np.ndarray, dx: int, dy: int) -> np.ndarray:
    shifted = np.full_like(image, 60)
    rows, columns = image.shape
    source_x, source_y = max(0, -dx), max(0, -dy)
    target_x, target_y = max(0, dx), max(0, dy)
    width, height = columns - abs(dx), rows - abs(dy)
    shifted[target_y : target_y + height, target_x : target_x + width] = image[
        source_y : source_y + height, source_x : source_x + width
    ]
    return shifted


def _series(tmp_path: Path, cine: np.ndarray, roi: np.ndarray | None, **metadata) -> Path:
    series = tmp_path / "series_02"
    series.mkdir()
    manifest = {
        "shape_frame_row_column": list(cine.shape),
        "source_record": {"series_number": 2, "frame_time_ms": 40.0},
        **metadata,
    }
    (series / "series_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    cine.tofile(series / "cine_u8.raw")
    if roi is not None:
        (series / "temporal").mkdir()
        roi.tofile(series / "temporal" / "temporal_centerlines_u8.raw")
    return series


def _assert_json_and_track_contract(result: dict, count: int) -> None:
    json.dumps(result, allow_nan=False)
    assert len(result["quality_by_frame"]) == count
    ids = [track["track_id"] for track in result["tracks"]]
    assert len(set(ids)) == len(ids)
    for track in result["tracks"]:
        assert track["semantic_status"] == "unreviewed_image_feature"
        for field in ("xy_px", "valid", "fb_error_px", "failure_reason"):
            assert len(track[field]) == count
        for xy, valid, error, reason in zip(
            track["xy_px"], track["valid"], track["fb_error_px"], track["failure_reason"]
        ):
            assert type(valid) is bool
            if valid:
                assert len(xy) == 2 and np.isfinite(xy).all()
                assert np.isfinite(error) and error >= 0
                assert reason is None
            else:
                assert xy is None and error is None
                assert isinstance(reason, str) and reason


def test_recovers_hand_placed_translation_in_xy_order() -> None:
    # Break caught: swapping row/column, returning stationary points, or wrong LK direction.
    cine = np.stack([_shift(_texture(), 2 * index, -index) for index in range(5)])
    result = _api("track_points_bidirectional")(cine, np.array([[34.0, 29.0]]), 0)
    track = result["tracks"][0]
    assert track["valid"] == [True] * 5
    np.testing.assert_allclose(
        track["xy_px"], [[34, 29], [36, 28], [38, 27], [40, 26], [42, 25]], atol=0.2
    )
    _assert_json_and_track_contract(result, 5)


def test_middle_reference_propagates_independently_to_both_sides() -> None:
    # Break caught: only tracking forward or initializing backward from the last frame.
    cine = np.stack([_shift(_texture(), 2 * index, index) for index in range(5)])
    result = _api("track_points_bidirectional")(cine, np.array([[38.0, 31.0]]), 2)
    assert result["reference_frame_index"] == 2
    assert result["tracks"][0]["valid"] == [True] * 5
    np.testing.assert_allclose(
        result["tracks"][0]["xy_px"],
        [[34, 29], [36, 30], [38, 31], [40, 32], [42, 33]],
        atol=0.2,
    )


def test_static_texture_brightness_changes_do_not_create_motion() -> None:
    # Break caught: feeding unnormalized intensities to LK under global affine brightness.
    image = _texture()
    cine = np.stack([image, image // 2 + 30, image + 35, image // 2 + 15])
    result = _api("track_points_bidirectional")(cine, np.array([[34.0, 29.0]]), 1)
    assert result["tracks"][0]["valid"] == [True] * 4
    np.testing.assert_allclose(result["tracks"][0]["xy_px"], [[34, 29]] * 4, atol=0.2)


@pytest.mark.parametrize("occlusion", ["blank_frame", "local_flat", "unrelated_texture"])
def test_occlusion_terminates_one_direction_without_resurrecting_identity(occlusion: str) -> None:
    # Break caught: forwarding old coordinates, interpolation, or re-detection after a loss.
    cine = np.repeat(_texture()[None], 6, axis=0)
    if occlusion == "blank_frame":
        cine[3] = 60
    elif occlusion == "local_flat":
        cine[3, 15:58, 18:65] = 60
        cine[3, 60:70, 75:85] = 180  # not a globally blank image
    else:
        rng = np.random.default_rng(19)
        cine[3, 18:54, 22:60] = rng.integers(60, 181, (36, 38), dtype=np.uint8)
    result = _api("track_points_bidirectional")(cine, np.array([[34.0, 29.0]]), 2)
    assert result["tracks"][0]["valid"] == [True, True, True, False, False, False]
    _assert_json_and_track_contract(result, 6)


def test_backward_loss_does_not_prevent_forward_tracking() -> None:
    cine = np.repeat(_texture()[None], 5, axis=0)
    cine[1] = 60
    result = _api("track_points_bidirectional")(cine, np.array([[34.0, 29.0]]), 2)
    assert result["tracks"][0]["valid"] == [False, False, True, True, True]
    _assert_json_and_track_contract(result, 5)


def test_reference_seed_requires_actual_two_dimensional_texture() -> None:
    # Break caught: declaring every seed valid at the reference, including blank/edge-only seeds.
    cine = np.repeat(_texture()[None], 3, axis=0)
    seeds = np.array([[34.0, 29.0], [75.0, 65.0], [2.0, 2.0], [41.0, 24.0]])
    result = _api("track_points_bidirectional")(cine, seeds, 1)
    assert result["tracks"][0]["valid"] == [True, True, True]
    for track in result["tracks"][1:]:
        assert track["valid"] == [False, False, False]
    _assert_json_and_track_contract(result, 3)


def test_boundary_exit_is_missing_and_never_clamped() -> None:
    cine = np.stack([_shift(_texture(), dx, 0) for dx in (0, 15, 30, 45, 60, 0)])
    result = _api("track_points_bidirectional")(cine, np.array([[34.0, 29.0]]), 0)
    track = result["tracks"][0]
    assert track["valid"][0] is True
    assert track["valid"][-2:] == [False, False]
    _assert_json_and_track_contract(result, 6)


def test_forward_backward_threshold_controls_acceptance_of_real_lk_residual() -> None:
    # Break caught: ignoring the caller's FB threshold while reporting valid tracks.
    cine = np.stack([_shift(_texture(), dx, 0) for dx in (0, 2, 4)])
    seeds = np.array([[34.25, 29.35]])
    loose = _api("track_points_bidirectional")(cine, seeds, 0, max_fb_error_px=1.5)
    strict = _api("track_points_bidirectional")(cine, seeds, 0, max_fb_error_px=0.00001)
    assert loose["tracks"][0]["valid"] == [True, True, True]
    assert strict["tracks"][0]["valid"] == [True, False, False]
    assert strict["tracks"][0]["failure_reason"][1] == "forward_backward_error"
    assert strict["quality_by_frame"][1]["valid_track_count"] == 0
    assert strict["quality_by_frame"][1]["missing_track_count"] == 1
    assert strict["quality_by_frame"][1]["median_fb_error_px"] is None
    _assert_json_and_track_contract(strict, 3)


@pytest.mark.parametrize(
    "cine,seeds,reference,threshold",
    [
        (np.zeros((8, 8), np.uint8), [[2, 2]], 0, 1.5),
        (np.zeros((0, 8, 8), np.uint8), [[2, 2]], 0, 1.5),
        (np.zeros((2, 8, 8), float), [[2, 2]], 0, 1.5),
        (np.zeros((2, 8, 8), np.uint8), [[2, 2]], -1, 1.5),
        (np.zeros((2, 8, 8), np.uint8), [[2, 2]], 2, 1.5),
        (np.zeros((2, 8, 8), np.uint8), [[2, 2]], 0.5, 1.5),
        (np.zeros((2, 8, 8), np.uint8), [[2, 2]], True, 1.5),
        (np.zeros((2, 8, 8), np.uint8), [2, 2], 0, 1.5),
        (np.zeros((2, 8, 8), np.uint8), [[float("nan"), 2]], 0, 1.5),
        (np.zeros((2, 8, 8), np.uint8), [[float("inf"), 2]], 0, 1.5),
        (np.zeros((2, 8, 8), np.uint8), [[-1, 2]], 0, 1.5),
        (np.zeros((2, 8, 8), np.uint8), [[8, 2]], 0, 1.5),
        (np.zeros((2, 8, 8), np.uint8), [[2, 2]], 0, 0),
        (np.zeros((2, 8, 8), np.uint8), [[2, 2]], 0, -1),
        (np.zeros((2, 8, 8), np.uint8), [[2, 2]], 0, float("inf")),
        (np.zeros((2, 8, 8), np.uint8), [[2, 2]], 0, float("nan")),
    ],
)
def test_invalid_tracking_inputs_are_rejected(cine, seeds, reference, threshold) -> None:
    with pytest.raises((TypeError, ValueError)):
        _api("track_points_bidirectional")(
            cine, np.asarray(seeds), reference, max_fb_error_px=threshold
        )


def test_empty_seed_set_returns_empty_json_safe_quality() -> None:
    result = _api("track_points_bidirectional")(
        np.repeat(_texture()[None], 3, axis=0), np.empty((0, 2)), 1
    )
    assert result["tracks"] == []
    _assert_json_and_track_contract(result, 3)


def test_report_uses_roi_richest_frame_without_trusting_display_intervals(tmp_path: Path) -> None:
    # Break caught: old-phase/full-field seeds or promoting untyped display intervals to acquisition.
    cine = np.repeat(_texture()[None], 4, axis=0)
    roi = np.zeros_like(cine)
    roi[1, 29, 34:38] = 255
    roi[2, 29, 34] = 255
    series = _series(tmp_path, cine, roi, frame_times_ms=[40, 50, 60, 70])
    (series / "phase_report.json").write_text("not a valid JSON document", encoding="utf-8")
    before = {path: path.read_bytes() for path in series.rglob("*") if path.is_file()}
    result = _api("build_observation_report")(series, max_points=2)
    assert result["run_id"] == "series_02"
    assert result["reference_frame_index"] == 1
    assert result["frame_times_ms"] == [None] * 4
    assert result["status"] == "provisional_unreviewed"
    assert 0 < len(result["tracks"]) <= 2
    for track in result["tracks"]:
        x, y = track["xy_px"][1]
        assert 31 <= x <= 40 and 26 <= y <= 32
    assert 1 in result["review_frame_indices"]
    assert result["limitations"]
    assert result["config"]["max_points"] == 2
    assert before == {path: path.read_bytes() for path in series.rglob("*") if path.is_file()}
    _assert_json_and_track_contract(result, 4)


def test_report_honors_explicit_reference_without_trusting_cache_scalar(tmp_path: Path) -> None:
    cine = np.repeat(_texture()[None], 12, axis=0)
    roi = np.zeros_like(cine)
    roi[:, 29, 34] = 255
    result = _api("build_observation_report")(
        _series(tmp_path, cine, roi), reference_frame=5, max_points=1
    )
    assert result["reference_frame_index"] == 5
    assert result["frame_times_ms"] == [None] * 12
    assert len(result["review_frame_indices"]) == 8
    assert {0, 5, 11}.issubset(result["review_frame_indices"])
    _assert_json_and_track_contract(result, 12)


def test_reference_seed_without_temporal_observations_is_insufficient(tmp_path: Path) -> None:
    # Break caught: promoting a one-frame corner into evidence of motion.
    cine = _texture()[None]
    roi = np.zeros_like(cine)
    roi[0, 29, 34] = 255
    result = _api("build_observation_report")(_series(tmp_path, cine, roi))
    assert result["status"] == "insufficient"
    assert result["tracks"] and result["tracks"][0]["valid"] == [True]
    assert result["review_frame_indices"] == [0]
    _assert_json_and_track_contract(result, 1)


@pytest.mark.parametrize("roi_kind", ["missing", "empty", "featureless"])
def test_missing_empty_or_featureless_roi_is_insufficient(tmp_path: Path, roi_kind: str) -> None:
    cine = np.repeat(_texture()[None], 3, axis=0)
    roi = None if roi_kind == "missing" else np.zeros_like(cine)
    if roi_kind == "featureless":
        roi[:, 65, 75] = 255  # texture exists elsewhere, but never in this ROI
    result = _api("build_observation_report")(_series(tmp_path, cine, roi))
    assert result["status"] == "insufficient"
    assert result["tracks"] == []
    assert result["limitations"]
    _assert_json_and_track_contract(result, 3)


@pytest.mark.parametrize("target,delta", [("cine_u8.raw", -1), ("cine_u8.raw", 1), ("temporal/temporal_centerlines_u8.raw", -1), ("temporal/temporal_centerlines_u8.raw", 1)])
def test_raw_lengths_must_match_exactly(tmp_path: Path, target: str, delta: int) -> None:
    cine = np.repeat(_texture()[None], 2, axis=0)
    series = _series(tmp_path, cine, np.ones_like(cine))
    path = series / target
    original = path.read_bytes()
    path.write_bytes(original[:-1] if delta < 0 else original + b"\0")
    with pytest.raises(ValueError, match="[Ll]ength|[Bb]yte|[Ss]ize"):
        _api("build_observation_report")(series)


@pytest.mark.parametrize("shape", [[2, 80], [2, 80, 96.0], [2, 0, 96], [True, 80, 96]])
def test_manifest_shape_is_strictly_validated(tmp_path: Path, shape) -> None:
    cine = np.repeat(_texture()[None], 2, axis=0)
    series = _series(tmp_path, cine, np.ones_like(cine), shape_frame_row_column=shape)
    with pytest.raises(ValueError):
        _api("build_observation_report")(series)


@pytest.mark.parametrize("value", [0, -1, 1.5, True])
def test_report_rejects_invalid_point_limit(tmp_path: Path, value) -> None:
    cine = np.repeat(_texture()[None], 2, axis=0)
    series = _series(tmp_path, cine, np.ones_like(cine))
    with pytest.raises((TypeError, ValueError)):
        _api("build_observation_report")(series, max_points=value)


@pytest.mark.parametrize("intervals", [[40], [40, float("nan")], [40, -1], [40, "bad"]])
def test_report_ignores_bad_display_timing_without_losing_spatial_observations(tmp_path: Path, intervals) -> None:
    # Break caught: untyped cache display data kills otherwise usable spatial tracks.
    cine = np.repeat(_texture()[None], 2, axis=0)
    series = _series(tmp_path, cine, np.ones_like(cine), frame_times_ms=intervals)
    result = _api("build_observation_report")(series)
    assert result["frame_times_ms"] == [None, None]
    assert result["status"] == "provisional_unreviewed"
    assert result["tracks"]
    _assert_json_and_track_contract(result, 2)


def test_report_without_timing_keeps_unknown_times_null(tmp_path: Path) -> None:
    cine = np.repeat(_texture()[None], 2, axis=0)
    series = _series(tmp_path, cine, np.ones_like(cine), source_record={"series_number": 2})
    result = _api("build_observation_report")(series)
    assert result["frame_times_ms"] == [None, None]
    assert "timing" in " ".join(result["limitations"]).lower()
    _assert_json_and_track_contract(result, 2)


@pytest.mark.parametrize("metadata", [
    {"frame_times_ms": [125, 125, 125, 125]},
    {"source_record": {"frame_time_ms": 125}},
    {"source_record": {}, "median_frame_time_ms": 125},
    {"source_record": {}, "frame_time_ms": 125},
    {"frame_times_ms": [0, 100, 120, 80]},
])
def test_standalone_display_metadata_never_establishes_acquisition_time(tmp_path: Path, metadata: dict) -> None:
    # Break caught: any exporter display duration/median is silently promoted to measured time.
    cine = np.repeat(_texture()[None], 4, axis=0)
    result = _api("build_observation_report")(_series(tmp_path, cine, np.ones_like(cine), **metadata))
    assert result["frame_times_ms"] == [None] * 4
    assert result["timing_status"] == "unverified"
    assert result["timing_source"] == "unknown"
    assert result["timing_provenance"]["source_verified"] is False
    assert result["status"] == "provisional_unreviewed"
    _assert_json_and_track_contract(result, 4)


def _trusted_timing(**overrides) -> dict:
    return {
        "source_verified": True, "frame_count": 4, "timing_status": "available",
        "timing_source": "FrameTimeVector", "frame_timestamps_ms": [0, 100, 220, 300],
        **overrides,
    }


def _report_with_timing(series: Path, timing) -> dict:
    import inspect

    build = _api("build_observation_report")
    assert "acquisition_timing" in inspect.signature(build).parameters, "explicit acquisition-timing contract is missing"
    return build(series, acquisition_timing=timing)


def test_explicit_verified_cumulative_timing_overrides_even_malformed_display_cache(tmp_path: Path) -> None:
    # Break caught: trusting cache values or cumulatively summing already cumulative audit times.
    cine = np.repeat(_texture()[None], 4, axis=0)
    series = _series(tmp_path, cine, np.ones_like(cine), frame_times_ms=[999])
    result = _report_with_timing(series, _trusted_timing())
    assert result["frame_times_ms"] == [0, 100, 220, 300]
    assert result["timing_status"] == "available"
    assert result["timing_source"] == "FrameTimeVector"
    assert result["timing_provenance"]["source_verified"] is True
    assert result["status"] == "provisional_unreviewed"
    _assert_json_and_track_contract(result, 4)


@pytest.mark.parametrize("timing,status", [
    (_trusted_timing(source_verified=False), "unverified"),
    (_trusted_timing(source_verified="true"), "unverified"),
    (_trusted_timing(source_link_status="mismatch"), "unverified"),
    (_trusted_timing(source_link_status="unknown"), "unverified"),
    (_trusted_timing(timing_status="insufficient", frame_timestamps_ms=[]), "insufficient"),
    (_trusted_timing(timing_status="unknown"), "insufficient"),
    (_trusted_timing(frame_count=3), "frame_count_mismatch"),
    (_trusted_timing(frame_count=True), "frame_count_mismatch"),
    (_trusted_timing(frame_timestamps_ms=[0, 100]), "invalid"),
    (_trusted_timing(frame_timestamps_ms=[0, 100, 90, 300]), "invalid"),
    (_trusted_timing(frame_timestamps_ms=[1, 100, 220, 300]), "invalid"),
    (_trusted_timing(frame_timestamps_ms=[0, 100, float("nan"), 300]), "invalid"),
    (_trusted_timing(frame_timestamps_ms=[0, 100, float("inf"), 300]), "invalid"),
    (_trusted_timing(frame_timestamps_ms=[False, 100, 220, 300]), "invalid"),
    (_trusted_timing(frame_timestamps_ms=[0, "100", 220, 300]), "invalid"),
    (_trusted_timing(timing_source=None), "invalid"),
    ([], "invalid"),
])
def test_insufficient_or_malformed_audit_never_falls_back_to_display_times(tmp_path: Path, timing, status: str) -> None:
    # Break caught: invalid/unverified override silently falls back or discards spatial observations.
    cine = np.repeat(_texture()[None], 4, axis=0)
    series = _series(tmp_path, cine, np.ones_like(cine), frame_times_ms=[125] * 4)
    result = _report_with_timing(series, timing)
    assert result["frame_times_ms"] == [None] * 4
    assert result["timing_status"] == status
    assert result["timing_source"] == "unknown"
    if status == "unverified":
        assert result["timing_provenance"]["source_verified"] is False
    assert "timing" in " ".join(result["limitations"]).lower()
    assert result["status"] == "provisional_unreviewed"
    assert any(all(track["valid"]) for track in result["tracks"])
    _assert_json_and_track_contract(result, 4)
