"""Provisional 2-D image-feature observations; never anatomical confirmation.

Coordinates are (column, row) pixels. A failed direction is terminal for that
track ID: no filling, interpolation, re-detection, or reuse of a stale point.
Only this module's two public functions form its interface.
"""

from __future__ import annotations

from collections import Counter
import json
import math
from numbers import Integral, Real
from pathlib import Path

import cv2
import numpy as np

from .input_policy import require_eligible_input


_PATCH_RADIUS = 5
_MIN_PATCH_STD = 3.0
_MIN_EIGENVALUE = 1e-4
_MIN_CORRELATION = 0.6
_ROI_DILATION_RADIUS = 3
_LK = {
    "winSize": (21, 21),
    "maxLevel": 3,
    "criteria": (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
    "minEigThreshold": _MIN_EIGENVALUE,
}
_TRACK_BORDER = max(_PATCH_RADIUS, _LK["winSize"][0] // 2)


def _positive_real(value: object, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{name} must be finite and positive")
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return result


def _reference_index(value: object, count: int) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise ValueError("reference_frame must be an integer frame index")
    if not 0 <= value < count:
        raise ValueError("reference_frame is outside the cine")
    return int(value)


def _normalize(frame: np.ndarray) -> np.ndarray:
    """Frame-local affine normalization; no temporal/motion information used."""
    low, high = np.percentile(frame, [1, 99])
    if high <= low:
        low, high = float(frame.min()), float(frame.max())
    if high <= low:
        return np.zeros_like(frame)
    return np.rint(np.clip((frame.astype(np.float32) - low) * (255.0 / (high - low)), 0, 255)).astype(np.uint8)


def _patch(frame: np.ndarray, xy: np.ndarray) -> np.ndarray:
    return cv2.getRectSubPix(frame, (2 * _PATCH_RADIUS + 1,) * 2, (float(xy[0]), float(xy[1])))


def _point_failure(raw: np.ndarray, eigenvalues: np.ndarray, xy: np.ndarray) -> str | None:
    if not np.isfinite(xy).all():
        return "nonfinite_position"
    x, y = (float(value) for value in xy)
    height, width = raw.shape
    if not (_TRACK_BORDER <= x <= width - 1 - _TRACK_BORDER and _TRACK_BORDER <= y <= height - 1 - _TRACK_BORDER):
        return "outside_tracking_window_border"
    if float(_patch(raw, xy).std()) < _MIN_PATCH_STD:
        return "low_texture_or_invisible"
    if float(eigenvalues[int(round(y)), int(round(x))]) < _MIN_EIGENVALUE:
        return "not_two_dimensional_texture"
    return None


def _correlation(first: np.ndarray, second: np.ndarray) -> float:
    a = first.astype(np.float64).ravel()
    b = second.astype(np.float64).ravel()
    a -= a.mean()
    b -= b.mean()
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / denominator) if denominator > 0 else 0.0


def _configuration(max_fb_error_px: float) -> dict:
    return {
        "method": "pyramidal_lucas_kanade_bidirectional",
        "coordinate_order": "xy_column_row_pixels",
        "normalization": "per_frame_affine_percentile_1_99_with_minmax_fallback",
        "lk_window_px": list(_LK["winSize"]),
        "lk_max_level": _LK["maxLevel"],
        "lk_max_iterations": 30,
        "lk_epsilon": 0.01,
        "max_fb_error_px": max_fb_error_px,
        "min_eigenvalue": _MIN_EIGENVALUE,
        "patch_radius_px": _PATCH_RADIUS,
        "tracking_border_px": _TRACK_BORDER,
        "min_raw_patch_std": _MIN_PATCH_STD,
        "min_patch_correlation": _MIN_CORRELATION,
        "redetection": False,
        "interpolation": False,
    }


def track_points_bidirectional(
    cine: np.ndarray,
    seed_xy: np.ndarray,
    reference_frame: int,
    *,
    max_fb_error_px: float = 1.5,
) -> dict:
    """Track corner-like seeds independently toward earlier and later frames.

    ``cine`` must be raw uint8 N,H,W; ``seed_xy`` is finite M,2 with
    coordinates inside the image. In-image seeds without a full textured
    patch remain invalid even at the reference. Failed observations have
    null coordinates AND null errors; reference FB errors are zero by
    convention (there is no temporal pair at the reference).
    """
    if not isinstance(cine, np.ndarray) or cine.ndim != 3 or any(size <= 0 for size in cine.shape):
        raise ValueError("cine must have nonempty shape (frame, row, column)")
    if cine.dtype != np.uint8:
        raise ValueError("cine must contain raw uint8 pixels")
    count, height, width = cine.shape
    reference = _reference_index(reference_frame, count)
    threshold = _positive_real(max_fb_error_px, "max_fb_error_px")
    if not isinstance(seed_xy, np.ndarray) or seed_xy.ndim != 2 or seed_xy.shape[1] != 2:
        raise ValueError("seed_xy must have shape (point, 2)")
    if seed_xy.dtype.kind not in "iuf" or not np.isfinite(seed_xy).all():
        raise ValueError("seed_xy must contain finite real coordinates")
    if np.any(seed_xy < 0) or np.any(seed_xy[:, 0] > width - 1) or np.any(seed_xy[:, 1] > height - 1):
        raise ValueError("seed_xy is outside the image")
    seeds = seed_xy.astype(np.float32)
    frames = [_normalize(frame) for frame in cine]
    eigenvalues = [cv2.cornerMinEigenVal(frame, blockSize=5, ksize=3) for frame in frames]
    tracks = []
    for track_id, xy in enumerate(seeds):
        reason = _point_failure(cine[reference], eigenvalues[reference], xy)
        track = {
            "track_id": track_id,
            "semantic_status": "unreviewed_image_feature",
            "xy_px": [None] * count,
            "valid": [False] * count,
            "fb_error_px": [None] * count,
            "failure_reason": ["reference_untrackable" if reason else "direction_terminated"] * count,
        }
        track["failure_reason"][reference] = reason
        if reason is None:
            track["xy_px"][reference] = [float(value) for value in xy]
            track["valid"][reference] = True
            track["fb_error_px"][reference] = 0.0
        tracks.append(track)

    for direction in (-1, 1):
        active = [index for index, track in enumerate(tracks) if track["valid"][reference]]
        source_index = reference
        for target_index in range(reference + direction, count if direction == 1 else -1, direction):
            if not active:
                break
            points = np.asarray([tracks[index]["xy_px"][source_index] for index in active], dtype=np.float32).reshape(-1, 1, 2)
            forward, forward_ok, _ = cv2.calcOpticalFlowPyrLK(frames[source_index], frames[target_index], points, None, **_LK)
            # Only finite, successful forward proposals may enter the reverse solve.
            proposal_indices = [] if forward is None or forward_ok is None else [
                offset for offset in range(len(active))
                if forward_ok[offset, 0] and np.isfinite(forward[offset]).all()
            ]
            reverse_by_offset = {}
            if proposal_indices:
                reverse, reverse_ok, _ = cv2.calcOpticalFlowPyrLK(
                    frames[target_index], frames[source_index], forward[proposal_indices], None, **_LK
                )
                if reverse is not None and reverse_ok is not None:
                    reverse_by_offset = {
                        offset: reverse[position, 0]
                        for position, offset in enumerate(proposal_indices)
                        if reverse_ok[position, 0] and np.isfinite(reverse[position]).all()
                    }
            next_active = []
            for offset, track_id in enumerate(active):
                track = tracks[track_id]
                reason = "lk_forward_failed"
                error = None
                xy = None
                if offset in reverse_by_offset:
                    xy = forward[offset, 0]
                    reason = _point_failure(cine[target_index], eigenvalues[target_index], xy)
                    error = float(np.linalg.norm(reverse_by_offset[offset] - points[offset, 0]))
                    if reason is None and (not math.isfinite(error) or error > threshold):
                        reason = "forward_backward_error"
                    if reason is None and _correlation(
                        _patch(frames[source_index], points[offset, 0]), _patch(frames[target_index], xy)
                    ) < _MIN_CORRELATION:
                        reason = "appearance_inconsistent_or_occluded"
                elif offset in proposal_indices:
                    reason = "lk_backward_failed"
                track["failure_reason"][target_index] = reason
                if reason is None:
                    track["xy_px"][target_index] = [float(value) for value in xy]
                    track["valid"][target_index] = True
                    track["fb_error_px"][target_index] = error
                    next_active.append(track_id)
            active = next_active
            source_index = target_index

    quality = []
    for frame_index in range(count):
        errors = [track["fb_error_px"][frame_index] for track in tracks if track["valid"][frame_index]]
        failures = Counter(track["failure_reason"][frame_index] for track in tracks if not track["valid"][frame_index])
        quality.append({
            "frame_index": frame_index,
            "valid_track_count": len(errors),
            "missing_track_count": len(tracks) - len(errors),
            "valid_fraction": len(errors) / len(tracks) if tracks else 0.0,
            "median_fb_error_px": float(np.median(errors)) if errors else None,
            "failure_reason_counts": dict(failures),
        })
    return {"reference_frame_index": reference, "tracks": tracks, "quality_by_frame": quality, "config": _configuration(threshold)}


def _frame_times(acquisition_timing: dict | None, count: int) -> tuple[list, dict]:
    # Exporter durations, including source_record medians, are untyped display
    # metadata. Only the explicit acquisition contract can establish timestamps.
    audit = acquisition_timing if isinstance(acquisition_timing, dict) else {}
    verified = (audit.get("source_verified") is True
                and ("source_link_status" not in audit or audit["source_link_status"] == "verified"))
    provenance = {"source_verified": verified}
    for field in ("source_link_status", "source_relative_path", "timing_status", "timing_source"):
        value = audit.get(field)
        key = f"audit_{field}" if field.startswith("timing_") else field
        provenance[key] = value if isinstance(value, str) else None
    metadata = {"timing_status": "unverified", "timing_source": "unknown", "timing_provenance": provenance}

    def unknown(status: str) -> tuple[list, dict]:
        metadata["timing_status"] = status
        return [None] * count, metadata

    if acquisition_timing is not None and not isinstance(acquisition_timing, dict):
        return unknown("invalid")
    if not verified:
        return unknown("unverified")
    if audit.get("timing_status") != "available":
        return unknown("insufficient")
    if type(audit.get("frame_count")) is not int or audit["frame_count"] != count:
        return unknown("frame_count_mismatch")
    times = audit.get("frame_timestamps_ms")
    source = audit.get("timing_source")
    if not isinstance(source, str) or not source.strip() or not isinstance(times, list) or len(times) != count:
        return unknown("invalid")
    if any(isinstance(value, (bool, np.bool_)) or not isinstance(value, Real) or not math.isfinite(value) for value in times):
        return unknown("invalid")
    if times[0] != 0 or any(second <= first for first, second in zip(times, times[1:])):
        return unknown("invalid")
    metadata.update(timing_status="available", timing_source=source)
    return [float(value) for value in times], metadata


def _read_raw(path: Path, shape: tuple) -> np.ndarray:
    require_eligible_input(path)
    expected = math.prod(shape)
    if path.stat().st_size != expected:
        raise ValueError(f"Raw byte length does not match shape: {path.name}")
    raw = np.fromfile(path, dtype=np.uint8)
    if raw.size != expected:
        raise ValueError(f"Raw byte length changed while reading: {path.name}")
    return raw.reshape(shape)


def _review_indices(count: int, reference: int) -> list[int]:
    # Always include the actual seed frame and endpoints in at most eight panels.
    target_count = min(8, count)
    indices = {0, reference, count - 1}
    for index in np.linspace(0, count - 1, target_count, dtype=int):
        if len(indices) < target_count:
            indices.add(int(index))
    return sorted(indices)


def build_observation_report(
    series_dir: Path,
    *,
    reference_frame: int | None = None,
    max_points: int = 60,
    max_fb_error_px: float = 1.5,
    acquisition_timing: dict | None = None,
) -> dict:
    """Read only the manifest, cine, and optional candidate centerline stack.

    ROI absence/emptiness is insufficiency, not permission for full-field
    detection. Invalid lengths/metadata raise ValueError. Unknown timing is
    null rather than a fabricated frame rate. This function writes nothing.

    ``acquisition_timing`` is an optional trusted acquisition contract, not
    exporter display metadata: ``source_verified`` must be the boolean True,
    ``frame_count`` the exact integer cine length, ``timing_status`` "available",
    ``timing_source`` a nonempty provenance string, and ``frame_timestamps_ms``
    N finite cumulative acquisition milliseconds, starting at zero and strictly
    increasing. Optional ``source_link_status`` and ``source_relative_path``
    describe the identity-matched audit; a supplied link status must be "verified".
    Standalone callers assert the trust;
    the runner establishes it from preflight source links. Insufficient or
    malformed contracts keep N null times and spatial observations. Untyped
    cache intervals/scalars/medians never supply acquisition timestamps.
    """
    if isinstance(max_points, (bool, np.bool_)) or not isinstance(max_points, Integral) or max_points <= 0:
        raise ValueError("max_points must be a positive integer")
    threshold = _positive_real(max_fb_error_px, "max_fb_error_px")
    series_dir = Path(series_dir)
    require_eligible_input(series_dir)
    manifest_path = series_dir / "series_manifest.json"
    require_eligible_input(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError("series manifest must be an object")
    shape = manifest.get("shape_frame_row_column")
    if not isinstance(shape, list) or len(shape) != 3 or any(type(value) is not int or value <= 0 for value in shape):
        raise ValueError("shape_frame_row_column must contain three positive integers")
    source_record = manifest.get("source_record", {})
    if not isinstance(source_record, dict):
        raise ValueError("source_record must be an object")
    # Validate copied provenance too: Python's decoder otherwise accepts NaN/Infinity.
    json.dumps(source_record, allow_nan=False)
    cine = _read_raw(series_dir / "cine_u8.raw", tuple(shape))
    count = shape[0]
    frame_times, timing_metadata = _frame_times(acquisition_timing, count)
    reference = None if reference_frame is None else _reference_index(reference_frame, count)
    roi_path = series_dir / "temporal" / "temporal_centerlines_u8.raw"
    require_eligible_input(roi_path)
    centerlines = _read_raw(roi_path, tuple(shape)) if roi_path.exists() else None
    if reference is None:
        reference = int(np.argmax(np.count_nonzero(centerlines, axis=(1, 2)))) if centerlines is not None else 0
    seeds = np.empty((0, 2), dtype=np.float32)
    limitations = [
        "All points are unreviewed image features, not confirmed vessels or anatomical bifurcations.",
        "Validity means image tracking checks only; no anatomical correctness or end-diastolic phase is established.",
        "2-D projected motion cannot identify unique 3-D cardiac motion or patient 3-D TRE.",
        "No complete cardiac-edge extraction or clinical validation is claimed.",
        "Texture, correlation, and forward-backward thresholds are heuristic and may reject true motion or retain wrong matches.",
        "Loss is terminal per direction; missing observations are not filled or re-detected.",
    ]
    if timing_metadata["timing_status"] != "available":
        limitations.append(f"Acquisition timing is {timing_metadata['timing_status']}; frame_times_ms entries are null. Display metadata is not acquisition evidence.")
    if centerlines is None:
        limitations.append("Candidate centerline ROI is missing; full-field fallback is disabled.")
    elif not np.any(centerlines[reference]):
        limitations.append("Candidate centerline ROI is empty at the reference; full-field fallback is disabled.")
    else:
        diameter = 2 * _ROI_DILATION_RADIUS + 1
        mask = cv2.dilate((centerlines[reference] != 0).astype(np.uint8) * 255, np.ones((diameter, diameter), dtype=np.uint8))
        # Exclude unsupported border patches before selecting corners.
        mask[:_TRACK_BORDER] = 0
        mask[-_TRACK_BORDER:] = 0
        mask[:, :_TRACK_BORDER] = 0
        mask[:, -_TRACK_BORDER:] = 0
        corners = cv2.goodFeaturesToTrack(
            _normalize(cine[reference]), maxCorners=int(max_points), qualityLevel=0.01,
            minDistance=6.0, mask=mask, blockSize=5,
        )
        if corners is not None:
            seeds = corners.reshape(-1, 2)
        else:
            limitations.append("No trackable corner-like texture was found inside the dilated candidate ROI.")
    result = track_points_bidirectional(cine, seeds, reference, max_fb_error_px=threshold)
    has_temporal_observations = any(sum(track["valid"]) >= 2 for track in result["tracks"])
    if not has_temporal_observations:
        limitations.append("Insufficient linked temporal observations; a reference seed alone is not a motion measurement.")
    result.update({
        "run_id": series_dir.name,
        "source_record": source_record,
        "frame_times_ms": frame_times,
        **timing_metadata,
        "review_frame_indices": _review_indices(count, reference),
        "status": "provisional_unreviewed" if has_temporal_observations else "insufficient",
        "limitations": limitations,
    })
    result["config"].update({
        "max_points": int(max_points),
        "roi_source": "temporal/temporal_centerlines_u8.raw",
        "roi_dilation_radius_px": _ROI_DILATION_RADIUS,
        "roi_dilation_shape": "square",
        "corner_quality_level": 0.01,
        "corner_min_distance_px": 6.0,
        "corner_block_size": 5,
        "reference_selection": "explicit" if reference_frame is not None else "maximum_centerline_pixel_count_first_tie",
    })
    json.dumps(result, allow_nan=False)
    return result
