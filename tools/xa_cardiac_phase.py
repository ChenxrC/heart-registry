"""Image-derived motion phase utilities for research-only XA playback."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np


MIN_PERIOD_FRAMES = 3
MAX_PERIOD_FRAMES = 16
MIN_TRACKING_FRACTION = 0.50
MIN_PERIODICITY_SCORE = 0.80
MIN_PEAK_PROMINENCE = 0.05
SCALE_TOLERANCE_FACTOR = 64.0
PHASE_GROUPS = {
    "rca": ("series_01", "series_02"),
    "lca": ("series_03", "series_04", "series_05", "series_06"),
}
FEATURE_SCALING = {
    "columns": [
        "phase_sin",
        "phase_cos",
        "smoothed_motion",
        "motion_direction",
        "valid_fraction",
        "tracking_stability",
    ],
    "phase_sin_cos": "sin(2*pi*phase) and cos(2*pi*phase), each divided by sqrt(2)",
    "smoothed_motion": "median-centered, divided by 3*IQR, then clipped to [-1,1]",
    "motion_direction": "first difference of smoothed motion, median-centered, divided by 3*IQR, then clipped to [-1,1]",
    "valid_fraction": "unchanged image-derived fraction in [0,1]",
    "tracking_stability": "unchanged per-run image-derived score in [0,1]",
    "excluded": ["frame-index percentage", "raw cross-view pixel coordinates"],
}


def fill_short_false_gaps(visible: np.ndarray, *, max_gap: int) -> np.ndarray:
    """Fill false runs bounded by visible frames when their length is small."""
    filled = visible.copy()
    start = 0
    while start < len(filled):
        if filled[start]:
            start += 1
            continue
        end = start
        while end < len(filled) and not filled[end]:
            end += 1
        if start > 0 and end < len(filled) and end - start <= max_gap:
            filled[start:end] = True
        start = end
    return filled


def contiguous_true_intervals(visible: np.ndarray) -> list[tuple[int, int]]:
    """Return inclusive intervals for each true run."""
    intervals: list[tuple[int, int]] = []
    start: int | None = None
    for index, value in enumerate(visible):
        if value and start is None:
            start = index
        elif not value and start is not None:
            intervals.append((start, index - 1))
            start = None
    if start is not None:
        intervals.append((start, len(visible) - 1))
    return intervals


def select_visible_frame_range(
    counts: Sequence[int],
    reference_frame: int,
    threshold_fraction: float = 0.30,
    max_gap: int = 2,
) -> tuple[int, int]:
    """Select an inclusive contrast-visible interval from centerline counts."""
    values = np.asarray(counts, dtype=np.float64)
    if values.size == 0:
        raise ValueError("counts must not be empty")
    peak = float(values.max())
    if peak <= 0:
        return 0, len(values) - 1
    visible = values >= peak * threshold_fraction
    visible = fill_short_false_gaps(visible, max_gap=max_gap)
    intervals = contiguous_true_intervals(visible)
    containing = [item for item in intervals if item[0] <= reference_frame <= item[1]]
    chosen = max(
        containing or intervals,
        key=lambda item: (item[1] - item[0] + 1, values[item[0] : item[1] + 1].sum()),
    )
    return int(chosen[0]), int(chosen[1])


def centerline_counts_from_raw(path: Path, frame_count: int, rows: int, columns: int) -> list[int]:
    """Count nonzero centerline pixels in each uint8 frame of a raw stack."""
    values = np.fromfile(path, dtype=np.uint8)
    expected_size = frame_count * rows * columns
    if values.size != expected_size:
        raise ValueError(f"Expected {expected_size} centerline bytes, found {values.size}")
    frames = values.reshape(frame_count, rows, columns)
    return np.count_nonzero(frames, axis=(1, 2)).astype(int).tolist()


def _validate_visible_range(visible_range: tuple[int, int], frame_count: int) -> tuple[int, int]:
    if (
        not isinstance(visible_range, tuple)
        or len(visible_range) != 2
        or not all(isinstance(value, (int, np.integer)) for value in visible_range)
    ):
        raise ValueError("visible_range must be a two-integer inclusive tuple")
    start, end = (int(visible_range[0]), int(visible_range[1]))
    if start < 0 or end < start or end >= frame_count:
        raise ValueError("visible_range must be inclusive and within the frame bounds")
    return start, end


def _normalize_cine_frames(cine: np.ndarray) -> np.ndarray:
    normalized = np.empty(cine.shape, dtype=np.uint8)
    for index, frame in enumerate(cine):
        low, high = np.percentile(frame, (1.0, 99.0))
        if high <= low:
            normalized[index] = 0
            continue
        scaled = (frame - low) * (255.0 / (high - low))
        normalized[index] = np.clip(scaled, 0.0, 255.0).astype(np.uint8)
    return normalized


def _smoothed_finite_signal(values: np.ndarray, start: int, end: int) -> np.ndarray:
    smoothed = np.full(values.shape, np.nan, dtype=np.float64)
    for index in range(start, end + 1):
        window = values[max(start, index - 1) : min(end, index + 1) + 1]
        finite = window[np.isfinite(window)]
        if finite.size:
            smoothed[index] = float(np.median(finite))
    return smoothed


def coronary_motion_signal(
    cine: np.ndarray,
    centerlines: np.ndarray,
    visible_range: tuple[int, int],
) -> dict[str, Any]:
    """Measure research-only local coronary motion after global-flow removal."""
    cine_values = np.asarray(cine)
    centerline_values = np.asarray(centerlines)
    if cine_values.ndim != 3 or any(size <= 0 for size in cine_values.shape):
        raise ValueError("cine must have non-empty (frames, rows, columns) shape")
    if cine_values.shape[0] < 2:
        raise ValueError("cine must contain at least two frames")
    if centerline_values.shape != cine_values.shape:
        raise ValueError("centerlines must have the same shape as cine")
    if not np.issubdtype(cine_values.dtype, np.number) or not np.isfinite(cine_values).all():
        raise ValueError("cine must contain finite numeric values")
    centerline_is_valid_type = any(
        np.issubdtype(centerline_values.dtype, valid_type)
        for valid_type in (np.bool_, np.integer, np.floating)
    )
    if (
        not centerline_is_valid_type
        or not np.isfinite(centerline_values).all()
        or np.any(centerline_values < 0)
    ):
        raise ValueError("centerlines must contain finite nonnegative numeric or boolean values")
    start, end = _validate_visible_range(visible_range, cine_values.shape[0])

    frame_count = cine_values.shape[0]
    raw_signal = np.full(frame_count, np.nan, dtype=np.float64)
    valid_fractions = np.zeros(frame_count, dtype=np.float64)
    normalized = _normalize_cine_frames(cine_values.astype(np.float32, copy=False))
    vessel_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    exclusion_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    stable_background_unavailable = False

    for index in range(max(start + 1, 1), end + 1):
        flow = cv2.calcOpticalFlowFarneback(
            normalized[index - 1],
            normalized[index],
            None,
            0.5,
            3,
            15,
            3,
            5,
            1.2,
            0,
        )
        centerline_pair = np.logical_or(centerline_values[index - 1] > 0, centerline_values[index] > 0)
        vessel_mask = cv2.dilate(centerline_pair.astype(np.uint8), vessel_kernel) > 0
        vessel_pixels = int(np.count_nonzero(vessel_mask))
        if vessel_pixels == 0:
            continue

        finite_flow = np.isfinite(flow).all(axis=2)
        non_vessel = np.logical_and(
            cv2.dilate(centerline_pair.astype(np.uint8), exclusion_kernel) == 0,
            finite_flow,
        )
        if not np.any(non_vessel):
            stable_background_unavailable = True
            continue

        background_flow = flow[non_vessel]
        initial_global = np.median(background_flow, axis=0)
        background_deviation = np.linalg.norm(background_flow - initial_global, axis=1)
        stable_cutoff = float(np.percentile(background_deviation, 75.0))
        stable_background = background_flow[background_deviation <= stable_cutoff]
        if stable_background.size == 0:
            stable_background_unavailable = True
            continue
        global_flow = np.median(stable_background, axis=0)

        valid_vessel = np.logical_and(vessel_mask, finite_flow)
        valid_count = int(np.count_nonzero(valid_vessel))
        valid_fractions[index] = valid_count / vessel_pixels
        if valid_count == 0:
            continue
        residual = flow[valid_vessel] - global_flow
        raw_signal[index] = float(np.median(np.linalg.norm(residual, axis=1)))

    if start < end and np.isfinite(raw_signal[start + 1]):
        raw_signal[start] = raw_signal[start + 1]
        valid_fractions[start] = valid_fractions[start + 1]

    smoothed_signal = _smoothed_finite_signal(raw_signal, start, end)
    valid_motion = np.isfinite(raw_signal[start : end + 1])
    if not np.any(valid_motion):
        return {
            "status": "insufficient",
            "reason": (
                "stable-background-unavailable"
                if stable_background_unavailable
                else "no-valid-centerline-motion"
            ),
            "raw_signal": raw_signal.tolist(),
            "smoothed_signal": smoothed_signal.tolist(),
            "valid_fractions": valid_fractions.tolist(),
            "tracking_stability": 0.0,
        }

    tracked = valid_fractions[start : end + 1][valid_motion]
    tracking_stability = float(np.clip(np.median(tracked), 0.0, 1.0))
    return {
        "status": "ok",
        "raw_signal": raw_signal.tolist(),
        "smoothed_signal": smoothed_signal.tolist(),
        "valid_fractions": valid_fractions.tolist(),
        "tracking_stability": tracking_stability,
    }


def _insufficient_period(reason: str, periodicity: float = 0.0) -> dict[str, float | int | str]:
    return {
        "status": "insufficient",
        "reason": reason,
        "periodicity": float(np.clip(periodicity, 0.0, 1.0)),
        "observable_cycles": 0.0,
    }


def _normalized_lag_correlation(signal: np.ndarray, lag: int) -> float | None:
    first = signal[:-lag]
    second = signal[lag:]
    first = first - np.mean(first)
    second = second - np.mean(second)
    first_norm = float(np.linalg.norm(first))
    second_norm = float(np.linalg.norm(second))
    segment_scale = max(
        float(np.max(np.abs(first), initial=0.0)),
        float(np.max(np.abs(second), initial=0.0)),
        1.0,
    )
    norm_tolerance = (
        SCALE_TOLERANCE_FACTOR
        * np.finfo(np.float64).eps
        * segment_scale
        * np.sqrt(max(first.size, second.size))
    )
    if first_norm <= norm_tolerance or second_norm <= norm_tolerance:
        return None
    return float(np.clip(np.dot(first, second) / (first_norm * second_norm), -1.0, 1.0))


def estimate_period(
    signal: Sequence[float],
    min_period: int = MIN_PERIOD_FRAMES,
    max_period: int = MAX_PERIOD_FRAMES,
) -> dict[str, float | int | str]:
    """Estimate an image-motion period from bounded autocorrelation peaks."""
    if (
        not isinstance(min_period, (int, np.integer))
        or not isinstance(max_period, (int, np.integer))
        or min_period < MIN_PERIOD_FRAMES
        or max_period > MAX_PERIOD_FRAMES
        or min_period > max_period
    ):
        raise ValueError("period bounds must satisfy 3 <= min_period <= max_period <= 16")
    values = np.asarray(signal, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError("signal must be one-dimensional")
    if values.size == 0 or not np.isfinite(values).all():
        return _insufficient_period("signal-must-be-finite")

    eligible_max = min(int(max_period), values.size // 2)
    if eligible_max < int(min_period):
        return _insufficient_period("fewer-than-two-observable-periods")

    raw_scale = max(float(np.max(np.abs(values))), 1.0)
    scale_tolerance = SCALE_TOLERANCE_FACTOR * np.finfo(np.float64).eps * raw_scale
    raw_span = float(np.max(values) - np.min(values))
    raw_robust_scale = float(np.percentile(values, 75.0) - np.percentile(values, 25.0))
    if raw_span <= scale_tolerance or raw_robust_scale <= scale_tolerance:
        return _insufficient_period("signal-has-no-motion-variation")

    frame_index = np.arange(values.size, dtype=np.float64)
    frame_index -= np.mean(frame_index)
    centered = values - np.mean(values)
    slope = float(np.dot(frame_index, centered) / np.dot(frame_index, frame_index))
    detrended = centered - slope * frame_index
    detrended_span = float(np.max(detrended) - np.min(detrended))
    detrended_deviation = float(np.std(detrended))
    if detrended_span <= scale_tolerance or detrended_deviation <= scale_tolerance:
        return _insufficient_period("signal-has-no-detrended-variation")

    correlation: dict[int, float | None] = {}
    neighbor_min = max(1, int(min_period) - 1)
    neighbor_max = min(int(max_period) + 1, values.size - 2)
    for lag in range(neighbor_min, neighbor_max + 1):
        correlation[lag] = _normalized_lag_correlation(detrended, lag)

    local_peaks: list[tuple[int, float, float]] = []
    for lag in range(int(min_period), eligible_max + 1):
        score = correlation.get(lag)
        left = correlation.get(lag - 1)
        right = correlation.get(lag + 1)
        if score is None or left is None or right is None:
            continue
        prominence = score - max(left, right)
        if (
            score >= MIN_PERIODICITY_SCORE
            and score > left
            and score > right
            and prominence >= MIN_PEAK_PROMINENCE
        ):
            local_peaks.append((lag, score, prominence))
    if not local_peaks:
        return _insufficient_period("no-autocorrelation-peak")

    strongest_score = max(item[1] for item in local_peaks)
    equally_strong = [item for item in local_peaks if strongest_score - item[1] <= 1.0e-9]
    period, periodicity, _prominence = min(equally_strong, key=lambda item: item[0])
    observable_cycles = float(values.size / period)
    return {
        "status": "ok",
        "period_frames": float(period),
        "periodicity": periodicity,
        "observable_cycles": observable_cycles,
    }


def phase_from_anchor(frame_count: int, period_frames: float, anchor_frame: int) -> list[float]:
    """Return wrapped research-only phase with zero at the image-derived anchor."""
    if not isinstance(frame_count, (int, np.integer)) or frame_count <= 0:
        raise ValueError("frame_count must be a positive integer")
    if not np.isfinite(period_frames) or period_frames <= 0:
        raise ValueError("period_frames must be finite and positive")
    if not isinstance(anchor_frame, (int, np.integer)) or not 0 <= anchor_frame < frame_count:
        raise ValueError("anchor_frame must be within frame bounds")
    phase = np.mod(np.arange(frame_count, dtype=np.float64) - anchor_frame, period_frames) / period_frames
    return phase.tolist()


def phase_confidence(
    periodicity: float,
    visibility: float,
    tracking_stability: float,
    observable_cycles: float,
) -> float:
    """Calculate the mandated bounded image-derived phase confidence score."""
    inputs = (periodicity, visibility, tracking_stability, observable_cycles)
    if not all(np.isfinite(value) for value in inputs):
        raise ValueError("confidence inputs must be finite")
    if not all(0.0 <= value <= 1.0 for value in inputs[:3]):
        raise ValueError("periodicity, visibility, and tracking_stability must be within [0, 1]")
    if observable_cycles < 0.0:
        raise ValueError("observable_cycles must be nonnegative")
    return float(
        np.clip(
            0.40 * periodicity
            + 0.25 * visibility
            + 0.20 * tracking_stability
            + 0.15 * min(observable_cycles / 3.0, 1.0),
            0.0,
            1.0,
        )
    )


def confidence_status(confidence: float) -> str:
    """Map a bounded confidence score to the global decision thresholds."""
    if not np.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
        raise ValueError("confidence must be finite and within [0, 1]")
    if confidence < 0.50:
        return "insufficient"
    if confidence < 0.80:
        return "provisional"
    return "automatic"


def estimate_anchor_frame(
    smoothed_signal: Sequence[float],
    visible_range: tuple[int, int],
    reference_frame: int,
    period_frames: float,
    valid_fractions: Sequence[float] | None = None,
) -> int | None:
    """Choose the lowest adequately tracked motion frame near a reference."""
    values = np.asarray(smoothed_signal, dtype=np.float64)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("smoothed_signal must be a non-empty one-dimensional sequence")
    start, end = _validate_visible_range(visible_range, values.size)
    if not isinstance(reference_frame, (int, np.integer)) or not 0 <= reference_frame < values.size:
        raise ValueError("reference_frame must be within signal bounds")
    if not np.isfinite(period_frames) or period_frames <= 0:
        raise ValueError("period_frames must be finite and positive")

    if valid_fractions is None:
        tracking = np.ones(values.size, dtype=np.float64)
    else:
        tracking = np.asarray(valid_fractions, dtype=np.float64)
        if tracking.ndim != 1 or tracking.shape != values.shape:
            raise ValueError("valid_fractions must match smoothed_signal shape")
        if not np.isfinite(tracking).all() or np.any((tracking < 0.0) | (tracking > 1.0)):
            raise ValueError("valid_fractions must be finite and within [0, 1]")

    half_period = period_frames / 2.0
    candidate_start = max(start, int(np.ceil(reference_frame - half_period)))
    candidate_end = min(end, int(np.floor(reference_frame + half_period)))
    candidates = [
        index
        for index in range(candidate_start, candidate_end + 1)
        if np.isfinite(values[index])
        and np.isfinite(tracking[index])
        and tracking[index] >= MIN_TRACKING_FRACTION
    ]
    if not candidates:
        return None
    return min(candidates, key=lambda index: (values[index], abs(index - reference_frame), index))


def partial_dtw(
    reference_features: np.ndarray,
    target_features: np.ndarray,
    skip_penalty: float = 0.25,
) -> list[tuple[int, int]]:
    """Return monotone matched local indices using Euclidean cost and skips."""
    reference = np.asarray(reference_features, dtype=np.float64)
    target = np.asarray(target_features, dtype=np.float64)
    if reference.ndim != 2 or target.ndim != 2 or not reference.size or not target.size:
        raise ValueError("feature arrays must be non-empty and two-dimensional")
    if reference.shape[1] != target.shape[1]:
        raise ValueError("feature arrays must have the same column count")
    if not np.isfinite(reference).all() or not np.isfinite(target).all():
        raise ValueError("feature arrays must contain finite values")
    if not np.isfinite(skip_penalty) or skip_penalty <= 0.0:
        raise ValueError("skip_penalty must be finite and positive")

    reference_count, target_count = reference.shape[0], target.shape[0]
    costs = np.full((reference_count + 1, target_count + 1), np.inf, dtype=np.float64)
    transitions = np.zeros((reference_count + 1, target_count + 1), dtype=np.uint8)
    costs[0, 0] = 0.0
    for reference_index in range(1, reference_count + 1):
        costs[reference_index, 0] = reference_index * skip_penalty
        transitions[reference_index, 0] = 2
    for target_index in range(1, target_count + 1):
        costs[0, target_index] = target_index * skip_penalty
        transitions[0, target_index] = 3

    for reference_index in range(1, reference_count + 1):
        for target_index in range(1, target_count + 1):
            match_cost = float(
                np.linalg.norm(reference[reference_index - 1] - target[target_index - 1])
            )
            candidates = (
                costs[reference_index - 1, target_index - 1] + match_cost,
                costs[reference_index - 1, target_index] + skip_penalty,
                costs[reference_index, target_index - 1] + skip_penalty,
            )
            transition = int(np.argmin(candidates)) + 1
            costs[reference_index, target_index] = candidates[transition - 1]
            transitions[reference_index, target_index] = transition

    pairs: list[tuple[int, int]] = []
    reference_index, target_index = reference_count, target_count
    while reference_index > 0 or target_index > 0:
        transition = int(transitions[reference_index, target_index])
        if transition == 1:
            reference_index -= 1
            target_index -= 1
            pairs.append((reference_index, target_index))
        elif transition == 2:
            reference_index -= 1
        elif transition == 3:
            target_index -= 1
        else:
            raise RuntimeError("DTW backtracking encountered an invalid transition")
    pairs.reverse()
    return pairs


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _input_record(path: Path, series_root: Path) -> dict[str, Any]:
    return {
        "path": path.relative_to(series_root).as_posix(),
        "sha256": _sha256(path),
        "byte_length": path.stat().st_size,
    }


def _empty_run(reason: str) -> dict[str, Any]:
    return {
        "visible_frame_range": None,
        "period_frames": None,
        "anchor_frame": None,
        "phase_per_frame": [],
        "confidence": 0.0,
        "status": "insufficient",
        "clinical_status": "research-only",
        "inputs": {},
        "reason": reason,
    }


def _load_u8_stack(path: Path, frame_count: int, rows: int, columns: int) -> np.ndarray:
    values = np.fromfile(path, dtype=np.uint8)
    expected = frame_count * rows * columns
    if values.size != expected:
        raise ValueError(f"Expected {expected} bytes in {path.name}, found {values.size}")
    return values.reshape(frame_count, rows, columns)


def _scaled_feature(values: np.ndarray) -> np.ndarray:
    median = float(np.median(values))
    scale = float(np.percentile(values, 75.0) - np.percentile(values, 25.0))
    tolerance = SCALE_TOLERANCE_FACTOR * np.finfo(np.float64).eps * max(
        float(np.max(np.abs(values), initial=0.0)), 1.0
    )
    if scale <= tolerance:
        return np.zeros(values.shape, dtype=np.float64)
    return np.clip((values - median) / (3.0 * scale), -1.0, 1.0)


def _phase_feature_rows(
    phase_values: Sequence[float],
    smoothed_signal: Sequence[float],
    valid_fractions: Sequence[float],
    tracking_stability: float,
    visible_range: tuple[int, int],
) -> np.ndarray:
    start, end = visible_range
    phase = np.asarray(phase_values[start : end + 1], dtype=np.float64)
    motion = np.asarray(smoothed_signal[start : end + 1], dtype=np.float64)
    valid = np.asarray(valid_fractions[start : end + 1], dtype=np.float64)
    direction = np.diff(motion, prepend=motion[0])
    return np.column_stack(
        (
            np.sin(2.0 * np.pi * phase) / np.sqrt(2.0),
            np.cos(2.0 * np.pi * phase) / np.sqrt(2.0),
            _scaled_feature(motion),
            _scaled_feature(direction),
            valid,
            np.full(phase.shape, tracking_stability, dtype=np.float64),
        )
    )


def _analyze_phase_run(
    series_root: Path,
    run_id: str,
) -> tuple[dict[str, Any], np.ndarray | None]:
    inputs: dict[str, Any] = {}
    try:
        series_dir = series_root / run_id
        if not series_dir.is_dir():
            return _empty_run("missing-run-directory"), None

        manifest_path = series_dir / "series_manifest.json"
        cine_path = series_dir / "cine_u8.raw"
        centerline_path = series_dir / "temporal" / "temporal_centerlines_u8.raw"
        dynamic_path = series_dir / "dynamic_3d" / "dynamic_vessel_3d.json"
        paths = {
            "manifest": manifest_path,
            "cine": cine_path,
            "temporal_centerlines": centerline_path,
            "dynamic_vessel_3d": dynamic_path,
        }
        missing: list[str] = []
        for name, path in paths.items():
            if not path.is_file():
                missing.append(name)
                continue
            inputs[name] = _input_record(path, series_root)
        if missing:
            result = _empty_run("missing-run-input")
            result["inputs"] = inputs
            result["error"] = f"Missing required inputs: {', '.join(missing)}"
            return result, None

        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        dynamic = json.loads(dynamic_path.read_text(encoding="utf-8"))
        if not isinstance(manifest, dict) or not isinstance(dynamic, dict):
            raise ValueError("manifest and dynamic vessel inputs must be JSON objects")
        source_record = manifest["source_record"]
        frame_count = int(manifest["frame_png_count"])
        rows = int(source_record["rows"])
        columns = int(source_record["columns"])
        if frame_count <= 0 or rows <= 0 or columns <= 0:
            raise ValueError("frame count, rows, and columns must be positive")
        reference_frame = int(dynamic["reference_frame"])
        if not 0 <= reference_frame < frame_count:
            raise ValueError("reference_frame is outside cine frame bounds")

        cine = _load_u8_stack(cine_path, frame_count, rows, columns)
        centerlines = _load_u8_stack(centerline_path, frame_count, rows, columns)
        counts = np.count_nonzero(centerlines, axis=(1, 2)).astype(int).tolist()
        visible_range = select_visible_frame_range(counts, reference_frame)
        motion = coronary_motion_signal(cine, centerlines, visible_range)
        visibility = float((visible_range[1] - visible_range[0] + 1) / frame_count)
        tracking_stability = float(motion["tracking_stability"])
        run: dict[str, Any] = {
            "visible_frame_range": list(visible_range),
            "period_frames": None,
            "anchor_frame": None,
            "phase_per_frame": [],
            "confidence": 0.0,
            "status": "insufficient",
            "clinical_status": "research-only",
            "inputs": inputs,
            "metrics": {
                "periodicity": 0.0,
                "visibility": visibility,
                "tracking_stability": tracking_stability,
                "observable_cycles": 0.0,
            },
        }
        source_hash = source_record.get("sha256")
        if source_hash:
            run["inputs"]["source_dicom"] = {
                "path": str(manifest.get("source_dicom", source_record.get("path", ""))),
                "sha256": str(source_hash),
            }
        if motion["status"] != "ok":
            run["reason"] = str(motion.get("reason", "insufficient-motion-signal"))
            run["confidence"] = phase_confidence(0.0, visibility, tracking_stability, 0.0)
            return run, None

        smoothed = np.asarray(motion["smoothed_signal"], dtype=np.float64)
        start, end = visible_range
        period = estimate_period(smoothed[start : end + 1])
        periodicity = float(period.get("periodicity", 0.0))
        observable_cycles = float(period.get("observable_cycles", 0.0))
        confidence = phase_confidence(
            periodicity,
            visibility,
            tracking_stability,
            observable_cycles,
        )
        run["confidence"] = confidence
        run["metrics"].update(
            {"periodicity": periodicity, "observable_cycles": observable_cycles}
        )
        if period["status"] != "ok":
            run["reason"] = str(period.get("reason", "insufficient-period"))
            return run, None

        period_frames = float(period["period_frames"])
        anchor = estimate_anchor_frame(
            smoothed,
            visible_range,
            reference_frame,
            period_frames,
            motion["valid_fractions"],
        )
        if anchor is None:
            run["reason"] = "no-adequately-tracked-anchor"
            return run, None

        phase_values = phase_from_anchor(frame_count, period_frames, anchor)
        run.update(
            {
                "period_frames": period_frames,
                "anchor_frame": anchor,
                "phase_per_frame": phase_values,
                "status": confidence_status(confidence),
            }
        )
        if run["status"] == "insufficient":
            run["reason"] = "confidence-below-usable-threshold"
            return run, None
        features = _phase_feature_rows(
            phase_values,
            motion["smoothed_signal"],
            motion["valid_fractions"],
            tracking_stability,
            visible_range,
        )
        return run, features
    except Exception as error:
        result = _empty_run("invalid-run-input")
        result["inputs"] = inputs
        result["error"] = f"{type(error).__name__}: {error}"
        return result, None


def build_phase_alignment(series_root: Path) -> dict[str, Any]:
    """Build a six-run research report while isolating malformed run inputs."""
    root = Path(series_root)
    if not root.is_dir():
        raise ValueError(f"Invalid series root: {root}")

    runs: dict[str, dict[str, Any]] = {}
    features: dict[str, np.ndarray] = {}
    for run_ids in PHASE_GROUPS.values():
        for run_id in run_ids:
            run, feature_rows = _analyze_phase_run(root, run_id)
            runs[run_id] = run
            if feature_rows is not None:
                features[run_id] = feature_rows

    groups: dict[str, dict[str, Any]] = {}
    correspondence: dict[str, dict[str, Any]] = {}
    for group_name, run_ids in PHASE_GROUPS.items():
        usable = [
            run_id
            for run_id in run_ids
            if runs[run_id]["period_frames"] is not None
            and runs[run_id]["status"] in {"automatic", "provisional"}
        ]
        group = {
            "run_ids": list(run_ids),
            "reference_run_id": None,
            "status": "insufficient",
        }
        if len(usable) >= 2:
            reference_run_id = max(usable, key=lambda run_id: runs[run_id]["confidence"])
            candidate_status = (
                "automatic"
                if all(runs[run_id]["status"] == "automatic" for run_id in usable)
                else "provisional"
            )
            reference_start = int(runs[reference_run_id]["visible_frame_range"][0])
            candidate_correspondence: dict[str, dict[str, Any]] = {}
            has_empty_correspondence = False
            for target_run_id in usable:
                if target_run_id == reference_run_id:
                    continue
                target_start = int(runs[target_run_id]["visible_frame_range"][0])
                local_pairs = partial_dtw(features[reference_run_id], features[target_run_id])
                if not local_pairs:
                    has_empty_correspondence = True
                    break
                key = f"{reference_run_id}_to_{target_run_id}"
                candidate_correspondence[key] = {
                    "group": group_name,
                    "reference_run_id": reference_run_id,
                    "target_run_id": target_run_id,
                    "pairs": [
                        [reference_start + reference_index, target_start + target_index]
                        for reference_index, target_index in local_pairs
                    ],
                }
            if has_empty_correspondence:
                group["reason"] = "no-credible-frame-correspondence"
            else:
                group["reference_run_id"] = reference_run_id
                group["status"] = candidate_status
                correspondence.update(candidate_correspondence)
        groups[group_name] = group

    usable_group_count = sum(group["status"] != "insufficient" for group in groups.values())
    top_status = ("insufficient", "partial", "ready")[usable_group_count]
    return {
        "schema_version": 1,
        "method": "image-derived coronary motion phase",
        "clinical_status": "research-only",
        "status": top_status,
        "feature_scaling": FEATURE_SCALING,
        "groups": groups,
        "runs": runs,
        "frame_correspondence": correspondence,
    }
