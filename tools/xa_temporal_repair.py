"""Motion-compensated temporal evidence for diagnostic XA processing."""

from __future__ import annotations

from dataclasses import dataclass, replace

import cv2
import numpy as np
from scipy import ndimage
from skimage import filters, morphology

try:
    from .xa_vessel_topology import (
        GapCandidate,
        VesselEdge,
        VesselGraph,
        VesselNode,
        _endpoint_pair_alignment_is_within_45_degrees,
        _endpoint_pair_geometry,
        _validated_graph,
        find_gap_candidates,
    )
except ImportError:  # direct script execution
    from xa_vessel_topology import (
        GapCandidate,
        VesselEdge,
        VesselGraph,
        VesselNode,
        _endpoint_pair_alignment_is_within_45_degrees,
        _endpoint_pair_geometry,
        _validated_graph,
        find_gap_candidates,
    )


__all__ = [
    "FlowEvidence",
    "GapCandidate",
    "RepairAudit",
    "bidirectional_flow_evidence",
    "build_vessel_evidence",
    "constrained_seed_reconstruction",
    "find_gap_candidates",
    "repair_short_gaps",
    "temporal_support",
]


@dataclass(frozen=True)
class FlowEvidence:
    warped_probability: np.ndarray
    warped_mask: np.ndarray
    confidence: np.ndarray
    forward_backward_error_px: np.ndarray


@dataclass(frozen=True)
class RepairAudit:
    accepted_pixels: int
    rejected_pixels: int
    area_growth_fraction: float
    status: str
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class _TemporalFusionEvidence:
    fused_probability: np.ndarray
    support_probability: np.ndarray
    valid_neighbor_count: np.ndarray
    valid_neighbor_frame_count: np.ndarray
    flow_confidence: np.ndarray
    forward_backward_error_p95_px: np.ndarray


CURRENT_PROBABILITY_WEIGHT = 0.60
NEIGHBOR_PROBABILITY_WEIGHT = 0.20


def _normalize_probability(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    low, high = np.percentile(values, (1.0, 99.0))
    if high <= low + 1e-6:
        return np.zeros_like(values)
    return np.clip((values - low) / (high - low), 0.0, 1.0).astype(
        np.float32, copy=False
    )


def _frangi_probability(frame: np.ndarray) -> np.ndarray:
    normalized = np.asarray(frame, dtype=np.float32) / 255.0
    vesselness = filters.frangi(
        normalized,
        sigmas=range(1, 5),
        black_ridges=True,
    )
    return _normalize_probability(vesselness)


def _normalized_dark_vessel_response(frame: np.ndarray) -> np.ndarray:
    dark_response = 1.0 - np.asarray(frame, dtype=np.float32) / 255.0
    return _normalize_probability(dark_response)


def build_vessel_evidence(
    frame: np.ndarray,
    model_probability: np.ndarray,
    temporal_probability: np.ndarray,
    temporal_support: np.ndarray,
) -> np.ndarray:
    """Combine model, image-derived, and aligned temporal vessel evidence."""
    frangi_probability = _frangi_probability(frame)
    evidence = (
        0.45 * np.asarray(model_probability, dtype=np.float32)
        + 0.20 * frangi_probability
        + 0.15 * _normalized_dark_vessel_response(frame)
        + 0.20
        * np.asarray(temporal_probability, dtype=np.float32)
        * np.asarray(temporal_support, dtype=np.float32)
    )
    return np.clip(evidence, 0.0, 1.0).astype(np.float32, copy=False)


def constrained_seed_reconstruction(
    evidence: np.ndarray,
    seed_mask: np.ndarray,
    exclusion_mask: np.ndarray,
    max_growth_fraction: float = 0.20,
) -> tuple[np.ndarray, RepairAudit]:
    """Grow seed-connected evidence and report the accepted expansion."""
    seed = np.asarray(seed_mask, dtype=bool).copy()
    exclusion = np.asarray(exclusion_mask, dtype=bool)
    if np.any(seed & exclusion):
        return seed, RepairAudit(
            accepted_pixels=0,
            rejected_pixels=0,
            area_growth_fraction=0.0,
            status="rejected",
            reasons=("exclusion_intersection",),
        )
    candidate = np.asarray(evidence, dtype=np.float32) >= 0.20
    candidate &= ~exclusion
    candidate |= seed
    reconstructed = morphology.reconstruction(
        seed.astype(np.uint8),
        candidate.astype(np.uint8),
        method="dilation",
    ).astype(bool)
    accepted_pixels = int(np.count_nonzero(reconstructed & ~seed))
    seed_pixels = int(np.count_nonzero(seed))
    area_growth_fraction = (
        float(accepted_pixels / seed_pixels) if seed_pixels else 0.0
    )
    border = np.zeros(seed.shape, dtype=bool)
    border[0, :] = True
    border[-1, :] = True
    border[:, 0] = True
    border[:, -1] = True
    reasons = []
    if area_growth_fraction > max_growth_fraction:
        reasons.append("growth_cap_exceeded")
    if np.any(reconstructed & border & ~seed):
        reasons.append("new_border_contact")
    if np.any(reconstructed & exclusion):
        reasons.append("exclusion_intersection")
    if reasons:
        return seed, RepairAudit(
            accepted_pixels=0,
            rejected_pixels=accepted_pixels,
            area_growth_fraction=area_growth_fraction,
            status="rejected",
            reasons=tuple(reasons),
        )
    return reconstructed, RepairAudit(
        accepted_pixels=accepted_pixels,
        rejected_pixels=0,
        area_growth_fraction=area_growth_fraction,
        status="accepted",
        reasons=(),
    )


def _rejected_gap(candidate: GapCandidate, reason: str) -> GapCandidate:
    reasons = tuple(dict.fromkeys((*candidate.reasons, reason)))
    return replace(candidate, decision="rejected", reasons=reasons)


def _malformed_gap(candidate: GapCandidate) -> GapCandidate:
    return replace(
        candidate,
        decision="rejected",
        reasons=("malformed_candidate",),
    )


def _foreground_hole_count(mask: np.ndarray) -> int:
    """Count holes using 4-connected background for 8-connected foreground."""
    background = ~np.asarray(mask, dtype=bool)
    background_labels, label_count = ndimage.label(
        background,
        structure=np.asarray(
            ((0, 1, 0), (1, 1, 1), (0, 1, 0)), dtype=np.uint8
        ),
    )
    border_labels = set(int(value) for value in background_labels[0, :])
    border_labels.update(int(value) for value in background_labels[-1, :])
    border_labels.update(int(value) for value in background_labels[:, 0])
    border_labels.update(int(value) for value in background_labels[:, -1])
    return sum(
        label_index not in border_labels
        for label_index in range(1, int(label_count) + 1)
    )


def _is_finite_real(value: object) -> bool:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        return False
    try:
        converted = float(value)
        return bool(np.isfinite(converted))
    except (TypeError, ValueError, OverflowError):
        return False


def _validate_candidate(
    candidate: GapCandidate,
    node_by_id: dict[str, VesselNode],
    incident_edges: dict[str, list[VesselEdge]],
    detector_shape: tuple[int, int],
) -> tuple[bool, np.ndarray | None, float | None, float | None]:
    numeric_values = (
        candidate.distance_px,
        candidate.angle_deg,
        candidate.mean_evidence,
        candidate.temporal_support,
        candidate.score,
    )
    if not all(_is_finite_real(value) for value in numeric_values):
        return False, None, None, None
    if not (
        0.0 < float(candidate.distance_px) <= 24.0
        and 0.0 <= float(candidate.angle_deg) <= 180.0
        and 0.0 <= float(candidate.mean_evidence) <= 1.0
        and 0.0 <= float(candidate.temporal_support) <= 1.0
        and float(candidate.score) >= 0.0
    ):
        return False, None, None, None
    if not isinstance(candidate.decision, str) or candidate.decision not in (
        "accepted",
        "review",
        "rejected",
    ):
        return False, None, None, None
    if (
        not isinstance(candidate.reasons, tuple)
        or not all(isinstance(reason, str) and reason for reason in candidate.reasons)
        or len(candidate.reasons) != len(set(candidate.reasons))
        or (candidate.decision == "accepted" and candidate.reasons)
    ):
        return False, None, None, None
    if (
        not isinstance(candidate.start_node_id, str)
        or not isinstance(candidate.end_node_id, str)
        or candidate.start_node_id == candidate.end_node_id
        or candidate.start_node_id not in node_by_id
        or candidate.end_node_id not in node_by_id
    ):
        return False, None, None, None

    start_node = node_by_id[candidate.start_node_id]
    end_node = node_by_id[candidate.end_node_id]
    start_coordinate = np.asarray(start_node.yx, dtype=np.float64)
    end_coordinate = np.asarray(end_node.yx, dtype=np.float64)
    actual_distance, actual_angle, _, _ = _endpoint_pair_geometry(
        candidate.start_node_id,
        candidate.end_node_id,
        node_by_id,
        incident_edges,
    )
    if not np.isclose(
        float(candidate.distance_px), actual_distance, rtol=0.0, atol=1e-12
    ) or not np.isclose(
        float(candidate.angle_deg), actual_angle, rtol=0.0, atol=1e-12
    ):
        return False, None, None, None

    try:
        path = np.asarray(candidate.path_yx)
        numeric_path = np.issubdtype(path.dtype, np.integer) or np.issubdtype(
            path.dtype, np.floating
        )
    except (TypeError, ValueError):
        return False, None, None, None
    if (
        path.ndim != 2
        or path.shape[1:] != (2,)
        or len(path) < 2
        or not numeric_path
        or np.issubdtype(path.dtype, np.bool_)
        or not np.all(np.isfinite(path))
    ):
        return False, None, None, None
    path_indices = np.rint(path).astype(np.int64)
    if not np.array_equal(path, path_indices):
        return False, None, None, None
    if (
        np.any(path_indices[:, 0] < 0)
        or np.any(path_indices[:, 0] >= detector_shape[0])
        or np.any(path_indices[:, 1] < 0)
        or np.any(path_indices[:, 1] >= detector_shape[1])
    ):
        return False, None, None, None
    start_pixel = np.rint(start_coordinate).astype(np.int64)
    end_pixel = np.rint(end_coordinate).astype(np.int64)
    if not (
        np.array_equal(path_indices[0], start_pixel)
        and np.array_equal(path_indices[-1], end_pixel)
    ):
        return False, None, None, None
    padding = 8
    minimum = np.maximum(
        np.minimum(start_pixel, end_pixel) - padding,
        np.asarray((0, 0), dtype=np.int64),
    )
    maximum = np.minimum(
        np.maximum(start_pixel, end_pixel) + padding,
        np.asarray(detector_shape, dtype=np.int64) - 1,
    )
    if np.any(path_indices < minimum) or np.any(path_indices > maximum):
        return False, None, None, None
    path_steps = np.max(np.abs(np.diff(path_indices, axis=0)), axis=1)
    if not np.all(path_steps == 1):
        return False, None, None, None
    return True, path_indices, actual_distance, actual_angle


def _candidate_sort_key(
    candidate: GapCandidate,
    path_indices: np.ndarray,
) -> tuple[object, ...]:
    pair = tuple(sorted((candidate.start_node_id, candidate.end_node_id)))
    oriented_path = (
        path_indices
        if candidate.start_node_id == pair[0]
        else path_indices[::-1]
    )
    path_key = tuple(tuple(int(value) for value in point) for point in oriented_path)
    decision_rank = {"accepted": 0, "review": 1, "rejected": 2}[
        candidate.decision
    ]
    return (
        pair,
        decision_rank,
        float(candidate.score),
        float(candidate.distance_px),
        float(candidate.angle_deg),
        -float(candidate.mean_evidence),
        -float(candidate.temporal_support),
        path_key,
        candidate.start_node_id,
        candidate.end_node_id,
        candidate.reasons,
    )


def repair_short_gaps(
    mask: np.ndarray,
    graph: VesselGraph,
    candidates: list[GapCandidate],
    auto_gap_px: int = 12,
    *,
    exclusion_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, list[GapCandidate]]:
    """Apply accepted gap candidates with radius and topology rollback gates."""
    source_mask = np.asarray(mask)
    if source_mask.ndim != 2:
        raise ValueError(f"mask must be 2D, got shape {source_mask.shape}")
    current = source_mask.astype(bool, copy=True)
    if (
        isinstance(auto_gap_px, (bool, np.bool_))
        or not isinstance(auto_gap_px, (int, np.integer))
        or not 1 <= int(auto_gap_px) <= 12
    ):
        raise ValueError("auto_gap_px must be an integer from 1 through 12")

    if exclusion_mask is None:
        exclusion = np.zeros(current.shape, dtype=bool)
    else:
        exclusion_array = np.asarray(exclusion_mask)
        if exclusion_array.ndim != 2 or exclusion_array.shape != current.shape:
            raise ValueError("exclusion_mask must have the same 2D shape as mask")
        exclusion = exclusion_array.astype(bool, copy=False)

    node_by_id, incident_edges = _validated_graph(graph, current.shape)
    source_candidates = tuple(candidates)
    if not all(isinstance(candidate, GapCandidate) for candidate in source_candidates):
        raise ValueError("candidates must contain only GapCandidate records")
    validated_candidates: list[
        tuple[
            tuple[object, ...],
            GapCandidate,
            bool,
            np.ndarray | None,
            float | None,
            float | None,
        ]
    ] = []
    for candidate_index, candidate in enumerate(source_candidates):
        try:
            valid, path_indices, actual_distance, actual_angle = _validate_candidate(
                candidate,
                node_by_id,
                incident_edges,
                current.shape,
            )
        except Exception:
            valid, path_indices, actual_distance, actual_angle = (
                False,
                None,
                None,
                None,
            )
        if valid:
            assert path_indices is not None
            sort_key = (0, *_candidate_sort_key(candidate, path_indices))
        else:
            sort_key = (1, candidate_index)
        validated_candidates.append(
            (
                sort_key,
                candidate,
                valid,
                path_indices,
                actual_distance,
                actual_angle,
            )
        )

    used_endpoints: set[str] = set()
    seen_pairs: set[tuple[str, str]] = set()
    audited: list[GapCandidate] = []
    for (
        _sort_key,
        candidate,
        valid,
        path_indices,
        actual_distance,
        actual_angle,
    ) in sorted(
        validated_candidates, key=lambda item: item[0]
    ):
        if not valid:
            audited.append(_malformed_gap(candidate))
            continue
        assert path_indices is not None
        assert actual_distance is not None
        assert actual_angle is not None
        endpoint_pair = tuple(
            sorted((candidate.start_node_id, candidate.end_node_id))
        )
        if endpoint_pair in seen_pairs:
            audited.append(_rejected_gap(candidate, "duplicate_endpoint_pair"))
            continue
        seen_pairs.add(endpoint_pair)
        within_hard_geometry = (
            actual_distance <= 12.0
            and _endpoint_pair_alignment_is_within_45_degrees(
                candidate.start_node_id,
                candidate.end_node_id,
                node_by_id,
                incident_edges,
            )
        )
        if candidate.decision != "accepted":
            audited.append(candidate)
            continue
        if (
            candidate.start_node_id in used_endpoints
            or candidate.end_node_id in used_endpoints
        ):
            audited.append(_rejected_gap(candidate, "endpoint_reused"))
            continue
        start_pixel = np.rint(node_by_id[candidate.start_node_id].yx).astype(np.int64)
        end_pixel = np.rint(node_by_id[candidate.end_node_id].yx).astype(np.int64)
        if (
            not within_hard_geometry
            or actual_distance > float(auto_gap_px)
            or candidate.mean_evidence < 0.45
            or candidate.temporal_support < 2.0 / 3.0
        ):
            audited.append(_rejected_gap(candidate, "candidate_gate_failed"))
            continue
        if np.any(exclusion[path_indices[:, 0], path_indices[:, 1]]):
            audited.append(_rejected_gap(candidate, "exclusion_intersection"))
            continue

        distance_map = ndimage.distance_transform_edt(current)
        endpoint_radii = distance_map[
            (start_pixel[0], end_pixel[0]),
            (start_pixel[1], end_pixel[1]),
        ]
        radius = int(np.rint(np.clip(np.median(endpoint_radii), 1.0, 5.0)))
        centerline = np.zeros(current.shape, dtype=bool)
        centerline[path_indices[:, 0], path_indices[:, 1]] = True
        repair_layer = morphology.dilation(centerline, morphology.disk(radius))
        if np.any(repair_layer & exclusion):
            audited.append(_rejected_gap(candidate, "exclusion_intersection"))
            continue

        labels_before, count_before = ndimage.label(
            current, structure=np.ones((3, 3), dtype=np.uint8)
        )
        start_label = int(labels_before[start_pixel[0], start_pixel[1]])
        end_label = int(labels_before[end_pixel[0], end_pixel[1]])
        proposal = current | repair_layer
        labels_after, count_after = ndimage.label(
            proposal, structure=np.ones((3, 3), dtype=np.uint8)
        )
        if _foreground_hole_count(proposal) > _foreground_hole_count(current):
            audited.append(_rejected_gap(candidate, "unsupported_loop"))
            continue
        merged_intended_components = (
            start_label > 0
            and end_label > 0
            and start_label != end_label
            and int(labels_after[start_pixel[0], start_pixel[1]])
            == int(labels_after[end_pixel[0], end_pixel[1]])
        )
        if not merged_intended_components or count_after >= count_before:
            audited.append(_rejected_gap(candidate, "no_component_merge"))
            continue
        if count_after != count_before - 1:
            audited.append(_rejected_gap(candidate, "unexpected_component_merge"))
            continue

        current = proposal
        used_endpoints.update((candidate.start_node_id, candidate.end_node_id))
        audited.append(candidate)
    return current, audited


def dense_flow(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Return flow from each source pixel to its corresponding target pixel."""
    return cv2.calcOpticalFlowFarneback(
        np.ascontiguousarray(source),
        np.ascontiguousarray(target),
        None,
        0.5,
        4,
        21,
        3,
        5,
        1.2,
        0,
    )


def _sampling_coordinates(flow: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    height, width = flow.shape[:2]
    grid_x, grid_y = np.meshgrid(
        np.arange(width, dtype=np.float32),
        np.arange(height, dtype=np.float32),
    )
    return grid_x + flow[..., 0], grid_y + flow[..., 1]


def remap_with_flow(
    values: np.ndarray,
    flow: np.ndarray,
    interpolation: int,
    border_mode: int = cv2.BORDER_CONSTANT,
) -> np.ndarray:
    """Sample a target-coordinate array at source-to-target flow locations."""
    map_x, map_y = _sampling_coordinates(flow)
    return cv2.remap(
        values,
        map_x,
        map_y,
        interpolation=interpolation,
        borderMode=border_mode,
        borderValue=0,
    )


def remap_vector_field(vector_field: np.ndarray, flow: np.ndarray) -> np.ndarray:
    """Sample a target-coordinate vector field at source-to-target locations."""
    return remap_with_flow(
        vector_field.astype(np.float32, copy=False), flow, cv2.INTER_LINEAR
    )


def bidirectional_flow_evidence(
    current_frame: np.ndarray,
    neighbor_frame: np.ndarray,
    neighbor_probability: np.ndarray,
    neighbor_mask: np.ndarray,
) -> FlowEvidence:
    """Warp one neighbor into the current frame with pixelwise validity confidence."""
    forward = dense_flow(current_frame, neighbor_frame)
    backward = dense_flow(neighbor_frame, current_frame)
    warped_backward = remap_vector_field(backward, forward)
    error = np.linalg.norm(forward + warped_backward, axis=-1)
    confidence = np.exp(-0.5 * (error / 1.5) ** 2)
    confidence[error > 1.5] = 0.0

    map_x, map_y = _sampling_coordinates(forward)
    height, width = current_frame.shape
    inside_detector = (
        (map_x >= 0.0)
        & (map_x <= width - 1)
        & (map_y >= 0.0)
        & (map_y <= height - 1)
    )
    warped_neighbor_frame = remap_with_flow(
        neighbor_frame.astype(np.float32, copy=False),
        forward,
        cv2.INTER_LINEAR,
    )
    appearance_residual = np.abs(
        current_frame.astype(np.float32, copy=False) - warped_neighbor_frame
    ) / 255.0
    confidence[~inside_detector | (appearance_residual >= 0.20)] = 0.0
    sampled_error = error.astype(np.float32, copy=True)
    sampled_error[~inside_detector] = np.nan

    warped_probability = remap_with_flow(
        neighbor_probability.astype(np.float32, copy=False),
        forward,
        cv2.INTER_LINEAR,
    )
    warped_mask = remap_with_flow(
        neighbor_mask.astype(np.uint8, copy=False),
        forward,
        cv2.INTER_NEAREST,
    ).astype(bool)
    return FlowEvidence(
        warped_probability=warped_probability.astype(np.float32, copy=False),
        warped_mask=warped_mask,
        confidence=confidence.astype(np.float32, copy=False),
        forward_backward_error_px=sampled_error,
    )


def _anisotropic_temporal_majority(
    frames: np.ndarray,
    masks: np.ndarray,
    combined_evidence: np.ndarray,
    exclusion_masks: np.ndarray,
) -> np.ndarray:
    """Close aligned 2D masks, then add target pixels with two-of-three support."""
    source_masks = np.asarray(masks, dtype=bool)
    evidence_maps = np.asarray(combined_evidence, dtype=np.float32)
    exclusions = np.asarray(exclusion_masks, dtype=bool)
    footprint = morphology.disk(1)
    repaired = []
    for index in range(len(frames)):
        current_mask = source_masks[index]
        aligned_closed = [morphology.closing(current_mask, footprint)]
        first = max(0, index - 1)
        last = min(len(frames), index + 2)
        for neighbor_index in range(first, last):
            if neighbor_index == index:
                continue
            flow_evidence = bidirectional_flow_evidence(
                frames[index],
                frames[neighbor_index],
                source_masks[neighbor_index].astype(np.float32),
                source_masks[neighbor_index],
            )
            valid = flow_evidence.confidence > 0.0
            aligned = flow_evidence.warped_mask & valid
            aligned_closed.append(morphology.closing(aligned, footprint) & valid)

        vote_count = np.sum(np.asarray(aligned_closed, dtype=np.uint8), axis=0)
        spatial_additions = aligned_closed[0] & (evidence_maps[index] >= 0.20)
        temporal_additions = (vote_count >= 2) & (evidence_maps[index] >= 0.35)
        allowed_additions = (
            (spatial_additions | temporal_additions) & ~exclusions[index]
        )
        repaired.append(current_mask | allowed_additions)
    return np.asarray(repaired, dtype=bool)


def _motion_compensated_temporal_corridor(
    index: int,
    frames: np.ndarray,
    masks: np.ndarray,
) -> np.ndarray:
    """Return a one-pixel dilation of masks aligned into one target frame."""
    source_masks = np.asarray(masks, dtype=bool)
    aligned_union = source_masks[index].copy()
    first = max(0, index - 1)
    last = min(len(frames), index + 2)
    for neighbor_index in range(first, last):
        if neighbor_index == index:
            continue
        flow_evidence = bidirectional_flow_evidence(
            frames[index],
            frames[neighbor_index],
            source_masks[neighbor_index].astype(np.float32),
            source_masks[neighbor_index],
        )
        aligned_union |= flow_evidence.warped_mask & (flow_evidence.confidence > 0.0)
    return morphology.dilation(aligned_union, morphology.disk(1))


def _frame_temporal_evidence(
    index: int,
    frames: np.ndarray,
    probabilities: np.ndarray,
    masks: np.ndarray,
    radius: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, np.ndarray, float]:
    shape = frames[index].shape
    weighted_probability = (
        probabilities[index].astype(np.float32, copy=False)
        * CURRENT_PROBABILITY_WEIGHT
    )
    probability_weight = np.full(shape, CURRENT_PROBABILITY_WEIGHT, dtype=np.float32)
    supported_neighbors = np.zeros(shape, dtype=np.float32)
    valid_neighbor_count = np.zeros(shape, dtype=np.uint8)
    valid_neighbor_frame_count = 0
    confidence_sum = np.zeros(shape, dtype=np.float32)
    error_samples: list[np.ndarray] = []
    available_neighbor_count = 0

    first = max(0, index - radius)
    last = min(len(frames), index + radius + 1)
    for neighbor_index in range(first, last):
        if neighbor_index == index:
            continue
        evidence = bidirectional_flow_evidence(
            frames[index],
            frames[neighbor_index],
            probabilities[neighbor_index],
            masks[neighbor_index],
        )
        valid = evidence.confidence > 0.0
        effective_weight = NEIGHBOR_PROBABILITY_WEIGHT * evidence.confidence
        weighted_probability += effective_weight * evidence.warped_probability
        probability_weight += effective_weight
        supported_neighbors += evidence.confidence * evidence.warped_mask
        valid_neighbor_count += valid.astype(np.uint8)
        valid_neighbor_frame_count += int(np.any(valid))
        confidence_sum += evidence.confidence
        sampled_errors = evidence.forward_backward_error_px[
            np.isfinite(evidence.forward_backward_error_px)
        ]
        if sampled_errors.size:
            error_samples.append(sampled_errors)
        available_neighbor_count += 1

    fused_probability = np.clip(
        weighted_probability / probability_weight,
        0.0,
        1.0,
    ).astype(np.float32, copy=False)
    support_probability = np.divide(
        supported_neighbors,
        confidence_sum,
        out=np.zeros(shape, dtype=np.float32),
        where=confidence_sum > 0.0,
    )
    if available_neighbor_count:
        flow_confidence = confidence_sum / float(available_neighbor_count)
    else:
        flow_confidence = confidence_sum
    if error_samples:
        forward_backward_error_p95_px = float(
            np.percentile(np.concatenate(error_samples), 95)
        )
    else:
        forward_backward_error_p95_px = 0.0
    return (
        fused_probability,
        support_probability,
        valid_neighbor_count,
        valid_neighbor_frame_count,
        flow_confidence.astype(np.float32, copy=False),
        forward_backward_error_p95_px,
    )


def temporal_support(
    index: int,
    frames: np.ndarray,
    probabilities: np.ndarray,
    masks: np.ndarray,
    radius: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Return aligned neighbor-mask support and its per-pixel valid count."""
    _, support, valid_neighbor_count, _, _, _ = _frame_temporal_evidence(
        index, frames, probabilities, masks, radius
    )
    return support, valid_neighbor_count


def _temporal_fusion_evidence(
    frames: np.ndarray,
    probabilities: np.ndarray,
    masks: np.ndarray,
    radius: int = 1,
) -> _TemporalFusionEvidence:
    """Compute fused probabilities and diagnostic temporal evidence once per frame."""
    fused = []
    support = []
    valid_counts = []
    valid_frame_counts = []
    flow_confidence = []
    error_p95 = []
    for index in range(len(frames)):
        frame_evidence = _frame_temporal_evidence(
            index, frames, probabilities, masks, radius
        )
        fused.append(frame_evidence[0])
        support.append(frame_evidence[1])
        valid_counts.append(frame_evidence[2])
        valid_frame_counts.append(frame_evidence[3])
        flow_confidence.append(frame_evidence[4])
        error_p95.append(frame_evidence[5])
    return _TemporalFusionEvidence(
        fused_probability=np.asarray(fused, dtype=np.float32),
        support_probability=np.asarray(support, dtype=np.float32),
        valid_neighbor_count=np.asarray(valid_counts, dtype=np.uint8),
        valid_neighbor_frame_count=np.asarray(valid_frame_counts, dtype=np.uint8),
        flow_confidence=np.asarray(flow_confidence, dtype=np.float32),
        forward_backward_error_p95_px=np.asarray(error_p95, dtype=np.float32),
    )
