"""Deterministic, observation-only topology summaries for XA vessel masks."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, replace
from typing import Any, Literal, Sequence

import numpy as np
from scipy import ndimage
from scipy.optimize import linear_sum_assignment
from skimage import draw, morphology
from skimage.graph import route_through_array


Pixel = tuple[int, int]


@dataclass(frozen=True)
class TopologyMetrics:
    component_count: int
    largest_component_fraction: float
    endpoint_count: int
    junction_count: int
    skeleton_length_px: int


@dataclass(frozen=True)
class VesselNode:
    node_id: str
    kind: Literal["root", "junction", "endpoint"]
    yx: tuple[float, float]
    topology_id: str | None = None
    parent_node_id: str | None = None
    confidence: float = 1.0
    visibility: Literal["visible", "occluded"] = "visible"


@dataclass(frozen=True)
class VesselEdge:
    edge_id: str
    source_node_id: str
    target_node_id: str
    points_yx: np.ndarray
    confidence: float
    topology_id: str | None = None
    parent_edge_id: str | None = None
    radius_px: np.ndarray | None = None
    visibility: Literal["visible", "occluded"] = "visible"


@dataclass(frozen=True)
class VesselGraph:
    frame_index: int
    nodes: tuple[VesselNode, ...]
    edges: tuple[VesselEdge, ...]
    root_node_id: str
    component_count: int
    qc: dict[str, Any] | None = None


@dataclass(frozen=True)
class GapCandidate:
    start_node_id: str
    end_node_id: str
    distance_px: float
    angle_deg: float
    path_yx: np.ndarray
    mean_evidence: float
    temporal_support: float
    score: float
    decision: Literal["accepted", "review", "rejected"]
    reasons: tuple[str, ...]

    @property
    def crosses_exclusion(self) -> bool:
        return "exclusion_intersection" in self.reasons


@dataclass(frozen=True)
class _NodeRegion:
    kind: Literal["root", "junction", "endpoint"]
    pixels: tuple[Pixel, ...]
    yx: tuple[float, float]


def canonical_edge_id(source: str, target: str) -> str:
    left, right = sorted((source, target))
    return f"{left}--{right}"


def _validated_gap_arrays(
    evidence: np.ndarray,
    temporal_support: np.ndarray,
    exclusion_mask: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    evidence_array = np.asarray(evidence)
    support_array = np.asarray(temporal_support)
    if (
        evidence_array.ndim != 2
        or support_array.ndim != 2
        or evidence_array.shape != support_array.shape
    ):
        raise ValueError("evidence and temporal_support must have the same 2D shape")
    try:
        evidence_float = evidence_array.astype(np.float64, copy=False)
        support_float = support_array.astype(np.float64, copy=False)
    except (TypeError, ValueError) as exc:
        raise ValueError("evidence and temporal_support must be numeric") from exc
    if not np.all(np.isfinite(evidence_float)) or not np.all(
        np.isfinite(support_float)
    ):
        raise ValueError("evidence and temporal_support must contain only finite values")
    if np.any((evidence_float < 0.0) | (evidence_float > 1.0)) or np.any(
        (support_float < 0.0) | (support_float > 1.0)
    ):
        raise ValueError("evidence and temporal_support must be within [0, 1]")

    if exclusion_mask is None:
        exclusion = np.zeros(evidence_array.shape, dtype=bool)
    else:
        exclusion_array = np.asarray(exclusion_mask)
        if exclusion_array.ndim != 2 or exclusion_array.shape != evidence_array.shape:
            raise ValueError("exclusion_mask must have the same 2D shape as evidence")
        exclusion = exclusion_array.astype(bool, copy=False)
    return evidence_float, support_float, exclusion


def _validated_graph(
    graph: VesselGraph,
    detector_shape: tuple[int, int],
) -> tuple[dict[str, VesselNode], dict[str, list[VesselEdge]]]:
    node_by_id: dict[str, VesselNode] = {}
    incident_edges: dict[str, list[VesselEdge]] = {}
    height, width = detector_shape
    for node in graph.nodes:
        if node.node_id in node_by_id:
            raise ValueError(f"Duplicate vessel node ID: {node.node_id}")
        try:
            coordinate = np.asarray(node.yx, dtype=np.float64)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(
                f"Node {node.node_id} has malformed yx coordinates"
            ) from exc
        if coordinate.shape != (2,) or not np.all(np.isfinite(coordinate)):
            raise ValueError(f"Node {node.node_id} has malformed yx coordinates")
        if not (
            0.0 <= coordinate[0] <= height - 1
            and 0.0 <= coordinate[1] <= width - 1
        ):
            raise ValueError(f"Node {node.node_id} lies outside the detector")
        node_by_id[node.node_id] = node
        incident_edges[node.node_id] = []

    edge_ids: set[str] = set()
    points_by_edge_id: dict[str, np.ndarray] = {}
    for edge in graph.edges:
        if edge.edge_id in edge_ids:
            raise ValueError(f"Duplicate vessel edge ID: {edge.edge_id}")
        edge_ids.add(edge.edge_id)
        if edge.source_node_id not in node_by_id or edge.target_node_id not in node_by_id:
            raise ValueError(f"Edge {edge.edge_id} refers to a missing node")
        try:
            points = np.asarray(edge.points_yx, dtype=np.float64)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"Edge {edge.edge_id} has malformed points_yx") from exc
        if (
            points.ndim != 2
            or points.shape[1] != 2
            or len(points) < 2
            or not np.all(np.isfinite(points))
        ):
            raise ValueError(f"Edge {edge.edge_id} has malformed points_yx")
        if np.any(
            (points[:, 0] < 0.0)
            | (points[:, 0] > height - 1)
            | (points[:, 1] < 0.0)
            | (points[:, 1] > width - 1)
        ):
            raise ValueError(
                f"Edge {edge.edge_id} points_yx must remain inside the detector"
            )
        rounded_points = np.rint(points)
        if not np.array_equal(points, rounded_points):
            raise ValueError(
                f"Edge {edge.edge_id} points_yx must contain integer pixels"
            )
        point_indices = rounded_points.astype(np.int64)
        steps = np.max(np.abs(np.diff(point_indices, axis=0)), axis=1)
        if not np.all(steps == 1):
            raise ValueError(
                f"Edge {edge.edge_id} points_yx must be an 8-connected pixel chain"
            )
        point_tuples = [tuple(int(value) for value in point) for point in point_indices]
        closed_self_loop = (
            edge.source_node_id == edge.target_node_id
            and point_tuples[0] == point_tuples[-1]
        )
        unique_chain = point_tuples[:-1] if closed_self_loop else point_tuples
        if len(unique_chain) != len(set(unique_chain)):
            raise ValueError(
                f"Edge {edge.edge_id} points_yx must not repeat pixels"
            )
        points_by_edge_id[edge.edge_id] = points
        incident_edges[edge.source_node_id].append(edge)
        incident_edges[edge.target_node_id].append(edge)

    for edge in graph.edges:
        if edge.source_node_id == edge.target_node_id:
            continue
        points = points_by_edge_id[edge.edge_id]
        for node_id, point, endpoint_name in (
            (edge.source_node_id, points[0], "source"),
            (edge.target_node_id, points[-1], "target"),
        ):
            node = node_by_id[node_id]
            is_terminal = node.kind == "endpoint" or (
                node.kind == "root" and len(incident_edges[node_id]) <= 1
            )
            if is_terminal and not np.allclose(
                point,
                np.asarray(node.yx, dtype=np.float64),
                rtol=0.0,
                atol=1e-9,
            ):
                raise ValueError(
                    f"Edge {edge.edge_id} ordered endpoint does not match its "
                    f"{endpoint_name} node"
                )
    return node_by_id, incident_edges


def _node_component_ids(
    node_by_id: dict[str, VesselNode],
    edges: tuple[VesselEdge, ...],
) -> dict[str, int]:
    adjacency = {node_id: set() for node_id in node_by_id}
    for edge in edges:
        adjacency[edge.source_node_id].add(edge.target_node_id)
        adjacency[edge.target_node_id].add(edge.source_node_id)

    component_ids: dict[str, int] = {}
    for component_index, seed in enumerate(sorted(node_by_id)):
        if seed in component_ids:
            continue
        component_ids[seed] = component_index
        queue = deque([seed])
        while queue:
            current = queue.popleft()
            for neighbor in sorted(adjacency[current]):
                if neighbor not in component_ids:
                    component_ids[neighbor] = component_index
                    queue.append(neighbor)
    return component_ids


def _endpoint_tangent(
    node_id: str,
    incident_edges: dict[str, list[VesselEdge]],
) -> np.ndarray | None:
    edges = incident_edges[node_id]
    if len(edges) != 1:
        return None
    edge = edges[0]
    points = np.asarray(edge.points_yx, dtype=np.float64)
    if edge.source_node_id == node_id:
        endpoint = points[0]
        interior_points = points[1:]
    else:
        endpoint = points[-1]
        interior_points = points[-2::-1]
    for interior in interior_points:
        tangent = endpoint - interior
        norm = float(np.linalg.norm(tangent))
        if norm > 1e-12:
            return tangent
    return None


def _alignment_angle_deg(tangent: np.ndarray, direction: np.ndarray) -> float:
    tangent_norm = float(np.linalg.norm(tangent))
    direction_norm = float(np.linalg.norm(direction))
    if tangent_norm <= 1e-12 or direction_norm <= 1e-12:
        return float("inf")
    cosine = float(np.dot(tangent, direction) / (tangent_norm * direction_norm))
    return float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))


def _alignment_is_within_45_degrees(
    tangent: np.ndarray,
    direction: np.ndarray,
) -> bool:
    projection = float(np.dot(tangent, direction))
    if projection < 0.0:
        return False
    tangent_squared = float(np.dot(tangent, tangent))
    direction_squared = float(np.dot(direction, direction))
    if tangent_squared <= 0.0 or direction_squared <= 0.0:
        return False
    return projection * projection >= 0.5 * tangent_squared * direction_squared


def _endpoint_pair_alignment_is_within_45_degrees(
    start_node_id: str,
    end_node_id: str,
    node_by_id: dict[str, VesselNode],
    incident_edges: dict[str, list[VesselEdge]],
) -> bool:
    start_tangent = _endpoint_tangent(start_node_id, incident_edges)
    end_tangent = _endpoint_tangent(end_node_id, incident_edges)
    if start_tangent is None or end_tangent is None:
        return False
    start_coordinate = np.asarray(node_by_id[start_node_id].yx, dtype=np.float64)
    end_coordinate = np.asarray(node_by_id[end_node_id].yx, dtype=np.float64)
    return _alignment_is_within_45_degrees(
        start_tangent,
        end_coordinate - start_coordinate,
    ) and _alignment_is_within_45_degrees(
        end_tangent,
        start_coordinate - end_coordinate,
    )


def _endpoint_pair_geometry(
    start_node_id: str,
    end_node_id: str,
    node_by_id: dict[str, VesselNode],
    incident_edges: dict[str, list[VesselEdge]],
) -> tuple[float, float, np.ndarray | None, np.ndarray | None]:
    start_coordinate = np.asarray(node_by_id[start_node_id].yx, dtype=np.float64)
    end_coordinate = np.asarray(node_by_id[end_node_id].yx, dtype=np.float64)
    distance = float(np.linalg.norm(end_coordinate - start_coordinate))
    start_tangent = _endpoint_tangent(start_node_id, incident_edges)
    end_tangent = _endpoint_tangent(end_node_id, incident_edges)
    if start_tangent is None or end_tangent is None:
        angle = 180.0
    else:
        angle = max(
            _alignment_angle_deg(start_tangent, end_coordinate - start_coordinate),
            _alignment_angle_deg(end_tangent, start_coordinate - end_coordinate),
        )
    return distance, angle, start_tangent, end_tangent


def _routing_penalties(
    shape: tuple[int, int],
    start_yx: tuple[int, int],
    end_yx: tuple[int, int],
    start_tangent: np.ndarray | None,
    end_tangent: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray]:
    grid_y, grid_x = np.mgrid[: shape[0], : shape[1]]
    points = np.stack((grid_y, grid_x), axis=-1).astype(np.float64)
    start = np.asarray(start_yx, dtype=np.float64)
    end = np.asarray(end_yx, dtype=np.float64)
    segment = end - start
    segment_length = float(np.linalg.norm(segment))

    direction_penalty = np.zeros(shape, dtype=np.float64)
    for origin, tangent in ((start, start_tangent), (end, end_tangent)):
        if tangent is None:
            direction_penalty[:] = 1.0
            break
        tangent_norm = float(np.linalg.norm(tangent))
        if tangent_norm <= 1e-12:
            direction_penalty[:] = 1.0
            break
        vectors = points - origin
        norms = np.linalg.norm(vectors, axis=-1)
        cosine = np.divide(
            np.sum(vectors * tangent, axis=-1),
            norms * tangent_norm,
            out=np.ones(shape, dtype=np.float64),
            where=norms > 1e-12,
        )
        endpoint_penalty = (1.0 - np.clip(cosine, -1.0, 1.0)) * 0.5
        direction_penalty = np.maximum(direction_penalty, endpoint_penalty)

    if segment_length <= 1e-12:
        curvature_penalty = np.ones(shape, dtype=np.float64)
    else:
        offsets = points - start
        perpendicular_distance = np.abs(
            segment[1] * offsets[..., 0] - segment[0] * offsets[..., 1]
        ) / segment_length
        curvature_penalty = np.clip(
            perpendicular_distance / max(segment_length, 1.0), 0.0, 1.0
        )
    return direction_penalty, curvature_penalty


def _gap_cost_image(
    evidence: np.ndarray,
    temporal_support: np.ndarray,
    direction_penalty: np.ndarray,
    curvature_penalty: np.ndarray,
    exclusion_mask: np.ndarray,
) -> np.ndarray:
    cost = (
        0.45 * (1.0 - evidence)
        + 0.25 * (1.0 - temporal_support)
        + 0.20 * direction_penalty
        + 0.10 * curvature_penalty
    )
    result = np.asarray(cost, dtype=np.float64).copy()
    result[np.asarray(exclusion_mask, dtype=bool)] = np.inf
    return result


def _candidate_path(
    start: VesselNode,
    end: VesselNode,
    evidence: np.ndarray,
    support: np.ndarray,
    exclusion: np.ndarray,
    start_tangent: np.ndarray | None,
    end_tangent: np.ndarray | None,
) -> tuple[np.ndarray, float, tuple[str, ...]]:
    start_global = np.rint(np.asarray(start.yx)).astype(np.int64)
    end_global = np.rint(np.asarray(end.yx)).astype(np.int64)
    padding = 8
    y0 = max(0, int(min(start_global[0], end_global[0])) - padding)
    y1 = min(
        evidence.shape[0], int(max(start_global[0], end_global[0])) + padding + 1
    )
    x0 = max(0, int(min(start_global[1], end_global[1])) - padding)
    x1 = min(
        evidence.shape[1], int(max(start_global[1], end_global[1])) + padding + 1
    )
    local_start = (int(start_global[0] - y0), int(start_global[1] - x0))
    local_end = (int(end_global[0] - y0), int(end_global[1] - x0))
    local_start_tangent = None if start_tangent is None else start_tangent.copy()
    local_end_tangent = None if end_tangent is None else end_tangent.copy()
    direction_penalty, curvature_penalty = _routing_penalties(
        (y1 - y0, x1 - x0),
        local_start,
        local_end,
        local_start_tangent,
        local_end_tangent,
    )
    local_exclusion = exclusion[y0:y1, x0:x1]
    cost = _gap_cost_image(
        evidence[y0:y1, x0:x1],
        support[y0:y1, x0:x1],
        direction_penalty,
        curvature_penalty,
        local_exclusion,
    )
    reasons: list[str] = []
    try:
        local_path, total_cost = route_through_array(
            cost,
            local_start,
            local_end,
            fully_connected=True,
            geometric=True,
        )
        path = np.asarray(local_path, dtype=np.int64)
        path[:, 0] += y0
        path[:, 1] += x0
        if not np.isfinite(total_cost):
            reasons.append("route_unavailable")
    except (RuntimeError, ValueError):
        path_y, path_x = draw.line(
            int(start_global[0]),
            int(start_global[1]),
            int(end_global[0]),
            int(end_global[1]),
        )
        path = np.column_stack((path_y, path_x)).astype(np.int64, copy=False)
        total_cost = float("nan")
        reasons.append("route_unavailable")

    if np.any(exclusion[path[:, 0], path[:, 1]]):
        reasons.append("exclusion_intersection")
    elif "route_unavailable" in reasons:
        passable = np.isfinite(cost)
        passable_labels, _ = ndimage.label(
            passable, structure=np.ones((3, 3), dtype=np.uint8)
        )
        start_label = int(passable_labels[local_start])
        end_label = int(passable_labels[local_end])
        if start_label == 0 or end_label == 0 or start_label != end_label:
            reasons.append("exclusion_intersection")
    path.setflags(write=False)
    if not np.isfinite(total_cost):
        finite_cost = cost[np.isfinite(cost)]
        fallback_cost = float(np.mean(finite_cost)) if finite_cost.size else 1.0
        total_cost = fallback_cost * max(len(path), 1)
    score = float(total_cost / max(len(path), 1))
    return path, score, tuple(dict.fromkeys(reasons))


def find_gap_candidates(
    graph: VesselGraph,
    evidence: np.ndarray,
    temporal_support: np.ndarray,
    max_review_gap_px: int = 24,
    *,
    exclusion_mask: np.ndarray | None = None,
) -> list[GapCandidate]:
    """Return deterministic reciprocal-nearest endpoint gap candidates."""
    if (
        isinstance(max_review_gap_px, (bool, np.bool_))
        or not isinstance(max_review_gap_px, (int, np.integer))
        or not 1 <= int(max_review_gap_px) <= 24
    ):
        raise ValueError("max_review_gap_px must be an integer from 1 through 24")
    evidence_array, support_array, exclusion = _validated_gap_arrays(
        evidence, temporal_support, exclusion_mask
    )
    node_by_id, incident_edges = _validated_graph(graph, evidence_array.shape)
    component_ids = _node_component_ids(node_by_id, graph.edges)
    endpoint_ids = sorted(
        node_id
        for node_id, node in node_by_id.items()
        if node.kind == "endpoint"
        or (node.kind == "root" and len(incident_edges[node_id]) <= 1)
    )

    nearest: dict[str, str] = {}
    for node_id in endpoint_ids:
        origin = np.asarray(node_by_id[node_id].yx, dtype=np.float64)
        options = []
        for other_id in endpoint_ids:
            if other_id == node_id or component_ids[other_id] == component_ids[node_id]:
                continue
            distance = float(
                np.linalg.norm(
                    np.asarray(node_by_id[other_id].yx, dtype=np.float64) - origin
                )
            )
            if 0.0 < distance <= float(max_review_gap_px):
                options.append((distance, other_id))
        if options:
            nearest[node_id] = min(options)[1]

    pairs: list[tuple[str, str]] = []
    used_endpoints: set[str] = set()
    for node_id in endpoint_ids:
        other_id = nearest.get(node_id)
        if (
            other_id is None
            or nearest.get(other_id) != node_id
            or node_id in used_endpoints
            or other_id in used_endpoints
        ):
            continue
        pair = tuple(sorted((node_id, other_id)))
        pairs.append(pair)
        used_endpoints.update(pair)

    candidates: list[GapCandidate] = []
    for start_id, end_id in sorted(pairs):
        start = node_by_id[start_id]
        end = node_by_id[end_id]
        distance, angle, start_tangent, end_tangent = _endpoint_pair_geometry(
            start_id,
            end_id,
            node_by_id,
            incident_edges,
        )
        reasons: list[str] = []
        if start_tangent is None or end_tangent is None:
            reasons.append("direction_unavailable")
        elif not _endpoint_pair_alignment_is_within_45_degrees(
            start_id,
            end_id,
            node_by_id,
            incident_edges,
        ):
            reasons.append("direction_misaligned")

        path, score, path_reasons = _candidate_path(
            start,
            end,
            evidence_array,
            support_array,
            exclusion,
            start_tangent,
            end_tangent,
        )
        reasons.extend(path_reasons)
        interior = path[1:-1]
        if len(interior) == 0:
            mean_evidence = 0.0
            mean_support = 0.0
            reasons.append("gap_interior_unavailable")
        else:
            mean_evidence = float(
                np.mean(evidence_array[interior[:, 0], interior[:, 1]])
            )
            mean_support = float(
                np.mean(support_array[interior[:, 0], interior[:, 1]])
            )
        if mean_evidence < 0.45:
            reasons.append("insufficient_evidence")
        if mean_support < 2.0 / 3.0:
            reasons.append("insufficient_temporal_support")

        unique_reasons = tuple(dict.fromkeys(reasons))
        if unique_reasons:
            decision: Literal["accepted", "review", "rejected"] = "rejected"
        elif distance <= 12.0:
            decision = "accepted"
        else:
            decision = "review"
            unique_reasons = ("manual_review_required",)
        candidates.append(
            GapCandidate(
                start_node_id=start_id,
                end_node_id=end_id,
                distance_px=distance,
                angle_deg=angle,
                path_yx=path,
                mean_evidence=mean_evidence,
                temporal_support=mean_support,
                score=score,
                decision=decision,
                reasons=unique_reasons,
            )
        )
    return candidates


def _binary_mask(mask: np.ndarray) -> np.ndarray:
    binary = np.asarray(mask, dtype=bool)
    if binary.ndim != 2:
        raise ValueError(f"Expected a 2D mask, got shape {binary.shape}")
    return binary


def _component_count(mask: np.ndarray) -> tuple[np.ndarray, int]:
    return ndimage.label(mask, structure=np.ones((3, 3), dtype=np.uint8))


def _skeleton_degrees(skeleton: np.ndarray) -> np.ndarray:
    neighbors = ndimage.convolve(
        skeleton.astype(np.uint8),
        np.ones((3, 3), dtype=np.uint8),
        mode="constant",
        cval=0,
    )
    return neighbors - skeleton.astype(np.uint8)


def mask_topology_metrics(mask: np.ndarray) -> TopologyMetrics:
    binary = _binary_mask(mask)
    labels, component_count = _component_count(binary)
    component_sizes = np.bincount(labels.ravel())[1:]
    foreground_count = int(np.count_nonzero(binary))
    largest_component_fraction = (
        float(np.max(component_sizes) / foreground_count) if foreground_count else 0.0
    )

    skeleton = morphology.skeletonize(binary)
    degrees = _skeleton_degrees(skeleton)
    endpoint_count = int(np.count_nonzero(skeleton & (degrees == 1)))
    _, junction_count = _component_count(skeleton & (degrees >= 3))
    return TopologyMetrics(
        component_count=int(component_count),
        largest_component_fraction=largest_component_fraction,
        endpoint_count=endpoint_count,
        junction_count=int(junction_count),
        skeleton_length_px=int(np.count_nonzero(skeleton)),
    )


def _pixel_neighbors(skeleton: np.ndarray) -> dict[Pixel, tuple[Pixel, ...]]:
    pixels = [tuple(int(value) for value in pixel) for pixel in np.argwhere(skeleton)]
    pixel_set = set(pixels)
    result: dict[Pixel, tuple[Pixel, ...]] = {}
    for y, x in pixels:
        result[(y, x)] = tuple(
            sorted(
                (y + dy, x + dx)
                for dy in (-1, 0, 1)
                for dx in (-1, 0, 1)
                if (dy or dx) and (y + dy, x + dx) in pixel_set
            )
        )
    return result


def _node_regions(skeleton: np.ndarray, degrees: np.ndarray) -> tuple[_NodeRegion, ...]:
    regions: list[_NodeRegion] = []
    junction_labels, junction_count = _component_count(skeleton & (degrees >= 3))
    for label_index in range(1, junction_count + 1):
        pixels = tuple(
            tuple(int(value) for value in pixel)
            for pixel in np.argwhere(junction_labels == label_index)
        )
        coordinates = np.asarray(pixels, dtype=np.float64)
        regions.append(
            _NodeRegion(
                kind="junction",
                pixels=pixels,
                yx=(float(np.mean(coordinates[:, 0])), float(np.mean(coordinates[:, 1]))),
            )
        )

    for pixel_array in np.argwhere(skeleton & (degrees == 1)):
        pixel = tuple(int(value) for value in pixel_array)
        regions.append(
            _NodeRegion(
                kind="endpoint",
                pixels=(pixel,),
                yx=(float(pixel[0]), float(pixel[1])),
            )
        )

    skeleton_labels, skeleton_component_count = _component_count(skeleton)
    node_pixels = {pixel for region in regions for pixel in region.pixels}
    for label_index in range(1, skeleton_component_count + 1):
        component_pixels = tuple(
            tuple(int(value) for value in pixel)
            for pixel in np.argwhere(skeleton_labels == label_index)
        )
        if component_pixels and not any(pixel in node_pixels for pixel in component_pixels):
            pixel = component_pixels[0]
            regions.append(
                _NodeRegion(
                    kind="root",
                    pixels=(pixel,),
                    yx=(float(pixel[0]), float(pixel[1])),
                )
            )
    return tuple(regions)


def _pixel_segment(left: Pixel, right: Pixel) -> tuple[Pixel, Pixel]:
    return (left, right) if left <= right else (right, left)


def _trace_edges(
    regions: tuple[_NodeRegion, ...],
    neighbors: dict[Pixel, tuple[Pixel, ...]],
) -> list[tuple[int, int, tuple[Pixel, ...]]]:
    pixel_to_node = {
        pixel: node_index
        for node_index, region in enumerate(regions)
        for pixel in region.pixels
    }
    visited_segments: set[tuple[Pixel, Pixel]] = set()
    edges: list[tuple[int, int, tuple[Pixel, ...]]] = []

    for start_node_index, region in enumerate(regions):
        for start_pixel in region.pixels:
            for next_pixel in neighbors[start_pixel]:
                if pixel_to_node.get(next_pixel) == start_node_index:
                    continue
                first_segment = _pixel_segment(start_pixel, next_pixel)
                if first_segment in visited_segments:
                    continue

                path = [start_pixel]
                previous_pixel = start_pixel
                current_pixel = next_pixel
                visited_segments.add(first_segment)
                while current_pixel not in pixel_to_node:
                    path.append(current_pixel)
                    onward = tuple(
                        pixel
                        for pixel in neighbors[current_pixel]
                        if pixel != previous_pixel
                    )
                    if not onward:
                        break
                    following_pixel = onward[0]
                    visited_segments.add(_pixel_segment(current_pixel, following_pixel))
                    previous_pixel, current_pixel = current_pixel, following_pixel

                end_node_index = pixel_to_node.get(current_pixel)
                if end_node_index is None:
                    continue
                path.append(current_pixel)
                edges.append((start_node_index, end_node_index, tuple(path)))
    return edges


def _root_first_order(
    regions: tuple[_NodeRegion, ...],
    edges: list[tuple[int, int, tuple[Pixel, ...]]],
    root_index: int,
) -> list[int]:
    adjacency: list[list[tuple[int, int]]] = [[] for _ in regions]
    for edge_index, (left, right, _points) in enumerate(edges):
        adjacency[left].append((right, edge_index))
        adjacency[right].append((left, edge_index))

    order: list[int] = []
    visited: set[int] = set()
    component_roots = [root_index] + [
        index
        for index in sorted(range(len(regions)), key=lambda item: regions[item].yx)
        if index != root_index
    ]
    for component_root in component_roots:
        if component_root in visited:
            continue
        visited.add(component_root)
        queue = deque([component_root])
        while queue:
            node_index = queue.popleft()
            order.append(node_index)
            for neighbor_index, _edge_index in sorted(
                adjacency[node_index],
                key=lambda item: (regions[item[0]].yx, item[1]),
            ):
                if neighbor_index not in visited:
                    visited.add(neighbor_index)
                    queue.append(neighbor_index)
    return order


def extract_vessel_graph(
    mask: np.ndarray,
    root_hint_yx: tuple[float, float] | None,
) -> VesselGraph:
    binary = _binary_mask(mask)
    _, component_count = _component_count(binary)
    radius_map = ndimage.distance_transform_edt(binary)
    skeleton = morphology.skeletonize(binary)
    degrees = _skeleton_degrees(skeleton)
    neighbors = _pixel_neighbors(skeleton)
    regions = _node_regions(skeleton, degrees)
    if not regions:
        return VesselGraph(
            frame_index=0,
            nodes=(),
            edges=(),
            root_node_id="",
            component_count=int(component_count),
        )

    if root_hint_yx is None:
        root_index = min(range(len(regions)), key=lambda index: regions[index].yx)
    else:
        root_y, root_x = root_hint_yx
        root_index = min(
            range(len(regions)),
            key=lambda index: (
                (regions[index].yx[0] - root_y) ** 2 + (regions[index].yx[1] - root_x) ** 2,
                regions[index].yx,
            ),
        )

    traced_edges = _trace_edges(regions, neighbors)
    node_order = _root_first_order(regions, traced_edges, root_index)
    node_ids = {
        region_index: f"node-{order_index:04d}"
        for order_index, region_index in enumerate(node_order)
    }
    order_positions = {region_index: index for index, region_index in enumerate(node_order)}
    nodes = tuple(
        VesselNode(
            node_id=node_ids[region_index],
            kind="root" if region_index == root_index else regions[region_index].kind,
            yx=regions[region_index].yx,
        )
        for region_index in node_order
    )

    ordered_edges: list[VesselEdge] = []
    edge_id_counts: dict[str, int] = {}
    for left_index, right_index, points in sorted(
        traced_edges,
        key=lambda item: (
            min(order_positions[item[0]], order_positions[item[1]]),
            max(order_positions[item[0]], order_positions[item[1]]),
        ),
    ):
        if order_positions[left_index] <= order_positions[right_index]:
            source_index, target_index = left_index, right_index
            ordered_points = points
        else:
            source_index, target_index = right_index, left_index
            ordered_points = tuple(reversed(points))
        source_id = node_ids[source_index]
        target_id = node_ids[target_index]
        base_edge_id = canonical_edge_id(source_id, target_id)
        duplicate_index = edge_id_counts.get(base_edge_id, 0)
        edge_id_counts[base_edge_id] = duplicate_index + 1
        edge_id = (
            base_edge_id
            if duplicate_index == 0
            else f"{base_edge_id}#{duplicate_index:04d}"
        )
        ordered_edges.append(
            VesselEdge(
                edge_id=edge_id,
                source_node_id=source_id,
                target_node_id=target_id,
                points_yx=np.asarray(ordered_points, dtype=np.float64),
                confidence=1.0,
                radius_px=radius_map[tuple(np.asarray(ordered_points).T)].astype(
                    np.float64
                ),
            )
        )

    return VesselGraph(
        frame_index=0,
        nodes=nodes,
        edges=tuple(ordered_edges),
        root_node_id=node_ids[root_index],
        component_count=int(component_count),
    )


_TRACKING_GATE_PX = 8.0
_TRACKED_EDGE_SAMPLE_COUNT = 64


def _validate_tracking_graph(
    graph: VesselGraph,
) -> tuple[dict[str, VesselNode], dict[str, VesselEdge]]:
    if not isinstance(graph, VesselGraph):
        raise ValueError("graphs must contain only VesselGraph values")
    node_by_id: dict[str, VesselNode] = {}
    for node in graph.nodes:
        if node.node_id in node_by_id:
            raise ValueError(f"Duplicate vessel node ID: {node.node_id}")
        coordinate = np.asarray(node.yx, dtype=np.float64)
        if coordinate.shape != (2,) or not np.all(np.isfinite(coordinate)):
            raise ValueError(f"Node {node.node_id} has malformed yx coordinates")
        confidence = float(node.confidence)
        if not np.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
            raise ValueError(f"Node {node.node_id} confidence must be within [0, 1]")
        node_by_id[node.node_id] = node
    if graph.root_node_id and graph.root_node_id not in node_by_id:
        raise ValueError("root_node_id must refer to a graph node")

    edge_by_id: dict[str, VesselEdge] = {}
    for edge in graph.edges:
        if edge.edge_id in edge_by_id:
            raise ValueError(f"Duplicate vessel edge ID: {edge.edge_id}")
        if edge.source_node_id not in node_by_id or edge.target_node_id not in node_by_id:
            raise ValueError(f"Edge {edge.edge_id} refers to a missing node")
        points = np.asarray(edge.points_yx, dtype=np.float64)
        if (
            points.ndim != 2
            or points.shape[1:] != (2,)
            or len(points) < 2
            or not np.all(np.isfinite(points))
        ):
            raise ValueError(f"Edge {edge.edge_id} has malformed points_yx")
        confidence = float(edge.confidence)
        if not np.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
            raise ValueError(f"Edge {edge.edge_id} confidence must be within [0, 1]")
        if edge.radius_px is not None:
            radii = np.asarray(edge.radius_px, dtype=np.float64)
            if (
                radii.shape != (len(points),)
                or not np.all(np.isfinite(radii))
                or np.any(radii <= 0.0)
            ):
                raise ValueError(
                    f"Edge {edge.edge_id} radius_px must contain one finite positive value per point"
                )
        edge_by_id[edge.edge_id] = edge
    return node_by_id, edge_by_id


def _validated_tracking_inputs(
    graphs: Sequence[VesselGraph], flows: Sequence[np.ndarray]
) -> tuple[tuple[VesselGraph, ...], tuple[np.ndarray, ...]]:
    try:
        graph_sequence = tuple(graphs)
        flow_sequence = tuple(flows)
    except TypeError as exc:
        raise ValueError("graphs and flows must be finite sequences") from exc
    if len(flow_sequence) != max(len(graph_sequence) - 1, 0):
        raise ValueError("flows must contain exactly len(graphs) - 1 fields")
    for graph in graph_sequence:
        _validate_tracking_graph(graph)

    validated_flows: list[np.ndarray] = []
    for flow in flow_sequence:
        array = np.asarray(flow)
        if array.ndim != 3 or array.shape[2] != 2:
            raise ValueError("each flow must have shape H x W x 2")
        if array.shape[0] <= 0 or array.shape[1] <= 0:
            raise ValueError("each flow must have positive H and W dimensions")
        if array.dtype == np.dtype(bool) or not np.issubdtype(array.dtype, np.number):
            raise ValueError("each flow must contain numeric dx, dy values")
        float_flow = array.astype(np.float64, copy=False)
        if not np.all(np.isfinite(float_flow)):
            raise ValueError("each flow must contain only finite dx, dy values")
        validated_flows.append(float_flow)
    return graph_sequence, tuple(validated_flows)


def _node_degrees(graph: VesselGraph) -> dict[str, int]:
    degrees = {node.node_id: 0 for node in graph.nodes}
    for edge in graph.edges:
        degrees[edge.source_node_id] += 1
        degrees[edge.target_node_id] += 1
    return degrees


def _root_first_nodes(graph: VesselGraph) -> tuple[VesselNode, ...]:
    node_by_id, _ = _validate_tracking_graph(graph)
    adjacency = {node_id: set() for node_id in node_by_id}
    for edge in graph.edges:
        adjacency[edge.source_node_id].add(edge.target_node_id)
        adjacency[edge.target_node_id].add(edge.source_node_id)

    ordered: list[VesselNode] = []
    visited: set[str] = set()
    component_roots = ([graph.root_node_id] if graph.root_node_id else []) + sorted(
        node_id for node_id in node_by_id if node_id != graph.root_node_id
    )
    for component_root in component_roots:
        if component_root in visited:
            continue
        visited.add(component_root)
        queue = deque([component_root])
        while queue:
            node_id = queue.popleft()
            ordered.append(node_by_id[node_id])
            for neighbor_id in sorted(adjacency[node_id]):
                if neighbor_id not in visited:
                    visited.add(neighbor_id)
                    queue.append(neighbor_id)
    return tuple(ordered)


def _root_first_edges(graph: VesselGraph) -> tuple[VesselEdge, ...]:
    node_positions = {
        node.node_id: index for index, node in enumerate(_root_first_nodes(graph))
    }

    def edge_key(edge: VesselEdge) -> tuple[Any, ...]:
        source_position = node_positions[edge.source_node_id]
        target_position = node_positions[edge.target_node_id]
        points = np.asarray(edge.points_yx, dtype=np.float64)
        if source_position > target_position:
            points = points[::-1]
        return (
            min(source_position, target_position),
            max(source_position, target_position),
            tuple(points.ravel()),
            edge.edge_id,
        )

    return tuple(sorted(graph.edges, key=edge_key))


def _sample_forward_flow(
    flow: np.ndarray, yx: tuple[float, float] | np.ndarray
) -> np.ndarray | None:
    y, x = np.asarray(yx, dtype=np.float64)
    height, width = flow.shape[:2]
    if not (0.0 <= y <= height - 1 and 0.0 <= x <= width - 1):
        return None
    y0 = int(np.floor(y))
    x0 = int(np.floor(x))
    y1 = min(y0 + 1, height - 1)
    x1 = min(x0 + 1, width - 1)
    wy = y - y0
    wx = x - x0
    displacement = (
        (1.0 - wy) * (1.0 - wx) * flow[y0, x0]
        + (1.0 - wy) * wx * flow[y0, x1]
        + wy * (1.0 - wx) * flow[y1, x0]
        + wy * wx * flow[y1, x1]
    )
    return np.asarray((y + displacement[1], x + displacement[0]), dtype=np.float64)


def _inside_detector(yx: np.ndarray, flow: np.ndarray) -> bool:
    height, width = flow.shape[:2]
    return bool(0.0 <= yx[0] <= height - 1 and 0.0 <= yx[1] <= width - 1)


def _resample_edge_geometry(edge: VesselEdge) -> tuple[np.ndarray, np.ndarray]:
    points = np.asarray(edge.points_yx, dtype=np.float64)
    radii = (
        np.ones(len(points), dtype=np.float64)
        if edge.radius_px is None
        else np.asarray(edge.radius_px, dtype=np.float64)
    )
    segment_lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    cumulative = np.concatenate(([0.0], np.cumsum(segment_lengths)))
    if cumulative[-1] <= np.finfo(np.float64).eps:
        sampled_points = np.repeat(points[:1], _TRACKED_EDGE_SAMPLE_COUNT, axis=0)
        sampled_radii = np.repeat(radii[:1], _TRACKED_EDGE_SAMPLE_COUNT)
    else:
        samples = np.linspace(0.0, cumulative[-1], _TRACKED_EDGE_SAMPLE_COUNT)
        sampled_points = np.column_stack(
            [np.interp(samples, cumulative, points[:, axis]) for axis in range(2)]
        )
        sampled_radii = np.interp(samples, cumulative, radii)
    return sampled_points, sampled_radii


def _canonical_loop_direction(
    points: np.ndarray, radii: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    forward = tuple(points.ravel())
    reverse = tuple(points[::-1].ravel())
    return (points, radii) if forward <= reverse else (points[::-1], radii[::-1])


def _normalized_edge(
    edge: VesselEdge, local_to_topology: dict[str, str]
) -> tuple[VesselEdge, tuple[str, str]]:
    source_topology = local_to_topology[edge.source_node_id]
    target_topology = local_to_topology[edge.target_node_id]
    points, radii = _resample_edge_geometry(edge)
    source_local = edge.source_node_id
    target_local = edge.target_node_id
    if source_topology > target_topology:
        source_topology, target_topology = target_topology, source_topology
        source_local, target_local = target_local, source_local
        points = points[::-1]
        radii = radii[::-1]
    elif source_topology == target_topology:
        points, radii = _canonical_loop_direction(points, radii)
    return (
        replace(
            edge,
            source_node_id=source_local,
            target_node_id=target_local,
            points_yx=np.asarray(points, dtype=np.float64),
            radius_px=np.asarray(radii, dtype=np.float64),
            visibility="visible",
        ),
        (source_topology, target_topology),
    )


def _warp_edge_points(points: np.ndarray, flow: np.ndarray) -> np.ndarray | None:
    warped: list[np.ndarray] = []
    for point in np.asarray(points, dtype=np.float64):
        prediction = _sample_forward_flow(flow, point)
        if prediction is None or not _inside_detector(prediction, flow):
            return None
        warped.append(prediction)
    return np.asarray(warped, dtype=np.float64)


def _edge_shape_distance(
    predicted: np.ndarray, observed: np.ndarray, *, is_loop: bool
) -> float:
    forward = float(np.mean(np.linalg.norm(predicted - observed, axis=1)))
    if not is_loop:
        return forward
    reverse = float(np.mean(np.linalg.norm(predicted - observed[::-1], axis=1)))
    return min(forward, reverse)


def _quality_status(matched_edge_fraction: float) -> str:
    if matched_edge_fraction >= 0.70:
        return "accepted"
    if matched_edge_fraction >= 0.40:
        return "review"
    return "rejected"


def _tracking_qc(
    *,
    visible_node_count: int,
    matched_node_count: int,
    visible_edge_count: int,
    matched_edge_count: int,
    new_nodes: int,
    retired_nodes: int,
    topology_changed: bool,
) -> dict[str, Any]:
    matched_node_fraction = (
        matched_node_count / visible_node_count if visible_node_count else 1.0
    )
    matched_edge_fraction = (
        matched_edge_count / visible_edge_count if visible_edge_count else 1.0
    )
    return {
        "matched_node_fraction": float(matched_node_fraction),
        "matched_edge_fraction": float(matched_edge_fraction),
        "new_nodes": int(new_nodes),
        "retired_nodes": int(retired_nodes),
        "topology_changed": bool(topology_changed),
        "quality_status": _quality_status(matched_edge_fraction),
    }


def _with_rooted_parents(
    nodes: tuple[VesselNode, ...],
    edges: tuple[VesselEdge, ...],
    root_node_id: str,
) -> tuple[tuple[VesselNode, ...], tuple[VesselEdge, ...]]:
    local_to_topology = {
        node.node_id: str(node.topology_id) for node in nodes if node.topology_id is not None
    }
    topology_to_local = {topology_id: local for local, topology_id in local_to_topology.items()}
    edge_by_topology = {
        str(edge.topology_id): edge for edge in edges if edge.topology_id is not None
    }
    adjacency = {topology_id: [] for topology_id in topology_to_local}
    for topology_id, edge in edge_by_topology.items():
        source = local_to_topology[edge.source_node_id]
        target = local_to_topology[edge.target_node_id]
        adjacency[source].append((target, topology_id))
        adjacency[target].append((source, topology_id))

    parent_node: dict[str, str | None] = {
        topology_id: None for topology_id in topology_to_local
    }
    parent_edge: dict[str, str | None] = {topology_id: None for topology_id in edge_by_topology}
    incoming_edge: dict[str, str | None] = {}
    visited: set[str] = set()
    root_topology = local_to_topology.get(root_node_id)
    component_roots = ([root_topology] if root_topology else []) + sorted(
        topology_id for topology_id in topology_to_local if topology_id != root_topology
    )
    for component_root in component_roots:
        if component_root in visited:
            continue
        visited.add(component_root)
        incoming_edge[component_root] = None
        queue = deque([component_root])
        while queue:
            topology_id = queue.popleft()
            for neighbor, edge_topology_id in sorted(adjacency[topology_id]):
                if neighbor in visited:
                    continue
                visited.add(neighbor)
                parent_node[neighbor] = topology_id
                parent_edge[edge_topology_id] = incoming_edge[topology_id]
                incoming_edge[neighbor] = edge_topology_id
                queue.append(neighbor)

    return (
        tuple(
            replace(node, parent_node_id=parent_node[str(node.topology_id)])
            for node in nodes
        ),
        tuple(
            replace(edge, parent_edge_id=parent_edge[str(edge.topology_id)])
            for edge in edges
        ),
    )


def track_vessel_graphs(
    graphs: Sequence[VesselGraph], flows: Sequence[np.ndarray]
) -> list[VesselGraph]:
    graph_sequence, flow_sequence = _validated_tracking_inputs(graphs, flows)
    if not graph_sequence:
        return []

    next_node_identity = 0
    next_edge_identity = 0
    node_misses: dict[str, int] = {}
    edge_misses: dict[str, int] = {}

    first_graph = graph_sequence[0]
    first_nodes: list[VesselNode] = []
    first_local_to_topology: dict[str, str] = {}
    for node in _root_first_nodes(first_graph):
        topology_id = f"node-v1-{next_node_identity:06d}"
        next_node_identity += 1
        first_local_to_topology[node.node_id] = topology_id
        node_misses[topology_id] = 0
        first_nodes.append(
            replace(
                node,
                topology_id=topology_id,
                parent_node_id=None,
                confidence=1.0,
                visibility="visible",
            )
        )

    first_edges: list[VesselEdge] = []
    for edge in _root_first_edges(first_graph):
        normalized, _ = _normalized_edge(edge, first_local_to_topology)
        topology_id = f"edge-v1-{next_edge_identity:06d}"
        next_edge_identity += 1
        edge_misses[topology_id] = 0
        first_edges.append(
            replace(
                normalized,
                topology_id=topology_id,
                parent_edge_id=None,
                confidence=1.0,
            )
        )
    first_root = first_graph.root_node_id
    parented_nodes, parented_edges = _with_rooted_parents(
        tuple(first_nodes), tuple(first_edges), first_root
    )
    tracked = [
        replace(
            first_graph,
            nodes=parented_nodes,
            edges=parented_edges,
            root_node_id=first_root,
            qc=_tracking_qc(
                visible_node_count=len(first_nodes),
                matched_node_count=len(first_nodes),
                visible_edge_count=len(first_edges),
                matched_edge_count=len(first_edges),
                new_nodes=0,
                retired_nodes=0,
                topology_changed=False,
            ),
        )
    ]
    root_topology_id = first_local_to_topology.get(first_root)

    for graph, flow in zip(graph_sequence[1:], flow_sequence, strict=True):
        previous = tracked[-1]
        previous_local_to_topology = {
            node.node_id: str(node.topology_id) for node in previous.nodes
        }
        previous_degrees = _node_degrees(previous)
        current_degrees = _node_degrees(graph)
        prior_nodes = sorted(previous.nodes, key=lambda node: str(node.topology_id))
        observations = list(_root_first_nodes(graph))
        predictions = [_sample_forward_flow(flow, node.yx) for node in prior_nodes]

        matched_prior: set[int] = set()
        matched_observations: set[int] = set()
        node_matches: dict[int, int] = {}
        node_costs: dict[int, float] = {}
        if prior_nodes and observations:
            costs = np.full((len(prior_nodes), len(observations)), 1.0e6)
            distances = np.full_like(costs, np.inf)
            for row, (prior, prediction) in enumerate(zip(prior_nodes, predictions, strict=True)):
                if prediction is None or not _inside_detector(prediction, flow):
                    continue
                for column, observation in enumerate(observations):
                    distance = float(
                        np.linalg.norm(prediction - np.asarray(observation.yx, dtype=np.float64))
                    )
                    distances[row, column] = distance
                    if distance > _TRACKING_GATE_PX:
                        continue
                    prior_degree = previous_degrees[prior.node_id]
                    current_degree = current_degrees[observation.node_id]
                    degree_mismatch = min(
                        abs(prior_degree - current_degree)
                        / max(prior_degree, current_degree, 1),
                        1.0,
                    )
                    costs[row, column] = (
                        0.55 * min(distance / _TRACKING_GATE_PX, 1.0)
                        + 0.25 * float(prior.kind != observation.kind)
                        + 0.20 * degree_mismatch
                    )
            rows, columns = linear_sum_assignment(costs)
            for row, column in zip(rows, columns, strict=True):
                if distances[row, column] <= _TRACKING_GATE_PX:
                    matched_prior.add(int(row))
                    matched_observations.add(int(column))
                    node_matches[int(column)] = int(row)
                    node_costs[int(column)] = float(costs[row, column])

        current_nodes: list[VesselNode] = []
        current_local_to_topology: dict[str, str] = {}
        for observation_index, observation in enumerate(observations):
            if observation_index in matched_observations:
                prior_index = node_matches[observation_index]
                topology_id = str(prior_nodes[prior_index].topology_id)
                confidence = float(np.clip(1.0 - node_costs[observation_index], 0.0, 1.0))
                node_misses[topology_id] = 0
            else:
                topology_id = f"node-v1-{next_node_identity:06d}"
                next_node_identity += 1
                confidence = 1.0
                node_misses[topology_id] = 0
            current_local_to_topology[observation.node_id] = topology_id
            current_nodes.append(
                replace(
                    observation,
                    topology_id=topology_id,
                    parent_node_id=None,
                    confidence=confidence,
                    visibility="visible",
                )
            )

        retired_nodes = 0
        for prior_index, prior in enumerate(prior_nodes):
            if prior_index in matched_prior:
                continue
            topology_id = str(prior.topology_id)
            misses = node_misses[topology_id] + 1
            node_misses[topology_id] = misses
            if misses > 2:
                retired_nodes += 1
                continue
            prediction = predictions[prior_index]
            if prediction is not None and not _inside_detector(prediction, flow):
                retired_nodes += 1
                continue
            predicted_yx = (
                tuple(np.asarray(prior.yx, dtype=np.float64))
                if prediction is None
                else tuple(prediction)
            )
            local_id = f"occluded-{topology_id}"
            current_local_to_topology[local_id] = topology_id
            current_nodes.append(
                replace(
                    prior,
                    node_id=local_id,
                    yx=predicted_yx,
                    parent_node_id=None,
                    confidence=float(np.clip(prior.confidence * 0.5, 0.0, 1.0)),
                    visibility="occluded",
                )
            )

        topology_to_current_local = {
            topology_id: local_id
            for local_id, topology_id in current_local_to_topology.items()
        }
        current_edge_observations: list[tuple[VesselEdge, tuple[str, str]]] = []
        for edge in _root_first_edges(graph):
            current_edge_observations.append(
                _normalized_edge(edge, current_local_to_topology)
            )

        prior_edge_records: list[tuple[VesselEdge, tuple[str, str], np.ndarray | None]] = []
        for edge in sorted(previous.edges, key=lambda item: str(item.topology_id)):
            endpoint_pair = (
                previous_local_to_topology[edge.source_node_id],
                previous_local_to_topology[edge.target_node_id],
            )
            prior_edge_records.append(
                (edge, endpoint_pair, _warp_edge_points(edge.points_yx, flow))
            )

        matched_prior_edges: set[int] = set()
        matched_current_edges: set[int] = set()
        edge_matches: dict[int, tuple[int, float]] = {}
        if prior_edge_records and current_edge_observations:
            costs = np.full(
                (len(prior_edge_records), len(current_edge_observations)), 1.0e6
            )
            for row, (prior_edge, prior_pair, predicted_points) in enumerate(
                prior_edge_records
            ):
                if predicted_points is None:
                    continue
                for column, (current_edge, current_pair) in enumerate(
                    current_edge_observations
                ):
                    if prior_pair != current_pair:
                        continue
                    distance = _edge_shape_distance(
                        predicted_points,
                        current_edge.points_yx,
                        is_loop=prior_pair[0] == prior_pair[1],
                    )
                    if distance <= _TRACKING_GATE_PX:
                        costs[row, column] = distance
            rows, columns = linear_sum_assignment(costs)
            for row, column in zip(rows, columns, strict=True):
                if costs[row, column] <= _TRACKING_GATE_PX:
                    matched_prior_edges.add(int(row))
                    matched_current_edges.add(int(column))
                    edge_matches[int(column)] = (int(row), float(costs[row, column]))

        current_edges: list[VesselEdge] = []
        new_edges = 0
        for edge_index, (edge, endpoint_pair) in enumerate(current_edge_observations):
            if edge_index in edge_matches:
                prior_index, distance = edge_matches[edge_index]
                topology_id = str(prior_edge_records[prior_index][0].topology_id)
                edge_misses[topology_id] = 0
                confidence = float(
                    np.clip(1.0 - distance / _TRACKING_GATE_PX, 0.0, 1.0)
                )
            else:
                topology_id = f"edge-v1-{next_edge_identity:06d}"
                next_edge_identity += 1
                edge_misses[topology_id] = 0
                confidence = 1.0
                new_edges += 1
            current_edges.append(
                replace(
                    edge,
                    topology_id=topology_id,
                    parent_edge_id=None,
                    confidence=confidence,
                    visibility="visible",
                )
            )

        retired_edges = 0
        active_topology_ids = set(topology_to_current_local)
        for prior_index, (prior_edge, endpoint_pair, predicted_points) in enumerate(
            prior_edge_records
        ):
            if prior_index in matched_prior_edges:
                continue
            topology_id = str(prior_edge.topology_id)
            misses = edge_misses[topology_id] + 1
            edge_misses[topology_id] = misses
            endpoints_active = set(endpoint_pair).issubset(active_topology_ids)
            if misses > 2 or not endpoints_active or predicted_points is None:
                retired_edges += 1
                continue
            current_edges.append(
                replace(
                    prior_edge,
                    edge_id=f"occluded-{topology_id}",
                    source_node_id=topology_to_current_local[endpoint_pair[0]],
                    target_node_id=topology_to_current_local[endpoint_pair[1]],
                    points_yx=predicted_points,
                    parent_edge_id=None,
                    confidence=float(
                        np.clip(prior_edge.confidence * 0.5, 0.0, 1.0)
                    ),
                    visibility="occluded",
                )
            )

        active_node_topologies = {
            str(node.topology_id) for node in current_nodes
        }
        if root_topology_id not in active_node_topologies:
            root_topology_id = current_local_to_topology.get(graph.root_node_id)
        root_node_id = (
            topology_to_current_local[root_topology_id]
            if root_topology_id in topology_to_current_local
            else ""
        )
        parented_nodes, parented_edges = _with_rooted_parents(
            tuple(current_nodes), tuple(current_edges), root_node_id
        )
        new_nodes = len(observations) - len(matched_observations)
        topology_changed = bool(
            new_nodes or retired_nodes or new_edges or retired_edges
        )
        tracked.append(
            replace(
                graph,
                nodes=parented_nodes,
                edges=parented_edges,
                root_node_id=root_node_id,
                qc=_tracking_qc(
                    visible_node_count=len(observations),
                    matched_node_count=len(matched_observations),
                    visible_edge_count=len(current_edge_observations),
                    matched_edge_count=len(matched_current_edges),
                    new_nodes=new_nodes,
                    retired_nodes=retired_nodes,
                    topology_changed=topology_changed,
                ),
            )
        )
    return tracked
