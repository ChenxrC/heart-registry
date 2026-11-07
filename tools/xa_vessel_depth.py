"""Graph-constrained monoplane XA depth placement in the CT LPS frame."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral

import numpy as np
import trimesh
from scipy import sparse
from scipy.optimize import lsq_linear
from scipy.spatial import cKDTree

try:
    from .xa_vessel_topology import VesselGraph
except ImportError:  # direct script execution
    from xa_vessel_topology import VesselGraph


@dataclass(frozen=True)
class DepthOptimizationResult:
    points_lps_by_edge: dict[str, np.ndarray]
    node_points_lps: dict[str, np.ndarray]
    objective_terms: dict[str, float]
    quality_status: str
    reasons: tuple[str, ...]


def _stable_key(local_id: str, topology_id: str | None) -> str:
    return topology_id if topology_id is not None else local_id


def _camera_vectors(camera: object) -> tuple[np.ndarray, ...]:
    try:
        source = np.asarray(camera.source, dtype=np.float64)
        detector_center = np.asarray(camera.detector_center, dtype=np.float64)
        u_axis = np.asarray(camera.u_axis, dtype=np.float64)
        v_axis = np.asarray(camera.v_axis, dtype=np.float64)
        n_axis = np.asarray(camera.n_axis, dtype=np.float64)
        principal_uv = np.asarray(camera.principal_uv, dtype=np.float64)
        row_spacing = float(camera.row_spacing_mm)
        col_spacing = float(camera.col_spacing_mm)
        rows = camera.rows
        columns = camera.columns
        sdd_mm = float(camera.sdd_mm)
    except (AttributeError, TypeError, ValueError, OverflowError) as exc:
        raise ValueError("camera has malformed geometry") from exc
    vectors = (source, detector_center, u_axis, v_axis, n_axis)
    if any(vector.shape != (3,) for vector in vectors):
        raise ValueError("camera LPS vectors must have shape (3,)")
    if principal_uv.shape != (2,):
        raise ValueError("camera principal_uv must have shape (2,)")
    if not all(np.all(np.isfinite(vector)) for vector in (*vectors, principal_uv)):
        raise ValueError("camera geometry must be finite")
    if not np.isfinite(row_spacing) or not np.isfinite(col_spacing):
        raise ValueError("camera spacing must be finite")
    if row_spacing <= 0.0 or col_spacing <= 0.0:
        raise ValueError("camera spacing must be positive")
    if (
        isinstance(rows, bool)
        or isinstance(columns, bool)
        or not isinstance(rows, Integral)
        or not isinstance(columns, Integral)
        or rows <= 0
        or columns <= 0
    ):
        raise ValueError("camera dimensions must be positive integers")
    axes = (u_axis, v_axis, n_axis)
    if any(not np.isclose(np.linalg.norm(axis), 1.0, atol=1e-8) for axis in axes):
        raise ValueError("camera axes must be orthonormal unit vectors")
    if any(
        not np.isclose(float(left @ right), 0.0, atol=1e-8)
        for left, right in ((u_axis, v_axis), (u_axis, n_axis), (v_axis, n_axis))
    ):
        raise ValueError("camera axes must be orthonormal unit vectors")
    source_to_detector = detector_center - source
    if not np.isfinite(sdd_mm) or sdd_mm <= 0.0:
        raise ValueError("camera sdd_mm must be finite and positive")
    if not np.isclose(np.linalg.norm(source_to_detector), sdd_mm, atol=1e-6):
        raise ValueError("camera sdd_mm must match the source-detector distance")
    if float(source_to_detector @ n_axis) <= 0.0:
        raise ValueError("camera detector plane must face away from the source")
    return (*vectors, principal_uv, np.asarray([row_spacing, col_spacing]))


def _validate_graph(graph: VesselGraph, camera: object) -> None:
    if not isinstance(graph, VesselGraph):
        raise ValueError("graph must be a VesselGraph")
    rows = int(camera.rows)
    columns = int(camera.columns)
    node_ids: set[str] = set()
    node_keys: set[str] = set()
    node_yx_by_id: dict[str, np.ndarray] = {}
    for node in graph.nodes:
        if not isinstance(node.node_id, str) or not node.node_id or node.node_id in node_ids:
            raise ValueError("graph node IDs must be unique non-empty strings")
        if node.topology_id is not None and (
            not isinstance(node.topology_id, str) or not node.topology_id
        ):
            raise ValueError(f"Node {node.node_id} has malformed topology_id")
        key = _stable_key(node.node_id, node.topology_id)
        if key in node_keys:
            raise ValueError(f"Duplicate vessel node topology key: {key}")
        coordinate = np.asarray(node.yx, dtype=np.float64)
        if coordinate.shape != (2,) or not np.all(np.isfinite(coordinate)):
            raise ValueError(f"Node {node.node_id} has malformed yx coordinates")
        if not (0.0 <= coordinate[0] < rows and 0.0 <= coordinate[1] < columns):
            raise ValueError(f"Node {node.node_id} lies outside the detector")
        node_ids.add(node.node_id)
        node_keys.add(key)
        node_yx_by_id[node.node_id] = coordinate
    if graph.root_node_id and graph.root_node_id not in node_ids:
        raise ValueError("root_node_id must refer to a graph node")

    edge_ids: set[str] = set()
    edge_keys: set[str] = set()
    for edge in graph.edges:
        if not isinstance(edge.edge_id, str) or not edge.edge_id or edge.edge_id in edge_ids:
            raise ValueError("graph edge IDs must be unique non-empty strings")
        if edge.source_node_id not in node_ids or edge.target_node_id not in node_ids:
            raise ValueError(f"Edge {edge.edge_id} refers to a missing node")
        if edge.topology_id is not None and (
            not isinstance(edge.topology_id, str) or not edge.topology_id
        ):
            raise ValueError(f"Edge {edge.edge_id} has malformed topology_id")
        key = _stable_key(edge.edge_id, edge.topology_id)
        if key in edge_keys:
            raise ValueError(f"Duplicate vessel edge topology key: {key}")
        points = np.asarray(edge.points_yx, dtype=np.float64)
        if points.ndim != 2 or points.shape[1:] != (2,) or len(points) < 2:
            raise ValueError(f"Edge {edge.edge_id} has malformed points_yx")
        if not np.all(np.isfinite(points)):
            raise ValueError(f"Edge {edge.edge_id} points_yx must be finite")
        if np.any(
            (points[:, 0] < 0.0)
            | (points[:, 0] >= rows)
            | (points[:, 1] < 0.0)
            | (points[:, 1] >= columns)
        ):
            raise ValueError(f"Edge {edge.edge_id} points_yx must remain inside the detector")
        if not np.array_equal(points[0], node_yx_by_id[edge.source_node_id]) or not np.array_equal(
            points[-1], node_yx_by_id[edge.target_node_id]
        ):
            raise ValueError(
                f"Edge {edge.edge_id} points_yx endpoints must exactly match its source and target nodes"
            )
        if edge.radius_px is not None:
            radii = np.asarray(edge.radius_px, dtype=np.float64)
            if radii.shape != (len(points),) or not np.all(np.isfinite(radii)) or np.any(radii <= 0.0):
                raise ValueError(
                    f"Edge {edge.edge_id} radius_px must contain one finite positive value per point"
                )
        edge_ids.add(edge.edge_id)
        edge_keys.add(key)


def _validate_previous_result(previous_result: DepthOptimizationResult | None) -> None:
    if previous_result is None:
        return
    if not isinstance(previous_result, DepthOptimizationResult):
        raise ValueError("previous_result must be a DepthOptimizationResult")
    for key, raw_points in previous_result.points_lps_by_edge.items():
        points = np.asarray(raw_points, dtype=np.float64)
        if not isinstance(key, str) or not key or points.ndim != 2 or points.shape[1:] != (3,):
            raise ValueError("previous_result edge points must have string keys and shape N x 3")
        if not np.all(np.isfinite(points)):
            raise ValueError("previous_result edge points must be finite")
    for key, raw_point in previous_result.node_points_lps.items():
        point = np.asarray(raw_point, dtype=np.float64)
        if not isinstance(key, str) or not key or point.shape != (3,):
            raise ValueError("previous_result node points must have string keys and shape (3,)")
        if not np.all(np.isfinite(point)):
            raise ValueError("previous_result node points must be finite")
    required_terms = {
        "heart_surface_distance",
        "first_derivative_smoothness",
        "second_derivative_smoothness",
        "junction_equality_penalty",
        "temporal_depth_penalty",
        "weighted_total",
    }
    if set(previous_result.objective_terms) != required_terms:
        raise ValueError("previous_result objective_terms are malformed")
    values = np.asarray(list(previous_result.objective_terms.values()), dtype=np.float64)
    if not np.all(np.isfinite(values)) or np.any(values < 0.0):
        raise ValueError("previous_result objective_terms must be finite and non-negative")
    if previous_result.quality_status not in {"accepted", "review", "rejected"}:
        raise ValueError("previous_result quality_status is malformed")
    if not isinstance(previous_result.reasons, tuple) or not all(
        isinstance(reason, str) for reason in previous_result.reasons
    ):
        raise ValueError("previous_result reasons must be a tuple of strings")


def _rays_for_yx(points_yx: np.ndarray, camera: object) -> tuple[np.ndarray, np.ndarray]:
    source, detector_center, u_axis, v_axis, _, principal, spacing = _camera_vectors(camera)
    points = np.asarray(points_yx, dtype=np.float64)
    uv = points[:, ::-1]
    detector_points = (
        detector_center[None, :]
        + ((uv[:, 0] - principal[0]) * spacing[1])[:, None] * u_axis[None, :]
        + ((uv[:, 1] - principal[1]) * spacing[0])[:, None] * v_axis[None, :]
    )
    directions = detector_points - source[None, :]
    detector_depths = np.linalg.norm(directions, axis=1)
    directions /= detector_depths[:, None]
    return directions, detector_depths


def _project_lps(points_lps: np.ndarray, camera: object) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    source, detector_center, u_axis, v_axis, n_axis, principal, spacing = _camera_vectors(camera)
    points = np.asarray(points_lps, dtype=np.float64)
    source_rays = points - source[None, :]
    denominator = source_rays @ n_axis
    plane_distance = float((detector_center - source) @ n_axis)
    scale = np.full(len(points), np.nan, dtype=np.float64)
    valid = np.abs(denominator) > 1e-12
    scale[valid] = plane_distance / denominator[valid]
    intersections = source[None, :] + scale[:, None] * source_rays
    local = intersections - detector_center[None, :]
    uv = np.column_stack(
        (
            principal[0] + (local @ u_axis) / spacing[1],
            principal[1] + (local @ v_axis) / spacing[0],
        )
    )
    depths = np.linalg.norm(source_rays, axis=1)
    detector_depths = np.linalg.norm(intersections - source[None, :], axis=1)
    return uv, depths, detector_depths


def _surface_vertices(heart_surface: np.ndarray | trimesh.Trimesh) -> np.ndarray:
    vertices = heart_surface.vertices if isinstance(heart_surface, trimesh.Trimesh) else heart_surface
    try:
        array = np.asarray(vertices, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("heart_surface must contain finite LPS vertices") from exc
    if array.ndim != 2 or array.shape[1:] != (3,) or len(array) == 0:
        raise ValueError("heart_surface must have non-empty shape N x 3")
    if not np.all(np.isfinite(array)):
        raise ValueError("heart_surface must contain finite LPS vertices")
    return array


def _surface_prior(heart_surface: np.ndarray | trimesh.Trimesh, camera: object) -> tuple[cKDTree, np.ndarray]:
    vertices = _surface_vertices(heart_surface)
    uv, depths, detector_depths = _project_lps(vertices, camera)
    valid = (
        np.all(np.isfinite(uv), axis=1)
        & np.isfinite(depths)
        & np.isfinite(detector_depths)
        & (depths > 0.0)
        & (depths < detector_depths)
    )
    if not np.any(valid):
        raise ValueError("heart_surface has no source-facing detector-visible prior")
    return cKDTree(uv[valid]), depths[valid]


def _initial_depths(
    points_yx: np.ndarray,
    camera: object,
    surface_tree: cKDTree,
    surface_depths: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    rays, detector_depths = _rays_for_yx(points_yx, camera)
    _, indices = surface_tree.query(np.asarray(points_yx, dtype=np.float64)[:, ::-1], k=min(8, surface_tree.n))
    indices = np.asarray(indices)
    if indices.ndim == 1:
        indices = indices[:, None]
    depths = np.min(surface_depths[indices], axis=1)
    if not np.all(np.isfinite(depths)) or np.any(depths <= 0.0) or np.any(depths >= detector_depths):
        raise ValueError("graph sample has no physically valid heart-surface prior")
    return depths, rays


def optimize_graph_depth(
    graph: VesselGraph,
    camera: object,
    heart_surface: np.ndarray | trimesh.Trimesh,
    previous_result: DepthOptimizationResult | None = None,
) -> DepthOptimizationResult:
    """Place every graph sample on its detector ray using shared node depths."""
    source = _camera_vectors(camera)[0]
    _validate_graph(graph, camera)
    _validate_previous_result(previous_result)
    surface_tree, surface_depths = _surface_prior(heart_surface, camera)

    variable_yx: list[np.ndarray] = []
    node_variable: dict[str, int] = {}
    for node in graph.nodes:
        node_variable[node.node_id] = len(variable_yx)
        variable_yx.append(np.asarray(node.yx, dtype=np.float64))

    edge_variables: dict[str, np.ndarray] = {}
    for edge in graph.edges:
        points_yx = np.asarray(edge.points_yx, dtype=np.float64)
        indices = [node_variable[edge.source_node_id]]
        for point_yx in points_yx[1:-1]:
            indices.append(len(variable_yx))
            variable_yx.append(point_yx)
        indices.append(node_variable[edge.target_node_id])
        edge_variables[edge.edge_id] = np.asarray(indices, dtype=np.int64)

    all_yx = np.asarray(variable_yx, dtype=np.float64)
    initial_depths, rays = _initial_depths(
        all_yx, camera, surface_tree, surface_depths
    )
    _, detector_depths = _rays_for_yx(all_yx, camera)
    epsilon = 1e-6
    lower = np.maximum(epsilon, initial_depths - 20.0)
    upper = np.minimum(detector_depths - epsilon, initial_depths + 20.0)
    if np.any(lower >= upper):
        raise ValueError("graph sample has an empty physical depth interval")

    coefficient_rows: list[dict[int, float]] = []
    targets: list[float] = []

    def add_constraint(entries: tuple[tuple[int, float], ...], target: float) -> None:
        coefficients: dict[int, float] = {}
        for column_index, value in entries:
            coefficients[column_index] = value
        coefficient_rows.append(coefficients)
        targets.append(target)

    variable_count = len(variable_yx)
    heart_scale = np.sqrt(1.0 / variable_count)
    for index, initial_depth in enumerate(initial_depths):
        add_constraint(
            ((index, heart_scale),), heart_scale * float(initial_depth)
        )

    first_pairs = [
        (int(left), int(right))
        for indices in edge_variables.values()
        for left, right in zip(indices[:-1], indices[1:], strict=True)
    ]
    if first_pairs:
        first_scale = np.sqrt(0.35 / len(first_pairs))
        for left, right in first_pairs:
            add_constraint(
                ((left, -first_scale), (right, first_scale)), 0.0
            )

    second_triples = [
        (int(left), int(middle), int(right))
        for indices in edge_variables.values()
        for left, middle, right in zip(indices[:-2], indices[1:-1], indices[2:], strict=True)
    ]
    if second_triples:
        second_scale = np.sqrt(0.20 / len(second_triples))
        for left, middle, right in second_triples:
            add_constraint(
                (
                    (left, second_scale),
                    (middle, -2.0 * second_scale),
                    (right, second_scale),
                ),
                0.0,
            )

    temporal_targets: list[tuple[int, float]] = []
    if previous_result is not None:
        for node in graph.nodes:
            key = _stable_key(node.node_id, node.topology_id)
            if key in previous_result.node_points_lps:
                previous_point = np.asarray(
                    previous_result.node_points_lps[key], dtype=np.float64
                )
                temporal_targets.append(
                    (node_variable[node.node_id], float(np.linalg.norm(previous_point - source)))
                )
        for edge in graph.edges:
            key = _stable_key(edge.edge_id, edge.topology_id)
            if key not in previous_result.points_lps_by_edge:
                continue
            previous_points = np.asarray(
                previous_result.points_lps_by_edge[key], dtype=np.float64
            )
            indices = edge_variables[edge.edge_id]
            if previous_points.shape != (len(indices), 3):
                raise ValueError(
                    f"previous_result edge {key} sample count does not match the current graph"
                )
            for variable_index, previous_point in zip(
                indices[1:-1], previous_points[1:-1], strict=True
            ):
                temporal_targets.append(
                    (int(variable_index), float(np.linalg.norm(previous_point - source)))
                )
    if temporal_targets:
        temporal_scale = np.sqrt(0.25 / len(temporal_targets))
        for variable_index, previous_depth in temporal_targets:
            add_constraint(
                ((variable_index, temporal_scale),),
                temporal_scale * previous_depth,
            )

    matrix_rows: list[int] = []
    matrix_columns: list[int] = []
    matrix_values: list[float] = []
    for row_index, coefficients in enumerate(coefficient_rows):
        for column_index in sorted(coefficients):
            matrix_rows.append(row_index)
            matrix_columns.append(column_index)
            matrix_values.append(coefficients[column_index])
    constraint_matrix = sparse.csr_matrix(
        (matrix_values, (matrix_rows, matrix_columns)),
        shape=(len(targets), variable_count),
        dtype=np.float64,
    )
    optimization = lsq_linear(
        constraint_matrix,
        np.asarray(targets, dtype=np.float64),
        bounds=(lower, upper),
        method="trf",
        lsmr_tol="auto",
    )
    success = getattr(optimization, "success", False)
    status = getattr(optimization, "status", None)
    if (
        not isinstance(success, (bool, np.bool_))
        or not bool(success)
        or not isinstance(status, Integral)
        or int(status) <= 0
    ):
        raise ValueError("depth optimization did not converge")
    try:
        optimized = np.asarray(optimization.x, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("depth optimization returned an invalid solution") from exc
    if optimized.shape != (variable_count,):
        raise ValueError("depth optimization returned an invalid solution shape")
    if not np.all(np.isfinite(optimized)):
        raise ValueError("depth optimization returned a non-finite solution")
    if np.any(optimized < lower) or np.any(optimized > upper):
        raise ValueError("depth optimization returned a solution outside bounds")
    variable_points = source[None, :] + optimized[:, None] * rays

    node_points = {
        _stable_key(node.node_id, node.topology_id): variable_points[node_variable[node.node_id]]
        for node in graph.nodes
    }
    edge_points = {
        _stable_key(edge.edge_id, edge.topology_id): variable_points[edge_variables[edge.edge_id]]
        for edge in graph.edges
    }

    reasons = sorted(
        {
            f"edge_step_exceeds_10mm:{_stable_key(edge.edge_id, edge.topology_id)}"
            for edge in graph.edges
            if np.any(
                np.linalg.norm(
                    np.diff(edge_points[_stable_key(edge.edge_id, edge.topology_id)], axis=0),
                    axis=1,
                )
                > 10.0
            )
        }
    )

    heart_term = float(np.mean((optimized - initial_depths) ** 2))
    first_values = np.asarray(
        [optimized[right] - optimized[left] for left, right in first_pairs],
        dtype=np.float64,
    )
    second_values = np.asarray(
        [
            optimized[left] - 2.0 * optimized[middle] + optimized[right]
            for left, middle, right in second_triples
        ],
        dtype=np.float64,
    )
    first_term = float(np.mean(first_values**2)) if len(first_values) else 0.0
    second_term = float(np.mean(second_values**2)) if len(second_values) else 0.0
    temporal_values = np.asarray(
        [optimized[index] - target for index, target in temporal_targets],
        dtype=np.float64,
    )
    temporal_term = float(np.mean(temporal_values**2)) if len(temporal_values) else 0.0
    objective_terms = {
        "heart_surface_distance": heart_term,
        "first_derivative_smoothness": first_term,
        "second_derivative_smoothness": second_term,
        "junction_equality_penalty": 0.0,
        "temporal_depth_penalty": temporal_term,
        "weighted_total": (
            heart_term
            + 0.35 * first_term
            + 0.20 * second_term
            + 0.25 * temporal_term
        ),
    }

    return DepthOptimizationResult(
        points_lps_by_edge=edge_points,
        node_points_lps=node_points,
        objective_terms=objective_terms,
        quality_status="rejected" if reasons else "accepted",
        reasons=tuple(reasons),
    )


def edge_depth_audit(
    result: DepthOptimizationResult,
    heart_surface: np.ndarray | trimesh.Trimesh,
) -> dict[str, dict[str, float]]:
    """Measure deterministic per-edge depth reconstruction quality."""
    surface_tree = cKDTree(_surface_vertices(heart_surface))
    audit: dict[str, dict[str, float]] = {}
    for edge_key, raw_points in result.points_lps_by_edge.items():
        points = np.asarray(raw_points, dtype=np.float64)
        steps = np.linalg.norm(np.diff(points, axis=0), axis=1)
        if len(points) >= 3:
            left = np.diff(points, axis=0)[:-1]
            right = np.diff(points, axis=0)[1:]
            left_length = np.linalg.norm(left, axis=1)
            right_length = np.linalg.norm(right, axis=1)
            valid = (left_length > 1e-12) & (right_length > 1e-12)
            curvature = np.zeros(len(left), dtype=np.float64)
            if np.any(valid):
                left_unit = left[valid] / left_length[valid, None]
                right_unit = right[valid] / right_length[valid, None]
                scale = 0.5 * (left_length[valid] + right_length[valid])
                curvature[valid] = np.linalg.norm(right_unit - left_unit, axis=1) / scale
        else:
            curvature = np.zeros(1, dtype=np.float64)
        surface_distances, _ = surface_tree.query(points, k=1)
        audit[edge_key] = {
            "edge_step_max_mm": float(np.max(steps)) if len(steps) else 0.0,
            "edge_step_p95_mm": float(np.percentile(steps, 95.0)) if len(steps) else 0.0,
            "curvature_p95": float(np.percentile(curvature, 95.0)),
            "surface_distance_p95_mm": float(np.percentile(surface_distances, 95.0)),
        }
    return audit


def reprojection_rms_px(
    graph: VesselGraph,
    result: DepthOptimizationResult,
    camera: object,
) -> float:
    """Return RMS detector error for optimized edge samples."""
    squared_errors: list[np.ndarray] = []
    for edge in graph.edges:
        key = _stable_key(edge.edge_id, edge.topology_id)
        projected_uv, _, _ = _project_lps(result.points_lps_by_edge[key], camera)
        expected_uv = np.asarray(edge.points_yx, dtype=np.float64)[:, ::-1]
        squared_errors.append(np.sum((projected_uv - expected_uv) ** 2, axis=1))
    if not squared_errors:
        return 0.0
    return float(np.sqrt(np.mean(np.concatenate(squared_errors))))
