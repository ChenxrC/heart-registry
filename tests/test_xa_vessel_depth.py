from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import trimesh
from scipy import sparse


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "tools"))

from build_dynamic_vessel_3d import Camera
import xa_vessel_depth as depth_module
from xa_vessel_depth import (
    DepthOptimizationResult,
    edge_depth_audit,
    optimize_graph_depth,
    reprojection_rms_px,
)
from xa_vessel_topology import VesselEdge, VesselGraph, VesselNode


def synthetic_camera() -> Camera:
    return Camera(
        source=np.asarray([0.0, 0.0, -1000.0]),
        detector_center=np.asarray([0.0, 0.0, 0.0]),
        u_axis=np.asarray([1.0, 0.0, 0.0]),
        v_axis=np.asarray([0.0, 1.0, 0.0]),
        n_axis=np.asarray([0.0, 0.0, 1.0]),
        principal_uv=np.asarray([31.5, 31.5]),
        row_spacing_mm=4.0,
        col_spacing_mm=4.0,
        rows=64,
        columns=64,
        sdd_mm=1000.0,
    )


def spherical_heart() -> trimesh.Trimesh:
    mesh = trimesh.creation.icosphere(subdivisions=3, radius=120.0)
    mesh.apply_translation([0.0, 0.0, -350.0])
    return mesh


def synthetic_projected_y_graph() -> VesselGraph:
    junction = VesselNode(
        "junction-local",
        "junction",
        (31.0, 31.0),
        topology_id="node-001",
    )
    upper_left = VesselNode(
        "upper-left-local",
        "root",
        (23.0, 23.0),
        topology_id="node-002",
    )
    upper_right = VesselNode(
        "upper-right-local",
        "endpoint",
        (23.0, 39.0),
        topology_id="node-003",
    )
    lower = VesselNode(
        "lower-local",
        "endpoint",
        (43.0, 31.0),
        topology_id="node-004",
    )
    branches = (
        ("edge-left", "edge-001", upper_left, np.arange(31, 22, -1), np.arange(31, 22, -1)),
        ("edge-right", "edge-002", upper_right, np.arange(31, 22, -1), np.arange(31, 40)),
        ("edge-lower", "edge-003", lower, np.arange(31, 44), np.full(13, 31)),
    )
    edges = tuple(
        VesselEdge(
            edge_id=local_id,
            source_node_id=junction.node_id,
            target_node_id=target.node_id,
            points_yx=np.column_stack((ys, xs)).astype(np.float64),
            confidence=1.0,
            topology_id=topology_id,
            radius_px=np.ones(len(ys), dtype=np.float64),
        )
        for local_id, topology_id, target, ys, xs in branches
    )
    return VesselGraph(
        frame_index=0,
        nodes=(junction, upper_left, upper_right, lower),
        edges=edges,
        root_node_id=upper_left.node_id,
        component_count=1,
    )


def single_edge_graph(
    topology_id: str = "edge-010",
    node_topology_ids: tuple[str, str] = ("node-010", "node-011"),
) -> VesselGraph:
    root = VesselNode(
        "root-local", "root", (31.0, 20.0), topology_id=node_topology_ids[0]
    )
    endpoint = VesselNode(
        "endpoint-local",
        "endpoint",
        (31.0, 28.0),
        topology_id=node_topology_ids[1],
    )
    points_yx = np.column_stack((np.full(9, 31), np.arange(20, 29))).astype(np.float64)
    edge = VesselEdge(
        "edge-local",
        root.node_id,
        endpoint.node_id,
        points_yx,
        1.0,
        topology_id=topology_id,
        radius_px=np.ones(len(points_yx), dtype=np.float64),
    )
    return VesselGraph(0, (root, endpoint), (edge,), root.node_id, 1)


def self_loop_graph(sample_count: int) -> VesselGraph:
    node = VesselNode(
        "loop-node", "root", (31.0, 20.0), topology_id="loop-node-stable"
    )
    points = (
        np.asarray([[31.0, 20.0], [31.0, 20.0]], dtype=np.float64)
        if sample_count == 2
        else np.asarray(
            [[31.0, 20.0], [31.0, 24.0], [31.0, 20.0]],
            dtype=np.float64,
        )
    )
    edge = VesselEdge(
        "loop-edge",
        node.node_id,
        node.node_id,
        points,
        1.0,
        topology_id="loop-edge-stable",
        radius_px=np.ones(len(points), dtype=np.float64),
    )
    return VesselGraph(0, (node,), (edge,), node.node_id, 1)


def point_on_detector_ray(camera: Camera, yx: np.ndarray, depth_mm: float) -> np.ndarray:
    y, x = np.asarray(yx, dtype=np.float64)
    detector_point = (
        camera.detector_center
        + (x - camera.principal_uv[0]) * camera.col_spacing_mm * camera.u_axis
        + (y - camera.principal_uv[1]) * camera.row_spacing_mm * camera.v_axis
    )
    direction = detector_point - camera.source
    direction /= np.linalg.norm(direction)
    return camera.source + depth_mm * direction


def exact_depth_surface(camera: Camera, points_yx: np.ndarray, depths_mm: np.ndarray) -> np.ndarray:
    points = [
        point_on_detector_ray(camera, yx, float(depth))
        for yx, depth in zip(points_yx, depths_mm, strict=True)
    ]
    return np.repeat(np.asarray(points, dtype=np.float64), 8, axis=0)


def test_joint_depth_keeps_reprojection_and_shared_junction() -> None:
    graph = synthetic_projected_y_graph()
    camera = synthetic_camera()

    result = optimize_graph_depth(graph, camera, spherical_heart())

    junction = result.node_points_lps["node-001"]
    touching = [result.points_lps_by_edge[str(edge.topology_id)][0] for edge in graph.edges]
    assert max(np.linalg.norm(point - junction) for point in touching) < 1e-6
    assert reprojection_rms_px(graph, result, camera) < 0.25


def test_bounded_objective_moves_depths_within_physical_and_initial_limits() -> None:
    camera = synthetic_camera()
    graph = single_edge_graph()
    initial_depths = np.asarray([650.0] * 4 + [690.0] * 5)
    points_yx = graph.edges[0].points_yx
    surface = exact_depth_surface(camera, points_yx, initial_depths)

    result = optimize_graph_depth(graph, camera, surface)

    optimized_points = result.points_lps_by_edge["edge-010"]
    optimized_depths = np.linalg.norm(optimized_points - camera.source[None, :], axis=1)
    detector_depths = np.asarray(
        [
            np.linalg.norm(
                camera.detector_center
                + (yx[1] - camera.principal_uv[0]) * camera.col_spacing_mm * camera.u_axis
                + (yx[0] - camera.principal_uv[1]) * camera.row_spacing_mm * camera.v_axis
                - camera.source
            )
            for yx in points_yx
        ]
    )
    assert np.max(np.abs(optimized_depths - initial_depths)) > 0.5
    assert np.all(optimized_depths > 0.0)
    assert np.all(optimized_depths < detector_depths)
    assert np.all(np.abs(optimized_depths - initial_depths) <= 20.0 + 1e-8)


def test_stable_topology_reduces_temporal_change_without_coupling_new_ids() -> None:
    camera = synthetic_camera()
    graph = single_edge_graph()
    points_yx = graph.edges[0].points_yx
    prior_surface = exact_depth_surface(camera, points_yx, np.full(9, 650.0))
    current_surface = exact_depth_surface(camera, points_yx, np.full(9, 690.0))
    previous = optimize_graph_depth(graph, camera, prior_surface)
    without_previous = optimize_graph_depth(graph, camera, current_surface)

    stable = optimize_graph_depth(graph, camera, current_surface, previous)
    new_graph = single_edge_graph(
        "edge-new", node_topology_ids=("node-new-root", "node-new-endpoint")
    )
    new_result = optimize_graph_depth(new_graph, camera, current_surface, previous)

    previous_points = previous.points_lps_by_edge["edge-010"]
    baseline_points = without_previous.points_lps_by_edge["edge-010"]
    stable_points = stable.points_lps_by_edge["edge-010"]
    assert np.mean(np.linalg.norm(stable_points - previous_points, axis=1)) < np.mean(
        np.linalg.norm(baseline_points - previous_points, axis=1)
    )
    np.testing.assert_allclose(
        new_result.points_lps_by_edge["edge-new"], baseline_points, atol=1e-9
    )


def test_large_prior_jump_preserves_one_complete_edge_without_hard_split() -> None:
    camera = synthetic_camera()
    graph = single_edge_graph()
    initial_depths = np.asarray([650.0] * 4 + [710.0] * 5)
    surface = exact_depth_surface(camera, graph.edges[0].points_yx, initial_depths)

    result = optimize_graph_depth(graph, camera, surface)
    audit = edge_depth_audit(result, surface)

    assert list(result.points_lps_by_edge) == ["edge-010"]
    assert len(result.points_lps_by_edge["edge-010"]) == len(graph.edges[0].points_yx)
    assert audit["edge-010"]["edge_step_max_mm"] > 14.0


def test_edge_step_over_ten_mm_is_rejected_with_audit_instead_of_split() -> None:
    camera = synthetic_camera()
    graph = single_edge_graph()
    initial_depths = np.asarray([650.0] * 4 + [710.0] * 5)
    surface = exact_depth_surface(camera, graph.edges[0].points_yx, initial_depths)

    result = optimize_graph_depth(graph, camera, surface)
    audit = edge_depth_audit(result, surface)

    assert audit["edge-010"]["edge_step_max_mm"] > 10.0
    assert result.quality_status == "rejected"
    assert result.reasons == ("edge_step_exceeds_10mm:edge-010",)
    assert len(result.points_lps_by_edge) == 1


def test_malformed_graph_camera_surface_and_previous_result_fail_closed() -> None:
    camera = synthetic_camera()
    graph = single_edge_graph()
    surface = exact_depth_surface(camera, graph.edges[0].points_yx, np.full(9, 680.0))

    missing_target = replace(graph.edges[0], target_node_id="missing-node")
    with pytest.raises(ValueError, match="missing node"):
        optimize_graph_depth(replace(graph, edges=(missing_target,)), camera, surface)

    with pytest.raises(ValueError, match="orthonormal"):
        optimize_graph_depth(
            graph, replace(camera, v_axis=camera.u_axis.copy()), surface
        )
    with pytest.raises(ValueError, match="dimensions"):
        optimize_graph_depth(graph, replace(camera, rows=0), surface)
    with pytest.raises(ValueError, match="heart_surface.*finite"):
        optimize_graph_depth(graph, camera, np.asarray([[np.nan, 0.0, 0.0]]))

    malformed_previous = DepthOptimizationResult(
        points_lps_by_edge={"edge-010": np.full((9, 3), np.nan)},
        node_points_lps={},
        objective_terms={"weighted_total": 0.0},
        quality_status="accepted",
        reasons=(),
    )
    with pytest.raises(ValueError, match="previous_result.*finite"):
        optimize_graph_depth(graph, camera, surface, malformed_previous)

    valid_previous = optimize_graph_depth(graph, camera, surface)
    mismatched_previous = replace(
        valid_previous,
        points_lps_by_edge={
            "edge-010": valid_previous.points_lps_by_edge["edge-010"][:-1]
        },
    )
    with pytest.raises(ValueError, match="previous_result.*sample count"):
        optimize_graph_depth(graph, camera, surface, mismatched_previous)


@pytest.mark.parametrize(
    "endpoint_mutation", ("source_mismatch", "target_mismatch", "swapped")
)
def test_optimize_graph_depth_rejects_edge_points_not_anchored_to_nodes(
    endpoint_mutation: str,
) -> None:
    camera = synthetic_camera()
    graph = single_edge_graph()
    points = np.asarray(graph.edges[0].points_yx, dtype=np.float64).copy()
    if endpoint_mutation == "source_mismatch":
        points[0, 0] += 1e-9
    elif endpoint_mutation == "target_mismatch":
        points[-1, 0] += 1e-9
    else:
        points = points[::-1].copy()
    malformed_edge = replace(graph.edges[0], points_yx=points)
    malformed_graph = replace(graph, edges=(malformed_edge,))
    surface = exact_depth_surface(
        camera, graph.edges[0].points_yx, np.full(9, 680.0)
    )

    with pytest.raises(ValueError, match="points_yx.*endpoint"):
        optimize_graph_depth(malformed_graph, camera, surface)


@pytest.mark.parametrize(
    "failure_mode",
    ("unsuccessful", "bad_status", "wrong_shape", "nonfinite", "outside_bounds"),
)
def test_invalid_lsq_linear_result_fails_closed(
    monkeypatch: pytest.MonkeyPatch, failure_mode: str
) -> None:
    camera = synthetic_camera()
    graph = single_edge_graph()
    surface = exact_depth_surface(
        camera, graph.edges[0].points_yx, np.full(9, 680.0)
    )
    real_lsq_linear = depth_module.lsq_linear

    def invalid_lsq_linear(*args: object, **kwargs: object) -> object:
        result = real_lsq_linear(*args, **kwargs)
        if failure_mode == "unsuccessful":
            result.success = False
        elif failure_mode == "bad_status":
            result.status = 0
        elif failure_mode == "wrong_shape":
            result.x = result.x[:-1]
        elif failure_mode == "nonfinite":
            result.x = result.x.copy()
            result.x[0] = np.nan
        else:
            result.x = result.x.copy()
            _, upper = kwargs["bounds"]
            result.x[0] = upper[0] + 1.0
        return result

    monkeypatch.setattr(depth_module, "lsq_linear", invalid_lsq_linear)

    with pytest.raises(ValueError, match="depth optimization"):
        optimize_graph_depth(graph, camera, surface)


def test_depth_optimizer_builds_a_sparse_constraint_matrix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    camera = synthetic_camera()
    graph = single_edge_graph()
    surface = exact_depth_surface(
        camera, graph.edges[0].points_yx, np.full(9, 680.0)
    )
    observed: dict[str, object] = {}
    real_lsq_linear = depth_module.lsq_linear

    def capture_matrix(matrix: object, *args: object, **kwargs: object) -> object:
        observed["is_sparse"] = sparse.issparse(matrix)
        observed["shape"] = matrix.shape
        observed["nnz"] = matrix.nnz if sparse.issparse(matrix) else np.count_nonzero(matrix)
        return real_lsq_linear(matrix, *args, **kwargs)

    monkeypatch.setattr(depth_module, "lsq_linear", capture_matrix)

    optimize_graph_depth(graph, camera, surface)

    assert observed == {"is_sparse": True, "shape": (24, 9), "nnz": 46}


@pytest.mark.parametrize("sample_count", (2, 3))
def test_sparse_self_loop_rows_preserve_legacy_dense_last_assignment_semantics(
    sample_count: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    camera = synthetic_camera()
    graph = self_loop_graph(sample_count)
    edge_points = graph.edges[0].points_yx
    surface = exact_depth_surface(
        camera, edge_points, np.full(sample_count, 680.0)
    )
    captured: dict[str, np.ndarray] = {}
    real_lsq_linear = depth_module.lsq_linear

    def capture_matrix(matrix: object, *args: object, **kwargs: object) -> object:
        captured["matrix"] = matrix.toarray()
        return real_lsq_linear(matrix, *args, **kwargs)

    monkeypatch.setattr(depth_module, "lsq_linear", capture_matrix)

    optimize_graph_depth(graph, camera, surface)

    if sample_count == 2:
        expected = np.asarray(
            [[1.0], [np.sqrt(0.35)]], dtype=np.float64
        )
    else:
        heart_scale = np.sqrt(0.5)
        first_scale = np.sqrt(0.35 / 2.0)
        second_scale = np.sqrt(0.20)
        expected = np.asarray(
            [
                [heart_scale, 0.0],
                [0.0, heart_scale],
                [-first_scale, first_scale],
                [first_scale, -first_scale],
                [second_scale, -2.0 * second_scale],
            ],
            dtype=np.float64,
        )
    np.testing.assert_allclose(captured["matrix"], expected, atol=0.0, rtol=0.0)


def test_objective_terms_and_edge_qc_are_finite_and_deterministic() -> None:
    camera = synthetic_camera()
    graph = single_edge_graph()
    surface = exact_depth_surface(
        camera,
        graph.edges[0].points_yx,
        np.asarray([650.0] * 4 + [690.0] * 5),
    )

    first = optimize_graph_depth(graph, camera, surface)
    second = optimize_graph_depth(graph, camera, surface)
    first_audit = edge_depth_audit(first, surface)
    second_audit = edge_depth_audit(second, surface)

    assert set(first.objective_terms) == {
        "heart_surface_distance",
        "first_derivative_smoothness",
        "second_derivative_smoothness",
        "junction_equality_penalty",
        "temporal_depth_penalty",
        "weighted_total",
    }
    objective_values = np.asarray(list(first.objective_terms.values()))
    assert np.all(np.isfinite(objective_values))
    assert np.all(objective_values >= 0.0)
    assert first.objective_terms["weighted_total"] == pytest.approx(
        first.objective_terms["heart_surface_distance"]
        + 0.35 * first.objective_terms["first_derivative_smoothness"]
        + 0.20 * first.objective_terms["second_derivative_smoothness"]
        + 0.50 * first.objective_terms["junction_equality_penalty"]
        + 0.25 * first.objective_terms["temporal_depth_penalty"]
    )
    assert first.objective_terms == second.objective_terms
    assert first_audit == second_audit
    assert set(first_audit["edge-010"]) == {
        "edge_step_max_mm",
        "edge_step_p95_mm",
        "curvature_p95",
        "surface_distance_p95_mm",
    }
    assert np.all(np.isfinite(list(first_audit["edge-010"].values())))
