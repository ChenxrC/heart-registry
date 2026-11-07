from __future__ import annotations

import copy
import sys
from pathlib import Path

import numpy as np
import pytest
import trimesh


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import xa_vessel_mesh as vessel_mesh_module
from xa_vessel_mesh import build_watertight_vessel_surface


def _edge(
    topology_id: str,
    source: str,
    target: str,
    start: tuple[float, float, float],
    end: tuple[float, float, float],
    radii: tuple[float, ...],
) -> dict[str, object]:
    return {
        "topology_id": topology_id,
        "source_topology_id": source,
        "target_topology_id": target,
        "points_lps_mm": np.linspace(start, end, len(radii)).tolist(),
        "radii_mm": list(radii),
    }


def synthetic_y_graph_frame() -> dict[str, object]:
    junction = (0.0, 0.0, 0.0)
    left = (-8.0, 8.0, 0.0)
    right = (8.0, 8.0, 0.0)
    lower = (0.0, -10.0, 0.0)
    return {
        "frame_index": 7,
        "quality_status": "accepted",
        "component_count": 1,
        "nodes": [
            {"topology_id": "node-junction", "point_lps_mm": list(junction)},
            {"topology_id": "node-left", "point_lps_mm": list(left)},
            {"topology_id": "node-right", "point_lps_mm": list(right)},
            {"topology_id": "node-lower", "point_lps_mm": list(lower)},
        ],
        "edges": [
            _edge(
                "edge-left",
                "node-junction",
                "node-left",
                junction,
                left,
                (2.0, 1.8, 1.6, 1.4, 1.2),
            ),
            _edge(
                "edge-right",
                "node-junction",
                "node-right",
                junction,
                right,
                (1.8, 1.7, 1.6, 1.5, 1.3),
            ),
            _edge(
                "edge-lower",
                "node-junction",
                "node-lower",
                junction,
                lower,
                (1.9, 1.75, 1.6, 1.45, 1.25),
            ),
        ],
        # Deliberately malformed: schema v2 must be authoritative.
        "paths": [{"points_lps_mm": "unused", "radius_mm": -1.0}],
    }


def _single_edge_frame(end: tuple[float, float, float]) -> dict[str, object]:
    return {
        "quality_status": "accepted",
        "nodes": [
            {"topology_id": "node-start", "point_lps_mm": [0.0, 0.0, 0.0]},
            {"topology_id": "node-end", "point_lps_mm": list(end)},
        ],
        "edges": [
            {
                "topology_id": "edge-main",
                "source_topology_id": "node-start",
                "target_topology_id": "node-end",
                "points_lps_mm": [[0.0, 0.0, 0.0], list(end)],
                "radii_mm": [1.0, 1.0],
            }
        ],
    }


def _legacy_y_frame() -> dict[str, object]:
    return {
        "paths": [
            {
                "points_lps_mm": [[-6.0, 5.0, 0.0], [0.0, 0.0, 0.0]],
                "radius_mm": 1.0,
            },
            {
                "points_lps_mm": [[0.0, 0.0, 0.0], [6.0, 5.0, 0.0]],
                "radius_mm": 1.0,
            },
            {
                "points_lps_mm": [[0.0, 0.0, 0.0], [0.0, -7.0, 0.0]],
                "radius_mm": 1.0,
            },
        ]
    }


def _translate_frame(
    frame: dict[str, object], offset: tuple[float, float, float]
) -> dict[str, object]:
    translated = copy.deepcopy(frame)
    delta = np.asarray(offset, dtype=np.float64)
    for node in translated["nodes"]:
        node["point_lps_mm"] = (
            np.asarray(node["point_lps_mm"], dtype=np.float64) + delta
        ).tolist()
    for edge in translated["edges"]:
        edge["points_lps_mm"] = (
            np.asarray(edge["points_lps_mm"], dtype=np.float64) + delta
        ).tolist()
    return translated


def test_y_graph_implicit_surface_covers_all_lps_branches_and_is_manifold(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    merge_tolerances: list[float] = []
    real_merge = vessel_mesh_module._merge_vertices

    def tracked_merge(
        vertices: np.ndarray, faces: np.ndarray, tolerance_mm: float
    ) -> tuple[np.ndarray, np.ndarray]:
        merge_tolerances.append(tolerance_mm)
        return real_merge(vertices, faces, tolerance_mm)

    monkeypatch.setattr(vessel_mesh_module, "_merge_vertices", tracked_merge)
    mesh = build_watertight_vessel_surface(synthetic_y_graph_frame())

    assert isinstance(mesh, trimesh.Trimesh)
    assert mesh.bounds[0, 0] < -9.0  # left branch endpoint plus radius
    assert mesh.bounds[1, 0] > 9.0  # right branch endpoint plus radius
    assert mesh.bounds[0, 1] < -11.0  # lower branch endpoint plus radius
    assert mesh.bounds[1, 1] > 9.0  # both upper branch endpoints plus radius
    assert mesh.bounds[0, 2] < -1.8
    assert mesh.bounds[1, 2] > 1.8
    assert len(mesh.split(only_watertight=False)) == 1
    assert mesh.is_watertight
    assert mesh.is_winding_consistent
    assert np.all(np.bincount(mesh.edges_unique_inverse) == 2)
    assert np.all(np.isfinite(mesh.vertex_normals))
    np.testing.assert_allclose(np.linalg.norm(mesh.vertex_normals, axis=1), 1.0)
    assert merge_tolerances == [0.05]


def test_clean_mesh_merges_a_known_short_manifold_edge_and_preserves_manifold() -> None:
    vertices = np.asarray(
        [
            [0.0, 0.0, 1.0],
            [0.01, 0.0, 1.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, 0.0, -1.0],
        ],
        dtype=np.float64,
    )
    faces = np.asarray(
        [
            [0, 1, 2],
            [0, 2, 3],
            [0, 3, 4],
            [0, 4, 1],
            [5, 2, 1],
            [5, 3, 2],
            [5, 4, 3],
            [5, 1, 4],
        ],
        dtype=np.int64,
    )

    mesh = vessel_mesh_module._clean_mesh(vertices, faces)

    assert len(mesh.vertices) == 5
    assert len(mesh.faces) == 6
    assert mesh.is_watertight
    assert mesh.is_winding_consistent
    assert np.all(np.bincount(mesh.edges_unique_inverse) == 2)
    assert not np.any(
        (mesh.faces[:, 0] == mesh.faces[:, 1])
        | (mesh.faces[:, 1] == mesh.faces[:, 2])
        | (mesh.faces[:, 0] == mesh.faces[:, 2])
    )
    assert len(np.unique(np.sort(mesh.faces, axis=1), axis=0)) == len(mesh.faces)
    assert np.min(np.linalg.norm(mesh.vertices - [0.005, 0.0, 1.0], axis=1)) < 1e-12


def test_y_graph_surface_preserves_lps_translation_and_axis_bounds() -> None:
    frame = synthetic_y_graph_frame()
    translation = np.asarray([31.0, -17.0, 9.0])

    baseline = build_watertight_vessel_surface(frame)
    translated = build_watertight_vessel_surface(
        _translate_frame(frame, tuple(translation))
    )

    np.testing.assert_allclose(
        translated.bounds - baseline.bounds,
        np.broadcast_to(translation, (2, 3)),
        atol=0.36,
    )
    np.testing.assert_allclose(translated.extents, baseline.extents, atol=0.36)


def test_larger_junction_radius_expands_surface_cross_section() -> None:
    small = synthetic_y_graph_frame()
    large = copy.deepcopy(small)
    for edge in small["edges"]:
        edge["radii_mm"] = [0.8] * len(edge["radii_mm"])
    for edge in large["edges"]:
        edge["radii_mm"] = [0.8] * len(edge["radii_mm"])
    large["edges"][0]["radii_mm"][0] = 3.0

    small_mesh = build_watertight_vessel_surface(small)
    large_mesh = build_watertight_vessel_surface(large)

    assert large_mesh.extents[2] > small_mesh.extents[2] + 3.5
    assert large_mesh.bounds[0, 2] < -2.7
    assert large_mesh.bounds[1, 2] > 2.7


def test_schema_v2_is_authoritative_over_legacy_paths() -> None:
    mesh = build_watertight_vessel_surface(synthetic_y_graph_frame())

    assert mesh.is_watertight
    assert np.max(mesh.bounds) < 20.0


def test_malformed_schema_v2_cannot_fall_back_to_valid_legacy_paths() -> None:
    frame = synthetic_y_graph_frame()
    frame["paths"] = [{
        "points_lps_mm": [[0.0, 0.0, 0.0], [4.0, 0.0, 0.0]],
        "radius_mm": 1.0,
    }]
    frame["edges"][0]["points_lps_mm"][0] = [0.25, 0.0, 0.0]
    before = copy.deepcopy(frame)

    with pytest.raises(ValueError):
        build_watertight_vessel_surface(frame)

    assert frame == before


@pytest.mark.parametrize(
    "case",
    [
        "empty-id",
        "duplicate-node-id",
        "duplicate-edge-id",
        "missing-reference",
        "nonfinite-point",
        "nonpositive-radius",
        "radius-count-mismatch",
        "endpoint-mismatch",
    ],
)
def test_malformed_schema_v2_graph_fails_closed(case: str) -> None:
    frame = synthetic_y_graph_frame()
    if case == "empty-id":
        frame["nodes"][0]["topology_id"] = ""
    elif case == "duplicate-node-id":
        frame["nodes"][1]["topology_id"] = "node-junction"
    elif case == "duplicate-edge-id":
        frame["edges"][1]["topology_id"] = "edge-left"
    elif case == "missing-reference":
        frame["edges"][0]["source_topology_id"] = "node-missing"
    elif case == "nonfinite-point":
        frame["edges"][0]["points_lps_mm"][2][1] = float("inf")
    elif case == "nonpositive-radius":
        frame["edges"][0]["radii_mm"][2] = 0.0
    elif case == "radius-count-mismatch":
        frame["edges"][0]["radii_mm"].pop()
    elif case == "endpoint-mismatch":
        frame["edges"][0]["points_lps_mm"][-1] = [-7.5, 8.0, 0.0]

    with pytest.raises(ValueError):
        build_watertight_vessel_surface(frame)


def test_rejected_frame_fails_closed() -> None:
    frame = synthetic_y_graph_frame()
    frame["quality_status"] = "rejected"
    frame["display_source_frame_index"] = 6

    with pytest.raises(ValueError):
        build_watertight_vessel_surface(frame)


def test_legacy_paths_without_quality_status_build_one_watertight_component() -> None:
    mesh = build_watertight_vessel_surface(_legacy_y_frame())

    assert len(mesh.split(only_watertight=False)) == 1
    assert mesh.is_watertight
    assert mesh.bounds[0, 0] < -6.5
    assert mesh.bounds[1, 0] > 6.5
    assert mesh.bounds[0, 1] < -7.5


def test_explicitly_rejected_legacy_frame_fails_closed() -> None:
    frame = _legacy_y_frame()
    frame["quality_status"] = "rejected"

    with pytest.raises(ValueError):
        build_watertight_vessel_surface(frame)


def test_schema_v2_without_accepted_quality_status_fails_closed() -> None:
    frame = synthetic_y_graph_frame()
    frame.pop("quality_status")

    with pytest.raises(ValueError):
        build_watertight_vessel_surface(frame)


def test_signed_zero_legacy_junction_endpoints_are_one_component() -> None:
    frame = _legacy_y_frame()
    frame["quality_status"] = "accepted"
    frame["paths"][0]["points_lps_mm"][-1] = [-0.0, 0.0, -0.0]
    frame["paths"][1]["points_lps_mm"][0] = [0.0, -0.0, 0.0]

    mesh = build_watertight_vessel_surface(frame)

    assert len(mesh.split(only_watertight=False)) == 1
    assert mesh.is_watertight


@pytest.mark.parametrize(
    ("case", "bad_value"),
    [
        ("coordinate-bool", True),
        ("coordinate-string", "1.0"),
        ("radius-bool", True),
        ("radius-string", "2.0"),
        ("legacy-radius-bool", True),
        ("legacy-radius-string", "2.0"),
        ("voxel-bool", True),
        ("voxel-string", "0.35"),
    ],
)
def test_bool_and_numeric_string_geometry_values_fail_closed(
    case: str, bad_value: object
) -> None:
    frame = synthetic_y_graph_frame()
    voxel_size: object = 0.35
    if case.startswith("coordinate"):
        frame["nodes"][1]["point_lps_mm"][0] = bad_value
        frame["edges"][0]["points_lps_mm"][-1][0] = bad_value
    elif case.startswith("radius"):
        frame["edges"][0]["radii_mm"][0] = bad_value
    elif case.startswith("legacy-radius"):
        frame = _legacy_y_frame()
        frame["paths"][0]["radius_mm"] = bad_value
    else:
        voxel_size = bad_value

    with pytest.raises(ValueError):
        build_watertight_vessel_surface(frame, voxel_size_mm=voxel_size)


@pytest.mark.parametrize(
    ("end", "voxel_size_mm"),
    [
        ((1.0e308, 0.0, 0.0), 0.35),
        ((8.0, 0.0, 0.0), 1.0e-300),
        ((2_100_000.0, 2_100_000.0, 2_100_000.0), 1.0),
    ],
)
def test_unsafe_grid_dimensions_fail_before_field_allocation(
    end: tuple[float, float, float],
    voxel_size_mm: float,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_full = vessel_mesh_module.np.full
    attempted_field_shapes: list[tuple[int, ...]] = []

    def guarded_full(shape: object, *args: object, **kwargs: object) -> np.ndarray:
        if isinstance(shape, tuple):
            attempted_field_shapes.append(shape)
            raise AssertionError("unsafe field allocation was attempted")
        return real_full(shape, *args, **kwargs)

    monkeypatch.setattr(vessel_mesh_module.np, "full", guarded_full)

    with pytest.raises(ValueError):
        build_watertight_vessel_surface(
            _single_edge_frame(end), voxel_size_mm=voxel_size_mm
        )

    assert attempted_field_shapes == []


def test_semantically_disconnected_graph_fails_closed() -> None:
    frame = {
        "quality_status": "accepted",
        # Deliberately stale metadata: connectivity must be computed from IDs.
        "component_count": 1,
        "nodes": [
            {"topology_id": "node-a", "point_lps_mm": [0.0, 0.0, 0.0]},
            {"topology_id": "node-b", "point_lps_mm": [5.0, 0.0, 0.0]},
            {"topology_id": "node-c", "point_lps_mm": [20.0, 0.0, 0.0]},
            {"topology_id": "node-d", "point_lps_mm": [25.0, 0.0, 0.0]},
        ],
        "edges": [
            _edge("edge-ab", "node-a", "node-b", (0, 0, 0), (5, 0, 0), (1, 1)),
            _edge("edge-cd", "node-c", "node-d", (20, 0, 0), (25, 0, 0), (1, 1)),
        ],
    }

    with pytest.raises(ValueError):
        build_watertight_vessel_surface(frame)
