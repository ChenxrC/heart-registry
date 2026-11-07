from __future__ import annotations

from collections import Counter
from dataclasses import FrozenInstanceError, fields, replace
import inspect
import sys
from pathlib import Path

import numpy as np
import pytest
from skimage import draw, morphology


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TOOLS = PROJECT_ROOT / "tools"
sys.path.insert(0, str(TOOLS))

import xa_vessel_topology as topology_module
from xa_vessel_topology import (
    GapCandidate,
    TopologyMetrics,
    VesselEdge,
    VesselGraph,
    VesselNode,
    extract_vessel_graph,
    find_gap_candidates,
    mask_topology_metrics,
)


def _two_segment_graph(
    distance_px: float,
    *,
    left_outward_yx: tuple[float, float] = (0.0, 1.0),
) -> VesselGraph:
    right_gap_x = 24.0 + distance_px
    if not float(right_gap_x).is_integer():
        raise ValueError("test graph requires an integral right-gap coordinate")
    left_gap = np.asarray((40, 24), dtype=np.int64)
    right_gap = np.asarray((40, int(right_gap_x)), dtype=np.int64)
    left_outer = np.asarray((40, 8), dtype=np.int64)
    right_outer = right_gap + np.asarray((0, 12), dtype=np.int64)
    left_outward = np.asarray(left_outward_yx, dtype=np.float64)
    if (
        left_outward.shape != (2,)
        or not np.array_equal(left_outward, np.rint(left_outward))
        or np.max(np.abs(left_outward)) != 1.0
    ):
        raise ValueError("test tangent must be one integral 8-neighbor step")
    outward_step = left_outward.astype(np.int64)
    left_previous = left_gap - outward_step
    if tuple(outward_step) == (0, -1):
        lead_y, lead_x = draw.line(40, 8, 39, 24)
        left_points = np.vstack(
            (
                np.column_stack((lead_y, lead_x)),
                np.asarray(((40, 25), (40, 24)), dtype=np.int64),
            )
        )
    else:
        lead_y, lead_x = draw.line(
            int(left_outer[0]),
            int(left_outer[1]),
            int(left_previous[0]),
            int(left_previous[1]),
        )
        left_points = np.vstack(
            (np.column_stack((lead_y, lead_x)), left_gap[np.newaxis, :])
        )
    right_y, right_x = draw.line(
        int(right_gap[0]),
        int(right_gap[1]),
        int(right_outer[0]),
        int(right_outer[1]),
    )
    return VesselGraph(
        frame_index=0,
        nodes=(
            VesselNode("left-outer", "root", tuple(left_outer.astype(float))),
            VesselNode("left-gap", "endpoint", tuple(left_gap.astype(float))),
            VesselNode("right-gap", "endpoint", tuple(right_gap.astype(float))),
            VesselNode("right-outer", "endpoint", tuple(right_outer.astype(float))),
        ),
        edges=(
            VesselEdge(
                "left-edge",
                "left-outer",
                "left-gap",
                left_points.astype(np.float64),
                1.0,
            ),
            VesselEdge(
                "right-edge",
                "right-gap",
                "right-outer",
                np.column_stack((right_y, right_x)).astype(np.float64),
                1.0,
            ),
        ),
        root_node_id="left-outer",
        component_count=2,
    )


def _gap_evidence(
    evidence_value: float = 1.0,
    support_value: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    evidence = np.full((80, 80), evidence_value, dtype=np.float64)
    support = np.full((80, 80), support_value, dtype=np.float64)
    return evidence, support


def _y_branch_mask(*, dy: int = 0, dx: int = 0) -> np.ndarray:
    mask = np.zeros((96, 96), dtype=bool)
    for start, end in (
        ((70 + dy, 48 + dx), (45 + dy, 48 + dx)),
        ((45 + dy, 48 + dx), (25 + dy, 30 + dx)),
        ((45 + dy, 48 + dx), (25 + dy, 66 + dx)),
    ):
        rr, cc = draw.line(*start, *end)
        mask[rr, cc] = True
    return morphology.dilation(mask, morphology.disk(2))


def _constant_forward_flow(*, dy: float = 0.0, dx: float = 0.0) -> np.ndarray:
    flow = np.zeros((96, 96, 2), dtype=np.float64)
    flow[..., 0] = dx
    flow[..., 1] = dy
    return flow


def _empty_graph(frame_index: int) -> VesselGraph:
    return VesselGraph(frame_index, (), (), "", 0)


def test_mask_topology_metrics_counts_disconnected_skeletons() -> None:
    mask = np.zeros((16, 16), dtype=bool)
    mask[2, 2:7] = True
    mask[10, 10] = True

    assert mask_topology_metrics(mask) == TopologyMetrics(
        component_count=2,
        largest_component_fraction=5.0 / 6.0,
        endpoint_count=2,
        junction_count=0,
        skeleton_length_px=6,
    )


def test_y_branch_becomes_one_graph_with_three_edges() -> None:
    mask = np.zeros((64, 64), dtype=bool)
    rr, cc = draw.line(50, 32, 25, 32); mask[rr, cc] = True
    rr, cc = draw.line(25, 32, 10, 15); mask[rr, cc] = True
    rr, cc = draw.line(25, 32, 10, 49); mask[rr, cc] = True
    mask = morphology.dilation(mask, morphology.disk(2))

    graph = extract_vessel_graph(mask, root_hint_yx=(50.0, 32.0))

    assert graph.component_count == 1
    assert len([node for node in graph.nodes if node.kind == "junction"]) == 1
    assert len(graph.edges) == 3
    assert {edge.source_node_id for edge in graph.edges} | {
        edge.target_node_id for edge in graph.edges
    } == {node.node_id for node in graph.nodes}


def test_graph_ids_and_edge_points_are_deterministic_and_root_first() -> None:
    mask = np.zeros((64, 64), dtype=bool)
    rr, cc = draw.line(50, 32, 25, 32); mask[rr, cc] = True
    rr, cc = draw.line(25, 32, 10, 15); mask[rr, cc] = True
    rr, cc = draw.line(25, 32, 10, 49); mask[rr, cc] = True
    mask = morphology.dilation(mask, morphology.disk(2))

    first = extract_vessel_graph(mask, root_hint_yx=(50.0, 32.0))
    second = extract_vessel_graph(mask, root_hint_yx=(50.0, 32.0))

    assert first.root_node_id == first.nodes[0].node_id
    assert first.nodes[0].kind == "root"
    assert [(node.node_id, node.kind, node.yx) for node in first.nodes] == [
        (node.node_id, node.kind, node.yx) for node in second.nodes
    ]
    assert [
        (edge.edge_id, edge.source_node_id, edge.target_node_id, edge.confidence)
        for edge in first.edges
    ] == [
        (edge.edge_id, edge.source_node_id, edge.target_node_id, edge.confidence)
        for edge in second.edges
    ]
    for first_edge, second_edge in zip(first.edges, second.edges, strict=True):
        np.testing.assert_array_equal(first_edge.points_yx, second_edge.points_yx)


def test_translated_y_branch_keeps_node_and_edge_topology_ids_with_dx_dy_flow() -> None:
    first = replace(
        extract_vessel_graph(_y_branch_mask(), root_hint_yx=(70.0, 48.0)),
        frame_index=0,
    )
    second = replace(
        extract_vessel_graph(
            _y_branch_mask(dy=3, dx=2), root_hint_yx=(73.0, 50.0)
        ),
        frame_index=1,
    )

    tracked = topology_module.track_vessel_graphs(
        [first, second], [_constant_forward_flow(dy=3.0, dx=2.0)]
    )

    assert [node.topology_id for node in tracked[0].nodes] == [
        node.topology_id for node in tracked[1].nodes
    ]
    assert [edge.topology_id for edge in tracked[0].edges] == [
        edge.topology_id for edge in tracked[1].edges
    ]


@pytest.mark.parametrize("missed_frames", [1, 2])
def test_short_occlusion_recovers_stable_ids(missed_frames: int) -> None:
    first = replace(
        extract_vessel_graph(_y_branch_mask(), root_hint_yx=(70.0, 48.0)),
        frame_index=0,
    )
    reappeared = replace(
        extract_vessel_graph(
            _y_branch_mask(dx=missed_frames + 1),
            root_hint_yx=(70.0, 48.0 + missed_frames + 1),
        ),
        frame_index=missed_frames + 1,
    )
    graphs = [first] + [
        _empty_graph(frame_index) for frame_index in range(1, missed_frames + 1)
    ] + [reappeared]
    flows = [
        _constant_forward_flow(dx=1.0) for _ in range(len(graphs) - 1)
    ]

    tracked = topology_module.track_vessel_graphs(graphs, flows)

    assert {node.topology_id for node in tracked[-1].nodes} == {
        node.topology_id for node in tracked[0].nodes
    }
    assert {edge.topology_id for edge in tracked[-1].edges} == {
        edge.topology_id for edge in tracked[0].edges
    }
    assert all(node.visibility == "visible" for node in tracked[-1].nodes)
    assert all(edge.visibility == "visible" for edge in tracked[-1].edges)


def test_third_missed_frame_retires_ids_and_later_observation_is_new() -> None:
    first = replace(
        extract_vessel_graph(_y_branch_mask(), root_hint_yx=(70.0, 48.0)),
        frame_index=0,
    )
    reappeared = replace(
        extract_vessel_graph(_y_branch_mask(dx=4), root_hint_yx=(70.0, 52.0)),
        frame_index=4,
    )
    tracked = topology_module.track_vessel_graphs(
        [first, _empty_graph(1), _empty_graph(2), _empty_graph(3), reappeared],
        [_constant_forward_flow(dx=1.0) for _ in range(4)],
    )

    assert tracked[3].nodes == ()
    assert tracked[3].edges == ()
    assert {node.topology_id for node in tracked[0].nodes}.isdisjoint(
        node.topology_id for node in tracked[4].nodes
    )
    assert {edge.topology_id for edge in tracked[0].edges}.isdisjoint(
        edge.topology_id for edge in tracked[4].edges
    )


@pytest.mark.parametrize(
    ("distance_px", "should_match"),
    [(8.0, True), (8.0 + 1e-6, False)],
)
def test_node_tracking_uses_inclusive_eight_pixel_gate(
    distance_px: float, should_match: bool
) -> None:
    first = VesselGraph(
        0, (VesselNode("first-root", "root", (20.0, 20.0)),), (), "first-root", 1
    )
    second = VesselGraph(
        1,
        (VesselNode("second-root", "root", (20.0, 20.0 + distance_px)),),
        (),
        "second-root",
        1,
    )

    tracked = topology_module.track_vessel_graphs(
        [first, second], [_constant_forward_flow()]
    )

    current_visible = next(
        node for node in tracked[1].nodes if node.visibility == "visible"
    )
    assert (current_visible.topology_id == tracked[0].nodes[0].topology_id) is should_match


def test_out_of_detector_prediction_cannot_match() -> None:
    first = VesselGraph(
        0, (VesselNode("first-root", "root", (20.0, 94.0)),), (), "first-root", 1
    )
    second = VesselGraph(
        1, (VesselNode("second-root", "root", (20.0, 94.0)),), (), "second-root", 1
    )

    tracked = topology_module.track_vessel_graphs(
        [first, second], [_constant_forward_flow(dx=4.0)]
    )

    visible = next(node for node in tracked[1].nodes if node.visibility == "visible")
    assert visible.topology_id != tracked[0].nodes[0].topology_id
    assert all(
        0.0 <= node.yx[0] < 96.0 and 0.0 <= node.yx[1] < 96.0
        for node in tracked[1].nodes
    )
    assert all(node.visibility != "occluded" for node in tracked[1].nodes)
    assert tracked[1].qc["retired_nodes"] == 1


def test_true_branch_appearance_is_new_topology_and_reported() -> None:
    root = VesselNode("root", "root", (20.0, 20.0))
    existing = VesselNode("existing", "endpoint", (20.0, 30.0))
    appeared = VesselNode("appeared", "endpoint", (30.0, 20.0))
    horizontal = np.column_stack(draw.line(20, 20, 20, 30)).astype(np.float64)
    vertical = np.column_stack(draw.line(20, 20, 30, 20)).astype(np.float64)
    first = VesselGraph(
        0,
        (root, existing),
        (VesselEdge("horizontal", "root", "existing", horizontal, 1.0),),
        "root",
        1,
    )
    second = VesselGraph(
        1,
        (root, existing, appeared),
        (
            VesselEdge("horizontal", "root", "existing", horizontal, 1.0),
            VesselEdge("vertical", "root", "appeared", vertical, 1.0),
        ),
        "root",
        1,
    )

    tracked = topology_module.track_vessel_graphs(
        [first, second], [_constant_forward_flow()]
    )

    first_node_ids = {node.topology_id for node in tracked[0].nodes}
    appeared_node = next(node for node in tracked[1].nodes if node.node_id == "appeared")
    assert appeared_node.topology_id not in first_node_ids
    assert tracked[1].qc == {
        "matched_node_fraction": 2.0 / 3.0,
        "matched_edge_fraction": 0.5,
        "new_nodes": 1,
        "retired_nodes": 0,
        "topology_changed": True,
        "quality_status": "review",
    }


def test_tracked_edge_endpoints_orientation_shape_and_radii_are_stable() -> None:
    nodes = (
        VesselNode("root", "root", (20.0, 20.0)),
        VesselNode("tip", "endpoint", (20.0, 30.0)),
    )
    points = np.column_stack(draw.line(20, 20, 20, 30)).astype(np.float64)
    first = VesselGraph(
        0,
        nodes,
        (VesselEdge("edge", "root", "tip", points, 1.0),),
        "root",
        1,
    )
    second = VesselGraph(
        1,
        tuple(reversed(nodes)),
        (VesselEdge("edge", "tip", "root", points[::-1], 1.0),),
        "root",
        1,
    )

    tracked = topology_module.track_vessel_graphs(
        [first, second], [_constant_forward_flow()]
    )

    for graph in tracked:
        local_to_topology = {
            node.node_id: node.topology_id for node in graph.nodes
        }
        edge = graph.edges[0]
        assert edge.points_yx.shape == (64, 2)
        assert edge.radius_px is not None and edge.radius_px.shape == (64,)
        assert np.all(np.isfinite(edge.radius_px))
        assert np.all(edge.radius_px > 0.0)
        assert local_to_topology[edge.source_node_id] < local_to_topology[edge.target_node_id]
    assert tracked[0].edges[0].topology_id == tracked[1].edges[0].topology_id
    np.testing.assert_allclose(
        tracked[0].edges[0].points_yx, tracked[1].edges[0].points_yx
    )


def test_extraction_samples_positive_radius_for_every_raw_edge_point() -> None:
    graph = extract_vessel_graph(_y_branch_mask(), root_hint_yx=(70.0, 48.0))

    for edge in graph.edges:
        assert edge.radius_px is not None
        assert edge.radius_px.shape == (len(edge.points_yx),)
        assert np.all(np.isfinite(edge.radius_px))
        assert np.all(edge.radius_px > 0.0)


def test_node_ties_and_parallel_edge_ties_ignore_input_tuple_order() -> None:
    points = np.column_stack(draw.line(20, 20, 20, 30)).astype(np.float64)
    nodes = (
        VesselNode("root", "root", (10.0, 10.0)),
        VesselNode("a", "endpoint", (20.0, 20.0)),
        VesselNode("b", "endpoint", (20.0, 20.0)),
        VesselNode("tip", "endpoint", (20.0, 30.0)),
    )
    edges = (
        VesselEdge("parallel-a", "a", "tip", points, 1.0),
        VesselEdge("parallel-b", "a", "tip", points.copy(), 1.0),
    )
    forward = VesselGraph(0, nodes, edges, "root", 2)
    reversed_order = VesselGraph(
        1, tuple(reversed(nodes)), tuple(reversed(edges)), "root", 2
    )

    first_run = topology_module.track_vessel_graphs(
        [forward, reversed_order], [_constant_forward_flow()]
    )
    second_run = topology_module.track_vessel_graphs(
        [
            replace(forward, nodes=tuple(reversed(nodes)), edges=tuple(reversed(edges))),
            replace(reversed_order, nodes=nodes, edges=edges),
        ],
        [_constant_forward_flow()],
    )

    for run in (first_run, second_run):
        assert len({node.topology_id for node in run[1].nodes}) == len(run[1].nodes)
    assert {
        node.node_id: node.topology_id for node in first_run[1].nodes
    } == {node.node_id: node.topology_id for node in second_run[1].nodes}
    assert {
        edge.edge_id: edge.topology_id for edge in first_run[1].edges
    } == {edge.edge_id: edge.topology_id for edge in second_run[1].edges}


def test_first_frame_edge_topology_ids_follow_root_first_graph_order() -> None:
    nodes = (
        VesselNode("root", "root", (20.0, 10.0)),
        VesselNode("middle", "junction", (20.0, 20.0)),
        VesselNode("tip", "endpoint", (20.0, 30.0)),
    )
    root_edge = VesselEdge(
        "z-root-edge",
        "root",
        "middle",
        np.column_stack(draw.line(20, 10, 20, 20)).astype(np.float64),
        1.0,
    )
    child_edge = VesselEdge(
        "a-child-edge",
        "middle",
        "tip",
        np.column_stack(draw.line(20, 20, 20, 30)).astype(np.float64),
        1.0,
    )

    tracked = topology_module.track_vessel_graphs(
        [VesselGraph(0, nodes, (child_edge, root_edge), "root", 1)], []
    )[0]

    edge_topologies = {edge.edge_id: edge.topology_id for edge in tracked.edges}
    assert edge_topologies == {
        "z-root-edge": "edge-v1-000000",
        "a-child-edge": "edge-v1-000001",
    }


def test_extracted_parallel_edges_have_unique_local_ids_and_can_be_tracked() -> None:
    mask = np.zeros((32, 32), dtype=bool)
    paths = (
        draw.line(16, 5, 16, 10),
        draw.line(16, 20, 16, 25),
        (
            np.asarray((16, 15, 14, 13, 12, 12, 12, 12, 12, 13, 14, 15, 16)),
            np.asarray((10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 20, 20)),
        ),
        (
            np.asarray((16, 17, 18, 19, 20, 20, 20, 20, 20, 19, 18, 17, 16)),
            np.asarray((10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 20, 20)),
        ),
    )
    for rr, cc in paths:
        mask[rr, cc] = True

    graph = extract_vessel_graph(mask, root_hint_yx=(16.0, 5.0))
    local_edge_ids = [edge.edge_id for edge in graph.edges]

    assert len(local_edge_ids) == len(set(local_edge_ids))
    tracked = topology_module.track_vessel_graphs([graph], [])
    assert len(tracked[0].edges) == len(graph.edges)


@pytest.mark.parametrize(
    ("graphs", "flows", "message"),
    [
        ([_empty_graph(0), _empty_graph(1)], [], "exactly"),
        ([_empty_graph(0), _empty_graph(1)], [np.zeros((8, 8))], "shape"),
        ([_empty_graph(0), _empty_graph(1)], [np.zeros((8, 8, 3))], "shape"),
        (
            [_empty_graph(0), _empty_graph(1)],
            [np.full((8, 8, 2), "bad")],
            "numeric",
        ),
        (
            [_empty_graph(0), _empty_graph(1)],
            [np.full((8, 8, 2), np.nan)],
            "finite",
        ),
        (
            [_empty_graph(0), _empty_graph(1)],
            [np.zeros((0, 8, 2))],
            "positive",
        ),
    ],
)
def test_tracking_rejects_malformed_flow_inputs(
    graphs: list[VesselGraph], flows: list[np.ndarray], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        topology_module.track_vessel_graphs(graphs, flows)


def test_empty_and_single_node_frames_have_finite_qc() -> None:
    single = VesselGraph(
        1, (VesselNode("root", "root", (4.0, 4.0)),), (), "root", 1
    )

    tracked = topology_module.track_vessel_graphs(
        [_empty_graph(0), single], [_constant_forward_flow()]
    )

    for graph in tracked:
        assert graph.qc is not None
        assert np.isfinite(graph.qc["matched_node_fraction"])
        assert np.isfinite(graph.qc["matched_edge_fraction"])
        assert graph.qc["quality_status"] == "accepted"


def test_pure_loop_synthesizes_root_and_closed_self_loop() -> None:
    mask = np.zeros((16, 16), dtype=bool)
    loop_pixels = [
        (2, 4),
        (3, 5),
        (4, 6),
        (5, 5),
        (6, 4),
        (5, 3),
        (4, 2),
        (3, 3),
    ]
    mask[tuple(zip(*loop_pixels))] = True

    graph = extract_vessel_graph(mask, root_hint_yx=(2.0, 4.0))

    assert graph.component_count == 1
    assert [(node.node_id, node.kind, node.yx) for node in graph.nodes] == [
        ("node-0000", "root", (2.0, 4.0))
    ]
    assert graph.root_node_id == "node-0000"
    assert len(graph.edges) == 1
    edge = graph.edges[0]
    assert (edge.edge_id, edge.source_node_id, edge.target_node_id) == (
        "node-0000--node-0000",
        "node-0000",
        "node-0000",
    )
    assert edge.points_yx.shape == (len(loop_pixels) + 1, 2)
    np.testing.assert_array_equal(edge.points_yx[0], (2.0, 4.0))
    np.testing.assert_array_equal(edge.points_yx[-1], (2.0, 4.0))
    assert {tuple(point) for point in edge.points_yx[:-1]} == set(loop_pixels)


def test_junction_chain_returning_to_itself_is_a_self_loop_edge() -> None:
    mask = np.zeros((20, 20), dtype=bool)
    pixels = [
        (8, 8),
        (7, 9),
        (6, 10),
        (7, 11),
        (8, 12),
        (9, 11),
        (10, 10),
        (9, 9),
        (8, 7),
        (8, 6),
        (8, 5),
    ]
    mask[tuple(zip(*pixels))] = True

    graph = extract_vessel_graph(mask, root_hint_yx=(8.0, 5.0))

    assert [(node.node_id, node.kind, node.yx) for node in graph.nodes] == [
        ("node-0000", "root", (8.0, 5.0)),
        ("node-0001", "junction", (8.0, 8.0)),
    ]
    assert [
        (edge.edge_id, edge.source_node_id, edge.target_node_id)
        for edge in graph.edges
    ] == [
        ("node-0000--node-0001", "node-0000", "node-0001"),
        ("node-0001--node-0001", "node-0001", "node-0001"),
    ]
    self_loop = graph.edges[1]
    assert self_loop.points_yx.shape == (9, 2)
    np.testing.assert_array_equal(self_loop.points_yx[0], (8.0, 8.0))
    np.testing.assert_array_equal(self_loop.points_yx[-1], (8.0, 8.0))


def test_node_ids_follow_breadth_first_levels_then_siblings_and_components() -> None:
    mask = np.zeros((32, 32), dtype=bool)
    pixels = [
        (14, 10),
        (13, 10),
        (12, 10),
        (11, 10),
        (10, 10),
        (9, 9),
        (8, 8),
        (7, 7),
        (9, 11),
        (8, 12),
        (7, 13),
        (6, 14),
        (5, 13),
        (4, 12),
        (3, 11),
        (5, 15),
        (4, 16),
        (3, 17),
        (18, 25),
        (19, 25),
        (20, 25),
        (20, 2),
        (21, 2),
        (22, 2),
    ]
    mask[tuple(zip(*pixels))] = True

    graph = extract_vessel_graph(mask, root_hint_yx=(14.0, 10.0))

    assert [(node.node_id, node.kind, node.yx) for node in graph.nodes] == [
        ("node-0000", "root", (14.0, 10.0)),
        ("node-0001", "junction", (10.0, 10.0)),
        ("node-0002", "junction", (6.0, 14.0)),
        ("node-0003", "endpoint", (7.0, 7.0)),
        ("node-0004", "endpoint", (3.0, 11.0)),
        ("node-0005", "endpoint", (3.0, 17.0)),
        ("node-0006", "endpoint", (18.0, 25.0)),
        ("node-0007", "endpoint", (20.0, 25.0)),
        ("node-0008", "endpoint", (20.0, 2.0)),
        ("node-0009", "endpoint", (22.0, 2.0)),
    ]


def test_gap_candidate_contract_is_frozen_exact_and_crossing_is_derived() -> None:
    assert [field.name for field in fields(GapCandidate)] == [
        "start_node_id",
        "end_node_id",
        "distance_px",
        "angle_deg",
        "path_yx",
        "mean_evidence",
        "temporal_support",
        "score",
        "decision",
        "reasons",
    ]
    candidate = GapCandidate(
        "left-gap",
        "right-gap",
        12.0,
        0.0,
        np.asarray(((40, 24), (40, 36)), dtype=np.int64),
        1.0,
        1.0,
        0.0,
        "rejected",
        ("exclusion_intersection",),
    )

    assert candidate.crosses_exclusion is True
    assert "crosses_exclusion" not in {field.name for field in fields(candidate)}
    with pytest.raises(FrozenInstanceError):
        candidate.crosses_exclusion = False  # type: ignore[misc]

    find_signature = inspect.signature(find_gap_candidates)
    assert list(find_signature.parameters) == [
        "graph",
        "evidence",
        "temporal_support",
        "max_review_gap_px",
        "exclusion_mask",
    ]
    assert find_signature.parameters["max_review_gap_px"].default == 24
    assert (
        find_signature.parameters["exclusion_mask"].kind
        is inspect.Parameter.KEYWORD_ONLY
    )
    assert find_signature.parameters["exclusion_mask"].default is None


@pytest.mark.parametrize(
    ("distance_px", "expected_decision"),
    [(12.0, "accepted"), (13.0, "review"), (24.0, "review")],
)
def test_gap_distance_boundaries_are_inclusive_and_audited(
    distance_px: float,
    expected_decision: str,
) -> None:
    evidence, support = _gap_evidence()

    candidates = find_gap_candidates(
        _two_segment_graph(distance_px), evidence, support
    )

    assert len(candidates) == 1
    assert candidates[0].distance_px == pytest.approx(distance_px)
    assert candidates[0].decision == expected_decision


def test_gap_beyond_review_limit_is_absent() -> None:
    evidence, support = _gap_evidence()

    assert find_gap_candidates(_two_segment_graph(25.0), evidence, support) == []


@pytest.mark.parametrize(
    ("left_outward_yx", "expected_angle", "expected_decision"),
    [
        ((-1.0, 1.0), 45.0, "accepted"),
        ((-1.0, 0.0), 90.0, "rejected"),
    ],
)
def test_worse_endpoint_angle_uses_exact_forty_five_degree_gate(
    left_outward_yx: tuple[float, float],
    expected_angle: float,
    expected_decision: str,
) -> None:
    evidence, support = _gap_evidence()

    candidate = find_gap_candidates(
        _two_segment_graph(12.0, left_outward_yx=left_outward_yx),
        evidence,
        support,
    )[0]

    assert candidate.angle_deg == pytest.approx(expected_angle, abs=1e-5)
    assert candidate.decision == expected_decision
    assert ("direction_misaligned" in candidate.reasons) is (
        expected_decision == "rejected"
    )


def test_opposed_endpoint_is_rejected_instead_of_cross_connected() -> None:
    evidence, support = _gap_evidence()

    candidate = find_gap_candidates(
        _two_segment_graph(12.0, left_outward_yx=(0.0, -1.0)),
        evidence,
        support,
    )[0]

    assert candidate.angle_deg == pytest.approx(180.0)
    assert candidate.decision == "rejected"
    assert "direction_misaligned" in candidate.reasons


@pytest.mark.parametrize(
    ("evidence_value", "support_value", "expected_decision", "reason"),
    [
        (0.45, 2.0 / 3.0, "accepted", None),
        (0.45 - 1e-6, 1.0, "rejected", "insufficient_evidence"),
        (1.0, 2.0 / 3.0 - 1e-6, "rejected", "insufficient_temporal_support"),
    ],
)
def test_gap_evidence_and_temporal_support_use_exact_inclusive_gates(
    evidence_value: float,
    support_value: float,
    expected_decision: str,
    reason: str | None,
) -> None:
    evidence, support = _gap_evidence(evidence_value, support_value)

    candidate = find_gap_candidates(
        _two_segment_graph(12.0), evidence, support
    )[0]

    assert candidate.mean_evidence == pytest.approx(evidence_value)
    assert candidate.temporal_support == pytest.approx(support_value)
    assert candidate.decision == expected_decision
    if reason is not None:
        assert reason in candidate.reasons


def test_endpoint_values_cannot_rescue_an_unsupported_gap_interior() -> None:
    graph = _two_segment_graph(12.0)
    evidence, support = _gap_evidence(0.44, 1.0)
    evidence[40, 24] = 1.0
    evidence[40, 36] = 1.0

    candidate = find_gap_candidates(graph, evidence, support)[0]

    assert candidate.mean_evidence == pytest.approx(0.44)
    assert candidate.decision == "rejected"
    assert "insufficient_evidence" in candidate.reasons


def test_exclusion_barrier_cannot_be_routed_through() -> None:
    graph = _two_segment_graph(12.0)
    evidence, support = _gap_evidence()
    exclusion = np.zeros(evidence.shape, dtype=bool)
    exclusion[32:49, 30] = True

    candidate = find_gap_candidates(
        graph, evidence, support, exclusion_mask=exclusion
    )[0]

    assert candidate.decision == "rejected"
    assert candidate.crosses_exclusion
    assert "exclusion_intersection" in candidate.reasons


def test_fully_excluded_local_window_still_produces_finite_rejected_audit() -> None:
    graph = _two_segment_graph(12.0)
    evidence, support = _gap_evidence()
    exclusion = np.ones(evidence.shape, dtype=bool)

    candidate = find_gap_candidates(
        graph, evidence, support, exclusion_mask=exclusion
    )[0]

    assert candidate.decision == "rejected"
    assert candidate.crosses_exclusion
    assert "route_unavailable" in candidate.reasons
    assert np.isfinite(candidate.score)


def test_routing_uses_exact_weighted_cost_inside_padded_local_window(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    graph = _two_segment_graph(12.0)
    evidence, support = _gap_evidence(0.8, 0.6)
    routed: dict[str, object] = {}

    def capture_route(
        cost: np.ndarray,
        start: tuple[int, int],
        end: tuple[int, int],
        *,
        fully_connected: bool,
        geometric: bool,
    ) -> tuple[list[tuple[int, int]], float]:
        routed["cost"] = cost.copy()
        routed["start"] = start
        routed["end"] = end
        routed["fully_connected"] = fully_connected
        routed["geometric"] = geometric
        rr, cc = draw.line(start[0], start[1], end[0], end[1])
        return list(zip(rr.tolist(), cc.tolist(), strict=True)), float(
            np.sum(cost[rr, cc])
        )

    monkeypatch.setattr(topology_module, "route_through_array", capture_route)

    candidate = find_gap_candidates(graph, evidence, support)[0]

    local_cost = routed["cost"]
    assert isinstance(local_cost, np.ndarray)
    assert local_cost.shape == (17, 29)
    assert routed["start"] == (8, 8)
    assert routed["end"] == (8, 20)
    assert routed["fully_connected"] is True
    assert routed["geometric"] is True
    assert local_cost[8, 14] == pytest.approx(
        0.45 * (1.0 - 0.8) + 0.25 * (1.0 - 0.6)
    )
    np.testing.assert_array_equal(candidate.path_yx[[0, -1]], ((40, 24), (40, 36)))


def test_cost_image_uses_all_four_exact_weights_and_infinite_exclusions() -> None:
    evidence = np.asarray([[0.2, 0.2]], dtype=np.float64)
    support = np.asarray([[0.3, 0.3]], dtype=np.float64)
    direction = np.asarray([[0.4, 0.4]], dtype=np.float64)
    curvature = np.asarray([[0.5, 0.5]], dtype=np.float64)
    exclusion = np.asarray([[False, True]])

    cost = topology_module._gap_cost_image(
        evidence, support, direction, curvature, exclusion
    )

    assert cost[0, 0] == pytest.approx(
        0.45 * 0.8 + 0.25 * 0.7 + 0.20 * 0.4 + 0.10 * 0.5
    )
    assert np.isinf(cost[0, 1])


def test_reciprocal_nearest_pairing_is_deterministic_without_endpoint_reuse() -> None:
    nodes = (
        VesselNode("endpoint-a", "endpoint", (40.0, 20.0)),
        VesselNode("endpoint-b", "endpoint", (40.0, 30.0)),
        VesselNode("endpoint-c", "endpoint", (40.0, 40.0)),
    )
    evidence, support = _gap_evidence()
    first_graph = VesselGraph(0, nodes, (), "endpoint-a", 3)
    second_graph = VesselGraph(0, tuple(reversed(nodes)), (), "endpoint-a", 3)

    first = find_gap_candidates(first_graph, evidence, support)
    second = find_gap_candidates(second_graph, evidence, support)

    assert [(item.start_node_id, item.end_node_id) for item in first] == [
        ("endpoint-a", "endpoint-b")
    ]
    assert [(item.start_node_id, item.end_node_id) for item in second] == [
        ("endpoint-a", "endpoint-b")
    ]
    assert len({first[0].start_node_id, first[0].end_node_id}) == 2
    assert first[0].decision == "rejected"
    assert "direction_unavailable" in first[0].reasons
    assert np.isfinite(first[0].angle_deg)
    assert np.isfinite(first[0].score)


def test_candidate_finding_rejects_malformed_inputs_without_mutation() -> None:
    graph = _two_segment_graph(12.0)
    evidence, support = _gap_evidence()
    exclusion = np.zeros(evidence.shape, dtype=bool)
    evidence_before = evidence.copy()
    support_before = support.copy()
    exclusion_before = exclusion.copy()
    edge_points_before = [edge.points_yx.copy() for edge in graph.edges]

    find_gap_candidates(graph, evidence, support, exclusion_mask=exclusion)

    np.testing.assert_array_equal(evidence, evidence_before)
    np.testing.assert_array_equal(support, support_before)
    np.testing.assert_array_equal(exclusion, exclusion_before)
    for edge, before in zip(graph.edges, edge_points_before, strict=True):
        np.testing.assert_array_equal(edge.points_yx, before)

    invalid_evidence = evidence.copy()
    invalid_evidence[0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        find_gap_candidates(graph, invalid_evidence, support)
    with pytest.raises(ValueError, match="same 2D shape"):
        find_gap_candidates(graph, evidence, support[:-1])
    with pytest.raises(ValueError, match="exclusion_mask"):
        find_gap_candidates(graph, evidence, support, exclusion_mask=exclusion[:-1])

    malformed_graph = VesselGraph(
        graph.frame_index,
        graph.nodes,
        (
            VesselEdge(
                graph.edges[0].edge_id,
                graph.edges[0].source_node_id,
                graph.edges[0].target_node_id,
                np.zeros((3,), dtype=np.float64),
                graph.edges[0].confidence,
            ),
            graph.edges[1],
        ),
        graph.root_node_id,
        graph.component_count,
    )
    with pytest.raises(ValueError, match="points_yx"):
        find_gap_candidates(malformed_graph, evidence, support)


def test_synthetic_candidate_decision_summary_is_one_each() -> None:
    cases = [
        (_two_segment_graph(12.0), *_gap_evidence()),
        (_two_segment_graph(13.0), *_gap_evidence()),
        (_two_segment_graph(12.0), *_gap_evidence(0.44, 1.0)),
    ]

    decisions = Counter(
        find_gap_candidates(graph, evidence, support)[0].decision
        for graph, evidence, support in cases
    )

    assert decisions == Counter({"accepted": 1, "review": 1, "rejected": 1})


@pytest.mark.parametrize("max_review_gap_px", [0, 25, True])
def test_review_gap_limit_cannot_relax_the_one_through_twenty_four_cap(
    max_review_gap_px: object,
) -> None:
    evidence, support = _gap_evidence()

    with pytest.raises(ValueError, match="1 through 24"):
        find_gap_candidates(
            _two_segment_graph(12.0),
            evidence,
            support,
            max_review_gap_px=max_review_gap_px,  # type: ignore[arg-type]
        )


def test_finder_rejects_an_angle_even_infinitesimally_above_forty_five() -> None:
    coordinate_delta = 4e-10
    left_y, left_x = draw.line(40, 8, 40, 24)
    right_y, right_x = draw.line(52, 36, 64, 48)
    graph = VesselGraph(
        frame_index=0,
        nodes=(
            VesselNode("left-outer", "root", (40.0, 8.0)),
            VesselNode("left-gap", "endpoint", (40.0, 24.0)),
            VesselNode(
                "right-gap", "endpoint", (52.0 + coordinate_delta, 36.0)
            ),
            VesselNode("right-outer", "endpoint", (64.0, 48.0)),
        ),
        edges=(
            VesselEdge(
                "left-edge",
                "left-outer",
                "left-gap",
                np.column_stack((left_y, left_x)).astype(np.float64),
                1.0,
            ),
            VesselEdge(
                "right-edge",
                "right-gap",
                "right-outer",
                np.column_stack((right_y, right_x)).astype(np.float64),
                1.0,
            ),
        ),
        root_node_id="left-outer",
        component_count=2,
    )
    evidence, support = _gap_evidence()

    candidate = find_gap_candidates(graph, evidence, support)[0]

    assert 45.0 < candidate.angle_deg < 45.0 + 2e-9
    assert candidate.decision == "rejected"
    assert "direction_misaligned" in candidate.reasons


@pytest.mark.parametrize("malformation", ["out-of-bounds", "reversed", "mismatch"])
def test_graph_validation_rejects_invalid_ordered_edge_geometry(
    malformation: str,
) -> None:
    graph = _two_segment_graph(12.0)
    edge = graph.edges[0]
    points = edge.points_yx.copy()
    if malformation == "out-of-bounds":
        points[1] = (-1.0, points[1, 1])
        message = "inside the detector"
    elif malformation == "reversed":
        points = points[::-1].copy()
        message = "ordered endpoint"
    else:
        points[-1] = points[-2]
        message = "8-connected"
    malformed = replace(
        graph,
        edges=(replace(edge, points_yx=points), graph.edges[1]),
    )
    evidence, support = _gap_evidence()

    with pytest.raises(ValueError, match=message):
        find_gap_candidates(malformed, evidence, support)


def test_graph_validation_preserves_valid_self_loops_and_junction_centroids() -> None:
    mask = np.zeros((20, 20), dtype=bool)
    pixels = [
        (8, 8),
        (7, 9),
        (6, 10),
        (7, 11),
        (8, 12),
        (9, 11),
        (10, 10),
        (9, 9),
        (8, 7),
        (8, 6),
        (8, 5),
    ]
    mask[tuple(zip(*pixels))] = True
    graph = extract_vessel_graph(mask, root_hint_yx=(8.0, 5.0))
    evidence = np.ones(mask.shape, dtype=np.float64)

    assert find_gap_candidates(graph, evidence, evidence) == []


@pytest.mark.parametrize(
    "malformation",
    ["repeated", "jump", "fractional"],
)
def test_graph_validation_rejects_non_pixelwise_edge_chains(
    malformation: str,
) -> None:
    graph = _two_segment_graph(12.0)
    edge = graph.edges[0]
    points = edge.points_yx.copy()
    if malformation == "repeated":
        points[1] = points[0]
    elif malformation == "jump":
        points = np.delete(points, 1, axis=0)
    else:
        points[1, 0] += 0.25
    malformed = replace(
        graph,
        edges=(replace(edge, points_yx=points), graph.edges[1]),
    )
    evidence, support = _gap_evidence()

    with pytest.raises(ValueError, match="points_yx"):
        find_gap_candidates(malformed, evidence, support)


def test_unrelated_exclusion_does_not_mislabel_an_unavailable_route(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    graph = _two_segment_graph(12.0)
    evidence, support = _gap_evidence()
    exclusion = np.zeros(evidence.shape, dtype=bool)
    exclusion[33, 17] = True

    def unavailable_route(*_args: object, **_kwargs: object) -> object:
        raise ValueError("synthetic route failure")

    monkeypatch.setattr(topology_module, "route_through_array", unavailable_route)

    candidate = find_gap_candidates(
        graph, evidence, support, exclusion_mask=exclusion
    )[0]

    assert candidate.decision == "rejected"
    assert "route_unavailable" in candidate.reasons
    assert "exclusion_intersection" not in candidate.reasons
    assert not candidate.crosses_exclusion
    assert np.isfinite(candidate.score)
    path_steps = np.max(np.abs(np.diff(candidate.path_yx, axis=0)), axis=1)
    assert np.all(path_steps == 1)
