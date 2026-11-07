"""Build research-only dynamic coronary 2.5D meshes in the CT LPS frame.

The XA series is monoplane and does not determine unique vessel depth.  This
module therefore uses a deliberately explicit prior: each segmented centerline
ray is placed on the source-facing TotalSegmentator heart surface.  The output
is suitable for visual exploration, not navigation or quantitative diagnosis.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from numbers import Real
import os
import shutil
import string
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as sitk
import trimesh
from scipy import ndimage
from scipy.spatial import cKDTree
from skimage import morphology

try:
    from .xa_vessel_depth import DepthOptimizationResult, edge_depth_audit, optimize_graph_depth, reprojection_rms_px
    from .xa_vessel_topology import VesselEdge, VesselGraph, VesselNode
    from .xa_vessel_mesh import build_watertight_vessel_surface
    from .xa_atomic_output import build_then_publish_directory
except ImportError:  # direct script execution
    from xa_vessel_depth import DepthOptimizationResult, edge_depth_audit, optimize_graph_depth, reprojection_rms_px
    from xa_vessel_topology import VesselEdge, VesselGraph, VesselNode
    from xa_vessel_mesh import build_watertight_vessel_surface
    from xa_atomic_output import build_then_publish_directory

from ctdr_pipeline.visualization.surface_mesh import (
    LEGACY_REGISTRATION_PROFILES,
    RAW_PROFILE,
    SurfaceProfile,
    ThoraxAssetRecord,
    ThoraxAssetReport,
    ThoraxDisplayRecord,
    ThoraxFailedCandidateRecord,
    ThoraxLabelRecord,
    ThoraxRegistrationRecord,
    _surface_geometry_audit,
    extract_registration_surface,
    extract_surface,
    export_thorax_mesh_variants,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Camera:
    source: np.ndarray
    detector_center: np.ndarray
    u_axis: np.ndarray
    v_axis: np.ndarray
    n_axis: np.ndarray
    principal_uv: np.ndarray
    row_spacing_mm: float
    col_spacing_mm: float
    rows: int
    columns: int
    sdd_mm: float


SURFACE_PROFILES = {
    "raw": RAW_PROFILE,
    "bone": LEGACY_REGISTRATION_PROFILES["rib"][0],
    "bone_fallback": LEGACY_REGISTRATION_PROFILES["rib"][1],
    "bone_safe": LEGACY_REGISTRATION_PROFILES["rib"][2],
    "heart": LEGACY_REGISTRATION_PROFILES["heart"][0],
}


def _validated_dynamic_paths(
    series_dir: Path, temporal_dir: Path, output_dir: Path
) -> tuple[Path, Path, Path, bool]:
    resolved_series = Path(series_dir).resolve(strict=False)
    resolved_temporal = Path(temporal_dir).resolve(strict=False)
    resolved_output = Path(output_dir).resolve(strict=False)
    actual_pair = (
        os.path.normcase(str(resolved_temporal)),
        os.path.normcase(str(resolved_output)),
    )
    published_pair = (
        os.path.normcase(str(resolved_series / "temporal")),
        os.path.normcase(str(resolved_series / "dynamic_3d")),
    )
    candidate_pair = (
        os.path.normcase(str(resolved_series / "temporal_candidate")),
        os.path.normcase(str(resolved_series / "dynamic_3d_candidate")),
    )
    if actual_pair == published_pair:
        return resolved_series, resolved_temporal, resolved_output, False
    if actual_pair == candidate_pair:
        return resolved_series, resolved_temporal, resolved_output, True
    raise ValueError(
        "Task 9 dynamic output requires one exact isolated sibling candidate pair: "
        "'temporal' -> 'dynamic_3d' or "
        "'temporal_candidate' -> 'dynamic_3d_candidate'."
    )


def _task9_nonnegative_int_metric(metrics: dict[str, Any], key: str) -> int:
    value = metrics.get(key)
    if type(value) is not int or value < 0:
        raise ValueError(f"{key} must be a non-negative integer")
    return value


def _task9_nonnegative_real_metric(
    metrics: dict[str, Any], key: str, *, maximum: float | None = None
) -> float:
    value = metrics.get(key)
    if type(value) not in (int, float):
        raise ValueError(f"{key} must be a finite non-negative number")
    number = float(value)
    if not math.isfinite(number) or number < 0 or (maximum is not None and number > maximum):
        raise ValueError(f"{key} must be a finite non-negative number")
    return number


def build_connectivity_acceptance_report(
    *,
    run_id: str,
    metrics: dict[str, Any],
    physician_review_status: str = "pending",
    browser_review_status: str = "pending",
    audit_errors: list[str] | None = None,
) -> dict[str, Any]:
    """Build the fail-closed Task 9 graph/depth acceptance record."""
    if type(run_id) is not str or not run_id or not isinstance(metrics, dict):
        raise ValueError("Task 9 connectivity acceptance inputs are malformed")
    accepted_frame_count = _task9_nonnegative_int_metric(
        metrics, "accepted_frame_count"
    )
    component_count = _task9_nonnegative_int_metric(
        metrics, "accepted_frame_graph_components_max"
    )
    stable_edge_fraction = _task9_nonnegative_real_metric(
        metrics, "stable_edge_id_fraction", maximum=1.0
    )
    edge_steps_over = _task9_nonnegative_int_metric(
        metrics, "edge_steps_over_10_mm"
    )
    watertight_components = _task9_nonnegative_int_metric(
        metrics, "watertight_surface_components"
    )
    max_junction_gap = _task9_nonnegative_real_metric(
        metrics, "max_junction_gap_mm"
    )
    reprojection_rms = (
        None
        if metrics.get("reprojection_rms_px") is None
        else _task9_nonnegative_real_metric(metrics, "reprojection_rms_px")
    )
    definitions = {
        "accepted_frame_graph_components_max": (
            "at_most",
            1,
            accepted_frame_count > 0 and component_count <= 1,
        ),
        "stable_edge_id_fraction": (
            "equal",
            1.0,
            stable_edge_fraction == 1.0,
        ),
        "edge_steps_over_10_mm": (
            "at_most",
            10,
            edge_steps_over <= 10,
        ),
        "watertight_surface_components": (
            "equal",
            1,
            watertight_components == 1,
        ),
        "max_junction_gap_mm": (
            "at_most",
            0.2,
            max_junction_gap <= 0.2,
        ),
        "reprojection_rms_px": (
            "at_most",
            0.5,
            reprojection_rms is not None and reprojection_rms <= 0.5,
        ),
    }
    gates = {
        key: {
            "value": metrics.get(key),
            "operator": operator,
            "threshold": threshold,
            "passed": bool(passed),
        }
        for key, (operator, threshold, passed) in definitions.items()
    }
    errors = list(audit_errors or [])
    quantitative_pass = not errors and all(item["passed"] for item in gates.values())
    physician_pass = physician_review_status == "passed"
    browser_pass = browser_review_status == "passed"
    return {
        "schema_version": 1,
        "run_id": run_id,
        "clinical_status": "research-only; not suitable for navigation or quantitative diagnosis",
        "depth_limitation": (
            "Monoplane XA does not uniquely determine depth; CT heart geometry is a prior only."
        ),
        "metrics": metrics,
        "gates": gates,
        "audit_errors": errors,
        "quantitative_gates_passed": quantitative_pass,
        "quality_status": "quantitative-pass" if quantitative_pass else "rejected",
        "physician_review": {"status": physician_review_status},
        "browser_review": {
            "status": browser_review_status,
            "required_frame_indices_zero_based": [3, 13, 24, 25, 28, 29],
            "required_modes": ["diagnostic", "realistic"],
        },
        "promotion_authorized": bool(
            quantitative_pass and physician_pass and browser_pass
        ),
    }


_DYNAMIC_OBJECTIVE_TERMS = {
    "heart_surface_distance",
    "first_derivative_smoothness",
    "second_derivative_smoothness",
    "junction_equality_penalty",
    "temporal_depth_penalty",
    "weighted_total",
}
_DYNAMIC_EDGE_AUDIT_FIELDS = {
    "edge_step_max_mm",
    "edge_step_p95_mm",
    "curvature_p95",
    "surface_distance_p95_mm",
}
_DYNAMIC_FRAME_FIELDS = {
    "frame_index",
    "root_node_id",
    "root_topology_id",
    "component_count",
    "nodes",
    "edges",
    "quality_status",
    "reasons",
    "depth_audit",
    "display_source_frame_index",
    "paths",
    "path_count",
    "point_count",
    "heart_projection_valid_fraction",
}
_DYNAMIC_NODE_FIELDS = {
    "node_id",
    "topology_id",
    "parent_node_id",
    "kind",
    "yx",
    "point_lps_mm",
    "confidence",
    "visibility",
}
_DYNAMIC_EDGE_FIELDS = {
    "edge_id",
    "topology_id",
    "parent_edge_id",
    "source_node_id",
    "target_node_id",
    "source_topology_id",
    "target_topology_id",
    "points_yx",
    "points_lps_mm",
    "radii_mm",
    "confidence",
    "visibility",
    "depth_audit",
}
_TEMPORAL_CANDIDATE_ASSETS = {
    "temporal_probabilities_u8.raw",
    "temporal_masks_u8.raw",
    "temporal_centerlines_u8.raw",
    "temporal_quality.csv",
    "repaired_probabilities_u8.raw",
    "repaired_masks_u8.raw",
    "repaired_centerlines_u8.raw",
    "repair_audit.json",
    "temporal_vessel_graphs.json",
    "temporal_report.json",
    "repair_before_after.png",
}
_TEMPORAL_BASELINE_ASSETS = {
    "temporal_probabilities_u8.raw",
    "temporal_masks_u8.raw",
    "temporal_centerlines_u8.raw",
    "temporal_quality.csv",
    "temporal_report.json",
}
_TEMPORAL_ACCEPTANCE_METRICS = {
    "frame_count",
    "multicomponent_frames",
    "median_warped_iou",
    "frames_below_iou_0_50",
    "accepted_gap_max_length_px",
    "max_component_growth_fraction",
    "rejected_border_leaks",
}
_TEMPORAL_ACCEPTANCE_CLINICAL_STATUS = (
    "research-only; quantitative gates are not clinical validation"
)
_TEMPORAL_REPAIR_CLINICAL_STATUS = (
    "research-only; repaired outputs require diagnostic review"
)
_TEMPORAL_PHYSICIAN_REVIEW_FRAMES = [0, 1, 2, 3, 13, 24, 25, 26, 27, 28, 29]
_TEMPORAL_BROWSER_REVIEW_FRAMES = [3, 13, 24, 25, 28, 29]
_TEMPORAL_BROWSER_REVIEW_MODES = ["diagnostic", "realistic"]
_TEMPORAL_HISTORICAL_DEPTH_NOTE = (
    "Prior audit count; surviving legacy paths contain no >14 mm within-path step "
    "because the old generator split at those events."
)


def _audit_real(
    value: object,
    label: str,
    errors: list[str],
    *,
    minimum: float = 0.0,
    maximum: float | None = None,
) -> float | None:
    if type(value) not in (int, float):
        errors.append(f"{label} must be a finite number")
        return None
    number = float(value)
    if (
        not math.isfinite(number)
        or number < minimum
        or (maximum is not None and number > maximum)
    ):
        errors.append(f"{label} must be a finite number within its valid range")
        return None
    return number


def _audit_vector_array(
    value: object,
    label: str,
    errors: list[str],
    *,
    columns: int,
    minimum_rows: int,
) -> np.ndarray | None:
    raw = np.asarray(value, dtype=object)
    if (
        raw.ndim != 2
        or raw.shape[0] < minimum_rows
        or raw.shape[1:] != (columns,)
        or any(isinstance(item, bool) or not isinstance(item, Real) for item in raw.flat)
    ):
        errors.append(
            f"{label} must have shape (N, {columns}) with at least "
            f"{minimum_rows} strictly numeric finite rows"
        )
        return None
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        errors.append(f"{label} is not a numeric coordinate array: {exc}")
        return None
    if (
        array.ndim != 2
        or array.shape[0] < minimum_rows
        or array.shape[1:] != (columns,)
        or not np.all(np.isfinite(array))
    ):
        errors.append(
            f"{label} must have shape (N, {columns}) with at least "
            f"{minimum_rows} finite rows"
        )
        return None
    return array


def _audit_positive_vector(
    value: object,
    label: str,
    errors: list[str],
    *,
    expected_length: int,
) -> np.ndarray | None:
    raw = np.asarray(value, dtype=object)
    if raw.shape != (expected_length,) or any(
        isinstance(item, bool) or not isinstance(item, Real) for item in raw.flat
    ):
        errors.append(f"{label} must match strictly numeric positive samples")
        return None
    array = np.asarray(value, dtype=np.float64)
    if not np.all(np.isfinite(array)) or np.any(array <= 0.0):
        errors.append(f"{label} must match finite positive samples")
        return None
    return array


def _graph_component_count(node_ids: set[str], edges: list[tuple[str, str]]) -> int:
    parent = {node_id: node_id for node_id in node_ids}

    def find(node_id: str) -> str:
        while parent[node_id] != node_id:
            parent[node_id] = parent[parent[node_id]]
            node_id = parent[node_id]
        return node_id

    for source, target in edges:
        source_root = find(source)
        target_root = find(target)
        if source_root != target_root:
            parent[target_root] = source_root
    return len({find(node_id) for node_id in node_ids})


def _dynamic_connectivity_metrics(depth_asset: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    errors: list[str] = []
    empty_metrics: dict[str, Any] = {
        "accepted_frame_count": 0,
        "accepted_frame_graph_components_max": 0,
        "stable_edge_id_fraction": 0.0,
        "edge_steps_over_10_mm": 0,
        "watertight_surface_components": 0,
        "max_junction_gap_mm": 0.0,
        "reprojection_rms_px": None,
    }
    if type(depth_asset) is not dict:
        return empty_metrics, ["dynamic vessel asset must be an object"]
    if type(depth_asset.get("schema_version")) is not int or depth_asset.get(
        "schema_version"
    ) != 2:
        errors.append("dynamic vessel schema_version must be integer 2")
    frames = depth_asset.get("frames")
    if type(frames) is not list:
        return empty_metrics, [*errors, "dynamic vessel frames must be an array"]
    frame_count = depth_asset.get("frame_count")
    if type(frame_count) is not int or frame_count < 0 or frame_count != len(frames):
        errors.append("dynamic vessel frame_count must match the frame array")
    reference_frame = depth_asset.get("reference_frame")
    if (
        type(reference_frame) is not int
        or reference_frame < 0
        or (frames and reference_frame >= len(frames))
    ):
        errors.append("dynamic vessel reference_frame must identify a frame")
    if depth_asset.get("legacy_paths") != {
        "authoritative": False,
        "lifecycle": "one-release compatibility data",
    } or type(depth_asset.get("legacy_paths")) is not dict:
        errors.append("dynamic vessel legacy_paths marker is malformed")

    edge_occurrences: dict[str, int] = {}
    edge_steps_over = 0
    max_junction_gap = 0.0
    reprojection_values: list[float] = []
    accepted: list[dict[str, Any]] = []
    accepted_components: list[int] = []
    accepted_indices: set[int] = set()
    accepted_display_edges: dict[int, dict[str, dict[str, Any]]] = {}
    display_references: list[tuple[str, int | None]] = []

    for expected_index, raw_frame in enumerate(frames):
        frame_label = f"frames[{expected_index}]"
        if type(raw_frame) is not dict:
            errors.append(f"{frame_label} must be an object")
            continue
        frame_valid = True

        def frame_error(message: str) -> None:
            nonlocal frame_valid
            frame_valid = False
            errors.append(message)

        if set(raw_frame) != _DYNAMIC_FRAME_FIELDS:
            frame_error(f"{frame_label} must match the producer frame schema")

        frame_index = raw_frame.get("frame_index")
        if type(frame_index) is not int or frame_index != expected_index:
            frame_error(f"{frame_label}.frame_index must equal its array position")
        quality_status = raw_frame.get("quality_status")
        if type(quality_status) is not str or quality_status not in {
            "accepted",
            "rejected",
        }:
            frame_error(f"{frame_label}.quality_status is invalid")
        component_count = raw_frame.get("component_count")
        if type(component_count) is not int or component_count < 0:
            frame_error(f"{frame_label}.component_count must be a non-negative integer")
            component_count = 0
        reasons = raw_frame.get("reasons")
        if type(reasons) is not list or any(type(reason) is not str for reason in reasons):
            frame_error(f"{frame_label}.reasons must be an array of strings")
        raw_nodes = raw_frame.get("nodes")
        raw_edges = raw_frame.get("edges")
        if type(raw_nodes) is not list:
            frame_error(f"{frame_label}.nodes must be an array")
            raw_nodes = []
        if type(raw_edges) is not list:
            frame_error(f"{frame_label}.edges must be an array")
            raw_edges = []

        node_points: dict[str, np.ndarray] = {}
        node_yx: dict[str, np.ndarray] = {}
        node_topology: dict[str, str] = {}
        topology_ids: set[str] = set()
        parent_node_refs: list[tuple[str, str]] = []
        for node_index, raw_node in enumerate(raw_nodes):
            node_label = f"{frame_label}.nodes[{node_index}]"
            if type(raw_node) is not dict:
                frame_error(f"{node_label} must be an object")
                continue
            if set(raw_node) != _DYNAMIC_NODE_FIELDS:
                frame_error(f"{node_label} must match the producer node schema")
            node_id = raw_node.get("node_id")
            topology_id = raw_node.get("topology_id")
            if type(node_id) is not str or not node_id or node_id in node_points:
                frame_error(f"{node_label}.node_id must be unique and non-empty")
                continue
            if (
                type(topology_id) is not str
                or not topology_id
                or topology_id in topology_ids
            ):
                frame_error(f"{node_label}.topology_id must be unique and non-empty")
                continue
            point_rows = _audit_vector_array(
                [raw_node.get("point_lps_mm")],
                f"{node_label}.point_lps_mm",
                errors,
                columns=3,
                minimum_rows=1,
            )
            yx_rows = _audit_vector_array(
                [raw_node.get("yx")],
                f"{node_label}.yx",
                errors,
                columns=2,
                minimum_rows=1,
            )
            if point_rows is None or yx_rows is None:
                frame_valid = False
            confidence = _audit_real(
                raw_node.get("confidence"),
                f"{node_label}.confidence",
                errors,
                maximum=1.0,
            )
            if confidence is None:
                frame_valid = False
            node_kind = raw_node.get("kind")
            if type(node_kind) is not str or node_kind not in {
                "root",
                "junction",
                "endpoint",
            }:
                frame_error(f"{node_label}.kind is invalid")
            node_visibility = raw_node.get("visibility")
            if type(node_visibility) is not str or node_visibility not in {
                "visible",
                "occluded",
            }:
                frame_error(f"{node_label}.visibility is invalid")
            parent_node_id = raw_node.get("parent_node_id")
            if parent_node_id is not None:
                if type(parent_node_id) is not str or not parent_node_id:
                    frame_error(f"{node_label}.parent_node_id is invalid")
                else:
                    parent_node_refs.append((topology_id, parent_node_id))
            if point_rows is not None:
                node_points[node_id] = point_rows[0]
            if yx_rows is not None:
                node_yx[node_id] = yx_rows[0]
            node_topology[node_id] = topology_id
            topology_ids.add(topology_id)

        for topology_id, parent_id in parent_node_refs:
            if parent_id not in topology_ids or parent_id == topology_id:
                frame_error(f"{frame_label} has an invalid parent_node_id reference")

        root_node_id = raw_frame.get("root_node_id")
        root_topology_id = raw_frame.get("root_topology_id")
        if node_points:
            if type(root_node_id) is not str or root_node_id not in node_points:
                frame_error(f"{frame_label}.root_node_id refers to a missing node")
            elif (
                type(root_topology_id) is not str
                or root_topology_id != node_topology[root_node_id]
            ):
                frame_error(f"{frame_label}.root_topology_id is inconsistent")
        elif root_node_id != "" or root_topology_id is not None:
            frame_error(f"{frame_label} empty graph root references are invalid")

        frame_edge_ids: set[str] = set()
        frame_edge_topology_ids: set[str] = set()
        valid_graph_edges: list[tuple[str, str]] = []
        edge_topology_refs: list[tuple[str, str]] = []
        valid_occurrences: set[str] = set()
        path_source_edges: dict[str, dict[str, Any]] = {}
        for edge_index, raw_edge in enumerate(raw_edges):
            edge_label = f"{frame_label}.edges[{edge_index}]"
            edge_valid = True
            if type(raw_edge) is not dict:
                frame_error(f"{edge_label} must be an object")
                continue
            if set(raw_edge) != _DYNAMIC_EDGE_FIELDS:
                frame_error(f"{edge_label} must match the producer edge schema")
            edge_id = raw_edge.get("edge_id")
            topology_id = raw_edge.get("topology_id")
            if type(edge_id) is not str or not edge_id or edge_id in frame_edge_ids:
                frame_error(f"{edge_label}.edge_id must be unique and non-empty")
                edge_valid = False
            else:
                frame_edge_ids.add(edge_id)
            if (
                type(topology_id) is not str
                or not topology_id
                or topology_id in frame_edge_topology_ids
            ):
                frame_error(f"{edge_label}.topology_id must be unique and non-empty")
                edge_valid = False
            else:
                frame_edge_topology_ids.add(topology_id)
            source = raw_edge.get("source_node_id")
            target = raw_edge.get("target_node_id")
            if (
                type(source) is not str
                or type(target) is not str
                or source not in node_points
                or target not in node_points
                or source == target
            ):
                frame_error(f"{edge_label} has invalid node references")
                edge_valid = False
            else:
                valid_graph_edges.append((source, target))
                if raw_edge.get("source_topology_id") != node_topology[source]:
                    frame_error(f"{edge_label}.source_topology_id is inconsistent")
                    edge_valid = False
                if raw_edge.get("target_topology_id") != node_topology[target]:
                    frame_error(f"{edge_label}.target_topology_id is inconsistent")
                    edge_valid = False
            points = _audit_vector_array(
                raw_edge.get("points_lps_mm"),
                f"{edge_label}.points_lps_mm",
                errors,
                columns=3,
                minimum_rows=2,
            )
            points_yx = _audit_vector_array(
                raw_edge.get("points_yx"),
                f"{edge_label}.points_yx",
                errors,
                columns=2,
                minimum_rows=2,
            )
            if points is None or points_yx is None or (
                points is not None
                and points_yx is not None
                and len(points) != len(points_yx)
            ):
                frame_valid = False
                edge_valid = False
                if points is not None and points_yx is not None:
                    errors.append(f"{edge_label} coordinate sample counts must match")
            radii = _audit_positive_vector(
                raw_edge.get("radii_mm"),
                f"{edge_label}.radii_mm",
                errors,
                expected_length=len(points) if points is not None else 0,
            )
            if points is None or radii is None:
                frame_valid = False
                edge_valid = False
            confidence = _audit_real(
                raw_edge.get("confidence"),
                f"{edge_label}.confidence",
                errors,
                maximum=1.0,
            )
            if confidence is None:
                frame_valid = False
                edge_valid = False
            edge_visibility = raw_edge.get("visibility")
            if type(edge_visibility) is not str or edge_visibility not in {
                "visible",
                "occluded",
            }:
                frame_error(f"{edge_label}.visibility is invalid")
                edge_valid = False
            parent_edge_id = raw_edge.get("parent_edge_id")
            if parent_edge_id is not None:
                if type(parent_edge_id) is not str or not parent_edge_id:
                    frame_error(f"{edge_label}.parent_edge_id is invalid")
                    edge_valid = False
                else:
                    edge_topology_refs.append((topology_id, parent_edge_id))
            edge_audit = raw_edge.get("depth_audit")
            if type(edge_audit) is not dict or set(edge_audit) != _DYNAMIC_EDGE_AUDIT_FIELDS:
                frame_error(f"{edge_label}.depth_audit is incomplete")
                edge_valid = False
            else:
                for field in sorted(_DYNAMIC_EDGE_AUDIT_FIELDS):
                    if _audit_real(
                        edge_audit.get(field), f"{edge_label}.depth_audit.{field}", errors
                    ) is None:
                        frame_valid = False
                        edge_valid = False
            if edge_valid and points is not None:
                steps = np.linalg.norm(np.diff(points, axis=0), axis=1)
                edge_steps_over += int(np.count_nonzero(steps > 10.0))
                max_junction_gap = max(
                    max_junction_gap,
                    float(np.linalg.norm(points[0] - node_points[source])),
                    float(np.linalg.norm(points[-1] - node_points[target])),
                )
                valid_occurrences.add(topology_id)
                assert radii is not None
                path_source_edges[edge_id] = {
                    "topology_id": topology_id,
                    "points_lps_mm": points,
                    "radius_mm": float(np.median(radii)),
                    "length_mm": float(steps.sum()),
                }

        for topology_id, parent_id in edge_topology_refs:
            if parent_id not in frame_edge_topology_ids or parent_id == topology_id:
                frame_error(f"{frame_label} has an invalid parent_edge_id reference")
        for topology_id in valid_occurrences:
            edge_occurrences[topology_id] = edge_occurrences.get(topology_id, 0) + 1

        if frame_valid:
            actual_components = _graph_component_count(
                set(node_points), valid_graph_edges
            )
            if component_count != actual_components:
                frame_error(
                    f"{frame_label}.component_count does not match graph connectivity"
                )

        frame_depth_audit = raw_frame.get("depth_audit")
        reprojection: float | None = None
        if type(frame_depth_audit) is not dict or set(frame_depth_audit) != {
            "objective_terms",
            "reprojection_rms_px",
        }:
            frame_error(f"{frame_label}.depth_audit must be an object")
        else:
            objective_terms = frame_depth_audit.get("objective_terms")
            if type(objective_terms) is not dict or set(objective_terms) != _DYNAMIC_OBJECTIVE_TERMS:
                frame_error(f"{frame_label}.depth_audit.objective_terms is malformed")
            else:
                for field in sorted(_DYNAMIC_OBJECTIVE_TERMS):
                    if _audit_real(
                        objective_terms.get(field),
                        f"{frame_label}.depth_audit.objective_terms.{field}",
                        errors,
                    ) is None:
                        frame_valid = False
            reprojection = _audit_real(
                frame_depth_audit.get("reprojection_rms_px"),
                f"{frame_label}.depth_audit.reprojection_rms_px",
                errors,
            )
            if reprojection is None:
                frame_valid = False

        raw_paths = raw_frame.get("paths")
        if type(raw_paths) is not list:
            frame_error(f"{frame_label}.paths must be an array")
            raw_paths = []
        parsed_paths: list[dict[str, Any]] = []
        path_ids: set[str] = set()
        path_topology_ids: set[str] = set()
        path_point_count = 0
        expected_path_fields = {
            "edge_id",
            "topology_id",
            "points_lps_mm",
            "radius_mm",
            "length_mm",
        }
        for path_index, raw_path in enumerate(raw_paths):
            path_label = f"{frame_label}.paths[{path_index}]"
            if type(raw_path) is not dict or set(raw_path) != expected_path_fields:
                frame_error(f"{path_label} must match the producer display-path schema")
                continue
            path_edge_id = raw_path["edge_id"]
            path_topology_id = raw_path["topology_id"]
            if (
                type(path_edge_id) is not str
                or not path_edge_id
                or path_edge_id in path_ids
            ):
                frame_error(f"{path_label}.edge_id must be unique and non-empty")
            else:
                path_ids.add(path_edge_id)
            if (
                type(path_topology_id) is not str
                or not path_topology_id
                or path_topology_id in path_topology_ids
            ):
                frame_error(f"{path_label}.topology_id must be unique and non-empty")
            else:
                path_topology_ids.add(path_topology_id)
            path_points = _audit_vector_array(
                raw_path["points_lps_mm"],
                f"{path_label}.points_lps_mm",
                errors,
                columns=3,
                minimum_rows=2,
            )
            if path_points is None:
                frame_valid = False
            else:
                path_point_count += len(path_points)
            radius = _audit_real(
                raw_path["radius_mm"], f"{path_label}.radius_mm", errors
            )
            if radius is None or radius <= 0.0:
                if radius is not None:
                    errors.append(f"{path_label}.radius_mm must be positive")
                frame_valid = False
            length = _audit_real(
                raw_path["length_mm"], f"{path_label}.length_mm", errors
            )
            if path_points is not None and length is not None:
                measured_length = float(
                    np.linalg.norm(np.diff(path_points, axis=0), axis=1).sum()
                )
                if not math.isclose(length, measured_length, rel_tol=1e-9, abs_tol=1e-9):
                    frame_error(f"{path_label}.length_mm is inconsistent with its points")
            parsed_paths.append(
                {
                    "edge_id": path_edge_id,
                    "topology_id": path_topology_id,
                    "points_lps_mm": path_points,
                    "radius_mm": radius,
                    "length_mm": length,
                }
            )
        path_count = raw_frame.get("path_count")
        if type(path_count) is not int or path_count < 0 or path_count != len(raw_paths):
            frame_error(f"{frame_label}.path_count must equal len(paths)")
        point_count = raw_frame.get("point_count")
        if (
            type(point_count) is not int
            or point_count < 0
            or point_count != path_point_count
        ):
            frame_error(f"{frame_label}.point_count must equal display-path samples")
        if _audit_real(
            raw_frame.get("heart_projection_valid_fraction"),
            f"{frame_label}.heart_projection_valid_fraction",
            errors,
            maximum=1.0,
        ) is None:
            frame_valid = False
        display_source = raw_frame.get("display_source_frame_index")
        if display_source is not None and (
            type(display_source) is not int
            or display_source < 0
            or display_source >= len(frames)
        ):
            frame_error(f"{frame_label}.display_source_frame_index is invalid")
            display_source = None
        if quality_status == "accepted" and display_source != expected_index:
            frame_error(
                f"{frame_label}.display_source_frame_index must identify itself when accepted"
            )
        if quality_status == "rejected" and display_source is not None:
            if display_source >= expected_index or display_source not in accepted_display_edges:
                frame_error(
                    f"{frame_label}.display_source_frame_index must identify an earlier accepted frame"
                )
        source_edges = (
            path_source_edges
            if quality_status == "accepted" and display_source == expected_index
            else accepted_display_edges.get(display_source, {})
        )
        if display_source is None and raw_paths:
            frame_error(f"{frame_label}.paths require a display source frame")
        for path_index, path_record in enumerate(parsed_paths):
            path_label = f"{frame_label}.paths[{path_index}]"
            source_edge = source_edges.get(path_record["edge_id"])
            if source_edge is None:
                frame_error(f"{path_label}.edge_id does not reference its display source")
                continue
            if path_record["topology_id"] != source_edge["topology_id"]:
                frame_error(f"{path_label}.topology_id is inconsistent")
            if path_record["points_lps_mm"] is not None and not np.array_equal(
                path_record["points_lps_mm"], source_edge["points_lps_mm"]
            ):
                frame_error(f"{path_label}.points_lps_mm is inconsistent")
            for field in ("radius_mm", "length_mm"):
                if path_record[field] is not None and not math.isclose(
                    path_record[field], source_edge[field], rel_tol=1e-9, abs_tol=1e-9
                ):
                    frame_error(f"{path_label}.{field} is inconsistent")
        display_references.append((frame_label, display_source))

        if quality_status == "accepted" and frame_valid:
            accepted.append(raw_frame)
            accepted_indices.add(expected_index)
            accepted_display_edges[expected_index] = path_source_edges
            accepted_components.append(component_count)
            if reprojection is not None:
                reprojection_values.append(reprojection)

    for frame_label, display_source in display_references:
        if display_source is not None and display_source not in accepted_indices:
            errors.append(
                f"{frame_label}.display_source_frame_index must reference an accepted frame"
            )

    stable_edge_count = sum(
        count >= min(2, len(frames)) for count in edge_occurrences.values()
    )
    watertight_components = 0
    if accepted and not errors:
        surface_frame = next(
            (
                frame
                for frame in accepted
                if frame["frame_index"] == reference_frame
            ),
            accepted[0],
        )
        try:
            mesh = build_watertight_vessel_surface(surface_frame)
            watertight_components = len(mesh.split(only_watertight=False))
            if not mesh.is_watertight:
                errors.append("audited vessel surface is not watertight")
        except Exception as error:
            errors.append(f"watertight surface audit failed: {error}")
    elif not accepted:
        errors.append("no accepted frame is available for surface audit")
    else:
        errors.append("watertight surface audit skipped because frame validation failed")

    metrics = {
        "accepted_frame_count": int(len(accepted)),
        "accepted_frame_graph_components_max": max(accepted_components, default=0),
        "stable_edge_id_fraction": (
            float(stable_edge_count / len(edge_occurrences))
            if edge_occurrences
            else 0.0
        ),
        "edge_steps_over_10_mm": int(edge_steps_over),
        "watertight_surface_components": int(watertight_components),
        "max_junction_gap_mm": float(max_junction_gap),
        "reprojection_rms_px": max(reprojection_values) if reprojection_values else None,
    }
    return metrics, errors


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _strict_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate key {key!r}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant {value}")


def _require_temporal_acceptance_real(
    value: object, label: str, *, maximum: float | None = None
) -> float:
    if type(value) not in (int, float):
        raise ValueError(f"{label} must be a finite non-negative number")
    number = float(value)
    if not math.isfinite(number) or number < 0 or (
        maximum is not None and number > maximum
    ):
        raise ValueError(f"{label} must be a finite non-negative number")
    return number


def _require_temporal_acceptance_int(value: object, label: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{label} must be a non-negative integer")
    return value


def _validate_temporal_acceptance_metrics(
    metrics: object, label: str
) -> dict[str, int | float]:
    if type(metrics) is not dict or set(metrics) != _TEMPORAL_ACCEPTANCE_METRICS:
        raise ValueError(f"{label} must contain the complete producer metric schema")
    values: dict[str, int | float] = {
        "frame_count": _require_temporal_acceptance_int(
            metrics.get("frame_count"), f"{label}.frame_count"
        ),
        "multicomponent_frames": _require_temporal_acceptance_int(
            metrics.get("multicomponent_frames"), f"{label}.multicomponent_frames"
        ),
        "median_warped_iou": _require_temporal_acceptance_real(
            metrics.get("median_warped_iou"),
            f"{label}.median_warped_iou",
            maximum=1.0,
        ),
        "frames_below_iou_0_50": _require_temporal_acceptance_int(
            metrics.get("frames_below_iou_0_50"),
            f"{label}.frames_below_iou_0_50",
        ),
        "accepted_gap_max_length_px": _require_temporal_acceptance_real(
            metrics.get("accepted_gap_max_length_px"),
            f"{label}.accepted_gap_max_length_px",
        ),
        "max_component_growth_fraction": _require_temporal_acceptance_real(
            metrics.get("max_component_growth_fraction"),
            f"{label}.max_component_growth_fraction",
        ),
        "rejected_border_leaks": _require_temporal_acceptance_int(
            metrics.get("rejected_border_leaks"),
            f"{label}.rejected_border_leaks",
        ),
    }
    for name in (
        "median_warped_iou",
        "accepted_gap_max_length_px",
        "max_component_growth_fraction",
    ):
        if type(metrics[name]) is not float:
            raise ValueError(f"{label}.{name} must use the producer float type")
    return values


def _validate_temporal_acceptance_assets(
    assets: object,
    *,
    expected_names: set[str],
    directory: Path,
    label: str,
) -> None:
    if type(assets) is not dict or set(assets) != expected_names:
        raise ValueError(f"{label} must contain the complete producer asset inventory")
    for name in sorted(expected_names):
        record = assets[name]
        if type(record) is not dict or set(record) != {"byte_length", "sha256"}:
            raise ValueError(f"{label}.{name} is malformed")
        byte_length = record["byte_length"]
        sha256 = record["sha256"]
        if type(byte_length) is not int or byte_length < 0:
            raise ValueError(f"{label}.{name}.byte_length is invalid")
        if (
            type(sha256) is not str
            or len(sha256) != 64
            or any(character not in string.hexdigits for character in sha256)
        ):
            raise ValueError(f"{label}.{name}.sha256 is invalid")
        asset_path = directory / name
        if (
            not asset_path.is_file()
            or asset_path.stat().st_size != byte_length
            or _sha256_file(asset_path).lower() != sha256.lower()
        ):
            raise ValueError(f"{label}.{name} does not match the bound bytes")


def _validate_temporal_acceptance_report(
    report: object,
    *,
    temporal_dir: Path,
    series_dir: Path,
    run_id: str,
) -> dict[str, Any]:
    if type(report) is not dict:
        raise ValueError("root must be an object")
    expected_top_level = {
        "schema_version",
        "run_id",
        "clinical_status",
        "baseline",
        "candidate",
        "gates",
        "quantitative_gates_passed",
        "quality_status",
        "physician_review",
        "browser_review",
        "promotion_authorized",
    }
    if set(report) != expected_top_level:
        raise ValueError("top-level fields are incomplete or unexpected")
    if type(report["schema_version"]) is not int or report["schema_version"] != 1:
        raise ValueError("schema_version must be integer 1")
    if report["run_id"] != run_id:
        raise ValueError("run_id does not match the dynamic candidate identity")
    if report["clinical_status"] != _TEMPORAL_ACCEPTANCE_CLINICAL_STATUS:
        raise ValueError("clinical_status does not match the temporal producer contract")
    baseline = report["baseline"]
    candidate = report["candidate"]
    if type(baseline) is not dict or type(candidate) is not dict:
        raise ValueError("baseline and candidate must be objects")
    if set(baseline) != {
        "directory",
        "metrics",
        "assets",
        "historical_hard_depth_cut_events",
        "historical_hard_depth_cut_note",
    }:
        raise ValueError("baseline fields are incomplete or unexpected")
    if set(candidate) != {
        "directory",
        "metrics",
        "assets",
        "comparisons",
        "link_decision_counts",
    }:
        raise ValueError("candidate fields are incomplete or unexpected")
    baseline_directory = baseline.get("directory")
    candidate_directory = candidate.get("directory")
    published_temporal_dir = (series_dir / "temporal").resolve(strict=False)
    if type(baseline_directory) is not str or Path(baseline_directory).resolve(
        strict=False
    ) != published_temporal_dir:
        raise ValueError("baseline.directory does not identify the published temporal sibling")
    if type(candidate_directory) is not str or Path(candidate_directory).resolve(
        strict=False
    ) != temporal_dir:
        raise ValueError("candidate.directory does not identify temporal_candidate")

    if (
        type(baseline["historical_hard_depth_cut_events"]) is not int
        or baseline["historical_hard_depth_cut_events"] != 250
        or baseline["historical_hard_depth_cut_note"]
        != _TEMPORAL_HISTORICAL_DEPTH_NOTE
    ):
        raise ValueError("baseline historical depth-cut audit fields are inconsistent")
    _validate_temporal_acceptance_metrics(baseline["metrics"], "baseline.metrics")
    _validate_temporal_acceptance_assets(
        baseline["assets"],
        expected_names=_TEMPORAL_BASELINE_ASSETS,
        directory=published_temporal_dir,
        label="baseline.assets",
    )

    metrics = candidate["metrics"]
    values = _validate_temporal_acceptance_metrics(metrics, "candidate.metrics")
    comparisons = candidate["comparisons"]
    if type(comparisons) is not dict or set(comparisons) != {
        "candidate_pre_repair_growth",
        "published_baseline_delta",
    }:
        raise ValueError("candidate.comparisons is incomplete or unexpected")
    candidate_growth = comparisons["candidate_pre_repair_growth"]
    if type(candidate_growth) is not dict or set(candidate_growth) != {
        "max_added_pixel_fraction"
    }:
        raise ValueError("candidate pre-repair comparison is malformed")
    growth_value = _require_temporal_acceptance_real(
        candidate_growth["max_added_pixel_fraction"],
        "candidate.comparisons.candidate_pre_repair_growth.max_added_pixel_fraction",
    )
    if growth_value != values["max_component_growth_fraction"]:
        raise ValueError("candidate pre-repair comparison is inconsistent")
    published_delta = comparisons["published_baseline_delta"]
    expected_delta_fields = {
        "max_added_pixel_fraction",
        "max_removed_pixel_fraction",
        "max_changed_pixel_fraction",
    }
    if type(published_delta) is not dict or set(published_delta) != expected_delta_fields:
        raise ValueError("published baseline comparison is malformed")
    for field in sorted(expected_delta_fields):
        _require_temporal_acceptance_real(
            published_delta[field],
            f"candidate.comparisons.published_baseline_delta.{field}",
        )
    link_counts = candidate["link_decision_counts"]
    if type(link_counts) is not dict or set(link_counts) != {
        "accepted",
        "review",
        "rejected",
    }:
        raise ValueError("candidate.link_decision_counts is malformed")
    for decision in ("accepted", "review", "rejected"):
        _require_temporal_acceptance_int(
            link_counts[decision], f"candidate.link_decision_counts.{decision}"
        )
    gate_definitions = {
        "frame_count": ("equal", 31, values["frame_count"] == 31),
        "multicomponent_frames": (
            "at_most",
            6,
            values["multicomponent_frames"] <= 6,
        ),
        "median_warped_iou": (
            "at_least",
            0.70,
            values["median_warped_iou"] >= 0.70,
        ),
        "frames_below_iou_0_50": (
            "at_most",
            2,
            values["frames_below_iou_0_50"] <= 2,
        ),
        "accepted_gap_max_length_px": (
            "at_most",
            12.0,
            values["accepted_gap_max_length_px"] <= 12.0,
        ),
        "max_component_growth_fraction": (
            "at_most",
            0.20,
            values["max_component_growth_fraction"] <= 0.20,
        ),
        "rejected_border_leaks": (
            "equal",
            0,
            values["rejected_border_leaks"] == 0,
        ),
    }
    gates = report["gates"]
    if type(gates) is not dict or set(gates) != set(gate_definitions):
        raise ValueError("gates are incomplete or unexpected")
    for name, (operator, threshold, passed) in gate_definitions.items():
        gate = gates[name]
        if type(gate) is not dict or set(gate) != {
            "value",
            "operator",
            "threshold",
            "passed",
        }:
            raise ValueError(f"gates.{name} is malformed")
        if (
            type(gate["value"]) is not type(metrics[name])
            or type(gate["threshold"]) is not type(threshold)
            or gate["value"] != metrics[name]
            or gate["operator"] != operator
            or gate["threshold"] != threshold
            or type(gate["passed"]) is not bool
            or gate["passed"] is not passed
        ):
            raise ValueError(f"gates.{name} is inconsistent")
    quantitative_passed = all(passed for _, _, passed in gate_definitions.values())
    if (
        type(report["quantitative_gates_passed"]) is not bool
        or report["quantitative_gates_passed"] is not quantitative_passed
        or report["quality_status"]
        != ("quantitative-pass" if quantitative_passed else "rejected")
    ):
        raise ValueError("quantitative acceptance decision is inconsistent")
    physician = report["physician_review"]
    browser = report["browser_review"]
    if (
        type(physician) is not dict
        or set(physician)
        != {"status", "required_frame_indices_zero_based", "contact_sheet"}
        or type(physician.get("status")) is not str
        or physician["status"] not in {"pending", "passed", "failed"}
        or physician["required_frame_indices_zero_based"]
        != _TEMPORAL_PHYSICIAN_REVIEW_FRAMES
        or any(
            type(value) is not int
            for value in physician["required_frame_indices_zero_based"]
        )
        or physician["contact_sheet"] != "repair_before_after.png"
    ):
        raise ValueError("physician_review is malformed")
    if (
        type(browser) is not dict
        or set(browser)
        != {"status", "required_frame_indices_zero_based", "required_modes"}
        or type(browser.get("status")) is not str
        or browser["status"] not in {"pending", "passed", "failed"}
        or browser["required_frame_indices_zero_based"]
        != _TEMPORAL_BROWSER_REVIEW_FRAMES
        or any(
            type(value) is not int
            for value in browser["required_frame_indices_zero_based"]
        )
        or browser["required_modes"] != _TEMPORAL_BROWSER_REVIEW_MODES
    ):
        raise ValueError("browser_review is malformed")
    expected_promotion = bool(
        quantitative_passed
        and physician["status"] == "passed"
        and browser["status"] == "passed"
    )
    if (
        type(report["promotion_authorized"]) is not bool
        or report["promotion_authorized"] is not expected_promotion
    ):
        raise ValueError("promotion_authorized is inconsistent")

    _validate_temporal_acceptance_assets(
        candidate["assets"],
        expected_names=_TEMPORAL_CANDIDATE_ASSETS,
        directory=temporal_dir,
        label="candidate.assets",
    )
    repair_audit_path = temporal_dir / "repair_audit.json"
    repair_audit = json.loads(
        repair_audit_path.read_text(encoding="utf-8"),
        object_pairs_hook=_strict_json_object,
        parse_constant=_reject_json_constant,
    )
    if type(repair_audit) is not dict or set(repair_audit) != {
        "schema_version",
        "method",
        "frame_count",
        "catheter_exclusion_available",
        "thresholds",
        "frames",
        "links",
        "clinical_status",
    }:
        raise ValueError("repair_audit.json does not match the temporal producer schema")
    if (
        type(repair_audit["schema_version"]) is not int
        or repair_audit["schema_version"] != 1
        or type(repair_audit["method"]) is not str
        or not repair_audit["method"]
        or type(repair_audit["frame_count"]) is not int
        or repair_audit["frame_count"] != values["frame_count"]
        or type(repair_audit["catheter_exclusion_available"]) is not bool
        or type(repair_audit["thresholds"]) is not dict
        or type(repair_audit["frames"]) is not list
        or type(repair_audit["links"]) is not list
        or repair_audit["clinical_status"] != _TEMPORAL_REPAIR_CLINICAL_STATUS
    ):
        raise ValueError("repair_audit.json metadata is inconsistent")
    observed_link_counts = {"accepted": 0, "review": 0, "rejected": 0}
    for index, link in enumerate(repair_audit["links"]):
        if type(link) is not dict or type(link.get("decision")) is not str:
            raise ValueError(f"repair_audit.json links[{index}] is malformed")
        decision = link["decision"]
        if decision not in observed_link_counts:
            raise ValueError(f"repair_audit.json links[{index}].decision is invalid")
        observed_link_counts[decision] += 1
    if link_counts != observed_link_counts:
        raise ValueError(
            "candidate.link_decision_counts does not match repair_audit.json links"
        )
    return report


def _load_temporal_candidate_acceptance(
    temporal_dir: Path, series_dir: Path, run_id: str
) -> tuple[dict[str, Any], list[str]]:
    path = temporal_dir / "repair_acceptance.json"
    summary: dict[str, Any] = {
        "path": str(path.resolve(strict=False)),
        "sha256": None,
        "validated": False,
        "quantitative_gates_passed": False,
        "promotion_authorized": False,
    }
    if not path.is_file():
        return summary, ["temporal acceptance report is missing"]
    try:
        summary["sha256"] = _sha256_file(path)
        report = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_strict_json_object,
            parse_constant=_reject_json_constant,
        )
        validated = _validate_temporal_acceptance_report(
            report,
            temporal_dir=temporal_dir,
            series_dir=series_dir,
            run_id=run_id,
        )
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        return summary, [f"temporal acceptance validation failed: {exc}"]
    summary.update(
        {
            "validated": True,
            "quantitative_gates_passed": validated[
                "quantitative_gates_passed"
            ],
            "promotion_authorized": validated["promotion_authorized"],
        }
    )
    errors: list[str] = []
    if validated["quantitative_gates_passed"] is not True:
        errors.append("temporal acceptance quantitative gates did not pass")
    return summary, errors


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def camera_from_calibration(report: dict[str, Any]) -> Camera:
    cal = report["calibration"]
    row_spacing, col_spacing = map(float, cal["pixel_spacing_row_col_mm"])
    detector = report["detector"]
    return Camera(
        source=np.asarray(cal["source_lps_mm"], dtype=np.float64),
        detector_center=np.asarray(cal["detector_center_lps_mm"], dtype=np.float64),
        u_axis=np.asarray(cal["u_lps"], dtype=np.float64),
        v_axis=np.asarray(cal["v_lps"], dtype=np.float64),
        n_axis=np.asarray(cal["n_lps"], dtype=np.float64),
        principal_uv=np.asarray(cal["principal_uv_px"], dtype=np.float64),
        row_spacing_mm=row_spacing,
        col_spacing_mm=col_spacing,
        rows=int(detector["rows"]),
        columns=int(detector["columns"]),
        sdd_mm=float(cal["sdd_mm"]),
    )


def surface_from_mask(
    path: Path,
    step_size: int | None = None,
    profile: str | SurfaceProfile | None = None,
) -> tuple[np.ndarray, np.ndarray, sitk.Image]:
    if profile is None:
        settings = SurfaceProfile(
            name="legacy_default",
            step_size=2 if step_size is None else int(step_size),
        )
    else:
        settings = SURFACE_PROFILES[profile] if isinstance(profile, str) else profile
        if step_size is not None:
            settings = replace(settings, step_size=int(step_size))
    result = extract_surface(path, settings)
    return result.vertices_lps_mm, result.faces, result.image


def foreground_points_from_mask(path: Path, stride: int = 2) -> np.ndarray:
    """Return a dense sampling of foreground voxel centers in patient LPS."""
    image = sitk.ReadImage(str(path))
    array = np.asarray(sitk.GetArrayViewFromImage(image)) > 0
    points_zyx = np.argwhere(array[::stride, ::stride, ::stride]) * stride
    points_xyz = points_zyx[:, ::-1].astype(np.float64)
    spacing = np.asarray(image.GetSpacing(), dtype=np.float64)
    origin = np.asarray(image.GetOrigin(), dtype=np.float64)
    direction = np.asarray(image.GetDirection(), dtype=np.float64).reshape(3, 3)
    return origin[None, :] + (points_xyz * spacing[None, :]) @ direction.T


def project_lps(points_lps: np.ndarray, camera: Camera, principal_uv: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Perspective-project LPS points to native DICOM (u=column, v=row)."""
    points = np.asarray(points_lps, dtype=np.float64)
    principal = camera.principal_uv if principal_uv is None else np.asarray(principal_uv, dtype=np.float64)
    rays = points - camera.source[None, :]
    denominator = rays @ camera.n_axis
    numerator = float((camera.detector_center - camera.source) @ camera.n_axis)
    valid = np.abs(denominator) > 1e-9
    t = np.full(len(points), np.nan, dtype=np.float64)
    t[valid] = numerator / denominator[valid]
    intersections = camera.source[None, :] + rays * t[:, None]
    local = intersections - camera.detector_center[None, :]
    uv = np.column_stack(
        [
            principal[0] + (local @ camera.u_axis) / camera.col_spacing_mm,
            principal[1] + (local @ camera.v_axis) / camera.row_spacing_mm,
        ]
    )
    depth = np.linalg.norm(rays, axis=1)
    return uv, depth


def rays_for_uv(uv: np.ndarray, camera: Camera, principal_uv: np.ndarray) -> np.ndarray:
    pixels = np.asarray(uv, dtype=np.float64)
    detector_points = (
        camera.detector_center[None, :]
        + ((pixels[:, 0] - principal_uv[0]) * camera.col_spacing_mm)[:, None] * camera.u_axis[None, :]
        + ((pixels[:, 1] - principal_uv[1]) * camera.row_spacing_mm)[:, None] * camera.v_axis[None, :]
    )
    directions = detector_points - camera.source[None, :]
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    return directions


def splat_silhouette(uv: np.ndarray, rows: int, columns: int, radius: int = 2) -> np.ndarray:
    finite = np.isfinite(uv).all(axis=1)
    rounded = np.rint(uv[finite]).astype(np.int32)
    image = np.zeros((rows, columns), dtype=bool)
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            x = rounded[:, 0] + dx
            y = rounded[:, 1] + dy
            valid = (x >= 0) & (x < columns) & (y >= 0) & (y < rows)
            image[y[valid], x[valid]] = True
    image = ndimage.binary_closing(image, iterations=2)
    labels, count = ndimage.label(image, structure=np.ones((3, 3), dtype=np.uint8))
    if count == 0:
        return image
    areas = np.bincount(labels.ravel())
    keep = np.flatnonzero(areas >= 256)
    keep = keep[keep != 0]
    return np.isin(labels, keep)


def largest_component(mask: np.ndarray, min_size: int = 80) -> np.ndarray:
    labels, count = ndimage.label(mask, structure=np.ones((3, 3), dtype=np.uint8))
    if count == 0:
        return np.zeros_like(mask, dtype=bool)
    areas = np.bincount(labels.ravel())
    areas[0] = 0
    label = int(np.argmax(areas))
    if areas[label] < min_size:
        return np.zeros_like(mask, dtype=bool)
    return labels == label


def keep_vessel_components(mask: np.ndarray, min_size: int = 60) -> np.ndarray:
    labels, count = ndimage.label(mask, structure=np.ones((3, 3), dtype=np.uint8))
    if count == 0:
        return np.zeros_like(mask, dtype=bool)
    areas = np.bincount(labels.ravel())
    keep = np.flatnonzero(areas >= min_size)
    keep = keep[keep != 0]
    return np.isin(labels, keep)


def bbox_center(mask: np.ndarray) -> np.ndarray:
    y, x = np.nonzero(mask)
    if len(x) == 0:
        raise ValueError("Cannot measure an empty mask")
    return np.asarray([(x.min() + x.max()) / 2.0, (y.min() + y.max()) / 2.0], dtype=np.float64)


def shift_mask(mask: np.ndarray, dx: int, dy: int) -> np.ndarray:
    height, width = mask.shape
    shifted = np.zeros_like(mask)
    source_x0 = max(0, -dx)
    source_x1 = min(width, width - dx)
    source_y0 = max(0, -dy)
    source_y1 = min(height, height - dy)
    if source_x0 >= source_x1 or source_y0 >= source_y1:
        return shifted
    shifted[source_y0 + dy : source_y1 + dy, source_x0 + dx : source_x1 + dx] = mask[
        source_y0:source_y1, source_x0:source_x1
    ]
    return shifted


def estimate_principal_shift(
    heart_silhouette: np.ndarray, masks: np.ndarray, search_radius: int = 36
) -> tuple[np.ndarray, int, dict[str, float], np.ndarray]:
    """Estimate only an in-plane anatomy-prior correction, never a 3D pose."""
    candidates: list[tuple[int, int]] = []
    for frame_index in range(len(masks)):
        component = largest_component(masks[frame_index] > 0)
        candidates.append((int(component.sum()), frame_index))
    _, reference_frame = max(candidates)
    target = largest_component(masks[reference_frame] > 0)
    target_line = morphology.skeletonize(target)
    target_line[:4, :] = False
    target_line[-4:, :] = False
    target_line[:, :4] = False
    target_line[:, -4:] = False

    initial_float = bbox_center(target) - bbox_center(heart_silhouette)
    initial = np.rint(initial_float).astype(int)
    initial[0] = int(np.clip(initial[0], -160, 160))
    initial[1] = int(np.clip(initial[1], -260, 160))

    dilated_target = ndimage.binary_dilation(target_line, iterations=2)
    target_count = max(int(dilated_target.sum()), 1)
    best: tuple[float, int, int, float] | None = None
    for dy in range(int(initial[1]) - search_radius, int(initial[1]) + search_radius + 1, 2):
        for dx in range(int(initial[0]) - search_radius, int(initial[0]) + search_radius + 1, 2):
            shifted = shift_mask(heart_silhouette, dx, dy)
            coverage = float(np.count_nonzero(shifted & dilated_target) / target_count)
            regularizer = 0.000025 * float((dx - initial[0]) ** 2 + (dy - initial[1]) ** 2)
            score = coverage - regularizer
            if best is None or score > best[0]:
                best = (score, dx, dy, coverage)
    assert best is not None
    _, dx, dy, coverage = best
    before = float(np.count_nonzero(heart_silhouette & dilated_target) / target_count)
    shifted = shift_mask(heart_silhouette, dx, dy)
    metrics = {
        "reference_centerline_coverage_before": before,
        "reference_centerline_coverage_after": coverage,
        "reference_component_area_px": float(target.sum()),
        "initial_bbox_shift_u_px": float(initial_float[0]),
        "initial_bbox_shift_v_px": float(initial_float[1]),
    }
    return np.asarray([dx, dy], dtype=np.float64), reference_frame, metrics, target


NEIGHBORS = [(dy, dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1) if not (dy == 0 and dx == 0)]


def trace_skeleton(skeleton: np.ndarray, min_length_px: float = 8.0) -> list[np.ndarray]:
    coords = {tuple(point) for point in np.argwhere(skeleton)}
    if not coords:
        return []

    def adjacent(point: tuple[int, int]) -> list[tuple[int, int]]:
        y, x = point
        return [(y + dy, x + dx) for dy, dx in NEIGHBORS if (y + dy, x + dx) in coords]

    adjacency = {point: adjacent(point) for point in coords}
    visited: set[frozenset[tuple[int, int]]] = set()
    paths: list[np.ndarray] = []

    def walk(start: tuple[int, int], neighbor: tuple[int, int]) -> list[tuple[int, int]]:
        path = [start, neighbor]
        previous, current = start, neighbor
        visited.add(frozenset((previous, current)))
        while len(adjacency[current]) == 2:
            candidates = [p for p in adjacency[current] if p != previous]
            if not candidates:
                break
            following = candidates[0]
            edge = frozenset((current, following))
            if edge in visited:
                break
            path.append(following)
            visited.add(edge)
            previous, current = current, following
        return path

    starts = sorted(point for point, neighbors in adjacency.items() if len(neighbors) != 2)
    for start in starts:
        for neighbor in adjacency[start]:
            if frozenset((start, neighbor)) in visited:
                continue
            path = walk(start, neighbor)
            paths.append(np.asarray(path, dtype=np.float64))

    for start in sorted(coords):
        for neighbor in adjacency[start]:
            if frozenset((start, neighbor)) in visited:
                continue
            path = walk(start, neighbor)
            paths.append(np.asarray(path, dtype=np.float64))

    filtered: list[np.ndarray] = []
    for path in paths:
        if len(path) < 2:
            continue
        length = float(np.linalg.norm(np.diff(path, axis=0), axis=1).sum())
        if length >= min_length_px:
            filtered.append(path)
    return filtered


def decimate_path(path_yx: np.ndarray, spacing_px: float = 3.0) -> np.ndarray:
    if len(path_yx) <= 2:
        return path_yx
    distances = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(path_yx, axis=0), axis=1))])
    keep = [0]
    next_distance = spacing_px
    while next_distance < distances[-1]:
        keep.append(int(np.searchsorted(distances, next_distance)))
        next_distance += spacing_px
    keep.append(len(path_yx) - 1)
    return path_yx[np.unique(keep)]


def map_path_to_heart(
    path_yx: np.ndarray,
    camera: Camera,
    principal_uv: np.ndarray,
    surface_tree: cKDTree,
    surface_depth: np.ndarray,
    distance_map_px: np.ndarray,
) -> tuple[list[np.ndarray], float]:
    del distance_map_px
    sampled = decimate_path(path_yx)
    uv = sampled[:, ::-1]
    distances, indices = surface_tree.query(uv, k=min(8, surface_tree.n))
    if distances.ndim == 1:
        distances = distances[:, None]
        indices = indices[:, None]
    candidate_depths = surface_depth[indices]
    depths = np.min(candidate_depths, axis=1)
    nearest_distance = distances[:, 0]
    directions = rays_for_uv(uv, camera, principal_uv)
    points = camera.source[None, :] + directions * depths[:, None]

    valid_fraction = float(np.mean(nearest_distance <= 12.0))
    return ([points] if len(points) >= 2 else []), valid_fraction


def _legacy_profile_dict(profile: SurfaceProfile) -> dict[str, int | float]:
    """Keep the pre-refactor report payload stable for legacy consumers."""
    return {
        "step_size": profile.step_size,
        "gaussian_sigma_mm": profile.gaussian_sigma_mm,
        "taubin_iterations": profile.taubin_iterations,
        "taubin_lambda": profile.taubin_lambda,
        "taubin_nu": profile.taubin_nu,
        "windowed_sinc_iterations": profile.windowed_sinc_iterations,
        "windowed_sinc_pass_band": profile.windowed_sinc_pass_band,
    }


def export_thorax_glb(
    masks_dir: Path,
    output_path: Path,
    step_size: int | None = None,
) -> list[dict[str, Any]]:
    paths = sorted(masks_dir.glob("rib_*.nii.gz")) + [masks_dir / "sternum.nii.gz", masks_dir / "heart.nii.gz"]
    scene = trimesh.Scene()
    records: list[dict[str, Any]] = []
    for path in paths:
        if not path.exists():
            continue
        label = path.name.replace(".nii.gz", "")
        if label == "heart":
            color = np.asarray([205, 65, 104, 72], dtype=np.uint8)
            kind = "heart"
        elif label == "sternum":
            color = np.asarray([240, 202, 126, 195], dtype=np.uint8)
            kind = "sternum"
        else:
            color = np.asarray([229, 218, 184, 190], dtype=np.uint8)
            kind = "rib"
        baseline_vertices, baseline_faces, _ = surface_from_mask(path, step_size=1)
        baseline_mesh = trimesh.Trimesh(
            vertices=baseline_vertices,
            faces=baseline_faces,
            process=False,
        )
        profile_names = ["heart"] if kind == "heart" else ["bone", "bone_fallback", "bone_safe", "raw"]
        mesh: trimesh.Trimesh | None = None
        audit: dict[str, float | bool | str] | None = None
        selected_profile_name = profile_names[0]
        for profile_name in profile_names:
            profile = SURFACE_PROFILES[profile_name]
            if step_size is not None:
                profile = replace(profile, step_size=int(step_size))
            if profile_name == "raw" and profile.step_size == 1:
                vertices, faces = baseline_vertices, baseline_faces
            else:
                vertices, faces, _ = surface_from_mask(path, profile=profile)
            mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
            audit = _surface_geometry_audit(baseline_mesh, mesh)
            selected_profile_name = profile_name
            if audit["within_visualization_tolerance"]:
                break
        assert mesh is not None and audit is not None
        mesh.visual.face_colors = np.tile(color, (len(faces), 1))
        scene.add_geometry(mesh, node_name=label, geom_name=label)
        records.append(
            {
                "label": label,
                "kind": kind,
                "vertex_count": int(len(vertices)),
                "face_count": int(len(faces)),
                "bbox_min_lps_mm": np.min(vertices, axis=0).round(3).tolist(),
                "bbox_max_lps_mm": np.max(vertices, axis=0).round(3).tolist(),
                "mesh_processing": _legacy_profile_dict(profile),
                "geometry_audit": audit,
                "smoothing_profile": selected_profile_name,
                "smoothing_fallback_applied": selected_profile_name != profile_names[0],
            }
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(trimesh.exchange.gltf.export_glb(scene))
    return records


def build_alignment_figure(
    output_path: Path,
    reference_mask: np.ndarray,
    before: np.ndarray,
    after: np.ndarray,
    reference_frame: int,
    shift_uv: np.ndarray,
    metrics: dict[str, float],
) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(13.2, 4.5), constrained_layout=True)
    for axis, silhouette, title in (
        (axes[0], before, "Transferred Series 7 pose"),
        (axes[1], after, "Heart-prior in-plane correction"),
    ):
        axis.imshow(silhouette, cmap="Reds", alpha=0.65)
        axis.contour(reference_mask, levels=[0.5], colors=["#34f5a6"], linewidths=1.0)
        axis.set_title(title)
        axis.axis("off")
    overlay = np.zeros((*after.shape, 3), dtype=np.float32)
    overlay[..., 0] = after.astype(np.float32)
    overlay[..., 1] = ndimage.binary_dilation(reference_mask, iterations=1).astype(np.float32)
    axes[2].imshow(overlay)
    axes[2].set_title(
        f"Frame {reference_frame}: shift ({shift_uv[0]:.0f}, {shift_uv[1]:.0f}) px\n"
        f"centerline coverage {metrics['reference_centerline_coverage_after']:.1%}"
    )
    axes[2].axis("off")
    figure.suptitle("Monoplane XA to CT heart-surface prior (research-only)")
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def _finite_number(value: object, label: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be a finite number")
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label} must be a finite number") from exc
    if not np.isfinite(number):
        raise ValueError(f"{label} must be a finite number")
    return number


def _nonempty_id(value: object, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a non-empty string")
    return value


def _optional_id(value: object, label: str) -> str | None:
    return None if value is None else _nonempty_id(value, label)


def _load_temporal_graphs(
    path: Path,
    expected_frame_count: int,
    detector_shape: tuple[int, int],
) -> list[VesselGraph]:
    def reject_constant(value: str) -> None:
        raise ValueError(f"temporal graph JSON contains invalid constant {value}")

    try:
        document = json.loads(
            path.read_text(encoding="utf-8"), parse_constant=reject_constant
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Unable to read temporal graph JSON: {path}") from exc
    if not isinstance(document, dict):
        raise ValueError("temporal graph document must be an object")
    if type(document.get("schema_version")) is not int or document["schema_version"] != 1:
        raise ValueError("temporal graph schema_version must be integer 1")
    expected_flow_convention = {
        "order": ["dx", "dy"],
        "direction": "previous_to_current",
        "mapping": "(y, x) -> (y + dy, x + dx)",
        "sampling": "deterministic_bilinear",
    }
    if document.get("flow_convention") != expected_flow_convention:
        raise ValueError("temporal graph flow_convention is malformed")
    if type(document.get("frame_count")) is not int:
        raise ValueError("temporal graph frame_count must be an integer")
    frames = document.get("frames")
    if (
        document["frame_count"] != expected_frame_count
        or not isinstance(frames, list)
        or len(frames) != expected_frame_count
    ):
        raise ValueError("temporal graph frame count does not match the series manifest")

    rows, columns = detector_shape
    graphs: list[VesselGraph] = []
    for expected_index, raw_frame in enumerate(frames):
        label = f"temporal graph frame {expected_index}"
        if not isinstance(raw_frame, dict):
            raise ValueError(f"{label} must be an object")
        if type(raw_frame.get("frame_index")) is not int or raw_frame["frame_index"] != expected_index:
            raise ValueError(f"{label} frame_index is inconsistent")
        if type(raw_frame.get("component_count")) is not int or raw_frame["component_count"] < 0:
            raise ValueError(f"{label} component_count must be a non-negative integer")
        raw_nodes = raw_frame.get("nodes")
        raw_edges = raw_frame.get("edges")
        qc = raw_frame.get("qc")
        if not isinstance(raw_nodes, list) or not isinstance(raw_edges, list):
            raise ValueError(f"{label} nodes and edges must be arrays")
        if not isinstance(qc, dict):
            raise ValueError(f"{label} qc must be an object")

        nodes: list[VesselNode] = []
        node_by_id: dict[str, VesselNode] = {}
        topology_by_local: dict[str, str] = {}
        topology_ids: set[str] = set()
        for node_index, raw_node in enumerate(raw_nodes):
            node_label = f"{label} node {node_index}"
            if not isinstance(raw_node, dict):
                raise ValueError(f"{node_label} must be an object")
            node_id = _nonempty_id(raw_node.get("node_id"), f"{node_label} node_id")
            topology_id = _nonempty_id(
                raw_node.get("topology_id"), f"{node_label} topology_id"
            )
            if node_id in node_by_id or topology_id in topology_ids:
                raise ValueError(f"{label} contains duplicate node identities")
            kind = raw_node.get("kind")
            visibility = raw_node.get("visibility")
            if kind not in {"root", "junction", "endpoint"}:
                raise ValueError(f"{node_label} kind is invalid")
            if visibility not in {"visible", "occluded"}:
                raise ValueError(f"{node_label} visibility is invalid")
            try:
                yx = np.asarray(raw_node.get("yx"), dtype=np.float64)
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValueError(f"{node_label} yx is malformed") from exc
            if yx.shape != (2,) or not np.all(np.isfinite(yx)):
                raise ValueError(f"{node_label} yx must contain two finite values")
            if not (0.0 <= yx[0] < rows and 0.0 <= yx[1] < columns):
                raise ValueError(f"{node_label} lies outside the detector")
            confidence = _finite_number(raw_node.get("confidence"), f"{node_label} confidence")
            if not 0.0 <= confidence <= 1.0:
                raise ValueError(f"{node_label} confidence must be within [0, 1]")
            node = VesselNode(
                node_id=node_id,
                kind=kind,
                yx=(float(yx[0]), float(yx[1])),
                topology_id=topology_id,
                parent_node_id=_optional_id(
                    raw_node.get("parent_node_id"), f"{node_label} parent_node_id"
                ),
                confidence=confidence,
                visibility=visibility,
            )
            nodes.append(node)
            node_by_id[node_id] = node
            topology_by_local[node_id] = topology_id
            topology_ids.add(topology_id)

        for node in nodes:
            parent = node.parent_node_id
            if parent is None:
                continue
            if parent in node_by_id or parent not in topology_ids:
                raise ValueError(
                    f"{label} node {node.node_id} parent_node_id must reference a stable node topology_id"
                )
            if parent == node.topology_id:
                raise ValueError(
                    f"{label} node {node.node_id} parent_node_id must not self-reference"
                )

        root_node_id = raw_frame.get("root_node_id")
        root_topology_id = raw_frame.get("root_topology_id")
        if nodes:
            root_node_id = _nonempty_id(root_node_id, f"{label} root_node_id")
            if root_node_id not in node_by_id:
                raise ValueError(f"{label} root_node_id refers to a missing node")
            if root_topology_id != topology_by_local[root_node_id]:
                raise ValueError(f"{label} root topology reference is inconsistent")
        elif root_node_id != "" or root_topology_id is not None:
            raise ValueError(f"{label} empty graph must have empty root references")

        edges: list[VesselEdge] = []
        edge_ids: set[str] = set()
        edge_topology_ids: set[str] = set()
        for edge_index, raw_edge in enumerate(raw_edges):
            edge_label = f"{label} edge {edge_index}"
            if not isinstance(raw_edge, dict):
                raise ValueError(f"{edge_label} must be an object")
            edge_id = _nonempty_id(raw_edge.get("edge_id"), f"{edge_label} edge_id")
            topology_id = _nonempty_id(
                raw_edge.get("topology_id"), f"{edge_label} topology_id"
            )
            if edge_id in edge_ids or topology_id in edge_topology_ids:
                raise ValueError(f"{label} contains duplicate edge identities")
            source_node_id = _nonempty_id(
                raw_edge.get("source_node_id"), f"{edge_label} source_node_id"
            )
            target_node_id = _nonempty_id(
                raw_edge.get("target_node_id"), f"{edge_label} target_node_id"
            )
            if source_node_id not in node_by_id or target_node_id not in node_by_id:
                raise ValueError(f"{edge_label} refers to a missing node")
            if raw_edge.get("source_topology_id") != topology_by_local[source_node_id]:
                raise ValueError(f"{edge_label} source topology reference is inconsistent")
            if raw_edge.get("target_topology_id") != topology_by_local[target_node_id]:
                raise ValueError(f"{edge_label} target topology reference is inconsistent")
            try:
                points = np.asarray(raw_edge.get("points_yx"), dtype=np.float64)
                radii = np.asarray(raw_edge.get("radius_px"), dtype=np.float64)
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValueError(f"{edge_label} sample arrays are malformed") from exc
            if points.shape != (64, 2) or not np.all(np.isfinite(points)):
                raise ValueError(f"{edge_label} points_yx must contain 64 finite samples")
            if radii.shape != (64,) or not np.all(np.isfinite(radii)) or np.any(radii <= 0.0):
                raise ValueError(f"{edge_label} radius_px must contain 64 finite positive samples")
            if np.any(
                (points[:, 0] < 0.0)
                | (points[:, 0] >= rows)
                | (points[:, 1] < 0.0)
                | (points[:, 1] >= columns)
            ):
                raise ValueError(f"{edge_label} points_yx lie outside the detector")
            if not np.array_equal(
                points[0], np.asarray(node_by_id[source_node_id].yx, dtype=np.float64)
            ) or not np.array_equal(
                points[-1], np.asarray(node_by_id[target_node_id].yx, dtype=np.float64)
            ):
                raise ValueError(
                    f"{edge_label} points_yx endpoints must exactly match its source and target nodes"
                )
            confidence = _finite_number(raw_edge.get("confidence"), f"{edge_label} confidence")
            if not 0.0 <= confidence <= 1.0:
                raise ValueError(f"{edge_label} confidence must be within [0, 1]")
            visibility = raw_edge.get("visibility")
            if visibility not in {"visible", "occluded"}:
                raise ValueError(f"{edge_label} visibility is invalid")
            edges.append(
                VesselEdge(
                    edge_id=edge_id,
                    source_node_id=source_node_id,
                    target_node_id=target_node_id,
                    points_yx=points,
                    confidence=confidence,
                    topology_id=topology_id,
                    parent_edge_id=_optional_id(
                        raw_edge.get("parent_edge_id"), f"{edge_label} parent_edge_id"
                    ),
                    radius_px=radii,
                    visibility=visibility,
                )
            )
            edge_ids.add(edge_id)
            edge_topology_ids.add(topology_id)

        for edge in edges:
            parent = edge.parent_edge_id
            if parent is None:
                continue
            if parent in edge_ids or parent not in edge_topology_ids:
                raise ValueError(
                    f"{label} edge {edge.edge_id} parent_edge_id must reference a stable edge topology_id"
                )
            if parent == edge.topology_id:
                raise ValueError(
                    f"{label} edge {edge.edge_id} parent_edge_id must not self-reference"
                )

        graphs.append(
            VesselGraph(
                frame_index=expected_index,
                nodes=tuple(nodes),
                edges=tuple(edges),
                root_node_id=str(root_node_id),
                component_count=int(raw_frame["component_count"]),
                qc=dict(qc),
            )
        )
    return graphs


def _empty_depth_result(reason: str) -> DepthOptimizationResult:
    return DepthOptimizationResult(
        points_lps_by_edge={},
        node_points_lps={},
        objective_terms={
            "heart_surface_distance": 0.0,
            "first_derivative_smoothness": 0.0,
            "second_derivative_smoothness": 0.0,
            "junction_equality_penalty": 0.0,
            "temporal_depth_penalty": 0.0,
            "weighted_total": 0.0,
        },
        quality_status="rejected",
        reasons=(reason,),
    )


def _edge_radii_mm(edge: VesselEdge, points_lps: np.ndarray, camera: Camera) -> np.ndarray:
    points_yx = np.asarray(edge.points_yx, dtype=np.float64)
    uv = points_yx[:, ::-1]
    detector_points = (
        camera.detector_center[None, :]
        + ((uv[:, 0] - camera.principal_uv[0]) * camera.col_spacing_mm)[:, None]
        * camera.u_axis[None, :]
        + ((uv[:, 1] - camera.principal_uv[1]) * camera.row_spacing_mm)[:, None]
        * camera.v_axis[None, :]
    )
    detector_depths = np.linalg.norm(detector_points - camera.source[None, :], axis=1)
    point_depths = np.linalg.norm(points_lps - camera.source[None, :], axis=1)
    detector_radius_mm = np.asarray(edge.radius_px, dtype=np.float64) * float(
        np.mean([camera.row_spacing_mm, camera.col_spacing_mm])
    )
    return np.clip(detector_radius_mm * point_depths / detector_depths, 0.55, 2.8)


def build_dynamic_3d(
    series_dir: Path,
    masks_dir: Path,
    output_dir: Path,
    multiview_path: Path | None = None,
    calibration_path: Path | None = None,
    temporal_dir: Path | None = None,
) -> dict[str, Any]:
    series_dir = Path(series_dir)
    temporal_dir = Path(temporal_dir) if temporal_dir is not None else series_dir / "temporal"
    series_dir, temporal_dir, output_dir, candidate_mode = _validated_dynamic_paths(
        series_dir, temporal_dir, output_dir
    )
    return build_then_publish_directory(
        output_dir,
        lambda staging_dir: _build_dynamic_3d_in_staging(
            series_dir,
            Path(masks_dir),
            staging_dir,
            multiview_path,
            calibration_path,
            temporal_dir,
            candidate_mode=candidate_mode,
            logical_output_dir=output_dir,
        ),
        seed_from_existing=False,
    )


def _build_dynamic_3d_in_staging(
    series_dir: Path,
    masks_dir: Path,
    output_dir: Path,
    multiview_path: Path | None,
    calibration_path: Path | None,
    temporal_dir: Path,
    *,
    candidate_mode: bool,
    logical_output_dir: Path,
) -> dict[str, Any]:
    if calibration_path is None:
        calibration_candidates = sorted(series_dir.glob("series_*_calibration.json"))
        if len(calibration_candidates) != 1:
            raise ValueError(
                f"Expected one calibration JSON in {series_dir}, found {len(calibration_candidates)}"
            )
        calibration_path = calibration_candidates[0]
    calibration_report = read_json(calibration_path)
    manifest = read_json(series_dir / "series_manifest.json")
    manifest_series = int(manifest["source_record"]["series_number"])
    calibration_series = int(calibration_report.get("series_number", manifest_series))
    if calibration_series != manifest_series:
        raise ValueError(
            f"Calibration series {calibration_series} does not match manifest series {manifest_series}"
        )
    run_id = str(calibration_report.get("run_id", f"series_{manifest_series:02d}"))
    temporal_acceptance_summary: dict[str, Any] | None = None
    temporal_acceptance_errors: list[str] = []
    if candidate_mode:
        (
            temporal_acceptance_summary,
            temporal_acceptance_errors,
        ) = _load_temporal_candidate_acceptance(
            temporal_dir, series_dir, run_id
        )
    camera = camera_from_calibration(calibration_report)
    frame_count = int(manifest["frame_png_count"])
    shape = (frame_count, camera.rows, camera.columns)
    repaired_mask_path = temporal_dir / "repaired_masks_u8.raw"
    temporal_mask_path = (
        repaired_mask_path
        if repaired_mask_path.is_file()
        else temporal_dir / "temporal_masks_u8.raw"
    )
    masks_raw = np.fromfile(temporal_mask_path, dtype=np.uint8)
    if masks_raw.size != int(np.prod(shape)):
        raise ValueError(f"Temporal mask size mismatch: {masks_raw.size} != {np.prod(shape)}")
    masks = masks_raw.reshape(shape) > 0
    graphs = _load_temporal_graphs(
        temporal_dir / "temporal_vessel_graphs.json",
        frame_count,
        (camera.rows, camera.columns),
    )

    heart_path = masks_dir / "heart.nii.gz"
    heart_vertices, _, _ = surface_from_mask(heart_path, profile="heart")
    heart_volume_points = foreground_points_from_mask(heart_path, stride=2)
    base_uv, heart_depth = project_lps(heart_vertices, camera)
    in_front = np.isfinite(base_uv).all(axis=1) & (heart_depth > 0)
    base_uv = base_uv[in_front]
    heart_depth = heart_depth[in_front]
    heart_volume_uv, _ = project_lps(heart_volume_points, camera)
    base_silhouette = splat_silhouette(heart_volume_uv, camera.rows, camera.columns, radius=1)
    shift_uv, reference_frame, alignment_metrics, reference_mask = estimate_principal_shift(base_silhouette, masks)
    corrected_principal = camera.principal_uv + shift_uv
    corrected_camera = replace(camera, principal_uv=corrected_principal)
    corrected_uv = base_uv + shift_uv[None, :]
    corrected_silhouette = shift_mask(base_silhouette, int(shift_uv[0]), int(shift_uv[1]))

    in_detector = (
        (corrected_uv[:, 0] >= -4)
        & (corrected_uv[:, 0] <= camera.columns + 3)
        & (corrected_uv[:, 1] >= -4)
        & (corrected_uv[:, 1] <= camera.rows + 3)
    )
    projected_surface = corrected_uv[in_detector]

    frames: list[dict[str, Any]] = []
    total_paths = 0
    total_points = 0
    total_nodes = 0
    total_edges = 0
    total_authoritative_points = 0
    all_validity: list[float] = []
    last_accepted_result: DepthOptimizationResult | None = None
    last_accepted_paths: list[dict[str, Any]] = []
    last_accepted_frame_index: int | None = None
    for graph in graphs:
        if graph.nodes:
            result = optimize_graph_depth(
                graph, corrected_camera, heart_vertices, last_accepted_result
            )
            if not graph.edges:
                result = replace(
                    result,
                    quality_status="rejected",
                    reasons=("no_observable_vessel_edges",),
                )
        else:
            result = _empty_depth_result("no_observable_vessel_graph")
        edge_audit = edge_depth_audit(result, heart_vertices)
        node_by_id = {node.node_id: node for node in graph.nodes}
        node_records = [
            {
                "node_id": node.node_id,
                "topology_id": node.topology_id,
                "parent_node_id": node.parent_node_id,
                "kind": node.kind,
                "yx": [float(node.yx[0]), float(node.yx[1])],
                "point_lps_mm": result.node_points_lps[
                    str(node.topology_id or node.node_id)
                ].tolist(),
                "confidence": float(node.confidence),
                "visibility": node.visibility,
            }
            for node in graph.nodes
        ]
        edge_records: list[dict[str, Any]] = []
        accepted_paths: list[dict[str, Any]] = []
        for edge in graph.edges:
            edge_key = str(edge.topology_id or edge.edge_id)
            points_lps = result.points_lps_by_edge[edge_key]
            radii_mm = _edge_radii_mm(edge, points_lps, corrected_camera)
            length_mm = float(np.linalg.norm(np.diff(points_lps, axis=0), axis=1).sum())
            edge_records.append(
                {
                    "edge_id": edge.edge_id,
                    "topology_id": edge.topology_id,
                    "parent_edge_id": edge.parent_edge_id,
                    "source_node_id": edge.source_node_id,
                    "target_node_id": edge.target_node_id,
                    "source_topology_id": node_by_id[edge.source_node_id].topology_id,
                    "target_topology_id": node_by_id[edge.target_node_id].topology_id,
                    "points_yx": np.asarray(edge.points_yx, dtype=np.float64).tolist(),
                    "points_lps_mm": points_lps.tolist(),
                    "radii_mm": radii_mm.tolist(),
                    "confidence": float(edge.confidence),
                    "visibility": edge.visibility,
                    "depth_audit": edge_audit[edge_key],
                }
            )
            accepted_paths.append(
                {
                    "edge_id": edge.edge_id,
                    "topology_id": edge.topology_id,
                    "points_lps_mm": points_lps.tolist(),
                    "radius_mm": float(np.median(radii_mm)),
                    "length_mm": length_mm,
                }
            )

        if result.quality_status == "accepted":
            frame_paths = accepted_paths
            display_source_frame_index = int(graph.frame_index)
            last_accepted_result = result
            last_accepted_paths = accepted_paths
            last_accepted_frame_index = int(graph.frame_index)
        else:
            frame_paths = last_accepted_paths
            display_source_frame_index = last_accepted_frame_index
        frame_record = {
            "frame_index": int(graph.frame_index),
            "root_node_id": graph.root_node_id,
            "root_topology_id": (
                node_by_id[graph.root_node_id].topology_id if graph.root_node_id else None
            ),
            "component_count": int(graph.component_count),
            "nodes": node_records,
            "edges": edge_records,
            "quality_status": result.quality_status,
            "reasons": list(result.reasons),
            "depth_audit": {
                "objective_terms": result.objective_terms,
                "reprojection_rms_px": reprojection_rms_px(
                    graph, result, corrected_camera
                ),
            },
            "display_source_frame_index": display_source_frame_index,
            "paths": frame_paths,
            "path_count": len(frame_paths),
            "point_count": int(sum(len(path["points_lps_mm"]) for path in frame_paths)),
            "heart_projection_valid_fraction": 1.0 if graph.edges else 0.0,
        }
        frames.append(frame_record)
        total_paths += frame_record["path_count"]
        total_points += frame_record["point_count"]
        total_nodes += len(node_records)
        total_edges += len(edge_records)
        total_authoritative_points += sum(len(edge["points_lps_mm"]) for edge in edge_records)
        if graph.edges:
            all_validity.extend([1.0] * len(graph.edges))

    output_dir.mkdir(parents=True, exist_ok=True)
    thorax_mesh_variants = export_thorax_mesh_variants(masks_dir, output_dir)
    registration_asset_path = (
        output_dir / thorax_mesh_variants.registration_asset.path
    )
    shutil.copyfile(registration_asset_path, output_dir / "thorax.glb")
    thorax_records = [item.to_legacy_dict() for item in thorax_mesh_variants.labels]
    build_alignment_figure(
        output_dir / "heart_prior_alignment.png",
        reference_mask,
        base_silhouette,
        corrected_silhouette,
        reference_frame,
        shift_uv,
        alignment_metrics,
    )

    multiview: dict[str, Any] | None = None
    if multiview_path and multiview_path.exists():
        summary = read_json(multiview_path)
        multiview = {
            "quality_status": summary.get("quality_status"),
            "final_edge_ncc": summary.get("final_edge_ncc"),
            "acceptance_edge_ncc": summary.get("acceptance_edge_ncc"),
            "used_for_depth": False,
        }

    depth_asset = {
        "schema_version": 2,
        "run_id": run_id,
        "series_number": manifest_series,
        "method": "graph-constrained monoplane ray depth optimization with a source-facing TotalSegmentator heart-surface prior",
        "quality_status": "diagnostic-only",
        "quality_gate_passed": False,
        "clinical_status": "research-only; not suitable for navigation or quantitative diagnosis",
        "interpretation": (
            "Each point stays on its original detector ray while graph-shared node depths preserve connected junctions. "
            "Monoplane XA cannot identify unique coronary depth; the CT heart surface and in-plane correction are priors, not proof of coronary position or an accepted 2D/3D registration."
        ),
        "depth_objective": {
            "heart_surface_distance": 1.0,
            "first_derivative_smoothness": 0.35,
            "second_derivative_smoothness": 0.20,
            "junction_equality_penalty": 0.50,
            "temporal_depth_penalty": 0.25,
            "monoplane_ambiguity": (
                "Depth is not uniquely observable; the CT heart surface is an initialization/prior."
            ),
        },
        "legacy_paths": {
            "authoritative": False,
            "lifecycle": "one-release compatibility data",
        },
        "frame_count": frame_count,
        "source_lps_mm": camera.source.round(6).tolist(),
        "original_principal_uv_px": camera.principal_uv.round(6).tolist(),
        "principal_shift_uv_px": shift_uv.round(3).tolist(),
        "corrected_principal_uv_px": corrected_principal.round(6).tolist(),
        "reference_frame": reference_frame,
        "alignment_metrics": alignment_metrics,
        "surface_mapping": {
            "heart_surface_vertices": int(len(heart_vertices)),
            "projected_surface_vertices": int(len(projected_surface)),
            "median_path_projection_valid_fraction": float(np.median(all_validity)) if all_validity else 0.0,
            "depth_ambiguity": "not uniquely observable along each monoplane ray; bounded optimization initialized by the source-facing heart-surface prior",
        },
        "multiview_attempt": multiview,
        "totals": {
            "paths": int(total_paths),
            "points": int(total_points),
            "nodes": int(total_nodes),
            "edges": int(total_edges),
            "authoritative_points": int(total_authoritative_points),
        },
        "frames": frames,
    }
    (output_dir / "dynamic_vessel_3d.json").write_text(
        json.dumps(
            depth_asset,
            ensure_ascii=False,
            separators=(",", ":"),
            allow_nan=False,
        ),
        encoding="utf-8",
    )
    report = {
        "schema_version": 2,
        "run_id": run_id,
        "series_number": manifest_series,
        "output": str(logical_output_dir),
        "thorax_glb": {"path": "thorax.glb", "byte_length": (output_dir / "thorax.glb").stat().st_size, "labels": thorax_records},
        "thorax_mesh_variants": thorax_mesh_variants.to_dict(),
        "dynamic_vessel": {
            "path": "dynamic_vessel_3d.json",
            "byte_length": (output_dir / "dynamic_vessel_3d.json").stat().st_size,
            "frames": frame_count,
            "paths": total_paths,
            "points": total_points,
            "nodes": total_nodes,
            "edges": total_edges,
            "authoritative_points": total_authoritative_points,
        },
        "alignment_figure": "heart_prior_alignment.png",
        "reference_frame": reference_frame,
        "principal_shift_uv_px": shift_uv.tolist(),
        "alignment_metrics": alignment_metrics,
        "quality_status": "diagnostic-only",
        "quality_gate_passed": False,
    }
    (output_dir / "dynamic_vessel_3d_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    if candidate_mode:
        connectivity_metrics, connectivity_errors = _dynamic_connectivity_metrics(
            depth_asset
        )
        acceptance = build_connectivity_acceptance_report(
            run_id=run_id,
            metrics=connectivity_metrics,
            audit_errors=[*temporal_acceptance_errors, *connectivity_errors],
        )
        assert temporal_acceptance_summary is not None
        acceptance["temporal_acceptance"] = temporal_acceptance_summary
        acceptance["promotion_authorized"] = False
        (output_dir / "connectivity_acceptance.json").write_text(
            json.dumps(
                acceptance, ensure_ascii=False, indent=2, allow_nan=False
            ),
            encoding="utf-8",
        )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--series-dir", type=Path, required=True)
    parser.add_argument("--masks-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--multiview", type=Path, default=None)
    parser.add_argument("--calibration", type=Path, default=None)
    parser.add_argument("--temporal-dir", type=Path, default=None)
    args = parser.parse_args()
    report = build_dynamic_3d(
        args.series_dir,
        args.masks_dir,
        args.output_dir,
        args.multiview,
        args.calibration,
        args.temporal_dir,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
