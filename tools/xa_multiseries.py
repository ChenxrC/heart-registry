"""Shared contracts for the grouped seven-run XA research workflow."""

from __future__ import annotations

import json
import hashlib
import math
import re
from pathlib import Path
from typing import Any, Iterable


ANATOMY_BY_SERIES = {
    1: ("rca", "右冠脉视角 1", True, "#ff9f43"),
    2: ("rca", "右冠脉视角 2", True, "#ffd166"),
    3: ("lca", "左冠脉视角 1", True, "#2ec4b6"),
    4: ("lca", "左冠脉视角 2", True, "#4cc9f0"),
    5: ("lca", "左冠脉视角 3", True, "#4895ef"),
    6: ("lca", "左冠脉视角 4", True, "#4361ee"),
    7: ("locator", "胸部定位序列", False, "#b7b7c9"),
}
PHASE_GROUPS = {
    "rca": ("series_01", "series_02"),
    "lca": ("series_03", "series_04", "series_05", "series_06"),
}
PHASE_RUN_STATUSES = {"automatic", "provisional", "insufficient"}
PHASE_METHOD = "image-derived coronary motion phase"
PRIMARY_PHASE_INPUTS = {
    "manifest",
    "cine",
    "temporal_centerlines",
    "dynamic_vessel_3d",
}


def run_id(series_number: int) -> str:
    return f"series_{series_number:02d}"


def load_run_specs(manifest_path: Path) -> list[dict[str, Any]]:
    """Load and classify the seven XA acquisitions from the audited manifest."""
    payload = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    source_root = Path(payload["source_root"])
    records = sorted(payload["records"], key=lambda item: int(item["series_number"]))
    specs: list[dict[str, Any]] = []
    for record in records:
        number = int(record["series_number"])
        if number not in ANATOMY_BY_SERIES:
            raise ValueError(f"Unsupported XA series number: {number}")
        anatomy_group, label, requires_vessel, color = ANATOMY_BY_SERIES[number]
        specs.append(
            {
                "run_id": run_id(number),
                "series_number": number,
                "series_label": label,
                "anatomy_group": anatomy_group,
                "requires_vessel": requires_vessel,
                "color": color,
                "dicom_path": str(source_root / record["path"]),
                "dicom_name": record["path"],
                "frame_count": int(record["frame_count"]),
                "frame_time_ms": float(record["frame_time_ms"]),
                "primary_angle_deg": float(record["primary_angle_deg"]),
                "secondary_angle_deg": float(record["secondary_angle_deg"]),
                "record": record,
            }
        )
    expected = list(range(1, 8))
    actual = [item["series_number"] for item in specs]
    if actual != expected:
        raise ValueError(f"Expected XA series {expected}, found {actual}")
    return specs


def build_pipeline_plan(
    runs: Iterable[dict[str, Any]],
    output_root: Path,
    *,
    python_executable: str,
    reference_report: Path | None = None,
    reference_scene: Path | None = None,
    masks_dir: Path | None = None,
) -> list[dict[str, Any]]:
    """Create explicit per-run commands without executing them."""
    output_root = Path(output_root)
    plan: list[dict[str, Any]] = []
    for run in runs:
        identifier = str(run["run_id"])
        series_dir = output_root / identifier
        dicom = str(run["dicom_path"])
        calibration = series_dir / f"{identifier}_calibration.json"
        steps: list[dict[str, Any]] = [
            {
                "name": "extract",
                "command": [python_executable, "tools/extract_xa_cine.py", "--dicom", dicom, "--output-dir", str(series_dir)],
            }
        ]
        if run["requires_vessel"]:
            source_record = run.get("record")
            if not isinstance(source_record, dict):
                raise ValueError(f"Missing source manifest record for {identifier}")
            frame_count = int(run["frame_count"])
            rows = int(source_record["rows"])
            columns = int(source_record["columns"])
            if frame_count <= 0 or rows <= 0 or columns <= 0:
                raise ValueError(f"Invalid source dimensions for {identifier}")
            temporal_validation = {
                "expected_frame_count": frame_count,
                "expected_raw_byte_length": frame_count * rows * columns,
            }
            temporal_dir = series_dir / "temporal"
            steps.extend(
                [
                    {
                        "name": "segment",
                        "command": [python_executable, "tools/run_xa_vessel_segmentation.py", "--dicom", dicom, "--output-dir", str(series_dir / "segmentation")],
                    },
                    {
                        "name": "temporal-repair",
                        "command": [python_executable, "tools/run_xa_temporal_fusion.py", "--dicom", dicom, "--segmentation-dir", str(series_dir / "segmentation"), "--output-dir", str(temporal_dir)],
                        "atomic_output_dir": str(temporal_dir),
                        "validation_profile": "temporal",
                        "validation_context": temporal_validation,
                    },
                    {
                        "name": "topology-graph",
                        "validate_dir": str(temporal_dir),
                        "validation_profile": "topology",
                        "validation_context": temporal_validation,
                    },
                ]
            )
        calibrate = [
            python_executable,
            "tools/calibrate_xa_run_to_ct.py",
            "--dicom",
            dicom,
            "--output",
            str(calibration),
        ]
        if reference_report is not None:
            calibrate.extend(["--reference", str(reference_report)])
        if reference_scene is not None:
            calibrate.extend(["--scene", str(reference_scene)])
        steps.append({"name": "calibrate", "command": calibrate})
        if run["requires_vessel"]:
            dynamic_validation = {
                "expected_run_id": identifier,
                "expected_series_number": int(run["series_number"]),
                "expected_frame_count": int(run["frame_count"]),
            }
            mesh = [
                python_executable,
                "tools/build_dynamic_vessel_3d.py",
                "--series-dir",
                str(series_dir),
                "--calibration",
                str(calibration),
                "--output-dir",
                str(series_dir / "dynamic_3d"),
            ]
            if masks_dir is not None:
                mesh.extend(["--masks-dir", str(masks_dir)])
            dynamic_dir = series_dir / "dynamic_3d"
            steps.extend(
                [
                    {
                        "name": "graph-depth",
                        "command": mesh,
                        "atomic_output_dir": str(dynamic_dir),
                        "validation_profile": "dynamic",
                        "validation_context": dynamic_validation,
                    },
                    {
                        "name": "connected-mesh-package",
                        "validate_dir": str(dynamic_dir),
                        "validation_profile": "connected-mesh",
                        "validation_context": dynamic_validation,
                    },
                ]
            )
        plan.append({"run_id": identifier, "series_dir": str(series_dir), "steps": steps})
    return plan


def _finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _exact_nonnegative_int(value: Any, label: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{label} must be a nonnegative integer")
    return value


def _strict_point3(value: Any, label: str) -> tuple[float, float, float]:
    if (
        not isinstance(value, list)
        or len(value) != 3
        or not all(_finite_number(component) for component in value)
    ):
        raise ValueError(f"{label} must be a finite numeric LPS point")
    return tuple(float(component) for component in value)


def validate_repair_links(report: dict[str, Any], frame_count: int) -> dict[str, int]:
    """Validate optional detector-space repair-link geometry and count decisions."""
    links = report.get("links")
    counts = {"accepted": 0, "review": 0, "rejected": 0}
    if links is None:
        return counts
    if not isinstance(links, list):
        raise ValueError("repair_audit.links must be an array")
    for link_position, link in enumerate(links):
        label = f"repair_audit.links[{link_position}]"
        if not isinstance(link, dict):
            raise ValueError(f"{label} must be an object")
        frame_index = _exact_nonnegative_int(link.get("frame_index"), f"{label}.frame_index")
        if frame_index >= frame_count:
            raise ValueError(f"{label}.frame_index is out of range")
        decision = link.get("decision")
        if decision not in counts:
            raise ValueError(f"{label}.decision must be accepted, review or rejected")
        confidence = link.get("confidence")
        if not _finite_number(confidence) or not 0.0 <= float(confidence) <= 1.0:
            raise ValueError(f"{label}.confidence must be finite and within [0, 1]")
        points = link.get("points_yx")
        if not isinstance(points, list) or len(points) < 2:
            raise ValueError(f"{label}.points_yx must contain at least two detector points")
        for point_position, point in enumerate(points):
            if (
                not isinstance(point, list)
                or len(point) != 2
                or not all(_finite_number(value) for value in point)
            ):
                raise ValueError(
                    f"{label}.points_yx[{point_position}] must be a finite numeric detector point"
                )
        counts[decision] += 1
    return counts


def _reject_nonfinite_numbers(value: Any, label: str) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"{label} contains a non-finite number")
    if isinstance(value, dict):
        for key, item in value.items():
            _reject_nonfinite_numbers(item, f"{label}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _reject_nonfinite_numbers(item, f"{label}[{index}]")


def _validate_depth_audit(value: Any, label: str) -> None:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    stack: list[tuple[Any, str]] = [(value, label)]
    while stack:
        current, current_label = stack.pop()
        if isinstance(current, bool):
            raise ValueError(f"{current_label} must be a numeric value, not boolean")
        if isinstance(current, (int, float)):
            if not _finite_number(current):
                raise ValueError(f"{current_label} must be finite")
            continue
        if isinstance(current, dict):
            stack.extend((item, f"{current_label}.{key}") for key, item in current.items())
        elif isinstance(current, list):
            stack.extend(
                (item, f"{current_label}[{index}]")
                for index, item in enumerate(current)
            )
        elif current is not None and not isinstance(current, str):
            raise ValueError(f"{current_label} contains an unsupported value")


def validate_dynamic_vessel_document(document: dict[str, Any]) -> None:
    """Fail closed on malformed authoritative schema-v2 vessel geometry."""
    if not isinstance(document, dict) or type(document.get("schema_version")) is not int:
        raise ValueError("Dynamic vessel document must declare integer schema_version 2")
    if document["schema_version"] != 2:
        raise ValueError("Dynamic vessel document schema_version must be 2")
    frame_count = _exact_nonnegative_int(document.get("frame_count"), "frame_count")
    if frame_count <= 0:
        raise ValueError("frame_count must be positive")
    frames = document.get("frames")
    if not isinstance(frames, list) or len(frames) != frame_count:
        raise ValueError("Dynamic vessel frame_count does not match frames")

    frame_statuses: list[str] = []
    display_source_indices: list[int | None] = []
    for frame_position, frame in enumerate(frames):
        label = f"frames[{frame_position}]"
        if not isinstance(frame, dict):
            raise ValueError(f"{label} must be an object")
        frame_index = _exact_nonnegative_int(frame.get("frame_index"), f"{label}.frame_index")
        if frame_index != frame_position:
            raise ValueError(f"{label}.frame_index must be ordered and exact")
        quality_status = frame.get("quality_status")
        if quality_status not in {"accepted", "rejected"}:
            raise ValueError(f"{label}.quality_status must be accepted or rejected")
        frame_statuses.append(quality_status)
        component_count = _exact_nonnegative_int(
            frame.get("component_count"), f"{label}.component_count"
        )
        if quality_status == "accepted" and component_count != 1:
            raise ValueError(f"{label} accepted graph must have exactly one component")
        nodes = frame.get("nodes")
        edges = frame.get("edges")
        if not isinstance(nodes, list) or not isinstance(edges, list):
            raise ValueError(f"{label} nodes and edges must be arrays")

        node_points: dict[str, tuple[float, float, float]] = {}
        local_to_topology: dict[str, str] = {}
        for node_position, node in enumerate(nodes):
            node_label = f"{label}.nodes[{node_position}]"
            if not isinstance(node, dict):
                raise ValueError(f"{node_label} must be an object")
            topology_id = node.get("topology_id")
            if (
                not isinstance(topology_id, str)
                or not topology_id
                or topology_id.strip() != topology_id
                or topology_id in node_points
            ):
                raise ValueError(f"{node_label}.topology_id must be unique and nonempty")
            node_points[topology_id] = _strict_point3(
                node.get("point_lps_mm"), f"{node_label}.point_lps_mm"
            )
            node_id = node.get("node_id")
            if (
                not isinstance(node_id, str)
                or not node_id
                or node_id.strip() != node_id
                or node_id in local_to_topology
            ):
                raise ValueError(f"{node_label}.node_id must be unique and nonempty")
            local_to_topology[node_id] = topology_id
            if "confidence" in node and (
                not _finite_number(node["confidence"])
                or not 0.0 <= float(node["confidence"]) <= 1.0
            ):
                raise ValueError(f"{node_label}.confidence must be finite and within [0, 1]")

        edge_ids: set[str] = set()
        local_edge_ids: set[str] = set()
        adjacency = {topology_id: set() for topology_id in node_points}
        for edge_position, edge in enumerate(edges):
            edge_label = f"{label}.edges[{edge_position}]"
            if not isinstance(edge, dict):
                raise ValueError(f"{edge_label} must be an object")
            topology_id = edge.get("topology_id")
            if (
                not isinstance(topology_id, str)
                or not topology_id
                or topology_id.strip() != topology_id
                or topology_id in edge_ids
            ):
                raise ValueError(f"{edge_label}.topology_id must be unique and nonempty")
            edge_ids.add(topology_id)
            edge_id = edge.get("edge_id")
            if (
                not isinstance(edge_id, str)
                or not edge_id
                or edge_id.strip() != edge_id
                or edge_id in local_edge_ids
            ):
                raise ValueError(f"{edge_label}.edge_id must be unique and nonempty")
            local_edge_ids.add(edge_id)
            if "confidence" in edge and (
                not _finite_number(edge["confidence"])
                or not 0.0 <= float(edge["confidence"]) <= 1.0
            ):
                raise ValueError(f"{edge_label}.confidence must be finite and within [0, 1]")
            source = edge.get("source_topology_id")
            target = edge.get("target_topology_id")
            if source not in node_points or target not in node_points:
                raise ValueError(f"{edge_label} references a missing node")
            if source == target:
                raise ValueError(f"{edge_label} must not be a self-loop")
            source_node_id = edge.get("source_node_id")
            target_node_id = edge.get("target_node_id")
            if (
                local_to_topology.get(source_node_id) != source
                or local_to_topology.get(target_node_id) != target
            ):
                raise ValueError(f"{edge_label} local and topology node references disagree")
            adjacency[source].add(target)
            adjacency[target].add(source)
            points = edge.get("points_lps_mm")
            radii = edge.get("radii_mm")
            if not isinstance(points, list) or len(points) < 2:
                raise ValueError(f"{edge_label}.points_lps_mm must contain two points")
            parsed_points = [
                _strict_point3(point, f"{edge_label}.points_lps_mm[{index}]")
                for index, point in enumerate(points)
            ]
            if (
                not isinstance(radii, list)
                or len(radii) != len(parsed_points)
                or not all(_finite_number(radius) and float(radius) > 0.0 for radius in radii)
            ):
                raise ValueError(f"{edge_label}.radii_mm must be finite, positive and sample-aligned")
            if parsed_points[0] != node_points[source] or parsed_points[-1] != node_points[target]:
                raise ValueError(f"{edge_label} endpoints must exactly equal referenced nodes")
            if "depth_audit" in edge:
                _validate_depth_audit(edge["depth_audit"], f"{edge_label}.depth_audit")
        if nodes:
            unseen = set(node_points)
            computed_components = 0
            while unseen:
                computed_components += 1
                pending = [next(iter(unseen))]
                while pending:
                    current = pending.pop()
                    if current not in unseen:
                        continue
                    unseen.remove(current)
                    pending.extend(adjacency[current] & unseen)
        else:
            computed_components = 0
        if component_count != computed_components:
            raise ValueError(f"{label}.component_count disagrees with graph connectivity")
        if quality_status == "accepted" and (
            not nodes
            or not edges
            or any(not neighbors for neighbors in adjacency.values())
            or computed_components != 1
        ):
            raise ValueError(f"{label} accepted graph must be one connected graph")
        _validate_depth_audit(frame.get("depth_audit", {}), f"{label}.depth_audit")
        display_index = frame.get("display_source_frame_index")
        if display_index is not None:
            display_index = _exact_nonnegative_int(
                display_index, f"{label}.display_source_frame_index"
            )
            if display_index >= frame_count:
                raise ValueError(f"{label}.display_source_frame_index is out of range")
        display_source_indices.append(display_index)

    for frame_position, (quality_status, display_index) in enumerate(
        zip(frame_statuses, display_source_indices, strict=True)
    ):
        if quality_status == "accepted":
            if display_index != frame_position:
                raise ValueError("Accepted frame must display its own observation")
        elif display_index is not None:
            if display_index == frame_position or frame_statuses[display_index] != "accepted":
                raise ValueError("Rejected frame fallback must reference another accepted frame")


def _load_json_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Invalid {label}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    return value


def _local_asset_path(directory: Path, relative_path: Any, label: str) -> Path:
    if not isinstance(relative_path, str) or not relative_path.strip():
        raise ValueError(f"{label}.path must be a nonempty relative path")
    relative = Path(relative_path)
    if relative.is_absolute():
        raise ValueError(f"{label}.path must remain inside the stage directory")
    root = directory.resolve()
    candidate = (directory / relative).resolve()
    if candidate != root and root not in candidate.parents:
        raise ValueError(f"{label}.path escapes the stage directory")
    if not candidate.is_file():
        raise ValueError(f"{label} asset is missing: {relative_path}")
    return candidate


def _validate_file_asset(
    directory: Path,
    record: Any,
    label: str,
    *,
    expected_path: str | None = None,
) -> Path:
    if not isinstance(record, dict):
        raise ValueError(f"{label} must be an asset object")
    relative_path = record.get("path")
    if expected_path is not None and relative_path != expected_path:
        raise ValueError(f"{label}.path must be {expected_path}")
    path = _local_asset_path(directory, relative_path, label)
    reported_size = _exact_nonnegative_int(record.get("byte_length"), f"{label}.byte_length")
    if reported_size <= 0 or path.stat().st_size != reported_size:
        raise ValueError(f"{label}.byte_length disagrees with the staged asset")
    digest = record.get("sha256")
    if digest is not None:
        if not isinstance(digest, str) or re.fullmatch(r"[0-9a-fA-F]{64}", digest) is None:
            raise ValueError(f"{label}.sha256 must contain 64 hexadecimal characters")
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual.lower() != digest.lower():
            raise ValueError(f"{label}.sha256 disagrees with the staged asset")
    return path


def _validate_temporal_playback_assets(
    directory: Path,
    temporal: dict[str, Any],
    *,
    expected_raw_byte_length: int | None = None,
) -> None:
    assets = temporal.get("assets")
    if not isinstance(assets, dict):
        raise ValueError("temporal_report.assets must be an object")
    expected = {
        "probabilities": "temporal_probabilities_u8.raw",
        "masks": "temporal_masks_u8.raw",
        "centerlines": "temporal_centerlines_u8.raw",
        "repaired_probabilities": "repaired_probabilities_u8.raw",
        "repaired_masks": "repaired_masks_u8.raw",
        "repaired_centerlines": "repaired_centerlines_u8.raw",
    }
    sizes: set[int] = set()
    for key, relative_path in expected.items():
        record = assets.get(key)
        _validate_file_asset(
            directory,
            record,
            f"temporal_report.assets.{key}",
            expected_path=relative_path,
        )
        sizes.add(record["byte_length"])
    if len(sizes) != 1:
        raise ValueError("Temporal baseline and repaired raw stacks must have equal byte lengths")
    actual_size = next(iter(sizes))
    if expected_raw_byte_length is not None:
        expected_size = _exact_nonnegative_int(
            expected_raw_byte_length, "expected_raw_byte_length"
        )
        if expected_size <= 0 or actual_size != expected_size:
            raise ValueError(
                "Temporal raw byte length disagrees with the source manifest dimensions"
            )


def _graph_component_count(
    node_ids: Iterable[str], edge_pairs: Iterable[tuple[str, str]]
) -> int:
    adjacency = {node_id: set() for node_id in node_ids}
    for source_id, target_id in edge_pairs:
        adjacency[source_id].add(target_id)
        adjacency[target_id].add(source_id)
    remaining = set(adjacency)
    components = 0
    while remaining:
        components += 1
        stack = [remaining.pop()]
        while stack:
            current = stack.pop()
            neighbours = adjacency[current] & remaining
            remaining.difference_update(neighbours)
            stack.extend(neighbours)
    return components


def _validate_thorax_assets(directory: Path, report: dict[str, Any]) -> None:
    _validate_file_asset(
        directory,
        report.get("thorax_glb"),
        "dynamic_vessel_3d_report.thorax_glb",
        expected_path="thorax.glb",
    )
    alignment = report.get("alignment_figure")
    alignment_path = _local_asset_path(
        directory, alignment, "dynamic_vessel_3d_report.alignment_figure"
    )
    if alignment != "heart_prior_alignment.png" or alignment_path.stat().st_size <= 0:
        raise ValueError("dynamic_vessel_3d_report.alignment_figure is invalid")
    variants = report.get("thorax_mesh_variants")
    if not isinstance(variants, dict):
        raise ValueError("dynamic_vessel_3d_report.thorax_mesh_variants must be an object")
    if type(variants.get("schema_version")) is not int or variants["schema_version"] != 2:
        raise ValueError("thorax_mesh_variants.schema_version must be integer 2")
    persisted = _load_json_object(
        directory / "thorax_surface_report.json", "thorax surface report"
    )
    if persisted != variants:
        raise ValueError("thorax_mesh_variants disagrees with thorax_surface_report.json")
    assets = variants.get("assets")
    if not isinstance(assets, dict):
        raise ValueError("thorax_mesh_variants.assets must be an object")
    for key in ("registration", "display"):
        _validate_file_asset(
            directory, assets.get(key), f"thorax_mesh_variants.assets.{key}"
        )


def _dynamic_geometry_totals(document: dict[str, Any]) -> dict[str, int]:
    totals = {key: 0 for key in ("paths", "points", "nodes", "edges", "authoritative_points")}
    for frame_position, frame in enumerate(document["frames"]):
        paths = frame.get("paths", [])
        if not isinstance(paths, list):
            raise ValueError(f"frames[{frame_position}].paths must be an array")
        point_count = 0
        for path_position, path in enumerate(paths):
            points = path.get("points_lps_mm") if isinstance(path, dict) else None
            if not isinstance(points, list):
                raise ValueError(
                    f"frames[{frame_position}].paths[{path_position}].points_lps_mm must be an array"
                )
            point_count += len(points)
        if "path_count" in frame and _exact_nonnegative_int(
            frame["path_count"], f"frames[{frame_position}].path_count"
        ) != len(paths):
            raise ValueError(f"frames[{frame_position}].path_count disagrees with paths")
        if "point_count" in frame and _exact_nonnegative_int(
            frame["point_count"], f"frames[{frame_position}].point_count"
        ) != point_count:
            raise ValueError(f"frames[{frame_position}].point_count disagrees with paths")
        totals["paths"] += len(paths)
        totals["points"] += point_count
        totals["nodes"] += len(frame["nodes"])
        totals["edges"] += len(frame["edges"])
        totals["authoritative_points"] += sum(
            len(edge["points_lps_mm"]) for edge in frame["edges"]
        )
    return totals


def _validate_indexed_frames(document: dict[str, Any], label: str) -> int:
    frame_count = _exact_nonnegative_int(document.get("frame_count"), f"{label}.frame_count")
    if frame_count <= 0:
        raise ValueError(f"{label}.frame_count must be positive")
    frames = document.get("frames")
    if not isinstance(frames, list) or len(frames) != frame_count:
        raise ValueError(f"{label} frame_count does not match frames")
    for position, frame in enumerate(frames):
        if not isinstance(frame, dict):
            raise ValueError(f"{label}.frames[{position}] must be an object")
        frame_index = _exact_nonnegative_int(
            frame.get("frame_index"), f"{label}.frames[{position}].frame_index"
        )
        if frame_index != position:
            raise ValueError(f"{label}.frames[{position}].frame_index must equal its position")
    _reject_nonfinite_numbers(document, label)
    return frame_count


def validate_stage_artifacts(
    profile: str,
    directory: Path,
    *,
    expected_run_id: str | None = None,
    expected_series_number: int | None = None,
    expected_frame_count: int | None = None,
    expected_raw_byte_length: int | None = None,
) -> dict[str, Any]:
    """Validate one pipeline stage without mutating its published directory."""
    directory = Path(directory)
    if profile in {"temporal", "topology"}:
        temporal = _load_json_object(directory / "temporal_report.json", "temporal report")
        repair = _load_json_object(directory / "repair_audit.json", "repair audit")
        graph = _load_json_object(
            directory / "temporal_vessel_graphs.json", "temporal vessel graph"
        )
        if type(temporal.get("schema_version")) is not int or temporal["schema_version"] != 2:
            raise ValueError("temporal_report schema_version must be integer 2")
        if type(repair.get("schema_version")) is not int or repair["schema_version"] != 1:
            raise ValueError("repair_audit schema_version must be integer 1")
        if type(graph.get("schema_version")) is not int or graph["schema_version"] != 1:
            raise ValueError("temporal_vessel_graphs schema_version must be integer 1")
        counts = {
            _validate_indexed_frames(temporal, "temporal_report"),
            _validate_indexed_frames(repair, "repair_audit"),
            _validate_indexed_frames(graph, "temporal_vessel_graphs"),
        }
        if len(counts) != 1:
            raise ValueError("Temporal stage frame counts disagree")
        frame_count = next(iter(counts))
        if expected_frame_count is not None:
            expected_count = _exact_nonnegative_int(
                expected_frame_count, "expected_frame_count"
            )
            if expected_count <= 0 or frame_count != expected_count:
                raise ValueError("Temporal stage frame count disagrees with source manifest")
        _validate_temporal_playback_assets(
            directory,
            temporal,
            expected_raw_byte_length=expected_raw_byte_length,
        )
        for field in ("method", "clinical_status"):
            value = repair.get(field)
            if type(value) is not str or not value.strip():
                raise ValueError(f"repair_audit.{field} must be a nonempty string")
        for position, frame in enumerate(repair["frames"]):
            repair_label = f"repair_audit.frames[{position}]"
            if frame.get("status") not in {"accepted", "rejected"}:
                raise ValueError(f"{repair_label}.status is invalid")
            for key in ("accepted_pixels", "rejected_pixels"):
                _exact_nonnegative_int(frame.get(key), f"{repair_label}.{key}")
            growth = frame.get("area_growth_fraction")
            if not _finite_number(growth) or float(growth) < 0.0:
                raise ValueError(f"{repair_label}.area_growth_fraction must be finite and nonnegative")
            reasons = frame.get("reasons")
            if not isinstance(reasons, list) or not all(isinstance(reason, str) for reason in reasons):
                raise ValueError(f"{repair_label}.reasons must be an array of strings")
        validate_repair_links(repair, frame_count)
        for position, frame in enumerate(graph["frames"]):
            graph_label = f"temporal_vessel_graphs.frames[{position}]"
            component_count = _exact_nonnegative_int(
                frame.get("component_count"), f"{graph_label}.component_count"
            )
            nodes = frame.get("nodes")
            edges = frame.get("edges")
            if not isinstance(nodes, list) or not isinstance(edges, list):
                raise ValueError(f"{graph_label} nodes and edges must be arrays")
            node_ids: dict[str, str] = {}
            node_topology_ids: set[str] = set()
            for node_position, node in enumerate(nodes):
                node_label = f"{graph_label}.nodes[{node_position}]"
                if not isinstance(node, dict):
                    raise ValueError(f"{node_label} must be an object")
                node_id = node.get("node_id")
                topology_id = node.get("topology_id")
                if not isinstance(node_id, str) or not node_id or node_id in node_ids:
                    raise ValueError(f"{node_label}.node_id must be unique and nonempty")
                if (
                    not isinstance(topology_id, str)
                    or not topology_id
                    or topology_id in node_topology_ids
                ):
                    raise ValueError(f"{node_label}.topology_id must be unique and nonempty")
                node_ids[node_id] = topology_id
                node_topology_ids.add(topology_id)
            edge_ids: set[str] = set()
            edge_topology_ids: set[str] = set()
            edge_pairs: list[tuple[str, str]] = []
            for edge_position, edge in enumerate(edges):
                edge_label = f"{graph_label}.edges[{edge_position}]"
                if not isinstance(edge, dict):
                    raise ValueError(f"{edge_label} must be an object")
                edge_id = edge.get("edge_id")
                topology_id = edge.get("topology_id")
                if not isinstance(edge_id, str) or not edge_id or edge_id in edge_ids:
                    raise ValueError(f"{edge_label}.edge_id must be unique and nonempty")
                if (
                    not isinstance(topology_id, str)
                    or not topology_id
                    or topology_id in edge_topology_ids
                ):
                    raise ValueError(f"{edge_label}.topology_id must be unique and nonempty")
                edge_ids.add(edge_id)
                edge_topology_ids.add(topology_id)
                source_id = edge.get("source_node_id")
                target_id = edge.get("target_node_id")
                if source_id not in node_ids or target_id not in node_ids or source_id == target_id:
                    raise ValueError(f"{edge_label} has invalid node references")
                if (
                    edge.get("source_topology_id") != node_ids[source_id]
                    or edge.get("target_topology_id") != node_ids[target_id]
                ):
                    raise ValueError(f"{edge_label} local and topology references disagree")
                edge_pairs.append((source_id, target_id))
            computed_components = _graph_component_count(node_ids, edge_pairs)
            if component_count != computed_components:
                raise ValueError(
                    f"{graph_label}.component_count disagrees with graph connectivity"
                )
        return {"profile": profile, "frame_count": frame_count}
    if profile in {"dynamic", "connected-mesh"}:
        depth = _load_json_object(
            directory / "dynamic_vessel_3d.json", "dynamic vessel document"
        )
        report = _load_json_object(
            directory / "dynamic_vessel_3d_report.json", "dynamic vessel report"
        )
        validate_dynamic_vessel_document(depth)
        if type(report.get("schema_version")) is not int or report["schema_version"] != 2:
            raise ValueError("dynamic vessel report schema_version must be integer 2")
        _reject_nonfinite_numbers(report, "dynamic_vessel_3d_report")
        dynamic_record = report.get("dynamic_vessel")
        if not isinstance(dynamic_record, dict):
            raise ValueError("Dynamic vessel report record is missing")
        reported_frames = dynamic_record.get("frames")
        if type(reported_frames) is not int or reported_frames != depth["frame_count"]:
            raise ValueError("Dynamic vessel report frame count disagrees")
        payload_run = depth.get("run_id")
        report_run = report.get("run_id")
        if type(payload_run) is not str or not payload_run.strip():
            raise ValueError("run_id must be a nonempty string")
        if type(report_run) is not str or not report_run.strip():
            raise ValueError("dynamic_vessel_3d_report.run_id must be a nonempty string")
        if report_run != payload_run:
            raise ValueError("Dynamic vessel payload/report run_id disagree")
        payload_series = _exact_nonnegative_int(depth.get("series_number"), "series_number")
        report_series = _exact_nonnegative_int(
            report.get("series_number"), "dynamic_vessel_3d_report.series_number"
        )
        if payload_series != report_series:
            raise ValueError("Dynamic vessel payload/report series_number disagree")
        if payload_series <= 0:
            raise ValueError("series_number must be positive")
        reference_frame = _exact_nonnegative_int(depth.get("reference_frame"), "reference_frame")
        report_reference = _exact_nonnegative_int(
            report.get("reference_frame"), "dynamic_vessel_3d_report.reference_frame"
        )
        if reference_frame >= depth["frame_count"] or reference_frame != report_reference:
            raise ValueError("Dynamic vessel payload/report reference_frame disagree")
        if dynamic_record.get("path") != "dynamic_vessel_3d.json":
            raise ValueError("Dynamic vessel report path is invalid")
        reported_size = dynamic_record.get("byte_length")
        actual_size = (directory / "dynamic_vessel_3d.json").stat().st_size
        if type(reported_size) is not int or reported_size != actual_size:
            raise ValueError("Dynamic vessel report byte_length disagrees")
        computed_totals = _dynamic_geometry_totals(depth)
        payload_totals = depth.get("totals")
        if not isinstance(payload_totals, dict):
            raise ValueError("Dynamic vessel payload totals are missing")
        for key, computed in computed_totals.items():
            payload_value = _exact_nonnegative_int(payload_totals.get(key), f"totals.{key}")
            report_value = _exact_nonnegative_int(
                dynamic_record.get(key), f"dynamic_vessel_3d_report.dynamic_vessel.{key}"
            )
            if payload_value != computed or report_value != computed:
                raise ValueError(f"Dynamic vessel {key} totals disagree")
        for owner, document in (("payload", depth), ("report", report)):
            if not isinstance(document.get("alignment_metrics"), dict):
                raise ValueError(f"{owner} alignment_metrics must be an object")
            quality_status = document.get("quality_status")
            if type(quality_status) is not str or not quality_status.strip():
                raise ValueError(f"{owner} quality_status must be a nonempty string")
            if type(document.get("quality_gate_passed")) is not bool:
                raise ValueError(f"{owner} quality_gate_passed must be boolean")
        for key in ("alignment_metrics", "quality_status", "quality_gate_passed"):
            if report[key] != depth[key]:
                raise ValueError(f"Dynamic vessel payload/report {key} disagree")
        if expected_run_id is not None:
            if type(expected_run_id) is not str or not expected_run_id.strip():
                raise ValueError("expected_run_id must be a nonempty string")
            if payload_run != expected_run_id:
                raise ValueError("Dynamic vessel run_id disagrees with pipeline manifest")
        if expected_series_number is not None:
            expected_series = _exact_nonnegative_int(
                expected_series_number, "expected_series_number"
            )
            if expected_series <= 0 or payload_series != expected_series:
                raise ValueError("Dynamic vessel series_number disagrees with pipeline manifest")
        if expected_frame_count is not None:
            expected_count = _exact_nonnegative_int(
                expected_frame_count, "expected_frame_count"
            )
            if expected_count <= 0 or depth["frame_count"] != expected_count:
                raise ValueError("Dynamic vessel frame_count disagrees with pipeline manifest")
        payload_shift = depth.get("principal_shift_uv_px")
        report_shift = report.get("principal_shift_uv_px")
        if (
            not isinstance(payload_shift, list)
            or not isinstance(report_shift, list)
            or len(payload_shift) != 2
            or len(report_shift) != 2
            or any(not _finite_number(value) for value in (*payload_shift, *report_shift))
            or any(
                not math.isclose(float(left), float(right), abs_tol=0.001, rel_tol=0.0)
                for left, right in zip(payload_shift, report_shift, strict=True)
            )
        ):
            raise ValueError("Dynamic vessel payload/report principal_shift_uv_px disagree")
        _validate_thorax_assets(directory, report)
        return {"profile": profile, "frame_count": depth["frame_count"]}
    raise ValueError(f"Unknown validation profile: {profile}")


def validate_phase_alignment_report(
    report: dict[str, Any],
    frame_counts: dict[str, int] | None = None,
) -> None:
    """Validate the complete research phase asset before viewer packaging."""
    if (
        not isinstance(report, dict)
        or type(report.get("schema_version")) is not int
        or report["schema_version"] != 1
    ):
        raise ValueError("Phase report has an incompatible schema version")
    if report.get("method") != PHASE_METHOD:
        raise ValueError("Phase report has an incompatible method")
    if report.get("clinical_status") != "research-only":
        raise ValueError("Phase report must be research-only")
    if report.get("status") not in {"ready", "partial", "insufficient"}:
        raise ValueError("Invalid top-level phase status")

    expected_run_ids = {run_id for run_ids in PHASE_GROUPS.values() for run_id in run_ids}
    runs = report.get("runs")
    if not isinstance(runs, dict) or set(runs) != expected_run_ids:
        raise ValueError("Phase report must contain exactly Series 1-6 runs")
    if frame_counts is not None and set(frame_counts) != expected_run_ids:
        raise ValueError("Phase frame counts must contain exactly Series 1-6 runs")

    for run_id, run in runs.items():
        if not isinstance(run, dict):
            raise ValueError(f"Invalid phase run record for {run_id}")
        status = run.get("status")
        confidence = run.get("confidence")
        period = run.get("period_frames")
        anchor = run.get("anchor_frame")
        phases = run.get("phase_per_frame")
        visible_range = run.get("visible_frame_range")
        if status not in PHASE_RUN_STATUSES:
            raise ValueError(f"Invalid phase status for {run_id}")
        if not _finite_number(confidence) or not 0.0 <= float(confidence) <= 1.0:
            raise ValueError(f"Invalid phase confidence for {run_id}")
        expected_status = (
            "insufficient"
            if confidence < 0.50
            else "provisional"
            if confidence < 0.80
            else "automatic"
        )
        if status != expected_status:
            raise ValueError(f"Phase confidence and status are inconsistent for {run_id}")
        if run.get("clinical_status") != "research-only":
            raise ValueError(f"Invalid clinical status for {run_id}")
        inputs = run.get("inputs")
        if not isinstance(inputs, dict) or not PRIMARY_PHASE_INPUTS.issubset(inputs):
            raise ValueError(f"Missing primary phase provenance for {run_id}")
        for input_name in PRIMARY_PHASE_INPUTS:
            provenance = inputs[input_name]
            if (
                not isinstance(provenance, dict)
                or not isinstance(provenance.get("path"), str)
                or not provenance["path"].strip()
                or not isinstance(provenance.get("sha256"), str)
                or re.fullmatch(r"[0-9a-fA-F]{64}", provenance["sha256"]) is None
                or type(provenance.get("byte_length")) is not int
                or provenance["byte_length"] < 0
            ):
                raise ValueError(f"Invalid primary phase provenance for {run_id}:{input_name}")
        if "source_dicom" in inputs:
            source_dicom = inputs["source_dicom"]
            if (
                not isinstance(source_dicom, dict)
                or not isinstance(source_dicom.get("path"), str)
                or not source_dicom["path"].strip()
                or not isinstance(source_dicom.get("sha256"), str)
                or re.fullmatch(r"[0-9a-fA-F]{64}", source_dicom["sha256"]) is None
            ):
                raise ValueError(f"Invalid source_dicom provenance for {run_id}")
        if not isinstance(phases, list):
            raise ValueError(f"Invalid phase array for {run_id}")

        if period is None:
            if status != "insufficient" or anchor is not None or phases:
                raise ValueError(f"Run {run_id} without a period must be fully insufficient")
            if visible_range is not None:
                if (
                    not isinstance(visible_range, list)
                    or len(visible_range) != 2
                    or not all(isinstance(value, int) and not isinstance(value, bool) for value in visible_range)
                    or visible_range[0] < 0
                    or visible_range[1] < visible_range[0]
                ):
                    raise ValueError(f"Invalid visible frame range for {run_id}")
                if frame_counts is not None and visible_range[1] >= frame_counts[run_id]:
                    raise ValueError(f"Visible frame range exceeds bounds for {run_id}")
            continue

        if not _finite_number(period) or float(period) <= 0.0:
            raise ValueError(f"Invalid period for {run_id}")
        if not isinstance(anchor, int) or isinstance(anchor, bool):
            raise ValueError(f"Invalid anchor frame for {run_id}")
        if not phases or not all(_finite_number(value) and 0.0 <= float(value) < 1.0 for value in phases):
            raise ValueError(f"Invalid phase array for {run_id}")
        if frame_counts is not None and len(phases) != int(frame_counts[run_id]):
            raise ValueError(f"Phase array length does not match frame count for {run_id}")
        if not 0 <= anchor < len(phases):
            raise ValueError(f"Anchor frame exceeds bounds for {run_id}")
        if (
            not isinstance(visible_range, list)
            or len(visible_range) != 2
            or not all(isinstance(value, int) and not isinstance(value, bool) for value in visible_range)
            or visible_range[0] < 0
            or visible_range[0] > visible_range[1]
            or visible_range[1] >= len(phases)
        ):
            raise ValueError(f"Visible frame range exceeds bounds for {run_id}")
        if not visible_range[0] <= anchor <= visible_range[1]:
            raise ValueError(f"Run {run_id} anchor is outside the visible frame range")

    groups = report.get("groups")
    if not isinstance(groups, dict) or set(groups) != set(PHASE_GROUPS):
        raise ValueError("Phase report must contain exact RCA and LCA groups")
    expected_relations: set[tuple[str, str, str]] = set()
    usable_group_count = 0
    for group_name, expected_members in PHASE_GROUPS.items():
        group = groups[group_name]
        if not isinstance(group, dict) or group.get("run_ids") != list(expected_members):
            raise ValueError(f"Invalid exact group membership for {group_name}")
        usable = [
            run_id
            for run_id in expected_members
            if runs[run_id]["period_frames"] is not None
            and runs[run_id]["status"] in {"automatic", "provisional"}
        ]
        reference = group.get("reference_run_id")
        if len(usable) < 2:
            if group.get("status") != "insufficient" or reference is not None:
                raise ValueError(f"Insufficient group {group_name} must have a null reference")
            continue
        if group.get("status") == "insufficient":
            if (
                reference is not None
                or group.get("reason") != "no-credible-frame-correspondence"
            ):
                raise ValueError(f"Invalid correspondence rejection for group {group_name}")
            continue
        expected_group_status = (
            "automatic"
            if all(runs[run_id]["status"] == "automatic" for run_id in usable)
            else "provisional"
        )
        if group.get("status") != expected_group_status:
            raise ValueError(f"Invalid derived group status for {group_name}")
        usable_group_count += 1
        if reference not in usable:
            raise ValueError(f"Invalid group reference for {group_name}")
        best_confidence = max(float(runs[run_id]["confidence"]) for run_id in usable)
        if float(runs[reference]["confidence"]) != best_confidence:
            raise ValueError(f"Group reference is not highest confidence for {group_name}")
        expected_relations.update(
            (group_name, reference, target) for target in usable if target != reference
        )

    expected_top_status = ("insufficient", "partial", "ready")[usable_group_count]
    if report.get("status") != expected_top_status:
        raise ValueError("Invalid derived top-level phase status")

    correspondence = report.get("frame_correspondence")
    if not isinstance(correspondence, dict):
        raise ValueError("Invalid frame correspondence object")
    actual_relations: set[tuple[str, str, str]] = set()
    for item in correspondence.values():
        if not isinstance(item, dict):
            raise ValueError("Invalid frame correspondence entry")
        group_name = item.get("group")
        reference = item.get("reference_run_id")
        target = item.get("target_run_id")
        relation = (group_name, reference, target)
        if relation not in expected_relations:
            raise ValueError("Frame correspondence crosses or contradicts its group")
        if relation in actual_relations:
            raise ValueError("Duplicate frame correspondence relation")
        actual_relations.add(relation)
        pairs = item.get("pairs")
        if not isinstance(pairs, list) or not pairs:
            raise ValueError("Frame correspondence pairs must be non-empty")
        previous: tuple[int, int] | None = None
        reference_visible = runs[reference]["visible_frame_range"]
        target_visible = runs[target]["visible_frame_range"]
        for pair in pairs:
            if (
                not isinstance(pair, list)
                or len(pair) != 2
                or not all(isinstance(value, int) and not isinstance(value, bool) for value in pair)
            ):
                raise ValueError("Frame correspondence pair must contain two integers")
            current = (pair[0], pair[1])
            if not (
                reference_visible[0] <= current[0] <= reference_visible[1]
                and target_visible[0] <= current[1] <= target_visible[1]
                and current[0] < len(runs[reference]["phase_per_frame"])
                and current[1] < len(runs[target]["phase_per_frame"])
            ):
                raise ValueError("Frame correspondence exceeds frame bounds")
            if previous is not None and not (
                previous[0] < current[0] and previous[1] < current[1]
            ):
                raise ValueError("Frame correspondence must be strictly monotone")
            previous = current
    if actual_relations != expected_relations:
        raise ValueError("Frame correspondence does not cover all usable group runs")


def validate_scene_contract(scene: dict[str, Any]) -> None:
    projection_runs = scene.get("dynamic_xa_runs", [])
    vessel_runs = scene.get("dynamic_vessel_3d_runs", [])
    projection_ids = [item.get("run_id") for item in projection_runs]
    vessel_ids = [item.get("run_id") for item in vessel_runs]
    if len(projection_runs) != 7 or len(set(projection_ids)) != 7:
        raise ValueError("Scene must contain seven unique dynamic_xa_runs")
    expected_projection_ids = {run_id(index) for index in range(1, 8)}
    if set(projection_ids) != expected_projection_ids:
        raise ValueError("Scene projection run IDs must be exactly series_01 through series_07")
    if "series_07" in vessel_ids:
        raise ValueError("series_07 is a locator and must not be a vessel mesh run")
    if len(vessel_runs) != 6 or len(set(vessel_ids)) != 6:
        raise ValueError("Scene must contain six unique dynamic_vessel_3d_runs")
    expected_vessel_ids = {run_id(index) for index in range(1, 7)}
    if set(vessel_ids) != expected_vessel_ids:
        raise ValueError("Scene vessel run IDs must be exactly series_01 through series_06")
    missing = set(vessel_ids) - set(projection_ids)
    if missing:
        raise ValueError(f"Mesh runs missing projection parents: {sorted(missing)}")
    for item in projection_runs:
        expected_vessel = item.get("run_id") != "series_07"
        if bool(item.get("has_vessel")) != expected_vessel:
            raise ValueError(f"Invalid has_vessel flag for {item.get('run_id')}")
        if not expected_vessel:
            continue
        if "visible_frame_range" not in item or "centerline_counts" not in item:
            raise ValueError(f"Incomplete visible-window fields for {item.get('run_id')}")
        if item.get("schema_version") == 2:
            frame_count = _exact_nonnegative_int(
                item.get("frame_count"), f"{item.get('run_id')}.frame_count"
            )
        else:
            frame_count = int(item["frame_count"])
        counts = item["centerline_counts"]
        visible_range = item["visible_frame_range"]
        if len(counts) != frame_count:
            raise ValueError(f"Invalid centerline_counts length for {item.get('run_id')}")
        if len(visible_range) != 2:
            raise ValueError(f"Invalid visible_frame_range for {item.get('run_id')}")
        start, end = visible_range
        if start < 0 or end >= frame_count or start > end:
            raise ValueError(f"Invalid visible_frame_range for {item.get('run_id')}")

    projections_by_id = {item["run_id"]: item for item in projection_runs}
    for item in vessel_runs:
        if item.get("schema_version") != 2:
            continue
        identifier = item.get("run_id")
        frame_count = _exact_nonnegative_int(
            item.get("frame_count"), f"{identifier}.frame_count"
        )
        if frame_count != projections_by_id[identifier].get("frame_count"):
            raise ValueError(f"Vessel/projection frame_count mismatch for {identifier}")
        display_source = item.get("display_source")
        fallback_reason = item.get("fallback_reason")
        if display_source not in {"repaired", "baseline-fallback"}:
            raise ValueError(f"Invalid display_source for {identifier}")
        if display_source == "repaired" and fallback_reason is not None:
            raise ValueError(f"Repaired display cannot have fallback_reason for {identifier}")
        if display_source == "baseline-fallback" and (
            not isinstance(fallback_reason, str) or not fallback_reason
        ):
            raise ValueError(f"Baseline fallback needs a reason for {identifier}")
        repair = item.get("repair_report")
        topology = item.get("topology_report")
        depth = item.get("depth_audit")
        if not all(isinstance(value, dict) for value in (repair, topology, depth)):
            raise ValueError(f"Missing vessel quality summaries for {identifier}")
        if type(repair.get("schema_version")) is not int or repair["schema_version"] != 1:
            raise ValueError(f"Invalid repair_report schema for {identifier}")
        repair_counts = [
            _exact_nonnegative_int(repair.get(name), f"{identifier}.repair_report.{name}")
            for name in (
                "accepted_frame_count",
                "review_frame_count",
                "rejected_frame_count",
            )
        ]
        if sum(repair_counts) != frame_count:
            raise ValueError(f"Repair summary counts disagree for {identifier}")
        stable_fraction = topology.get("stable_edge_fraction")
        if not _finite_number(stable_fraction) or not 0.0 <= float(stable_fraction) <= 1.0:
            raise ValueError(f"Invalid stable_edge_fraction for {identifier}")
        accepted_depth = _exact_nonnegative_int(
            depth.get("accepted_frame_count"), f"{identifier}.depth_audit.accepted_frame_count"
        )
        rejected_depth = _exact_nonnegative_int(
            depth.get("rejected_frame_count"), f"{identifier}.depth_audit.rejected_frame_count"
        )
        if accepted_depth + rejected_depth not in {0, frame_count}:
            raise ValueError(f"Depth summary counts disagree for {identifier}")
        _reject_nonfinite_numbers(repair, f"{identifier}.repair_report")
        _reject_nonfinite_numbers(topology, f"{identifier}.topology_report")
        _reject_nonfinite_numbers(depth, f"{identifier}.depth_audit")

    phase = scene.get("cardiac_phase_alignment")
    if phase is None:
        return
    if not isinstance(phase, dict) or phase.get("clinical_status") != "research-only":
        raise ValueError("Invalid cardiac_phase_alignment scene summary")
    if phase.get("status") == "missing":
        if "asset" in phase:
            raise ValueError("Missing phase summary cannot expose an asset")
        return
    if phase.get("status") not in {"ready", "partial", "insufficient"}:
        raise ValueError("Invalid cardiac_phase_alignment status")
    if set(phase.get("groups", {})) != set(PHASE_GROUPS):
        raise ValueError("Invalid cardiac phase groups in scene")
    if set(phase.get("runs", {})) != {
        run_id for run_ids in PHASE_GROUPS.values() for run_id in run_ids
    }:
        raise ValueError("Scene cardiac phase summary must contain Series 1-6 only")
    if not isinstance(phase.get("frame_correspondence"), dict) or not isinstance(
        phase.get("asset"), dict
    ):
        raise ValueError("Incomplete cardiac phase scene summary")
