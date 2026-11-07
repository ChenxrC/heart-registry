from __future__ import annotations

import copy
import hashlib
import json
import inspect
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import SimpleITK as sitk
import trimesh
from scipy.spatial import cKDTree

import ctdr_pipeline.visualization.surface_mesh as surface_mesh


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "tools"))

import build_dynamic_vessel_3d as dynamic_module
import run_xa_temporal_fusion as temporal_module

from build_dynamic_vessel_3d import (
    Camera,
    _surface_geometry_audit,
    build_dynamic_3d,
    export_thorax_glb,
    export_thorax_mesh_variants,
    map_path_to_heart,
    project_lps,
    rays_for_uv,
    surface_from_mask,
    trace_skeleton,
)


def synthetic_camera() -> Camera:
    return Camera(
        source=np.asarray([0.0, 0.0, -1000.0]),
        detector_center=np.asarray([0.0, 0.0, 0.0]),
        u_axis=np.asarray([1.0, 0.0, 0.0]),
        v_axis=np.asarray([0.0, 1.0, 0.0]),
        n_axis=np.asarray([0.0, 0.0, 1.0]),
        principal_uv=np.asarray([255.5, 255.5]),
        row_spacing_mm=0.5,
        col_spacing_mm=0.5,
        rows=512,
        columns=512,
        sdd_mm=1000.0,
    )


def _point_on_ray(camera: Camera, yx: np.ndarray, depth_mm: float) -> np.ndarray:
    y, x = np.asarray(yx, dtype=np.float64)
    detector_point = (
        camera.detector_center
        + (x - camera.principal_uv[0]) * camera.col_spacing_mm * camera.u_axis
        + (y - camera.principal_uv[1]) * camera.row_spacing_mm * camera.v_axis
    )
    direction = detector_point - camera.source
    direction /= np.linalg.norm(direction)
    return camera.source + depth_mm * direction


def _line_yx(start: tuple[float, float], stop: tuple[float, float]) -> np.ndarray:
    return np.column_stack(
        (
            np.linspace(start[0], stop[0], 64),
            np.linspace(start[1], stop[1], 64),
        )
    )


def _task5_node(
    node_id: str,
    topology_id: str,
    kind: str,
    yx: tuple[float, float],
    parent_node_id: str | None,
) -> dict[str, object]:
    return {
        "node_id": node_id,
        "topology_id": topology_id,
        "parent_node_id": parent_node_id,
        "kind": kind,
        "yx": list(yx),
        "confidence": 1.0,
        "visibility": "visible",
    }


def _task5_edge(
    edge_id: str,
    topology_id: str,
    source_node_id: str,
    target_node_id: str,
    source_topology_id: str,
    target_topology_id: str,
    points_yx: np.ndarray,
    parent_edge_id: str | None = None,
) -> dict[str, object]:
    return {
        "edge_id": edge_id,
        "topology_id": topology_id,
        "parent_edge_id": parent_edge_id,
        "source_node_id": source_node_id,
        "target_node_id": target_node_id,
        "source_topology_id": source_topology_id,
        "target_topology_id": target_topology_id,
        "points_yx": points_yx.tolist(),
        "radius_px": np.full(64, 2.0).tolist(),
        "confidence": 1.0,
        "visibility": "visible",
    }


def _write_task6_build_inputs(
    tmp_path: Path,
) -> tuple[Path, Path, Path, Path, bytes, bytes, np.ndarray]:
    series_dir = tmp_path / "series"
    temporal_dir = series_dir / "temporal"
    masks_dir = tmp_path / "thorax_masks"
    output_dir = series_dir / "dynamic_3d"
    temporal_dir.mkdir(parents=True)
    masks_dir.mkdir()

    camera = replace(
        synthetic_camera(),
        principal_uv=np.asarray([31.5, 31.5]),
        row_spacing_mm=4.0,
        col_spacing_mm=4.0,
        rows=64,
        columns=64,
    )
    calibration = {
        "run_id": "synthetic-task6",
        "series_number": 1,
        "calibration": {
            "pixel_spacing_row_col_mm": [camera.row_spacing_mm, camera.col_spacing_mm],
            "source_lps_mm": camera.source.tolist(),
            "detector_center_lps_mm": camera.detector_center.tolist(),
            "u_lps": camera.u_axis.tolist(),
            "v_lps": camera.v_axis.tolist(),
            "n_lps": camera.n_axis.tolist(),
            "principal_uv_px": camera.principal_uv.tolist(),
            "sdd_mm": camera.sdd_mm,
        },
        "detector": {"rows": camera.rows, "columns": camera.columns},
    }
    calibration_path = series_dir / "series_01_calibration.json"
    calibration_path.write_text(json.dumps(calibration), encoding="utf-8")
    (series_dir / "series_manifest.json").write_text(
        json.dumps(
            {"source_record": {"series_number": 1}, "frame_png_count": 2}
        ),
        encoding="utf-8",
    )

    root_yx = (8.0, 8.0)
    junction_yx = (31.0, 31.0)
    right_yx = (8.0, 54.0)
    lower_yx = (55.0, 31.0)
    first_edges = (
        _task5_edge(
            "edge-root",
            "edge-v1-000000",
            "root",
            "junction",
            "node-v1-000000",
            "node-v1-000001",
            _line_yx(root_yx, junction_yx),
        ),
        _task5_edge(
            "edge-right",
            "edge-v1-000001",
            "junction",
            "right",
            "node-v1-000001",
            "node-v1-000002",
            _line_yx(junction_yx, right_yx),
            "edge-v1-000000",
        ),
        _task5_edge(
            "edge-lower",
            "edge-v1-000002",
            "junction",
            "lower",
            "node-v1-000001",
            "node-v1-000003",
            _line_yx(junction_yx, lower_yx),
            "edge-v1-000000",
        ),
    )
    first_frame = {
        "frame_index": 0,
        "root_node_id": "root",
        "root_topology_id": "node-v1-000000",
        "component_count": 1,
        "qc": {"quality_status": "accepted"},
        "nodes": [
            _task5_node("root", "node-v1-000000", "root", root_yx, None),
            _task5_node(
                "junction",
                "node-v1-000001",
                "junction",
                junction_yx,
                "node-v1-000000",
            ),
            _task5_node(
                "right",
                "node-v1-000002",
                "endpoint",
                right_yx,
                "node-v1-000001",
            ),
            _task5_node(
                "lower",
                "node-v1-000003",
                "endpoint",
                lower_yx,
                "node-v1-000001",
            ),
        ],
        "edges": list(first_edges),
    }

    rejected_points = _line_yx((50.0, 0.0), (50.0, 63.0))
    second_frame = {
        "frame_index": 1,
        "root_node_id": "root-1",
        "root_topology_id": "node-v1-new-root",
        "component_count": 1,
        "qc": {"quality_status": "accepted"},
        "nodes": [
            _task5_node(
                "root-1", "node-v1-new-root", "root", (50.0, 0.0), None
            ),
            _task5_node(
                "end-1",
                "node-v1-new-end",
                "endpoint",
                (50.0, 63.0),
                "node-v1-new-root",
            ),
        ],
        "edges": [
            _task5_edge(
                "edge-rejected",
                "edge-v1-new",
                "root-1",
                "end-1",
                "node-v1-new-root",
                "node-v1-new-end",
                rejected_points,
            )
        ],
    }
    graph_document = {
        "schema_version": 1,
        "frame_count": 2,
        "flow_convention": {
            "order": ["dx", "dy"],
            "direction": "previous_to_current",
            "mapping": "(y, x) -> (y + dy, x + dx)",
            "sampling": "deterministic_bilinear",
        },
        "frames": [first_frame, second_frame],
    }
    graph_bytes = json.dumps(graph_document, separators=(",", ":")).encode("utf-8")
    graph_path = temporal_dir / "temporal_vessel_graphs.json"
    graph_path.write_bytes(graph_bytes)

    mask_bytes = np.ones((2, 64, 64), dtype=np.uint8).tobytes(order="C")
    mask_path = temporal_dir / "temporal_masks_u8.raw"
    mask_path.write_bytes(mask_bytes)

    surface_points: list[np.ndarray] = []
    for edge in first_edges:
        points_yx = np.asarray(edge["points_yx"], dtype=np.float64)
        surface_points.extend(_point_on_ray(camera, yx, 650.0) for yx in points_yx)
    rejected_depths = np.asarray([650.0] * 32 + [710.0] * 32)
    surface_points.extend(
        _point_on_ray(camera, yx, float(depth))
        for yx, depth in zip(rejected_points, rejected_depths, strict=True)
    )
    surface = np.repeat(np.asarray(surface_points), 8, axis=0)
    return (
        series_dir,
        masks_dir,
        output_dir,
        calibration_path,
        graph_bytes,
        mask_bytes,
        surface,
    )


def _patch_task6_build_dependencies(
    monkeypatch: pytest.MonkeyPatch, surface: np.ndarray
) -> None:
    monkeypatch.setattr(
        dynamic_module,
        "surface_from_mask",
        lambda *_args, **_kwargs: (
            surface.copy(),
            np.empty((0, 3), dtype=np.int64),
            None,
        ),
    )
    monkeypatch.setattr(
        dynamic_module, "foreground_points_from_mask", lambda *_args, **_kwargs: surface.copy()
    )
    monkeypatch.setattr(
        dynamic_module,
        "estimate_principal_shift",
        lambda _silhouette, masks: (
            np.zeros(2, dtype=np.float64),
            0,
            {
                "reference_centerline_coverage_before": 1.0,
                "reference_centerline_coverage_after": 1.0,
                "reference_component_area_px": float(masks[0].sum()),
                "initial_bbox_shift_u_px": 0.0,
                "initial_bbox_shift_v_px": 0.0,
            },
            masks[0],
        ),
    )

    def fake_export(_masks_dir: Path, output_dir: Path) -> SimpleNamespace:
        (output_dir / "thorax_registration.glb").write_bytes(b"synthetic-glb")
        return SimpleNamespace(
            registration_asset=SimpleNamespace(path="thorax_registration.glb"),
            labels=(),
            to_dict=lambda: {"schema_version": 2, "assets": {}, "labels": []},
        )

    monkeypatch.setattr(dynamic_module, "export_thorax_mesh_variants", fake_export)
    monkeypatch.setattr(
        dynamic_module,
        "build_alignment_figure",
        lambda output_path, *_args, **_kwargs: output_path.write_bytes(b"synthetic-png"),
    )


def _task9_temporal_acceptance_metrics() -> dict[str, object]:
    return {
        "frame_count": 31,
        "multicomponent_frames": 6,
        "median_warped_iou": 0.70,
        "frames_below_iou_0_50": 2,
        "accepted_gap_max_length_px": 12.0,
        "max_component_growth_fraction": 0.20,
        "rejected_border_leaks": 0,
    }


def _prepare_task9_dynamic_candidate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[Path, Path, Path, Path, Path]:
    (
        series_dir,
        masks_dir,
        _legacy_output,
        calibration_path,
        _graph_bytes,
        mask_bytes,
        surface,
    ) = _write_task6_build_inputs(tmp_path)
    candidate_temporal = series_dir / "temporal_candidate"
    (series_dir / "temporal").rename(candidate_temporal)
    published_temporal = series_dir / "temporal"
    published_temporal.mkdir()
    (published_temporal / "published.sentinel").write_bytes(b"published-temporal")
    required_assets = (
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
    )
    for name in required_assets:
        path = candidate_temporal / name
        if path.exists():
            continue
        payload = mask_bytes if name.endswith("_u8.raw") else f"asset:{name}".encode()
        path.write_bytes(payload)
    (candidate_temporal / "repair_audit.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "method": "synthetic constrained repair",
                "frame_count": 31,
                "catheter_exclusion_available": False,
                "thresholds": {
                    "seed": 0.70,
                    "corridor": 0.20,
                    "temporal_addition": 0.35,
                    "max_growth_fraction": 0.20,
                    "automatic_gap_max_px": 12,
                    "review_gap_max_px": 24,
                    "maximum_direction_angle_deg": 45.0,
                },
                "frames": [],
                "links": [
                    {
                        "frame_index": index,
                        "start_node_id": f"n{index}",
                        "end_node_id": f"n{index + 1}",
                        "distance_px": 2.0,
                        "angle_deg": 0.0,
                        "mean_evidence": 0.9,
                        "temporal_support": 0.9,
                        "score": 1.0,
                        "confidence": 0.9,
                        "decision": decision,
                        "reasons": [],
                        "points_yx": [[1.0, 1.0], [1.0, 3.0]],
                    }
                    for index, decision in enumerate(
                        ("accepted", "review", "rejected")
                    )
                ],
                "clinical_status": (
                    "research-only; repaired outputs require diagnostic review"
                ),
            },
            allow_nan=False,
        ),
        encoding="utf-8",
    )
    for name in (
        "temporal_probabilities_u8.raw",
        "temporal_masks_u8.raw",
        "temporal_centerlines_u8.raw",
        "temporal_quality.csv",
        "temporal_report.json",
    ):
        (published_temporal / name).write_bytes(
            (candidate_temporal / name).read_bytes()
        )
    published_dynamic = series_dir / "dynamic_3d"
    published_dynamic.mkdir()
    (published_dynamic / "published.sentinel").write_bytes(b"published-dynamic")
    _patch_task6_build_dependencies(monkeypatch, surface)
    return (
        series_dir,
        masks_dir,
        calibration_path,
        candidate_temporal,
        series_dir / "dynamic_3d_candidate",
    )


def _write_task9_temporal_acceptance(
    series_dir: Path,
    candidate_temporal: Path,
    *,
    run_id: str = "synthetic-task6",
    candidate_directory: Path | None = None,
    corrupt_hash: bool = False,
) -> Path:
    assets = {
        path.name: {
            "byte_length": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        for path in sorted(candidate_temporal.iterdir())
        if path.is_file() and path.name != "repair_acceptance.json"
    }
    if corrupt_hash:
        assets["temporal_vessel_graphs.json"]["sha256"] = "0" * 64
    baseline_dir = series_dir / "temporal"
    baseline_assets = {
        name: {
            "byte_length": (baseline_dir / name).stat().st_size,
            "sha256": hashlib.sha256((baseline_dir / name).read_bytes()).hexdigest(),
        }
        for name in (
            "temporal_probabilities_u8.raw",
            "temporal_masks_u8.raw",
            "temporal_centerlines_u8.raw",
            "temporal_quality.csv",
            "temporal_report.json",
        )
    }
    report = temporal_module.build_repair_acceptance_report(
        run_id=run_id,
        baseline={
            "directory": str(baseline_dir.resolve()),
            "metrics": _task9_temporal_acceptance_metrics(),
            "assets": baseline_assets,
            "historical_hard_depth_cut_events": 250,
            "historical_hard_depth_cut_note": (
                "Prior audit count; surviving legacy paths contain no >14 mm "
                "within-path step because the old generator split at those events."
            ),
        },
        candidate={
            "directory": str(
                (candidate_directory or candidate_temporal).resolve()
            ),
            "metrics": _task9_temporal_acceptance_metrics(),
            "assets": assets,
            "comparisons": {
                "candidate_pre_repair_growth": {
                    "max_added_pixel_fraction": 0.20
                },
                "published_baseline_delta": {
                    "max_added_pixel_fraction": 0.0,
                    "max_removed_pixel_fraction": 0.0,
                    "max_changed_pixel_fraction": 0.0,
                },
            },
            "link_decision_counts": {"accepted": 1, "review": 1, "rejected": 1},
        },
    )
    path = candidate_temporal / "repair_acceptance.json"
    path.write_text(json.dumps(report, allow_nan=False), encoding="utf-8")
    return path


def _output_tree_hashes(directory: Path) -> dict[str, str]:
    return {
        path.relative_to(directory).as_posix(): hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
        for path in sorted(directory.rglob("*"))
        if path.is_file()
    }


def test_projection_and_ray_backprojection_are_consistent() -> None:
    camera = synthetic_camera()
    points = np.asarray([[10.0, -20.0, -500.0], [-15.0, 5.0, -250.0]])
    uv, depth = project_lps(points, camera)
    directions = rays_for_uv(uv, camera, camera.principal_uv)
    reconstructed = camera.source[None, :] + directions * depth[:, None]
    np.testing.assert_allclose(reconstructed, points, atol=1e-9)


def test_skeleton_tracing_splits_a_branching_tree() -> None:
    skeleton = np.zeros((32, 32), dtype=bool)
    skeleton[5:25, 15] = True
    for offset in range(8):
        skeleton[12 + offset, 15 + offset] = True
    paths = trace_skeleton(skeleton, min_length_px=3.0)
    assert len(paths) >= 3
    assert sum(len(path) for path in paths) >= 20


def test_legacy_ray_mapper_never_smooths_off_ray_or_splits_large_steps() -> None:
    camera = replace(
        synthetic_camera(),
        principal_uv=np.asarray([31.5, 31.5]),
        row_spacing_mm=4.0,
        col_spacing_mm=4.0,
        rows=64,
        columns=64,
    )
    path_yx = np.column_stack((np.full(9, 31), np.arange(20, 29))).astype(np.float64)
    sampled_yx = path_yx[[0, 3, 6, 8]]
    sampled_uv = sampled_yx[:, ::-1]
    depths = np.asarray([650.0, 710.0, 650.0, 710.0])
    projected_candidates = np.repeat(sampled_uv, 8, axis=0)
    candidate_depths = np.repeat(depths, 8)

    segments, _ = map_path_to_heart(
        path_yx,
        camera,
        camera.principal_uv,
        cKDTree(projected_candidates),
        candidate_depths,
        np.zeros((64, 64), dtype=np.float64),
    )

    assert len(segments) == 1
    projected, _ = project_lps(segments[0], camera)
    np.testing.assert_allclose(projected, sampled_uv, atol=1e-9)


def test_dynamic_mesh_builder_accepts_explicit_calibration_path() -> None:
    parameters = inspect.signature(build_dynamic_3d).parameters
    assert "calibration_path" in parameters


def test_task9_dynamic_builder_accepts_candidate_temporal_directory() -> None:
    parameters = inspect.signature(build_dynamic_3d).parameters
    assert "temporal_dir" in parameters


@pytest.mark.parametrize(
    ("temporal_name", "output_name"),
    (
        ("temporal_candidate", "dynamic_3d"),
        ("temporal", "dynamic_3d_candidate"),
        ("temporal_candidate", "dynamic_3d_candidate_other"),
    ),
)
def test_task9_dynamic_candidate_layout_fails_before_mkdir_or_write(
    tmp_path: Path,
    temporal_name: str,
    output_name: str,
) -> None:
    series_dir = tmp_path / "series_01"
    temporal_dir = series_dir / temporal_name
    temporal_dir.mkdir(parents=True)
    published_dir = series_dir / "dynamic_3d"
    published_dir.mkdir()
    sentinel = published_dir / "published.sentinel"
    sentinel.write_bytes(b"published-dynamic")
    published_hash = hashlib.sha256(sentinel.read_bytes()).hexdigest()
    output_dir = series_dir / output_name

    with pytest.raises(ValueError, match="isolated sibling candidate"):
        build_dynamic_3d(
            series_dir,
            tmp_path / "thorax_masks",
            output_dir,
            multiview_path=None,
            temporal_dir=temporal_dir,
        )

    assert hashlib.sha256(sentinel.read_bytes()).hexdigest() == published_hash
    if output_dir != published_dir:
        assert not output_dir.exists()


def test_task9_dynamic_windows_case_candidate_cannot_target_published_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (
        series_dir,
        masks_dir,
        calibration_path,
        candidate_temporal,
        _candidate_output,
    ) = _prepare_task9_dynamic_candidate(tmp_path, monkeypatch)
    windows_case_candidate = candidate_temporal.with_name("TEMPORAL_CANDIDATE")
    assert windows_case_candidate.is_dir()
    published_output = series_dir / "dynamic_3d"
    before = _output_tree_hashes(published_output)

    with pytest.raises(ValueError, match="isolated sibling candidate"):
        build_dynamic_3d(
            series_dir,
            masks_dir,
            published_output,
            multiview_path=None,
            calibration_path=calibration_path,
            temporal_dir=windows_case_candidate,
        )

    assert _output_tree_hashes(published_output) == before


def test_task9_dynamic_candidate_alias_cannot_target_published_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (
        series_dir,
        masks_dir,
        calibration_path,
        candidate_temporal,
        _candidate_output,
    ) = _prepare_task9_dynamic_candidate(tmp_path, monkeypatch)
    temporal_alias = series_dir / "temporal_alias"
    try:
        temporal_alias.symlink_to(candidate_temporal, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"directory symlinks unavailable: {exc}")
    published_output = series_dir / "dynamic_3d"
    before = _output_tree_hashes(published_output)

    with pytest.raises(ValueError, match="isolated sibling candidate"):
        build_dynamic_3d(
            series_dir,
            masks_dir,
            published_output,
            multiview_path=None,
            calibration_path=calibration_path,
            temporal_dir=temporal_alias,
        )

    assert _output_tree_hashes(published_output) == before


def test_dynamic_builder_never_serializes_nonfinite_json(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (
        series_dir,
        masks_dir,
        output_dir,
        calibration_path,
        _graph_bytes,
        _mask_bytes,
        surface,
    ) = _write_task6_build_inputs(tmp_path)
    _patch_task6_build_dependencies(monkeypatch, surface)
    monkeypatch.setattr(
        dynamic_module,
        "estimate_principal_shift",
        lambda _silhouette, masks: (
            np.zeros(2, dtype=np.float64),
            0,
            {
                "reference_centerline_coverage_before": float("nan"),
                "reference_centerline_coverage_after": 1.0,
                "reference_component_area_px": float(masks[0].sum()),
                "initial_bbox_shift_u_px": 0.0,
                "initial_bbox_shift_v_px": 0.0,
            },
            masks[0],
        ),
    )

    with pytest.raises(ValueError, match="Out of range float values"):
        build_dynamic_3d(
            series_dir,
            masks_dir,
            output_dir,
            multiview_path=None,
            calibration_path=calibration_path,
        )

    assert not (output_dir / "dynamic_vessel_3d.json").exists()


def test_task9_dynamic_candidate_validates_upstream_acceptance_and_preserves_published(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (
        series_dir,
        masks_dir,
        calibration_path,
        candidate_temporal,
        output_dir,
    ) = _prepare_task9_dynamic_candidate(tmp_path, monkeypatch)
    acceptance_path = _write_task9_temporal_acceptance(
        series_dir, candidate_temporal
    )
    acceptance_hash = hashlib.sha256(acceptance_path.read_bytes()).hexdigest()
    published_temporal = series_dir / "temporal" / "published.sentinel"
    published_dynamic = series_dir / "dynamic_3d" / "published.sentinel"
    published_hashes = {
        "temporal": hashlib.sha256(published_temporal.read_bytes()).hexdigest(),
        "dynamic": hashlib.sha256(published_dynamic.read_bytes()).hexdigest(),
    }

    build_dynamic_3d(
        series_dir,
        masks_dir,
        output_dir,
        multiview_path=None,
        calibration_path=calibration_path,
        temporal_dir=candidate_temporal,
    )

    connectivity = json.loads(
        (output_dir / "connectivity_acceptance.json").read_text(encoding="utf-8")
    )
    assert connectivity["temporal_acceptance"] == {
        "path": str(acceptance_path.resolve()),
        "sha256": acceptance_hash,
        "validated": True,
        "quantitative_gates_passed": True,
        "promotion_authorized": False,
    }
    assert hashlib.sha256(published_temporal.read_bytes()).hexdigest() == published_hashes[
        "temporal"
    ]
    assert hashlib.sha256(published_dynamic.read_bytes()).hexdigest() == published_hashes[
        "dynamic"
    ]


def test_task9_dynamic_candidate_late_acceptance_failure_preserves_entire_output_tree(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (
        series_dir,
        masks_dir,
        calibration_path,
        candidate_temporal,
        output_dir,
    ) = _prepare_task9_dynamic_candidate(tmp_path, monkeypatch)
    _write_task9_temporal_acceptance(series_dir, candidate_temporal)
    output_dir.mkdir()
    (output_dir / "published.sentinel").write_bytes(b"old-candidate")
    before = _output_tree_hashes(output_dir)

    def fail_late(*_args: object, **_kwargs: object) -> dict[str, object]:
        raise RuntimeError("simulated late acceptance failure")

    monkeypatch.setattr(
        dynamic_module, "build_connectivity_acceptance_report", fail_late
    )
    with pytest.raises(RuntimeError, match="late acceptance failure"):
        build_dynamic_3d(
            series_dir,
            masks_dir,
            output_dir,
            multiview_path=None,
            calibration_path=calibration_path,
            temporal_dir=candidate_temporal,
        )

    assert _output_tree_hashes(output_dir) == before
    assert sorted(path.name for path in output_dir.parent.iterdir()) == [
        "dynamic_3d",
        "dynamic_3d_candidate",
        "series_01_calibration.json",
        "series_manifest.json",
        "temporal",
        "temporal_candidate",
    ]


def test_task9_dynamic_published_late_render_failure_preserves_entire_output_tree(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (
        series_dir,
        masks_dir,
        output_dir,
        calibration_path,
        _graph_bytes,
        _mask_bytes,
        surface,
    ) = _write_task6_build_inputs(tmp_path)
    _patch_task6_build_dependencies(monkeypatch, surface)
    output_dir.mkdir()
    (output_dir / "published.sentinel").write_bytes(b"old-published")
    before = _output_tree_hashes(output_dir)

    def fail_render(output_path: Path, *_args: object, **_kwargs: object) -> None:
        output_path.write_bytes(b"partial-render")
        raise RuntimeError("simulated late render failure")

    monkeypatch.setattr(dynamic_module, "build_alignment_figure", fail_render)
    with pytest.raises(RuntimeError, match="late render failure"):
        build_dynamic_3d(
            series_dir,
            masks_dir,
            output_dir,
            multiview_path=None,
            calibration_path=calibration_path,
        )

    assert _output_tree_hashes(output_dir) == before
    assert not any(
        ".stage-" in path.name or ".backup-" in path.name
        for path in output_dir.parent.iterdir()
    )


def test_task9_atomic_directory_publication_restores_old_tree_on_swap_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output_dir = tmp_path / "dynamic_3d_candidate"
    output_dir.mkdir()
    (output_dir / "published.sentinel").write_bytes(b"old-tree")
    before = _output_tree_hashes(output_dir)
    real_replace = Path.replace

    def fail_staging_swap(source: Path, target: Path) -> Path:
        if ".stage-" in source.name and target == output_dir:
            raise OSError("simulated directory swap failure")
        return real_replace(source, target)

    def build_staging(staging_dir: Path) -> None:
        (staging_dir / "new.json").write_text("{}", encoding="utf-8")

    monkeypatch.setattr(Path, "replace", fail_staging_swap)
    with pytest.raises(OSError, match="directory swap failure"):
        dynamic_module.build_then_publish_directory(
            output_dir, build_staging, seed_from_existing=False
        )

    assert _output_tree_hashes(output_dir) == before
    assert not any(
        ".stage-" in path.name or ".backup-" in path.name
        for path in tmp_path.iterdir()
    )


@pytest.mark.parametrize(
    "acceptance_problem",
    (
        "missing",
        "run-id",
        "candidate-directory",
        "candidate-hash",
        "baseline-fields",
        "baseline-metric",
        "baseline-asset-hash",
        "clinical-status",
        "physician-frames",
        "physician-contact-sheet",
        "browser-frames",
        "browser-modes",
        "gate-value-type",
        "gate-threshold-type",
        "comparison",
        "link-count",
    ),
)
def test_task9_dynamic_candidate_rejects_unverified_upstream_acceptance_with_audit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    acceptance_problem: str,
) -> None:
    (
        series_dir,
        masks_dir,
        calibration_path,
        candidate_temporal,
        output_dir,
    ) = _prepare_task9_dynamic_candidate(tmp_path, monkeypatch)
    if acceptance_problem != "missing":
        acceptance_path = _write_task9_temporal_acceptance(
            series_dir,
            candidate_temporal,
            run_id=(
                "wrong-run" if acceptance_problem == "run-id" else "synthetic-task6"
            ),
            candidate_directory=(
                series_dir / "other_candidate"
                if acceptance_problem == "candidate-directory"
                else None
            ),
            corrupt_hash=acceptance_problem == "candidate-hash",
        )
        if acceptance_problem in {
            "baseline-fields",
            "baseline-metric",
            "baseline-asset-hash",
            "clinical-status",
            "physician-frames",
            "physician-contact-sheet",
            "browser-frames",
            "browser-modes",
            "gate-value-type",
            "gate-threshold-type",
            "comparison",
            "link-count",
        }:
            acceptance = json.loads(acceptance_path.read_text(encoding="utf-8"))
            if acceptance_problem == "baseline-fields":
                acceptance["baseline"].pop("historical_hard_depth_cut_note")
            elif acceptance_problem == "baseline-metric":
                acceptance["baseline"]["metrics"]["frame_count"] = True
            elif acceptance_problem == "baseline-asset-hash":
                acceptance["baseline"]["assets"]["temporal_masks_u8.raw"][
                    "sha256"
                ] = "0" * 64
            elif acceptance_problem == "clinical-status":
                acceptance["clinical_status"] = "research-only but forged"
            elif acceptance_problem == "physician-frames":
                acceptance["physician_review"][
                    "required_frame_indices_zero_based"
                ] = [0]
            elif acceptance_problem == "physician-contact-sheet":
                acceptance["physician_review"]["contact_sheet"] = "other.png"
            elif acceptance_problem == "browser-frames":
                acceptance["browser_review"][
                    "required_frame_indices_zero_based"
                ] = [3]
            elif acceptance_problem == "browser-modes":
                acceptance["browser_review"]["required_modes"] = ["diagnostic"]
            elif acceptance_problem == "gate-value-type":
                acceptance["gates"]["frame_count"]["value"] = 31.0
            elif acceptance_problem == "gate-threshold-type":
                acceptance["gates"]["frame_count"]["threshold"] = 31.0
            elif acceptance_problem == "comparison":
                acceptance["candidate"]["comparisons"][
                    "candidate_pre_repair_growth"
                ]["max_added_pixel_fraction"] = True
            else:
                acceptance["candidate"]["link_decision_counts"]["accepted"] = 2
            acceptance_path.write_text(
                json.dumps(acceptance, allow_nan=False), encoding="utf-8"
            )

    build_dynamic_3d(
        series_dir,
        masks_dir,
        output_dir,
        multiview_path=None,
        calibration_path=calibration_path,
        temporal_dir=candidate_temporal,
    )

    connectivity = json.loads(
        (output_dir / "connectivity_acceptance.json").read_text(encoding="utf-8")
    )
    assert connectivity["quantitative_gates_passed"] is False
    assert connectivity["quality_status"] == "rejected"
    assert any(
        "temporal acceptance" in error.lower()
        for error in connectivity["audit_errors"]
    )
    assert connectivity["temporal_acceptance"]["validated"] is False


def test_task9_connectivity_acceptance_requires_all_quantitative_and_review_gates() -> None:
    passing_metrics = {
        "accepted_frame_count": 1,
        "accepted_frame_graph_components_max": 1,
        "stable_edge_id_fraction": 1.0,
        "edge_steps_over_10_mm": 10,
        "watertight_surface_components": 1,
        "max_junction_gap_mm": 0.2,
        "reprojection_rms_px": 0.5,
    }
    report = dynamic_module.build_connectivity_acceptance_report(
        run_id="series_01", metrics=passing_metrics
    )

    assert report["schema_version"] == 1
    assert report["quantitative_gates_passed"] is True
    assert all(item["passed"] for item in report["gates"].values())
    assert report["physician_review"]["status"] == "pending"
    assert report["browser_review"]["status"] == "pending"
    assert report["promotion_authorized"] is False
    assert "monoplane" in report["depth_limitation"].lower()

    failed = dynamic_module.build_connectivity_acceptance_report(
        run_id="series_01",
        metrics={**passing_metrics, "max_junction_gap_mm": 0.21},
    )
    assert failed["quantitative_gates_passed"] is False
    assert failed["gates"]["max_junction_gap_mm"]["passed"] is False
    assert failed["promotion_authorized"] is False


@pytest.mark.parametrize(
    ("metric_name", "invalid_value"),
    (
        ("accepted_frame_count", True),
        ("accepted_frame_graph_components_max", "1"),
        ("stable_edge_id_fraction", float("nan")),
        ("edge_steps_over_10_mm", -1),
        ("watertight_surface_components", 1.0),
        ("max_junction_gap_mm", float("inf")),
        ("reprojection_rms_px", -0.01),
    ),
)
def test_task9_connectivity_report_rejects_malformed_metric_types_and_ranges(
    metric_name: str, invalid_value: object
) -> None:
    metrics = {
        "accepted_frame_count": 1,
        "accepted_frame_graph_components_max": 1,
        "stable_edge_id_fraction": 1.0,
        "edge_steps_over_10_mm": 10,
        "watertight_surface_components": 1,
        "max_junction_gap_mm": 0.2,
        "reprojection_rms_px": 0.5,
    }
    metrics[metric_name] = invalid_value

    with pytest.raises(ValueError, match=metric_name):
        dynamic_module.build_connectivity_acceptance_report(
            run_id="series_01", metrics=metrics
        )


def test_task9_connectivity_report_records_unavailable_reprojection_as_failed_gate() -> None:
    report = dynamic_module.build_connectivity_acceptance_report(
        run_id="series_01",
        metrics={
            "accepted_frame_count": 0,
            "accepted_frame_graph_components_max": 0,
            "stable_edge_id_fraction": 0.0,
            "edge_steps_over_10_mm": 0,
            "watertight_surface_components": 0,
            "max_junction_gap_mm": 0.0,
            "reprojection_rms_px": None,
        },
        audit_errors=["no accepted frame is available for surface audit"],
    )

    assert report["metrics"]["reprojection_rms_px"] is None
    assert report["gates"]["reprojection_rms_px"]["passed"] is False
    assert report["quantitative_gates_passed"] is False
    assert report["quality_status"] == "rejected"
    assert report["promotion_authorized"] is False


def _task9_valid_connectivity_asset(*, quality_status: str = "accepted") -> dict[str, object]:
    point_a = [0.0, 0.0, 0.0]
    point_b = [1.0, 0.0, 0.0]
    display_paths = (
        [
            {
                "edge_id": "edge-a-b",
                "topology_id": "edge-stable",
                "points_lps_mm": [point_a.copy(), point_b.copy()],
                "radius_mm": 1.0,
                "length_mm": 1.0,
            }
        ]
        if quality_status == "accepted"
        else []
    )
    return {
        "schema_version": 2,
        "frame_count": 1,
        "reference_frame": 0,
        "legacy_paths": {
            "authoritative": False,
            "lifecycle": "one-release compatibility data",
        },
        "frames": [
            {
                "frame_index": 0,
                "root_node_id": "a",
                "root_topology_id": "node-a",
                "quality_status": quality_status,
                "component_count": 1,
                "reasons": [] if quality_status == "accepted" else ["test-rejection"],
                "nodes": [
                    {
                        "node_id": "a",
                        "topology_id": "node-a",
                        "parent_node_id": None,
                        "kind": "root",
                        "yx": [1.0, 1.0],
                        "point_lps_mm": point_a,
                        "confidence": 1.0,
                        "visibility": "visible",
                    },
                    {
                        "node_id": "b",
                        "topology_id": "node-b",
                        "parent_node_id": "node-a",
                        "kind": "endpoint",
                        "yx": [1.0, 2.0],
                        "point_lps_mm": point_b,
                        "confidence": 1.0,
                        "visibility": "visible",
                    },
                ],
                "edges": [
                    {
                        "edge_id": "edge-a-b",
                        "topology_id": "edge-stable",
                        "parent_edge_id": None,
                        "source_node_id": "a",
                        "target_node_id": "b",
                        "source_topology_id": "node-a",
                        "target_topology_id": "node-b",
                        "points_yx": [[1.0, 1.0], [1.0, 2.0]],
                        "points_lps_mm": [point_a, point_b],
                        "radii_mm": [1.0, 1.0],
                        "confidence": 1.0,
                        "visibility": "visible",
                        "depth_audit": {
                            "edge_step_max_mm": 1.0,
                            "edge_step_p95_mm": 1.0,
                            "curvature_p95": 0.0,
                            "surface_distance_p95_mm": 0.0,
                        },
                    }
                ],
                "depth_audit": {
                    "objective_terms": {
                        "heart_surface_distance": 0.0,
                        "first_derivative_smoothness": 0.0,
                        "second_derivative_smoothness": 0.0,
                        "junction_equality_penalty": 0.0,
                        "temporal_depth_penalty": 0.0,
                        "weighted_total": 0.0,
                    },
                    "reprojection_rms_px": 0.0,
                },
                "display_source_frame_index": 0 if quality_status == "accepted" else None,
                "paths": display_paths,
                "path_count": len(display_paths),
                "point_count": sum(
                    len(path["points_lps_mm"]) for path in display_paths
                ),
                "heart_projection_valid_fraction": 1.0,
            }
        ],
    }


@pytest.mark.parametrize(
    "mutation",
    (
        "paths-not-list",
        "path-not-object",
        "path-extra-field",
        "path-edge-id-bool",
        "path-missing-edge-reference",
        "path-topology-reference",
        "path-points-shape",
        "path-points-nonfinite",
        "path-radius-bool",
        "path-length-mismatch",
        "path-count-mismatch",
        "point-count-mismatch",
    ),
)
def test_task9_connectivity_metrics_audit_accepted_display_paths_fail_closed(
    mutation: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    class WatertightMesh:
        is_watertight = True

        def split(self, *, only_watertight: bool) -> list[object]:
            assert only_watertight is False
            return [self]

    monkeypatch.setattr(
        dynamic_module,
        "build_watertight_vessel_surface",
        lambda _frame: WatertightMesh(),
    )
    asset = _task9_valid_connectivity_asset()
    frame = asset["frames"][0]
    path = frame["paths"][0]
    if mutation == "paths-not-list":
        frame["paths"] = {}
    elif mutation == "path-not-object":
        frame["paths"][0] = []
    elif mutation == "path-extra-field":
        path["type"] = "legacy"
    elif mutation == "path-edge-id-bool":
        path["edge_id"] = True
    elif mutation == "path-missing-edge-reference":
        path["edge_id"] = "missing"
    elif mutation == "path-topology-reference":
        path["topology_id"] = "wrong"
    elif mutation == "path-points-shape":
        path["points_lps_mm"] = [[0.0, 0.0], [1.0, 0.0]]
    elif mutation == "path-points-nonfinite":
        path["points_lps_mm"][0][0] = float("nan")
    elif mutation == "path-radius-bool":
        path["radius_mm"] = True
    elif mutation == "path-length-mismatch":
        path["length_mm"] = 2.0
    elif mutation == "path-count-mismatch":
        frame["path_count"] = 0
    elif mutation == "point-count-mismatch":
        frame["point_count"] = 1

    metrics, errors = dynamic_module._dynamic_connectivity_metrics(asset)
    report = dynamic_module.build_connectivity_acceptance_report(
        run_id="series_01", metrics=metrics, audit_errors=errors
    )

    assert errors, mutation
    assert report["quality_status"] == "rejected"
    json.dumps(report, allow_nan=False)


def test_task9_connectivity_metrics_audit_rejected_display_paths_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class WatertightMesh:
        is_watertight = True

        def split(self, *, only_watertight: bool) -> list[object]:
            assert only_watertight is False
            return [self]

    monkeypatch.setattr(
        dynamic_module,
        "build_watertight_vessel_surface",
        lambda _frame: WatertightMesh(),
    )
    asset = _task9_valid_connectivity_asset()
    accepted = asset["frames"][0]
    rejected = copy.deepcopy(accepted)
    rejected["frame_index"] = 1
    rejected["quality_status"] = "rejected"
    rejected["reasons"] = ["synthetic-rejection"]
    rejected["display_source_frame_index"] = 0
    rejected["paths"][0]["points_lps_mm"][0][0] = float("nan")
    asset["frames"].append(rejected)
    asset["frame_count"] = 2

    metrics, errors = dynamic_module._dynamic_connectivity_metrics(asset)
    report = dynamic_module.build_connectivity_acceptance_report(
        run_id="series_01", metrics=metrics, audit_errors=errors
    )

    assert errors
    assert report["quality_status"] == "rejected"
    json.dumps(report, allow_nan=False)


@pytest.mark.parametrize(
    "mutation",
    (
        "frame-index-bool",
        "frame-state",
        "frame-state-list",
        "frame-extra-field",
        "frame-component-bool",
        "frame-reprojection-inf",
        "frame-objective-nan",
        "node-duplicate-id",
        "node-point-shape",
        "node-point-inf",
        "node-point-bool",
        "node-yx-shape",
        "node-confidence-bool",
        "node-kind-list",
        "node-state",
        "node-state-list",
        "node-extra-field",
        "edge-duplicate-id",
        "edge-missing-reference",
        "edge-points-shape",
        "edge-points-ragged",
        "edge-points-nan",
        "edge-points-string",
        "edge-yx-shape",
        "edge-radius-inf",
        "edge-radius-bool",
        "edge-confidence-bool",
        "edge-state",
        "edge-state-list",
        "edge-extra-field",
        "edge-depth-audit-inf",
        "legacy-path-marker",
    ),
)
def test_task9_connectivity_metrics_audit_malformed_geometry_fail_closed(
    mutation: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    class WatertightMesh:
        is_watertight = True

        def split(self, *, only_watertight: bool) -> list[object]:
            assert only_watertight is False
            return [self]

    monkeypatch.setattr(
        dynamic_module,
        "build_watertight_vessel_surface",
        lambda _frame: WatertightMesh(),
    )
    asset = _task9_valid_connectivity_asset()
    frame = asset["frames"][0]
    nodes = frame["nodes"]
    edge = frame["edges"][0]
    if mutation == "frame-index-bool":
        frame["frame_index"] = False
    elif mutation == "frame-state":
        frame["quality_status"] = "review"
    elif mutation == "frame-state-list":
        frame["quality_status"] = []
    elif mutation == "frame-extra-field":
        frame["unexpected"] = 1
    elif mutation == "frame-component-bool":
        frame["component_count"] = True
    elif mutation == "frame-reprojection-inf":
        frame["depth_audit"]["reprojection_rms_px"] = float("inf")
    elif mutation == "frame-objective-nan":
        frame["depth_audit"]["objective_terms"]["weighted_total"] = float("nan")
    elif mutation == "node-duplicate-id":
        nodes[1]["node_id"] = "a"
    elif mutation == "node-point-shape":
        nodes[0]["point_lps_mm"] = [0.0, 0.0]
    elif mutation == "node-point-inf":
        nodes[0]["point_lps_mm"][0] = float("inf")
    elif mutation == "node-point-bool":
        nodes[0]["point_lps_mm"][0] = True
    elif mutation == "node-yx-shape":
        nodes[0]["yx"] = [1.0]
    elif mutation == "node-confidence-bool":
        nodes[0]["confidence"] = True
    elif mutation == "node-kind-list":
        nodes[0]["kind"] = []
    elif mutation == "node-state":
        nodes[0]["visibility"] = "missing"
    elif mutation == "node-state-list":
        nodes[0]["visibility"] = []
    elif mutation == "node-extra-field":
        nodes[0]["unexpected"] = 1
    elif mutation == "edge-duplicate-id":
        frame["edges"].append(copy.deepcopy(edge))
    elif mutation == "edge-missing-reference":
        edge["source_node_id"] = "missing"
    elif mutation == "edge-points-shape":
        edge["points_lps_mm"] = [[0.0, 0.0], [1.0, 0.0]]
    elif mutation == "edge-points-ragged":
        edge["points_lps_mm"] = [[0.0, 0.0, 0.0], [1.0]]
    elif mutation == "edge-points-nan":
        edge["points_lps_mm"][1][0] = float("nan")
    elif mutation == "edge-points-string":
        edge["points_lps_mm"][1][0] = "1.0"
    elif mutation == "edge-yx-shape":
        edge["points_yx"] = [[1.0, 1.0]]
    elif mutation == "edge-radius-inf":
        edge["radii_mm"][0] = float("inf")
    elif mutation == "edge-radius-bool":
        edge["radii_mm"][0] = True
    elif mutation == "edge-confidence-bool":
        edge["confidence"] = True
    elif mutation == "edge-state":
        edge["visibility"] = "missing"
    elif mutation == "edge-state-list":
        edge["visibility"] = []
    elif mutation == "edge-extra-field":
        edge["unexpected"] = 1
    elif mutation == "edge-depth-audit-inf":
        edge["depth_audit"]["edge_step_max_mm"] = float("inf")
    elif mutation == "legacy-path-marker":
        asset["legacy_paths"]["authoritative"] = True

    metrics, errors = dynamic_module._dynamic_connectivity_metrics(asset)
    report = dynamic_module.build_connectivity_acceptance_report(
        run_id="series_01", metrics=metrics, audit_errors=errors
    )

    assert errors, mutation
    assert report["quantitative_gates_passed"] is False
    assert report["quality_status"] == "rejected"
    json.dumps(report, allow_nan=False)


def test_task9_connectivity_metrics_count_stable_unique_edges_not_observations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class WatertightMesh:
        is_watertight = True

        def split(self, *, only_watertight: bool) -> list[object]:
            assert only_watertight is False
            return [self]

    monkeypatch.setattr(
        dynamic_module,
        "build_watertight_vessel_surface",
        lambda _frame: WatertightMesh(),
    )
    def frame(index: int, topology_ids: list[str]) -> dict[str, object]:
        record = _task9_valid_connectivity_asset()["frames"][0]
        record["frame_index"] = index
        record["display_source_frame_index"] = index
        record["edges"] = []
        record["paths"] = []
        record["component_count"] = 1 if topology_ids else 0
        if not topology_ids:
            record["nodes"] = []
            record["root_node_id"] = ""
            record["root_topology_id"] = None
        for edge_index, topology_id in enumerate(topology_ids):
            item = copy.deepcopy(
                _task9_valid_connectivity_asset()["frames"][0]["edges"][0]
            )
            item["edge_id"] = f"edge-{edge_index}"
            item["topology_id"] = topology_id
            record["edges"].append(item)
            record["paths"].append(
                {
                    "edge_id": item["edge_id"],
                    "topology_id": topology_id,
                    "points_lps_mm": copy.deepcopy(item["points_lps_mm"]),
                    "radius_mm": 1.0,
                    "length_mm": 1.0,
                }
            )
        record["path_count"] = len(record["paths"])
        record["point_count"] = sum(
            len(path["points_lps_mm"]) for path in record["paths"]
        )
        return record

    metrics, errors = dynamic_module._dynamic_connectivity_metrics(
        {
            "schema_version": 2,
            "frame_count": 3,
            "reference_frame": 0,
            "legacy_paths": {
                "authoritative": False,
                "lifecycle": "one-release compatibility data",
            },
            "frames": [frame(0, ["stable", "single"]), frame(1, ["stable"]), frame(2, [])],
        }
    )

    assert errors == []
    assert metrics["stable_edge_id_fraction"] == 0.5


def test_task9_connectivity_metrics_audit_rejected_geometry_instead_of_hiding_it() -> None:
    asset = _task9_valid_connectivity_asset(quality_status="rejected")
    frame = asset["frames"][0]
    frame["nodes"][1]["point_lps_mm"] = [11.0, 0.0, 0.0]
    frame["edges"][0]["points_lps_mm"] = [
        [0.0, 0.0, 0.0],
        [11.0, 0.0, 0.0],
    ]
    frame["edges"][0]["depth_audit"]["edge_step_max_mm"] = 11.0
    frame["edges"][0]["depth_audit"]["edge_step_p95_mm"] = 11.0

    metrics, errors = dynamic_module._dynamic_connectivity_metrics(asset)

    assert metrics["accepted_frame_count"] == 0
    assert metrics["stable_edge_id_fraction"] == 1.0
    assert metrics["edge_steps_over_10_mm"] == 1
    assert errors == ["no accepted frame is available for surface audit"]


def test_schema_v2_build_uses_task5_graph_and_falls_back_after_rejection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (
        series_dir,
        masks_dir,
        output_dir,
        calibration_path,
        graph_bytes,
        mask_bytes,
        surface,
    ) = _write_task6_build_inputs(tmp_path)
    _patch_task6_build_dependencies(monkeypatch, surface)

    report = build_dynamic_3d(
        series_dir,
        masks_dir,
        output_dir,
        multiview_path=None,
        calibration_path=calibration_path,
    )

    payload = json.loads(
        (output_dir / "dynamic_vessel_3d.json").read_text(encoding="utf-8")
    )
    assert payload["schema_version"] == 2
    assert payload["frame_count"] == 2
    assert payload["quality_gate_passed"] is False
    assert payload["quality_status"] == "diagnostic-only"
    assert "research-only" in payload["clinical_status"]
    assert payload["legacy_paths"] == {
        "authoritative": False,
        "lifecycle": "one-release compatibility data",
    }
    assert payload["depth_objective"] == {
        "heart_surface_distance": 1.0,
        "first_derivative_smoothness": 0.35,
        "second_derivative_smoothness": 0.2,
        "junction_equality_penalty": 0.5,
        "temporal_depth_penalty": 0.25,
        "monoplane_ambiguity": (
            "Depth is not uniquely observable; the CT heart surface is an initialization/prior."
        ),
    }

    accepted, rejected = payload["frames"]
    assert accepted["quality_status"] == "accepted"
    assert accepted["display_source_frame_index"] == 0
    assert len(accepted["edges"]) == len(accepted["paths"]) == 3
    assert accepted["path_count"] == 3
    node_points = {
        node["node_id"]: node["point_lps_mm"] for node in accepted["nodes"]
    }
    junction = node_points["junction"]
    touching = []
    for edge in accepted["edges"]:
        assert len(edge["points_lps_mm"]) == 64
        assert len(edge["radii_mm"]) == 64
        assert edge["points_lps_mm"][0] == node_points[edge["source_node_id"]]
        assert edge["points_lps_mm"][-1] == node_points[edge["target_node_id"]]
        if edge["source_node_id"] == "junction":
            touching.append(edge["points_lps_mm"][0])
        if edge["target_node_id"] == "junction":
            touching.append(edge["points_lps_mm"][-1])
        assert set(edge["depth_audit"]) == {
            "edge_step_max_mm",
            "edge_step_p95_mm",
            "curvature_p95",
            "surface_distance_p95_mm",
        }
        assert np.all(np.isfinite(list(edge["depth_audit"].values())))
    assert touching == [junction, junction, junction]
    assert set(accepted["depth_audit"]) == {
        "objective_terms",
        "reprojection_rms_px",
    }
    assert accepted["depth_audit"]["reprojection_rms_px"] < 0.25

    assert rejected["quality_status"] == "rejected"
    assert rejected["display_source_frame_index"] == 0
    assert len(rejected["edges"]) == 1
    assert rejected["edges"][0]["depth_audit"]["edge_step_max_mm"] > 10.0
    assert rejected["reasons"] == ["edge_step_exceeds_10mm:edge-v1-new"]
    assert rejected["paths"] == accepted["paths"]
    assert rejected["edges"][0]["points_lps_mm"] != accepted["edges"][0][
        "points_lps_mm"
    ]

    assert (series_dir / "temporal" / "temporal_vessel_graphs.json").read_bytes() == graph_bytes
    assert (series_dir / "temporal" / "temporal_masks_u8.raw").read_bytes() == mask_bytes
    assert report["schema_version"] == 2
    assert report["dynamic_vessel"]["frames"] == 2
    assert report["quality_gate_passed"] is False


def test_schema_v2_build_audits_node_only_and_empty_frames_with_legacy_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (
        series_dir,
        masks_dir,
        output_dir,
        calibration_path,
        graph_bytes,
        _mask_bytes,
        surface,
    ) = _write_task6_build_inputs(tmp_path)
    document = json.loads(graph_bytes)
    single_node = {
        "frame_index": 1,
        "root_node_id": "single-local",
        "root_topology_id": "node-v1-single",
        "component_count": 1,
        "qc": {"quality_status": "rejected"},
        "nodes": [
            _task5_node(
                "single-local",
                "node-v1-single",
                "root",
                (31.0, 31.0),
                None,
            )
        ],
        "edges": [],
    }
    empty = {
        "frame_index": 2,
        "root_node_id": "",
        "root_topology_id": None,
        "component_count": 0,
        "qc": {"quality_status": "rejected"},
        "nodes": [],
        "edges": [],
    }
    document["frame_count"] = 3
    document["frames"] = [document["frames"][0], single_node, empty]
    (series_dir / "temporal" / "temporal_vessel_graphs.json").write_text(
        json.dumps(document), encoding="utf-8"
    )
    (series_dir / "series_manifest.json").write_text(
        json.dumps({"source_record": {"series_number": 1}, "frame_png_count": 3}),
        encoding="utf-8",
    )
    (series_dir / "temporal" / "temporal_masks_u8.raw").write_bytes(
        np.ones((3, 64, 64), dtype=np.uint8).tobytes(order="C")
    )
    _patch_task6_build_dependencies(monkeypatch, surface)

    build_dynamic_3d(
        series_dir,
        masks_dir,
        output_dir,
        multiview_path=None,
        calibration_path=calibration_path,
    )

    payload = json.loads(
        (output_dir / "dynamic_vessel_3d.json").read_text(encoding="utf-8")
    )
    accepted, node_only, empty_frame = payload["frames"]
    assert node_only["quality_status"] == "rejected"
    assert node_only["reasons"] == ["no_observable_vessel_edges"]
    assert len(node_only["nodes"]) == 1
    assert node_only["edges"] == []
    node_point = np.asarray(node_only["nodes"][0]["point_lps_mm"])
    assert np.all(np.isfinite(node_point))
    projected_uv, _ = project_lps(node_point[None, :], replace(
        synthetic_camera(),
        principal_uv=np.asarray([31.5, 31.5]),
        row_spacing_mm=4.0,
        col_spacing_mm=4.0,
        rows=64,
        columns=64,
    ))
    np.testing.assert_allclose(projected_uv[0], [31.0, 31.0], atol=1e-9)
    assert node_only["paths"] == accepted["paths"]
    assert node_only["display_source_frame_index"] == 0
    assert np.all(np.isfinite(list(node_only["depth_audit"]["objective_terms"].values())))

    assert empty_frame["quality_status"] == "rejected"
    assert empty_frame["reasons"] == ["no_observable_vessel_graph"]
    assert empty_frame["nodes"] == []
    assert empty_frame["edges"] == []
    assert empty_frame["paths"] == accepted["paths"]
    assert empty_frame["display_source_frame_index"] == 0


def test_schema_v2_build_temporally_couples_only_stable_ids_after_topology_change(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (
        series_dir,
        masks_dir,
        output_dir,
        calibration_path,
        _graph_bytes,
        _mask_bytes,
        _surface,
    ) = _write_task6_build_inputs(tmp_path)
    camera = replace(
        synthetic_camera(),
        principal_uv=np.asarray([31.5, 31.5]),
        row_spacing_mm=4.0,
        col_spacing_mm=4.0,
        rows=64,
        columns=64,
    )
    previous_yx = _line_yx((10.0, 10.0), (10.0, 30.0))
    stable_yx = _line_yx((20.0, 10.0), (20.0, 30.0))
    new_yx = _line_yx((40.0, 10.0), (40.0, 30.0))
    first_frame = {
        "frame_index": 0,
        "root_node_id": "root-0",
        "root_topology_id": "node-stable-root",
        "component_count": 1,
        "qc": {"quality_status": "accepted"},
        "nodes": [
            _task5_node(
                "root-0", "node-stable-root", "root", (10.0, 10.0), None
            ),
            _task5_node(
                "end-0",
                "node-stable-end",
                "endpoint",
                (10.0, 30.0),
                "node-stable-root",
            ),
        ],
        "edges": [
            _task5_edge(
                "stable-0",
                "edge-stable",
                "root-0",
                "end-0",
                "node-stable-root",
                "node-stable-end",
                previous_yx,
            )
        ],
    }
    second_frame = {
        "frame_index": 1,
        "root_node_id": "root-1",
        "root_topology_id": "node-stable-root",
        "component_count": 2,
        "qc": {"quality_status": "accepted"},
        "nodes": [
            _task5_node(
                "root-1", "node-stable-root", "root", (20.0, 10.0), None
            ),
            _task5_node(
                "end-1",
                "node-stable-end",
                "endpoint",
                (20.0, 30.0),
                "node-stable-root",
            ),
            _task5_node(
                "new-root", "node-new-root", "root", (40.0, 10.0), None
            ),
            _task5_node(
                "new-end",
                "node-new-end",
                "endpoint",
                (40.0, 30.0),
                "node-new-root",
            ),
        ],
        "edges": [
            _task5_edge(
                "stable-1",
                "edge-stable",
                "root-1",
                "end-1",
                "node-stable-root",
                "node-stable-end",
                stable_yx,
            ),
            _task5_edge(
                "new-edge",
                "edge-new",
                "new-root",
                "new-end",
                "node-new-root",
                "node-new-end",
                new_yx,
            ),
        ],
    }
    graph_document = {
        "schema_version": 1,
        "frame_count": 2,
        "flow_convention": {
            "order": ["dx", "dy"],
            "direction": "previous_to_current",
            "mapping": "(y, x) -> (y + dy, x + dx)",
            "sampling": "deterministic_bilinear",
        },
        "frames": [first_frame, second_frame],
    }
    (series_dir / "temporal" / "temporal_vessel_graphs.json").write_text(
        json.dumps(graph_document), encoding="utf-8"
    )
    prior_points = [
        _point_on_ray(camera, yx, 650.0) for yx in previous_yx
    ]
    current_points = [
        _point_on_ray(camera, yx, 690.0)
        for points_yx in (stable_yx, new_yx)
        for yx in points_yx
    ]
    surface = np.repeat(np.asarray(prior_points + current_points), 8, axis=0)
    _patch_task6_build_dependencies(monkeypatch, surface)

    build_dynamic_3d(
        series_dir,
        masks_dir,
        output_dir,
        multiview_path=None,
        calibration_path=calibration_path,
    )

    payload = json.loads(
        (output_dir / "dynamic_vessel_3d.json").read_text(encoding="utf-8")
    )
    current = payload["frames"][1]
    edges = {edge["topology_id"]: edge for edge in current["edges"]}
    source = np.asarray(payload["source_lps_mm"])
    stable_depth = float(
        np.mean(
            np.linalg.norm(
                np.asarray(edges["edge-stable"]["points_lps_mm"]) - source,
                axis=1,
            )
        )
    )
    new_depth = float(
        np.mean(
            np.linalg.norm(
                np.asarray(edges["edge-new"]["points_lps_mm"]) - source,
                axis=1,
            )
        )
    )
    assert stable_depth < new_depth - 5.0
    assert new_depth == pytest.approx(690.0, abs=0.5)
    assert current["depth_audit"]["objective_terms"]["temporal_depth_penalty"] > 0.0


def test_task5_graph_schema_mismatches_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (
        series_dir,
        masks_dir,
        output_dir,
        calibration_path,
        graph_bytes,
        _mask_bytes,
        surface,
    ) = _write_task6_build_inputs(tmp_path)
    _patch_task6_build_dependencies(monkeypatch, surface)
    original = json.loads(graph_bytes)
    cases: list[tuple[str, dict[str, object], str]] = []

    wrong_flow = copy.deepcopy(original)
    wrong_flow["flow_convention"]["order"] = ["dy", "dx"]
    cases.append(("flow", wrong_flow, "flow_convention"))

    wrong_schema = copy.deepcopy(original)
    wrong_schema["schema_version"] = 2
    cases.append(("schema", wrong_schema, "schema_version"))

    wrong_count = copy.deepcopy(original)
    wrong_count["frame_count"] = 1
    cases.append(("count", wrong_count, "frame count"))

    missing_reference = copy.deepcopy(original)
    missing_reference["frames"][0]["edges"][0]["source_node_id"] = "missing"
    cases.append(("reference", missing_reference, "missing node"))

    wrong_samples = copy.deepcopy(original)
    wrong_samples["frames"][0]["edges"][0]["points_yx"] = wrong_samples[
        "frames"
    ][0]["edges"][0]["points_yx"][:-1]
    cases.append(("samples", wrong_samples, "64 finite samples"))

    nonfinite_radius = copy.deepcopy(original)
    nonfinite_radius["frames"][0]["edges"][0]["radius_px"][5] = np.nan
    cases.append(("finite", nonfinite_radius, "invalid constant"))

    local_node_parent = copy.deepcopy(original)
    local_node_parent["frames"][0]["nodes"][1]["parent_node_id"] = "root"
    cases.append(("local-node-parent", local_node_parent, "parent_node_id"))

    missing_node_parent = copy.deepcopy(original)
    missing_node_parent["frames"][0]["nodes"][1]["parent_node_id"] = "node-missing"
    cases.append(("missing-node-parent", missing_node_parent, "parent_node_id"))

    self_node_parent = copy.deepcopy(original)
    self_node_parent["frames"][0]["nodes"][1]["parent_node_id"] = "node-v1-000001"
    cases.append(("self-node-parent", self_node_parent, "parent_node_id"))

    local_edge_parent = copy.deepcopy(original)
    local_edge_parent["frames"][0]["edges"][1]["parent_edge_id"] = "edge-root"
    cases.append(("local-edge-parent", local_edge_parent, "parent_edge_id"))

    missing_edge_parent = copy.deepcopy(original)
    missing_edge_parent["frames"][0]["edges"][1]["parent_edge_id"] = "edge-missing"
    cases.append(("missing-edge-parent", missing_edge_parent, "parent_edge_id"))

    self_edge_parent = copy.deepcopy(original)
    self_edge_parent["frames"][0]["edges"][1]["parent_edge_id"] = "edge-v1-000001"
    cases.append(("self-edge-parent", self_edge_parent, "parent_edge_id"))

    graph_path = series_dir / "temporal" / "temporal_vessel_graphs.json"
    for case_name, document, message in cases:
        graph_path.write_text(json.dumps(document), encoding="utf-8")
        with pytest.raises(ValueError, match=message):
            build_dynamic_3d(
                series_dir,
                masks_dir,
                output_dir,
                multiview_path=None,
                calibration_path=calibration_path,
            )


@pytest.mark.parametrize(
    "endpoint_mutation", ("source_mismatch", "target_mismatch", "swapped")
)
def test_strict_temporal_graph_loader_rejects_edge_points_not_anchored_to_nodes(
    tmp_path: Path, endpoint_mutation: str
) -> None:
    (
        series_dir,
        _masks_dir,
        _output_dir,
        _calibration_path,
        graph_bytes,
        _mask_bytes,
        _surface,
    ) = _write_task6_build_inputs(tmp_path)
    document = json.loads(graph_bytes)
    edge = document["frames"][0]["edges"][0]
    points = np.asarray(edge["points_yx"], dtype=np.float64)
    if endpoint_mutation == "source_mismatch":
        points[0, 0] += 1e-9
    elif endpoint_mutation == "target_mismatch":
        points[-1, 0] += 1e-9
    else:
        points = points[::-1].copy()
    edge["points_yx"] = points.tolist()
    graph_path = series_dir / "temporal" / "temporal_vessel_graphs.json"
    graph_path.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(ValueError, match="points_yx.*endpoint"):
        dynamic_module._load_temporal_graphs(graph_path, 2, (64, 64))


def _write_spherical_mask(path: Path) -> None:
    z, y, x = np.indices((41, 41, 41))
    array = (((z - 20) ** 2 + (y - 20) ** 2 + (x - 20) ** 2) <= 12**2).astype(np.uint8)
    image = sitk.GetImageFromArray(array)
    image.SetSpacing((0.7, 0.8, 1.2))
    image.SetOrigin((10.0, 20.0, -30.0))
    image.SetDirection((-1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 1.0))
    sitk.WriteImage(image, str(path))


def _mesh_from_surface(surface: tuple[np.ndarray, np.ndarray, sitk.Image]) -> trimesh.Trimesh:
    vertices, faces, _ = surface
    return trimesh.Trimesh(vertices=vertices, faces=faces, process=False)


def test_heart_surface_profile_densifies_and_smooths_without_moving_lps_frame(tmp_path: Path) -> None:
    mask_path = tmp_path / "heart.nii.gz"
    _write_spherical_mask(mask_path)

    coarse = _mesh_from_surface(surface_from_mask(mask_path, step_size=2))
    full_resolution = _mesh_from_surface(surface_from_mask(mask_path, step_size=1))
    smoothed = _mesh_from_surface(surface_from_mask(mask_path, profile="heart"))

    assert len(smoothed.faces) > 3 * len(coarse.faces)
    assert np.percentile(smoothed.face_adjacency_angles, 95) < 0.4 * np.percentile(
        full_resolution.face_adjacency_angles, 95
    )
    assert np.linalg.norm(smoothed.centroid - full_resolution.centroid) < 0.25
    assert np.max(np.abs(smoothed.bounds - full_resolution.bounds)) < 2.0
    assert 0.85 < abs(smoothed.volume / full_resolution.volume) < 1.02


def test_thorax_export_reports_differential_smoothing_and_geometry_drift(tmp_path: Path) -> None:
    masks_dir = tmp_path / "masks"
    masks_dir.mkdir()
    _write_spherical_mask(masks_dir / "heart.nii.gz")
    _write_spherical_mask(masks_dir / "rib_left_1.nii.gz")

    output_path = tmp_path / "thorax.glb"
    records = export_thorax_glb(masks_dir, output_path)
    by_label = {record["label"]: record for record in records}

    assert output_path.is_file()
    assert set(by_label) == {"heart", "rib_left_1"}
    assert by_label["heart"]["mesh_processing"]["step_size"] == 1
    assert by_label["rib_left_1"]["mesh_processing"]["step_size"] == 1
    assert (
        by_label["heart"]["mesh_processing"]["gaussian_sigma_mm"]
        > by_label["rib_left_1"]["mesh_processing"]["gaussian_sigma_mm"]
    )
    assert (
        by_label["heart"]["mesh_processing"]["taubin_iterations"]
        > by_label["rib_left_1"]["mesh_processing"]["taubin_iterations"]
    )
    for record in records:
        audit = record["geometry_audit"]
        assert audit["centroid_shift_mm"] < 0.5
        assert audit["bbox_max_abs_shift_mm"] < 2.0
        assert 0.85 < audit["absolute_volume_ratio"] < 1.05
        assert audit["within_visualization_tolerance"]


def test_thorax_export_writes_registration_and_display_variants(tmp_path: Path) -> None:
    masks_dir = tmp_path / "masks"
    output_dir = tmp_path / "out"
    masks_dir.mkdir()
    _write_spherical_mask(masks_dir / "heart.nii.gz")
    _write_spherical_mask(masks_dir / "rib_left_1.nii.gz")

    report = export_thorax_mesh_variants(masks_dir, output_dir)
    registration_path = output_dir / "thorax_registration.glb"
    display_path = output_dir / "thorax_display.glb"
    report_path = output_dir / "thorax_surface_report.json"

    assert registration_path.is_file()
    assert display_path.is_file()
    assert report_path.is_file()
    assert report.schema_version == 2
    assert report.coordinate_system == "DICOM patient LPS"
    assert report.units == "mm"
    assert report.display_to_registration_row_major == np.eye(4).tolist()
    assert {item.label for item in report.labels} == {"heart", "rib_left_1"}

    payload = report.to_dict()
    assert json.loads(report_path.read_text(encoding="utf-8")) == payload
    for variant, asset_path in (
        ("registration", registration_path),
        ("display", display_path),
    ):
        asset = payload["assets"][variant]
        assert asset["path"] == asset_path.name
        assert asset["byte_length"] == asset_path.stat().st_size > 0
        assert len(asset["sha256"]) == 64
        int(asset["sha256"], 16)

    labels = {item["label"]: item for item in payload["labels"]}
    assert set(labels) == {"heart", "rib_left_1"}
    for item in labels.values():
        for variant in ("registration", "display"):
            assert item[variant]["vertex_count"] > 0
            assert item[variant]["face_count"] > 0
        assert isinstance(item["display"]["selected_profile"], str)
        assert {
            "centroid_shift_mm",
            "absolute_volume_ratio",
            "bidirectional_surface_distance_p95_mm",
            "adjacency_angle_p95_deg",
            "non_manifold_edge_delta",
            "within_tolerance",
            "failed_metrics",
        } <= set(item["display"]["geometry_audit"])
        assert isinstance(item["display"]["failed_candidates"], list)
        assert isinstance(item["display"]["fallback_applied"], bool)


def test_thorax_export_evaluates_display_candidates_lazily(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    masks_dir = tmp_path / "masks"
    masks_dir.mkdir()
    _write_spherical_mask(masks_dir / "heart.nii.gz")
    original_extract = surface_mesh.extract_surface
    display_calls: list[str] = []

    def tracked_extract(path: Path, profile: object):
        name = str(getattr(profile, "name", ""))
        if name.startswith("heart_sdf_"):
            display_calls.append(name)
        return original_extract(path, profile)

    def accept_first(registration, candidates, anatomy_kind):
        name, mesh = next(iter(candidates))
        audit = surface_mesh.audit_surface_pair(
            registration,
            registration.copy(),
            anatomy_kind,
            sample_count=64,
        )
        return surface_mesh.SurfaceSelection(mesh, name, audit, (), False)

    monkeypatch.setattr(surface_mesh, "extract_surface", tracked_extract)
    monkeypatch.setattr(surface_mesh, "select_display_surface", accept_first)
    surface_mesh.export_thorax_mesh_variants(masks_dir, tmp_path / "out")
    assert display_calls == ["heart_sdf_strong"]


def test_thorax_export_hashes_without_reading_whole_glb_back(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    masks_dir = tmp_path / "masks"
    masks_dir.mkdir()
    _write_spherical_mask(masks_dir / "rib_left_1.nii.gz")

    def reject_read_bytes(_path: Path) -> bytes:
        raise AssertionError("whole-file GLB reread")

    monkeypatch.setattr(Path, "read_bytes", reject_read_bytes)
    report = surface_mesh.export_thorax_mesh_variants(masks_dir, tmp_path / "out")
    assert report.registration_asset.byte_length > 0
    assert report.display_asset.byte_length > 0


def test_thorax_asset_publication_rolls_back_on_mid_commit_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    masks_dir = tmp_path / "masks"
    output_dir = tmp_path / "out"
    masks_dir.mkdir()
    output_dir.mkdir()
    _write_spherical_mask(masks_dir / "rib_left_1.nii.gz")
    old_payloads = {
        "thorax_registration.glb": b"old-registration",
        "thorax_display.glb": b"old-display",
        "thorax_surface_report.json": b"old-report",
    }
    for name, payload in old_payloads.items():
        (output_dir / name).write_bytes(payload)

    real_replace = Path.replace
    failed_once = False

    def fail_display_publish(source: Path, target: Path):
        nonlocal failed_once
        if (
            not failed_once
            and target.name == "thorax_display.glb"
            and ".stage-" in source.name
        ):
            failed_once = True
            raise OSError("simulated publication failure")
        return real_replace(source, target)

    monkeypatch.setattr(surface_mesh, "_replace_path", fail_display_publish)
    with pytest.raises(OSError, match="simulated publication failure"):
        surface_mesh.export_thorax_mesh_variants(masks_dir, output_dir)

    assert failed_once
    for name, payload in old_payloads.items():
        assert (output_dir / name).read_bytes() == payload


def test_dynamic_build_reuses_versioned_registration_export() -> None:
    source = inspect.getsource(dynamic_module._build_dynamic_3d_in_staging)
    assert "export_thorax_glb(" not in source
    assert "to_legacy_dict" in source


def test_geometry_audit_uses_surface_centroid_for_open_meshes() -> None:
    vertices = np.asarray(
        [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 2.0]]
    )
    faces = np.asarray([[0, 1, 3], [1, 2, 3], [2, 0, 3]])
    baseline = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    processed = trimesh.Trimesh(vertices=vertices + [0.1, 0.0, 0.0], faces=faces, process=False)

    audit = _surface_geometry_audit(baseline, processed)

    assert audit["centroid_definition"] == "surface-area centroid (open mesh)"
    assert audit["centroid_shift_mm"] == 0.1


def test_built_heart_surface_removes_visible_axial_stair_steps() -> None:
    dynamic_dir = (
        PROJECT_ROOT
        / "artifacts"
        / "xa_dynamic_vessel_1234567"
        / "all_series"
        / "series_01"
        / "dynamic_3d"
    )
    if not (dynamic_dir / "thorax.glb").is_file():
        pytest.skip("patient-specific generated meshes are intentionally not shipped")
    scene = trimesh.load(dynamic_dir / "thorax.glb", force="scene")
    heart = scene.geometry["heart"]
    p95_angle_degrees = float(np.degrees(np.percentile(heart.face_adjacency_angles, 95)))
    rib = scene.geometry["rib_right_6"]
    rib_p95_angle_degrees = float(np.degrees(np.percentile(rib.face_adjacency_angles, 95)))
    report = json.loads((dynamic_dir / "dynamic_vessel_3d_report.json").read_text(encoding="utf-8"))
    heart_record = next(item for item in report["thorax_glb"]["labels"] if item["label"] == "heart")

    assert p95_angle_degrees < 6.2
    assert rib_p95_angle_degrees < 20.0
    rib_record = next(item for item in report["thorax_glb"]["labels"] if item["label"] == "rib_right_6")
    assert rib_record["mesh_processing"]["windowed_sinc_iterations"] == 50
    assert all(
        item["geometry_audit"]["within_visualization_tolerance"]
        for item in report["thorax_glb"]["labels"]
    )
    assert heart_record["geometry_audit"]["centroid_shift_mm"] < 0.5
    assert 0.98 < heart_record["geometry_audit"]["absolute_volume_ratio"] < 1.02


def test_built_thorax_display_meshes_are_audited_or_exact_registration_fallbacks() -> None:
    dynamic_dir = (
        PROJECT_ROOT
        / "artifacts"
        / "xa_dynamic_vessel_1234567"
        / "all_series"
        / "series_01"
        / "dynamic_3d"
    )
    report_path = dynamic_dir / "thorax_surface_report.json"
    if not report_path.is_file():
        pytest.skip("patient-specific generated mesh audits are intentionally not shipped")
    report = json.loads(report_path.read_text(encoding="utf-8"))

    assert report["schema_version"] == 2
    assert report["coordinate_system"] == "DICOM patient LPS"
    assert report["units"] == "mm"
    assert report["display_to_registration_row_major"] == [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
    for asset in report["assets"].values():
        path = dynamic_dir / asset["path"]
        assert path.stat().st_size == asset["byte_length"]
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            while chunk := stream.read(1024 * 1024):
                digest.update(chunk)
        assert digest.hexdigest() == asset["sha256"]

    for item in report["labels"]:
        display = item["display"]
        audit = display["geometry_audit"]
        if display["fallback_applied"]:
            assert display["selected_profile"] == "registration_fallback"
            assert audit["centroid_shift_mm"] == 0.0
            assert audit["absolute_volume_ratio"] == 1.0
            assert audit["bidirectional_surface_distance_p95_mm"] < 1e-10
            assert audit["non_manifold_edge_delta"] == 0
        else:
            assert audit["within_tolerance"] is True
            assert audit["failed_metrics"] == []

    heart = next(item for item in report["labels"] if item["label"] == "heart")
    rib_right_6 = next(
        item for item in report["labels"] if item["label"] == "rib_right_6"
    )
    assert heart["display"]["fallback_applied"] is False
    assert heart["display"]["geometry_audit"]["adjacency_angle_p95_deg"] <= 4.0
    assert rib_right_6["display"]["fallback_applied"] is False
    assert rib_right_6["display"]["geometry_audit"]["adjacency_angle_p95_deg"] <= 12.0


def test_built_dynamic_vessel_3d_asset_contract() -> None:
    viewer = PROJECT_ROOT / "artifacts" / "xa_dynamic_vessel_1234567" / "viewer"
    if not (viewer / "scene_geometry.json").is_file():
        pytest.skip("patient-specific generated viewer assets are intentionally not shipped")
    scene = json.loads((viewer / "scene_geometry.json").read_text(encoding="utf-8"))
    depth = scene["dynamic_vessel_3d"]
    assert depth["frame_count"] == scene["dynamic_xa"]["frame_count"] == 31
    assert depth["quality_status"] == "diagnostic-only"
    assert depth["quality_gate_passed"] is False
    assert depth["multiview_attempt"]["used_for_depth"] is False
    assert depth["alignment_metrics"]["reference_centerline_coverage_after"] > depth["alignment_metrics"]["reference_centerline_coverage_before"]
    for asset in depth["assets"].values():
        path = viewer / asset["path"]
        assert path.is_file()
        assert path.stat().st_size == asset["byte_length"]

    payload = json.loads((viewer / depth["assets"]["vessel_paths"]["path"]).read_text(encoding="utf-8"))
    assert len(payload["frames"]) == 31
    assert payload["totals"]["paths"] > 0
    assert payload["totals"]["points"] > payload["totals"]["paths"]
