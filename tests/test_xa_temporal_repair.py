from __future__ import annotations

from dataclasses import replace
import hashlib
import inspect
import json
import sys
from pathlib import Path
from typing import get_type_hints

import cv2
import numpy as np
import pytest
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, SecondaryCaptureImageStorage, generate_uid


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TOOLS = PROJECT_ROOT / "tools"
sys.path.insert(0, str(TOOLS))

import run_xa_temporal_fusion as temporal_module
import build_dynamic_vessel_3d as dynamic_module
import xa_temporal_repair as repair_module
from run_xa_temporal_fusion import temporal_fuse
from xa_temporal_repair import (
    FlowEvidence,
    GapCandidate,
    bidirectional_flow_evidence,
    find_gap_candidates,
    repair_short_gaps,
    temporal_support,
)
from xa_cine import write_frames as write_valid_frames
from xa_cine import write_json as write_strict_json
from xa_cine import write_mp4 as write_valid_mp4
from xa_vessel_topology import (
    VesselEdge,
    VesselGraph,
    VesselNode,
    extract_vessel_graph,
    mask_topology_metrics,
)


def synthetic_dark_vessel(shift_x: int) -> np.ndarray:
    height = width = 96
    grid_y, grid_x = np.mgrid[:height, :width]
    texture = 190.0 + 18.0 * np.sin(grid_x / 5.0) + 14.0 * np.cos(grid_y / 7.0)
    base = np.clip(texture, 128, 255).astype(np.uint8)
    cv2.line(base, (47, 12), (47, 83), 35, 5, lineType=cv2.LINE_AA)
    transform = np.asarray([[1.0, 0.0, float(shift_x)], [0.0, 1.0, 0.0]], dtype=np.float32)
    return cv2.warpAffine(base, transform, (width, height), borderMode=cv2.BORDER_REFLECT101)


def binary_iou(left: np.ndarray, right: np.ndarray) -> float:
    union = np.count_nonzero(left | right)
    return 1.0 if union == 0 else float(np.count_nonzero(left & right) / union)


def write_test_cine(path: Path, frames: np.ndarray) -> None:
    meta = FileMetaDataset()
    meta.MediaStorageSOPClassUID = SecondaryCaptureImageStorage
    meta.MediaStorageSOPInstanceUID = generate_uid()
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    dataset = FileDataset(str(path), {}, file_meta=meta, preamble=b"\0" * 128)
    dataset.SOPClassUID = meta.MediaStorageSOPClassUID
    dataset.SOPInstanceUID = meta.MediaStorageSOPInstanceUID
    dataset.Modality = "XA"
    dataset.Rows = frames.shape[1]
    dataset.Columns = frames.shape[2]
    dataset.NumberOfFrames = frames.shape[0]
    dataset.SamplesPerPixel = 1
    dataset.PhotometricInterpretation = "MONOCHROME2"
    dataset.BitsAllocated = 8
    dataset.BitsStored = 8
    dataset.HighBit = 7
    dataset.PixelRepresentation = 0
    dataset.FrameTime = 125.0
    dataset.PixelData = np.ascontiguousarray(frames, dtype=np.uint8).tobytes()
    dataset.save_as(path, enforce_file_format=True)


def write_catheter_exclusions(
    segmentation_dir: Path, exclusions: np.ndarray
) -> Path:
    path = segmentation_dir / "catheter_exclusions_u8.raw"
    path.write_bytes(np.asarray(exclusions, dtype=np.uint8).tobytes(order="C"))
    return path


def test_task9_temporal_interfaces_accept_baseline_and_build_fail_closed_gate_report() -> None:
    assert "baseline_dir" in inspect.signature(temporal_module.run_temporal).parameters
    candidate_metrics = {
        "frame_count": 31,
        "multicomponent_frames": 6,
        "median_warped_iou": 0.70,
        "frames_below_iou_0_50": 2,
        "accepted_gap_max_length_px": 12.0,
        "max_component_growth_fraction": 0.20,
        "rejected_border_leaks": 0,
    }

    report = temporal_module.build_repair_acceptance_report(
        run_id="series_01",
        baseline={"metrics": {"multicomponent_frames": 13}, "assets": {}},
        candidate={"metrics": candidate_metrics, "assets": {}},
    )

    assert report["schema_version"] == 1
    assert report["quantitative_gates_passed"] is True
    assert all(item["passed"] for item in report["gates"].values())
    assert report["physician_review"]["status"] == "pending"
    assert report["browser_review"]["status"] == "pending"
    assert report["promotion_authorized"] is False
    assert "research-only" in report["clinical_status"]

    failed = temporal_module.build_repair_acceptance_report(
        run_id="series_01",
        baseline={"metrics": {}, "assets": {}},
        candidate={
            "metrics": {**candidate_metrics, "rejected_border_leaks": 1},
            "assets": {},
        },
    )
    assert failed["quantitative_gates_passed"] is False
    assert failed["gates"]["rejected_border_leaks"]["passed"] is False
    assert failed["promotion_authorized"] is False


@pytest.mark.parametrize(
    ("output_name", "provide_baseline"),
    (
        ("temporal", True),
        ("TEMPORAL", True),
        ("temporal_candidate_other", True),
        ("temporal_candidate", False),
        ("TEMPORAL_CANDIDATE", False),
    ),
)
def test_task9_temporal_candidate_layout_fails_before_any_output_change(
    tmp_path: Path,
    output_name: str,
    provide_baseline: bool,
) -> None:
    series_dir = tmp_path / "series_01"
    baseline_dir = series_dir / "temporal"
    baseline_dir.mkdir(parents=True)
    sentinel = baseline_dir / "published.sentinel"
    sentinel.write_bytes(b"published-baseline")
    baseline_state = output_tree_state(baseline_dir)
    output_dir = series_dir / output_name

    with pytest.raises(ValueError, match="isolated sibling candidate"):
        temporal_module.run_temporal(
            tmp_path / "missing.dcm",
            tmp_path / "missing-segmentation",
            output_dir,
            baseline_dir=baseline_dir if provide_baseline else None,
        )

    assert output_tree_state(baseline_dir) == baseline_state
    if output_dir != baseline_dir:
        assert not output_dir.exists()


def test_task9_temporal_candidate_alias_to_published_baseline_is_zero_write(
    tmp_path: Path,
) -> None:
    series_dir = tmp_path / "series_01"
    baseline_dir = series_dir / "temporal"
    baseline_dir.mkdir(parents=True)
    sentinel = baseline_dir / "published.sentinel"
    sentinel.write_bytes(b"published-baseline")
    baseline_state = output_tree_state(baseline_dir)
    output_alias = series_dir / "temporal_candidate"
    try:
        output_alias.symlink_to(baseline_dir, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"directory symlinks unavailable: {exc}")

    with pytest.raises(ValueError, match="isolated sibling candidate"):
        temporal_module.run_temporal(
            tmp_path / "missing.dcm",
            tmp_path / "missing-segmentation",
            output_alias,
            baseline_dir=baseline_dir,
        )

    assert output_tree_state(baseline_dir) == baseline_state


@pytest.mark.parametrize(
    ("metric_name", "invalid_value"),
    (
        ("frame_count", True),
        ("multicomponent_frames", "6"),
        ("median_warped_iou", float("nan")),
        ("frames_below_iou_0_50", -1),
        ("accepted_gap_max_length_px", float("inf")),
        ("max_component_growth_fraction", -0.01),
        ("rejected_border_leaks", 0.0),
    ),
)
def test_task9_acceptance_report_rejects_malformed_metric_types_and_ranges(
    metric_name: str, invalid_value: object
) -> None:
    metrics = {
        "frame_count": 31,
        "multicomponent_frames": 6,
        "median_warped_iou": 0.70,
        "frames_below_iou_0_50": 2,
        "accepted_gap_max_length_px": 12.0,
        "max_component_growth_fraction": 0.20,
        "rejected_border_leaks": 0,
    }
    metrics[metric_name] = invalid_value

    with pytest.raises(ValueError, match=metric_name):
        temporal_module.build_repair_acceptance_report(
            run_id="series_01",
            baseline={
                "directory": "temporal",
                "metrics": dict(metrics),
                "assets": {
                    name: {"byte_length": 1, "sha256": "0" * 64}
                    for name in (
                        "temporal_probabilities_u8.raw",
                        "temporal_masks_u8.raw",
                        "temporal_centerlines_u8.raw",
                        "temporal_quality.csv",
                        "temporal_report.json",
                    )
                },
                "historical_hard_depth_cut_events": 250,
                "historical_hard_depth_cut_note": "producer contract fixture",
            },
            candidate={"metrics": metrics, "assets": {}},
        )


def test_task9_contact_sheet_contains_five_same_frame_review_columns(
    tmp_path: Path,
) -> None:
    frames = np.stack(
        [np.full((16, 16), 80, dtype=np.uint8), np.full((16, 16), 120, dtype=np.uint8)]
    )
    baseline = np.zeros_like(frames, dtype=bool)
    repaired = baseline.copy()
    baseline[:, 4:12, 7] = True
    repaired[:] = baseline
    repaired[0, 8, 8:11] = True
    skeletons = repaired.copy()
    links = [
        {
            "frame_index": 0,
            "decision": "accepted",
            "confidence": 0.9,
            "distance_px": 3.0,
            "points_yx": [[8.0, 7.0], [8.0, 10.0]],
            "reasons": [],
        }
    ]
    output = tmp_path / "repair_before_after.png"

    temporal_module.write_repair_contact_sheet(
        output,
        frames,
        baseline,
        repaired,
        skeletons,
        links,
        review_frame_indices=[0, 1],
    )

    rendered = cv2.imread(str(output), cv2.IMREAD_COLOR)
    assert rendered is not None
    assert rendered.shape[1] == 5 * 160
    assert rendered.shape[0] == 2 * 180


def test_json_writer_rejects_nonfinite_values_before_creating_file(
    tmp_path: Path,
) -> None:
    output = tmp_path / "nonfinite.json"

    with pytest.raises(ValueError, match="Out of range float values"):
        write_strict_json(output, {"invalid": float("nan")})

    assert not output.exists()


def test_task9_candidate_rejects_invalid_published_baseline_before_writing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    frames = np.zeros((2, 8, 8), dtype=np.uint8)
    dicom_path = tmp_path / "cine.dcm"
    write_test_cine(dicom_path, frames)
    segmentation_dir = tmp_path / "segmentation"
    segmentation_dir.mkdir()
    (segmentation_dir / "probabilities_u8.raw").write_bytes(
        np.zeros(frames.shape, dtype=np.uint8).tobytes(order="C")
    )
    series_dir = tmp_path / "series_01"
    baseline_dir = series_dir / "temporal"
    baseline_dir.mkdir(parents=True)
    (baseline_dir / "published.sentinel").write_bytes(b"published")
    baseline_state = output_tree_state(baseline_dir)
    output_dir = series_dir / "temporal_candidate"
    install_test_output_writers(monkeypatch)

    with pytest.raises(ValueError, match="published baseline"):
        temporal_module.run_temporal(
            dicom_path,
            segmentation_dir,
            output_dir,
            baseline_dir=baseline_dir,
        )

    assert not output_dir.exists()
    assert output_tree_state(baseline_dir) == baseline_state


def test_task9_candidate_rejects_nonfinite_published_baseline_metrics_before_writing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    frames = np.zeros((2, 8, 8), dtype=np.uint8)
    dicom_path = tmp_path / "cine.dcm"
    write_test_cine(dicom_path, frames)
    segmentation_dir = tmp_path / "segmentation"
    segmentation_dir.mkdir()
    (segmentation_dir / "probabilities_u8.raw").write_bytes(
        np.zeros(frames.shape, dtype=np.uint8).tobytes(order="C")
    )
    install_test_output_writers(monkeypatch)
    series_dir = tmp_path / "series_01"
    baseline_dir = series_dir / "temporal"
    temporal_module.run_temporal(dicom_path, segmentation_dir, baseline_dir)
    quality_path = baseline_dir / "temporal_quality.csv"
    quality_text = quality_path.read_text(encoding="utf-8-sig")
    assert "1.0,1.0,1.0,provisional" in quality_text
    quality_path.write_text(
        quality_text.replace(
            "1.0,1.0,1.0,provisional", "nan,1.0,1.0,provisional", 1
        ),
        encoding="utf-8-sig",
    )
    baseline_state = output_tree_state(baseline_dir)
    output_dir = series_dir / "temporal_candidate"

    with pytest.raises(ValueError, match="published baseline.*warped_iou"):
        temporal_module.run_temporal(
            dicom_path,
            segmentation_dir,
            output_dir,
            baseline_dir=baseline_dir,
        )

    assert not output_dir.exists()
    assert output_tree_state(baseline_dir) == baseline_state


def test_task9_candidate_integration_uses_immutable_published_baseline_and_preserves_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    frame = np.tile(np.arange(16, dtype=np.uint8), (16, 1))
    frames = np.stack([frame, frame, frame])
    dicom_path = tmp_path / "cine.dcm"
    write_test_cine(dicom_path, frames)
    segmentation_dir = tmp_path / "segmentation"
    segmentation_dir.mkdir()
    (segmentation_dir / "probabilities_u8.raw").write_bytes(
        np.zeros(frames.shape, dtype=np.uint8).tobytes(order="C")
    )
    write_catheter_exclusions(segmentation_dir, np.zeros(frames.shape, dtype=bool))
    published_masks = np.zeros(frames.shape, dtype=bool)
    published_masks[:, 6, 3:10] = True
    candidate_local_masks = np.zeros(frames.shape, dtype=bool)
    candidate_local_masks[:, 9, 4:12] = True
    mask_source = {"value": published_masks}
    monkeypatch.setattr(
        temporal_module,
        "postprocess_probability",
        lambda _probability: mask_source["value"][0].copy(),
    )
    install_test_output_writers(monkeypatch)
    series_dir = tmp_path / "series_01"
    baseline_dir = series_dir / "temporal"
    temporal_module.run_temporal(dicom_path, segmentation_dir, baseline_dir)
    sentinel = baseline_dir / "published.sentinel"
    sentinel.write_bytes(b"do-not-touch")
    published_state = output_tree_state(baseline_dir)
    published_mask_hash = hashlib.sha256(
        (baseline_dir / "temporal_masks_u8.raw").read_bytes()
    ).hexdigest()

    captured: dict[str, np.ndarray] = {}
    real_writer = temporal_module.write_repair_contact_sheet

    def capture_contact_sheet(
        output_path: Path,
        sheet_frames: np.ndarray,
        baseline_masks: np.ndarray,
        repaired_masks: np.ndarray,
        repaired_skeletons: np.ndarray,
        links: list[dict[str, object]],
        *,
        review_frame_indices: list[int] | None = None,
    ) -> None:
        captured["baseline"] = np.asarray(baseline_masks, dtype=bool).copy()
        real_writer(
            output_path,
            sheet_frames,
            baseline_masks,
            repaired_masks,
            repaired_skeletons,
            links,
            review_frame_indices=review_frame_indices,
        )

    monkeypatch.setattr(
        temporal_module, "write_repair_contact_sheet", capture_contact_sheet
    )
    mask_source["value"] = candidate_local_masks
    output_dir = series_dir / "temporal_candidate"
    temporal_module.run_temporal(
        dicom_path,
        segmentation_dir,
        output_dir,
        baseline_dir=baseline_dir,
    )

    acceptance_path = output_dir / "repair_acceptance.json"
    acceptance = json.loads(acceptance_path.read_text(encoding="utf-8"))
    np.testing.assert_array_equal(captured["baseline"], published_masks)
    assert acceptance["baseline"]["assets"]["temporal_masks_u8.raw"][
        "sha256"
    ] == published_mask_hash
    comparisons = acceptance["candidate"]["comparisons"]
    assert set(comparisons) == {
        "candidate_pre_repair_growth",
        "published_baseline_delta",
    }
    assert comparisons["candidate_pre_repair_growth"] == {
        "max_added_pixel_fraction": acceptance["candidate"]["metrics"][
            "max_component_growth_fraction"
        ]
    }
    assert comparisons["published_baseline_delta"]["max_changed_pixel_fraction"] > 0
    assert acceptance_path.is_file()
    validation, validation_errors = dynamic_module._load_temporal_candidate_acceptance(
        output_dir, series_dir, "series_01"
    )
    assert validation["validated"] is True
    assert validation["quantitative_gates_passed"] is False
    assert validation_errors == [
        "temporal acceptance quantitative gates did not pass"
    ]
    assert output_tree_state(baseline_dir) == published_state
    assert sentinel.read_bytes() == b"do-not-touch"


def test_task9_temporal_candidate_late_failure_preserves_entire_existing_tree(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    frame = np.tile(np.arange(16, dtype=np.uint8), (16, 1))
    frames = np.stack([frame, frame])
    dicom_path = tmp_path / "cine.dcm"
    write_test_cine(dicom_path, frames)
    segmentation_dir = tmp_path / "segmentation"
    segmentation_dir.mkdir()
    (segmentation_dir / "probabilities_u8.raw").write_bytes(
        np.zeros(frames.shape, dtype=np.uint8).tobytes(order="C")
    )
    write_catheter_exclusions(segmentation_dir, np.zeros(frames.shape, dtype=bool))
    masks = np.zeros(frames.shape, dtype=bool)
    masks[:, 8, 3:12] = True
    monkeypatch.setattr(
        temporal_module,
        "postprocess_probability",
        lambda _probability: masks[0].copy(),
    )
    install_test_output_writers(monkeypatch)
    series_dir = tmp_path / "series_01"
    baseline_dir = series_dir / "temporal"
    temporal_module.run_temporal(dicom_path, segmentation_dir, baseline_dir)
    candidate_dir = series_dir / "temporal_candidate"
    temporal_module.run_temporal(
        dicom_path,
        segmentation_dir,
        candidate_dir,
        baseline_dir=baseline_dir,
    )
    (candidate_dir / "candidate.sentinel").write_bytes(b"old-candidate")
    before = output_tree_state(candidate_dir)

    def fail_contact_sheet(output_path: Path, *_args: object, **_kwargs: object) -> None:
        output_path.write_bytes(b"partial-contact-sheet")
        raise RuntimeError("simulated late contact-sheet failure")

    monkeypatch.setattr(
        temporal_module, "write_repair_contact_sheet", fail_contact_sheet
    )
    with pytest.raises(RuntimeError, match="late contact-sheet failure"):
        temporal_module.run_temporal(
            dicom_path,
            segmentation_dir,
            candidate_dir,
            baseline_dir=baseline_dir,
        )

    assert output_tree_state(candidate_dir) == before
    assert not any(
        ".stage-" in path.name or ".backup-" in path.name
        for path in series_dir.iterdir()
    )


def test_task9_persisted_temporal_graph_drops_unsupported_self_loop_edges() -> None:
    graph = VesselGraph(
        frame_index=0,
        nodes=(
            VesselNode("n0", "root", (1.0, 1.0), topology_id="tn0"),
            VesselNode("n1", "endpoint", (1.0, 3.0), topology_id="tn1"),
        ),
        edges=(
            VesselEdge(
                "e0",
                "n0",
                "n1",
                np.asarray([[1, 0], [1, 4]], dtype=np.float64),
                1.0,
                topology_id="te0",
                radius_px=np.ones(2),
            ),
            VesselEdge(
                "loop",
                "n0",
                "n0",
                np.asarray([[1, 1], [2, 2], [1, 1]], dtype=np.float64),
                1.0,
                topology_id="loop-id",
                radius_px=np.ones(3),
            ),
        ),
        root_node_id="n0",
        component_count=1,
    )

    document = temporal_module._temporal_graph_document([graph])

    assert [edge["edge_id"] for edge in document["frames"][0]["edges"]] == ["e0"]
    assert document["frames"][0]["edges"][0]["points_yx"] == [
        [1.0, 1.0],
        [1.0, 3.0],
    ]


def install_test_output_writers(
    monkeypatch: pytest.MonkeyPatch,
) -> list[str]:
    mp4_names: list[str] = []

    def write_test_frames(
        frames: np.ndarray, output_dir: Path, prefix: str
    ) -> list[Path]:
        return write_valid_frames(frames, output_dir, prefix)

    def write_test_mp4(
        path: Path,
        frames: np.ndarray,
        frame_times: list[float],
        *,
        color: bool = False,
    ) -> None:
        write_valid_mp4(path, frames, frame_times, color=color)
        mp4_names.append(path.name)

    monkeypatch.setattr(temporal_module, "write_frames", write_test_frames)
    monkeypatch.setattr(temporal_module, "write_mp4", write_test_mp4)
    return mp4_names


def output_file_hashes(output_dir: Path) -> dict[str, str]:
    return {
        path.relative_to(output_dir).as_posix(): hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
        for path in sorted(output_dir.rglob("*"))
        if path.is_file()
    }


def output_tree_state(output_dir: Path) -> dict[str, object]:
    return {
        "files": output_file_hashes(output_dir),
        "directories": sorted(
            path.relative_to(output_dir).as_posix()
            for path in output_dir.rglob("*")
            if path.is_dir()
        ),
    }


def install_no_output_call_spies(
    monkeypatch: pytest.MonkeyPatch,
) -> list[str]:
    calls: list[str] = []
    real_path_open = Path.open

    def forbidden(name: str):
        def fail(*_args: object, **_kwargs: object) -> None:
            calls.append(name)
            raise AssertionError(f"output writer called during rejected rerun: {name}")

        return fail

    monkeypatch.setattr(temporal_module, "write_frames", forbidden("write_frames"))
    monkeypatch.setattr(temporal_module, "write_mp4", forbidden("write_mp4"))
    monkeypatch.setattr(
        temporal_module, "write_raw_stack", forbidden("write_raw_stack")
    )
    monkeypatch.setattr(temporal_module, "write_json", forbidden("write_json"))

    def forbid_output_open(
        path: Path, mode: str = "r", *args: object, **kwargs: object
    ):
        if any(marker in mode for marker in ("w", "a", "x", "+")):
            calls.append(f"Path.open:{path.name}")
            raise AssertionError(
                f"output file opened during rejected rerun: {path.name}"
            )
        return real_path_open(path, mode, *args, **kwargs)

    monkeypatch.setattr(Path, "open", forbid_output_open)
    return calls


def evidence_with_fixed_flows(
    monkeypatch: pytest.MonkeyPatch,
    current: np.ndarray,
    neighbor: np.ndarray,
    forward: np.ndarray,
    backward: np.ndarray,
) -> FlowEvidence:
    flows = iter((forward, backward))
    monkeypatch.setattr(repair_module, "dense_flow", lambda *_args: next(flows))
    probability = np.ones(current.shape, dtype=np.float32)
    return bidirectional_flow_evidence(
        current, neighbor, probability, probability > 0.5
    )


def broken_collinear_gap(
    distance_px: int = 12,
) -> tuple[np.ndarray, VesselGraph, np.ndarray, np.ndarray]:
    mask = np.zeros((80, 80), dtype=bool)
    mask[40, 8:25] = True
    mask[40, 24 + distance_px : 41 + distance_px] = True
    graph = extract_vessel_graph(mask, root_hint_yx=(40.0, 8.0))
    evidence = np.ones(mask.shape, dtype=np.float64)
    support = np.ones(mask.shape, dtype=np.float64)
    return mask, graph, evidence, support


def accepted_candidate_for_graph(
    graph: VesselGraph,
    start_yx: tuple[int, int],
    end_yx: tuple[int, int],
) -> GapCandidate:
    nodes = graph.nodes
    start_node = min(
        nodes,
        key=lambda node: (node.yx[0] - start_yx[0]) ** 2
        + (node.yx[1] - start_yx[1]) ** 2,
    )
    end_node = min(
        nodes,
        key=lambda node: (node.yx[0] - end_yx[0]) ** 2
        + (node.yx[1] - end_yx[1]) ** 2,
    )
    start_coordinate = np.asarray(start_node.yx, dtype=np.float64)
    end_coordinate = np.asarray(end_node.yx, dtype=np.float64)

    def outward_tangent(node_id: str) -> np.ndarray:
        incident = [
            edge
            for edge in graph.edges
            if node_id in (edge.source_node_id, edge.target_node_id)
        ]
        assert len(incident) == 1
        edge = incident[0]
        points = np.asarray(edge.points_yx, dtype=np.float64)
        if edge.source_node_id == node_id:
            tangent = points[0] - points[1]
        else:
            tangent = points[-1] - points[-2]
        return tangent

    def angle_deg(tangent: np.ndarray, direction: np.ndarray) -> float:
        cosine = float(
            np.dot(tangent, direction)
            / (np.linalg.norm(tangent) * np.linalg.norm(direction))
        )
        return float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))

    start_pixel = np.rint(start_coordinate).astype(np.int64)
    end_pixel = np.rint(end_coordinate).astype(np.int64)
    steps = int(np.max(np.abs(end_pixel - start_pixel)))
    path = np.rint(
        np.column_stack(
            (
                np.linspace(start_pixel[0], end_pixel[0], steps + 1),
                np.linspace(start_pixel[1], end_pixel[1], steps + 1),
            )
        )
    ).astype(np.int64)
    angle = max(
        angle_deg(outward_tangent(start_node.node_id), end_coordinate - start_coordinate),
        angle_deg(outward_tangent(end_node.node_id), start_coordinate - end_coordinate),
    )
    return GapCandidate(
        start_node.node_id,
        end_node.node_id,
        float(np.linalg.norm(end_coordinate - start_coordinate)),
        angle,
        path,
        1.0,
        1.0,
        0.0,
        "accepted",
        (),
    )


def test_seed_reconstruction_recovers_supported_gap_without_leaking() -> None:
    evidence = np.zeros((64, 64), np.float32)
    evidence[31:34, 8:55] = 0.65
    evidence[31:34, 29:33] = 0.35
    evidence[10:25, 40:55] = 0.25  # nearby distractor
    seed = evidence >= 0.70
    seed[31:34, 8:29] = True
    seed[31:34, 33:55] = True

    repaired, audit = repair_module.constrained_seed_reconstruction(
        evidence, seed, np.zeros_like(seed), max_growth_fraction=0.20
    )

    assert repaired[32, 30]
    assert not repaired[15, 45]
    assert audit.accepted_pixels == 12
    assert audit.rejected_pixels == 0
    assert audit.area_growth_fraction == pytest.approx(12 / 129)
    assert audit.status == "accepted"
    assert audit.reasons == ()


def test_seed_reconstruction_rejects_growth_above_twenty_percent() -> None:
    evidence = np.zeros((64, 64), np.float32)
    evidence[30:35, 8:55] = 0.65
    evidence[30:35, 29:33] = 0.35
    evidence[10:25, 40:55] = 0.25
    seed = evidence >= 0.70
    seed[31:34, 8:29] = True
    seed[31:34, 33:55] = True

    repaired, audit = repair_module.constrained_seed_reconstruction(
        evidence, seed, np.zeros_like(seed), max_growth_fraction=0.20
    )

    np.testing.assert_array_equal(repaired, seed)
    assert audit.accepted_pixels == 0
    assert audit.rejected_pixels == 106
    assert audit.area_growth_fraction == pytest.approx(106 / 129)
    assert audit.status == "rejected"
    assert "growth_cap_exceeded" in audit.reasons


def test_seed_reconstruction_rejects_new_detector_border_contact() -> None:
    evidence = np.zeros((16, 16), np.float32)
    evidence[7:10, 0:10] = 0.65
    seed = np.zeros_like(evidence, dtype=bool)
    seed[7:10, 1:10] = True

    repaired, audit = repair_module.constrained_seed_reconstruction(
        evidence, seed, np.zeros_like(seed), max_growth_fraction=0.20
    )

    np.testing.assert_array_equal(repaired, seed)
    assert audit.accepted_pixels == 0
    assert audit.rejected_pixels == 3
    assert audit.area_growth_fraction == pytest.approx(3 / 27)
    assert audit.status == "rejected"
    assert "new_border_contact" in audit.reasons


def test_seed_reconstruction_removes_exclusion_pixels_from_candidate() -> None:
    evidence = np.zeros((16, 16), np.float32)
    evidence[7:10, 2:14] = 0.65
    seed = evidence >= 0.20
    seed[7:10, 8] = False
    exclusion = np.zeros_like(seed)
    exclusion[7:10, 8] = True

    repaired, audit = repair_module.constrained_seed_reconstruction(
        evidence, seed, exclusion, max_growth_fraction=0.20
    )

    np.testing.assert_array_equal(repaired, seed)
    assert not np.any(repaired & exclusion)
    assert audit.accepted_pixels == 0
    assert audit.rejected_pixels == 0
    assert audit.status == "accepted"


def test_seed_reconstruction_rejects_seed_intersecting_exclusion() -> None:
    evidence = np.zeros((16, 16), np.float32)
    evidence[7:10, 2:14] = 0.65
    seed = np.zeros_like(evidence, dtype=bool)
    seed[7:10, 2:14] = True
    exclusion = np.zeros_like(seed)
    exclusion[8, 8] = True

    repaired, audit = repair_module.constrained_seed_reconstruction(
        evidence, seed, exclusion, max_growth_fraction=0.20
    )

    np.testing.assert_array_equal(repaired, seed)
    assert audit.accepted_pixels == 0
    assert audit.status == "rejected"
    assert "exclusion_intersection" in audit.reasons


def test_seed_reconstruction_includes_exact_corridor_threshold_only() -> None:
    evidence = np.zeros((16, 16), np.float32)
    evidence[8, 2:8] = 0.65
    evidence[8, 8] = 0.20
    evidence[8, 9] = 0.199
    seed = np.zeros_like(evidence, dtype=bool)
    seed[8, 2:8] = True

    repaired, audit = repair_module.constrained_seed_reconstruction(
        evidence, seed, np.zeros_like(seed), max_growth_fraction=0.20
    )

    assert repaired[8, 8]
    assert not repaired[8, 9]
    assert audit.accepted_pixels == 1
    assert audit.status == "accepted"


def test_seed_reconstruction_accepts_growth_equal_to_twenty_percent() -> None:
    evidence = np.zeros((16, 16), np.float32)
    evidence[8, 2:8] = 0.65
    seed = np.zeros_like(evidence, dtype=bool)
    seed[8, 2:7] = True

    repaired, audit = repair_module.constrained_seed_reconstruction(
        evidence, seed, np.zeros_like(seed), max_growth_fraction=0.20
    )

    assert repaired[8, 7]
    assert audit.accepted_pixels == 1
    assert audit.area_growth_fraction == pytest.approx(0.20)
    assert audit.status == "accepted"


def test_build_vessel_evidence_uses_internal_image_channels_and_exact_weights() -> None:
    frame = np.full((8, 8), 127, dtype=np.uint8)
    model_probability = np.full(frame.shape, 0.8, dtype=np.float32)
    temporal_probability = np.full(frame.shape, 0.6, dtype=np.float32)
    support = np.full(frame.shape, 0.5, dtype=np.float32)
    inputs_before = tuple(
        item.copy()
        for item in (frame, model_probability, temporal_probability, support)
    )

    actual = repair_module.build_vessel_evidence(
        frame, model_probability, temporal_probability, support
    )

    np.testing.assert_allclose(actual, 0.42, atol=1e-6)
    assert actual.dtype == np.float32
    assert actual.shape == frame.shape
    for item, before in zip(
        (frame, model_probability, temporal_probability, support), inputs_before
    ):
        np.testing.assert_array_equal(item, before)


@pytest.mark.parametrize(
    ("channel", "expected"),
    [
        ("model", 0.45),
        ("frangi", 0.20),
        ("dark", 0.15),
        ("temporal", 0.20),
    ],
)
def test_build_vessel_evidence_protects_each_channel_weight_independently(
    monkeypatch: pytest.MonkeyPatch,
    channel: str,
    expected: float,
) -> None:
    shape = (4, 4)
    zeros = np.zeros(shape, dtype=np.float32)
    ones = np.ones(shape, dtype=np.float32)
    monkeypatch.setattr(
        repair_module,
        "_frangi_probability",
        lambda _frame: ones.copy() if channel == "frangi" else zeros.copy(),
    )
    monkeypatch.setattr(
        repair_module,
        "_normalized_dark_vessel_response",
        lambda _frame: ones.copy() if channel == "dark" else zeros.copy(),
    )

    actual = repair_module.build_vessel_evidence(
        np.zeros(shape, dtype=np.uint8),
        ones if channel == "model" else zeros,
        ones if channel == "temporal" else zeros,
        ones if channel == "temporal" else zeros,
    )

    np.testing.assert_allclose(actual, expected, atol=1e-6)


def test_public_api_is_limited_to_authorized_temporal_interfaces() -> None:
    assert repair_module.__all__ == [
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
    assert list(inspect.signature(repair_module.build_vessel_evidence).parameters) == [
        "frame",
        "model_probability",
        "temporal_probability",
        "temporal_support",
    ]
    assert not hasattr(repair_module, "TemporalFusionEvidence")
    assert not hasattr(repair_module, "temporal_fusion_evidence")

    repair_signature = inspect.signature(repair_module.repair_short_gaps)
    assert list(repair_signature.parameters) == [
        "mask",
        "graph",
        "candidates",
        "auto_gap_px",
        "exclusion_mask",
    ]
    assert repair_signature.parameters["auto_gap_px"].default == 12
    assert (
        repair_signature.parameters["exclusion_mask"].kind
        is inspect.Parameter.KEYWORD_ONLY
    )
    assert repair_signature.parameters["exclusion_mask"].default is None


def test_repair_audit_and_reconstruction_have_public_type_annotations() -> None:
    assert get_type_hints(repair_module.RepairAudit) == {
        "accepted_pixels": int,
        "rejected_pixels": int,
        "area_growth_fraction": float,
        "status": str,
        "reasons": tuple[str, ...],
    }


def test_build_vessel_evidence_has_exact_public_type_hints() -> None:
    assert get_type_hints(repair_module.build_vessel_evidence) == {
        "frame": np.ndarray,
        "model_probability": np.ndarray,
        "temporal_probability": np.ndarray,
        "temporal_support": np.ndarray,
        "return": np.ndarray,
    }
    assert get_type_hints(repair_module.constrained_seed_reconstruction) == {
        "evidence": np.ndarray,
        "seed_mask": np.ndarray,
        "exclusion_mask": np.ndarray,
        "max_growth_fraction": float,
        "return": tuple[np.ndarray, repair_module.RepairAudit],
    }


def test_motion_compensation_aligns_one_pixel_translation() -> None:
    current = synthetic_dark_vessel(shift_x=0)
    neighbor = synthetic_dark_vessel(shift_x=1)
    probability = (neighbor < 100).astype(np.float32)
    evidence = bidirectional_flow_evidence(
        current, neighbor, probability, probability > 0.5
    )
    assert binary_iou(evidence.warped_mask, current < 100) > 0.90


def test_forward_backward_inconsistency_has_low_confidence_when_appearance_matches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    current = synthetic_dark_vessel(shift_x=0)
    forward = np.zeros((*current.shape, 2), dtype=np.float32)
    backward = forward.copy()
    backward[..., 0] = 2.0

    evidence = evidence_with_fixed_flows(
        monkeypatch, current, current.copy(), forward, backward
    )

    assert np.count_nonzero(evidence.confidence) == 0


def test_backward_field_is_sampled_at_forward_coordinates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    current = np.zeros((2, 5), dtype=np.uint8)
    forward = np.zeros((*current.shape, 2), dtype=np.float32)
    forward[..., 0] = 1.0
    backward = np.zeros_like(forward)
    backward[..., 0] = np.asarray([-9.0, -1.0, -0.5, 0.5, 2.0], dtype=np.float32)

    evidence = evidence_with_fixed_flows(
        monkeypatch, current, current.copy(), forward, backward
    )

    np.testing.assert_allclose(
        evidence.forward_backward_error_px[:, :4],
        np.asarray([[0.0, 0.5, 1.5, 3.0], [0.0, 0.5, 1.5, 3.0]]),
        atol=1e-6,
    )


def test_flow_confidence_cutoff_keeps_threshold_and_rejects_above(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    current = np.zeros((2, 3), dtype=np.uint8)
    forward = np.zeros((*current.shape, 2), dtype=np.float32)
    backward = np.zeros_like(forward)
    backward[..., 0] = np.asarray([1.49, 1.50, 1.51], dtype=np.float32)

    evidence = evidence_with_fixed_flows(
        monkeypatch, current, current.copy(), forward, backward
    )

    assert evidence.confidence[0, 0] > 0.0
    assert evidence.confidence[0, 1] == pytest.approx(np.exp(-0.5), abs=1e-6)
    assert evidence.confidence[0, 2] == 0.0


def test_outside_detector_flow_is_invalid_and_has_no_sampled_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    current = np.zeros((2, 4), dtype=np.uint8)
    forward = np.zeros((*current.shape, 2), dtype=np.float32)
    forward[..., 0] = 1.0
    backward = np.zeros_like(forward)
    backward[..., 0] = -1.0

    evidence = evidence_with_fixed_flows(
        monkeypatch, current, current.copy(), forward, backward
    )

    np.testing.assert_array_equal(evidence.confidence[:, :3], 1.0)
    np.testing.assert_array_equal(evidence.confidence[:, 3], 0.0)
    np.testing.assert_array_equal(evidence.forward_backward_error_px[:, :3], 0.0)
    assert np.all(np.isnan(evidence.forward_backward_error_px[:, 3]))


def test_appearance_gate_is_strict_at_normalized_point_two(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    current = np.zeros((2, 3), dtype=np.uint8)
    neighbor = np.tile(np.asarray([50, 51, 52], dtype=np.uint8), (2, 1))
    forward = np.zeros((*current.shape, 2), dtype=np.float32)

    evidence = evidence_with_fixed_flows(
        monkeypatch, current, neighbor, forward, forward.copy()
    )

    np.testing.assert_array_equal(
        evidence.confidence,
        np.asarray([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float32),
    )


def test_translated_neighbor_contributes_only_after_motion_compensation() -> None:
    current = synthetic_dark_vessel(shift_x=0)
    neighbor = synthetic_dark_vessel(shift_x=7)
    neighbor_probability = (neighbor < 100).astype(np.float32)
    probabilities = np.stack(
        [np.zeros_like(neighbor_probability), neighbor_probability]
    )
    current_vessel = current < 100

    assert np.count_nonzero(neighbor_probability[current_vessel]) == 0

    fused = temporal_fuse(np.stack([current, neighbor]), probabilities)

    assert float(np.mean(fused[0][current_vessel])) > 0.20


def test_anisotropic_temporal_majority_uses_two_votes_and_point_three_five_evidence() -> None:
    frame = synthetic_dark_vessel(shift_x=0)
    frames = np.stack([frame, frame, frame])
    masks = np.zeros(frames.shape, dtype=bool)
    masks[0, 40, 40:42] = True
    masks[2, 40, 40:42] = True
    combined_evidence = np.zeros(frames.shape, dtype=np.float32)
    combined_evidence[1, 40, 40] = 0.35
    combined_evidence[1, 40, 41] = 0.349
    inputs_before = (masks.copy(), combined_evidence.copy())

    repaired = repair_module._anisotropic_temporal_majority(
        frames,
        masks,
        combined_evidence,
        np.zeros_like(masks),
    )

    assert repaired[1, 40, 40]
    assert not repaired[1, 40, 41]
    np.testing.assert_array_equal(masks, inputs_before[0])
    np.testing.assert_array_equal(combined_evidence, inputs_before[1])


def test_anisotropic_temporal_majority_warps_before_spatial_closing() -> None:
    frames = np.stack(
        [
            synthetic_dark_vessel(shift_x=0),
            synthetic_dark_vessel(shift_x=1),
            synthetic_dark_vessel(shift_x=2),
        ]
    )
    masks = np.zeros(frames.shape, dtype=bool)
    masks[0, 40, 47] = True
    masks[2, 40, 49] = True
    combined_evidence = np.zeros(frames.shape, dtype=np.float32)
    combined_evidence[1, 40, 48] = 0.35

    repaired = repair_module._anisotropic_temporal_majority(
        frames,
        masks,
        combined_evidence,
        np.zeros_like(masks),
    )

    assert repaired[1, 40, 48]


def test_temporal_corridor_warps_before_spatial_dilation() -> None:
    frames = np.stack(
        [
            synthetic_dark_vessel(shift_x=0),
            synthetic_dark_vessel(shift_x=2),
            synthetic_dark_vessel(shift_x=4),
        ]
    )
    masks = np.zeros(frames.shape, dtype=bool)
    masks[0, 40, 47] = True
    masks[2, 40, 51] = True

    corridor = repair_module._motion_compensated_temporal_corridor(
        1, frames, masks
    )

    assert corridor[40, 49]
    assert corridor[39, 49]
    assert not corridor[40, 47]
    assert not corridor[40, 51]


def test_repair_sequence_includes_exact_seed_threshold(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frames = np.zeros((1, 16, 16), dtype=np.uint8)
    probabilities = np.zeros(frames.shape, dtype=np.float32)
    baseline_masks = np.zeros(frames.shape, dtype=bool)
    baseline_masks[0, 8, 4:9] = True
    controlled_evidence = np.zeros(frames.shape[1:], dtype=np.float32)
    controlled_evidence[8, 4:9] = 0.70
    controlled_evidence[8, 9] = 0.699
    monkeypatch.setattr(
        temporal_module,
        "build_vessel_evidence",
        lambda *_args: controlled_evidence.copy(),
    )
    monkeypatch.setattr(
        temporal_module,
        "_motion_compensated_temporal_corridor",
        lambda *_args: np.ones(frames.shape[1:], dtype=bool),
    )

    _, repaired_masks, audits = temporal_module._repair_temporal_sequence(
        frames,
        probabilities,
        probabilities,
        probabilities,
        baseline_masks,
        np.zeros_like(baseline_masks),
    )

    np.testing.assert_array_equal(
        repaired_masks[0, 8, 4:10], np.ones(6, dtype=bool)
    )
    assert audits[0].accepted_pixels == 1
    assert audits[0].area_growth_fraction == pytest.approx(0.20)
    assert audits[0].status == "accepted"


@pytest.mark.parametrize(
    ("proposal_pixel", "extra_pixels", "catheter_pixel", "expected_reason"),
    [
        ((4, 4), (), None, "temporal_corridor_violation"),
        ((0, 4), (), None, "exclusion_intersection"),
        ((4, 4), (), (4, 4), "exclusion_intersection"),
        ((4, 4), ((4, 5),), None, "growth_cap_exceeded"),
    ],
)
def test_morphology_proposal_cannot_bypass_final_hard_gates(
    monkeypatch: pytest.MonkeyPatch,
    proposal_pixel: tuple[int, int],
    extra_pixels: tuple[tuple[int, int], ...],
    catheter_pixel: tuple[int, int] | None,
    expected_reason: str,
) -> None:
    frames = np.zeros((1, 12, 12), dtype=np.uint8)
    probabilities = np.zeros(frames.shape, dtype=np.float32)
    baseline_masks = np.zeros(frames.shape, dtype=bool)
    baseline_masks[0, 6, 3:8] = True
    catheter_exclusions = np.zeros_like(baseline_masks)
    if catheter_pixel is not None:
        catheter_exclusions[(0, *catheter_pixel)] = True
    corridor = np.zeros(frames.shape[1:], dtype=bool)
    corridor[3:9, 2:10] = True
    if expected_reason == "temporal_corridor_violation":
        corridor[proposal_pixel] = False
    monkeypatch.setattr(
        temporal_module,
        "build_vessel_evidence",
        lambda *_args: np.zeros(frames.shape[1:], dtype=np.float32),
    )
    monkeypatch.setattr(
        temporal_module,
        "_motion_compensated_temporal_corridor",
        lambda *_args: corridor.copy(),
    )

    def adversarial_morphology(*_args: object, **_kwargs: object) -> np.ndarray:
        proposal = baseline_masks.copy()
        proposal[(0, *proposal_pixel)] = True
        for pixel in extra_pixels:
            proposal[(0, *pixel)] = True
        return proposal

    monkeypatch.setattr(
        temporal_module, "_anisotropic_temporal_majority", adversarial_morphology
    )

    _, repaired_masks, audits = temporal_module._repair_temporal_sequence(
        frames,
        probabilities,
        probabilities,
        probabilities,
        baseline_masks,
        catheter_exclusions,
    )

    np.testing.assert_array_equal(repaired_masks, baseline_masks)
    assert audits[0].accepted_pixels == 0
    assert audits[0].rejected_pixels == 1 + len(extra_pixels)
    assert audits[0].area_growth_fraction == pytest.approx(
        (1 + len(extra_pixels)) / 5
    )
    assert audits[0].status == "rejected"
    assert expected_reason in audits[0].reasons


def test_repair_sequence_fails_closed_without_authoritative_catheter_exclusions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frames = np.zeros((2, 12, 12), dtype=np.uint8)
    probabilities = np.zeros(frames.shape, dtype=np.float32)
    baseline_masks = np.zeros(frames.shape, dtype=bool)
    baseline_masks[:, 6, 3:8] = True
    controlled_evidence = np.zeros(frames.shape[1:], dtype=np.float32)
    controlled_evidence[6, 3:9] = 0.70
    monkeypatch.setattr(
        temporal_module,
        "build_vessel_evidence",
        lambda *_args: controlled_evidence.copy(),
    )

    _, repaired_masks, audits = temporal_module._repair_temporal_sequence(
        frames,
        probabilities,
        probabilities,
        probabilities,
        baseline_masks,
        None,
    )

    np.testing.assert_array_equal(repaired_masks, baseline_masks)
    assert [audit.status for audit in audits] == ["rejected", "rejected"]
    assert all(
        audit.reasons == ("catheter_exclusion_unavailable",) for audit in audits
    )


def test_run_temporal_persists_real_repair_and_reconciled_audit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    frame = np.tile(np.arange(32, dtype=np.uint8), (32, 1))
    frames = np.stack([frame, frame, frame])
    dicom_path = tmp_path / "cine.dcm"
    write_test_cine(dicom_path, frames)
    segmentation_dir = tmp_path / "segmentation"
    segmentation_dir.mkdir()
    probability_path = segmentation_dir / "probabilities_u8.raw"
    probability_path.write_bytes(np.zeros(frames.shape, dtype=np.uint8).tobytes())
    write_catheter_exclusions(segmentation_dir, np.zeros(frames.shape, dtype=bool))

    baseline_masks = np.zeros(frames.shape, dtype=bool)
    baseline_masks[:, 16, 10:15] = True
    baseline_masks[:, 16, 16:21] = True
    mask_results = iter(baseline_masks)
    monkeypatch.setattr(
        temporal_module,
        "postprocess_probability",
        lambda _probability: next(mask_results).copy(),
    )
    controlled_evidence = np.zeros(frames.shape[1:], dtype=np.float32)
    controlled_evidence[16, 10:15] = 0.70
    controlled_evidence[16, 16:21] = 0.70
    controlled_evidence[16, 15] = 0.35
    monkeypatch.setattr(
        temporal_module,
        "build_vessel_evidence",
        lambda *_args: controlled_evidence.copy(),
    )
    mp4_names = install_test_output_writers(monkeypatch)

    output_dir = tmp_path / "temporal"
    report = temporal_module.run_temporal(dicom_path, segmentation_dir, output_dir)

    baseline_raw = (baseline_masks.astype(np.uint8) * 255).tobytes(order="C")
    assert (output_dir / "temporal_masks_u8.raw").read_bytes() == baseline_raw
    persisted_masks = (
        np.fromfile(output_dir / "repaired_masks_u8.raw", dtype=np.uint8)
        .reshape(frames.shape)
        .astype(bool)
    )
    assert np.all(persisted_masks[:, 16, 15])
    assert (output_dir / "repaired_centerlines_u8.raw").stat().st_size == int(
        np.prod(frames.shape)
    )
    assert sorted(mp4_names) == ["repaired_overlay.mp4", "temporal_overlay.mp4"]

    repair_audit = json.loads(
        (output_dir / "repair_audit.json").read_text(encoding="utf-8")
    )
    assert repair_audit["schema_version"] == 1
    assert repair_audit["catheter_exclusion_available"] is True
    assert repair_audit["thresholds"] == {
        "seed": 0.70,
        "corridor": 0.20,
        "temporal_addition": 0.35,
        "max_growth_fraction": 0.20,
        "automatic_gap_max_px": 12,
        "review_gap_max_px": 24,
        "maximum_direction_angle_deg": 45.0,
    }
    for index, frame_audit in enumerate(repair_audit["frames"]):
        additions = persisted_masks[index] & ~baseline_masks[index]
        assert frame_audit["accepted_pixels"] == int(np.count_nonzero(additions)) == 1
        assert frame_audit["rejected_pixels"] == 0
        assert frame_audit["area_growth_fraction"] == pytest.approx(0.10)
        assert frame_audit["status"] == "accepted"
        assert frame_audit["reasons"] == []
        assert frame_audit["topology_after"] == report["frames"][index][
            "topology_after"
        ]
    assert report["schema_version"] == 2
    for key, filename in (
        ("repaired_probabilities", "repaired_probabilities_u8.raw"),
        ("repaired_masks", "repaired_masks_u8.raw"),
        ("repaired_centerlines", "repaired_centerlines_u8.raw"),
        ("repair_audit", "repair_audit.json"),
        ("repaired_overlay", "repaired_overlay.mp4"),
    ):
        assert report["assets"][key]["path"] == filename


def test_run_temporal_rejects_malformed_catheter_exclusion_stack(
    tmp_path: Path,
) -> None:
    frames = np.zeros((2, 8, 8), dtype=np.uint8)
    dicom_path = tmp_path / "cine.dcm"
    write_test_cine(dicom_path, frames)
    segmentation_dir = tmp_path / "segmentation"
    segmentation_dir.mkdir()
    (segmentation_dir / "probabilities_u8.raw").write_bytes(
        np.zeros(frames.shape, dtype=np.uint8).tobytes()
    )
    (segmentation_dir / "catheter_exclusions_u8.raw").write_bytes(b"too-short")

    with pytest.raises(
        ValueError,
        match=r"Catheter exclusion stack size mismatch: 9 != 128",
    ):
        temporal_module.run_temporal(
            dicom_path, segmentation_dir, tmp_path / "temporal"
        )


def test_run_temporal_rejects_changed_probability_input_before_any_output_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    frames = np.stack(
        [
            np.tile(np.arange(8, dtype=np.uint8), (8, 1)),
            np.tile(np.arange(8, dtype=np.uint8), (8, 1)),
        ]
    )
    dicom_path = tmp_path / "cine.dcm"
    write_test_cine(dicom_path, frames)
    segmentation_dir = tmp_path / "segmentation"
    segmentation_dir.mkdir()
    probability_path = segmentation_dir / "probabilities_u8.raw"
    probability_path.write_bytes(np.zeros(frames.shape, dtype=np.uint8).tobytes())
    write_catheter_exclusions(segmentation_dir, np.zeros(frames.shape, dtype=bool))
    monkeypatch.setattr(
        temporal_module,
        "postprocess_probability",
        lambda probability: np.asarray(probability) >= 0.5,
    )
    install_test_output_writers(monkeypatch)
    output_dir = tmp_path / "temporal"

    temporal_module.run_temporal(dicom_path, segmentation_dir, output_dir)
    hashes_before = output_file_hashes(output_dir)

    probability_path.write_bytes(
        np.full(frames.shape, 255, dtype=np.uint8).tobytes(order="C")
    )
    writer_calls = install_no_output_call_spies(monkeypatch)
    with pytest.raises(ValueError, match=r"fresh output directory"):
        temporal_module.run_temporal(dicom_path, segmentation_dir, output_dir)

    assert output_file_hashes(output_dir) == hashes_before
    assert writer_calls == []


def test_run_temporal_same_input_rerun_validates_manifest_and_preserves_baseline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    frames = np.stack(
        [
            np.tile(np.arange(8, dtype=np.uint8), (8, 1)),
            np.tile(np.arange(8, dtype=np.uint8), (8, 1)),
        ]
    )
    dicom_path = tmp_path / "cine.dcm"
    write_test_cine(dicom_path, frames)
    segmentation_dir = tmp_path / "segmentation"
    segmentation_dir.mkdir()
    probability_path = segmentation_dir / "probabilities_u8.raw"
    probability_path.write_bytes(np.zeros(frames.shape, dtype=np.uint8).tobytes())
    write_catheter_exclusions(segmentation_dir, np.zeros(frames.shape, dtype=bool))
    monkeypatch.setattr(
        temporal_module,
        "postprocess_probability",
        lambda probability: np.asarray(probability) >= 0.5,
    )
    install_test_output_writers(monkeypatch)
    output_dir = tmp_path / "temporal"

    temporal_module.run_temporal(dicom_path, segmentation_dir, output_dir)
    manifest_path = output_dir / "baseline_provenance.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest_hash = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    expected_targets = {
        "temporal_probabilities_u8.raw",
        "temporal_support_u8.raw",
        "flow_confidence_u8.raw",
        "temporal_masks_u8.raw",
        "temporal_centerlines_u8.raw",
        "probabilities",
        "masks",
        "centerlines",
        "overlays",
        "temporal_overlay.mp4",
    }
    assert manifest["schema_version"] == 1
    assert manifest["inputs"] == {
        "dicom": {
            "byte_length": dicom_path.stat().st_size,
            "sha256": hashlib.sha256(dicom_path.read_bytes()).hexdigest(),
        },
        "probabilities_u8.raw": {
            "byte_length": probability_path.stat().st_size,
            "sha256": hashlib.sha256(probability_path.read_bytes()).hexdigest(),
        },
    }
    assert set(manifest["baseline_targets"]) == expected_targets
    for directory, prefix in (
        ("probabilities", "temporal_probability"),
        ("masks", "temporal_mask"),
        ("centerlines", "temporal_centerline"),
        ("overlays", "temporal_overlay"),
    ):
        record = manifest["baseline_targets"][directory]
        assert record["kind"] == "directory"
        assert [item["path"] for item in record["files"]] == [
            f"{prefix}_0000.png",
            f"{prefix}_0001.png",
        ]
        assert len(record["sha256"]) == 64
    baseline_hashes_before = {
        path: digest
        for path, digest in output_file_hashes(output_dir).items()
        if path == "baseline_provenance.json"
        or path in expected_targets
        or path.split("/", 1)[0]
        in {"probabilities", "masks", "centerlines", "overlays"}
    }

    report = temporal_module.run_temporal(dicom_path, segmentation_dir, output_dir)

    assert report["schema_version"] == 2
    assert hashlib.sha256(manifest_path.read_bytes()).hexdigest() == manifest_hash
    baseline_hashes_after = {
        path: digest
        for path, digest in output_file_hashes(output_dir).items()
        if path == "baseline_provenance.json"
        or path in expected_targets
        or path.split("/", 1)[0]
        in {"probabilities", "masks", "centerlines", "overlays"}
    }
    assert baseline_hashes_after == baseline_hashes_before


def test_same_input_rerun_uses_persisted_flow_diagnostics_and_metrics(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    frame = np.tile(np.arange(16, dtype=np.uint8), (16, 1))
    frames = np.stack([frame, frame, frame])
    dicom_path = tmp_path / "cine.dcm"
    write_test_cine(dicom_path, frames)
    segmentation_dir = tmp_path / "segmentation"
    segmentation_dir.mkdir()
    (segmentation_dir / "probabilities_u8.raw").write_bytes(
        np.zeros(frames.shape, dtype=np.uint8).tobytes()
    )
    write_catheter_exclusions(segmentation_dir, np.zeros(frames.shape, dtype=bool))
    baseline_masks = np.zeros(frames.shape, dtype=bool)
    baseline_masks[:, 8, 4:12] = True
    monkeypatch.setattr(
        temporal_module,
        "postprocess_probability",
        lambda _probability: baseline_masks[0].copy(),
    )
    install_test_output_writers(monkeypatch)
    output_dir = tmp_path / "temporal"

    first_report = temporal_module.run_temporal(
        dicom_path, segmentation_dir, output_dir
    )
    first_frame_diagnostics = [
        (
            item["forward_backward_error_p95_px"],
            item["valid_neighbor_count"],
        )
        for item in first_report["frames"]
    ]
    first_quality_summary = first_report["quality_summary"]
    first_csv = (output_dir / "temporal_quality.csv").read_bytes()
    manifest_path = output_dir / "baseline_provenance.json"
    first_manifest = manifest_path.read_bytes()
    first_manifest_json = json.loads(first_manifest)
    assert set(first_manifest_json["diagnostics"]) == {
        "frames",
        "temporal_metrics",
        "quality_summary",
    }
    baseline_files_before = {
        relative: digest
        for relative, digest in output_file_hashes(output_dir).items()
        if relative == "baseline_provenance.json"
        or relative in first_manifest_json["baseline_targets"]
        or relative.split("/", 1)[0]
        in {"probabilities", "masks", "centerlines", "overlays"}
    }

    def adversarial_flow(source: np.ndarray, _target: np.ndarray) -> np.ndarray:
        flow = np.zeros((*source.shape, 2), dtype=np.float32)
        flow[..., 0] = 4.0
        return flow

    monkeypatch.setattr(repair_module, "dense_flow", adversarial_flow)
    monkeypatch.setattr(temporal_module, "dense_flow", adversarial_flow)
    second_report = temporal_module.run_temporal(
        dicom_path, segmentation_dir, output_dir
    )

    assert [
        (
            item["forward_backward_error_p95_px"],
            item["valid_neighbor_count"],
        )
        for item in second_report["frames"]
    ] == first_frame_diagnostics
    assert second_report["quality_summary"] == first_quality_summary
    assert (output_dir / "temporal_quality.csv").read_bytes() == first_csv
    assert manifest_path.read_bytes() == first_manifest
    baseline_files_after = {
        relative: digest
        for relative, digest in output_file_hashes(output_dir).items()
        if relative == "baseline_provenance.json"
        or relative in first_manifest_json["baseline_targets"]
        or relative.split("/", 1)[0]
        in {"probabilities", "masks", "centerlines", "overlays"}
    }
    assert baseline_files_after == baseline_files_before


def test_run_temporal_rejects_existing_baseline_without_manifest_before_writes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    frames = np.zeros((2, 8, 8), dtype=np.uint8)
    dicom_path = tmp_path / "cine.dcm"
    write_test_cine(dicom_path, frames)
    segmentation_dir = tmp_path / "segmentation"
    segmentation_dir.mkdir()
    (segmentation_dir / "probabilities_u8.raw").write_bytes(
        np.zeros(frames.shape, dtype=np.uint8).tobytes()
    )
    output_dir = tmp_path / "temporal"
    output_dir.mkdir()
    (output_dir / "temporal_masks_u8.raw").write_bytes(b"baseline-sentinel")
    (output_dir / "repair_audit.json").write_bytes(b"repair-sentinel")
    hashes_before = output_file_hashes(output_dir)
    writer_calls = install_no_output_call_spies(monkeypatch)

    with pytest.raises(ValueError, match=r"manifest.*fresh output directory"):
        temporal_module.run_temporal(dicom_path, segmentation_dir, output_dir)

    assert output_file_hashes(output_dir) == hashes_before
    assert writer_calls == []


@pytest.mark.parametrize(
    ("relative_target", "mutation"),
    [
        ("temporal_masks_u8.raw", "tamper"),
        ("probabilities/temporal_probability_0000.png", "delete"),
        ("temporal_overlay.mp4", "delete"),
    ],
)
def test_run_temporal_rejects_tampered_or_missing_baseline_before_writes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    relative_target: str,
    mutation: str,
) -> None:
    frames = np.zeros((2, 8, 8), dtype=np.uint8)
    dicom_path = tmp_path / "cine.dcm"
    write_test_cine(dicom_path, frames)
    segmentation_dir = tmp_path / "segmentation"
    segmentation_dir.mkdir()
    (segmentation_dir / "probabilities_u8.raw").write_bytes(
        np.zeros(frames.shape, dtype=np.uint8).tobytes()
    )
    write_catheter_exclusions(segmentation_dir, np.zeros(frames.shape, dtype=bool))
    install_test_output_writers(monkeypatch)
    output_dir = tmp_path / "temporal"
    temporal_module.run_temporal(dicom_path, segmentation_dir, output_dir)
    target = output_dir / Path(relative_target)
    if mutation == "delete":
        target.unlink()
    else:
        target.write_bytes(b"tampered-baseline")
    hashes_before = output_file_hashes(output_dir)
    writer_calls = install_no_output_call_spies(monkeypatch)

    with pytest.raises(ValueError, match=r"baseline target.*fresh output directory"):
        temporal_module.run_temporal(dicom_path, segmentation_dir, output_dir)

    assert output_file_hashes(output_dir) == hashes_before
    assert writer_calls == []


def test_run_temporal_rejects_unexpected_empty_nested_baseline_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    frames = np.zeros((2, 8, 8), dtype=np.uint8)
    dicom_path = tmp_path / "cine.dcm"
    write_test_cine(dicom_path, frames)
    segmentation_dir = tmp_path / "segmentation"
    segmentation_dir.mkdir()
    (segmentation_dir / "probabilities_u8.raw").write_bytes(
        np.zeros(frames.shape, dtype=np.uint8).tobytes()
    )
    write_catheter_exclusions(segmentation_dir, np.zeros(frames.shape, dtype=bool))
    install_test_output_writers(monkeypatch)
    output_dir = tmp_path / "temporal"
    temporal_module.run_temporal(dicom_path, segmentation_dir, output_dir)
    (output_dir / "probabilities" / "unexpected-empty").mkdir()
    state_before = output_tree_state(output_dir)
    writer_calls = install_no_output_call_spies(monkeypatch)

    with pytest.raises(ValueError, match=r"nested directory.*fresh output directory"):
        temporal_module.run_temporal(dicom_path, segmentation_dir, output_dir)

    assert output_tree_state(output_dir) == state_before
    assert writer_calls == []


@pytest.mark.parametrize(
    ("field", "tampered_value"),
    [
        ("valid_neighbor_count", 99),
        ("relative_area_jump", 999.0),
    ],
)
def test_run_temporal_rejects_semantically_tampered_diagnostics_before_writes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    tampered_value: int | float,
) -> None:
    frames = np.zeros((2, 16, 16), dtype=np.uint8)
    dicom_path = tmp_path / "cine.dcm"
    write_test_cine(dicom_path, frames)
    segmentation_dir = tmp_path / "segmentation"
    segmentation_dir.mkdir()
    (segmentation_dir / "probabilities_u8.raw").write_bytes(
        np.zeros(frames.shape, dtype=np.uint8).tobytes()
    )
    write_catheter_exclusions(segmentation_dir, np.zeros(frames.shape, dtype=bool))
    baseline_masks = np.zeros(frames.shape, dtype=bool)
    baseline_masks[0, 8, 4:8] = True
    baseline_masks[1, 8, 4:10] = True
    mask_results = iter(baseline_masks)
    monkeypatch.setattr(
        temporal_module,
        "postprocess_probability",
        lambda _probability: next(mask_results).copy(),
    )
    install_test_output_writers(monkeypatch)
    output_dir = tmp_path / "temporal"
    temporal_module.run_temporal(dicom_path, segmentation_dir, output_dir)

    manifest_path = output_dir / "baseline_provenance.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if field == "valid_neighbor_count":
        manifest["diagnostics"]["frames"][0][field] = tampered_value
    else:
        manifest["diagnostics"]["temporal_metrics"][1][field] = tampered_value
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    state_before = output_tree_state(output_dir)
    writer_calls = install_no_output_call_spies(monkeypatch)

    with pytest.raises(ValueError, match=rf"{field}.*fresh output directory"):
        temporal_module.run_temporal(dicom_path, segmentation_dir, output_dir)

    assert output_tree_state(output_dir) == state_before
    assert writer_calls == []


@pytest.mark.parametrize(
    "media_case",
    [
        "png-silent",
        "png-partial",
        "png-arbitrary-bytes",
        "png-wrong-dimensions",
        "png-wrong-channels",
        "mp4-silent",
        "mp4-arbitrary-bytes",
        "mp4-truncated",
        "mp4-wrong-dimensions",
    ],
)
def test_fresh_run_rejects_invalid_baseline_media_before_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    media_case: str,
) -> None:
    frames = np.zeros((2, 16, 16), dtype=np.uint8)
    dicom_path = tmp_path / "cine.dcm"
    write_test_cine(dicom_path, frames)
    segmentation_dir = tmp_path / "segmentation"
    segmentation_dir.mkdir()
    (segmentation_dir / "probabilities_u8.raw").write_bytes(
        np.zeros(frames.shape, dtype=np.uint8).tobytes()
    )
    write_catheter_exclusions(segmentation_dir, np.zeros(frames.shape, dtype=bool))
    output_dir = tmp_path / "temporal"
    mp4_calls: list[str] = []

    def adversarial_frames(
        media_frames: np.ndarray, frame_dir: Path, prefix: str
    ) -> list[Path]:
        if not media_case.startswith("png-") or prefix != "temporal_probability":
            return write_valid_frames(media_frames, frame_dir, prefix)
        if media_case == "png-silent":
            return []
        paths = write_valid_frames(media_frames, frame_dir, prefix)
        if media_case == "png-partial":
            paths[-1].unlink()
        elif media_case == "png-arbitrary-bytes":
            paths[0].write_bytes(b"not-a-png")
        elif media_case == "png-wrong-dimensions":
            assert cv2.imwrite(
                str(paths[0]), np.zeros((20, 16), dtype=np.uint8)
            )
        elif media_case == "png-wrong-channels":
            assert cv2.imwrite(
                str(paths[0]), np.zeros((16, 16, 3), dtype=np.uint8)
            )
        return paths

    def adversarial_mp4(
        path: Path,
        media_frames: np.ndarray,
        frame_times: list[float],
        *,
        color: bool = False,
    ) -> None:
        mp4_calls.append(path.name)
        if path.name != "temporal_overlay.mp4" or not media_case.startswith("mp4-"):
            write_valid_mp4(
                path, media_frames, frame_times, color=color
            )
        elif media_case == "mp4-arbitrary-bytes":
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"not-an-mp4")
        elif media_case == "mp4-truncated":
            write_valid_mp4(
                path, media_frames[:1], frame_times[:1], color=color
            )
        elif media_case == "mp4-wrong-dimensions":
            wrong_shape = np.asarray(
                [
                    cv2.resize(
                        frame,
                        (24, 20),
                        interpolation=cv2.INTER_NEAREST,
                    )
                    for frame in media_frames
                ]
            )
            write_valid_mp4(path, wrong_shape, frame_times, color=color)

    monkeypatch.setattr(temporal_module, "write_frames", adversarial_frames)
    monkeypatch.setattr(temporal_module, "write_mp4", adversarial_mp4)

    with pytest.raises(ValueError, match=r"baseline media.*fresh output directory"):
        temporal_module.run_temporal(dicom_path, segmentation_dir, output_dir)

    assert not (output_dir / "baseline_provenance.json").exists()
    assert not (output_dir / "repaired_probabilities_u8.raw").exists()
    assert not (output_dir / "repaired_masks_u8.raw").exists()
    assert not (output_dir / "repaired_centerlines_u8.raw").exists()
    assert not (output_dir / "repaired_overlay.mp4").exists()
    assert not (output_dir / "repair_audit.json").exists()
    assert not (output_dir / "temporal_quality.csv").exists()
    assert not (output_dir / "temporal_report.json").exists()
    assert "repaired_overlay.mp4" not in mp4_calls
    assert not output_dir.exists()
    assert not any(
        ".stage-" in path.name or ".backup-" in path.name
        for path in tmp_path.iterdir()
    )


@pytest.mark.parametrize(
    "media_case",
    [
        "png-arbitrary-bytes",
        "png-wrong-dimensions",
        "png-wrong-channels",
        "mp4-arbitrary-bytes",
        "mp4-truncated",
        "mp4-wrong-dimensions",
    ],
)
def test_rerun_rejects_hash_consistent_invalid_baseline_media_before_writes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    media_case: str,
) -> None:
    frames = np.zeros((2, 16, 16), dtype=np.uint8)
    dicom_path = tmp_path / "cine.dcm"
    write_test_cine(dicom_path, frames)
    segmentation_dir = tmp_path / "segmentation"
    segmentation_dir.mkdir()
    (segmentation_dir / "probabilities_u8.raw").write_bytes(
        np.zeros(frames.shape, dtype=np.uint8).tobytes()
    )
    write_catheter_exclusions(segmentation_dir, np.zeros(frames.shape, dtype=bool))
    install_test_output_writers(monkeypatch)
    output_dir = tmp_path / "temporal"
    temporal_module.run_temporal(dicom_path, segmentation_dir, output_dir)

    manifest_path = output_dir / "baseline_provenance.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if media_case.startswith("png-"):
        target = output_dir / "probabilities" / "temporal_probability_0000.png"
        if media_case == "png-arbitrary-bytes":
            target.write_bytes(b"not-a-png")
        elif media_case == "png-wrong-dimensions":
            assert cv2.imwrite(
                str(target), np.zeros((20, 16), dtype=np.uint8)
            )
        elif media_case == "png-wrong-channels":
            assert cv2.imwrite(
                str(target), np.zeros((16, 16, 3), dtype=np.uint8)
            )
        directory_record = manifest["baseline_targets"]["probabilities"]
        file_record = directory_record["files"][0]
        file_record["byte_length"] = target.stat().st_size
        file_record["sha256"] = hashlib.sha256(target.read_bytes()).hexdigest()
        canonical = json.dumps(
            directory_record["files"],
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
        directory_record["sha256"] = hashlib.sha256(canonical).hexdigest()
    else:
        target = output_dir / "temporal_overlay.mp4"
        if media_case == "mp4-arbitrary-bytes":
            target.write_bytes(b"not-an-mp4")
        elif media_case == "mp4-truncated":
            write_valid_mp4(target, np.zeros((1, 16, 16, 3), np.uint8), [125.0], color=True)
        elif media_case == "mp4-wrong-dimensions":
            write_valid_mp4(
                target,
                np.zeros((2, 20, 24, 3), np.uint8),
                [125.0, 125.0],
                color=True,
            )
        video_record = manifest["baseline_targets"]["temporal_overlay.mp4"]
        video_record["byte_length"] = target.stat().st_size
        video_record["sha256"] = hashlib.sha256(target.read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    state_before = output_tree_state(output_dir)
    writer_calls = install_no_output_call_spies(monkeypatch)

    with pytest.raises(ValueError, match=r"baseline media.*fresh output directory"):
        temporal_module.run_temporal(dicom_path, segmentation_dir, output_dir)

    assert output_tree_state(output_dir) == state_before
    assert writer_calls == []


@pytest.mark.parametrize(
    "case",
    [
        "malformed-json",
        "top-level-list",
        "top-level-extra",
        "schema-bool",
        "schema-float",
        "missing-inputs",
        "inputs-list",
        "input-byte-length-bool",
        "input-hash-nonhex",
        "target-record-list",
        "target-extra-field",
        "target-byte-length-negative",
        "directory-file-count-float",
        "directory-files-dict",
        "directory-path-parent",
        "diagnostics-list",
        "diagnostic-nonfinite",
        "diagnostic-index-bool",
        "metric-missing-field",
        "summary-count-bool",
    ],
)
def test_run_temporal_strictly_rejects_malformed_baseline_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    case: str,
) -> None:
    frames = np.zeros((2, 8, 8), dtype=np.uint8)
    dicom_path = tmp_path / "cine.dcm"
    write_test_cine(dicom_path, frames)
    segmentation_dir = tmp_path / "segmentation"
    segmentation_dir.mkdir()
    (segmentation_dir / "probabilities_u8.raw").write_bytes(
        np.zeros(frames.shape, dtype=np.uint8).tobytes()
    )
    write_catheter_exclusions(segmentation_dir, np.zeros(frames.shape, dtype=bool))
    install_test_output_writers(monkeypatch)
    output_dir = tmp_path / "temporal"
    initial_report = temporal_module.run_temporal(
        dicom_path, segmentation_dir, output_dir
    )
    manifest_path = output_dir / "baseline_provenance.json"
    manifest: object = json.loads(manifest_path.read_text(encoding="utf-8"))
    if isinstance(manifest, dict) and "diagnostics" not in manifest:
        manifest["diagnostics"] = {
            "frames": [
                {
                    "frame_index": index,
                    "forward_backward_error_p95_px": frame_record[
                        "forward_backward_error_p95_px"
                    ],
                    "valid_neighbor_count": frame_record["valid_neighbor_count"],
                }
                for index, frame_record in enumerate(initial_report["frames"])
            ],
            "temporal_metrics": [
                {
                    "frame_index": 0,
                    "warped_iou_to_previous": None,
                    "relative_area_jump": None,
                    "relative_centerline_jump": None,
                    "quality_flag": "provisional",
                },
                {
                    "frame_index": 1,
                    "warped_iou_to_previous": 1.0,
                    "relative_area_jump": 0.0,
                    "relative_centerline_jump": 0.0,
                    "quality_flag": "provisional",
                },
            ],
            "quality_summary": initial_report["quality_summary"],
        }

    if case == "malformed-json":
        manifest_path.write_text("{", encoding="utf-8")
    elif case == "top-level-list":
        manifest_path.write_text("[]", encoding="utf-8")
    else:
        assert isinstance(manifest, dict)
        if case == "top-level-extra":
            manifest["unexpected"] = True
        elif case == "schema-bool":
            manifest["schema_version"] = True
        elif case == "schema-float":
            manifest["schema_version"] = 1.0
        elif case == "missing-inputs":
            manifest.pop("inputs")
        elif case == "inputs-list":
            manifest["inputs"] = []
        elif case == "input-byte-length-bool":
            manifest["inputs"]["dicom"]["byte_length"] = True
        elif case == "input-hash-nonhex":
            manifest["inputs"]["dicom"]["sha256"] = "g" * 64
        elif case == "target-record-list":
            manifest["baseline_targets"]["temporal_masks_u8.raw"] = []
        elif case == "target-extra-field":
            manifest["baseline_targets"]["temporal_masks_u8.raw"][
                "unexpected"
            ] = 1
        elif case == "target-byte-length-negative":
            manifest["baseline_targets"]["temporal_masks_u8.raw"][
                "byte_length"
            ] = -1
        elif case == "directory-file-count-float":
            manifest["baseline_targets"]["probabilities"]["file_count"] = 2.0
        elif case == "directory-files-dict":
            manifest["baseline_targets"]["probabilities"]["files"] = {}
        elif case == "directory-path-parent":
            manifest["baseline_targets"]["probabilities"]["files"][0][
                "path"
            ] = "../escape.png"
        elif case == "diagnostics-list":
            manifest["diagnostics"] = []
        elif case == "diagnostic-nonfinite":
            manifest["diagnostics"]["frames"][0][
                "forward_backward_error_p95_px"
            ] = float("nan")
        elif case == "diagnostic-index-bool":
            manifest["diagnostics"]["frames"][0]["frame_index"] = True
        elif case == "metric-missing-field":
            manifest["diagnostics"]["temporal_metrics"][0].pop("quality_flag")
        elif case == "summary-count-bool":
            manifest["diagnostics"]["quality_summary"]["review_frames"] = True
        manifest_path.write_text(
            json.dumps(manifest, indent=2, allow_nan=True), encoding="utf-8"
        )

    state_before = output_tree_state(output_dir)
    writer_calls = install_no_output_call_spies(monkeypatch)
    with pytest.raises(ValueError, match=r"fresh output directory"):
        temporal_module.run_temporal(dicom_path, segmentation_dir, output_dir)

    assert output_tree_state(output_dir) == state_before
    assert writer_calls == []


def test_run_temporal_rejects_missing_probability_input_before_output_changes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    frames = np.zeros((2, 8, 8), dtype=np.uint8)
    dicom_path = tmp_path / "cine.dcm"
    write_test_cine(dicom_path, frames)
    segmentation_dir = tmp_path / "segmentation"
    segmentation_dir.mkdir()
    output_dir = tmp_path / "temporal"
    output_dir.mkdir()
    (output_dir / "unrelated.sentinel").write_bytes(b"unchanged")
    state_before = output_tree_state(output_dir)
    writer_calls = install_no_output_call_spies(monkeypatch)

    with pytest.raises(ValueError, match=r"Probability input.*missing"):
        temporal_module.run_temporal(dicom_path, segmentation_dir, output_dir)

    assert output_tree_state(output_dir) == state_before
    assert writer_calls == []


def test_topology_before_comes_from_validated_persisted_baseline_mask(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    frames = np.zeros((2, 8, 8), dtype=np.uint8)
    dicom_path = tmp_path / "cine.dcm"
    write_test_cine(dicom_path, frames)
    segmentation_dir = tmp_path / "segmentation"
    segmentation_dir.mkdir()
    (segmentation_dir / "probabilities_u8.raw").write_bytes(
        np.zeros(frames.shape, dtype=np.uint8).tobytes()
    )
    write_catheter_exclusions(segmentation_dir, np.zeros(frames.shape, dtype=bool))
    baseline = np.zeros(frames.shape, dtype=bool)
    baseline[:, 3, 2:7] = True
    changed_recomputation = np.zeros(frames.shape, dtype=bool)
    changed_recomputation[:, 2, 2:4] = True
    changed_recomputation[:, 5, 5:7] = True
    selected_masks = [baseline]
    monkeypatch.setattr(
        temporal_module,
        "postprocess_probability",
        lambda _probability: selected_masks[0][0].copy(),
    )
    install_test_output_writers(monkeypatch)
    output_dir = tmp_path / "temporal"
    temporal_module.run_temporal(dicom_path, segmentation_dir, output_dir)
    selected_masks[0] = changed_recomputation

    report = temporal_module.run_temporal(dicom_path, segmentation_dir, output_dir)

    expected_topology = {
        "component_count": 1,
        "largest_component_fraction": 1.0,
        "endpoint_count": 2,
        "junction_count": 0,
        "skeleton_length_px": 5,
    }
    assert [item["topology_before"] for item in report["frames"]] == [
        expected_topology,
        expected_topology,
    ]
    persisted = np.fromfile(
        output_dir / "temporal_masks_u8.raw", dtype=np.uint8
    ).reshape(frames.shape)
    np.testing.assert_array_equal(persisted != 0, baseline)


@pytest.mark.parametrize("displacement_px", [1.0, 8.0])
def test_report_flow_error_p95_uses_only_sampled_backward_locations(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    displacement_px: float,
) -> None:
    frames = np.stack(
        [
            np.full((8, 8), 10, dtype=np.uint8),
            np.full((8, 8), 20, dtype=np.uint8),
        ]
    )
    dicom_path = tmp_path / "cine.dcm"
    write_test_cine(dicom_path, frames)
    segmentation_dir = tmp_path / "segmentation"
    segmentation_dir.mkdir()
    probabilities = np.zeros(frames.shape, dtype=np.uint8)
    (segmentation_dir / "probabilities_u8.raw").write_bytes(probabilities.tobytes())

    def translated_flow(source: np.ndarray, target: np.ndarray) -> np.ndarray:
        direction = 1.0 if int(source[0, 0]) < int(target[0, 0]) else -1.0
        flow = np.zeros((*source.shape, 2), dtype=np.float32)
        flow[..., 0] = direction * displacement_px
        return flow

    monkeypatch.setattr(repair_module, "dense_flow", translated_flow)
    monkeypatch.setattr(
        temporal_module,
        "dense_flow",
        lambda source, _target: np.zeros((*source.shape, 2), dtype=np.float32),
    )
    install_test_output_writers(monkeypatch)

    report = temporal_module.run_temporal(
        dicom_path, segmentation_dir, tmp_path / "temporal"
    )

    error_p95 = report["frames"][0]["forward_backward_error_p95_px"]
    assert np.isfinite(error_p95)
    assert error_p95 == pytest.approx(0.0)


def test_report_counts_disjoint_valid_neighbor_frames_separately_from_pixel_counts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frames = np.stack(
        [
            np.full((8, 8), 10, dtype=np.uint8),
            np.full((8, 8), 20, dtype=np.uint8),
            np.full((8, 8), 30, dtype=np.uint8),
        ]
    )
    probabilities = np.zeros(frames.shape, dtype=np.float32)
    masks = probabilities > 0.5
    dicom_path = tmp_path / "cine.dcm"
    write_test_cine(dicom_path, frames)
    segmentation_dir = tmp_path / "segmentation"
    segmentation_dir.mkdir()
    (segmentation_dir / "probabilities_u8.raw").write_bytes(
        np.zeros(frames.shape, dtype=np.uint8).tobytes()
    )

    def disjoint_evidence(
        current: np.ndarray,
        neighbor: np.ndarray,
        neighbor_probability: np.ndarray,
        neighbor_mask: np.ndarray,
    ) -> FlowEvidence:
        confidence = np.zeros(current.shape, dtype=np.float32)
        if int(current[0, 0]) == 20 and int(neighbor[0, 0]) == 10:
            confidence[:, :4] = 1.0
        elif int(current[0, 0]) == 20 and int(neighbor[0, 0]) == 30:
            confidence[:, 4:] = 1.0
        return FlowEvidence(
            warped_probability=neighbor_probability.copy(),
            warped_mask=neighbor_mask.copy(),
            confidence=confidence,
            forward_backward_error_px=np.zeros(current.shape, dtype=np.float32),
        )

    monkeypatch.setattr(
        repair_module, "bidirectional_flow_evidence", disjoint_evidence
    )
    monkeypatch.setattr(
        temporal_module,
        "dense_flow",
        lambda source, _target: np.zeros((*source.shape, 2), dtype=np.float32),
    )
    install_test_output_writers(monkeypatch)

    _, valid_neighbor_count = temporal_support(
        1, frames, probabilities, masks
    )
    report = temporal_module.run_temporal(
        dicom_path, segmentation_dir, tmp_path / "temporal"
    )

    assert int(np.max(valid_neighbor_count)) == 1
    assert report["frames"][1]["valid_neighbor_count"] == 2


def test_legacy_metric_warp_retains_reflected_border(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    neighbor = np.asarray([[10, 20, 30], [40, 50, 60]], dtype=np.float32)
    frame = np.zeros(neighbor.shape, dtype=np.uint8)
    flow = np.zeros((*neighbor.shape, 2), dtype=np.float32)
    flow[..., 0] = 1.0
    monkeypatch.setattr(temporal_module, "dense_flow", lambda *_args: flow)

    actual = temporal_module.warp_neighbor_to_current(neighbor, frame, frame)

    np.testing.assert_array_equal(
        actual,
        np.asarray([[20, 30, 20], [50, 60, 50]], dtype=np.float32),
    )


def test_temporal_support_and_fusion_use_valid_adjacent_frames_without_mutation() -> None:
    frame = synthetic_dark_vessel(shift_x=0)
    frames = np.stack([frame, frame, frame])
    probabilities = np.stack(
        [
            np.full(frame.shape, 0.2, dtype=np.float32),
            np.full(frame.shape, 0.4, dtype=np.float32),
            np.full(frame.shape, 0.8, dtype=np.float32),
        ]
    )
    masks = probabilities > 0.5
    frames_before = frames.copy()
    probabilities_before = probabilities.copy()
    masks_before = masks.copy()

    support, valid_neighbor_count = temporal_support(
        1, frames, probabilities, masks
    )
    fused = temporal_fuse(frames, probabilities)

    interior = np.s_[1:-1, 1:-1]
    np.testing.assert_allclose(support[interior], 0.5, atol=1e-6)
    np.testing.assert_array_equal(valid_neighbor_count[interior], 2)
    np.testing.assert_allclose(fused[1][interior], 0.44, atol=1e-5)
    np.testing.assert_allclose(fused[0][interior], 0.25, atol=1e-5)
    np.testing.assert_array_equal(frames, frames_before)
    np.testing.assert_array_equal(probabilities, probabilities_before)
    np.testing.assert_array_equal(masks, masks_before)


def test_fusion_uses_continuous_pixelwise_flow_confidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frames = np.stack(
        [
            np.full((4, 4), 0, dtype=np.uint8),
            np.full((4, 4), 10, dtype=np.uint8),
            np.full((4, 4), 20, dtype=np.uint8),
        ]
    )
    probabilities = np.stack(
        [
            np.zeros((4, 4), dtype=np.float32),
            np.full((4, 4), 0.4, dtype=np.float32),
            np.ones((4, 4), dtype=np.float32),
        ]
    )
    masks = probabilities > 0.5

    def fixed_evidence(
        _current: np.ndarray,
        neighbor: np.ndarray,
        neighbor_probability: np.ndarray,
        neighbor_mask: np.ndarray,
    ) -> FlowEvidence:
        confidence_value = 0.5 if int(neighbor[0, 0]) == 20 else 1.0
        return FlowEvidence(
            warped_probability=neighbor_probability.copy(),
            warped_mask=neighbor_mask.copy(),
            confidence=np.full(neighbor.shape, confidence_value, dtype=np.float32),
            forward_backward_error_px=np.zeros(neighbor.shape, dtype=np.float32),
        )

    monkeypatch.setattr(repair_module, "bidirectional_flow_evidence", fixed_evidence)

    support, valid_neighbor_count = temporal_support(
        1, frames, probabilities, masks
    )
    fused = temporal_fuse(frames, probabilities)

    np.testing.assert_allclose(fused[1], 0.34 / 0.90, atol=1e-6)
    np.testing.assert_allclose(support, 1.0 / 3.0, atol=1e-6)
    np.testing.assert_array_equal(valid_neighbor_count, 2)


def test_temporal_report_preserves_contract_and_writes_pixelwise_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    frame = np.tile(np.arange(32, dtype=np.uint8), (32, 1))
    frames = np.stack([frame, frame])
    dicom_path = tmp_path / "cine.dcm"
    write_test_cine(dicom_path, frames)

    segmentation_dir = tmp_path / "segmentation"
    segmentation_dir.mkdir()
    probability_bytes = np.zeros(frames.shape, dtype=np.uint8).tobytes(order="C")
    probability_path = segmentation_dir / "probabilities_u8.raw"
    probability_path.write_bytes(probability_bytes)
    dicom_hash_before = hashlib.sha256(dicom_path.read_bytes()).hexdigest()
    probability_hash_before = hashlib.sha256(probability_path.read_bytes()).hexdigest()

    install_test_output_writers(monkeypatch)

    output_dir = tmp_path / "temporal"
    report = temporal_module.run_temporal(dicom_path, segmentation_dir, output_dir)

    assert report["schema_version"] == 2
    assert report["clinical_status"] == "research-only; temporal smoothing does not create missing anatomy"
    assert report["assets"]["probabilities"]["path"] == "temporal_probabilities_u8.raw"
    assert report["assets"]["masks"]["path"] == "temporal_masks_u8.raw"
    assert report["assets"]["centerlines"]["path"] == "temporal_centerlines_u8.raw"
    expected_byte_length = int(np.prod(frames.shape))
    for key, filename in (
        ("temporal_support", "temporal_support_u8.raw"),
        ("flow_confidence", "flow_confidence_u8.raw"),
    ):
        asset = report["assets"][key]
        assert asset == {"path": filename, "byte_length": expected_byte_length}
        assert (output_dir / filename).stat().st_size == expected_byte_length

    assert (output_dir / "temporal_support_u8.raw").read_bytes() == bytes(expected_byte_length)
    flow_confidence_raw = np.fromfile(output_dir / "flow_confidence_u8.raw", dtype=np.uint8)
    assert np.median(flow_confidence_raw) == 255
    quality_fields = {
        "median_flow_confidence",
        "supported_pixel_fraction",
        "forward_backward_error_p95_px",
        "valid_neighbor_count",
    }
    for index, frame_record in enumerate(report["frames"]):
        assert frame_record["frame_index"] == index
        assert frame_record["topology_before"] == frame_record["topology_after"]
        assert quality_fields <= frame_record.keys()
        assert frame_record["median_flow_confidence"] == pytest.approx(1.0)
        assert frame_record["supported_pixel_fraction"] == pytest.approx(0.0)
        assert frame_record["forward_backward_error_p95_px"] == pytest.approx(0.0)
        assert frame_record["valid_neighbor_count"] == 1

    assert json.loads((output_dir / "temporal_report.json").read_text(encoding="utf-8")) == report
    assert hashlib.sha256(dicom_path.read_bytes()).hexdigest() == dicom_hash_before
    assert hashlib.sha256(probability_path.read_bytes()).hexdigest() == probability_hash_before


def test_minimum_cost_path_connects_collinear_supported_endpoints() -> None:
    mask, graph, evidence, support = broken_collinear_gap()

    candidates = find_gap_candidates(graph, evidence, support)
    repaired, audited = repair_short_gaps(mask, graph, candidates)

    assert mask_topology_metrics(repaired).component_count == 1
    assert [item.decision for item in audited] == ["accepted"]


def test_review_and_rejected_candidates_are_never_written() -> None:
    mask, graph, evidence, support = broken_collinear_gap(distance_px=13)
    review_candidate = find_gap_candidates(graph, evidence, support)[0]
    rejected_candidate = replace(
        review_candidate,
        distance_px=12.0,
        decision="rejected",
        reasons=("insufficient_evidence",),
    )

    repaired, audited = repair_short_gaps(
        mask, graph, [review_candidate, rejected_candidate]
    )

    np.testing.assert_array_equal(repaired, mask)
    assert {item.decision for item in audited} == {"review", "rejected"}


def test_exclusion_in_restored_radius_rejects_whole_candidate() -> None:
    mask, graph, evidence, support = broken_collinear_gap()
    exclusion = np.zeros(mask.shape, dtype=bool)
    exclusion[41, 30] = True
    candidates = find_gap_candidates(
        graph, evidence, support, exclusion_mask=exclusion
    )
    assert candidates[0].decision == "accepted"

    repaired, audited = repair_short_gaps(
        mask, graph, candidates, exclusion_mask=exclusion
    )

    np.testing.assert_array_equal(repaired, mask)
    assert audited[0].decision == "rejected"
    assert audited[0].crosses_exclusion
    assert "exclusion_intersection" in audited[0].reasons


@pytest.mark.parametrize(
    ("endpoint_radius", "expected_radius"),
    [(0.25, 1), (9.0, 5)],
)
def test_radius_restoration_clips_endpoint_median_to_one_through_five_pixels(
    endpoint_radius: float,
    expected_radius: int,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mask, graph, evidence, support = broken_collinear_gap()
    candidate = find_gap_candidates(graph, evidence, support)[0]
    distance_map = np.ones(mask.shape, dtype=np.float64)
    node_by_id = {node.node_id: node for node in graph.nodes}
    for node_id in (candidate.start_node_id, candidate.end_node_id):
        y, x = np.rint(node_by_id[node_id].yx).astype(int)
        distance_map[y, x] = endpoint_radius
    monkeypatch.setattr(
        repair_module.ndimage,
        "distance_transform_edt",
        lambda _mask: distance_map.copy(),
    )

    repaired, audited = repair_short_gaps(mask, graph, [candidate])

    midpoint_y, midpoint_x = candidate.path_yx[len(candidate.path_yx) // 2]
    assert audited[0].decision == "accepted"
    assert repaired[midpoint_y + expected_radius, midpoint_x]
    assert not repaired[midpoint_y + expected_radius + 1, midpoint_x]


def test_duplicate_endpoint_pair_is_rejected_after_canonical_selection() -> None:
    mask, graph, evidence, support = broken_collinear_gap()
    first = find_gap_candidates(graph, evidence, support)[0]
    lower_priority_duplicate = replace(first, score=first.score + 1.0)

    repaired, audited = repair_short_gaps(
        mask, graph, [lower_priority_duplicate, first]
    )

    assert mask_topology_metrics(repaired).component_count == 1
    assert [item.decision for item in audited] == ["accepted", "rejected"]
    assert "duplicate_endpoint_pair" in audited[1].reasons


def test_loop_increase_is_rolled_back_with_audited_reason() -> None:
    mask = np.zeros((40, 40), dtype=bool)
    mask[10, 10:29] = True
    mask[10:21, 10] = True
    mask[20, 10:29] = True
    for y, x in ((11, 29), (12, 30), (19, 29), (18, 30)):
        mask[y, x] = True
    graph = extract_vessel_graph(mask, root_hint_yx=(12.0, 30.0))
    candidate = accepted_candidate_for_graph(graph, (12, 30), (18, 30))

    repaired, audited = repair_short_gaps(mask, graph, [candidate])

    np.testing.assert_array_equal(repaired, mask)
    assert audited[0].decision == "rejected"
    assert "unsupported_loop" in audited[0].reasons


def test_candidate_that_does_not_merge_components_is_rolled_back() -> None:
    mask, graph, evidence, support = broken_collinear_gap()
    candidate = find_gap_candidates(graph, evidence, support)[0]
    mask[candidate.path_yx[:, 0], candidate.path_yx[:, 1]] = True

    repaired, audited = repair_short_gaps(mask, graph, [candidate])

    np.testing.assert_array_equal(repaired, mask)
    assert audited[0].decision == "rejected"
    assert "no_component_merge" in audited[0].reasons


def test_gap_repair_rejects_malformed_shapes_and_preserves_every_input() -> None:
    mask, graph, evidence, support = broken_collinear_gap()
    exclusion = np.zeros(mask.shape, dtype=bool)
    mask_before = mask.copy()
    evidence_before = evidence.copy()
    support_before = support.copy()
    exclusion_before = exclusion.copy()
    graph_points_before = [edge.points_yx.copy() for edge in graph.edges]

    candidates = find_gap_candidates(
        graph, evidence, support, exclusion_mask=exclusion
    )
    path_before = [candidate.path_yx.copy() for candidate in candidates]
    repair_short_gaps(mask, graph, candidates, exclusion_mask=exclusion)

    np.testing.assert_array_equal(mask, mask_before)
    np.testing.assert_array_equal(evidence, evidence_before)
    np.testing.assert_array_equal(support, support_before)
    np.testing.assert_array_equal(exclusion, exclusion_before)
    for edge, before in zip(graph.edges, graph_points_before, strict=True):
        np.testing.assert_array_equal(edge.points_yx, before)
    for candidate, before in zip(candidates, path_before, strict=True):
        np.testing.assert_array_equal(candidate.path_yx, before)

    with pytest.raises(ValueError, match="exclusion_mask"):
        repair_short_gaps(mask, graph, candidates, exclusion_mask=exclusion[:-1])

    malformed = replace(candidates[0], path_yx=np.zeros((4,), dtype=np.float64))
    repaired, audited = repair_short_gaps(mask, graph, [malformed])
    np.testing.assert_array_equal(repaired, mask)
    assert audited[0].decision == "rejected"
    assert "malformed_candidate" in audited[0].reasons


@pytest.mark.parametrize("auto_gap_px", [0, 13, True])
def test_auto_gap_limit_cannot_relax_the_one_through_twelve_cap(
    auto_gap_px: object,
) -> None:
    mask, graph, evidence, support = broken_collinear_gap()
    candidate = find_gap_candidates(graph, evidence, support)[0]

    with pytest.raises(ValueError, match="1 through 12"):
        repair_short_gaps(
            mask,
            graph,
            [candidate],
            auto_gap_px=auto_gap_px,  # type: ignore[arg-type]
        )


def test_repair_rejects_a_claimed_angle_that_does_not_match_the_graph() -> None:
    mask, graph, evidence, support = broken_collinear_gap()
    candidate = replace(
        find_gap_candidates(graph, evidence, support)[0],
        angle_deg=45.0 + 5e-10,
    )

    repaired, audited = repair_short_gaps(mask, graph, [candidate])

    np.testing.assert_array_equal(repaired, mask)
    assert audited[0].decision == "rejected"
    assert audited[0].reasons == ("malformed_candidate",)


def test_repair_rejects_a_tightly_forged_endpoint_distance() -> None:
    mask, graph, evidence, support = broken_collinear_gap()
    candidate = find_gap_candidates(graph, evidence, support)[0]
    node_by_id = {node.node_id: node for node in graph.nodes}
    start = np.asarray(node_by_id[candidate.start_node_id].yx, dtype=np.float64)
    end = np.asarray(node_by_id[candidate.end_node_id].yx, dtype=np.float64)
    direction = (end - start) / np.linalg.norm(end - start)
    forged_end = end + direction * 5e-10
    forged_graph = replace(
        graph,
        nodes=tuple(
            replace(node, yx=tuple(forged_end))
            if node.node_id == candidate.end_node_id
            else node
            for node in graph.nodes
        ),
    )

    repaired, audited = repair_short_gaps(mask, forged_graph, [candidate])

    np.testing.assert_array_equal(repaired, mask)
    assert audited[0].decision == "rejected"
    assert audited[0].reasons == ("malformed_candidate",)


def test_repair_rejects_a_forged_zero_angle_for_perpendicular_tangents() -> None:
    mask = np.zeros((80, 80), dtype=bool)
    mask[24:41, 24] = True
    mask[40:57, 36] = True
    graph = extract_vessel_graph(mask, root_hint_yx=(24.0, 24.0))
    evidence = np.ones(mask.shape, dtype=np.float64)
    candidate = find_gap_candidates(graph, evidence, evidence)[0]
    assert candidate.angle_deg == pytest.approx(90.0)
    forged = replace(candidate, angle_deg=0.0, decision="accepted", reasons=())

    repaired, audited = repair_short_gaps(mask, graph, [forged])

    np.testing.assert_array_equal(repaired, mask)
    assert audited[0].decision == "rejected"
    assert audited[0].reasons == ("malformed_candidate",)


def test_repair_rejects_a_path_outside_the_padded_endpoint_window() -> None:
    mask, graph, evidence, support = broken_collinear_gap()
    candidate = find_gap_candidates(graph, evidence, support)[0]
    detour = np.asarray(
        [
            *((y, 24) for y in range(40, 30, -1)),
            *((31, x) for x in range(25, 37)),
            *((y, 36) for y in range(32, 41)),
        ],
        dtype=np.int64,
    )
    forged = replace(candidate, path_yx=detour)

    repaired, audited = repair_short_gaps(mask, graph, [forged])

    np.testing.assert_array_equal(repaired, mask)
    assert audited[0].decision == "rejected"
    assert audited[0].reasons == ("malformed_candidate",)


@pytest.mark.parametrize(
    ("field_name", "field_value"),
    [
        ("distance_px", 10**10000),
        ("decision", []),
    ],
    ids=("huge-integer", "unhashable-decision"),
)
def test_exceptional_candidate_values_are_audited_as_malformed(
    field_name: str,
    field_value: object,
) -> None:
    mask, graph, evidence, support = broken_collinear_gap()
    candidate = find_gap_candidates(graph, evidence, support)[0]
    malformed = replace(candidate, **{field_name: field_value})

    repaired, audited = repair_short_gaps(mask, graph, [malformed])

    np.testing.assert_array_equal(repaired, mask)
    assert audited[0].decision == "rejected"
    assert audited[0].reasons == ("malformed_candidate",)


def test_diagonal_tree_merge_is_not_mistaken_for_a_new_loop() -> None:
    mask = np.zeros((48, 48), dtype=bool)
    for offset in range(8):
        mask[20 + offset, 20 + offset] = True
        mask[12 - offset, 12 - offset] = True
    graph = extract_vessel_graph(mask, root_hint_yx=(27.0, 27.0))
    candidate = accepted_candidate_for_graph(graph, (20, 20), (12, 12))

    repaired, audited = repair_short_gaps(mask, graph, [candidate])

    assert mask_topology_metrics(repaired).component_count == 1
    assert audited[0].decision == "accepted"
    assert "unsupported_loop" not in audited[0].reasons


def test_non_candidate_is_rejected_before_sorting_without_mask_mutation() -> None:
    mask, graph, _evidence, _support = broken_collinear_gap()
    before = mask.copy()

    with pytest.raises(ValueError, match="GapCandidate"):
        repair_short_gaps(mask, graph, [object()])  # type: ignore[list-item]

    np.testing.assert_array_equal(mask, before)


@pytest.mark.parametrize(
    "case",
    [
        "bool-distance",
        "bool-angle",
        "bool-evidence",
        "bool-support",
        "bool-score",
        "nonfinite-distance",
        "distance-low",
        "distance-high",
        "angle-low",
        "angle-high",
        "evidence-low",
        "evidence-high",
        "support-low",
        "support-high",
        "score-low",
        "decision",
        "reasons-list",
        "reason-nonstring",
        "reason-duplicate",
        "accepted-reason",
        "missing-endpoint",
        "same-endpoint",
        "distance-mismatch",
        "fractional-path",
        "out-of-bounds-path",
        "disconnected-path",
        "reversed-path",
    ],
)
def test_malformed_candidate_is_audited_without_changing_mask(case: str) -> None:
    mask, graph, evidence, support = broken_collinear_gap()
    candidate = find_gap_candidates(graph, evidence, support)[0]
    path = candidate.path_yx.copy()
    if case == "bool-distance":
        malformed = replace(candidate, distance_px=True)
    elif case == "bool-angle":
        malformed = replace(candidate, angle_deg=True)
    elif case == "bool-evidence":
        malformed = replace(candidate, mean_evidence=True)
    elif case == "bool-support":
        malformed = replace(candidate, temporal_support=False)
    elif case == "bool-score":
        malformed = replace(candidate, score=True)
    elif case == "nonfinite-distance":
        malformed = replace(candidate, distance_px=float("nan"))
    elif case == "distance-low":
        malformed = replace(candidate, distance_px=0.0)
    elif case == "distance-high":
        malformed = replace(candidate, distance_px=25.0)
    elif case == "angle-low":
        malformed = replace(candidate, angle_deg=-0.1)
    elif case == "angle-high":
        malformed = replace(candidate, angle_deg=180.1)
    elif case == "evidence-low":
        malformed = replace(candidate, mean_evidence=-0.1)
    elif case == "evidence-high":
        malformed = replace(candidate, mean_evidence=1.1)
    elif case == "support-low":
        malformed = replace(candidate, temporal_support=-0.1)
    elif case == "support-high":
        malformed = replace(candidate, temporal_support=1.1)
    elif case == "score-low":
        malformed = replace(candidate, score=-0.1)
    elif case == "decision":
        malformed = replace(candidate, decision="unknown")  # type: ignore[arg-type]
    elif case == "reasons-list":
        malformed = replace(candidate, decision="rejected", reasons=["bad"])  # type: ignore[arg-type]
    elif case == "reason-nonstring":
        malformed = replace(candidate, decision="rejected", reasons=(1,))  # type: ignore[arg-type]
    elif case == "reason-duplicate":
        malformed = replace(
            candidate,
            decision="rejected",
            reasons=("insufficient_evidence", "insufficient_evidence"),
        )
    elif case == "accepted-reason":
        malformed = replace(candidate, reasons=("insufficient_evidence",))
    elif case == "missing-endpoint":
        malformed = replace(candidate, end_node_id="missing")
    elif case == "same-endpoint":
        malformed = replace(candidate, end_node_id=candidate.start_node_id)
    elif case == "distance-mismatch":
        malformed = replace(candidate, distance_px=11.0)
    elif case == "fractional-path":
        path = path.astype(np.float64)
        path[1, 1] += 0.25
        malformed = replace(candidate, path_yx=path)
    elif case == "out-of-bounds-path":
        path[1] = (-1, -1)
        malformed = replace(candidate, path_yx=path)
    elif case == "disconnected-path":
        malformed = replace(candidate, path_yx=path[[0, -1]])
    else:
        malformed = replace(candidate, path_yx=path[::-1].copy())

    before = mask.copy()
    repaired, audited = repair_short_gaps(mask, graph, [malformed])

    np.testing.assert_array_equal(mask, before)
    np.testing.assert_array_equal(repaired, before)
    assert audited[0].decision == "rejected"
    assert "malformed_candidate" in audited[0].reasons


def test_duplicate_pair_selection_and_audit_are_input_order_independent() -> None:
    mask, graph, evidence, support = broken_collinear_gap()
    direct = find_gap_candidates(graph, evidence, support)[0]
    detour_path = np.asarray(
        [(40, 24), *[(39, x) for x in range(25, 36)], (40, 36)],
        dtype=np.int64,
    )
    detour = replace(direct, path_yx=detour_path)
    reversed_detour = replace(
        detour,
        start_node_id=detour.end_node_id,
        end_node_id=detour.start_node_id,
        path_yx=detour_path[::-1].copy(),
    )

    first_mask, first_audit = repair_short_gaps(
        mask, graph, [direct, detour, reversed_detour]
    )
    second_mask, second_audit = repair_short_gaps(
        mask, graph, [reversed_detour, detour, direct]
    )

    np.testing.assert_array_equal(first_mask, second_mask)
    first_summary = [
        (item.start_node_id, item.end_node_id, item.decision, item.reasons)
        for item in first_audit
    ]
    second_summary = [
        (item.start_node_id, item.end_node_id, item.decision, item.reasons)
        for item in second_audit
    ]
    assert first_summary == second_summary
    assert [item.decision for item in first_audit] == [
        "accepted",
        "rejected",
        "rejected",
    ]
    assert all(
        "duplicate_endpoint_pair" in item.reasons for item in first_audit[1:]
    )
