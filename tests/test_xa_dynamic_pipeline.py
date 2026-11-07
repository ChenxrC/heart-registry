from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pydicom
import pytest
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, SecondaryCaptureImageStorage, generate_uid


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TOOLS = PROJECT_ROOT / "tools"
sys.path.insert(0, str(TOOLS))

from calibrate_xa_run_to_ct import calibrate_series
import run_xa_temporal_fusion as temporal_module
from run_xa_temporal_fusion import temporal_fuse
from run_xa_vessel_segmentation import postprocess_probability
from xa_cine import read_cine, write_frames, write_mp4


def write_test_cine(path: Path, frames: np.ndarray, photometric: str = "MONOCHROME2") -> None:
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
    dataset.PhotometricInterpretation = photometric
    dataset.BitsAllocated = 8
    dataset.BitsStored = 8
    dataset.HighBit = 7
    dataset.PixelRepresentation = 0
    dataset.FrameTime = 125.0
    dataset.PixelData = np.ascontiguousarray(frames, dtype=np.uint8).tobytes()
    dataset.save_as(path, enforce_file_format=True)


def test_cine_reader_preserves_rows_columns_and_timing(tmp_path: Path) -> None:
    expected = np.asarray(
        [
            [[1, 2, 3], [4, 5, 6]],
            [[7, 8, 9], [10, 11, 12]],
        ],
        dtype=np.uint8,
    )
    path = tmp_path / "cine.dcm"
    write_test_cine(path, expected)
    _, actual, frame_times = read_cine(path)
    np.testing.assert_array_equal(actual, expected)
    assert frame_times == [125.0, 125.0]


def test_cine_reader_only_applies_photometric_inversion(tmp_path: Path) -> None:
    expected = np.asarray([[[0, 50], [100, 255]]], dtype=np.uint8)
    path = tmp_path / "mono1.dcm"
    write_test_cine(path, expected, "MONOCHROME1")
    _, actual, _ = read_cine(path)
    np.testing.assert_array_equal(actual, 255 - expected)


def test_probability_postprocess_keeps_large_curvilinear_component() -> None:
    probability = np.zeros((128, 128), dtype=np.float32)
    probability[25:105, 48:56] = 0.95
    probability[88:96, 48:112] = 0.85
    mask = postprocess_probability(probability, min_size=120)
    assert mask[50, 52]
    assert mask[92, 90]
    assert 800 < np.count_nonzero(mask) < 2000


def test_temporal_fusion_is_identity_for_static_sequence() -> None:
    frame = np.tile(np.arange(64, dtype=np.uint8), (64, 1))
    frames = np.stack([frame, frame, frame])
    probability = np.zeros((64, 64), dtype=np.float32)
    probability[10:50, 30:34] = 0.8
    probabilities = np.stack([probability, probability, probability])
    actual = temporal_fuse(frames, probabilities)
    np.testing.assert_allclose(actual, probabilities, atol=1e-5)


def test_series1_calibration_is_geometrically_consistent_and_not_promoted() -> None:
    dicom = PROJECT_ROOT / "1234567" / "XA35397" / "DB201234503" / "-0001-0001-0032-W256L128.DCM"
    report = PROJECT_ROOT / "artifacts" / "annotation_registration" / "first_pair_1234567" / "ct_dr_3d_viewer" / "calibration_report.json"
    scene = PROJECT_ROOT / "artifacts" / "annotation_registration" / "first_pair_1234567" / "ct_dr_3d_viewer" / "scene_geometry.json"
    if not all(path.is_file() for path in (dicom, report, scene)):
        pytest.skip("patient-specific calibration inputs are intentionally not shipped")
    result = calibrate_series(
        dicom,
        report,
        scene,
    )
    assert result["series_number"] == 1
    assert result["run_id"] == "series_01"
    assert result["series"] == "XA Series 1"
    assert result["quality_status"] == "diagnostic-only"
    assert result["quality_gate_passed"] is False
    consistency = result["geometry_consistency"]
    np.testing.assert_allclose(consistency["source_to_detector_distance_mm"], 1029.0, atol=1e-6)
    np.testing.assert_allclose(consistency["source_to_isocenter_distance_mm"], 810.0, atol=1e-6)
    assert consistency["basis_max_abs_dot"] < 1e-12
    assert consistency["isocenter_distance_to_ct_bbox_mm"] == 0.0


def test_calibration_uses_the_input_dicom_series_identity() -> None:
    dicom = PROJECT_ROOT / "1234567" / "XA35397" / "DB201234503" / "-0002-0002-0064-W256L128.DCM"
    report = PROJECT_ROOT / "artifacts" / "annotation_registration" / "first_pair_1234567" / "ct_dr_3d_viewer" / "calibration_report.json"
    scene = PROJECT_ROOT / "artifacts" / "annotation_registration" / "first_pair_1234567" / "ct_dr_3d_viewer" / "scene_geometry.json"
    if not all(path.is_file() for path in (dicom, report, scene)):
        pytest.skip("patient-specific calibration inputs are intentionally not shipped")
    result = calibrate_series(
        dicom,
        report,
        scene,
    )
    assert result["series_number"] == 2
    assert result["run_id"] == "series_02"
    assert result["series"] == "XA Series 2"
    assert "Series 2" in " ".join(result["quality_reasons"])


def test_built_dynamic_viewer_asset_contract() -> None:
    output = PROJECT_ROOT / "artifacts" / "xa_dynamic_vessel_1234567" / "viewer"
    if not (output / "scene_geometry.json").is_file():
        pytest.skip("patient-specific generated XA viewer assets are intentionally not shipped")
    scene = json.loads((output / "scene_geometry.json").read_text(encoding="utf-8"))
    dynamic = scene["dynamic_xa"]
    expected = dynamic["frame_count"] * dynamic["rows"] * dynamic["columns"]
    assert dynamic["quality_status"] == "diagnostic-only"
    assert dynamic["quality_gate_passed"] is False
    assert dynamic["frame_count"] == 31
    for asset in dynamic["assets"].values():
        path = output / asset["path"]
        assert path.stat().st_size == expected == asset["byte_length"]
    assert "unique 3D coronary depth" in dynamic["interpretation"]
    source = np.asarray(dynamic["source_lps_mm"])
    detector = np.asarray(dynamic["detector_center_lps_mm"])
    np.testing.assert_allclose(np.linalg.norm(detector - source), dynamic["sdd_mm"], atol=1e-6)


def test_temporal_report_v2_observes_topology_without_changing_raw_masks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    frame = np.tile(np.arange(64, dtype=np.uint8), (64, 1))
    frames = np.stack([frame, frame])
    dicom_path = tmp_path / "cine.dcm"
    write_test_cine(dicom_path, frames)

    segmentation_dir = tmp_path / "segmentation"
    segmentation_dir.mkdir()
    probabilities_u8 = np.zeros(frames.shape, dtype=np.uint8)
    (segmentation_dir / "probabilities_u8.raw").write_bytes(probabilities_u8.tobytes(order="C"))

    first_mask = np.zeros((64, 64), dtype=bool)
    first_mask[2, 2:7] = True
    second_mask = first_mask.copy()
    second_mask[10, 10] = True
    expected_masks = np.stack([first_mask, second_mask])
    mask_results = iter(expected_masks)
    monkeypatch.setattr(
        temporal_module,
        "postprocess_probability",
        lambda _probability: next(mask_results).copy(),
    )
    monkeypatch.setattr(temporal_module, "write_frames", write_frames)
    monkeypatch.setattr(temporal_module, "write_mp4", write_mp4)

    output_dir = tmp_path / "temporal"
    report = temporal_module.run_temporal(dicom_path, segmentation_dir, output_dir)

    expected_topology = [
        {
            "component_count": 1,
            "largest_component_fraction": 1.0,
            "endpoint_count": 2,
            "junction_count": 0,
            "skeleton_length_px": 5,
        },
        {
            "component_count": 2,
            "largest_component_fraction": 5.0 / 6.0,
            "endpoint_count": 2,
            "junction_count": 0,
            "skeleton_length_px": 6,
        },
    ]
    assert report["schema_version"] == 2
    topology_records = [
        {
            "frame_index": frame_record["frame_index"],
            "topology_before": frame_record["topology_before"],
            "topology_after": frame_record["topology_after"],
        }
        for frame_record in report["frames"]
    ]
    assert topology_records == [
        {
            "frame_index": index,
            "topology_before": item,
            "topology_after": item,
        }
        for index, item in enumerate(expected_topology)
    ]
    assert json.loads((output_dir / "temporal_report.json").read_text(encoding="utf-8")) == report

    expected_raw = (expected_masks.astype(np.uint8) * 255).tobytes(order="C")
    actual_raw = (output_dir / "temporal_masks_u8.raw").read_bytes()
    assert actual_raw == expected_raw
    assert hashlib.sha256(actual_raw).hexdigest() == hashlib.sha256(expected_raw).hexdigest()


def test_run_temporal_writes_deterministic_graph_json_without_rewriting_legacy_raw_assets(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    frame = np.tile(np.arange(64, dtype=np.uint8), (64, 1))
    frames = np.stack([frame, frame])
    dicom_path = tmp_path / "cine.dcm"
    write_test_cine(dicom_path, frames)

    segmentation_dir = tmp_path / "segmentation"
    segmentation_dir.mkdir()
    probabilities_u8 = np.zeros(frames.shape, dtype=np.uint8)
    (segmentation_dir / "probabilities_u8.raw").write_bytes(
        probabilities_u8.tobytes(order="C")
    )

    first_mask = np.zeros((64, 64), dtype=bool)
    first_mask[20, 20:31] = True
    second_mask = np.zeros((64, 64), dtype=bool)
    second_mask[20, 21:32] = True
    mask_results = iter((first_mask, second_mask))
    monkeypatch.setattr(
        temporal_module,
        "postprocess_probability",
        lambda _probability: next(mask_results).copy(),
    )
    monkeypatch.setattr(temporal_module, "write_frames", write_frames)
    monkeypatch.setattr(temporal_module, "write_mp4", write_mp4)
    monkeypatch.setattr(
        temporal_module,
        "dense_flow",
        lambda source, _target: np.zeros((*source.shape, 2), dtype=np.float32),
    )

    output_dir = tmp_path / "temporal"
    first_report = temporal_module.run_temporal(
        dicom_path, segmentation_dir, output_dir
    )
    legacy_names = (
        "temporal_probabilities_u8.raw",
        "temporal_support_u8.raw",
        "flow_confidence_u8.raw",
        "temporal_masks_u8.raw",
        "temporal_centerlines_u8.raw",
    )
    legacy_before = {
        name: (
            (output_dir / name).stat().st_size,
            hashlib.sha256((output_dir / name).read_bytes()).hexdigest(),
        )
        for name in legacy_names
    }
    graph_path = output_dir / "temporal_vessel_graphs.json"
    first_graph_bytes = graph_path.read_bytes()
    document = json.loads(first_graph_bytes.decode("utf-8"))

    assert document["schema_version"] == 1
    assert document["frame_count"] == 2
    assert document["flow_convention"] == {
        "order": ["dx", "dy"],
        "direction": "previous_to_current",
        "mapping": "(y, x) -> (y + dy, x + dx)",
        "sampling": "deterministic_bilinear",
    }
    assert [frame_record["frame_index"] for frame_record in document["frames"]] == [
        0,
        1,
    ]
    for frame_record in document["frames"]:
        local_to_topology = {
            node["node_id"]: node["topology_id"] for node in frame_record["nodes"]
        }
        assert frame_record["root_topology_id"] == local_to_topology[
            frame_record["root_node_id"]
        ]
        assert set(frame_record["qc"]) == {
            "matched_node_fraction",
            "matched_edge_fraction",
            "new_nodes",
            "retired_nodes",
            "topology_changed",
            "quality_status",
        }
        for edge in frame_record["edges"]:
            assert edge["source_topology_id"] == local_to_topology[
                edge["source_node_id"]
            ]
            assert edge["target_topology_id"] == local_to_topology[
                edge["target_node_id"]
            ]
            assert len(edge["points_yx"]) == 64
            assert len(edge["radius_px"]) == 64
    assert first_report["assets"]["temporal_vessel_graphs"] == {
        "path": "temporal_vessel_graphs.json",
        "schema_version": 1,
        "byte_length": len(first_graph_bytes),
    }

    second_report = temporal_module.run_temporal(
        dicom_path, segmentation_dir, output_dir
    )

    assert graph_path.read_bytes() == first_graph_bytes
    assert second_report["assets"]["temporal_vessel_graphs"] == first_report["assets"][
        "temporal_vessel_graphs"
    ]
    assert {
        name: (
            (output_dir / name).stat().st_size,
            hashlib.sha256((output_dir / name).read_bytes()).hexdigest(),
        )
        for name in legacy_names
    } == legacy_before
