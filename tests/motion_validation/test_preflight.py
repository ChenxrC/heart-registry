"""Hand-authored metadata fixtures; no source images or external services."""

from __future__ import annotations

import hashlib
import importlib
import json
from pathlib import Path

import pytest
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import CTImageStorage, ExplicitVRLittleEndian, XRayAngiographicImageStorage


def _dicom(path: Path, *, modality: str = "CT", uid: str = "1.2.3", **tags) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    meta = FileMetaDataset()
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    meta.MediaStorageSOPClassUID = CTImageStorage if modality == "CT" else XRayAngiographicImageStorage
    meta.MediaStorageSOPInstanceUID = f"{uid}.1"
    ds = FileDataset(str(path), {}, file_meta=meta, preamble=b"\0" * 128)
    ds.SOPClassUID = meta.MediaStorageSOPClassUID
    ds.SOPInstanceUID = meta.MediaStorageSOPInstanceUID
    ds.SeriesInstanceUID = uid
    ds.Modality = modality
    ds.SeriesNumber = 1
    ds.Rows = 2
    ds.Columns = 2
    ds.PatientName = "DO_NOT_EXPORT^PRIVATE"
    ds.PatientID = "DO_NOT_EXPORT_ID"
    for name, value in tags.items():
        setattr(ds, name, value)
    ds.save_as(path, enforce_file_format=True)
    return path


def _json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _audit(case: Path, assets: Path) -> dict:
    return importlib.import_module("ctdr_pipeline.motion_validation.preflight").audit_case(case, assets)


def _roots(tmp_path: Path) -> tuple[Path, Path]:
    case, assets = tmp_path / "1234567", tmp_path / "assets"
    case.mkdir()
    assets.mkdir()
    return case, assets


def _calibrations(assets: Path, first: object = True, second: object = True) -> None:
    for run_id, flag in (("series_01", first), ("series_02", second)):
        _json(assets / "all_series" / run_id / f"{run_id}_calibration.json", {
            "quality_gate_passed": flag, "quality_status": "accepted" if flag is True else "diagnostic-only",
        })


def test_ct_spacing_uses_iop_normal_not_slice_thickness_or_filename(tmp_path: Path) -> None:
    # Break caught: sorting by filename, z coordinate, or SliceThickness instead of dot(IPP, IOP normal).
    case, assets = _roots(tmp_path)
    for name, x in (("a.dcm", 12), ("c.dcm", 10), ("b.dcm", 11)):
        _dicom(case / name, ImageOrientationPatient=[0, 1, 0, 0, 0, 1],
               ImagePositionPatient=[x, 20, 40], SliceThickness=2, ConvolutionKernel="B30f")
    ct = _audit(case, assets)["ct_series"][0]
    assert ct["series_instance_uid"] == "1.2.3"
    assert ct["unique_position_count"] == 3
    assert ct["spacing_mm"] == {"median": 1.0, "min": 1.0, "max": 1.0}
    assert [item["relative_path"] for item in ct["instances"]] == ["c.dcm", "b.dcm", "a.dcm"]
    assert ct["slice_thickness_mm"] == [2.0]
    assert ct["dynamic_ct_confirmed"] is False


def test_duplicate_ct_positions_are_not_zero_spacing_or_time_phases(tmp_path: Path) -> None:
    # Break caught: duplicate reconstructions counted as independent slices or phases.
    case, assets = _roots(tmp_path)
    for i, (z, kernel) in enumerate(((2, "B30f"), (0, "B30f"), (0, "B70f"), (1, "B30f"))):
        _dicom(case / f"{i}.DCM", ImageOrientationPatient=[1, 0, 0, 0, 1, 0],
               ImagePositionPatient=[0, 0, z], SliceThickness=2, ConvolutionKernel=kernel)
    ct = _audit(case, assets)["ct_series"][0]
    assert ct["unique_position_count"] == 3
    assert ct["duplicate_position_count"] == 1
    assert ct["duplicate_positions"][0]["count"] == 2
    assert ct["spacing_mm"] == {"median": 1.0, "min": 1.0, "max": 1.0}
    assert ct["convolution_kernels"] == ["B30f", "B70f"]
    assert ct["dynamic_ct_confirmed"] is False


def test_ct_series_are_grouped_by_uid_not_reconstruction_or_series_number(tmp_path: Path) -> None:
    case, assets = _roots(tmp_path)
    for i, uid in enumerate(("1.2.3", "1.2.4")):
        _dicom(case / f"{i}.dcm", uid=uid, ConvolutionKernel="B30f")
    assert len(_audit(case, assets)["ct_series"]) == 2


def test_xa_frame_span_is_n_minus_one_intervals_and_geometry_is_explicit(tmp_path: Path) -> None:
    # Break caught: old N*FrameTime duration, missing ECG/geometry interpreted as available.
    case, assets = _roots(tmp_path)
    _dicom(case / "xa.DCM", modality="XA", NumberOfFrames=31, FrameTime=133.333,
           PositionerPrimaryAngle=37.9, PositionerSecondaryAngle=-0.16,
           DistanceSourceToDetector=1029, DistanceSourceToPatient=810)
    xa = _audit(case, assets)["xa_series"][0]["instances"][0]
    assert xa["frame_count"] == 31
    assert xa["duration_ms"] == pytest.approx(3999.99)
    assert xa["frame_timestamps_ms"][0] == 0
    assert xa["frame_timestamps_ms"][-1] == pytest.approx(3999.99)
    assert xa["primary_angle_deg"] == 37.9
    assert xa["source_to_detector_mm"] == 1029
    assert "FrameOfReferenceUID" in xa["missing_geometry"]
    assert xa["has_ecg"] is False
    assert xa["phase_inferred_from_name"] is False


def test_frame_time_vector_uses_increment_semantics_and_precedes_scalar(tmp_path: Path) -> None:
    case, assets = _roots(tmp_path)
    _dicom(case / "xa.dcm", modality="XA", NumberOfFrames=4,
           FrameTime=999, FrameTimeVector=[0, 100, 120, 80], TriggerTime=20)
    xa = _audit(case, assets)["xa_series"][0]["instances"][0]
    assert xa["frame_timestamps_ms"] == [0.0, 100.0, 220.0, 300.0]
    assert xa["duration_ms"] == 300.0
    assert xa["timing_source"] == "FrameTimeVector"
    assert xa["ecg_tags"]["TriggerTime"] == 20.0


@pytest.mark.parametrize("vector", [[0, 100], [1, 100, 100], [0, -1, 100]])
def test_invalid_frame_time_vector_is_reported_without_scalar_fallback(tmp_path: Path, vector: list) -> None:
    case, assets = _roots(tmp_path)
    _dicom(case / "xa.dcm", modality="XA", NumberOfFrames=3, FrameTime=100, FrameTimeVector=vector)
    xa = _audit(case, assets)["xa_series"][0]["instances"][0]
    assert xa["timing_status"] == "insufficient"
    assert xa["duration_ms"] is None
    assert any(issue["field"] == "FrameTimeVector" for issue in xa["metadata_issues"])


@pytest.mark.parametrize("flag", [False, None, "true", 1])
def test_missing_failed_or_nonboolean_calibration_cannot_pass_gate(tmp_path: Path, flag: object) -> None:
    case, assets = _roots(tmp_path)
    _calibrations(assets, True, flag)
    report = _audit(case, assets)
    assert report["gates"]["absolute_geometry_accepted"] is False
    assert all(type(value) is bool for value in report["gates"].values())
    assert report["gates"]["clinical_validation"] is False
    assert report["gates"]["true_3d_reference_available"] is False
    assert report["patient_true_3d_tre_mm"] is None


def test_both_explicitly_accepted_calibrations_do_not_accept_insufficient_phase(tmp_path: Path) -> None:
    case, assets = _roots(tmp_path)
    _calibrations(assets)
    _json(assets / "all_series/cardiac_phase/phase_alignment.json", {
        "status": "insufficient", "quality_gate_passed": True,
        "runs": {"series_01": {"status": "accepted"}, "series_02": {"status": "accepted"}},
    })
    report = _audit(case, assets)
    assert report["gates"]["absolute_geometry_accepted"] is True
    assert report["gates"]["phase_sync_accepted"] is False


def test_assets_are_exact_allowlist_hashed_and_source_bytes_unchanged(tmp_path: Path) -> None:
    # Break caught: recursive asset discovery imports candidate/history; audit mutates inputs or leaks PHI.
    case, assets = _roots(tmp_path)
    path = _dicom(case / "ct.dcm")
    _json(assets / "all_series/series_01/series_manifest.json", {"quality_status": "diagnostic-only", "PatientName": "SECRET"})
    raw = assets / "all_series/series_01/cine_u8.raw"
    raw.write_bytes(b"abc")
    _json(assets / "all_series_candidate/series_02/series_02_calibration.json", {"quality_gate_passed": True})
    _json(assets / "history/all_series/series_02/series_02_calibration.json", {"quality_gate_passed": True})
    before = {p: p.read_bytes() for root in (case, assets) for p in root.rglob("*") if p.is_file()}
    report = _audit(case, assets)
    assert report["case_id"] == "1234567"
    source = report["ct_series"][0]["instances"][0]
    assert source["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    record = report["assets"]["all_series/series_01/cine_u8.raw"]
    assert record["sha256"] == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    assert record["byte_length"] == 3
    assert record["exists"] is True
    assert report["assets"]["all_series/series_02/series_02_calibration.json"]["exists"] is False
    assert not any("candidate" in key or "history" in key for key in report["assets"])
    serialized = json.dumps(report, allow_nan=False)
    assert "DO_NOT_EXPORT" not in serialized
    assert "SECRET" not in serialized
    assert {p: p.read_bytes() for root in (case, assets) for p in root.rglob("*") if p.is_file()} == before


def test_corrupt_dicom_and_assets_remain_visible_as_read_errors(tmp_path: Path) -> None:
    case, assets = _roots(tmp_path)
    (case / "bad.dcm").write_bytes(b"not a DICOM")
    bad_json = assets / "all_series/series_01/series_01_calibration.json"
    bad_json.parent.mkdir(parents=True)
    bad_json.write_bytes(b"{broken")
    report = _audit(case, assets)
    assert {item["relative_path"] for item in report["read_errors"]} >= {
        "bad.dcm", "all_series/series_01/series_01_calibration.json",
    }
    assert report["source_files"][0]["sha256"] == hashlib.sha256(b"not a DICOM").hexdigest()
    assert report["gates"]["absolute_geometry_accepted"] is False
    json.dumps(report, allow_nan=False)


def test_invalid_and_nonfinite_metadata_is_null_and_not_silently_dropped(tmp_path: Path) -> None:
    case, assets = _roots(tmp_path)
    path = _dicom(case / "invalid.dcm", modality="XA", NumberOfFrames=3)
    import pydicom
    ds = pydicom.dcmread(path)
    ds.add_new(0x00181063, "LO", "NaN")  # Invalid numeric VR/value, serialized without fixture warnings.
    ds.add_new(0x00181510, "LO", "not-a-number")
    ds.save_as(path, enforce_file_format=True)
    _json(assets / "all_series/series_01/series_01_calibration.json", {"quality_gate_passed": True, "metric": float("nan")})
    report = _audit(case, assets)
    xa = report["xa_series"][0]["instances"][0]
    assert xa["duration_ms"] is None
    assert xa["primary_angle_deg"] is None
    assert {item["field"] for item in xa["metadata_issues"]} >= {"FrameTime", "PositionerPrimaryAngle"}
    assert report["read_errors"]
    json.dumps(report, allow_nan=False)


def test_missing_case_is_reported_not_an_empty_success(tmp_path: Path) -> None:
    report = _audit(tmp_path / "absent", tmp_path / "missing_assets")
    assert report["status"] == "insufficient"
    assert report["read_errors"]
    assert report["limitations"]
    assert not any(report["gates"].values())


def _manifest_for_source(assets: Path, source: Path, *, uid: str = "1.2.3", **overrides) -> None:
    record = {"sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
              "sop_instance_uid": f"{uid}.1", "series_instance_uid": uid}
    record.update(overrides)
    _json(assets / "all_series/series_01/series_manifest.json", {
        "source_dicom": "/old-server/unavailable/another-case.dcm", "source_record": record,
        "shape_frame_row_column": [3, 2, 2], "dtype": "uint8",
    })


def test_source_link_matches_actual_identity_and_hash_not_stale_absolute_path(tmp_path: Path) -> None:
    # Break caught: matching on old absolute source_dicom instead of actual SOP/series identity.
    case, assets = _roots(tmp_path)
    source = _dicom(case / "relocated/renamed.DCM", modality="XA", NumberOfFrames=3, FrameTime=100)
    _manifest_for_source(assets, source)
    link = _audit(case, assets)["source_links"]["series_01"]
    assert link["status"] == "verified"
    assert link["source_relative_path"] == "relocated/renamed.DCM"
    assert link["checks"] == {"sop_instance_uid": True, "series_instance_uid": True, "sha256": True}


@pytest.mark.parametrize("overrides,failed_field", [
    ({"sha256": "0" * 64}, "sha256"),
    ({"series_instance_uid": "1.2.999"}, "series_instance_uid"),
    ({"sop_instance_uid": "1.2.999.1"}, "sop_instance_uid"),
])
def test_mismatched_cache_source_is_explicit_and_does_not_change_geometry_gate(
    tmp_path: Path, overrides: dict, failed_field: str,
) -> None:
    case, assets = _roots(tmp_path)
    source = _dicom(case / "xa.dcm", modality="XA", NumberOfFrames=3, FrameTime=100)
    _manifest_for_source(assets, source, **overrides)
    _calibrations(assets)
    report = _audit(case, assets)
    link = report["source_links"]["series_01"]
    assert link["status"] == "mismatch"
    assert link["checks"][failed_field] is False
    assert report["assets"]["all_series/series_01/series_manifest.json"]["quality_summary"]["source_link_status"] == "mismatch"
    assert report["gates"]["absolute_geometry_accepted"] is True


def test_source_link_missing_manifest_identity_and_source_are_explicit(tmp_path: Path) -> None:
    case, assets = _roots(tmp_path)
    assert _audit(case, assets)["source_links"]["series_01"]["status"] == "missing_manifest"
    source = _dicom(case / "xa.dcm", modality="XA")
    _manifest_for_source(assets, source, sop_instance_uid=None)
    assert _audit(case, assets)["source_links"]["series_01"]["status"] == "missing_identity"
    _manifest_for_source(assets, source, sop_instance_uid="9.8.7.1", series_instance_uid="9.8.7")
    assert _audit(case, assets)["source_links"]["series_01"]["status"] == "missing_source"


def test_source_link_does_not_use_ct_or_ambiguous_sop_as_xa_source(tmp_path: Path) -> None:
    case, assets = _roots(tmp_path)
    source = _dicom(case / "ct.dcm")
    _manifest_for_source(assets, source)
    assert _audit(case, assets)["source_links"]["series_01"]["status"] == "mismatch"
    source = _dicom(case / "one.dcm", modality="XA")
    _manifest_for_source(assets, source)
    (case / "two.dcm").write_bytes(source.read_bytes())
    assert _audit(case, assets)["source_links"]["series_01"]["status"] == "ambiguous_source"


def test_existing_empty_roots_return_insufficient_false_gates(tmp_path: Path) -> None:
    case, assets = _roots(tmp_path)
    report = _audit(case, assets)
    assert report["status"] == "insufficient"
    assert not any(report["gates"].values())
    assert report["source_files"] == []


def test_invalid_iop_is_not_used_to_fabricate_spacing(tmp_path: Path) -> None:
    case, assets = _roots(tmp_path)
    for i in range(2):
        _dicom(case / f"{i}.dcm", ImagePositionPatient=[0, 0, i],
               ImageOrientationPatient=[1, 0, 0, 1, 0, 0])
    ct = _audit(case, assets)["ct_series"][0]
    assert ct["spacing_mm"] == {"median": None, "min": None, "max": None}
    assert ct["geometry_complete"] is False
    assert all(r["metadata_issues"] for r in ct["instances"])


def test_missing_second_calibration_and_candidate_only_phase_remain_false(tmp_path: Path) -> None:
    case, assets = _roots(tmp_path)
    _json(assets / "all_series/series_01/series_01_calibration.json", {"quality_gate_passed": True})
    _json(assets / "all_series_candidate/cardiac_phase/phase_alignment.json", {
        "status": "accepted", "runs": {"series_01": {"status": "accepted"}, "series_02": {"status": "accepted"}},
    })
    report = _audit(case, assets)
    assert report["gates"]["absolute_geometry_accepted"] is False
    assert report["gates"]["phase_sync_accepted"] is False


def test_malformed_phase_group_is_reported_without_crash(tmp_path: Path) -> None:
    case, assets = _roots(tmp_path)
    _json(assets / "all_series/cardiac_phase/phase_alignment.json", {
        "status": "accepted", "groups": {"rca": {"run_ids": None, "status": "accepted"}},
        "runs": {"series_01": {"status": "accepted"}, "series_02": {"status": "accepted"}},
    })
    report = _audit(case, assets)
    assert report["gates"]["phase_sync_accepted"] is False
    assert report["assets"]["all_series/cardiac_phase/phase_alignment.json"]["quality_summary"]["metadata_issues"]


def test_contradictory_calibration_rejection_cannot_be_overridden(tmp_path: Path) -> None:
    case, assets = _roots(tmp_path)
    _calibrations(assets)
    _json(assets / "all_series/series_01/series_01_calibration.json", {
        "quality_gate_passed": True, "quality_status": "accepted", "status": "failed",
    })
    assert _audit(case, assets)["gates"]["absolute_geometry_accepted"] is False


def test_single_frame_vector_has_zero_span(tmp_path: Path) -> None:
    # Break caught: pydicom represents a one-value DS vector as a scalar.
    case, assets = _roots(tmp_path)
    _dicom(case / "xa.dcm", modality="XA", NumberOfFrames=1, FrameTimeVector=[0])
    xa = _audit(case, assets)["xa_series"][0]["instances"][0]
    assert xa["frame_timestamps_ms"] == [0.0]
    assert xa["duration_ms"] == 0.0


def test_overflowing_ct_spacing_is_reported_and_stays_json_safe(tmp_path: Path) -> None:
    case, assets = _roots(tmp_path)
    for i, z in enumerate((-1e308, 1e308)):
        _dicom(case / f"{i}.dcm", ImagePositionPatient=[0, 0, z],
               ImageOrientationPatient=[1, 0, 0, 0, 1, 0])
    report = _audit(case, assets)
    assert report["ct_series"][0]["spacing_mm"] == {"median": None, "min": None, "max": None}
    assert any(issue["field"] == "projected_spacing_mm" for issue in report["metadata_issues"])
    json.dumps(report, allow_nan=False)


def test_corrupt_quality_fields_cannot_copy_nested_phi_into_summary(tmp_path: Path) -> None:
    case, assets = _roots(tmp_path)
    _json(assets / "all_series/series_01/series_01_calibration.json", {
        "quality_gate_passed": True, "quality_status": {"PatientName": "SECRET_PATIENT"},
    })
    report = _audit(case, assets)
    summary = report["assets"]["all_series/series_01/series_01_calibration.json"]["quality_summary"]
    assert summary["metadata_issues"]
    assert "SECRET_PATIENT" not in json.dumps(report, allow_nan=False)
    assert report["gates"]["absolute_geometry_accepted"] is False


def test_missing_hash_and_accepted_phase_have_distinct_evidence(tmp_path: Path) -> None:
    case, assets = _roots(tmp_path)
    source = _dicom(case / "xa.dcm", modality="XA")
    _manifest_for_source(assets, source, sha256=None)
    _json(assets / "all_series/cardiac_phase/phase_alignment.json", {
        "status": "accepted", "quality_gate_passed": True,
        "groups": {"rca": {"run_ids": ["series_01", "series_02"], "status": "accepted"}},
        "runs": {"series_01": {"status": "accepted"}, "series_02": {"status": "accepted"}},
    })
    report = _audit(case, assets)
    assert report["source_links"]["series_01"]["status"] == "missing_hash"
    assert report["gates"]["phase_sync_accepted"] is True
    assert report["gates"]["clinical_validation"] is False
