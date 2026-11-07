from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import numpy as np

from ctdr_pipeline.motion_validation.runner import (
    _heart_shape_points,
    _read_cine,
    _snapshot_inputs,
    _summary,
    run_validation,
    validate_output_paths,
)


def test_output_inside_or_above_source_is_rejected_before_writing(tmp_path: Path) -> None:
    case = tmp_path / "case"
    assets = tmp_path / "assets"
    case.mkdir()
    assets.mkdir()
    for output in (case / "new", assets / "new", case, tmp_path):
        with pytest.raises((ValueError, FileExistsError)):
            validate_output_paths(case, assets, output)
    assert not (case / "new").exists()
    assert not (assets / "new").exists()


def test_existing_output_even_empty_is_never_reused(tmp_path: Path) -> None:
    case = tmp_path / "case"
    assets = tmp_path / "assets"
    output = tmp_path / "run"
    for directory in (case, assets, output):
        directory.mkdir()
    with pytest.raises(FileExistsError):
        validate_output_paths(case, assets, output)
    assert list(output.iterdir()) == []


def test_missing_input_is_rejected_without_creating_output(tmp_path: Path) -> None:
    assets = tmp_path / "assets"
    assets.mkdir()
    output = tmp_path / "outputs" / "run"
    with pytest.raises(FileNotFoundError):
        run_validation(tmp_path / "missing_case", assets, output)
    assert not output.parent.exists()


def test_incomplete_case_records_insufficiency_without_promoting_gates(tmp_path: Path) -> None:
    case = tmp_path / "case"
    assets = tmp_path / "assets"
    case.mkdir()
    assets.mkdir()
    marker = case / "keep.bin"
    marker.write_bytes(b"immutable input")
    before = hashlib.sha256(marker.read_bytes()).hexdigest()
    output = tmp_path / "run"

    report = run_validation(case, assets, output)

    assert report["patient_3d_optimization_enabled"] is False
    assert report["patient_true_3d_tre_mm"] is None
    assert report["source_integrity"]["unchanged"] is True
    assert hashlib.sha256(marker.read_bytes()).hexdigest() == before
    preflight = json.loads((output / "preflight.json").read_text(encoding="utf-8"))
    assert preflight["gates"]["absolute_geometry_accepted"] is False
    assert preflight["gates"]["clinical_validation"] is False
    for run_id in ("series_01", "series_02"):
        observation = json.loads((output / "observations" / f"{run_id}.json").read_text(encoding="utf-8"))
        assert observation["status"] == "insufficient"
    review = json.loads((output / "review_template.json").read_text(encoding="utf-8"))
    assert review["reviewed"] is False
    assert (output / "synthetic_benchmark.json").is_file()
    assert (output / "code_manifest.json").is_file()
    assert (output / "summary.md").is_file()
    json.dumps(report, allow_nan=False)


def test_cli_help_describes_research_only_scope() -> None:
    import subprocess
    import sys

    project_root = Path(__file__).resolve().parents[2]
    result = subprocess.run(
        [sys.executable, str(project_root / "run_heart_motion_validation.py"), "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    assert "--output-dir" in result.stdout
    assert "research" in result.stdout.lower()


def test_real_cine_assets_produce_unreviewed_overlays_without_changing_inputs(tmp_path: Path) -> None:
    case = tmp_path / "case"
    assets = tmp_path / "assets"
    case.mkdir()
    assets.mkdir()
    texture = np.random.default_rng(147).integers(20, 210, size=(32, 32), dtype=np.uint8)
    frame = np.full((64, 64), 110, dtype=np.uint8)
    frame[16:48, 16:48] = texture
    cine = np.repeat(frame[None], 31, axis=0)
    roi = np.zeros_like(cine)
    roi[:, 16:48, 16:48] = 255
    for run_id in ("series_01", "series_02"):
        series = assets / "all_series" / run_id
        (series / "temporal").mkdir(parents=True)
        (series / "series_manifest.json").write_text(
            json.dumps({
                "shape_frame_row_column": [31, 64, 64],
                "frame_times_ms": [133.333] * 31,
                "median_frame_time_ms": 133.333,
                "source_record": {"rows": 64, "columns": 64, "frame_count": 31, "frame_time_ms": 133.333},
            }), encoding="utf-8",
        )
        (series / "cine_u8.raw").write_bytes(cine.tobytes())
        (series / "temporal" / "temporal_centerlines_u8.raw").write_bytes(roi.tobytes())
    source = assets / "all_series" / "series_01" / "cine_u8.raw"
    before = hashlib.sha256(source.read_bytes()).hexdigest()
    output = tmp_path / "new_run"

    report = run_validation(case, assets, output)

    assert report["source_integrity"]["unchanged"] is True
    assert hashlib.sha256(source.read_bytes()).hexdigest() == before
    for run_id in ("series_01", "series_02"):
        observation = json.loads((output / "observations" / f"{run_id}.json").read_text(encoding="utf-8"))
        assert observation["status"] == "provisional_unreviewed"
        assert len(observation["tracks"]) > 0
        assert all(track["semantic_status"] == "unreviewed_image_feature" for track in observation["tracks"])
        from PIL import Image

        with Image.open(output / "observations" / f"{run_id}_review.png") as image:
            assert image.width > 0 and image.height > 0
    assert report["gates"]["absolute_geometry_accepted"] is False


def test_heart_shape_sampling_preserves_physical_geometry_and_mask(tmp_path: Path) -> None:
    import SimpleITK as sitk

    voxels = np.zeros((8, 8, 8), dtype=np.uint8)
    voxels[2:6, 2:6, 2:6] = 1
    image = sitk.GetImageFromArray(voxels)
    image.SetOrigin((10.0, 20.0, 30.0))
    image.SetSpacing((1.0, 2.0, 3.0))
    image.SetDirection((-1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 1.0))
    path = tmp_path / "heart.nii.gz"
    sitk.WriteImage(image, str(path))
    before = hashlib.sha256(path.read_bytes()).hexdigest()

    points, provenance = _heart_shape_points(path)

    assert points is not None
    np.testing.assert_allclose(points[0], [8.0, 16.0, 36.0])
    np.testing.assert_allclose(points.min(axis=0), [5.0, 10.0, 36.0])
    np.testing.assert_allclose(points.max(axis=0), [8.0, 16.0, 45.0])
    assert provenance["kind"] == "CT_heart_surface_shape_only"
    assert hashlib.sha256(path.read_bytes()).hexdigest() == before


def test_summary_keeps_support_and_heldout_errors_separate() -> None:
    synthetic = {
        "cases": {
            "translation": {
                "models": {
                    "rigid_xy": {"metrics": {
                        "support": {"rmse_2d_px": 0.125},
                        "heldout": {"rmse_2d_px": 3.5, "synthetic_rmse_3d_mm": 4.25},
                    }},
                },
            },
        },
        "depth_ambiguity": {"max_projection_difference_px": 0.0, "max_3d_difference_mm": 100.0},
    }
    summary = _summary(
        {"gates": {"absolute_geometry_accepted": False}},
        [],
        {"source_integrity": {"unchanged": True}},
        synthetic,
    )
    assert "| translation | rigid_xy | 0.1250 | 3.5000 | 4.2500 |" in summary
    assert "100.0000" in summary


def _artificial_assets(root: Path, count: int = 31) -> Path:
    """Real cached pixels/ROI; no patient data or fitted-motion dependencies."""
    root.mkdir(parents=True, exist_ok=True)
    frame = np.full((64, 64), 110, dtype=np.uint8)
    frame[16:48, 16:48] = np.random.default_rng(147).integers(20, 210, (32, 32), dtype=np.uint8)
    cine = np.repeat(frame[None], count, axis=0)
    roi = np.zeros_like(cine)
    roi[:, 16:48, 16:48] = 255
    for run_id in ("series_01", "series_02"):
        series = root / "all_series" / run_id
        (series / "temporal").mkdir(parents=True)
        (series / "series_manifest.json").write_text(json.dumps({
            "shape_frame_row_column": list(cine.shape), "dtype": "uint8",
            "frame_times_ms": [125] * count,
            "source_record": {"frame_time_ms": 125},
        }), encoding="utf-8")
        (series / "cine_u8.raw").write_bytes(cine.tobytes())
        (series / "temporal/temporal_centerlines_u8.raw").write_bytes(roi.tobytes())
    return root


def _artificial_mask(path: Path) -> None:
    import SimpleITK as sitk

    path.parent.mkdir(parents=True, exist_ok=True)
    voxels = np.zeros((8, 8, 8), np.uint8)
    voxels[2:6, 2:6, 2:6] = 1
    sitk.WriteImage(sitk.GetImageFromArray(voxels), str(path))


def _symlink(link: Path, target: Path, *, directory: bool = False) -> None:
    try:
        link.symlink_to(target, target_is_directory=directory)
    except OSError as error:
        pytest.skip(f"Real symlink creation unavailable: {error}")


def _record_forbidden_reads(monkeypatch, *forbidden: Path) -> list[str]:
    """Observe real IO at Python, NumPy and image-reader boundaries."""
    import SimpleITK as sitk

    attempts = []
    roots = [path.absolute() for path in forbidden]
    resolved = [path.resolve() for path in forbidden]

    def record(path):
        if isinstance(path, (str, Path)):
            path = Path(path)
            if any(path.absolute().is_relative_to(root) for root in roots) or any(
                path.resolve().is_relative_to(root) for root in resolved
            ):
                attempts.append(str(path))

    original_open, original_fromfile, original_image = Path.open, np.fromfile, sitk.ReadImage

    def path_open(path, *args, **kwargs):
        record(path)
        return original_open(path, *args, **kwargs)

    def fromfile(path, *args, **kwargs):
        record(path)
        return original_fromfile(path, *args, **kwargs)

    def read_image(path, *args, **kwargs):
        record(path)
        return original_image(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", path_open)
    monkeypatch.setattr(np, "fromfile", fromfile)
    monkeypatch.setattr(sitk, "ReadImage", read_image)
    return attempts


@pytest.mark.parametrize("root_kind", ["case", "assets"])
def test_candidate_input_root_is_rejected_without_reading_or_writing(tmp_path: Path, monkeypatch, root_kind: str) -> None:
    # Break caught: snapshot/audit dereferences a candidate root before validation.
    case = tmp_path / ("case_candidate" if root_kind == "case" else "case")
    case.mkdir()
    (case / "marker.bin").write_bytes(b"artificial")
    assets = _artificial_assets(tmp_path / ("assets_candidate" if root_kind == "assets" else "assets"))
    output = tmp_path / "new_run"
    attempts = _record_forbidden_reads(monkeypatch, case if root_kind == "case" else assets)
    with pytest.raises(ValueError, match="[Ee]xcluded|authoritative"):
        run_validation(case, assets, output)
    assert attempts == []
    assert not output.exists()


def test_snapshot_prunes_nested_candidate_case_data_before_hashing(tmp_path: Path, monkeypatch) -> None:
    # Break caught: recursive snapshot includes a case subtree excluded by preflight.
    case, assets = tmp_path / "case", tmp_path / "assets"
    case.mkdir()
    assets.mkdir()
    good = case / "source.bin"
    good.write_bytes(b"allowed")
    excluded = case / "nested" / "derived_CANDIDATE"
    excluded.mkdir(parents=True)
    (excluded / "do_not_read.bin").write_bytes(b"not authoritative")
    attempts = _record_forbidden_reads(monkeypatch, excluded)
    snapshot = _snapshot_inputs(case, assets)
    assert attempts == []
    assert snapshot == {str(good): hashlib.sha256(b"allowed").hexdigest()}


@pytest.mark.parametrize("consumer", ["observation", "render", "mask"])
def test_candidate_consumer_paths_are_never_opened(tmp_path: Path, monkeypatch, consumer: str) -> None:
    # Break caught: standalone readers bypass the runner's input boundary.
    from ctdr_pipeline.motion_validation.observations import build_observation_report

    assets = _artificial_assets(tmp_path / "assets_candidate")
    mask = assets / "thorax_masks/heart.nii.gz"
    _artificial_mask(mask)
    attempts = _record_forbidden_reads(monkeypatch, assets)
    if consumer == "mask":
        points, provenance = _heart_shape_points(mask)
        assert points is None
        assert provenance["kind"] == "digital_phantom"
        assert "excluded" in provenance["reason"].lower()
    else:
        with pytest.raises(ValueError, match="[Ee]xcluded|authoritative"):
            (build_observation_report if consumer == "observation" else _read_cine)(assets / "all_series/series_01")
    assert attempts == []


@pytest.mark.parametrize("root_kind", ["case", "assets"])
def test_linked_input_root_is_rejected_before_resolution(tmp_path: Path, monkeypatch, root_kind: str) -> None:
    case, assets = tmp_path / "case", _artificial_assets(tmp_path / "assets")
    case.mkdir()
    (case / "source.bin").write_bytes(b"allowed only directly")
    target = case if root_kind == "case" else assets
    link = tmp_path / "linked_input"
    _symlink(link, target, directory=True)
    attempts = _record_forbidden_reads(monkeypatch, target)
    with pytest.raises(ValueError, match="[Ee]xcluded|authoritative"):
        run_validation(link if root_kind == "case" else case, link if root_kind == "assets" else assets, tmp_path / "run")
    assert attempts == []
    assert not (tmp_path / "run").exists()


@pytest.mark.parametrize("relative", [
    "all_series/series_01/series_manifest.json", "all_series/series_01/cine_u8.raw",
    "all_series/series_01/temporal/temporal_centerlines_u8.raw", "thorax_masks/heart.nii.gz",
])
def test_linked_asset_is_never_hashed_tracked_sampled_or_rendered(tmp_path: Path, monkeypatch, relative: str) -> None:
    # Break caught: consumers ignore an excluded audit asset, including optional masks.
    case, assets = tmp_path / "case", _artificial_assets(tmp_path / "assets")
    case.mkdir()
    _artificial_mask(assets / "thorax_masks/heart.nii.gz")
    path = assets / relative
    target = tmp_path / "history_candidate" / path.name
    target.parent.mkdir()
    path.rename(target)
    _symlink(path, target)
    attempts = _record_forbidden_reads(monkeypatch, path, target)
    output = tmp_path / "run"
    report = run_validation(case, assets, output)
    assert attempts == []
    assert str(path) not in report["source_sha256"]
    if relative.endswith(".nii.gz"):
        result = json.loads((output / "synthetic_benchmark.json").read_text())
        assert result["static_shape_provenance"]["kind"] == "digital_phantom"
        assert "excluded" in result["static_shape_provenance"]["reason"].lower()
    else:
        observation = json.loads((output / "observations/series_01.json").read_text())
        assert observation["status"] == "insufficient"
        assert not (output / "observations/series_01_review.png").exists()


def test_snapshot_skips_nested_link_file_and_directory(tmp_path: Path, monkeypatch) -> None:
    case, assets, history = tmp_path / "case", tmp_path / "assets", tmp_path / "history"
    for root in (case, assets, history):
        root.mkdir()
    target = history / "do_not_read.bin"
    target.write_bytes(b"history")
    _symlink(case / "linked.bin", target)
    _symlink(case / "linked_directory", history, directory=True)
    attempts = _record_forbidden_reads(monkeypatch, history)
    assert _snapshot_inputs(case, assets) == {}
    assert attempts == []


def test_dangling_output_link_is_not_reused(tmp_path: Path) -> None:
    case, assets = tmp_path / "case", tmp_path / "assets"
    case.mkdir()
    assets.mkdir()
    output = tmp_path / "run"
    target = tmp_path / "not_created"
    _symlink(output, target, directory=True)
    with pytest.raises(FileExistsError):
        validate_output_paths(case, assets, output)
    assert not target.exists()


def test_runner_honors_excluded_audit_records_on_otherwise_eligible_paths(tmp_path: Path, monkeypatch) -> None:
    # Break caught: boundary checks alone pass, but the consumer ignores audit authority.
    import ctdr_pipeline.motion_validation.preflight as preflight_module

    case, assets = tmp_path / "case", _artificial_assets(tmp_path / "assets")
    case.mkdir()
    mask = assets / "thorax_masks/heart.nii.gz"
    _artificial_mask(mask)
    audit = preflight_module.audit_case(case, assets)
    manifest = assets / "all_series/series_01/series_manifest.json"
    for path in (manifest, mask):
        audit["assets"][path.relative_to(assets).as_posix()]["read_status"] = "excluded"
    monkeypatch.setattr(preflight_module, "audit_case", lambda *args: audit)
    attempts = _record_forbidden_reads(monkeypatch, manifest, mask)
    output = tmp_path / "run"
    report = run_validation(case, assets, output)
    assert attempts == []
    assert str(manifest) not in report["source_sha256"]
    assert str(mask) not in report["source_sha256"]
    observation = json.loads((output / "observations/series_01.json").read_text())
    assert observation["status"] == "insufficient"
    result = json.loads((output / "synthetic_benchmark.json").read_text())
    assert result["static_shape_provenance"]["kind"] == "digital_phantom"


def _exporter_timing_fixture(tmp_path: Path, *, count: int, timing_kind: str, identity: str = "verified", cache_count: int | None = None):
    """Actual exporter metadata helpers plus tiny hand-authored DICOM headers."""
    from pydicom.dataset import FileDataset, FileMetaDataset
    from pydicom.uid import ExplicitVRLittleEndian, XRayAngiographicImageStorage
    from tools.xa_cine import audit_record, frame_times_ms

    case = tmp_path / "case"
    case.mkdir()
    assets = _artificial_assets(tmp_path / "assets", count if cache_count is None else cache_count)
    for index, run_id in enumerate(("series_01", "series_02"), 1):
        # Reverse lexical source names: do not infer source path from run number.
        path = case / f"source_{3-index}.dcm"
        meta = FileMetaDataset()
        meta.TransferSyntaxUID = ExplicitVRLittleEndian
        meta.MediaStorageSOPClassUID = XRayAngiographicImageStorage
        meta.MediaStorageSOPInstanceUID = f"1.2.3.{index}.1"
        ds = FileDataset(str(path), {}, file_meta=meta, preamble=b"\0" * 128)
        ds.SOPClassUID = meta.MediaStorageSOPClassUID
        ds.SOPInstanceUID = meta.MediaStorageSOPInstanceUID
        ds.SeriesInstanceUID = f"1.2.3.{index}"
        ds.Modality = "XA"
        ds.NumberOfFrames, ds.Rows, ds.Columns = count, 64, 64
        if timing_kind == "vector":
            ds.FrameTimeVector = ([0, 100, 120, 80] if index == 1 else [0, 80, 110, 90]) + [100] * (count - 4)
            ds.FrameTime = 999  # vector wins, not this scalar or the cache median
        elif timing_kind == "scalar":
            ds.FrameTime = 40 if index == 1 else 60
        elif timing_kind == "malformed":
            ds.FrameTimeVector = [0, 100, -2, 80] + [100] * (count - 4)
            ds.FrameTime = 999  # audit deliberately forbids this fallback
        elif timing_kind == "display_rate":
            ds.RecommendedDisplayFrameRate = 8
        ds.save_as(path, enforce_file_format=True)
        series = assets / "all_series" / run_id
        manifest_path = series / "series_manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["frame_times_ms"] = frame_times_ms(ds, count)
        manifest["source_record"] = audit_record(path, root=case)
        manifest["source_record"]["path"] = "ignored/old/absolute/path.dcm"
        if identity == "mismatch":
            manifest["source_record"]["sha256"] = "0" * 64
        elif identity == "missing_identity":
            manifest["source_record"].pop("sop_instance_uid")
        elif identity == "missing_hash":
            manifest["source_record"].pop("sha256")
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return case, assets


@pytest.mark.parametrize("kind,expected,status,source", [
    ("vector", [0, 100, 220, 300], "available", "FrameTimeVector"),
    ("no_timing", [None] * 4, "insufficient", "unknown"),
])
def test_four_frame_exporter_audit_observation_contract(tmp_path: Path, kind: str, expected: list, status: str, source: str) -> None:
    # Break caught: actual initial-zero DICOM increments rejected or display fallback promoted.
    import ctdr_pipeline.motion_validation.runner as runner
    from ctdr_pipeline.motion_validation.observations import build_observation_report
    from ctdr_pipeline.motion_validation.preflight import audit_case

    case, assets = _exporter_timing_fixture(tmp_path, count=4, timing_kind=kind)
    audit = audit_case(case, assets)
    link = audit["source_links"]["series_01"]
    assert link["status"] == "verified"
    source_record = next(record for record in audit["source_files"] if record["relative_path"] == link["source_relative_path"])
    assert source_record["frame_timestamps_ms"] == ([0, 100, 220, 300] if kind == "vector" else [])
    manifest = json.loads((assets / "all_series/series_01/series_manifest.json").read_text())
    assert manifest["frame_times_ms"] == ([0, 100, 120, 80] if kind == "vector" else [125] * 4)
    contract = getattr(runner, "_acquisition_timing", None)
    assert callable(contract), "runner has no source-audit-to-acquisition contract"
    observation = build_observation_report(assets / "all_series/series_01", acquisition_timing=contract(audit, "series_01"))
    assert observation["frame_times_ms"] == expected
    assert observation["timing_status"] == status
    assert observation["timing_source"] == source
    assert observation["status"] == "provisional_unreviewed"
    json.dumps(observation, allow_nan=False)


@pytest.mark.parametrize("kind,identity,cache_count,status,expected_prefix", [
    ("vector", "verified", 31, "available", [0, 100, 220, 300]),
    ("scalar", "verified", 31, "available", [0, 40, 80, 120]),
    ("no_timing", "verified", 31, "insufficient", [None] * 4),
    ("display_rate", "verified", 31, "insufficient", [None] * 4),
    ("malformed", "verified", 31, "insufficient", [None] * 4),
    ("scalar", "mismatch", 31, "unverified", [None] * 4),
    ("scalar", "missing_identity", 31, "unverified", [None] * 4),
    ("scalar", "missing_hash", 31, "unverified", [None] * 4),
    ("scalar", "verified", 32, "frame_count_mismatch", [None] * 4),
])
def test_runner_wires_only_identity_matched_acquisition_times(tmp_path: Path, kind: str, identity: str, cache_count: int, status: str, expected_prefix: list) -> None:
    # Break caught: runner omits/misbinds audited timing or labels an identity mismatch verified.
    case, assets = _exporter_timing_fixture(tmp_path, count=31, timing_kind=kind, identity=identity, cache_count=cache_count)
    output = tmp_path / "run"
    report = run_validation(case, assets, output)
    audit = json.loads((output / "preflight.json").read_text())
    for run_id in ("series_01", "series_02"):
        observation = json.loads((output / "observations" / f"{run_id}.json").read_text())
        assert observation["status"] == "provisional_unreviewed"
        prefix = expected_prefix
        if status == "available" and run_id == "series_02":
            prefix = [0, 80, 190, 280] if kind == "vector" else [0, 60, 120, 180]
        assert observation["frame_times_ms"][:4] == prefix
        assert observation["timing_status"] == status
        provenance = observation["timing_provenance"]
        assert provenance["source_verified"] is (identity == "verified")
        assert provenance["source_link_status"] == identity
        if status == "available":
            link = audit["source_links"][run_id]
            source_record = next(record for record in audit["source_files"] if record["relative_path"] == link["source_relative_path"])
            assert observation["frame_times_ms"] == source_record["frame_timestamps_ms"]
            assert provenance["source_relative_path"] == link["source_relative_path"]
            assert observation["timing_source"] == source_record["timing_source"]
        else:
            assert observation["frame_times_ms"] == [None] * cache_count
        assert all(track["semantic_status"] == "unreviewed_image_feature" for track in observation["tracks"])
        assert (output / "observations" / f"{run_id}_review.png").is_file()
        json.dumps(observation, allow_nan=False)
    assert report["gates"]["absolute_geometry_accepted"] is False
    assert report["patient_3d_optimization_enabled"] is False
    assert report["patient_true_3d_tre_mm"] is None
    assert report["clinical_validation"] is False


def _junction(link: Path, target: Path) -> None:
    import os
    import subprocess

    if os.name != "nt":
        pytest.skip("NTFS junction tests require Windows")
    # Only artificial test directories, no deletion or cross-shell path discovery.
    result = subprocess.run(["cmd.exe", "/c", "mklink", "/J", str(link), str(target)], capture_output=True, text=True)
    if result.returncode:
        pytest.skip(f"Real NTFS junction creation unavailable: {result.stderr.strip()} (exit {result.returncode})")


@pytest.mark.parametrize("root_kind", ["case", "assets", "ancestor"])
def test_ntfs_junction_input_roots_are_rejected_before_resolution(tmp_path: Path, monkeypatch, root_kind: str) -> None:
    # Break caught: Python 3.11 has no is_junction and resolves a linked root into authority.
    history = tmp_path / "history"
    case, assets = history / "case", _artificial_assets(history / "assets")
    case.mkdir()
    (case / "source.bin").write_bytes(b"historical fixture")
    link = tmp_path / "linked"
    _junction(link, history if root_kind == "ancestor" else case if root_kind == "case" else assets)
    attempts = _record_forbidden_reads(monkeypatch, history)
    with pytest.raises(ValueError, match="[Ee]xcluded|authoritative"):
        run_validation(link / "case" if root_kind == "ancestor" else link if root_kind == "case" else case,
                       link if root_kind == "assets" else assets, tmp_path / "run")
    assert attempts == []
    assert not (tmp_path / "run").exists()


@pytest.mark.parametrize("consumer", ["observation", "render", "mask"])
def test_ntfs_junction_consumers_do_not_open_historical_targets(tmp_path: Path, monkeypatch, consumer: str) -> None:
    from ctdr_pipeline.motion_validation.observations import build_observation_report

    history = _artificial_assets(tmp_path / "history")
    _artificial_mask(history / "thorax_masks/heart.nii.gz")
    link = tmp_path / "linked"
    _junction(link, history)
    attempts = _record_forbidden_reads(monkeypatch, history)
    if consumer == "mask":
        points, provenance = _heart_shape_points(link / "thorax_masks/heart.nii.gz")
        assert points is None
        assert provenance["kind"] == "digital_phantom"
        assert "excluded" in provenance["reason"].lower()
    else:
        with pytest.raises(ValueError, match="[Ee]xcluded|authoritative"):
            (build_observation_report if consumer == "observation" else _read_cine)(link / "all_series/series_01")
    assert attempts == []


def test_ntfs_junction_case_subtree_is_not_hashed(tmp_path: Path, monkeypatch) -> None:
    case, assets, history = tmp_path / "case", tmp_path / "assets", tmp_path / "history"
    for root in (case, assets, history):
        root.mkdir()
    (history / "source.bin").write_bytes(b"historical fixture")
    _junction(case / "linked", history)
    attempts = _record_forbidden_reads(monkeypatch, history)
    assert _snapshot_inputs(case, assets) == {}
    assert attempts == []


@pytest.mark.parametrize("location", ["case", "all_series/series_01", "all_series/series_01/temporal", "thorax_masks"])
def test_runner_rejects_ntfs_junctions_before_unaware_preflight_can_read(tmp_path: Path, monkeypatch, location: str) -> None:
    # Break caught: fixed consumers still let the unchanged preflight dereference junctions.
    case, assets = tmp_path / "case", _artificial_assets(tmp_path / "assets")
    case.mkdir()
    _artificial_mask(assets / "thorax_masks/heart.nii.gz")
    history = tmp_path / "history"
    if location == "case":
        history.mkdir()
        (history / "source.bin").write_bytes(b"historical fixture")
        link = case / "linked"
    else:
        link = assets / location
        link.rename(history)
    _junction(link, history)
    attempts = _record_forbidden_reads(monkeypatch, history)
    with pytest.raises(ValueError, match="[Ee]xcluded|authoritative"):
        run_validation(case, assets, tmp_path / "run")
    assert attempts == []
    assert not (tmp_path / "run").exists()


def test_ntfs_junction_output_parent_is_rejected_before_target_creation(tmp_path: Path) -> None:
    case, assets, history = tmp_path / "case", tmp_path / "assets", tmp_path / "history"
    for root in (case, assets, history):
        root.mkdir()
    link = tmp_path / "linked"
    _junction(link, history)
    with pytest.raises(ValueError, match="linked"):
        validate_output_paths(case, assets, link / "new_run")
    assert not (history / "new_run").exists()


@pytest.mark.parametrize("location", ["case_root", "assets_root", "case_nested", "asset_nested", "output_parent"])
def test_reparse_metadata_blocks_runner_before_forbidden_io(tmp_path: Path, monkeypatch, location: str) -> None:
    # OS-boundary simulation only; does NOT claim actual NTFS junction coverage.
    import stat

    case, assets = tmp_path / "case", _artificial_assets(tmp_path / "assets")
    case.mkdir()
    (case / "source.bin").write_bytes(b"artificial")
    output_parent = tmp_path / "outputs"
    output_parent.mkdir()
    nested = case / "historical"
    nested.mkdir()
    (nested / "source.bin").write_bytes(b"artificial historical fixture")
    marked = {"case_root": case, "assets_root": assets, "case_nested": nested,
              "asset_nested": assets / "all_series/series_01", "output_parent": output_parent}[location]
    original = Path.lstat

    class ReparseStat:
        def __init__(self, value):
            self.value = value
            self.st_file_attributes = getattr(value, "st_file_attributes", 0) | getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)

        def __getattr__(self, key):
            return getattr(self.value, key)

    def lstat(path, *args, **kwargs):
        value = original(path, *args, **kwargs)
        return ReparseStat(value) if path == marked else value

    monkeypatch.setattr(Path, "lstat", lstat)
    attempts = _record_forbidden_reads(monkeypatch, marked)
    with pytest.raises(ValueError, match="[Ee]xcluded|authoritative|linked"):
        run_validation(case, assets, output_parent / "run")
    assert attempts == []
    assert not (output_parent / "run").exists()
