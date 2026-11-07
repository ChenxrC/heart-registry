"""Review must not promote automatic points or use stale/catheter seeds."""

import copy
import hashlib
import importlib
import json
from pathlib import Path

import numpy as np
import pytest


def api():
    return importlib.import_module("ctdr_pipeline.motion_validation.review")


def bundle_api():
    return importlib.import_module("ctdr_pipeline.motion_validation.review_bundle")


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, allow_nan=False), encoding="utf-8")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def inputs(tmp_path):
    assets, validation = tmp_path / "assets", tmp_path / "validation"
    assets.mkdir()
    validation.mkdir()
    records = {}
    for run_id in ("series_01", "series_02"):
        series = assets / "all_series" / run_id
        series.mkdir(parents=True)
        rng = np.random.default_rng(73)
        frame = rng.integers(0, 256, size=(48, 48), dtype=np.uint8)
        np.repeat(frame[None], 4, axis=0).tofile(series / "cine_u8.raw")
        write_json(series / "series_manifest.json", {"shape_frame_row_column": [4, 48, 48]})
        write_json(series / f"{run_id}_calibration.json", {
            "quality_gate_passed": False, "quality_status": "diagnostic-only",
            "transferred_from": {"reference_series": "XA Series 7", "reference_rms_error_px": 6.65},
            "calibration": {"u_lps": [1, 0, 0], "v_lps": [0, 1, 0], "n_lps": [0, 0, 1],
                "source_lps_mm": [0, 0, 0], "detector_center_lps_mm": [0, 0, 1000],
                "isocenter_lps_mm": [0, 0, 800], "sdd_mm": 1000, "sod_mm": 800,
                "principal_uv_px": [23.5, 23.5]}})
        for path in series.iterdir():
            records[path.relative_to(assets).as_posix()] = {"sha256": sha(path), "read_status": "read"}
        write_json(validation / "observations" / f"{run_id}.json", {
            "run_id": run_id, "reference_frame_index": 1,
            "frame_times_ms": [0, 100, 200, 300], "timing_status": "available",
            "timing_provenance": {"source_verified": True},
            "tracks": [{"track_id": 0, "xy_px": [[24, 24]] * 4, "valid": [True] * 4}],
        })
    write_json(validation / "preflight.json", {"case_id": "fixture", "assets": records,
        "source_links": {r: {"status": "verified", "actual_sha256": "a" * 64} for r in ("series_01", "series_02")},
        "gates": {"absolute_geometry_accepted": False, "phase_sync_accepted": False}})
    return validation, assets, tmp_path / "review_pack"


def build(inputs):
    validation, assets, output = inputs
    bundle_api().build_review_bundle(validation, assets, output)
    return output, json.loads((output / "manifest.json").read_text(encoding="utf-8"))


def included(document):
    doc = copy.deepcopy(document)
    doc["reviewer"] = "synthetic test only"
    doc["review_status"] = "submitted"
    point = doc["runs"][0]["points"][0]
    point.update(decision="include", anatomical_label="fixture branch")
    point["observations"] = [{"frame_index": 1, "state": "visible", "xy_px": [24, 24], "reviewed": True}]
    return doc


def test_new_bundle_leaves_candidates_pending_and_inputs_unchanged(inputs):
    before = {p: sha(p) for root in inputs[:2] for p in root.rglob("*") if p.is_file()}
    output, manifest = build(inputs)
    doc = json.loads((output / "review_template.json").read_text(encoding="utf-8"))
    assert doc["review_status"] == "draft"
    assert doc["reviewer"] == ""
    assert all(p["decision"] == "pending" and p["observations"] == [] for r in doc["runs"] for p in r["points"])
    assert manifest["patient_3d_optimization_enabled"] is False
    assert manifest["runs"][0]["geometry"]["geometry_accepted"] is False
    assert manifest["runs"][0]["geometry"]["current_series_reprojection_rms_px"] is None
    assert manifest["runs"][0]["geometry"]["reference_series_rms_px"] == 6.65
    assert (output / "frames/series_01/0001.png").is_file()
    assert (output / "index.html").is_file()
    assert before == {p: sha(p) for p in before}


def test_pack_id_is_stable_and_bound_to_observations(inputs):
    output, manifest = build(inputs)
    validation, assets, _ = inputs
    other = output.with_name("second")
    bundle_api().build_review_bundle(validation, assets, other)
    assert manifest["pack_id"] == json.loads((other / "manifest.json").read_text())["pack_id"]
    observation = validation / "observations/series_01.json"
    data = json.loads(observation.read_text())
    data["tracks"][0]["xy_px"][1] = [25, 24]
    write_json(observation, data)
    third = output.with_name("third")
    bundle_api().build_review_bundle(validation, assets, third)
    assert manifest["pack_id"] != json.loads((third / "manifest.json").read_text())["pack_id"]


@pytest.mark.parametrize("name", ["cine_u8.raw", "series_manifest.json", "series_01_calibration.json"])
def test_changed_audited_asset_is_rejected_before_output(inputs, name):
    validation, assets, output = inputs
    path = assets / "all_series/series_01" / name
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(ValueError, match="SHA256"):
        bundle_api().build_review_bundle(validation, assets, output)
    assert not output.exists()


def test_existing_output_is_never_reused(inputs):
    build(inputs)
    with pytest.raises(FileExistsError):
        bundle_api().build_review_bundle(*inputs)


def test_candidate_input_is_rejected(inputs):
    validation, assets, output = inputs
    candidate = assets.with_name("assets_candidate")
    assets.rename(candidate)
    with pytest.raises(ValueError):
        bundle_api().build_review_bundle(validation, candidate, output)
    assert not output.exists()


@pytest.fixture
def review_fixture(inputs):
    output, manifest = build(inputs)
    draft = json.loads((output / "review_template.json").read_text(encoding="utf-8"))
    return manifest, included(draft), output


def test_reviewed_seed_does_not_accept_patient_geometry(review_fixture):
    manifest, document, _ = review_fixture
    result = api().validate_review(document, manifest)
    assert result["included_point_count"] == 1
    assert result["reviewer_identity_verified"] is False
    assert result["patient_3d_optimization_enabled"] is False
    assert result["patient_true_3d_tre_mm"] is None


@pytest.mark.parametrize("field,value", [("pack_id", "wrong"), ("case_id", "other"),
    ("research_only", False), ("reviewer", ""), ("schema_version", True), ("review_status", "approved_clinical")])
def test_invalid_review_identity_or_submission_rejected(review_fixture, field, value):
    manifest, document, _ = review_fixture
    document[field] = value
    with pytest.raises(ValueError):
        api().validate_review(document, manifest)


@pytest.mark.parametrize("xy", [[-1, 20], [48, 20], [20, float("nan")], [True, 20], [24], None])
def test_visible_review_requires_finite_in_bounds_xy(review_fixture, xy):
    manifest, document, _ = review_fixture
    document["runs"][0]["points"][0]["observations"][0]["xy_px"] = xy
    with pytest.raises(ValueError):
        api().validate_review(document, manifest)


@pytest.mark.parametrize("index", [-1, 4, True, 1.0])
def test_frame_index_is_exact_integer_in_range(review_fixture, index):
    manifest, document, _ = review_fixture
    document["runs"][0]["points"][0]["observations"][0]["frame_index"] = index
    with pytest.raises(ValueError):
        api().validate_review(document, manifest)


def test_include_needs_explicit_anatomy_and_reviewed_visible_seed(review_fixture):
    manifest, document, _ = review_fixture
    point = document["runs"][0]["points"][0]
    point["observations"][0]["reviewed"] = False
    with pytest.raises(ValueError):
        api().validate_review(document, manifest)
    point["observations"][0]["reviewed"] = True
    point["anatomical_label"] = ""
    with pytest.raises(ValueError):
        api().validate_review(document, manifest)


def test_invisible_point_cannot_have_a_coordinate(review_fixture):
    manifest, document, _ = review_fixture
    point = document["runs"][0]["points"][0]
    point["observations"].append({"frame_index": 2, "state": "not_visible", "xy_px": [24, 24], "reviewed": True})
    with pytest.raises(ValueError):
        api().validate_review(document, manifest)


def test_duplicate_ids_frames_and_wrong_source_track_rejected(review_fixture):
    manifest, document, _ = review_fixture
    point = document["runs"][0]["points"][0]
    point["source_track_id"] = 99
    with pytest.raises(ValueError):
        api().validate_review(document, manifest)
    point["source_track_id"] = 0
    point["observations"] *= 2
    with pytest.raises(ValueError):
        api().validate_review(document, manifest)
    point["observations"] = point["observations"][:1]
    document["runs"][0]["points"].append(copy.deepcopy(point))
    with pytest.raises(ValueError):
        api().validate_review(document, manifest)


def test_retracking_consumes_only_included_seeds_and_keeps_manual_observations_separate(inputs, review_fixture):
    manifest, document, pack = review_fixture
    point = document["runs"][0]["points"][0]
    point["observations"].append({"frame_index": 2, "state": "not_visible", "xy_px": None, "reviewed": True})
    reviewed_path = inputs[2].parent / "test_review.json"
    write_json(reviewed_path, document)
    output = inputs[2].parent / "retracked"
    result = bundle_api().retrack_review(reviewed_path, pack, inputs[1], output)
    tracks = result["runs"][0]["tracked_candidates"]
    assert len(tracks) == 1
    assert tracks[0]["semantic_status"] == "unreviewed_tracking_suggestion"
    assert tracks[0]["valid"][2] is False
    assert tracks[0]["xy_px"][2] is None
    assert result["runs"][0]["manual_observations"][0]["observations"][1]["state"] == "not_visible"
    assert result["patient_3d_optimization_enabled"] is False
    assert (output / "retracking_report.json").is_file()


def test_draft_empty_review_generates_no_fictional_tracks(inputs):
    output, manifest = build(inputs)
    result = bundle_api().retrack_review(output / "review_template.json", output, inputs[1], output.with_name("empty"))
    assert result["status"] == "no_included_manual_seeds"
    assert all(r["tracked_candidates"] == [] for r in result["runs"])


def test_excluded_catheter_is_not_a_tracking_seed(inputs, review_fixture):
    _, document, pack = review_fixture
    point = document["runs"][0]["points"][0]
    point.update(decision="exclude", exclusion_reason="catheter")
    path = inputs[2].parent / "excluded.json"
    write_json(path, document)
    result = bundle_api().retrack_review(path, pack, inputs[1], pack.with_name("excluded"))
    assert result["review_summary"]["excluded_point_count"] == 1
    assert all(r["tracked_candidates"] == [] for r in result["runs"])


def test_retracking_refuses_changed_cine_or_tampered_pack(inputs, review_fixture):
    _, document, pack = review_fixture
    path = inputs[2].parent / "review.json"
    write_json(path, document)
    cine = inputs[1] / "all_series/series_01/cine_u8.raw"
    cine.write_bytes(b"wrong")
    with pytest.raises(ValueError, match="SHA256"):
        bundle_api().retrack_review(path, pack, inputs[1], pack.with_name("bad"))
    assert not pack.with_name("bad").exists()


@pytest.mark.parametrize("mutation", ["path", "run_id", "shape", "hash", "duplicate_run", "missing_cine", "candidate", "absolute"])
def test_manifest_digest_is_not_permission_for_arbitrary_paths(review_fixture, mutation):
    manifest, _, _ = review_fixture
    bad = copy.deepcopy(manifest)
    run = bad["runs"][0]
    if mutation == "path":
        run["asset_sha256"]["../../outside.json"] = "a" * 64
    elif mutation == "run_id":
        run["run_id"] = "../../elsewhere"
    elif mutation == "shape":
        run["shape_frame_row_column"] = [4, True, 48]
    elif mutation == "hash":
        run["asset_sha256"]["cine_u8.raw"] = "invalid"
    elif mutation == "missing_cine":
        del run["asset_sha256"]["cine_u8.raw"]
    elif mutation == "candidate":
        run["run_id"] = "../series_01_candidate"
    elif mutation == "absolute":
        run["asset_sha256"][str(Path.cwd() / "outside.raw")] = "a" * 64
    else:
        bad["runs"][1] = copy.deepcopy(run)
    bad["binding"]["runs"] = bad["runs"]
    bad["pack_id"] = api().pack_digest(bad["binding"])
    with pytest.raises(ValueError):
        api().validate_manifest(bad)


def test_omitted_auto_candidates_cannot_silently_shrink_review_denominator(review_fixture):
    manifest, document, _ = review_fixture
    document["runs"][1]["points"] = []
    with pytest.raises(ValueError):
        api().validate_review(document, manifest)


@pytest.mark.parametrize("target_kind", ["preflight", "observation", "manifest", "review"])
def test_json_save_during_snapshot_cannot_bind_old_object_to_new_hash(inputs, monkeypatch, target_kind):
    validation, assets, output = inputs
    pack, _ = build(inputs)
    doc = included(json.loads((pack / "review_template.json").read_text(encoding="utf-8")))
    review_path = output.parent / "racing_review.json"
    write_json(review_path, doc)
    target = {"preflight": validation / "preflight.json", "observation": validation / "observations/series_01.json",
        "manifest": pack / "manifest.json", "review": review_path}[target_kind]
    original_bytes, original_text = Path.read_bytes, Path.read_text
    fired = False

    def replace_after_read(path, content):
        nonlocal fired
        if path == target and not fired:
            fired = True
            changed = json.loads(content)
            changed["concurrent_save"] = "new bytes, not the parsed snapshot"
            path.write_text(json.dumps(changed), encoding="utf-8")
        return content

    def racing_bytes(path):
        return replace_after_read(path, original_bytes(path))

    def racing_text(path, *args, **kwargs):
        return replace_after_read(path, original_text(path, *args, **kwargs))

    monkeypatch.setattr(Path, "read_bytes", racing_bytes)
    monkeypatch.setattr(Path, "read_text", racing_text)
    destination = output.with_name("race_result")
    with pytest.raises(ValueError, match="[Ii]nputs changed"):
        if target_kind in ("preflight", "observation"):
            bundle_api().build_review_bundle(validation, assets, destination)
        else:
            bundle_api().retrack_review(review_path, pack, assets, destination)
    assert fired
    assert not destination.exists()
