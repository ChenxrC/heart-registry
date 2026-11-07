"""Offline review-pack export and conservative retracking of curated 2-D seeds."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

from .input_policy import require_eligible_input
from .observations import track_points_bidirectional
from .review import pack_digest, validate_manifest, validate_review
from .runner import validate_output_paths


def _json_snapshot(path: Path) -> tuple[dict, str]:
    """Bind parsed content and digest to the *same* eligible byte read."""
    require_eligible_input(path)
    content = path.read_bytes()
    data = json.loads(content.decode("utf-8"))
    json.dumps(data, allow_nan=False)
    if not isinstance(data, dict):
        raise ValueError(f"JSON object required: {path.name}")
    return data, hashlib.sha256(content).hexdigest()


def _read_json(path: Path) -> dict:
    return _json_snapshot(path)[0]


def _sha(path: Path) -> str:
    require_eligible_input(path)
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _geometry_diagnostic(report: dict) -> dict:
    cal = report.get("calibration", {})
    transfer = report.get("transferred_from", {})
    result = {
        "geometry_accepted": False,
        "reported_quality_gate_passed": report.get("quality_gate_passed") is True,
        "quality_status": report.get("quality_status", "missing"),
        "reference_series": transfer.get("reference_series"),
        "reference_series_rms_px": transfer.get("reference_rms_error_px"),
        "current_series_reprojection_rms_px": None,
        "basis_orthogonality_error": None, "basis_determinant": None,
        "sdd_distance_error_mm": None, "sod_distance_error_mm": None,
        "principal_uv_px": cal.get("principal_uv_px"),
        "limitations": [
            "Transferred reference RMS is not this series' measured reprojection error.",
            "An orthonormal basis and correct source distances do not prove anatomical alignment.",
            "No new CT-to-XA 2D-3D correspondences, geometry acceptance, or patient optimization here.",
        ],
    }
    try:
        basis = np.column_stack([cal[k] for k in ("u_lps", "v_lps", "n_lps")]).astype(float)
        source = np.asarray(cal["source_lps_mm"], dtype=float)
        detector = np.asarray(cal["detector_center_lps_mm"], dtype=float)
        iso = np.asarray(cal["isocenter_lps_mm"], dtype=float)
        if basis.shape != (3, 3) or any(v.shape != (3,) for v in (source, detector, iso)):
            raise ValueError("Invalid camera shape")
        values = np.concatenate((basis.ravel(), source, detector, iso, [cal["sdd_mm"], cal["sod_mm"]]))
        if not np.isfinite(values).all():
            raise ValueError("Nonfinite camera")
        result.update(
            basis_orthogonality_error=float(np.abs(basis.T @ basis - np.eye(3)).max()),
            basis_determinant=float(np.linalg.det(basis)),
            sdd_distance_error_mm=float(np.linalg.norm(detector - source) - cal["sdd_mm"]),
            sod_distance_error_mm=float(np.linalg.norm(iso - source) - cal["sod_mm"]),
        )
    except (KeyError, TypeError, ValueError):
        result["limitations"].append("Incomplete or invalid camera: numerical diagnostics unavailable.")
    return result


def build_review_bundle(validation_run: Path, assets_root: Path, output_dir: Path) -> dict:
    validation_run, assets_root, output_dir = validate_output_paths(validation_run, assets_root, output_dir)
    preflight_path = validation_run / "preflight.json"
    preflight, preflight_sha = _json_snapshot(preflight_path)
    snapshots = {preflight_path: preflight_sha}
    runs, observations, cines = [], {}, {}
    for run_id in ("series_01", "series_02"):
        observation_path = validation_run / "observations" / f"{run_id}.json"
        observation, observation_sha = _json_snapshot(observation_path)
        snapshots[observation_path] = observation_sha
        asset_hashes = {}
        series = assets_root / "all_series" / run_id
        for name in ("series_manifest.json", "cine_u8.raw", f"{run_id}_calibration.json"):
            path = series / name
            actual = _sha(path)
            record = preflight.get("assets", {}).get(path.relative_to(assets_root).as_posix(), {})
            if record.get("read_status") != "read" or record.get("sha256") != actual:
                raise ValueError(f"SHA256 differs from frozen audit: {run_id}/{name}")
            snapshots[path] = asset_hashes[name] = actual
        series_manifest = _read_json(series / "series_manifest.json")
        shape = series_manifest.get("shape_frame_row_column")
        if not isinstance(shape, list) or len(shape) != 3 or any(type(v) is not int or v <= 0 for v in shape):
            raise ValueError("Invalid cine shape")
        require_eligible_input(series / "cine_u8.raw")
        raw = np.fromfile(series / "cine_u8.raw", dtype=np.uint8)
        if raw.size != int(np.prod(shape)):
            raise ValueError("Cine byte length differs from native shape")
        if observation.get("run_id") != run_id:
            raise ValueError("Observation series identity mismatch")
        tracks = observation.get("tracks", [])
        ids = [p["track_id"] for p in tracks]
        if len(set(ids)) != len(ids) or any(type(v) is not int or v < 0 for v in ids):
            raise ValueError("Invalid source track IDs")
        for track in tracks:
            if len(track["xy_px"]) != shape[0] or len(track["valid"]) != shape[0]:
                raise ValueError("Observation/cine frame count mismatch")
        reference = observation.get("reference_frame_index")
        if type(reference) is not int or not 0 <= reference < shape[0]:
            raise ValueError("Invalid reference frame")
        geometry = _geometry_diagnostic(_read_json(series / f"{run_id}_calibration.json"))
        runs.append({"run_id": run_id, "shape_frame_row_column": shape,
            "reference_frame_index": reference, "source_track_ids": ids,
            "asset_sha256": asset_hashes, "observation_sha256": snapshots[observation_path],
            "source_link": preflight.get("source_links", {}).get(run_id, {}), "geometry": geometry})
        observations[run_id], cines[run_id] = observation, raw.reshape(shape)
    binding = {"case_id": preflight["case_id"], "preflight_sha256": snapshots[preflight_path], "runs": runs}
    manifest = {"schema_version": 1, "pack_id": pack_digest(binding), "binding": binding,
        "case_id": preflight["case_id"], "runs": runs, "research_only": True,
        "patient_3d_optimization_enabled": False, "patient_true_3d_tre_mm": None,
        "clinical_validation": False, "inherited_gates": preflight.get("gates", {})}
    template = {"schema_version": 1, "pack_id": manifest["pack_id"], "case_id": manifest["case_id"],
        "research_only": True, "review_status": "draft", "reviewer": "", "runs": []}
    for run in runs:
        template["runs"].append({"run_id": run["run_id"], "points": [
            {"point_id": f"auto_{i}", "origin": "automatic", "source_track_id": i,
             "decision": "pending", "anatomical_label": "", "exclusion_reason": None, "observations": []}
            for i in run["source_track_ids"]]})
    validate_review(template, manifest)
    ui_root = Path(__file__).resolve().parent / "ui"
    ui_bytes = {name: (ui_root / name).read_bytes() for name in ("index.html", "review.js")}
    if any(_sha(path) != value for path, value in snapshots.items()):
        raise ValueError("Inputs changed while preparing review")
    output_dir.mkdir(parents=True, exist_ok=False)
    for name, content in ui_bytes.items():
        (output_dir / name).write_bytes(content)
    for run_id, cine in cines.items():
        frame_dir = output_dir / "frames" / run_id
        frame_dir.mkdir(parents=True)
        for index, frame in enumerate(cine):
            Image.fromarray(frame).save(frame_dir / f"{index:04d}.png")
    _write_json(output_dir / "manifest.json", manifest)
    _write_json(output_dir / "review_template.json", template)
    payload = json.dumps({"manifest": manifest, "template": template, "observations": observations}, ensure_ascii=False, allow_nan=False)
    payload = payload.replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026").replace("\u2028", "\\u2028").replace("\u2029", "\\u2029")
    (output_dir / "data.js").write_text("window.REVIEW_DATA=" + payload + ";\n", encoding="utf-8")
    _write_json(output_dir / "input_integrity.json", {"unchanged": all(_sha(p) == h for p, h in snapshots.items()),
        "sha256": {str(p): h for p, h in snapshots.items()}})
    return manifest


def retrack_review(review_path: Path, bundle_dir: Path, assets_root: Path, output_dir: Path) -> dict:
    bundle_dir, assets_root, output_dir = validate_output_paths(bundle_dir, assets_root, output_dir)
    require_eligible_input(review_path)
    if output_dir == Path(review_path).resolve() or output_dir in Path(review_path).resolve().parents:
        raise ValueError("Output overlaps review input")
    manifest, manifest_sha = _json_snapshot(bundle_dir / "manifest.json")
    validate_manifest(manifest)
    document, review_sha = _json_snapshot(Path(review_path))
    summary = validate_review(document, manifest)
    snapshots = {Path(review_path): review_sha, bundle_dir / "manifest.json": manifest_sha}
    runs = []
    for bound in manifest["runs"]:
        run_id = bound["run_id"]
        series = assets_root / "all_series" / run_id
        for name, expected in bound["asset_sha256"].items():
            actual = _sha(series / name)
            if actual != expected:
                raise ValueError(f"SHA256 differs from review pack: {run_id}/{name}")
            snapshots[series / name] = actual
        cine_path = series / "cine_u8.raw"
        require_eligible_input(cine_path)
        if not cine_path.resolve().is_relative_to(assets_root):
            raise ValueError("Cine path escapes the asset root")
        cine = np.fromfile(cine_path, dtype=np.uint8).reshape(bound["shape_frame_row_column"])
        reviewed_run = next(r for r in document["runs"] if r["run_id"] == run_id)
        selected = [p for p in reviewed_run["points"] if p["decision"] == "include"]
        candidates = []
        for point in selected:
            anchors = [o for o in point["observations"] if o["state"] == "visible"]
            anchor = min(anchors, key=lambda o: (o["frame_index"] != bound["reference_frame_index"], o["frame_index"]))
            result = track_points_bidirectional(cine, np.asarray([anchor["xy_px"]], dtype=np.float32), anchor["frame_index"])
            track = result["tracks"][0]
            track.update(point_id=point["point_id"], anatomical_label=point["anatomical_label"],
                seed_frame_index=anchor["frame_index"], semantic_status="unreviewed_tracking_suggestion")
            # Do not replace manual observations with predictions, or silently use
            # secondary corrections as re-detection seeds. Null flags override suggestions.
            for o in point["observations"]:
                if o["state"] != "visible":
                    index = o["frame_index"]
                    track["valid"][index] = False
                    track["xy_px"][index] = track["fb_error_px"][index] = None
                    track["failure_reason"][index] = "manual_" + o["state"]
            candidates.append(track)
        runs.append({"run_id": run_id, "manual_observations": copy.deepcopy(selected), "tracked_candidates": candidates})
    if any(_sha(path) != expected for path, expected in snapshots.items()):
        raise ValueError("Inputs changed during retracking")
    result = {"schema_version": 1, "pack_id": manifest["pack_id"], "case_id": manifest["case_id"],
        "research_only": True, "review_summary": summary, "runs": runs,
        "status": "unreviewed_tracking_suggestions" if summary["included_point_count"] else "no_included_manual_seeds",
        "patient_3d_optimization_enabled": False, "patient_true_3d_tre_mm": None, "clinical_validation": False,
        "input_sha256": {str(p): h for p, h in snapshots.items()}, "source_integrity_unchanged": True,
        "limitations": ["Reviewer identity/anatomical correctness are not authenticated by this local tool.",
            "Only explicitly reviewed frames are manual observations; propagated suggestions need independent review.",
            "No new 3D correspondence, geometry acceptance, phase synchronization or clinical validation."]}
    output_dir.mkdir(parents=True, exist_ok=False)
    _write_json(output_dir / "retracking_report.json", result)
    _write_json(output_dir / "submitted_review.json", document)
    return result
