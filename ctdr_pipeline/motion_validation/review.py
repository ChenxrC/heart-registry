"""Strict, source-bound manual 2-D observations; never clinical acceptance."""

from __future__ import annotations

import hashlib
import json
import math
import re
from typing import Any


def pack_digest(binding: dict) -> str:
    return hashlib.sha256(json.dumps(binding, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def validate_manifest(manifest: dict) -> None:
    if not isinstance(manifest, dict) or manifest.get("pack_id") != pack_digest(manifest.get("binding", {})):
        raise ValueError("Review pack binding/pack_id mismatch")
    binding = manifest["binding"]
    if manifest.get("case_id") != binding.get("case_id") or manifest.get("runs") != binding.get("runs"):
        raise ValueError("Review manifest differs from bound runs/case")
    if type(manifest.get("schema_version")) is not int or manifest["schema_version"] != 1:
        raise ValueError("Unsupported manifest schema")
    if not isinstance(manifest.get("case_id"), str) or not manifest["case_id"].strip():
        raise ValueError("Invalid bound case identity")
    if manifest.get("research_only") is not True or manifest.get("patient_3d_optimization_enabled") is not False:
        raise ValueError("Only research review manifests are allowed")
    def valid_hash(value):
        return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None
    if not valid_hash(binding.get("preflight_sha256")):
        raise ValueError("Invalid preflight SHA256")
    runs = manifest.get("runs")
    if not isinstance(runs, list) or len(runs) != 2 or any(not isinstance(r, dict) for r in runs):
        raise ValueError("Manifest requires exactly Series 1 and 2")
    if {r.get("run_id") for r in runs} != {"series_01", "series_02"}:
        raise ValueError("Unsupported/duplicate manifest run identity")
    for run in runs:
        run_id = run["run_id"]
        hashes = run.get("asset_sha256")
        expected = {"series_manifest.json", "cine_u8.raw", f"{run_id}_calibration.json"}
        if not isinstance(hashes, dict) or set(hashes) != expected or not all(valid_hash(v) for v in hashes.values()):
            raise ValueError("Manifest asset SHA256 keys must be the exact filename whitelist")
        shape, reference, ids = run.get("shape_frame_row_column"), run.get("reference_frame_index"), run.get("source_track_ids")
        if not isinstance(shape, list) or len(shape) != 3 or any(type(v) is not int or v <= 0 for v in shape):
            raise ValueError("Invalid manifest shape")
        if type(reference) is not int or not 0 <= reference < shape[0]:
            raise ValueError("Invalid bound reference frame")
        if not isinstance(ids, list) or any(type(i) is not int or i < 0 for i in ids) or len(set(ids)) != len(ids):
            raise ValueError("Invalid bound source track identities")
        if not valid_hash(run.get("observation_sha256")):
            raise ValueError("Invalid observation SHA256")


def validate_review(document: dict, manifest: dict) -> dict[str, Any]:
    """Validate explicit observations. Reviewer identity is user-asserted, not authenticated.

    An include decision approves ONLY supplied, reviewed frames; automatic
    coordinates are not implicitly confirmed at other frames. Missing or
    uncertain observations have null coordinates. Partial drafts remain usable.
    """
    validate_manifest(manifest)
    if not isinstance(document, dict):
        raise ValueError("Review must be an object")
    try:
        json.dumps(document, allow_nan=False)
    except (TypeError, ValueError) as error:
        raise ValueError("Review must contain finite JSON values") from error
    if type(document.get("schema_version")) is not int or document["schema_version"] != 1:
        raise ValueError("Unsupported review schema")
    for field in ("pack_id", "case_id"):
        if document.get(field) != manifest[field]:
            raise ValueError(f"Review {field} does not match this pack")
    if document.get("research_only") is not True:
        raise ValueError("Research-only review is required")
    status, reviewer = document.get("review_status"), document.get("reviewer")
    if status not in ("draft", "submitted") or not isinstance(reviewer, str):
        raise ValueError("Invalid review status/reviewer")
    if status == "submitted" and not reviewer.strip():
        raise ValueError("Submitted review needs a reviewer identifier")
    runs = document.get("runs")
    if not isinstance(runs, list):
        raise ValueError("Review runs must be a list")
    expected = {r["run_id"]: r for r in manifest["runs"]}
    seen_runs = set()
    included = excluded = pending = manual_count = 0
    for run in runs:
        if not isinstance(run, dict) or run.get("run_id") not in expected or run["run_id"] in seen_runs:
            raise ValueError("Unknown or duplicate review run")
        run_id = run["run_id"]
        seen_runs.add(run_id)
        source = expected[run_id]
        frames, rows, columns = source["shape_frame_row_column"]
        if not isinstance(run.get("points"), list):
            raise ValueError("Review points must be a list")
        seen_points, seen_sources = set(), set()
        for point in run["points"]:
            if not isinstance(point, dict):
                raise ValueError("Invalid point")
            pid = point.get("point_id")
            if not isinstance(pid, str) or re.fullmatch(r"[A-Za-z0-9_-]{1,80}", pid) is None or pid in seen_points:
                raise ValueError("Invalid or duplicate point_id")
            seen_points.add(pid)
            origin, track_id = point.get("origin"), point.get("source_track_id")
            if origin == "automatic":
                if type(track_id) is not int or track_id not in source["source_track_ids"] or pid != f"auto_{track_id}" or track_id in seen_sources:
                    raise ValueError("Automatic point must retain its source track identity")
                seen_sources.add(track_id)
            elif origin != "manual" or track_id is not None or not pid.startswith("manual_"):
                raise ValueError("Manual point must not impersonate a source track")
            decision = point.get("decision")
            if decision not in ("pending", "include", "exclude"):
                raise ValueError("Invalid point decision")
            label = point.get("anatomical_label")
            if not isinstance(label, str):
                raise ValueError("Anatomical label must be text")
            reason = point.get("exclusion_reason")
            reasons = ("catheter", "background", "ambiguous", "tracking_error", "other")
            if reason is not None and reason not in reasons:
                raise ValueError("Invalid exclusion reason")
            if decision == "exclude" and reason is None:
                raise ValueError("Exclusion needs a reason")
            if decision == "include" and (not reviewer.strip() or not label.strip() or reason is not None):
                raise ValueError("Included anatomy needs reviewer/label and no exclusion reason")
            observations = point.get("observations")
            if not isinstance(observations, list):
                raise ValueError("Manual observations must be a list")
            seen_frames, visible = set(), 0
            for observation in observations:
                if not isinstance(observation, dict):
                    raise ValueError("Invalid manual observation")
                frame = observation.get("frame_index")
                if type(frame) is not int or not 0 <= frame < frames or frame in seen_frames:
                    raise ValueError("Invalid or duplicate frame index")
                seen_frames.add(frame)
                if observation.get("reviewed") is not True:
                    raise ValueError("Stored manual observations must be explicitly reviewed")
                state, xy = observation.get("state"), observation.get("xy_px")
                if state == "visible":
                    if not isinstance(xy, list) or len(xy) != 2 or any(type(v) not in (int, float) or not math.isfinite(v) for v in xy):
                        raise ValueError("Visible observation needs finite numeric xy")
                    if not (0 <= xy[0] < columns and 0 <= xy[1] < rows):
                        raise ValueError("Manual xy outside native image")
                    visible += 1
                elif state not in ("not_visible", "uncertain") or xy is not None:
                    raise ValueError("Invisible/uncertain observations must have null xy")
            if decision == "include" and not visible:
                raise ValueError("Include needs at least one reviewed visible seed")
            included += decision == "include"
            excluded += decision == "exclude"
            pending += decision == "pending"
            manual_count += len(observations)
        if seen_sources != set(source["source_track_ids"]):
            raise ValueError("Keep all source candidates; explicitly exclude instead of silently omitting")
    if seen_runs != set(expected):
        raise ValueError("Review must retain every run, even when empty")
    return {
        "included_point_count": included, "excluded_point_count": excluded,
        "pending_point_count": pending, "explicit_observation_count": manual_count,
        "reviewer_identity_verified": False, "reviewer_provenance": "user_asserted",
        "patient_3d_optimization_enabled": False, "patient_true_3d_tre_mm": None,
        "clinical_validation": False,
    }
