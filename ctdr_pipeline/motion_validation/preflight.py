"""Read-only, research-only source freezing and CT/XA metadata audit.

No image-derived geometry, phase labels, or existing visualization can establish
an independent patient 3D reference. Only explicit upstream acceptance is carried
forward; missing, malformed, and rejected evidence never becomes acceptance.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pydicom


_RUNS = ("series_01", "series_02")
_PHASE = "all_series/cardiac_phase/phase_alignment.json"
_ASSETS = (
    "thorax_masks/heart.nii.gz",
    *(f"all_series/{run}/{suffix}" for run in _RUNS for suffix in (
        "series_manifest.json", "cine_u8.raw", "temporal/temporal_centerlines_u8.raw",
        f"{run}_calibration.json", "dynamic_3d/dynamic_vessel_3d.json",
    )),
    _PHASE,
    "cine_manifest.json",  # Optional comparison, never authoritative timing.
)
_ECG_NUMERIC = (
    "TriggerTime", "NominalInterval", "HeartRate", "CardiacNumberOfImages",
    "LowRRValue", "HighRRValue", "IntervalsAcquired", "IntervalsRejected", "SkipBeats",
)
_TAGS = [
    "SeriesInstanceUID", "SOPInstanceUID", "SOPClassUID", "Modality", "SeriesNumber",
    "InstanceNumber", "ImagePositionPatient", "ImageOrientationPatient", "SliceThickness",
    "SpacingBetweenSlices", "PixelSpacing", "ConvolutionKernel", "ReconstructionDiameter",
    "ImageType", "TemporalPositionIdentifier", "NumberOfTemporalPositions", "NumberOfFrames",
    "FrameTime", "FrameTimeVector", "FrameIncrementPointer", "Rows", "Columns",
    "PositionerPrimaryAngle", "PositionerSecondaryAngle", "DistanceSourceToDetector",
    "DistanceSourceToPatient", "ImagerPixelSpacing", "FrameOfReferenceUID",
    "TableHeight", "TableLateralIncrement", "TableLongitudinalIncrement", "PositionerMotion",
    "TableMotion", "RWaveTimeVector", "BeatRejectionFlag", "WaveformSequence", *_ECG_NUMERIC,
]
_QUALITY_FIELDS = (
    "schema_version", "quality_status", "quality_gate_passed", "status", "clinical_status",
    "frame_count", "shape_frame_row_column", "dtype", "duration_ms", "median_frame_time_ms",
)


def _issue(record: dict, field: str, reason: str) -> None:
    record["metadata_issues"].append({"field": field, "reason": reason})


def _number(ds: pydicom.Dataset, name: str, record: dict, *, integer: bool = False) -> Any:
    value = getattr(ds, name, None)
    if value is None or value == "":
        return None
    try:
        number = float(value)
        if not math.isfinite(number) or (integer and number != int(number)):
            raise ValueError
        return int(number) if integer else number
    except (TypeError, ValueError, OverflowError):
        _issue(record, name, "invalid or nonfinite numeric metadata")
        return None


def _vector(ds: pydicom.Dataset, name: str, record: dict, size: int | None = None) -> list | None:
    value = getattr(ds, name, None)
    if value is None:
        return None
    try:
        # MultiValue is iterable, but a scalar DS and an arbitrary string are not vectors.
        if isinstance(value, (str, bytes)):
            raise ValueError
        values = [value] if size == 1 and isinstance(value, (int, float)) else value
        result = [float(v) for v in values]
        if not result or (size is not None and len(result) != size) or not all(map(math.isfinite, result)):
            raise ValueError
        return result
    except (TypeError, ValueError, OverflowError):
        _issue(record, name, "invalid length, type, or nonfinite vector metadata")
        return None


def _normal(iop: list | None, record: dict) -> np.ndarray | None:
    if iop is None:
        return None
    row, col = np.asarray(iop[:3]), np.asarray(iop[3:])
    if (abs(float(np.linalg.norm(row)) - 1) > 1e-4
            or abs(float(np.linalg.norm(col)) - 1) > 1e-4 or abs(float(row @ col)) > 1e-4):
        _issue(record, "ImageOrientationPatient", "direction cosines are not orthonormal")
        return None
    normal = np.cross(row, col)
    return normal / np.linalg.norm(normal)


def _strings(ds: pydicom.Dataset, name: str) -> list[str]:
    value = getattr(ds, name, None)
    if value is None:
        return []
    return [str(value)] if isinstance(value, str) else [str(v) for v in value]


def _ct_instance(ds: pydicom.Dataset, record: dict) -> None:
    record.update({
        "image_position_patient": _vector(ds, "ImagePositionPatient", record, 3),
        "image_orientation_patient": _vector(ds, "ImageOrientationPatient", record, 6),
        "slice_thickness_mm": _number(ds, "SliceThickness", record),
        "spacing_between_slices_tag_mm": _number(ds, "SpacingBetweenSlices", record),
        "pixel_spacing_mm": _vector(ds, "PixelSpacing", record, 2),
        "convolution_kernels": _strings(ds, "ConvolutionKernel"),
        "image_type": _strings(ds, "ImageType"),
        "reconstruction_diameter_mm": _number(ds, "ReconstructionDiameter", record),
        "temporal_position_identifier": _number(ds, "TemporalPositionIdentifier", record, integer=True),
        "number_of_temporal_positions": _number(ds, "NumberOfTemporalPositions", record, integer=True),
        "number_of_frames": _number(ds, "NumberOfFrames", record, integer=True),
    })
    for name, field in (("ImagePositionPatient", "image_position_patient"),
                        ("ImageOrientationPatient", "image_orientation_patient")):
        if record[field] is None and name not in ds:
            _issue(record, name, "missing geometry metadata")
    _normal(record["image_orientation_patient"], record)


def _timing(ds: pydicom.Dataset, record: dict) -> None:
    n = _number(ds, "NumberOfFrames", record, integer=True)
    record.update(frame_count=n, frame_time_ms=_number(ds, "FrameTime", record),
                  frame_time_vector_ms=None, frame_timestamps_ms=[], duration_ms=None,
                  timing_status="insufficient", timing_source=None)
    if n is None or not 1 <= n <= 1_000_000:
        _issue(record, "NumberOfFrames", "missing or outside supported positive frame count")
        return
    if "FrameTimeVector" in ds:
        vector = _vector(ds, "FrameTimeVector", record, n)
        record["frame_time_vector_ms"] = vector
        if vector is None or vector[0] != 0 or any(v <= 0 for v in vector[1:]):
            _issue(record, "FrameTimeVector", "requires N increments, first zero, remaining positive; no scalar fallback")
            return
        timestamps = np.cumsum(vector).tolist()
        source = "FrameTimeVector"
    else:
        interval = record["frame_time_ms"]
        if interval is None or interval <= 0:
            _issue(record, "FrameTime", "missing or nonpositive interval")
            return
        timestamps = [i * interval for i in range(n)]
        source = "FrameTime"
    if not all(map(math.isfinite, timestamps)):
        _issue(record, source, "cumulative timing overflow")
        return
    record.update(frame_timestamps_ms=timestamps, duration_ms=timestamps[-1],
                  timing_status="available", timing_source=source)


def _xa_instance(ds: pydicom.Dataset, record: dict) -> None:
    _timing(ds, record)
    record.update({
        "primary_angle_deg": _number(ds, "PositionerPrimaryAngle", record),
        "secondary_angle_deg": _number(ds, "PositionerSecondaryAngle", record),
        "source_to_detector_mm": _number(ds, "DistanceSourceToDetector", record),
        "source_to_patient_mm": _number(ds, "DistanceSourceToPatient", record),
        "imager_pixel_spacing_row_col_mm": _vector(ds, "ImagerPixelSpacing", record, 2),
        "image_position_patient": _vector(ds, "ImagePositionPatient", record, 3),
        "image_orientation_patient": _vector(ds, "ImageOrientationPatient", record, 6),
        "frame_of_reference_uid": str(ds.FrameOfReferenceUID) if "FrameOfReferenceUID" in ds else None,
        "positioner_motion": str(ds.PositionerMotion) if "PositionerMotion" in ds else None,
        "table_motion": str(ds.TableMotion) if "TableMotion" in ds else None,
        "table_height_mm": _number(ds, "TableHeight", record),
        "table_lateral_increment_mm": _number(ds, "TableLateralIncrement", record),
        "table_longitudinal_increment_mm": _number(ds, "TableLongitudinalIncrement", record),
        "rows": _number(ds, "Rows", record, integer=True),
        "columns": _number(ds, "Columns", record, integer=True),
        "phase_inferred_from_name": False,
    })
    normal = _normal(record["image_orientation_patient"], record)
    required = {
        "FrameOfReferenceUID": record["frame_of_reference_uid"],
        "ImagePositionPatient": record["image_position_patient"],
        "ImageOrientationPatient": record["image_orientation_patient"] if normal is not None else None,
        "PositionerPrimaryAngle": record["primary_angle_deg"],
        "PositionerSecondaryAngle": record["secondary_angle_deg"],
        "DistanceSourceToDetector": record["source_to_detector_mm"],
        "DistanceSourceToPatient": record["source_to_patient_mm"],
        "ImagerPixelSpacing": record["imager_pixel_spacing_row_col_mm"],
        # Incremental motion is not an absolute CT-to-table transform.
        "absolute_table_position": None,
    }
    record["missing_geometry"] = [name for name, value in required.items() if value is None or value == ""]
    ecg = {name: _number(ds, name, record) for name in _ECG_NUMERIC if name in ds}
    if "RWaveTimeVector" in ds:
        ecg["RWaveTimeVector"] = _vector(ds, "RWaveTimeVector", record)
    if "BeatRejectionFlag" in ds:
        ecg["BeatRejectionFlag"] = str(ds.BeatRejectionFlag)
    record["ecg_tags"] = ecg
    record["waveform_sequence_present"] = "WaveformSequence" in ds
    record["has_ecg"] = any(v is not None for v in ecg.values()) or record["waveform_sequence_present"]


def _ct_series(uid: str, instances: list[dict]) -> dict:
    reference = next((r["image_orientation_patient"] for r in instances
                      if r["image_orientation_patient"] is not None
                      and not any(i["field"] == "ImageOrientationPatient" for i in r["metadata_issues"])), None)
    normal = _normal(reference, {"metadata_issues": []})
    for record in instances:
        iop, ipp = record["image_orientation_patient"], record["image_position_patient"]
        consistent = normal is not None and iop is not None and np.allclose(iop, reference, atol=1e-5, rtol=0)
        with np.errstate(over="ignore", invalid="ignore"):
            position = float(np.dot(ipp, normal)) if consistent and ipp is not None else None
        if position is not None and not math.isfinite(position):
            _issue(record, "ImagePositionPatient", "normal projection overflow; excluded from spacing")
            position = None
        record["projected_position_mm"] = position
        if iop is not None and reference is not None and not consistent:
            _issue(record, "ImageOrientationPatient", "inconsistent series orientation; excluded from spacing")
    instances.sort(key=lambda r: (r["projected_position_mm"] is None,
                                 r["projected_position_mm"] or 0, r["relative_path"]))
    groups: list[dict] = []
    for record in instances:
        position = record["projected_position_mm"]
        if position is None:
            continue
        if groups and abs(groups[-1]["projected_position_mm"] - position) <= 1e-4:
            groups[-1]["relative_paths"].append(record["relative_path"])
            groups[-1]["count"] += 1
        else:
            groups.append({"projected_position_mm": position, "relative_paths": [record["relative_path"]], "count": 1})
    gaps = []
    for a, b in zip(groups, groups[1:]):
        gap = b["projected_position_mm"] - a["projected_position_mm"]
        if math.isfinite(gap):
            gaps.append(gap)
        else:
            _issue(instances[0], "projected_spacing_mm", "spacing overflow; interval excluded")
    ordered = sorted(gaps)
    mid = len(ordered) // 2
    median = (ordered[mid] if len(ordered) % 2 else ordered[mid - 1] / 2 + ordered[mid] / 2) if ordered else None
    return {
        "series_instance_uid": uid, "modality": "CT", "file_count": len(instances), "instances": instances,
        "slice_normal_lps": normal.tolist() if normal is not None else None,
        "unique_position_count": len(groups),
        "duplicate_position_count": sum(g["count"] - 1 for g in groups),
        "duplicate_positions": [g for g in groups if g["count"] > 1],
        "spacing_mm": {"median": median,
                       "min": min(gaps) if gaps else None, "max": max(gaps) if gaps else None},
        "spacing_basis": "sorted dot(ImagePositionPatient, normalized cross(IOP row, IOP column)); unique positions, 0.0001 mm tolerance",
        "geometry_complete": all(r["projected_position_mm"] is not None for r in instances),
        "slice_thickness_mm": sorted({r["slice_thickness_mm"] for r in instances if r["slice_thickness_mm"] is not None}),
        "convolution_kernels": sorted({k for r in instances for k in r["convolution_kernels"]}),
        "dynamic_ct_confirmed": False,
    }


def _error(errors: list, scope: str, relative: str, stage: str, exc: Exception) -> None:
    # Exception text can contain arbitrary metadata/PHI. Keep only its technical class.
    errors.append({"scope": scope, "relative_path": relative, "stage": stage, "error_type": type(exc).__name__})


def _file_record(path: Path, relative: str, errors: list, scope: str) -> dict:
    result = {"relative_path": relative, "exists": path.exists(), "sha256": None,
              "byte_length": None, "read_status": "missing"}
    if not result["exists"]:
        return result
    try:
        digest = hashlib.sha256()
        byte_length = 0
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
                byte_length += len(chunk)
        result.update(sha256=digest.hexdigest(), byte_length=byte_length, read_status="read")
    except OSError as exc:
        result["read_status"] = "error"
        _error(errors, scope, relative, "hash", exc)
    return result


def _candidate(path: Path) -> bool:
    return any(part.lower().endswith("_candidate") for part in path.parts)


def _strict_constant(value: str) -> None:
    raise ValueError("nonfinite JSON number")


def _finite_json(value: Any) -> bool:
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, dict):
        return all(_finite_json(v) for v in value.values())
    if isinstance(value, list):
        return all(_finite_json(v) for v in value)
    return True


def _accepted(data: dict | None) -> bool:
    if not data or data.get("quality_gate_passed") is not True:
        return False
    return all(data.get(field) is None or data[field] in ("accepted", "passed", "ok")
               for field in ("quality_status", "status"))


def _quality_summary(doc: dict) -> dict:
    """Keep technical scalar fields only, never arbitrary nested metadata."""
    summary: dict = {"metadata_issues": []}
    for field in _QUALITY_FIELDS:
        if field not in doc:
            continue
        value = doc[field]
        valid = value is None or type(value) in (bool, int, float, str)
        if field == "shape_frame_row_column":
            valid = isinstance(value, list) and len(value) == 3 and all(type(n) is int and n > 0 for n in value)
        if not valid:
            summary["metadata_issues"].append({"field": field, "reason": "invalid quality metadata type"})
            continue
        summary[field] = value
    for key in ("groups", "runs"):
        if isinstance(doc.get(key), dict):
            summary[f"{key}_status"] = {
                k: v.get("status") if isinstance(v.get("status"), str) else None
                for k, v in doc[key].items() if isinstance(v, dict) and k in (*_RUNS, "rca", "lca")
            }
    if isinstance(doc.get("source_record"), dict):
        value = doc["source_record"].get("duration_ms")
        summary["manifest_duration_ms"] = value if type(value) in (float, int) else None
    return summary


def _asset_audit(root: Path, errors: list) -> tuple[dict, dict]:
    records, documents = {}, {}
    for relative in _ASSETS:
        path = root / relative
        # Never follow an allowlisted pathname into a candidate or a symlinked history.
        unsafe = _candidate(path) or any(p.is_symlink() for p in (path, *path.parents))
        if unsafe:
            records[relative] = {"relative_path": relative, "exists": path.exists(), "sha256": None,
                                 "byte_length": None, "read_status": "excluded", "quality_summary": {"status": "non-authoritative"}}
            continue
        record = _file_record(path, relative, errors, "assets")
        record["required"] = relative != "cine_manifest.json"
        record["quality_summary"] = {"status": "not_assessed" if record["exists"] else "missing"}
        records[relative] = record
        if record["read_status"] != "read":
            continue
        try:
            if path.suffix == ".json":
                with path.open(encoding="utf-8-sig") as stream:
                    doc = json.load(stream, parse_constant=_strict_constant)
                if not isinstance(doc, dict) or not _finite_json(doc):
                    raise ValueError("invalid JSON document")
                documents[relative] = doc
                # Do not copy arbitrary JSON strings, provenance paths, or PHI into the audit.
                record["quality_summary"] = _quality_summary(doc)
            elif relative.endswith(".nii.gz"):
                import SimpleITK as sitk
                reader = sitk.ImageFileReader()
                reader.SetFileName(str(path))
                reader.ReadImageInformation()
                spacing = list(reader.GetSpacing())
                if not all(math.isfinite(v) and v > 0 for v in spacing):
                    raise ValueError("invalid NIfTI spacing")
                record["quality_summary"] = {"status": "header_read", "shape": list(reader.GetSize()),
                                             "spacing_mm": spacing, "independent_3d_truth": False}
        except Exception as exc:
            documents.pop(relative, None)
            record["read_status"] = "error"
            record["quality_summary"] = {"status": "unreadable"}
            _error(errors, "assets", relative, "metadata", exc)
    for run in _RUNS:
        manifest = documents.get(f"all_series/{run}/series_manifest.json", {})
        shape = manifest.get("shape_frame_row_column")
        expected = math.prod(shape) if (isinstance(shape, list) and len(shape) == 3
                                      and all(type(n) is int and n > 0 for n in shape)
                                      and manifest.get("dtype") == "uint8") else None
        for suffix in ("cine_u8.raw", "temporal/temporal_centerlines_u8.raw"):
            record = records[f"all_series/{run}/{suffix}"]
            record["quality_summary"] = {
                "status": "missing" if not record["exists"] else "size_only",
                "expected_byte_length": expected,
                "byte_length_matches_manifest": record["byte_length"] == expected if expected is not None else None,
                "pixel_quality_assessed": False,
            }
    return records, documents


def _phase_issues(doc: dict) -> list[dict]:
    issues = []
    runs, groups = doc.get("runs"), doc.get("groups", {})
    if not isinstance(runs, dict) or any(not isinstance(runs.get(run), dict) for run in _RUNS):
        issues.append({"field": "runs", "reason": "both first-batch run records are required"})
    if not isinstance(groups, dict):
        issues.append({"field": "groups", "reason": "expected group mapping"})
    else:
        for group in groups.values():
            if (not isinstance(group, dict) or not isinstance(group.get("run_ids"), list)
                    or not all(isinstance(run, str) for run in group["run_ids"])):
                issues.append({"field": "groups.run_ids", "reason": "expected list of run identifiers"})
    return issues


def _phase_accepted(doc: dict | None) -> bool:
    if not doc or doc.get("status") != "accepted":
        return False
    if _phase_issues(doc):
        return False
    if "quality_gate_passed" in doc and doc["quality_gate_passed"] is not True:
        return False
    runs = doc.get("runs")
    if not isinstance(runs, dict):
        return False
    for run in _RUNS:
        item = runs.get(run)
        if not isinstance(item, dict) or item.get("status") != "accepted":
            return False
        if "quality_gate_passed" in item and item["quality_gate_passed"] is not True:
            return False
    groups = doc.get("groups", {})
    if not isinstance(groups, dict):
        return False
    for group in groups.values():
        if not isinstance(group, dict):
            return False
        if set(group.get("run_ids", ())) & set(_RUNS) and group.get("status") != "accepted":
            return False
    return True


def _source_links(source_files: list[dict], documents: dict, assets: dict) -> dict:
    """Bind cached cine provenance to actual headers/hashes, never old paths."""
    links = {}
    for run in _RUNS:
        manifest_key = f"all_series/{run}/series_manifest.json"
        manifest = documents.get(manifest_key)
        link = {"status": "missing_manifest", "source_relative_path": None,
                "checks": {"sop_instance_uid": None, "series_instance_uid": None, "sha256": None},
                "claimed_sha256": None, "actual_sha256": None,
                "path_policy": "matched by SOP and series identity in case_dir; manifest absolute paths ignored"}
        links[run] = link
        if manifest is not None:
            claimed = manifest.get("source_record")
            if not isinstance(claimed, dict):
                claimed = {}
            sop, series = claimed.get("sop_instance_uid"), claimed.get("series_instance_uid")
            sha = claimed.get("sha256")
            if not isinstance(sop, str) or not sop or not isinstance(series, str) or not series:
                link["status"] = "missing_identity"
            else:
                sources = [r for r in source_files if r.get("read_status") == "read"]
                exact = [r for r in sources if r.get("sop_instance_uid") == sop and r.get("series_instance_uid") == series]
                candidates = exact or [r for r in sources if r.get("sop_instance_uid") == sop]
                candidates = candidates or [r for r in sources if r.get("series_instance_uid") == series]
                if not candidates:
                    link["status"] = "missing_source"
                elif len(candidates) != 1:
                    link["status"] = "ambiguous_source"
                else:
                    source = candidates[0]
                    valid_hash = (isinstance(sha, str) and len(sha) == 64
                                  and all(c in "0123456789abcdefABCDEF" for c in sha))
                    checks = {"sop_instance_uid": source.get("sop_instance_uid") == sop,
                              "series_instance_uid": source.get("series_instance_uid") == series,
                              "sha256": source["sha256"] == sha.lower() if valid_hash else None}
                    link.update(source_relative_path=source["relative_path"], checks=checks,
                                claimed_sha256=sha.lower() if valid_hash else None,
                                actual_sha256=source["sha256"], source_modality=source.get("modality"))
                    if source.get("modality") != "XA" or any(value is False for value in checks.values()):
                        link["status"] = "mismatch"
                    elif not valid_hash:
                        link["status"] = "missing_hash"
                    else:
                        link["status"] = "verified"
        assets[manifest_key]["quality_summary"]["source_link_status"] = link["status"]
        for suffix in ("cine_u8.raw", "temporal/temporal_centerlines_u8.raw"):
            assets[f"all_series/{run}/{suffix}"]["quality_summary"]["source_link_status"] = link["status"]
    return links


def audit_case(case_dir: Path, assets_root: Path) -> dict:
    """Return a strict-JSON-compatible audit without writing either input tree.

    Existing empty inputs are insufficient, not exceptional. Each source file is
    hashed before header parsing; failed headers stay in source_files/read_errors.
    CT/XA series are grouped by SeriesInstanceUID, never by descriptions or kernels.
    """
    case_dir, assets_root = Path(case_dir), Path(assets_root)
    report: dict = {
        "schema_version": 1, "case_id": case_dir.name, "status": "insufficient",
        "ct_series": [], "xa_series": [], "source_files": [], "other_series": [],
        "assets": {}, "read_errors": [], "metadata_issues": [],
        "gates": {"absolute_geometry_accepted": False, "phase_sync_accepted": False,
                  "true_3d_reference_available": False, "clinical_validation": False},
        "patient_true_3d_tre_mm": None, "dynamic_ct_confirmed": False,
        "limitations": [
            "Research metadata audit only; not clinical acceptance or navigation validation.",
            "Dynamic CT and an independent 3D patient reference are not confirmed; true patient TRE is null.",
            "Reconstruction kernels, duplicate slices, filenames and sequence order do not establish cardiac phase.",
            "Existing depth-prior vessel reconstructions and heart masks are not independent 3D truth.",
            "Hashes freeze observed bytes; concurrent source changes during an audit are not a transactional snapshot.",
            "Enhanced multi-frame CT functional-group geometry is not expanded; no dynamic CT is inferred.",
        ],
    }
    errors = report["read_errors"]
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    if not case_dir.is_dir() or _candidate(case_dir):
        _error(errors, "case", ".", "scan", FileNotFoundError())
    else:
        def walk_error(exc: OSError) -> None:
            _error(errors, "case", ".", "scan", exc)

        for parent, directories, names in os.walk(case_dir, onerror=walk_error, followlinks=False):
            directories[:] = sorted(d for d in directories if not d.lower().endswith("_candidate")
                                    and not (Path(parent) / d).is_symlink())
            for name in sorted(names):
                path = Path(parent) / name
                relative = path.relative_to(case_dir).as_posix()
                if path.is_symlink():
                    _error(errors, "case", relative, "scan", ValueError())
                    continue
                record = _file_record(path, relative, errors, "case")
                report["source_files"].append(record)
                if record["read_status"] != "read":
                    continue
                record["metadata_issues"] = []
                try:
                    with warnings.catch_warnings(record=True) as caught:
                        warnings.simplefilter("always")
                        ds = pydicom.dcmread(path, stop_before_pixels=True, force=False, specific_tags=_TAGS)
                        uid = str(ds.SeriesInstanceUID) if "SeriesInstanceUID" in ds else ""
                        modality = str(ds.Modality) if "Modality" in ds else ""
                        record.update(series_instance_uid=uid or None, modality=modality or None,
                                      sop_instance_uid=str(ds.SOPInstanceUID) if "SOPInstanceUID" in ds else None,
                                      series_number=_number(ds, "SeriesNumber", record, integer=True),
                                      instance_number=_number(ds, "InstanceNumber", record, integer=True))
                        if not uid or not modality:
                            _issue(record, "SeriesInstanceUID/Modality", "missing grouping identity; source retained")
                            record["read_status"] = "invalid_metadata"
                        elif modality == "CT":
                            _ct_instance(ds, record)
                        elif modality == "XA":
                            _xa_instance(ds, record)
                        if caught:
                            _issue(record, "DICOM", "parser warning; raw warning text suppressed for privacy")
                    if uid and modality:
                        grouped[(modality, uid)].append(record)
                except Exception as exc:
                    record["read_status"] = "error"
                    _error(errors, "case", relative, "dicom_metadata", exc)
    for (modality, uid), instances in sorted(grouped.items()):
        if modality == "CT":
            report["ct_series"].append(_ct_series(uid, instances))
        else:
            record = {"series_instance_uid": uid, "modality": modality,
                      "file_count": len(instances), "instances": instances}
            report["xa_series" if modality == "XA" else "other_series"].append(record)
    assets, documents = _asset_audit(assets_root, errors)
    report["assets"] = assets
    report["source_links"] = _source_links(report["source_files"], documents, assets)
    phase = documents.get(_PHASE)
    if phase is not None:
        assets[_PHASE]["quality_summary"]["metadata_issues"].extend(_phase_issues(phase))
    report["gates"]["absolute_geometry_accepted"] = all(
        _accepted(documents.get(f"all_series/{run}/{run}_calibration.json")) for run in _RUNS
    )
    report["gates"]["phase_sync_accepted"] = _phase_accepted(phase)
    for record in report["source_files"]:
        report["metadata_issues"].extend({"relative_path": record["relative_path"], **issue}
                                         for issue in record.get("metadata_issues", []))
    for relative, record in assets.items():
        report["metadata_issues"].extend({"scope": "assets", "relative_path": relative, **issue}
                                         for issue in record["quality_summary"].get("metadata_issues", []))
    if not report["ct_series"]:
        report["limitations"].append("No readable CT series available.")
    if not report["xa_series"]:
        report["limitations"].append("No readable XA series available.")
    if any(link["status"] != "verified" for link in report["source_links"].values()):
        report["limitations"].append("One or more first-batch cached cines have missing, ambiguous, or mismatched source links; do not treat them as source-verified.")
    if any(r["required"] and r["read_status"] != "read" for r in assets.values() if "required" in r):
        report["limitations"].append("One or more authoritative assets are missing or unreadable.")
    if report["ct_series"] and report["xa_series"] and not errors and not report["metadata_issues"]:
        report["status"] = "audited"
    # Defense against nonfinite values hidden in otherwise legal JSON exponents.
    json.dumps(report, allow_nan=False)
    return report
