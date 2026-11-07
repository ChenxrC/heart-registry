from __future__ import annotations

import hashlib
import json
import os
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable, Mapping

from ctdr_pipeline.config import PipelineConfig
from ctdr_pipeline.evaluation.metrics import CaseMetrics
from ctdr_pipeline.evaluation.quality import QualityDecision
from ctdr_pipeline.evaluation.report import CaseReport, write_case_report, write_run_summary
from ctdr_pipeline.schemas import CaseStatus


STAGE_ORDER = ("pptx", "dicom_linkage", "registration3d", "evaluation", "report")
STAGE_DEPENDENCIES = {
    "pptx": (),
    "dicom_linkage": ("pptx",),
    "registration3d": ("dicom_linkage",),
    "evaluation": ("registration3d",),
    "report": ("evaluation",),
}


class CaseTerminalError(RuntimeError):
    def __init__(self, status: CaseStatus, message: str) -> None:
        super().__init__(message)
        self.status = status


@dataclass(frozen=True)
class CaseRunRequest:
    case_id: str
    bundle_path: Path
    dicom_manifest_path: Path
    output_dir: Path
    config: PipelineConfig
    device: str | None = None
    handlers: Mapping[str, "StageHandler"] | None = None


@dataclass(frozen=True)
class BatchRunRequest:
    case_ids: tuple[str, ...]
    bundle_path: Path
    dicom_manifest_path: Path
    output_dir: Path
    config: PipelineConfig
    handlers: Mapping[str, "StageHandler"] | None = None


@dataclass
class StageContext:
    request: CaseRunRequest
    stage: str
    stage_dir: Path
    state: dict[str, dict[str, Any]]


StageHandler = Callable[[StageContext], dict[str, Any]]


@dataclass(frozen=True)
class RunSummary:
    reports: tuple[CaseReport, ...]
    exit_code: int


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, CaseStatus):
        return value.value
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if hasattr(value, "model_dump"):
        return _json_safe(value.model_dump(mode="json"))
    return value


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        _json_safe(value), ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str
    ).encode("utf-8")


def _sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _hash_path(path: Path | None) -> str | None:
    if path is None:
        return None
    path = Path(path)
    if not path.exists():
        return "missing"
    digest = hashlib.sha256()
    if path.is_file():
        digest.update(path.read_bytes())
    else:
        for member in sorted(item for item in path.rglob("*") if item.is_file()):
            digest.update(member.relative_to(path).as_posix().encode("utf-8"))
            with member.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(_json_safe(value), ensure_ascii=False, sort_keys=True, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _stage_base_inputs(request: CaseRunRequest, stage: str) -> dict[str, Any]:
    inputs: dict[str, Any] = {
        "case_id": request.case_id,
        "stage": stage,
        "pipeline_implementation_hash": _hash_path(Path(__file__)),
    }
    if stage == "pptx":
        inputs["bundle_hash"] = _hash_path(request.bundle_path)
    elif stage == "dicom_linkage":
        inputs["manifest_hash"] = _hash_path(request.dicom_manifest_path)
        inputs["dicom_overrides_hash"] = _hash_path(request.config.dicom_overrides)
    elif stage == "registration3d":
        inputs["registration_config"] = request.config.registration.model_dump(mode="json")
        inputs["device"] = request.device or request.config.gpus[0]
    elif stage == "evaluation":
        inputs["quality_config"] = request.config.quality.model_dump(mode="json")
    elif stage == "report":
        inputs["anonymize_reports"] = request.config.anonymize_reports
    return inputs


def _stage_input_hash(
    request: CaseRunRequest,
    stage: str,
    records: dict[str, dict[str, Any]],
) -> str:
    value = _stage_base_inputs(request, stage)
    value["dependencies"] = {
        dependency: records[dependency]["output_hash"]
        for dependency in STAGE_DEPENDENCIES[stage]
    }
    return _sha256(value)


def _load_cached_stage(stage_dir: Path, input_hash: str) -> tuple[dict[str, Any], dict[str, Any]] | None:
    marker_path = stage_dir / "stage.json"
    payload_path = stage_dir / "payload.json"
    if not marker_path.exists() or not payload_path.exists():
        return None
    try:
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
        payload = json.loads(payload_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if marker.get("status") != "success" or marker.get("input_hash") != input_hash:
        return None
    if marker.get("output_hash") != _sha256(payload):
        return None
    return payload, marker


def _default_missing_handler(context: StageContext) -> dict[str, Any]:
    raise RuntimeError(
        f"no default implementation is registered for stage {context.stage}; "
        "supply explicit stage handlers"
    )


def _bundle_root(path: Path) -> Path:
    return path if path.is_dir() else path.parent


def _default_pptx_stage(context: StageContext) -> dict[str, Any]:
    import csv

    import numpy as np
    from PIL import Image

    from ctdr_pipeline.bundle import load_bundle

    root = _bundle_root(context.request.bundle_path)
    index = load_bundle(root)
    record = next((item for item in index.cases if item.case_id == context.request.case_id), None)
    if record is None:
        raise CaseTerminalError(
            CaseStatus.SKIPPED_INCOMPLETE_PPTX_PAIR, "case_not_present_in_pptx_bundle"
        )
    if record.pair_status != "ready" or record.image_count != 2:
        raise CaseTerminalError(
            CaseStatus.SKIPPED_INCOMPLETE_PPTX_PAIR,
            ";".join(record.reasons) or "incomplete_pptx_pair",
        )
    if record.registration_status != "accepted" or record.matrix_3x3 is None:
        raise CaseTerminalError(
            CaseStatus.REVIEW_REQUIRED_2D,
            ";".join(record.reasons) or "pptx_affine_not_accepted",
        )
    landmark_rows: list[dict[str, str]] = []
    landmarks_path = root / "landmarks.csv"
    if landmarks_path.exists():
        with landmarks_path.open("r", encoding="utf-8-sig", newline="") as handle:
            landmark_rows = [
                row for row in csv.DictReader(handle) if row.get("case_id") == record.case_id
            ]
    source_points = [
        [float(row["source_x_px"]), float(row["source_y_px"])] for row in landmark_rows
    ]
    target_points = [
        [float(row["target_x_px"]), float(row["target_y_px"])] for row in landmark_rows
    ]
    residuals = np.asarray(
        [float(row["residual_px"]) for row in landmark_rows], dtype=np.float64
    )
    inliers = [str(row.get("inlier", "")).lower() == "true" for row in landmark_rows]
    target_image = root / str(record.target_image)
    with Image.open(target_image) as image:
        diagonal = float(np.hypot(image.width, image.height))
    matrix = np.asarray(record.matrix_3x3, dtype=np.float64).reshape(3, 3)
    gated = residuals[np.asarray(inliers, dtype=bool)] if residuals.size else residuals
    if gated.size == 0:
        gated = residuals
    metrics = {
        "rms_px": float(np.sqrt(np.mean(np.square(gated)))) if gated.size else None,
        "p95_px": float(np.percentile(gated, 95)) if gated.size else None,
        "normalized_rms": float(np.sqrt(np.mean(np.square(gated))) / diagonal) if gated.size else None,
        "normalized_p95": float(np.percentile(gated, 95) / diagonal) if gated.size else None,
        "inlier_ratio": float(np.mean(inliers)) if inliers else None,
        "condition_number": float(np.linalg.cond(matrix[:2, :2])),
        "determinant": float(np.linalg.det(matrix[:2, :2])),
    }
    return {
        "bundle_case": record.model_dump(mode="json"),
        "matrix_3x3": matrix.reshape(-1).tolist(),
        "source_points": source_points,
        "target_points": target_points,
        "image_diagonal": diagonal,
        "metrics": metrics,
        "metadata": {"bundle_source_sha256": index.source_sha256},
    }


def _default_dicom_linkage_stage(context: StageContext) -> dict[str, Any]:
    from ctdr_pipeline.bundle import BundleCaseRecord
    from ctdr_pipeline.dicom.linker import link_case_images
    from ctdr_pipeline.dicom.scanner import load_dicom_manifest
    from ctdr_pipeline.dicom.selector import apply_dicom_overrides, rank_series

    manifest = load_dicom_manifest(context.request.dicom_manifest_path)
    try:
        case = manifest.case(context.request.case_id)
    except KeyError as exc:
        raise CaseTerminalError(
            CaseStatus.SKIPPED_CASE_DIRECTORY_MISSING, "case_directory_missing"
        ) from exc
    selection = apply_dicom_overrides(
        rank_series(case), context.request.config.dicom_overrides
    )
    if selection.ct_series_uid is None:
        raise CaseTerminalError(CaseStatus.SKIPPED_CT_MISSING, "ct_series_missing")
    if selection.xa_series_uid is None:
        raise CaseTerminalError(CaseStatus.SKIPPED_XA_MISSING, "xa_series_missing")
    if selection.localizer_series_uid is None:
        raise CaseTerminalError(
            CaseStatus.REVIEW_REQUIRED_DICOM_AMBIGUITY, "ct_localizer_missing"
        )
    linkage = link_case_images(
        _bundle_root(context.request.bundle_path),
        BundleCaseRecord.model_validate(context.state["pptx"]["bundle_case"]),
        Path(manifest.data_root),
        case,
        selection,
    )
    if linkage.status != "accepted":
        raise CaseTerminalError(
            CaseStatus.REVIEW_REQUIRED_DICOM_AMBIGUITY,
            ";".join(linkage.reasons) or f"dicom_linkage_{linkage.status}",
        )
    return {
        "data_root": manifest.data_root,
        "case": case.model_dump(mode="json"),
        "selection": selection.model_dump(mode="json"),
        "linkage": linkage.model_dump(mode="json"),
        "metadata": {
            "source_confidence": linkage.source.confidence,
            "target_confidence": linkage.target.confidence,
        },
    }


def _orient_projection_frame(image: Any, orientation: str) -> Any:
    import numpy as np

    if orientation == "identity":
        return image
    if orientation == "rotate180":
        return np.rot90(image, 2).copy()
    if orientation == "horizontal_flip":
        return np.fliplr(image).copy()
    if orientation == "vertical_flip":
        return np.flipud(image).copy()
    raise ValueError(f"unknown projection orientation: {orientation}")


def _default_registration_stage(context: StageContext) -> dict[str, Any]:
    import cv2
    import numpy as np
    import pydicom
    import torch

    from ctdr_pipeline.dicom.geometry import projection_geometry
    from ctdr_pipeline.dicom.linker import DicomLinkageResult, normalize_u8
    from ctdr_pipeline.dicom.scanner import CaseDicomManifest
    from ctdr_pipeline.dicom.selector import SeriesSelection
    from ctdr_pipeline.registration.affine2d import estimate_affine_ransac
    from ctdr_pipeline.registration.diffdrr_runner import (
        DiffDrrRenderer,
        Registration3DRequest,
        run_diffdrr_registration,
    )
    from ctdr_pipeline.registration.initializer import generate_pose_seeds
    from ctdr_pipeline.registration.volume import load_ct_volume

    linkage_payload = context.state["dicom_linkage"]
    data_root = Path(linkage_payload["data_root"])
    case = CaseDicomManifest.model_validate(linkage_payload["case"])
    selection = SeriesSelection.model_validate(linkage_payload["selection"])
    linkage = DicomLinkageResult.model_validate(linkage_payload["linkage"])
    if linkage.target.selected is None or selection.ct_series_uid is None:
        raise CaseTerminalError(
            CaseStatus.REVIEW_REQUIRED_DICOM_AMBIGUITY, "selected_projection_or_ct_missing"
        )
    ct_volume = load_ct_volume(case.series_by_uid(selection.ct_series_uid), data_root)
    selected = linkage.target.selected
    dataset = pydicom.dcmread(selected.path, force=True)
    pixels = np.asarray(dataset.pixel_array)
    frames = pixels[None] if pixels.ndim == 2 else pixels
    if selected.frame_index >= len(frames):
        raise IndexError("linked XA frame index is outside the DICOM pixel array")
    target = normalize_u8(
        frames[selected.frame_index],
        str(getattr(dataset, "PhotometricInterpretation", "MONOCHROME2")),
    ).astype(np.float32) / 255.0
    target = _orient_projection_frame(target, selected.orientation)
    settings = context.request.config.registration
    evaluation_height = int(max(settings.resolutions))
    evaluation_width = max(16, int(round(evaluation_height * target.shape[1] / target.shape[0])))
    if target.shape != (evaluation_height, evaluation_width):
        target = cv2.resize(
            target,
            (evaluation_width, evaluation_height),
            interpolation=cv2.INTER_AREA,
        )
    geometry = projection_geometry(dataset)
    if geometry.sid_mm is None or geometry.pixel_spacing_row_mm is None:
        raise CaseTerminalError(CaseStatus.REJECTED_GEOMETRY, "projection_geometry_incomplete")

    pptx = context.state["pptx"]
    source_points = np.asarray(pptx["source_points"], dtype=np.float64)
    target_points = np.asarray(pptx["target_points"], dtype=np.float64)
    affine = estimate_affine_ransac(
        source_points,
        target_points,
        image_diagonal=float(pptx["image_diagonal"]),
        ransac_threshold_px=max(3.0, float(pptx["image_diagonal"]) * 0.02),
    )
    seeds = generate_pose_seeds(
        linkage,
        affine,
        geometry,
        seed=context.request.config.registration.random_seed,
        count=context.request.config.registration.multistart_count,
    )
    device = context.request.device or context.request.config.gpus[0]
    renderer = DiffDrrRenderer.from_ct_volume(
        ct_volume,
        geometry,
        device=device,
        reverse_x_axis=True,
        checkpoint_gradients=True,
    )
    result = run_diffdrr_registration(
        Registration3DRequest(
            renderer=renderer,
            target=torch.from_numpy(target),
            roi=torch.ones_like(torch.from_numpy(target)),
            seeds=seeds,
            resolutions=tuple(settings.resolutions),
            coarse_iterations=settings.coarse_iterations,
            refine_iterations=settings.refine_iterations,
            refine_top_k=settings.refine_top_k,
            rotation_learning_rate_deg=settings.rotation_learning_rate_deg,
            translation_learning_rate_mm=settings.translation_learning_rate_mm,
            rotation_bounds_deg=tuple(tuple(value) for value in settings.rotation_bounds_deg),
            translation_bounds_mm=tuple(tuple(value) for value in settings.translation_bounds_mm),
            device=device,
        )
    )
    if result.selected is None:
        raise RuntimeError(result.failed_reason or "DiffDRR returned no physical candidate")
    arrays_dir = context.stage_dir / "arrays"
    arrays_dir.mkdir(parents=True, exist_ok=True)
    target_path = arrays_dir / "target.npy"
    drr_path = arrays_dir / "final_drr.npy"
    np.save(target_path, target.astype(np.float32))
    final_drr = result.selected.rendered_image.squeeze().numpy().astype(np.float32)
    np.save(drr_path, final_drr)
    candidates: list[dict[str, Any]] = []
    for candidate in result.candidates:
        candidates.append(
            {
                "name": candidate.name,
                "diagnostic_only": candidate.diagnostic_only,
                "provenance": candidate.provenance,
                "initial_pose": list(candidate.initial_pose),
                "final_pose": list(candidate.final_pose),
                "initial_score": candidate.initial_score,
                "final_score": candidate.final_score,
                "improvement": candidate.improvement,
                "objective_components": candidate.objective_components,
                "boundary_flags": candidate.boundary_flags,
                "failed_reason": candidate.failed_reason,
                "trace": candidate.trace,
            }
        )
    return {
        "selected_name": result.selected.name,
        "selected_pose": list(result.selected.final_pose),
        "selected_initial_score": result.selected.initial_score,
        "selected_final_score": result.selected.final_score,
        "selected_boundary_flags": result.selected.boundary_flags,
        "target_array": str(target_path),
        "drr_array": str(drr_path),
        "candidates": candidates,
        "geometry": {
            "rows": geometry.rows,
            "columns": geometry.columns,
            "pixel_spacing_row_mm": geometry.pixel_spacing_row_mm,
            "pixel_spacing_col_mm": geometry.pixel_spacing_col_mm,
            "sid_mm": geometry.sid_mm,
            "sod_mm": geometry.sod_mm,
            "primary_angle_deg": geometry.primary_angle_deg,
            "secondary_angle_deg": geometry.secondary_angle_deg,
            "missing_fields": geometry.missing_fields,
        },
        "metadata": result.metadata,
    }


def _normalized_ncc(first: Any, second: Any) -> float:
    import numpy as np

    left = np.asarray(first, dtype=np.float64).reshape(-1)
    right = np.asarray(second, dtype=np.float64).reshape(-1)
    left -= left.mean()
    right -= right.mean()
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    return float(np.dot(left, right) / denominator) if denominator > 0 else 0.0


def _default_evaluation_stage(context: StageContext) -> dict[str, Any]:
    import numpy as np

    from ctdr_pipeline.evaluation.metrics import MetricInputs, calculate_case_metrics
    from ctdr_pipeline.evaluation.quality import decide_quality

    pptx = context.state["pptx"]
    linkage_payload = context.state["dicom_linkage"]["linkage"]
    registration = context.state["registration3d"]
    target = np.load(registration["target_array"])
    drr = np.load(registration["drr_array"])
    if target.shape != drr.shape:
        raise ValueError("native DRR and XA target shapes differ")
    nonzero = drr[np.isfinite(drr) & (drr > 0)]
    drr_threshold = float(np.percentile(nonzero, 2)) if nonzero.size else 0.0
    target_threshold = float(np.percentile(target[np.isfinite(target)], 1))
    drr_fov = np.isfinite(drr) & (drr > drr_threshold)
    target_fov = np.isfinite(target) & (target > target_threshold)

    # A flipped image may score better even when optimization converges. Treat this as
    # orientation evidence and fail closed rather than silently changing coordinates.
    identity_ncc = _normalized_ncc(drr, target)
    vertical_ncc = _normalized_ncc(np.flipud(drr), target)
    horizontal_ncc = _normalized_ncc(np.fliplr(drr), target)
    vertical_flip = vertical_ncc > identity_ncc + 0.05
    mirror = horizontal_ncc > identity_ncc + 0.05

    physical = [
        candidate
        for candidate in registration["candidates"]
        if not candidate["diagnostic_only"] and candidate["failed_reason"] is None
    ]
    physical.sort(key=lambda candidate: candidate["final_score"], reverse=True)
    top = physical[: context.request.config.registration.refine_top_k]
    poses = np.asarray([candidate["final_pose"] for candidate in top], dtype=np.float64)
    scores = np.asarray([candidate["final_score"] for candidate in top], dtype=np.float64)
    source_link = linkage_payload["source"]
    target_link = linkage_payload["target"]
    pptx_metrics = pptx["metrics"]
    metrics = calculate_case_metrics(
        MetricInputs(
            source_points_2d=np.asarray(pptx["source_points"], dtype=np.float64),
            target_points_2d=np.asarray(pptx["target_points"], dtype=np.float64),
            affine_2d=np.asarray(pptx["matrix_3x3"], dtype=np.float64).reshape(3, 3),
            image_diagonal=float(pptx["image_diagonal"]),
            drr=drr,
            target=target,
            roi=np.ones_like(target),
            drr_fov_mask=drr_fov,
            target_fov_mask=target_fov,
            candidate_poses=poses if len(poses) else None,
            candidate_scores=scores if len(scores) else None,
            objective_initial=float(registration["selected_initial_score"]),
            objective_final=float(registration["selected_final_score"]),
            pptx_inlier_ratio=pptx_metrics["inlier_ratio"],
            affine_condition_number=pptx_metrics["condition_number"],
            dicom_confidence=min(float(source_link["confidence"]), float(target_link["confidence"])),
            dicom_candidate_margin=min(
                float(source_link["candidate_margin"]), float(target_link["candidate_margin"])
            ),
            mirror_detected=mirror,
            vertical_flip_detected=vertical_flip,
            parameter_boundary=bool(registration["selected_boundary_flags"]),
        )
    )
    decision = decide_quality(metrics, context.request.config.quality)
    difference_path = Path(registration["drr_array"]).with_name("absolute_difference.npy")
    np.save(difference_path, np.abs(drr - target).astype(np.float32))
    checkerboard = target.copy()
    block = max(4, target.shape[0] // 8)
    yy, xx = np.indices(target.shape)
    take_drr = ((yy // block) + (xx // block)) % 2 == 0
    checkerboard[take_drr] = drr[take_drr]
    checkerboard_path = difference_path.with_name("checkerboard.npy")
    np.save(checkerboard_path, checkerboard.astype(np.float32))
    from scipy import ndimage

    target_edges = np.hypot(ndimage.sobel(target, 0), ndimage.sobel(target, 1))
    drr_edges = np.hypot(ndimage.sobel(drr, 0), ndimage.sobel(drr, 1))
    target_edges /= max(float(target_edges.max()), 1e-6)
    drr_edges /= max(float(drr_edges.max()), 1e-6)
    edge_overlay = np.stack([target_edges, drr_edges, np.zeros_like(target_edges)], axis=-1)
    edge_overlay_path = difference_path.with_name("edge_overlay.npy")
    np.save(edge_overlay_path, edge_overlay.astype(np.float32))
    return {
        "status": decision.status.value,
        "accepted": decision.accepted,
        "failed_rules": list(decision.failed_rules),
        "metrics": asdict(metrics),
        "diagnostic_arrays": {
            "target_xa": registration["target_array"],
            "final_drr": registration["drr_array"],
            "absolute_difference": str(difference_path),
            "checkerboard": str(checkerboard_path),
            "edge_overlay": str(edge_overlay_path),
        },
        "orientation_diagnostics": {
            "identity_ncc": identity_ncc,
            "vertical_flip_ncc": vertical_ncc,
            "horizontal_flip_ncc": horizontal_ncc,
        },
    }


def _default_report_stage(context: StageContext) -> dict[str, Any]:
    return {"report_schema": "1.0", "status": context.state["evaluation"]["status"]}


def default_stage_handlers() -> dict[str, StageHandler]:
    return {
        "pptx": _default_pptx_stage,
        "dicom_linkage": _default_dicom_linkage_stage,
        "registration3d": _default_registration_stage,
        "evaluation": _default_evaluation_stage,
        "report": _default_report_stage,
    }


def _resolve_handlers(request: CaseRunRequest) -> dict[str, StageHandler]:
    resolved = default_stage_handlers()
    resolved.update(dict(request.handlers or {}))
    return {stage: resolved.get(stage, _default_missing_handler) for stage in STAGE_ORDER}


def _direct_decision(status: CaseStatus, reason: str | None = None) -> QualityDecision:
    return QualityDecision(
        status=status,
        accepted=status == CaseStatus.ACCEPTED_RESEARCH_CANDIDATE,
        failed_rules=(reason,) if reason else (),
    )


def _report_from_state(
    request: CaseRunRequest,
    state: dict[str, dict[str, Any]],
    terminal: tuple[CaseStatus, str] | None,
) -> CaseReport:
    if terminal is not None:
        status, reason = terminal
        decision = _direct_decision(status, reason)
    else:
        evaluation = state.get("evaluation", {})
        raw_status = evaluation.get("status", CaseStatus.REVIEW_REQUIRED_3D.value)
        try:
            status = CaseStatus(str(raw_status))
        except ValueError:
            status = CaseStatus.FAILED_RUNTIME
        decision = QualityDecision(
            status=status,
            accepted=status == CaseStatus.ACCEPTED_RESEARCH_CANDIDATE,
            failed_rules=tuple(str(item) for item in evaluation.get("failed_rules", [])),
        )
    evaluation = state.get("evaluation", {})
    try:
        metrics = CaseMetrics(**evaluation.get("metrics", {}))
    except TypeError:
        metrics = CaseMetrics(runtime_failure="invalid_evaluation_metric_schema")
    registration = state.get("registration3d", {})
    metadata = {
        "device": request.device,
        "stage_payloads": {
            name: payload.get("metadata", {}) for name, payload in state.items()
        },
    }
    diagnostic_images: dict[str, Any] = {}
    for name, path in evaluation.get("diagnostic_arrays", {}).items():
        try:
            import numpy as np

            diagnostic_images[str(name)] = np.load(path)
        except Exception:
            continue
    return CaseReport(
        case_id=request.case_id,
        decision=decision,
        metrics=metrics,
        candidates=list(registration.get("candidates", [])),
        metadata=metadata,
        diagnostic_images=diagnostic_images,
    )


def run_case(request: CaseRunRequest) -> CaseReport:
    if not request.case_id.isdigit():
        raise ValueError("case_id must contain only digits")
    output_dir = Path(request.output_dir)
    handlers = _resolve_handlers(request)
    state: dict[str, dict[str, Any]] = {}
    records: dict[str, dict[str, Any]] = {}
    terminal: tuple[CaseStatus, str] | None = None

    for stage in STAGE_ORDER:
        stage_dir = output_dir / "stages" / stage
        input_hash = _stage_input_hash(request, stage, records)
        cached = _load_cached_stage(stage_dir, input_hash)
        if cached is not None:
            payload, marker = cached
            state[stage] = payload
            records[stage] = marker
            continue
        started = _now()
        marker: dict[str, Any] = {
            "stage": stage,
            "input_hash": input_hash,
            "started_at": started,
            "status": "running",
            "warnings": [],
        }
        _atomic_json(stage_dir / "stage.json", marker)
        try:
            payload = handlers[stage](
                StageContext(request=request, stage=stage, stage_dir=stage_dir, state=state)
            )
            if not isinstance(payload, dict):
                raise TypeError("stage handlers must return dictionaries")
            payload = _json_safe(payload)
            output_hash = _sha256(payload)
            _atomic_json(stage_dir / "payload.json", payload)
            marker.update(
                {
                    "status": "success",
                    "output_hash": output_hash,
                    "ended_at": _now(),
                }
            )
            _atomic_json(stage_dir / "stage.json", marker)
            state[stage] = payload
            records[stage] = marker
        except CaseTerminalError as exc:
            marker.update(
                {
                    "status": "terminal",
                    "ended_at": _now(),
                    "exception": {"type": type(exc).__name__, "message": str(exc)},
                }
            )
            _atomic_json(stage_dir / "stage.json", marker)
            terminal = (exc.status, str(exc))
            break
        except Exception as exc:
            marker.update(
                {
                    "status": "failed",
                    "ended_at": _now(),
                    "exception": {
                        "type": type(exc).__name__,
                        "message": str(exc),
                        "traceback": "".join(traceback.format_exception(exc))[-8000:],
                    },
                }
            )
            _atomic_json(stage_dir / "stage.json", marker)
            terminal = (CaseStatus.FAILED_RUNTIME, f"{type(exc).__name__}: {exc}")
            break

    report = _report_from_state(request, state, terminal)
    write_case_report(output_dir, report)
    return report


def run_all(request: BatchRunRequest) -> RunSummary:
    output_dir = Path(request.output_dir)
    devices = request.config.gpus or ["cpu"]
    worker_devices: list[str] = []
    for device in devices:
        worker_devices.extend([device] * max(1, request.config.gpu_workers.get(device, 1)))
    worker_devices = worker_devices or ["cpu"]

    def submit_one(index: int, case_id: str) -> CaseReport:
        device = worker_devices[index % len(worker_devices)]
        return run_case(
            CaseRunRequest(
                case_id=case_id,
                bundle_path=request.bundle_path,
                dicom_manifest_path=request.dicom_manifest_path,
                output_dir=output_dir / "cases" / case_id,
                config=request.config,
                device=device,
                handlers=request.handlers,
            )
        )

    reports: list[CaseReport] = []
    with ThreadPoolExecutor(max_workers=len(worker_devices)) as executor:
        futures = {
            executor.submit(submit_one, index, case_id): case_id
            for index, case_id in enumerate(request.case_ids)
        }
        for future in as_completed(futures):
            case_id = futures[future]
            try:
                reports.append(future.result())
            except Exception as exc:
                report = CaseReport(
                    case_id=case_id,
                    decision=_direct_decision(
                        CaseStatus.FAILED_RUNTIME, f"batch_worker:{type(exc).__name__}: {exc}"
                    ),
                    metrics=CaseMetrics(runtime_failure=str(exc)),
                )
                write_case_report(output_dir / "cases" / case_id, report)
                reports.append(report)
    reports.sort(key=lambda item: item.case_id)
    write_run_summary(output_dir, reports)
    nonaccepted = any(not report.decision.accepted for report in reports)
    exit_code = 3 if request.config.fail_on_review and nonaccepted else 0
    return RunSummary(reports=tuple(reports), exit_code=exit_code)
