"""Temporally stabilize per-frame XA vessel probabilities with optical flow."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import string
from dataclasses import asdict, replace
from pathlib import Path, PurePosixPath
from typing import Any

import cv2
import numpy as np
from skimage import morphology

try:
    from .run_xa_vessel_segmentation import make_overlays, postprocess_probability
    from .xa_cine import read_cine, write_frames, write_json, write_mp4, write_raw_stack
    from .xa_atomic_output import build_then_publish_directory
    from .xa_temporal_repair import (
        RepairAudit, _anisotropic_temporal_majority, _motion_compensated_temporal_corridor,
        _temporal_fusion_evidence, build_vessel_evidence, constrained_seed_reconstruction,
        dense_flow, remap_with_flow, repair_short_gaps,
    )
    from .xa_vessel_topology import (
        GapCandidate, VesselGraph, extract_vessel_graph, find_gap_candidates,
        mask_topology_metrics, track_vessel_graphs,
    )
except ImportError:  # direct script execution
    from run_xa_vessel_segmentation import make_overlays, postprocess_probability
    from xa_cine import read_cine, write_frames, write_json, write_mp4, write_raw_stack
    from xa_atomic_output import build_then_publish_directory
    from xa_temporal_repair import (
        RepairAudit, _anisotropic_temporal_majority, _motion_compensated_temporal_corridor,
        _temporal_fusion_evidence, build_vessel_evidence, constrained_seed_reconstruction,
        dense_flow, remap_with_flow, repair_short_gaps,
    )
    from xa_vessel_topology import (
        GapCandidate, VesselGraph, extract_vessel_graph, find_gap_candidates,
        mask_topology_metrics, track_vessel_graphs,
    )

_BASELINE_RAW_TARGETS = (
    "temporal_probabilities_u8.raw",
    "temporal_support_u8.raw",
    "flow_confidence_u8.raw",
    "temporal_masks_u8.raw",
    "temporal_centerlines_u8.raw",
)
_BASELINE_FRAME_TARGETS = {
    "probabilities": "temporal_probability",
    "masks": "temporal_mask",
    "centerlines": "temporal_centerline",
    "overlays": "temporal_overlay",
}
_BASELINE_VIDEO_TARGET = "temporal_overlay.mp4"
_BASELINE_MANIFEST = "baseline_provenance.json"

_MANIFEST_TOP_LEVEL_KEYS = {
    "schema_version",
    "inputs",
    "baseline_targets",
    "diagnostics",
}
_INPUT_KEYS = {"dicom", "probabilities_u8.raw"}
_FILE_IDENTITY_KEYS = {"byte_length", "sha256"}
_FILE_TARGET_KEYS = {"kind", "byte_length", "sha256"}
_DIRECTORY_TARGET_KEYS = {"kind", "file_count", "files", "sha256"}
_DIRECTORY_FILE_KEYS = {"path", "byte_length", "sha256"}
_DIAGNOSTIC_KEYS = {"frames", "temporal_metrics", "quality_summary"}
_FRAME_DIAGNOSTIC_KEYS = {
    "frame_index",
    "forward_backward_error_p95_px",
    "valid_neighbor_count",
}
_TEMPORAL_METRIC_KEYS = {
    "frame_index",
    "warped_iou_to_previous",
    "relative_area_jump",
    "relative_centerline_jump",
    "quality_flag",
}
_QUALITY_SUMMARY_KEYS = {
    "median_warped_iou",
    "review_frames",
    "provisional_frames",
}
_TASK9_REVIEW_FRAME_INDICES = (0, 1, 2, 3, 13, 24, 25, 26, 27, 28, 29)


def _validated_temporal_paths(
    output_dir: Path, baseline_dir: Path | None
) -> tuple[Path, Path | None]:
    resolved_output = Path(output_dir).resolve(strict=False)
    output_key = os.path.normcase(str(resolved_output))
    published_key = os.path.normcase(str(resolved_output.parent / "temporal"))
    candidate_key = os.path.normcase(
        str(resolved_output.parent / "temporal_candidate")
    )
    if baseline_dir is None and output_key == published_key:
        return resolved_output, None
    if baseline_dir is None and output_key == candidate_key:
        raise ValueError(
            "Task 9 candidate mode requires an explicit immutable baseline_dir "
            "and an isolated sibling candidate output."
        )
    if baseline_dir is None:
        raise ValueError(
            "Task 9 temporal output must be the exact published sibling 'temporal' "
            "or the isolated candidate sibling 'temporal_candidate'."
        )
    raw_baseline = Path(baseline_dir)
    try:
        resolved_baseline = raw_baseline.resolve(strict=True)
    except OSError as exc:
        raise ValueError(
            "Task 9 candidate mode requires an existing published baseline_dir "
            "and an isolated sibling candidate output."
        ) from exc
    expected_output = resolved_baseline.parent / "temporal_candidate"
    expected_baseline = resolved_baseline.parent / "temporal"
    if (
        not resolved_baseline.is_dir()
        or os.path.normcase(str(resolved_baseline))
        != os.path.normcase(str(expected_baseline))
        or output_key != os.path.normcase(str(expected_output))
        or output_key == os.path.normcase(str(resolved_baseline))
    ):
        raise ValueError(
            "Task 9 candidate mode requires published baseline_dir 'temporal' "
            "and the exact isolated sibling candidate output 'temporal_candidate'."
        )
    return resolved_output, resolved_baseline


def _task9_nonnegative_int_metric(metrics: dict[str, Any], key: str) -> int:
    value = metrics.get(key)
    if type(value) is not int or value < 0:
        raise ValueError(f"{key} must be a non-negative integer")
    return value


def _task9_nonnegative_real_metric(
    metrics: dict[str, Any], key: str, *, maximum: float | None = None
) -> float:
    value = metrics.get(key)
    if type(value) not in (int, float):
        raise ValueError(f"{key} must be a finite non-negative number")
    number = float(value)
    if not math.isfinite(number) or number < 0 or (maximum is not None and number > maximum):
        raise ValueError(f"{key} must be a finite non-negative number")
    return number


def build_repair_acceptance_report(
    *,
    run_id: str,
    baseline: dict[str, Any],
    candidate: dict[str, Any],
    physician_review_status: str = "pending",
    browser_review_status: str = "pending",
) -> dict[str, Any]:
    """Build a fail-closed quantitative and human-review decision record."""
    metrics = candidate.get("metrics")
    if type(run_id) is not str or not run_id or not isinstance(metrics, dict):
        raise ValueError("Task 9 acceptance inputs are malformed")
    frame_count = _task9_nonnegative_int_metric(metrics, "frame_count")
    multicomponent_frames = _task9_nonnegative_int_metric(
        metrics, "multicomponent_frames"
    )
    median_warped_iou = _task9_nonnegative_real_metric(
        metrics, "median_warped_iou", maximum=1.0
    )
    frames_below_iou = _task9_nonnegative_int_metric(
        metrics, "frames_below_iou_0_50"
    )
    accepted_gap_max_length = _task9_nonnegative_real_metric(
        metrics, "accepted_gap_max_length_px"
    )
    max_component_growth = _task9_nonnegative_real_metric(
        metrics, "max_component_growth_fraction"
    )
    rejected_border_leaks = _task9_nonnegative_int_metric(
        metrics, "rejected_border_leaks"
    )
    definitions = {
        "frame_count": ("equal", 31, frame_count == 31),
        "multicomponent_frames": (
            "at_most",
            6,
            multicomponent_frames <= 6,
        ),
        "median_warped_iou": (
            "at_least",
            0.70,
            median_warped_iou >= 0.70,
        ),
        "frames_below_iou_0_50": (
            "at_most",
            2,
            frames_below_iou <= 2,
        ),
        "accepted_gap_max_length_px": (
            "at_most",
            12.0,
            accepted_gap_max_length <= 12.0,
        ),
        "max_component_growth_fraction": (
            "at_most",
            0.20,
            max_component_growth <= 0.20,
        ),
        "rejected_border_leaks": (
            "equal",
            0,
            rejected_border_leaks == 0,
        ),
    }
    gates = {
        key: {
            "value": metrics.get(key),
            "operator": operator,
            "threshold": threshold,
            "passed": bool(passed),
        }
        for key, (operator, threshold, passed) in definitions.items()
    }
    quantitative_pass = all(item["passed"] for item in gates.values())
    physician_pass = physician_review_status == "passed"
    browser_pass = browser_review_status == "passed"
    return {
        "schema_version": 1,
        "run_id": run_id,
        "clinical_status": "research-only; quantitative gates are not clinical validation",
        "baseline": baseline,
        "candidate": candidate,
        "gates": gates,
        "quantitative_gates_passed": quantitative_pass,
        "quality_status": "quantitative-pass" if quantitative_pass else "rejected",
        "physician_review": {
            "status": physician_review_status,
            "required_frame_indices_zero_based": list(_TASK9_REVIEW_FRAME_INDICES),
            "contact_sheet": "repair_before_after.png",
        },
        "browser_review": {
            "status": browser_review_status,
            "required_frame_indices_zero_based": [3, 13, 24, 25, 28, 29],
            "required_modes": ["diagnostic", "realistic"],
        },
        "promotion_authorized": bool(
            quantitative_pass and physician_pass and browser_pass
        ),
    }


def _mask_overlay(image: np.ndarray, mask: np.ndarray, color: tuple[int, int, int]) -> np.ndarray:
    base = cv2.cvtColor(np.asarray(image, dtype=np.uint8), cv2.COLOR_GRAY2BGR)
    overlay = base.copy()
    overlay[np.asarray(mask, dtype=bool)] = color
    return cv2.addWeighted(base, 0.55, overlay, 0.45, 0.0)


def write_repair_contact_sheet(
    output_path: Path,
    frames: np.ndarray,
    baseline_masks: np.ndarray,
    repaired_masks: np.ndarray,
    repaired_skeletons: np.ndarray,
    links: list[dict[str, Any]],
    *,
    review_frame_indices: list[int] | None = None,
) -> None:
    """Render same-frame raw/baseline/repaired/centerline/link evidence."""
    frame_stack = np.asarray(frames, dtype=np.uint8)
    baseline = np.asarray(baseline_masks, dtype=bool)
    repaired = np.asarray(repaired_masks, dtype=bool)
    skeletons = np.asarray(repaired_skeletons, dtype=bool)
    if (
        frame_stack.ndim != 3
        or baseline.shape != frame_stack.shape
        or repaired.shape != frame_stack.shape
        or skeletons.shape != frame_stack.shape
    ):
        raise ValueError("Contact-sheet stacks must share one (frame, row, column) shape")
    link_frames = {
        int(link["frame_index"])
        for link in links
        if isinstance(link, dict) and type(link.get("frame_index")) is int
    }
    requested = (
        set(_TASK9_REVIEW_FRAME_INDICES)
        if review_frame_indices is None
        else {int(index) for index in review_frame_indices}
    )
    indices = sorted(
        index for index in requested | link_frames if 0 <= index < len(frame_stack)
    )
    if not indices:
        raise ValueError("Contact sheet has no valid review frames")
    tile_size = 160
    header = 20
    row_height = tile_size + header
    sheet = np.zeros((len(indices) * row_height, 5 * tile_size, 3), dtype=np.uint8)
    decision_colors = {
        "accepted": (91, 227, 45),
        "review": (77, 184, 255),
        "rejected": (115, 93, 255),
    }
    labels = ("Raw XA", "Baseline", "Repaired", "Centerline", "Gap audit")
    for row, frame_index in enumerate(indices):
        raw = cv2.cvtColor(frame_stack[frame_index], cv2.COLOR_GRAY2BGR)
        centerline = _mask_overlay(
            frame_stack[frame_index], repaired_skeletons[frame_index], (255, 220, 40)
        )
        audit = raw.copy()
        for link in links:
            if link.get("frame_index") != frame_index:
                continue
            points = np.asarray(link.get("points_yx", []), dtype=np.float64)
            if points.ndim != 2 or points.shape[1:] != (2,) or len(points) < 2:
                continue
            polyline = np.rint(points[:, ::-1]).astype(np.int32).reshape(-1, 1, 2)
            cv2.polylines(
                audit,
                [polyline],
                False,
                decision_colors.get(str(link.get("decision")), (255, 255, 255)),
                2,
                cv2.LINE_AA,
            )
        panels = (
            raw,
            _mask_overlay(frame_stack[frame_index], baseline[frame_index], (80, 220, 80)),
            _mask_overlay(frame_stack[frame_index], repaired[frame_index], (30, 160, 255)),
            centerline,
            audit,
        )
        y0 = row * row_height
        for column, (label, panel) in enumerate(zip(labels, panels, strict=True)):
            x0 = column * tile_size
            sheet[y0 : y0 + header, x0 : x0 + tile_size] = (20, 20, 20)
            cv2.putText(
                sheet,
                f"F{frame_index + 1:02d} {label}",
                (x0 + 4, y0 + 14),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.38,
                (240, 240, 240),
                1,
                cv2.LINE_AA,
            )
            sheet[
                y0 + header : y0 + row_height, x0 : x0 + tile_size
            ] = cv2.resize(panel, (tile_size, tile_size), interpolation=cv2.INTER_AREA)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), sheet):
        raise OSError(f"Failed to write repair contact sheet: {output_path}")


def warp_neighbor_to_current(neighbor: np.ndarray, current_frame: np.ndarray, neighbor_frame: np.ndarray) -> np.ndarray:
    """Sample a neighbor probability at corresponding positions in current."""
    flow_current_to_neighbor = dense_flow(current_frame, neighbor_frame)
    return remap_with_flow(
        neighbor.astype(np.float32, copy=False),
        flow_current_to_neighbor,
        cv2.INTER_LINEAR,
        border_mode=cv2.BORDER_REFLECT101,
    )


def temporal_fuse(frames: np.ndarray, probabilities: np.ndarray) -> np.ndarray:
    source_masks = probabilities >= 0.5
    return _temporal_fusion_evidence(
        frames, probabilities, source_masks
    ).fused_probability


def binary_iou(left: np.ndarray, right: np.ndarray) -> float:
    union = np.count_nonzero(left | right)
    if union == 0:
        return 1.0
    return float(np.count_nonzero(left & right) / union)


def temporal_metrics(frames: np.ndarray, masks: np.ndarray, skeletons: np.ndarray) -> list[dict[str, Any]]:
    metrics = []
    for index in range(len(frames)):
        if index == 0:
            warped_iou = None
            area_jump = None
            length_jump = None
        else:
            warped_previous = warp_neighbor_to_current(masks[index - 1].astype(np.float32), frames[index], frames[index - 1]) >= 0.5
            warped_iou = binary_iou(masks[index], warped_previous)
            previous_area = max(int(np.count_nonzero(masks[index - 1])), 1)
            previous_length = max(int(np.count_nonzero(skeletons[index - 1])), 1)
            area_jump = abs(int(np.count_nonzero(masks[index])) - previous_area) / previous_area
            length_jump = abs(int(np.count_nonzero(skeletons[index])) - previous_length) / previous_length
        metrics.append({
            "frame_index": index,
            "warped_iou_to_previous": warped_iou,
            "relative_area_jump": area_jump,
            "relative_centerline_jump": length_jump,
            "quality_flag": "review" if warped_iou is not None and warped_iou < 0.35 else "provisional",
        })
    return metrics


def _detector_border_exclusions(shape: tuple[int, ...]) -> np.ndarray:
    exclusions = np.zeros(shape, dtype=bool)
    exclusions[:, 0, :] = True
    exclusions[:, -1, :] = True
    exclusions[:, :, 0] = True
    exclusions[:, :, -1] = True
    return exclusions


def _finalize_repair_proposal(
    baseline_mask: np.ndarray,
    proposal_mask: np.ndarray,
    temporal_corridor: np.ndarray,
    exclusion_mask: np.ndarray,
    max_growth_fraction: float = 0.20,
) -> tuple[np.ndarray, RepairAudit]:
    baseline = np.asarray(baseline_mask, dtype=bool)
    proposal = baseline | np.asarray(proposal_mask, dtype=bool)
    additions = proposal & ~baseline
    attempted_pixels = int(np.count_nonzero(additions))
    baseline_pixels = int(np.count_nonzero(baseline))
    if attempted_pixels == 0:
        area_growth_fraction = 0.0
    elif baseline_pixels == 0:
        area_growth_fraction = 1.0
    else:
        area_growth_fraction = float(attempted_pixels / baseline_pixels)

    reasons = []
    if np.any(additions & ~np.asarray(temporal_corridor, dtype=bool)):
        reasons.append("temporal_corridor_violation")
    if np.any(additions & np.asarray(exclusion_mask, dtype=bool)):
        reasons.append("exclusion_intersection")
    if area_growth_fraction > max_growth_fraction:
        reasons.append("growth_cap_exceeded")
    if reasons:
        return baseline.copy(), RepairAudit(
            accepted_pixels=0,
            rejected_pixels=attempted_pixels,
            area_growth_fraction=area_growth_fraction,
            status="rejected",
            reasons=tuple(reasons),
        )
    return proposal, RepairAudit(
        accepted_pixels=attempted_pixels,
        rejected_pixels=0,
        area_growth_fraction=area_growth_fraction,
        status="accepted",
        reasons=(),
    )


def _repair_temporal_sequence(
    frames: np.ndarray,
    model_probabilities: np.ndarray,
    temporal_probabilities: np.ndarray,
    support_probabilities: np.ndarray,
    baseline_masks: np.ndarray,
    catheter_exclusion_masks: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, list[RepairAudit]]:
    evidence_maps = np.asarray(
        [
            build_vessel_evidence(frame, model, temporal, support)
            for frame, model, temporal, support in zip(
                frames,
                model_probabilities,
                temporal_probabilities,
                support_probabilities,
            )
        ],
        dtype=np.float32,
    )
    baseline = np.asarray(baseline_masks, dtype=bool)
    if catheter_exclusion_masks is None:
        audits = [
            RepairAudit(
                accepted_pixels=0,
                rejected_pixels=0,
                area_growth_fraction=0.0,
                status="rejected",
                reasons=("catheter_exclusion_unavailable",),
            )
            for _ in frames
        ]
        return evidence_maps, baseline.copy(), audits

    catheter_exclusions = np.asarray(catheter_exclusion_masks, dtype=bool)
    if catheter_exclusions.shape != frames.shape:
        raise ValueError(
            "Catheter exclusion shape mismatch: "
            f"{catheter_exclusions.shape} != {frames.shape}"
        )
    exclusions = _detector_border_exclusions(frames.shape) | catheter_exclusions
    reconstruction_proposals = []
    corridors = []
    for index, evidence in enumerate(evidence_maps):
        corridor = _motion_compensated_temporal_corridor(
            index, frames, baseline
        )
        corridors.append(corridor)
        constrained_evidence = np.where(corridor, evidence, 0.0)
        seed = (evidence >= 0.70) & corridor & ~exclusions[index]
        reconstructed, _ = constrained_seed_reconstruction(
            constrained_evidence,
            seed,
            exclusions[index],
            max_growth_fraction=float("inf"),
        )
        reconstruction_proposals.append(baseline[index] | reconstructed)
    morphology_proposals = _anisotropic_temporal_majority(
        frames,
        np.asarray(reconstruction_proposals, dtype=bool),
        evidence_maps,
        exclusions,
    )
    repaired_masks = []
    audits = []
    for index, proposal in enumerate(morphology_proposals):
        repaired, audit = _finalize_repair_proposal(
            baseline[index],
            proposal,
            corridors[index],
            exclusions[index],
            max_growth_fraction=0.20,
        )
        repaired_masks.append(repaired)
        audits.append(audit)
    return evidence_maps, np.asarray(repaired_masks, dtype=bool), audits


def _gap_confidence(candidate: GapCandidate) -> float:
    direction = max(0.0, 1.0 - float(candidate.angle_deg) / 180.0)
    distance = max(0.0, 1.0 - float(candidate.distance_px) / 24.0)
    return float(
        np.clip(
            0.40 * float(candidate.mean_evidence)
            + 0.35 * float(candidate.temporal_support)
            + 0.15 * direction
            + 0.10 * distance,
            0.0,
            1.0,
        )
    )


def _gap_link_record(frame_index: int, candidate: GapCandidate) -> dict[str, Any]:
    return {
        "frame_index": int(frame_index),
        "start_node_id": candidate.start_node_id,
        "end_node_id": candidate.end_node_id,
        "distance_px": float(candidate.distance_px),
        "angle_deg": float(candidate.angle_deg),
        "mean_evidence": float(candidate.mean_evidence),
        "temporal_support": float(candidate.temporal_support),
        "score": float(candidate.score),
        "confidence": _gap_confidence(candidate),
        "decision": candidate.decision,
        "reasons": list(candidate.reasons),
        "points_yx": np.asarray(candidate.path_yx, dtype=np.float64).tolist(),
    }


def _apply_short_gap_repairs(
    frames: np.ndarray,
    baseline_masks: np.ndarray,
    morphology_masks: np.ndarray,
    evidence_maps: np.ndarray,
    support_probabilities: np.ndarray,
    catheter_exclusion_masks: np.ndarray | None,
) -> tuple[np.ndarray, list[RepairAudit], list[dict[str, Any]]]:
    """Apply Task 4 repairs after morphology while retaining the frame hard gates."""
    baseline = np.asarray(baseline_masks, dtype=bool)
    morphology_result = np.asarray(morphology_masks, dtype=bool)
    if catheter_exclusion_masks is None:
        audits = [
            RepairAudit(
                accepted_pixels=0,
                rejected_pixels=0,
                area_growth_fraction=0.0,
                status="rejected",
                reasons=("catheter_exclusion_unavailable",),
            )
            for _ in frames
        ]
        return baseline.copy(), audits, []
    catheter = np.asarray(catheter_exclusion_masks, dtype=bool)
    if catheter.shape != baseline.shape:
        raise ValueError("Catheter exclusion shape mismatch during gap repair")
    exclusions = _detector_border_exclusions(frames.shape) | catheter
    final_masks: list[np.ndarray] = []
    final_audits: list[RepairAudit] = []
    links: list[dict[str, Any]] = []
    for index in range(len(frames)):
        graph = extract_vessel_graph(morphology_result[index], root_hint_yx=None)
        candidates = find_gap_candidates(
            graph,
            evidence_maps[index],
            support_probabilities[index],
            exclusion_mask=exclusions[index],
        )
        gap_mask, audited = repair_short_gaps(
            morphology_result[index],
            graph,
            candidates,
            exclusion_mask=exclusions[index],
        )
        supported = (
            np.asarray(support_probabilities[index], dtype=np.float32) >= (2.0 / 3.0)
        )
        temporal_corridor = (
            morphology_result[index]
            | morphology.dilation(supported, morphology.disk(5))
            | baseline[index]
        )
        final_mask, frame_audit = _finalize_repair_proposal(
            baseline[index],
            gap_mask,
            temporal_corridor,
            exclusions[index],
            max_growth_fraction=0.20,
        )
        if frame_audit.status == "rejected":
            reason = "frame_finalization_rejected"
            audited = [
                replace(
                    candidate,
                    decision="rejected",
                    reasons=tuple(dict.fromkeys((*candidate.reasons, reason))),
                )
                if candidate.decision == "accepted"
                else candidate
                for candidate in audited
            ]
        final_masks.append(final_mask)
        final_audits.append(frame_audit)
        links.extend(_gap_link_record(index, candidate) for candidate in audited)
    return np.asarray(final_masks, dtype=bool), final_audits, links


def _raw_asset_records(directory: Path, names: tuple[str, ...]) -> dict[str, Any]:
    records: dict[str, Any] = {}
    for name in names:
        path = Path(directory) / name
        if not path.is_file():
            raise FileNotFoundError(path)
        records[name] = {"byte_length": int(path.stat().st_size), "sha256": _sha256_file(path)}
    return records


def _load_published_temporal_baseline(
    baseline_dir: Path, shape: tuple[int, ...]
) -> dict[str, Any]:
    expected_size = int(np.prod(shape))
    required_assets = (
        "temporal_probabilities_u8.raw",
        "temporal_masks_u8.raw",
        "temporal_centerlines_u8.raw",
        "temporal_quality.csv",
        "temporal_report.json",
    )
    try:
        assets = _raw_asset_records(baseline_dir, required_assets)
        mask_raw = np.fromfile(
            baseline_dir / "temporal_masks_u8.raw", dtype=np.uint8
        )
        centerline_raw = np.fromfile(
            baseline_dir / "temporal_centerlines_u8.raw", dtype=np.uint8
        )
    except (OSError, FileNotFoundError) as exc:
        raise ValueError(
            "Task 9 published baseline is missing required immutable assets"
        ) from exc
    if mask_raw.size != expected_size or centerline_raw.size != expected_size:
        raise ValueError("Task 9 published baseline temporal stack size mismatch")
    if not np.all(np.isin(mask_raw, (0, 255))) or not np.all(
        np.isin(centerline_raw, (0, 255))
    ):
        raise ValueError("Task 9 published baseline masks must be binary uint8 stacks")
    metric_path = baseline_dir / "temporal_quality.csv"
    try:
        metric_rows = _metrics_from_csv(metric_path)
    except (OSError, KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            "Task 9 published baseline warped_iou metrics are invalid"
        ) from exc
    if len(metric_rows) != int(shape[0]):
        raise ValueError("Task 9 published baseline temporal metrics are incomplete")
    return {
        "masks": mask_raw.reshape(shape) != 0,
        "skeletons": centerline_raw.reshape(shape) != 0,
        "metric_rows": metric_rows,
        "assets": assets,
    }


def _metrics_from_csv(path: Path) -> list[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        expected_fields = [
            "frame_index",
            "warped_iou_to_previous",
            "relative_area_jump",
            "relative_centerline_jump",
            "quality_flag",
        ]
        if reader.fieldnames != expected_fields:
            raise ValueError("Temporal quality CSV columns are invalid")
        rows = list(reader)
    result: list[dict[str, Any]] = []
    for position, row in enumerate(rows):
        frame_index = int(row["frame_index"])
        values = [
            None if row[field] == "" else float(row[field])
            for field in expected_fields[1:4]
        ]
        if frame_index != position:
            raise ValueError("Temporal quality CSV frames are out of order")
        if position == 0:
            if values != [None, None, None]:
                raise ValueError("Temporal quality CSV first deltas must be empty")
        elif any(value is None for value in values):
            raise ValueError("Temporal quality CSV deltas are incomplete")
        numeric_values = [value for value in values if value is not None]
        if any(not math.isfinite(value) or value < 0.0 for value in numeric_values):
            raise ValueError("Temporal quality CSV contains non-finite deltas")
        warped_iou = values[0]
        if warped_iou is not None and warped_iou > 1.0:
            raise ValueError("Temporal quality CSV warped IoU is outside [0, 1]")
        quality_flag = row["quality_flag"]
        expected_flag = (
            "review" if warped_iou is not None and warped_iou < 0.35 else "provisional"
        )
        if quality_flag != expected_flag:
            raise ValueError("Temporal quality CSV quality flag is inconsistent")
        result.append(
            {
                "frame_index": frame_index,
                "warped_iou_to_previous": warped_iou,
            }
        )
    return result


def _acceptance_metrics(
    topology: list[Any],
    temporal_metric_rows: list[dict[str, Any]],
    baseline_masks: np.ndarray,
    repaired_masks: np.ndarray,
    links: list[dict[str, Any]],
) -> dict[str, Any]:
    ious = [
        float(item["warped_iou_to_previous"])
        for item in temporal_metric_rows
        if item.get("warped_iou_to_previous") is not None
    ]
    baseline = np.asarray(baseline_masks, dtype=bool)
    repaired = np.asarray(repaired_masks, dtype=bool)
    growth = []
    border = np.zeros(baseline.shape[1:], dtype=bool)
    border[0, :] = border[-1, :] = True
    border[:, 0] = border[:, -1] = True
    border_leaks = 0
    for before, after in zip(baseline, repaired, strict=True):
        additions = after & ~before
        denominator = max(int(np.count_nonzero(before)), 1)
        growth.append(float(np.count_nonzero(additions) / denominator))
        border_leaks += int(np.count_nonzero(additions & border))
    accepted_lengths = [
        float(link["distance_px"])
        for link in links
        if link.get("decision") == "accepted"
    ]
    return {
        "frame_count": int(len(repaired)),
        "multicomponent_frames": int(
            sum(int(item.component_count) > 1 for item in topology)
        ),
        "median_warped_iou": float(np.median(ious)) if ious else 1.0,
        "frames_below_iou_0_50": int(sum(value < 0.50 for value in ious)),
        "accepted_gap_max_length_px": max(accepted_lengths, default=0.0),
        "max_component_growth_fraction": max(growth, default=0.0),
        "rejected_border_leaks": int(border_leaks),
    }


def _mask_delta_metrics(
    published_masks: np.ndarray, candidate_masks: np.ndarray
) -> dict[str, float]:
    published = np.asarray(published_masks, dtype=bool)
    candidate = np.asarray(candidate_masks, dtype=bool)
    if published.shape != candidate.shape or published.ndim != 3:
        raise ValueError("Task 9 published-baseline delta stacks must share a 3D shape")
    added_fractions: list[float] = []
    removed_fractions: list[float] = []
    changed_fractions: list[float] = []
    for baseline, repaired in zip(published, candidate, strict=True):
        denominator = max(int(np.count_nonzero(baseline)), 1)
        added_fractions.append(
            float(np.count_nonzero(repaired & ~baseline) / denominator)
        )
        removed_fractions.append(
            float(np.count_nonzero(baseline & ~repaired) / denominator)
        )
        changed_fractions.append(
            float(np.count_nonzero(baseline ^ repaired) / denominator)
        )
    return {
        "max_added_pixel_fraction": max(added_fractions, default=0.0),
        "max_removed_pixel_fraction": max(removed_fractions, default=0.0),
        "max_changed_pixel_fraction": max(changed_fractions, default=0.0),
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_identity(path: Path) -> dict[str, Any]:
    return {
        "byte_length": int(path.stat().st_size),
        "sha256": _sha256_file(path),
    }


def _require_exact_dict(
    value: object, expected_keys: set[str], label: str
) -> dict[str, Any]:
    if type(value) is not dict or set(value) != expected_keys:
        raise _baseline_provenance_error(
            f"manifest {label} must be an object with exactly "
            f"{sorted(expected_keys)}"
        )
    return value


def _require_nonnegative_int(value: object, label: str) -> int:
    if type(value) is not int or value < 0:
        raise _baseline_provenance_error(
            f"manifest {label} must be a nonnegative integer"
        )
    return value


def _require_sha256(value: object, label: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in string.hexdigits for character in value)
    ):
        raise _baseline_provenance_error(
            f"manifest {label} must be a 64-character hexadecimal SHA-256"
        )
    return value


def _require_finite_number(value: object, label: str) -> float:
    if type(value) not in (int, float) or not np.isfinite(value):
        raise _baseline_provenance_error(
            f"manifest {label} must be a finite number"
        )
    return float(value)


def _require_optional_finite_number(
    value: object, label: str
) -> float | None:
    if value is None:
        return None
    return _require_finite_number(value, label)


def _validate_file_identity_record(value: object, label: str) -> dict[str, Any]:
    record = _require_exact_dict(value, _FILE_IDENTITY_KEYS, label)
    _require_nonnegative_int(record["byte_length"], f"{label}.byte_length")
    _require_sha256(record["sha256"], f"{label}.sha256")
    return record


def _validate_directory_relative_path(value: object, label: str) -> str:
    if type(value) is not str or not value or "\\" in value:
        raise _baseline_provenance_error(
            f"manifest {label} must be a canonical relative POSIX path"
        )
    relative = PurePosixPath(value)
    if (
        relative.is_absolute()
        or not relative.parts
        or any(part in (".", "..") for part in relative.parts)
        or relative.as_posix() != value
    ):
        raise _baseline_provenance_error(
            f"manifest {label} must be a canonical relative POSIX path"
        )
    return value


def _validate_baseline_target_record(
    value: object, label: str
) -> dict[str, Any]:
    if type(value) is not dict:
        raise _baseline_provenance_error(f"manifest {label} must be an object")
    kind = value.get("kind")
    if kind == "file":
        record = _require_exact_dict(value, _FILE_TARGET_KEYS, label)
        _require_nonnegative_int(record["byte_length"], f"{label}.byte_length")
        _require_sha256(record["sha256"], f"{label}.sha256")
        return record
    if kind == "directory":
        record = _require_exact_dict(value, _DIRECTORY_TARGET_KEYS, label)
        file_count = _require_nonnegative_int(
            record["file_count"], f"{label}.file_count"
        )
        files = record["files"]
        if type(files) is not list:
            raise _baseline_provenance_error(
                f"manifest {label}.files must be a list"
            )
        validated_paths = []
        for index, item in enumerate(files):
            file_label = f"{label}.files[{index}]"
            file_record = _require_exact_dict(
                item, _DIRECTORY_FILE_KEYS, file_label
            )
            validated_paths.append(
                _validate_directory_relative_path(
                    file_record["path"], f"{file_label}.path"
                )
            )
            _require_nonnegative_int(
                file_record["byte_length"], f"{file_label}.byte_length"
            )
            _require_sha256(file_record["sha256"], f"{file_label}.sha256")
        if file_count != len(files):
            raise _baseline_provenance_error(
                f"manifest {label}.file_count does not match its files list"
            )
        if validated_paths != sorted(validated_paths) or len(
            set(validated_paths)
        ) != len(validated_paths):
            raise _baseline_provenance_error(
                f"manifest {label}.files paths must be unique and sorted"
            )
        _require_sha256(record["sha256"], f"{label}.sha256")
        return record
    raise _baseline_provenance_error(
        f"manifest {label}.kind must be 'file' or 'directory'"
    )


def _validate_diagnostics(value: object) -> dict[str, Any]:
    diagnostics = _require_exact_dict(value, _DIAGNOSTIC_KEYS, "diagnostics")

    frame_records = diagnostics["frames"]
    if type(frame_records) is not list:
        raise _baseline_provenance_error("manifest diagnostics.frames must be a list")
    for index, item in enumerate(frame_records):
        label = f"diagnostics.frames[{index}]"
        record = _require_exact_dict(item, _FRAME_DIAGNOSTIC_KEYS, label)
        _require_nonnegative_int(record["frame_index"], f"{label}.frame_index")
        error_p95 = _require_finite_number(
            record["forward_backward_error_p95_px"],
            f"{label}.forward_backward_error_p95_px",
        )
        if error_p95 < 0.0:
            raise _baseline_provenance_error(
                f"manifest {label}.forward_backward_error_p95_px must be nonnegative"
            )
        _require_nonnegative_int(
            record["valid_neighbor_count"], f"{label}.valid_neighbor_count"
        )

    metric_records = diagnostics["temporal_metrics"]
    if type(metric_records) is not list:
        raise _baseline_provenance_error(
            "manifest diagnostics.temporal_metrics must be a list"
        )
    for index, item in enumerate(metric_records):
        label = f"diagnostics.temporal_metrics[{index}]"
        record = _require_exact_dict(item, _TEMPORAL_METRIC_KEYS, label)
        _require_nonnegative_int(record["frame_index"], f"{label}.frame_index")
        for field in (
            "warped_iou_to_previous",
            "relative_area_jump",
            "relative_centerline_jump",
        ):
            number = _require_optional_finite_number(
                record[field], f"{label}.{field}"
            )
            if number is not None and number < 0.0:
                raise _baseline_provenance_error(
                    f"manifest {label}.{field} must be nonnegative or null"
                )
        if record["quality_flag"] not in ("review", "provisional"):
            raise _baseline_provenance_error(
                f"manifest {label}.quality_flag is unsupported"
            )

    summary = _require_exact_dict(
        diagnostics["quality_summary"],
        _QUALITY_SUMMARY_KEYS,
        "diagnostics.quality_summary",
    )
    median_iou = _require_optional_finite_number(
        summary["median_warped_iou"],
        "diagnostics.quality_summary.median_warped_iou",
    )
    if median_iou is not None and not 0.0 <= median_iou <= 1.0:
        raise _baseline_provenance_error(
            "manifest diagnostics.quality_summary.median_warped_iou "
            "must be between zero and one or null"
        )
    _require_nonnegative_int(
        summary["review_frames"],
        "diagnostics.quality_summary.review_frames",
    )
    _require_nonnegative_int(
        summary["provisional_frames"],
        "diagnostics.quality_summary.provisional_frames",
    )
    return diagnostics


def _validate_manifest_structure(value: object) -> dict[str, Any]:
    manifest = _require_exact_dict(value, _MANIFEST_TOP_LEVEL_KEYS, "root")
    if type(manifest["schema_version"]) is not int or manifest["schema_version"] != 1:
        raise _baseline_provenance_error(
            "manifest schema_version must be integer 1"
        )

    inputs = _require_exact_dict(manifest["inputs"], _INPUT_KEYS, "inputs")
    for name in sorted(_INPUT_KEYS):
        _validate_file_identity_record(inputs[name], f"inputs.{name}")

    target_paths = set(_BASELINE_RAW_TARGETS) | set(_BASELINE_FRAME_TARGETS) | {
        _BASELINE_VIDEO_TARGET
    }
    targets = _require_exact_dict(
        manifest["baseline_targets"], target_paths, "baseline_targets"
    )
    for name in sorted(target_paths):
        _validate_baseline_target_record(
            targets[name], f"baseline_targets.{name}"
        )

    _validate_diagnostics(manifest["diagnostics"])
    return manifest


def _file_provenance(path: Path) -> dict[str, Any]:
    return {"kind": "file", **_file_identity(path)}


def _directory_provenance(path: Path) -> dict[str, Any]:
    if not path.is_dir():
        raise FileNotFoundError(path)
    nested_directories = sorted(
        candidate.relative_to(path).as_posix()
        for candidate in path.rglob("*")
        if candidate.is_dir()
    )
    if nested_directories:
        raise _baseline_provenance_error(
            f"baseline target {path.name} contains unexpected nested directory "
            f"{nested_directories[0]}"
        )
    files = [
        {
            "path": item.relative_to(path).as_posix(),
            **_file_identity(item),
        }
        for item in sorted(
            (candidate for candidate in path.rglob("*") if candidate.is_file()),
            key=lambda candidate: candidate.relative_to(path).as_posix(),
        )
    ]
    canonical = json.dumps(
        files,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return {
        "kind": "directory",
        "file_count": len(files),
        "files": files,
        "sha256": hashlib.sha256(canonical).hexdigest(),
    }


def _baseline_target_paths(output_dir: Path) -> dict[str, Path]:
    return {
        **{name: output_dir / name for name in _BASELINE_RAW_TARGETS},
        **{name: output_dir / name for name in _BASELINE_FRAME_TARGETS},
        _BASELINE_VIDEO_TARGET: output_dir / _BASELINE_VIDEO_TARGET,
    }


def _baseline_target_provenance(path: Path) -> dict[str, Any]:
    if path.is_dir():
        return _directory_provenance(path)
    if path.is_file():
        return _file_provenance(path)
    raise FileNotFoundError(path)


def _baseline_provenance_error(reason: str) -> ValueError:
    return ValueError(
        f"Baseline provenance validation failed: {reason}; "
        "use a fresh output directory."
    )


def _input_provenance(
    dicom_path: Path, probability_path: Path
) -> dict[str, dict[str, Any]]:
    return {
        "dicom": _file_identity(dicom_path),
        "probabilities_u8.raw": _file_identity(probability_path),
    }


def _file_identity_matches(
    recorded: dict[str, Any], actual: dict[str, Any]
) -> bool:
    return (
        recorded["byte_length"] == actual["byte_length"]
        and recorded["sha256"].lower() == actual["sha256"].lower()
    )


def _target_provenance_matches(
    recorded: dict[str, Any], actual: dict[str, Any]
) -> bool:
    if recorded["kind"] != actual["kind"]:
        return False
    if recorded["kind"] == "file":
        return _file_identity_matches(recorded, actual)
    if (
        recorded["file_count"] != actual["file_count"]
        or recorded["sha256"].lower() != actual["sha256"].lower()
        or len(recorded["files"]) != len(actual["files"])
    ):
        return False
    return all(
        left["path"] == right["path"] and _file_identity_matches(left, right)
        for left, right in zip(recorded["files"], actual["files"])
    )


def _strict_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate manifest key {key!r}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant {value}")


def _validate_existing_baseline(
    dicom_path: Path,
    probability_path: Path,
    output_dir: Path,
) -> dict[str, Any] | None:
    manifest_path = output_dir / _BASELINE_MANIFEST
    target_paths = _baseline_target_paths(output_dir)
    if not manifest_path.exists():
        existing = [name for name, path in target_paths.items() if path.exists()]
        if existing:
            raise _baseline_provenance_error(
                "manifest is absent while baseline targets already exist"
            )
        return None

    try:
        parsed_manifest = json.loads(
            manifest_path.read_text(encoding="utf-8"),
            object_pairs_hook=_strict_json_object,
            parse_constant=_reject_json_constant,
        )
    except (OSError, json.JSONDecodeError, ValueError) as error:
        raise _baseline_provenance_error("manifest is unreadable or invalid") from error
    manifest = _validate_manifest_structure(parsed_manifest)

    actual_inputs = _input_provenance(dicom_path, probability_path)
    if any(
        not _file_identity_matches(manifest["inputs"][name], actual_inputs[name])
        for name in _INPUT_KEYS
    ):
        raise _baseline_provenance_error("input DICOM or probabilities hash mismatch")

    recorded_targets = manifest["baseline_targets"]
    for name, path in target_paths.items():
        try:
            actual = _baseline_target_provenance(path)
        except (OSError, FileNotFoundError) as error:
            raise _baseline_provenance_error(
                f"baseline target {name} is missing"
            ) from error
        if not _target_provenance_matches(recorded_targets[name], actual):
            raise _baseline_provenance_error(
                f"baseline target {name} does not match its recorded hash"
            )
    return manifest


def _validate_baseline_completeness(
    manifest: dict[str, Any], shape: tuple[int, ...]
) -> None:
    manifest = _validate_manifest_structure(manifest)
    records = manifest["baseline_targets"]
    expected_byte_length = int(np.prod(shape))
    for name in _BASELINE_RAW_TARGETS:
        record = records[name]
        if (
            record.get("kind") != "file"
            or record.get("byte_length") != expected_byte_length
        ):
            raise _baseline_provenance_error(
                f"baseline target {name} has an invalid byte length"
            )
    for directory, prefix in _BASELINE_FRAME_TARGETS.items():
        expected_files = [
            f"{prefix}_{index:04d}.png" for index in range(int(shape[0]))
        ]
        record = records[directory]
        actual_files = [item.get("path") for item in record.get("files", [])]
        if (
            record.get("kind") != "directory"
            or record.get("file_count") != len(expected_files)
            or actual_files != expected_files
        ):
            raise _baseline_provenance_error(
                f"baseline target {directory} has incomplete frame contents"
            )
    video = records[_BASELINE_VIDEO_TARGET]
    if video["kind"] != "file" or video["byte_length"] <= 0:
        raise _baseline_provenance_error(
            f"baseline target {_BASELINE_VIDEO_TARGET} is incomplete"
        )

    frame_count = int(shape[0])
    diagnostics = manifest["diagnostics"]
    frame_records = diagnostics["frames"]
    metric_records = diagnostics["temporal_metrics"]
    expected_indices = list(range(frame_count))
    if (
        len(frame_records) != frame_count
        or [item["frame_index"] for item in frame_records] != expected_indices
    ):
        raise _baseline_provenance_error(
            "persisted frame diagnostics are incomplete or misindexed"
        )
    for index, frame_record in enumerate(frame_records):
        available_adjacent_frames = int(index > 0) + int(index + 1 < frame_count)
        if frame_record["valid_neighbor_count"] > available_adjacent_frames:
            raise _baseline_provenance_error(
                f"diagnostics.frames[{index}].valid_neighbor_count exceeds "
                "the available adjacent frame count"
            )
    if (
        len(metric_records) != frame_count
        or [item["frame_index"] for item in metric_records] != expected_indices
    ):
        raise _baseline_provenance_error(
            "persisted temporal metrics are incomplete or misindexed"
        )
    for index, metric in enumerate(metric_records):
        numeric_values = [
            metric["warped_iou_to_previous"],
            metric["relative_area_jump"],
            metric["relative_centerline_jump"],
        ]
        if index == 0:
            if numeric_values != [None, None, None]:
                raise _baseline_provenance_error(
                    "first persisted temporal metric row must use null deltas"
                )
        elif any(value is None for value in numeric_values):
            raise _baseline_provenance_error(
                "persisted temporal metric deltas are incomplete"
            )
        warped_iou = metric["warped_iou_to_previous"]
        if warped_iou is not None and warped_iou > 1.0:
            raise _baseline_provenance_error(
                "persisted warped IoU must be between zero and one"
            )
        expected_flag = (
            "review"
            if warped_iou is not None and warped_iou < 0.35
            else "provisional"
        )
        if metric["quality_flag"] != expected_flag:
            raise _baseline_provenance_error(
                "persisted temporal metric quality flag is inconsistent"
            )
    if diagnostics["quality_summary"] != _quality_summary(metric_records):
        raise _baseline_provenance_error(
            "persisted quality summary is inconsistent with temporal metrics"
        )


def _quality_summary(metrics: list[dict[str, Any]]) -> dict[str, Any]:
    valid_iou = [
        item["warped_iou_to_previous"]
        for item in metrics
        if item["warped_iou_to_previous"] is not None
    ]
    return {
        "median_warped_iou": float(np.median(valid_iou)) if valid_iou else None,
        "review_frames": sum(item["quality_flag"] == "review" for item in metrics),
        "provisional_frames": sum(
            item["quality_flag"] == "provisional" for item in metrics
        ),
    }


def _baseline_diagnostics(
    flow_evidence: Any, metrics: list[dict[str, Any]]
) -> dict[str, Any]:
    return {
        "frames": [
            {
                "frame_index": index,
                "forward_backward_error_p95_px": float(
                    flow_evidence.forward_backward_error_p95_px[index]
                ),
                "valid_neighbor_count": int(
                    flow_evidence.valid_neighbor_frame_count[index]
                ),
            }
            for index in range(len(metrics))
        ],
        "temporal_metrics": metrics,
        "quality_summary": _quality_summary(metrics),
    }


def _build_baseline_manifest(
    dicom_path: Path,
    probability_path: Path,
    output_dir: Path,
    diagnostics: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "inputs": _input_provenance(dicom_path, probability_path),
        "baseline_targets": {
            name: _baseline_target_provenance(path)
            for name, path in _baseline_target_paths(output_dir).items()
        },
        "diagnostics": diagnostics,
    }


def _load_baseline_raw(
    output_dir: Path, name: str, shape: tuple[int, ...]
) -> np.ndarray:
    raw = np.fromfile(output_dir / name, dtype=np.uint8)
    if raw.size != int(np.prod(shape)):
        raise _baseline_provenance_error(
            f"baseline target {name} has an invalid byte length"
        )
    return raw.reshape(shape)


def _validate_baseline_media(
    output_dir: Path, shape: tuple[int, ...]
) -> None:
    frame_count, height, width = (int(value) for value in shape)
    expected_hw = (height, width)
    for directory, prefix in _BASELINE_FRAME_TARGETS.items():
        expected_channels = 3 if directory == "overlays" else 1
        for index in range(frame_count):
            path = output_dir / directory / f"{prefix}_{index:04d}.png"
            if not path.is_file():
                raise _baseline_provenance_error(
                    f"baseline media {path.name} is missing"
                )
            image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
            if image is None:
                raise _baseline_provenance_error(
                    f"baseline media {path.name} is not a decodable PNG"
                )
            if image.shape[:2] != expected_hw:
                raise _baseline_provenance_error(
                    f"baseline media {path.name} has shape {image.shape[:2]}, "
                    f"expected {expected_hw}"
                )
            actual_channels = (
                1
                if image.ndim == 2
                else int(image.shape[2])
                if image.ndim == 3
                else 0
            )
            if actual_channels != expected_channels:
                raise _baseline_provenance_error(
                    f"baseline media {path.name} has {actual_channels} channels, "
                    f"expected {expected_channels}"
                )

    video_path = output_dir / _BASELINE_VIDEO_TARGET
    if not video_path.is_file():
        raise _baseline_provenance_error(
            f"baseline media {_BASELINE_VIDEO_TARGET} is missing"
        )
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        capture.release()
        raise _baseline_provenance_error(
            f"baseline media {_BASELINE_VIDEO_TARGET} is not a decodable MP4"
        )
    decoded_frame_count = 0
    try:
        while True:
            decoded, frame = capture.read()
            if not decoded:
                break
            if frame is None or frame.shape[:2] != expected_hw:
                actual_shape = None if frame is None else frame.shape[:2]
                raise _baseline_provenance_error(
                    f"baseline media {_BASELINE_VIDEO_TARGET} frame "
                    f"{decoded_frame_count} has shape {actual_shape}, "
                    f"expected {expected_hw}"
                )
            decoded_frame_count += 1
    finally:
        capture.release()
    if decoded_frame_count != frame_count:
        raise _baseline_provenance_error(
            f"baseline media {_BASELINE_VIDEO_TARGET} has "
            f"{decoded_frame_count} decoded frames, expected {frame_count}"
        )


def _validate_persisted_diagnostic_jumps(
    manifest: dict[str, Any],
    masks: np.ndarray,
    skeletons: np.ndarray,
) -> None:
    metrics = manifest["diagnostics"]["temporal_metrics"]
    for index in range(1, len(metrics)):
        previous_area = max(int(np.count_nonzero(masks[index - 1])), 1)
        previous_length = max(int(np.count_nonzero(skeletons[index - 1])), 1)
        expected = {
            "relative_area_jump": abs(
                int(np.count_nonzero(masks[index])) - previous_area
            )
            / previous_area,
            "relative_centerline_jump": abs(
                int(np.count_nonzero(skeletons[index])) - previous_length
            )
            / previous_length,
        }
        for field, expected_value in expected.items():
            stored_value = metrics[index][field]
            if not math.isclose(
                stored_value,
                expected_value,
                rel_tol=1e-12,
                abs_tol=1e-12,
            ):
                raise _baseline_provenance_error(
                    f"diagnostics.temporal_metrics[{index}].{field} does not "
                    "match the persisted baseline raw stack"
                )


def _temporal_graph_document(graphs: list[VesselGraph]) -> dict[str, Any]:
    frame_records: list[dict[str, Any]] = []
    for graph in graphs:
        local_to_topology = {
            node.node_id: node.topology_id for node in graph.nodes
        }
        edge_records = [
            {
                "edge_id": edge.edge_id,
                "topology_id": edge.topology_id,
                "parent_edge_id": edge.parent_edge_id,
                "source_node_id": edge.source_node_id,
                "target_node_id": edge.target_node_id,
                "source_topology_id": local_to_topology[edge.source_node_id],
                "target_topology_id": local_to_topology[edge.target_node_id],
                "points_yx": np.asarray(edge.points_yx, dtype=np.float64).tolist(),
                "radius_px": np.asarray(edge.radius_px, dtype=np.float64).tolist(),
                "confidence": float(edge.confidence),
                "visibility": edge.visibility,
            }
            for edge in graph.edges
            if edge.source_node_id != edge.target_node_id
        ]
        local_to_yx = {
            node.node_id: [float(node.yx[0]), float(node.yx[1])]
            for node in graph.nodes
        }
        for edge in edge_records:
            edge["points_yx"][0] = local_to_yx[edge["source_node_id"]]
            edge["points_yx"][-1] = local_to_yx[edge["target_node_id"]]
        adjacency = {node.node_id: set() for node in graph.nodes}
        for edge in edge_records:
            source_id = edge["source_node_id"]
            target_id = edge["target_node_id"]
            adjacency[source_id].add(target_id)
            adjacency[target_id].add(source_id)
        remaining = set(adjacency)
        component_count = 0
        while remaining:
            component_count += 1
            stack = [remaining.pop()]
            while stack:
                neighbours = adjacency[stack.pop()] & remaining
                remaining.difference_update(neighbours)
                stack.extend(neighbours)
        frame_records.append(
            {
                "frame_index": int(graph.frame_index),
                "root_node_id": graph.root_node_id,
                "root_topology_id": local_to_topology.get(graph.root_node_id),
                "component_count": int(component_count),
                "qc": dict(graph.qc or {}),
                "nodes": [
                    {
                        "node_id": node.node_id,
                        "topology_id": node.topology_id,
                        "parent_node_id": node.parent_node_id,
                        "kind": node.kind,
                        "yx": [float(node.yx[0]), float(node.yx[1])],
                        "confidence": float(node.confidence),
                        "visibility": node.visibility,
                    }
                    for node in graph.nodes
                ],
                "edges": edge_records,
            }
        )
    return {
        "schema_version": 1,
        "frame_count": len(graphs),
        "flow_convention": {
            "order": ["dx", "dy"],
            "direction": "previous_to_current",
            "mapping": "(y, x) -> (y + dy, x + dx)",
            "sampling": "deterministic_bilinear",
        },
        "frames": frame_records,
    }


def run_temporal(
    dicom_path: Path,
    segmentation_dir: Path,
    output_dir: Path,
    baseline_dir: Path | None = None,
) -> dict[str, Any]:
    dicom_path = Path(dicom_path)
    segmentation_dir = Path(segmentation_dir)
    output_dir, baseline_dir = _validated_temporal_paths(output_dir, baseline_dir)
    return build_then_publish_directory(
        output_dir,
        lambda staging_dir: _run_temporal_in_staging(
            dicom_path,
            segmentation_dir,
            staging_dir,
            baseline_dir,
            logical_output_dir=output_dir,
        ),
        seed_from_existing=True,
    )


def _run_temporal_in_staging(
    dicom_path: Path,
    segmentation_dir: Path,
    output_dir: Path,
    baseline_dir: Path | None,
    *,
    logical_output_dir: Path,
) -> dict[str, Any]:
    probability_path = segmentation_dir / "probabilities_u8.raw"
    if not probability_path.is_file():
        raise ValueError(
            f"Probability input is missing: {probability_path}; no output was changed."
        )
    _, frames, frame_times = read_cine(dicom_path)
    shape = frames.shape
    published_baseline = (
        _load_published_temporal_baseline(baseline_dir, shape)
        if baseline_dir is not None
        else None
    )
    baseline_manifest = _validate_existing_baseline(
        dicom_path, probability_path, output_dir
    )
    if baseline_manifest is not None:
        _validate_baseline_completeness(baseline_manifest, shape)
        _validate_baseline_media(output_dir, shape)
    raw = np.fromfile(probability_path, dtype=np.uint8)
    if raw.size != int(np.prod(shape)):
        raise ValueError(f"Probability stack size mismatch: {raw.size} != {np.prod(shape)}")
    probabilities = raw.reshape(shape).astype(np.float32) / 255.0
    catheter_path = segmentation_dir / "catheter_exclusions_u8.raw"
    catheter_exclusion_available = catheter_path.exists()
    catheter_exclusions = None
    if catheter_exclusion_available:
        catheter_raw = np.fromfile(catheter_path, dtype=np.uint8)
        if catheter_raw.size != int(np.prod(shape)):
            raise ValueError(
                "Catheter exclusion stack size mismatch: "
                f"{catheter_raw.size} != {np.prod(shape)}"
            )
        catheter_exclusions = catheter_raw.reshape(shape) != 0

    if baseline_manifest is None:
        flow_evidence = _temporal_fusion_evidence(
            frames, probabilities, probabilities >= 0.5
        )
        computed_fused = flow_evidence.fused_probability
        computed_masks = np.asarray(
            [postprocess_probability(frame) for frame in computed_fused], dtype=bool
        )
        computed_skeletons = np.asarray(
            [morphology.skeletonize(mask) for mask in computed_masks], dtype=bool
        )
        fused_u8 = np.round(computed_fused * 255).astype(np.uint8)
        temporal_support_u8 = np.round(
            flow_evidence.support_probability * 255
        ).astype(np.uint8)
        flow_confidence_u8 = np.round(
            flow_evidence.flow_confidence * 255
        ).astype(np.uint8)
        masks_u8 = computed_masks.astype(np.uint8) * 255
        skeletons_u8 = computed_skeletons.astype(np.uint8) * 255
        overlays = make_overlays(
            frames, computed_fused, computed_masks, computed_skeletons
        )
        computed_metrics = temporal_metrics(
            frames, computed_masks, computed_skeletons
        )
        diagnostics = _baseline_diagnostics(flow_evidence, computed_metrics)

        write_frames(
            fused_u8, output_dir / "probabilities", "temporal_probability"
        )
        write_frames(
            masks_u8, output_dir / "masks", "temporal_mask"
        )
        write_frames(
            skeletons_u8, output_dir / "centerlines", "temporal_centerline"
        )
        write_frames(
            overlays, output_dir / "overlays", "temporal_overlay"
        )
        write_raw_stack(output_dir / "temporal_probabilities_u8.raw", fused_u8)
        write_raw_stack(
            output_dir / "temporal_support_u8.raw", temporal_support_u8
        )
        write_raw_stack(
            output_dir / "flow_confidence_u8.raw", flow_confidence_u8
        )
        write_raw_stack(output_dir / "temporal_masks_u8.raw", masks_u8)
        write_raw_stack(
            output_dir / "temporal_centerlines_u8.raw", skeletons_u8
        )
        write_mp4(
            output_dir / _BASELINE_VIDEO_TARGET,
            overlays,
            frame_times,
            color=True,
        )
        _validate_baseline_media(output_dir, shape)
        baseline_manifest = _build_baseline_manifest(
            dicom_path, probability_path, output_dir, diagnostics
        )
        _validate_baseline_completeness(baseline_manifest, shape)
        write_json(output_dir / _BASELINE_MANIFEST, baseline_manifest)

    fused_u8 = _load_baseline_raw(
        output_dir, "temporal_probabilities_u8.raw", shape
    )
    temporal_support_u8 = _load_baseline_raw(
        output_dir, "temporal_support_u8.raw", shape
    )
    flow_confidence_u8 = _load_baseline_raw(
        output_dir, "flow_confidence_u8.raw", shape
    )
    masks_u8 = _load_baseline_raw(output_dir, "temporal_masks_u8.raw", shape)
    skeletons_u8 = _load_baseline_raw(
        output_dir, "temporal_centerlines_u8.raw", shape
    )
    fused = fused_u8.astype(np.float32) / 255.0
    support_probability = temporal_support_u8.astype(np.float32) / 255.0
    persisted_flow_confidence = flow_confidence_u8.astype(np.float32) / 255.0
    masks = masks_u8 != 0
    skeletons = skeletons_u8 != 0
    _validate_persisted_diagnostic_jumps(
        baseline_manifest, masks, skeletons
    )

    repaired_probabilities, morphology_masks, _morphology_audits = _repair_temporal_sequence(
        frames,
        probabilities,
        fused,
        support_probability,
        masks,
        catheter_exclusions,
    )
    repaired_masks, repair_audits, gap_links = _apply_short_gap_repairs(
        frames,
        masks,
        morphology_masks,
        repaired_probabilities,
        support_probability,
        catheter_exclusions,
    )
    repaired_skeletons = np.asarray(
        [morphology.skeletonize(mask) for mask in repaired_masks], dtype=bool
    )
    observed_graphs = [
        replace(
            extract_vessel_graph(mask, root_hint_yx=None), frame_index=frame_index
        )
        for frame_index, mask in enumerate(repaired_masks)
    ]
    forward_flows = [
        dense_flow(frames[index], frames[index + 1])
        for index in range(len(frames) - 1)
    ]
    tracked_graphs = track_vessel_graphs(observed_graphs, forward_flows)
    graph_path = output_dir / "temporal_vessel_graphs.json"
    write_json(graph_path, _temporal_graph_document(tracked_graphs))
    diagnostics = baseline_manifest["diagnostics"]
    frame_diagnostics = diagnostics["frames"]
    metrics = diagnostics["temporal_metrics"]
    topology = [asdict(mask_topology_metrics(mask)) for mask in masks]
    repaired_topology_objects = [mask_topology_metrics(mask) for mask in repaired_masks]
    repaired_topology = [asdict(item) for item in repaired_topology_objects]

    repaired_probabilities_u8 = np.round(repaired_probabilities * 255).astype(np.uint8)
    repaired_masks_u8 = repaired_masks.astype(np.uint8) * 255
    repaired_skeletons_u8 = repaired_skeletons.astype(np.uint8) * 255
    repaired_overlays = make_overlays(
        frames, repaired_probabilities, repaired_masks, repaired_skeletons
    )
    baseline_targets = baseline_manifest["baseline_targets"]
    assets = {
        "probabilities": {"path": "temporal_probabilities_u8.raw", "byte_length": baseline_targets["temporal_probabilities_u8.raw"]["byte_length"]},
        "temporal_support": {"path": "temporal_support_u8.raw", "byte_length": baseline_targets["temporal_support_u8.raw"]["byte_length"]},
        "flow_confidence": {"path": "flow_confidence_u8.raw", "byte_length": baseline_targets["flow_confidence_u8.raw"]["byte_length"]},
        "masks": {"path": "temporal_masks_u8.raw", "byte_length": baseline_targets["temporal_masks_u8.raw"]["byte_length"]},
        "centerlines": {"path": "temporal_centerlines_u8.raw", "byte_length": baseline_targets["temporal_centerlines_u8.raw"]["byte_length"]},
        "baseline_provenance": {"path": _BASELINE_MANIFEST, "schema_version": 1},
        "repaired_probabilities": {"path": "repaired_probabilities_u8.raw", "byte_length": write_raw_stack(output_dir / "repaired_probabilities_u8.raw", repaired_probabilities_u8)},
        "repaired_masks": {"path": "repaired_masks_u8.raw", "byte_length": write_raw_stack(output_dir / "repaired_masks_u8.raw", repaired_masks_u8)},
        "repaired_centerlines": {"path": "repaired_centerlines_u8.raw", "byte_length": write_raw_stack(output_dir / "repaired_centerlines_u8.raw", repaired_skeletons_u8)},
        "repair_audit": {"path": "repair_audit.json"},
        "repaired_overlay": {"path": "repaired_overlay.mp4"},
        "temporal_vessel_graphs": {
            "path": "temporal_vessel_graphs.json",
            "schema_version": 1,
            "byte_length": graph_path.stat().st_size,
        },
    }
    write_mp4(
        output_dir / "repaired_overlay.mp4",
        repaired_overlays,
        frame_times,
        color=True,
    )
    repair_audit = {
        "schema_version": 1,
        "method": (
            "Constrained seed reconstruction, aligned anisotropic morphology, "
            "and direction-aware minimum-cost short-gap repair"
        ),
        "frame_count": int(len(frames)),
        "catheter_exclusion_available": catheter_exclusion_available,
        "thresholds": {
            "seed": 0.70,
            "corridor": 0.20,
            "temporal_addition": 0.35,
            "max_growth_fraction": 0.20,
            "automatic_gap_max_px": 12,
            "review_gap_max_px": 24,
            "maximum_direction_angle_deg": 45.0,
        },
        "frames": [
            {
                "frame_index": index,
                **asdict(audit),
                "topology_before": topology[index],
                "topology_after": repaired_topology[index],
            }
            for index, audit in enumerate(repair_audits)
        ],
        "links": gap_links,
        "clinical_status": "research-only; repaired outputs require diagnostic review",
    }
    write_json(output_dir / "repair_audit.json", repair_audit)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "temporal_quality.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(metrics[0]))
        writer.writeheader()
        writer.writerows(metrics)
    report = {
        "schema_version": 2,
        "method": "DeepSA probabilities plus bidirectional neighboring-frame Farneback flow fusion",
        "frame_count": int(len(frames)),
        "frames": [
            {
                "frame_index": index,
                "topology_before": item,
                "topology_after": repaired_topology[index],
                "median_flow_confidence": float(np.median(persisted_flow_confidence[index])),
                "supported_pixel_fraction": float(np.mean(support_probability[index] > 0.0)),
                "forward_backward_error_p95_px": frame_diagnostics[index][
                    "forward_backward_error_p95_px"
                ],
                "valid_neighbor_count": frame_diagnostics[index][
                    "valid_neighbor_count"
                ],
            }
            for index, item in enumerate(topology)
        ],
        "assets": assets,
        "quality_summary": diagnostics["quality_summary"],
        "clinical_status": "research-only; temporal smoothing does not create missing anatomy",
        "upgrade_path": "Replace Farneback with validated SEA-RAFT/RAFT flow after domain evaluation.",
    }
    write_json(output_dir / "temporal_report.json", report)
    if baseline_dir is not None:
        assert published_baseline is not None
        published_masks = published_baseline["masks"]
        published_skeletons = published_baseline["skeletons"]
        baseline_metric_rows = published_baseline["metric_rows"]
        if baseline_metric_rows is None:
            baseline_metric_rows = temporal_metrics(
                frames, published_masks, published_skeletons
            )
        candidate_metric_rows = temporal_metrics(
            frames, repaired_masks, repaired_skeletons
        )
        baseline_metrics = _acceptance_metrics(
            [mask_topology_metrics(mask) for mask in published_masks],
            baseline_metric_rows,
            published_masks,
            published_masks,
            [],
        )
        candidate_metrics = _acceptance_metrics(
            repaired_topology_objects,
            candidate_metric_rows,
            masks,
            repaired_masks,
            gap_links,
        )
        contact_sheet_path = output_dir / "repair_before_after.png"
        write_repair_contact_sheet(
            contact_sheet_path,
            frames,
            published_masks,
            repaired_masks,
            repaired_skeletons,
            gap_links,
        )
        baseline_assets = published_baseline["assets"]
        candidate_assets = _raw_asset_records(
            output_dir,
            (
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
            ),
        )
        acceptance = build_repair_acceptance_report(
            run_id=logical_output_dir.parent.name,
            baseline={
                "directory": str(baseline_dir),
                "metrics": baseline_metrics,
                "assets": baseline_assets,
                "historical_hard_depth_cut_events": 250,
                "historical_hard_depth_cut_note": (
                    "Prior audit count; surviving legacy paths contain no >14 mm within-path step "
                    "because the old generator split at those events."
                ),
            },
            candidate={
                "directory": str(logical_output_dir),
                "metrics": candidate_metrics,
                "assets": candidate_assets,
                "comparisons": {
                    "candidate_pre_repair_growth": {
                        "max_added_pixel_fraction": candidate_metrics[
                            "max_component_growth_fraction"
                        ]
                    },
                    "published_baseline_delta": _mask_delta_metrics(
                        published_masks, repaired_masks
                    ),
                },
                "link_decision_counts": {
                    decision: sum(link["decision"] == decision for link in gap_links)
                    for decision in ("accepted", "review", "rejected")
                },
            },
        )
        write_json(output_dir / "repair_acceptance.json", acceptance)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dicom", type=Path, required=True)
    parser.add_argument("--segmentation-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--baseline-dir", type=Path, default=None)
    args = parser.parse_args()
    report = run_temporal(
        args.dicom,
        args.segmentation_dir,
        args.output_dir,
        baseline_dir=args.baseline_dir,
    )
    print(
        json.dumps(
            report["quality_summary"], ensure_ascii=False, allow_nan=False
        )
    )


if __name__ == "__main__":
    main()
