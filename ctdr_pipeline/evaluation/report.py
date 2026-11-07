from __future__ import annotations

import csv
import hashlib
import importlib.metadata
import json
import os
import platform
import re
import sys
from collections import Counter
from dataclasses import asdict, dataclass, field, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import xlsxwriter
from PIL import Image, ImageDraw

from ctdr_pipeline.evaluation.metrics import CaseMetrics
from ctdr_pipeline.evaluation.quality import QualityDecision


_CASE_ID = re.compile(r"^\d{6,12}$")
_PHI_KEYS = {
    "patientname",
    "patientid",
    "patientbirthdate",
    "patientbirthtime",
    "patientaddress",
    "patienttelephone",
    "institutionname",
    "referringphysicianname",
    "performingphysicianname",
    "operatorsname",
    "accessionnumber",
}


@dataclass(frozen=True)
class CaseReport:
    case_id: str
    decision: QualityDecision
    metrics: CaseMetrics
    candidates: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    diagnostic_images: dict[str, np.ndarray] = field(default_factory=dict, repr=False)


def _json_value(value: Any) -> Any:
    if is_dataclass(value):
        return _json_value(asdict(value))
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return _json_value(value.tolist())
    if torch.is_tensor(value):
        return _json_value(value.detach().cpu().tolist())
    if isinstance(value, np.generic):
        return _json_value(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_value(item) for item in value]
    return value


def _normalized_key(key: str) -> str:
    return re.sub(r"[^a-z0-9]", "", key.lower())


def _sanitize_metadata(value: Any, key: str | None = None) -> Any:
    normalized = _normalized_key(key or "")
    if normalized in _PHI_KEYS:
        return None
    if key and normalized.endswith("uid") and value not in (None, ""):
        digest = hashlib.sha256(str(value).encode("utf-8")).hexdigest()
        return f"sha256:{digest}"
    if isinstance(value, dict):
        cleaned: dict[str, Any] = {}
        for child_key, child_value in value.items():
            child_normalized = _normalized_key(str(child_key))
            if child_normalized in _PHI_KEYS:
                continue
            cleaned[str(child_key)] = _sanitize_metadata(child_value, str(child_key))
        return cleaned
    if isinstance(value, (list, tuple)):
        return [_sanitize_metadata(item) for item in value]
    return _json_value(value)


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def _write_json(path: Path, value: Any) -> None:
    _atomic_text(
        path,
        json.dumps(_json_value(value), ensure_ascii=False, indent=2, allow_nan=False) + "\n",
    )


def _atomic_csv(path: Path, rows: Sequence[dict[str, Any]], headers: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_value(row.get(key)) for key in headers})
    temporary.replace(path)


def _csv_value(value: Any) -> Any:
    value = _json_value(value)
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    return value


def _candidate_row(candidate: dict[str, Any]) -> dict[str, Any]:
    sanitized = _sanitize_metadata(candidate)
    row = dict(sanitized)
    pose = row.pop("final_pose", None)
    if isinstance(pose, list) and len(pose) == 6:
        for name, value in zip(
            ("rotation_z_deg", "rotation_x_deg", "rotation_y_deg", "translation_x_mm", "translation_y_mm", "translation_z_mm"),
            pose,
        ):
            row[name] = value
    return row


def _save_image(path: Path, array: np.ndarray, label: str | None = None) -> None:
    image = np.asarray(array).squeeze()
    if image.ndim not in (2, 3):
        raise ValueError(f"diagnostic image {path.name} must be 2D or RGB")
    image = image.astype(np.float32, copy=False)
    finite = image[np.isfinite(image)]
    if finite.size == 0:
        normalized = np.zeros(image.shape, dtype=np.uint8)
    else:
        low, high = np.percentile(finite, [1.0, 99.0])
        normalized = np.clip((np.nan_to_num(image, nan=low) - low) / max(high - low, 1e-6), 0.0, 1.0)
        normalized = np.rint(normalized * 255.0).astype(np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.stem}.{os.getpid()}.tmp.png")
    rendered = Image.fromarray(normalized)
    if label:
        if rendered.mode != "RGB":
            rendered = rendered.convert("RGB")
        draw = ImageDraw.Draw(rendered)
        width = max(72, 8 * len(label) + 12)
        draw.rectangle((0, 0, width, 22), fill=(150, 0, 0))
        draw.text((6, 4), label, fill=(255, 255, 255))
    rendered.save(temporary)
    temporary.replace(path)


def write_case_report(case_dir: Path, report: CaseReport) -> None:
    if not _CASE_ID.fullmatch(report.case_id):
        raise ValueError("case_id must contain only 6-12 digits")
    case_dir = Path(case_dir)
    status = {
        "case_id": report.case_id,
        "status": report.decision.status.value,
        "accepted": report.decision.accepted,
        "failed_rules": list(report.decision.failed_rules),
    }
    metrics = _json_value(report.metrics)
    candidates = [_candidate_row(candidate) for candidate in report.candidates]
    sanitized_metadata = _sanitize_metadata(report.metadata)
    _write_json(case_dir / "status.json", status)
    _write_json(case_dir / "evaluation" / "metrics.json", metrics)
    _write_json(case_dir / "metadata.json", sanitized_metadata)
    _write_json(
        case_dir / "report.json",
        {"status": status, "metrics": metrics, "metadata": sanitized_metadata, "candidates": candidates},
    )
    headers = sorted({key for row in candidates for key in row}) or ["name"]
    _atomic_csv(case_dir / "registration3d" / "candidates.csv", candidates, headers)
    diagnostics = case_dir / "evaluation" / "diagnostics"
    image_label = None if report.decision.accepted else report.decision.status.value.upper()
    for name, image in sorted(report.diagnostic_images.items()):
        safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("._") or "image"
        _save_image(diagnostics / f"{safe_name}.png", image, image_label)


def _summary_row(report: CaseReport) -> dict[str, Any]:
    metrics = _json_value(report.metrics)
    return {
        "case_id": report.case_id,
        "status": report.decision.status.value,
        "accepted": report.decision.accepted,
        "failed_rules": list(report.decision.failed_rules),
        "pptx_rms_diagonal": metrics.get("pptx_rms_diagonal"),
        "pptx_p95_diagonal": metrics.get("pptx_p95_diagonal"),
        "pptx_inlier_ratio": metrics.get("pptx_inlier_ratio"),
        "affine_condition_number": metrics.get("affine_condition_number"),
        "dicom_confidence": metrics.get("dicom_confidence"),
        "dicom_candidate_margin": metrics.get("dicom_candidate_margin"),
        "drr_roi_ncc": metrics.get("drr_roi_ncc"),
        "drr_gradient_ncc": metrics.get("drr_gradient_ncc"),
        "drr_objective_improvement": metrics.get("drr_objective_improvement"),
        "pose_translation_spread_mm": metrics.get("pose_translation_spread_mm"),
        "pose_rotation_spread_deg": metrics.get("pose_rotation_spread_deg"),
        "fov_iou": metrics.get("fov_iou"),
        "tre_mm": metrics.get("tre_mm"),
        "tre_reason": metrics.get("tre_reason"),
    }


def capture_environment() -> dict[str, Any]:
    try:
        diffdrr_version = importlib.metadata.version("diffdrr")
    except importlib.metadata.PackageNotFoundError:
        diffdrr_version = None
    cuda_devices = []
    if torch.cuda.is_available():
        cuda_devices = [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())]
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "diffdrr": diffdrr_version,
        "cuda_runtime": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "cuda_devices": cuda_devices,
    }


def _write_table_sheet(
    workbook: xlsxwriter.Workbook,
    name: str,
    rows: list[dict[str, Any]],
    headers: list[str],
) -> None:
    worksheet = workbook.add_worksheet(name)
    title = workbook.add_format(
        {"bold": True, "font_color": "#FFFFFF", "bg_color": "#1F4E78", "border": 1}
    )
    number = workbook.add_format({"num_format": "0.0000"})
    for column, header in enumerate(headers):
        worksheet.write(0, column, header, title)
    for row_index, row in enumerate(rows, start=1):
        for column, header in enumerate(headers):
            value = _csv_value(row.get(header))
            worksheet.write(row_index, column, value, number if isinstance(value, float) else None)
    worksheet.freeze_panes(1, 0)
    worksheet.autofilter(0, 0, max(1, len(rows)), len(headers) - 1)
    worksheet.set_column(0, len(headers) - 1, 22)
    if "status" in headers and rows:
        status_column = headers.index("status")
        worksheet.conditional_format(
            1,
            status_column,
            len(rows),
            status_column,
            {"type": "text", "criteria": "containing", "value": "accepted", "format": workbook.add_format({"bg_color": "#C6EFCE", "font_color": "#006100"})},
        )
        worksheet.conditional_format(
            1,
            status_column,
            len(rows),
            status_column,
            {"type": "text", "criteria": "containing", "value": "review", "format": workbook.add_format({"bg_color": "#FFEB9C", "font_color": "#9C6500"})},
        )
        worksheet.conditional_format(
            1,
            status_column,
            len(rows),
            status_column,
            {"type": "text", "criteria": "containing", "value": "reject", "format": workbook.add_format({"bg_color": "#FFC7CE", "font_color": "#9C0006"})},
        )


def _write_summary_xlsx(
    path: Path, rows: list[dict[str, Any]], environment: dict[str, Any]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    workbook = xlsxwriter.Workbook(str(temporary), {"constant_memory": True})
    _write_table_sheet(
        workbook,
        "run_summary",
        rows,
        ["case_id", "status", "accepted", "failed_rules", "tre_mm", "tre_reason"],
    )
    _write_table_sheet(
        workbook,
        "errors_2d",
        rows,
        ["case_id", "pptx_rms_diagonal", "pptx_p95_diagonal", "pptx_inlier_ratio", "affine_condition_number"],
    )
    _write_table_sheet(
        workbook,
        "dicom_linkage",
        rows,
        ["case_id", "dicom_confidence", "dicom_candidate_margin"],
    )
    _write_table_sheet(
        workbook,
        "registration_3d",
        rows,
        [
            "case_id",
            "drr_roi_ncc",
            "drr_gradient_ncc",
            "drr_objective_improvement",
            "pose_translation_spread_mm",
            "pose_rotation_spread_deg",
            "fov_iou",
            "tre_mm",
            "tre_reason",
        ],
    )
    _write_table_sheet(
        workbook,
        "quality_reasons",
        rows,
        ["case_id", "status", "accepted", "failed_rules"],
    )
    environment_rows = [
        {"key": key, "value": _csv_value(value)} for key, value in sorted(environment.items())
    ]
    _write_table_sheet(workbook, "environment", environment_rows, ["key", "value"])
    workbook.close()
    temporary.replace(path)


def write_run_summary(run_dir: Path, cases: Sequence[CaseReport]) -> None:
    run_dir = Path(run_dir)
    for report in cases:
        if not _CASE_ID.fullmatch(report.case_id):
            raise ValueError("case_id must contain only 6-12 digits")
    rows = [_summary_row(report) for report in sorted(cases, key=lambda item: item.case_id)]
    counts = Counter(row["status"] for row in rows)
    _write_json(
        run_dir / "summary.json",
        {"case_count": len(rows), "status_counts": dict(sorted(counts.items())), "cases": rows},
    )
    headers = list(rows[0]) if rows else ["case_id", "status"]
    _atomic_csv(run_dir / "summary.csv", rows, headers)
    environment = capture_environment()
    _write_summary_xlsx(run_dir / "summary.xlsx", rows, environment)
    _write_json(run_dir / "environment.json", environment)
