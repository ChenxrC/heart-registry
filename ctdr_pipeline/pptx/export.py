from __future__ import annotations

import csv
import io
import json
import math
import shutil
import uuid
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable
from zipfile import ZipFile

import cv2
import numpy as np
import xlsxwriter
from PIL import Image, ImageDraw

from ctdr_pipeline.bundle import BundleCaseRecord, BundleIndex, sha256_file
from ctdr_pipeline.pptx.annotations import AnnotationSet, extract_annotations
from ctdr_pipeline.pptx.grouping import PptxCaseGroup, group_case_slides
from ctdr_pipeline.pptx.reader import ImageGeometry, SlideRecord, read_pptx
from ctdr_pipeline.registration.affine2d import (
    Registration2DResult,
    estimate_affine_ransac,
    register_landmarks,
    transform_points,
)


PAIR_FIELDS = [
    "case_id",
    "slide_numbers",
    "image_count",
    "pair_status",
    "registration_status",
    "source_image",
    "target_image",
    "reasons",
]
LANDMARK_FIELDS = [
    "case_id",
    "landmark_id",
    "source_x_px",
    "source_y_px",
    "target_x_px",
    "target_y_px",
    "predicted_x_px",
    "predicted_y_px",
    "residual_px",
    "inlier",
]
MATRIX_FIELDS = [
    "case_id",
    "source_space",
    "target_space",
    "m00",
    "m01",
    "m02",
    "m10",
    "m11",
    "m12",
    "m20",
    "m21",
    "m22",
    "rotation_deg",
    "scale_x",
    "scale_y",
    "shear",
    "translation_x_px",
    "translation_y_px",
    "determinant",
]
VALIDATION_FIELDS = [
    "case_id",
    "status",
    "accepted",
    "rms_px",
    "p95_px",
    "normalized_rms",
    "normalized_p95",
    "inlier_ratio",
    "condition_number",
    "order_consistent",
    "reasons",
]


def _json_dump(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _write_csv(path: Path, fields: list[str], rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _largest_images(group: PptxCaseGroup) -> list[tuple[str, ImageGeometry]]:
    selected: list[tuple[str, ImageGeometry]] = []
    for slide in group.slides:
        candidates = list(zip(slide.image_members, slide.image_geometry, strict=False))
        candidates.sort(
            key=lambda item: item[1].extent_x * item[1].extent_y,
            reverse=True,
        )
        if candidates:
            selected.append(candidates[0])
    if len(selected) == 2:
        return selected
    flattened: list[tuple[str, ImageGeometry]] = []
    for slide in group.slides:
        flattened.extend(zip(slide.image_members, slide.image_geometry, strict=False))
    return flattened[:2]


def _save_png(data: bytes, path: Path) -> np.ndarray:
    image = Image.open(io.BytesIO(data)).convert("RGB")
    image.save(path)
    return np.asarray(image)


def _load_overrides(path: Path | None) -> dict[str, dict[str, Any]]:
    if path is None:
        return {}
    raw = json.loads(path.read_text(encoding="utf-8"))
    if "case_id" in raw:
        return {str(raw["case_id"]): raw}
    if "cases" in raw and isinstance(raw["cases"], list):
        return {str(case["case_id"]): case for case in raw["cases"]}
    raise ValueError("landmark overrides must contain case_id or a cases list")


def _bbox_points(
    override: dict[str, Any], role: str
) -> tuple[np.ndarray, list[str]]:
    points: list[list[float]] = []
    ids: list[str] = []
    key = f"{role}_bbox_xywh"
    for item in override["landmarks"]:
        x, y, width, height = (float(value) for value in item[key])
        points.append([x + width / 2.0, y + height / 2.0])
        ids.append(str(item["id"]))
    return np.asarray(points, dtype=np.float64), ids


def _register_pair(
    case_id: str,
    source_rgb: np.ndarray,
    target_rgb: np.ndarray,
    overrides: dict[str, dict[str, Any]],
) -> tuple[AnnotationSet, AnnotationSet, Registration2DResult | None, list[str]]:
    source_annotations = extract_annotations(source_rgb)
    target_annotations = extract_annotations(target_rgb)
    diagonal = float(math.hypot(target_rgb.shape[1], target_rgb.shape[0]))
    try:
        if case_id in overrides:
            source_points, ids = _bbox_points(overrides[case_id], "source")
            target_points, target_ids = _bbox_points(overrides[case_id], "target")
            if ids != target_ids:
                raise ValueError("source and target override IDs differ")
            result = estimate_affine_ransac(
                source_points,
                target_points,
                image_diagonal=diagonal,
                ransac_threshold_px=max(3.0, diagonal * 0.02),
                landmark_ids=ids,
            )
        else:
            result = register_landmarks(
                source_annotations,
                target_annotations,
                image_diagonal=diagonal,
            )
        return source_annotations, target_annotations, result, list(result.reasons)
    except (ValueError, cv2.error) as exc:
        return source_annotations, target_annotations, None, [str(exc)]


def _result_rows(
    case_id: str, result: Registration2DResult
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    source = np.asarray([landmark.source_xy for landmark in result.landmarks])
    predicted = transform_points(source, result.matrix)
    landmark_rows: list[dict[str, Any]] = []
    for index, landmark in enumerate(result.landmarks):
        landmark_rows.append(
            {
                "case_id": case_id,
                "landmark_id": landmark.landmark_id,
                "source_x_px": landmark.source_xy[0],
                "source_y_px": landmark.source_xy[1],
                "target_x_px": landmark.target_xy[0],
                "target_y_px": landmark.target_xy[1],
                "predicted_x_px": float(predicted[index, 0]),
                "predicted_y_px": float(predicted[index, 1]),
                "residual_px": result.residuals_px[index],
                "inlier": result.inliers[index],
            }
        )
    flat = result.matrix.reshape(-1).tolist()
    metrics = result.metrics
    matrix_row = {
        "case_id": case_id,
        "source_space": "embedded_pixel",
        "target_space": "embedded_pixel",
        **{f"m{row}{column}": flat[row * 3 + column] for row in range(3) for column in range(3)},
        "rotation_deg": metrics.rotation_deg,
        "scale_x": metrics.scale_x,
        "scale_y": metrics.scale_y,
        "shear": metrics.shear,
        "translation_x_px": metrics.translation_x,
        "translation_y_px": metrics.translation_y,
        "determinant": metrics.determinant,
    }
    validation_row = {
        "case_id": case_id,
        "status": "accepted" if result.accepted else "review_required_2d",
        "accepted": result.accepted,
        "rms_px": metrics.rms_px,
        "p95_px": metrics.p95_px,
        "normalized_rms": metrics.normalized_rms,
        "normalized_p95": metrics.normalized_p95,
        "inlier_ratio": metrics.inlier_ratio,
        "condition_number": metrics.condition_number,
        "order_consistent": metrics.order_consistent,
        "reasons": "|".join(result.reasons),
    }
    return landmark_rows, matrix_row, validation_row


def _save_overlay(
    source_rgb: np.ndarray,
    target_rgb: np.ndarray,
    result: Registration2DResult | None,
    output: Path,
) -> None:
    width = source_rgb.shape[1] + target_rgb.shape[1]
    height = max(source_rgb.shape[0], target_rgb.shape[0])
    canvas = Image.new("RGB", (width, height), "black")
    canvas.paste(Image.fromarray(source_rgb), (0, 0))
    canvas.paste(Image.fromarray(target_rgb), (source_rgb.shape[1], 0))
    if result is not None:
        draw = ImageDraw.Draw(canvas)
        for landmark in result.landmarks:
            sx, sy = landmark.source_xy
            tx, ty = landmark.target_xy
            draw.ellipse((sx - 4, sy - 4, sx + 4, sy + 4), outline="lime", width=2)
            draw.ellipse(
                (
                    source_rgb.shape[1] + tx - 4,
                    ty - 4,
                    source_rgb.shape[1] + tx + 4,
                    ty + 4,
                ),
                outline="yellow",
                width=2,
            )
    canvas.save(output)


def _write_table_sheet(
    workbook: xlsxwriter.Workbook,
    name: str,
    fields: list[str],
    rows: list[dict[str, Any]],
    formats: dict[str, xlsxwriter.format.Format],
) -> None:
    sheet = workbook.add_worksheet(name)
    sheet.hide_gridlines(2)
    sheet.freeze_panes(1, 1)
    for column, field in enumerate(fields):
        sheet.write(0, column, field, formats["header"])
        width = min(38, max(12, len(field) + 2))
        sheet.set_column(column, column, width)
    for row_number, row in enumerate(rows, start=1):
        for column, field in enumerate(fields):
            value = row.get(field)
            cell_format = formats["number"] if isinstance(value, float) else formats["body"]
            sheet.write(row_number, column, value, cell_format)
    if fields:
        sheet.autofilter(0, 0, max(0, len(rows)), len(fields) - 1)
    if "status" in fields and rows:
        status_column = fields.index("status")
        sheet.conditional_format(
            1,
            status_column,
            len(rows),
            status_column,
            {"type": "text", "criteria": "containing", "value": "accepted", "format": formats["accepted"]},
        )
        sheet.conditional_format(
            1,
            status_column,
            len(rows),
            status_column,
            {"type": "text", "criteria": "containing", "value": "review", "format": formats["review"]},
        )


def _write_workbook(
    path: Path,
    pair_rows: list[dict[str, Any]],
    landmark_rows: list[dict[str, Any]],
    matrix_rows: list[dict[str, Any]],
    validation_rows: list[dict[str, Any]],
    skipped_rows: list[dict[str, Any]],
) -> None:
    workbook = xlsxwriter.Workbook(path)
    header = workbook.add_format({"bold": True, "font_color": "#FFFFFF", "bg_color": "#0F766E", "align": "center"})
    body = workbook.add_format({"font_color": "#1F2937", "valign": "top"})
    number = workbook.add_format({"font_color": "#1F2937", "num_format": "0.000"})
    accepted = workbook.add_format({"bg_color": "#DCFCE7", "font_color": "#166534"})
    review = workbook.add_format({"bg_color": "#FEF3C7", "font_color": "#92400E"})
    formats = {"header": header, "body": body, "number": number, "accepted": accepted, "review": review}
    readme = workbook.add_worksheet("README")
    readme.hide_gridlines(2)
    readme.set_column("A:A", 28)
    readme.set_column("B:B", 90)
    readme.write_row("A1", ["项目", "说明"], header)
    notes = [
        ("用途", "医生审阅用；服务器计算以同目录 JSON/CSV 为准。"),
        ("病例号", "来自 PPTX 文本框中的 8 位数字。"),
        ("坐标", "x 向右、y 向下，单位为原始嵌入图像像素。"),
        ("矩阵", "3×3 齐次 source-to-target 仿射矩阵；不允许镜像自动通过。"),
        ("状态颜色", "绿色=accepted，黄色=review，跳过原因见 skipped_cases。"),
        ("临床声明", "研究候选，不等同于临床级配准结果。"),
    ]
    for row, values in enumerate(notes, start=1):
        readme.write_row(row, 0, values, body)
    _write_table_sheet(workbook, "pair_index", PAIR_FIELDS, pair_rows, formats)
    _write_table_sheet(workbook, "landmarks", LANDMARK_FIELDS, landmark_rows, formats)
    _write_table_sheet(workbook, "matrices", MATRIX_FIELDS, matrix_rows, formats)
    _write_table_sheet(workbook, "validation", VALIDATION_FIELDS, validation_rows, formats)
    _write_table_sheet(
        workbook,
        "dicom_linkage",
        ["case_id", "ct_series_uid", "localizer_instance", "xa_series_uid", "xa_frame", "status"],
        [],
        formats,
    )
    _write_table_sheet(workbook, "skipped_cases", PAIR_FIELDS, skipped_rows, formats)
    workbook.close()


def prepare_pptx(
    pptx_path: Path,
    output_dir: Path,
    overrides: Path | None = None,
) -> BundleIndex:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"output directory is not empty: {output_dir}")
    inventory = read_pptx(pptx_path)
    groups = group_case_slides(inventory.slides)
    override_map = _load_overrides(overrides)
    staging = output_dir.parent / f".{output_dir.name}.staging-{uuid.uuid4().hex}"
    staging.mkdir(parents=True, exist_ok=False)
    pair_rows: list[dict[str, Any]] = []
    landmark_rows: list[dict[str, Any]] = []
    matrix_rows: list[dict[str, Any]] = []
    validation_rows: list[dict[str, Any]] = []
    skipped_rows: list[dict[str, Any]] = []
    case_records: list[BundleCaseRecord] = []
    try:
        with ZipFile(io.BytesIO(pptx_path.read_bytes())) as archive:
            for group in groups:
                source_relative = None
                target_relative = None
                matrix_values = None
                reasons = list(group.issues)
                registration_status = group.pair_status
                if group.image_count == 2:
                    images = _largest_images(group)
                    if len(images) != 2:
                        registration_status = "review_required_pptx"
                        reasons.append("could_not_select_two_primary_images")
                    else:
                        case_dir = staging / "cases" / group.case_id
                        case_dir.mkdir(parents=True, exist_ok=True)
                        source_rgb = _save_png(archive.read(images[0][0]), case_dir / "source.png")
                        target_rgb = _save_png(archive.read(images[1][0]), case_dir / "target.png")
                        source_relative = f"cases/{group.case_id}/source.png"
                        target_relative = f"cases/{group.case_id}/target.png"
                        source_annotations, target_annotations, result, registration_reasons = _register_pair(
                            group.case_id, source_rgb, target_rgb, override_map
                        )
                        Image.fromarray(source_annotations.mask * 255).save(case_dir / "source_mask.png")
                        Image.fromarray(target_annotations.mask * 255).save(case_dir / "target_mask.png")
                        _save_overlay(source_rgb, target_rgb, result, case_dir / "landmark_overlay.png")
                        reasons.extend(registration_reasons)
                        if result is None:
                            registration_status = "review_required_2d"
                            pair_payload: dict[str, Any] = {
                                "case_id": group.case_id,
                                "slide_numbers": group.slide_numbers,
                                "status": registration_status,
                                "reasons": reasons,
                            }
                        else:
                            registration_status = "accepted" if result.accepted else "review_required_2d"
                            matrix_values = result.matrix.reshape(-1).tolist()
                            rows, matrix_row, validation_row = _result_rows(group.case_id, result)
                            landmark_rows.extend(rows)
                            matrix_rows.append(matrix_row)
                            validation_rows.append(validation_row)
                            pair_payload = {
                                "case_id": group.case_id,
                                "slide_numbers": group.slide_numbers,
                                "status": registration_status,
                                "reasons": reasons,
                                "matrix_3x3": result.matrix.tolist(),
                                "inverse_matrix_3x3": result.inverse_matrix.tolist(),
                                "metrics": asdict(result.metrics),
                                "landmarks": rows,
                            }
                        _json_dump(case_dir / "pair.json", pair_payload)
                record = BundleCaseRecord(
                    case_id=group.case_id,
                    slide_numbers=group.slide_numbers,
                    image_count=group.image_count,
                    pair_status=group.pair_status,
                    registration_status=registration_status,
                    source_image=source_relative,
                    target_image=target_relative,
                    matrix_3x3=matrix_values,
                    reasons=list(dict.fromkeys(reasons)),
                )
                case_records.append(record)
                row = {
                    "case_id": record.case_id,
                    "slide_numbers": ",".join(str(value) for value in record.slide_numbers),
                    "image_count": record.image_count,
                    "pair_status": record.pair_status,
                    "registration_status": record.registration_status,
                    "source_image": record.source_image or "",
                    "target_image": record.target_image or "",
                    "reasons": "|".join(record.reasons),
                }
                pair_rows.append(row)
                if group.image_count != 2:
                    skipped_rows.append(row)

        _write_csv(staging / "pairs.csv", PAIR_FIELDS, pair_rows)
        _write_csv(staging / "landmarks.csv", LANDMARK_FIELDS, landmark_rows)
        _write_csv(staging / "transforms.csv", MATRIX_FIELDS, matrix_rows)
        _json_dump(staging / "transforms.json", matrix_rows)
        _write_workbook(
            staging / "landmark_transforms.xlsx",
            pair_rows,
            landmark_rows,
            matrix_rows,
            validation_rows,
            skipped_rows,
        )
        checksum_files = [
            path
            for path in staging.rglob("*")
            if path.is_file() and path.name != "bundle.json"
        ]
        checksums = {
            path.relative_to(staging).as_posix(): sha256_file(path)
            for path in sorted(checksum_files)
        }
        index = BundleIndex(
            source_pptx=pptx_path.name,
            source_sha256=inventory.source_sha256,
            total_cases=len(groups),
            complete_pairs=sum(group.image_count == 2 for group in groups),
            incomplete_pairs=sum(group.image_count != 2 for group in groups),
            cases=case_records,
            checksums=checksums,
        )
        _json_dump(staging / "bundle.json", index.model_dump(mode="json"))
        output_dir.parent.mkdir(parents=True, exist_ok=True)
        staging.replace(output_dir)
        return index
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
