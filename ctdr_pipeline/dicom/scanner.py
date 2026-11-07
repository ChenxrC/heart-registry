from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import pydicom
import yaml
from pydantic import Field

from ctdr_pipeline.schemas import StrictModel


CASE_DIRECTORY_PATTERN = re.compile(r"^(\d+)")
DICOM_TAGS = [
    "StudyInstanceUID",
    "SeriesInstanceUID",
    "SOPInstanceUID",
    "SOPClassUID",
    "Modality",
    "SeriesNumber",
    "InstanceNumber",
    "ImageType",
    "Rows",
    "Columns",
    "PixelSpacing",
    "ImagerPixelSpacing",
    "ImageOrientationPatient",
    "ImagePositionPatient",
    "NumberOfFrames",
    "PositionerPrimaryAngle",
    "PositionerSecondaryAngle",
    "DistanceSourceToDetector",
    "DistanceSourceToPatient",
]


class DicomSeriesRecord(StrictModel):
    study_uid: str
    series_uid: str
    modality: str
    series_number: str = ""
    sop_class_uids: list[str] = Field(default_factory=list)
    image_types: list[list[str]] = Field(default_factory=list)
    relative_paths: list[str] = Field(default_factory=list)
    file_count: int = 0
    frame_count: int = 0
    rows: int | None = None
    columns: int | None = None
    pixel_spacing: list[float] | None = None
    orientation: list[float] | None = None
    positions: list[list[float]] = Field(default_factory=list)
    primary_angle_deg: float | None = None
    secondary_angle_deg: float | None = None
    sid_mm: float | None = None
    sod_mm: float | None = None

    @property
    def unique_slice_positions(self) -> int:
        return len({tuple(round(value, 5) for value in position) for position in self.positions})


class CaseDicomManifest(StrictModel):
    case_id: str
    directory: str | None = None
    candidate_directories: list[str] = Field(default_factory=list)
    status: str = "scanned"
    series: list[DicomSeriesRecord] = Field(default_factory=list)
    unreadable_files: int = 0

    def series_by_uid(self, uid: str) -> DicomSeriesRecord:
        for record in self.series:
            if record.series_uid == uid:
                return record
        raise KeyError(uid)


class DicomManifest(StrictModel):
    schema_version: str = "1.0"
    data_root: str
    cases: list[CaseDicomManifest]

    def case(self, case_id: str) -> CaseDicomManifest:
        for record in self.cases:
            if record.case_id == case_id:
                return record
        raise KeyError(case_id)


def _float_list(value: Any) -> list[float] | None:
    if value is None:
        return None
    try:
        return [float(item) for item in value]
    except (TypeError, ValueError):
        return None


def _optional_float(dataset: pydicom.Dataset, name: str) -> float | None:
    value = getattr(dataset, name, None)
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _scan_case(data_root: Path, case_id: str, directory: Path) -> CaseDicomManifest:
    grouped: dict[tuple[str, str], list[tuple[Path, pydicom.Dataset]]] = defaultdict(list)
    unreadable = 0
    for path in sorted(item for item in directory.rglob("*") if item.is_file()):
        try:
            dataset = pydicom.dcmread(
                path,
                stop_before_pixels=True,
                force=False,
                specific_tags=DICOM_TAGS,
            )
        except Exception:
            unreadable += 1
            continue
        study_uid = str(getattr(dataset, "StudyInstanceUID", ""))
        series_uid = str(getattr(dataset, "SeriesInstanceUID", ""))
        if not study_uid or not series_uid:
            unreadable += 1
            continue
        grouped[(study_uid, series_uid)].append((path, dataset))

    records: list[DicomSeriesRecord] = []
    for (study_uid, series_uid), members in sorted(grouped.items()):
        first = members[0][1]
        image_types: list[list[str]] = []
        positions: list[list[float]] = []
        sop_classes: set[str] = set()
        frame_count = 0
        for _, dataset in members:
            image_type = getattr(dataset, "ImageType", [])
            values = [str(value).upper() for value in image_type]
            if values and values not in image_types:
                image_types.append(values)
            position = _float_list(getattr(dataset, "ImagePositionPatient", None))
            if position is not None:
                positions.append(position)
            sop_class = str(getattr(dataset, "SOPClassUID", ""))
            if sop_class:
                sop_classes.add(sop_class)
            frame_count += int(getattr(dataset, "NumberOfFrames", 1) or 1)
        pixel_spacing = _float_list(getattr(first, "PixelSpacing", None))
        if pixel_spacing is None:
            pixel_spacing = _float_list(getattr(first, "ImagerPixelSpacing", None))
        records.append(
            DicomSeriesRecord(
                study_uid=study_uid,
                series_uid=series_uid,
                modality=str(getattr(first, "Modality", "")).upper(),
                series_number=str(getattr(first, "SeriesNumber", "")),
                sop_class_uids=sorted(sop_classes),
                image_types=image_types,
                relative_paths=[
                    path.relative_to(data_root).as_posix() for path, _ in members
                ],
                file_count=len(members),
                frame_count=frame_count,
                rows=int(getattr(first, "Rows", 0)) or None,
                columns=int(getattr(first, "Columns", 0)) or None,
                pixel_spacing=pixel_spacing,
                orientation=_float_list(
                    getattr(first, "ImageOrientationPatient", None)
                ),
                positions=positions,
                primary_angle_deg=_optional_float(first, "PositionerPrimaryAngle"),
                secondary_angle_deg=_optional_float(first, "PositionerSecondaryAngle"),
                sid_mm=_optional_float(first, "DistanceSourceToDetector"),
                sod_mm=_optional_float(first, "DistanceSourceToPatient"),
            )
        )
    return CaseDicomManifest(
        case_id=case_id,
        directory=directory.relative_to(data_root).as_posix(),
        candidate_directories=[directory.relative_to(data_root).as_posix()],
        status="scanned",
        series=records,
        unreadable_files=unreadable,
    )


def scan_data_root(root: Path) -> DicomManifest:
    data_root = root.resolve()
    candidates: dict[str, list[Path]] = defaultdict(list)
    for directory in sorted(item for item in data_root.iterdir() if item.is_dir()):
        match = CASE_DIRECTORY_PATTERN.match(directory.name)
        if match:
            candidates[match.group(1)].append(directory)
    cases: list[CaseDicomManifest] = []
    for case_id, directories in sorted(candidates.items()):
        if len(directories) > 1:
            cases.append(
                CaseDicomManifest(
                    case_id=case_id,
                    status="directory_conflict",
                    candidate_directories=[
                        directory.relative_to(data_root).as_posix()
                        for directory in directories
                    ],
                )
            )
        else:
            cases.append(_scan_case(data_root, case_id, directories[0]))
    return DicomManifest(data_root=str(data_root), cases=cases)


def write_dicom_manifest(manifest: DicomManifest, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(
            manifest.model_dump(mode="json"),
            allow_unicode=True,
            sort_keys=False,
        ),
        encoding="utf-8",
    )


def load_dicom_manifest(path: Path) -> DicomManifest:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    return DicomManifest.model_validate(raw)
