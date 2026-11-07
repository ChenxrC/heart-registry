from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import Field, ValidationError

from ctdr_pipeline.dicom.scanner import CaseDicomManifest, DicomSeriesRecord
from ctdr_pipeline.schemas import StrictModel


class SeriesSelection(StrictModel):
    case_id: str
    ct_series_uid: str | None = None
    localizer_series_uid: str | None = None
    localizer_instance_uid: str | None = None
    xa_series_uid: str | None = None
    xa_frame_index: int = 0
    ct_candidates: list[str] = Field(default_factory=list)
    localizer_candidates: list[str] = Field(default_factory=list)
    xa_candidates: list[str] = Field(default_factory=list)
    available_series_uids: list[str] = Field(default_factory=list)
    series_frame_counts: dict[str, int] = Field(default_factory=dict)
    status: str
    reasons: list[str] = Field(default_factory=list)
    override_fields: list[str] = Field(default_factory=list)


class DicomCaseOverride(StrictModel):
    case_id: str
    ct_series_uid: str | None = None
    localizer_series_uid: str | None = None
    localizer_instance_uid: str | None = None
    xa_series_uid: str | None = None
    xa_frame_index: int | None = None


class DicomOverrides(StrictModel):
    cases: list[DicomCaseOverride]


def _contains_image_type(series: DicomSeriesRecord, *tokens: str) -> bool:
    expected = {token.upper() for token in tokens}
    return any(expected.intersection(values) for values in series.image_types)


def _orientation_consistent(series: DicomSeriesRecord) -> bool:
    return series.orientation is not None and len(series.orientation) == 6


def _volume_score(series: DicomSeriesRecord) -> tuple[int, int, int]:
    return (
        int(_orientation_consistent(series)),
        series.unique_slice_positions,
        series.file_count,
    )


def _projection_score(series: DicomSeriesRecord) -> tuple[int, int, int]:
    modality_priority = {"XA": 4, "RF": 3, "DX": 2, "CR": 1}.get(
        series.modality, 0
    )
    geometry_fields = sum(
        value is not None
        for value in (series.pixel_spacing, series.primary_angle_deg, series.sid_mm)
    )
    return modality_priority, geometry_fields, series.frame_count


def rank_series(case: CaseDicomManifest) -> SeriesSelection:
    if case.status == "directory_conflict":
        return SeriesSelection(
            case_id=case.case_id,
            status="directory_conflict",
            reasons=["multiple_directories_share_case_id"],
        )
    ct_candidates = [
        series
        for series in case.series
        if series.modality == "CT"
        and series.unique_slice_positions >= 16
        and _orientation_consistent(series)
        and not _contains_image_type(series, "LOCALIZER", "SCOUT")
    ]
    ct_candidates.sort(key=_volume_score, reverse=True)
    localizer_candidates = [
        series
        for series in case.series
        if series.modality == "CT"
        and _contains_image_type(series, "LOCALIZER", "SCOUT")
    ]
    if not localizer_candidates:
        localizer_candidates = [
            series
            for series in case.series
            if series.modality == "CT" and series.file_count <= 4 and series not in ct_candidates
        ]
    localizer_candidates.sort(key=lambda series: series.file_count)
    xa_candidates = [
        series for series in case.series if series.modality in {"XA", "RF", "DX", "CR"}
    ]
    xa_candidates.sort(key=_projection_score, reverse=True)
    reasons: list[str] = []
    if not ct_candidates:
        reasons.append("ct_series_missing")
    if not localizer_candidates:
        reasons.append("ct_localizer_missing")
    if not xa_candidates:
        reasons.append("xa_series_missing")
    return SeriesSelection(
        case_id=case.case_id,
        ct_series_uid=ct_candidates[0].series_uid if ct_candidates else None,
        localizer_series_uid=(
            localizer_candidates[0].series_uid if localizer_candidates else None
        ),
        xa_series_uid=xa_candidates[0].series_uid if xa_candidates else None,
        xa_frame_index=0,
        ct_candidates=[series.series_uid for series in ct_candidates],
        localizer_candidates=[series.series_uid for series in localizer_candidates],
        xa_candidates=[series.series_uid for series in xa_candidates],
        available_series_uids=[series.series_uid for series in case.series],
        series_frame_counts={
            series.series_uid: series.frame_count for series in case.series
        },
        status="ready" if not reasons else "incomplete",
        reasons=reasons,
    )


def apply_dicom_overrides(
    selection: SeriesSelection, path: Path | None
) -> SeriesSelection:
    if path is None:
        return selection
    try:
        document = DicomOverrides.model_validate(
            yaml.safe_load(path.read_text(encoding="utf-8"))
        )
    except ValidationError as exc:
        raise ValueError(str(exc)) from exc
    override = next(
        (record for record in document.cases if record.case_id == selection.case_id),
        None,
    )
    if override is None:
        return selection
    update = override.model_dump(exclude_none=True)
    update.pop("case_id", None)
    override_fields = sorted(update)
    for field in ("ct_series_uid", "localizer_series_uid", "xa_series_uid"):
        uid = update.get(field)
        if uid is not None and uid not in selection.available_series_uids:
            raise ValueError(f"{field} references unavailable series UID {uid}")
    xa_uid = str(update.get("xa_series_uid", selection.xa_series_uid or ""))
    frame_index = int(update.get("xa_frame_index", selection.xa_frame_index))
    if xa_uid and frame_index >= selection.series_frame_counts.get(xa_uid, 0):
        raise ValueError(f"xa_frame_index {frame_index} is outside series {xa_uid}")
    if "ct_series_uid" in update:
        update["ct_candidates"] = [update["ct_series_uid"]]
    if "localizer_series_uid" in update:
        update["localizer_candidates"] = [update["localizer_series_uid"]]
    if "xa_series_uid" in update:
        update["xa_candidates"] = [update["xa_series_uid"]]
    update["override_fields"] = override_fields
    updated = selection.model_copy(update=update)
    reasons = [
        reason
        for reason in updated.reasons
        if not (
            (reason == "ct_series_missing" and updated.ct_series_uid)
            or (reason == "ct_localizer_missing" and updated.localizer_series_uid)
            or (reason == "xa_series_missing" and updated.xa_series_uid)
        )
    ]
    return updated.model_copy(
        update={"reasons": reasons, "status": "ready" if not reasons else "incomplete"}
    )
