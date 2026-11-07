from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pydicom


@dataclass(frozen=True)
class ProjectionGeometry:
    rows: int | None
    columns: int | None
    pixel_spacing_row_mm: float | None
    pixel_spacing_col_mm: float | None
    detector_origin_lps: np.ndarray | None
    row_direction_lps: np.ndarray | None
    column_direction_lps: np.ndarray | None
    normal_lps: np.ndarray | None
    source_lps: np.ndarray | None
    sid_mm: float | None
    sod_mm: float | None
    primary_angle_deg: float | None
    secondary_angle_deg: float | None
    provenance: dict[str, str]
    missing_fields: list[str]

    @property
    def complete(self) -> bool:
        return not self.missing_fields


@dataclass(frozen=True)
class Ray3D:
    origin: np.ndarray
    direction: np.ndarray


def _optional_float(dataset: pydicom.Dataset, name: str) -> float | None:
    value = getattr(dataset, name, None)
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _vector(dataset: pydicom.Dataset, name: str, length: int) -> np.ndarray | None:
    value = getattr(dataset, name, None)
    if value is None or len(value) != length:
        return None
    try:
        return np.asarray([float(item) for item in value], dtype=np.float64)
    except (TypeError, ValueError):
        return None


def projection_geometry(dataset: pydicom.Dataset) -> ProjectionGeometry:
    rows = int(getattr(dataset, "Rows", 0)) or None
    columns = int(getattr(dataset, "Columns", 0)) or None
    spacing = _vector(dataset, "PixelSpacing", 2)
    if spacing is None:
        spacing = _vector(dataset, "ImagerPixelSpacing", 2)
    orientation = _vector(dataset, "ImageOrientationPatient", 6)
    detector_origin = _vector(dataset, "ImagePositionPatient", 3)
    sid = _optional_float(dataset, "DistanceSourceToDetector")
    sod = _optional_float(dataset, "DistanceSourceToPatient")
    primary = _optional_float(dataset, "PositionerPrimaryAngle")
    secondary = _optional_float(dataset, "PositionerSecondaryAngle")
    row_direction = orientation[:3] if orientation is not None else None
    column_direction = orientation[3:] if orientation is not None else None
    normal = None
    source = None
    if row_direction is not None and column_direction is not None:
        row_direction = row_direction / np.linalg.norm(row_direction)
        column_direction = column_direction / np.linalg.norm(column_direction)
        normal = np.cross(row_direction, column_direction)
        normal = normal / np.linalg.norm(normal)
    if (
        detector_origin is not None
        and row_direction is not None
        and column_direction is not None
        and spacing is not None
        and rows is not None
        and columns is not None
        and sid is not None
    ):
        detector_center = (
            detector_origin
            + row_direction * ((columns - 1) * float(spacing[1]) / 2.0)
            + column_direction * ((rows - 1) * float(spacing[0]) / 2.0)
        )
        source = detector_center - normal * sid
    values = {
        "rows": rows,
        "columns": columns,
        "pixel_spacing": spacing,
        "orientation": orientation,
        "detector_origin": detector_origin,
        "sid_mm": sid,
        "sod_mm": sod,
        "primary_angle_deg": primary,
        "secondary_angle_deg": secondary,
    }
    missing = [name for name, value in values.items() if value is None]
    provenance = {
        name: "dicom" if value is not None else "unknown"
        for name, value in values.items()
    }
    return ProjectionGeometry(
        rows=rows,
        columns=columns,
        pixel_spacing_row_mm=float(spacing[0]) if spacing is not None else None,
        pixel_spacing_col_mm=float(spacing[1]) if spacing is not None else None,
        detector_origin_lps=detector_origin,
        row_direction_lps=row_direction,
        column_direction_lps=column_direction,
        normal_lps=normal,
        source_lps=source,
        sid_mm=sid,
        sod_mm=sod,
        primary_angle_deg=primary,
        secondary_angle_deg=secondary,
        provenance=provenance,
        missing_fields=missing,
    )


def pixel_to_lps_ray(pixel_xy: np.ndarray, geometry: ProjectionGeometry) -> Ray3D:
    if (
        geometry.source_lps is None
        or geometry.detector_origin_lps is None
        or geometry.row_direction_lps is None
        or geometry.column_direction_lps is None
        or geometry.pixel_spacing_row_mm is None
        or geometry.pixel_spacing_col_mm is None
    ):
        raise ValueError("projection geometry is incomplete; pixel ray is undefined")
    x, y = (float(value) for value in np.asarray(pixel_xy).reshape(2))
    detector_point = (
        geometry.detector_origin_lps
        + geometry.row_direction_lps * x * geometry.pixel_spacing_col_mm
        + geometry.column_direction_lps * y * geometry.pixel_spacing_row_mm
    )
    direction = detector_point - geometry.source_lps
    direction /= np.linalg.norm(direction)
    return Ray3D(origin=geometry.source_lps.copy(), direction=direction)
