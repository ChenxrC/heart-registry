from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pydicom
import SimpleITK as sitk

from ctdr_pipeline.dicom.scanner import DicomSeriesRecord


class CtGeometryError(ValueError):
    """Raised when a CT series cannot form a coherent physical volume."""


@dataclass(frozen=True)
class CtVolume:
    series_uid: str
    data_hu: np.ndarray
    voxel_to_lps: np.ndarray
    spacing_xyz: tuple[float, float, float]
    origin_lps: tuple[float, float, float]
    direction: np.ndarray
    sitk_image: sitk.Image
    source_paths: list[str]


def _array(dataset: pydicom.Dataset, name: str, length: int) -> np.ndarray:
    value = getattr(dataset, name, None)
    if value is None or len(value) != length:
        raise CtGeometryError(f"missing_or_invalid_{name}")
    return np.asarray([float(item) for item in value], dtype=np.float64)


def load_ct_volume(series: DicomSeriesRecord, data_root: Path) -> CtVolume:
    if series.modality != "CT":
        raise CtGeometryError("series_is_not_ct")
    slices: list[tuple[str, pydicom.Dataset]] = []
    for relative in series.relative_paths:
        path = data_root / Path(relative)
        dataset = pydicom.dcmread(path, force=True)
        slices.append((relative, dataset))
    if len(slices) < 2:
        raise CtGeometryError("fewer_than_two_slices")

    reference_orientation = _array(slices[0][1], "ImageOrientationPatient", 6)
    row_direction = reference_orientation[:3]
    column_direction = reference_orientation[3:]
    row_direction /= np.linalg.norm(row_direction)
    column_direction /= np.linalg.norm(column_direction)
    normal = np.cross(row_direction, column_direction)
    normal /= np.linalg.norm(normal)
    positioned: list[tuple[float, str, pydicom.Dataset, np.ndarray]] = []
    for relative, dataset in slices:
        orientation = _array(dataset, "ImageOrientationPatient", 6)
        if float(np.max(np.abs(orientation - reference_orientation))) > 1e-4:
            raise CtGeometryError("inconsistent_orientation")
        position = _array(dataset, "ImagePositionPatient", 3)
        positioned.append((float(np.dot(position, normal)), relative, dataset, position))
    positioned.sort(key=lambda item: item[0])
    coordinates = np.asarray([item[0] for item in positioned], dtype=np.float64)
    differences = np.diff(coordinates)
    if np.any(np.abs(differences) < 1e-5):
        raise CtGeometryError("duplicate_slice_positions")
    slice_spacing = float(np.median(differences))
    if slice_spacing <= 0:
        raise CtGeometryError("nonpositive_slice_spacing")
    if float(np.std(differences) / max(abs(np.mean(differences)), 1e-12)) > 0.05:
        raise CtGeometryError("irregular_slice_spacing")

    first = positioned[0][2]
    pixel_spacing = _array(first, "PixelSpacing", 2)
    row_spacing = float(pixel_spacing[0])
    column_spacing = float(pixel_spacing[1])
    expected_shape = (int(first.Rows), int(first.Columns))
    arrays: list[np.ndarray] = []
    source_paths: list[str] = []
    for _, relative, dataset, _ in positioned:
        pixels = np.asarray(dataset.pixel_array)
        if pixels.shape != expected_shape:
            raise CtGeometryError("inconsistent_slice_shape")
        slope = float(getattr(dataset, "RescaleSlope", 1.0))
        intercept = float(getattr(dataset, "RescaleIntercept", 0.0))
        arrays.append(pixels.astype(np.float32) * slope + intercept)
        source_paths.append(relative)
    data_hu = np.stack(arrays).astype(np.float32, copy=False)
    origin = positioned[0][3]
    direction = np.column_stack([row_direction, column_direction, normal])
    voxel_to_lps = np.eye(4, dtype=np.float64)
    voxel_to_lps[:3, 0] = row_direction * column_spacing
    voxel_to_lps[:3, 1] = column_direction * row_spacing
    voxel_to_lps[:3, 2] = normal * slice_spacing
    voxel_to_lps[:3, 3] = origin

    image = sitk.GetImageFromArray(data_hu)
    image.SetSpacing((column_spacing, row_spacing, slice_spacing))
    image.SetOrigin(tuple(float(value) for value in origin))
    image.SetDirection(tuple(float(value) for value in direction.reshape(-1)))
    return CtVolume(
        series_uid=series.series_uid,
        data_hu=data_hu,
        voxel_to_lps=voxel_to_lps,
        spacing_xyz=(column_spacing, row_spacing, slice_spacing),
        origin_lps=tuple(float(value) for value in origin),
        direction=direction,
        sitk_image=image,
        source_paths=source_paths,
    )
