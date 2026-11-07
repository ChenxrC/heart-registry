from __future__ import annotations

import tempfile
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import SimpleITK as sitk
from scipy import ndimage

from ctdr_pipeline.dicom.geometry import ProjectionGeometry, pixel_to_lps_ray
from ctdr_pipeline.registration.affine2d import LandmarkPair
from ctdr_pipeline.registration.volume import CtVolume


@dataclass(frozen=True)
class RoiResult:
    volume_mask: np.ndarray
    projection_mask: np.ndarray
    method: str
    warnings: list[str]


def _remove_small_components(mask: np.ndarray, minimum_voxels: int) -> np.ndarray:
    labels, count = ndimage.label(mask)
    if count == 0:
        return np.zeros_like(mask, dtype=bool)
    sizes = np.bincount(labels.reshape(-1))
    keep = sizes >= minimum_voxels
    keep[0] = False
    return keep[labels]


def _landmark_corridor(
    volume: CtVolume,
    landmarks: list[LandmarkPair],
    geometry: ProjectionGeometry | None,
    radius_mm: float,
) -> np.ndarray:
    if geometry is None or not geometry.complete or not landmarks:
        return np.ones(volume.data_hu.shape, dtype=bool)
    zyx = np.column_stack(np.nonzero(volume.data_hu >= 150.0))
    if len(zyx) == 0:
        return np.zeros(volume.data_hu.shape, dtype=bool)
    xyz = zyx[:, [2, 1, 0]].astype(np.float64)
    homogeneous = np.column_stack([xyz, np.ones(len(xyz), dtype=np.float64)])
    points_lps = (homogeneous @ volume.voxel_to_lps.T)[:, :3]
    keep = np.zeros(len(points_lps), dtype=bool)
    for landmark in landmarks:
        ray = pixel_to_lps_ray(np.asarray(landmark.target_xy), geometry)
        relative = points_lps - ray.origin
        distance_along = relative @ ray.direction
        closest = ray.origin + distance_along[:, None] * ray.direction
        distance = np.linalg.norm(points_lps - closest, axis=1)
        keep |= (distance_along >= 0.0) & (distance <= radius_mm)
    corridor = np.zeros(volume.data_hu.shape, dtype=bool)
    selected = zyx[keep]
    corridor[selected[:, 0], selected[:, 1], selected[:, 2]] = True
    return corridor


def _baseline_roi(
    volume: CtVolume,
    landmarks: list[LandmarkPair],
    geometry: ProjectionGeometry | None,
    corridor_radius_mm: float,
    minimum_component_voxels: int,
) -> RoiResult:
    bone = volume.data_hu >= 150.0
    corridor = _landmark_corridor(
        volume, landmarks, geometry, corridor_radius_mm
    )
    mask = _remove_small_components(
        bone & corridor, minimum_component_voxels
    )
    return RoiResult(
        volume_mask=mask,
        projection_mask=np.max(mask, axis=0),
        method="baseline",
        warnings=[],
    )


def _run_totalsegmentator(
    volume: CtVolume,
    cache_dir: Path | None = None,
) -> np.ndarray:
    cache_path = (
        cache_dir / f"{volume.series_uid.replace('.', '_')}-roi.npy"
        if cache_dir is not None
        else None
    )
    if cache_path is not None and cache_path.exists():
        return np.load(cache_path).astype(bool)
    from totalsegmentator.python_api import totalsegmentator

    vertebrae = [
        *(f"vertebrae_C{index}" for index in range(1, 8)),
        *(f"vertebrae_T{index}" for index in range(1, 13)),
        *(f"vertebrae_L{index}" for index in range(1, 6)),
        "sacrum",
        "heart",
    ]
    with tempfile.TemporaryDirectory(prefix="ctdr-totalseg-") as temporary:
        root = Path(temporary)
        input_path = root / "ct.nii.gz"
        output_dir = root / "segmentations"
        sitk.WriteImage(volume.sitk_image, str(input_path))
        totalsegmentator(
            input=input_path,
            output=output_dir,
            roi_subset=vertebrae,
            fast=True,
            quiet=True,
        )
        mask = np.zeros(volume.data_hu.shape, dtype=bool)
        for path in sorted(output_dir.glob("*.nii.gz")):
            values = sitk.GetArrayFromImage(sitk.ReadImage(str(path)))
            if values.shape == mask.shape:
                mask |= values > 0
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache_path, mask.astype(np.uint8))
    return mask


def build_roi(
    volume: CtVolume,
    landmarks: list[LandmarkPair],
    geometry: ProjectionGeometry | None,
    *,
    method: str = "baseline",
    corridor_radius_mm: float = 40.0,
    minimum_component_voxels: int = 500,
    cache_dir: Path | None = None,
) -> RoiResult:
    baseline = _baseline_roi(
        volume,
        landmarks,
        geometry,
        corridor_radius_mm,
        minimum_component_voxels,
    )
    if method == "baseline":
        return baseline
    if method != "totalsegmentator":
        raise ValueError(f"unknown ROI method: {method}")
    try:
        segmented = _run_totalsegmentator(volume, cache_dir)
        combined = segmented & (
            baseline.volume_mask if baseline.volume_mask.any() else segmented
        )
        combined = _remove_small_components(combined, minimum_component_voxels)
        return RoiResult(
            volume_mask=combined,
            projection_mask=np.max(combined, axis=0),
            method="totalsegmentator",
            warnings=[],
        )
    except Exception as exc:
        return replace(
            baseline,
            method="baseline_fallback",
            warnings=[f"{type(exc).__name__}: {exc}"],
        )
