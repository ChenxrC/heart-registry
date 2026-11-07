from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ctdr_pipeline.dicom.geometry import ProjectionGeometry
from ctdr_pipeline.dicom.linker import DicomLinkageResult
from ctdr_pipeline.registration.affine2d import Registration2DResult


@dataclass(frozen=True)
class PoseSeed:
    name: str
    rotation_deg: tuple[float, float, float]
    translation_mm: tuple[float, float, float]
    diagnostic_only: bool = False
    provenance: str = "dicom_affine"

    @property
    def as_vector(self) -> tuple[float, float, float, float, float, float, bool]:
        return (*self.rotation_deg, *self.translation_mm, self.diagnostic_only)


def _homography(linkage: DicomLinkageResult, role: str) -> np.ndarray:
    decision = linkage.source if role == "source" else linkage.target
    selected = decision.selected
    if selected is None or selected.homography_candidate_to_slide is None:
        return np.eye(3, dtype=np.float64)
    return np.asarray(
        selected.homography_candidate_to_slide, dtype=np.float64
    ).reshape(3, 3)


def generate_pose_seeds(
    linkage: DicomLinkageResult,
    affine: Registration2DResult,
    geometry: ProjectionGeometry,
    *,
    seed: int,
    count: int = 12,
) -> list[PoseSeed]:
    if count < 1:
        raise ValueError("count must be at least one")
    source_to_slide = _homography(linkage, "source")
    target_to_slide = _homography(linkage, "target")
    source_dicom_to_target_dicom = (
        np.linalg.inv(target_to_slide) @ affine.matrix @ source_to_slide
    )
    linear = source_dicom_to_target_dicom[:2, :2]
    affine_angle = math.degrees(math.atan2(float(linear[1, 0]), float(linear[0, 0])))
    primary = geometry.primary_angle_deg or 0.0
    secondary = geometry.secondary_angle_deg or 0.0
    distance = geometry.sod_mm or 810.0
    column_spacing = geometry.pixel_spacing_col_mm or 1.0
    row_spacing = geometry.pixel_spacing_row_mm or 1.0
    translation_x = float(source_dicom_to_target_dicom[0, 2]) * column_spacing
    translation_z = float(source_dicom_to_target_dicom[1, 2]) * row_spacing
    base = PoseSeed(
        name="dicom_affine_base",
        rotation_deg=(primary + affine_angle, secondary, 0.0),
        translation_mm=(translation_x, distance, translation_z),
    )
    rng = np.random.default_rng(seed)
    seeds = [base]
    for index in range(1, count):
        rotation_noise = rng.normal(0.0, [8.0, 5.0, 3.0])
        translation_noise = rng.normal(0.0, [20.0, 40.0, 20.0])
        rotations = np.asarray(base.rotation_deg) + rotation_noise
        translations = np.asarray(base.translation_mm) + translation_noise
        translations[1] = np.clip(translations[1], 450.0, 950.0)
        seeds.append(
            PoseSeed(
                name=f"multistart_{index:02d}",
                rotation_deg=tuple(float(value) for value in rotations),
                translation_mm=tuple(float(value) for value in translations),
            )
        )
    seeds.extend(
        [
            PoseSeed(
                name="diagnostic_rotate180",
                rotation_deg=(base.rotation_deg[0] + 180.0, *base.rotation_deg[1:]),
                translation_mm=base.translation_mm,
                diagnostic_only=True,
                provenance="orientation_diagnostic",
            ),
            PoseSeed(
                name="diagnostic_vertical_flip",
                rotation_deg=(base.rotation_deg[0], -base.rotation_deg[1], 180.0),
                translation_mm=base.translation_mm,
                diagnostic_only=True,
                provenance="orientation_diagnostic",
            ),
        ]
    )
    return seeds
