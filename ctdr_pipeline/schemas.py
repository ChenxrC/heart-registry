from __future__ import annotations

import math
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, field_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class CaseStatus(StrEnum):
    ACCEPTED_RESEARCH_CANDIDATE = "accepted_research_candidate"
    REVIEW_REQUIRED_PPTX = "review_required_pptx"
    REVIEW_REQUIRED_2D = "review_required_2d"
    REVIEW_REQUIRED_DICOM_AMBIGUITY = "review_required_dicom_ambiguity"
    REVIEW_REQUIRED_3D = "review_required_3d"
    SKIPPED_INCOMPLETE_PPTX_PAIR = "skipped_incomplete_pptx_pair"
    SKIPPED_CASE_DIRECTORY_MISSING = "skipped_case_directory_missing"
    SKIPPED_CT_MISSING = "skipped_ct_missing"
    SKIPPED_XA_MISSING = "skipped_xa_missing"
    REJECTED_GEOMETRY = "rejected_geometry"
    REJECTED_3D = "rejected_3d"
    FAILED_RUNTIME = "failed_runtime"


class CoordinateSystem(StrEnum):
    SLIDE_EMU = "slide_emu"
    SLIDE_PIXEL = "slide_pixel"
    EMBEDDED_PIXEL = "embedded_pixel"
    DICOM_PIXEL = "dicom_pixel"
    PATIENT_LPS_MM = "patient_lps_mm"


class Matrix3x3(StrictModel):
    values: list[float]
    source_space: CoordinateSystem
    target_space: CoordinateSystem

    @field_validator("values")
    @classmethod
    def validate_values(cls, values: list[float]) -> list[float]:
        if len(values) != 9:
            raise ValueError("Matrix3x3 requires exactly nine values")
        if not all(math.isfinite(value) for value in values):
            raise ValueError("Matrix3x3 values must be finite")
        return values

    def rows(self) -> list[list[float]]:
        return [self.values[0:3], self.values[3:6], self.values[6:9]]
