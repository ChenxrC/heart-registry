from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from ctdr_pipeline.pptx.reader import SlideRecord


@dataclass(frozen=True)
class PptxCaseGroup:
    case_id: str
    slides: list[SlideRecord]
    pair_status: str
    issues: list[str]

    @property
    def slide_numbers(self) -> list[int]:
        return [slide.number for slide in self.slides]

    @property
    def image_count(self) -> int:
        return sum(slide.image_count for slide in self.slides)


def _finalize(case_id: str, slides: list[SlideRecord]) -> PptxCaseGroup:
    image_count = sum(slide.image_count for slide in slides)
    issues: list[str] = []
    if any(len(slide.case_ids) > 1 for slide in slides):
        status = "review_required_multiple_case_ids"
        issues.append("multiple_case_ids_on_slide")
    elif image_count < 2:
        status = "skipped_incomplete_pptx_pair"
        issues.append(f"expected_2_images_found_{image_count}")
    elif image_count > 2:
        status = "review_required_too_many_images"
        issues.append(f"expected_2_images_found_{image_count}")
    else:
        status = "ready"
    return PptxCaseGroup(
        case_id=case_id,
        slides=list(slides),
        pair_status=status,
        issues=issues,
    )


def group_case_slides(slides: Sequence[SlideRecord]) -> list[PptxCaseGroup]:
    groups: list[PptxCaseGroup] = []
    current_case_id: str | None = None
    current_slides: list[SlideRecord] = []

    for slide in slides:
        slide_case_id = slide.case_ids[0] if slide.case_ids else None
        if slide_case_id and slide_case_id != current_case_id:
            if current_case_id is not None:
                groups.append(_finalize(current_case_id, current_slides))
            current_case_id = slide_case_id
            current_slides = []
        if current_case_id is not None:
            current_slides.append(slide)

    if current_case_id is not None:
        groups.append(_finalize(current_case_id, current_slides))
    return groups
