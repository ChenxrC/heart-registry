from ctdr_pipeline.pptx.grouping import group_case_slides
from ctdr_pipeline.pptx.reader import SlideRecord


def slide(number: int, case_id: str | None, images: int) -> SlideRecord:
    return SlideRecord(
        number=number,
        texts=([case_id] if case_id else []),
        image_refs=[f"rId{x + 1}" for x in range(images)],
    )


def test_grouping_uses_case_state_not_parity() -> None:
    groups = group_case_slides(
        [
            slide(1, "90000001", 1),
            slide(2, None, 1),
            slide(3, "90000002", 0),
            slide(4, "90000007", 1),
            slide(5, None, 1),
        ]
    )

    assert [(group.case_id, group.slide_numbers, group.image_count) for group in groups] == [
        ("90000001", [1, 2], 2),
        ("90000002", [3], 0),
        ("90000007", [4, 5], 2),
    ]


def test_repeated_case_id_stays_in_same_group() -> None:
    groups = group_case_slides(
        [slide(164, "90000008", 1), slide(165, "90000008", 1)]
    )

    assert len(groups) == 1
    assert groups[0].image_count == 2
    assert groups[0].pair_status == "ready"


def test_incomplete_and_excess_image_groups_are_explicit() -> None:
    groups = group_case_slides(
        [
            slide(1, "90000004", 0),
            slide(2, None, 1),
            slide(3, "90000009", 2),
            slide(4, None, 1),
        ]
    )

    assert groups[0].pair_status == "skipped_incomplete_pptx_pair"
    assert groups[1].pair_status == "review_required_too_many_images"
