from pathlib import Path

import cv2
import numpy as np

from ctdr_pipeline.dicom.linker import (
    evaluate_orientations,
    select_image_match,
)


def textured_image(seed: int = 7) -> np.ndarray:
    rng = np.random.default_rng(seed)
    image = rng.integers(20, 220, size=(256, 256), dtype=np.uint8)
    image = cv2.GaussianBlur(image, (5, 5), 0)
    cv2.circle(image, (70, 80), 30, 245, 4)
    cv2.rectangle(image, (140, 40), (220, 120), 15, 5)
    cv2.line(image, (30, 220), (230, 150), 250, 4)
    cv2.putText(image, "CT", (90, 205), cv2.FONT_HERSHEY_SIMPLEX, 1.2, 240, 3)
    return image


def as_rgb(image: np.ndarray) -> np.ndarray:
    return np.repeat(image[:, :, None], 3, axis=2)


def test_correct_candidate_ranks_above_distractor() -> None:
    source = textured_image()
    distractor = textured_image(seed=19)
    candidates = []
    candidates.extend(
        evaluate_orientations(
            source,
            as_rgb(source),
            path=Path("correct.dcm"),
            series_uid="correct",
            frame_index=0,
        )
    )
    candidates.extend(
        evaluate_orientations(
            distractor,
            as_rgb(source),
            path=Path("distractor.dcm"),
            series_uid="distractor",
            frame_index=0,
        )
    )

    decision = select_image_match(candidates)

    assert decision.selected is not None
    assert decision.selected.series_uid == "correct"
    assert decision.selected.orientation == "identity"
    assert decision.status == "accepted"


def test_vertical_flip_required_is_rejected() -> None:
    source = textured_image()
    target = np.flipud(source)
    candidates = evaluate_orientations(
        source,
        as_rgb(target),
        path=Path("vertical.dcm"),
        series_uid="vertical",
        frame_index=0,
    )

    decision = select_image_match(candidates)

    assert decision.selected is not None
    assert decision.selected.vertical_flip
    assert decision.status == "rejected"
    assert "mirror_or_vertical_flip_required" in decision.reasons


def test_close_top_two_candidates_require_review() -> None:
    source = textured_image()
    candidates = evaluate_orientations(
        source,
        as_rgb(source),
        path=Path("first.dcm"),
        series_uid="first",
        frame_index=0,
    )
    candidates.extend(
        evaluate_orientations(
            source.copy(),
            as_rgb(source),
            path=Path("second.dcm"),
            series_uid="second",
            frame_index=0,
        )
    )

    decision = select_image_match(candidates)

    assert decision.status == "review_required"
    assert "candidate_margin_below_threshold" in decision.reasons
