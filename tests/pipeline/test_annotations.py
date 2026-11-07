import numpy as np

from ctdr_pipeline.pptx.annotations import extract_annotations


def test_gray_bone_is_excluded_but_three_annotation_colors_are_kept() -> None:
    image = np.full((100, 100, 3), 128, np.uint8)
    image[10:20, 10:20] = (255, 255, 0)
    image[40:50, 40:50] = (0, 255, 0)
    image[70:80, 70:80] = (0, 100, 255)

    features = extract_annotations(image, min_area=20)

    assert len(features.landmarks) == 3
    assert features.mask[30, 30] == 0
    assert features.direction_axis is not None


def test_long_annotation_component_is_used_as_direction_line() -> None:
    image = np.full((120, 100, 3), 120, np.uint8)
    image[10:105, 48:51] = (255, 255, 0)
    image[20:28, 20:28] = (0, 255, 0)
    image[70:78, 70:78] = (0, 100, 255)

    features = extract_annotations(image, min_area=10)

    assert features.direction_line is not None
    assert abs(features.direction_axis[1]) > 0.95
    assert len(features.landmarks) == 2
