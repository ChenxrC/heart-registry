from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class AnnotationFeature:
    component_id: int
    kind: str
    center_x: float
    center_y: float
    bbox_xywh: tuple[int, int, int, int]
    area_px: int
    major_axis: tuple[float, float]
    aspect_ratio: float

    @property
    def point(self) -> tuple[float, float]:
        return self.center_x, self.center_y


@dataclass(frozen=True)
class AnnotationSet:
    mask: np.ndarray
    landmarks: list[AnnotationFeature]
    direction_line: AnnotationFeature | None
    direction_axis: tuple[float, float] | None

    @property
    def points(self) -> np.ndarray:
        return np.asarray([feature.point for feature in self.landmarks], dtype=np.float64)


def colored_annotation_mask(rgb: np.ndarray) -> np.ndarray:
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError(f"expected RGB image with shape HxWx3, got {rgb.shape}")
    image = np.asarray(rgb, dtype=np.uint8)
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    rgb_i = image.astype(np.int16)
    chroma = rgb_i.max(axis=2) - rgb_i.min(axis=2)
    mask = (hsv[:, :, 1] >= 65) & (hsv[:, :, 2] >= 75) & (chroma >= 40)
    return mask.astype(np.uint8)


def _oriented_axis(points_xy: np.ndarray) -> tuple[float, float]:
    if len(points_xy) < 2:
        return 0.0, 1.0
    centered = points_xy - points_xy.mean(axis=0, keepdims=True)
    _, _, vectors = np.linalg.svd(centered, full_matrices=False)
    axis = vectors[0].astype(np.float64)
    if axis[1] < 0 or (abs(axis[1]) < 1e-8 and axis[0] < 0):
        axis *= -1.0
    norm = float(np.linalg.norm(axis))
    if norm == 0:
        return 0.0, 1.0
    return float(axis[0] / norm), float(axis[1] / norm)


def _component_feature(
    label: int,
    labels: np.ndarray,
    stats: np.ndarray,
    centroids: np.ndarray,
) -> AnnotationFeature:
    x, y, width, height, area = (int(value) for value in stats[label])
    rows, columns = np.where(labels == label)
    points_xy = np.column_stack([columns, rows]).astype(np.float64)
    axis = _oriented_axis(points_xy)
    aspect_ratio = max(width, height) / max(1.0, min(width, height))
    kind = "direction_line" if aspect_ratio >= 4.0 and area >= 2 * min(width, height) else "landmark"
    return AnnotationFeature(
        component_id=label,
        kind=kind,
        center_x=float(centroids[label, 0]),
        center_y=float(centroids[label, 1]),
        bbox_xywh=(x, y, width, height),
        area_px=area,
        major_axis=axis,
        aspect_ratio=float(aspect_ratio),
    )


def extract_annotations(rgb: np.ndarray, min_area: int = 20) -> AnnotationSet:
    mask = colored_annotation_mask(rgb)
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, 8)
    features = [
        _component_feature(label, labels, stats, centroids)
        for label in range(1, count)
        if int(stats[label, cv2.CC_STAT_AREA]) >= min_area
    ]
    line_candidates = [feature for feature in features if feature.kind == "direction_line"]
    direction_line = max(line_candidates, key=lambda item: item.area_px, default=None)
    landmarks = [feature for feature in features if feature is not direction_line]
    if direction_line is not None:
        direction_axis = direction_line.major_axis
        if direction_axis[1] < 0:
            direction_axis = (-direction_axis[0], -direction_axis[1])
    elif len(landmarks) >= 2:
        direction_axis = _oriented_axis(
            np.asarray([feature.point for feature in landmarks], dtype=np.float64)
        )
    else:
        direction_axis = None

    if direction_axis is not None:
        axis = np.asarray(direction_axis)
        landmarks.sort(key=lambda feature: float(np.dot(feature.point, axis)))
    else:
        landmarks.sort(key=lambda feature: (feature.center_y, feature.center_x))
    return AnnotationSet(
        mask=mask,
        landmarks=landmarks,
        direction_line=direction_line,
        direction_axis=direction_axis,
    )
