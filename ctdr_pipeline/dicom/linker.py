from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np
import pydicom
from PIL import Image
from pydantic import Field

from ctdr_pipeline.bundle import BundleCaseRecord
from ctdr_pipeline.dicom.scanner import CaseDicomManifest
from ctdr_pipeline.dicom.selector import SeriesSelection
from ctdr_pipeline.pptx.annotations import colored_annotation_mask
from ctdr_pipeline.schemas import StrictModel


class ImageMatchCandidate(StrictModel):
    path: str
    series_uid: str
    frame_index: int
    orientation: str
    good_matches: int
    inliers: int
    inlier_ratio: float
    reprojection_error_px: float | None = None
    masked_ncc: float | None = None
    gradient_ncc: float | None = None
    fov_iou: float
    confidence: float
    homography_candidate_to_slide: list[float] | None = None
    diagnostic_only: bool = False
    mirrored: bool = False
    vertical_flip: bool = False


class ImageMatchDecision(StrictModel):
    selected: ImageMatchCandidate | None = None
    candidates: list[ImageMatchCandidate] = Field(default_factory=list)
    status: str
    confidence: float = 0.0
    candidate_margin: float = 0.0
    reasons: list[str] = Field(default_factory=list)


class DicomLinkageResult(StrictModel):
    case_id: str
    source: ImageMatchDecision
    target: ImageMatchDecision
    status: str
    reasons: list[str] = Field(default_factory=list)


def normalize_u8(array: np.ndarray, photometric: str = "MONOCHROME2") -> np.ndarray:
    image = np.asarray(array, dtype=np.float32)
    if image.ndim != 2:
        raise ValueError(f"expected a 2D image, got {image.shape}")
    if photometric == "MONOCHROME1":
        image = image.max() - image
    low, high = np.percentile(image, [0.5, 99.5])
    image = np.clip((image - low) / max(float(high - low), 1e-6), 0.0, 1.0)
    return np.round(image * 255.0).astype(np.uint8)


def _ncc(first: np.ndarray, second: np.ndarray, mask: np.ndarray) -> float | None:
    valid = np.asarray(mask, dtype=bool)
    if int(valid.sum()) < 100:
        return None
    left = first[valid].astype(np.float64)
    right = second[valid].astype(np.float64)
    left -= left.mean()
    right -= right.mean()
    denominator = math.sqrt(float(np.dot(left, left) * np.dot(right, right)))
    return float(np.dot(left, right) / denominator) if denominator > 0 else None


def _gradient(image: np.ndarray) -> np.ndarray:
    values = np.asarray(image, dtype=np.float32)
    gx = cv2.Sobel(values, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(values, cv2.CV_32F, 0, 1, ksize=3)
    return cv2.magnitude(gx, gy)


def _orientation_variants(candidate: np.ndarray) -> list[tuple[str, np.ndarray, bool, bool]]:
    return [
        ("identity", candidate, False, False),
        ("rotate180", np.rot90(candidate, 2).copy(), False, False),
        ("horizontal_flip", np.fliplr(candidate).copy(), True, False),
        ("vertical_flip", np.flipud(candidate).copy(), True, True),
    ]


def _match_one(
    candidate: np.ndarray,
    slide_rgb: np.ndarray,
    *,
    path: Path,
    series_uid: str,
    frame_index: int,
    orientation: str,
    diagnostic_only: bool,
    vertical_flip: bool,
) -> ImageMatchCandidate:
    candidate_u8 = normalize_u8(candidate)
    slide_gray = cv2.cvtColor(np.asarray(slide_rgb, dtype=np.uint8), cv2.COLOR_RGB2GRAY)
    annotation = colored_annotation_mask(np.asarray(slide_rgb, dtype=np.uint8))
    usable = (1 - cv2.dilate(annotation, np.ones((9, 9), np.uint8))) * 255
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    prepared_slide = clahe.apply(slide_gray)
    prepared_candidate = clahe.apply(candidate_u8)
    sift = cv2.SIFT_create(nfeatures=3000, contrastThreshold=0.015)
    slide_keypoints, slide_descriptors = sift.detectAndCompute(prepared_slide, usable)
    candidate_keypoints, candidate_descriptors = sift.detectAndCompute(
        prepared_candidate, None
    )
    good: list[cv2.DMatch] = []
    homography = None
    inlier_mask = None
    reprojection_error = None
    if candidate_descriptors is not None and slide_descriptors is not None:
        matches = cv2.BFMatcher().knnMatch(
            candidate_descriptors, slide_descriptors, k=2
        )
        good = [
            first
            for pair in matches
            if len(pair) == 2
            for first, second in [pair]
            if first.distance < 0.75 * second.distance
        ]
        if len(good) >= 4:
            source = np.float32(
                [candidate_keypoints[match.queryIdx].pt for match in good]
            )
            target = np.float32(
                [slide_keypoints[match.trainIdx].pt for match in good]
            )
            homography, inlier_mask = cv2.findHomography(
                source, target, cv2.RANSAC, 5.0
            )
            if homography is not None and inlier_mask is not None:
                mask_values = inlier_mask.reshape(-1).astype(bool)
                projected = cv2.perspectiveTransform(
                    source.reshape(-1, 1, 2), homography
                ).reshape(-1, 2)
                if mask_values.any():
                    reprojection_error = float(
                        np.mean(
                            np.linalg.norm(
                                projected[mask_values] - target[mask_values], axis=1
                            )
                        )
                    )
    inliers = int(inlier_mask.sum()) if inlier_mask is not None else 0
    inlier_ratio = inliers / max(len(good), 1)
    masked_ncc = None
    gradient_ncc = None
    fov_iou = 0.0
    mirrored_homography = False
    if homography is not None:
        height, width = slide_gray.shape
        warped = cv2.warpPerspective(candidate_u8, homography, (width, height))
        valid = cv2.warpPerspective(
            np.ones(candidate_u8.shape, np.uint8) * 255,
            homography,
            (width, height),
        )
        valid_mask = (valid > 0) & (annotation == 0)
        masked_ncc = _ncc(warped, slide_gray, valid_mask)
        gradient_ncc = _ncc(_gradient(warped), _gradient(slide_gray), valid_mask)
        fov_iou = float(valid_mask.sum() / max(valid_mask.size, 1))
        mirrored_homography = float(np.linalg.det(homography[:2, :2])) < 0
    feature_score = min(inliers / 20.0, 1.0) * inlier_ratio
    ncc_score = ((masked_ncc + 1.0) / 2.0) if masked_ncc is not None else 0.0
    gradient_score = (
        (gradient_ncc + 1.0) / 2.0 if gradient_ncc is not None else 0.0
    )
    confidence = (
        0.35 * feature_score
        + 0.25 * ncc_score
        + 0.25 * gradient_score
        + 0.15 * fov_iou
    )
    if mirrored_homography:
        confidence = max(0.0, confidence - 0.15)
    return ImageMatchCandidate(
        path=str(path),
        series_uid=series_uid,
        frame_index=frame_index,
        orientation=orientation,
        good_matches=len(good),
        inliers=inliers,
        inlier_ratio=inlier_ratio,
        reprojection_error_px=reprojection_error,
        masked_ncc=masked_ncc,
        gradient_ncc=gradient_ncc,
        fov_iou=fov_iou,
        confidence=float(np.clip(confidence, 0.0, 1.0)),
        homography_candidate_to_slide=(
            homography.reshape(-1).astype(float).tolist()
            if homography is not None
            else None
        ),
        diagnostic_only=diagnostic_only or mirrored_homography,
        mirrored=diagnostic_only or mirrored_homography,
        vertical_flip=vertical_flip,
    )


def evaluate_orientations(
    candidate: np.ndarray,
    slide_rgb: np.ndarray,
    *,
    path: Path,
    series_uid: str,
    frame_index: int,
    photometric: str = "MONOCHROME2",
) -> list[ImageMatchCandidate]:
    normalized = normalize_u8(candidate, photometric)
    return [
        _match_one(
            oriented,
            slide_rgb,
            path=path,
            series_uid=series_uid,
            frame_index=frame_index,
            orientation=name,
            diagnostic_only=diagnostic,
            vertical_flip=vertical,
        )
        for name, oriented, diagnostic, vertical in _orientation_variants(normalized)
    ]


def select_image_match(
    candidates: list[ImageMatchCandidate],
    *,
    minimum_confidence: float = 0.60,
    minimum_margin: float = 0.05,
) -> ImageMatchDecision:
    if not candidates:
        return ImageMatchDecision(
            status="rejected", reasons=["no_image_candidates"]
        )
    orientation_priority = {
        "identity": 3,
        "rotate180": 2,
        "horizontal_flip": 1,
        "vertical_flip": 0,
    }
    best_by_physical: dict[tuple[str, int], ImageMatchCandidate] = {}
    for candidate in candidates:
        key = (candidate.path, candidate.frame_index)
        previous = best_by_physical.get(key)
        if previous is None or (
            candidate.confidence,
            orientation_priority.get(candidate.orientation, -1),
        ) > (
            previous.confidence,
            orientation_priority.get(previous.orientation, -1),
        ):
            best_by_physical[key] = candidate
    ranked = sorted(
        best_by_physical.values(),
        key=lambda candidate: (
            candidate.confidence,
            orientation_priority.get(candidate.orientation, -1),
        ),
        reverse=True,
    )
    selected = ranked[0]
    margin = (
        selected.confidence - ranked[1].confidence if len(ranked) > 1 else 1.0
    )
    reasons: list[str] = []
    if selected.confidence < minimum_confidence:
        reasons.append("confidence_below_threshold")
    if len(ranked) > 1 and margin < minimum_margin:
        reasons.append("candidate_margin_below_threshold")
    if selected.diagnostic_only or selected.mirrored or selected.vertical_flip:
        reasons.append("mirror_or_vertical_flip_required")
    if "mirror_or_vertical_flip_required" in reasons:
        status = "rejected"
    elif reasons:
        status = "review_required"
    else:
        status = "accepted"
    return ImageMatchDecision(
        selected=selected,
        candidates=sorted(
            candidates, key=lambda candidate: candidate.confidence, reverse=True
        ),
        status=status,
        confidence=selected.confidence,
        candidate_margin=margin,
        reasons=reasons,
    )


def _series_candidates(
    data_root: Path,
    case: CaseDicomManifest,
    series_uids: list[str],
    slide_rgb: np.ndarray,
    requested_frame_index: int | None = None,
    instance_uid: str | None = None,
) -> list[ImageMatchCandidate]:
    first_pass: list[
        tuple[ImageMatchCandidate, np.ndarray, Path, str, int]
    ] = []
    for series_uid in series_uids:
        series = case.series_by_uid(series_uid)
        for relative in series.relative_paths:
            path = data_root / Path(relative)
            dataset = pydicom.dcmread(path, force=True)
            if instance_uid is not None and str(getattr(dataset, "SOPInstanceUID", "")) != instance_uid:
                continue
            pixels = np.asarray(dataset.pixel_array)
            frames = pixels[None] if pixels.ndim == 2 else pixels
            photometric = str(
                getattr(dataset, "PhotometricInterpretation", "MONOCHROME2")
            )
            for candidate_frame_index, frame in enumerate(frames):
                if (
                    requested_frame_index is not None
                    and candidate_frame_index != requested_frame_index
                ):
                    continue
                normalized = normalize_u8(frame, photometric)
                identity = _match_one(
                    normalized,
                    slide_rgb,
                    path=path,
                    series_uid=series_uid,
                    frame_index=candidate_frame_index,
                    orientation="identity",
                    diagnostic_only=False,
                    vertical_flip=False,
                )
                first_pass.append(
                    (identity, normalized, path, series_uid, candidate_frame_index)
                )
    candidates = [record[0] for record in first_pass]
    for _, normalized, path, series_uid, frame_index in sorted(
        first_pass, key=lambda record: record[0].confidence, reverse=True
    )[:12]:
        for name, oriented, diagnostic, vertical in _orientation_variants(normalized)[1:]:
            candidates.append(
                _match_one(
                    oriented,
                    slide_rgb,
                    path=path,
                    series_uid=series_uid,
                    frame_index=frame_index,
                    orientation=name,
                    diagnostic_only=diagnostic,
                    vertical_flip=vertical,
                )
            )
    return candidates


def link_case_images(
    bundle_root: Path,
    bundle_case: BundleCaseRecord,
    data_root: Path,
    case_manifest: CaseDicomManifest,
    selection: SeriesSelection,
) -> DicomLinkageResult:
    if bundle_case.source_image is None or bundle_case.target_image is None:
        rejected = ImageMatchDecision(
            status="rejected", reasons=["bundle_pair_images_missing"]
        )
        return DicomLinkageResult(
            case_id=bundle_case.case_id,
            source=rejected,
            target=rejected,
            status="rejected",
            reasons=["bundle_pair_images_missing"],
        )
    source_rgb = np.asarray(Image.open(bundle_root / bundle_case.source_image).convert("RGB"))
    target_rgb = np.asarray(Image.open(bundle_root / bundle_case.target_image).convert("RGB"))
    source_uids = selection.localizer_candidates or (
        [selection.localizer_series_uid] if selection.localizer_series_uid else []
    )
    target_uids = selection.xa_candidates or (
        [selection.xa_series_uid] if selection.xa_series_uid else []
    )
    source = select_image_match(
        _series_candidates(
            data_root,
            case_manifest,
            source_uids,
            source_rgb,
            instance_uid=(
                selection.localizer_instance_uid
                if "localizer_instance_uid" in selection.override_fields
                else None
            ),
        )
    )
    target = select_image_match(
        _series_candidates(
            data_root,
            case_manifest,
            target_uids,
            target_rgb,
            requested_frame_index=(
                selection.xa_frame_index
                if "xa_frame_index" in selection.override_fields
                else None
            ),
        )
    )
    reasons = [
        f"source:{reason}" for reason in source.reasons
    ] + [f"target:{reason}" for reason in target.reasons]
    if source.status == "rejected" or target.status == "rejected":
        status = "rejected"
    elif source.status != "accepted" or target.status != "accepted":
        status = "review_required"
    else:
        status = "accepted"
    return DicomLinkageResult(
        case_id=bundle_case.case_id,
        source=source,
        target=target,
        status=status,
        reasons=reasons,
    )
