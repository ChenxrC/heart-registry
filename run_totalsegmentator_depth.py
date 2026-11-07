"""Segment vertebrae and recover landmark depth from explicit case inputs.

This script deliberately separates the three coordinate spaces involved here:

1. PowerPoint pixels, where the physician annotations were extracted.
2. CT localizer pixels, recovered by the previously validated SIFT homography.
3. DICOM patient LPS millimetres, shared by the localizer and axial CT series.

The localizer supplies two coordinates on its reference plane.  The missing
coordinate is estimated from the labelled vertebral-body mask along the
localizer ray.  Because a single projection cannot uniquely determine depth,
the output includes a ray interval and a quality flag in addition to the
mid-depth estimate.
"""

from __future__ import annotations

import argparse
import csv
import importlib.metadata
import json
import os
import subprocess
import sys
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import pydicom
import SimpleITK as sitk


PROJECT_ROOT = Path(__file__).resolve().parent

VERTEBRAE = [
    *(f"vertebrae_C{level}" for level in range(1, 8)),
    *(f"vertebrae_T{level}" for level in range(1, 13)),
    *(f"vertebrae_L{level}" for level in range(1, 6)),
]
SEMANTIC_LANDMARK_LABELS = {
    "T1": "vertebrae_T1",
    "T2": "vertebrae_T2",
}


@dataclass(frozen=True)
class LocalizerGeometry:
    origin_lps_mm: np.ndarray
    row_direction_lps: np.ndarray
    column_direction_lps: np.ndarray
    normal_direction_lps: np.ndarray
    row_spacing_mm: float
    column_spacing_mm: float
    rows: int
    columns: int


@dataclass(frozen=True)
class MaskGeometry:
    origin_lps_mm: np.ndarray
    spacing_mm: np.ndarray
    direction_lps: np.ndarray
    size_xyz: tuple[int, int, int]


def dicom_series(directory: Path) -> dict[str, list[Path]]:
    grouped: dict[str, list[Path]] = defaultdict(list)
    for path in directory.glob("*.DCM"):
        dataset = pydicom.dcmread(str(path), stop_before_pixels=True, force=True)
        uid = str(getattr(dataset, "SeriesInstanceUID", ""))
        if uid:
            grouped[uid].append(path)
    return dict(grouped)


def select_ct_series(directory: Path, series_number: str) -> tuple[list[Path], object]:
    candidates: list[tuple[list[Path], object]] = []
    available: list[dict[str, object]] = []
    for paths in dicom_series(directory).values():
        header = pydicom.dcmread(str(paths[0]), stop_before_pixels=True, force=True)
        row = {
            "modality": str(getattr(header, "Modality", "")),
            "series_number": str(getattr(header, "SeriesNumber", "")),
            "description": str(getattr(header, "SeriesDescription", "")),
            "slice_count": len(paths),
        }
        available.append(row)
        if row["modality"] == "CT" and row["series_number"] == str(series_number):
            candidates.append((paths, header))
    if len(candidates) != 1:
        raise RuntimeError(
            f"Expected one CT series {series_number} in {directory}; "
            f"found {len(candidates)}. Available={available}"
        )
    return candidates[0]


def read_ct_series(directory: Path, series_number: str) -> tuple[sitk.Image, object, list[str]]:
    paths, header = select_ct_series(directory, series_number)
    # GDCM returns this hospital series in reverse InstanceNumber order while
    # SimpleITK still keeps a -z third direction.  That combination produces a
    # physically impossible volume range.  Sort the explicitly selected series
    # by its unique acquisition InstanceNumber and validate both endpoints.
    slice_records: list[tuple[int, Path, np.ndarray]] = []
    for path in paths:
        slice_header = pydicom.dcmread(str(path), stop_before_pixels=True, force=True)
        slice_records.append(
            (
                int(getattr(slice_header, "InstanceNumber", 0)),
                path,
                np.asarray(slice_header.ImagePositionPatient, dtype=np.float64),
            )
        )
    instance_numbers = [record[0] for record in slice_records]
    if len(set(instance_numbers)) != len(slice_records):
        raise RuntimeError("CT InstanceNumber values are not unique; refusing ambiguous slice order")
    slice_records.sort(key=lambda record: record[0])
    ordered = [str(record[1]) for record in slice_records]
    reader = sitk.ImageSeriesReader()
    reader.SetFileNames(ordered)
    reader.MetaDataDictionaryArrayUpdateOn()
    image = reader.Execute()
    if image.GetDimension() != 3 or image.GetSize()[2] < 16:
        raise RuntimeError(f"Series {series_number} is not a usable 3D CT volume: {image.GetSize()}")
    actual_first = np.asarray(image.TransformIndexToPhysicalPoint((0, 0, 0)))
    actual_last = np.asarray(
        image.TransformIndexToPhysicalPoint((0, 0, image.GetSize()[2] - 1))
    )
    expected_first = slice_records[0][2]
    expected_last = slice_records[-1][2]
    if not (
        np.allclose(actual_first, expected_first, atol=0.05)
        and np.allclose(actual_last, expected_last, atol=0.05)
    ):
        raise RuntimeError(
            "DICOM-to-volume geometry validation failed: "
            f"actual=({actual_first.tolist()}, {actual_last.tolist()}) "
            f"expected=({expected_first.tolist()}, {expected_last.tolist()})"
        )
    return image, header, ordered


def mask_geometry(image: sitk.Image) -> MaskGeometry:
    return MaskGeometry(
        origin_lps_mm=np.asarray(image.GetOrigin(), dtype=np.float64),
        spacing_mm=np.asarray(image.GetSpacing(), dtype=np.float64),
        direction_lps=np.asarray(image.GetDirection(), dtype=np.float64).reshape(3, 3),
        size_xyz=tuple(int(value) for value in image.GetSize()),
    )


def geometry_payload(image: sitk.Image) -> dict[str, object]:
    array = sitk.GetArrayViewFromImage(image)
    # Computing five percentiles separately on the full 512x512x330 volume can
    # create a large temporary-array peak.  A regular 4x subsample still gives
    # >1.3 million deterministic HU samples and is more than sufficient for an
    # input-integrity check.
    sample = np.asarray(array[::4, ::4, ::4]).reshape(-1)
    percentile_levels = np.asarray([0.1, 1.0, 50.0, 99.0, 99.9])
    percentile_values = np.percentile(sample, percentile_levels)
    return {
        "size_xyz": list(map(int, image.GetSize())),
        "spacing_xyz_mm": list(map(float, image.GetSpacing())),
        "origin_lps_mm": list(map(float, image.GetOrigin())),
        "direction_lps": np.asarray(image.GetDirection()).reshape(3, 3).tolist(),
        "pixel_type": image.GetPixelIDTypeAsString(),
        "hu_min": float(np.min(array)),
        "hu_max": float(np.max(array)),
        "hu_percentiles": {
            str(float(percentile)): float(value)
            for percentile, value in zip(percentile_levels, percentile_values)
        },
        "hu_percentile_sample_stride_zyx": [4, 4, 4],
    }


def write_ct_nifti(
    ct_dir: Path,
    series_number: str,
    output_path: Path,
    overwrite: bool,
) -> tuple[sitk.Image, dict[str, object]]:
    image, header, ordered = read_ct_series(ct_dir, series_number)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if overwrite or not output_path.exists():
        sitk.WriteImage(image, str(output_path), useCompression=True)
    metadata = {
        "source_directory": str(ct_dir.resolve()),
        "series_number": str(series_number),
        "series_instance_uid": str(getattr(header, "SeriesInstanceUID", "")),
        "series_description": str(getattr(header, "SeriesDescription", "")),
        "frame_of_reference_uid": str(getattr(header, "FrameOfReferenceUID", "")),
        "dicom_slice_count": len(ordered),
        "nifti_path": str(output_path.resolve()),
        **geometry_payload(image),
    }
    return image, metadata


def configure_private_totalsegmentator_home(home: Path) -> Path:
    home.mkdir(parents=True, exist_ok=True)
    config_path = home / "config.json"
    if config_path.exists():
        config = json.loads(config_path.read_text(encoding="utf-8"))
    else:
        config = {
            "totalseg_id": "local-research-no-telemetry",
            "prediction_counter": 0,
        }
    config["send_usage_stats"] = False
    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
    return config_path


def totalsegmentator_executable() -> Path:
    executable = Path(sys.executable).with_name("TotalSegmentator.exe")
    if not executable.exists():
        raise RuntimeError(
            "TotalSegmentator executable not found beside the active Python. "
            "Run this script with .venv-totalseg\\Scripts\\python.exe."
        )
    return executable


def run_totalsegmentator(
    input_nifti: Path,
    masks_dir: Path,
    cache_dir: Path,
    task: str,
    device: str,
    overwrite: bool,
) -> None:
    expected = masks_dir / "vertebrae_T1.nii.gz"
    if expected.exists() and not overwrite:
        print(f"Segmentation already exists: {masks_dir}")
        return
    masks_dir.mkdir(parents=True, exist_ok=True)
    configure_private_totalsegmentator_home(cache_dir)
    env = os.environ.copy()
    env["TOTALSEG_HOME_DIR"] = str(cache_dir.resolve())
    command = [
        str(totalsegmentator_executable()),
        "-i",
        str(input_nifti.resolve()),
        "-o",
        str(masks_dir.resolve()),
        "--task",
        task,
        "--device",
        device,
        "--nr_thr_resamp",
        "2",
        "--nr_thr_saving",
        "1",
        "--higher_order_resampling",
        "--report",
        str((masks_dir.parent / "totalsegmentator_run_report.json").resolve()),
    ]
    print("Running:", subprocess.list2cmdline(command), flush=True)
    subprocess.run(command, check=True, env=env)


def localizer_geometry(dataset: object) -> LocalizerGeometry:
    orientation = np.asarray(dataset.ImageOrientationPatient, dtype=np.float64)
    row_direction = orientation[:3]
    column_direction = orientation[3:]
    row_direction /= np.linalg.norm(row_direction)
    column_direction /= np.linalg.norm(column_direction)
    normal = np.cross(row_direction, column_direction)
    normal /= np.linalg.norm(normal)
    spacing = np.asarray(dataset.PixelSpacing, dtype=np.float64)
    return LocalizerGeometry(
        origin_lps_mm=np.asarray(dataset.ImagePositionPatient, dtype=np.float64),
        row_direction_lps=row_direction,
        column_direction_lps=column_direction,
        normal_direction_lps=normal,
        row_spacing_mm=float(spacing[0]),
        column_spacing_mm=float(spacing[1]),
        rows=int(dataset.Rows),
        columns=int(dataset.Columns),
    )


def localizer_pixel_to_lps(
    uv: np.ndarray,
    geometry: LocalizerGeometry,
    depth_mm: float | np.ndarray = 0.0,
) -> np.ndarray:
    points = np.atleast_2d(np.asarray(uv, dtype=np.float64))
    depth = np.broadcast_to(np.asarray(depth_mm, dtype=np.float64), (len(points),))
    return (
        geometry.origin_lps_mm[None, :]
        + points[:, 0:1] * geometry.column_spacing_mm * geometry.row_direction_lps[None, :]
        + points[:, 1:2] * geometry.row_spacing_mm * geometry.column_direction_lps[None, :]
        + depth[:, None] * geometry.normal_direction_lps[None, :]
    )


def lps_to_localizer_pixel(points_lps: np.ndarray, geometry: LocalizerGeometry) -> tuple[np.ndarray, np.ndarray]:
    points = np.atleast_2d(np.asarray(points_lps, dtype=np.float64))
    delta = points - geometry.origin_lps_mm[None, :]
    u = delta @ geometry.row_direction_lps / geometry.column_spacing_mm
    v = delta @ geometry.column_direction_lps / geometry.row_spacing_mm
    depth = delta @ geometry.normal_direction_lps
    return np.column_stack([u, v]), depth


def transform_points(points_xy: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    points = np.atleast_2d(np.asarray(points_xy, dtype=np.float64))
    homogeneous = np.column_stack([points, np.ones(len(points), dtype=np.float64)])
    transformed = (np.asarray(matrix, dtype=np.float64) @ homogeneous.T).T
    return transformed[:, :2] / transformed[:, 2:3]


def source_landmarks_in_localizer(summary_path: Path) -> tuple[list[dict[str, object]], Path]:
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    physical = next(
        row for row in summary["dicom_linkage"] if row.get("role") == "ct_physical_best"
    )
    physical_path = Path(physical["path"])
    candidate = next(
        row
        for row in summary["dicom_search"]["ct_candidates"]
        if Path(row["path"]).resolve() == physical_path.resolve()
    )
    dicom_to_slide = np.asarray(candidate["homography_dicom_to_slide"], dtype=np.float64)
    slide_points = np.asarray(
        [[row["source_x_px"], row["source_y_px"]] for row in summary["landmarks"]],
        dtype=np.float64,
    )
    native_points = transform_points(slide_points, np.linalg.inv(dicom_to_slide))
    rows: list[dict[str, object]] = []
    for source, native in zip(summary["landmarks"], native_points):
        rows.append(
            {
                "landmark_id": source["landmark_id"],
                "kind": source["kind"],
                "slide_xy_px": [float(source["source_x_px"]), float(source["source_y_px"])],
                "localizer_uv_px": native.tolist(),
            }
        )
    return rows, physical_path


def validate_frame_of_reference(ct_uid: str, localizer_uid: str) -> None:
    if not ct_uid or not localizer_uid:
        raise RuntimeError("Missing FrameOfReferenceUID in CT or physical localizer")
    if ct_uid != localizer_uid:
        raise RuntimeError(
            "CT and physical localizer do not share FrameOfReferenceUID: "
            f"ct={ct_uid}, localizer={localizer_uid}"
        )


def image_geometry_matches(reference: sitk.Image, candidate: sitk.Image, tolerance: float = 1e-4) -> bool:
    return (
        reference.GetSize() == candidate.GetSize()
        and np.allclose(reference.GetSpacing(), candidate.GetSpacing(), atol=tolerance)
        and np.allclose(reference.GetOrigin(), candidate.GetOrigin(), atol=tolerance)
        and np.allclose(reference.GetDirection(), candidate.GetDirection(), atol=tolerance)
    )


def voxel_indices_to_lps(indices_zyx: np.ndarray, geometry: MaskGeometry) -> np.ndarray:
    indices_xyz = np.asarray(indices_zyx, dtype=np.float64)[:, ::-1]
    scaled = indices_xyz * geometry.spacing_mm[None, :]
    return geometry.origin_lps_mm[None, :] + scaled @ geometry.direction_lps.T


def load_vertebra_masks(
    ct_image: sitk.Image,
    masks_dir: Path,
) -> tuple[dict[str, np.ndarray], dict[str, dict[str, object]]]:
    masks: dict[str, np.ndarray] = {}
    metrics: dict[str, dict[str, object]] = {}
    geometry = mask_geometry(ct_image)
    for label in VERTEBRAE:
        path = masks_dir / f"{label}.nii.gz"
        if not path.exists():
            continue
        image = sitk.ReadImage(str(path))
        if not image_geometry_matches(ct_image, image):
            raise RuntimeError(f"Mask geometry differs from CT geometry: {path}")
        indices = np.argwhere(sitk.GetArrayViewFromImage(image) > 0)
        if len(indices) == 0:
            continue
        points = voxel_indices_to_lps(indices, geometry)
        masks[label] = points
        metrics[label] = {
            "path": str(path.resolve()),
            "voxel_count": int(len(points)),
            "volume_mm3": float(len(points) * np.prod(geometry.spacing_mm)),
            "centroid_lps_mm": np.mean(points, axis=0).tolist(),
            "bbox_lps_min_mm": np.min(points, axis=0).tolist(),
            "bbox_lps_max_mm": np.max(points, axis=0).tolist(),
            "touches_volume_border": bool(
                np.any(indices == 0)
                or np.any(indices[:, 0] == geometry.size_xyz[2] - 1)
                or np.any(indices[:, 1] == geometry.size_xyz[1] - 1)
                or np.any(indices[:, 2] == geometry.size_xyz[0] - 1)
            ),
        }
    if not masks:
        raise RuntimeError(f"No non-empty vertebra masks found in {masks_dir}")
    return masks, metrics


def projected_distance_mm(
    points_lps: np.ndarray,
    reference_lps: np.ndarray,
    geometry: LocalizerGeometry,
) -> np.ndarray:
    delta = points_lps - np.asarray(reference_lps, dtype=np.float64)[None, :]
    horizontal = delta @ geometry.row_direction_lps
    vertical = delta @ geometry.column_direction_lps
    return np.hypot(horizontal, vertical)


def select_label_for_landmark(
    landmark_id: str,
    reference_lps: np.ndarray,
    masks: dict[str, np.ndarray],
    geometry: LocalizerGeometry,
) -> tuple[str, list[tuple[str, float]], str]:
    distances = sorted(
        (
            (label, float(np.min(projected_distance_mm(points, reference_lps, geometry))))
            for label, points in masks.items()
        ),
        key=lambda item: item[1],
    )
    semantic = SEMANTIC_LANDMARK_LABELS.get(landmark_id)
    if semantic in masks:
        return semantic, distances, "physician_anatomical_label"
    return distances[0][0], distances, "nearest_projected_vertebra"


def recover_landmark_depths(
    landmarks: list[dict[str, object]],
    masks: dict[str, np.ndarray],
    geometry: LocalizerGeometry,
    neighborhood_mm: float,
) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    for landmark in landmarks:
        uv = np.asarray(landmark["localizer_uv_px"], dtype=np.float64)
        reference = localizer_pixel_to_lps(uv, geometry)[0]
        label, distances, selection_reason = select_label_for_landmark(
            str(landmark["landmark_id"]), reference, masks, geometry
        )
        points = masks[label]
        distance = projected_distance_mm(points, reference, geometry)
        minimum = float(np.min(distance))
        adaptive_radius = max(float(neighborhood_mm), minimum + 1.5)
        local_points = points[distance <= adaptive_radius]
        local_depth = (local_points - reference[None, :]) @ geometry.normal_direction_lps
        quantiles = np.percentile(local_depth, [0, 10, 25, 50, 75, 90, 100])
        estimate_depth = float(quantiles[3])
        estimate_lps = reference + estimate_depth * geometry.normal_direction_lps
        second_distance = next((value for name, value in distances if name != label), float("nan"))
        interval_width = float(quantiles[5] - quantiles[1])
        label_margin = float(second_distance - minimum)
        if minimum <= 2.0 and len(local_points) >= 20 and interval_width <= 10.0:
            quality = "high"
        elif minimum <= 8.0 and len(local_points) >= 10 and interval_width <= 20.0:
            quality = "medium"
        else:
            quality = "low"
        if selection_reason == "physician_anatomical_label" and minimum > 12.0:
            quality = "low"
        output.append(
            {
                **landmark,
                "selected_vertebra": label,
                "selection_reason": selection_reason,
                "reference_plane_lps_mm": reference.tolist(),
                "estimated_lps_mm": estimate_lps.tolist(),
                "estimated_depth_from_localizer_plane_mm": estimate_depth,
                "depth_interval_p10_p90_mm": [float(quantiles[1]), float(quantiles[5])],
                "depth_interval_width_p10_p90_mm": interval_width,
                "depth_interval_full_mask_mm": [float(quantiles[0]), float(quantiles[6])],
                "depth_iqr_mm": float(quantiles[4] - quantiles[2]),
                "projected_distance_to_selected_mask_mm": minimum,
                "nearest_other_label_distance_mm": second_distance,
                "projected_label_margin_mm": label_margin,
                "label_ambiguity": (
                    "resolved_by_physician_anatomical_label"
                    if selection_reason == "physician_anatomical_label" and label_margin < 2.0
                    else ("projected_labels_overlap" if label_margin < 2.0 else "none")
                ),
                "adaptive_neighborhood_radius_mm": adaptive_radius,
                "local_mask_voxel_count": int(len(local_points)),
                "quality": quality,
                "interpretation": (
                    "ray-mask median depth estimate; not unique single-view triangulation"
                ),
                "nearest_labels": [
                    {"label": name, "projected_distance_mm": value}
                    for name, value in distances[:4]
                ],
            }
        )
    return output


def normalized_dicom_pixels(dataset: object) -> np.ndarray:
    image = np.asarray(dataset.pixel_array, dtype=np.float32)
    if image.ndim > 2:
        image = image[0]
    slope = float(getattr(dataset, "RescaleSlope", 1.0))
    intercept = float(getattr(dataset, "RescaleIntercept", 0.0))
    image = image * slope + intercept
    low, high = np.percentile(image, [1.0, 99.5])
    image = np.clip((image - low) / max(high - low, 1e-6), 0.0, 1.0)
    if str(getattr(dataset, "PhotometricInterpretation", "")) == "MONOCHROME1":
        image = 1.0 - image
    return image


def label_colors(count: int) -> np.ndarray:
    base = plt.get_cmap("turbo")(np.linspace(0.05, 0.95, max(count, 2)))[:, :3]
    return np.vstack([[0.0, 0.0, 0.0], base[:count]])


def rasterize_projected_masks(
    masks: dict[str, np.ndarray],
    geometry: LocalizerGeometry,
) -> tuple[np.ndarray, list[str]]:
    labels = sorted(masks, key=lambda name: float(np.mean(masks[name][:, 2])), reverse=True)
    raster = np.zeros((geometry.rows, geometry.columns), dtype=np.int16)
    for index, label in enumerate(labels, start=1):
        uv, _ = lps_to_localizer_pixel(masks[label], geometry)
        ij = np.rint(uv).astype(np.int32)
        valid = (
            (ij[:, 0] >= 0)
            & (ij[:, 0] < geometry.columns)
            & (ij[:, 1] >= 0)
            & (ij[:, 1] < geometry.rows)
        )
        raster[ij[valid, 1], ij[valid, 0]] = index
    return raster, labels


def save_localizer_overlay(
    dataset: object,
    geometry: LocalizerGeometry,
    masks: dict[str, np.ndarray],
    recovered: list[dict[str, object]],
    output_path: Path,
) -> None:
    background = normalized_dicom_pixels(dataset)
    raster, labels = rasterize_projected_masks(masks, geometry)
    colors = label_colors(len(labels))
    overlay = colors[raster]
    alpha = np.where(raster > 0, 0.38, 0.0)

    fig, axes = plt.subplots(1, 2, figsize=(12, 6.5), dpi=170)
    axes[0].imshow(background, cmap="gray", vmin=0.0, vmax=1.0)
    axes[0].set_title("CT localizer + projected vertebral-body masks")
    axes[0].imshow(overlay, alpha=alpha)
    axes[1].imshow(background, cmap="gray", vmin=0.0, vmax=1.0)
    axes[1].set_title("Depth-assigned physician landmarks")
    for axis in axes:
        for row in recovered:
            u, v = row["localizer_uv_px"]
            color = "lime" if row["quality"] != "low" else "red"
            axis.scatter([u], [v], s=65, facecolors="none", edgecolors=color, linewidths=1.8)
            axis.text(
                u + 5,
                v - 5,
                f"{row['landmark_id']} -> {str(row['selected_vertebra']).replace('vertebrae_', '')}\n"
                f"d={row['estimated_depth_from_localizer_plane_mm']:.1f} mm",
                color=color,
                fontsize=7,
                bbox={"facecolor": "black", "alpha": 0.55, "edgecolor": "none", "pad": 1.5},
            )
        axis.set_xlim(0, geometry.columns)
        axis.set_ylim(geometry.rows, 0)
        axis.set_axis_off()
    fig.suptitle("TotalSegmentator vertebrae_pp_refined (research-only depth prior)")
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def window_ct(array: np.ndarray, center: float = 300.0, width: float = 1800.0) -> np.ndarray:
    low = center - width / 2.0
    high = center + width / 2.0
    return np.clip((array.astype(np.float32) - low) / width, 0.0, 1.0)


def save_mpr_overlay(
    ct_image: sitk.Image,
    masks_dir: Path,
    masks: dict[str, np.ndarray],
    output_path: Path,
) -> None:
    ct = np.asarray(sitk.GetArrayViewFromImage(ct_image))
    names = list(masks)
    centre_lps = np.median(np.concatenate(list(masks.values()), axis=0), axis=0)
    x_float, y_float, z_float = ct_image.TransformPhysicalPointToContinuousIndex(
        tuple(map(float, centre_lps))
    )
    x_index = int(np.clip(round(x_float), 0, ct_image.GetSize()[0] - 1))
    y_index = int(np.clip(round(y_float), 0, ct_image.GetSize()[1] - 1))
    z_index = int(np.clip(round(z_float), 0, ct_image.GetSize()[2] - 1))
    axial_labels = np.zeros(ct[z_index].shape, dtype=np.int16)
    coronal_labels = np.zeros(ct[:, y_index, :].shape, dtype=np.int16)
    sagittal_labels = np.zeros(ct[:, :, x_index].shape, dtype=np.int16)
    for label_index, label in enumerate(names, start=1):
        image = sitk.ReadImage(str(masks_dir / f"{label}.nii.gz"))
        array = sitk.GetArrayViewFromImage(image)
        axial_labels[array[z_index] > 0] = label_index
        coronal_labels[array[:, y_index, :] > 0] = label_index
        sagittal_labels[array[:, :, x_index] > 0] = label_index
    panels = [
        (ct[z_index], axial_labels, f"Axial z-index {z_index}"),
        (ct[:, y_index, :], coronal_labels, f"Coronal y-index {y_index}"),
        (ct[:, :, x_index], sagittal_labels, f"Sagittal x-index {x_index}"),
    ]
    colors = ListedColormap(label_colors(len(names)))
    fig, axes = plt.subplots(1, 3, figsize=(15, 7), dpi=170)
    for axis, (background, mask, title) in zip(axes, panels):
        axis.imshow(window_ct(background), cmap="gray", vmin=0.0, vmax=1.0)
        axis.imshow(np.ma.masked_where(mask == 0, mask), cmap=colors, vmin=0, vmax=len(names), alpha=0.42)
        axis.set_title(title)
        axis.set_axis_off()
    fig.suptitle(f"CT Series 4 with {len(names)} non-empty labelled vertebra masks")
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def write_depth_csv(path: Path, recovered: Iterable[dict[str, object]]) -> None:
    rows: list[dict[str, object]] = []
    for row in recovered:
        estimate = row["estimated_lps_mm"]
        p10, p90 = row["depth_interval_p10_p90_mm"]
        rows.append(
            {
                "landmark_id": row["landmark_id"],
                "selected_vertebra": row["selected_vertebra"],
                "selection_reason": row["selection_reason"],
                "x_lps_mm": estimate[0],
                "y_lps_mm": estimate[1],
                "z_lps_mm": estimate[2],
                "depth_from_localizer_plane_mm": row["estimated_depth_from_localizer_plane_mm"],
                "depth_p10_mm": p10,
                "depth_p90_mm": p90,
                "depth_iqr_mm": row["depth_iqr_mm"],
                "projected_mask_distance_mm": row["projected_distance_to_selected_mask_mm"],
                "local_mask_voxel_count": row["local_mask_voxel_count"],
                "quality": row["quality"],
            }
        )
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-id", required=True, help="Case key from the caller's approved dataset")
    parser.add_argument("--ct-dir", type=Path, required=True)
    parser.add_argument("--ct-series", required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--task", default="vertebrae_pp_refined")
    parser.add_argument("--device", default="gpu")
    parser.add_argument("--neighborhood-mm", type=float, default=4.0)
    parser.add_argument("--skip-inference", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    input_nifti = output_dir / "ct_series4_native_geometry.nii.gz"
    masks_dir = output_dir / "vertebrae_masks"
    cache_dir = output_dir / "totalsegmentator_cache"

    ct_image, ct_metadata = write_ct_nifti(
        args.ct_dir.resolve(), args.ct_series, input_nifti, args.overwrite
    )
    print(
        f"ct_ready size={ct_image.GetSize()} spacing={ct_image.GetSpacing()} "
        f"hu=[{ct_metadata['hu_min']:.0f}, {ct_metadata['hu_max']:.0f}]",
        flush=True,
    )
    if not args.skip_inference:
        run_totalsegmentator(
            input_nifti,
            masks_dir,
            cache_dir,
            args.task,
            args.device,
            args.overwrite,
        )

    masks, mask_metrics = load_vertebra_masks(ct_image, masks_dir)
    print(f"loaded_non_empty_masks={sorted(masks)}", flush=True)
    landmarks, localizer_path = source_landmarks_in_localizer(args.summary.resolve())
    localizer_dataset = pydicom.dcmread(str(localizer_path), force=True)
    localizer_frame_uid = str(getattr(localizer_dataset, "FrameOfReferenceUID", ""))
    validate_frame_of_reference(
        str(ct_metadata["frame_of_reference_uid"]), localizer_frame_uid
    )
    localizer = localizer_geometry(localizer_dataset)
    recovered = recover_landmark_depths(
        landmarks, masks, localizer, neighborhood_mm=args.neighborhood_mm
    )
    print("landmark_depths_recovered", flush=True)

    save_mpr_overlay(
        ct_image,
        masks_dir,
        masks,
        output_dir / "vertebrae_segmentation_mpr.png",
    )
    print("mpr_overlay_saved", flush=True)
    save_localizer_overlay(
        localizer_dataset,
        localizer,
        masks,
        recovered,
        output_dir / "localizer_vertebrae_depth_overlay.png",
    )
    write_depth_csv(output_dir / "landmark_depths.csv", recovered)

    try:
        version = importlib.metadata.version("TotalSegmentator")
    except importlib.metadata.PackageNotFoundError:
        version = "unknown"
    result = {
        "metadata": {
            "case_id": args.case_id,
            "clinical_status": "research_only_not_clinically_validated",
            "method": "TotalSegmentator vertebral-body mask constrained single-view depth estimate",
            "totalsegmentator_version": version,
            "totalsegmentator_task": args.task,
            "device": args.device,
            "usage_statistics_disabled": True,
            "depth_definition": (
                "median signed distance along the CT-localizer normal among selected-vertebra "
                "voxels within the adaptive projected neighbourhood"
            ),
            "single_view_limitation": (
                "A localizer pixel defines a ray, not a unique 3D point. The reported point uses "
                "a segmented-vertebra prior and must retain its depth interval."
            ),
        },
        "ct": ct_metadata,
        "localizer": {
            "path": str(localizer_path.resolve()),
            "frame_of_reference_uid": localizer_frame_uid,
            **{
                key: value.tolist() if isinstance(value, np.ndarray) else value
                for key, value in asdict(localizer).items()
            },
        },
        "vertebrae": mask_metrics,
        "landmarks": recovered,
        "outputs": {
            "ct_nifti": str(input_nifti.resolve()),
            "masks_dir": str(masks_dir.resolve()),
            "mpr_overlay": str((output_dir / "vertebrae_segmentation_mpr.png").resolve()),
            "localizer_depth_overlay": str(
                (output_dir / "localizer_vertebrae_depth_overlay.png").resolve()
            ),
            "depth_csv": str((output_dir / "landmark_depths.csv").resolve()),
        },
    }
    result_path = output_dir / "landmark_depths.json"
    result_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"non_empty_vertebrae={len(masks)}")
    for row in recovered:
        estimate = row["estimated_lps_mm"]
        print(
            f"{row['landmark_id']}: {row['selected_vertebra']} "
            f"LPS=({estimate[0]:.2f}, {estimate[1]:.2f}, {estimate[2]:.2f}) mm "
            f"depth={row['estimated_depth_from_localizer_plane_mm']:.2f} mm "
            f"quality={row['quality']}"
        )
    print(f"result={result_path}")


if __name__ == "__main__":
    main()
