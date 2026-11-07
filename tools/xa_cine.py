"""Shared helpers for reading, auditing, and exporting XA cine DICOM files.

The functions in this module deliberately keep the DICOM row/column order.
No implicit vertical or horizontal flip is performed.  MONOCHROME1 is inverted
only for display so that returned frames use the conventional bright-vessel
/ bright-bone presentation.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np
import pydicom
from PIL import Image


def _plain(value: Any) -> Any:
    """Convert pydicom/numpy values into JSON-safe Python values."""
    if value is None:
        return None
    if isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (bytes, bytearray)):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, Iterable):
        return [_plain(item) for item in value]
    return str(value)


def dicom_value(dataset: pydicom.Dataset, keyword: str, default: Any = None) -> Any:
    value = getattr(dataset, keyword, default)
    return _plain(value)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def frame_times_ms(dataset: pydicom.Dataset, frame_count: int) -> list[float]:
    """Return one display duration per frame, preferring exact DICOM timing."""
    vector = getattr(dataset, "FrameTimeVector", None)
    if vector is not None:
        values = [float(value) for value in vector]
        if len(values) == frame_count:
            return values
    frame_time = getattr(dataset, "FrameTime", None)
    if frame_time is not None and float(frame_time) > 0:
        return [float(frame_time)] * frame_count
    cine_rate = getattr(dataset, "CineRate", None) or getattr(dataset, "RecommendedDisplayFrameRate", None)
    if cine_rate is not None and float(cine_rate) > 0:
        return [1000.0 / float(cine_rate)] * frame_count
    return [125.0] * frame_count


def read_cine(path: str | Path) -> tuple[pydicom.Dataset, np.ndarray, list[float]]:
    """Read one XA cine as display-ready uint8 frames without geometric flips."""
    path = Path(path)
    dataset = pydicom.dcmread(path)
    pixels = np.asarray(dataset.pixel_array)
    if pixels.ndim == 2:
        pixels = pixels[np.newaxis, ...]
    if pixels.ndim != 3:
        raise ValueError(f"Expected 2D or multi-frame grayscale XA, got shape {pixels.shape}")

    if pixels.dtype != np.uint8:
        # XA can be 10/12/16-bit.  Export is display-oriented in that case;
        # the original DICOM remains the lossless source of truth.
        display = np.empty(pixels.shape, dtype=np.uint8)
        for index, frame in enumerate(pixels.astype(np.float32)):
            low, high = np.percentile(frame, (0.5, 99.5))
            if high <= low:
                display[index] = 0
            else:
                display[index] = np.clip((frame - low) * 255.0 / (high - low), 0, 255).astype(np.uint8)
        pixels = display
    else:
        pixels = pixels.copy()

    if str(getattr(dataset, "PhotometricInterpretation", "MONOCHROME2")) == "MONOCHROME1":
        pixels = 255 - pixels
    times = frame_times_ms(dataset, int(pixels.shape[0]))
    return dataset, np.ascontiguousarray(pixels), times


def audit_record(path: str | Path, root: str | Path | None = None) -> dict[str, Any]:
    path = Path(path)
    dataset = pydicom.dcmread(path, stop_before_pixels=True)
    frame_count = int(getattr(dataset, "NumberOfFrames", 1))
    times = frame_times_ms(dataset, frame_count)
    relative = str(path.relative_to(Path(root))) if root is not None else str(path)
    record = {
        "path": relative.replace("\\", "/"),
        "sha256": sha256_file(path),
        "sop_instance_uid": dicom_value(dataset, "SOPInstanceUID"),
        "series_instance_uid": dicom_value(dataset, "SeriesInstanceUID"),
        "study_instance_uid": dicom_value(dataset, "StudyInstanceUID"),
        "series_number": dicom_value(dataset, "SeriesNumber"),
        "instance_number": dicom_value(dataset, "InstanceNumber"),
        "modality": dicom_value(dataset, "Modality"),
        "rows": int(dataset.Rows),
        "columns": int(dataset.Columns),
        "frame_count": frame_count,
        "frame_time_ms": float(np.median(times)),
        "duration_ms": float(np.sum(times)),
        "photometric_interpretation": dicom_value(dataset, "PhotometricInterpretation"),
        "bits_allocated": dicom_value(dataset, "BitsAllocated"),
        "bits_stored": dicom_value(dataset, "BitsStored"),
        "pixel_representation": dicom_value(dataset, "PixelRepresentation"),
        "primary_angle_deg": dicom_value(dataset, "PositionerPrimaryAngle"),
        "secondary_angle_deg": dicom_value(dataset, "PositionerSecondaryAngle"),
        "source_to_detector_mm": dicom_value(dataset, "DistanceSourceToDetector"),
        "source_to_patient_mm": dicom_value(dataset, "DistanceSourceToPatient"),
        "imager_pixel_spacing_row_col_mm": dicom_value(dataset, "ImagerPixelSpacing"),
        "positioner_motion": dicom_value(dataset, "PositionerMotion"),
        "table_motion": dicom_value(dataset, "TableMotion"),
        "frame_of_reference_uid": dicom_value(dataset, "FrameOfReferenceUID"),
        "image_position_patient": dicom_value(dataset, "ImagePositionPatient"),
        "image_orientation_patient": dicom_value(dataset, "ImageOrientationPatient"),
        "has_ecg": any(hasattr(dataset, key) for key in ("TriggerTime", "HeartRate", "CardiacNumberOfImages")),
    }
    record["geometry_limitations"] = [
        label
        for present, label in (
            (record["frame_of_reference_uid"], "missing FrameOfReferenceUID"),
            (record["image_position_patient"], "missing ImagePositionPatient"),
            (record["image_orientation_patient"], "missing ImageOrientationPatient"),
            (record["has_ecg"], "missing ECG/trigger metadata"),
        )
        if not present
    ]
    return record


def write_json(path: str | Path, value: Any) -> None:
    path = Path(path)
    serialized = json.dumps(
        value, indent=2, ensure_ascii=False, allow_nan=False
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(serialized, encoding="utf-8")


def write_frames(frames: np.ndarray, output_dir: str | Path, prefix: str = "frame") -> list[Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for index, frame in enumerate(frames):
        path = output_dir / f"{prefix}_{index:04d}.png"
        mode = "L" if frame.ndim == 2 else "RGB"
        Image.fromarray(frame, mode=mode).save(path)
        paths.append(path)
    return paths


def write_raw_stack(path: str | Path, frames: np.ndarray) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = np.ascontiguousarray(frames, dtype=np.uint8)
    path.write_bytes(data.tobytes(order="C"))
    return int(data.nbytes)


def write_mp4(path: str | Path, frames: np.ndarray, frame_times: list[float], *, color: bool = False) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    height, width = frames.shape[1:3]
    fps = 1000.0 / max(float(np.median(frame_times)), 1.0)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height), isColor=True)
    if not writer.isOpened():
        raise RuntimeError(f"Could not create MP4: {path}")
    try:
        for frame in frames:
            if frame.ndim == 2:
                bgr = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            elif color:
                bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            else:
                bgr = frame
            writer.write(np.ascontiguousarray(bgr, dtype=np.uint8))
    finally:
        writer.release()
