"""Run a DeepSA-based coronary vessel segmentation baseline on XA cine.

This is an engineering baseline, not a clinically validated segmentation.
DeepSA is fused with a weak multi-scale Frangi prior to reduce gross failures
on an unseen hospital acquisition domain.  Every output is labelled for
physician review.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from scipy import ndimage
from skimage import filters, morphology

try:
    from .xa_cine import read_cine, write_frames, write_json, write_mp4, write_raw_stack
except ImportError:  # direct script execution
    from xa_cine import read_cine, write_frames, write_json, write_mp4, write_raw_stack


def enhanced_tophat(frame: np.ndarray, kernel_size: int = 50) -> np.ndarray:
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    white = cv2.morphologyEx(frame, cv2.MORPH_TOPHAT, kernel).astype(np.float32)
    black = cv2.morphologyEx(frame, cv2.MORPH_BLACKHAT, kernel).astype(np.float32)
    return np.clip(frame.astype(np.float32) + white - black, 0, 255).astype(np.uint8)


def normalize_probability(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    low, high = np.percentile(values, (1.0, 99.0))
    if high <= low + 1e-6:
        return np.zeros_like(values)
    return np.clip((values - low) / (high - low), 0.0, 1.0)


def load_deepsa(model_code_dir: Path, checkpoint_path: Path, device: torch.device) -> torch.nn.Module:
    model_code_dir = Path(model_code_dir).resolve()
    if not (model_code_dir / "models.py").is_file():
        raise FileNotFoundError(f"DeepSA model source not found: {model_code_dir / 'models.py'}")
    sys.path.insert(0, str(model_code_dir))
    try:
        from models import UNet
    finally:
        sys.path.pop(0)
    model = UNet(1, 1, 32, bilinear=True)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    state = checkpoint["netE"] if isinstance(checkpoint, dict) and "netE" in checkpoint else checkpoint
    state = {str(key).replace("module.", ""): value for key, value in state.items()}
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def deepsa_probabilities(frames: np.ndarray, model: torch.nn.Module, device: torch.device, batch_size: int = 4) -> np.ndarray:
    all_probabilities: list[np.ndarray] = []
    for start in range(0, len(frames), batch_size):
        batch = frames[start : start + batch_size]
        variants = [batch, np.stack([enhanced_tophat(frame) for frame in batch])]
        variant_outputs = []
        for variant in variants:
            tensor = torch.from_numpy(variant[:, None].astype(np.float32) / 127.5 - 1.0).to(device)
            with torch.inference_mode():
                output = model.model(tensor).detach().float().cpu().numpy()[:, 0]
            variant_outputs.append(np.stack([normalize_probability(frame) for frame in output]))
        merged = np.maximum(variant_outputs[0], variant_outputs[1])
        all_probabilities.extend(merged)
    return np.asarray(all_probabilities, dtype=np.float32)


def frangi_probabilities(frames: np.ndarray) -> np.ndarray:
    probabilities = []
    for frame in frames:
        normalized = frame.astype(np.float32) / 255.0
        vesselness = filters.frangi(normalized, sigmas=range(1, 5), black_ridges=True)
        probabilities.append(normalize_probability(vesselness))
    return np.asarray(probabilities, dtype=np.float32)


def postprocess_probability(probability: np.ndarray, min_size: int = 350) -> np.ndarray:
    probability = np.clip(np.asarray(probability, dtype=np.float32), 0.0, 1.0)
    nonzero = probability[probability > 0]
    if nonzero.size < 16 or float(np.std(nonzero)) < 1e-6:
        return np.zeros(probability.shape, dtype=bool)
    global_threshold = float(filters.threshold_yen(probability))
    block_size = min(31, min(probability.shape) // 2 * 2 - 1)
    block_size = max(block_size, 3)
    local_threshold = filters.threshold_local(probability, block_size=block_size, offset=-0.015)
    mask = (probability >= max(global_threshold * 0.72, 0.18)) & (probability >= local_threshold)
    mask = morphology.closing(mask, morphology.disk(1))
    mask = morphology.remove_small_objects(mask, max_size=min_size - 1)
    mask = morphology.remove_small_holes(mask, max_size=79)
    return np.asarray(mask, dtype=bool)


def segmentation_metrics(frame: np.ndarray, probability: np.ndarray, mask: np.ndarray, skeleton: np.ndarray, index: int) -> dict[str, Any]:
    area_fraction = float(np.mean(mask))
    centerline_pixels = int(np.count_nonzero(skeleton))
    components = int(ndimage.label(mask)[1])
    vessel_values = frame[mask]
    background_values = frame[~mask]
    contrast = float(np.mean(background_values) - np.mean(vessel_values)) if vessel_values.size and background_values.size else 0.0
    return {
        "frame_index": index,
        "mask_area_pixels": int(np.count_nonzero(mask)),
        "mask_area_fraction": area_fraction,
        "centerline_pixels": centerline_pixels,
        "connected_components": components,
        "dark_vessel_contrast_gray": contrast,
        "mean_probability_in_mask": float(np.mean(probability[mask])) if np.any(mask) else 0.0,
        "quality_flag": "review" if not 0.005 <= area_fraction <= 0.35 or centerline_pixels < 100 else "provisional",
    }


def make_overlays(frames: np.ndarray, probabilities: np.ndarray, masks: np.ndarray, skeletons: np.ndarray) -> np.ndarray:
    overlays = []
    for frame, probability, mask, skeleton in zip(frames, probabilities, masks, skeletons):
        rgb = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB).astype(np.float32)
        strength = probability[..., None] * mask[..., None]
        vessel_color = np.asarray([15, 235, 145], dtype=np.float32)
        rgb = rgb * (1.0 - 0.58 * strength) + vessel_color * (0.58 * strength)
        rgb[skeleton] = np.asarray([255, 224, 40], dtype=np.float32)
        overlays.append(np.clip(rgb, 0, 255).astype(np.uint8))
    return np.asarray(overlays)


def run_segmentation(dicom_path: Path, checkpoint_path: Path, output_dir: Path, device_name: str = "auto", model_code_dir: Path | None = None) -> dict[str, Any]:
    if model_code_dir is None:
        raise ValueError("DeepSA source must be supplied explicitly via --model-code-dir")
    _, frames, frame_times = read_cine(dicom_path)
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)
    model = load_deepsa(model_code_dir, checkpoint_path, device)
    deep = deepsa_probabilities(frames, model, device)
    classical = frangi_probabilities(frames)
    probabilities = np.maximum(deep, 0.30 * classical)
    masks = np.asarray([postprocess_probability(frame) for frame in probabilities], dtype=bool)
    skeletons = np.asarray([morphology.skeletonize(mask) for mask in masks], dtype=bool)
    metrics = [segmentation_metrics(frame, probability, mask, skeleton, index) for index, (frame, probability, mask, skeleton) in enumerate(zip(frames, probabilities, masks, skeletons))]

    probabilities_u8 = np.round(probabilities * 255.0).astype(np.uint8)
    masks_u8 = masks.astype(np.uint8) * 255
    skeletons_u8 = skeletons.astype(np.uint8) * 255
    overlays = make_overlays(frames, probabilities, masks, skeletons)
    write_frames(probabilities_u8, output_dir / "probabilities", "probability")
    write_frames(masks_u8, output_dir / "masks", "mask")
    write_frames(skeletons_u8, output_dir / "centerlines", "centerline")
    write_frames(overlays, output_dir / "overlays", "overlay")
    assets = {
        "probabilities": {"path": "probabilities_u8.raw", "byte_length": write_raw_stack(output_dir / "probabilities_u8.raw", probabilities_u8)},
        "masks": {"path": "masks_u8.raw", "byte_length": write_raw_stack(output_dir / "masks_u8.raw", masks_u8)},
        "centerlines": {"path": "centerlines_u8.raw", "byte_length": write_raw_stack(output_dir / "centerlines_u8.raw", skeletons_u8)},
    }
    write_mp4(output_dir / "vessel_overlay.mp4", overlays, frame_times, color=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "segmentation_quality.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(metrics[0]))
        writer.writeheader()
        writer.writerows(metrics)
    report = {
        "schema_version": 1,
        "method": "DeepSA FS checkpoint plus weak Frangi safety prior",
        "checkpoint": str(checkpoint_path),
        "device": str(device),
        "frame_count": int(len(frames)),
        "rows": int(frames.shape[1]),
        "columns": int(frames.shape[2]),
        "assets": assets,
        "quality_summary": {
            "provisional_frames": sum(item["quality_flag"] == "provisional" for item in metrics),
            "review_frames": sum(item["quality_flag"] == "review" for item in metrics),
            "median_mask_area_fraction": float(np.median([item["mask_area_fraction"] for item in metrics])),
            "median_centerline_pixels": float(np.median([item["centerline_pixels"] for item in metrics])),
            "median_dark_vessel_contrast_gray": float(np.median([item["dark_vessel_contrast_gray"] for item in metrics])),
        },
        "clinical_status": "research-only; physician review required",
        "known_limitations": [
            "DeepSA checkpoint was not trained or validated on this hospital series.",
            "Mask quality changes with contrast-agent filling and overlapping anatomy.",
            "The result is a 2D projection mask, not a 3D vessel segmentation.",
        ],
    }
    write_json(output_dir / "segmentation_report.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dicom", type=Path, required=True)
    parser.add_argument("--model-code-dir", type=Path, required=True, help="External DeepSA source directory containing models.py")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Externally obtained model weights; not bundled")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    report = run_segmentation(args.dicom, args.checkpoint, args.output_dir, args.device, args.model_code_dir)
    print(json.dumps(report["quality_summary"], ensure_ascii=False))


if __name__ == "__main__":
    main()
