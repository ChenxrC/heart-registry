"""Stable, path-explicit Python facade over the research pipeline components.

The API deliberately does not discover patient data or choose a pose implicitly.
All inputs, outputs, configuration, and model assets are supplied by the caller.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ctdr_pipeline.bundle import load_bundle
from ctdr_pipeline.config import load_config
from ctdr_pipeline.dicom.scanner import scan_data_root, write_dicom_manifest
from ctdr_pipeline.pipeline import (
    BatchRunRequest,
    CaseRunRequest,
    RunSummary,
    run_all,
    run_case as _run_case,
)
from ctdr_pipeline.pptx.export import prepare_pptx


def prepare_annotation_bundle(
    pptx_path: Path,
    output_dir: Path,
    overrides_path: Path | None = None,
) -> Any:
    """Extract reviewed PPTX pairs, colored marks, and 2-D transform records."""

    return prepare_pptx(Path(pptx_path), Path(output_dir), overrides_path)


def scan_dicom_tree(data_root: Path, output_manifest: Path | None = None) -> Any:
    """Index case folders by DICOM metadata; optionally write a YAML manifest."""

    manifest = scan_data_root(Path(data_root))
    if output_manifest is not None:
        write_dicom_manifest(manifest, Path(output_manifest))
    return manifest


def register_case(
    case_id: str,
    bundle_dir: Path,
    dicom_manifest: Path,
    output_dir: Path,
    config_path: Path,
    device: str | None = None,
) -> Any:
    """Run one auditable 2-D/3-D registration case and quality evaluation."""

    return _run_case(
        CaseRunRequest(
            case_id=str(case_id),
            bundle_path=Path(bundle_dir),
            dicom_manifest_path=Path(dicom_manifest),
            output_dir=Path(output_dir),
            config=load_config(Path(config_path)),
            device=device,
        )
    )


def register_cases(
    bundle_dir: Path,
    dicom_manifest: Path,
    output_dir: Path,
    config_path: Path,
    *,
    fail_on_review: bool | None = None,
) -> RunSummary:
    """Run all cases represented in the annotation bundle with resumable stages."""

    config = load_config(Path(config_path))
    if fail_on_review is not None:
        config = config.model_copy(update={"fail_on_review": fail_on_review})
    bundle = load_bundle(Path(bundle_dir))
    return run_all(
        BatchRunRequest(
            case_ids=tuple(record.case_id for record in bundle.cases),
            bundle_path=Path(bundle_dir),
            dicom_manifest_path=Path(dicom_manifest),
            output_dir=Path(output_dir),
            config=config,
        )
    )


def extract_xa_cine(dicom_path: Path, output_dir: Path) -> dict[str, Any]:
    """Export XA frames, frame timing, a raw stack, and a preview manifest."""

    from tools.extract_xa_cine import extract_cine

    return extract_cine(Path(dicom_path), Path(output_dir))


def segment_xa_vessels(
    dicom_path: Path,
    model_code_dir: Path,
    checkpoint_path: Path,
    output_dir: Path,
    *,
    device: str = "auto",
) -> dict[str, Any]:
    """Run the DeepSA+Frangi XA baseline using separately supplied model assets."""

    from tools.run_xa_vessel_segmentation import run_segmentation

    return run_segmentation(
        Path(dicom_path),
        Path(checkpoint_path),
        Path(output_dir),
        device,
        Path(model_code_dir),
    )


def fuse_xa_temporal_masks(
    dicom_path: Path,
    segmentation_dir: Path,
    output_dir: Path,
    *,
    baseline_dir: Path | None = None,
) -> dict[str, Any]:
    """Motion-compensate and audit a temporal XA vessel-mask sequence."""

    from tools.run_xa_temporal_fusion import run_temporal

    return run_temporal(
        Path(dicom_path),
        Path(segmentation_dir),
        Path(output_dir),
        Path(baseline_dir) if baseline_dir is not None else None,
    )


def build_xa_dynamic_vessel_mesh(
    series_dir: Path,
    masks_dir: Path,
    output_dir: Path,
    *,
    multiview_path: Path | None = None,
    calibration_path: Path | None = None,
    temporal_dir: Path | None = None,
) -> dict[str, Any]:
    """Build audited research-only 2.5-D vessel meshes from XA topology."""

    from tools.build_dynamic_vessel_3d import build_dynamic_3d

    return build_dynamic_3d(
        Path(series_dir),
        Path(masks_dir),
        Path(output_dir),
        Path(multiview_path) if multiview_path is not None else None,
        Path(calibration_path) if calibration_path is not None else None,
        Path(temporal_dir) if temporal_dir is not None else None,
    )
