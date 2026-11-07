from __future__ import annotations

import inspect

import heart_registry


def test_public_api_exports_path_explicit_workflows() -> None:
    expected = {
        "prepare_annotation_bundle",
        "scan_dicom_tree",
        "register_case",
        "register_cases",
        "extract_xa_cine",
        "segment_xa_vessels",
        "fuse_xa_temporal_masks",
        "build_xa_dynamic_vessel_mesh",
    }
    assert expected <= set(heart_registry.__all__)


def test_registration_api_requires_explicit_inputs() -> None:
    parameters = inspect.signature(heart_registry.register_case).parameters
    assert {"bundle_dir", "dicom_manifest", "output_dir", "config_path"} <= set(parameters)
    assert parameters["case_id"].default is inspect.Parameter.empty


def test_segmentation_api_requires_external_model_assets() -> None:
    parameters = inspect.signature(heart_registry.segment_xa_vessels).parameters
    assert "model_code_dir" in parameters
    assert "checkpoint_path" in parameters
