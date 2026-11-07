from pathlib import Path

import pytest

from ctdr_pipeline.bundle import load_bundle
from ctdr_pipeline.dicom.linker import link_case_images
from ctdr_pipeline.dicom.scanner import load_dicom_manifest
from ctdr_pipeline.dicom.selector import rank_series


@pytest.mark.medical
def test_first_case_links_to_confirmed_localizer_and_xa_frame() -> None:
    bundle_root = Path("artifacts/server-pipeline/bundles/标注-OKOK-pure")
    manifest_path = Path("artifacts/server-pipeline/manifests/local.generated.yaml")
    if not bundle_root.exists() or not manifest_path.exists():
        pytest.skip("local first-case artifacts are unavailable")
    bundle = load_bundle(bundle_root)
    bundle_case = next(case for case in bundle.cases if case.case_id == "1234567")
    manifest = load_dicom_manifest(manifest_path)
    case_manifest = manifest.case("1234567")
    selection = rank_series(case_manifest)
    series7 = next(
        series
        for series in case_manifest.series
        if series.modality == "XA" and series.series_number == "7"
    )

    result = link_case_images(
        bundle_root,
        bundle_case,
        Path(manifest.data_root),
        case_manifest,
        selection,
    )

    assert result.source.selected is not None
    assert result.source.selected.path.endswith("-0001-0001-0003-W1500L200.DCM")
    assert result.target.selected is not None
    assert result.target.selected.series_uid == series7.series_uid
    assert result.target.selected.frame_index == 0
