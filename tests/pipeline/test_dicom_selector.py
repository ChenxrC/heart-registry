from pathlib import Path

import pytest
import yaml

from ctdr_pipeline.dicom.scanner import scan_data_root
from ctdr_pipeline.dicom.selector import apply_dicom_overrides, rank_series
from tests.pipeline.test_dicom_scanner import build_case


def test_selector_finds_volume_localizer_and_xa(tmp_path: Path) -> None:
    volume_uid, localizer_uid, xa_uid = build_case(tmp_path)
    case = scan_data_root(tmp_path).case("1234567")

    selection = rank_series(case)

    assert selection.ct_series_uid == volume_uid
    assert selection.localizer_series_uid == localizer_uid
    assert selection.xa_series_uid == xa_uid
    assert selection.status == "ready"


def test_override_rejects_unknown_fields(tmp_path: Path) -> None:
    build_case(tmp_path)
    selection = rank_series(scan_data_root(tmp_path).case("1234567"))
    overrides = tmp_path / "overrides.yaml"
    overrides.write_text(
        yaml.safe_dump({"cases": [{"case_id": "1234567", "mystery": "x"}]}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="mystery"):
        apply_dicom_overrides(selection, overrides)


def test_override_can_select_an_existing_xa_frame(tmp_path: Path) -> None:
    _, _, xa_uid = build_case(tmp_path)
    selection = rank_series(scan_data_root(tmp_path).case("1234567"))
    overrides = tmp_path / "overrides.yaml"
    overrides.write_text(
        yaml.safe_dump(
            {
                "cases": [
                    {
                        "case_id": "1234567",
                        "xa_series_uid": xa_uid,
                        "xa_frame_index": 3,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    updated = apply_dicom_overrides(selection, overrides)

    assert updated.xa_series_uid == xa_uid
    assert updated.xa_frame_index == 3
