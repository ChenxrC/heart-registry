from pathlib import Path

import numpy as np
import pytest
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, generate_uid

from ctdr_pipeline.dicom.scanner import scan_data_root
from ctdr_pipeline.registration.volume import CtGeometryError, load_ct_volume


CT_SOP_CLASS = "1.2.840.10008.5.1.4.1.1.2"


def write_ct_slice(
    path: Path,
    *,
    study_uid: str,
    series_uid: str,
    z: float,
    value: int,
    orientation: list[float] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sop_uid = str(generate_uid())
    meta = FileMetaDataset()
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    meta.MediaStorageSOPClassUID = CT_SOP_CLASS
    meta.MediaStorageSOPInstanceUID = sop_uid
    meta.ImplementationClassUID = str(generate_uid())
    dataset = FileDataset(str(path), {}, file_meta=meta, preamble=b"\0" * 128)
    dataset.SOPClassUID = CT_SOP_CLASS
    dataset.SOPInstanceUID = sop_uid
    dataset.StudyInstanceUID = study_uid
    dataset.SeriesInstanceUID = series_uid
    dataset.Modality = "CT"
    dataset.ImageType = ["ORIGINAL", "PRIMARY", "AXIAL"]
    dataset.Rows = 4
    dataset.Columns = 5
    dataset.PixelSpacing = [1.0, 0.5]
    dataset.ImageOrientationPatient = orientation or [1, 0, 0, 0, 1, 0]
    dataset.ImagePositionPatient = [10.0, 20.0, z]
    dataset.RescaleSlope = 2.0
    dataset.RescaleIntercept = -1000.0
    dataset.SamplesPerPixel = 1
    dataset.PhotometricInterpretation = "MONOCHROME2"
    dataset.BitsAllocated = 16
    dataset.BitsStored = 16
    dataset.HighBit = 15
    dataset.PixelRepresentation = 1
    pixels = np.full((4, 5), value, dtype=np.int16)
    dataset.PixelData = pixels.tobytes()
    dataset.save_as(path, enforce_file_format=True)


def build_volume_case(root: Path, positions: list[float]) -> str:
    study_uid = str(generate_uid())
    series_uid = str(generate_uid())
    for index, z in enumerate(positions):
        write_ct_slice(
            root / "1234567" / "CT" / f"slice-{index}.dcm",
            study_uid=study_uid,
            series_uid=series_uid,
            z=z,
            value=100 + index,
        )
    return series_uid


def test_load_ct_volume_sorts_slices_builds_lps_affine_and_applies_hu(tmp_path: Path) -> None:
    series_uid = build_volume_case(tmp_path, [4.0, 0.0, 2.0, 6.0])
    series = scan_data_root(tmp_path).case("1234567").series_by_uid(series_uid)

    volume = load_ct_volume(series, tmp_path)

    assert volume.data_hu.shape == (4, 4, 5)
    assert volume.spacing_xyz == pytest.approx((0.5, 1.0, 2.0))
    assert np.allclose(volume.voxel_to_lps[:3, 3], [10.0, 20.0, 0.0])
    assert volume.data_hu[0, 0, 0] == -798.0
    assert volume.data_hu[-1, 0, 0] == -794.0


def test_duplicate_slice_positions_are_rejected(tmp_path: Path) -> None:
    series_uid = build_volume_case(tmp_path, [0.0, 2.0, 2.0, 4.0])
    series = scan_data_root(tmp_path).case("1234567").series_by_uid(series_uid)

    with pytest.raises(CtGeometryError, match="duplicate_slice_positions"):
        load_ct_volume(series, tmp_path)


def test_inconsistent_orientation_is_rejected(tmp_path: Path) -> None:
    study_uid = str(generate_uid())
    series_uid = str(generate_uid())
    write_ct_slice(
        tmp_path / "1234567" / "CT" / "a.dcm",
        study_uid=study_uid,
        series_uid=series_uid,
        z=0.0,
        value=100,
    )
    write_ct_slice(
        tmp_path / "1234567" / "CT" / "b.dcm",
        study_uid=study_uid,
        series_uid=series_uid,
        z=2.0,
        value=101,
        orientation=[0, 1, 0, 1, 0, 0],
    )
    series = scan_data_root(tmp_path).case("1234567").series_by_uid(series_uid)

    with pytest.raises(CtGeometryError, match="inconsistent_orientation"):
        load_ct_volume(series, tmp_path)
