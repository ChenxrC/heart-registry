from pathlib import Path

from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, generate_uid
from typer.testing import CliRunner

from ctdr_pipeline.cli import app
from ctdr_pipeline.dicom.scanner import load_dicom_manifest, scan_data_root


CT_SOP_CLASS = "1.2.840.10008.5.1.4.1.1.2"
XA_SOP_CLASS = "1.2.840.10008.5.1.4.1.1.12.1"


def write_dicom(
    path: Path,
    *,
    study_uid: str,
    series_uid: str,
    modality: str,
    sop_class_uid: str,
    instance_number: int,
    image_type: list[str],
    position: list[float] | None = None,
    frames: int = 1,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sop_instance_uid = generate_uid()
    meta = FileMetaDataset()
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    meta.MediaStorageSOPClassUID = sop_class_uid
    meta.MediaStorageSOPInstanceUID = sop_instance_uid
    meta.ImplementationClassUID = generate_uid()
    dataset = FileDataset(str(path), {}, file_meta=meta, preamble=b"\0" * 128)
    dataset.SOPClassUID = sop_class_uid
    dataset.SOPInstanceUID = sop_instance_uid
    dataset.StudyInstanceUID = study_uid
    dataset.SeriesInstanceUID = series_uid
    dataset.Modality = modality
    dataset.InstanceNumber = instance_number
    dataset.ImageType = image_type
    dataset.Rows = 64
    dataset.Columns = 80
    dataset.PixelSpacing = [1.0, 1.0]
    dataset.ImageOrientationPatient = [1, 0, 0, 0, 1, 0]
    if position is not None:
        dataset.ImagePositionPatient = position
    if frames > 1:
        dataset.NumberOfFrames = frames
    dataset.save_as(path, enforce_file_format=True)


def build_case(root: Path) -> tuple[str, str, str]:
    case = root / "1234567"
    study_uid = str(generate_uid())
    volume_uid = str(generate_uid())
    localizer_uid = str(generate_uid())
    xa_uid = str(generate_uid())
    for index in range(16):
        write_dicom(
            case / "CT1715346" / "DB1" / f"ct-{index:03}.dcm",
            study_uid=study_uid,
            series_uid=volume_uid,
            modality="CT",
            sop_class_uid=CT_SOP_CLASS,
            instance_number=index + 1,
            image_type=["ORIGINAL", "PRIMARY", "AXIAL"],
            position=[0.0, 0.0, float(index) * 2.0],
        )
    write_dicom(
        case / "CT1715346" / "DB1" / "localizer.dcm",
        study_uid=study_uid,
        series_uid=localizer_uid,
        modality="CT",
        sop_class_uid=CT_SOP_CLASS,
        instance_number=1,
        image_type=["ORIGINAL", "PRIMARY", "LOCALIZER"],
        position=[0.0, 0.0, 0.0],
    )
    write_dicom(
        case / "XA35397" / "DB2" / "cine.dcm",
        study_uid=study_uid,
        series_uid=xa_uid,
        modality="XA",
        sop_class_uid=XA_SOP_CLASS,
        instance_number=1,
        image_type=["ORIGINAL", "PRIMARY"],
        frames=7,
    )
    return volume_uid, localizer_uid, xa_uid


def test_scanner_uses_dicom_tags_and_groups_series(tmp_path: Path) -> None:
    volume_uid, localizer_uid, xa_uid = build_case(tmp_path)
    (tmp_path / "1234502没有ct").mkdir()

    manifest = scan_data_root(tmp_path)
    case = manifest.case("1234567")

    assert len(case.series) == 3
    assert {series.series_uid for series in case.series} == {
        volume_uid,
        localizer_uid,
        xa_uid,
    }
    assert next(series for series in case.series if series.series_uid == xa_uid).frame_count == 7
    assert manifest.case("1234502").series == []


def test_duplicate_numeric_prefix_is_reported_as_conflict(tmp_path: Path) -> None:
    (tmp_path / "1234501PRTct外院").mkdir()
    (tmp_path / "1234501本院有ct").mkdir()

    manifest = scan_data_root(tmp_path)
    case = manifest.case("1234501")

    assert case.status == "directory_conflict"
    assert len(case.candidate_directories) == 2


def test_scan_dicom_cli_writes_manifest(tmp_path: Path) -> None:
    build_case(tmp_path)
    output = tmp_path / "manifest.yaml"

    result = CliRunner().invoke(
        app,
        ["scan-dicom", "--data-root", str(tmp_path), "--output", str(output)],
    )

    assert result.exit_code == 0
    assert "cases=1" in result.stdout
    assert load_dicom_manifest(output).case("1234567").series
