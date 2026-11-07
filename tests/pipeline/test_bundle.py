from pathlib import Path
from zipfile import ZipFile

import numpy as np
from PIL import Image
from pptx import Presentation
from pptx.util import Inches
from typer.testing import CliRunner

from ctdr_pipeline.bundle import load_bundle, validate_bundle
from ctdr_pipeline.cli import app
from ctdr_pipeline.pptx.export import prepare_pptx


def annotation_image(points: list[tuple[int, int]]) -> np.ndarray:
    image = np.full((160, 160, 3), 120, dtype=np.uint8)
    colors = [(255, 255, 0), (0, 255, 0), (0, 100, 255), (255, 180, 0)]
    for (x, y), color in zip(points, colors, strict=True):
        image[y - 4 : y + 5, x - 4 : x + 5] = color
    return image


def add_slide(
    presentation: Presentation,
    image_path: Path,
    case_id: str | None,
) -> None:
    slide = presentation.slides.add_slide(presentation.slide_layouts[6])
    if case_id:
        box = slide.shapes.add_textbox(Inches(0.5), Inches(0.2), Inches(3), Inches(0.5))
        box.text = case_id
    slide.shapes.add_picture(str(image_path), Inches(1), Inches(1), width=Inches(4))


def make_pptx(path: Path, image_dir: Path) -> None:
    source = annotation_image([(20, 20), (120, 25), (25, 120), (120, 120)])
    target = annotation_image([(30, 35), (130, 40), (35, 135), (130, 135)])
    source_path = image_dir / "source.png"
    target_path = image_dir / "target.png"
    Image.fromarray(source).save(source_path)
    Image.fromarray(target).save(target_path)
    presentation = Presentation()
    presentation.slides._sldIdLst.clear()
    add_slide(presentation, source_path, "90000001")
    add_slide(presentation, target_path, None)
    add_slide(presentation, source_path, "90000004")
    presentation.save(path)


def workbook_sheet_names(path: Path) -> list[str]:
    with ZipFile(path) as archive:
        xml = archive.read("xl/workbook.xml").decode("utf-8")
    expected = [
        "README",
        "pair_index",
        "landmarks",
        "matrices",
        "validation",
        "dicom_linkage",
        "skipped_cases",
    ]
    return [name for name in expected if f'name="{name}"' in xml]


def test_bundle_round_trip_and_workbook_sheets(tmp_path: Path) -> None:
    pptx = tmp_path / "annotations.pptx"
    make_pptx(pptx, tmp_path)
    output = tmp_path / "bundle"

    index = prepare_pptx(pptx, output)
    loaded = load_bundle(output)
    validation = validate_bundle(output)

    assert index.total_cases == 2
    assert index.complete_pairs == 1
    assert index.incomplete_pairs == 1
    assert loaded.model_dump() == index.model_dump()
    assert validation.valid
    assert workbook_sheet_names(output / "landmark_transforms.xlsx") == [
        "README",
        "pair_index",
        "landmarks",
        "matrices",
        "validation",
        "dicom_linkage",
        "skipped_cases",
    ]
    assert (output / "cases" / "90000001" / "source.png").exists()
    assert (output / "cases" / "90000001" / "pair.json").exists()


def test_checksum_validation_detects_machine_artifact_change(tmp_path: Path) -> None:
    pptx = tmp_path / "annotations.pptx"
    make_pptx(pptx, tmp_path)
    output = tmp_path / "bundle"
    prepare_pptx(pptx, output)
    pairs = output / "pairs.csv"
    pairs.write_text(pairs.read_text(encoding="utf-8") + "tampered", encoding="utf-8")

    validation = validate_bundle(output)

    assert not validation.valid
    assert "pairs.csv" in validation.changed_files


def test_prepare_and_validate_bundle_cli(tmp_path: Path) -> None:
    pptx = tmp_path / "annotations.pptx"
    make_pptx(pptx, tmp_path)
    output = tmp_path / "bundle"
    runner = CliRunner()

    prepared = runner.invoke(
        app,
        ["prepare-pptx", "--pptx", str(pptx), "--output", str(output)],
    )
    validated = runner.invoke(
        app,
        ["validate-bundle", "--bundle", str(output)],
    )

    assert prepared.exit_code == 0
    assert "total=2" in prepared.stdout
    assert validated.exit_code == 0
    assert "valid" in validated.stdout
