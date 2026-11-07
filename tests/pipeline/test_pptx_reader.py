from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

from ctdr_pipeline.pptx.reader import read_pptx


PRESENTATION_XML = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<p:presentation xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main"
 xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"
 xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main">
  <p:sldIdLst><p:sldId id="256" r:id="rId1"/><p:sldId id="257" r:id="rId2"/></p:sldIdLst>
</p:presentation>"""

PRESENTATION_RELS = """<?xml version="1.0" encoding="UTF-8"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
 <Relationship Id="rId1" Type="x" Target="slides/slide1.xml"/>
 <Relationship Id="rId2" Type="x" Target="slides/slide2.xml"/>
</Relationships>"""


def slide_xml(text: str, rel_id: str) -> str:
    return f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<p:sld xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main"
 xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"
 xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main">
 <p:cSld><p:spTree>
  <p:sp><p:txBody><a:p><a:r><a:t>{text}</a:t></a:r></a:p></p:txBody></p:sp>
  <p:pic><p:blipFill><a:blip r:embed="{rel_id}"/></p:blipFill>
   <p:spPr><a:xfrm><a:off x="100" y="200"/><a:ext cx="300" cy="400"/></a:xfrm></p:spPr>
  </p:pic>
 </p:spTree></p:cSld>
</p:sld>"""


def slide_rels(image_name: str) -> str:
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
 <Relationship Id="rIdImage" Type="image" Target="../media/{image_name}"/>
</Relationships>"""


def write_minimal_pptx(path: Path) -> None:
    with ZipFile(path, "w", ZIP_DEFLATED) as archive:
        archive.writestr("ppt/presentation.xml", PRESENTATION_XML)
        archive.writestr("ppt/_rels/presentation.xml.rels", PRESENTATION_RELS)
        archive.writestr("ppt/slides/slide1.xml", slide_xml("90000001", "rIdImage"))
        archive.writestr("ppt/slides/slide2.xml", slide_xml("", "rIdImage"))
        archive.writestr("ppt/slides/_rels/slide1.xml.rels", slide_rels("image1.png"))
        archive.writestr("ppt/slides/_rels/slide2.xml.rels", slide_rels("image2.png"))
        archive.writestr("ppt/media/image1.png", b"one")
        archive.writestr("ppt/media/image2.png", b"two")


def test_read_pptx_preserves_slide_order_text_images_and_geometry(tmp_path: Path) -> None:
    path = tmp_path / "sample.pptx"
    write_minimal_pptx(path)

    inventory = read_pptx(path)

    assert [slide.number for slide in inventory.slides] == [1, 2]
    assert inventory.slides[0].case_ids == ["90000001"]
    assert inventory.slides[0].image_members == ["ppt/media/image1.png"]
    assert inventory.slides[0].image_geometry[0].offset_x == 100
    assert inventory.slides[0].image_geometry[0].extent_y == 400
    assert inventory.source_sha256
