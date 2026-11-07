from __future__ import annotations

import hashlib
import io
import posixpath
import re
from dataclasses import dataclass, field
from pathlib import Path
from xml.etree import ElementTree as ET
from zipfile import ZipFile


NS = {
    "a": "http://schemas.openxmlformats.org/drawingml/2006/main",
    "p": "http://schemas.openxmlformats.org/presentationml/2006/main",
    "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
    "rel": "http://schemas.openxmlformats.org/package/2006/relationships",
}
CASE_ID_PATTERN = re.compile(r"(?<!\d)\d{8}(?!\d)")


@dataclass(frozen=True)
class ImageGeometry:
    offset_x: int = 0
    offset_y: int = 0
    extent_x: int = 0
    extent_y: int = 0
    crop_left: float = 0.0
    crop_top: float = 0.0
    crop_right: float = 0.0
    crop_bottom: float = 0.0


@dataclass(frozen=True)
class SlideRecord:
    number: int
    texts: list[str]
    image_refs: list[str]
    image_members: list[str] = field(default_factory=list)
    image_geometry: list[ImageGeometry] = field(default_factory=list)
    image_sha256: list[str] = field(default_factory=list)
    slide_member: str = ""

    @property
    def case_ids(self) -> list[str]:
        return list(dict.fromkeys(CASE_ID_PATTERN.findall(" ".join(self.texts))))

    @property
    def image_count(self) -> int:
        return len(self.image_members) if self.image_members else len(self.image_refs)


@dataclass(frozen=True)
class PptxInventory:
    path: str
    source_sha256: str
    slides: list[SlideRecord]
    media_count: int


def _normalize_member(base_member: str, target: str) -> str:
    return posixpath.normpath(posixpath.join(posixpath.dirname(base_member), target))


def _relationship_map(archive: ZipFile, member: str) -> dict[str, str]:
    if member not in archive.namelist():
        return {}
    root = ET.fromstring(archive.read(member))
    return {
        node.attrib["Id"]: node.attrib["Target"]
        for node in root.findall("rel:Relationship", NS)
    }


def _integer_attribute(node: ET.Element | None, name: str) -> int:
    return int(node.attrib.get(name, "0")) if node is not None else 0


def _crop_attribute(node: ET.Element | None, name: str) -> float:
    return float(node.attrib.get(name, "0")) / 100000.0 if node is not None else 0.0


def _picture_geometry(picture: ET.Element) -> ImageGeometry:
    transform = picture.find(".//a:xfrm", NS)
    offset = transform.find("a:off", NS) if transform is not None else None
    extent = transform.find("a:ext", NS) if transform is not None else None
    crop = picture.find(".//a:srcRect", NS)
    return ImageGeometry(
        offset_x=_integer_attribute(offset, "x"),
        offset_y=_integer_attribute(offset, "y"),
        extent_x=_integer_attribute(extent, "cx"),
        extent_y=_integer_attribute(extent, "cy"),
        crop_left=_crop_attribute(crop, "l"),
        crop_top=_crop_attribute(crop, "t"),
        crop_right=_crop_attribute(crop, "r"),
        crop_bottom=_crop_attribute(crop, "b"),
    )


def read_pptx(path: Path) -> PptxInventory:
    source = path.read_bytes()
    source_sha256 = hashlib.sha256(source).hexdigest()
    with ZipFile(io.BytesIO(source)) as archive:
        presentation = ET.fromstring(archive.read("ppt/presentation.xml"))
        presentation_rels = _relationship_map(
            archive, "ppt/_rels/presentation.xml.rels"
        )
        slide_members: list[str] = []
        for slide_id in presentation.findall(".//p:sldId", NS):
            relationship_id = slide_id.attrib[f"{{{NS['r']}}}id"]
            target = presentation_rels[relationship_id]
            slide_members.append(_normalize_member("ppt/presentation.xml", target))

        slides: list[SlideRecord] = []
        names = set(archive.namelist())
        for number, member in enumerate(slide_members, start=1):
            root = ET.fromstring(archive.read(member))
            texts = [node.text for node in root.findall(".//a:t", NS) if node.text]
            relationships_member = posixpath.join(
                posixpath.dirname(member),
                "_rels",
                f"{posixpath.basename(member)}.rels",
            )
            relationships = _relationship_map(archive, relationships_member)
            refs: list[str] = []
            members: list[str] = []
            geometries: list[ImageGeometry] = []
            hashes: list[str] = []
            for picture in root.findall(".//p:pic", NS):
                blip = picture.find(".//a:blip", NS)
                if blip is None:
                    continue
                relationship_id = blip.attrib.get(f"{{{NS['r']}}}embed", "")
                refs.append(relationship_id)
                geometries.append(_picture_geometry(picture))
                target = relationships.get(relationship_id)
                if target is None:
                    continue
                image_member = _normalize_member(member, target)
                if image_member not in names:
                    continue
                members.append(image_member)
                hashes.append(hashlib.sha256(archive.read(image_member)).hexdigest())
            slides.append(
                SlideRecord(
                    number=number,
                    texts=texts,
                    image_refs=refs,
                    image_members=members,
                    image_geometry=geometries,
                    image_sha256=hashes,
                    slide_member=member,
                )
            )

        media_count = sum(name.startswith("ppt/media/") for name in names)
    return PptxInventory(
        path=str(path),
        source_sha256=source_sha256,
        slides=slides,
        media_count=media_count,
    )
