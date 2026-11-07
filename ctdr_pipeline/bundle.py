from __future__ import annotations

import hashlib
import json
from pathlib import Path

from pydantic import Field

from ctdr_pipeline.schemas import StrictModel


class BundleCaseRecord(StrictModel):
    case_id: str
    slide_numbers: list[int]
    image_count: int
    pair_status: str
    registration_status: str
    source_image: str | None = None
    target_image: str | None = None
    matrix_3x3: list[float] | None = None
    reasons: list[str] = Field(default_factory=list)


class BundleIndex(StrictModel):
    schema_version: str = "1.0"
    source_pptx: str
    source_sha256: str
    total_cases: int
    complete_pairs: int
    incomplete_pairs: int
    cases: list[BundleCaseRecord]
    checksums: dict[str, str]


class BundleValidation(StrictModel):
    valid: bool
    missing_files: list[str] = Field(default_factory=list)
    changed_files: list[str] = Field(default_factory=list)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_bundle(path: Path) -> BundleIndex:
    raw = json.loads((path / "bundle.json").read_text(encoding="utf-8"))
    return BundleIndex.model_validate(raw)


def validate_bundle(path: Path) -> BundleValidation:
    index = load_bundle(path)
    missing: list[str] = []
    changed: list[str] = []
    for relative, expected in index.checksums.items():
        artifact = path / relative
        if not artifact.exists():
            missing.append(relative)
        elif sha256_file(artifact) != expected:
            changed.append(relative)
    return BundleValidation(
        valid=not missing and not changed,
        missing_files=sorted(missing),
        changed_files=sorted(changed),
    )
