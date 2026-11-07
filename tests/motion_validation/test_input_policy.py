"""Portable authority checks; Windows attributes are simulated, not real links."""

from pathlib import Path
import stat

import pytest

from ctdr_pipeline.motion_validation.input_policy import require_eligible_input


@pytest.mark.parametrize("at_ancestor", [False, True])
def test_reparse_attribute_excludes_path_without_path_is_junction(tmp_path: Path, monkeypatch, at_ancestor: bool) -> None:
    # Break caught: Python 3.11's absent Path.is_junction silently permits junctions.
    directory = tmp_path / "historical"
    directory.mkdir()
    path = directory / "input.bin"
    path.write_bytes(b"artificial")
    marked = directory if at_ancestor else path
    original = Path.lstat

    class ReparseStat:
        def __init__(self, value):
            self.value = value
            self.st_file_attributes = getattr(value, "st_file_attributes", 0) | getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)

        def __getattr__(self, key):
            return getattr(self.value, key)

    def lstat(candidate, *args, **kwargs):
        value = original(candidate, *args, **kwargs)
        return ReparseStat(value) if candidate == marked else value

    monkeypatch.setattr(Path, "lstat", lstat)
    with pytest.raises(ValueError, match="[Ee]xcluded|authoritative"):
        require_eligible_input(path)
