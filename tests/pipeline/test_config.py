from pathlib import Path

import pytest

from ctdr_pipeline.config import load_config


def test_config_rejects_unknown_keys(tmp_path: Path) -> None:
    path = tmp_path / "server.yaml"
    path.write_text("data_root: /data/cases\nunknown_key: true\n", encoding="utf-8")

    with pytest.raises(ValueError, match="unknown_key"):
        load_config(path)


def test_default_quality_thresholds(tmp_path: Path) -> None:
    path = tmp_path / "server.yaml"
    path.write_text("data_root: /data/cases\n", encoding="utf-8")

    config = load_config(path)

    assert config.quality.pptx_rms_diagonal == 0.02
    assert config.quality.drr_gradient_ncc == 0.35
    assert config.gpus == ["cuda:0"]
