from typer.testing import CliRunner

from ctdr_pipeline.cli import app


def test_cli_version() -> None:
    result = CliRunner().invoke(app, ["version"])

    assert result.exit_code == 0
    assert "ctdr-pipeline" in result.stdout
