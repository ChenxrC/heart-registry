from __future__ import annotations

from pathlib import Path

import typer

from ctdr_pipeline import __version__
from ctdr_pipeline.bundle import load_bundle, validate_bundle
from ctdr_pipeline.config import load_config
from ctdr_pipeline.dicom.scanner import (
    load_dicom_manifest,
    scan_data_root,
    write_dicom_manifest,
)
from ctdr_pipeline.evaluation.metrics import CaseMetrics
from ctdr_pipeline.evaluation.quality import QualityDecision
from ctdr_pipeline.evaluation.report import CaseReport, write_run_summary
from ctdr_pipeline.pipeline import BatchRunRequest, CaseRunRequest, run_all, run_case
from ctdr_pipeline.pptx.export import prepare_pptx
from ctdr_pipeline.schemas import CaseStatus

app = typer.Typer(
    name="ctdr",
    help="Auditable PPTX-to-CT/DR registration pipeline.",
    no_args_is_help=True,
)


@app.callback()
def main() -> None:
    """Run the CT-DR pipeline command group."""


@app.command()
def version() -> None:
    """Print the installed pipeline version."""

    typer.echo(f"ctdr-pipeline {__version__}")


@app.command("prepare-pptx")
def prepare_pptx_command(
    pptx: Path = typer.Option(..., exists=True, dir_okay=False, readable=True),
    output: Path = typer.Option(..., file_okay=False),
    overrides: Path | None = typer.Option(None, exists=True, dir_okay=False),
) -> None:
    """Create a portable annotation bundle from a physician PPTX."""

    index = prepare_pptx(pptx, output, overrides)
    typer.echo(
        f"total={index.total_cases} complete={index.complete_pairs} "
        f"incomplete={index.incomplete_pairs} output={output}"
    )


@app.command("validate-bundle")
def validate_bundle_command(
    bundle: Path = typer.Option(..., exists=True, file_okay=False, readable=True),
) -> None:
    """Verify every authoritative file in a portable bundle."""

    result = validate_bundle(bundle)
    if result.valid:
        typer.echo(f"valid bundle={bundle}")
        return
    typer.echo(
        f"invalid bundle={bundle} missing={','.join(result.missing_files)} "
        f"changed={','.join(result.changed_files)}",
        err=True,
    )
    raise typer.Exit(code=2)


@app.command("scan-dicom")
def scan_dicom_command(
    data_root: Path = typer.Option(..., exists=True, file_okay=False, readable=True),
    output: Path = typer.Option(..., dir_okay=False),
) -> None:
    """Scan first-level case directories and write a DICOM metadata manifest."""

    manifest = scan_data_root(data_root)
    write_dicom_manifest(manifest, output)
    conflicts = sum(case.status == "directory_conflict" for case in manifest.cases)
    typer.echo(
        f"cases={len(manifest.cases)} conflicts={conflicts} output={output}"
    )


@app.command("validate")
def validate_command(
    bundle: Path = typer.Option(..., exists=True, file_okay=False, readable=True),
    dicom_manifest: Path = typer.Option(..., exists=True, dir_okay=False, readable=True),
) -> None:
    """Validate a portable bundle and DICOM manifest without running registration."""

    result = validate_bundle(bundle)
    if not result.valid:
        typer.echo(
            f"invalid bundle missing={','.join(result.missing_files)} "
            f"changed={','.join(result.changed_files)}",
            err=True,
        )
        raise typer.Exit(code=2)
    try:
        manifest = load_dicom_manifest(dicom_manifest)
    except Exception as exc:
        typer.echo(f"invalid DICOM manifest: {exc}", err=True)
        raise typer.Exit(code=2) from exc
    typer.echo(f"valid bundle_cases={load_bundle(bundle).total_cases} manifest_cases={len(manifest.cases)}")


@app.command("run-case")
def run_case_command(
    case_id: str = typer.Option(...),
    bundle: Path = typer.Option(..., exists=True, file_okay=False, readable=True),
    dicom_manifest: Path = typer.Option(..., exists=True, dir_okay=False, readable=True),
    output: Path = typer.Option(..., file_okay=False),
    config: Path = typer.Option(Path("configs/server.example.yaml"), exists=True, dir_okay=False),
    device: str | None = typer.Option(None),
) -> None:
    """Run one case to a terminal accepted, review, rejected, skipped, or failed status."""

    settings = load_config(config)
    report = run_case(
        CaseRunRequest(
            case_id=case_id,
            bundle_path=bundle,
            dicom_manifest_path=dicom_manifest,
            output_dir=output,
            config=settings,
            device=device,
        )
    )
    typer.echo(f"case_id={case_id} status={report.decision.status.value} output={output}")


@app.command("run-all")
def run_all_command(
    bundle: Path = typer.Option(..., exists=True, file_okay=False, readable=True),
    dicom_manifest: Path = typer.Option(..., exists=True, dir_okay=False, readable=True),
    config: Path = typer.Option(Path("configs/server.example.yaml"), exists=True, dir_okay=False),
    output: Path | None = typer.Option(None, file_okay=False),
    fail_on_review: bool | None = typer.Option(None),
) -> None:
    """Run every PPTX case with resumable per-case stages and bounded GPU workers."""

    settings = load_config(config)
    if fail_on_review is not None:
        settings = settings.model_copy(update={"fail_on_review": fail_on_review})
    index = load_bundle(bundle)
    run_dir = output or settings.output_root
    summary = run_all(
        BatchRunRequest(
            case_ids=tuple(record.case_id for record in index.cases),
            bundle_path=bundle,
            dicom_manifest_path=dicom_manifest,
            output_dir=run_dir,
            config=settings,
        )
    )
    accepted = sum(report.decision.accepted for report in summary.reports)
    typer.echo(f"cases={len(summary.reports)} accepted={accepted} output={run_dir}")
    if summary.exit_code:
        raise typer.Exit(code=summary.exit_code)


@app.command("report")
def report_command(
    run_dir: Path = typer.Option(..., exists=True, file_okay=False, readable=True),
) -> None:
    """Rebuild batch summaries from existing per-case report JSON files."""

    reports: list[CaseReport] = []
    for path in sorted((run_dir / "cases").glob("*/report.json")):
        import json

        raw = json.loads(path.read_text(encoding="utf-8"))
        status_raw = raw["status"]
        status = CaseStatus(status_raw["status"])
        reports.append(
            CaseReport(
                case_id=str(status_raw["case_id"]),
                decision=QualityDecision(
                    status=status,
                    accepted=bool(status_raw["accepted"]),
                    failed_rules=tuple(status_raw.get("failed_rules", [])),
                ),
                metrics=CaseMetrics(**raw.get("metrics", {})),
                candidates=list(raw.get("candidates", [])),
                metadata=dict(raw.get("metadata", {})),
            )
        )
    write_run_summary(run_dir, reports)
    typer.echo(f"rebuilt cases={len(reports)} output={run_dir}")


if __name__ == "__main__":
    app()
