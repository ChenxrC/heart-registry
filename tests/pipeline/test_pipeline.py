from __future__ import annotations

from pathlib import Path

from ctdr_pipeline.config import PipelineConfig
from ctdr_pipeline.pipeline import (
    BatchRunRequest,
    CaseRunRequest,
    CaseTerminalError,
    run_all,
    run_case,
)
from ctdr_pipeline.schemas import CaseStatus


def _inputs(tmp_path: Path) -> tuple[Path, Path, PipelineConfig]:
    bundle = tmp_path / "bundle.json"
    manifest = tmp_path / "manifest.yaml"
    bundle.write_text('{"test": true}', encoding="utf-8")
    manifest.write_text("cases: []\n", encoding="utf-8")
    config = PipelineConfig(data_root=tmp_path, output_root=tmp_path / "runs", gpus=["cpu"])
    return bundle, manifest, config


def test_partial_failure_does_not_stop_other_cases(tmp_path: Path) -> None:
    bundle, manifest, config = _inputs(tmp_path)
    attempted: list[str] = []

    def pptx(context):
        attempted.append(context.request.case_id)
        if context.request.case_id == "1234505":
            raise CaseTerminalError(CaseStatus.SKIPPED_CT_MISSING, "ct missing")
        return {"ready": True}

    def linkage(context):
        if context.request.case_id == "1234506":
            raise RuntimeError("synthetic CUDA failure")
        return {"linked": True}

    handlers = {
        "pptx": pptx,
        "dicom_linkage": linkage,
        "registration3d": lambda context: {"score": 0.8},
        "evaluation": lambda context: {
            "status": CaseStatus.ACCEPTED_RESEARCH_CANDIDATE.value,
            "metrics": {},
        },
        "report": lambda context: {"written": True},
    }
    request = BatchRunRequest(
        case_ids=("1234567", "1234505", "1234506"),
        bundle_path=bundle,
        dicom_manifest_path=manifest,
        output_dir=tmp_path / "batch",
        config=config,
        handlers=handlers,
    )
    summary = run_all(request)

    assert sorted(attempted) == ["1234505", "1234506", "1234567"]
    assert summary.exit_code == 0
    assert {report.decision.status for report in summary.reports} == {
        CaseStatus.ACCEPTED_RESEARCH_CANDIDATE,
        CaseStatus.SKIPPED_CT_MISSING,
        CaseStatus.FAILED_RUNTIME,
    }
    for case_id in request.case_ids:
        assert (request.output_dir / "cases" / case_id / "status.json").exists()


def test_fail_on_review_changes_only_batch_exit_code(tmp_path: Path) -> None:
    bundle, manifest, config = _inputs(tmp_path)
    handlers = {
        stage: (lambda context: {"status": CaseStatus.REVIEW_REQUIRED_3D.value})
        for stage in ("pptx", "dicom_linkage", "registration3d", "evaluation", "report")
    }
    summary = run_all(
        BatchRunRequest(
            case_ids=("1234567",),
            bundle_path=bundle,
            dicom_manifest_path=manifest,
            output_dir=tmp_path / "strict",
            config=config.model_copy(update={"fail_on_review": True}),
            handlers=handlers,
        )
    )
    assert summary.exit_code == 3


def test_single_case_returns_terminal_report(tmp_path: Path) -> None:
    bundle, manifest, config = _inputs(tmp_path)
    report = run_case(
        CaseRunRequest(
            case_id="1234567",
            bundle_path=bundle,
            dicom_manifest_path=manifest,
            output_dir=tmp_path / "single",
            config=config,
            handlers={
                stage: (lambda context: {"status": CaseStatus.REVIEW_REQUIRED_3D.value})
                for stage in ("pptx", "dicom_linkage", "registration3d", "evaluation", "report")
            },
        )
    )
    assert report.case_id == "1234567"
    assert report.decision.status == CaseStatus.REVIEW_REQUIRED_3D

