from __future__ import annotations

import json
import re
import zipfile
from pathlib import Path

import numpy as np

from ctdr_pipeline.evaluation.metrics import CaseMetrics
from ctdr_pipeline.evaluation.quality import QualityDecision
from ctdr_pipeline.evaluation.report import CaseReport, write_case_report, write_run_summary
from ctdr_pipeline.schemas import CaseStatus


def _report(case_id: str, status: CaseStatus) -> CaseReport:
    return CaseReport(
        case_id=case_id,
        decision=QualityDecision(
            status=status,
            accepted=status == CaseStatus.ACCEPTED_RESEARCH_CANDIDATE,
            failed_rules=() if status == CaseStatus.ACCEPTED_RESEARCH_CANDIDATE else ("test_rule",),
        ),
        metrics=CaseMetrics(drr_roi_ncc=0.8, tre_reason="independent_3d_ground_truth_unavailable"),
        candidates=[
            {
                "name": "seed_0",
                "diagnostic_only": False,
                "final_score": 0.8,
                "final_pose": [1, 2, 3, 4, 700, 6],
            }
        ],
        metadata={
            "PatientName": "SECRET NAME",
            "PatientID": "SECRET-ID-123",
            "SeriesInstanceUID": "1.2.3.4",
            "safe": "kept",
        },
        diagnostic_images={"overlay": np.eye(16, dtype=np.float32)},
    )


def test_case_and_batch_reports_have_stable_files_and_remove_phi(tmp_path: Path) -> None:
    reports = [
        _report("1234567", CaseStatus.ACCEPTED_RESEARCH_CANDIDATE),
        _report("1234505", CaseStatus.REVIEW_REQUIRED_3D),
        _report("1234506", CaseStatus.SKIPPED_CT_MISSING),
    ]
    for report in reports:
        write_case_report(tmp_path / "cases" / report.case_id, report)
    write_run_summary(tmp_path, reports)

    assert (tmp_path / "summary.json").exists()
    assert (tmp_path / "summary.csv").exists()
    assert (tmp_path / "summary.xlsx").exists()
    assert (tmp_path / "environment.json").exists()
    case_dir = tmp_path / "cases" / "1234567"
    assert (case_dir / "status.json").exists()
    assert (case_dir / "registration3d" / "candidates.csv").exists()
    assert (case_dir / "evaluation" / "diagnostics" / "overlay.png").exists()
    summary = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert summary["status_counts"][CaseStatus.ACCEPTED_RESEARCH_CANDIDATE.value] == 1
    with zipfile.ZipFile(tmp_path / "summary.xlsx") as workbook:
        workbook_xml = workbook.read("xl/workbook.xml").decode("utf-8")
    sheet_names = set(re.findall(r'<sheet name="([^"]+)"', workbook_xml))
    assert sheet_names == {
        "run_summary",
        "errors_2d",
        "dicom_linkage",
        "registration_3d",
        "quality_reasons",
        "environment",
    }

    exported_text = "\n".join(
        path.read_text(encoding="utf-8", errors="ignore")
        for path in tmp_path.rglob("*")
        if path.is_file() and path.suffix.lower() in {".json", ".csv"}
    )
    assert "SECRET NAME" not in exported_text
    assert "SECRET-ID-123" not in exported_text
    assert "PatientName" not in exported_text
    assert "PatientID" not in exported_text
    assert "kept" in exported_text


def test_report_refuses_path_like_case_id(tmp_path: Path) -> None:
    report = _report("../escape", CaseStatus.REVIEW_REQUIRED_3D)
    try:
        write_case_report(tmp_path / "case", report)
    except ValueError as exc:
        assert "case_id" in str(exc)
    else:
        raise AssertionError("path-like case IDs must be rejected")
