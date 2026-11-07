from __future__ import annotations

from collections import Counter
from pathlib import Path

from ctdr_pipeline.config import PipelineConfig
from ctdr_pipeline.pipeline import CaseRunRequest, run_case
from ctdr_pipeline.schemas import CaseStatus


def test_resume_uses_stage_hashes_and_invalidates_only_downstream(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle.json"
    manifest = tmp_path / "manifest.yaml"
    overrides = tmp_path / "overrides.yaml"
    bundle.write_text('{"bundle": 1}', encoding="utf-8")
    manifest.write_text("cases: []\n", encoding="utf-8")
    overrides.write_text("cases: []\n", encoding="utf-8")
    counts: Counter[str] = Counter()

    def handler(stage: str):
        def run(context):
            counts[stage] += 1
            payload = {"stage": stage, "count": counts[stage]}
            if stage == "evaluation":
                payload["status"] = CaseStatus.ACCEPTED_RESEARCH_CANDIDATE.value
            return payload

        return run

    handlers = {stage: handler(stage) for stage in ("pptx", "dicom_linkage", "registration3d", "evaluation", "report")}
    request = CaseRunRequest(
        case_id="1234567",
        bundle_path=bundle,
        dicom_manifest_path=manifest,
        output_dir=tmp_path / "run",
        config=PipelineConfig(
            data_root=tmp_path,
            output_root=tmp_path / "runs",
            gpus=["cpu"],
            dicom_overrides=overrides,
        ),
        handlers=handlers,
    )

    run_case(request)
    first_counts = counts.copy()
    run_case(request)
    assert counts == first_counts

    overrides.write_text("cases:\n  - case_id: '1234567'\n    xa_frame_index: 1\n", encoding="utf-8")
    run_case(request)
    assert counts["pptx"] == 1
    assert counts["dicom_linkage"] == 2
    assert counts["registration3d"] == 2
    assert counts["evaluation"] == 2
    assert counts["report"] == 2

    for stage in handlers:
        assert (request.output_dir / "stages" / stage / "stage.json").exists()
