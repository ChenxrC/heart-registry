from __future__ import annotations

import pytest
import torch

from ctdr_pipeline.registration.diffdrr_runner import (
    Registration3DRequest,
    run_diffdrr_registration,
)
from ctdr_pipeline.registration.initializer import PoseSeed
from tests.pipeline.test_diffdrr_runner import GaussianRenderer


@pytest.mark.slow
def test_known_synthetic_pose_is_recovered() -> None:
    renderer = GaussianRenderer()
    truth = torch.tensor([22.0, 0.0, 0.0, 14.0, 700.0, -10.0])
    target = renderer.render(truth[None], 96, 96).detach()
    result = run_diffdrr_registration(
        Registration3DRequest(
            renderer=renderer,
            target=target,
            roi=torch.ones_like(target),
            seeds=[PoseSeed("seed", (10.0, 0.0, 0.0), (5.0, 700.0, -2.0))],
            resolutions=(48, 96),
            coarse_iterations=60,
            refine_iterations=100,
            refine_top_k=1,
            rotation_learning_rate_deg=2.0,
            translation_learning_rate_mm=2.0,
            device="cpu",
        )
    )
    assert result.selected is not None
    pose = torch.tensor(result.selected.final_pose)
    assert abs(float(pose[0] - truth[0])) <= 5.0
    assert torch.linalg.vector_norm(pose[3:] - truth[3:]) <= 10.0
    assert result.selected.final_score - result.selected.initial_score >= 0.03
