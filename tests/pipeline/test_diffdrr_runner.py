from __future__ import annotations

import torch

from ctdr_pipeline.registration.diffdrr_runner import (
    Registration3DRequest,
    run_diffdrr_registration,
)
from ctdr_pipeline.registration.initializer import PoseSeed


class GaussianRenderer:
    """Small differentiable renderer used to test optimizer control flow."""

    convention = "synthetic_xyz_degrees_translation_mm"

    def render(self, pose: torch.Tensor, height: int, width: int) -> torch.Tensor:
        y = torch.linspace(-1.0, 1.0, height, device=pose.device, dtype=pose.dtype)
        x = torch.linspace(-1.0, 1.0, width, device=pose.device, dtype=pose.dtype)
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        center_x = pose[:, 3, None, None] / 40.0
        center_y = pose[:, 5, None, None] / 40.0
        angle = torch.deg2rad(pose[:, 0, None, None])
        cosine, sine = torch.cos(angle), torch.sin(angle)
        shifted_x = xx - center_x
        shifted_y = yy - center_y
        rotated_x = cosine * shifted_x + sine * shifted_y
        rotated_y = -sine * shifted_x + cosine * shifted_y
        image = torch.exp(
            -(rotated_x.square() / 0.12 + rotated_y.square() / 0.035)
        )
        return image[:, None]


def _target(renderer: GaussianRenderer, pose: list[float], size: int = 64) -> torch.Tensor:
    return renderer.render(torch.tensor([pose], dtype=torch.float32), size, size).detach()


def test_multistart_runner_improves_score_and_excludes_diagnostics() -> None:
    renderer = GaussianRenderer()
    target_pose = [18.0, 0.0, 0.0, 12.0, 700.0, -8.0]
    target = _target(renderer, target_pose)
    request = Registration3DRequest(
        renderer=renderer,
        target=target,
        roi=torch.ones_like(target),
        seeds=[
            PoseSeed("physical_far", (0.0, 0.0, 0.0), (-18.0, 700.0, 18.0)),
            PoseSeed("physical_near", (8.0, 0.0, 0.0), (5.0, 700.0, -2.0)),
            PoseSeed(
                "diagnostic_exact",
                tuple(target_pose[:3]),
                tuple(target_pose[3:]),
                diagnostic_only=True,
            ),
        ],
        resolutions=(32, 64),
        coarse_iterations=35,
        refine_iterations=45,
        refine_top_k=2,
        rotation_learning_rate_deg=2.0,
        translation_learning_rate_mm=2.5,
        device="cpu",
    )

    result = run_diffdrr_registration(request)

    assert result.selected is not None
    assert not result.selected.diagnostic_only
    assert result.selected.final_score > result.selected.initial_score + 0.03
    assert result.selected.trace
    assert all(not flag for flag in result.selected.boundary_flags.values())
    assert any(candidate.diagnostic_only for candidate in result.candidates)
    assert result.metadata["pose_units"] == "degrees_mm"


def test_runner_clamps_pose_and_marks_boundary_solution() -> None:
    renderer = GaussianRenderer()
    target = _target(renderer, [0.0, 0.0, 0.0, 80.0, 700.0, 0.0], size=32)
    result = run_diffdrr_registration(
        Registration3DRequest(
            renderer=renderer,
            target=target,
            roi=torch.ones_like(target),
            seeds=[PoseSeed("outside", (500.0, 0.0, 0.0), (900.0, 2000.0, 0.0))],
            resolutions=(32,),
            coarse_iterations=3,
            refine_iterations=0,
            refine_top_k=1,
            device="cpu",
        )
    )

    candidate = result.candidates[0]
    assert -180.0 <= candidate.final_pose[0] <= 180.0
    assert -500.0 <= candidate.final_pose[3] <= 500.0
    assert 300.0 <= candidate.final_pose[4] <= 1500.0
    assert candidate.boundary_flags

