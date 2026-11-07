from __future__ import annotations

import math
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import torch
import torch.nn.functional as functional

from ctdr_pipeline.dicom.geometry import ProjectionGeometry
from ctdr_pipeline.registration.initializer import PoseSeed
from ctdr_pipeline.registration.objectives import ObjectiveBreakdown, registration_objective
from ctdr_pipeline.registration.volume import CtVolume


@runtime_checkable
class DrrRenderer(Protocol):
    """Differentiable renderer accepting degrees/mm at the pipeline boundary."""

    convention: str

    def render(self, pose: torch.Tensor, height: int, width: int) -> torch.Tensor:
        """Render BCHW images from a Bx6 [rotation_deg, translation_mm] pose."""


@dataclass(frozen=True)
class Registration3DRequest:
    renderer: DrrRenderer
    target: torch.Tensor
    roi: torch.Tensor
    seeds: list[PoseSeed]
    resolutions: tuple[int, ...] = (256, 512)
    coarse_iterations: int = 30
    refine_iterations: int = 80
    refine_top_k: int = 3
    rotation_learning_rate_deg: float = 1.5
    translation_learning_rate_mm: float = 3.0
    rotation_bounds_deg: tuple[tuple[float, float], ...] = (
        (-180.0, 180.0),
        (-180.0, 180.0),
        (-180.0, 180.0),
    )
    translation_bounds_mm: tuple[tuple[float, float], ...] = (
        (-500.0, 500.0),
        (300.0, 1500.0),
        (-500.0, 500.0),
    )
    device: str = "cpu"

    def __post_init__(self) -> None:
        if not self.seeds:
            raise ValueError("at least one pose seed is required")
        if not self.resolutions or any(value < 16 for value in self.resolutions):
            raise ValueError("resolutions must contain values >= 16")
        if self.coarse_iterations < 0 or self.refine_iterations < 0:
            raise ValueError("iteration counts cannot be negative")
        if self.refine_top_k < 1:
            raise ValueError("refine_top_k must be at least one")


@dataclass
class Candidate3DResult:
    name: str
    diagnostic_only: bool
    provenance: str
    initial_pose: tuple[float, ...]
    final_pose: tuple[float, ...]
    initial_score: float
    final_score: float
    objective_components: dict[str, float]
    trace: list[dict[str, float | int | str]] = field(default_factory=list)
    boundary_flags: dict[str, bool] = field(default_factory=dict)
    failed_reason: str | None = None
    rendered_image: torch.Tensor | None = field(default=None, repr=False)

    @property
    def improvement(self) -> float:
        return self.final_score - self.initial_score


@dataclass(frozen=True)
class Registration3DResult:
    candidates: list[Candidate3DResult]
    selected: Candidate3DResult | None
    metadata: dict[str, Any]
    failed_reason: str | None = None


def _as_bchw(value: torch.Tensor, device: torch.device) -> torch.Tensor:
    tensor = torch.as_tensor(value, dtype=torch.float32, device=device)
    if tensor.ndim == 2:
        tensor = tensor[None, None]
    elif tensor.ndim == 3:
        tensor = tensor[:, None]
    if tensor.ndim != 4 or tensor.shape[1] != 1:
        raise ValueError("target and ROI must be 2D, BHW, or single-channel BCHW tensors")
    return tensor


def _stage_shape(target: torch.Tensor, height: int) -> tuple[int, int]:
    width = max(16, int(round(height * target.shape[-1] / target.shape[-2])))
    return height, width


def _resize(value: torch.Tensor, height: int, width: int) -> torch.Tensor:
    if value.shape[-2:] == (height, width):
        return value
    return functional.interpolate(value, size=(height, width), mode="bilinear", align_corners=False)


def _normalize(image: torch.Tensor) -> torch.Tensor:
    low = image.amin(dim=(-2, -1), keepdim=True)
    high = image.amax(dim=(-2, -1), keepdim=True)
    return (image - low) / (high - low).clamp_min(1e-6)


def _pose_tensor(seed: PoseSeed, device: torch.device) -> torch.Tensor:
    return torch.tensor(
        [[*seed.rotation_deg, *seed.translation_mm]],
        dtype=torch.float32,
        device=device,
    )


def _limits(request: Registration3DRequest, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    bounds = (*request.rotation_bounds_deg, *request.translation_bounds_mm)
    lower = torch.tensor([[item[0] for item in bounds]], device=device, dtype=torch.float32)
    upper = torch.tensor([[item[1] for item in bounds]], device=device, dtype=torch.float32)
    if bool(torch.any(lower >= upper)):
        raise ValueError("each pose lower bound must be less than its upper bound")
    return lower, upper


def _clamp_pose(pose: torch.Tensor, lower: torch.Tensor, upper: torch.Tensor) -> torch.Tensor:
    return torch.maximum(torch.minimum(pose, upper), lower)


def _boundary_flags(
    pose: torch.Tensor,
    lower: torch.Tensor,
    upper: torch.Tensor,
) -> dict[str, bool]:
    names = ("rotation_z", "rotation_x", "rotation_y", "translation_x", "translation_y", "translation_z")
    tolerances = torch.tensor(
        [[0.1, 0.1, 0.1, 0.5, 0.5, 0.5]], device=pose.device, dtype=pose.dtype
    )
    at_boundary = ((pose - lower).abs() <= tolerances) | ((upper - pose).abs() <= tolerances)
    return {name: True for name, flag in zip(names, at_boundary[0].tolist()) if flag}


def _components(breakdown: ObjectiveBreakdown) -> dict[str, float]:
    return {
        "intensity_ncc": float(breakdown.intensity_ncc.detach().item()),
        "gradient_ncc": float(breakdown.gradient_ncc.detach().item()),
        "contour_score": float(breakdown.contour_score.detach().item()),
        "total": float(breakdown.total.detach().item()),
    }


def _memory_snapshot(device: torch.device) -> dict[str, int | None]:
    if device.type != "cuda" or not torch.cuda.is_available():
        return {"free_bytes": None, "total_bytes": None}
    free, total = torch.cuda.mem_get_info(device)
    return {"free_bytes": int(free), "total_bytes": int(total)}


def _render_and_score(
    renderer: DrrRenderer,
    pose: torch.Tensor,
    target: torch.Tensor,
    roi: torch.Tensor,
    height: int,
    width: int,
) -> tuple[torch.Tensor, ObjectiveBreakdown]:
    estimate = renderer.render(pose, height, width)
    estimate = _as_bchw(estimate, pose.device)
    if estimate.shape[0] != pose.shape[0]:
        raise ValueError("renderer batch dimension does not match pose batch dimension")
    estimate = _normalize(estimate)
    breakdown = registration_objective(
        estimate,
        _normalize(_resize(target, height, width)),
        _resize(roi, height, width).clamp(0.0, 1.0),
    )
    return estimate, breakdown


def _optimize_stage(
    request: Registration3DRequest,
    candidate: Candidate3DResult,
    start_pose: torch.Tensor,
    target: torch.Tensor,
    roi: torch.Tensor,
    *,
    height: int,
    width: int,
    iterations: int,
    stage: str,
    lower: torch.Tensor,
    upper: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, ObjectiveBreakdown]:
    clamped = _clamp_pose(start_pose.detach(), lower, upper)
    rotations = torch.nn.Parameter(clamped[:, :3].clone())
    translations = torch.nn.Parameter(clamped[:, 3:].clone())
    optimizer = torch.optim.Adam(
        [
            {"params": [rotations], "lr": request.rotation_learning_rate_deg},
            {"params": [translations], "lr": request.translation_learning_rate_mm},
        ]
    )
    with torch.no_grad():
        initial_image, initial_breakdown = _render_and_score(
            request.renderer,
            torch.cat([rotations, translations], dim=1),
            target,
            roi,
            height,
            width,
        )
    if not initial_breakdown.finite:
        raise FloatingPointError("nonfinite_initial_objective")
    best_pose = torch.cat([rotations, translations], dim=1).detach().clone()
    best_image = initial_image.detach().clone()
    best_breakdown = initial_breakdown

    for iteration in range(1, iterations + 1):
        optimizer.zero_grad(set_to_none=True)
        pose = torch.cat([rotations, translations], dim=1)
        estimate, breakdown = _render_and_score(
            request.renderer, pose, target, roi, height, width
        )
        if not breakdown.finite:
            raise FloatingPointError(f"nonfinite_objective_at_iteration_{iteration}")
        score = float(breakdown.total.detach().item())
        candidate.trace.append(
            {
                "stage": stage,
                "iteration": iteration,
                "height": height,
                "width": width,
                **_components(breakdown),
            }
        )
        if score > float(best_breakdown.total.detach().item()):
            best_pose = pose.detach().clone()
            best_image = estimate.detach().clone()
            best_breakdown = breakdown
        (-breakdown.total).backward()
        if rotations.grad is None or translations.grad is None:
            raise RuntimeError("renderer_did_not_provide_pose_gradients")
        if not bool(torch.isfinite(rotations.grad).all() and torch.isfinite(translations.grad).all()):
            raise FloatingPointError(f"nonfinite_gradient_at_iteration_{iteration}")
        optimizer.step()
        with torch.no_grad():
            rotations.copy_(
                _clamp_pose(torch.cat([rotations, translations], dim=1), lower, upper)[:, :3]
            )
            translations.copy_(
                _clamp_pose(torch.cat([rotations, translations], dim=1), lower, upper)[:, 3:]
            )

    with torch.no_grad():
        final_pose = torch.cat([rotations, translations], dim=1)
        final_image, final_breakdown = _render_and_score(
            request.renderer, final_pose, target, roi, height, width
        )
    if final_breakdown.finite and float(final_breakdown.total.item()) > float(best_breakdown.total.item()):
        best_pose = final_pose.detach().clone()
        best_image = final_image.detach().clone()
        best_breakdown = final_breakdown
    return best_pose, best_image, best_breakdown


def _is_cuda_oom(exc: BaseException) -> bool:
    return isinstance(exc, torch.cuda.OutOfMemoryError) or "out of memory" in str(exc).lower()


def _clear_renderer_cache(renderer: DrrRenderer, device: torch.device) -> None:
    clear_cache = getattr(renderer, "clear_cache", None)
    if callable(clear_cache):
        clear_cache()
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()


def _evaluate_diagnostic(
    request: Registration3DRequest,
    seed: PoseSeed,
    target: torch.Tensor,
    roi: torch.Tensor,
    height: int,
    width: int,
    lower: torch.Tensor,
    upper: torch.Tensor,
) -> Candidate3DResult:
    pose = _clamp_pose(_pose_tensor(seed, target.device), lower, upper)
    initial = tuple(float(value) for value in pose[0].detach().cpu().tolist())
    try:
        with torch.no_grad():
            image, breakdown = _render_and_score(
                request.renderer, pose, target, roi, height, width
            )
        if not breakdown.finite:
            raise FloatingPointError("nonfinite_diagnostic_objective")
        score = float(breakdown.total.item())
        return Candidate3DResult(
            name=seed.name,
            diagnostic_only=True,
            provenance=seed.provenance,
            initial_pose=initial,
            final_pose=initial,
            initial_score=score,
            final_score=score,
            objective_components=_components(breakdown),
            trace=[{"stage": "diagnostic", "iteration": 0, "height": height, "width": width, **_components(breakdown)}],
            boundary_flags=_boundary_flags(pose, lower, upper),
            rendered_image=image.detach().cpu(),
        )
    except Exception as exc:  # diagnostic evidence must never terminate physical registration
        return Candidate3DResult(
            name=seed.name,
            diagnostic_only=True,
            provenance=seed.provenance,
            initial_pose=initial,
            final_pose=initial,
            initial_score=float("-inf"),
            final_score=float("-inf"),
            objective_components={},
            failed_reason=f"{type(exc).__name__}: {exc}",
        )


def run_diffdrr_registration(request: Registration3DRequest) -> Registration3DResult:
    """Run fail-closed multi-start and multi-resolution 2D-3D registration."""

    device = torch.device(request.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device requested but unavailable: {device}")
    target = _as_bchw(request.target, device)
    roi = _as_bchw(request.roi, device)
    if target.shape[-2:] != roi.shape[-2:]:
        raise ValueError("target and ROI spatial dimensions must match")
    lower, upper = _limits(request, device)
    stage_memory: list[dict[str, Any]] = []
    first_height = int(request.resolutions[0])
    first_height, first_width = _stage_shape(target, first_height)
    candidates: list[Candidate3DResult] = []

    physical_seeds = [seed for seed in request.seeds if not seed.diagnostic_only]
    diagnostic_seeds = [seed for seed in request.seeds if seed.diagnostic_only]
    for seed in diagnostic_seeds:
        candidates.append(
            _evaluate_diagnostic(
                request, seed, target, roi, first_height, first_width, lower, upper
            )
        )
    if not physical_seeds:
        return Registration3DResult(
            candidates=candidates,
            selected=None,
            metadata={
                "pose_units": "degrees_mm",
                "renderer_convention": request.renderer.convention,
                "stage_memory": stage_memory,
            },
            failed_reason="no_physical_pose_seeds",
        )

    physical: list[Candidate3DResult] = []
    for seed in physical_seeds:
        original_pose = _pose_tensor(seed, device)
        clamped_pose = _clamp_pose(original_pose, lower, upper)
        candidate = Candidate3DResult(
            name=seed.name,
            diagnostic_only=False,
            provenance=seed.provenance,
            initial_pose=tuple(float(value) for value in clamped_pose[0].cpu().tolist()),
            final_pose=tuple(float(value) for value in clamped_pose[0].cpu().tolist()),
            initial_score=float("-inf"),
            final_score=float("-inf"),
            objective_components={},
            boundary_flags=_boundary_flags(clamped_pose, lower, upper),
        )
        stage_memory.append(
            {"candidate": seed.name, "stage": "coarse", **_memory_snapshot(device)}
        )
        try:
            with torch.no_grad():
                _, initial_breakdown = _render_and_score(
                    request.renderer,
                    clamped_pose,
                    target,
                    roi,
                    first_height,
                    first_width,
                )
            candidate.initial_score = float(initial_breakdown.total.item())
            pose, image, breakdown = _optimize_stage(
                request,
                candidate,
                clamped_pose,
                target,
                roi,
                height=first_height,
                width=first_width,
                iterations=request.coarse_iterations,
                stage="coarse",
                lower=lower,
                upper=upper,
            )
            candidate.final_pose = tuple(float(value) for value in pose[0].cpu().tolist())
            candidate.final_score = float(breakdown.total.detach().item())
            candidate.objective_components = _components(breakdown)
            candidate.boundary_flags = _boundary_flags(pose, lower, upper)
            candidate.rendered_image = image.detach().cpu()
        except Exception as exc:
            candidate.failed_reason = f"{type(exc).__name__}: {exc}"
            if _is_cuda_oom(exc):
                _clear_renderer_cache(request.renderer, device)
        physical.append(candidate)

    successful = [item for item in physical if item.failed_reason is None]
    successful.sort(key=lambda item: item.final_score, reverse=True)
    active = successful[: request.refine_top_k]
    previous_height = first_height
    for stage_index, configured_height in enumerate(request.resolutions[1:], start=1):
        height, width = _stage_shape(target, int(configured_height))
        for candidate in active:
            start_pose = torch.tensor([candidate.final_pose], dtype=torch.float32, device=device)
            stage_name = f"refine_{stage_index}"
            stage_memory.append(
                {"candidate": candidate.name, "stage": stage_name, **_memory_snapshot(device)}
            )
            try:
                pose, image, breakdown = _optimize_stage(
                    request,
                    candidate,
                    start_pose,
                    target,
                    roi,
                    height=height,
                    width=width,
                    iterations=request.refine_iterations,
                    stage=stage_name,
                    lower=lower,
                    upper=upper,
                )
            except Exception as exc:
                if _is_cuda_oom(exc):
                    _clear_renderer_cache(request.renderer, device)
                    fallback_height, fallback_width = _stage_shape(target, previous_height)
                    try:
                        pose, image, breakdown = _optimize_stage(
                            request,
                            candidate,
                            start_pose,
                            target,
                            roi,
                            height=fallback_height,
                            width=fallback_width,
                            iterations=request.refine_iterations,
                            stage=f"{stage_name}_oom_retry",
                            lower=lower,
                            upper=upper,
                        )
                    except Exception as retry_exc:
                        candidate.failed_reason = f"oom_retry_failed: {type(retry_exc).__name__}: {retry_exc}"
                        continue
                else:
                    candidate.failed_reason = f"{type(exc).__name__}: {exc}"
                    continue
            candidate.final_pose = tuple(float(value) for value in pose[0].cpu().tolist())
            candidate.final_score = float(breakdown.total.detach().item())
            candidate.objective_components = _components(breakdown)
            candidate.boundary_flags = _boundary_flags(pose, lower, upper)
            candidate.rendered_image = image.detach().cpu()
        active = sorted(
            [item for item in active if item.failed_reason is None],
            key=lambda item: item.final_score,
            reverse=True,
        )
        previous_height = height

    # The final comparison is always performed at the native target resolution.
    native_height, native_width = target.shape[-2:]
    for candidate in active:
        pose = torch.tensor([candidate.final_pose], dtype=torch.float32, device=device)
        try:
            with torch.no_grad():
                image, breakdown = _render_and_score(
                    request.renderer,
                    pose,
                    target,
                    roi,
                    native_height,
                    native_width,
                )
            if not breakdown.finite:
                raise FloatingPointError("nonfinite_native_objective")
            candidate.final_score = float(breakdown.total.item())
            candidate.objective_components = _components(breakdown)
            candidate.rendered_image = image.detach().cpu()
            candidate.boundary_flags = _boundary_flags(pose, lower, upper)
        except Exception as exc:
            candidate.failed_reason = f"native_evaluation_failed: {type(exc).__name__}: {exc}"

    selectable = [
        item
        for item in physical
        if item.failed_reason is None and not item.boundary_flags and math.isfinite(item.final_score)
    ]
    if not selectable:
        selectable = [
            item
            for item in physical
            if item.failed_reason is None and math.isfinite(item.final_score)
        ]
    selected = max(selectable, key=lambda item: item.final_score) if selectable else None
    candidates.extend(physical)
    failed_reason = None if selected is not None else "all_physical_candidates_failed"
    return Registration3DResult(
        candidates=candidates,
        selected=selected,
        metadata={
            "pose_units": "degrees_mm",
            "pose_order": [
                "rotation_z",
                "rotation_x",
                "rotation_y",
                "translation_x",
                "translation_y",
                "translation_z",
            ],
            "renderer_convention": request.renderer.convention,
            "device": str(device),
            "resolutions": list(request.resolutions),
            "native_target_shape": [int(native_height), int(native_width)],
            "stage_memory": stage_memory,
            "diagnostic_candidates_excluded_from_selection": True,
        },
        failed_reason=failed_reason,
    )


class DiffDrrRenderer:
    """Adapter around the vendored DiffDRR renderer with dynamic resolution caching."""

    convention = "DiffDRR euler_angles ZXY; external degrees/mm; internal radians/mm"

    def __init__(
        self,
        subject: Any,
        *,
        sdd_mm: float,
        detector_spacing_mm: tuple[float, float],
        original_detector_shape: tuple[int, int],
        device: str | torch.device,
        reverse_x_axis: bool = True,
        renderer: str = "siddon",
        checkpoint_gradients: bool = False,
    ) -> None:
        if sdd_mm <= 0:
            raise ValueError("sdd_mm must be positive")
        self.subject = subject
        self.sdd_mm = float(sdd_mm)
        self.detector_spacing_mm = tuple(float(value) for value in detector_spacing_mm)
        self.original_detector_shape = tuple(int(value) for value in original_detector_shape)
        self.device = torch.device(device)
        self.reverse_x_axis = reverse_x_axis
        self.renderer_name = renderer
        self.checkpoint_gradients = checkpoint_gradients
        self._cache: dict[tuple[int, int], Any] = {}

    @classmethod
    def from_ct_volume(
        cls,
        volume: CtVolume,
        geometry: ProjectionGeometry,
        *,
        device: str | torch.device,
        reverse_x_axis: bool = True,
        renderer: str = "siddon",
        checkpoint_gradients: bool = False,
    ) -> "DiffDrrRenderer":
        required = (
            geometry.sid_mm,
            geometry.pixel_spacing_row_mm,
            geometry.pixel_spacing_col_mm,
            geometry.rows,
            geometry.columns,
        )
        if any(value is None for value in required):
            raise ValueError("complete detector dimensions, spacing, and SID are required")
        import SimpleITK as sitk
        from diffdrr.data import read

        with tempfile.TemporaryDirectory(prefix="ctdr_diffdrr_") as temporary:
            path = Path(temporary) / "ct.nii.gz"
            sitk.WriteImage(volume.sitk_image, str(path))
            subject = read(path, orientation="AP", center_volume=True)
        return cls(
            subject,
            sdd_mm=float(geometry.sid_mm),
            detector_spacing_mm=(
                float(geometry.pixel_spacing_row_mm),
                float(geometry.pixel_spacing_col_mm),
            ),
            original_detector_shape=(int(geometry.rows), int(geometry.columns)),
            device=device,
            reverse_x_axis=reverse_x_axis,
            renderer=renderer,
            checkpoint_gradients=checkpoint_gradients,
        )

    def _drr(self, height: int, width: int) -> Any:
        key = (height, width)
        if key not in self._cache:
            from diffdrr.drr import DRR

            original_rows, original_columns = self.original_detector_shape
            row_spacing, column_spacing = self.detector_spacing_mm
            delx = column_spacing * original_columns / width
            dely = row_spacing * original_rows / height
            self._cache[key] = DRR(
                self.subject,
                sdd=self.sdd_mm,
                height=height,
                width=width,
                delx=delx,
                dely=dely,
                reverse_x_axis=self.reverse_x_axis,
                renderer=self.renderer_name,
                checkpoint_gradients=self.checkpoint_gradients,
            ).to(self.device)
        return self._cache[key]

    def render(self, pose: torch.Tensor, height: int, width: int) -> torch.Tensor:
        if pose.ndim != 2 or pose.shape[1] != 6:
            raise ValueError("pose must have shape Bx6")
        pose = pose.to(device=self.device, dtype=torch.float32)
        rotations = torch.deg2rad(pose[:, :3])
        translations = pose[:, 3:]
        return self._drr(height, width)(
            rotations,
            translations,
            parameterization="euler_angles",
            convention="ZXY",
        )

    def clear_cache(self) -> None:
        self._cache.clear()
