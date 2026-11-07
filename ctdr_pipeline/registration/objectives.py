from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as functional


@dataclass(frozen=True)
class ObjectiveBreakdown:
    intensity_ncc: torch.Tensor
    gradient_ncc: torch.Tensor
    contour_score: torch.Tensor
    total: torch.Tensor
    finite: bool


def weighted_ncc(
    first: torch.Tensor,
    second: torch.Tensor,
    weight: torch.Tensor,
) -> torch.Tensor:
    dimensions = (-2, -1)
    denominator = weight.sum(dim=dimensions, keepdim=True).clamp_min(1e-6)
    first_mean = (first * weight).sum(dim=dimensions, keepdim=True) / denominator
    second_mean = (second * weight).sum(dim=dimensions, keepdim=True) / denominator
    first_centered = first - first_mean
    second_centered = second - second_mean
    numerator = (weight * first_centered * second_centered).sum(dim=dimensions)
    first_energy = (weight * first_centered.square()).sum(dim=dimensions)
    second_energy = (weight * second_centered.square()).sum(dim=dimensions)
    return numerator / (first_energy * second_energy).sqrt().clamp_min(1e-6)


def gradient_magnitude(image: torch.Tensor) -> torch.Tensor:
    kernel_x = torch.tensor(
        [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
        device=image.device,
        dtype=image.dtype,
    )[None, None]
    kernel_y = kernel_x.transpose(-1, -2)
    grad_x = functional.conv2d(image, kernel_x, padding=1)
    grad_y = functional.conv2d(image, kernel_y, padding=1)
    return torch.sqrt(grad_x.square() + grad_y.square() + 1e-8)


def registration_objective(
    drr: torch.Tensor,
    target: torch.Tensor,
    roi: torch.Tensor,
) -> ObjectiveBreakdown:
    if drr.shape != target.shape or drr.shape != roi.shape:
        raise ValueError("drr, target, and roi must have identical BCHW shapes")
    weight = roi.clamp(0.0, 1.0)
    intensity = weighted_ncc(drr, target, weight).mean()
    drr_gradient = gradient_magnitude(drr)
    target_gradient = gradient_magnitude(target)
    gradient = weighted_ncc(drr_gradient, target_gradient, weight).mean()
    weighted_difference = (weight * (drr_gradient - target_gradient).abs()).sum()
    contour_denominator = (weight * target_gradient.abs()).sum().clamp_min(1e-6)
    contour = (1.0 - weighted_difference / contour_denominator).clamp(-1.0, 1.0)
    total = 0.30 * intensity + 0.55 * gradient + 0.15 * contour
    finite_tensor = torch.stack([intensity, gradient, contour, total])
    return ObjectiveBreakdown(
        intensity_ncc=intensity,
        gradient_ncc=gradient,
        contour_score=contour,
        total=total,
        finite=bool(torch.isfinite(finite_tensor).all().item()),
    )
