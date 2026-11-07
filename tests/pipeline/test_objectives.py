import torch

from ctdr_pipeline.registration.objectives import registration_objective


def asymmetric_image() -> torch.Tensor:
    image = torch.zeros((1, 1, 64, 64), dtype=torch.float32)
    image[:, :, 8:20, 10:34] = 0.8
    image[:, :, 30:55, 42:50] = 1.0
    return image


def test_perfect_images_have_near_one_ncc_and_finite_total() -> None:
    image = asymmetric_image()
    mask = torch.ones_like(image)

    result = registration_objective(image, image, mask)

    assert result.intensity_ncc.item() > 0.999
    assert result.gradient_ncc.item() > 0.999
    assert torch.isfinite(result.total)
    assert result.finite


def test_constant_images_remain_finite() -> None:
    image = torch.ones((1, 1, 32, 32), dtype=torch.float32)

    result = registration_objective(image, image, torch.ones_like(image))

    assert torch.isfinite(result.total)
    assert result.finite


def test_vertical_flip_scores_worse_than_aligned_image() -> None:
    target = asymmetric_image()
    mask = torch.ones_like(target)

    aligned = registration_objective(target, target, mask)
    flipped = registration_objective(torch.flip(target, dims=(-2,)), target, mask)

    assert aligned.gradient_ncc > flipped.gradient_ncc
    assert aligned.total > flipped.total
