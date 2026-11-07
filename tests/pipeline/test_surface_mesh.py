from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest
import SimpleITK as sitk
import trimesh
from scipy import ndimage
from skimage.measure import marching_cubes

import ctdr_pipeline.visualization.surface_mesh as surface_mesh
from ctdr_pipeline.visualization.surface_mesh import (
    DISPLAY_PROFILES,
    LEGACY_REGISTRATION_PROFILES,
    SurfaceProfile,
    audit_surface_pair,
    extract_registration_surface,
    extract_surface,
    select_display_surface,
)
from tools.build_dynamic_vessel_3d import export_thorax_glb, surface_from_mask


def _write_spherical_mask(path: Path) -> None:
    z, y, x = np.indices((41, 41, 41))
    array = (((z - 20) ** 2 + (y - 20) ** 2 + (x - 20) ** 2) <= 12**2).astype(np.uint8)
    image = sitk.GetImageFromArray(array)
    image.SetSpacing((0.7, 0.8, 1.2))
    image.SetOrigin((10.0, 20.0, -30.0))
    image.SetDirection((-1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 1.0))
    sitk.WriteImage(image, str(path))


def _write_curved_tube_mask(
    path: Path,
    *,
    spacing_xyz: tuple[float, float, float],
    origin_xyz: tuple[float, float, float],
    direction: tuple[float, ...],
) -> None:
    z, y, x = np.ogrid[:80, :128, :128]
    center_x = 64.0 + 0.18 * (z - 40.0)
    center_y = 64.0 + 10.0 * np.sin((z - 12.0) / 14.0)
    array = (
        (z >= 12)
        & (z <= 67)
        & ((x - center_x) ** 2 + (y - center_y) ** 2 <= 5.0**2)
    ).astype(np.uint8)
    image = sitk.GetImageFromArray(array)
    image.SetSpacing(spacing_xyz)
    image.SetOrigin(origin_xyz)
    image.SetDirection(direction)
    sitk.WriteImage(image, str(path))


def _legacy_reference_surface(
    path: Path,
    *,
    step_size: int,
    gaussian_sigma_mm: float = 0.0,
    taubin_iterations: int = 0,
) -> tuple[np.ndarray, np.ndarray, sitk.Image]:
    """Frozen pre-refactor numerical path used as an independent oracle."""
    image = sitk.ReadImage(str(path))
    array = np.asarray(sitk.GetArrayViewFromImage(image)) > 0
    spacing = np.asarray(image.GetSpacing(), dtype=np.float64)
    if gaussian_sigma_mm > 0:
        field = ndimage.gaussian_filter(
            array.astype(np.float32),
            sigma=tuple(map(float, gaussian_sigma_mm / spacing[::-1])),
            mode="nearest",
            truncate=3.0,
        )
    else:
        field = array.astype(np.uint8)
    vertices_zyx, faces, _, _ = marching_cubes(
        field,
        level=0.5,
        step_size=step_size,
        allow_degenerate=False,
    )
    vertices_xyz = vertices_zyx[:, ::-1]
    origin = np.asarray(image.GetOrigin(), dtype=np.float64)
    direction = np.asarray(image.GetDirection(), dtype=np.float64).reshape(3, 3)
    vertices_lps = origin[None, :] + (vertices_xyz * spacing[None, :]) @ direction.T
    faces = faces.astype(np.int64)
    if taubin_iterations > 0:
        mesh = trimesh.Trimesh(vertices=vertices_lps, faces=faces, process=False)
        trimesh.smoothing.filter_taubin(
            mesh,
            lamb=0.5,
            nu=0.526,
            iterations=taubin_iterations,
        )
        vertices_lps = np.asarray(mesh.vertices, dtype=np.float64)
    return vertices_lps, faces, image


def test_surface_profile_preserves_legacy_constructor_contract() -> None:
    positional = SurfaceProfile(1, 1.0, 16)
    keyword = SurfaceProfile(
        step_size=1,
        gaussian_sigma_mm=1.0,
        taubin_iterations=16,
    )
    for profile in (positional, keyword):
        assert profile.step_size == 1
        assert profile.gaussian_sigma_mm == 1.0
        assert profile.taubin_iterations == 16


def test_extracted_registration_surface_matches_frozen_legacy_path(tmp_path: Path) -> None:
    mask_path = tmp_path / "heart.nii.gz"
    _write_spherical_mask(mask_path)

    coarse_expected = _legacy_reference_surface(mask_path, step_size=2)
    coarse_actual = surface_from_mask(mask_path)
    np.testing.assert_allclose(coarse_actual[0], coarse_expected[0])
    np.testing.assert_array_equal(coarse_actual[1], coarse_expected[1])

    expected = _legacy_reference_surface(
        mask_path,
        step_size=1,
        gaussian_sigma_mm=2.2,
        taubin_iterations=60,
    )
    actual = extract_surface(mask_path, LEGACY_REGISTRATION_PROFILES["heart"][0])
    wrapped = surface_from_mask(mask_path, profile="heart")
    selected = extract_registration_surface(mask_path, "heart")

    np.testing.assert_allclose(actual.vertices_lps_mm, expected[0])
    np.testing.assert_array_equal(actual.faces, expected[1])
    np.testing.assert_allclose(wrapped[0], expected[0])
    np.testing.assert_array_equal(wrapped[1], expected[1])
    np.testing.assert_allclose(selected.vertices_lps_mm, expected[0])
    assert wrapped[2].GetSpacing() == expected[2].GetSpacing()
    assert tuple(profile.name for profile in LEGACY_REGISTRATION_PROFILES["rib"]) == (
        "bone",
        "bone_fallback",
        "bone_safe",
        "raw",
    )


def test_legacy_thorax_report_keeps_original_mesh_processing_keys(tmp_path: Path) -> None:
    masks_dir = tmp_path / "masks"
    masks_dir.mkdir()
    _write_spherical_mask(masks_dir / "heart.nii.gz")
    records = export_thorax_glb(masks_dir, tmp_path / "thorax.glb")
    assert set(records[0]["mesh_processing"]) == {
        "step_size",
        "gaussian_sigma_mm",
        "taubin_iterations",
        "taubin_lambda",
        "taubin_nu",
        "windowed_sinc_iterations",
        "windowed_sinc_pass_band",
    }


def test_sdf_supersampling_preserves_anisotropic_lps_frame_and_densifies_mesh(
    tmp_path: Path,
) -> None:
    mask_path = tmp_path / "rib_left_1.nii.gz"
    origin_xyz = (31.0, -144.0, 512.0)
    _write_curved_tube_mask(
        mask_path,
        spacing_xyz=(0.72, 0.84, 1.4),
        origin_xyz=origin_xyz,
        direction=(-1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 1.0),
    )
    baseline = extract_surface(
        mask_path,
        LEGACY_REGISTRATION_PROFILES["rib"][0],
    ).as_trimesh()
    sdf_profile = SurfaceProfile(
        step_size=1,
        gaussian_sigma_mm=1.0,
        taubin_iterations=16,
        windowed_sinc_iterations=50,
        windowed_sinc_pass_band=0.04,
        sdf_supersample_factor=2.0,
        name="test_sdf",
    )

    display = extract_surface(mask_path, sdf_profile).as_trimesh()

    assert np.linalg.norm(display.centroid - baseline.centroid) < 0.75
    assert np.all(display.bounds[:, 0] < origin_xyz[0])
    assert np.all(display.bounds[:, 1] < origin_xyz[1])
    assert np.all(display.bounds[:, 2] > origin_xyz[2])
    assert display.faces.shape[0] > baseline.faces.shape[0]


def test_signed_distance_crop_is_bounded_around_sparse_foreground() -> None:
    array = np.zeros((80, 128, 128), dtype=np.uint8)
    array[35:45, 60:68, 61:67] = 1
    image = sitk.GetImageFromArray(array)
    image.SetSpacing((0.8, 0.9, 1.6))

    crop = surface_mesh._signed_distance_crop(image, margin_mm=10.0)

    assert crop.sdf_zyx_mm.shape == (24, 32, 32)
    assert np.prod(crop.sdf_zyx_mm.shape) < 0.05 * np.prod(array.shape)
    np.testing.assert_array_equal(crop.start_zyx, np.array([28, 48, 48]))
    np.testing.assert_allclose(crop.source_spacing_zyx_mm, np.array([1.6, 0.9, 0.8]))


def test_signed_distance_crop_rejects_empty_mask() -> None:
    image = sitk.GetImageFromArray(np.zeros((12, 16, 20), dtype=np.uint8))

    with pytest.raises(ValueError, match="empty"):
        surface_mesh._signed_distance_crop(image, margin_mm=4.0)


def test_sdf_supersampling_preserves_analytic_bounds_with_permuted_axes(
    tmp_path: Path,
) -> None:
    array = np.zeros((36, 48, 56), dtype=np.uint8)
    array[10:20, 15:25, 20:30] = 1
    image = sitk.GetImageFromArray(array)
    spacing_xyz = np.array([0.73, 1.10, 1.70])
    origin_xyz = np.array([31.0, -144.0, 512.0])
    direction = np.array(
        [
            [0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    image.SetSpacing(tuple(spacing_xyz))
    image.SetOrigin(tuple(origin_xyz))
    image.SetDirection(tuple(direction.ravel()))
    path = tmp_path / "analytic_cuboid.nii.gz"
    sitk.WriteImage(image, str(path))

    profile = SurfaceProfile(sdf_supersample_factor=1.7, name="analytic_sdf")
    mesh = extract_surface(path, profile).as_trimesh()

    local_min_xyz = np.array([19.5, 14.5, 9.5]) * spacing_xyz
    local_max_xyz = np.array([29.5, 24.5, 19.5]) * spacing_xyz
    corners = np.array(
        [
            [x, y, z]
            for x in (local_min_xyz[0], local_max_xyz[0])
            for y in (local_min_xyz[1], local_max_xyz[1])
            for z in (local_min_xyz[2], local_max_xyz[2])
        ]
    )
    expected_lps = origin_xyz + corners @ direction.T
    expected_bounds = np.vstack([expected_lps.min(axis=0), expected_lps.max(axis=0)])
    np.testing.assert_allclose(mesh.bounds, expected_bounds, atol=0.08)


def test_signed_distance_crop_does_not_allocate_foreground_coordinate_table(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    array = np.zeros((64, 96, 128), dtype=np.uint8)
    array[8:56, 12:84, 16:112] = 1
    image = sitk.GetImageFromArray(array)
    image.SetSpacing((0.8, 0.9, 1.4))

    def reject_argwhere(*_args: object, **_kwargs: object) -> np.ndarray:
        raise AssertionError("foreground-sized coordinate table allocated")

    monkeypatch.setattr(surface_mesh.np, "argwhere", reject_argwhere)
    crop = surface_mesh._signed_distance_crop(image, margin_mm=4.0)
    assert crop.sdf_zyx_mm.ndim == 3
    assert crop.sdf_zyx_mm.size <= array.size


@pytest.mark.parametrize("margin_mm", [-1.0, float("nan"), float("inf")])
def test_signed_distance_crop_rejects_invalid_margin(margin_mm: float) -> None:
    image = sitk.GetImageFromArray(np.ones((8, 8, 8), dtype=np.uint8))
    with pytest.raises(ValueError, match="margin_mm"):
        surface_mesh._signed_distance_crop(image, margin_mm=margin_mm)


def test_surface_audit_rejects_two_mm_translated_heart() -> None:
    registration = trimesh.creation.icosphere(subdivisions=2, radius=40.0)
    translated = registration.copy()
    translated.apply_translation([2.0, 0.0, 0.0])

    audit = audit_surface_pair(
        registration,
        translated,
        anatomy_kind="heart",
        sample_count=2000,
    )

    assert audit.centroid_shift_mm == pytest.approx(2.0)
    assert audit.within_tolerance is False
    assert "centroid_shift_mm" in audit.failed_metrics


def test_display_selector_falls_back_to_exact_registration_copy_for_rib() -> None:
    registration = trimesh.creation.cylinder(radius=3.0, height=80.0, sections=24)
    too_small = registration.copy()
    too_small.apply_scale(0.5)

    selected = select_display_surface(
        registration,
        [("rib_sdf_too_strong", too_small)],
        anatomy_kind="rib",
    )

    assert selected.selected_profile == "registration_fallback"
    assert selected.fallback_applied is True
    assert selected.mesh is not registration
    np.testing.assert_array_equal(selected.mesh.vertices, registration.vertices)
    np.testing.assert_array_equal(selected.mesh.faces, registration.faces)
    assert tuple(name for name, _audit in selected.failed_candidates) == (
        "rib_sdf_too_strong",
    )


def test_surface_audit_is_deterministic_for_fixed_sample_count() -> None:
    registration = trimesh.creation.icosphere(subdivisions=2, radius=25.0)
    candidate = registration.copy()
    candidate.apply_translation([0.1, -0.2, 0.3])

    first = audit_surface_pair(
        registration,
        candidate,
        anatomy_kind="heart",
        sample_count=512,
    )
    second = audit_surface_pair(
        registration,
        candidate,
        anatomy_kind="heart",
        sample_count=512,
    )

    assert asdict(first) == asdict(second)


def test_display_profiles_are_ordered_strongest_to_safe_and_use_sdf() -> None:
    expected_names = {
        "heart": (
            "heart_sdf_strong",
            "heart_sdf_medium",
            "heart_sdf_safe",
        ),
        "rib": (
            "rib_sdf_strong",
            "rib_sdf_medium",
            "rib_sdf_safe",
        ),
        "sternum": (
            "rib_sdf_strong",
            "rib_sdf_medium",
            "rib_sdf_safe",
        ),
    }

    assert set(DISPLAY_PROFILES) == set(expected_names)
    for anatomy_kind, names in expected_names.items():
        profiles = DISPLAY_PROFILES[anatomy_kind]
        assert tuple(profile.name for profile in profiles) == names
        assert all(profile.sdf_supersample_factor > 1.0 for profile in profiles)


def test_display_selector_chooses_first_passing_candidate() -> None:
    registration = trimesh.creation.icosphere(subdivisions=4, radius=20.0)
    first_passing = registration.copy()
    later_passing = registration.copy()

    selected = select_display_surface(
        registration,
        [
            ("heart_sdf_strong", first_passing),
            ("heart_sdf_safe", later_passing),
        ],
        anatomy_kind="heart",
    )

    assert selected.selected_profile == "heart_sdf_strong"
    assert selected.fallback_applied is False
    assert selected.failed_candidates == ()
