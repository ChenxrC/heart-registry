"""Independent geometry fixtures and behavioral checks for Task 3."""

from __future__ import annotations

import importlib
import importlib.util
import json

import numpy as np
import pytest


def synthetic_module():
    # A missing implementation is an assertion failure, not a collection error.
    name = "ctdr_pipeline.motion_validation.synthetic"
    assert importlib.util.find_spec(name) is not None, "Task 3 module is missing"
    return importlib.import_module(name)


def test_projection_matches_hand_calculated_pinhole_coordinates() -> None:
    # Catches swapped axes, a missing principal point, or missing z division.
    points = np.array([[2.0, -1.0, 4.0], [-3.0, 6.0, 12.0]])
    result = synthetic_module().project_points(
        points, focal_px=100.0, principal_xy=(10.0, 20.0)
    )
    np.testing.assert_allclose(result, [[60.0, -5.0], [-15.0, 70.0]])


def test_projection_defaults_are_hand_calculated() -> None:
    result = synthetic_module().project_points(np.array([[1.0, 2.0, 10.0]]))
    np.testing.assert_allclose(result, [[316.0, 376.0]])


@pytest.mark.parametrize(
    "points",
    [
        [[1.0, 2.0, 0.0]],
        [[1.0, 2.0, -1.0]],
        [[np.nan, 0.0, 1.0]],
        [[0.0, np.inf, 1.0]],
        [[0.0, 0.0, np.inf]],
        [1.0, 2.0, 3.0],
        [[1.0, 2.0]],
        np.empty((0, 3)),
        [[1.0 + 2.0j, 0.0, 1.0]],
    ],
)
def test_projection_rejects_invalid_geometry(points) -> None:
    with pytest.raises(ValueError):
        synthetic_module().project_points(np.asarray(points))


@pytest.mark.parametrize("focal_px", [0.0, -1.0, np.nan, np.inf, True, [600.0]])
def test_projection_rejects_invalid_focal_length(focal_px) -> None:
    with pytest.raises(ValueError):
        synthetic_module().project_points(np.array([[0.0, 0.0, 1.0]]), focal_px=focal_px)


@pytest.mark.parametrize("principal_xy", [(np.nan, 0.0), (0.0, np.inf), (1.0,), (1, 2, 3)])
def test_projection_rejects_invalid_principal_point(principal_xy) -> None:
    with pytest.raises(ValueError):
        synthetic_module().project_points(
            np.array([[0.0, 0.0, 1.0]]), principal_xy=principal_xy
        )


def test_projection_does_not_return_infinity_for_finite_overflowing_inputs() -> None:
    with pytest.raises(ValueError):
        synthetic_module().project_points(np.array([[1e308, 0.0, 1e-308]]))


def test_projection_does_not_mutate_input() -> None:
    points = np.array([[2.0, -1.0, 4.0]])
    original = points.copy()
    synthetic_module().project_points(points)
    np.testing.assert_array_equal(points, original)


def reference_points() -> np.ndarray:
    return np.array([[-2.0, 0.0, 10.0], [2.0, 0.0, 10.0], [0.0, 2.0, 20.0], [0.0, -2.0, 20.0]])


def endpoint_observations(*, radial: bool) -> np.ndarray:
    # Independent hand calculation: f=100, principal=(0,0), t4=(4,-2), a4=-0.08.
    observations = np.full((5, 4, 2), np.nan)
    observations[0] = [[-20, 0], [20, 0], [0, 10], [0, -10]]
    observations[4] = (
        [[21.6, -20], [58.4, -20], [20, -0.8], [20, -19.2]]
        if radial else [[20, -20], [60, -20], [20, 0], [20, -20]]
    )
    return observations


@pytest.mark.parametrize(
    "model,radial,expected_parameters,expected_middle",
    [
        ("rigid_xy", False, [[0, 0], [4, -2]],
         [[0, -1, 10], [4, -1, 10], [2, 1, 20], [2, -3, 20]]),
        ("rigid_xy_radial", True, [[0, 0, 0], [4, -2, -0.08]],
         [[0.08, -1, 10], [3.92, -1, 10], [2, 0.92, 20], [2, -2.92, 20]]),
    ],
)
def test_known_depth_fit_recovers_mm_parameters_and_interpolates(
    model, radial, expected_parameters, expected_middle
) -> None:
    # Catches pixel-vs-mm confusion, wrong radial basis, and nearest-frame prediction.
    module = synthetic_module()
    fit = module.fit_motion_parameters(
        reference_points(), endpoint_observations(radial=radial), [4, 0],
        model=model, focal_px=100, principal_xy=(0, 0),
    )
    assert fit["support_indices"] == [0, 4]
    np.testing.assert_allclose(fit["support_parameters"], expected_parameters, atol=1e-12)
    predicted = module.predict_motion_points(reference_points(), fit, [2])
    np.testing.assert_allclose(predicted[0], expected_middle, atol=1e-12)
    np.testing.assert_array_equal(predicted[0, :, 2], [10, 10, 20, 20])


@pytest.mark.parametrize("model", ["rigid_xy", "rigid_xy_radial"])
def test_changing_or_masking_heldout_targets_cannot_change_fit_or_prediction(model) -> None:
    module = synthetic_module()
    masked = endpoint_observations(radial=True)
    poisoned = masked.copy()
    poisoned[1:4] = 123456.0
    before = module.fit_motion_parameters(
        reference_points(), masked, [0, 4], model=model, focal_px=100, principal_xy=(0, 0)
    )
    after = module.fit_motion_parameters(
        reference_points(), poisoned, [0, 4], model=model, focal_px=100, principal_xy=(0, 0)
    )
    assert before == after
    np.testing.assert_array_equal(
        module.predict_motion_points(reference_points(), before, [1, 2, 3]),
        module.predict_motion_points(reference_points(), after, [1, 2, 3]),
    )


def test_static_baseline_does_not_fit_a_translation() -> None:
    module = synthetic_module()
    fit = module.fit_motion_parameters(
        reference_points(), endpoint_observations(radial=False), [0, 4], model="static"
    )
    assert fit["support_parameters"] == [[], []]
    np.testing.assert_array_equal(
        module.predict_motion_points(reference_points(), fit, [0, 2, 4]),
        np.stack([reference_points()] * 3),
    )


@pytest.mark.parametrize("support", [[], [0, 0], [-1, 4], [0, 5], [0, 1.5], [True, False]])
def test_fit_rejects_invalid_support_indices(support) -> None:
    with pytest.raises(ValueError):
        synthetic_module().fit_motion_parameters(
            reference_points(), endpoint_observations(radial=False), support, model="rigid_xy"
        )


def test_fit_rejects_nonfinite_support_even_though_masked_holdout_is_allowed() -> None:
    observations = endpoint_observations(radial=False)
    observations[0, 0, 0] = np.nan
    with pytest.raises(ValueError):
        synthetic_module().fit_motion_parameters(reference_points(), observations, [0, 4], model="rigid_xy")


def test_fit_rejects_rank_deficient_radial_shape() -> None:
    with pytest.raises(ValueError):
        synthetic_module().fit_motion_parameters(
            np.array([[0, 0, 10], [0, 0, 20]]), np.zeros((3, 2, 2)),
            [0, 2], model="rigid_xy_radial",
        )


def test_fit_rejects_unknown_motion_model() -> None:
    with pytest.raises(ValueError):
        synthetic_module().fit_motion_parameters(
            reference_points(), endpoint_observations(radial=False), [0, 4], model="se3"
        )


@pytest.mark.parametrize("indices", [[-1], [5], [1.5], []])
def test_prediction_rejects_extrapolation_or_invalid_frame_indices(indices) -> None:
    module = synthetic_module()
    fit = module.fit_motion_parameters(
        reference_points(), endpoint_observations(radial=False), [0, 4], model="rigid_xy"
    )
    with pytest.raises(ValueError):
        module.predict_motion_points(reference_points(), fit, indices)


@pytest.fixture(scope="module")
def benchmark():
    return synthetic_module().run_synthetic_benchmark()


def test_benchmark_retains_every_case_model_and_contiguous_holdout(benchmark) -> None:
    assert set(benchmark["cases"]) == {"stationary", "translation", "contraction", "mixed", "local_shear"}
    support = benchmark["split"]["support_indices"]
    heldout = benchmark["split"]["heldout_indices"]
    assert support and heldout
    assert heldout == list(range(heldout[0], heldout[-1] + 1))
    assert support[0] < heldout[0] < heldout[-1] < support[-1]
    assert set(support).isdisjoint(heldout)
    assert sorted(support + heldout) == list(range(24))
    for case in benchmark["cases"].values():
        assert set(case["models"]) == {"static", "rigid_xy", "rigid_xy_radial"}
        for result in case["models"].values():
            assert result["fit"]["support_indices"] == support
            assert set(result["metrics"]) == {"support", "heldout"}


def test_benchmark_truth_static_translation_and_radial_contraction_are_distinct(benchmark) -> None:
    reference = np.asarray(benchmark["reference_xyz_mm"])
    cases = benchmark["cases"]
    stationary = np.asarray(cases["stationary"]["truth_xyz_mm"])
    np.testing.assert_array_equal(stationary, np.broadcast_to(reference, stationary.shape))
    translation = np.asarray(cases["translation"]["truth_xyz_mm"])
    displacement = translation - reference
    np.testing.assert_allclose(np.ptp(displacement[:, :, 0], axis=1), 0, atol=1e-12)
    np.testing.assert_allclose(np.ptp(displacement[:, :, 1], axis=1), 0, atol=1e-12)
    assert np.max(np.abs(displacement[:, :, :2])) > 1.0
    contraction = np.asarray(cases["contraction"]["truth_xyz_mm"])
    assert np.linalg.norm(contraction[12, :, :2]) < np.linalg.norm(reference[:, :2])
    # All truth cases preserve the declared known depth, including out-of-model shear.
    for case in cases.values():
        np.testing.assert_array_equal(np.asarray(case["truth_xyz_mm"])[:, :, 2], np.broadcast_to(reference[:, 2], (24, len(reference))))


@pytest.mark.parametrize("case,model", [("translation", "rigid_xy"), ("contraction", "rigid_xy_radial")])
def test_in_model_motion_improves_support_and_heldout_over_static(benchmark, case, model) -> None:
    results = benchmark["cases"][case]["models"]
    for split in ("support", "heldout"):
        assert results[model]["metrics"][split]["rmse_2d_px"] < results["static"]["metrics"][split]["rmse_2d_px"]
        assert results[model]["metrics"][split]["synthetic_rmse_3d_mm"] < results["static"]["metrics"][split]["synthetic_rmse_3d_mm"]


def test_local_shear_keeps_out_of_model_residuals(benchmark) -> None:
    cases = benchmark["cases"]
    shear = cases["local_shear"]["models"]["rigid_xy_radial"]
    stationary = cases["stationary"]["models"]["rigid_xy_radial"]
    # Numerical separation from the stationary noise floor, not a clinical gate.
    assert shear["metrics"]["support"]["rmse_2d_px"] > 2 * stationary["metrics"]["support"]["rmse_2d_px"]
    assert shear["metrics"]["heldout"]["synthetic_rmse_3d_mm"] > 0.5


def test_all_residuals_and_split_metrics_are_recomputable(benchmark) -> None:
    for case in benchmark["cases"].values():
        observed = np.asarray(case["observed_xy_px"])
        truth = np.asarray(case["truth_xyz_mm"])
        for result in case["models"].values():
            delta2 = np.asarray(result["predicted_xy_px"]) - observed
            delta3 = np.asarray(result["predicted_xyz_mm"]) - truth
            np.testing.assert_allclose(result["residual_xy_px"], delta2, atol=1e-12)
            np.testing.assert_allclose(result["synthetic_residual_xyz_mm"], delta3, atol=1e-12)
            for name, key in (("support", "support_indices"), ("heldout", "heldout_indices")):
                indices = benchmark["split"][key]
                metrics = result["metrics"][name]
                assert metrics["rmse_2d_px"] == pytest.approx(np.sqrt(np.mean(np.sum(delta2[indices] ** 2, axis=-1))))
                assert metrics["synthetic_rmse_3d_mm"] == pytest.approx(np.sqrt(np.mean(np.sum(delta3[indices] ** 2, axis=-1))))


def test_ray_scaled_points_have_identical_pixels_but_distinct_3d(benchmark) -> None:
    ambiguity = benchmark["depth_ambiguity"]
    first = np.asarray(ambiguity["points_a_xyz_mm"])
    second = np.asarray(ambiguity["points_b_xyz_mm"])
    scales = np.asarray(ambiguity["positive_scale_factors"])
    assert np.all(scales > 0)
    np.testing.assert_allclose(second, first * scales[:, None])
    # Independent ratios, not project_points used as its own oracle.
    np.testing.assert_allclose(first[:, :2] / first[:, 2, None], second[:, :2] / second[:, 2, None], atol=1e-14)
    assert ambiguity["max_projection_difference_px"] < 1e-10
    assert ambiguity["max_3d_difference_mm"] > 1.0


def test_seed_reproduces_noise_and_changing_seed_changes_only_observations() -> None:
    module = synthetic_module()
    first = module.run_synthetic_benchmark(seed=123, frame_count=6)
    assert first == module.run_synthetic_benchmark(seed=123, frame_count=6)
    other = module.run_synthetic_benchmark(seed=124, frame_count=6)
    assert first["reference_xyz_mm"] == other["reference_xyz_mm"]
    for name in first["cases"]:
        assert first["cases"][name]["truth_xyz_mm"] == other["cases"][name]["truth_xyz_mm"]
        assert first["cases"][name]["observed_xy_px"] != other["cases"][name]["observed_xy_px"]
    assert first["noise"]["std_px"] > 0
    assert first["noise"]["seed"] == 123


def test_optional_ct_points_are_only_centered_and_translated_as_shape() -> None:
    points = np.array([[10, 20, 0], [14, 20, 0], [10, 24, 2], [14, 24, 2]], dtype=float)
    original = points.copy()
    result = synthetic_module().run_synthetic_benchmark(points, frame_count=3)
    np.testing.assert_array_equal(points, original)
    np.testing.assert_allclose(result["reference_xyz_mm"], [[-2, -2, 599], [2, -2, 599], [-2, 2, 601], [2, 2, 601]])
    np.testing.assert_allclose(result["shape"]["input_to_camera_translation_xyz_mm"], [-12, -22, 599])
    assert result["shape"]["used_as_shape_only"] is True
    assert result["shape"]["patient_xa_pose_estimated"] is False


def test_synthetic_report_never_claims_clinical_validation_or_patient_tre(benchmark) -> None:
    assert benchmark["single_view_unique_depth"] is False
    assert benchmark["patient_true_3d_tre_mm"] is None
    assert benchmark["clinical_validation"] is False
    assert benchmark["limitations"]
    assert json.loads(json.dumps(benchmark, allow_nan=False))["patient_true_3d_tre_mm"] is None


@pytest.mark.parametrize("frame_count", [0, 1, 2, -1, 3.5, True])
def test_benchmark_rejects_frame_counts_without_a_valid_support_holdout_split(frame_count) -> None:
    with pytest.raises(ValueError):
        synthetic_module().run_synthetic_benchmark(frame_count=frame_count)


@pytest.mark.parametrize("seed", [-1, 1.5, True])
def test_benchmark_rejects_invalid_seed(seed) -> None:
    with pytest.raises(ValueError):
        synthetic_module().run_synthetic_benchmark(seed=seed)


@pytest.mark.parametrize("points", [np.empty((0, 3)), [[np.nan, 0, 0]], [[0, 0, 0]], [[0, 0]], np.full((3, 3), 1e308)])
def test_benchmark_rejects_nonfinite_degenerate_or_unrepresentable_shapes(points) -> None:
    with pytest.raises(ValueError):
        synthetic_module().run_synthetic_benchmark(np.asarray(points), frame_count=3)
