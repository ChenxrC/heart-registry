"""Research-only synthetic point motion with explicitly known, fixed depth.

This is not patient pose estimation, depth recovery, a DRR renderer, or a
myocardial mechanics/contrast transport simulator. All coordinates are in a
synthetic pinhole camera; optional CT points supply a shape only.
"""

from __future__ import annotations

from numbers import Integral

import numpy as np


_PARAMETER_NAMES = {
    "static": [],
    "rigid_xy": ["tx_mm", "ty_mm"],
    "rigid_xy_radial": ["tx_mm", "ty_mm", "radial_xy_fraction"],
}


def _real_array(value, name: str) -> np.ndarray:
    try:
        raw = np.asarray(value)
        if raw.dtype.kind not in "iuf":
            raise ValueError(f"{name} must contain real numbers")
        return np.asarray(raw, dtype=np.float64)
    except (TypeError, OverflowError) as exc:
        raise ValueError(f"{name} must contain real numbers") from exc


def _points(value, *, positive_depth: bool = True) -> np.ndarray:
    points = _real_array(value, "points_xyz_mm")
    if points.ndim != 2 or points.shape[1] != 3 or len(points) == 0:
        raise ValueError("points_xyz_mm must be a nonempty N x 3 array")
    if not np.isfinite(points).all():
        raise ValueError("points_xyz_mm must be finite")
    if positive_depth and np.any(points[:, 2] <= 0):
        raise ValueError("camera depth z must be strictly positive")
    return points


def _camera(focal_px, principal_xy) -> tuple[float, np.ndarray]:
    focal = _real_array(focal_px, "focal_px")
    principal = _real_array(principal_xy, "principal_xy")
    if focal.ndim != 0 or not np.isfinite(focal) or focal <= 0:
        raise ValueError("focal_px must be a finite positive scalar")
    if principal.shape != (2,) or not np.isfinite(principal).all():
        raise ValueError("principal_xy must contain two finite numbers")
    return float(focal), principal


def project_points(
    points_xyz_mm: np.ndarray,
    *,
    focal_px: float = 600.0,
    principal_xy: tuple[float, float] = (256.0, 256.0),
) -> np.ndarray:
    """Project finite nonempty N x 3 camera points with z > 0 to N x 2 pixels.

    ``u = focal_px*x/z + cx`` and ``v = focal_px*y/z + cy``. Invalid
    geometry/camera parameters and unrepresentable results raise ValueError.
    Inputs are never changed.
    """
    points = _points(points_xyz_mm)
    focal, principal = _camera(focal_px, principal_xy)
    try:
        with np.errstate(over="raise", divide="raise", invalid="raise"):
            projected = focal * (points[:, :2] / points[:, 2, None]) + principal
    except FloatingPointError as exc:
        raise ValueError("projection is not representable as finite pixels") from exc
    if not np.isfinite(projected).all():
        raise ValueError("projection is not representable as finite pixels")
    return projected


def _indices(value, frame_count: int, *, unique: bool) -> np.ndarray:
    raw = np.asarray(value)
    if raw.ndim != 1 or raw.size == 0 or raw.dtype.kind not in "iu":
        raise ValueError("frame indices must be a nonempty integer vector")
    if np.any(raw < 0) or np.any(raw >= frame_count):
        raise ValueError("frame indices must be in the observation sequence")
    indices = raw.astype(np.int64)
    if unique and len(np.unique(indices)) != len(indices):
        raise ValueError("support indices must be unique")
    return indices


def _model_parameter_names(model: str) -> list[str]:
    if not isinstance(model, str) or model not in _PARAMETER_NAMES:
        raise ValueError("model must be static, rigid_xy, or rigid_xy_radial")
    return list(_PARAMETER_NAMES[model])


def fit_motion_parameters(
    reference_xyz_mm: np.ndarray,
    observed_xy_px: np.ndarray,
    support_indices,
    *,
    model: str,
    focal_px: float = 600.0,
    principal_xy: tuple[float, float] = (256.0, 256.0),
) -> dict:
    """Fit each support frame using known reference z (not unknown 3D depth).

    ``observed_xy_px`` has shape (F,N,2). Only support targets are inspected
    for finiteness or used in least squares; other frames may even be NaN.
    The two dynamic models use translation (tx_mm,ty_mm), optionally followed
    by a dimensionless radial coefficient a with dx=a*x0, dy=a*y0, dz=0.
    The returned JSON-safe dict contains sorted support indices/parameters.
    Rank-deficient shapes are rejected rather than assigned unique parameters.
    """
    reference = _points(reference_xyz_mm)
    names = _model_parameter_names(model)
    focal, principal = _camera(focal_px, principal_xy)
    observations = _real_array(observed_xy_px, "observed_xy_px")
    if observations.ndim != 3 or observations.shape[1:] != (len(reference), 2):
        raise ValueError("observed_xy_px must have shape (F,N,2) matching reference")
    support = np.sort(_indices(support_indices, len(observations), unique=True))
    selected = observations[support]
    if not np.isfinite(selected).all():
        raise ValueError("support observations must be finite")
    parameters = np.empty((len(support), len(names)), dtype=float)
    rank = 0
    if names:
        baseline = project_points(reference, focal_px=focal, principal_xy=tuple(principal))
        try:
            with np.errstate(over="raise", divide="raise", invalid="raise"):
                design = np.zeros((2 * len(reference), len(names)))
                scale = focal / reference[:, 2]
                design[0::2, 0] = scale
                design[1::2, 1] = scale
                if model == "rigid_xy_radial":
                    design[:, 2] = (scale[:, None] * reference[:, :2]).reshape(-1)
                residuals = (selected - baseline).reshape(len(support), -1)
                solution, _, rank, _ = np.linalg.lstsq(design, residuals.T, rcond=None)
        except (FloatingPointError, np.linalg.LinAlgError) as exc:
            raise ValueError("motion least squares is not numerically representable") from exc
        if rank != len(names):
            raise ValueError("reference shape is rank deficient for the chosen motion model")
        parameters = solution.T
        if not np.isfinite(parameters).all():
            raise ValueError("motion parameters must be finite")
    return {
        "model": model,
        "parameter_names": names,
        "support_indices": support.tolist(),
        "support_parameters": parameters.tolist(),
        "frame_count": len(observations),
        "point_count": len(reference),
        "focal_px": focal,
        "principal_xy": principal.tolist(),
        "design_rank": int(rank),
        "depth_assumption": "reference_z_known_and_fixed",
        "interpolation": "piecewise_linear_support_parameters",
    }


def predict_motion_points(
    reference_xyz_mm: np.ndarray, fit: dict, frame_indices
) -> np.ndarray:
    """Predict (K,N,3) points by linear parameter interpolation, without targets.

    Both dynamic models use exactly the same temporal interpolation. No
    extrapolation beyond the support range is allowed. Static repeats the
    reference shape; all models leave reference z unchanged.
    """
    reference = _points(reference_xyz_mm)
    names = _model_parameter_names(fit.get("model"))
    count = fit.get("frame_count")
    if isinstance(count, bool) or not isinstance(count, Integral) or count < 1:
        raise ValueError("fit must contain a positive integer frame_count")
    support = _indices(fit.get("support_indices"), count, unique=True)
    if np.any(np.diff(support) <= 0):
        raise ValueError("fit support indices must be sorted")
    indices = _indices(frame_indices, count, unique=False)
    if np.any(indices < support[0]) or np.any(indices > support[-1]):
        raise ValueError("prediction outside the support range would be extrapolation")
    parameters = _real_array(fit.get("support_parameters"), "support_parameters")
    if parameters.shape != (len(support), len(names)) or not np.isfinite(parameters).all():
        raise ValueError("fit support parameters have invalid shape or values")
    if fit.get("point_count") != len(reference):
        raise ValueError("fit point_count does not match reference")
    prediction = np.broadcast_to(reference, (len(indices), *reference.shape)).copy()
    if names:
        interpolated = np.column_stack([
            np.interp(indices, support, parameters[:, column])
            for column in range(len(names))
        ])
        try:
            with np.errstate(over="raise", invalid="raise"):
                prediction[:, :, :2] += interpolated[:, None, :2]
                if fit["model"] == "rigid_xy_radial":
                    prediction[:, :, :2] += interpolated[:, None, 2, None] * reference[None, :, :2]
        except FloatingPointError as exc:
            raise ValueError("predicted points are not finite") from exc
    if not np.isfinite(prediction).all():
        raise ValueError("predicted points are not finite")
    return prediction


def _prepare_shape(points_xyz_mm: np.ndarray | None) -> tuple[np.ndarray, dict]:
    if points_xyz_mm is None:
        # Deterministic Fibonacci ellipsoid, independent of the observation seed.
        indices = np.arange(160, dtype=float)
        z = 1.0 - 2.0 * (indices + 0.5) / len(indices)
        angle = indices * np.pi * (3.0 - np.sqrt(5.0))
        radius = np.sqrt(1.0 - z * z)
        points = np.column_stack((32 * radius * np.cos(angle), 25 * radius * np.sin(angle), 42 * z))
        source = "deterministic_digital_ellipsoid"
    else:
        points = _points(points_xyz_mm, positive_depth=False)
        source = "supplied_static_points"
    try:
        with np.errstate(over="raise", invalid="raise"):
            center = points.mean(axis=0)
            centered = points - center
            camera_distance = max(600.0, 4.0 * float(np.max(np.linalg.norm(centered, axis=1))))
            translation = -center + np.array([0.0, 0.0, camera_distance])
            reference = centered + np.array([0.0, 0.0, camera_distance])
    except FloatingPointError as exc:
        raise ValueError("shape centering/camera placement is not finite") from exc
    _points(reference)
    if not np.isfinite(translation).all():
        raise ValueError("input-to-camera translation must be finite")
    matrix = np.eye(4)
    matrix[:3, 3] = translation
    return reference, {
        "source": source,
        "used_as_shape_only": True,
        "patient_xa_pose_estimated": False,
        "input_center_xyz_mm": center.tolist(),
        "camera_translation_after_centering_xyz_mm": [0.0, 0.0, camera_distance],
        "input_to_camera_translation_xyz_mm": translation.tolist(),
        "input_to_camera_matrix_4x4": matrix.tolist(),
        "scale_factor": 1.0,
    }


def _generate_ground_truth(reference: np.ndarray, case: str, frame_count: int) -> np.ndarray:
    """Analytic truth generator; deliberately independent of fitting/prediction."""
    phase = np.linspace(0.0, 2.0 * np.pi, frame_count)
    pulse = (1.0 - np.cos(phase)) / 2.0
    truth = np.broadcast_to(reference, (frame_count, *reference.shape)).copy()
    if case == "translation":
        truth[:, :, 0] += (8.0 * np.sin(phase))[:, None]
        truth[:, :, 1] += (5.0 * (np.cos(phase) - 1.0))[:, None]
    elif case == "contraction":
        truth[:, :, :2] *= (1.0 - 0.12 * pulse)[:, None, None]
    elif case == "mixed":
        truth[:, :, :2] *= (1.0 - 0.10 * pulse)[:, None, None]
        truth[:, :, 0] += (6.0 * np.sin(phase))[:, None]
        truth[:, :, 1] += (4.0 * np.sin(2.0 * phase))[:, None]
    elif case == "local_shear":
        # Only the upper-right patch moves in x as a function of y: neither
        # a global XY translation nor an isotropic XY radial displacement.
        local_patch = (reference[:, 0] > 0) & (reference[:, 1] > 0)
        truth[:, :, 0] += 0.35 * pulse[:, None] * reference[None, :, 1] * local_patch
    elif case != "stationary":
        raise ValueError("unknown ground-truth motion case")
    return truth


def _project_frames(points: np.ndarray) -> np.ndarray:
    return project_points(points.reshape(-1, 3)).reshape(*points.shape[:2], 2)


def _split_metrics(residual_xy: np.ndarray, residual_xyz: np.ndarray, indices: np.ndarray) -> dict:
    error2 = np.linalg.norm(residual_xy[indices], axis=-1)
    error3 = np.linalg.norm(residual_xyz[indices], axis=-1)
    return {
        "frame_count": len(indices),
        "point_sample_count": int(error2.size),
        "mean_2d_error_px": float(np.mean(error2)),
        "rmse_2d_px": float(np.sqrt(np.mean(error2 ** 2))),
        "max_2d_error_px": float(np.max(error2)),
        "synthetic_mean_3d_error_mm": float(np.mean(error3)),
        "synthetic_rmse_3d_mm": float(np.sqrt(np.mean(error3 ** 2))),
        "synthetic_max_3d_error_mm": float(np.max(error3)),
    }


def run_synthetic_benchmark(
    points_xyz_mm: np.ndarray | None = None,
    *,
    seed: int = 20261017,
    frame_count: int = 24,
) -> dict:
    """Compare three constrained baselines on five seeded synthetic point cases.

    Optional points are centered and translated in front of an artificial
    camera, never treated as a calibrated patient XA pose. The middle third
    of frames is frozen before truth/noise creation. Only the complement is
    fitted; support parameters are linearly interpolated across the holdout.
    The return value is JSON-safe (including raw targets/predictions/residuals)
    and contains no clinical threshold, filtering, or clinical pass decision.
    """
    if isinstance(seed, bool) or not isinstance(seed, Integral) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if isinstance(frame_count, bool) or not isinstance(frame_count, Integral) or frame_count < 3:
        raise ValueError("frame_count must be an integer of at least three")
    seed, frame_count = int(seed), int(frame_count)
    heldout = np.arange(frame_count // 3, (2 * frame_count) // 3, dtype=int)
    support = np.setdiff1d(np.arange(frame_count), heldout)
    reference, shape = _prepare_shape(points_xyz_mm)
    rng = np.random.default_rng(seed)
    noise_std_px = 0.15
    cases = {}
    for name in ("stationary", "translation", "contraction", "mixed", "local_shear"):
        truth = _generate_ground_truth(reference, name, frame_count)
        truth_xy = _project_frames(truth)
        observations = truth_xy + rng.normal(0.0, noise_std_px, size=truth_xy.shape)
        results = {}
        for model in _PARAMETER_NAMES:
            fit = fit_motion_parameters(reference, observations, support, model=model)
            predicted = predict_motion_points(reference, fit, np.arange(frame_count))
            predicted_xy = _project_frames(predicted)
            residual_xy = predicted_xy - observations
            residual_xyz = predicted - truth
            results[model] = {
                "fit": fit,
                "predicted_xyz_mm": predicted.tolist(),
                "predicted_xy_px": predicted_xy.tolist(),
                "residual_xy_px": residual_xy.tolist(),
                "synthetic_residual_xyz_mm": residual_xyz.tolist(),
                "metrics": {
                    "support": _split_metrics(residual_xy, residual_xyz, support),
                    "heldout": _split_metrics(residual_xy, residual_xyz, heldout),
                },
            }
        cases[name] = {
            "truth_xyz_mm": truth.tolist(),
            "truth_xy_px": truth_xy.tolist(),
            "observed_xy_px": observations.tolist(),
            "out_of_model": name == "local_shear",
            "models": results,
        }
    positive_scales = np.linspace(0.7, 1.4, len(reference))
    alternative = reference * positive_scales[:, None]
    first_xy, second_xy = project_points(reference), project_points(alternative)
    return {
        "schema_version": 1,
        "benchmark_type": "synthetic_point_motion_known_depth",
        "seed": seed,
        "frame_count": frame_count,
        "point_count": len(reference),
        "camera": {"focal_px": 600.0, "principal_xy": [256.0, 256.0], "coordinates": "synthetic_camera_xyz_mm_z_positive"},
        "shape": shape,
        "reference_xyz_mm": reference.tolist(),
        "split": {
            "method": "frozen_contiguous_middle_third",
            "support_indices": support.tolist(),
            "heldout_indices": heldout.tolist(),
            "prediction": "piecewise_linear_support_parameters",
            "heldout_targets_used_for_fit": False,
        },
        "noise": {"distribution": "independent_zero_mean_gaussian_xy", "std_px": noise_std_px, "seed": seed, "rng": "numpy.default_rng_PCG64"},
        "cases": cases,
        "depth_ambiguity": {
            "positive_scale_factors": positive_scales.tolist(),
            "points_a_xyz_mm": reference.tolist(),
            "points_b_xyz_mm": alternative.tolist(),
            "points_a_xy_px": first_xy.tolist(),
            "points_b_xy_px": second_xy.tolist(),
            "max_projection_difference_px": float(np.max(np.linalg.norm(first_xy - second_xy, axis=-1))),
            "max_3d_difference_mm": float(np.max(np.linalg.norm(reference - alternative, axis=-1))),
            "single_view_unique_depth": False,
        },
        "single_view_unique_depth": False,
        "patient_true_3d_tre_mm": None,
        "clinical_validation": False,
        "limitations": [
            "Point-motion benchmark only; not full DRR, contrast transport, tissue simulation, or true myocardial mechanics.",
            "Optional CT points provide synthetic shape only; the recorded centering/translation is not patient XA pose.",
            "Known fixed z and known point correspondence are assumed; fitted motion is XY translation plus at most one XY radial mode, not full SE3 or unknown 3D depth.",
            "Synthetic conditional 3D errors use generated ground truth; they are not patient true 3D TRE.",
            "A single projection cannot distinguish positive ray-scaled points with different 3D depths.",
            "The middle third is predicted by parameter interpolation, not by an unseen-frame optimizer; smoothness and selected motion modes are assumptions.",
            "All five cases and their residuals are retained, including local shear outside the fitted modes; no clinical pass threshold is applied.",
            "Brightness negative controls belong to Task 2 image tests, not this point benchmark.",
        ],
    }
