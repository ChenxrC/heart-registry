"""Physical-space surface extraction for thorax visualization assets.

The legacy extraction path is intentionally kept byte-for-byte equivalent in
its numerical operations so existing registration-facing meshes remain in the
same DICOM patient LPS frame.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from uuid import uuid4

import numpy as np
import SimpleITK as sitk
import trimesh
from scipy import ndimage
from skimage.measure import marching_cubes


@dataclass(frozen=True)
class SurfaceProfile:
    """Physical-space surface extraction settings for one anatomy class."""

    step_size: int = 1
    gaussian_sigma_mm: float = 0.0
    taubin_iterations: int = 0
    taubin_lambda: float = 0.5
    taubin_nu: float = 0.526
    windowed_sinc_iterations: int = 0
    windowed_sinc_pass_band: float = 0.1
    sdf_supersample_factor: float = 1.0
    target_edge_length_mm: float | None = None
    name: str = field(default="custom", kw_only=True)


@dataclass(frozen=True)
class SurfaceResult:
    """A surface mesh and its source image in DICOM patient LPS millimetres."""

    vertices_lps_mm: np.ndarray
    faces: np.ndarray
    image: sitk.Image
    profile: SurfaceProfile

    def as_trimesh(self) -> trimesh.Trimesh:
        return trimesh.Trimesh(
            vertices=self.vertices_lps_mm,
            faces=self.faces,
            process=False,
        )


@dataclass(frozen=True)
class SdfCrop:
    """A cropped signed-distance field in source-image ZYX coordinates."""

    sdf_zyx_mm: np.ndarray
    start_zyx: np.ndarray
    source_spacing_zyx_mm: np.ndarray


@dataclass(frozen=True)
class SurfaceTolerance:
    centroid_shift_mm: float
    volume_ratio_min: float
    volume_ratio_max: float
    surface_distance_p95_mm: float
    adjacency_angle_p95_deg: float


@dataclass(frozen=True)
class SurfaceAudit:
    centroid_shift_mm: float
    absolute_volume_ratio: float
    bidirectional_surface_distance_p95_mm: float
    adjacency_angle_p95_deg: float
    non_manifold_edge_delta: int
    within_tolerance: bool
    failed_metrics: tuple[str, ...]
    distance_method: str
    sample_count: int

    def to_dict(self) -> dict[str, object]:
        """Return a stable, JSON-ready audit payload."""
        return {
            "centroid_shift_mm": self.centroid_shift_mm,
            "absolute_volume_ratio": self.absolute_volume_ratio,
            "bidirectional_surface_distance_p95_mm": (
                self.bidirectional_surface_distance_p95_mm
            ),
            "adjacency_angle_p95_deg": self.adjacency_angle_p95_deg,
            "non_manifold_edge_delta": self.non_manifold_edge_delta,
            "within_tolerance": self.within_tolerance,
            "failed_metrics": list(self.failed_metrics),
            "distance_method": self.distance_method,
            "sample_count": self.sample_count,
        }


@dataclass(frozen=True)
class SurfaceSelection:
    mesh: trimesh.Trimesh
    selected_profile: str
    audit: SurfaceAudit
    failed_candidates: tuple[tuple[str, SurfaceAudit], ...]
    fallback_applied: bool


@dataclass(frozen=True)
class RegistrationSurfaceSelection:
    surface: SurfaceResult
    legacy_geometry_audit: dict[str, float | bool | str]
    fallback_applied: bool


@dataclass(frozen=True)
class ThoraxAssetRecord:
    path: str
    byte_length: int
    sha256: str

    def to_dict(self) -> dict[str, object]:
        return {
            "path": self.path,
            "byte_length": self.byte_length,
            "sha256": self.sha256,
        }


@dataclass(frozen=True)
class ThoraxRegistrationRecord:
    vertex_count: int
    face_count: int
    bbox_min_lps_mm: tuple[float, float, float]
    bbox_max_lps_mm: tuple[float, float, float]
    selected_profile: str
    mesh_processing: dict[str, int | float]
    geometry_audit: dict[str, float | bool | str]
    smoothing_fallback_applied: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "vertex_count": self.vertex_count,
            "face_count": self.face_count,
            "bbox_min_lps_mm": list(self.bbox_min_lps_mm),
            "bbox_max_lps_mm": list(self.bbox_max_lps_mm),
            "selected_profile": self.selected_profile,
            "mesh_processing": dict(self.mesh_processing),
            "geometry_audit": dict(self.geometry_audit),
            "smoothing_fallback_applied": self.smoothing_fallback_applied,
        }


@dataclass(frozen=True)
class ThoraxFailedCandidateRecord:
    profile: str
    audit: SurfaceAudit

    def to_dict(self) -> dict[str, object]:
        return {"profile": self.profile, "audit": self.audit.to_dict()}


@dataclass(frozen=True)
class ThoraxDisplayRecord:
    vertex_count: int
    face_count: int
    bbox_min_lps_mm: tuple[float, float, float]
    bbox_max_lps_mm: tuple[float, float, float]
    selected_profile: str
    geometry_audit: SurfaceAudit
    failed_candidates: tuple[ThoraxFailedCandidateRecord, ...]
    fallback_applied: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "vertex_count": self.vertex_count,
            "face_count": self.face_count,
            "bbox_min_lps_mm": list(self.bbox_min_lps_mm),
            "bbox_max_lps_mm": list(self.bbox_max_lps_mm),
            "selected_profile": self.selected_profile,
            "geometry_audit": self.geometry_audit.to_dict(),
            "failed_candidates": [item.to_dict() for item in self.failed_candidates],
            "fallback_applied": self.fallback_applied,
        }


@dataclass(frozen=True)
class ThoraxLabelRecord:
    label: str
    kind: str
    registration: ThoraxRegistrationRecord
    display: ThoraxDisplayRecord

    def to_dict(self) -> dict[str, object]:
        return {
            "label": self.label,
            "kind": self.kind,
            "registration": self.registration.to_dict(),
            "display": self.display.to_dict(),
        }

    def to_legacy_dict(self) -> dict[str, object]:
        """Project the version-2 registration record onto the legacy schema."""
        registration = self.registration
        return {
            "label": self.label,
            "kind": self.kind,
            "vertex_count": registration.vertex_count,
            "face_count": registration.face_count,
            "bbox_min_lps_mm": [round(value, 3) for value in registration.bbox_min_lps_mm],
            "bbox_max_lps_mm": [round(value, 3) for value in registration.bbox_max_lps_mm],
            "mesh_processing": dict(registration.mesh_processing),
            "geometry_audit": dict(registration.geometry_audit),
            "smoothing_profile": registration.selected_profile,
            "smoothing_fallback_applied": registration.smoothing_fallback_applied,
        }


@dataclass(frozen=True)
class ThoraxAssetReport:
    schema_version: int
    coordinate_system: str
    units: str
    display_to_registration_row_major: list[list[float]]
    registration_asset: ThoraxAssetRecord
    display_asset: ThoraxAssetRecord
    labels: tuple[ThoraxLabelRecord, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "coordinate_system": self.coordinate_system,
            "units": self.units,
            "display_to_registration_row_major": self.display_to_registration_row_major,
            "assets": {
                "registration": self.registration_asset.to_dict(),
                "display": self.display_asset.to_dict(),
            },
            "labels": [item.to_dict() for item in self.labels],
        }


TOLERANCES = {
    "heart": SurfaceTolerance(0.5, 0.97, 1.03, 1.5, 4.0),
    "rib": SurfaceTolerance(0.75, 0.90, 1.10, 1.0, 12.0),
    "sternum": SurfaceTolerance(0.75, 0.90, 1.10, 1.0, 12.0),
}


RAW_PROFILE = SurfaceProfile(name="raw")

LEGACY_REGISTRATION_PROFILES: dict[str, tuple[SurfaceProfile, ...]] = {
    "heart": (
        SurfaceProfile(name="heart", gaussian_sigma_mm=2.2, taubin_iterations=60),
    ),
    "rib": (
        SurfaceProfile(
            name="bone",
            gaussian_sigma_mm=1.0,
            taubin_iterations=16,
            windowed_sinc_iterations=50,
            windowed_sinc_pass_band=0.04,
        ),
        SurfaceProfile(
            name="bone_fallback",
            gaussian_sigma_mm=0.8,
            taubin_iterations=12,
            windowed_sinc_iterations=50,
            windowed_sinc_pass_band=0.04,
        ),
        SurfaceProfile(
            name="bone_safe",
            gaussian_sigma_mm=0.6,
            taubin_iterations=8,
            windowed_sinc_iterations=50,
            windowed_sinc_pass_band=0.04,
        ),
        RAW_PROFILE,
    ),
}
LEGACY_REGISTRATION_PROFILES["sternum"] = LEGACY_REGISTRATION_PROFILES["rib"]

DISPLAY_PROFILES: dict[str, tuple[SurfaceProfile, ...]] = {
    "heart": (
        SurfaceProfile(
            name="heart_sdf_strong",
            gaussian_sigma_mm=3.0,
            windowed_sinc_iterations=100,
            windowed_sinc_pass_band=0.03,
            sdf_supersample_factor=2.0,
        ),
        SurfaceProfile(
            name="heart_sdf_medium",
            gaussian_sigma_mm=2.7,
            windowed_sinc_iterations=90,
            windowed_sinc_pass_band=0.04,
            sdf_supersample_factor=2.0,
        ),
        SurfaceProfile(
            name="heart_sdf_safe",
            gaussian_sigma_mm=2.5,
            windowed_sinc_iterations=80,
            windowed_sinc_pass_band=0.05,
            sdf_supersample_factor=2.0,
        ),
    ),
    "rib": (
        SurfaceProfile(
            name="rib_sdf_strong",
            gaussian_sigma_mm=1.2,
            windowed_sinc_iterations=120,
            windowed_sinc_pass_band=0.03,
            sdf_supersample_factor=2.0,
            target_edge_length_mm=0.9,
        ),
        SurfaceProfile(
            name="rib_sdf_medium",
            gaussian_sigma_mm=1.0,
            windowed_sinc_iterations=100,
            windowed_sinc_pass_band=0.04,
            sdf_supersample_factor=2.0,
            target_edge_length_mm=0.9,
        ),
        SurfaceProfile(
            name="rib_sdf_safe",
            gaussian_sigma_mm=0.8,
            windowed_sinc_iterations=80,
            windowed_sinc_pass_band=0.05,
            sdf_supersample_factor=2.0,
            target_edge_length_mm=1.0,
        ),
    ),
}
DISPLAY_PROFILES["sternum"] = DISPLAY_PROFILES["rib"]


def _extract_legacy_isosurface(
    image: sitk.Image,
    profile: SurfaceProfile,
) -> tuple[np.ndarray, np.ndarray]:
    """Run the pre-SDF extraction path without changing numerical ordering."""
    array = np.asarray(sitk.GetArrayViewFromImage(image)) > 0
    spacing = np.asarray(image.GetSpacing(), dtype=np.float64)
    field: np.ndarray
    if profile.gaussian_sigma_mm > 0:
        sigma_zyx = profile.gaussian_sigma_mm / spacing[::-1]
        field = ndimage.gaussian_filter(
            array.astype(np.float32),
            sigma=tuple(map(float, sigma_zyx)),
            mode="nearest",
            truncate=3.0,
        )
    else:
        field = array.astype(np.uint8)

    vertices_zyx, faces, _, _ = marching_cubes(
        field,
        level=0.5,
        step_size=profile.step_size,
        allow_degenerate=False,
    )
    vertices_xyz = vertices_zyx[:, ::-1]
    origin = np.asarray(image.GetOrigin(), dtype=np.float64)
    direction = np.asarray(image.GetDirection(), dtype=np.float64).reshape(3, 3)
    vertices_lps = origin[None, :] + (vertices_xyz * spacing[None, :]) @ direction.T
    faces = faces.astype(np.int64)

    if profile.taubin_iterations > 0:
        mesh = trimesh.Trimesh(vertices=vertices_lps, faces=faces, process=False)
        trimesh.smoothing.filter_taubin(
            mesh,
            lamb=profile.taubin_lambda,
            nu=profile.taubin_nu,
            iterations=profile.taubin_iterations,
        )
        vertices_lps = np.asarray(mesh.vertices, dtype=np.float64)

    if profile.windowed_sinc_iterations > 0:
        import pyvista as pv

        vtk_faces = np.column_stack(
            [np.full(len(faces), 3, dtype=np.int64), faces]
        ).ravel()
        surface = pv.PolyData(vertices_lps, vtk_faces)
        smoothed = surface.smooth_taubin(
            n_iter=profile.windowed_sinc_iterations,
            pass_band=profile.windowed_sinc_pass_band,
            boundary_smoothing=True,
            feature_smoothing=False,
            non_manifold_smoothing=True,
            normalize_coordinates=True,
        )
        vertices_lps = np.asarray(smoothed.points, dtype=np.float64)

    return vertices_lps, faces


def _signed_distance_crop(image: sitk.Image, margin_mm: float) -> SdfCrop:
    """Crop a binary mask and compute its signed distance in millimetres."""
    margin_mm = float(margin_mm)
    if not np.isfinite(margin_mm) or margin_mm < 0.0:
        raise ValueError("margin_mm must be finite and non-negative")
    mask = np.asarray(sitk.GetArrayViewFromImage(image), dtype=bool)
    occupied_axes = (
        np.flatnonzero(mask.any(axis=(1, 2))),
        np.flatnonzero(mask.any(axis=(0, 2))),
        np.flatnonzero(mask.any(axis=(0, 1))),
    )
    if any(indices.size == 0 for indices in occupied_axes):
        raise ValueError("surface mask is empty")

    spacing_zyx = np.asarray(image.GetSpacing(), dtype=np.float64)[::-1]
    margin_voxels = np.ceil(margin_mm / spacing_zyx).astype(np.int64)
    foreground_start = np.asarray([indices[0] for indices in occupied_axes], dtype=np.int64)
    foreground_stop = np.asarray([indices[-1] + 1 for indices in occupied_axes], dtype=np.int64)
    start = np.maximum(foreground_start - margin_voxels, 0)
    stop = np.minimum(foreground_stop + margin_voxels, mask.shape)
    slices = tuple(slice(int(first), int(last)) for first, last in zip(start, stop))
    cropped = mask[slices]

    outside = ndimage.distance_transform_edt(~cropped, sampling=spacing_zyx)
    inside = ndimage.distance_transform_edt(cropped, sampling=spacing_zyx)
    return SdfCrop(
        sdf_zyx_mm=(outside - inside).astype(np.float32),
        start_zyx=start.astype(np.int64),
        source_spacing_zyx_mm=spacing_zyx,
    )


def _extract_sdf_isosurface(
    image: sitk.Image,
    profile: SurfaceProfile,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract a supersampled zero-isosurface in DICOM patient LPS millimetres."""
    margin_mm = max(4.0, 3.0 * profile.gaussian_sigma_mm)
    crop = _signed_distance_crop(image, margin_mm)
    requested_spacing_mm = float(
        np.min(crop.source_spacing_zyx_mm) / profile.sdf_supersample_factor
    )
    requested_zoom = crop.source_spacing_zyx_mm / requested_spacing_mm
    sdf = ndimage.zoom(
        crop.sdf_zyx_mm,
        zoom=requested_zoom,
        order=1,
        mode="nearest",
        prefilter=False,
        grid_mode=False,
    )

    source_shape = np.asarray(crop.sdf_zyx_mm.shape, dtype=np.int64)
    output_shape = np.asarray(sdf.shape, dtype=np.int64)
    source_extent_mm = (source_shape - 1) * crop.source_spacing_zyx_mm
    effective_spacing_zyx_mm = crop.source_spacing_zyx_mm.copy()
    resampled_axes = (source_shape > 1) & (output_shape > 1)
    effective_spacing_zyx_mm[resampled_axes] = (
        source_extent_mm[resampled_axes] / (output_shape[resampled_axes] - 1)
    )

    if profile.gaussian_sigma_mm > 0:
        sigma_zyx = profile.gaussian_sigma_mm / effective_spacing_zyx_mm
        sdf = ndimage.gaussian_filter(
            sdf,
            sigma=tuple(map(float, sigma_zyx)),
            mode="nearest",
            truncate=3.0,
        )

    vertices_zyx_mm, faces, _, _ = marching_cubes(
        sdf,
        level=0.0,
        spacing=tuple(map(float, effective_spacing_zyx_mm)),
        step_size=profile.step_size,
        allow_degenerate=False,
    )
    source_zyx_mm = (
        vertices_zyx_mm
        + crop.start_zyx[None, :] * crop.source_spacing_zyx_mm[None, :]
    )
    vertices_xyz_mm = source_zyx_mm[:, ::-1]
    origin = np.asarray(image.GetOrigin(), dtype=np.float64)
    direction = np.asarray(image.GetDirection(), dtype=np.float64).reshape(3, 3)
    vertices_lps = origin[None, :] + vertices_xyz_mm @ direction.T
    faces = faces.astype(np.int64)

    if profile.taubin_iterations > 0:
        mesh = trimesh.Trimesh(vertices=vertices_lps, faces=faces, process=False)
        trimesh.smoothing.filter_taubin(
            mesh,
            lamb=profile.taubin_lambda,
            nu=profile.taubin_nu,
            iterations=profile.taubin_iterations,
        )
        vertices_lps = np.asarray(mesh.vertices, dtype=np.float64)

    if profile.windowed_sinc_iterations > 0:
        import pyvista as pv

        vtk_faces = np.column_stack(
            [np.full(len(faces), 3, dtype=np.int64), faces]
        ).ravel()
        surface = pv.PolyData(vertices_lps, vtk_faces)
        smoothed = surface.smooth_taubin(
            n_iter=profile.windowed_sinc_iterations,
            pass_band=profile.windowed_sinc_pass_band,
            boundary_smoothing=True,
            feature_smoothing=False,
            non_manifold_smoothing=True,
            normalize_coordinates=True,
        )
        vertices_lps = np.asarray(smoothed.points, dtype=np.float64)

    return vertices_lps, faces


def extract_surface(path: Path, profile: SurfaceProfile) -> SurfaceResult:
    """Extract one surface while preserving the source image's LPS affine."""
    image = sitk.ReadImage(str(path))
    if profile.sdf_supersample_factor > 1.0:
        vertices, faces = _extract_sdf_isosurface(image, profile)
    else:
        vertices, faces = _extract_legacy_isosurface(image, profile)
    return SurfaceResult(vertices, faces, image, profile)


def _surface_geometry_audit(
    baseline: trimesh.Trimesh,
    processed: trimesh.Trimesh,
) -> dict[str, float | bool | str]:
    """Apply the legacy visualization-tolerance gate unchanged."""
    baseline_volume = abs(float(baseline.volume))
    processed_volume = abs(float(processed.volume))
    volume_ratio = processed_volume / baseline_volume if baseline_volume > 1e-9 else float("nan")
    use_volume_centroid = bool(baseline.is_watertight and processed.is_watertight)
    baseline_centroid = baseline.center_mass if use_volume_centroid else baseline.centroid
    processed_centroid = processed.center_mass if use_volume_centroid else processed.centroid
    centroid_shift = float(np.linalg.norm(processed_centroid - baseline_centroid))
    surface_centroid_shift = float(np.linalg.norm(processed.centroid - baseline.centroid))
    bbox_shift = float(np.max(np.abs(processed.bounds - baseline.bounds)))
    area_ratio = float(processed.area / baseline.area) if baseline.area > 1e-9 else float("nan")
    return {
        "centroid_shift_mm": round(centroid_shift, 6),
        "centroid_definition": (
            "enclosed-volume center of mass"
            if use_volume_centroid
            else "surface-area centroid (open mesh)"
        ),
        "surface_centroid_shift_mm": round(surface_centroid_shift, 6),
        "bbox_max_abs_shift_mm": round(bbox_shift, 6),
        "absolute_volume_ratio": round(volume_ratio, 6),
        "surface_area_ratio": round(area_ratio, 6),
        "within_visualization_tolerance": bool(
            centroid_shift < 0.75
            and bbox_shift < 2.0
            and np.isfinite(volume_ratio)
            and 0.85 < volume_ratio < 1.10
        ),
    }


def _select_registration_surface(
    path: Path,
    anatomy_kind: str,
) -> RegistrationSurfaceSelection:
    """Select once and retain the legacy audit needed by both report schemas."""
    try:
        profiles = LEGACY_REGISTRATION_PROFILES[anatomy_kind]
    except KeyError as exc:
        allowed = ", ".join(sorted(LEGACY_REGISTRATION_PROFILES))
        raise ValueError(f"Unknown anatomy kind {anatomy_kind!r}; expected one of: {allowed}") from exc

    baseline = extract_surface(path, RAW_PROFILE)
    baseline_mesh = baseline.as_trimesh()
    selected = baseline
    selected_audit = _surface_geometry_audit(baseline_mesh, baseline_mesh)
    for profile in profiles:
        candidate = baseline if profile is RAW_PROFILE else extract_surface(path, profile)
        selected = candidate
        selected_audit = _surface_geometry_audit(baseline_mesh, candidate.as_trimesh())
        if selected_audit["within_visualization_tolerance"]:
            break
    return RegistrationSurfaceSelection(
        surface=selected,
        legacy_geometry_audit=selected_audit,
        fallback_applied=selected.profile.name != profiles[0].name,
    )


def extract_registration_surface(path: Path, anatomy_kind: str) -> SurfaceResult:
    """Select the first legacy profile passing the existing geometry gate."""
    return _select_registration_surface(path, anatomy_kind).surface


def _non_manifold_edge_count(mesh: trimesh.Trimesh) -> int:
    """Count edges which are not shared by exactly two triangle faces."""
    counts = np.bincount(
        np.asarray(mesh.edges_unique_inverse, dtype=np.int64),
        minlength=len(mesh.edges_unique),
    )
    return int(np.count_nonzero(counts != 2))


def _validate_audit_mesh(mesh: trimesh.Trimesh, label: str) -> None:
    if not isinstance(mesh, trimesh.Trimesh):
        raise TypeError(f"{label} must be a trimesh.Trimesh")
    vertices = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.faces)
    if vertices.ndim != 2 or vertices.shape[0] == 0:
        raise ValueError(f"{label} mesh has no vertices")
    if faces.ndim != 2 or faces.shape[0] == 0 or faces.shape[1] != 3:
        raise ValueError(f"{label} mesh has no triangle faces")
    if not np.all(np.isfinite(vertices)):
        raise ValueError(f"{label} mesh contains non-finite vertices")
    area = float(mesh.area)
    if not np.isfinite(area) or area <= 0.0:
        raise ValueError(f"{label} mesh has degenerate surface area")


def _sample_surface_fixed(
    mesh: trimesh.Trimesh,
    sample_count: int,
    seed: int,
) -> np.ndarray:
    """Sample by triangle area with a fixed seed across trimesh versions."""
    try:
        points, _ = trimesh.sample.sample_surface(mesh, sample_count, seed=seed)
    except TypeError:
        state = np.random.get_state()
        np.random.seed(seed)
        try:
            points, _ = trimesh.sample.sample_surface(mesh, sample_count)
        finally:
            np.random.set_state(state)
    return np.asarray(points, dtype=np.float64)


def _pyvista_triangle_surface(mesh: trimesh.Trimesh):
    import pyvista as pv

    faces = np.column_stack(
        [np.full(len(mesh.faces), 3, dtype=np.int64), np.asarray(mesh.faces)]
    ).ravel()
    return pv.PolyData(np.asarray(mesh.vertices, dtype=np.float64), faces)


def _implicit_distances(points: np.ndarray, surface) -> np.ndarray:
    import pyvista as pv

    sampled = pv.PolyData(points)
    measured = sampled.compute_implicit_distance(surface, inplace=False)
    return np.abs(np.asarray(measured["implicit_distance"], dtype=np.float64))


def _adjacency_angle_p95_deg(mesh: trimesh.Trimesh) -> float:
    angles = np.degrees(
        np.abs(np.asarray(mesh.face_adjacency_angles, dtype=np.float64))
    )
    finite = angles[np.isfinite(angles)]
    if finite.size == 0:
        return float("inf")
    return float(np.percentile(finite, 95.0))


def audit_surface_pair(
    registration: trimesh.Trimesh,
    candidate: trimesh.Trimesh,
    anatomy_kind: str,
    sample_count: int = 1024,
) -> SurfaceAudit:
    """Audit a display candidate against registration geometry in LPS mm."""
    try:
        tolerance = TOLERANCES[anatomy_kind]
    except KeyError as exc:
        allowed = ", ".join(sorted(TOLERANCES))
        raise ValueError(
            f"Unknown anatomy kind {anatomy_kind!r}; expected one of: {allowed}"
        ) from exc
    if (
        isinstance(sample_count, bool)
        or not isinstance(sample_count, (int, np.integer))
        or int(sample_count) <= 0
    ):
        raise ValueError("sample_count must be a positive integer")
    sample_count = int(sample_count)
    _validate_audit_mesh(registration, "registration")
    _validate_audit_mesh(candidate, "candidate")

    use_volume_centroid = bool(registration.is_watertight and candidate.is_watertight)
    registration_centroid = (
        registration.center_mass if use_volume_centroid else registration.centroid
    )
    candidate_centroid = candidate.center_mass if use_volume_centroid else candidate.centroid
    centroid_shift_mm = float(
        np.linalg.norm(
            np.asarray(candidate_centroid, dtype=np.float64)
            - np.asarray(registration_centroid, dtype=np.float64)
        )
    )

    registration_volume = abs(float(registration.volume))
    candidate_volume = abs(float(candidate.volume))
    absolute_volume_ratio = (
        candidate_volume / registration_volume
        if registration_volume > 1e-12
        else float("nan")
    )

    registration_points = _sample_surface_fixed(registration, sample_count, 20260904)
    candidate_points = _sample_surface_fixed(candidate, sample_count, 20260905)
    registration_surface = _pyvista_triangle_surface(registration)
    candidate_surface = _pyvista_triangle_surface(candidate)
    bidirectional_distances = np.concatenate(
        [
            _implicit_distances(registration_points, candidate_surface),
            _implicit_distances(candidate_points, registration_surface),
        ]
    )
    surface_distance_p95_mm = float(np.percentile(bidirectional_distances, 95.0))
    adjacency_angle_p95_deg = _adjacency_angle_p95_deg(candidate)
    non_manifold_edge_delta = _non_manifold_edge_count(
        candidate
    ) - _non_manifold_edge_count(registration)

    failed_metrics: list[str] = []
    if not np.isfinite(centroid_shift_mm) or centroid_shift_mm > tolerance.centroid_shift_mm:
        failed_metrics.append("centroid_shift_mm")
    if (
        not np.isfinite(absolute_volume_ratio)
        or absolute_volume_ratio < tolerance.volume_ratio_min
        or absolute_volume_ratio > tolerance.volume_ratio_max
    ):
        failed_metrics.append("absolute_volume_ratio")
    if (
        not np.isfinite(surface_distance_p95_mm)
        or surface_distance_p95_mm > tolerance.surface_distance_p95_mm
    ):
        failed_metrics.append("bidirectional_surface_distance_p95_mm")
    if (
        not np.isfinite(adjacency_angle_p95_deg)
        or adjacency_angle_p95_deg > tolerance.adjacency_angle_p95_deg
    ):
        failed_metrics.append("adjacency_angle_p95_deg")
    if non_manifold_edge_delta > 0:
        failed_metrics.append("non_manifold_edge_delta")

    return SurfaceAudit(
        centroid_shift_mm=centroid_shift_mm,
        absolute_volume_ratio=absolute_volume_ratio,
        bidirectional_surface_distance_p95_mm=surface_distance_p95_mm,
        adjacency_angle_p95_deg=adjacency_angle_p95_deg,
        non_manifold_edge_delta=non_manifold_edge_delta,
        within_tolerance=not failed_metrics,
        failed_metrics=tuple(failed_metrics),
        distance_method=(
            "trimesh.sample.sample_surface+pyvista.compute_implicit_distance"
        ),
        sample_count=sample_count,
    )


def select_display_surface(
    registration: trimesh.Trimesh,
    candidates: Iterable[tuple[str, trimesh.Trimesh]],
    anatomy_kind: str,
    sample_count: int = 1024,
) -> SurfaceSelection:
    """Select the first passing display mesh, or an exact registration copy."""
    failed_candidates: list[tuple[str, SurfaceAudit]] = []
    for profile_name, candidate in candidates:
        audit = audit_surface_pair(
            registration,
            candidate,
            anatomy_kind=anatomy_kind,
            sample_count=sample_count,
        )
        if audit.within_tolerance:
            return SurfaceSelection(
                mesh=candidate,
                selected_profile=str(profile_name),
                audit=audit,
                failed_candidates=tuple(failed_candidates),
                fallback_applied=False,
            )
        failed_candidates.append((str(profile_name), audit))

    fallback_mesh = registration.copy()
    fallback_audit = audit_surface_pair(
        registration,
        fallback_mesh,
        anatomy_kind=anatomy_kind,
        sample_count=sample_count,
    )
    return SurfaceSelection(
        mesh=fallback_mesh,
        selected_profile="registration_fallback",
        audit=fallback_audit,
        failed_candidates=tuple(failed_candidates),
        fallback_applied=True,
    )


def _ordered_thorax_masks(masks_dir: Path) -> tuple[Path, ...]:
    """Return the same deterministic anatomy order as the legacy exporter."""
    masks_dir = Path(masks_dir)
    candidates = (
        *sorted(masks_dir.glob("rib_*.nii.gz")),
        masks_dir / "sternum.nii.gz",
        masks_dir / "heart.nii.gz",
    )
    paths = tuple(path for path in candidates if path.is_file())
    if not paths:
        raise ValueError(f"No thorax masks found in {masks_dir}")
    return paths


def _anatomy_kind(path: Path) -> str:
    label = path.name.removesuffix(".nii.gz")
    if label == "heart":
        return "heart"
    if label == "sternum":
        return "sternum"
    if label.startswith("rib_"):
        return "rib"
    raise ValueError(f"Unsupported thorax mask filename: {path.name}")


def _thorax_color(anatomy_kind: str) -> np.ndarray:
    if anatomy_kind == "heart":
        return np.asarray([205, 65, 104, 72], dtype=np.uint8)
    if anatomy_kind == "sternum":
        return np.asarray([240, 202, 126, 195], dtype=np.uint8)
    if anatomy_kind == "rib":
        return np.asarray([229, 218, 184, 190], dtype=np.uint8)
    raise ValueError(f"Unsupported anatomy kind: {anatomy_kind}")


def _add_colored_geometry(
    scene: trimesh.Scene,
    mesh: trimesh.Trimesh,
    path: Path,
    anatomy_kind: str,
) -> None:
    label = path.name.removesuffix(".nii.gz")
    mesh.visual.face_colors = np.tile(_thorax_color(anatomy_kind), (len(mesh.faces), 1))
    scene.add_geometry(mesh, node_name=label, geom_name=label)


def _mesh_bounds(
    mesh: trimesh.Trimesh,
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    bounds = np.asarray(mesh.bounds, dtype=np.float64)
    minimum = tuple(float(value) for value in np.round(bounds[0], 6))
    maximum = tuple(float(value) for value in np.round(bounds[1], 6))
    return minimum, maximum


def _legacy_profile_payload(profile: SurfaceProfile) -> dict[str, int | float]:
    return {
        "step_size": profile.step_size,
        "gaussian_sigma_mm": profile.gaussian_sigma_mm,
        "taubin_iterations": profile.taubin_iterations,
        "taubin_lambda": profile.taubin_lambda,
        "taubin_nu": profile.taubin_nu,
        "windowed_sinc_iterations": profile.windowed_sinc_iterations,
        "windowed_sinc_pass_band": profile.windowed_sinc_pass_band,
    }


def _surface_record(
    path: Path,
    anatomy_kind: str,
    registration_selection: RegistrationSurfaceSelection,
    selection: SurfaceSelection,
) -> ThoraxLabelRecord:
    registration = registration_selection.surface
    registration_mesh = registration.as_trimesh()
    registration_min, registration_max = _mesh_bounds(registration_mesh)
    display_min, display_max = _mesh_bounds(selection.mesh)
    return ThoraxLabelRecord(
        label=path.name.removesuffix(".nii.gz"),
        kind=anatomy_kind,
        registration=ThoraxRegistrationRecord(
            vertex_count=int(len(registration_mesh.vertices)),
            face_count=int(len(registration_mesh.faces)),
            bbox_min_lps_mm=registration_min,
            bbox_max_lps_mm=registration_max,
            selected_profile=registration.profile.name,
            mesh_processing=_legacy_profile_payload(registration.profile),
            geometry_audit=dict(registration_selection.legacy_geometry_audit),
            smoothing_fallback_applied=registration_selection.fallback_applied,
        ),
        display=ThoraxDisplayRecord(
            vertex_count=int(len(selection.mesh.vertices)),
            face_count=int(len(selection.mesh.faces)),
            bbox_min_lps_mm=display_min,
            bbox_max_lps_mm=display_max,
            selected_profile=selection.selected_profile,
            geometry_audit=selection.audit,
            failed_candidates=tuple(
                ThoraxFailedCandidateRecord(profile=name, audit=audit)
                for name, audit in selection.failed_candidates
            ),
            fallback_applied=selection.fallback_applied,
        ),
    )


def _asset_record_for_payload(name: str, payload: bytes) -> ThoraxAssetRecord:
    return ThoraxAssetRecord(
        path=name,
        byte_length=len(payload),
        sha256=hashlib.sha256(payload).hexdigest(),
    )


def _replace_path(source: Path, target: Path) -> Path:
    """Small seam for testing the best-effort multi-file publication rollback."""
    return source.replace(target)


def _publish_asset_set(output_dir: Path, payloads: dict[str, bytes]) -> None:
    """Stage, verify and publish a related asset set with rollback on failure.

    The report is supplied last by the caller and therefore acts as the commit
    marker. Consumers must validate the hashes in that report before use.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    token = uuid4().hex
    staged: dict[str, Path] = {}
    backups: dict[str, Path] = {}
    published: list[str] = []
    try:
        for name, payload in payloads.items():
            stage = output_dir / f".{name}.stage-{token}"
            written = stage.write_bytes(payload)
            actual = stage.stat().st_size if stage.is_file() else -1
            if written != len(payload) or actual != len(payload):
                raise OSError(
                    f"Staged asset write mismatch for {name}: "
                    f"expected {len(payload)} bytes, wrote {actual}"
                )
            staged[name] = stage

        for name in payloads:
            final = output_dir / name
            if final.exists():
                backup = output_dir / f".{name}.backup-{token}"
                _replace_path(final, backup)
                backups[name] = backup

        for name in payloads:
            _replace_path(staged[name], output_dir / name)
            published.append(name)
    except Exception:
        for name in reversed(published):
            (output_dir / name).unlink(missing_ok=True)
        for name, backup in backups.items():
            if backup.exists():
                _replace_path(backup, output_dir / name)
        raise
    finally:
        for stage in staged.values():
            stage.unlink(missing_ok=True)
        for backup in backups.values():
            backup.unlink(missing_ok=True)


def export_thorax_mesh_variants(
    masks_dir: Path,
    output_dir: Path,
) -> ThoraxAssetReport:
    """Export audited registration and display thorax meshes in DICOM LPS mm."""
    paths = _ordered_thorax_masks(Path(masks_dir))
    output_dir = Path(output_dir)
    registration_scene = trimesh.Scene()
    display_scene = trimesh.Scene()
    records: list[ThoraxLabelRecord] = []

    for path in paths:
        anatomy_kind = _anatomy_kind(path)
        registration_selection = _select_registration_surface(path, anatomy_kind)
        registration = registration_selection.surface
        registration_mesh = registration.as_trimesh()
        candidates = (
            (profile.name, extract_surface(path, profile).as_trimesh())
            for profile in DISPLAY_PROFILES[anatomy_kind]
        )
        selection = select_display_surface(registration_mesh, candidates, anatomy_kind)
        _add_colored_geometry(registration_scene, registration_mesh, path, anatomy_kind)
        _add_colored_geometry(display_scene, selection.mesh, path, anatomy_kind)
        records.append(
            _surface_record(path, anatomy_kind, registration_selection, selection)
        )

    registration_payload = trimesh.exchange.gltf.export_glb(registration_scene)
    display_payload = trimesh.exchange.gltf.export_glb(display_scene)
    registration_asset = _asset_record_for_payload(
        "thorax_registration.glb", registration_payload
    )
    display_asset = _asset_record_for_payload("thorax_display.glb", display_payload)
    report = ThoraxAssetReport(
        schema_version=2,
        coordinate_system="DICOM patient LPS",
        units="mm",
        display_to_registration_row_major=[
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        registration_asset=registration_asset,
        display_asset=display_asset,
        labels=tuple(records),
    )
    report_payload = (json.dumps(report.to_dict(), ensure_ascii=False, indent=2) + "\n").encode(
        "utf-8"
    )
    _publish_asset_set(
        output_dir,
        {
            registration_asset.path: registration_payload,
            display_asset.path: display_payload,
            "thorax_surface_report.json": report_payload,
        },
    )
    return report


__all__ = [
    "DISPLAY_PROFILES",
    "LEGACY_REGISTRATION_PROFILES",
    "RAW_PROFILE",
    "SdfCrop",
    "SurfaceAudit",
    "SurfaceProfile",
    "SurfaceResult",
    "SurfaceSelection",
    "SurfaceTolerance",
    "TOLERANCES",
    "ThoraxAssetRecord",
    "ThoraxAssetReport",
    "ThoraxDisplayRecord",
    "ThoraxFailedCandidateRecord",
    "ThoraxLabelRecord",
    "ThoraxRegistrationRecord",
    "_extract_sdf_isosurface",
    "_non_manifold_edge_count",
    "_signed_distance_crop",
    "audit_surface_pair",
    "extract_registration_surface",
    "extract_surface",
    "export_thorax_mesh_variants",
    "select_display_surface",
]
