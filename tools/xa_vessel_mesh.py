from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from numbers import Real
from typing import Any

import numpy as np
import trimesh
from scipy.spatial import cKDTree
from skimage.measure import marching_cubes


_MERGE_TOLERANCE_MM = 0.05
_MAX_GRID_VOXELS = 40_000_000


@dataclass(frozen=True)
class _EdgeGeometry:
    points: np.ndarray
    radii: np.ndarray


@dataclass(frozen=True)
class _ValidatedGeometry:
    edges: tuple[_EdgeGeometry, ...]
    node_spheres: tuple[tuple[np.ndarray, float], ...]


def _stable_id(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{label} must be an exact nonempty topology_id")
    return value


def _records(value: Any, label: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(f"{label} must be an array")
    return value


def _vector3(value: Any, label: str) -> np.ndarray:
    raw = np.asarray(value, dtype=object)
    if raw.shape != (3,) or any(
        isinstance(item, (bool, np.bool_)) or not isinstance(item, Real)
        for item in raw.flat
    ):
        raise ValueError(f"{label} must contain numeric coordinates")
    try:
        vector = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label} must contain numeric coordinates") from exc
    if vector.shape != (3,) or not np.all(np.isfinite(vector)):
        raise ValueError(f"{label} must be one finite LPS coordinate")
    return vector.copy()


def _points(value: Any, label: str) -> np.ndarray:
    raw = np.asarray(value, dtype=object)
    if raw.ndim != 2 or raw.shape[0] < 2 or raw.shape[1] != 3 or any(
        isinstance(item, (bool, np.bool_)) or not isinstance(item, Real)
        for item in raw.flat
    ):
        raise ValueError(f"{label} must contain at least two numeric LPS coordinates")
    try:
        points = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label} must contain numeric coordinates") from exc
    if (
        points.ndim != 2
        or points.shape[0] < 2
        or points.shape[1] != 3
        or not np.all(np.isfinite(points))
    ):
        raise ValueError(f"{label} must contain at least two finite LPS coordinates")
    return points.copy()


def _positive_radii(value: Any, count: int, label: str) -> np.ndarray:
    raw = np.asarray(value, dtype=object)
    if raw.shape != (count,) or any(
        isinstance(item, (bool, np.bool_)) or not isinstance(item, Real)
        for item in raw.flat
    ):
        raise ValueError(f"{label} must contain one numeric radius per point")
    try:
        radii = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label} must contain numeric radii") from exc
    if (
        radii.shape != (count,)
        or not np.all(np.isfinite(radii))
        or not np.all(radii > 0.0)
    ):
        raise ValueError(f"{label} must contain one finite positive radius per point")
    return radii.copy()


def _require_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    return value


def _require_one_component(
    node_ids: Sequence[str], edge_endpoints: Sequence[tuple[str, str]]
) -> None:
    if not node_ids or not edge_endpoints:
        raise ValueError("vessel graph must contain nodes and edges")
    adjacency = {node_id: set() for node_id in node_ids}
    for source, target in edge_endpoints:
        adjacency[source].add(target)
        adjacency[target].add(source)
    visited: set[str] = set()
    stack = [node_ids[0]]
    while stack:
        current = stack.pop()
        if current in visited:
            continue
        visited.add(current)
        stack.extend(adjacency[current] - visited)
    if len(visited) != len(adjacency):
        raise ValueError("vessel graph must contain exactly one connected component")


def _validate_schema_v2(frame: Mapping[str, Any]) -> _ValidatedGeometry:
    if "nodes" not in frame or "edges" not in frame:
        raise ValueError("schema-v2 frames require both nodes and edges")
    node_records = _records(frame["nodes"], "nodes")
    edge_records = _records(frame["edges"], "edges")
    if not node_records or not edge_records:
        raise ValueError("vessel graph must contain nodes and edges")

    nodes: dict[str, np.ndarray] = {}
    for index, raw_node in enumerate(node_records):
        node = _require_mapping(raw_node, f"nodes[{index}]")
        topology_id = _stable_id(node.get("topology_id"), f"nodes[{index}]")
        if topology_id in nodes:
            raise ValueError("node topology_id values must be unique")
        nodes[topology_id] = _vector3(
            node.get("point_lps_mm"), f"nodes[{index}].point_lps_mm"
        )

    edge_ids: set[str] = set()
    edge_endpoints: list[tuple[str, str]] = []
    edges: list[_EdgeGeometry] = []
    incident_radii: dict[str, list[float]] = {node_id: [] for node_id in nodes}
    for index, raw_edge in enumerate(edge_records):
        edge = _require_mapping(raw_edge, f"edges[{index}]")
        topology_id = _stable_id(edge.get("topology_id"), f"edges[{index}]")
        if topology_id in edge_ids:
            raise ValueError("edge topology_id values must be unique")
        edge_ids.add(topology_id)
        source = _stable_id(
            edge.get("source_topology_id"),
            f"edges[{index}].source_topology_id",
        )
        target = _stable_id(
            edge.get("target_topology_id"),
            f"edges[{index}].target_topology_id",
        )
        if source not in nodes or target not in nodes:
            raise ValueError("edge topology references must identify existing nodes")
        points = _points(edge.get("points_lps_mm"), f"edges[{index}].points_lps_mm")
        radii = _positive_radii(
            edge.get("radii_mm"), len(points), f"edges[{index}].radii_mm"
        )
        if not np.array_equal(points[0], nodes[source]):
            raise ValueError("edge first point must exactly equal its source node")
        if not np.array_equal(points[-1], nodes[target]):
            raise ValueError("edge last point must exactly equal its target node")
        edges.append(_EdgeGeometry(points=points, radii=radii))
        edge_endpoints.append((source, target))
        incident_radii[source].append(float(radii[0]))
        incident_radii[target].append(float(radii[-1]))

    node_ids = list(nodes)
    _require_one_component(node_ids, edge_endpoints)
    node_spheres = []
    for node_id, point in nodes.items():
        radii = incident_radii[node_id]
        if not radii:
            raise ValueError("every node must be incident to an edge")
        node_spheres.append((point, max(radii)))
    return _ValidatedGeometry(tuple(edges), tuple(node_spheres))


def _validate_legacy_paths(frame: Mapping[str, Any]) -> _ValidatedGeometry:
    path_records = _records(frame.get("paths"), "paths")
    if not path_records:
        raise ValueError("legacy frame must contain at least one path")

    edges: list[_EdgeGeometry] = []
    endpoints: list[tuple[str, str]] = []
    points_by_id: dict[str, np.ndarray] = {}
    incident_radii: dict[str, list[float]] = {}

    def endpoint_id(point: np.ndarray) -> str:
        return "|".join(
            (0.0 if float(value) == 0.0 else float(value)).hex()
            for value in point
        )

    for index, raw_path in enumerate(path_records):
        path = _require_mapping(raw_path, f"paths[{index}]")
        points = _points(path.get("points_lps_mm"), f"paths[{index}].points_lps_mm")
        if "radii_mm" in path:
            radii = _positive_radii(
                path["radii_mm"], len(points), f"paths[{index}].radii_mm"
            )
        else:
            raw_radius = path.get("radius_mm")
            if isinstance(raw_radius, (bool, np.bool_)) or not isinstance(
                raw_radius, Real
            ):
                raise ValueError(f"paths[{index}].radius_mm must be numeric")
            try:
                radius = float(raw_radius)
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValueError(f"paths[{index}].radius_mm must be positive") from exc
            if not np.isfinite(radius) or radius <= 0.0:
                raise ValueError(f"paths[{index}].radius_mm must be positive")
            radii = np.full(len(points), radius, dtype=np.float64)
        source = endpoint_id(points[0])
        target = endpoint_id(points[-1])
        points_by_id.setdefault(source, points[0])
        points_by_id.setdefault(target, points[-1])
        incident_radii.setdefault(source, []).append(float(radii[0]))
        incident_radii.setdefault(target, []).append(float(radii[-1]))
        endpoints.append((source, target))
        edges.append(_EdgeGeometry(points=points, radii=radii))

    node_ids = list(points_by_id)
    _require_one_component(node_ids, endpoints)
    node_spheres = tuple(
        (points_by_id[node_id], max(incident_radii[node_id])) for node_id in node_ids
    )
    return _ValidatedGeometry(tuple(edges), node_spheres)


def _validate_frame(frame: Any) -> _ValidatedGeometry:
    frame_record = _require_mapping(frame, "frame")
    if "nodes" in frame_record or "edges" in frame_record:
        if frame_record.get("quality_status") != "accepted":
            raise ValueError("schema-v2 vessel surface requires an accepted frame")
        return _validate_schema_v2(frame_record)
    if frame_record.get("quality_status") == "rejected":
        raise ValueError("rejected legacy vessel frame is not displayable")
    return _validate_legacy_paths(frame_record)


def _grid_bounds(
    geometry: _ValidatedGeometry, voxel_size_mm: float
) -> tuple[np.ndarray, tuple[int, int, int]]:
    lower = np.full(3, np.inf, dtype=np.float64)
    upper = np.full(3, -np.inf, dtype=np.float64)
    for edge in geometry.edges:
        lower = np.minimum(lower, np.min(edge.points - edge.radii[:, None], axis=0))
        upper = np.maximum(upper, np.max(edge.points + edge.radii[:, None], axis=0))
    for point, radius in geometry.node_spheres:
        lower = np.minimum(lower, point - radius)
        upper = np.maximum(upper, point + radius)
    padding = 2.0 * voxel_size_mm
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        lower_indices = np.floor((lower - padding) / voxel_size_mm)
        upper_indices = np.ceil((upper + padding) / voxel_size_mm)
    if not np.all(np.isfinite(lower_indices)) or not np.all(np.isfinite(upper_indices)):
        raise ValueError("vessel surface grid bounds are not finite")
    # Beyond 2**52 a float cannot represent every integer grid coordinate, so
    # origin/shape arithmetic would no longer be trustworthy even before the
    # allocation cap is considered.
    if np.any(np.abs(lower_indices) > 2**52) or np.any(np.abs(upper_indices) > 2**52):
        raise ValueError("vessel surface grid coordinates exceed the safe range")

    shape: list[int] = []
    voxel_count = 1
    for lower_index, upper_index in zip(lower_indices, upper_indices, strict=True):
        dimension_float = float(upper_index - lower_index + 1.0)
        if not np.isfinite(dimension_float) or dimension_float < 1.0:
            raise ValueError("vessel surface grid dimensions are invalid")
        if dimension_float > _MAX_GRID_VOXELS:
            raise ValueError("vessel surface grid exceeds the safe offline size")
        dimension = max(3, int(dimension_float))
        if voxel_count > _MAX_GRID_VOXELS // dimension:
            raise ValueError("vessel surface grid exceeds the safe offline size")
        voxel_count *= dimension
        shape.append(dimension)

    origin = lower_indices * voxel_size_mm
    if not np.all(np.isfinite(origin)):
        raise ValueError("vessel surface grid origin is not finite")
    return origin, tuple(shape)


def _local_slices(
    center_min: np.ndarray,
    center_max: np.ndarray,
    radius: float,
    origin: np.ndarray,
    shape: tuple[int, int, int],
    voxel_size_mm: float,
) -> tuple[slice, slice, slice]:
    low = np.floor((center_min - radius - origin) / voxel_size_mm).astype(np.int64)
    high = np.ceil((center_max + radius - origin) / voxel_size_mm).astype(np.int64)
    low = np.maximum(low, 0)
    high = np.minimum(high, np.asarray(shape, dtype=np.int64) - 1)
    return tuple(slice(int(low[axis]), int(high[axis]) + 1) for axis in range(3))


def _local_coordinates(
    slices: tuple[slice, slice, slice], origin: np.ndarray, voxel_size_mm: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    axes = [
        origin[axis]
        + np.arange(slices[axis].start, slices[axis].stop, dtype=np.float64)
        * voxel_size_mm
        for axis in range(3)
    ]
    return np.meshgrid(*axes, indexing="ij", sparse=True)


def _union_sphere(
    field: np.ndarray,
    point: np.ndarray,
    radius: float,
    origin: np.ndarray,
    voxel_size_mm: float,
) -> None:
    slices = _local_slices(
        point, point, radius + voxel_size_mm, origin, field.shape, voxel_size_mm
    )
    x, y, z = _local_coordinates(slices, origin, voxel_size_mm)
    distance = np.sqrt(
        (x - point[0]) ** 2 + (y - point[1]) ** 2 + (z - point[2]) ** 2
    ) - radius
    field[slices] = np.minimum(field[slices], distance)


def _union_capsule_segment(
    field: np.ndarray,
    start: np.ndarray,
    end: np.ndarray,
    start_radius: float,
    end_radius: float,
    origin: np.ndarray,
    voxel_size_mm: float,
) -> None:
    radius = max(start_radius, end_radius) + voxel_size_mm
    slices = _local_slices(
        np.minimum(start, end),
        np.maximum(start, end),
        radius,
        origin,
        field.shape,
        voxel_size_mm,
    )
    x, y, z = _local_coordinates(slices, origin, voxel_size_mm)
    direction = end - start
    length_squared = float(np.dot(direction, direction))
    if length_squared <= np.finfo(np.float64).eps:
        distance = np.sqrt(
            (x - start[0]) ** 2
            + (y - start[1]) ** 2
            + (z - start[2]) ** 2
        ) - max(start_radius, end_radius)
    else:
        projection = (
            (x - start[0]) * direction[0]
            + (y - start[1]) * direction[1]
            + (z - start[2]) * direction[2]
        ) / length_squared
        projection = np.clip(projection, 0.0, 1.0)
        center_x = start[0] + projection * direction[0]
        center_y = start[1] + projection * direction[1]
        center_z = start[2] + projection * direction[2]
        local_radius = start_radius + projection * (end_radius - start_radius)
        distance = np.sqrt(
            (x - center_x) ** 2 + (y - center_y) ** 2 + (z - center_z) ** 2
        ) - local_radius
    field[slices] = np.minimum(field[slices], distance)


def _implicit_field(
    geometry: _ValidatedGeometry, voxel_size_mm: float
) -> tuple[np.ndarray, np.ndarray]:
    origin, shape = _grid_bounds(geometry, voxel_size_mm)
    outside = float(np.linalg.norm(np.asarray(shape) * voxel_size_mm) + 1.0)
    field = np.full(shape, outside, dtype=np.float64)
    for edge in geometry.edges:
        for index in range(len(edge.points) - 1):
            _union_capsule_segment(
                field,
                edge.points[index],
                edge.points[index + 1],
                float(edge.radii[index]),
                float(edge.radii[index + 1]),
                origin,
                voxel_size_mm,
            )
    for point, radius in geometry.node_spheres:
        _union_sphere(field, point, radius, origin, voxel_size_mm)
    return field, origin


def _merge_vertices(
    vertices: np.ndarray, faces: np.ndarray, tolerance_mm: float
) -> tuple[np.ndarray, np.ndarray]:
    pairs = cKDTree(vertices).query_pairs(tolerance_mm, output_type="ndarray")
    if len(pairs) == 0:
        return vertices, faces
    distances = np.linalg.norm(vertices[pairs[:, 0]] - vertices[pairs[:, 1]], axis=1)
    pairs = pairs[np.argsort(distances, kind="stable")]

    neighbors = [set() for _ in range(len(vertices))]
    for first, second, third in faces:
        first = int(first)
        second = int(second)
        third = int(third)
        neighbors[first].update((second, third))
        neighbors[second].update((first, third))
        neighbors[third].update((first, second))

    # Collapse only mutually isolated short manifold edges.  The link condition
    # preserves a closed triangular surface; blocking each selected edge's star
    # prevents otherwise-safe neighboring collapses from interacting.
    parent = np.arange(len(vertices), dtype=np.int64)
    blocked: set[int] = set()
    for raw_first, raw_second in pairs:
        first = int(raw_first)
        second = int(raw_second)
        if second not in neighbors[first]:
            continue
        common_neighbors = neighbors[first] & neighbors[second]
        if len(common_neighbors) != 2:
            continue
        star = {first, second, *neighbors[first], *neighbors[second]}
        if star & blocked:
            continue
        parent[second] = first
        blocked.update(star)

    _, inverse = np.unique(parent, return_inverse=True)
    merged = np.zeros((int(inverse.max()) + 1, 3), dtype=np.float64)
    np.add.at(merged, inverse, vertices)
    counts = np.bincount(inverse)
    merged /= counts[:, None]
    return merged, inverse[faces]


def _clean_mesh(vertices: np.ndarray, faces: np.ndarray) -> trimesh.Trimesh:
    vertices, faces = _merge_vertices(vertices, faces, _MERGE_TOLERANCE_MM)
    distinct = (
        (faces[:, 0] != faces[:, 1])
        & (faces[:, 1] != faces[:, 2])
        & (faces[:, 0] != faces[:, 2])
    )
    faces = faces[distinct]
    if len(faces) == 0:
        raise ValueError("polygonization produced no nondegenerate faces")
    triangle_vertices = vertices[faces]
    doubled_area = np.linalg.norm(
        np.cross(
            triangle_vertices[:, 1] - triangle_vertices[:, 0],
            triangle_vertices[:, 2] - triangle_vertices[:, 0],
        ),
        axis=1,
    )
    faces = faces[doubled_area > np.finfo(np.float64).eps]
    canonical_faces = np.sort(faces, axis=1)
    _, unique_indices = np.unique(canonical_faces, axis=0, return_index=True)
    faces = faces[np.sort(unique_indices)]
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    mesh.remove_unreferenced_vertices()
    trimesh.repair.fix_normals(mesh, multibody=False)
    return mesh


def build_watertight_vessel_surface(
    frame: Mapping[str, Any], voxel_size_mm: float = 0.35
) -> trimesh.Trimesh:
    """Build the authoritative offline vessel surface for one accepted frame."""
    if isinstance(voxel_size_mm, (bool, np.bool_)) or not isinstance(
        voxel_size_mm, Real
    ):
        raise ValueError("voxel_size_mm must be a finite positive number")
    voxel_size = float(voxel_size_mm)
    if not np.isfinite(voxel_size) or voxel_size <= 0.0:
        raise ValueError("voxel_size_mm must be finite and positive")

    geometry = _validate_frame(frame)
    field, origin = _implicit_field(geometry, voxel_size)
    if not np.any(field < 0.0) or not np.any(field > 0.0):
        raise ValueError("implicit vessel surface does not cross the zero level")
    try:
        vertices, faces, _normals, _values = marching_cubes(
            field,
            level=0.0,
            spacing=(voxel_size, voxel_size, voxel_size),
            allow_degenerate=False,
        )
    except (RuntimeError, ValueError) as exc:
        raise ValueError("implicit vessel surface could not be polygonized") from exc
    mesh = _clean_mesh(vertices + origin, faces)
    components = mesh.split(only_watertight=False)
    if len(components) != 1 or not mesh.is_watertight:
        raise ValueError("vessel surface must be one watertight component")
    return mesh
