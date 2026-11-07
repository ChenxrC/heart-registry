const disposedGeometries = new WeakSet();
const disposedMaterials = new WeakSet();
const unregisteredMeshes = new WeakSet();


function disposeGeometryOnce(geometry) {
  if (geometry?.dispose && !disposedGeometries.has(geometry)) {
    geometry.dispose();
    disposedGeometries.add(geometry);
  }
}


function hasOwn(value, key) {
  return Object.prototype.hasOwnProperty.call(value ?? {}, key);
}

function schemaV2Attempted(frame) {
  return hasOwn(frame, "nodes") || hasOwn(frame, "edges");
}

function stableTopologyId(value, label) {
  if (typeof value !== "string" || value.length === 0 || value.trim() !== value) {
    throw new TypeError(`${label}.topology_id must be an exact nonempty string`);
  }
  return value;
}

function finitePoint(value, label) {
  if (!Array.isArray(value) || value.length !== 3 || !value.every(Number.isFinite)) {
    throw new TypeError(`${label} must be a finite LPS [x,y,z] point`);
  }
  return value;
}

function pointsEqual(left, right) {
  return left.length === right.length && left.every((value, index) => value === right[index]);
}

function validateSchemaV2Frame(frame, { requireAccepted = true } = {}) {
  if (!frame || typeof frame !== "object" || Array.isArray(frame)) {
    throw new TypeError("Vessel frame must be an object");
  }
  if (!schemaV2Attempted(frame)) throw new TypeError("Schema-v2 nodes and edges are absent");
  if (!Array.isArray(frame.nodes) || !Array.isArray(frame.edges)) {
    throw new TypeError("Schema-v2 nodes and edges must both be arrays");
  }
  if (frame.nodes.length === 0 || frame.edges.length === 0) {
    throw new RangeError("Schema-v2 nodes and edges must be nonempty");
  }
  if (requireAccepted && frame.quality_status !== "accepted") {
    throw new RangeError("Schema-v2 display geometry requires an accepted frame");
  }

  const nodesByTopologyId = new Map();
  const nodesByLocalId = new Map();
  for (const [index, node] of frame.nodes.entries()) {
    if (!node || typeof node !== "object" || Array.isArray(node)) {
      throw new TypeError(`nodes[${index}] must be an object`);
    }
    const topologyId = stableTopologyId(node.topology_id, `nodes[${index}]`);
    if (nodesByTopologyId.has(topologyId)) throw new RangeError(`Duplicate node topology_id: ${topologyId}`);
    finitePoint(node.point_lps_mm, `nodes[${index}].point_lps_mm`);
    nodesByTopologyId.set(topologyId, node);
    if (typeof node.node_id === "string" && node.node_id.length > 0) {
      if (nodesByLocalId.has(node.node_id)) throw new RangeError(`Duplicate node_id: ${node.node_id}`);
      nodesByLocalId.set(node.node_id, node);
    }
  }

  const edgesByTopologyId = new Map();
  const incidentEndpointRadii = new Map(
    [...nodesByTopologyId.keys()].map((topologyId) => [topologyId, []]),
  );
  for (const [index, edge] of frame.edges.entries()) {
    if (!edge || typeof edge !== "object" || Array.isArray(edge)) {
      throw new TypeError(`edges[${index}] must be an object`);
    }
    const topologyId = stableTopologyId(edge.topology_id, `edges[${index}]`);
    if (edgesByTopologyId.has(topologyId)) throw new RangeError(`Duplicate edge topology_id: ${topologyId}`);
    const sourceTopologyId = stableTopologyId(
      edge.source_topology_id,
      `edges[${index}].source_topology_id`,
    );
    const targetTopologyId = stableTopologyId(
      edge.target_topology_id,
      `edges[${index}].target_topology_id`,
    );
    const sourceNode = nodesByTopologyId.get(sourceTopologyId);
    const targetNode = nodesByTopologyId.get(targetTopologyId);
    if (!sourceNode || !targetNode) {
      throw new RangeError(`Edge ${topologyId} source/target topology reference is missing`);
    }
    if (hasOwn(edge, "source_node_id") && nodesByLocalId.get(edge.source_node_id) !== sourceNode) {
      throw new RangeError(`Edge ${topologyId} source_node_id disagrees with source_topology_id`);
    }
    if (hasOwn(edge, "target_node_id") && nodesByLocalId.get(edge.target_node_id) !== targetNode) {
      throw new RangeError(`Edge ${topologyId} target_node_id disagrees with target_topology_id`);
    }
    if (!Array.isArray(edge.points_lps_mm) || edge.points_lps_mm.length < 2) {
      throw new RangeError(`Edge ${topologyId} points_lps_mm must contain at least two ordered samples`);
    }
    edge.points_lps_mm.forEach((point, pointIndex) => {
      finitePoint(point, `edges[${index}].points_lps_mm[${pointIndex}]`);
    });
    if (
      !Array.isArray(edge.radii_mm)
      || edge.radii_mm.length !== edge.points_lps_mm.length
      || !edge.radii_mm.every((radius) => Number.isFinite(radius) && radius > 0)
    ) {
      throw new RangeError(`Edge ${topologyId} radii_mm must be positive and match its sample count`);
    }
    if (!pointsEqual(edge.points_lps_mm[0], sourceNode.point_lps_mm)) {
      throw new RangeError(`Edge ${topologyId} source endpoint must equal its shared node exactly`);
    }
    if (!pointsEqual(edge.points_lps_mm.at(-1), targetNode.point_lps_mm)) {
      throw new RangeError(`Edge ${topologyId} target endpoint must equal its shared node exactly`);
    }
    incidentEndpointRadii.get(sourceTopologyId).push(edge.radii_mm[0]);
    incidentEndpointRadii.get(targetTopologyId).push(edge.radii_mm.at(-1));
    edgesByTopologyId.set(topologyId, edge);
  }

  for (const [topologyId, radii] of incidentEndpointRadii) {
    if (radii.length === 0) throw new RangeError(`Node ${topologyId} has no incident edge`);
  }

  return { nodesByTopologyId, edgesByTopologyId, incidentEndpointRadii };
}

function validateLegacyFrame(frame) {
  if (!frame || typeof frame !== "object" || Array.isArray(frame) || !Array.isArray(frame.paths)) {
    throw new TypeError("Legacy vessel frame paths must be an array");
  }
  if (frame.quality_status === "rejected") throw new RangeError("Rejected legacy frame is not displayable");
  return frame.paths.map((path, index) => {
    if (!path || typeof path !== "object" || !Array.isArray(path.points_lps_mm)) {
      throw new TypeError(`paths[${index}] must contain points_lps_mm`);
    }
    if (path.points_lps_mm.length < 2) throw new RangeError(`paths[${index}] needs at least two points`);
    path.points_lps_mm.forEach((point, pointIndex) => finitePoint(point, `paths[${index}][${pointIndex}]`));
    if (!Number.isFinite(path.radius_mm) || path.radius_mm <= 0) {
      throw new RangeError(`paths[${index}].radius_mm must be positive`);
    }
    return path;
  });
}

function requireThree(options) {
  const THREE = options?.THREE;
  for (const name of ["Group", "Mesh", "Vector3", "CatmullRomCurve3", "TubeGeometry", "SphereGeometry"]) {
    if (typeof THREE?.[name] !== "function") throw new TypeError(`options.THREE.${name} is required`);
  }
  return THREE;
}

function radiusAt(radii, t) {
  const sample = Math.max(0, Math.min(radii.length - 1, t * (radii.length - 1)));
  const lower = Math.floor(sample);
  const upper = Math.min(radii.length - 1, lower + 1);
  const alpha = sample - lower;
  return radii[lower] + (radii[upper] - radii[lower]) * alpha;
}

function makeVariableRadiusTube(THREE, points, radii, options) {
  const curve = new THREE.CatmullRomCurve3(points, false, "centripetal");
  const tubularSegments = Number.isInteger(options.tubularSegments)
    ? Math.max(2, options.tubularSegments)
    : Math.min(96, Math.max(4, (points.length - 1) * (options.tubularSegmentsPerSample ?? 2)));
  const radialSegments = Number.isInteger(options.radialSegments) ? Math.max(3, options.radialSegments) : 6;
  let geometry = null;
  try {
    geometry = new THREE.TubeGeometry(curve, tubularSegments, 1, radialSegments, false);
    const position = geometry.getAttribute("position");
    const center = new THREE.Vector3();
    for (let segment = 0; segment <= tubularSegments; segment += 1) {
      const t = segment / tubularSegments;
      curve.getPointAt(t, center);
      const radius = radiusAt(radii, t);
      for (let radial = 0; radial <= radialSegments; radial += 1) {
        const vertex = segment * (radialSegments + 1) + radial;
        position.setXYZ(
          vertex,
          center.x + (position.getX(vertex) - center.x) * radius,
          center.y + (position.getY(vertex) - center.y) * radius,
          center.z + (position.getZ(vertex) - center.z) * radius,
        );
      }
    }
    position.needsUpdate = true;
    geometry.computeVertexNormals();
    geometry.computeBoundingBox();
    geometry.computeBoundingSphere();
    geometry.userData.radiiMm = [...radii];
    geometry.userData.samplePointsLpsMm = points.map((point) => point.toArray());
    geometry.userData.radialSegments = radialSegments;
    geometry.userData.tubularSegments = tubularSegments;
    return geometry;
  } catch (error) {
    disposeGeometryOnce(geometry);
    throw error;
  }
}

function buildSchemaV2Group(frame, material, options, THREE) {
  const { incidentEndpointRadii } = validateSchemaV2Frame(frame);
  const group = new THREE.Group();
  group.name = `Connected vessel frame ${frame.frame_index ?? "unknown"}`;
  group.userData = {
    vesselSchemaVersion: 2,
    frameIndex: frame.frame_index ?? null,
    ownedMaterials: options.ownsMaterial === true ? [material] : [],
  };
  try {
    for (const edge of frame.edges) {
      const points = edge.points_lps_mm.map((point) => new THREE.Vector3(...point));
      const geometry = makeVariableRadiusTube(THREE, points, edge.radii_mm, options);
      try {
        const mesh = new THREE.Mesh(geometry, material);
        mesh.name = `Vessel edge ${edge.topology_id}`;
        mesh.userData = {
          vesselPrimitive: "edge-tube",
          topologyId: edge.topology_id,
          sourceTopologyId: edge.source_topology_id,
          targetTopologyId: edge.target_topology_id,
        };
        group.add(mesh);
      } catch (error) {
        disposeGeometryOnce(geometry);
        throw error;
      }
    }

    const radialSegments = Number.isInteger(options.radialSegments) ? Math.max(3, options.radialSegments) : 6;
    for (const node of frame.nodes) {
      const radius = Math.max(...incidentEndpointRadii.get(node.topology_id));
      const geometry = new THREE.SphereGeometry(radius, Math.max(8, radialSegments * 2), Math.max(6, radialSegments));
      try {
        geometry.userData.radiusMm = radius;
        const mesh = new THREE.Mesh(geometry, material);
        mesh.name = `Vessel node ${node.topology_id}`;
        mesh.position.set(...node.point_lps_mm);
        mesh.userData = {
          vesselPrimitive: "node-sphere",
          topologyId: node.topology_id,
          pointLpsMm: [...node.point_lps_mm],
          radiusMm: radius,
        };
        group.add(mesh);
      } catch (error) {
        disposeGeometryOnce(geometry);
        throw error;
      }
    }
  } catch (error) {
    disposeConnectedVesselGroup(group);
    throw error;
  }
  return group;
}

function buildLegacyGroup(frame, material, options, THREE) {
  const paths = validateLegacyFrame(frame);
  const group = new THREE.Group();
  group.name = `Legacy vessel frame ${frame.frame_index ?? "unknown"}`;
  group.userData = {
    vesselSchemaVersion: 1,
    frameIndex: frame.frame_index ?? null,
    ownedMaterials: options.ownsMaterial === true ? [material] : [],
  };
  const radialSegments = Number.isInteger(options.radialSegments) ? Math.max(3, options.radialSegments) : 6;
  try {
    for (const [index, path] of paths.entries()) {
      const points = path.points_lps_mm.map((point) => new THREE.Vector3(...point));
      const curve = new THREE.CatmullRomCurve3(points, false, "centripetal");
      const tubularSegments = Math.min(96, Math.max(4, (points.length - 1) * (options.tubularSegmentsPerSample ?? 2)));
      const geometry = new THREE.TubeGeometry(curve, tubularSegments, path.radius_mm, radialSegments, false);
      try {
        geometry.userData.radiiMm = Array(points.length).fill(path.radius_mm);
        geometry.userData.samplePointsLpsMm = points.map((point) => point.toArray());
        const mesh = new THREE.Mesh(geometry, material);
        const topologyId = path.topology_id ?? path.path_id ?? `legacy-path-${index}`;
        mesh.name = `Legacy vessel path ${topologyId}`;
        mesh.userData = { vesselPrimitive: "edge-tube", topologyId, legacyPathIndex: index };
        group.add(mesh);
      } catch (error) {
        disposeGeometryOnce(geometry);
        throw error;
      }
    }
  } catch (error) {
    disposeConnectedVesselGroup(group);
    throw error;
  }
  return group;
}

/** Build browser display geometry without importing Three.js through Node's package resolver. */
export function buildConnectedVesselGroup(frame, material, options = {}) {
  const THREE = requireThree(options);
  if (!material || typeof material !== "object") throw new TypeError("material is required");
  try {
    return schemaV2Attempted(frame)
      ? buildSchemaV2Group(frame, material, options, THREE)
      : buildLegacyGroup(frame, material, options, THREE);
  } catch (error) {
    if (
      options.ownsMaterial === true
      && material?.dispose
      && !disposedMaterials.has(material)
    ) {
      material.dispose();
      disposedMaterials.add(material);
    }
    throw error;
  }
}

function graphComponentCount(frame) {
  const { nodesByTopologyId } = validateSchemaV2Frame(frame);
  const adjacency = new Map([...nodesByTopologyId.keys()].map((topologyId) => [topologyId, new Set()]));
  for (const edge of frame.edges) {
    adjacency.get(edge.source_topology_id).add(edge.target_topology_id);
    adjacency.get(edge.target_topology_id).add(edge.source_topology_id);
  }
  let count = 0;
  const visited = new Set();
  for (const topologyId of adjacency.keys()) {
    if (visited.has(topologyId)) continue;
    count += 1;
    const pending = [topologyId];
    while (pending.length) {
      const current = pending.pop();
      if (visited.has(current)) continue;
      visited.add(current);
      for (const neighbor of adjacency.get(current)) pending.push(neighbor);
    }
  }
  return count;
}

function distance(left, right) {
  return Math.hypot(left[0] - right[0], left[1] - right[1], left[2] - right[2]);
}

function builtTubeEndpointCenter(mesh, endpointIndex) {
  const position = mesh.geometry?.getAttribute?.("position");
  const radialSegments = mesh.geometry?.userData?.radialSegments;
  const tubularSegments = mesh.geometry?.userData?.tubularSegments;
  if (
    !position
    || !Number.isInteger(radialSegments)
    || radialSegments < 3
    || !Number.isInteger(tubularSegments)
    || tubularSegments < 2
  ) return null;
  const segment = endpointIndex === 0 ? 0 : tubularSegments;
  const start = segment * (radialSegments + 1);
  const center = [0, 0, 0];
  for (let radial = 0; radial < radialSegments; radial += 1) {
    const vertex = start + radial;
    center[0] += position.getX(vertex);
    center[1] += position.getY(vertex);
    center[2] += position.getZ(vertex);
  }
  return center.map((value) => value / radialSegments);
}

/** Audit semantic graph connectivity and the actual built node/endpoint placement. */
export function auditVesselGeometry(group, frame) {
  const { nodesByTopologyId } = validateSchemaV2Frame(frame);
  if (!group || typeof group.traverse !== "function") throw new TypeError("group must be a Three.js Group");
  const nodeMeshes = new Map();
  const edgeMeshes = new Map();
  group.traverse((object) => {
    if (object?.userData?.vesselPrimitive === "node-sphere") nodeMeshes.set(object.userData.topologyId, object);
    if (object?.userData?.vesselPrimitive === "edge-tube") edgeMeshes.set(object.userData.topologyId, object);
  });

  const missingTopologyIds = new Set();
  for (const topologyId of nodesByTopologyId.keys()) {
    if (!nodeMeshes.has(topologyId)) missingTopologyIds.add(topologyId);
  }
  for (const edge of frame.edges) {
    if (!edgeMeshes.has(edge.topology_id)) missingTopologyIds.add(edge.topology_id);
  }

  let maxJunctionGapMm = 0;
  for (const edge of frame.edges) {
    const edgeMesh = edgeMeshes.get(edge.topology_id);
    if (!edgeMesh) continue;
    for (const [endpointIndex, nodeTopologyId] of [edge.source_topology_id, edge.target_topology_id].entries()) {
      const nodeMesh = nodeMeshes.get(nodeTopologyId);
      const builtEndpoint = builtTubeEndpointCenter(edgeMesh, endpointIndex);
      if (!builtEndpoint) {
        missingTopologyIds.add(edge.topology_id);
        continue;
      }
      if (!nodeMesh) continue;
      const nodePosition = [nodeMesh.position.x, nodeMesh.position.y, nodeMesh.position.z];
      const authoritativePosition = nodesByTopologyId.get(nodeTopologyId).point_lps_mm;
      maxJunctionGapMm = Math.max(
        maxJunctionGapMm,
        distance(builtEndpoint, nodePosition),
        distance(nodePosition, authoritativePosition),
      );
    }
  }
  return {
    graphComponentCount: graphComponentCount(frame),
    maxJunctionGapMm,
    missingTopologyIds: [...missingTopologyIds].sort(),
  };
}

export function schemaV2FramesHaveMatchingTopology(fromFrame, toFrame) {
  try {
    const from = validateSchemaV2Frame(fromFrame);
    const to = validateSchemaV2Frame(toFrame);
    if (from.nodesByTopologyId.size !== to.nodesByTopologyId.size) return false;
    if (from.edgesByTopologyId.size !== to.edgesByTopologyId.size) return false;
    for (const topologyId of from.nodesByTopologyId.keys()) {
      if (!to.nodesByTopologyId.has(topologyId)) return false;
    }
    for (const [topologyId, edge] of from.edgesByTopologyId) {
      const candidate = to.edgesByTopologyId.get(topologyId);
      if (
        !candidate
        || edge.source_topology_id !== candidate.source_topology_id
        || edge.target_topology_id !== candidate.target_topology_id
        || edge.points_lps_mm.length !== candidate.points_lps_mm.length
        || edge.radii_mm.length !== candidate.radii_mm.length
      ) return false;
    }
    return true;
  } catch (error) {
    if (error instanceof TypeError || error instanceof RangeError) return false;
    throw error;
  }
}

export function isAcceptedVesselFrame(frame) {
  if (schemaV2Attempted(frame)) {
    try {
      validateSchemaV2Frame(frame);
      return true;
    } catch (error) {
      if (error instanceof TypeError || error instanceof RangeError) return false;
      throw error;
    }
  }
  try {
    validateLegacyFrame(frame);
    return true;
  } catch (error) {
    if (error instanceof TypeError || error instanceof RangeError) return false;
    throw error;
  }
}

function interpolatePoint(from, to, alpha) {
  return from.map((value, index) => value + (to[index] - value) * alpha);
}

/** Interpolate authoritative graph samples by stable topology ID and re-pin shared endpoints. */
export function interpolateVesselFrame(fromFrame, toFrame, alpha) {
  if (!Number.isFinite(alpha) || alpha < 0 || alpha > 1) throw new RangeError("alpha must be finite in [0,1]");
  if (!schemaV2FramesHaveMatchingTopology(fromFrame, toFrame)) {
    throw new RangeError("Accepted schema-v2 vessel frames must have matching stable topology");
  }
  const toNodes = new Map(toFrame.nodes.map((node) => [node.topology_id, node]));
  const toEdges = new Map(toFrame.edges.map((edge) => [edge.topology_id, edge]));
  const nodes = fromFrame.nodes.map((node) => ({
    ...node,
    point_lps_mm: interpolatePoint(node.point_lps_mm, toNodes.get(node.topology_id).point_lps_mm, alpha),
  }));
  const nodePoints = new Map(nodes.map((node) => [node.topology_id, node.point_lps_mm]));
  const edges = fromFrame.edges.map((edge) => {
    const candidate = toEdges.get(edge.topology_id);
    const points = edge.points_lps_mm.map((point, index) => (
      interpolatePoint(point, candidate.points_lps_mm[index], alpha)
    ));
    points[0] = [...nodePoints.get(edge.source_topology_id)];
    points[points.length - 1] = [...nodePoints.get(edge.target_topology_id)];
    return {
      ...edge,
      points_lps_mm: points,
      radii_mm: edge.radii_mm.map((radius, index) => (
        radius + (candidate.radii_mm[index] - radius) * alpha
      )),
    };
  });
  return {
    ...fromFrame,
    frame_index: fromFrame.frame_index ?? null,
    quality_status: "accepted",
    display_source_frame_index: null,
    nodes,
    edges,
    paths: [],
    path_count: edges.length,
    point_count: edges.reduce((sum, edge) => sum + edge.points_lps_mm.length, 0),
    interpolation: {
      from_frame_index: fromFrame.frame_index ?? null,
      to_frame_index: toFrame.frame_index ?? null,
      alpha,
    },
  };
}

/** Unregister before releasing each generated geometry and each group-owned material exactly once. */
export function disposeConnectedVesselGroup(group, { unregister } = {}) {
  if (!group || typeof group.traverse !== "function") return;
  const meshes = [];
  group.traverse((object) => {
    if (object?.isMesh) meshes.push(object);
  });
  for (const mesh of meshes) {
    if (typeof unregister === "function" && !unregisteredMeshes.has(mesh)) {
      unregister(mesh);
      unregisteredMeshes.add(mesh);
    }
    disposeGeometryOnce(mesh.geometry);
  }
  for (const material of group.userData?.ownedMaterials ?? []) {
    if (material?.dispose && !disposedMaterials.has(material)) {
      material.dispose();
      disposedMaterials.add(material);
    }
  }
  group.parent?.remove(group);
  group.clear?.();
  group.userData.disposed = true;
}
