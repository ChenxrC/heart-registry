import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

import * as motion from "../../web/ct_dr_3d_viewer/multiseries_motion.js";


function frameBlendAtElapsed(...args) {
  assert.equal(typeof motion.frameBlendAtElapsed, "function", "frameBlendAtElapsed export is missing");
  return motion.frameBlendAtElapsed(...args);
}

function vesselFramesHaveMatchingTopology(...args) {
  assert.equal(
    typeof motion.vesselFramesHaveMatchingTopology,
    "function",
    "vesselFramesHaveMatchingTopology export is missing",
  );
  return motion.vesselFramesHaveMatchingTopology(...args);
}

function resolveVesselFrameBlend(...args) {
  assert.equal(
    typeof motion.resolveVesselFrameBlend,
    "function",
    "resolveVesselFrameBlend export is missing",
  );
  return motion.resolveVesselFrameBlend(...args);
}


test("frame blend returns a safe sentinel for no frames", () => {
  assert.deepEqual(frameBlendAtElapsed([], 25), {
    fromIndex: -1,
    toIndex: -1,
    alpha: 0,
  });
});

test("frame blend always returns the only observed frame", () => {
  for (const elapsedMs of [-100, 0, 20, 100, Number.NaN]) {
    assert.deepEqual(frameBlendAtElapsed([40], elapsedMs), {
      fromIndex: 0,
      toIndex: 0,
      alpha: 0,
    });
  }
});

test("frame blend preserves exact irregular-duration keyframes", () => {
  const frameTimesMs = [100, 200, 50];
  assert.deepEqual(frameBlendAtElapsed(frameTimesMs, 0), { fromIndex: 0, toIndex: 0, alpha: 0 });
  assert.deepEqual(frameBlendAtElapsed(frameTimesMs, 100), { fromIndex: 1, toIndex: 1, alpha: 0 });
  assert.deepEqual(frameBlendAtElapsed(frameTimesMs, 300), { fromIndex: 2, toIndex: 2, alpha: 0 });
  assert.deepEqual(frameBlendAtElapsed(frameTimesMs, 350), { fromIndex: 0, toIndex: 0, alpha: 0 });
});

test("frame blend uses adjacent frames and DICOM-derived interval durations", () => {
  const frameTimesMs = [100, 200, 50];
  assert.deepEqual(frameBlendAtElapsed(frameTimesMs, 50), { fromIndex: 0, toIndex: 1, alpha: 0.5 });
  assert.deepEqual(frameBlendAtElapsed(frameTimesMs, 200), { fromIndex: 1, toIndex: 2, alpha: 0.5 });
  assert.deepEqual(frameBlendAtElapsed(frameTimesMs, 325), { fromIndex: 2, toIndex: 0, alpha: 0.5 });
});

test("frame blend wraps negative elapsed time and sanitizes invalid durations", () => {
  assert.deepEqual(frameBlendAtElapsed([100, 200, 50], -25), { fromIndex: 2, toIndex: 0, alpha: 0.5 });
  assert.deepEqual(frameBlendAtElapsed([0, Number.NaN, -10], 1), { fromIndex: 1, toIndex: 1, alpha: 0 });
  assert.deepEqual(frameBlendAtElapsed([0, Number.NaN, -10], 1.5), { fromIndex: 1, toIndex: 2, alpha: 0.5 });
  assert.deepEqual(frameBlendAtElapsed([25, 25], Number.POSITIVE_INFINITY), { fromIndex: 0, toIndex: 0, alpha: 0 });
});

function schemaFrame({ frameIndex = 0, status = "accepted", edgeTopology = "edge-main" } = {}) {
  return {
    frame_index: frameIndex,
    quality_status: status,
    display_source_frame_index: status === "accepted" ? frameIndex : null,
    nodes: [
      { node_id: "local-a", topology_id: "node-a", point_lps_mm: [0, 0, frameIndex] },
      { node_id: "local-b", topology_id: "node-b", point_lps_mm: [2, 0, frameIndex] },
    ],
    edges: [{
      edge_id: "local-edge",
      topology_id: edgeTopology,
      source_node_id: "local-a",
      target_node_id: "local-b",
      source_topology_id: "node-a",
      target_topology_id: "node-b",
      points_lps_mm: [[0, 0, frameIndex], [1, 0, frameIndex], [2, 0, frameIndex]],
      radii_mm: [1, 0.9, 0.8],
    }],
  };
}

test("schema-v2 topology matches by stable IDs independent of node and edge array order", () => {
  const from = schemaFrame({ frameIndex: 0 });
  const to = schemaFrame({ frameIndex: 1 });
  to.nodes.reverse();
  to.edges.reverse();
  assert.equal(vesselFramesHaveMatchingTopology(from, to), true);

  const changedEndpoint = structuredClone(to);
  changedEndpoint.edges[0].source_topology_id = "node-b";
  assert.equal(vesselFramesHaveMatchingTopology(from, changedEndpoint), false);

  const changedSampleCount = structuredClone(to);
  changedSampleCount.edges[0].points_lps_mm.splice(1, 1);
  changedSampleCount.edges[0].radii_mm.splice(1, 1);
  assert.equal(vesselFramesHaveMatchingTopology(from, changedSampleCount), false);
  assert.equal(vesselFramesHaveMatchingTopology(from, schemaFrame({ frameIndex: 1, status: "rejected" })), false);
  assert.equal(vesselFramesHaveMatchingTopology(from, { paths: [] }), false, "mixed schema must fail closed");
});

test("legacy topology compatibility remains strict and array-order dependent", () => {
  const pathA = { path_id: "a", topology_id: "root-a", points_lps_mm: [[0, 0, 0], [1, 0, 0]] };
  const pathB = { path_id: "b", topology_id: "root-b", points_lps_mm: [[0, 1, 0], [1, 1, 0]] };
  const movedA = { ...pathA, points_lps_mm: [[0, 0, 1], [1, 0, 1]] };
  const movedB = { ...pathB, points_lps_mm: [[0, 1, 1], [1, 1, 1]] };

  assert.equal(vesselFramesHaveMatchingTopology({ paths: [pathA, pathB] }, { paths: [movedA, movedB] }), true);
  assert.equal(vesselFramesHaveMatchingTopology({ paths: [pathA, pathB] }, { paths: [movedB, movedA] }), false);
});

test("matching legacy frames use a nearest keyframe instead of schema-v2 interpolation", () => {
  const frames = [0, 1].map((z) => ({
    paths: [{
      path_id: "main",
      topology_id: "legacy-main",
      points_lps_mm: [[0, 0, z], [1, 0, z]],
      radius_mm: 1,
    }],
  }));
  assert.deepEqual(
    resolveVesselFrameBlend(frames, { fromIndex: 0, toIndex: 1, alpha: 0.75 }),
    { mode: "fallback", frameIndex: 1, reason: "legacy-keyframe" },
  );
  const unidentified = frames.map((frame) => ({
    paths: frame.paths.map(({ points_lps_mm, radius_mm }) => ({ points_lps_mm, radius_mm })),
  }));
  assert.deepEqual(
    resolveVesselFrameBlend(unidentified, { fromIndex: 0, toIndex: 1, alpha: 0.75 }),
    { mode: "fallback", frameIndex: 1, reason: "topology-change" },
  );
});

test("rejected and topology-change blends resolve to deterministic accepted observations", () => {
  const frames = [
    schemaFrame({ frameIndex: 0 }),
    schemaFrame({ frameIndex: 1, status: "rejected" }),
    schemaFrame({ frameIndex: 2, edgeTopology: "edge-changed" }),
    schemaFrame({ frameIndex: 3, status: "rejected" }),
    schemaFrame({ frameIndex: 4 }),
  ];
  frames[1].display_source_frame_index = 0;
  frames[3].display_source_frame_index = 4;

  assert.deepEqual(
    resolveVesselFrameBlend(frames, { fromIndex: 1, toIndex: 1, alpha: 0 }),
    { mode: "fallback", frameIndex: 0, reason: "rejected-frame" },
  );
  assert.deepEqual(
    resolveVesselFrameBlend(frames, { fromIndex: 0, toIndex: 2, alpha: 0.75 }),
    { mode: "fallback", frameIndex: 2, reason: "topology-change" },
  );
  assert.deepEqual(
    resolveVesselFrameBlend(frames, { fromIndex: 3, toIndex: 3, alpha: 0 }),
    { mode: "fallback", frameIndex: 4, reason: "rejected-frame" },
  );

  frames[3].display_source_frame_index = 99;
  assert.deepEqual(
    resolveVesselFrameBlend(frames, { fromIndex: 3, toIndex: 3, alpha: 0 }),
    { mode: "fallback", frameIndex: 2, reason: "rejected-frame" },
    "nearest accepted search must prefer the earlier frame on a tie",
  );
});

test("app delegates observed frames to connected vessel group construction", async () => {
  const source = await readFile(
    new URL("../../web/ct_dr_3d_viewer/app.js", import.meta.url),
    "utf8",
  );
  const start = source.indexOf("function buildRunVesselFrame(");
  const end = source.indexOf("\nfunction updateRunVesselFrame(", start);
  assert.notEqual(start, -1, "buildRunVesselFrame is missing");
  assert.notEqual(end, -1, "buildRunVesselFrame boundary is missing");
  const build = source.slice(start, end);

  assert.match(build, /buildConnectedVesselGroup\(/);
  assert.match(build, /registerVesselGroup\(/);
  assert.doesNotMatch(build, /mergeGeometries|new THREE\.TubeGeometry/);
  assert.match(build, /visible = false/);
});

test("a newly cached observed frame remains hidden until frame selection", async () => {
  const source = await readFile(
    new URL("../../web/ct_dr_3d_viewer/app.js", import.meta.url),
    "utf8",
  );
  const start = source.indexOf("function buildRunVesselFrame(");
  const end = source.indexOf("\nfunction updateRunVesselFrame(", start);
  const builtGroup = {
    visible: true,
    name: "",
    userData: {},
    traverse() {},
  };
  let cached = null;
  const buildRunVesselFrame = new Function(
    "buildConnectedVesselGroup",
    "createRunVesselMaterial",
    "THREE",
    "registerVesselGroup",
    `${source.slice(start, end)}\nreturn buildRunVesselFrame;`,
  )(
    () => builtGroup,
    () => ({}),
    {},
    () => {},
  );
  const state = {
    run: { series_label: "Series 1" },
    payload: { frames: [{ paths: [{ points_lps_mm: [[0, 0, 0], [1, 0, 0]], radius_mm: 1 }] }] },
    cache: { get: () => undefined, set: (_key, value) => { cached = value; } },
    group: { add() {} },
  };
  assert.equal(buildRunVesselFrame(state, 0), builtGroup);
  assert.equal(builtGroup.visible, false);
  assert.equal(cached, builtGroup);
});

test("app rebuilds interpolation from stable graph samples and exposes fallback reason", async () => {
  const source = await readFile(
    new URL("../../web/ct_dr_3d_viewer/app.js", import.meta.url),
    "utf8",
  );
  const start = source.indexOf("function updateRunVesselBlend(");
  const end = source.indexOf("\nfunction updateDynamicVesselFrame(", start);
  assert.notEqual(start, -1, "updateRunVesselBlend is missing");
  assert.notEqual(end, -1, "updateRunVesselBlend boundary is missing");
  const update = source.slice(start, end);

  assert.match(update, /resolveVesselFrameBlend\(/);
  assert.match(update, /interpolateVesselFrame\(/);
  assert.match(update, /buildConnectedVesselGroup\(/);
  assert.match(update, /state\.interpolation\.group/);
  assert.match(update, /vesselFallbackReason/);
  assert.doesNotMatch(
    update,
    /getAttribute\(["']position["']\)|interpolatedPosition|interpolatedNormal|fromPosition|toPosition/,
  );
});

test("app delegates interpolation and cache teardown to exact-once group disposal", async () => {
  const source = await readFile(
    new URL("../../web/ct_dr_3d_viewer/app.js", import.meta.url),
    "utf8",
  );
  const frameDisposalStart = source.indexOf("function disposeFrameGroup(");
  const frameDisposalEnd = source.indexOf("\nfunction createComparisonRunState(", frameDisposalStart);
  const interpolationStart = source.indexOf("function disposeRunInterpolation(");
  const interpolationEnd = source.indexOf("\nfunction disposeComparisonRunStates(", interpolationStart);
  assert.notEqual(frameDisposalStart, -1);
  assert.notEqual(frameDisposalEnd, -1);
  assert.notEqual(interpolationStart, -1);
  assert.notEqual(interpolationEnd, -1);
  const disposal = [
    source.slice(frameDisposalStart, frameDisposalEnd),
    source.slice(interpolationStart, interpolationEnd),
  ].join("\n");

  assert.match(disposal, /disposeConnectedVesselGroup\(/);
  assert.match(disposal, /unregisterVesselMesh/);
  assert.match(disposal, /state\.interpolation\.group = null/);
  assert.doesNotMatch(disposal, /mesh\.geometry\.dispose\(|material\?\.dispose\(/);
});

async function createBlendRuntimeHarness(overrides = {}) {
  const source = await readFile(
    new URL("../../web/ct_dr_3d_viewer/app.js", import.meta.url),
    "utf8",
  );
  const start = source.indexOf("function updateRunVesselBlend(");
  const end = source.indexOf("\nfunction updateDynamicVesselFrame(", start);
  assert.notEqual(start, -1, "updateRunVesselBlend is missing");
  assert.notEqual(end, -1, "updateRunVesselBlend boundary is missing");
  return new Function(
    "runMeshIsVisible",
    "resolveVesselFrameBlend",
    "updateRunVesselFrame",
    "disposeRunInterpolation",
    "interpolateVesselFrame",
    "buildConnectedVesselGroup",
    "createRunVesselMaterial",
    "THREE",
    "registerVesselGroup",
    `${source.slice(start, end)}\nreturn updateRunVesselBlend;`,
  )(
    overrides.runMeshIsVisible ?? ((state) => state.group.visible && Boolean(state.parent?.visible)),
    motion.resolveVesselFrameBlend,
    overrides.updateRunVesselFrame ?? (() => {}),
    overrides.disposeRunInterpolation ?? (() => {}),
    overrides.interpolateVesselFrame ?? (() => { throw new Error("unexpected interpolation"); }),
    overrides.buildConnectedVesselGroup ?? (() => { throw new Error("unexpected build"); }),
    overrides.createRunVesselMaterial ?? (() => ({})),
    overrides.THREE ?? {},
    overrides.registerVesselGroup ?? (() => {}),
  );
}

function blendRuntimeState(frames) {
  return {
    run: { run_id: "series_01" },
    payload: { frames },
    group: { visible: true, userData: {}, add() {} },
    parent: { visible: true },
    currentFrame: null,
    requestedFrame: null,
    vesselFallbackReason: null,
    interpolation: { group: null },
    cache: { get: () => undefined },
  };
}

test("legacy playback, exact keyframes, run guards, and hidden guards avoid graph interpolation", async () => {
  const frames = [0, 1].map((z) => ({
    paths: [{
      path_id: "main",
      topology_id: "legacy-main",
      points_lps_mm: [[0, 0, z], [1, 0, z]],
      radius_mm: 1,
    }],
  }));
  const selected = [];
  let interpolationCalls = 0;
  const update = await createBlendRuntimeHarness({
    updateRunVesselFrame: (_state, frameIndex, reason) => selected.push([frameIndex, reason]),
    interpolateVesselFrame: () => { interpolationCalls += 1; throw new Error("legacy entered graph interpolation"); },
  });
  const state = blendRuntimeState(frames);
  update(state, { runId: "series_01", fromIndex: 0, toIndex: 1, alpha: 0.75 });
  update(state, { runId: "series_01", fromIndex: 1, toIndex: 1, alpha: 0 });
  update(state, { runId: "series_02", fromIndex: 0, toIndex: 1, alpha: 0.5 });
  state.parent.visible = false;
  update(state, { runId: "series_01", fromIndex: 0, toIndex: 1, alpha: 0.5 });
  assert.deepEqual(selected, [[1, "legacy-keyframe"], [1, null]]);
  assert.equal(interpolationCalls, 0);
});

test("rejected-frame fallback preserves the originally requested frame index", async () => {
  const accepted = schemaFrame({ frameIndex: 0 });
  const rejected = schemaFrame({ frameIndex: 1, status: "rejected" });
  rejected.display_source_frame_index = 0;
  const selected = [];
  const update = await createBlendRuntimeHarness({
    updateRunVesselFrame: (_state, frameIndex, reason) => selected.push([frameIndex, reason]),
  });
  update(
    blendRuntimeState([accepted, rejected]),
    { runId: "series_01", fromIndex: 1, toIndex: 1, alpha: 0 },
  );
  assert.deepEqual(selected, [[1, "rejected-frame"]]);
});

test("interpolated group registration is semantic and unpickable", async () => {
  const source = await readFile(
    new URL("../../web/ct_dr_3d_viewer/app.js", import.meta.url),
    "utf8",
  );
  const start = source.indexOf("function registerVesselGroup(");
  const end = source.indexOf("\nfunction buildRunVesselFrame(", start);
  assert.notEqual(start, -1);
  assert.notEqual(end, -1);
  const registrations = [];
  const pickables = [];
  const registerVesselGroup = new Function(
    "renderingController",
    "activeDynamicRunId",
    "pickables",
    `${source.slice(start, end)}\nreturn registerVesselGroup;`,
  )(
    { registerMesh: (mesh, kind) => registrations.push([mesh, kind]) },
    "series_01",
    pickables,
  );
  const meshes = [{ isMesh: true, userData: {} }, { isMesh: true, userData: {} }];
  const group = {
    userData: {},
    traverse(callback) { meshes.forEach(callback); },
  };
  const state = { run: { run_id: "series_01" } };
  registerVesselGroup(state, group, { frame_index: 0 }, 0, { pickable: false });
  assert.equal(registrations.length, 2);
  assert.equal(registrations.every(([, kind]) => kind === "vessel-primary"), true);
  assert.equal(pickables.length, 0);
  assert.equal(meshes.every((mesh) => mesh.userData.kind === "multiseries-vessel-3d"), true);
});

async function createVisibilityHarness(options) {
  const source = await readFile(
    new URL("../../web/ct_dr_3d_viewer/app.js", import.meta.url),
    "utf8",
  );
  const extract = (startToken, endToken) => {
    const start = source.indexOf(startToken);
    const end = source.indexOf(endToken, start);
    assert.notEqual(start, -1, `${startToken} is missing`);
    assert.notEqual(end, -1, `${endToken} is missing`);
    return source.slice(start, end);
  };
  const runtimeSource = [
    extract("function resynchronizeVisibleRun(", "\nfunction refreshComparisonVisibility("),
    extract("function refreshComparisonVisibility(", "\nfunction lineBetween("),
    extract("function setViewerGroupVisibility(", "\nfunction bindUi("),
  ].join("\n");
  return new Function(
    "groups",
    "$",
    "groupEnabled",
    "multiseriesPlayer",
    "dynamicCine",
    "comparisonRunStates",
    "activeDynamicRunId",
    "runMeshIsVisible",
    "frameBlendForRun",
    "updateRunVesselBlend",
    "updateEstimatedPlayback",
    "renderMultiseriesFrameStatus",
    `${runtimeSource}\nreturn { refreshComparisonVisibility, setViewerGroupVisibility };`,
  )(
    options.groups,
    options.$,
    options.groupEnabled,
    options.multiseriesPlayer,
    options.dynamicCine,
    options.comparisonRunStates,
    options.activeDynamicRunId,
    options.runMeshIsVisible,
    options.frameBlendForRun,
    options.updateRunVesselBlend,
    options.updateEstimatedPlayback,
    options.renderMultiseriesFrameStatus,
  );
}

test("revealing a comparison while paused resynchronizes once to multiseries elapsed time", async () => {
  const vesselComparisons = { visible: false, children: [] };
  const state = {
    run: { run_id: "series_02", anatomy_group: "rca" },
    group: { visible: true, userData: { anatomyGroup: "rca" }, parent: vesselComparisons },
    parent: vesselComparisons,
  };
  vesselComparisons.children.push(state.group);
  const dynamicCine = { playing: false, elapsedMs: 25 };
  const multiseriesPlayer = { mode: "independent", elapsedMs: 175 };
  const updates = [];
  const harness = await createVisibilityHarness({
    groups: { xaComparisons: { visible: false, children: [] }, vesselComparisons, dynamicVessel3d: { visible: true } },
    $: (id) => ({ checked: id === "show-all-vessel-meshes" }),
    groupEnabled: () => true,
    multiseriesPlayer,
    dynamicCine,
    comparisonRunStates: new Map([["series_02", state]]),
    activeDynamicRunId: "series_01",
    runMeshIsVisible: (candidate) => candidate.group.visible && candidate.parent.visible,
    frameBlendForRun: (run, elapsedMs) => ({ runId: run.run_id, fromIndex: 1, toIndex: 2, alpha: elapsedMs / 1000 }),
    updateRunVesselBlend: (candidate, blend) => updates.push({ candidate, blend }),
    updateEstimatedPlayback: () => { throw new Error("independent reveal entered estimated playback"); },
    renderMultiseriesFrameStatus: () => {},
  });

  harness.refreshComparisonVisibility();
  assert.equal(dynamicCine.playing, false);
  assert.equal(vesselComparisons.visible, true);
  assert.equal(updates.length, 1);
  assert.equal(updates[0].candidate, state);
  assert.equal(updates[0].blend.alpha, 0.175);
});

test("revealing the active vessel while paused resynchronizes once to active elapsed time", async () => {
  const dynamicVessel3d = { visible: false };
  const state = {
    run: { run_id: "series_01", anatomy_group: "rca" },
    group: { visible: true },
    parent: dynamicVessel3d,
  };
  const dynamicCine = { playing: false, elapsedMs: 225 };
  const multiseriesPlayer = { mode: "single", elapsedMs: 900 };
  const updates = [];
  const harness = await createVisibilityHarness({
    groups: { xaComparisons: { visible: false, children: [] }, vesselComparisons: { visible: false, children: [] }, dynamicVessel3d },
    $: () => ({ checked: false }),
    groupEnabled: () => true,
    multiseriesPlayer,
    dynamicCine,
    comparisonRunStates: new Map([["series_01", state]]),
    activeDynamicRunId: "series_01",
    runMeshIsVisible: (candidate) => candidate.group.visible && candidate.parent.visible,
    frameBlendForRun: (run, elapsedMs) => ({ runId: run.run_id, fromIndex: 2, toIndex: 3, alpha: elapsedMs / 1000 }),
    updateRunVesselBlend: (candidate, blend) => updates.push({ candidate, blend }),
    updateEstimatedPlayback: () => { throw new Error("single reveal entered estimated playback"); },
    renderMultiseriesFrameStatus: () => {},
  });

  harness.setViewerGroupVisibility("show-vessel-3d", "dynamicVessel3d", true);
  assert.equal(dynamicCine.playing, false);
  assert.equal(dynamicVessel3d.visible, true);
  assert.equal(updates.length, 1);
  assert.equal(updates[0].candidate, state);
  assert.equal(updates[0].blend.alpha, 0.225);
});
