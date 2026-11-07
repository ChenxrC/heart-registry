import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { GLTFLoader } from "three/addons/loaders/GLTFLoader.js";
import { normalizeRenderMode } from "./rendering_mode.mjs";
import { createRenderingController, loadGtaoAddons } from "./rendering_scene.mjs";
import {
  buildConnectedVesselGroup,
  disposeConnectedVesselGroup,
  interpolateVesselFrame,
} from "./vessel_mesh.js";
import {
  resolveThoraxAssetRecords,
  resolveThoraxVariant,
  setThoraxVariantVisibility,
} from "./thorax_variants.mjs";
import {
  createPhaseOverridePayload,
  elapsedBeforeFrame,
  estimatedGroupPlaybackState,
  estimatedPlaybackRequestAllowed,
  frameAtElapsed,
  frameBlendAtElapsed,
  isEffectivelyVisible,
  phaseStatusRows,
  phaseWithAnchorOverride,
  validatePhasePlaybackReport,
  LruFrameCache,
  multiseriesSelectionPlaybackText,
  playbackFrameRange,
  resolveVesselFrameBlend,
} from "./multiseries_motion.js";

const gtaoAddons = await loadGtaoAddons();

const $ = (id) => document.getElementById(id);

function selectedRenderMode() {
  return normalizeRenderMode(new URLSearchParams(window.location.search).get("render"));
}

const renderModeSelector = $("render-mode");
renderModeSelector.value = selectedRenderMode();
const thoraxVariantSelector = $("thorax-variant");
thoraxVariantSelector.value = "auto";

const canvas = $("viewport");
const viewportShell = $("viewport-shell");
const loading = $("loading");
const loadingMessage = $("loading-message");

const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: false });
renderer.localClippingEnabled = true;

const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(38, 1, 0.5, 6000);
camera.up.set(0, 0, 1);
const lowPower = window.innerWidth <= 768 || (navigator.hardwareConcurrency ?? 8) <= 4;
const maxPixelRatio = lowPower ? 1.5 : 2.0;
const renderingController = createRenderingController({
  THREE,
  renderer,
  scene,
  camera,
  requestRender,
  lowPower,
  addons: gtaoAddons,
});
renderingController.setMode(renderModeSelector.value);
renderModeSelector.addEventListener("change", (event) => {
  event.stopPropagation();
  const mode = normalizeRenderMode(event.target.value);
  renderModeSelector.value = mode;
  const url = new URL(window.location.href);
  url.searchParams.set("render", mode);
  history.replaceState(null, "", url);
  renderingController.setMode(mode);
  setThoraxVariant(resolveThoraxVariant(renderModeSelector.value, thoraxVariantSelector.value));
  requestRender();
});
thoraxVariantSelector.addEventListener("change", (event) => {
  event.stopPropagation();
  setThoraxVariant(resolveThoraxVariant(renderModeSelector.value, event.target.value));
  requestRender();
});

const controls = new OrbitControls(camera, canvas);
controls.enableDamping = false;
controls.screenSpacePanning = true;

const groups = {
  ct: new THREE.Group(),
  vertebrae: new THREE.Group(),
  dr: new THREE.Group(),
  frustum: new THREE.Group(),
  landmarks: new THREE.Group(),
  rays: new THREE.Group(),
  axes: new THREE.Group(),
  dynamicXa: new THREE.Group(),
  ribs: new THREE.Group(),
  heart: new THREE.Group(),
  dynamicVessel3d: new THREE.Group(),
  vesselDiagnostics: new THREE.Group(),
  xaComparisons: new THREE.Group(),
  vesselComparisons: new THREE.Group(),
};
Object.values(groups).forEach((group) => scene.add(group));

const thoraxVariants = {
  registration: { heart: new THREE.Group(), ribs: new THREE.Group() },
  display: { heart: new THREE.Group(), ribs: new THREE.Group() },
};
groups.heart.add(thoraxVariants.registration.heart, thoraxVariants.display.heart);
groups.ribs.add(thoraxVariants.registration.ribs, thoraxVariants.display.ribs);
let currentThoraxVariant = setThoraxVariantVisibility(thoraxVariants, "registration");

const pickables = [];
const sliceMeshes = {};
let geometry;
let ctTexture;
let defaultView;
let dynamicCine = null;
let renderRequest = null;
let dynamicVessel3d = null;
let activeDynamicRunId = null;
const comparisonRunStates = new Map();
const repairLinkDecisionColors = Object.freeze({
  accepted: 0x2de3b7,
  review: 0xffb84d,
  rejected: 0xff5d73,
});
const multiseriesPlayer = { mode: "single", elapsedMs: 0, groupElapsedMs: { rca: 0, lca: 0 } };
const phasePlayback = {
  report: null,
  sourceAsset: null,
  loadStatus: "unavailable",
  reason: "尚未载入估计相位报告。",
  overrides: new Map(),
};

function vector(values) { return new THREE.Vector3(...values); }
function setLoading(message) { loadingMessage.textContent = message; }
function fmt(value, digits = 2) { return Number(value).toFixed(digits); }
function fmtPoint(values) { return values.map((v) => fmt(v, 1)).join(", "); }

function setThoraxVariant(variant) {
  currentThoraxVariant = setThoraxVariantVisibility(thoraxVariants, variant);
  thoraxVariantSelector.dataset.activeVariant = currentThoraxVariant;
  const mode = renderModeSelector.value;
  if (mode === "realistic" && $("show-topology-nodes")) {
    $("show-topology-nodes").checked = false;
  }
  refreshVesselDiagnosticStatus();
}

async function loadThoraxVariant(name, thoraxScene) {
  const thoraxMeshes = [];
  thoraxScene.traverse((object) => { if (object.isMesh) thoraxMeshes.push(object); });
  for (const object of thoraxMeshes) {
    const isHeart = object.name.toLowerCase().includes("heart");
    object.geometry?.computeVertexNormals();
    const base = Array.isArray(object.material) ? object.material[0] : object.material;
    object.material = base.clone();
    object.material.transparent = true;
    object.material.depthWrite = false;
    object.material.side = THREE.DoubleSide;
    object.material.roughness = isHeart ? 0.55 : 0.78;
    object.material.metalness = 0.0;
    object.material.color.setHex(isHeart ? 0xc94168 : 0xe5dab8);
    object.material.opacity = isHeart ? Number($("heart-opacity").value) : Number($("rib-opacity").value);
    object.userData = {
      kind: isHeart ? "heart-surface" : "thorax-bone",
      label: object.name,
      thoraxVariant: name,
    };
    renderingController.registerMesh(object, isHeart ? "heart" : "bone");
    thoraxVariants[name][isHeart ? "heart" : "ribs"].add(object);
    pickables.push(object);
  }
}

function configureDynamicRuns() {
  const runs = geometry.dynamic_xa_runs ?? (geometry.dynamic_xa ? [geometry.dynamic_xa] : []);
  if (!runs.length) return;
  const requested = new URLSearchParams(window.location.search).get("run");
  const fallback = geometry.active_dynamic_xa_run_id ?? runs[0].run_id ?? "series_01";
  const selected = runs.find((item) => item.run_id === requested)
    ?? runs.find((item) => item.run_id === fallback)
    ?? runs[0];
  activeDynamicRunId = selected.run_id;
  geometry.dynamic_xa = selected;
  geometry.dynamic_vessel_3d = (geometry.dynamic_vessel_3d_runs ?? []).find(
    (item) => item.run_id === activeDynamicRunId,
  ) ?? (geometry.dynamic_xa_runs ? null : geometry.dynamic_vessel_3d);

  const vesselOnlyIds = ["vessel-opacity", "vessel-threshold", "show-centerline"];
  const hasVessel = selected.has_vessel !== false;
  $("vessel-mask-toggle")?.classList.toggle("hidden", !hasVessel);
  if (!hasVessel && $("show-vessel-mask")) $("show-vessel-mask").checked = false;
  vesselOnlyIds.forEach((id) => $(id)?.closest("label")?.classList.toggle("hidden", !hasVessel));
  const projectionWarning = document.querySelector("#dynamic-controls .projection-warning");
  if (projectionWarning && !hasVessel) {
    projectionWarning.textContent = "Series 7 为胸部定位序列，没有冠脉造影剂充盈；本页只显示其 XA 投影平面，不生成血管分割或 Mesh。";
  }

  if (geometry.dynamic_xa_runs) {
    document.querySelectorAll(".multiseries-only").forEach((node) => node.classList.remove("hidden"));
    const selector = $("xa-run-select");
    selector.replaceChildren();
    for (const run of runs) {
      const option = document.createElement("option");
      option.value = run.run_id;
      option.textContent = `${run.series_label} · ${run.frame_count} 帧`;
      option.selected = run.run_id === activeDynamicRunId;
      selector.appendChild(option);
    }
  }
}

const phaseRunIds = ["series_01", "series_02", "series_03", "series_04", "series_05", "series_06"];
const phaseGroupNames = ["rca", "lca"];

function phaseGroupLabel(groupName) {
  return groupName === "rca" ? "RCA" : "LCA";
}

function phaseStatusLabel(status) {
  return ({
    automatic: "自动",
    provisional: "待人工确认",
    insufficient: "不足",
    "manual-confirmed": "人工确认",
  })[status] ?? status;
}

function phaseGroupForRun(runId) {
  return ["series_01", "series_02"].includes(runId) ? "rca" : "lca";
}

function fallbackPhaseRows() {
  const compactRuns = geometry?.cardiac_phase_alignment?.runs ?? {};
  return phaseRunIds.filter((runId) => compactRuns[runId]).map((runId) => {
    const run = compactRuns[runId];
    return {
      run_id: runId,
      period_frames: Number.isFinite(run.period_frames) ? run.period_frames : null,
      confidence: Number.isFinite(run.confidence) ? run.confidence : null,
      status: run.status ?? "insufficient",
      reason: phasePlayback.reason,
      anchor_frame: Number.isInteger(run.anchor_frame) ? run.anchor_frame : null,
    };
  });
}

function renderPhaseConfidenceRows() {
  const list = $("phase-confidence-list");
  if (!list) return;
  list.replaceChildren();
  const rows = phasePlayback.report
    ? phaseStatusRows(phasePlayback.report, phasePlayback.overrides)
    : fallbackPhaseRows();
  rows.forEach((row) => {
    const item = document.createElement("div");
    item.className = `phase-confidence-row phase-${row.status}`;
    const heading = document.createElement("strong");
    heading.textContent = `${seriesLabel({ run_id: row.run_id })} · ${phaseGroupLabel(phaseGroupForRun(row.run_id))}`;
    const metrics = document.createElement("span");
    const confidence = Number.isFinite(row.confidence) ? `${(row.confidence * 100).toFixed(1)}%` : "—";
    const period = Number.isFinite(row.period_frames) ? `${Number(row.period_frames).toFixed(2)} 帧` : "—";
    const anchor = Number.isInteger(row.anchor_frame) ? String(row.anchor_frame) : "—";
    metrics.textContent = `${phaseStatusLabel(row.status)} · 置信度 ${confidence} · 周期 ${period} · 锚点 ${anchor}`;
    const reason = document.createElement("small");
    reason.textContent = `原因：${row.reason || "—"}`;
    item.append(heading, metrics, reason);
    list.appendChild(item);
  });
}

function phaseAnchorCandidates() {
  if (!phasePlayback.report) return [];
  return phaseRunIds.filter((runId) => {
    const run = phasePlayback.report.runs[runId];
    if (!run || run.status === "insufficient") return false;
    try {
      phaseWithAnchorOverride(run, run.anchor_frame);
      return true;
    } catch (error) {
      return false;
    }
  });
}

function configurePhaseAnchorSlider() {
  const selector = $("phase-anchor-run");
  const slider = $("phase-anchor-frame");
  const output = $("phase-anchor-frame-value");
  const applyButton = $("phase-anchor-apply");
  if (!selector || !slider || !applyButton) return;
  const candidates = phaseAnchorCandidates();
  const previous = selector.value;
  selector.replaceChildren();
  candidates.forEach((runId) => {
    const option = document.createElement("option");
    option.value = runId;
    option.textContent = `${seriesLabel({ run_id: runId })} · ${phaseStatusLabel(phasePlayback.report.runs[runId].status)}`;
    selector.appendChild(option);
  });
  if (candidates.includes(previous)) selector.value = previous;
  const selectedRunId = selector.value;
  const run = phasePlayback.report?.runs?.[selectedRunId];
  const enabled = Boolean(run);
  selector.disabled = !enabled;
  slider.disabled = !enabled;
  applyButton.disabled = !enabled;
  if (!enabled) {
    slider.min = "0";
    slider.max = "0";
    slider.value = "0";
    if (output) output.value = "—";
    return;
  }
  const visible = Array.isArray(run.visible_frame_range) ? run.visible_frame_range : [0, run.phase_per_frame.length - 1];
  slider.min = String(visible[0]);
  slider.max = String(visible[1]);
  const override = phasePlayback.overrides.get(selectedRunId);
  slider.value = String(Number.isInteger(override) ? override : run.anchor_frame);
  if (output) output.value = slider.value;
}

function selectedPhaseGroups() {
  return phaseGroupNames.filter((groupName) => groupEnabled(groupName));
}

function estimatedModeGate() {
  if (!phasePlayback.report) return { enabled: false, reason: phasePlayback.reason };
  const selected = selectedPhaseGroups();
  if (!selected.length) return { enabled: false, reason: "请至少选择一个 RCA 或 LCA 解剖组。" };
  const failed = selected.map((groupName) => ({
    groupName,
    state: estimatedGroupPlaybackState(
      phasePlayback.report,
      geometry.dynamic_xa_runs,
      groupName,
      multiseriesPlayer.groupElapsedMs[groupName],
      phasePlayback.overrides,
    ),
  })).filter(({ state }) => !state.enabled);
  if (failed.length) {
    const labels = failed.map(({ groupName }) => phaseGroupLabel(groupName)).join("、");
    return { enabled: false, reason: `${labels} 需要至少 2 段已确认可用序列和非空参考序列。` };
  }
  return { enabled: true, reason: "估计相位同步可用；RCA 与 LCA 使用独立参考时钟。" };
}

function refreshEstimatedModeAvailability() {
  const estimatedOption = $("multiseries-play-mode")?.querySelector('option[value="estimated"]');
  const status = $("phase-mode-status");
  if (!estimatedOption) return;
  const gate = estimatedModeGate();
  estimatedOption.disabled = !gate.enabled;
  if (status) status.textContent = gate.reason;
  if (!gate.enabled && $("multiseries-play-mode").value === "estimated") {
    if (dynamicCine?.playing) setMultiseriesPlaying(false);
    $("multiseries-play-mode").value = "independent";
    multiseriesPlayer.mode = "independent";
  }
}

function renderPhaseControls() {
  renderPhaseConfidenceRows();
  configurePhaseAnchorSlider();
  const exportButton = $("phase-overrides-export");
  if (exportButton) exportButton.disabled = !phasePlayback.sourceAsset || phasePlayback.overrides.size === 0;
  refreshEstimatedModeAvailability();
}

async function loadPhaseAlignmentReport() {
  const summary = geometry.cardiac_phase_alignment;
  const assetPath = summary?.asset?.path;
  if (!assetPath) {
    phasePlayback.reason = summary?.status === "missing"
      ? "估计相位报告缺失或不兼容；单序列与未同步播放仍可用。"
      : "估计相位报告不可用；单序列与未同步播放仍可用。";
    renderPhaseControls();
    return;
  }
  try {
    const response = await fetch(assetPath);
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    phasePlayback.report = validatePhasePlaybackReport(await response.json());
    phasePlayback.sourceAsset = assetPath;
    phasePlayback.loadStatus = "loaded";
    phasePlayback.reason = phasePlayback.report.status === "insufficient"
      ? "真实估计报告全部不足；估计相位同步保持禁用。"
      : "估计相位报告已载入。";
  } catch (error) {
    console.warn("Estimated phase report disabled:", error);
    phasePlayback.report = null;
    phasePlayback.sourceAsset = null;
    phasePlayback.loadStatus = "invalid";
    phasePlayback.reason = `估计相位报告载入或架构校验失败：${error.message}；单序列与未同步播放仍可用。`;
  }
  renderPhaseControls();
}

function applySelectedPhaseAnchor() {
  const runId = $("phase-anchor-run")?.value;
  const anchorFrame = Number($("phase-anchor-frame")?.value);
  const run = phasePlayback.report?.runs?.[runId];
  if (!run || run.status === "insufficient") return;
  try {
    phaseWithAnchorOverride(run, anchorFrame);
  } catch (error) {
    console.warn("Manual phase anchor rejected:", error);
    return;
  }
  phasePlayback.overrides.set(runId, anchorFrame);
  renderPhaseControls();
  if (multiseriesPlayer.mode === "estimated") updateEstimatedPlayback();
}

function downloadPhaseOverrides() {
  if (!phasePlayback.sourceAsset || phasePlayback.overrides.size === 0) return;
  const payload = createPhaseOverridePayload(phasePlayback.sourceAsset, phasePlayback.overrides);
  const blob = new Blob([`${JSON.stringify(payload, null, 2)}\n`], { type: "application/json;charset=utf-8" });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = "phase_anchor_overrides.json";
  document.body.appendChild(link);
  link.click();
  link.remove();
  URL.revokeObjectURL(url);
}

function groupEnabled(anatomyGroup) {
  const checkbox = $(`show-group-${anatomyGroup}`);
  return checkbox ? checkbox.checked : true;
}

function resynchronizeVisibleRun(state) {
  if (!state || dynamicCine?.playing || !runMeshIsVisible(state)) return;
  if (multiseriesPlayer.mode === "estimated") {
    updateEstimatedPlayback();
    return;
  }
  const elapsedMs = multiseriesPlayer.mode === "independent"
    ? multiseriesPlayer.elapsedMs
    : (dynamicCine?.elapsedMs ?? 0);
  const useFullClip = Boolean($("multiseries-full-clip")?.checked);
  updateRunVesselBlend(state, frameBlendForRun(state.run, elapsedMs, useFullClip));
}

function refreshComparisonVisibility() {
  const wasVisible = new Map(
    [...comparisonRunStates.values()].map((state) => [state, runMeshIsVisible(state)]),
  );
  groups.xaComparisons.visible = Boolean($("show-all-xa-planes")?.checked);
  groups.vesselComparisons.visible = Boolean($("show-all-vessel-meshes")?.checked);
  for (const group of [...groups.xaComparisons.children, ...groups.vesselComparisons.children]) {
    group.visible = groupEnabled(group.userData.anatomyGroup);
  }
  if (multiseriesPlayer.mode === "estimated") {
    updateEstimatedPlayback();
    return;
  }
  comparisonRunStates.forEach((state) => {
    state.group.visible = groupEnabled(state.run.anatomy_group);
  });
  if (!dynamicCine?.playing) {
    comparisonRunStates.forEach((state) => {
      if (state.run.run_id !== activeDynamicRunId && !wasVisible.get(state) && runMeshIsVisible(state)) {
        resynchronizeVisibleRun(state);
      }
    });
  }
  renderMultiseriesFrameStatus();
}

function lineBetween(a, b, color, opacity = 1, dashed = false, kind = "helper") {
  const material = dashed
    ? new THREE.LineDashedMaterial({ color, transparent: opacity < 1, opacity, dashSize: 9, gapSize: 5 })
    : new THREE.LineBasicMaterial({ color, transparent: opacity < 1, opacity });
  const line = new THREE.Line(new THREE.BufferGeometry().setFromPoints([a, b]), material);
  if (dashed) line.computeLineDistances();
  renderingController.registerMesh(line, kind);
  return line;
}

function sphereAt(point, radius, color, materialOptions = {}, kind = "helper") {
  const mesh = new THREE.Mesh(
    new THREE.SphereGeometry(radius, 20, 14),
    new THREE.MeshStandardMaterial({ color, roughness: 0.35, metalness: 0.05, ...materialOptions }),
  );
  mesh.position.copy(point);
  renderingController.registerMesh(mesh, kind);
  return mesh;
}

function makeLabel(text, color = "#ffffff", scale = 34) {
  const labelCanvas = document.createElement("canvas");
  const context = labelCanvas.getContext("2d");
  const fontSize = 42;
  context.font = `700 ${fontSize}px Segoe UI, sans-serif`;
  const width = Math.ceil(context.measureText(text).width + 28);
  labelCanvas.width = width;
  labelCanvas.height = 64;
  context.font = `700 ${fontSize}px Segoe UI, sans-serif`;
  context.fillStyle = "rgba(3, 12, 16, 0.78)";
  context.beginPath();
  context.roundRect(0, 2, width, 58, 10);
  context.fill();
  context.fillStyle = color;
  context.textBaseline = "middle";
  context.fillText(text, 14, 31);
  const texture = new THREE.CanvasTexture(labelCanvas);
  texture.colorSpace = THREE.SRGBColorSpace;
  const sprite = new THREE.Sprite(new THREE.SpriteMaterial({ map: texture, depthTest: false, transparent: true }));
  sprite.scale.set((width / 64) * scale, scale, 1);
  sprite.renderOrder = 20;
  renderingController.registerMesh(sprite, "annotation");
  return sprite;
}

function lpsFromIndex(index) {
  const volume = geometry.ct.volume;
  const scaled = index.map((value, axis) => value * volume.spacing_xyz_mm[axis]);
  const direction = volume.direction_lps;
  return new THREE.Vector3(
    volume.origin_lps_mm[0] + direction[0][0] * scaled[0] + direction[0][1] * scaled[1] + direction[0][2] * scaled[2],
    volume.origin_lps_mm[1] + direction[1][0] * scaled[0] + direction[1][1] * scaled[1] + direction[1][2] * scaled[2],
    volume.origin_lps_mm[2] + direction[2][0] * scaled[0] + direction[2][1] * scaled[1] + direction[2][2] * scaled[2],
  );
}

function makeSliceGeometry(axis, index) {
  const [nx, ny, nz] = geometry.ct.volume.dimensions_xyz;
  let indices;
  if (axis === "x") indices = [[index, 0, 0], [index, ny - 1, 0], [index, ny - 1, nz - 1], [index, 0, nz - 1]];
  if (axis === "y") indices = [[0, index, 0], [nx - 1, index, 0], [nx - 1, index, nz - 1], [0, index, nz - 1]];
  if (axis === "z") indices = [[0, 0, index], [nx - 1, 0, index], [nx - 1, ny - 1, index], [0, ny - 1, index]];
  const positions = indices.flatMap((p) => lpsFromIndex(p).toArray());
  const volumeCoords = indices.flatMap((p) => [p[0] / (nx - 1), p[1] / (ny - 1), p[2] / (nz - 1)]);
  const buffer = new THREE.BufferGeometry();
  buffer.setAttribute("position", new THREE.Float32BufferAttribute(positions, 3));
  buffer.setAttribute("volumeCoord", new THREE.Float32BufferAttribute(volumeCoords, 3));
  buffer.setIndex([0, 1, 2, 0, 2, 3]);
  buffer.computeVertexNormals();
  return buffer;
}

const sliceVertexShader = `
  precision highp float;
  uniform mat4 modelViewMatrix;
  uniform mat4 projectionMatrix;
  in vec3 position;
  in vec3 volumeCoord;
  out vec3 vVolumeCoord;
  void main() {
    vVolumeCoord = volumeCoord;
    gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
  }
`;

const sliceFragmentShader = `
  precision highp float;
  precision highp sampler3D;
  uniform sampler3D volumeTexture;
  uniform float windowCenter;
  uniform float windowWidth;
  uniform float opacity;
  in vec3 vVolumeCoord;
  out vec4 outColor;
  void main() {
    float encoded = texture(volumeTexture, clamp(vVolumeCoord, 0.0, 1.0)).r;
    float hu = encoded * 4095.0 - 1024.0;
    float gray = clamp((hu - (windowCenter - 0.5 * windowWidth)) / max(windowWidth, 1.0), 0.0, 1.0);
    outColor = vec4(vec3(gray), opacity);
  }
`;

function createSliceMaterial() {
  return new THREE.RawShaderMaterial({
    glslVersion: THREE.GLSL3,
    vertexShader: sliceVertexShader,
    fragmentShader: sliceFragmentShader,
    uniforms: {
      volumeTexture: { value: ctTexture },
      windowCenter: { value: Number($("window-center").value) },
      windowWidth: { value: Number($("window-width").value) },
      opacity: { value: Number($("ct-opacity").value) },
    },
    transparent: true,
    depthWrite: false,
    side: THREE.DoubleSide,
  });
}

async function loadCtVolume() {
  setLoading("正在载入 CT 体数据与三正交切片…");
  const volume = geometry.ct.volume;
  const response = await fetch(volume.path);
  if (!response.ok) throw new Error(`CT 体数据读取失败：${response.status}`);
  const buffer = await response.arrayBuffer();
  if (buffer.byteLength !== volume.byte_length) throw new Error(`CT 体数据长度异常：${buffer.byteLength} / ${volume.byte_length}`);
  const values = new Uint16Array(buffer);
  const [nx, ny, nz] = volume.dimensions_xyz;
  ctTexture = new THREE.Data3DTexture(values, nx, ny, nz);
  ctTexture.format = THREE.RedFormat;
  ctTexture.type = THREE.UnsignedShortType;
  ctTexture.minFilter = THREE.LinearFilter;
  ctTexture.magFilter = THREE.LinearFilter;
  ctTexture.unpackAlignment = 1;
  ctTexture.needsUpdate = true;

  const initial = geometry.ct.initial_slices.index_xyz.map((v, axis) => Math.round(v / volume.stride_zyx[axis]));
  const limits = [nx, ny, nz];
  ["x", "y", "z"].forEach((axis, axisIndex) => {
    const slider = $(`slice-${axis}`);
    slider.max = limits[axisIndex] - 1;
    slider.value = THREE.MathUtils.clamp(initial[axisIndex], 0, limits[axisIndex] - 1);
    const mesh = new THREE.Mesh(makeSliceGeometry(axis, Number(slider.value)), createSliceMaterial());
    mesh.name = `CT ${axis.toUpperCase()} 切面`;
    mesh.renderOrder = 2;
    mesh.userData = { kind: "ct-slice", axis };
    renderingController.registerMesh(mesh, "ct-image");
    groups.ct.add(mesh);
    sliceMeshes[axis] = mesh;
    updateSlice(axis);
  });
}

function updateSlice(axis) {
  if (!sliceMeshes[axis]) return;
  const index = Number($(`slice-${axis}`).value);
  sliceMeshes[axis].geometry.dispose();
  sliceMeshes[axis].geometry = makeSliceGeometry(axis, index);
  const p = [0, 0, 0];
  const component = { x: 0, y: 1, z: 2 }[axis];
  p[component] = index;
  const position = lpsFromIndex(p).getComponent(component);
  $(`slice-${axis}-value`).value = `${index} · ${fmt(position, 1)} mm`;
}

function addCtBox() {
  const box = new THREE.Box3(vector(geometry.ct.bounds_min_lps_mm), vector(geometry.ct.bounds_max_lps_mm));
  const helper = new THREE.Box3Helper(box, 0x4c8995);
  helper.material.transparent = true;
  helper.material.opacity = 0.42;
  renderingController.registerMesh(helper, "helper");
  groups.ct.add(helper);
}

async function loadVertebrae() {
  setLoading("正在载入逐椎体三维网格…");
  const gltf = await new GLTFLoader().loadAsync(geometry.vertebrae.glb_path);
  gltf.scene.traverse((object) => {
    if (!object.isMesh) return;
    const base = Array.isArray(object.material) ? object.material[0] : object.material;
    object.material = base.clone();
    object.material.transparent = true;
    object.material.opacity = Number($("mesh-opacity").value);
    object.material.depthWrite = false;
    object.material.side = THREE.DoubleSide;
    const record = geometry.vertebrae.labels.find((item) => object.name.includes(item.label));
    object.userData = { kind: "vertebra", record, label: record?.label ?? object.name };
    renderingController.registerMesh(object, "bone");
    pickables.push(object);
  });
  groups.vertebrae.add(gltf.scene);
}

async function addDetectorPlane() {
  setLoading("正在读取 DR 原生灰度纹理…");
  const corners = geometry.xa.detector.corners_lps_mm.map(vector);
  const detectorGeometry = new THREE.BufferGeometry();
  detectorGeometry.setAttribute("position", new THREE.Float32BufferAttribute(corners.flatMap((p) => p.toArray()), 3));
  detectorGeometry.setAttribute("uv", new THREE.Float32BufferAttribute([[0, 0], [1, 0], [1, 1], [0, 1]].flat(), 2));
  detectorGeometry.setIndex([0, 1, 2, 0, 2, 3]);
  detectorGeometry.computeVertexNormals();
  const textureResponse = await fetch(geometry.xa.texture_raw_path);
  if (!textureResponse.ok) throw new Error(`DR 纹理读取失败：${textureResponse.status}`);
  setLoading("正在解码 DR 原生灰度纹理…");
  const textureBytes = new Uint8Array(await textureResponse.arrayBuffer());
  const expectedBytes = geometry.xa.detector.columns * geometry.xa.detector.rows;
  if (textureBytes.byteLength !== expectedBytes) throw new Error(`DR 纹理长度异常：${textureBytes.byteLength} / ${expectedBytes}`);
  const rgbaBytes = new Uint8Array(expectedBytes * 4);
  for (let index = 0; index < textureBytes.length; index += 1) {
    const value = textureBytes[index];
    rgbaBytes[index * 4] = value;
    rgbaBytes[index * 4 + 1] = value;
    rgbaBytes[index * 4 + 2] = value;
    rgbaBytes[index * 4 + 3] = 255;
  }
  const texture = new THREE.DataTexture(
    rgbaBytes,
    geometry.xa.detector.columns,
    geometry.xa.detector.rows,
    THREE.RGBAFormat,
    THREE.UnsignedByteType,
  );
  texture.needsUpdate = true;
  texture.flipY = false;
  texture.unpackAlignment = 1;
  texture.minFilter = THREE.LinearFilter;
  texture.magFilter = THREE.LinearFilter;
  setLoading("正在创建 DR 物理探测器平面…");
  const material = new THREE.MeshBasicMaterial({ map: texture, transparent: true, opacity: Number($("dr-opacity").value), side: THREE.DoubleSide, depthWrite: false });
  const plane = new THREE.Mesh(detectorGeometry, material);
  plane.name = "XA / DR 探测器平面";
  plane.renderOrder = 5;
  plane.userData = { kind: "detector" };
  renderingController.registerMesh(plane, "dr-image");
  groups.dr.add(plane);
  pickables.push(plane);

  const outline = new THREE.Line(
    new THREE.BufferGeometry().setFromPoints([...corners, corners[0]]),
    new THREE.LineBasicMaterial({ color: 0x56dce5 }),
  );
  outline.renderOrder = 6;
  renderingController.registerMesh(outline, "helper");
  groups.dr.add(outline);
  const topLabel = makeLabel("DR TOP / 头侧", "#79f4dd", 24);
  topLabel.position.copy(corners[0]).lerp(corners[1], 0.5).addScaledVector(vector(geometry.xa.v_lps), -18);
  groups.dr.add(topLabel);
  setLoading("DR 物理探测器平面已完成…");
}

function detectorGeometryFromCorners(corners, offset = new THREE.Vector3()) {
  const detectorGeometry = new THREE.BufferGeometry();
  const shifted = corners.map((point) => point.clone().add(offset));
  detectorGeometry.setAttribute("position", new THREE.Float32BufferAttribute(shifted.flatMap((point) => point.toArray()), 3));
  detectorGeometry.setAttribute("uv", new THREE.Float32BufferAttribute([[0, 0], [1, 0], [1, 1], [0, 1]].flat(), 2));
  detectorGeometry.setIndex([0, 1, 2, 0, 2, 3]);
  detectorGeometry.computeVertexNormals();
  return detectorGeometry;
}

async function fetchRawAsset(asset, expectedBytes, label) {
  const response = await fetch(asset.path);
  if (!response.ok) throw new Error(`${label} 读取失败：${response.status}`);
  const values = new Uint8Array(await response.arrayBuffer());
  if (values.byteLength !== expectedBytes || asset.byte_length !== expectedBytes) {
    throw new Error(`${label} 长度异常：${values.byteLength} / ${expectedBytes}`);
  }
  return values;
}

function makeRgbaTexture(bytes, width, height) {
  const texture = new THREE.DataTexture(bytes, width, height, THREE.RGBAFormat, THREE.UnsignedByteType);
  texture.flipY = false;
  texture.unpackAlignment = 1;
  texture.minFilter = THREE.LinearFilter;
  texture.magFilter = THREE.LinearFilter;
  texture.needsUpdate = true;
  return texture;
}

function updateDynamicFrame(frameIndex, vesselBlend = null) {
  if (!dynamicCine) return;
  const data = geometry.dynamic_xa;
  const pixelCount = data.rows * data.columns;
  const bounded = ((Number(frameIndex) % data.frame_count) + data.frame_count) % data.frame_count;
  const offset = bounded * pixelCount;
  const threshold = Number($("vessel-threshold").value);
  const showVessel = $("show-vessel-mask").checked;
  const showCenterline = $("show-centerline").checked;
  const showComparison = Boolean(
    $("show-repair-comparison")?.checked
    && dynamicCine.baselineMasks
    && dynamicCine.repairedMasks
  );
  const grayRgba = dynamicCine.grayTexture.image.data;
  const overlayRgba = dynamicCine.overlayTexture.image.data;
  const comparisonPanel = $("repair-comparison-panel");
  const baselinePreviewRgba = dynamicCine.baselinePreviewImage?.data;
  const repairedPreviewRgba = dynamicCine.repairedPreviewImage?.data;
  comparisonPanel?.classList.toggle("hidden", !showComparison);
  for (let index = 0; index < pixelCount; index += 1) {
    const value = dynamicCine.cine[offset + index];
    const probability = dynamicCine.probabilities[offset + index] / 255;
    const inMask = dynamicCine.masks[offset + index] > 0;
    const baselineMask = showComparison && dynamicCine.baselineMasks[offset + index] > 0;
    const repairedMask = showComparison && dynamicCine.repairedMasks[offset + index] > 0;
    const centerline = dynamicCine.centerlines[offset + index] > 0;
    const rgba = index * 4;
    grayRgba[rgba] = value;
    grayRgba[rgba + 1] = value;
    grayRgba[rgba + 2] = value;
    grayRgba[rgba + 3] = 255;
    if (showComparison) {
      overlayRgba[rgba] = baselineMask && repairedMask ? 72 : repairedMask ? 255 : 67;
      overlayRgba[rgba + 1] = baselineMask && repairedMask ? 225 : repairedMask ? 184 : 156;
      overlayRgba[rgba + 2] = baselineMask && repairedMask ? 161 : repairedMask ? 71 : 255;
      overlayRgba[rgba + 3] = showVessel && (baselineMask || repairedMask) ? 230 : 0;
      const previewGray = Math.round(value * 0.62);
      baselinePreviewRgba[rgba] = baselineMask ? 67 : previewGray;
      baselinePreviewRgba[rgba + 1] = baselineMask ? 156 : previewGray;
      baselinePreviewRgba[rgba + 2] = baselineMask ? 255 : previewGray;
      baselinePreviewRgba[rgba + 3] = 255;
      repairedPreviewRgba[rgba] = repairedMask ? 255 : previewGray;
      repairedPreviewRgba[rgba + 1] = repairedMask ? 184 : previewGray;
      repairedPreviewRgba[rgba + 2] = repairedMask ? 71 : previewGray;
      repairedPreviewRgba[rgba + 3] = 255;
    } else {
      overlayRgba[rgba] = 20;
      overlayRgba[rgba + 1] = 235;
      overlayRgba[rgba + 2] = 145;
      overlayRgba[rgba + 3] = showVessel && inMask && probability >= threshold ? Math.round(80 + 175 * probability) : 0;
    }
    if (showCenterline && centerline) {
      overlayRgba[rgba] = 255;
      overlayRgba[rgba + 1] = 224;
      overlayRgba[rgba + 2] = 40;
      overlayRgba[rgba + 3] = 255;
    }
  }
  dynamicCine.grayTexture.needsUpdate = true;
  dynamicCine.overlayTexture.needsUpdate = true;
  if (showComparison) {
    dynamicCine.baselinePreviewContext.putImageData(dynamicCine.baselinePreviewImage, 0, 0);
    dynamicCine.repairedPreviewContext.putImageData(dynamicCine.repairedPreviewImage, 0, 0);
  }
  dynamicCine.frameIndex = bounded;
  $("cine-frame").value = bounded;
  $("cine-frame-value").value = `${bounded + 1} / ${data.frame_count}`;
  const elapsed = data.frame_times_ms.slice(0, bounded).reduce((sum, value) => sum + value, 0) / 1000;
  $("cine-time").textContent = `${elapsed.toFixed(2)} s`;
  refreshVesselDiagnosticStatus();
  if (vesselBlend) updateRunVesselBlend(comparisonRunStates.get(activeDynamicRunId), vesselBlend);
  else updateDynamicVesselFrame(bounded);
}

function singleUseFullClip() {
  return Boolean($("multiseries-full-clip")?.checked);
}

function frameBlendForRun(run, elapsedMs, useFullClip = false) {
  const [start, end] = playbackFrameRange(run, useFullClip);
  const fallback = Number.isFinite(Number(run?.median_frame_time_ms)) && Number(run.median_frame_time_ms) > 0
    ? Number(run.median_frame_time_ms)
    : 1;
  const source = Array.isArray(run?.frame_times_ms) ? run.frame_times_ms : [];
  const activeDurations = Array.from({ length: end - start + 1 }, (_, offset) => {
    const duration = Number(source[start + offset]);
    return Number.isFinite(duration) && duration > 0 ? duration : fallback;
  });
  const blend = frameBlendAtElapsed(activeDurations, elapsedMs);
  return {
    runId: run.run_id,
    fromIndex: start + blend.fromIndex,
    toIndex: start + blend.toIndex,
    alpha: blend.alpha,
  };
}

function updateSinglePlayback() {
  if (!dynamicCine) return;
  const useFullClip = singleUseFullClip();
  const frameIndex = frameAtElapsed(geometry.dynamic_xa, dynamicCine.elapsedMs, useFullClip);
  const blend = frameBlendForRun(geometry.dynamic_xa, dynamicCine.elapsedMs, useFullClip);
  updateDynamicFrame(frameIndex, blend);
}

function configureSinglePlaybackRange() {
  if (!dynamicCine) return;
  const [start, end] = playbackFrameRange(geometry.dynamic_xa, singleUseFullClip());
  const slider = $("cine-frame");
  slider.min = start;
  slider.max = end;
  dynamicCine.elapsedMs = elapsedBeforeFrame(
    geometry.dynamic_xa,
    Number(slider.value),
    singleUseFullClip(),
  );
  updateSinglePlayback();
}

function scrubSinglePlayback(frameIndex) {
  if (!dynamicCine) return;
  dynamicCine.elapsedMs = elapsedBeforeFrame(geometry.dynamic_xa, frameIndex, singleUseFullClip());
  updateSinglePlayback();
}

function cineAnimation(timestamp) {
  if (!dynamicCine?.playing) return;
  dynamicCine.animationRequest = null;
  if (document.hidden) {
    dynamicCine.lastTimestamp = null;
    return;
  }
  if (dynamicCine.lastTimestamp === null) dynamicCine.lastTimestamp = timestamp;
  const delta = timestamp - dynamicCine.lastTimestamp;
  dynamicCine.lastTimestamp = timestamp;
  if (multiseriesPlayer.mode === "estimated") {
    const speed = Number($("multiseries-speed").value);
    selectedPhaseGroups().forEach((groupName) => {
      multiseriesPlayer.groupElapsedMs[groupName] += delta * speed;
    });
    updateEstimatedPlayback();
  } else if (multiseriesPlayer.mode === "independent") {
    multiseriesPlayer.elapsedMs += delta * Number($("multiseries-speed").value);
    updateIndependentPlayback();
  } else {
    dynamicCine.elapsedMs += delta * Number($("cine-speed").value);
    updateSinglePlayback();
  }
  renderFrame(delta);
  if (!document.hidden) dynamicCine.animationRequest = requestAnimationFrame(cineAnimation);
}

function setCinePlaying(playing) {
  if (!dynamicCine) return;
  if (playing) cancelPendingRender();
  const switchedToSingle = multiseriesPlayer.mode !== "single";
  multiseriesPlayer.mode = "single";
  if ($("multiseries-play-mode")) $("multiseries-play-mode").value = "single";
  if (switchedToSingle) {
    dynamicCine.elapsedMs = 0;
    configureSinglePlaybackRange();
  }
  dynamicCine.playing = playing;
  dynamicCine.lastTimestamp = null;
  $("cine-play").textContent = playing ? "暂停" : "播放";
  $("multiseries-play").textContent = playing ? "暂停" : "播放全部";
  if (playing && !document.hidden && !dynamicCine.animationRequest) dynamicCine.animationRequest = requestAnimationFrame(cineAnimation);
  else if (!playing && dynamicCine.animationRequest) {
    cancelAnimationFrame(dynamicCine.animationRequest);
    dynamicCine.animationRequest = null;
  }
  refreshComparisonVisibility();
}

function setMultiseriesPlaying(playing) {
  if (!dynamicCine) return;
  const mode = $("multiseries-play-mode").value;
  const estimatedEnabled = !$("multiseries-play-mode").querySelector('option[value="estimated"]').disabled;
  if (!estimatedPlaybackRequestAllowed(mode, playing, estimatedEnabled)) return;
  if (playing) cancelPendingRender();
  multiseriesPlayer.mode = mode;
  if (mode === "independent" && playing) {
    $("show-all-vessel-meshes").checked = true;
    refreshComparisonVisibility();
    multiseriesPlayer.elapsedMs = 0;
    updateIndependentPlayback();
  }
  if (mode === "estimated" && playing) {
    $("show-all-vessel-meshes").checked = true;
    groups.vesselComparisons.visible = true;
    if ($("show-vessel-3d")) {
      $("show-vessel-3d").checked = true;
      groups.dynamicVessel3d.visible = true;
    }
    selectedPhaseGroups().forEach((groupName) => { multiseriesPlayer.groupElapsedMs[groupName] = 0; });
    updateEstimatedPlayback();
  }
  dynamicCine.playing = playing;
  dynamicCine.lastTimestamp = null;
  $("cine-play").textContent = playing && mode === "single" ? "暂停" : "播放";
  $("multiseries-play").textContent = playing ? "暂停" : "播放全部";
  if (playing && !document.hidden && !dynamicCine.animationRequest) dynamicCine.animationRequest = requestAnimationFrame(cineAnimation);
  else if (!playing && dynamicCine.animationRequest) {
    cancelAnimationFrame(dynamicCine.animationRequest);
    dynamicCine.animationRequest = null;
  }
}

async function addDynamicXaCine() {
  if (!geometry.dynamic_xa) return;
  document.querySelectorAll(".dynamic-only").forEach((node) => node.classList.remove("hidden"));
  setLoading("正在载入动态 XA 与血管分割帧栈…");
  const data = geometry.dynamic_xa;
  const pixelCount = data.rows * data.columns;
  const expectedBytes = pixelCount * data.frame_count;
  const cine = await fetchRawAsset(data.assets.cine, expectedBytes, "动态 XA");
  let probabilities = new Uint8Array(expectedBytes);
  let masks = new Uint8Array(expectedBytes);
  let centerlines = new Uint8Array(expectedBytes);
  let baselineMasks = null;
  let repairedMasks = null;
  let repairLinks = [];
  if (data.has_vessel !== false) {
    [probabilities, masks, centerlines] = await Promise.all([
      fetchRawAsset(data.assets.probabilities, expectedBytes, "血管概率图"),
      fetchRawAsset(data.assets.masks, expectedBytes, "血管掩膜"),
      fetchRawAsset(data.assets.centerlines, expectedBytes, "血管中心线"),
    ]);
    if (data.assets.baseline_masks && data.assets.repaired_masks) {
      [baselineMasks, repairedMasks] = await Promise.all([
        fetchRawAsset(data.assets.baseline_masks, expectedBytes, "基线血管掩膜"),
        fetchRawAsset(data.assets.repaired_masks, expectedBytes, "修复血管掩膜"),
      ]);
    }
  }
  const repairReport = geometry.dynamic_vessel_3d?.repair_report;
  if (repairReport?.link_geometry_available === true && repairReport.asset?.path) {
    const response = await fetch(repairReport.asset.path);
    if (!response.ok) throw new Error(`血管修复连线报告读取失败：${response.status}`);
    const reportPayload = await response.json();
    repairLinks = Array.isArray(reportPayload.links) ? reportPayload.links : [];
  }
  const baselinePreviewCanvas = $("baseline-repair-preview");
  const repairedPreviewCanvas = $("repaired-repair-preview");
  baselinePreviewCanvas.width = data.columns;
  baselinePreviewCanvas.height = data.rows;
  repairedPreviewCanvas.width = data.columns;
  repairedPreviewCanvas.height = data.rows;
  const baselinePreviewContext = baselinePreviewCanvas.getContext("2d");
  const repairedPreviewContext = repairedPreviewCanvas.getContext("2d");
  if (!baselinePreviewContext || !repairedPreviewContext) {
    throw new Error("浏览器不支持血管分割并排对照画布");
  }
  const baselinePreviewImage = baselinePreviewContext.createImageData(data.columns, data.rows);
  const repairedPreviewImage = repairedPreviewContext.createImageData(data.columns, data.rows);
  const grayTexture = makeRgbaTexture(new Uint8Array(pixelCount * 4), data.columns, data.rows);
  const overlayTexture = makeRgbaTexture(new Uint8Array(pixelCount * 4), data.columns, data.rows);
  const corners = data.detector.corners_lps_mm.map(vector);
  const towardSource = vector(data.n_lps).multiplyScalar(-0.75);
  const baseMaterial = new THREE.MeshBasicMaterial({ map: grayTexture, transparent: true, opacity: Number($("cine-opacity").value), side: THREE.DoubleSide, depthWrite: false });
  const overlayMaterial = new THREE.MeshBasicMaterial({ map: overlayTexture, transparent: true, opacity: Number($("vessel-opacity").value), side: THREE.DoubleSide, depthWrite: false });
  const plane = new THREE.Mesh(detectorGeometryFromCorners(corners), baseMaterial);
  plane.name = `${data.series_label ?? "动态 XA"} 探测器`;
  plane.renderOrder = 7;
  plane.userData = { kind: "dynamic-detector", runId: data.run_id };
  renderingController.registerMesh(plane, "xa-image");
  groups.dynamicXa.add(plane);
  pickables.push(plane);
  if (data.has_vessel !== false) {
    const overlay = new THREE.Mesh(detectorGeometryFromCorners(corners, towardSource), overlayMaterial);
    overlay.name = `${data.series_label ?? "动态 XA"} 血管分割`;
    overlay.renderOrder = 8;
    overlay.userData = { kind: "dynamic-vessel", runId: data.run_id };
    renderingController.registerMesh(overlay, "xa-image");
    groups.dynamicXa.add(overlay);
    pickables.push(overlay);
  }

  const source = vector(data.source_lps_mm);
  const sourceMarker = sphereAt(source, 9, 0xff8a3c, { emissive: 0x4d2100, emissiveIntensity: 0.8 });
  sourceMarker.userData = { kind: "dynamic-source", runId: data.run_id };
  groups.dynamicXa.add(sourceMarker);
  pickables.push(sourceMarker);
  corners.forEach((corner) => groups.dynamicXa.add(lineBetween(source, corner, 0x2de3b7, 0.34)));
  const detectorOutline = new THREE.Line(
    new THREE.BufferGeometry().setFromPoints([...corners, corners[0]]),
    new THREE.LineBasicMaterial({ color: 0x2de3b7 }),
  );
  renderingController.registerMesh(detectorOutline, "helper");
  groups.dynamicXa.add(detectorOutline);
  const label = makeLabel(data.series_label ?? "DYNAMIC XA", data.color ?? "#7df8d9", 22);
  label.position.copy(corners[0]).lerp(corners[1], 0.5).addScaledVector(vector(data.v_lps), -14);
  groups.dynamicXa.add(label);

  dynamicCine = {
    cine,
    probabilities,
    masks,
    centerlines,
    baselineMasks,
    repairedMasks,
    repairLinks,
    baselinePreviewContext,
    repairedPreviewContext,
    baselinePreviewImage,
    repairedPreviewImage,
    grayTexture,
    overlayTexture,
    baseMaterial,
    overlayMaterial,
    frameIndex: 0,
    playing: false,
    lastTimestamp: null,
    elapsedMs: 0,
    animationRequest: null,
  };
  configureSinglePlaybackRange();
}

function clearVesselDiagnostics() {
  const disposed = new Set();
  while (groups.vesselDiagnostics.children.length) {
    const child = groups.vesselDiagnostics.children[0];
    groups.vesselDiagnostics.remove(child);
    child.traverse((object) => {
      renderingController.unregisterMesh(object);
      if (object.geometry && !disposed.has(object.geometry)) {
        disposed.add(object.geometry);
        object.geometry.dispose();
      }
      const materials = Array.isArray(object.material) ? object.material : [object.material];
      materials.filter(Boolean).forEach((material) => {
        if (material.map && !disposed.has(material.map)) {
          disposed.add(material.map);
          material.map.dispose();
        }
        if (!disposed.has(material)) {
          disposed.add(material);
          material.dispose();
        }
      });
    });
  }
}

function detectorPointFromYx(yx, data, normalOffsetMm = -1.5) {
  const corners = data.detector.corners_lps_mm.map(vector);
  const u = data.columns > 1 ? THREE.MathUtils.clamp(Number(yx[1]) / (data.columns - 1), 0, 1) : 0.5;
  const v = data.rows > 1 ? THREE.MathUtils.clamp(Number(yx[0]) / (data.rows - 1), 0, 1) : 0.5;
  const top = corners[0].clone().lerp(corners[1], u);
  const bottom = corners[3].clone().lerp(corners[2], u);
  return top.lerp(bottom, v).addScaledVector(vector(data.n_lps), normalOffsetMm);
}

function refreshVesselDiagnosticObjects() {
  clearVesselDiagnostics();
  if (renderModeSelector.value !== "diagnostic" || !dynamicCine) return;
  const requestedIndex = dynamicCine.frameIndex ?? 0;
  if ($("show-low-confidence-links")?.checked) {
    dynamicCine.repairLinks
      .filter((link) => link.frame_index === requestedIndex)
      .forEach((link) => {
        const color = repairLinkDecisionColors[link.decision];
        if (color === undefined) return;
        const points = link.points_yx.map((point) => detectorPointFromYx(point, geometry.dynamic_xa));
        const line = new THREE.Line(
          new THREE.BufferGeometry().setFromPoints(points),
          new THREE.LineBasicMaterial({ color, transparent: true, opacity: 0.92 }),
        );
        line.name = `repair-link:${link.decision}`;
        line.renderOrder = 19;
        line.userData = { kind: "repair-link", decision: link.decision, frameIndex: requestedIndex };
        renderingController.registerMesh(line, "annotation");
        groups.vesselDiagnostics.add(line);
      });
  }
  if ($("show-topology-nodes")?.checked && dynamicVessel3d?.payload?.frames?.length) {
    const requestedFrame = dynamicVessel3d.payload.frames[
      THREE.MathUtils.clamp(requestedIndex, 0, dynamicVessel3d.payload.frames.length - 1)
    ];
    const displayIndex = Number.isInteger(requestedFrame?.display_source_frame_index)
      ? requestedFrame.display_source_frame_index
      : requestedIndex;
    const displayFrame = dynamicVessel3d.payload.frames[
      THREE.MathUtils.clamp(displayIndex, 0, dynamicVessel3d.payload.frames.length - 1)
    ];
    (displayFrame?.nodes ?? []).forEach((node) => {
      if (typeof node.topology_id !== "string" || !Array.isArray(node.point_lps_mm)) return;
      const label = makeLabel(node.topology_id, "#9cf7e8", 11);
      label.position.copy(vector(node.point_lps_mm));
      label.userData = { kind: "topology-node-label", topologyId: node.topology_id };
      groups.vesselDiagnostics.add(label);
    });
  }
}

function refreshVesselDiagnosticStatus() {
  const status = $("vessel-quality-status");
  if (!status) return;
  const vessel = geometry?.dynamic_vessel_3d;
  const comparison = $("show-repair-comparison");
  const lowConfidence = $("show-low-confidence-links");
  const topology = $("show-topology-nodes");
  if (!vessel) {
    clearVesselDiagnostics();
    status.textContent = "当前序列没有血管修复或三维血管数据。";
    if (comparison) comparison.disabled = true;
    if (lowConfidence) lowConfidence.disabled = true;
    if (topology) topology.disabled = true;
    return;
  }
  const repaired = vessel.display_source === "repaired";
  if (comparison) {
    comparison.disabled = !repaired || !dynamicCine?.baselineMasks || !dynamicCine?.repairedMasks;
  }
  const linkGeometryAvailable = (
    vessel.repair_report?.link_geometry_available === true
    && Array.isArray(dynamicCine?.repairLinks)
    && dynamicCine.repairLinks.length > 0
  );
  if (lowConfidence) {
    lowConfidence.disabled = !linkGeometryAvailable || renderModeSelector.value !== "diagnostic";
    if (lowConfidence.disabled) lowConfidence.checked = false;
  }
  if (topology) {
    topology.disabled = renderModeSelector.value !== "diagnostic";
    if (topology.disabled) topology.checked = false;
  }
  const sourceText = repaired
    ? "修复栈已验证；未通过帧保留基线"
    : `显示基线栈（${vessel.fallback_reason ?? "unknown"}）`;
  const messages = [sourceText];
  if (!linkGeometryAvailable) {
    messages.push("修复报告没有可绘制候选路径，低置信度连线已禁用");
  }
  if (topology?.checked && dynamicVessel3d?.payload?.frames?.length) {
    const frameIndex = Math.max(
      0,
      Math.min(dynamicVessel3d.payload.frames.length - 1, dynamicCine?.frameIndex ?? 0),
    );
    const ids = (dynamicVessel3d.payload.frames[frameIndex].nodes ?? [])
      .map((node) => node.topology_id)
      .filter((value) => typeof value === "string");
    messages.push(ids.length ? `拓扑节点：${ids.join("、")}` : "本帧没有可显示的拓扑节点");
  }
  status.textContent = messages.join("；");
  refreshVesselDiagnosticObjects();
}

function unregisterVesselMesh(mesh) {
  const pickableIndex = pickables.indexOf(mesh);
  if (pickableIndex >= 0) pickables.splice(pickableIndex, 1);
  renderingController.unregisterMesh(mesh);
}

function disposeFrameGroup(frameGroup) {
  disposeConnectedVesselGroup(frameGroup, { unregister: unregisterVesselMesh });
}

function createComparisonRunState(run, payload, parent) {
  const group = new THREE.Group();
  group.name = `${run.series_label} 动态 Mesh`;
  group.userData = {
    runId: run.run_id,
    anatomyGroup: run.anatomy_group,
    vesselFallbackReason: null,
  };
  parent.add(group);
  const state = {
    run,
    payload,
    group,
    cache: null,
    currentFrame: null,
    requestedFrame: null,
    vesselFallbackReason: null,
    parent,
    interpolation: { group: null },
  };
  state.cache = new LruFrameCache(8, disposeFrameGroup);
  comparisonRunStates.set(run.run_id, state);
  return state;
}

function createRunVesselMaterial(state) {
  const color = new THREE.Color(state.run.color ?? "#55d9d1");
  return new THREE.MeshStandardMaterial({
    color,
    emissive: color.clone().multiplyScalar(0.25),
    emissiveIntensity: 0.55,
    roughness: 0.32,
    transparent: true,
    opacity: Number($("vessel-3d-opacity").value),
    depthWrite: true,
  });
}

function registerVesselGroup(
  state,
  vesselGroup,
  frame,
  frameIndex,
  { pickable = true, requestedFrameIndex = frameIndex, fallbackReason = null } = {},
) {
  const kind = state.run.run_id === activeDynamicRunId ? "vessel-primary" : "vessel-comparison";
  vesselGroup.userData = {
    ...vesselGroup.userData,
    kind: "multiseries-vessel-3d",
    run: state.run,
    frameIndex,
    requestedFrameIndex,
    frame,
    vesselFallbackReason: fallbackReason,
  };
  vesselGroup.traverse((object) => {
    if (!object.isMesh) return;
    object.userData = {
      ...object.userData,
      kind: "multiseries-vessel-3d",
      run: state.run,
      frameIndex,
      requestedFrameIndex,
      frame,
      vesselFallbackReason: fallbackReason,
    };
    renderingController.registerMesh(object, kind);
    if (pickable) pickables.push(object);
  });
}

function buildRunVesselFrame(state, frameIndex) {
  const cached = state.cache.get(frameIndex);
  if (cached) return cached;
  const frame = state.payload.frames[frameIndex];
  if (!frame) throw new RangeError(`Missing vessel frame ${frameIndex}`);
  const frameGroup = buildConnectedVesselGroup(frame, createRunVesselMaterial(state), {
    THREE,
    radialSegments: 6,
    tubularSegmentsPerSample: 2,
    ownsMaterial: true,
  });
  frameGroup.visible = false;
  frameGroup.name = `${state.run.series_label} 3D frame ${frameIndex}`;
  registerVesselGroup(state, frameGroup, frame, frameIndex);
  state.group.add(frameGroup);
  state.cache.set(frameIndex, frameGroup);
  return frameGroup;
}

function disposeRunInterpolation(state) {
  const group = state?.interpolation?.group;
  if (!group) return;
  state.interpolation.group = null;
  disposeConnectedVesselGroup(group, { unregister: unregisterVesselMesh });
}

function updateRunVesselFrame(state, frameIndex, fallbackReason = null) {
  if (!state) return;
  disposeRunInterpolation(state);
  if (state.currentFrame !== null) {
    const previous = state.cache.get(state.currentFrame);
    if (previous) previous.visible = false;
  }

  const resolution = resolveVesselFrameBlend(
    state.payload.frames,
    { fromIndex: frameIndex, toIndex: frameIndex, alpha: 0 },
  );
  const reason = fallbackReason ?? resolution.reason;
  state.requestedFrame = frameIndex;
  state.vesselFallbackReason = reason;
  state.group.userData.vesselFallbackReason = reason;
  if (resolution.mode === "unavailable") {
    state.currentFrame = null;
    return;
  }

  const current = buildRunVesselFrame(state, resolution.frameIndex);
  current.userData.requestedFrameIndex = frameIndex;
  current.userData.vesselFallbackReason = reason;
  current.traverse((object) => {
    if (!object.isMesh) return;
    object.userData.requestedFrameIndex = frameIndex;
    object.userData.vesselFallbackReason = reason;
  });
  current.visible = true;
  state.currentFrame = resolution.frameIndex;
}

function disposeComparisonRunStates() {
  comparisonRunStates.forEach((state) => {
    disposeRunInterpolation(state);
    state.cache.clear();
  });
  comparisonRunStates.clear();
}

function updateRunVesselBlend(state, blend) {
  if (!state || blend.fromIndex < 0 || blend.toIndex < 0) return;
  if (blend.runId !== state.run.run_id) return;
  if (!runMeshIsVisible(state)) return;
  const resolution = resolveVesselFrameBlend(state.payload.frames, blend);
  if (resolution.mode === "frame" || resolution.mode === "fallback") {
    const requestedFrameIndex = blend.alpha <= 0.5 ? blend.fromIndex : blend.toIndex;
    updateRunVesselFrame(
      state,
      resolution.mode === "fallback" ? requestedFrameIndex : resolution.frameIndex,
      resolution.reason,
    );
    return;
  }
  if (resolution.mode === "unavailable") {
    disposeRunInterpolation(state);
    if (state.currentFrame !== null) {
      const previous = state.cache.get(state.currentFrame);
      if (previous) previous.visible = false;
    }
    state.currentFrame = null;
    state.requestedFrame = blend.alpha <= 0.5 ? blend.fromIndex : blend.toIndex;
    state.vesselFallbackReason = resolution.reason;
    state.group.userData.vesselFallbackReason = resolution.reason;
    return;
  }

  if (state.currentFrame !== null) {
    const previous = state.cache.get(state.currentFrame);
    if (previous) previous.visible = false;
  }
  disposeRunInterpolation(state);
  const fromFrame = state.payload.frames[resolution.fromIndex];
  const toFrame = state.payload.frames[resolution.toIndex];
  const interpolatedFrame = interpolateVesselFrame(fromFrame, toFrame, resolution.alpha);
  const interpolationGroup = buildConnectedVesselGroup(
    interpolatedFrame,
    createRunVesselMaterial(state),
    {
      THREE,
      radialSegments: 6,
      tubularSegmentsPerSample: 2,
      ownsMaterial: true,
    },
  );
  interpolationGroup.name = `${state.run.series_label} graph interpolation`;
  interpolationGroup.userData = {
    ...interpolationGroup.userData,
    fromFrameIndex: resolution.fromIndex,
    toFrameIndex: resolution.toIndex,
    alpha: resolution.alpha,
    vesselFallbackReason: null,
  };
  const displayFrameIndex = resolution.alpha <= 0.5 ? resolution.fromIndex : resolution.toIndex;
  registerVesselGroup(
    state,
    interpolationGroup,
    interpolatedFrame,
    displayFrameIndex,
    {
      pickable: false,
      requestedFrameIndex: displayFrameIndex,
      fallbackReason: null,
    },
  );
  state.group.add(interpolationGroup);
  state.interpolation.group = interpolationGroup;
  state.currentFrame = displayFrameIndex;
  state.requestedFrame = displayFrameIndex;
  state.vesselFallbackReason = null;
  state.group.userData.vesselFallbackReason = null;
}

function updateDynamicVesselFrame(frameIndex) {
  updateRunVesselFrame(comparisonRunStates.get(activeDynamicRunId), frameIndex);
}

function runMeshIsVisible(state) {
  return state.group.visible && Boolean(state.parent?.visible);
}

function seriesLabel(run) {
  const seriesNumber = Number(String(run.run_id).match(/(\d+)$/)?.[1]);
  return Number.isInteger(seriesNumber) ? `Series ${seriesNumber}` : run.series_label;
}

function vesselFallbackText(state) {
  const labels = {
    "rejected-frame": "拒绝帧，显示最近可接受观测",
    "topology-change": "拓扑变化，显示最近可接受观测",
    "legacy-keyframe": "旧版数据，显示最近原始帧",
  };
  const reason = state?.vesselFallbackReason;
  return reason ? ` · 回退：${labels[reason] ?? reason}` : "";
}

function renderMultiseriesFrameStatus() {
  const status = $("multiseries-frame-status");
  if (!status) return;
  status.replaceChildren();
  if (multiseriesPlayer.mode === "estimated") {
    phaseRunIds.forEach((runId) => {
      const state = comparisonRunStates.get(runId);
      const line = document.createElement("div");
      if (state?.phasePlaybackStatus === "synchronized" && state.currentFrame !== null) {
        line.textContent = `${seriesLabel(state.run)} · frame ${state.currentFrame + 1}/${state.run.frame_count} · 估计相位${vesselFallbackText(state)}`;
      } else {
        line.textContent = `${seriesLabel(state?.run ?? { run_id: runId })} · 已隐藏 · ${state?.phasePlaybackStatus ?? "无可用 Mesh"}`;
      }
      status.appendChild(line);
    });
    return;
  }
  if (multiseriesPlayer.mode !== "independent") return;
  [...comparisonRunStates.values()].sort((left, right) => left.run.run_id.localeCompare(right.run.run_id)).forEach((state) => {
    if (!runMeshIsVisible(state) || state.currentFrame === null) return;
    const line = document.createElement("div");
    line.textContent = `${seriesLabel(state.run)} · frame ${state.currentFrame + 1}/${state.run.frame_count} · 未同步${vesselFallbackText(state)}`;
    status.appendChild(line);
  });
}

function updateIndependentPlayback() {
  const useFullClip = $("multiseries-full-clip").checked;
  const activeState = comparisonRunStates.get(activeDynamicRunId);
  if (!activeState) updateDynamicFrame(frameAtElapsed(geometry.dynamic_xa, multiseriesPlayer.elapsedMs, useFullClip));
  comparisonRunStates.forEach((state) => {
    const frameIndex = frameAtElapsed(state.run, multiseriesPlayer.elapsedMs, useFullClip);
    const blend = frameBlendForRun(state.run, multiseriesPlayer.elapsedMs, useFullClip);
    if (state.run.run_id === activeDynamicRunId) updateDynamicFrame(frameIndex, blend);
    else updateRunVesselBlend(state, blend);
  });
  renderMultiseriesFrameStatus();
}

function updateEstimatedPlayback() {
  if (!phasePlayback.report) return;
  const selected = new Set(selectedPhaseGroups());
  const groupStates = new Map(phaseGroupNames.map((groupName) => [
    groupName,
    estimatedGroupPlaybackState(
      phasePlayback.report,
      geometry.dynamic_xa_runs,
      groupName,
      multiseriesPlayer.groupElapsedMs[groupName],
      phasePlayback.overrides,
    ),
  ]));
  phaseRunIds.forEach((runId) => {
    const state = comparisonRunStates.get(runId);
    if (!state) return;
    const groupName = phaseGroupForRun(runId);
    const runPhase = phasePlayback.report.runs[runId];
    if (!selected.has(groupName)) {
      state.group.visible = false;
      state.phasePlaybackStatus = `${phaseGroupLabel(groupName)} 未选择`;
      return;
    }
    const groupState = groupStates.get(groupName);
    const frameIndex = groupState.enabled ? groupState.frames[runId] : null;
    if (frameIndex === null) {
      state.group.visible = false;
      state.phasePlaybackStatus = runPhase.status === "insufficient"
        ? `相位不足：${runPhase.reason ?? "无可用周期"}`
        : "临时估计尚未人工确认";
      return;
    }
    state.group.visible = true;
    state.phasePlaybackStatus = "synchronized";
    if (runId === activeDynamicRunId) updateDynamicFrame(frameIndex);
    else updateRunVesselFrame(state, frameIndex);
  });
  renderMultiseriesFrameStatus();
}

async function addDynamicVessel3d() {
  if (!geometry.dynamic_vessel_3d) return;
  document.querySelectorAll(".dynamic-3d-only").forEach((node) => node.classList.remove("hidden"));
  setLoading("正在载入CT肋骨、心脏与动态血管3D路径…");
  const data = geometry.dynamic_vessel_3d;
  const thoraxAssets = resolveThoraxAssetRecords(data.assets.thorax);
  const [registrationThorax, displayThorax, pathResponse] = await Promise.all([
    new GLTFLoader().loadAsync(thoraxAssets.registration.path),
    new GLTFLoader().loadAsync(thoraxAssets.display.path),
    fetch(data.assets.vessel_paths.path),
  ]);
  if (!pathResponse.ok) throw new Error(`动态血管3D路径读取失败：${pathResponse.status}`);
  const payload = await pathResponse.json();
  await Promise.all([
    loadThoraxVariant("registration", registrationThorax.scene),
    loadThoraxVariant("display", displayThorax.scene),
  ]);
  setThoraxVariant(resolveThoraxVariant(renderModeSelector.value, thoraxVariantSelector.value));

  const heartBounds = new THREE.Box3().setFromObject(thoraxVariants.registration.heart);
  dynamicVessel3d = createComparisonRunState(geometry.dynamic_xa, payload, groups.dynamicVessel3d);
  dynamicVessel3d.focusCenter = heartBounds.getCenter(new THREE.Vector3());
  for (const [inputId, groupName] of [
    ["show-ct", "ct"],
    ["show-vertebrae", "vertebrae"],
    ["show-dr", "dr"],
    ["show-frustum", "frustum"],
    ["show-landmarks", "landmarks"],
    ["show-rays", "rays"],
    ["show-dynamic-xa", "dynamicXa"],
  ]) {
    $(inputId).checked = false;
    groups[groupName].visible = false;
  }
  updateDynamicVesselFrame(dynamicCine?.frameIndex ?? 0);
}

async function addMultiseriesComparisons() {
  if (!geometry.dynamic_xa_runs) return;
  setLoading("正在构建七序列投影平面与动态血管 Mesh…");
  for (const run of geometry.dynamic_xa_runs) {
    if (run.run_id === activeDynamicRunId) continue;
    const runGroup = new THREE.Group();
    runGroup.name = `${run.series_label} 投影平面`;
    runGroup.userData = { runId: run.run_id, anatomyGroup: run.anatomy_group };
    const corners = run.detector.corners_lps_mm.map(vector);
    const color = new THREE.Color(run.color ?? "#55d9d1");
    const outline = new THREE.Line(
      new THREE.BufferGeometry().setFromPoints([...corners, corners[0]]),
      new THREE.LineBasicMaterial({ color, transparent: true, opacity: 0.72 }),
    );
    renderingController.registerMesh(outline, "helper");
    runGroup.add(outline);
    const label = makeLabel(run.series_label, run.color ?? "#ffffff", 18);
    label.position.copy(corners[0]).lerp(corners[1], 0.5).addScaledVector(vector(run.v_lps), -12);
    runGroup.add(label);
    groups.xaComparisons.add(runGroup);
  }

  const runsById = new Map(geometry.dynamic_xa_runs.map((run) => [run.run_id, run]));
  await Promise.all((geometry.dynamic_vessel_3d_runs ?? []).map(async (vesselRun) => {
    if (vesselRun.run_id === activeDynamicRunId) return;
    const run = runsById.get(vesselRun.run_id) ?? vesselRun;
    if (run.has_vessel === false) return;
    const response = await fetch(vesselRun.assets.vessel_paths.path);
    if (!response.ok) throw new Error(`${run.series_label} 动态 Mesh 读取失败：${response.status}`);
    const payload = await response.json();
    const state = createComparisonRunState(run, payload, groups.vesselComparisons);
    updateRunVesselFrame(state, dynamicCine?.frameIndex ?? 0);
  }));
  refreshComparisonVisibility();
}

function addSourceAndFrustum() {
  const source = vector(geometry.xa.source_lps_mm);
  const isocenter = vector(geometry.xa.isocenter_lps_mm);
  const detectorCenter = vector(geometry.xa.detector_center_lps_mm);
  const sourceMesh = sphereAt(source, 11, 0xffa23f, { emissive: 0x5b2600, emissiveIntensity: 0.8 });
  sourceMesh.userData = { kind: "source" };
  groups.frustum.add(sourceMesh);
  pickables.push(sourceMesh);
  const sourceLabel = makeLabel("X-RAY SOURCE", "#ffc06b", 27);
  sourceLabel.position.copy(source).add(new THREE.Vector3(0, 0, 25));
  groups.frustum.add(sourceLabel);

  const isoMesh = sphereAt(isocenter, 6, 0x54e6ff, { emissive: 0x173d47, emissiveIntensity: 0.7 });
  isoMesh.userData = { kind: "isocenter" };
  groups.frustum.add(isoMesh);
  pickables.push(isoMesh);
  groups.frustum.add(lineBetween(source, detectorCenter, 0x8fc5ce, 0.5, true));
  geometry.xa.detector.corners_lps_mm.map(vector).forEach((corner) => groups.frustum.add(lineBetween(source, corner, 0x5ab7c2, 0.5)));
}

function addLandmarksAndRays() {
  const source = vector(geometry.xa.source_lps_mm);
  geometry.landmarks.forEach((landmark) => {
    const point = vector(landmark.point_lps_mm);
    const predicted = vector(landmark.predicted_detector_intersection_lps_mm);
    const observed = vector(landmark.observed_detector_point_lps_mm);
    const marker = sphereAt(point, 5.5, 0xffd84e, { emissive: 0x4d3900, emissiveIntensity: 0.65 }, "annotation");
    marker.userData = { kind: "landmark", landmark };
    marker.name = `标定点 ${landmark.id}`;
    groups.landmarks.add(marker);
    pickables.push(marker);
    const label = makeLabel(`${landmark.id} · ${landmark.vertebra}`, "#ffe58b", 21);
    label.position.copy(point).add(new THREE.Vector3(7, 7, 10));
    groups.landmarks.add(label);

    groups.rays.add(lineBetween(source, predicted, 0xffd84e, 0.76, false, "annotation"));
    const predictedMarker = new THREE.Mesh(
      new THREE.TorusGeometry(5.2, 1.4, 10, 24),
      new THREE.MeshBasicMaterial({ color: 0xff4de1, depthTest: false }),
    );
    predictedMarker.position.copy(predicted);
    predictedMarker.lookAt(source);
    predictedMarker.renderOrder = 14;
    renderingController.registerMesh(predictedMarker, "annotation");
    groups.rays.add(predictedMarker);
    const observedMarker = new THREE.Mesh(
      new THREE.RingGeometry(4.6, 6.2, 24),
      new THREE.MeshBasicMaterial({ color: 0x53f28f, side: THREE.DoubleSide, depthTest: false }),
    );
    observedMarker.position.copy(observed);
    observedMarker.lookAt(source);
    observedMarker.renderOrder = 15;
    renderingController.registerMesh(observedMarker, "annotation");
    groups.rays.add(observedMarker);
    groups.rays.add(lineBetween(predicted, observed, 0xff6d55, 1, false, "annotation"));
  });
}

function addLpsAxes() {
  const origin = vector(geometry.ct.bounds_min_lps_mm).add(new THREE.Vector3(18, 18, 18));
  const specifications = [
    [new THREE.Vector3(1, 0, 0), 0xff5c62, "+X · L"],
    [new THREE.Vector3(0, 1, 0), 0x68df87, "+Y · P"],
    [new THREE.Vector3(0, 0, 1), 0x62a8ff, "+Z · S"],
  ];
  specifications.forEach(([direction, color, text]) => {
    const arrow = new THREE.ArrowHelper(direction, origin, 70, color, 10, 6);
    arrow.traverse((object) => {
      if (object.material) renderingController.registerMesh(object, "annotation");
    });
    groups.axes.add(arrow);
    const label = makeLabel(text, `#${new THREE.Color(color).getHexString()}`, 20);
    label.position.copy(origin).addScaledVector(direction, 82);
    groups.axes.add(label);
  });
}

function updateSceneMetadata() {
  const calibration = geometry.xa.calibration;
  const status = calibration.quality_status;
  const badge = $("quality-badge");
  badge.textContent = status;
  badge.className = `badge ${status === "accepted" ? "badge-accepted" : status === "warning" ? "badge-warning" : "badge-diagnostic"}`;
  $("metric-status").textContent = status;
  $("metric-rms").textContent = `${fmt(calibration.rms_error_px)} px`;
  $("metric-max").textContent = `${fmt(calibration.max_error_px)} px`;
  $("metric-candidate").textContent = calibration.candidate_name;
  $("quality-reasons").textContent = calibration.quality_reasons.length ? calibration.quality_reasons.join("；") : "物理约束、方向顺序和重投影残差均通过。";
  $("case-id").textContent = geometry.metadata.case_id;
  $("ct-size").textContent = `${geometry.ct.size_xyz.join(" × ")} voxel`;
  $("xa-size").textContent = `${geometry.xa.detector.columns} × ${geometry.xa.detector.rows} px`;
  $("source-distances").textContent = `${fmt(geometry.xa.sdd_mm, 0)} / ${fmt(geometry.xa.sod_mm, 0)} mm`;
  $("gantry-angles").textContent = `${fmt(geometry.xa.effective_primary_deg, 2)}° / ${fmt(geometry.xa.effective_secondary_deg, 2)}°`;
  if (geometry.dynamic_xa) {
    const dynamic = geometry.dynamic_xa;
    $("dynamic-pose-status").textContent = dynamic.quality_status;
    $("dynamic-series-size").textContent = `${dynamic.columns} × ${dynamic.rows} × ${dynamic.frame_count}`;
    $("dynamic-temporal-iou").textContent = dynamic.temporal
      ? fmt(dynamic.temporal.quality_summary.median_warped_iou, 3)
      : "定位序列";
    $("dynamic-review-frames").textContent = dynamic.temporal
      ? `${dynamic.temporal.quality_summary.review_frames} / ${dynamic.frame_count}`
      : `0 / ${dynamic.frame_count}`;
    $("dynamic-quality-reasons").textContent = dynamic.quality_reasons.join("；");
  }
  if (geometry.dynamic_vessel_3d) {
    const depth = geometry.dynamic_vessel_3d;
    $("dynamic-depth-status").textContent = depth.quality_status;
    $("dynamic-principal-shift").textContent = `${fmt(depth.principal_shift_uv_px[0], 0)}, ${fmt(depth.principal_shift_uv_px[1], 0)} px`;
    $("dynamic-heart-coverage").textContent = `${(100 * depth.alignment_metrics.reference_centerline_coverage_after).toFixed(1)}%`;
    $("dynamic-multiview").textContent = depth.multiview_attempt?.quality_status ?? "未运行";
    $("dynamic-depth-reasons").textContent = depth.interpretation;
  }
}

function selectionDetails(userData) {
  if (userData.kind === "landmark") {
    const p = userData.landmark;
    return `<h3>标定点 ${p.id}</h3><dl>
      <div><dt>对应椎体</dt><dd>${p.vertebra}</dd></div>
      <div><dt>LPS 坐标</dt><dd>${fmtPoint(p.point_lps_mm)} mm</dd></div>
      <div><dt>深度 P10–P90</dt><dd>${p.depth_p10_p90_mm.map((v) => fmt(v, 1)).join("–")} mm</dd></div>
      <div><dt>医生像素</dt><dd>${fmtPoint(p.observed_uv_px)} px</dd></div>
      <div><dt>重投影像素</dt><dd>${fmtPoint(p.reprojected_uv_px)} px</dd></div>
      <div><dt>残差</dt><dd>${fmt(p.reprojection_error_px)} px</dd></div>
      <div><dt>深度质量</dt><dd>${p.quality}</dd></div>
    </dl>`;
  }
  if (userData.kind === "vertebra") {
    const r = userData.record;
    return `<h3>${userData.label}</h3><dl>
      <div><dt>中心 LPS</dt><dd>${r ? fmtPoint(r.centroid_lps_mm) : "–"} mm</dd></div>
      <div><dt>三角面</dt><dd>${r ? r.face_count.toLocaleString() : "–"}</dd></div>
      <div><dt>顶点</dt><dd>${r ? r.vertex_count.toLocaleString() : "–"}</dd></div>
    </dl>`;
  }
  if (userData.kind === "detector") return `<h3>DR 探测器平面</h3><p>原生 Series 7 / Frame 0 纹理；物理尺寸 ${fmt(geometry.xa.detector.physical_width_mm, 1)} × ${fmt(geometry.xa.detector.physical_height_mm, 1)} mm。</p>`;
  if (userData.kind === "dynamic-detector") return `<h3>${geometry.dynamic_xa.series_label}</h3><p>${geometry.dynamic_xa.frame_count} 帧二维透视影像，姿态状态：${geometry.dynamic_xa.quality_status}。</p><p>${geometry.dynamic_xa.interpretation}</p>`;
  if (userData.kind === "dynamic-vessel") return `<h3>动态血管分割</h3><p>${geometry.dynamic_xa.segmentation.method}</p><p>${geometry.dynamic_xa.segmentation.clinical_status}</p>`;
  if (userData.kind === "dynamic-vessel-3d") return `<h3>动态血管 3D Mesh</h3><p>帧 ${userData.frameIndex + 1}：${userData.frame.path_count} 段管状路径，${userData.frame.point_count} 个3D采样点。</p><p>${geometry.dynamic_vessel_3d.interpretation}</p>`;
  if (userData.kind === "multiseries-vessel-3d") return `<h3>${userData.run.series_label} 动态 3D Mesh</h3><p>帧 ${userData.frameIndex + 1}：${userData.frame.path_count} 段路径，${userData.frame.point_count} 个3D采样点。</p><p>${multiseriesSelectionPlaybackText(multiseriesPlayer.mode, comparisonRunStates.get(userData.run.run_id)?.phasePlaybackStatus)}</p>`;
  if (userData.kind === "heart-surface") return `<h3>CT 心脏表面</h3><p>TotalSegmentator 心脏分割；当前为 ${userData.thoraxVariant === "display" ? "平滑显示" : "配准原始"} 网格，作为单视角血管深度的近侧表面先验。</p>`;
  if (userData.kind === "thorax-bone") return `<h3>${userData.label}</h3><p>来自同一CT坐标系的肋骨或胸骨${userData.thoraxVariant === "display" ? "平滑显示" : "配准原始"}网格。</p>`;
  if (userData.kind === "dynamic-source") return `<h3>${geometry.dynamic_xa.series_label} X 射线源</h3><p>LPS：${fmtPoint(geometry.dynamic_xa.source_lps_mm)} mm</p>`;
  if (userData.kind === "source") return `<h3>X 射线源</h3><p>LPS：${fmtPoint(geometry.xa.source_lps_mm)} mm</p>`;
  if (userData.kind === "isocenter") return `<h3>等中心</h3><p>LPS：${fmtPoint(geometry.xa.isocenter_lps_mm)} mm</p>`;
  return `<p class="muted">未选择可检查对象。</p>`;
}

function focusCamera(position, target, up = new THREE.Vector3(0, 0, 1)) {
  camera.up.copy(up);
  camera.position.copy(position);
  controls.target.copy(target);
  camera.lookAt(target);
  controls.update();
}

function ctCenter() {
  return vector(geometry.ct.bounds_min_lps_mm).add(vector(geometry.ct.bounds_max_lps_mm)).multiplyScalar(0.5);
}

function applyView(name) {
  const center = ctCenter();
  const extent = vector(geometry.ct.bounds_max_lps_mm).sub(vector(geometry.ct.bounds_min_lps_mm));
  const distance = Math.max(extent.length() * 1.55, 650);
  const source = vector(geometry.xa.source_lps_mm);
  const detector = vector(geometry.xa.detector_center_lps_mm);
  const rayDirection = detector.clone().sub(source).normalize();
  if (name === "ap") focusCamera(center.clone().add(new THREE.Vector3(0, -distance, 0)), center);
  if (name === "lateral") focusCamera(center.clone().add(new THREE.Vector3(-distance, 0, 0)), center);
  if (name === "ray") focusCamera(center.clone().addScaledVector(rayDirection, -distance), center, vector(geometry.xa.v_lps).multiplyScalar(-1));
  if (name === "detector") focusCamera(detector.clone().addScaledVector(rayDirection, -450), detector, vector(geometry.xa.v_lps).multiplyScalar(-1));
  if (name === "dynamic-detector" && geometry.dynamic_xa) {
    const dynamicDetector = vector(geometry.dynamic_xa.detector_center_lps_mm);
    const dynamicRay = dynamicDetector.clone().sub(vector(geometry.dynamic_xa.source_lps_mm)).normalize();
    focusCamera(dynamicDetector.clone().addScaledVector(dynamicRay, -450), dynamicDetector, vector(geometry.dynamic_xa.v_lps).multiplyScalar(-1));
  }
  if (name === "vessel-3d" && dynamicVessel3d) {
    const heartCenter = dynamicVessel3d.focusCenter;
    const dynamicRay = vector(geometry.dynamic_xa.detector_center_lps_mm).sub(vector(geometry.dynamic_xa.source_lps_mm)).normalize();
    focusCamera(heartCenter.clone().addScaledVector(dynamicRay, -360).add(new THREE.Vector3(0, 0, 70)), heartCenter);
  }
  if (name === "fit") {
    const midpoint = source.clone().lerp(detector, 0.5);
    focusCamera(midpoint.clone().add(new THREE.Vector3(-1050, -1250, 700)), midpoint);
  }
  if (name === "reset") focusCamera(defaultView.position.clone(), defaultView.target.clone(), defaultView.up.clone());
}

function setViewerGroupVisibility(inputId, groupName, visible) {
  const group = groups[groupName];
  const wasVisible = group.visible;
  group.visible = visible;
  if (
    inputId === "show-vessel-3d"
    && visible
    && !wasVisible
    && !dynamicCine?.playing
  ) {
    resynchronizeVisibleRun(comparisonRunStates.get(activeDynamicRunId));
  }
}

function bindUi() {
  const visibility = { "show-ct": "ct", "show-vertebrae": "vertebrae", "show-dr": "dr", "show-frustum": "frustum", "show-landmarks": "landmarks", "show-rays": "rays", "show-dynamic-xa": "dynamicXa", "show-ribs": "ribs", "show-heart": "heart", "show-vessel-3d": "dynamicVessel3d" };
  Object.entries(visibility).forEach(([inputId, groupName]) => $(inputId).addEventListener("change", (event) => {
    setViewerGroupVisibility(inputId, groupName, event.target.checked);
  }));
  if (geometry.dynamic_xa_runs) {
    $("xa-run-select").addEventListener("change", (event) => {
      setCinePlaying(false);
      const url = new URL(window.location.href);
      url.searchParams.set("run", event.target.value);
      window.location.assign(url);
    });
    [
      "show-all-xa-planes",
      "show-all-vessel-meshes",
      "show-group-rca",
      "show-group-lca",
      "show-group-locator",
    ].forEach((id) => $(id).addEventListener("change", () => {
      refreshEstimatedModeAvailability();
      refreshComparisonVisibility();
    }));
    $("multiseries-play-mode").addEventListener("change", () => {
      if (dynamicCine?.playing) setMultiseriesPlaying(false);
      multiseriesPlayer.mode = $("multiseries-play-mode").value;
      if (multiseriesPlayer.mode === "estimated") {
        selectedPhaseGroups().forEach((groupName) => { multiseriesPlayer.groupElapsedMs[groupName] = 0; });
        updateEstimatedPlayback();
      } else {
        if (multiseriesPlayer.mode === "single") {
          dynamicCine.elapsedMs = 0;
          configureSinglePlaybackRange();
        }
        refreshComparisonVisibility();
      }
    });
    $("multiseries-play").addEventListener("click", () => setMultiseriesPlaying(!dynamicCine.playing));
    $("multiseries-full-clip").addEventListener("change", () => {
      if (multiseriesPlayer.mode === "independent") updateIndependentPlayback();
      if (multiseriesPlayer.mode === "single") configureSinglePlaybackRange();
    });
    $("phase-anchor-run").addEventListener("change", configurePhaseAnchorSlider);
    $("phase-anchor-frame").addEventListener("input", (event) => {
      $("phase-anchor-frame-value").value = event.target.value;
    });
    $("phase-anchor-apply").addEventListener("click", applySelectedPhaseAnchor);
    $("phase-overrides-export").addEventListener("click", downloadPhaseOverrides);
  }
  ["x", "y", "z"].forEach((axis) => $(`slice-${axis}`).addEventListener("input", () => updateSlice(axis)));

  const updateWindow = () => {
    Object.values(sliceMeshes).forEach((mesh) => {
      mesh.material.uniforms.windowCenter.value = Number($("window-center").value);
      mesh.material.uniforms.windowWidth.value = Math.max(1, Number($("window-width").value));
    });
  };
  $("window-center").addEventListener("input", updateWindow);
  $("window-width").addEventListener("input", updateWindow);
  $("window-presets").addEventListener("click", (event) => {
    const button = event.target.closest("button[data-window]");
    if (!button) return;
    const preset = geometry.ct.window_presets[button.dataset.window];
    $("window-center").value = preset.center;
    $("window-width").value = preset.width;
    document.querySelectorAll("[data-window]").forEach((node) => node.classList.toggle("active", node === button));
    updateWindow();
  });
  $("ct-opacity").addEventListener("input", (event) => {
    $("ct-opacity-value").value = Number(event.target.value).toFixed(2);
    Object.values(sliceMeshes).forEach((mesh) => { mesh.material.uniforms.opacity.value = Number(event.target.value); });
  });
  $("mesh-opacity").addEventListener("input", (event) => {
    event.stopPropagation();
    $("mesh-opacity-value").value = Number(event.target.value).toFixed(2);
    renderingController.setOpacity("bone", Number(event.target.value), groups.vertebrae);
  });
  $("dr-opacity").addEventListener("input", (event) => {
    $("dr-opacity-value").value = Number(event.target.value).toFixed(2);
    groups.dr.traverse((object) => { if (object.isMesh && object.material.map) object.material.opacity = Number(event.target.value); });
  });
  if (geometry.dynamic_vessel_3d) {
    $("rib-opacity").addEventListener("input", (event) => {
      event.stopPropagation();
      $("rib-opacity-value").value = Number(event.target.value).toFixed(2);
      renderingController.setOpacity("bone", Number(event.target.value), groups.ribs);
    });
    $("heart-opacity").addEventListener("input", (event) => {
      event.stopPropagation();
      $("heart-opacity-value").value = Number(event.target.value).toFixed(2);
      renderingController.setOpacity("heart", Number(event.target.value), groups.heart);
    });
    $("vessel-3d-opacity").addEventListener("input", (event) => {
      event.stopPropagation();
      $("vessel-3d-opacity-value").value = Number(event.target.value).toFixed(2);
      renderingController.setOpacity("vessel-primary", Number(event.target.value));
    });
  }
  if (geometry.dynamic_xa) {
    $("cine-play").addEventListener("click", () => setCinePlaying(!dynamicCine.playing));
    $("cine-frame").addEventListener("input", (event) => { setCinePlaying(false); scrubSinglePlayback(Number(event.target.value)); });
    $("cine-opacity").addEventListener("input", (event) => {
      $("cine-opacity-value").value = Number(event.target.value).toFixed(2);
      dynamicCine.baseMaterial.opacity = Number(event.target.value);
    });
    $("vessel-opacity").addEventListener("input", (event) => {
      $("vessel-opacity-value").value = Number(event.target.value).toFixed(2);
      dynamicCine.overlayMaterial.opacity = Number(event.target.value);
    });
    ["show-vessel-mask", "show-centerline", "vessel-threshold"].forEach((id) => $(id).addEventListener("input", () => {
      $("vessel-threshold-value").value = Number($("vessel-threshold").value).toFixed(2);
      updateDynamicFrame(dynamicCine.frameIndex);
    }));
    $("show-repair-comparison").addEventListener("change", () => {
      updateDynamicFrame(dynamicCine.frameIndex);
      requestRender();
    });
    ["show-low-confidence-links", "show-topology-nodes"].forEach((id) => {
      $(id).addEventListener("change", () => {
        refreshVesselDiagnosticStatus();
        requestRender();
      });
    });
  }
  document.querySelector(".view-grid").addEventListener("click", (event) => {
    const button = event.target.closest("button[data-view]");
    if (button) applyView(button.dataset.view);
  });

  const raycaster = new THREE.Raycaster();
  const pointer = new THREE.Vector2();
  canvas.addEventListener("pointerdown", (event) => {
    const rect = canvas.getBoundingClientRect();
    pointer.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
    pointer.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;
    raycaster.setFromCamera(pointer, camera);
    const hit = raycaster.intersectObjects(pickables.filter(isEffectivelyVisible), true)[0];
    if (!hit) return;
    let object = hit.object;
    while (object && !object.userData.kind) object = object.parent;
    if (object?.userData.kind) $("selection").innerHTML = selectionDetails(object.userData);
  });
}

function resize() {
  const width = viewportShell.clientWidth;
  const height = viewportShell.clientHeight;
  renderingController.resize(width, height, Math.min(window.devicePixelRatio || 1, maxPixelRatio));
  camera.aspect = width / Math.max(height, 1);
  camera.updateProjectionMatrix();
}

function requestRender() {
  if (document.hidden || renderRequest !== null || dynamicCine?.playing) return;
  renderRequest = requestAnimationFrame(() => {
    renderRequest = null;
    if (document.hidden || dynamicCine?.playing) return;
    renderFrame();
  });
}

function cancelPendingRender() {
  if (renderRequest !== null) cancelAnimationFrame(renderRequest);
  renderRequest = null;
}

function syncRenderingObservation() {
  const status = renderingController.getGtaoStatus();
  canvas.dataset.gtaoState = status.state;
  canvas.dataset.gtaoReason = status.reason ?? "";
  canvas.dataset.renderWarmupFrameCount = String(status.warmupFrameCount);
  canvas.dataset.renderSampleFrameCount = String(status.sampleFrameCount);
  canvas.dataset.renderCompletedWindowCount = String(status.completedWindowCount);
  canvas.dataset.renderP95FrameMs = status.lastP95FrameMs === null
    ? ""
    : status.lastP95FrameMs.toFixed(3);
}

function renderFrame(frameIntervalMs = null) {
  if (document.hidden) return;
  renderingController.render(camera);
  if (dynamicCine?.playing && Number.isFinite(frameIntervalMs) && frameIntervalMs > 0) {
    renderingController.reportFrameTiming(frameIntervalMs, true);
  }
  syncRenderingObservation();
}

async function initialize() {
  try {
    if (!renderer.capabilities.isWebGL2) throw new Error("当前浏览器不支持 WebGL2，无法显示 CT 三维纹理。");
    const response = await fetch("./scene_geometry.json");
    if (!response.ok) throw new Error(`scene_geometry.json 读取失败：${response.status}`);
    geometry = await response.json();
    configureDynamicRuns();
    await loadPhaseAlignmentReport();
    updateSceneMetadata();
    await loadCtVolume();
    addCtBox();
    await loadVertebrae();
    setLoading("正在构建 DR 探测器、射线与标定点…");
    await addDetectorPlane();
    await addDynamicXaCine();
    await addDynamicVessel3d();
    await addMultiseriesComparisons();
    setLoading("正在添加 X 射线源与视锥…");
    addSourceAndFrustum();
    setLoading("正在添加标定点与重投影射线…");
    addLandmarksAndRays();
    setLoading("正在添加 LPS 方向轴…");
    addLpsAxes();
    setLoading("正在绑定交互控件…");
    bindUi();
    refreshVesselDiagnosticStatus();

    const center = dynamicVessel3d?.focusCenter ?? ctCenter();
    defaultView = {
      position: dynamicVessel3d
        ? center.clone().add(new THREE.Vector3(-320, -380, 190))
        : center.clone().add(new THREE.Vector3(-760, -920, 460)),
      target: center.clone(),
      up: new THREE.Vector3(0, 0, 1),
    };
    applyView("reset");
    resize();
    setLoading("正在启动 WebGL 渲染…");
    loading.classList.add("hidden");
    controls.addEventListener("change", requestRender);
    document.addEventListener("input", requestRender);
    document.addEventListener("change", requestRender);
    document.addEventListener("click", requestRender);
    requestRender();
  } catch (error) {
    console.error(error);
    loadingMessage.textContent = `加载失败：${error.message}`;
    loading.querySelector(".spinner").style.display = "none";
  }
}

window.addEventListener("resize", resize);
document.addEventListener("visibilitychange", () => {
  if (document.hidden) {
    if (renderRequest !== null) cancelAnimationFrame(renderRequest);
    renderRequest = null;
    if (dynamicCine?.animationRequest) cancelAnimationFrame(dynamicCine.animationRequest);
    if (dynamicCine) {
      dynamicCine.animationRequest = null;
      dynamicCine.lastTimestamp = null;
    }
    return;
  }
  if (dynamicCine?.playing && !dynamicCine.animationRequest) {
    dynamicCine.animationRequest = requestAnimationFrame(cineAnimation);
  } else {
    requestRender();
  }
});
window.addEventListener("beforeunload", () => {
  if (renderRequest !== null) cancelAnimationFrame(renderRequest);
  clearVesselDiagnostics();
  disposeComparisonRunStates();
  renderingController.dispose();
});
initialize();
