import {
  isAcceptedVesselFrame,
  schemaV2FramesHaveMatchingTopology,
} from "./vessel_mesh.js";


function positiveDuration(value, fallback) {
  return Number.isFinite(value) && value > 0 ? value : fallback;
}

function frameCountFor(run) {
  const count = Number(run?.frame_count);
  if (!Number.isInteger(count) || count <= 0) {
    throw new RangeError("run.frame_count must be a positive integer");
  }
  return count;
}

export function playbackFrameRange(run, useFullClip = false) {
  const frameCount = frameCountFor(run);
  if (useFullClip) return [0, frameCount - 1];
  if (!Object.prototype.hasOwnProperty.call(run, "visible_frame_range")) {
    return [0, frameCount - 1];
  }

  const range = run.visible_frame_range;
  if (Array.isArray(range) && range.length === 2) {
    const start = range[0];
    const end = range[1];
    if (Number.isInteger(start) && Number.isInteger(end) && start >= 0 && end < frameCount && start <= end) {
      return [start, end];
    }
  }
  throw new RangeError("run.visible_frame_range must contain inclusive in-bounds frame endpoints");
}

function frameDurations(run, frameCount) {
  const fallback = positiveDuration(Number(run?.median_frame_time_ms), 1);
  const source = Array.isArray(run?.frame_times_ms) ? run.frame_times_ms : [];
  return Array.from({ length: frameCount }, (_, index) => positiveDuration(Number(source[index]), fallback));
}

function positiveModulo(value, modulus) {
  return ((value % modulus) + modulus) % modulus;
}

/**
 * Return the two observed frames surrounding elapsed time on a looping clock.
 * Invalid durations degrade to a one millisecond interval instead of creating
 * a non-finite clock or skipping an observed frame.
 */
export function frameBlendAtElapsed(frameTimesMs, elapsedMs) {
  const source = Array.isArray(frameTimesMs) ? frameTimesMs : [];
  if (source.length === 0) return { fromIndex: -1, toIndex: -1, alpha: 0 };
  if (source.length === 1) return { fromIndex: 0, toIndex: 0, alpha: 0 };

  const durations = source.map((value) => positiveDuration(Number(value), 1));
  const totalDuration = durations.reduce((sum, duration) => sum + duration, 0);
  const elapsed = Number.isFinite(Number(elapsedMs)) ? Number(elapsedMs) : 0;
  const wrapped = positiveModulo(elapsed, totalDuration);

  let boundary = 0;
  for (let fromIndex = 0; fromIndex < durations.length; fromIndex += 1) {
    if (wrapped === boundary) return { fromIndex, toIndex: fromIndex, alpha: 0 };
    const nextBoundary = boundary + durations[fromIndex];
    if (wrapped < nextBoundary) {
      return {
        fromIndex,
        toIndex: (fromIndex + 1) % durations.length,
        alpha: (wrapped - boundary) / durations[fromIndex],
      };
    }
    boundary = nextBoundary;
  }
  return { fromIndex: 0, toIndex: 0, alpha: 0 };
}

const VESSEL_PATH_IDENTITY_FIELDS = ["path_id", "topology_id"];
const VESSEL_PATH_CONNECTIVITY_FIELDS = [
  "parent_path_id",
  "source_node_id",
  "target_node_id",
  "start_node_id",
  "end_node_id",
];

function stablePathField(path, field) {
  if (!Object.prototype.hasOwnProperty.call(path, field)) return null;
  const value = path[field];
  return (typeof value === "string" && value.length > 0) || Number.isFinite(value) ? value : null;
}

function vesselFrameHasSchemaV2(frame) {
  return Object.prototype.hasOwnProperty.call(frame ?? {}, "nodes")
    || Object.prototype.hasOwnProperty.call(frame ?? {}, "edges");
}

/** Return true when corresponding buildable vessel paths have stable identical topology. */
export function vesselFramesHaveMatchingTopology(fromFrame, toFrame) {
  const fromHasSchemaV2 = vesselFrameHasSchemaV2(fromFrame);
  const toHasSchemaV2 = vesselFrameHasSchemaV2(toFrame);
  if (fromHasSchemaV2 || toHasSchemaV2) {
    return fromHasSchemaV2 && toHasSchemaV2
      && schemaV2FramesHaveMatchingTopology(fromFrame, toFrame);
  }
  if (!Array.isArray(fromFrame?.paths) || !Array.isArray(toFrame?.paths)) return false;
  const buildablePaths = (frame) => frame.paths.filter(
    (path) => Array.isArray(path?.points_lps_mm) && path.points_lps_mm.length >= 2,
  );
  const fromPaths = buildablePaths(fromFrame);
  const toPaths = buildablePaths(toFrame);
  const stableFields = [...VESSEL_PATH_IDENTITY_FIELDS, ...VESSEL_PATH_CONNECTIVITY_FIELDS];
  return fromPaths.length === toPaths.length && fromPaths.every((path, index) => {
    const candidate = toPaths[index];
    const hasStableIdentity = VESSEL_PATH_IDENTITY_FIELDS.some(
      (field) => stablePathField(path, field) !== null && stablePathField(candidate, field) !== null,
    );
    return hasStableIdentity
      && path.points_lps_mm.length === candidate.points_lps_mm.length
      && stableFields.every((field) => stablePathField(path, field) === stablePathField(candidate, field));
  });
}

function nearestAcceptedFrameIndex(frames, targetIndex) {
  let selected = null;
  let selectedDistance = Infinity;
  frames.forEach((frame, frameIndex) => {
    if (!isAcceptedVesselFrame(frame)) return;
    const distance = Math.abs(frameIndex - targetIndex);
    if (distance < selectedDistance || (distance === selectedDistance && frameIndex < selected)) {
      selected = frameIndex;
      selectedDistance = distance;
    }
  });
  return selected;
}

/** Resolve safe interpolation or a deterministic nearest accepted observation. */
export function resolveVesselFrameBlend(frames, blend) {
  if (!Array.isArray(frames)) throw new TypeError("frames must be an array");
  const fromIndex = blend?.fromIndex;
  const toIndex = blend?.toIndex;
  const alpha = Number(blend?.alpha);
  if (
    !Number.isInteger(fromIndex)
    || !Number.isInteger(toIndex)
    || fromIndex < 0
    || toIndex < 0
    || fromIndex >= frames.length
    || toIndex >= frames.length
    || !Number.isFinite(alpha)
    || alpha < 0
    || alpha > 1
  ) throw new RangeError("blend must reference in-bounds frames with alpha in [0,1]");

  const fromFrame = frames[fromIndex];
  const toFrame = frames[toIndex];
  const preferredIndex = alpha <= 0.5 ? fromIndex : toIndex;
  if (fromIndex === toIndex || alpha === 0) {
    if (isAcceptedVesselFrame(fromFrame)) return { mode: "frame", frameIndex: fromIndex, reason: null };
  } else if (
    isAcceptedVesselFrame(fromFrame)
    && isAcceptedVesselFrame(toFrame)
    && vesselFramesHaveMatchingTopology(fromFrame, toFrame)
  ) {
    if (vesselFrameHasSchemaV2(fromFrame) && vesselFrameHasSchemaV2(toFrame)) {
      return { mode: "interpolate", fromIndex, toIndex, alpha, reason: null };
    }
    return { mode: "fallback", frameIndex: preferredIndex, reason: "legacy-keyframe" };
  }

  const preferredFrame = frames[preferredIndex];
  const reason = !isAcceptedVesselFrame(fromFrame) || !isAcceptedVesselFrame(toFrame)
    ? "rejected-frame"
    : "topology-change";
  if (isAcceptedVesselFrame(preferredFrame)) {
    return { mode: "fallback", frameIndex: preferredIndex, reason };
  }
  const displaySource = preferredFrame?.display_source_frame_index;
  if (
    Number.isInteger(displaySource)
    && displaySource >= 0
    && displaySource < frames.length
    && isAcceptedVesselFrame(frames[displaySource])
  ) return { mode: "fallback", frameIndex: displaySource, reason };
  const nearest = nearestAcceptedFrameIndex(frames, preferredIndex);
  return nearest === null
    ? { mode: "unavailable", frameIndex: null, reason }
    : { mode: "fallback", frameIndex: nearest, reason };
}

const PHASE_RUN_IDS = ["series_01", "series_02", "series_03", "series_04", "series_05", "series_06"];
const PHASE_GROUP_MEMBERS = {
  rca: ["series_01", "series_02"],
  lca: ["series_03", "series_04", "series_05", "series_06"],
};

function normalizedPhaseSamples(runPhase) {
  const samples = runPhase?.phase_per_frame;
  if (!Array.isArray(samples)) throw new TypeError("runPhase.phase_per_frame must be an array");
  if (samples.length === 0) throw new RangeError("runPhase.phase_per_frame must not be empty");
  if (!samples.every((value) => Number.isFinite(value) && value >= 0 && value < 1)) {
    throw new RangeError("runPhase.phase_per_frame must contain finite normalized phases in [0,1)");
  }
  return samples;
}

function anchorOverrideFor(anchorOverrides, runId) {
  if (anchorOverrides instanceof Map) return anchorOverrides.has(runId) ? anchorOverrides.get(runId) : null;
  if (anchorOverrides && Object.prototype.hasOwnProperty.call(anchorOverrides, runId)) return anchorOverrides[runId];
  return null;
}

/**
 * Return the frame at elapsed time using an inclusive visible range unless the full clip is requested.
 * This is a timing-only clock and intentionally does not infer cardiac phase.
 */
export function frameAtElapsed(run, elapsedMs, useFullClip = false) {
  const frameCount = frameCountFor(run);
  const [start, end] = playbackFrameRange(run, useFullClip);
  const durations = frameDurations(run, frameCount);
  const intervalDuration = durations.slice(start, end + 1).reduce((sum, duration) => sum + duration, 0);
  const elapsed = Number.isFinite(Number(elapsedMs)) ? Number(elapsedMs) : 0;
  const wrapped = positiveModulo(elapsed, intervalDuration);

  let cursor = 0;
  for (let frame = start; frame <= end; frame += 1) {
    cursor += durations[frame];
    if (wrapped < cursor || frame === end) return frame;
  }
  return start;
}

/** Return a valid elapsed offset for a scrubbed frame, clamped to the active playback range. */
export function elapsedBeforeFrame(run, frameIndex, useFullClip = false) {
  const frameCount = frameCountFor(run);
  const [start, end] = playbackFrameRange(run, useFullClip);
  if (!Number.isInteger(frameIndex) || frameIndex < 0 || frameIndex >= frameCount) {
    throw new RangeError("frameIndex must be an in-bounds integer");
  }
  const frame = Math.min(Math.max(frameIndex, start), end);
  return frameDurations(run, frameCount).slice(start, frame).reduce((sum, duration) => sum + duration, 0);
}

/** Human-readable state for a selected multi-series mesh without promoting unusable phase data. */
export function multiseriesSelectionPlaybackText(mode, phasePlaybackStatus) {
  if (mode === "estimated" && phasePlaybackStatus === "synchronized") {
    return "该序列按估计相位播放；Mesh 使用单目心脏表面深度先验。";
  }
  return "该序列与其他序列未同步，Mesh 使用单目心脏表面深度先验。";
}

/**
 * Return the phase-sampled frame nearest to a normalized phase on a circular domain.
 * Missing or unusable phase data returns null so callers can retain unsynchronized playback.
 */
export function nearestFrameForPhase(runPhase, phase) {
  const samples = runPhase?.phase_per_frame;
  const targetValue = Number(phase);
  if (!Array.isArray(samples) || samples.length === 0 || !Number.isFinite(targetValue)) return null;

  const target = positiveModulo(targetValue, 1);
  let nearestFrame = null;
  let nearestDistance = Infinity;
  samples.forEach((value, frame) => {
    const sampleValue = Number(value);
    if (!Number.isFinite(sampleValue)) return;
    const sample = positiveModulo(sampleValue, 1);
    const difference = Math.abs(sample - target);
    const distance = Math.min(difference, 1 - difference);
    if (distance < nearestDistance) {
      nearestDistance = distance;
      nearestFrame = frame;
    }
  });
  return nearestFrame;
}

/** Return a new phase array whose selected anchor frame is phase zero. */
export function phaseWithAnchorOverride(runPhase, anchorFrame) {
  const samples = normalizedPhaseSamples(runPhase);
  if (!Number.isInteger(anchorFrame) || typeof anchorFrame === "boolean") {
    throw new TypeError("anchorFrame must be an integer");
  }
  if (anchorFrame < 0 || anchorFrame >= samples.length) {
    throw new RangeError("anchorFrame must be in bounds for phase_per_frame");
  }
  const anchorPhase = samples[anchorFrame];
  return samples.map((value) => {
    const shifted = positiveModulo(value - anchorPhase, 1);
    return Object.is(shifted, -0) ? 0 : shifted;
  });
}

/** Resolve a synchronized frame while failing closed for unusable or unconfirmed estimates. */
export function synchronizedFrame(runPhase, globalPhase, anchorOverride = null) {
  if (!Number.isFinite(globalPhase) || globalPhase < 0 || globalPhase >= 1) return null;
  if (!runPhase || !["automatic", "provisional", "insufficient"].includes(runPhase.status)) return null;
  if (runPhase.status === "insufficient") return null;
  const hasOverride = anchorOverride !== null && anchorOverride !== undefined;
  if (runPhase.status === "provisional" && runPhase.manual_confirmed !== true && !hasOverride) return null;

  try {
    const phases = hasOverride
      ? phaseWithAnchorOverride(runPhase, anchorOverride)
      : [...normalizedPhaseSamples(runPhase)];
    return nearestFrameForPhase({ phase_per_frame: phases }, globalPhase);
  } catch (error) {
    if (error instanceof TypeError || error instanceof RangeError) return null;
    throw error;
  }
}

/** Validate the browser-facing subset of a packaged phase report. */
export function validatePhasePlaybackReport(report) {
  if (!report || typeof report !== "object" || Array.isArray(report)) {
    throw new TypeError("Phase report must be an object");
  }
  if (report.schema_version !== 1) throw new RangeError("Incompatible phase report schema version");
  if (report.method !== "image-derived coronary motion phase") {
    throw new RangeError("Incompatible phase report method");
  }
  if (report.clinical_status !== "research-only") {
    throw new RangeError("Phase report must remain research-only");
  }
  if (!["ready", "partial", "insufficient"].includes(report.status)) {
    throw new RangeError("Invalid phase report status");
  }
  if (!report.runs || typeof report.runs !== "object" || Array.isArray(report.runs)) {
    throw new TypeError("Phase report runs must be an object");
  }
  if (Object.keys(report.runs).sort().join("|") !== [...PHASE_RUN_IDS].sort().join("|")) {
    throw new RangeError("Phase report must contain exactly Series 1-6");
  }
  if (!report.groups || typeof report.groups !== "object" || Array.isArray(report.groups)) {
    throw new TypeError("Phase report groups must be an object");
  }
  for (const [groupName, expectedMembers] of Object.entries(PHASE_GROUP_MEMBERS)) {
    const group = report.groups[groupName];
    if (!group || !Array.isArray(group.run_ids) || group.run_ids.join("|") !== expectedMembers.join("|")) {
      throw new RangeError(`Invalid ${groupName} phase group membership`);
    }
    if (group.reference_run_id !== null && !expectedMembers.includes(group.reference_run_id)) {
      throw new RangeError(`Invalid ${groupName} phase group reference`);
    }
  }
  for (const runId of PHASE_RUN_IDS) {
    const run = report.runs[runId];
    if (!run || !["automatic", "provisional", "insufficient"].includes(run.status)) {
      throw new RangeError(`Invalid phase status for ${runId}`);
    }
    if (!Number.isFinite(run.confidence) || run.confidence < 0 || run.confidence > 1) {
      throw new RangeError(`Invalid phase confidence for ${runId}`);
    }
    if (run.manual_confirmed !== undefined && typeof run.manual_confirmed !== "boolean") {
      throw new TypeError(`Invalid manual confirmation for ${runId}`);
    }
    if (run.status === "insufficient") {
      if (!Array.isArray(run.phase_per_frame)) throw new TypeError(`Invalid phase array for ${runId}`);
      continue;
    }
    if (!Number.isFinite(run.period_frames) || run.period_frames <= 0) {
      throw new RangeError(`Invalid phase period for ${runId}`);
    }
    phaseWithAnchorOverride(run, run.anchor_frame);
  }
  return report;
}

/** Build display rows without promoting insufficient runs when manual overrides exist. */
export function phaseStatusRows(report, anchorOverrides = {}) {
  validatePhasePlaybackReport(report);
  return PHASE_RUN_IDS.map((runId) => {
    const run = report.runs[runId];
    const override = anchorOverrideFor(anchorOverrides, runId);
    const manuallyConfirmed = run.status !== "insufficient"
      && (run.manual_confirmed === true || override !== null);
    return {
      run_id: runId,
      period_frames: Number.isFinite(run.period_frames) ? run.period_frames : null,
      confidence: run.confidence,
      status: manuallyConfirmed ? "manual-confirmed" : run.status,
      reason: typeof run.reason === "string" ? run.reason : "",
      anchor_frame: manuallyConfirmed && override !== null ? override : run.anchor_frame,
    };
  });
}

/** Derive one anatomical group's clock and synchronized frames from its own reference timing. */
export function estimatedGroupPlaybackState(report, projectionRuns, groupName, elapsedMs, anchorOverrides = {}) {
  const group = report?.groups?.[groupName];
  const members = Array.isArray(group?.run_ids) ? group.run_ids : [];
  const projectionById = new Map((Array.isArray(projectionRuns) ? projectionRuns : []).map((run) => [run.run_id, run]));
  const sceneCompatible = (runId) => {
    const run = report?.runs?.[runId];
    const projection = projectionById.get(runId);
    return Number.isInteger(projection?.frame_count)
      && projection.frame_count > 0
      && Array.isArray(run?.phase_per_frame)
      && run.phase_per_frame.length === projection.frame_count;
  };
  const usableRunIds = members.filter((runId) => {
    const run = report?.runs?.[runId];
    const override = anchorOverrideFor(anchorOverrides, runId);
    return sceneCompatible(runId)
      && Number.isFinite(run?.period_frames)
      && run.period_frames > 0
      && synchronizedFrame(run, 0, override) !== null;
  });
  const referenceRunId = group?.reference_run_id ?? null;
  const referencePhase = report?.runs?.[referenceRunId];
  const referenceProjection = projectionById.get(referenceRunId);
  const medianFrameTimeMs = referenceProjection?.median_frame_time_ms;
  const emptyFrames = Object.fromEntries(members.map((runId) => [runId, null]));
  let disabledReason = null;
  if (referenceRunId === null) disabledReason = "missing-reference";
  else if (!sceneCompatible(referenceRunId)) disabledReason = "incompatible-reference-projection";
  else if (!usableRunIds.includes(referenceRunId)) disabledReason = "unusable-reference-phase-policy";
  else if (usableRunIds.length < 2) disabledReason = "fewer-than-two-scene-compatible-runs";
  else if (!Number.isFinite(medianFrameTimeMs) || medianFrameTimeMs <= 0) disabledReason = "invalid-reference-clock";
  if (disabledReason !== null) {
    return {
      enabled: false,
      reason: disabledReason,
      usable_run_ids: usableRunIds,
      reference_run_id: referenceRunId,
      period_ms: null,
      global_phase: null,
      frames: emptyFrames,
    };
  }

  const periodMs = referencePhase.period_frames * medianFrameTimeMs;
  if (!Number.isFinite(periodMs) || periodMs <= 0 || !Number.isFinite(elapsedMs)) {
    return {
      enabled: false,
      reason: "invalid-reference-clock",
      usable_run_ids: usableRunIds,
      reference_run_id: referenceRunId,
      period_ms: null,
      global_phase: null,
      frames: emptyFrames,
    };
  }
  const globalPhase = positiveModulo(elapsedMs, periodMs) / periodMs;
  const frames = Object.fromEntries(members.map((runId) => [
    runId,
    synchronizedFrame(report.runs[runId], globalPhase, anchorOverrideFor(anchorOverrides, runId)),
  ]));
  return {
    enabled: true,
    reason: null,
    usable_run_ids: usableRunIds,
    reference_run_id: referenceRunId,
    period_ms: periodMs,
    global_phase: globalPhase,
    frames,
  };
}

/** Estimated-mode availability gates starts, never the stop needed to leave a disabled mode safely. */
export function estimatedPlaybackRequestAllowed(mode, requestedPlaying, estimatedEnabled) {
  return mode !== "estimated" || requestedPlaying !== true || estimatedEnabled === true;
}

/** Create the JSON data for a browser-only override download. */
export function createPhaseOverridePayload(sourcePhaseAsset, anchorOverrides, exportedAt = new Date()) {
  if (typeof sourcePhaseAsset !== "string" || !sourcePhaseAsset.trim()) {
    throw new TypeError("sourcePhaseAsset must be a nonempty path");
  }
  const entries = anchorOverrides instanceof Map
    ? [...anchorOverrides.entries()]
    : Object.entries(anchorOverrides ?? {});
  const runs = Object.fromEntries(entries.sort(([left], [right]) => left.localeCompare(right)).map(([runId, anchorFrame]) => {
    if (typeof runId !== "string" || !runId || !Number.isInteger(anchorFrame) || anchorFrame < 0) {
      throw new TypeError("Each phase override must have a run id and nonnegative integer anchor");
    }
    return [runId, { anchor_frame: anchorFrame, status: "manual-confirmed" }];
  }));
  return {
    schema_version: 1,
    source_phase_asset: sourcePhaseAsset,
    exported_at: new Date(exportedAt).toISOString(),
    runs,
  };
}

/** Return true only when an object and every ancestor are effectively visible. */
export function isEffectivelyVisible(object) {
  if (!object || (typeof object !== "object" && typeof object !== "function")) return false;
  let current = object;
  while (current) {
    if (current.visible === false) return false;
    current = current.parent;
  }
  return true;
}

export class LruFrameCache {
  constructor(maxEntries, disposeValue = () => {}) {
    if (!Number.isInteger(maxEntries) || maxEntries <= 0) {
      throw new RangeError("maxEntries must be a positive integer");
    }
    if (typeof disposeValue !== "function") {
      throw new TypeError("disposeValue must be a function");
    }
    this.maxEntries = maxEntries;
    this.disposeValue = disposeValue;
    this.entries = new Map();
  }

  get size() {
    return this.entries.size;
  }

  has(key) {
    return this.entries.has(key);
  }

  get(key) {
    if (!this.entries.has(key)) return undefined;
    const value = this.entries.get(key);
    this.entries.delete(key);
    this.entries.set(key, value);
    return value;
  }

  set(key, value) {
    if (this.entries.has(key)) {
      const previous = this.entries.get(key);
      this.entries.delete(key);
      if (previous !== value) this.disposeValue(previous);
    }
    this.entries.set(key, value);
    while (this.entries.size > this.maxEntries) {
      const oldestKey = this.entries.keys().next().value;
      const oldestValue = this.entries.get(oldestKey);
      this.entries.delete(oldestKey);
      this.disposeValue(oldestValue);
    }
  }

  clear() {
    const values = [...this.entries.values()];
    this.entries.clear();
    values.forEach((value) => this.disposeValue(value));
  }
}
