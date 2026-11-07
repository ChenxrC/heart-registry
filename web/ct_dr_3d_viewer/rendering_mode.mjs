export const RENDER_MODES = Object.freeze({
  DIAGNOSTIC: "diagnostic",
  REALISTIC: "realistic",
});

const IMAGE_KINDS = new Set([
  "ct-image",
  "dr-image",
  "xa-image",
  "annotation",
  "helper",
]);


export function normalizeRenderMode(value) {
  return value === RENDER_MODES.REALISTIC
    ? RENDER_MODES.REALISTIC
    : RENDER_MODES.DIAGNOSTIC;
}


export function rendererProfile(mode, lowPower = false) {
  const selected = normalizeRenderMode(mode);
  return Object.freeze({
    mode: selected,
    toneMapping: selected === RENDER_MODES.REALISTIC ? "aces-filmic" : "none",
    exposure: 1.0,
    maxPixelRatio: lowPower ? 1.5 : 2.0,
    gtao: selected === RENDER_MODES.REALISTIC && !lowPower,
  });
}


export function materialProfile(kind, mode, opacity) {
  if (!Number.isFinite(opacity) || opacity < 0 || opacity > 1) {
    throw new RangeError("Material opacity must be a finite number in [0, 1]");
  }

  if (normalizeRenderMode(mode) !== RENDER_MODES.REALISTIC) {
    return Object.freeze({ type: "standard", opacity });
  }

  const realisticProfiles = {
    heart: {
      type: "physical",
      color: 0x7a2633,
      roughness: 0.62,
      metalness: 0,
      clearcoat: 0.08,
      sheen: 0.22,
    },
    "vessel-primary": {
      type: "physical",
      color: 0xb8323e,
      roughness: 0.34,
      metalness: 0,
      clearcoat: 0.16,
      sheen: 0.08,
    },
    "vessel-comparison": {
      type: "physical",
      roughness: 0.40,
      metalness: 0,
      clearcoat: 0.10,
      sheen: 0.05,
    },
    bone: {
      type: "standard",
      color: 0xd8cdb3,
      roughness: 0.72,
      metalness: 0,
    },
  };

  return Object.freeze({
    ...(realisticProfiles[kind] ?? { type: "standard" }),
    opacity,
  });
}


export function shouldBypassToneMapping(kind) {
  return IMAGE_KINDS.has(kind);
}
