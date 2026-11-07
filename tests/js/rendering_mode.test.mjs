import assert from "node:assert/strict";
import test from "node:test";

import {
  RENDER_MODES,
  materialProfile,
  normalizeRenderMode,
  rendererProfile,
  shouldBypassToneMapping,
} from "../../web/ct_dr_3d_viewer/rendering_mode.mjs";


test("render modes expose only diagnostic and realistic values", () => {
  assert.deepEqual(RENDER_MODES, {
    DIAGNOSTIC: "diagnostic",
    REALISTIC: "realistic",
  });
  assert.equal(Object.isFrozen(RENDER_MODES), true);
});

test("render mode normalization fails closed to diagnostic", () => {
  assert.equal(normalizeRenderMode("diagnostic"), RENDER_MODES.DIAGNOSTIC);
  assert.equal(normalizeRenderMode("realistic"), RENDER_MODES.REALISTIC);
  for (const value of [undefined, null, "", "REALISTIC", "unknown", 1]) {
    assert.equal(normalizeRenderMode(value), RENDER_MODES.DIAGNOSTIC);
  }
});

test("renderer profiles select tone mapping, pixel ratio, and GTAO safely", () => {
  assert.deepEqual(rendererProfile("diagnostic"), {
    mode: "diagnostic",
    toneMapping: "none",
    exposure: 1.0,
    maxPixelRatio: 2.0,
    gtao: false,
  });
  assert.deepEqual(rendererProfile("realistic"), {
    mode: "realistic",
    toneMapping: "aces-filmic",
    exposure: 1.0,
    maxPixelRatio: 2.0,
    gtao: true,
  });
  assert.deepEqual(rendererProfile("realistic", true), {
    mode: "realistic",
    toneMapping: "aces-filmic",
    exposure: 1.0,
    maxPixelRatio: 1.5,
    gtao: false,
  });
  assert.equal(Object.isFrozen(rendererProfile("realistic")), true);
});

test("realistic heart material uses the approved physical profile", () => {
  assert.deepEqual(materialProfile("heart", "realistic", 0.72), {
    type: "physical",
    color: 0x7a2633,
    roughness: 0.62,
    metalness: 0,
    clearcoat: 0.08,
    sheen: 0.22,
    opacity: 0.72,
  });
});

test("realistic primary vessel uses the approved physical profile", () => {
  assert.deepEqual(materialProfile("vessel-primary", "realistic", 0.8), {
    type: "physical",
    color: 0xb8323e,
    roughness: 0.34,
    metalness: 0,
    clearcoat: 0.16,
    sheen: 0.08,
    opacity: 0.8,
  });
});

test("realistic comparison vessel preserves the caller color", () => {
  const profile = materialProfile("vessel-comparison", "realistic", 0.4);
  assert.deepEqual(profile, {
    type: "physical",
    roughness: 0.40,
    metalness: 0,
    clearcoat: 0.10,
    sheen: 0.05,
    opacity: 0.4,
  });
  assert.equal(Object.hasOwn(profile, "color"), false);
});

test("realistic bone uses the approved standard profile", () => {
  assert.deepEqual(materialProfile("bone", "realistic", 1), {
    type: "standard",
    color: 0xd8cdb3,
    roughness: 0.72,
    metalness: 0,
    opacity: 1,
  });
});

test("diagnostic and unknown mesh kinds receive a safe standard profile", () => {
  assert.deepEqual(materialProfile("heart", "diagnostic", 0.5), {
    type: "standard",
    opacity: 0.5,
  });
  assert.deepEqual(materialProfile("unexpected-mesh", "realistic", 0.25), {
    type: "standard",
    opacity: 0.25,
  });
});

test("material opacity preserves boundaries and rejects out-of-range values", () => {
  assert.equal(materialProfile("heart", "realistic", 0).opacity, 0);
  assert.equal(materialProfile("heart", "realistic", 1).opacity, 1);
  for (const opacity of [-0.001, 1.001]) {
    assert.throws(
      () => materialProfile("heart", "realistic", opacity),
      RangeError,
    );
  }
});

test("every returned material profile is immutable", () => {
  for (const [kind, mode] of [
    ["heart", "realistic"],
    ["vessel-primary", "realistic"],
    ["vessel-comparison", "realistic"],
    ["bone", "realistic"],
    ["heart", "diagnostic"],
    ["unexpected-mesh", "realistic"],
  ]) {
    assert.equal(Object.isFrozen(materialProfile(kind, mode, 0.5)), true);
  }
});

test("only medical images, annotations, and helpers bypass display effects", () => {
  for (const kind of ["ct-image", "dr-image", "xa-image", "annotation", "helper"]) {
    assert.equal(shouldBypassToneMapping(kind), true);
  }
  for (const kind of ["heart", "vessel-primary", "vessel-comparison", "bone", "ct_image", undefined]) {
    assert.equal(shouldBypassToneMapping(kind), false);
  }
});
