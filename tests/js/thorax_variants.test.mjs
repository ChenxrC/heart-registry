import assert from "node:assert/strict";
import test from "node:test";

import {
  resolveThoraxAssetRecords,
  resolveThoraxVariant,
  setThoraxVariantVisibility,
} from "../../web/ct_dr_3d_viewer/thorax_variants.mjs";


test("automatic and manual thorax variant selection is deterministic", () => {
  assert.equal(resolveThoraxVariant("diagnostic", "auto"), "registration");
  assert.equal(resolveThoraxVariant("realistic", "auto"), "display");
  assert.equal(resolveThoraxVariant("realistic", "registration"), "registration");
  assert.equal(resolveThoraxVariant("diagnostic", "display"), "display");
  assert.throws(
    () => resolveThoraxVariant("realistic", "shifted"),
    /Unknown thorax variant/,
  );
});


test("nested and legacy thorax asset records resolve without changing paths", () => {
  const nested = {
    registration: { path: "assets/registration.glb", sha256: "a" },
    display: { path: "assets/display.glb", sha256: "b" },
    path: "assets/registration.glb",
  };
  assert.deepEqual(resolveThoraxAssetRecords(nested), {
    registration: nested.registration,
    display: nested.display,
    hasDistinctDisplay: true,
  });

  const legacy = { path: "assets/thorax.glb", byte_length: 123 };
  assert.deepEqual(resolveThoraxAssetRecords(legacy), {
    registration: legacy,
    display: legacy,
    hasDistinctDisplay: false,
  });
  assert.throws(() => resolveThoraxAssetRecords({}), /path/);
});


test("variant switching mutates visibility only", () => {
  function makeGroup(seed) {
    return {
      visible: true,
      position: { x: seed, y: seed + 1, z: seed + 2 },
      rotation: { x: seed + 3, y: seed + 4, z: seed + 5 },
      scale: { x: 1, y: 1, z: 1 },
      matrix: { elements: Array.from({ length: 16 }, (_value, index) => seed + index) },
      geometry: {
        attributes: {
          position: { array: new Float32Array([seed, seed + 0.5, seed + 1]) },
        },
      },
    };
  }

  const groups = {
    registration: { heart: makeGroup(10), ribs: makeGroup(20) },
    display: { heart: makeGroup(30), ribs: makeGroup(40) },
  };
  const snapshot = () => Object.fromEntries(
    Object.entries(groups).map(([variant, structures]) => [
      variant,
      Object.fromEntries(Object.entries(structures).map(([name, group]) => [name, {
        position: structuredClone(group.position),
        rotation: structuredClone(group.rotation),
        scale: structuredClone(group.scale),
        matrix: [...group.matrix.elements],
        vertices: [...group.geometry.attributes.position.array],
      }])),
    ]),
  );

  const before = snapshot();
  assert.equal(setThoraxVariantVisibility(groups, "display"), "display");
  assert.equal(groups.registration.heart.visible, false);
  assert.equal(groups.registration.ribs.visible, false);
  assert.equal(groups.display.heart.visible, true);
  assert.equal(groups.display.ribs.visible, true);
  assert.deepEqual(snapshot(), before);

  assert.equal(setThoraxVariantVisibility(groups, "registration"), "registration");
  assert.equal(groups.registration.heart.visible, true);
  assert.equal(groups.display.heart.visible, false);
  assert.deepEqual(snapshot(), before);
  assert.throws(() => setThoraxVariantVisibility(groups, "auto"), /concrete/);
});
