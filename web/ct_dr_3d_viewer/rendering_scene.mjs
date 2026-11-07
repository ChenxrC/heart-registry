import {
  materialProfile,
  normalizeRenderMode,
  RENDER_MODES,
  rendererProfile,
  shouldBypassToneMapping,
} from "./rendering_mode.mjs";

const DIAGNOSTIC_BACKGROUND = 0x03090c;
const ANATOMY_KINDS = new Set(["heart", "vessel-primary", "vessel-comparison", "bone"]);
const SMOOTH_SURFACE_KINDS = new Set(["heart", "bone"]);
const AO_BYPASS_KINDS = new Set(["ct-image", "dr-image", "xa-image", "annotation", "helper"]);
const GTAO_WARMUP_FRAMES = 30;
const GTAO_SAMPLE_FRAMES = 120;
const GTAO_FRAME_BUDGET_MS = 33.3;

export function prepareAnatomyMeshGeometry(mesh, kind) {
  if (!SMOOTH_SURFACE_KINDS.has(kind) || !mesh?.geometry?.getAttribute?.("position")) {
    return false;
  }
  mesh.geometry.deleteAttribute?.("normal");
  mesh.geometry.computeVertexNormals?.();
  mesh.geometry.normalizeNormals?.();
  (Array.isArray(mesh.material) ? mesh.material : [mesh.material]).forEach((material) => {
    if (!material) return;
    material.flatShading = false;
    material.needsUpdate = true;
  });
  return true;
}

function setLightPosition(light, x, y, z) {
  light.position.x = x;
  light.position.y = y;
  light.position.z = z;
}

function disposeMaterial(material) {
  if (Array.isArray(material)) {
    material.forEach((item) => item.dispose?.());
    return;
  }
  material?.dispose?.();
}

function disposeEnvironmentSource(source) {
  for (const object of [...source.children]) {
    source.remove(object);
    object.geometry?.dispose?.();
    disposeMaterial(object.material);
    object.dispose?.();
  }
}

function createProgrammaticEnvironment(THREE) {
  const source = new THREE.Scene();
  source.background = new THREE.Color(0x101820);

  const enclosure = new THREE.Mesh(
    new THREE.BoxGeometry(12, 12, 12),
    new THREE.MeshBasicMaterial({ color: 0x18232d, side: THREE.BackSide }),
  );
  source.add(enclosure);

  const warm = new THREE.DirectionalLight(0xffd2ad, 4.0);
  setLightPosition(warm, -4, -5, 6);
  const cool = new THREE.DirectionalLight(0xa9cfff, 2.0);
  setLightPosition(cool, 5, 2, 3);
  const ambient = new THREE.AmbientLight(0xcad8e8, 0.8);
  source.add(warm, cool, ambient);
  return source;
}

export async function loadGtaoAddons(
  importModule = (specifier) => import(specifier),
  warn = (message) => console.warn(message),
) {
  const specifiers = [
    "three/addons/postprocessing/EffectComposer.js",
    "three/addons/postprocessing/RenderPass.js",
    "three/addons/postprocessing/GTAOPass.js",
    "three/addons/postprocessing/OutputPass.js",
  ];
  try {
    const [composerModule, renderModule, gtaoModule, outputModule] = await Promise.all(
      specifiers.map((specifier) => importModule(specifier)),
    );
    const loaded = {
      EffectComposer: composerModule.EffectComposer,
      RenderPass: renderModule.RenderPass,
      GTAOPass: gtaoModule.GTAOPass,
      OutputPass: outputModule.OutputPass,
    };
    if (!Object.values(loaded).every((value) => typeof value === "function")) {
      throw new TypeError("one or more addon exports are missing");
    }
    return loaded;
  } catch (error) {
    warn(`[viewer] GTAO addons unavailable; using direct rendering: ${error?.message ?? error}`);
    return null;
  }
}

export function createRenderingController({
  THREE,
  renderer,
  scene,
  camera = null,
  requestRender,
  lowPower = false,
  addons = null,
  warn = (message) => console.warn(message),
}) {
  const registrations = new Map();
  const opacities = new Map();
  const ownedMaterials = new Set();
  const realisticMaterials = new Map();
  let modeLights = [];
  let environmentTarget = null;
  let currentMode = RENDER_MODES.DIAGNOSTIC;
  let disposed = false;
  let composer = null;
  let renderPass = null;
  let gtaoPass = null;
  let outputPass = null;
  let gtaoState = "idle";
  let gtaoReason = null;
  let gtaoWarningEmitted = false;
  let renderWidth = 1;
  let renderHeight = 1;
  let renderPixelRatio = 1;
  let performanceWarmup = 0;
  let performanceSamples = [];
  let completedPerformanceWindows = 0;
  let lastP95FrameMs = null;
  let consecutiveSlowWindows = 0;

  function safeDispose(resource) {
    try {
      resource?.dispose?.();
    } catch (_error) {
      // Disposal is best effort; the direct renderer remains usable.
    }
  }

  function releaseGtao() {
    safeDispose(gtaoPass);
    safeDispose(renderPass);
    safeDispose(outputPass);
    safeDispose(composer);
    gtaoPass = null;
    renderPass = null;
    outputPass = null;
    composer = null;
  }

  function getGtaoStatus() {
    return Object.freeze({
      state: gtaoState,
      reason: gtaoReason,
      warmupFrameCount: performanceWarmup,
      sampleFrameCount: performanceSamples.length,
      completedWindowCount: completedPerformanceWindows,
      lastP95FrameMs,
    });
  }

  function disableGtao(reason) {
    if (gtaoState === "disabled") return getGtaoStatus();
    releaseGtao();
    gtaoState = "disabled";
    gtaoReason = String(reason || "unknown");
    if (!gtaoWarningEmitted) {
      gtaoWarningEmitted = true;
      warn(`[viewer] GTAO disabled for this session: ${gtaoReason}`);
    }
    return getGtaoStatus();
  }

  function hasAoBypassRegistration() {
    for (const record of registrations.values()) {
      if (AO_BYPASS_KINDS.has(record.kind)) return true;
    }
    return false;
  }

  function initializeGtao() {
    if (gtaoState !== "idle" || currentMode !== RENDER_MODES.REALISTIC) return;
    if (!rendererProfile(currentMode, lowPower).gtao) {
      disableGtao("low-power-device");
      return;
    }
    if (!renderer.capabilities?.isWebGL2) {
      disableGtao("webgl2-unavailable");
      return;
    }
    if (hasAoBypassRegistration()) {
      // Postprocessing has no compatible depth hand-off for the existing
      // medical planes. Fail closed instead of changing their occlusion.
      disableGtao("medical-overlay-present");
      return;
    }
    const { EffectComposer, RenderPass, GTAOPass, OutputPass } = addons ?? {};
    if (![EffectComposer, RenderPass, GTAOPass, OutputPass].every((value) => typeof value === "function") || !camera) {
      disableGtao("addon-or-camera-unavailable");
      return;
    }
    try {
      composer = new EffectComposer(renderer);
      renderPass = new RenderPass(scene, camera);
      gtaoPass = new GTAOPass(scene, camera, renderWidth, renderHeight);
      outputPass = new OutputPass();
      gtaoPass.blendIntensity = 0.3;
      gtaoPass.updateGtaoMaterial?.({ radius: 0.2, distanceExponent: 1, thickness: 1 });
      gtaoPass.updatePdMaterial?.({ radius: 4, radiusExponent: 2, rings: 2, samples: 8 });
      composer.addPass(renderPass);
      composer.addPass(gtaoPass);
      composer.addPass(outputPass);
      composer.setPixelRatio?.(renderPixelRatio);
      composer.setSize(renderWidth, renderHeight);
      gtaoState = "enabled";
    } catch (_error) {
      disableGtao("allocation-failed");
    }
  }

  function eachMaterial(material, callback) {
    (Array.isArray(material) ? material : [material]).forEach((item) => {
      if (item) callback(item);
    });
  }

  function materialOpacity(material) {
    const first = Array.isArray(material) ? material[0] : material;
    return Number.isFinite(first?.opacity) ? first.opacity : 1;
  }

  function setMaterialOpacity(material, opacity) {
    eachMaterial(material, (item) => {
      item.opacity = opacity;
      if (item.uniforms?.opacity) item.uniforms.opacity.value = opacity;
    });
  }

  function diagnosticColor(material) {
    const first = Array.isArray(material) ? material[0] : material;
    if (!first?.color) return null;
    return typeof first.color.getHex === "function" ? first.color.getHex() : first.color;
  }

  function realisticMaterial(kind, opacity, diagnosticMaterial) {
    const profile = materialProfile(kind, RENDER_MODES.REALISTIC, opacity);
    const comparisonColor = kind === "vessel-comparison" ? diagnosticColor(diagnosticMaterial) : null;
    const diagnosticSide = (Array.isArray(diagnosticMaterial) ? diagnosticMaterial[0] : diagnosticMaterial)?.side;
    const options = {
      ...profile,
      ...(comparisonColor === null ? {} : { color: comparisonColor }),
      ...(diagnosticSide === undefined ? {} : { side: diagnosticSide }),
      transparent: opacity < 1,
      depthWrite: kind === "heart" ? false : opacity >= 1,
    };
    delete options.type;
    const key = JSON.stringify([kind, options]);
    if (realisticMaterials.has(key)) return realisticMaterials.get(key);
    const Material = profile.type === "physical" ? THREE.MeshPhysicalMaterial : THREE.MeshStandardMaterial;
    const material = new Material(options);
    realisticMaterials.set(key, material);
    ownedMaterials.add(material);
    return material;
  }

  function createHeartDepthHelper(mesh) {
    if (typeof mesh.add !== "function" || !mesh.geometry || !THREE.MeshDepthMaterial) return null;
    const material = new THREE.MeshDepthMaterial({
      depthPacking: THREE.RGBADepthPacking,
      colorWrite: false,
      depthWrite: true,
      depthTest: true,
      side: THREE.DoubleSide,
    });
    ownedMaterials.add(material);
    const helper = new THREE.Mesh(mesh.geometry, material);
    helper.name = "realistic-heart-depth-prepass";
    helper.renderOrder = (mesh.renderOrder ?? 0) - 1;
    helper.visible = currentMode === RENDER_MODES.REALISTIC;
    helper.raycast = () => {};
    mesh.add(helper);
    return helper;
  }

  function applyRegistration(record) {
    if (ANATOMY_KINDS.has(record.kind)) {
      record.mesh.material = currentMode === RENDER_MODES.REALISTIC
        ? record.realisticMaterial
        : record.diagnosticMaterial;
    }
    if (record.depthHelper) record.depthHelper.visible = currentMode === RENDER_MODES.REALISTIC;
  }

  function isWithinTarget(mesh, target) {
    if (!target) return true;
    for (let object = mesh; object; object = object.parent) {
      if (object === target) return true;
    }
    return false;
  }

  function removeModeLights() {
    for (const light of modeLights) {
      scene.remove(light);
      light.dispose?.();
    }
    modeLights = [];
  }

  function releaseEnvironment() {
    if (environmentTarget && scene.environment === environmentTarget.texture) {
      scene.environment = null;
    }
    environmentTarget?.dispose?.();
    environmentTarget = null;
  }

  function addDiagnosticLights() {
    const hemisphere = new THREE.HemisphereLight(0xd9f9ff, 0x102128, 1.25);
    const key = new THREE.DirectionalLight(0xffffff, 2.2);
    setLightPosition(key, -300, -500, 500);
    modeLights = [hemisphere, key];
    scene.add(...modeLights);
  }

  function addRealisticLights() {
    const hemisphere = new THREE.HemisphereLight(0xdcecff, 0x24170f, 0.7);
    const ambient = new THREE.AmbientLight(0xffead8, 0.25);
    const key = new THREE.DirectionalLight(0xffd1ad, 3.4);
    const fill = new THREE.DirectionalLight(0x9fc9ff, 1.15);
    const rim = new THREE.DirectionalLight(0xffbf8a, 1.8);
    setLightPosition(key, -320, -420, 520);
    setLightPosition(fill, 380, -160, 260);
    setLightPosition(rim, 180, 420, 360);
    for (const light of [key, fill, rim]) light.castShadow = false;
    modeLights = [hemisphere, ambient, key, fill, rim];
    scene.add(...modeLights);
  }

  function allocateEnvironment() {
    const source = createProgrammaticEnvironment(THREE);
    let generator = null;
    try {
      generator = new THREE.PMREMGenerator(renderer);
      generator.compileCubemapShader?.();
      environmentTarget = generator.fromScene(source);
      scene.environment = environmentTarget.texture;
    } catch (_error) {
      environmentTarget?.dispose?.();
      environmentTarget = null;
      scene.environment = null;
    } finally {
      generator?.dispose?.();
      disposeEnvironmentSource(source);
    }
  }

  function setMode(nextMode) {
    if (disposed) return;
    const normalizedMode = normalizeRenderMode(nextMode);
    currentMode = normalizedMode;
    removeModeLights();
    releaseEnvironment();

    renderer.outputColorSpace = THREE.SRGBColorSpace;
    renderer.toneMappingExposure = 1.0;
    renderer.shadowMap.enabled = false;
    renderer.setClearColor(DIAGNOSTIC_BACKGROUND, 1);

    if (normalizedMode === RENDER_MODES.REALISTIC) {
      renderer.toneMapping = THREE.ACESFilmicToneMapping;
      scene.fog = null;
      addRealisticLights();
      allocateEnvironment();
      registrations.forEach(applyRegistration);
      initializeGtao();
      return;
    }

    renderer.toneMapping = THREE.NoToneMapping;
    scene.fog = new THREE.FogExp2(DIAGNOSTIC_BACKGROUND, 0.0007);
    addDiagnosticLights();
    registrations.forEach(applyRegistration);
  }

  function registerMesh(mesh, kind) {
    if (disposed || !mesh) return;
    if (registrations.has(mesh)) unregisterMesh(mesh);
    prepareAnatomyMeshGeometry(mesh, kind);
    const diagnosticMaterial = mesh.material;
    if (shouldBypassToneMapping(kind)) {
      eachMaterial(diagnosticMaterial, (material) => {
        material.toneMapped = false;
        material.fog = false;
      });
    }
    const opacity = opacities.get(kind) ?? materialOpacity(diagnosticMaterial);
    if (opacities.has(kind)) setMaterialOpacity(diagnosticMaterial, opacity);
    const record = {
      mesh,
      kind,
      diagnosticMaterial,
      realisticMaterial: ANATOMY_KINDS.has(kind) && diagnosticMaterial
        ? realisticMaterial(kind, opacity, diagnosticMaterial)
        : null,
      depthHelper: kind === "heart" ? createHeartDepthHelper(mesh) : null,
    };
    registrations.set(mesh, record);
    applyRegistration(record);
    if (currentMode === RENDER_MODES.REALISTIC && AO_BYPASS_KINDS.has(kind)) {
      disableGtao("medical-overlay-present");
    }
  }

  function unregisterMesh(mesh) {
    const record = registrations.get(mesh);
    if (!record) return;
    mesh.material = record.diagnosticMaterial;
    if (record.depthHelper) {
      mesh.remove?.(record.depthHelper);
      record.depthHelper.material.dispose?.();
      ownedMaterials.delete(record.depthHelper.material);
    }
    registrations.delete(mesh);
  }

  function setOpacity(kind, opacity, target = null) {
    if (!Number.isFinite(opacity) || opacity < 0 || opacity > 1) {
      throw new RangeError("Opacity must be a finite number in [0, 1]");
    }
    const affectedKinds = kind === "vessel-primary"
      ? new Set(["vessel-primary", "vessel-comparison"])
      : new Set([kind]);
    if (!target) affectedKinds.forEach((affectedKind) => opacities.set(affectedKind, opacity));
    registrations.forEach((record) => {
      if (!affectedKinds.has(record.kind) || !isWithinTarget(record.mesh, target)) return;
      setMaterialOpacity(record.diagnosticMaterial, opacity);
      if (record.realisticMaterial) {
        record.realisticMaterial = realisticMaterial(record.kind, opacity, record.diagnosticMaterial);
        if (currentMode === RENDER_MODES.REALISTIC) record.mesh.material = record.realisticMaterial;
      }
    });
    requestRender();
  }

  function resize(width, height, pixelRatio) {
    renderWidth = Math.max(1, width);
    renderHeight = Math.max(1, height);
    renderPixelRatio = pixelRatio;
    renderer.setPixelRatio(pixelRatio);
    renderer.setSize(width, height, false);
    if (gtaoState === "enabled") {
      try {
        composer.setPixelRatio?.(pixelRatio);
        composer.setSize(renderWidth, renderHeight);
      } catch (_error) {
        disableGtao("resize-failed");
      }
    }
  }

  function render(activeCamera = camera) {
    if (disposed) return;
    if (currentMode === RENDER_MODES.REALISTIC && gtaoState === "idle") initializeGtao();
    if (gtaoState === "enabled" && currentMode === RENDER_MODES.REALISTIC) {
      if (activeCamera && activeCamera !== camera) {
        renderPass.camera = activeCamera;
        gtaoPass.camera = activeCamera;
      }
      try {
        composer.render();
        return;
      } catch (_error) {
        disableGtao("render-failed");
      }
    }
    renderer.render(scene, activeCamera);
  }

  function reportFrameTiming(durationMs, cinePlaying) {
    if (!cinePlaying || !Number.isFinite(durationMs) || durationMs < 0) return;
    if (performanceWarmup < GTAO_WARMUP_FRAMES) {
      performanceWarmup += 1;
      return;
    }
    performanceSamples.push(durationMs);
    if (performanceSamples.length < GTAO_SAMPLE_FRAMES) return;
    const ordered = [...performanceSamples].sort((left, right) => left - right);
    const p95Index = Math.ceil(0.95 * ordered.length) - 1;
    lastP95FrameMs = ordered[p95Index];
    completedPerformanceWindows += 1;
    performanceSamples = [];
    if (gtaoState === "enabled") {
      consecutiveSlowWindows = lastP95FrameMs > GTAO_FRAME_BUDGET_MS ? consecutiveSlowWindows + 1 : 0;
      if (consecutiveSlowWindows >= 2) disableGtao("performance-p95");
    }
  }

  function dispose() {
    if (disposed) return;
    disposed = true;
    releaseGtao();
    removeModeLights();
    releaseEnvironment();
    [...registrations.keys()].forEach(unregisterMesh);
    ownedMaterials.forEach((material) => material.dispose?.());
    ownedMaterials.clear();
    realisticMaterials.clear();
    registrations.clear();
    opacities.clear();
  }

  return {
    setMode,
    registerMesh,
    unregisterMesh,
    setOpacity,
    resize,
    render,
    reportFrameTiming,
    getGtaoStatus,
    disableGtao,
    dispose,
  };
}
