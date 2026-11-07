from __future__ import annotations

import re
import shutil
import subprocess
import tempfile
from html.parser import HTMLParser
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
VIEWER = ROOT / "web/ct_dr_3d_viewer"
PROTECTED_GEOMETRY_SEGMENTS = {"calibration", "detector", "metrics"}
_JS_MEMBER_EXPRESSION = (
    r"[A-Za-z_$][\w$]*"
    r"(?:(?:\.[A-Za-z_$][\w$]*)|(?:\[(?:\"[^\"]+\"|'[^']+'|\d+)\]))*"
)
_ARRAY_MUTATORS = "push|pop|shift|unshift|splice|sort|reverse|fill|copyWithin"


def _member_segments(expression: str) -> list[str]:
    return re.findall(r"[A-Za-z_$][\w$]*", expression)


def _is_protected_geometry_path(segments: list[str]) -> bool:
    for segment in segments[1:]:
        lowered = segment.lower()
        if lowered in PROTECTED_GEOMETRY_SEGMENTS or lowered.endswith("_metrics"):
            return True
        if "lps" in lowered or "frame_time" in lowered or "frame_timing" in lowered:
            return True
    return False


def _canonical_geometry_path(
    expression: str,
    aliases: dict[str, list[str]],
) -> list[str] | None:
    segments = _member_segments(expression)
    if not segments or segments[0] not in aliases:
        return None
    return [*aliases[segments[0]], *segments[1:]]


def _geometry_aliases_before(source: str, end: int) -> dict[str, list[str]]:
    aliases = {"geometry": ["geometry"]}
    for match in re.finditer(
        r"\b(?:const|let|var)\s+([A-Za-z_$][\w$]*)\s*=\s*([^;]+);",
        source[:end],
    ):
        alias, expression = match.group(1), match.group(2).strip()
        if re.fullmatch(_JS_MEMBER_EXPRESSION, expression):
            canonical = _canonical_geometry_path(expression, aliases)
        else:
            canonical = None
        if canonical is None:
            aliases.pop(alias, None)
        else:
            aliases[alias] = canonical
    return aliases


def _balanced_brace_content(source: str, opening_brace: int) -> str | None:
    depth = 0
    quote: str | None = None
    escaped = False
    for index in range(opening_brace, len(source)):
        character = source[index]
        if quote is not None:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == quote:
                quote = None
            continue
        if character in {'"', "'", "`"}:
            quote = character
        elif character == "{":
            depth += 1
        elif character == "}":
            depth -= 1
            if depth == 0:
                return source[opening_brace + 1:index]
    return None


def _top_level_object_properties(body: str) -> list[str]:
    properties: list[str] = []
    start = 0
    depths = {"{": 0, "[": 0, "(": 0}
    closing_to_opening = {"}": "{", "]": "[", ")": "("}
    quote: str | None = None
    escaped = False
    for index, character in enumerate(body):
        if quote is not None:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == quote:
                quote = None
            continue
        if character in {'"', "'", "`"}:
            quote = character
        elif character in depths:
            depths[character] += 1
        elif character in closing_to_opening:
            opening = closing_to_opening[character]
            depths[opening] = max(0, depths[opening] - 1)
        elif character == "," and not any(depths.values()):
            properties.append(body[start:index])
            start = index + 1
    properties.append(body[start:])
    return properties


def _object_property_name(fragment: str) -> str | None:
    candidate = fragment.strip()
    if not candidate or candidate.startswith("..."):
        return None
    match = re.match(
        r'(?:"([A-Za-z_$][\w$]*)"|\'([A-Za-z_$][\w$]*)\'|([A-Za-z_$][\w$]*))'
        r"\s*(?::|\(|$)",
        candidate,
    )
    if match is None:
        return None
    return next(group for group in match.groups() if group is not None)


def _find_protected_medical_writes(source: str) -> list[str]:
    violations: list[str] = []
    for match in re.finditer(
        rf"\b({_JS_MEMBER_EXPRESSION})\s*"
        r"(?:>>>=|<<=|>>=|\*\*=|&&=|\|\|=|\?\?=|[+\-*/%&|^]=|=(?!=)|\+\+|--)",
        source,
    ):
        declaration_prefix = source[max(0, match.start() - 12):match.start()]
        if re.search(r"\b(?:const|let|var)\s+$", declaration_prefix):
            continue
        aliases = _geometry_aliases_before(source, match.start())
        canonical = _canonical_geometry_path(match.group(1), aliases)
        if canonical is not None and _is_protected_geometry_path(canonical):
            violations.append(match.group(0))

    for match in re.finditer(
        rf"\b({_JS_MEMBER_EXPRESSION})\.({_ARRAY_MUTATORS})\s*\(",
        source,
    ):
        aliases = _geometry_aliases_before(source, match.start())
        canonical = _canonical_geometry_path(match.group(1), aliases)
        if canonical is not None and _is_protected_geometry_path(canonical):
            violations.append(match.group(0))

    for match in re.finditer(
        rf"\bObject\.(?:assign|defineProperty)\s*\(\s*({_JS_MEMBER_EXPRESSION})",
        source,
    ):
        aliases = _geometry_aliases_before(source, match.start())
        canonical = _canonical_geometry_path(match.group(1), aliases)
        if canonical is not None and _is_protected_geometry_path(canonical):
            violations.append(match.group(0))

    for match in re.finditer(
        rf"\bObject\.defineProperty\s*\(\s*({_JS_MEMBER_EXPRESSION})\s*,\s*"
        r"[\"']([A-Za-z_$][\w$]*)[\"']",
        source,
    ):
        aliases = _geometry_aliases_before(source, match.start())
        canonical = _canonical_geometry_path(match.group(1), aliases)
        if canonical is not None and _is_protected_geometry_path([*canonical, match.group(2)]):
            violations.append(match.group(0))

    for match in re.finditer(
        rf"\bObject\.assign\s*\(\s*({_JS_MEMBER_EXPRESSION})\s*,\s*\{{",
        source,
    ):
        aliases = _geometry_aliases_before(source, match.start())
        canonical = _canonical_geometry_path(match.group(1), aliases)
        body = _balanced_brace_content(source, match.end() - 1)
        if canonical is None or body is None:
            continue
        for fragment in _top_level_object_properties(body):
            property_name = _object_property_name(fragment)
            if property_name is not None and _is_protected_geometry_path(
                [*canonical, property_name]
            ):
                violations.append(f"{match.group(0)}...{property_name}")
    return violations


class _RenderModeMarkupParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.select_values: list[str] = []
        self.option_labels: dict[str, str] = {}
        self.note_text = ""
        self._inside_select = False
        self._option_value: str | None = None
        self._inside_note = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attributes = dict(attrs)
        if tag == "select" and attributes.get("id") == "render-mode":
            self.select_values.append(attributes["id"])
            self._inside_select = True
        if self._inside_select and tag == "option":
            self._option_value = attributes.get("value")
        if attributes.get("id") == "render-mode-note":
            self._inside_note = True

    def handle_endtag(self, tag: str) -> None:
        if tag == "option":
            self._option_value = None
        if tag == "select" and self._inside_select:
            self._inside_select = False
        if tag == "p" and self._inside_note:
            self._inside_note = False

    def handle_data(self, data: str) -> None:
        text = data.strip()
        if self._option_value and text:
            self.option_labels[self._option_value] = text
        if self._inside_note and text:
            self.note_text += text


def test_ct_dr_builder_packages_only_explicit_web_sources_byte_identically() -> None:
    from build_ct_dr_3d_viewer import copy_web_sources

    payloads = {
        "index.html": b"<main>synthetic</main>\n",
        "app.js": b"export const app = true;\n",
        "styles.css": b"main { color: #abc; }\n",
        "rendering_mode.mjs": b"export const mode = '\xe7\x9c\x9f\xe5\xae\x9e';\n",
        "rendering_scene.mjs": b"export const scene = new Uint8Array([0, 255]);\n",
        "thorax_variants.mjs": b"export const variant = 'display';\n",
    }
    with tempfile.TemporaryDirectory(dir=ROOT) as temporary_dir:
        root = Path(temporary_dir)
        web_source = root / "web_source"
        output = root / "viewer"
        web_source.mkdir()
        output.mkdir()
        for filename, payload in payloads.items():
            (web_source / filename).write_bytes(payload)
        (web_source / "unrelated-addon.mjs").write_bytes(b"must not be packaged")

        copy_web_sources(web_source, output)

        for filename, payload in payloads.items():
            assert (output / filename).read_bytes() == payload
        assert not (output / "unrelated-addon.mjs").exists()


def test_ct_dr_builder_packages_exact_gtao_dependency_closure() -> None:
    from build_ct_dr_3d_viewer import copy_three_vendor

    required = {
        "build/three.module.min.js": "vendor/three.module.min.js",
        "build/three.core.min.js": "vendor/three.core.min.js",
        "examples/jsm/controls/OrbitControls.js": "vendor/addons/controls/OrbitControls.js",
        "examples/jsm/loaders/GLTFLoader.js": "vendor/addons/loaders/GLTFLoader.js",
        "examples/jsm/utils/BufferGeometryUtils.js": "vendor/addons/utils/BufferGeometryUtils.js",
        "examples/jsm/utils/SkeletonUtils.js": "vendor/addons/utils/SkeletonUtils.js",
        "examples/jsm/postprocessing/EffectComposer.js": "vendor/addons/postprocessing/EffectComposer.js",
        "examples/jsm/postprocessing/RenderPass.js": "vendor/addons/postprocessing/RenderPass.js",
        "examples/jsm/postprocessing/GTAOPass.js": "vendor/addons/postprocessing/GTAOPass.js",
        "examples/jsm/postprocessing/OutputPass.js": "vendor/addons/postprocessing/OutputPass.js",
        "examples/jsm/postprocessing/Pass.js": "vendor/addons/postprocessing/Pass.js",
        "examples/jsm/postprocessing/ShaderPass.js": "vendor/addons/postprocessing/ShaderPass.js",
        "examples/jsm/postprocessing/MaskPass.js": "vendor/addons/postprocessing/MaskPass.js",
        "examples/jsm/shaders/CopyShader.js": "vendor/addons/shaders/CopyShader.js",
        "examples/jsm/shaders/OutputShader.js": "vendor/addons/shaders/OutputShader.js",
        "examples/jsm/shaders/GTAOShader.js": "vendor/addons/shaders/GTAOShader.js",
        "examples/jsm/shaders/PoissonDenoiseShader.js": "vendor/addons/shaders/PoissonDenoiseShader.js",
        "examples/jsm/math/SimplexNoise.js": "vendor/addons/math/SimplexNoise.js",
        "LICENSE": "vendor/THREE_LICENSE.txt",
    }
    with tempfile.TemporaryDirectory(dir=ROOT) as temporary_dir:
        root = Path(temporary_dir)
        three_source = root / "three"
        output = root / "viewer"
        for index, source_name in enumerate(required):
            source = three_source / source_name
            source.parent.mkdir(parents=True, exist_ok=True)
            source.write_bytes(f"required-{index}".encode())
        unrelated = three_source / "examples/jsm/postprocessing/UnrelatedPass.js"
        unrelated.parent.mkdir(parents=True, exist_ok=True)
        unrelated.write_bytes(b"must-not-copy")

        copy_three_vendor(three_source, output)

        packaged_files = {
            path.relative_to(output).as_posix()
            for path in output.rglob("*")
            if path.is_file()
        }
        assert packaged_files == set(required.values())
        for source_name, target_name in required.items():
            assert (output / target_name).read_bytes() == (three_source / source_name).read_bytes()


def test_render_mode_selector_dom_and_styles_contract() -> None:
    html = (VIEWER / "index.html").read_text(encoding="utf-8")
    parser = _RenderModeMarkupParser()
    parser.feed(html)

    assert "<h2>渲染模式</h2>" in html
    assert parser.select_values == ["render-mode"]
    assert parser.option_labels == {
        "diagnostic": "配准核查",
        "realistic": "真实解剖",
    }
    assert parser.note_text == "真实模式只改变显示，不提高配准精度。"

    styles = (VIEWER / "styles.css").read_text(encoding="utf-8")
    assert ".render-mode-row" in styles
    assert "#render-mode-note" in styles


def test_thorax_variant_selector_dom_and_styles_contract() -> None:
    html = (VIEWER / "index.html").read_text(encoding="utf-8")

    assert 'id="thorax-variant"' in html
    assert '<option value="auto">随渲染模式</option>' in html
    assert '<option value="registration">配准原网格</option>' in html
    assert '<option value="display">平滑显示网格</option>' in html
    assert 'id="thorax-variant-note"' in html
    assert "平滑网格只用于显示" in html

    styles = (VIEWER / "styles.css").read_text(encoding="utf-8")
    assert ".thorax-variant-row" in styles
    assert "#thorax-variant-note" in styles


def test_thorax_variant_app_integration_changes_visibility_only() -> None:
    source = (VIEWER / "app.js").read_text(encoding="utf-8")

    assert 'from "./thorax_variants.mjs"' in source
    assert 'const thoraxVariantSelector = $("thorax-variant");' in source
    assert "resolveThoraxAssetRecords(data.assets.thorax)" in source
    assert "async function loadThoraxVariant(" in source
    assert "function setThoraxVariant(" in source
    assert "setThoraxVariantVisibility(thoraxVariants, variant)" in source
    assert "resolveThoraxVariant(renderModeSelector.value, thoraxVariantSelector.value)" in source

    setter_start = source.index("function setThoraxVariant(")
    setter_end = source.index("\n}", setter_start)
    setter = source[setter_start:setter_end]
    assert ".position" not in setter
    assert ".rotation" not in setter
    assert ".scale" not in setter
    assert ".matrix" not in setter
    assert ".geometry" not in setter


def test_thorax_variant_contract_with_node() -> None:
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is not installed")

    completed = subprocess.run(
        [node, str(ROOT / "tests/js/thorax_variants.test.mjs")],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr


def test_render_mode_url_state_is_isolated_from_scene_contract() -> None:
    source = (VIEWER / "app.js").read_text(encoding="utf-8")

    assert 'import { normalizeRenderMode } from "./rendering_mode.mjs";' in source
    assert "function selectedRenderMode()" in source
    assert 'new URLSearchParams(window.location.search).get("render")' in source
    assert 'const renderModeSelector = $("render-mode");' in source
    assert "renderModeSelector.value = selectedRenderMode();" in source
    assert 'url.searchParams.set("render", mode);' in source
    assert 'history.replaceState(null, "", url);' in source

    handler_start = source.index('renderModeSelector.addEventListener("change"')
    handler_end = source.index("\n});", handler_start)
    handler = source[handler_start:handler_end]
    assert "normalizeRenderMode" in handler
    assert "new URL(window.location.href)" in handler
    for forbidden in ("renderer", "scene", "material", "light", "geometry", "transform", "quality"):
        assert forbidden not in handler


def test_rendering_controller_source_and_app_integration_contract() -> None:
    controller_source = (VIEWER / "rendering_scene.mjs").read_text(encoding="utf-8")
    app_source = (VIEWER / "app.js").read_text(encoding="utf-8")

    assert "export function createRenderingController" in controller_source
    for forbidden in (
        ".position.set(",
        ".rotation.set(",
        ".scale.set(",
        ".matrix.copy(",
        '.geometry.setAttribute("position"',
    ):
        assert forbidden not in controller_source

    assert "const lowPower =" in app_source
    assert "navigator.hardwareConcurrency" in app_source
    assert "lowPower ? 1.5 : 2.0" in app_source
    assert "createRenderingController({" in app_source
    assert "renderingController.setMode(renderModeSelector.value);" in app_source
    assert "renderingController.setMode(mode);" in app_source
    assert "requestRender();" in app_source
    assert 'import { createRenderingController, loadGtaoAddons } from "./rendering_scene.mjs";' in app_source
    assert "const gtaoAddons = await loadGtaoAddons();" in app_source
    assert "addons: gtaoAddons" in app_source
    for export_name in ("EffectComposer", "RenderPass", "GTAOPass", "OutputPass"):
        assert f"import {{ {export_name} }} from" not in app_source
    assert "lowPower," in app_source
    assert "renderingController.render(camera)" in app_source
    assert "renderingController.reportFrameTiming" in app_source
    assert "if (document.hidden) return;" in app_source
    assert 'document.addEventListener("visibilitychange"' in app_source
    assert "renderer.render(scene, camera);" not in app_source
    assert "scene.add(new THREE.HemisphereLight(0xd9f9ff, 0x102128, 1.25));" not in app_source
    assert "keyLight.position.set(-300, -500, 500);" not in app_source


def test_gtao_addon_loader_catches_module_graph_failure_and_returns_null() -> None:
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is not installed")

    script = r'''
import assert from "node:assert/strict";
import { pathToFileURL } from "node:url";

const moduleUrl = pathToFileURL("web/ct_dr_3d_viewer/rendering_scene.mjs").href;
const { loadGtaoAddons } = await import(moduleUrl);
const expected = [
  "three/addons/postprocessing/EffectComposer.js",
  "three/addons/postprocessing/RenderPass.js",
  "three/addons/postprocessing/GTAOPass.js",
  "three/addons/postprocessing/OutputPass.js",
];
const exportsBySpecifier = new Map(expected.map((specifier, index) => [
  specifier,
  { [specifier.split("/").at(-1).replace(".js", "")]: class { static index = index; } },
]));
const loadedSpecifiers = [];
const loaded = await loadGtaoAddons(async (specifier) => {
  loadedSpecifiers.push(specifier);
  return exportsBySpecifier.get(specifier);
}, () => { throw new Error("successful load warned"); });
assert.deepEqual(loadedSpecifiers, expected);
assert.deepEqual(Object.keys(loaded).sort(), ["EffectComposer", "GTAOPass", "OutputPass", "RenderPass"]);

const warnings = [];
const failed = await loadGtaoAddons(
  async () => { throw new TypeError("404 or invalid MIME type"); },
  (message) => warnings.push(message),
);
assert.equal(failed, null);
assert.equal(warnings.length, 1);
assert.match(warnings[0], /GTAO addons unavailable/);
'''
    completed = subprocess.run(
        [node, "--input-type=module", "--eval", script],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr


def test_anatomy_geometry_is_recomputed_for_smooth_shading_with_node() -> None:
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is not installed")

    script = r'''
import assert from "node:assert/strict";
import { pathToFileURL } from "node:url";

const moduleUrl = pathToFileURL("web/ct_dr_3d_viewer/rendering_scene.mjs").href;
const { prepareAnatomyMeshGeometry } = await import(moduleUrl);
const calls = [];
const material = { flatShading: true, needsUpdate: false };
const mesh = {
  geometry: {
    getAttribute(name) { return name === "position" ? { count: 8 } : null; },
    deleteAttribute(name) { calls.push(`delete:${name}`); },
    computeVertexNormals() { calls.push("compute"); },
    normalizeNormals() { calls.push("normalize"); },
  },
  material,
};

assert.equal(prepareAnatomyMeshGeometry(mesh, "heart"), true);
assert.deepEqual(calls, ["delete:normal", "compute", "normalize"]);
assert.equal(material.flatShading, false);
assert.equal(material.needsUpdate, true);

calls.length = 0;
assert.equal(prepareAnatomyMeshGeometry(mesh, "xa-image"), false);
assert.deepEqual(calls, []);
'''
    completed = subprocess.run(
        [node, "--input-type=module", "--eval", script],
        cwd=ROOT,
        text=True,
        capture_output=True,
        timeout=30,
    )
    assert completed.returncode == 0, completed.stderr


def test_cine_performance_sampling_uses_raf_frame_interval_only() -> None:
    source = (VIEWER / "app.js").read_text(encoding="utf-8")
    cine = source[source.index("function cineAnimation("):source.index("function setCinePlaying(")]
    render = source[source.index("function renderFrame("):source.index("async function initialize(")]

    assert "renderFrame(delta);" in cine
    assert "function renderFrame(frameIntervalMs = null)" in render
    assert "frameIntervalMs > 0" in render
    assert "reportFrameTiming(frameIntervalMs, true)" in render
    assert "performance.now()" not in render


def test_render_frame_publishes_stable_performance_observation_to_canvas_dataset() -> None:
    source = (VIEWER / "app.js").read_text(encoding="utf-8")
    observation = source[source.index("function syncRenderingObservation("):source.index("function renderFrame(")]
    render = source[source.index("function renderFrame("):source.index("async function initialize(")]

    assert render.index("renderingController.reportFrameTiming") < render.index("syncRenderingObservation();")
    assert "const status = renderingController.getGtaoStatus();" in observation
    assert "canvas.dataset.gtaoState = status.state;" in observation
    assert "canvas.dataset.gtaoReason = status.reason ?? \"\";" in observation
    assert "canvas.dataset.renderP95FrameMs" in observation
    assert "canvas.dataset.renderSampleFrameCount" in observation
    assert "canvas.dataset.renderCompletedWindowCount" in observation
    assert "geometry" not in observation


def test_single_and_multiseries_playback_cancel_pending_on_demand_render() -> None:
    source = (VIEWER / "app.js").read_text(encoding="utf-8")
    single = source[source.index("function setCinePlaying("):source.index("function setMultiseriesPlaying(")]
    multi = source[source.index("function setMultiseriesPlaying("):source.index("async function addDynamicXaCine(")]
    request = source[source.index("function requestRender("):source.index("function renderFrame(")]

    assert "if (playing) cancelPendingRender();" in single
    assert "if (playing) cancelPendingRender();" in multi
    assert "function cancelPendingRender()" in source
    assert "if (document.hidden || dynamicCine?.playing) return;" in request


def test_rendering_sources_cannot_write_medical_geometry_or_timing() -> None:
    app_source = (VIEWER / "app.js").read_text(encoding="utf-8")
    assert _find_protected_medical_writes(app_source) == []

    for module_name in ("rendering_mode.mjs", "rendering_scene.mjs", "thorax_variants.mjs"):
        module_source = (VIEWER / module_name).read_text(encoding="utf-8")
        assert '.geometry.setAttribute("position"' not in module_source
        assert ".geometry.attributes.position" not in module_source
        assert not re.search(
            r"\b(?!light\b)[A-Za-z_$][\w$]*\.(?:position|rotation|scale|matrix)"
            r"(?:\.[A-Za-z_$][\w$]*)?\s*=",
            module_source,
        )


@pytest.mark.parametrize(
    "unsafe_source",
    [
        "geometry.xa.calibration = replacement;",
        "geometry.xa.detector.corners_lps_mm += delta;",
        "Object.assign(geometry.xa.calibration, patch);",
        'Object.defineProperty(geometry.xa.detector, "corners_lps_mm", descriptor);',
        'Object.defineProperty(geometry.xa, "calibration", descriptor);',
        "Object.assign(geometry.xa, { detector: replacement });",
        "Object.assign(geometry.xa, { safe: 1, detector: replacement });",
        "Object.assign(geometry.xa, { detector });",
        "geometry.xa.calibration.flags <<= 1;",
        "geometry.xa.detector.flags >>= 1;",
        "geometry.dynamic_xa.frame_times_ms[0] >>>= 1;",
        "geometry.dynamic_xa.frame_times_ms.splice(0, 1, 40);",
        "const detector = geometry.xa.detector; detector.corners_lps_mm.push(point);",
        "const timing = geometry.dynamic_xa.frame_times_ms; timing[0] = 20;",
        "const xa = geometry.xa; const source = xa.source_lps_mm; source.fill(0);",
        "const metrics = geometry.xa.calibration.depth_sensitivity; Object.assign(metrics, patch);",
    ],
)
def test_protected_write_guard_rejects_representative_unsafe_mutations(
    unsafe_source: str,
) -> None:
    assert _find_protected_medical_writes(unsafe_source), unsafe_source


def test_every_medical_mesh_family_has_an_exact_semantic_registration() -> None:
    source = (VIEWER / "app.js").read_text(encoding="utf-8")

    for kind in (
        "heart",
        "vessel-primary",
        "vessel-comparison",
        "bone",
        "ct-image",
        "dr-image",
        "xa-image",
        "annotation",
        "helper",
    ):
        assert f'renderingController.registerMesh(' in source
        assert f'"{kind}"' in source

    assert 'renderingController.registerMesh(mesh, "ct-image");' in source
    assert 'renderingController.registerMesh(object, "bone");' in source
    assert 'renderingController.registerMesh(object, isHeart ? "heart" : "bone");' in source
    assert 'renderingController.registerMesh(plane, "dr-image");' in source
    assert 'renderingController.registerMesh(plane, "xa-image");' in source
    assert 'renderingController.registerMesh(overlay, "xa-image");' in source
    assert 'state.run.run_id === activeDynamicRunId ? "vessel-primary" : "vessel-comparison"' in source
    assert 'renderingController.registerMesh(sprite, "annotation");' in source
    assert 'renderingController.registerMesh(helper, "helper");' in source

    cleanup = source[source.index("function disposeFrameGroup"):source.index("function createComparisonRunState")]
    assert "disposeConnectedVesselGroup(frameGroup" in cleanup
    assert "unregister: unregisterVesselMesh" in cleanup
    unregister = source[source.index("function unregisterVesselMesh"):source.index("function disposeFrameGroup")]
    assert "renderingController.unregisterMesh(mesh)" in unregister

    assert 'renderingController.setOpacity("bone", Number(event.target.value), groups.vertebrae);' in source
    assert 'renderingController.setOpacity("bone", Number(event.target.value), groups.ribs);' in source
    assert 'renderingController.setOpacity("heart", Number(event.target.value), groups.heart);' in source
    assert 'renderingController.setOpacity("vessel-primary", Number(event.target.value));' in source
    assert 'dynamicCine.overlayMaterial.opacity = Number(event.target.value);' in source


def test_semantic_material_controller_runtime_contract_with_node() -> None:
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is not installed")

    script = r'''
import assert from "node:assert/strict";
import { pathToFileURL } from "node:url";

const moduleUrl = pathToFileURL("web/ct_dr_3d_viewer/rendering_scene.mjs").href;
const { createRenderingController } = await import(moduleUrl);

const disposals = new Map();
class Disposable {
  constructor(type) { this.type = type; }
  dispose() { disposals.set(this, (disposals.get(this) ?? 0) + 1); }
}
class Color {
  constructor(value) { this.value = value; }
  clone() { return new Color(this.value); }
  getHex() { return this.value; }
}
class Position { constructor() { this.x = 0; this.y = 0; this.z = 0; } }
class Object3D {
  constructor() { this.children = []; this.parent = null; this.visible = true; }
  add(...objects) {
    for (const object of objects) { object.parent = this; this.children.push(object); }
  }
  remove(object) {
    this.children = this.children.filter((item) => item !== object);
    if (object.parent === this) object.parent = null;
  }
}
class Scene extends Object3D { constructor() { super(); this.environment = null; this.fog = null; } }
class Group extends Object3D {}
class Mesh extends Object3D {
  constructor(geometry, material) {
    super();
    Object.assign(this, { geometry, material, renderOrder: 0, userData: {}, matrixWorld: { token: "same" } });
  }
}
class Material extends Disposable {
  constructor(type, options = {}) {
    super(type);
    Object.assign(this, options);
    if (Object.hasOwn(options, "color") && !(options.color instanceof Color)) this.color = new Color(options.color);
  }
}
class MeshBasicMaterial extends Material { constructor(options) { super("basic", options); } }
class MeshStandardMaterial extends Material { constructor(options) { super("standard", options); } }
class MeshPhysicalMaterial extends Material { constructor(options) { super("physical", options); } }
class MeshDepthMaterial extends Material { constructor(options) { super("depth", options); } }
class Light extends Disposable {
  constructor(type) { super(type); this.position = new Position(); }
}
class HemisphereLight extends Light { constructor() { super("hemisphere"); } }
class DirectionalLight extends Light { constructor() { super("directional"); this.castShadow = true; } }
class AmbientLight extends Light { constructor() { super("ambient"); } }
class FogExp2 { constructor(color, density) { Object.assign(this, { color, density }); } }
class BoxGeometry extends Disposable { constructor() { super("environment-geometry"); } }
class PMREMGenerator extends Disposable {
  constructor() { super("pmrem-generator"); }
  compileCubemapShader() {}
  fromScene() {
    return new class extends Disposable {
      constructor() { super("pmrem-target"); this.texture = { kind: "environment" }; }
    }();
  }
}
const THREE = {
  SRGBColorSpace: "srgb",
  NoToneMapping: "none",
  ACESFilmicToneMapping: "aces",
  BackSide: "back",
  DoubleSide: "double",
  RGBADepthPacking: "rgba-depth",
  Color,
  Scene,
  Group,
  Mesh,
  MeshBasicMaterial,
  MeshStandardMaterial,
  MeshPhysicalMaterial,
  MeshDepthMaterial,
  HemisphereLight,
  DirectionalLight,
  AmbientLight,
  FogExp2,
  BoxGeometry,
  PMREMGenerator,
};
const renderer = {
  shadowMap: { enabled: true },
  setClearColor() {},
  setPixelRatio() {},
  setSize() {},
};
const scene = new Scene();
let renderRequests = 0;
const controller = createRenderingController({
  THREE,
  renderer,
  scene,
  requestRender: () => { renderRequests += 1; },
});

const bypassMaterials = [];
for (const kind of ["ct-image", "dr-image", "xa-image"]) {
  const imageMaterial = new MeshBasicMaterial({ opacity: 0.8, toneMapped: true, fog: true });
  const image = new Mesh({ id: `${kind}-geometry` }, imageMaterial);
  controller.registerMesh(image, kind);
  assert.equal(image.material, imageMaterial, `${kind} material identity changed`);
  assert.equal(imageMaterial.toneMapped, false);
  assert.equal(imageMaterial.fog, false);
  assert.equal(imageMaterial.envMap, undefined);
  bypassMaterials.push(imageMaterial);
}

const annotationMaterial = new MeshBasicMaterial({ toneMapped: true, fog: true });
const annotation = new Mesh({ id: "annotation-geometry" }, annotationMaterial);
controller.registerMesh(annotation, "annotation");
assert.equal(annotation.material, annotationMaterial);
assert.equal(annotationMaterial.toneMapped, false);
assert.equal(annotationMaterial.fog, false);
bypassMaterials.push(annotationMaterial);

const helperMaterial = new MeshBasicMaterial({ color: 0x00ffff, toneMapped: true, fog: true });
const helper = new Mesh({ id: "helper-geometry" }, helperMaterial);
controller.registerMesh(helper, "helper");
assert.equal(helperMaterial.toneMapped, false);
assert.equal(helperMaterial.fog, false);
bypassMaterials.push(helperMaterial);

const arrayMaterials = [
  new MeshBasicMaterial({ toneMapped: true, fog: true }),
  new MeshBasicMaterial({ toneMapped: true, fog: true }),
];
const arrayImage = new Mesh({ id: "array-xa-geometry" }, arrayMaterials);
controller.registerMesh(arrayImage, "xa-image");
assert.ok(arrayMaterials.every((material) => material.toneMapped === false && material.fog === false));
bypassMaterials.push(...arrayMaterials);

const vertebraGroup = new Group();
const ribGroup = new Group();
const vertebraDiagnostic = new MeshStandardMaterial({ color: 0xeeeeee, opacity: 0.6 });
const ribDiagnostic = new MeshStandardMaterial({ color: 0xdddddd, opacity: 0.6 });
const vertebra = new Mesh({ id: "vertebra-geometry" }, vertebraDiagnostic);
const rib = new Mesh({ id: "rib-geometry" }, ribDiagnostic);
vertebraGroup.add(vertebra);
ribGroup.add(rib);
controller.registerMesh(vertebra, "bone");
controller.registerMesh(rib, "bone");

const comparisonDiagnostic = new MeshStandardMaterial({ color: new Color(0x55d9d1), opacity: 0.9 });
const comparison = new Mesh({ id: "comparison-geometry" }, comparisonDiagnostic);
controller.registerMesh(comparison, "vessel-comparison");
const primaryDiagnostic = new MeshStandardMaterial({ color: new Color(0x884455), opacity: 0.9 });
const primary = new Mesh({ id: "primary-geometry" }, primaryDiagnostic);
controller.registerMesh(primary, "vessel-primary");

const heartDiagnostic = new MeshStandardMaterial({
  color: 0xc94168,
  opacity: 0.16,
  side: THREE.DoubleSide,
});
const heartGeometry = { id: "heart-geometry", dispose() { throw new Error("caller geometry disposed"); } };
const heart = new Mesh(heartGeometry, heartDiagnostic);
const heartState = {
  geometry: heart.geometry,
  matrixWorld: heart.matrixWorld,
  userData: heart.userData,
  visible: heart.visible,
};
controller.registerMesh(heart, "heart");
assert.equal(heart.children.length, 1, "heart depth helper missing");
const depthHelper = heart.children[0];
assert.equal(depthHelper.geometry, heartGeometry);
assert.equal(depthHelper.material.colorWrite, false);
assert.equal(depthHelper.material.depthWrite, true);
assert.equal(depthHelper.visible, false);
const intersections = [];
depthHelper.raycast(null, intersections);
assert.deepEqual(intersections, [], "heart depth helper became pickable");

controller.setMode("realistic");
assert.ok(bypassMaterials.every((material) => material.toneMapped === false && material.fog === false));
assert.ok(heart.material instanceof MeshPhysicalMaterial);
assert.equal(heart.material.color.getHex(), 0x7a2633);
assert.equal(heart.material.metalness, 0);
assert.equal(heart.material.opacity, 0.16, "realistic mode changed the heart opacity control value");
assert.equal(heart.material.side, THREE.DoubleSide, "realistic mode culled inward-wound heart front surfaces");
assert.equal(depthHelper.material.side, THREE.DoubleSide, "depth prepass no longer covers the same heart surfaces");
assert.ok(vertebra.material instanceof MeshStandardMaterial);
assert.equal(vertebra.material.color.getHex(), 0xd8cdb3);
assert.equal(vertebra.material.metalness, 0);
assert.ok(comparison.material instanceof MeshPhysicalMaterial);
assert.equal(comparison.material.color.getHex(), 0x55d9d1, "comparison series color was not preserved");
assert.ok(primary.material instanceof MeshPhysicalMaterial);
assert.equal(primary.material.color.getHex(), 0xb8323e);
assert.equal(helper.material, helperMaterial, "helper became a tissue material");
assert.equal(depthHelper.visible, true);
assert.equal(heart.geometry, heartState.geometry);
assert.equal(heart.matrixWorld, heartState.matrixWorld);
assert.equal(heart.userData, heartState.userData);
assert.equal(heart.visible, heartState.visible);

controller.setOpacity("vessel-primary", 0.45);
assert.equal(renderRequests, 1);
assert.equal(primaryDiagnostic.opacity, 0.45);
assert.equal(primary.material.opacity, 0.45);
assert.equal(comparisonDiagnostic.opacity, 0.45, "comparison vessel opacity no longer follows the 3D vessel control");
assert.equal(comparison.material.opacity, 0.45);
assert.equal(comparison.material.color.getHex(), 0x55d9d1, "comparison series color changed after opacity update");
assert.deepEqual(
  [comparison.material.roughness, comparison.material.clearcoat, comparison.material.sheen],
  [0.40, 0.10, 0.05],
  "comparison profile changed after opacity update",
);

controller.setOpacity("bone", 0.25, vertebraGroup);
assert.equal(renderRequests, 2);
assert.equal(vertebraDiagnostic.opacity, 0.25);
assert.equal(vertebra.material.opacity, 0.25);
assert.equal(ribDiagnostic.opacity, 0.6, "thorax opacity leaked from vertebra scope");
assert.equal(rib.material.opacity, 0.6, "realistic thorax opacity leaked from vertebra scope");
assert.equal(vertebraDiagnostic.fog, undefined, "anatomy diagnostic fog participation changed");
assert.equal(vertebra.material.fog, undefined, "anatomy realistic fog participation changed");

controller.setMode("diagnostic");
assert.ok(bypassMaterials.every((material) => material.toneMapped === false && material.fog === false));
assert.equal(heart.material, heartDiagnostic);
assert.equal(vertebra.material, vertebraDiagnostic);
assert.equal(comparison.material, comparisonDiagnostic);
assert.equal(depthHelper.visible, false);

controller.unregisterMesh(heart);
assert.equal(heart.children.length, 0, "heart depth helper survived unregister");
assert.equal(disposals.get(depthHelper.material), 1);
controller.dispose();
for (const [resource, count] of disposals) {
  if (["physical", "standard", "depth"].includes(resource.type)) {
    assert.equal(count, 1, `${resource.type} controller material disposed ${count} times`);
  }
}
assert.equal(disposals.has(annotationMaterial), false, "caller annotation material was disposed");
assert.equal(disposals.has(helperMaterial), false, "caller helper material was disposed");
'''
    completed = subprocess.run(
        [node, "--input-type=module", "--eval", script],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr


def test_rendering_controller_runtime_contract_with_node() -> None:
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is not installed")

    script = r'''
import assert from "node:assert/strict";
import { pathToFileURL } from "node:url";

const moduleUrl = pathToFileURL("web/ct_dr_3d_viewer/rendering_scene.mjs").href;
const { createRenderingController } = await import(moduleUrl);

const disposed = [];
class Disposable {
  constructor(type) { this.type = type; }
  dispose() { disposed.push(this.type); }
}
class Position {
  constructor() { this.x = 0; this.y = 0; this.z = 0; }
}
class HemisphereLight extends Disposable {
  constructor(skyColor, groundColor, intensity) {
    super("HemisphereLight");
    Object.assign(this, { skyColor, groundColor, intensity, position: new Position() });
  }
}
class DirectionalLight extends Disposable {
  constructor(color, intensity) {
    super("DirectionalLight");
    Object.assign(this, { color, intensity, position: new Position(), castShadow: true });
  }
}
class AmbientLight extends Disposable {
  constructor(color, intensity) {
    super("AmbientLight");
    Object.assign(this, { color, intensity, position: new Position() });
  }
}
class Scene {
  constructor() { this.children = []; this.environment = null; this.fog = null; }
  add(...objects) { this.children.push(...objects); }
  remove(object) { this.children = this.children.filter((item) => item !== object); }
}
class FogExp2 {
  constructor(color, density) { Object.assign(this, { color, density }); }
}
class Color {
  constructor(value) { this.value = value; }
}
class BoxGeometry extends Disposable { constructor() { super("environment-geometry"); } }
class MeshBasicMaterial extends Disposable {
  constructor(options) { super("environment-material"); this.options = options; }
}
class Mesh { constructor(geometry, material) { Object.assign(this, { geometry, material }); } }
class PMREMGenerator extends Disposable {
  constructor() { super("pmrem-generator"); }
  compileCubemapShader() {}
  fromScene(source) {
    assert.ok(source.children.length > 0, "programmatic environment source is empty");
    return new class extends Disposable {
      constructor() { super("pmrem-target"); this.texture = { kind: "pmrem-texture" }; }
    }();
  }
}
const THREE = {
  SRGBColorSpace: "srgb",
  NoToneMapping: "none",
  ACESFilmicToneMapping: "aces",
  BackSide: "back",
  HemisphereLight,
  DirectionalLight,
  AmbientLight,
  Scene,
  FogExp2,
  Color,
  BoxGeometry,
  MeshBasicMaterial,
  Mesh,
  PMREMGenerator,
};
const renderer = {
  shadowMap: { enabled: true },
  setClearColor(color, alpha) { this.clearColor = [color, alpha]; },
  setPixelRatio(value) { this.pixelRatio = value; },
  setSize(width, height, updateStyle) { this.size = [width, height, updateStyle]; },
};
const scene = new Scene();
let renderRequests = 0;
const controller = createRenderingController({
  THREE,
  renderer,
  scene,
  requestRender: () => { renderRequests += 1; },
});
assert.deepEqual(
  Object.keys(controller).sort(),
  ["disableGtao", "dispose", "getGtaoStatus", "registerMesh", "render", "reportFrameTiming", "resize", "setMode", "setOpacity", "unregisterMesh"],
);

controller.setMode("diagnostic");
assert.equal(renderer.outputColorSpace, "srgb");
assert.equal(renderer.toneMapping, "none");
assert.equal(renderer.toneMappingExposure, 1);
assert.equal(renderer.shadowMap.enabled, false);
assert.deepEqual(renderer.clearColor, [0x03090c, 1]);
assert.equal(scene.fog.color, 0x03090c);
assert.equal(scene.fog.density, 0.0007);
const diagnosticHemisphere = scene.children.find((item) => item instanceof HemisphereLight);
const diagnosticKey = scene.children.find((item) => item instanceof DirectionalLight);
assert.deepEqual(
  [diagnosticHemisphere.skyColor, diagnosticHemisphere.groundColor, diagnosticHemisphere.intensity],
  [0xd9f9ff, 0x102128, 1.25],
);
assert.deepEqual(
  [diagnosticKey.color, diagnosticKey.intensity, diagnosticKey.position.x, diagnosticKey.position.y, diagnosticKey.position.z],
  [0xffffff, 2.2, -300, -500, 500],
);

controller.setMode("realistic");
assert.equal(renderer.outputColorSpace, "srgb");
assert.equal(renderer.toneMapping, "aces");
assert.equal(renderer.toneMappingExposure, 1);
assert.equal(renderer.shadowMap.enabled, false);
assert.equal(scene.fog, null);
assert.equal(scene.environment.kind, "pmrem-texture");
assert.ok(scene.children.filter((item) => item instanceof DirectionalLight).length >= 3);
assert.ok(scene.children.some((item) => item instanceof HemisphereLight));
assert.ok(scene.children.some((item) => item instanceof AmbientLight));
assert.ok(disposed.includes("environment-geometry"));
assert.ok(disposed.includes("environment-material"));
assert.ok(disposed.includes("pmrem-generator"));

const callerGeometry = { dispose() { throw new Error("caller geometry was disposed"); } };
const callerMesh = { geometry: callerGeometry };
controller.registerMesh(callerMesh, "heart");
controller.setOpacity("heart", 0.35);
controller.unregisterMesh(callerMesh);
controller.resize(900, 600, 1.5);
assert.equal(renderRequests, 1);
assert.equal(renderer.pixelRatio, 1.5);
assert.deepEqual(renderer.size, [900, 600, false]);

controller.dispose();
assert.equal(scene.environment, null);
assert.equal(scene.children.length, 0);
assert.ok(disposed.includes("pmrem-target"));

class FailingPMREMGenerator extends PMREMGenerator {
  fromScene() { throw new Error("allocation failed"); }
}
const fallbackThree = { ...THREE, PMREMGenerator: FailingPMREMGenerator };
const fallbackRenderer = { ...renderer, shadowMap: { enabled: true } };
const fallbackScene = new Scene();
const fallbackController = createRenderingController({
  THREE: fallbackThree,
  renderer: fallbackRenderer,
  scene: fallbackScene,
  requestRender: () => {},
});
assert.doesNotThrow(() => fallbackController.setMode("realistic"));
assert.equal(fallbackRenderer.toneMapping, "aces");
assert.equal(fallbackRenderer.shadowMap.enabled, false);
assert.equal(fallbackScene.environment, null);
assert.ok(fallbackScene.children.filter((item) => item instanceof DirectionalLight).length >= 3);
fallbackController.dispose();
'''
    completed = subprocess.run(
        [node, "--input-type=module", "--eval", script],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr


def test_gtao_runtime_is_optional_failure_safe_and_performance_bounded() -> None:
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is not installed")

    script = r'''
import assert from "node:assert/strict";
import { pathToFileURL } from "node:url";

const moduleUrl = pathToFileURL("web/ct_dr_3d_viewer/rendering_scene.mjs").href;
const { createRenderingController } = await import(moduleUrl);

class Position { constructor() { this.x = 0; this.y = 0; this.z = 0; } }
class Disposable { constructor(type) { this.type = type; this.disposeCount = 0; } dispose() { this.disposeCount += 1; } }
class Scene {
  constructor() { this.children = []; this.environment = null; this.fog = null; }
  add(...objects) { this.children.push(...objects); }
  remove(object) { this.children = this.children.filter((item) => item !== object); }
}
class Light extends Disposable { constructor(type) { super(type); this.position = new Position(); } }
class HemisphereLight extends Light { constructor() { super("hemisphere"); } }
class DirectionalLight extends Light { constructor() { super("directional"); this.castShadow = true; } }
class AmbientLight extends Light { constructor() { super("ambient"); } }
class Color { constructor(value) { this.value = value; } }
class FogExp2 { constructor(color, density) { Object.assign(this, { color, density }); } }
class BoxGeometry extends Disposable { constructor() { super("box"); } }
class Material extends Disposable { constructor(options = {}) { super("material"); Object.assign(this, options); } }
class MeshBasicMaterial extends Material {}
class MeshStandardMaterial extends Material {}
class MeshPhysicalMaterial extends Material {}
class MeshDepthMaterial extends Material {}
class Mesh {
  constructor(geometry, material) { Object.assign(this, { geometry, material, children: [], visible: true, renderOrder: 0 }); }
  add(object) { this.children.push(object); object.parent = this; }
  remove(object) { this.children = this.children.filter((item) => item !== object); }
}
class PMREMGenerator extends Disposable {
  constructor() { super("pmrem"); }
  compileCubemapShader() {}
  fromScene() { return { texture: {}, dispose() {} }; }
}
const THREE = {
  SRGBColorSpace: "srgb", NoToneMapping: "none", ACESFilmicToneMapping: "aces",
  BackSide: "back", DoubleSide: "double", RGBADepthPacking: "depth",
  Scene, HemisphereLight, DirectionalLight, AmbientLight, Color, FogExp2,
  BoxGeometry, MeshBasicMaterial, MeshStandardMaterial, MeshPhysicalMaterial,
  MeshDepthMaterial, Mesh, PMREMGenerator,
};

const resources = [];
class EffectComposer extends Disposable {
  constructor(renderer) { super("composer"); this.renderer = renderer; this.passes = []; resources.push(this); }
  addPass(pass) { this.passes.push(pass); }
  setSize(width, height) { this.size = [width, height]; }
  setPixelRatio(pixelRatio) { this.pixelRatio = pixelRatio; }
  render() { this.renderCount = (this.renderCount ?? 0) + 1; }
}
class RenderPass extends Disposable { constructor(scene, camera) { super("render-pass"); Object.assign(this, { scene, camera }); resources.push(this); } }
class GTAOPass extends Disposable {
  constructor(scene, camera, width, height) { super("gtao-pass"); Object.assign(this, { scene, camera, width, height }); resources.push(this); }
  updateGtaoMaterial(options) { this.gtaoOptions = options; }
  updatePdMaterial(options) { this.pdOptions = options; }
}
class OutputPass extends Disposable { constructor() { super("output-pass"); resources.push(this); } }
const addons = { EffectComposer, RenderPass, GTAOPass, OutputPass };

function makeRenderer() {
  return {
    capabilities: { isWebGL2: true }, shadowMap: { enabled: true }, directRenders: 0,
    setClearColor() {}, setPixelRatio(value) { this.pixelRatio = value; },
    setSize(width, height) { this.size = [width, height]; },
    getPixelRatio() { return this.pixelRatio ?? 1; },
    getSize(target) { target.width = 640; target.height = 480; return target; },
    render() { this.directRenders += 1; },
  };
}

const renderer = makeRenderer();
const scene = new Scene();
const warnings = [];
const camera = { near: 0.5, far: 6000, isPerspectiveCamera: true };
const controller = createRenderingController({
  THREE, renderer, scene, camera, requestRender() {}, lowPower: false, addons,
  warn: (message) => warnings.push(message),
});
controller.resize(800, 600, 1.5);
controller.setMode("realistic");
assert.deepEqual(controller.getGtaoStatus(), {
  state: "enabled",
  reason: null,
  warmupFrameCount: 0,
  sampleFrameCount: 0,
  completedWindowCount: 0,
  lastP95FrameMs: null,
});
assert.ok(Object.isFrozen(controller.getGtaoStatus()), "rendering status snapshot is mutable");
const gtao = resources.find((item) => item.type === "gtao-pass");
assert.ok(gtao, "GTAO pass was not allocated");
assert.ok(gtao.blendIntensity <= 0.35, "AO intensity is not conservative");
assert.ok(gtao.gtaoOptions.radius <= 0.25, "AO radius is not conservative");
controller.render(camera);
assert.equal(resources.find((item) => item.type === "composer").renderCount, 1);
assert.equal(renderer.directRenders, 0);

const image = new Mesh({}, new MeshBasicMaterial({ toneMapped: true }));
controller.registerMesh(image, "xa-image");
assert.equal(image.material.toneMapped, false);
assert.equal(controller.getGtaoStatus().state, "disabled");
assert.equal(controller.getGtaoStatus().reason, "medical-overlay-present");
assert.equal(warnings.length, 1);
for (let index = 0; index < 30; index += 1) controller.reportFrameTiming(99, true);
for (let index = 0; index < 120; index += 1) controller.reportFrameTiming(16, true);
assert.deepEqual(controller.getGtaoStatus(), {
  state: "disabled",
  reason: "medical-overlay-present",
  warmupFrameCount: 30,
  sampleFrameCount: 0,
  completedWindowCount: 1,
  lastP95FrameMs: 16,
});
controller.disableGtao("second-reason-must-not-replace-first");
assert.equal(controller.getGtaoStatus().reason, "medical-overlay-present");
assert.equal(controller.getGtaoStatus().lastP95FrameMs, 16);
assert.equal(warnings.length, 1);
controller.render(camera);
assert.equal(renderer.directRenders, 1);
for (const resource of resources) assert.equal(resource.disposeCount, 1, `${resource.type} not disposed exactly once`);
controller.dispose();
for (const resource of resources) assert.equal(resource.disposeCount, 1, `${resource.type} double-disposed`);

class FailingComposer extends EffectComposer { render() { throw new Error("framebuffer lost"); } }
const failingRenderer = makeRenderer();
const failingWarnings = [];
const failingController = createRenderingController({
  THREE, renderer: failingRenderer, scene: new Scene(), camera, requestRender() {}, lowPower: false,
  addons: { ...addons, EffectComposer: FailingComposer }, warn: (message) => failingWarnings.push(message),
});
failingController.setMode("realistic");
assert.doesNotThrow(() => failingController.render(camera));
assert.equal(failingRenderer.directRenders, 1, "render failure did not fall back in the same frame");
assert.equal(failingController.getGtaoStatus().reason, "render-failed");
assert.equal(failingWarnings.length, 1);

class FailingRenderPass extends RenderPass { constructor() { super(...arguments); throw new Error("pass allocation failed"); } }
const allocationController = createRenderingController({
  THREE, renderer: makeRenderer(), scene: new Scene(), camera, requestRender() {}, lowPower: false,
  addons: { ...addons, RenderPass: FailingRenderPass }, warn() {},
});
assert.doesNotThrow(() => allocationController.setMode("realistic"));
assert.equal(allocationController.getGtaoStatus().state, "disabled");
assert.equal(allocationController.getGtaoStatus().reason, "allocation-failed");

class ResizeFailComposer extends EffectComposer {
  setSize(width, height) {
    this.resizeCalls = (this.resizeCalls ?? 0) + 1;
    if (this.resizeCalls > 1) throw new Error("resize failed");
    super.setSize(width, height);
  }
}
const resizeController = createRenderingController({
  THREE, renderer: makeRenderer(), scene: new Scene(), camera, requestRender() {}, lowPower: false,
  addons: { ...addons, EffectComposer: ResizeFailComposer }, warn() {},
});
resizeController.setMode("realistic");
assert.doesNotThrow(() => resizeController.resize(1024, 768, 1.25));
assert.equal(resizeController.getGtaoStatus().state, "disabled");
assert.equal(resizeController.getGtaoStatus().reason, "resize-failed");

const performanceRenderer = makeRenderer();
const performanceController = createRenderingController({
  THREE, renderer: performanceRenderer, scene: new Scene(), camera, requestRender() {}, lowPower: false,
  addons, warn() {},
});
performanceController.setMode("realistic");
for (let index = 0; index < 30; index += 1) performanceController.reportFrameTiming(100, true);
assert.equal(performanceController.getGtaoStatus().state, "enabled", "warm-up samples disabled AO");
for (let index = 0; index < 120; index += 1) performanceController.reportFrameTiming(index < 114 ? 40 : 1, true);
assert.equal(performanceController.getGtaoStatus().state, "enabled", "one slow window disabled AO");
assert.equal(performanceController.getGtaoStatus().completedWindowCount, 1);
assert.equal(performanceController.getGtaoStatus().lastP95FrameMs, 40);
performanceController.reportFrameTiming(1000, false);
for (let index = 0; index < 120; index += 1) performanceController.reportFrameTiming(index < 114 ? 40 : 1, true);
assert.equal(performanceController.getGtaoStatus().state, "disabled");
assert.equal(performanceController.getGtaoStatus().reason, "performance-p95");
assert.equal(performanceController.getGtaoStatus().completedWindowCount, 2);
assert.equal(performanceController.getGtaoStatus().lastP95FrameMs, 40);

const lowPowerController = createRenderingController({
  THREE, renderer: makeRenderer(), scene: new Scene(), camera, requestRender() {}, lowPower: true, addons, warn() {},
});
lowPowerController.setMode("realistic");
assert.equal(lowPowerController.getGtaoStatus().state, "disabled");
assert.equal(lowPowerController.getGtaoStatus().reason, "low-power-device");
'''
    completed = subprocess.run(
        [node, "--input-type=module", "--eval", script],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr


def test_render_mode_change_does_not_reach_document_render_listener() -> None:
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is not installed")

    script = r'''
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { pathToFileURL } from "node:url";

const source = readFileSync("web/ct_dr_3d_viewer/app.js", "utf8");
const startToken = 'renderModeSelector.addEventListener("change", (event) => {';
const handlerStart = source.indexOf(startToken);
assert.notEqual(handlerStart, -1, "render-mode change handler is missing");
const bodyStart = handlerStart + startToken.length;
const bodyEnd = source.indexOf("\n});", bodyStart);
assert.notEqual(bodyEnd, -1, "render-mode change handler is incomplete");
const handlerBody = source.slice(bodyStart, bodyEnd);

const moduleUrl = pathToFileURL("web/ct_dr_3d_viewer/rendering_mode.mjs").href;
const { normalizeRenderMode } = await import(moduleUrl);
const thoraxModuleUrl = pathToFileURL("web/ct_dr_3d_viewer/thorax_variants.mjs").href;
const { resolveThoraxVariant } = await import(thoraxModuleUrl);
const renderModeSelector = { value: "diagnostic" };
const thoraxVariantSelector = { value: "auto" };
const window = { location: { href: "https://viewer.test/?run=series_03&render=diagnostic" } };
let replacedUrl = null;
const history = { replaceState: (_state, _title, url) => { replacedUrl = url; } };
const renderingController = { modes: [], setMode(mode) { this.modes.push(mode); } };
const selectedThoraxVariants = [];
const setThoraxVariant = (variant) => { selectedThoraxVariants.push(variant); };
let explicitRenderRequests = 0;
const requestRender = () => { explicitRenderRequests += 1; };
const makeHandler = new Function(
  "normalizeRenderMode",
  "resolveThoraxVariant",
  "renderModeSelector",
  "thoraxVariantSelector",
  "window",
  "history",
  "URL",
  "renderingController",
  "setThoraxVariant",
  "requestRender",
  `return (event) => {${handlerBody}\n};`,
);
const handler = makeHandler(
  normalizeRenderMode,
  resolveThoraxVariant,
  renderModeSelector,
  thoraxVariantSelector,
  window,
  history,
  URL,
  renderingController,
  setThoraxVariant,
  requestRender,
);

let documentRenderRequests = 0;
const event = {
  target: { value: "realistic" },
  propagationStopped: false,
  stopPropagation() { this.propagationStopped = true; },
};
handler(event);
if (!event.propagationStopped) documentRenderRequests += 1;

assert.equal(
  documentRenderRequests,
  0,
  "render-mode change reached the document change listener and scheduled renderFrame",
);
assert.equal(renderModeSelector.value, "realistic");
assert.equal(replacedUrl.searchParams.get("render"), "realistic");
assert.equal(replacedUrl.searchParams.get("run"), "series_03");
assert.deepEqual(renderingController.modes, ["realistic"]);
assert.deepEqual(selectedThoraxVariants, ["display"]);
assert.equal(explicitRenderRequests, 1);
'''
    completed = subprocess.run(
        [node, "--input-type=module", "--eval", script],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr


def test_rendering_mode_contract_with_node() -> None:
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is not installed")

    completed = subprocess.run(
        [node, str(ROOT / "tests/js/rendering_mode.test.mjs")],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr


def test_task5_vessel_motion_contract_with_node() -> None:
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is not installed")

    completed = subprocess.run(
        [
            node,
            "--experimental-default-type=module",
            str(ROOT / "tests/js/task5_vessel_interpolation.test.mjs"),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr


def test_task7_app_rebuilds_connected_graph_samples_and_fails_closed() -> None:
    source = (VIEWER / "app.js").read_text(encoding="utf-8")

    for symbol in (
        "buildConnectedVesselGroup",
        "disposeConnectedVesselGroup",
        "interpolateVesselFrame",
        "resolveVesselFrameBlend",
    ):
        assert symbol in source

    build = source[
        source.index("function buildRunVesselFrame("):
        source.index("function updateRunVesselFrame(")
    ]
    assert "buildConnectedVesselGroup(" in build
    assert "registerVesselGroup(" in build
    assert "mergeGeometries(" not in build
    assert "new THREE.TubeGeometry(" not in build

    update = source[
        source.index("function updateRunVesselBlend("):
        source.index("function updateDynamicVesselFrame(")
    ]
    assert "resolveVesselFrameBlend(" in update
    assert "interpolateVesselFrame(" in update
    assert "buildConnectedVesselGroup(" in update
    assert "state.interpolation.group" in update
    assert "vesselFallbackReason" in update
    for forbidden in (
        "fromPosition",
        "toPosition",
        "interpolatedPosition",
        "interpolatedNormal",
        'getAttribute("position")',
    ):
        assert forbidden not in update

    cleanup = source[
        source.index("function disposeFrameGroup("):
        source.index("function createComparisonRunState(")
    ]
    interpolation_cleanup = source[
        source.index("function disposeRunInterpolation("):
        source.index("function disposeComparisonRunStates(")
    ]
    assert "disposeConnectedVesselGroup(" in cleanup
    assert "unregisterVesselMesh" in cleanup
    assert "disposeConnectedVesselGroup(" in interpolation_cleanup
    assert "state.interpolation.group = null" in interpolation_cleanup


def test_task5_static_heart_and_medical_timing_remain_observed() -> None:
    source = (VIEWER / "app.js").read_text(encoding="utf-8")
    lower_source = source.lower()

    for forbidden in (
        "heartpulse",
        "heartbeatscale",
        "morphtarget",
    ):
        assert forbidden not in lower_source

    single_playback = source[source.index("function updateSinglePlayback("):source.index("function configureSinglePlaybackRange(")]
    assert 'updateDynamicFrame(frameAtElapsed(' not in single_playback
    assert "frameAtElapsed(geometry.dynamic_xa" in source
    assert "estimatedGroupPlaybackState(" in source
