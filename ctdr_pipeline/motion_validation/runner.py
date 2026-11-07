"""Isolated, research-only orchestration of motion observability checks."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from .input_policy import exclusion_reason, is_linked_path, require_eligible_input


_ASSET_PATHS = (
    "cine_manifest.json", "thorax_masks/heart.nii.gz", "all_series/cardiac_phase/phase_alignment.json",
    *(f"all_series/{run_id}/{name}" for run_id in ("series_01", "series_02") for name in (
        "series_manifest.json", "cine_u8.raw", "temporal/temporal_centerlines_u8.raw",
        f"{run_id}_calibration.json", "dynamic_3d/dynamic_vessel_3d.json",
    )),
)


def validate_output_paths(case_dir: Path, assets_root: Path, output_dir: Path) -> tuple[Path, Path, Path]:
    """Reject source overlap and *any* existing destination before a write."""
    case_dir, assets_root, output_dir = Path(case_dir), Path(assets_root), Path(output_dir)
    for source in (case_dir, assets_root):
        require_eligible_input(source)
    # lexists includes dangling links; do not resolve their evidence away.
    if is_linked_path(output_dir) or os.path.lexists(output_dir):
        raise FileExistsError(f"Output must be a new directory: {output_dir}")
    for parent in output_dir.parents:
        if is_linked_path(parent):
            raise ValueError(f"Output must not traverse a linked directory: {parent}")
    case_dir, assets_root, output_dir = case_dir.resolve(), assets_root.resolve(), output_dir.resolve()
    for source in (case_dir, assets_root):
        if not source.is_dir():
            raise FileNotFoundError(f"Input directory is missing: {source}")
    if output_dir.exists():
        raise FileExistsError(f"Output must be a new directory: {output_dir}")
    for source in (case_dir, assets_root):
        if output_dir.is_relative_to(source) or source.is_relative_to(output_dir):
            raise ValueError(f"Output must not overlap a source directory: {source}")
    return case_dir, assets_root, output_dir


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _snapshot_inputs(case_dir: Path, assets_root: Path, audit_assets: dict | None = None) -> dict[str, str]:
    require_eligible_input(case_dir)
    require_eligible_input(assets_root)
    paths = set()
    for parent, directories, names in os.walk(case_dir, followlinks=False):
        directories[:] = sorted(name for name in directories if exclusion_reason(Path(parent) / name) is None)
        for name in names:
            path = Path(parent) / name
            if exclusion_reason(path) is None and path.is_file():
                paths.add(path)
    for name in _ASSET_PATHS:
        path = assets_root / name
        if exclusion_reason(path) is not None:
            continue
        if audit_assets is not None and audit_assets.get(name, {}).get("read_status") != "read":
            continue
        if path.is_file():
            paths.add(path)
    return {str(path): _sha256(path) for path in sorted(paths)}


def _reject_unauditable_reparse_points(case_dir: Path, assets_root: Path) -> None:
    """Fail closed before the unchanged preflight's symlink-only exclusions.

    Preflight handles ordinary symlinks/candidate subtrees itself. On Windows
    3.11 a non-symlink reparse point is not recognized there, so reject such a
    request before audit IO instead of pretending preflight has excluded it.
    """
    def check(path: Path) -> None:
        for component in (*reversed(path.parents), path):
            if is_linked_path(component):
                if not component.is_symlink():
                    require_eligible_input(component)
                return

    for parent, directories, names in os.walk(case_dir, followlinks=False):
        for name in (*directories, *names):
            if not name.lower().endswith("_candidate"):
                check(Path(parent) / name)
        directories[:] = [name for name in directories if exclusion_reason(Path(parent) / name) is None]
    for relative in _ASSET_PATHS:
        check(assets_root / relative)


def _require_audited_assets(root: Path, relatives: list[str], audit_assets: dict) -> None:
    for relative in relatives:
        require_eligible_input(root / relative)
        status = audit_assets.get(relative, {}).get("read_status", "missing")
        if status != "read":
            raise ValueError(f"Authoritative asset is {status}: {relative}")


def _acquisition_timing(preflight: dict, run_id: str) -> dict:
    """Use only the exact audited source named by a verified cache identity link.

    No manifest paths are followed and no source is reopened or re-decoded.
    Source identity and timing sufficiency remain separate from geometry gates.
    """
    link = preflight.get("source_links", {}).get(run_id, {})
    relative = link.get("source_relative_path")
    matches = [record for record in preflight.get("source_files", []) if relative is not None
               and record.get("relative_path") == relative and record.get("read_status") == "read"]
    source = matches[0] if len(matches) == 1 else {}
    verified = (link.get("status") == "verified" and len(matches) == 1
                and all(link.get("checks", {}).get(key) is True for key in ("sop_instance_uid", "series_instance_uid", "sha256"))
                and source.get("modality") == "XA" and bool(source.get("sha256"))
                and source["sha256"] == link.get("actual_sha256"))
    return {
        "source_verified": verified,
        "source_link_status": link.get("status", "unknown"),
        "source_relative_path": relative if verified else None,
        "frame_count": source.get("frame_count") if verified else None,
        "timing_status": source.get("timing_status", "insufficient") if verified else "unverified",
        "timing_source": source.get("timing_source") if verified else None,
        "frame_timestamps_ms": source.get("frame_timestamps_ms", []) if verified else [],
    }


def _write_json(path: Path, payload: Any) -> None:
    text = json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text + "\n", encoding="utf-8")


def _code_manifest() -> dict[str, Any]:
    project_root = Path(__file__).resolve().parents[2]
    files = sorted((project_root / "ctdr_pipeline" / "motion_validation").glob("*.py"))
    files.extend(sorted((project_root / "tests" / "motion_validation").glob("test_*.py")))
    files.append(project_root / "run_heart_motion_validation.py")
    versions = {}
    for distribution in ("numpy", "scipy", "pydicom", "opencv-python-headless", "Pillow", "SimpleITK"):
        try:
            versions[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            versions[distribution] = None
    return {
        "python": platform.python_version(),
        "platform": platform.system(),
        "dependencies": versions,
        "git_commit": None,
        "checkpoint_method": "explicit added files and SHA256; workspace is not a Git repository",
        "files": [
            {"path": path.relative_to(project_root).as_posix(), "sha256": _sha256(path)}
            for path in files
            if path.is_file()
        ],
    }


def _heart_shape_points(mask_path: Path, *, max_points: int = 192) -> tuple[np.ndarray | None, dict[str, Any]]:
    reason = exclusion_reason(mask_path)
    if reason is not None:
        return None, {"kind": "digital_phantom", "reason": f"CT heart mask excluded: {reason}"}
    if not mask_path.is_file():
        return None, {"kind": "digital_phantom", "reason": "CT heart mask is unavailable"}
    try:
        import SimpleITK as sitk

        image = sitk.ReadImage(str(mask_path))
        if image.GetDimension() != 3:
            raise ValueError("heart mask must be a 3D image")
        contour = sitk.BinaryContour(sitk.Cast(image != 0, sitk.sitkUInt8))
        locations_zyx = np.argwhere(sitk.GetArrayViewFromImage(contour) != 0)
        if len(locations_zyx) < 12:
            raise ValueError("heart mask has too few boundary points")
        selected = locations_zyx[np.linspace(0, len(locations_zyx) - 1, min(max_points, len(locations_zyx)), dtype=int)]
        points = np.asarray(
            [image.TransformIndexToPhysicalPoint(tuple(int(value) for value in index[::-1])) for index in selected],
            dtype=np.float64,
        )
        if not np.isfinite(points).all():
            raise ValueError("heart mask physical coordinates are non-finite")
        return points, {
            "kind": "CT_heart_surface_shape_only",
            "path": str(mask_path),
            "sha256": _sha256(mask_path),
            "coordinate_system": "DICOM LPS mm",
            "point_count": len(points),
            "interpretation": "Static shape for synthetic motion only; not measured coronary or myocardial motion.",
        }
    except (RuntimeError, ValueError) as error:
        return None, {
            "kind": "digital_phantom",
            "reason": f"CT heart mask could not be sampled: {type(error).__name__}: {error}",
            "path": str(mask_path),
            "sha256": _sha256(mask_path),
        }


def _read_cine(series_dir: Path) -> np.ndarray:
    for path in (series_dir, series_dir / "series_manifest.json", series_dir / "cine_u8.raw"):
        require_eligible_input(path)
    manifest = json.loads((series_dir / "series_manifest.json").read_text(encoding="utf-8"))
    shape = manifest["shape_frame_row_column"]
    if len(shape) != 3 or any(type(value) is not int or value <= 0 for value in shape):
        raise ValueError("Invalid frame/row/column shape")
    raw = np.fromfile(series_dir / "cine_u8.raw", dtype=np.uint8)
    if raw.size != int(np.prod(shape)):
        raise ValueError("Cine byte length does not match the manifest")
    return raw.reshape(shape)


def _render_review_panels(series_dir: Path, report: dict[str, Any], output_path: Path) -> None:
    cine = _read_cine(series_dir)
    frame_indices = report.get("review_frame_indices") or np.linspace(0, len(cine) - 1, min(8, len(cine)), dtype=int).tolist()
    frame_indices = sorted({int(index) for index in frame_indices if 0 <= int(index) < len(cine)})
    tile_width, tile_height = 384, 414
    rows = (len(frame_indices) + 3) // 4
    sheet = Image.new("RGB", (tile_width * 4, tile_height * max(rows, 1)), "#121820")
    for tile_index, frame_index in enumerate(frame_indices):
        image = Image.fromarray(cine[frame_index]).convert("RGB")
        draw = ImageDraw.Draw(image)
        valid_count = 0
        for track in report.get("tracks", []):
            xy = track["xy_px"][frame_index]
            if track["valid"][frame_index] and xy is not None:
                x, y = (float(value) for value in xy)
                draw.ellipse((x - 3, y - 3, x + 3, y + 3), outline="#45e9c5", width=2)
                draw.text((x + 4, y - 5), str(track["track_id"]), fill="#fff35a")
                valid_count += 1
        image.thumbnail((tile_width, tile_width), Image.Resampling.LANCZOS)
        tile_x = (tile_index % 4) * tile_width
        tile_y = (tile_index // 4) * tile_height
        sheet.paste(image, (tile_x, tile_y + 30))
        ImageDraw.Draw(sheet).text(
            (tile_x + 5, tile_y + 5),
            f"{report['run_id']} frame {frame_index} | {valid_count} UNREVIEWED tracks",
            fill="white",
        )
    sheet.save(output_path)


def _review_template(observations: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "research_only": True,
        "reviewed": False,
        "instruction": "核查候选点是否沿同一可见血管特征运动；自动点不是已确认分叉。看不清请保留缺失，不补造坐标。",
        "runs": [
            {
                "run_id": report["run_id"],
                "reviewed": False,
                "reference_frame_index": report.get("reference_frame_index"),
                "review_frame_indices": report.get("review_frame_indices", []),
                "reviewer": None,
                "points": [
                    {
                        "track_id": track["track_id"],
                        "anatomical_label": None,
                        "approved": None,
                        "rejected_frames": [],
                        "corrected_xy_by_frame": {},
                        "notes": "",
                    }
                    for track in report.get("tracks", [])
                ],
            }
            for report in observations
        ],
    }


def _summary(
    preflight: dict[str, Any],
    observations: list[dict[str, Any]],
    report: dict[str, Any],
    synthetic: dict[str, Any],
) -> str:
    lines = [
        "# CT–XA 心脏动态方案校验：首轮结果",
        "",
        "本轮是数据审计、未复核二维运动与合成点运动验证，不是患者真实三维心脏运动重建。",
        "",
        "## 质量门控",
        "",
    ]
    lines.extend(f"- `{name}`：`{value}`" for name, value in preflight.get("gates", {}).items())
    lines.extend([
        "- 本轮未执行真实患者三维非刚性优化；真实三维 TRE 为 null。",
        "- 源数据完整性：" + ("运行前后 SHA256 一致。" if report["source_integrity"]["unchanged"] else "发生变化，结果不可接受。"),
        "",
        "## 二维候选轨迹（需人工复核）",
        "",
        "| 序列 | 状态 | 自动候选点数 | 有效点帧数 / 全部点帧数 |",
        "|---|---|---:|---:|",
    ])
    for observation in observations:
        tracks = observation.get("tracks", [])
        valid = sum(sum(bool(value) for value in track["valid"]) for track in tracks)
        total = sum(len(track["valid"]) for track in tracks)
        lines.append(f"| {observation['run_id']} | {observation['status']} | {len(tracks)} | {valid} / {total} |")
    lines.extend([
        "",
        "有效性来自图像跟踪质控，不等于分叉身份或解剖对应正确。失踪点不补成观测；关键帧 PNG 与 review_template.json 用于复核。",
        "",
        "## 合成实验边界",
        "",
        "synthetic_benchmark.json 比较静态、受限 XY 刚体平移、XY 平移加径向模式，并单独报告支持/留出误差。深度已知是该模式实验的条件，不是算法从单幅 XA 恢复的结果。",
        "",
        "CT 心脏表面若被使用，仅作为施加已知合成运动的静态形状；合成相机不是实际 XA 相机。射线歧义对照显示同一投影可对应不同三维位置。该基准不替代完整 DRR/组织仿真或独立临床验证。",
        "",
        "## 合成结果数值（不是患者真实三维误差）",
        "",
        "| 合成案例 | 模型 | 支持帧 2D RMSE / px | 留出帧 2D RMSE / px | 留出帧合成 3D RMSE / mm |",
        "|---|---|---:|---:|---:|",
    ])
    for case_name, case in synthetic["cases"].items():
        for model_name, model in case["models"].items():
            support = model["metrics"]["support"]
            heldout = model["metrics"]["heldout"]
            lines.append(
                f"| {case_name} | {model_name} | {support['rmse_2d_px']:.4f} | "
                f"{heldout['rmse_2d_px']:.4f} | {heldout['synthetic_rmse_3d_mm']:.4f} |"
            )
    ambiguity = synthetic["depth_ambiguity"]
    lines.extend([
        "",
        f"射线歧义对照：最大投影差 {ambiguity['max_projection_difference_px']:.6g} px，"
        f"但最大三维差 {ambiguity['max_3d_difference_mm']:.4f} mm。该差值由人为构造的射线缩放产生，不是临床误差测量。",
        "",
        "连续中段留出使用支持帧参数插值；留出误差可能高于支持帧，也可能不优于简单基线。保留这种结果，不据此删去困难案例或宣称全心运动已恢复。",
        "",
        "## 下一步",
        "",
        "1. 审查 preflight.json 的缺项与现有标定失败原因。",
        "2. 在关键帧上确认/否定二维轨迹身份，必要时补少量局部心脏可见边缘。",
        "3. 完成 P2 完整投影压力测试；几何与观测支持后，才进入真实低自由度非刚性拟合。",
        "4. 不把旧心脏表面深度先验或新合成三维误差作为真实患者验证依据。",
        "",
    ])
    return "\n".join(lines)


def run_validation(case_dir: Path, assets_root: Path, output_dir: Path) -> dict[str, Any]:
    """Run the first validation batch without changing any input assets."""
    case_dir, assets_root, output_dir = validate_output_paths(case_dir, assets_root, output_dir)
    # Imports are deliberately after validation, so an unsafe request never writes.
    from .observations import build_observation_report
    from .preflight import audit_case
    from .synthetic import run_synthetic_benchmark

    _reject_unauditable_reparse_points(case_dir, assets_root)
    preflight = audit_case(case_dir, assets_root)
    audit_assets = preflight["assets"]
    before = _snapshot_inputs(case_dir, assets_root, audit_assets)
    observations = []
    for run_id, reference_frame in (("series_01", 20), ("series_02", 16)):
        series_dir = assets_root / "all_series" / run_id
        try:
            _require_audited_assets(assets_root, [f"all_series/{run_id}/{suffix}" for suffix in (
                "series_manifest.json", "cine_u8.raw", "temporal/temporal_centerlines_u8.raw",
            )], audit_assets)
            observation = build_observation_report(
                series_dir, reference_frame=reference_frame,
                acquisition_timing=_acquisition_timing(preflight, run_id),
            )
        except (OSError, ValueError, KeyError) as error:
            observation = {
                "run_id": run_id,
                "status": "insufficient",
                "tracks": [],
                "quality_by_frame": [],
                "review_frame_indices": [],
                "limitations": [f"{type(error).__name__}: {error}"],
            }
        observations.append(observation)
    try:
        _require_audited_assets(assets_root, ["thorax_masks/heart.nii.gz"], audit_assets)
    except ValueError as error:
        points, shape_provenance = None, {"kind": "digital_phantom", "reason": str(error)}
    else:
        points, shape_provenance = _heart_shape_points(assets_root / "thorax_masks" / "heart.nii.gz")
    synthetic = run_synthetic_benchmark(points)
    synthetic["static_shape_provenance"] = shape_provenance
    after = _snapshot_inputs(case_dir, assets_root, audit_assets)
    changed = sorted(path for path in set(before) | set(after) if before.get(path) != after.get(path))
    report = {
        "schema_version": 1,
        "case_id": case_dir.name,
        "research_only": True,
        "scope": "P0/P1 automated foundations and P2 constrained point benchmark only",
        "patient_3d_optimization_enabled": False,
        "patient_true_3d_tre_mm": None,
        "clinical_validation": False,
        "gates": preflight.get("gates", {}),
        "source_integrity": {"unchanged": not changed, "file_count": len(before), "changed_paths": changed},
        "source_sha256": before,
        "config": {"seed": 20261017, "series_reference_frames": {"series_01": 20, "series_02": 16}},
        "output_dir": str(output_dir),
    }
    if changed:
        raise RuntimeError("Source inputs changed during validation; no result will be published")
    # Exclusive directory creation protects against a concurrent/repeated run.
    output_dir.mkdir(parents=True, exist_ok=False)
    _write_json(output_dir / "preflight.json", preflight)
    _write_json(output_dir / "synthetic_benchmark.json", synthetic)
    _write_json(output_dir / "review_template.json", _review_template(observations))
    for observation in observations:
        run_id = observation["run_id"]
        _write_json(output_dir / "observations" / f"{run_id}.json", observation)
        if observation.get("reference_frame_index") is not None:
            _require_audited_assets(assets_root, [f"all_series/{run_id}/{suffix}" for suffix in (
                "series_manifest.json", "cine_u8.raw",
            )], audit_assets)
            _render_review_panels(
                assets_root / "all_series" / run_id,
                observation,
                output_dir / "observations" / f"{run_id}_review.png",
            )
    _write_json(output_dir / "code_manifest.json", _code_manifest())
    (output_dir / "summary.md").write_text(_summary(preflight, observations, report, synthetic), encoding="utf-8")
    _write_json(output_dir / "run_report.json", report)
    return report
