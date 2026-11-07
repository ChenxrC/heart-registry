"""Standalone research probe, NOT a differentiable or accepted registration backend.

For an oriented closed triangle mesh, a source-to-detector ray's total length
inside the solid is sum(exit distances) - sum(entry distances). Perspective-
correct triangle rasterization computes those intersections at native pixel
centres without filling gaps between separate solid intervals. All tissues
inside this heart envelope share one explicitly illustrative attenuation.
"""
from __future__ import annotations

import numpy as np
import trimesh

from ctdr_pipeline.heart_mesh_motion.geometry import project_lps
from ._paths import research_roots


def mesh_path_lengths(vertices, faces, camera):
    """Return exact triangle-mesh chord lengths in mm, without shading/blur."""
    surface = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    if not surface.is_watertight or not surface.is_winding_consistent:
        raise ValueError("A closed, consistently oriented mesh is required")
    volume = float(surface.volume)
    if not np.isfinite(volume) or abs(volume) < 1e-8:
        raise ValueError("A nonzero closed solid volume is required")
    uv, z = project_lps(np.asarray(surface.vertices), camera)
    orientation = 1 if volume > 0 else -1
    height, width = camera.image_hw
    signed_depth = np.zeros((height, width), dtype=np.float64)
    balance = np.zeros((height, width), dtype=np.int32)
    crossings = np.zeros((height, width), dtype=np.int32)
    skipped = 0

    def edge(a, b, xx, yy):
        return (b[0] - a[0]) * (yy - a[1]) - (b[1] - a[1]) * (xx - a[0])

    def inclusive_edge(a, b):
        delta = b - a
        return delta[1] < 0 or (delta[1] == 0 and delta[0] > 0)

    for face in surface.faces:
        triangle = uv[face].copy()
        depths = z[face].copy()
        area = float(edge(triangle[0], triangle[1], *triangle[2]))
        if abs(area) < 1e-12:
            skipped += 1
            continue
        contribution = (1 if area > 0 else -1) * orientation
        if area < 0:
            triangle[[1, 2]] = triangle[[2, 1]]
            depths[[1, 2]] = depths[[2, 1]]
            area = -area
        lo = np.maximum(0, np.ceil(triangle.min(0) - 1e-9)).astype(int)
        hi = np.minimum([width - 1, height - 1], np.floor(triangle.max(0) + 1e-9)).astype(int)
        if np.any(lo > hi):
            continue
        xx, yy = np.meshgrid(np.arange(lo[0], hi[0] + 1), np.arange(lo[1], hi[1] + 1))
        pairs = ((triangle[1], triangle[2]), (triangle[2], triangle[0]), (triangle[0], triangle[1]))
        values = [edge(a, b, xx, yy) for a, b in pairs]
        epsilon = max(area, 1) * 1e-10
        # Half-open triangle boundaries count a shared edge only once.
        inside = np.ones(xx.shape, dtype=bool)
        for e, (a, b) in zip(values, pairs):
            inside &= (e > epsilon) | ((np.abs(e) <= epsilon) & inclusive_edge(a, b))
        if not inside.any():
            continue
        inverse_depth = sum(e / area / depth for e, depth in zip(values, depths))
        fragment_depth = 1. / inverse_depth[inside]
        row, col = yy[inside], xx[inside]
        signed_depth[row, col] += contribution * fragment_depth
        balance[row, col] += contribution
        crossings[row, col] += 1

    yy, xx = np.mgrid[:height, :width]
    source_detector_mm = float((camera.detector_center_lps_mm - camera.source_lps_mm) @ camera.n_lps)
    slope_x = (xx - camera.principal_xy_px[0]) * camera.spacing_row_col_mm[1] / source_detector_mm
    slope_y = (yy - camera.principal_xy_px[1]) * camera.spacing_row_col_mm[0] / source_detector_mm
    length = signed_depth * np.sqrt(1 + slope_x ** 2 + slope_y ** 2)
    negative = int(np.count_nonzero(length < -1e-6))
    unbalanced = int(np.count_nonzero(balance))
    if negative or unbalanced:
        raise ValueError(f"Invalid solid intersections: {negative} negative, {unbalanced} unbalanced rays")
    length = np.maximum(length, 0.)
    return {"length_mm": length, "audit": {
        "algorithm": "signed closed-surface intersections; perspective-correct depth; half-open triangle edges",
        "unbalanced_ray_pixels": unbalanced, "negative_length_pixels": negative,
        "maximum_crossings_per_ray": int(crossings.max()),
        "grazing_projected_triangles_skipped": skipped,
        "input_winding_volume_sign": orientation, "mesh_volume_mm3": abs(volume),
        "projection_image_hw": [height, width], "pixel_centres": "integer native column,row",
        "non_differentiable_probe": True,
    }}


def beer_lambert(length_mm, *, mu_per_mm):
    length = np.asarray(length_mm, dtype=float)
    if not np.isfinite(length).all() or np.any(length < 0) or not np.isfinite(mu_per_mm) or mu_per_mm < 0:
        raise ValueError("Finite nonnegative path length and attenuation required")
    absorbance = length * mu_per_mm
    return {"absorbance": absorbance, "transmission": np.exp(-absorbance),
            "opacity": -np.expm1(-absorbance)}


def visible_boundary(length_mm):
    from scipy import ndimage
    mask = np.asarray(length_mm) > 1e-6
    boundary = mask & ~ndimage.binary_erosion(mask, border_value=1)
    boundary[[0, -1]] = False
    boundary[:, [0, -1]] = False
    return boundary


def encode_length_map(length_mm):
    import base64
    length = np.asarray(length_mm)
    if not np.isfinite(length).all() or np.any(length < 0) or np.any(length > 655.35):
        raise ValueError("Viewer path length must fit an unsigned 0.01 mm integer")
    return base64.b64encode(np.rint(length * 100).astype('<u2').tobytes()).decode('ascii')


def independent_ray_lengths(vertices, faces, camera, pixels_yx):
    """Independent Moller-Trumbore intersections in 3D, for numerical audit only."""
    triangles = np.asarray(vertices)[np.asarray(faces)]
    a = triangles[:, 0]
    edge1, edge2 = triangles[:, 1] - a, triangles[:, 2] - a
    start = camera.source_lps_mm
    offset = start - a
    q = np.cross(offset, edge1)
    result = []
    for row, col in pixels_yx:
        detector = (camera.detector_center_lps_mm
                    + (col - camera.principal_xy_px[0]) * camera.spacing_row_col_mm[1] * camera.u_lps
                    + (row - camera.principal_xy_px[1]) * camera.spacing_row_col_mm[0] * camera.v_lps)
        direction = detector - start
        detector_distance = np.linalg.norm(direction)
        direction /= detector_distance
        p = np.cross(direction, edge2)
        determinant = np.einsum('ij,ij->i', edge1, p)
        valid = np.abs(determinant) > 1e-12
        inv = np.zeros_like(determinant)
        inv[valid] = 1 / determinant[valid]
        u = np.einsum('ij,ij->i', offset, p) * inv
        v = q @ direction * inv
        distance = np.einsum('ij,ij->i', edge2, q) * inv
        valid &= (u >= -1e-10) & (v >= -1e-10) & (u + v <= 1 + 1e-10)
        valid &= (distance >= 0) & (distance <= detector_distance)
        hits = np.unique(np.round(distance[valid], 8))
        if len(hits) % 2:
            raise ValueError("Independent check encountered an unmatched/grazing intersection")
        result.append(float((hits[1::2] - hits[::2]).sum()))
    return np.asarray(result)


def _png_url(array):
    import base64
    import io
    from PIL import Image
    stream = io.BytesIO()
    Image.fromarray(array).save(stream, format='PNG')
    return 'data:image/png;base64,' + base64.b64encode(stream.getvalue()).decode('ascii')


def _comparison(cine, records, output_path, mu, scale_mm, run_id, frame_index):
    from PIL import Image, ImageDraw, ImageFont
    font = ImageFont.truetype('C:/Windows/Fonts/msyh.ttc', 17)
    small = ImageFont.truetype('C:/Windows/Fonts/msyh.ttc', 14)
    sheet = Image.new('RGB', (1024, 732), '#12202b')
    draw = ImageDraw.Draw(sheet)
    draw.text((14, 8), f'{run_id} · CT心脏闭合mesh的透视投影 · 原生512×512', font=font, fill='white')
    draw.text((14, 35), '科研诊断：两套相机均未验收；心脏静止，不是动态重建。青色线=模型轮廓，不是实测心缘。', font=small, fill='#ffcc76')
    for j, title in enumerate(('原始XA（保留组织背景）', f'穿行厚度 0–{scale_mm:.0f} mm', f'透射率 exp(-μL), μ={mu:.3f}/mm', '模型轮廓叠加（非误差图）')):
        draw.text((j * 256 + 8, 64), title, font=small, fill='white')
    for row, item in enumerate(records):
        length = item['length_mm']
        frame = cine[frame_index]
        transmission = beer_lambert(length, mu_per_mm=mu)['transmission']
        overlay = np.repeat(frame[..., None], 3, axis=-1)
        overlay[visible_boundary(length)] = [35, 235, 230]
        panels = (frame, np.rint(np.clip(length / scale_mm, 0, 1) * 255).astype('uint8'),
                  np.rint(transmission * 255).astype('uint8'), overlay)
        top = 96 + row * 295
        name = '继承的初始相机' if item['camera_id'] == 'inherited' else '此前2DOF骨性搜索候选（未通过解剖验收）'
        draw.text((12, top), f'{name} · 心脏覆盖视野 {item["report"]["fov_coverage_fraction"]:.1%}', font=small, fill='white')
        for col, panel in enumerate(panels):
            sheet.paste(Image.fromarray(panel).convert('RGB').resize((256, 256), Image.Resampling.LANCZOS), (col * 256, top + 25))
    draw.text((12, 691), 'μ仅用于均匀组织演示，未按设备/能谱标定；未模拟肺、肋骨、血液及造影剂。', font=small, fill='#ffcc76')
    draw.text((12, 713), f'XA帧号：{frame_index}（从0开始）；不能用全图灰度差作为心缘或三维配准误差。', font=small, fill='white')
    sheet.save(output_path)


def run_probe(output_dir):
    import argparse
    import io
    import json
    import sys
    import time
    from pathlib import Path
    import pydicom
    from PIL import Image
    from ctdr_pipeline.heart_mesh_motion.contracts import CameraData
    from ctdr_pipeline.heart_mesh_motion.geometry import load_camera
    from ctdr_pipeline.heart_mesh_motion.inputs import (
        freeze_inputs, read_bytes_snapshot, read_json_snapshot, sha256,
        assert_unchanged, validate_output, reserve_output, write_json,
    )
    from ctdr_pipeline.heart_mesh_motion.reference_raster import render_hard

    root, case_id = research_roots()
    case = root / 'cases' / case_id
    assets = root / 'artifacts/xa_dynamic_vessel' / case_id
    audit = root / 'artifacts/heart_motion_validation' / case_id / 'run_002/preflight.json'
    previous = root / 'artifacts/heart_mesh_dynamic' / case_id / 'static_preflight'
    output = Path(output_dir)
    validate_output([case, assets, audit.parent, previous], output)
    frozen = freeze_inputs(case, assets, audit)
    snapshots = {Path(p): value for p, value in frozen['sha256'].items()}
    old_report, digest = read_json_snapshot(previous / 'static_preflight.json')
    snapshots[previous / 'static_preflight.json'] = digest
    for path, value in old_report['source_sha256'].items():
        if Path(path) in snapshots and snapshots[Path(path)] != value:
            raise ValueError('Previous preflight was computed from different inputs')
    mesh_bytes, digest = read_bytes_snapshot(previous / 'mesh_optimization.npz')
    snapshots[previous / 'mesh_optimization.npz'] = digest
    with np.load(io.BytesIO(mesh_bytes), allow_pickle=False) as archive:
        vertices, faces = archive['vertices_lps_mm'], archive['faces']
        if str(archive['source_sha256'].item()) != snapshots[assets / 'thorax_masks/heart.nii.gz']:
            raise ValueError('Mesh does not identify the frozen CT-heart segmentation')
    mu = .010  # Illustrative uniform tissue, NOT a scanner-calibrated coefficient.
    records, series, report_series, payload = [], {}, [], {'series': [], 'projections': []}
    for run_id, reference_frame in (('series_01', 20), ('series_02', 21)):
        folder = assets / 'all_series' / run_id
        metadata, _ = read_json_snapshot(folder / 'series_manifest.json', expected_sha256=snapshots[folder / 'series_manifest.json'])
        raw, _ = read_bytes_snapshot(folder / 'cine_u8.raw', expected_sha256=snapshots[folder / 'cine_u8.raw'])
        shape = metadata['shape_frame_row_column']
        if metadata['dtype'] != 'uint8' or len(raw) != int(np.prod(shape)):
            raise ValueError('Cine shape or dtype mismatch')
        cine = np.frombuffer(raw, dtype='uint8').reshape(shape)
        series[run_id] = cine
        link = next(item['source_link'] for item in frozen['runs'] if item['run_id'] == run_id)
        source_path = case / link['source_relative_path']
        source_bytes, _ = read_bytes_snapshot(source_path, expected_sha256=snapshots[source_path])
        header = pydicom.dcmread(io.BytesIO(source_bytes), stop_before_pixels=True)
        image_type = [str(s) for s in getattr(header, 'ImageType', [])]
        subtraction = bool(getattr(header, 'MaskSubtractionSequence', None))
        inherited = load_camera(folder / f'{run_id}_calibration.json', image_hw=tuple(shape[1:]),
                                expected_sha256=snapshots[folder / f'{run_id}_calibration.json'])
        candidate_path = previous / f'{run_id}_camera_in_ct.json'
        candidate_dict, digest = read_json_snapshot(candidate_path)
        snapshots[candidate_path] = digest
        candidate_dict['image_hw'] = tuple(candidate_dict['image_hw'])
        candidate = CameraData(**candidate_dict)
        old_series = next(s for s in old_report['series'] if s['run_id'] == run_id)
        offset = np.asarray(old_series['ct_to_xa_lps_matrix'])[:3, 3]
        if not (np.allclose(candidate.source_lps_mm, inherited.source_lps_mm - offset)
                and np.allclose(candidate.detector_center_lps_mm, inherited.detector_center_lps_mm - offset)):
            raise ValueError('Candidate camera does not match the recorded rigid transform')
        # Do not certify subtraction solely from visual appearance or missing tags.
        report_series.append({'run_id': run_id, 'reference_frame_index': reference_frame,
                              'source_image_type': image_type, 'has_mask_subtraction_sequence': subtraction,
                              'interpretation': 'XA cine retaining anatomical background; not assumed to be a subtraction image'})
        payload['series'].append({'run_id': run_id, 'reference_frame': reference_frame,
                                  'frame_duration_ms': metadata.get('median_frame_time_ms', 133.333),
                                  'frames': [_png_url(frame) for frame in cine]})
        for camera_id, camera in (('inherited', inherited), ('static_candidate', candidate)):
            print(f'Projecting {run_id}/{camera_id} at {shape[1]}x{shape[2]}...', flush=True)
            start = time.perf_counter()
            projection = mesh_path_lengths(vertices, faces, camera)
            length = projection['length_mm']
            hard = render_hard(vertices, faces, camera, image_hw=camera.image_hw)
            mask = length > 1e-6
            rng = np.random.default_rng(20261017)
            pixels = []
            for region in (mask, ~mask):
                yyxx = np.argwhere(region)
                if len(yyxx):
                    pixels.extend(yyxx[rng.choice(len(yyxx), min(48, len(yyxx)), replace=False)].tolist())
            independent = independent_ray_lengths(vertices, faces, camera, pixels)
            tested = length[tuple(np.asarray(pixels).T)]
            error = float(np.max(np.abs(independent - tested)))
            if error > 1e-4:
                raise ValueError(f'Independent 3D ray check failed: {error} mm')
            statistics = {**projection['audit'], 'run_id': run_id, 'camera_id': camera_id,
                          'maximum_path_length_mm': float(length.max()),
                          'fov_coverage_fraction': float(mask.mean()),
                          'mask_touches_fov_border': bool(mask[0].any() or mask[-1].any() or mask[:, 0].any() or mask[:, -1].any()),
                          'mesh_silhouette_iou_not_xa_agreement': float((mask & hard).sum() / max(1, (mask | hard).sum())),
                          'independent_3d_ray_check_count': len(pixels),
                          'independent_3d_ray_max_difference_mm': error,
                          'elapsed_seconds': time.perf_counter() - start,
                          'absolute_geometry_accepted': False, 'measured_xa_heart_boundary_error_px': None}
            item = {'run_id': run_id, 'camera_id': camera_id, 'length_mm': length,
                    'camera': camera.to_dict(), 'report': statistics, 'reference_frame': reference_frame}
            records.append(item)
            payload['projections'].append({'run_id': run_id, 'camera_id': camera_id,
                                           'length_base64_u16_le': encode_length_map(length), 'report': statistics})
    assert_unchanged(snapshots)
    output = reserve_output([case, assets, audit.parent, previous], output)
    scale_mm = float(np.ceil(max(r['length_mm'].max() for r in records) / 20) * 20)
    payload.update({'image_hw': [512, 512], 'default_mu_per_mm': mu, 'thickness_display_max_mm': scale_mm})
    for item in records:
        prefix = item['run_id'] + '_' + item['camera_id']
        physical = beer_lambert(item['length_mm'], mu_per_mm=mu)
        np.savez_compressed(output / f'{prefix}_projection.npz', length_mm=item['length_mm'].astype('float32'),
                            absorbance=physical['absorbance'].astype('float32'),
                            transmission=physical['transmission'].astype('float32'), mu_per_mm=np.asarray(mu))
        write_json(output / f'{prefix}_camera.json', item['camera'])
        Image.fromarray(np.rint(255 * physical['transmission']).astype('uint8')).save(output / f'{prefix}_transmission.png')
        length = item['length_mm']
        overlay = np.repeat(series[item['run_id']][item['reference_frame']][..., None], 3, axis=-1)
        overlay[visible_boundary(length)] = [35, 235, 230]
        Image.fromarray(overlay).save(output / f'{prefix}_overlay.png')
    for run_id, cine in series.items():
        selected = [r for r in records if r['run_id'] == run_id]
        _comparison(cine, selected, output / f'{run_id}_comparison.png', mu, scale_mm, run_id, selected[0]['reference_frame'])
    (output / 'data.js').write_text('window.HEART_PROJECTION_DATA=' + json.dumps(payload, ensure_ascii=False, allow_nan=False) + ';\n', encoding='utf-8')
    template = Path(__file__).with_name('beer_lambert_viewer.html')
    (output / 'index.html').write_bytes(template.read_bytes())
    report = {'schema_version': 1, 'status': 'research_static_mesh_projection_visualization_only',
              'case_id': case.name, 'mesh_vertices': len(vertices), 'mesh_faces': len(faces),
              'equation': 'I/I0=exp(-mu*sum(length_inside_closed_mesh_intervals_mm))',
              'uniform_mu_per_mm': mu, 'mu_is_scanner_calibrated': False,
              'series': report_series, 'projections': [r['report'] for r in records],
              'source_sha256': {str(p): value for p, value in snapshots.items()},
              'source_integrity_unchanged': True, 'absolute_geometry_accepted': False,
              'real_patient_mesh_fit_executed': False, 'clinical_validation': False,
              'patient_true_3d_tre_mm': None, 'measured_xa_heart_boundary_error_px': None,
              'old_artifacts_overwritten': False,
              'limitations': ['Uniform whole-heart envelope attenuation, not chamber-specific blood/myocardium.',
                             'No ribs/lung/diaphragm/iodine/scatter/spectrum/detector postprocessing.',
                             'Two inherited/candidate cameras, neither independently accepted.',
                             'XA plays but CT mesh is static. This is NOT reconstructed cardiac motion.',
                             'No annotated XA heart boundary; image differences are not registration accuracy.',
                             'Standalone non-differentiable probe; not integrated into the optimization backend.']}
    write_json(output / 'report.json', report)
    summary = ['# CT心脏 Beer–Lambert 投影：静态可视化验证', '',
               '本次复用既有20,000面闭合心脏mesh；逐原生像素计算射线在mesh内的总穿行长度，'
               '不是表面着色，也不将第一交点到最后交点之间的空隙全部填满。', '',
               '按 `I/I0=exp(-μL)` 生成均匀心脏包络透射率；μ=0.010/mm是演示参数，并非设备标定。'
               '图中没有肺、肋骨、膈肌、血液/心肌分区或造影剂，因此不能把全图灰度残差当配准误差。', '',
               '**两套相机均未通过独立解剖验证。** inherited为旧S7转移初始化；static_candidate为此前仅u/v平移搜索候选，'
               '不是已接受修正。两者原样展示，本轮未挑选/拟合新的位置。', '',
               '| 序列 | 相机 | 视野内模型覆盖 | 最大穿行长度(mm) | 独立射线计算最大差(mm) |',
               '|---|---|---:|---:|---:|']
    for r in records:
        s = r['report']
        summary.append(f'| {r["run_id"]} | {r["camera_id"]} | {s["fov_coverage_fraction"]:.2%} | '
                       f'{s["maximum_path_length_mm"]:.3f} | {s["independent_3d_ray_max_difference_mm"]:.3g} |')
    summary += ['', '上述覆盖率只描述投影模型占据多少画面，不代表与XA心脏的重合率。'
                '独立射线检查只验证计算器的一致性，不证明患者相机或分割正确。', '',
                '## 本次图像检查所见', '',
                'S1/S2中，已有相机下的心脏投影主要位于画面下部并被视野裁剪；此前2DOF候选并未给出'
                '可独立确认的心缘吻合，模型进入画面的比例反而下降。覆盖率下降本身不是错误证明，'
                '但这些结果不足以支持相机已正确配准。应先核对静态位姿和CT/XA解剖范围，'
                '不能通过改变μ或非刚性拉伸来掩盖该问题。本轮没有用图像观感选择新相机。', '',
                '## 查看', '', '`index.html` 可切换S1/S2、继承/候选相机、XA帧、μ与叠加透明度。'
                '改变μ只改变灰度，不改变几何轮廓；播放只播放原始XA，心脏mesh投影保持静态。', '',
                '青色线是模型边缘，不是自动识别或医生标注的XA心缘；裁剪边界不画成心缘。'
                '目前无可信心缘标签，故不报告心缘误差、真实三维TRE或临床通过。', '',
                '## 复现', '', '在项目根运行：', '', '```powershell',
                '& experiments/heart_mesh_dynamic/.venv/Scripts/python.exe -m '
                'experiments.heart_mesh_dynamic.probes.beer_lambert_visualization --output-dir '
                'artifacts/heart_mesh_dynamic/<case-key>/beer_lambert_visualization', '```', '',
                '这是独立实验探针，未接入正式可微优化/旧3D页面；代码和输入摘要保存在report.json与checkpoint.json。']
    (output / 'summary.md').write_text('\n'.join(summary) + '\n', encoding='utf-8')
    code_paths = [Path(__file__), template]
    write_json(output / 'checkpoint.json', {'python': sys.version, 'numpy': np.__version__,
               'trimesh': trimesh.__version__, 'code_sha256': {str(p): sha256(p) for p in code_paths},
               'git': 'workspace is not a Git repository; no commit or worktree created',
               'old_data_written': False})
    assert_unchanged(snapshots)
    print(json.dumps({'output': str(output), 'projection_count': len(records), 'source_integrity_unchanged': True}, ensure_ascii=False))
    return report


if __name__ == '__main__':
    import argparse
    from pathlib import Path
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    run_probe(parser.parse_args().output_dir)
