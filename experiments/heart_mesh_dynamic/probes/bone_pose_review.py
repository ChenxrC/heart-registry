"""Bone-only static pose diagnostic and human review pack; never auto-accepts geometry."""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import hashlib
import io
import json
import tempfile

import numpy as np
from scipy import ndimage

from ctdr_pipeline.heart_mesh_motion.bone_projection import render_bone, load_bone_volume
from ctdr_pipeline.heart_mesh_motion.contracts import CameraData
from ctdr_pipeline.heart_mesh_motion.geometry import load_camera, project_lps
from ctdr_pipeline.heart_mesh_motion.inputs import (
    read_bytes_snapshot, read_json_snapshot, freeze_inputs, sha256,
    validate_output, reserve_output, write_json, assert_unchanged,
)
from ctdr_pipeline.heart_mesh_motion.reference_raster import render_hard
from .beer_lambert_visualization import _png_url, visible_boundary
from ._paths import research_roots


def render_tiled(volume, camera, *, rows_per_tile=16, samples=384):
    """Reuse tested LPS ray marcher without allocating a full native 3D ray grid."""
    import torch
    if type(rows_per_tile) is not int or rows_per_tile < 1:
        raise ValueError('Positive row tile budget required')
    h, w = camera.image_hw
    result = np.empty((h, w), np.float32)
    with torch.no_grad():
        for first in range(0, h, rows_per_tile):
            last = min(first + rows_per_tile, h)
            tile = replace(camera, image_hw=(last - first, w),
                           principal_xy_px=camera.principal_xy_px - [0, first])
            result[first:last] = render_bone(volume, tile, torch.zeros(2),
                                            image_hw=tile.image_hw, samples=samples).numpy()
    return result


def context_camera(camera, *, scale=.5, margin=128):
    """Expand FOV, not a new pose: original pixel centres retain explicit mapping."""
    if not np.isfinite(scale) or scale <= 0 or type(margin) is not int or margin < 0:
        raise ValueError('Invalid context size')
    h, w = camera.image_hw
    new = replace(camera, image_hw=(int(np.ceil(h * scale)) + 2 * margin,
                                    int(np.ceil(w * scale)) + 2 * margin),
                  principal_xy_px=(camera.principal_xy_px + .5) * scale - .5 + margin,
                  spacing_row_col_mm=camera.spacing_row_col_mm / scale)
    return new, [margin - .5, margin - .5, margin - .5 + w * scale, margin - .5 + h * scale]


def check_mask_geometry(ct, mask):
    if ct.GetSize() != mask.GetSize() or any(
        not np.allclose(a, b, atol=1e-6, rtol=0) for a, b in (
            (ct.GetSpacing(), mask.GetSpacing()), (ct.GetOrigin(), mask.GetOrigin()),
            (ct.GetDirection(), mask.GetDirection()))):
        raise ValueError('CT/segmentation physical geometry mismatch')


def check_candidate(inherited, candidate, matrix):
    m = np.asarray(matrix, dtype=float)
    if (m.shape != (4, 4) or not np.isfinite(m).all() or not np.allclose(m[:3, :3], np.eye(3))
            or not np.allclose(m[3], [0, 0, 0, 1]) or abs(m[:3, 3] @ inherited.n_lps) > 1e-5):
        raise ValueError('Candidate is not detector-plane translation only')
    locked = ('u_lps', 'v_lps', 'n_lps', 'principal_xy_px', 'spacing_row_col_mm', 'image_hw')
    if (inherited.geometry_accepted or candidate.geometry_accepted
            or any(not np.allclose(getattr(inherited, k), getattr(candidate, k), atol=1e-7, rtol=0) for k in locked)
            or not np.allclose(candidate.source_lps_mm, inherited.source_lps_mm - m[:3, 3])
            or not np.allclose(candidate.detector_center_lps_mm, inherited.detector_center_lps_mm - m[:3, 3])):
        raise ValueError('Unaccepted 2DOF candidate changed locked camera parameters')


def verify_ppt_reference(deck_bytes, slide_index, expected_sha256):
    import zipfile
    import posixpath
    from xml.etree import ElementTree as ET
    with zipfile.ZipFile(io.BytesIO(deck_bytes)) as archive:
        slide = ET.fromstring(archive.read(f'ppt/slides/slide{slide_index}.xml'))
        used = {n.attrib.get('{http://schemas.openxmlformats.org/officeDocument/2006/relationships}embed')
                for n in slide.iter('{http://schemas.openxmlformats.org/drawingml/2006/main}blip')}
        relations = ET.fromstring(archive.read(f'ppt/slides/_rels/slide{slide_index}.xml.rels'))
        for relation in relations:
            if (relation.attrib.get('Id') in used and relation.attrib.get('Type', '').endswith('/image')
                    and relation.attrib.get('TargetMode') != 'External'):
                member = posixpath.normpath(posixpath.join('ppt/slides', relation.attrib['Target']))
                if member.startswith('ppt/media/') and hashlib.sha256(archive.read(member)).hexdigest() == expected_sha256:
                    return {'slide': slide_index, 'media_member': member, 'sha256': expected_sha256}
    raise ValueError('PPT slide media does not match extracted reference image')


def validate_review(review, pack):
    if (review.get('schema_version') != 1 or review.get('pack_id') != pack['pack_id']
            or review.get('case_id') != pack['case_id']):
        raise ValueError('Review does not identify this frozen pack')
    if review.get('display_material_sha256') != pack.get('display_material_sha256'):
        raise ValueError('Actual browser display material differs from canonical pack')
    runs = {r['run_id']: r for r in pack['series']}
    labels = {s['id']: s['split'] for s in pack['structures']}
    observations = review.get('observations')
    if not isinstance(observations, list):
        raise ValueError('Observation list required')
    used = set()
    for item in observations:
        run = runs.get(item.get('run_id'))
        key = item.get('run_id'), item.get('structure_id')
        if key in used:
            raise ValueError('Duplicate structure observation')
        used.add(key)
        if (run is None or item.get('structure_id') not in labels
                or item.get('split') != labels[item['structure_id']]
                or type(item.get('frame_index')) is not int or item['frame_index'] != run['frame_index']):
            raise ValueError('Run, frame or frozen split mismatch')
        state = item.get('review_state')
        if state not in ('confirmed_visible_edge', 'unknown', 'identity_uncertain'):
            raise ValueError('Unknown review state')
        xy = np.asarray(item.get('xy_px', []), dtype=float)
        if state != 'confirmed_visible_edge':
            if xy.size:
                raise ValueError('Unknown anatomy must not supply target coordinates')
            continue
        if not isinstance(review.get('reviewer'), str) or not review['reviewer'].strip() or item.get('identity_confirmed') is not True:
            raise ValueError('Independent reviewer and explicit anatomical identity required')
        if (xy.ndim != 2 or xy.shape[1] != 2 or len(xy) < 2 or not np.isfinite(xy).all()
                or np.any(xy < 0) or np.any(xy[:, 0] > run['image_hw'][1] - 1)
                or np.any(xy[:, 1] > run['image_hw'][0] - 1)
                or np.linalg.norm(np.diff(xy, axis=0), axis=1).sum() < 1):
            raise ValueError('Finite nontrivial native-pixel visible edge required')
    return observations


def evaluate_review(review, pack, model_edges):
    """Partial observed edges -> model edges, not full Dice/TRE or an acceptance rule."""
    observations = validate_review(review, pack)
    records, distances = [], []
    for item in observations:
        if item['review_state'] != 'confirmed_visible_edge':
            continue
        key = item['run_id'], item['structure_id']
        edge = model_edges.get(key)
        expected_hw = next(r['image_hw'] for r in pack['series'] if r['run_id'] == item['run_id'])
        if edge is not None and np.asarray(edge).shape != tuple(expected_hw):
            raise ValueError('Model edge shape differs from native image')
        result = {k: item[k] for k in ('run_id', 'structure_id', 'split', 'frame_index')}
        if edge is None or not np.any(edge):
            result.update(status='model_structure_outside_fov_or_no_edge', mean_px=None, p95_px=None)
        else:
            xy = np.asarray(item['xy_px'], dtype=float)
            curve = np.concatenate([np.linspace(a, b, int(np.ceil(np.linalg.norm(b - a))) + 1)[:-1]
                                    for a, b in zip(xy[:-1], xy[1:]) if np.linalg.norm(b - a) > 0] + [xy[-1:]])
            dt = ndimage.distance_transform_edt(~np.asarray(edge, bool))
            values = ndimage.map_coordinates(dt, curve[:, ::-1].T, order=1)
            distances.extend(values.tolist())
            result.update(status='descriptive_only', mean_px=float(values.mean()),
                          p95_px=float(np.percentile(values, 95)), sample_count=len(values))
        records.append(result)
    total = len(pack['structures']) * len(pack['series'])
    failed = sum(r['mean_px'] is None for r in records)
    split_summary = {}
    for split in ('fit', 'holdout'):
        subset = [r for r in records if r['split'] == split]
        means = [r['mean_px'] for r in subset if r['mean_px'] is not None]
        failures = len(subset) - len(means)
        split_summary[split] = {'reviewed_count': len(subset), 'failed_count': failures,
                                'mean_per_structure_px': float(np.mean(means)) if means and not failures else None,
                                'success_conditional_mean_per_structure_px': float(np.mean(means)) if means else None}
    return {'status': ('reviewed_projection_failures' if failed else 'descriptive_review_only') if records else 'insufficient_reviewed_observations',
            'per_structure': records, 'split_summary': split_summary, 'expected_structure_count': total,
            'reviewed_visible_structure_count': len(records), 'missing_structure_count': total - len(records),
            'failed_structure_count': failed,
            'mean_visible_edge_to_model_px': float(np.mean(distances)) if distances and not failed else None,
            'p95_visible_edge_to_model_px': float(np.percentile(distances, 95)) if distances and not failed else None,
            'success_conditional_mean_visible_edge_to_model_px': float(np.mean(distances)) if distances else None,
            'success_conditional_p95_visible_edge_to_model_px': float(np.percentile(distances, 95)) if distances else None,
            'metric': 'native-pixel directed distance from reviewed visible bone arc to same labelled CT whole-vertebra projected edge; not symmetric full-silhouette error',
            'fit_executed': False, 'absolute_geometry_accepted': False, 'clinical_validation': False,
            'patient_true_3d_tre_mm': None}


def verify_materials(pack, manifest, edge_bytes):
    material = {k: v for k, v in pack.items() if k not in ('pack_id', 'display_material_sha256')}
    digest = lambda value: hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()
    if (digest(manifest) != pack['pack_id'] or digest(material) != manifest['review_material_sha256']
            or pack['display_material_sha256'] != manifest['review_material_sha256']
            or hashlib.sha256(edge_bytes).hexdigest() != manifest['model_edges_sha256']):
        raise ValueError('Review pack/model edge integrity mismatch')


def _snapshot_image(path, snapshots):
    import SimpleITK as sitk
    data, digest = read_bytes_snapshot(path)
    snapshots[path] = digest
    scratch = Path(__file__).resolve().parents[3] / 'tmp'
    scratch.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='bone_review_', dir=scratch) as folder:
        snapshot = Path(folder) / 'image.nii.gz'
        snapshot.write_bytes(data)
        return sitk.ReadImage(str(snapshot))


def _mask_mesh(image):
    import SimpleITK as sitk
    from skimage.measure import marching_cubes
    array = sitk.GetArrayFromImage(image)
    indices = np.argwhere(array > 0)
    if not len(indices):
        return None
    low, high = indices.min(0), indices.max(0) + 1
    cropped = np.pad((array[tuple(slice(a, b) for a, b in zip(low, high))] > 0).astype('uint8'), 2)
    try:
        vertices, faces, _, _ = marching_cubes(cropped, .5, step_size=2, allow_degenerate=False)
    except RuntimeError:
        # Thin nonempty masks can fall entirely between the step-two sample lattice.
        vertices, faces, _, _ = marching_cubes(cropped, .5, step_size=1, allow_degenerate=False)
    ijk = (vertices + low - 2)[:, ::-1]
    direction = np.asarray(image.GetDirection()).reshape(3, 3)
    lps = (ijk * image.GetSpacing()) @ direction.T + image.GetOrigin()
    centroid = (indices[:, ::-1].mean(0) * image.GetSpacing()) @ direction.T + image.GetOrigin()
    return lps, faces, centroid


def _outline_map(meshes, camera):
    edges, reports = {}, []
    for name, surface in meshes.items():
        if surface is None:
            edges[name] = np.zeros(camera.image_hw, dtype=bool)
            reports.append({'structure_id': name, 'model_state': 'empty_segmentation',
                            'projected_volume_centroid_guide_xy': None,
                            'in_fov_pixels': 0, 'clipped_or_outside_fov': True})
            continue
        vertices, faces, centroid = surface
        mask = render_hard(vertices, faces, camera, image_hw=camera.image_hw)
        uv, _ = project_lps(vertices, camera)
        guide, _ = project_lps(centroid[None], camera)
        edges[name] = visible_boundary(mask.astype(float))
        reports.append({'structure_id': name, 'model_state': 'available', 'projected_volume_centroid_guide_xy': guide[0].tolist(),
                        'guide_is_not_2d_region_centroid_or_correspondence': True,
                        'in_fov_pixels': int(mask.sum()),
                        'clipped_or_outside_fov': bool((uv < -.5).any() or (uv[:, 0] > camera.image_hw[1] - .5).any()
                                                     or (uv[:, 1] > camera.image_hw[0] - .5).any())})
    return edges, reports


def run_pack(output_dir):
    import torch
    import SimpleITK as sitk
    from PIL import Image, ImageDraw, ImageFont

    torch.set_num_threads(4)
    root, case_id = research_roots()
    case = root / 'cases' / case_id
    assets = root / 'artifacts/xa_dynamic_vessel' / case_id
    annotation = root / 'artifacts/annotation_registration' / case_id
    depth = annotation / 'totalsegmentator_depth'
    ppt_path = root / '标注-OKOK-pure.pptx'
    previous = root / 'artifacts/heart_mesh_dynamic' / case_id / 'static_preflight'
    audit = root / 'artifacts/heart_motion_validation' / case_id / 'run_002/preflight.json'
    inputs = [case, assets, annotation, previous, audit.parent, ppt_path]
    validate_output(inputs, output_dir)
    frozen = freeze_inputs(case, assets, audit)
    snapshots = {Path(k): v for k, v in frozen['sha256'].items()}
    previous_report, digest = read_json_snapshot(previous / 'static_preflight.json')
    snapshots[previous / 'static_preflight.json'] = digest
    ct_path = depth / 'ct_series4_native_geometry.nii.gz'
    ct = _snapshot_image(ct_path, snapshots)
    if previous_report['source_sha256'].get(str(ct_path)) != snapshots[ct_path]:
        raise ValueError('CT identity differs from previous static preflight')
    volume = load_bone_volume(ct_path, expected_sha256=snapshots[ct_path], sampling_mm=1.5)
    meshes = {}
    for index in range(1, 13):
        name = f'T{index}'
        mask = _snapshot_image(depth / f'vertebrae_masks/vertebrae_{name}.nii.gz', snapshots)
        check_mask_geometry(ct, mask)
        surface = _mask_mesh(mask)
        meshes[name] = surface
    lineage, digest = read_json_snapshot(root / 'configs/annotation_landmarks.json')
    snapshots[root / 'configs/annotation_landmarks.json'] = digest
    workbook, digest = read_json_snapshot(annotation / 'workbook_data.json')
    snapshots[annotation / 'workbook_data.json'] = digest
    pair = next(p for p in workbook['pair_index'] if p['case_id'] == case.name and p['pair_index'] == 1)
    ppt_bytes, digest = read_bytes_snapshot(ppt_path)
    snapshots[ppt_path] = digest
    references = []
    for role, slide_index, name in (('source', 1, 'slide_0001_source.png'), ('target', 2, 'slide_0002_target.png')):
        path = root / 'artifacts/annotation_registration/extracted_images' / case_id / name
        data, digest = read_bytes_snapshot(path)
        if digest != pair[f'{role}_image_sha256'] or pair[f'{role}_slide'] != slide_index:
            raise ValueError('PPT extraction lineage differs from existing workbook')
        verified = verify_ppt_reference(ppt_bytes, slide_index, digest)
        snapshots[path] = digest
        references.append({'name': name, 'ppt_lineage': verified, 'url': _png_url(np.asarray(Image.open(io.BytesIO(data)).convert('RGB')))})
    payload = {'case_id': case.name, 'schema_version': 1,
               'structures': [{'id': f'T{i}', 'split': 'fit' if i % 2 else 'holdout'} for i in range(1, 13)],
               'series': [], 'projections': [], 'references': references,
               'label_scope': 'CT whole vertebrae (including posterior elements), NOT isolated vertebral bodies',
               'ppt_scope': 'Existing slide1/2 landmarks refer to scout and S7, not new S1/S2 labels'}
    arrays, panels, statistics = {}, {}, []
    palette = [(255, 199, 77), (101, 225, 222), (181, 161, 255), (255, 151, 158)]
    font = ImageFont.truetype('C:/Windows/Fonts/msyh.ttc', 14)
    for run_id, frame_index in (('series_01', 20), ('series_02', 21)):
        folder = assets / 'all_series' / run_id
        metadata, _ = read_json_snapshot(folder / 'series_manifest.json', expected_sha256=snapshots[folder / 'series_manifest.json'])
        raw, _ = read_bytes_snapshot(folder / 'cine_u8.raw', expected_sha256=snapshots[folder / 'cine_u8.raw'])
        shape = metadata['shape_frame_row_column']
        if metadata['dtype'] != 'uint8' or len(raw) != int(np.prod(shape)):
            raise ValueError('Cine shape/dtype mismatch')
        cine = np.frombuffer(raw, dtype='uint8').reshape(shape)
        payload['series'].append({'run_id': run_id, 'frame_index': frame_index, 'image_hw': list(shape[1:]),
                                  'xa': _png_url(cine[frame_index]), 'frame_count': shape[0]})
        inherited = load_camera(folder / f'{run_id}_calibration.json', image_hw=tuple(shape[1:]),
                                expected_sha256=snapshots[folder / f'{run_id}_calibration.json'])
        candidate_dict, digest = read_json_snapshot(previous / f'{run_id}_camera_in_ct.json')
        snapshots[previous / f'{run_id}_camera_in_ct.json'] = digest
        candidate_dict['image_hw'] = tuple(candidate_dict['image_hw'])
        candidate = CameraData(**candidate_dict)
        matrix = np.asarray(next(s for s in previous_report['series'] if s['run_id'] == run_id)['ct_to_xa_lps_matrix'])
        check_candidate(inherited, candidate, matrix)
        for camera_id, camera in (('inherited', inherited), ('static_candidate', candidate)):
            print(f'{run_id}/{camera_id}: native512 bone DRR + T1-T12 outlines', flush=True)
            drr = render_tiled(volume, camera)
            edges, bone_reports = _outline_map(meshes, camera)
            context, box = context_camera(camera, scale=.375, margin=96)
            context_edges, context_reports = _outline_map(meshes, context)
            guide_image = Image.new('RGB', (context.image_hw[1], context.image_hw[0]), '#111c26')
            guide = np.array(guide_image)
            color = np.zeros((*camera.image_hw, 4), dtype='uint8')
            for i, name in enumerate(meshes):
                rgb = palette[i % len(palette)]
                guide[context_edges[name]] = rgb
                color[edges[name]] = [*rgb, 255]
                arrays[f'{run_id}__{camera_id}__{name}'] = edges[name]
            guide_image = Image.fromarray(guide)
            draw = ImageDraw.Draw(guide_image)
            draw.rectangle(box, outline='white', width=2)
            for i, report in enumerate(context_reports):
                if report['projected_volume_centroid_guide_xy'] is None:
                    continue
                x, y = report['projected_volume_centroid_guide_xy']
                if 0 <= x < context.image_hw[1] and 0 <= y < context.image_hw[0]:
                    draw.text((x + 3, y), report['structure_id'], font=font, fill=palette[i % len(palette)], stroke_width=1, stroke_fill='#111c26')
            item = {'run_id': run_id, 'camera_id': camera_id, 'camera': camera.to_dict(),
                    'bone_reports': bone_reports, 'drr': drr, 'guide': color, 'context_image': guide_image,
                    'bone_proxy_max_mm': float(drr.max()), 'absolute_geometry_accepted': False}
            statistics.append(item)
            panels[(run_id, camera_id)] = cine[frame_index]
    assert_unchanged(snapshots)
    # A common display scale across all four maps, never per-camera auto-contrast selection.
    nonzero = np.concatenate([s['drr'][s['drr'] > 0] for s in statistics])
    scale = float(np.percentile(nonzero, 99))
    for item in statistics:
        item['drr_image'] = np.rint(255 * (1 - np.clip(item['drr'] / scale, 0, 1))).astype('uint8')
        payload['projections'].append({k: item[k] for k in ('run_id', 'camera_id', 'bone_reports')})
        payload['projections'][-1].update(drr=_png_url(item['drr_image']), guide=_png_url(item['guide']),
                                         context=_png_url(np.array(item['context_image'])))
    edge_stream = io.BytesIO()
    np.savez_compressed(edge_stream, **arrays)
    edge_bytes = edge_stream.getvalue()
    manifest = {'source_sha256': {str(p): h for p, h in snapshots.items()},
                'model_edges_sha256': hashlib.sha256(edge_bytes).hexdigest(),
                'review_material_sha256': hashlib.sha256(json.dumps(payload, sort_keys=True, allow_nan=False).encode()).hexdigest(),
                'series': [{k: v for k, v in s.items() if k != 'xa'} for s in payload['series']],
                'structures': payload['structures'],
                'camera_parameters': [s['camera'] for s in statistics],
                'projection': {'sampling_mm': 1.5, 'ray_samples': 384, 'mask_marching_cubes_step': '2; fallback 1 if thin mask has no sampled isosurface'},
                'code_sha256': {str(p): sha256(p) for p in (Path(__file__), Path(__file__).with_name('bone_pose_review.html'))}}
    material_bytes = json.dumps(payload, sort_keys=True, allow_nan=False).encode()
    payload['pack_id'] = hashlib.sha256(json.dumps(manifest, sort_keys=True, allow_nan=False).encode()).hexdigest()
    payload['display_material_sha256'] = manifest['review_material_sha256']
    output = reserve_output(inputs, output_dir)
    (output / 'model_edges.npz').write_bytes(edge_bytes)
    write_json(output / 'pack.json', payload)
    write_json(output / 'manifest.json', manifest)
    review_template = {k: payload[k] for k in ('schema_version', 'case_id', 'pack_id', 'display_material_sha256')}
    review_template.update(reviewer='', observations=[])
    write_json(output / 'review_template.json', review_template)
    import base64
    (output / 'data.js').write_text('window.BONE_REVIEW_B64=' + json.dumps(base64.b64encode(material_bytes).decode('ascii'))
        + ';window.BONE_REVIEW_PACK_ID=' + json.dumps(payload['pack_id']) + ';\n', encoding='utf-8')
    (output / 'index.html').write_bytes(Path(__file__).with_name('bone_pose_review.html').read_bytes())
    font_big = ImageFont.truetype('C:/Windows/Fonts/msyh.ttc', 18)
    for run_id in ('series_01', 'series_02'):
        sheet = Image.new('RGB', (1536, 1168), '#14212b')
        draw = ImageDraw.Draw(sheet)
        draw.text((12, 8), f'{run_id} 骨性位姿复核：原生512像素；两套相机均未验收', font=font_big, fill='white')
        draw.text((12, 40), '左：原始XA　中：CT骨密度代理DRR（不是定量透射率）　右：CT逐椎骨轮廓叠加；彩线不是XA标注', font=font_big, fill='#ffcc76')
        for row, item in enumerate(s for s in statistics if s['run_id'] == run_id):
            y = 80 + row * 544
            draw.text((12, y), item['camera_id'], font=font_big, fill='white')
            xa = panels[(run_id, item['camera_id'])]
            overlay = np.repeat(xa[..., None], 3, axis=-1)
            mask = item['guide'][..., 3] > 0
            overlay[mask] = item['guide'][mask, :3]
            for col, image in enumerate((xa, item['drr_image'], overlay)):
                sheet.paste(Image.fromarray(image).convert('RGB'), (512 * col, y + 26))
            Image.fromarray(overlay).save(output / f'{run_id}_{item["camera_id"]}_overlay.png')
            item['context_image'].save(output / f'{run_id}_{item["camera_id"]}_context.png')
            np.savez_compressed(output / f'{run_id}_{item["camera_id"]}_bone_proxy.npz', bone_proxy_mm=item['drr'])
        sheet.save(output / f'{run_id}_comparison.png')
    report = {'case_id': case.name, 'pack_id': payload['pack_id'], 'status': 'bone_pose_review_prepared_not_calibrated',
              'projection': manifest['projection'], 'common_display_scale': scale,
              'density_definition': volume.report['density_definition'], 'quantitative_attenuation': False,
              'bone_geometry': [{k: s[k] for k in ('run_id', 'camera_id', 'bone_reports', 'bone_proxy_max_mm')} for s in statistics],
              'existing_ppt_landmarks_reused_as_reference_only': True,
              'ppt_slide_media_identity_verified': True,
              'new_xa_landmarks_inferred_from_ppt': False, 'source_integrity_unchanged': True,
              'fit_executed': False, 'absolute_geometry_accepted': False, 'clinical_validation': False,
              'patient_true_3d_tre_mm': None, 'measured_xa_bone_error_px': None,
              'empty_review_evaluation': evaluate_review(review_template, payload, {})}
    write_json(output / 'report.json', report)
    write_json(output / 'checkpoint.json', {'source_integrity_unchanged': True, **manifest,
               'test_sha256': sha256(Path(__file__).with_name('test_bone_pose_review.py')),
               'git': 'not a Git repository; no commit/worktree created'})
    assert_unchanged(snapshots)
    print(json.dumps({'output': str(output), 'pack_id': payload['pack_id'], 'geometry_accepted': False}))
    return report


def run_evaluation(pack_dir, review_path, output_dir):
    """Read-only evaluation of all cameras, without fitting or selecting a winner."""
    pack_dir, review_path = Path(pack_dir), Path(review_path)
    inputs = [pack_dir, review_path]
    validate_output(inputs, output_dir)
    pack, pack_hash = read_json_snapshot(pack_dir / 'pack.json')
    review, review_hash = read_json_snapshot(review_path)
    manifest, manifest_hash = read_json_snapshot(pack_dir / 'manifest.json')
    edge_bytes, edge_hash = read_bytes_snapshot(pack_dir / 'model_edges.npz')
    verify_materials(pack, manifest, edge_bytes)
    results = {}
    with np.load(io.BytesIO(edge_bytes), allow_pickle=False) as archive:
        for camera_id in ('inherited', 'static_candidate'):
            edges = {(s['run_id'], st['id']): archive[f'{s["run_id"]}__{camera_id}__{st["id"]}']
                     for s in pack['series'] for st in pack['structures']}
            results[camera_id] = evaluate_review(review, pack, edges)
    snapshots = {pack_dir / 'pack.json': pack_hash, pack_dir / 'manifest.json': manifest_hash,
                 pack_dir / 'model_edges.npz': edge_hash, review_path: review_hash}
    assert_unchanged(snapshots)
    output = reserve_output(inputs, output_dir)
    write_json(output / 'evaluation.json', {'pack_id': pack['pack_id'], 'camera_results': results,
               'input_sha256': {str(p): h for p, h in snapshots.items()},
               'candidate_selection_executed': False, 'absolute_geometry_accepted': False})
    assert_unchanged(snapshots)
    return results


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--pack-dir', type=Path)
    parser.add_argument('--review-json', type=Path)
    args = parser.parse_args()
    if bool(args.pack_dir) != bool(args.review_json):
        parser.error('--pack-dir and --review-json must be supplied together')
    if args.pack_dir:
        run_evaluation(args.pack_dir, args.review_json, args.output_dir)
    else:
        run_pack(args.output_dir)
