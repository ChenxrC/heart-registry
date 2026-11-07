"""Explicitly conditional, non-accepted S7 pose experiments on manual open arcs.

Independent from the legacy anatomical-identity gate. This module cannot accept
anatomy/absolute geometry, change splits, or fit depth, scale or heart shape.
"""
from __future__ import annotations

from copy import deepcopy
import re

import numpy as np
from scipy import ndimage
from scipy.optimize import OptimizeResult, minimize

from .bone_correspondence_review import validate_correspondence_review
from .geometry import project_lps
from .reference_raster import render_hard
from .reviewed_static_pose import camera_in_original_ct, inplane_matrix, model_edge, sample_curve


DEFAULT_CONFIG = dict(resolutions=[128,256], max_evaluations_per_level=160,
                      translation_bound_mm=60., rotation_bound_deg=10.,
                      initial_rotations_deg=[0.,-5.,5.])


def conditional_records(evidence, material, *, thoracic_offset=0):
    validated = validate_correspondence_review(evidence, material)
    if type(thoracic_offset) is not int or thoracic_offset not in (-1,0,1):
        raise ValueError('Only the declared neighboring thoracic hypotheses are supported')
    sources = material['source'].get('regions', [])
    roles = {}
    for source in sources:
        rid, role = source.get('region_id'), source.get('source_split')
        if rid not in material['region_ids'] or rid in roles or role not in ('fit','holdout'):
            raise ValueError('Unambiguous original source roles are required; never reassign splits')
        roles[rid] = role
    if set(roles) != set(material['region_ids']):
        raise ValueError('Original roles missing')
    records = []
    for obs in validated['observations']:
        level = obs['anatomical_level']
        if obs['review_state'] != 'same_region_visible' or level is None:
            continue
        if re.fullmatch(r'T(?:[1-9]|1[0-2])', level) is None:
            raise ValueError('This bounded experiment only supports thoracic-level hypotheses')
        index = int(level[1:]) + thoracic_offset
        if not 1 <= index <= 12:
            raise ValueError('Neighbor hypothesis falls outside the thoracic CT labels')
        rid = obs['source_region_id']
        records.append(dict(region_id=rid, structure_id=f'T{index}', role=roles[rid],
                            claimed_level=level, thoracic_offset=thoracic_offset,
                            arcs_xy_px=deepcopy(obs['target_arcs_xy_px']),
                            anatomical_identity_verified=False,
                            independent_blinded_truth=False))
    return records


def sample_record(record):
    arcs = record.get('arcs_xy_px')
    if not isinstance(arcs,list) or not arcs:
        raise ValueError('Actual separate observed arcs are required')
    return np.concatenate([sample_curve(arc) for arc in arcs])


def evaluate_records(meshes, camera, records, *, image_hw=None):
    hw = tuple(image_hw or camera.image_hw)
    native = np.asarray(camera.image_hw, dtype=float)
    sy, sx = np.asarray(hw) / native
    rows = []
    for record in records:
        row = dict(region_id=record['region_id'], structure_id=record['structure_id'],
                   role=record['role'], mean_observed_arc_distance_px=None,
                   p95_observed_arc_distance_px=None, model_state='missing_segmentation',
                   clipped_or_outside_fov=None)
        curve = sample_record(record)
        if (np.any(curve < 0) or np.any(curve[:,0] > native[1]-1)
                or np.any(curve[:,1] > native[0]-1)):
            raise ValueError('Observed arcs must be inside the native image')
        mesh = meshes.get(record['structure_id'])
        if mesh is not None:
            vertices, faces, _ = mesh
            if np.any((np.asarray(vertices)-camera.source_lps_mm)@camera.n_lps <= 1e-8):
                row.update(model_state='invalid_source_plane_projection',
                           failure_reason='vertices on or behind source plane')
                rows.append(row)
                continue
            uv, _ = project_lps(vertices, camera)
            edge = model_edge(render_hard(vertices, faces, camera, image_hw=hw))
            row['clipped_or_outside_fov'] = bool(
                (uv < -.5).any() or (uv[:,0] > native[1]-.5).any() or (uv[:,1] > native[0]-.5).any())
            row['projected_vertex_bounds_xy_px'] = [uv.min(0).tolist(), uv.max(0).tolist()]
            row['projected_boundary_pixels'] = int(edge.sum())
            if edge.any():
                distance = ndimage.distance_transform_edt(~edge, sampling=(1/sy,1/sx))
                resized = (curve + .5)*[sx,sy]-.5
                values = ndimage.map_coordinates(distance,resized[:,::-1].T,order=1,mode='nearest')
                row.update(model_state='projected', mean_observed_arc_distance_px=float(values.mean()),
                           p95_observed_arc_distance_px=float(np.quantile(values,.95)))
            else:
                row['model_state'] = 'no_visible_model_boundary'
        rows.append(row)
    failures = sum(r['mean_observed_arc_distance_px'] is None for r in rows)
    successes = [r['mean_observed_arc_distance_px'] for r in rows if r['mean_observed_arc_distance_px'] is not None]
    return dict(rows=rows, expected_region_count=len(records), failure_count=failures,
                mean_per_region_px=float(np.mean(successes)) if successes and not failures else None,
                success_conditional_mean_px=float(np.mean(successes)) if successes else None,
                metric='directed observed open-arc to conditional CT-body projection; native pixels, NOT 3D TRE')


def fit_conditional_pose(meshes, camera, fit_records, pivot_lps_mm, *,
                         allow_unverified_hypothesis=False, config=None):
    if allow_unverified_hypothesis is not True:
        raise ValueError('Explicit conditional research authorization required')
    if not fit_records or any(r.get('role') != 'fit' for r in fit_records):
        raise ValueError('Only original fit records may cross the optimizer boundary')
    if len({r['region_id'] for r in fit_records}) != len(fit_records):
        raise ValueError('Duplicate fit region')
    cfg = deepcopy(DEFAULT_CONFIG)
    if config is not None:
        if set(config)-set(cfg):
            raise ValueError('Unknown fitting configuration; scale/depth cannot be fitted')
        cfg.update(deepcopy(config))
    if (not cfg['resolutions'] or any(type(r) is not int or r < 16 for r in cfg['resolutions'])
            or type(cfg['max_evaluations_per_level']) is not int or cfg['max_evaluations_per_level'] < 1
            or not np.isfinite([cfg['translation_bound_mm'],cfg['rotation_bound_deg']]).all()
            or min(cfg['translation_bound_mm'],cfg['rotation_bound_deg']) <= 0
            or not isinstance(cfg['initial_rotations_deg'],list) or not cfg['initial_rotations_deg']
            or not np.isfinite(cfg['initial_rotations_deg']).all()
            or any(abs(r) > cfg['rotation_bound_deg'] for r in cfg['initial_rotations_deg'])):
        raise ValueError('Invalid fixed bounded search configuration')
    for r in fit_records:
        sample_record(r)
    bounds = np.asarray([cfg['translation_bound_mm']]*2+[cfg['rotation_bound_deg']],dtype=float)
    penalty = float(np.hypot(*camera.image_hw)*2)
    base = dict(anatomical_identity_verified=False, absolute_geometry_accepted=False,
                clinical_validation=False, patient_true_3d_tre_mm=None,
                holdout_consumed=False, independent_blinded_truth=False,
                conditional_research_fit_executed=True, production_rigid_refit_allowed=False,
                heart_deformation_executed=False, optimized_dofs=['translation_u_mm','translation_v_mm','rotation_n_deg'],
                locked_dofs=['translation_n','rotation_u','rotation_v','intrinsics','scale','heart_shape'],
                fit_region_ids=[r['region_id'] for r in fit_records],
                fit_structure_ids=[r['structure_id'] for r in fit_records],
                single_structure_fit=len(fit_records)==1, config=cfg,
                pivot_lps_mm=None if pivot_lps_mm is None else np.asarray(pivot_lps_mm,dtype=float).tolist(),
                final_parameters=None, final_fit_loss_native_px=None)
    if any(meshes.get(r['structure_id']) is None for r in fit_records):
        return {**base,'status':'insufficient_fit_geometry','conditional_research_fit_executed':False,
                'initial_fit_loss_native_px':None,'stages':[],'history':[],
                'all_optimizer_stages_converged':False,
                'initial_fit_evaluation':evaluate_records(meshes,camera,fit_records)}
    inplane_matrix(camera,[0,0,0],pivot_lps_mm)

    def evaluate(parameters, hw):
        corrected = camera_in_original_ct(camera,inplane_matrix(camera,parameters,pivot_lps_mm))
        return evaluate_records(meshes,corrected,fit_records,image_hw=hw)

    def score(parameters, hw):
        value = evaluate(parameters,hw)['mean_per_region_px']
        return penalty if value is None else value

    zero = np.zeros(3)
    initial = evaluate(zero,camera.image_hw)
    candidates = [zero.copy()]
    history, stages = [], []
    for start_index, roll in enumerate(cfg['initial_rotations_deg']):
        parameters = np.asarray([0.,0.,roll])
        for resolution in cfg['resolutions']:
            hw = (resolution,max(16,int(round(camera.image_hw[1]*resolution/camera.image_hw[0]))))
            best = [float('inf'),parameters.copy()]
            count = 0

            def objective(p):
                nonlocal count
                if not np.isfinite(p).all() or np.any(np.abs(p) > bounds+1e-8):
                    return penalty
                value=score(p,hw);count+=1
                history.append(dict(start=start_index,resolution=resolution,evaluation=count,
                                    parameters=np.asarray(p).tolist(),search_objective_native_px=value))
                if value < best[0]:
                    best[:] = [value,np.asarray(p).copy()]
                return value

            print(f'Conditional fit {base["fit_structure_ids"]}: start{start_index}, {hw}, max{cfg["max_evaluations_per_level"]}',flush=True)
            objective(parameters)  # Initial level score counts within the frozen cap.
            remaining=cfg['max_evaluations_per_level']-1
            result = (minimize(objective,parameters,method='Powell',bounds=[(-b,b) for b in bounds],
                               options=dict(maxfev=remaining,xtol=.02,ftol=.0005)) if remaining else
                      OptimizeResult(success=False,message='Level budget consumed by initial evaluation'))
            parameters=best[1];candidates.append(parameters.copy())
            stages.append(dict(start=start_index,resolution=resolution,evaluations=count,
                               optimizer_success=bool(result.success),message=str(result.message),
                               parameters=parameters.tolist(),search_objective_native_px=best[0]))
    native = [score(p,camera.image_hw) for p in candidates]
    chosen = int(np.argmin(native));parameters=candidates[chosen]
    final = evaluate(parameters,camera.image_hw)
    diagnostics = dict(stages=stages,history=history,initial_fit_loss_native_px=initial['mean_per_region_px'],
                       native_fit_only_candidate_scores=native,selected_fit_only_candidate_index=chosen,
                       initial_fit_evaluation=initial,final_fit_evaluation=final,
                       all_optimizer_stages_converged=all(s['optimizer_success'] for s in stages),
                       per_start_final_parameters=[s['parameters'] for s in stages if s['resolution']==cfg['resolutions'][-1]])
    if final['mean_per_region_px'] is None:
        return {**base,**diagnostics,'status':'fit_projection_failure'}
    matrix=inplane_matrix(camera,parameters,pivot_lps_mm)
    at_bound=bool(np.any(np.abs(parameters) >= .99*bounds))
    return {**base,**diagnostics,'status':'conditional_candidate_at_search_boundary' if at_bound else 'conditional_candidate_requires_review',
            'final_parameters':parameters.tolist(),'final_fit_loss_native_px':final['mean_per_region_px'],
            'ct_to_base_camera_frame_lps_matrix':matrix.tolist(),'at_search_boundary':at_bound}
