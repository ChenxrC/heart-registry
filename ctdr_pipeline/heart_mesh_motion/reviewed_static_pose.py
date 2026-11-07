"""Research-only 3DOF rigid correction driven exclusively by reviewed fit bones.

No nonrigid deformation, intrinsic/depth/scale fitting, holdout selection or
automatic geometry acceptance. Native hard-raster boundaries define the metric.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import numpy as np
from scipy import ndimage
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation

from .reference_raster import render_hard


DEFAULT_CONFIG = {'resolutions': [128, 256], 'max_evaluations_per_level': 240,
                  'translation_bound_mm': 20., 'rotation_bound_deg': 10.}


def inplane_matrix(camera, parameters, pivot_lps_mm):
    p = np.asarray(parameters, dtype=float)
    pivot = np.asarray(pivot_lps_mm, dtype=float)
    if p.shape != (3,) or pivot.shape != (3,) or not np.isfinite(p).all() or not np.isfinite(pivot).all():
        raise ValueError('Three finite pose parameters and LPS pivot required')
    rotation = Rotation.from_rotvec(np.deg2rad(p[2]) * camera.n_lps).as_matrix()
    translation = p[0] * camera.u_lps + p[1] * camera.v_lps
    matrix = np.eye(4)
    matrix[:3,:3] = rotation
    matrix[:3,3] = pivot - rotation @ pivot + translation
    return matrix


def camera_in_original_ct(camera, matrix):
    m = np.asarray(matrix, dtype=float)
    if m.shape != (4,4) or not np.isfinite(m).all() or not np.allclose(m[3], [0,0,0,1]):
        raise ValueError('Finite rigid homogeneous matrix required')
    r = m[:3,:3]
    if (not np.allclose(r.T@r, np.eye(3), atol=1e-8) or not np.isclose(np.linalg.det(r), 1)
            or not np.allclose(r@camera.n_lps, camera.n_lps, atol=1e-8)
            or abs(m[:3,3]@camera.n_lps) > 1e-6):
        raise ValueError('Only in-plane rigid motion is allowed; scale/depth are locked')
    inverse = np.linalg.inv(m)
    point = lambda value: (inverse @ np.r_[value, 1.])[:3]
    return replace(camera, source_lps_mm=point(camera.source_lps_mm),
                   detector_center_lps_mm=point(camera.detector_center_lps_mm),
                   u_lps=r.T@camera.u_lps, v_lps=r.T@camera.v_lps, n_lps=r.T@camera.n_lps,
                   provenance={**camera.provenance, 'reviewed_bone_fit': '3DOF conditional research candidate; unaccepted'},
                   geometry_accepted=False)


def training_observations(observations, *, run_id):
    selected = [deepcopy(o) for o in observations if o.get('run_id') == run_id
                and o.get('structure_id') in ('T3','T5') and o.get('split') == 'fit'
                and o.get('review_state') == 'confirmed_visible_edge' and o.get('identity_confirmed') is True]
    if len({o['structure_id'] for o in selected}) != len(selected):
        raise ValueError('Duplicate fit structure')
    return sorted(selected, key=lambda o: o['structure_id'])


def sample_curve(xy):
    xy = np.asarray(xy, dtype=float)
    if xy.ndim != 2 or xy.shape[1] != 2 or len(xy) < 2 or not np.isfinite(xy).all():
        raise ValueError('Finite native-pixel polyline required')
    lengths = np.linalg.norm(np.diff(xy,axis=0),axis=1)
    if lengths.sum() < 1:
        raise ValueError('Nontrivial visible edge required')
    return np.concatenate([np.linspace(a,b,int(np.ceil(d))+1)[:-1]
                           for a,b,d in zip(xy[:-1],xy[1:],lengths) if d > 0] + [xy[-1:]])


def model_edge(mask):
    result = mask & ~ndimage.binary_erosion(mask, border_value=1)
    result[[0,-1]] = False
    result[:,[0,-1]] = False
    return result


def visible_edge_loss(edge, native_curve, *, native_hw):
    edge = np.asarray(edge, dtype=bool)
    curve = np.asarray(native_curve, dtype=float)
    native = np.asarray(native_hw, dtype=float)
    if (edge.ndim != 2 or min(edge.shape)<2 or native.shape!=(2,) or np.any(native<=0)
            or not np.isfinite(native).all() or curve.ndim!=2 or curve.shape[1]!=2
            or not len(curve) or not np.isfinite(curve).all()
            or np.any(curve<0) or np.any(curve[:,0]>native[1]-1) or np.any(curve[:,1]>native[0]-1)):
        raise ValueError('Valid edge and native-pixel observations required')
    if not edge.any():
        return float(np.linalg.norm(native))
    sy, sx = np.asarray(edge.shape)/native
    resized_xy = (curve + .5)*[sx,sy]-.5
    # Distance remains in native pixels even with anisotropic resizing.
    distance = ndimage.distance_transform_edt(~edge, sampling=(1/sy,1/sx))
    return float(ndimage.map_coordinates(distance, resized_xy[:,::-1].T, order=1, mode='nearest').mean())


def fit_inplane_pose(meshes, camera, fit_records, pivot_lps_mm, *, config=None):
    if any(o.get('split') != 'fit' or o.get('structure_id') not in ('T3','T5') for o in fit_records):
        raise ValueError('Only fit observations may cross the optimizer boundary')
    base = {'absolute_geometry_accepted':False, 'clinical_validation':False,
            'patient_true_3d_tre_mm':None, 'holdout_consumed':False,
            'optimization_attempted':False,
            'optimized_dofs':['translation_u_mm','translation_v_mm','rotation_n_deg'],
            'locked_dofs':['translation_n','rotation_u','rotation_v','intrinsics','scale','heart_shape'],
            'objective':'equal-structure mean reviewed-visible-edge to same CT whole-vertebra boundary; native pixels',
            'final_parameters':None, 'final_fit_loss_native_px':None}
    if len(fit_records)<2 or len({o['structure_id'] for o in fit_records})!=len(fit_records):
        return {**base, 'status':'insufficient_fit_observations'}
    if any(meshes.get(o['structure_id']) is None for o in fit_records):
        return {**base, 'status':'insufficient_fit_geometry'}
    cfg = deepcopy(DEFAULT_CONFIG)
    if config is not None:
        if set(config)-set(cfg):
            raise ValueError('Unknown fitting configuration')
        cfg.update(deepcopy(config))
    if (not cfg['resolutions'] or any(type(r) is not int or r<16 for r in cfg['resolutions'])
            or type(cfg['max_evaluations_per_level']) is not int or cfg['max_evaluations_per_level']<1
            or not np.isfinite([cfg['translation_bound_mm'],cfg['rotation_bound_deg']]).all()
            or min(cfg['translation_bound_mm'],cfg['rotation_bound_deg'])<=0):
        raise ValueError('Invalid bounded fitting configuration')
    curves = {o['structure_id']:sample_curve(o['xy_px']) for o in fit_records}
    bounds = np.array([cfg['translation_bound_mm']]*2+[cfg['rotation_bound_deg']],dtype=float)
    history, stages = [], []
    h,w = camera.image_hw

    def loss(parameters, image_hw):
        if not np.isfinite(parameters).all() or np.any(np.abs(parameters)>bounds+1e-8):
            return float(np.hypot(h,w)*2)
        corrected = camera_in_original_ct(camera,inplane_matrix(camera,parameters,pivot_lps_mm))
        scores=[]
        for name, curve in curves.items():
            vertices, faces, _ = meshes[name]
            edge = model_edge(render_hard(vertices,faces,corrected,image_hw=image_hw))
            if not edge.any():
                return float(np.hypot(h,w)*2)  # search penalty, never a measured error
            scores.append(visible_edge_loss(edge,curve,native_hw=camera.image_hw))
        return float(np.mean(scores))

    def missing_native_projections(p):
        corrected=camera_in_original_ct(camera,inplane_matrix(camera,p,pivot_lps_mm))
        return [name for name in curves if not model_edge(render_hard(
            meshes[name][0],meshes[name][1],corrected,image_hw=camera.image_hw)).any()]

    zero = np.zeros(3)
    initial = loss(zero,camera.image_hw)
    parameters=zero.copy()
    candidates=[zero.copy()]
    for resolution in cfg['resolutions']:
        image_hw=(resolution,max(16,int(round(w*resolution/h))))
        best=[loss(parameters,image_hw),parameters.copy()]
        count=0
        def objective(p):
            nonlocal count
            value=loss(p,image_hw);count+=1
            history.append({'resolution':resolution,'evaluation':count,'parameters':np.asarray(p).tolist(),'fit_loss_native_px':value})
            if value<best[0]:best[:]=[value,np.array(p).copy()]
            return value
        print(f'Fit only {list(curves)}: {image_hw}, max {cfg["max_evaluations_per_level"]} evaluations',flush=True)
        result=minimize(objective,parameters,method='Powell',bounds=[(-b,b) for b in bounds],
                        options={'maxfev':cfg['max_evaluations_per_level'],'xtol':.02,'ftol':.0005})
        parameters=best[1]
        candidates.append(parameters.copy())
        stages.append({'resolution':resolution,'evaluations':count,'optimizer_success':bool(result.success),
                       'message':str(result.message),'parameters':parameters.tolist(),'fit_loss_native_px':best[0]})
    # Native-resolution selection uses fit data only, including the frozen zero baseline.
    native_scores=[loss(p,camera.image_hw) for p in candidates]
    chosen=int(np.argmin(native_scores));parameters=candidates[chosen]
    failed=missing_native_projections(parameters)
    initial_failed=missing_native_projections(zero)
    if failed:
        return {**base,'status':'fit_projection_failure','optimization_attempted':True,
                'failed_fit_structures':failed,'attempted_parameters':parameters.tolist(),
                'initial_fit_loss_native_px':initial if not initial_failed else None,
                'initial_training_penalty_native_px':initial,'final_training_penalty_native_px':native_scores[chosen],
                'config':cfg,'stages':stages,'history':history}
    matrix=inplane_matrix(camera,parameters,pivot_lps_mm)
    at_bound=bool(np.any(np.abs(parameters)>=bounds*.99))
    return {**base, 'status':'research_candidate_at_search_boundary' if at_bound else 'research_candidate_requires_review',
            'config':cfg, 'fit_structure_ids':list(curves), 'pivot_lps_mm':np.asarray(pivot_lps_mm).tolist(),
            'optimization_attempted':True,'failed_fit_structures':[],
            'initial_fit_loss_native_px':initial if not initial_failed else None,
            'initial_training_penalty_native_px':initial,'final_fit_loss_native_px':native_scores[chosen],
            'final_parameters':parameters.tolist(), 'ct_to_base_camera_frame_lps_matrix':matrix.tolist(),
            'at_search_boundary':at_bound, 'all_optimizer_stages_converged':all(s['optimizer_success'] for s in stages),
            'native_fit_only_candidate_scores':native_scores,'selected_fit_only_candidate_index':chosen,
            'stages':stages,'history':history}
