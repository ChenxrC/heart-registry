"""Conditional pose must not promote labels, leak check regions, or invent edges."""
from copy import deepcopy
from dataclasses import replace
import importlib

import numpy as np
import pytest
import trimesh

from ctdr_pipeline.heart_mesh_motion.contracts import CameraData
from tests.heart_mesh_motion.test_bone_correspondence_review import material, observation, review


def api():
    return importlib.import_module('ctdr_pipeline.heart_mesh_motion.conditional_bone_pose')


def inputs():
    pack = material()
    pack['source']['regions'] = [dict(region_id=r, source_split=s) for r,s in
                                [('R1','holdout'),('R2','fit'),('R3','holdout')]]
    value = review()
    value['observations'] = [observation(r,l) for r,l in [('R1','T7'),('R2','T8'),('R3','T9')]]
    return value, pack


def camera():
    return CameraData([0,0,0],[1,0,0],[0,1,0],[0,0,1],[0,0,100],
                      [40,40],[1,1],(81,81),{},False)


def record():
    return dict(region_id='R2', structure_id='T8', role='fit',
                arcs_xy_px=[[[36,30],[48,30],[48,50],[36,50]]])


def test_adapter_preserves_original_roles_and_actual_disjoint_arcs_without_promotion():
    value,pack=inputs();before=deepcopy((value,pack))
    result=api().conditional_records(value,pack)
    assert [(r['region_id'],r['structure_id'],r['role']) for r in result]==[
        ('R1','T7','holdout'),('R2','T8','fit'),('R3','T9','holdout')]
    assert result[0]['arcs_xy_px']==value['observations'][0]['target_arcs_xy_px']
    assert all(r['anatomical_identity_verified'] is False for r in result)
    assert (value,pack)==before


def test_unknown_level_is_not_silently_assigned_from_order():
    value,pack=inputs();value['observations'][0].update(anatomical_level=None,level_confirmed_by_reviewer=False)
    assert [r['region_id'] for r in api().conditional_records(value,pack)]==['R2','R3']


def test_neighbor_sensitivity_changes_labels_not_data_or_original_roles():
    value,pack=inputs()
    result=api().conditional_records(value,pack,thoracic_offset=-1)
    assert [r['structure_id'] for r in result]==['T6','T7','T8']
    assert result[1]['arcs_xy_px']==value['observations'][1]['target_arcs_xy_px']
    assert result[1]['role']=='fit'


@pytest.mark.parametrize('mutation', ['digest','role','duplicate_source'])
def test_adapter_rejects_wrong_lineage_or_ambiguous_legacy_roles(mutation):
    value,pack=inputs()
    if mutation=='digest':value['display_material_sha256']='d'*64
    elif mutation=='role':pack['source']['regions'][1]['source_split']=None
    else:pack['source']['regions'].append(deepcopy(pack['source']['regions'][0]))
    with pytest.raises(ValueError):api().conditional_records(value,pack)


def test_sampling_never_connects_separate_visible_arcs_or_closes_polyline():
    item=record();item['arcs_xy_px']=[[[20,10],[20,20]],[[20,50],[20,60]]]
    points=api().sample_record(item)
    assert not np.any((points[:,1]>20)&(points[:,1]<50))
    assert len(points)==22


def test_optimizer_rejects_holdout_and_requires_explicit_conditional_mode():
    with pytest.raises(ValueError,match='conditional'):
        api().fit_conditional_pose({},camera(),[record()],[0,0,40])
    item=record();item['role']='holdout'
    with pytest.raises(ValueError,match='fit'):
        api().fit_conditional_pose({},camera(),[item],[0,0,40],allow_unverified_hypothesis=True)


def test_synthetic_physical_translation_reduces_observed_distance_without_promoting_pose():
    from skimage.measure import find_contours
    from ctdr_pipeline.heart_mesh_motion.reference_raster import render_hard
    cam=camera();mesh=trimesh.creation.box(extents=[8,12,4]);mesh.apply_translation([0,0,40])
    # Independently created target: a physical +2,-1.5 mm translation, not the optimizer helper.
    moved=mesh.vertices+np.array([2,-1.5,0])
    mask=render_hard(moved,mesh.faces,cam,image_hw=cam.image_hw)
    curve=max(find_contours(mask.astype(float),1-1e-6),key=len)[:,::-1].tolist()
    item=record();item['arcs_xy_px']=[curve]
    result=api().fit_conditional_pose({'T8':(mesh.vertices,mesh.faces,np.array([0,0,40]))},
        cam,[item],[0,0,40],allow_unverified_hypothesis=True,
        config=dict(resolutions=[81],max_evaluations_per_level=180,
                    translation_bound_mm=4.,rotation_bound_deg=8.,initial_rotations_deg=[0.]))
    assert result['final_fit_loss_native_px'] < result['initial_fit_loss_native_px']*.4
    np.testing.assert_allclose(result['final_parameters'][:2],[2,-1.5],atol=.8)
    assert result['anatomical_identity_verified'] is False
    assert result['absolute_geometry_accepted'] is False
    assert result['clinical_validation'] is False
    assert result['patient_true_3d_tre_mm'] is None
    assert result['holdout_consumed'] is False
    assert result['fit_region_ids']==['R2']
    matrix=np.asarray(result['ct_to_base_camera_frame_lps_matrix'])
    np.testing.assert_allclose(matrix[:3,:3].T@matrix[:3,:3],np.eye(3),atol=1e-10)
    assert np.linalg.det(matrix[:3,:3])==pytest.approx(1)


def test_missing_or_offscreen_projection_is_failure_not_zero_error():
    item=record();cam=camera()
    result=api().evaluate_records({},cam,[item])
    assert result['mean_per_region_px'] is None
    assert result['failure_count']==1
    assert result['rows'][0]['mean_observed_arc_distance_px'] is None
    mesh=trimesh.creation.box(extents=[2,2,2]);mesh.apply_translation([1000,1000,40])
    fitted=api().fit_conditional_pose({'T8':(mesh.vertices,mesh.faces,np.array([1000,1000,40]))},
        cam,[item],[0,0,40],allow_unverified_hypothesis=True,
        config=dict(resolutions=[16],max_evaluations_per_level=4,
                    translation_bound_mm=1.,rotation_bound_deg=1.,initial_rotations_deg=[0.]))
    assert fitted['final_parameters'] is None
    assert fitted['final_fit_loss_native_px'] is None
    assert fitted['status']=='fit_projection_failure'


@pytest.mark.parametrize('config',[
    {'scale':1.1},{'initial_rotations_deg':[15.]},{'translation_bound_mm':0.},
    {'resolutions':[]},{'max_evaluations_per_level':True}])
def test_config_cannot_sneak_in_free_scale_or_invalid_search(config):
    with pytest.raises(ValueError):
        api().fit_conditional_pose({},camera(),[record()],[0,0,40],
            allow_unverified_hypothesis=True,config=config)


def test_behind_source_geometry_is_explicit_failure_not_exception():
    mesh=trimesh.creation.box(extents=[2,2,2]);mesh.apply_translation([0,0,-40])
    result=api().evaluate_records({'T8':(mesh.vertices,mesh.faces,np.array([0,0,-40]))},camera(),[record()])
    assert result['failure_count']==1
    assert result['mean_per_region_px'] is None
    assert result['rows'][0]['model_state']=='invalid_source_plane_projection'


def test_real_level_raster_calls_include_initial_score_in_evaluation_cap(monkeypatch):
    module=api();actual=module.render_hard;calls=[]
    def counted(*args,**kwargs):
        if kwargs['image_hw']==(16,16):calls.append(1)
        return actual(*args,**kwargs)
    monkeypatch.setattr(module,'render_hard',counted)
    mesh=trimesh.creation.box(extents=[8,12,4]);mesh.apply_translation([0,0,40])
    module.fit_conditional_pose({'T8':(mesh.vertices,mesh.faces,np.array([0,0,40]))},camera(),[record()],[0,0,40],
        allow_unverified_hypothesis=True,config=dict(resolutions=[16],max_evaluations_per_level=1,
                    translation_bound_mm=4.,rotation_bound_deg=8.,initial_rotations_deg=[0.]))
    assert len(calls)==1
