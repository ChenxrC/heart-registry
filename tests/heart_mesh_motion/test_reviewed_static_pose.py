"""Bounded 3DOF static correction, with no holdout or shape degrees of freedom."""
from dataclasses import replace
import importlib
import numpy as np
import pytest
import trimesh

from ctdr_pipeline.heart_mesh_motion.contracts import CameraData
from ctdr_pipeline.heart_mesh_motion.geometry import project_lps


def api():
    return importlib.import_module('ctdr_pipeline.heart_mesh_motion.reviewed_static_pose')


def camera():
    return CameraData([0,0,0],[1,0,0],[0,1,0],[0,0,1], [0,0,100],
                      [40,30],[1.5,2],(61,81),{},False)


def test_rigid_matrix_is_physical_lps_translation_and_rotation_not_pixel_warp():
    matrix = api().inplane_matrix(camera(), [2,-3,90], [0,0,40])
    actual = np.c_[[[1,0,40],[0,1,45]],np.ones(2)] @ matrix.T
    np.testing.assert_allclose(actual[:,:3], [[2,-2,40],[1,-3,45]], atol=1e-10)
    np.testing.assert_allclose(matrix[:3,:3].T@matrix[:3,:3],np.eye(3),atol=1e-10)
    assert np.linalg.det(matrix[:3,:3]) == pytest.approx(1)


def test_camera_inverse_equals_moving_ct_without_double_transform():
    cam = camera()
    matrix = api().inplane_matrix(cam,[2,-3,7],[0,0,40])
    corrected = api().camera_in_original_ct(cam,matrix)
    points=np.array([[-4,2,30],[5,-3,45],[1,3,60]])
    moved=np.c_[points,np.ones(3)]@matrix.T
    expected,_=project_lps(moved[:,:3],cam)
    actual,_=project_lps(points,corrected)
    np.testing.assert_allclose(actual,expected,atol=1e-10)
    assert corrected.geometry_accepted is False
    np.testing.assert_array_equal(corrected.principal_xy_px,cam.principal_xy_px)
    np.testing.assert_array_equal(corrected.spacing_row_col_mm,cam.spacing_row_col_mm)
    np.testing.assert_allclose(corrected.n_lps,cam.n_lps)


@pytest.mark.parametrize('parameters', [[np.nan,0,0],[0,0],[0,0,0,0]])
def test_nonfinite_or_wrong_pose_cannot_be_exported(parameters):
    with pytest.raises(ValueError):
        api().inplane_matrix(camera(),parameters,[0,0,40])


def test_inverse_camera_refuses_scale_shear_reflection_or_out_of_plane_motion():
    for bad in (np.diag([2,1,1,1]),np.diag([-1,1,1,1])):
        with pytest.raises(ValueError):
            api().camera_in_original_ct(camera(),bad)
    bad=np.eye(4);bad[2,3]=2
    with pytest.raises(ValueError):
        api().camera_in_original_ct(camera(),bad)


def record(label='T3',split='fit',xy=None):
    return {'structure_id':label,'split':split,'xy_px':xy or [[20,20],[25,20]],
            'run_id':'series_02','frame_index':21,'review_state':'confirmed_visible_edge','identity_confirmed':True}


def test_solver_boundary_rejects_holdout_even_if_it_has_valid_points():
    with pytest.raises(ValueError,match='fit'):
        api().fit_inplane_pose({},camera(),[record(split='holdout')],[0,0,40])


def test_holdout_label_cannot_be_disguised_by_changing_its_split_field():
    with pytest.raises(ValueError,match='fit'):
        api().fit_inplane_pose({},camera(),[record('T3'),record('T4','fit')],[0,0,40])


def test_missing_training_structure_returns_insufficient_not_zero_error():
    result=api().fit_inplane_pose({'T3':None,'T5':None},camera(),[record(),record('T5')],[0,0,40])
    assert result['status']=='insufficient_fit_geometry'
    assert result['final_parameters'] is None
    assert result['final_fit_loss_native_px'] is None


def test_training_selection_is_fixed_and_unreviewed_or_holdout_never_enters_fit():
    records=[record(),record('T4','holdout'),record('T5'),record('T7')]
    result=api().training_observations(records,run_id='series_02')
    assert [r['structure_id'] for r in result]==['T3','T5']
    records[1]['xy_px']=[[1e6,1e6],[2e6,2e6]]
    assert api().training_observations(records,run_id='series_02')==result


def test_partial_edge_distance_does_not_reward_empty_model_or_border_sampling():
    curve=np.array([[0.,0.],[1.,0.]])
    empty=np.zeros((16,16),bool)
    assert api().visible_edge_loss(empty,curve,native_hw=(32,32)) > 40
    edge=empty.copy();edge[:,8]=True
    assert api().visible_edge_loss(edge,curve,native_hw=(32,32)) > 10


def test_empty_or_single_training_structure_is_not_a_pose_fit():
    result=api().fit_inplane_pose({},camera(),[],[0,0,40])
    assert result['status']=='insufficient_fit_observations'
    assert result['final_parameters'] is None


def test_independent_synthetic_translation_rotation_reduces_error_and_respects_bounds():
    from scipy import ndimage
    from skimage.measure import find_contours
    from ctdr_pipeline.heart_mesh_motion.reference_raster import render_hard
    cam=replace(camera(),principal_xy_px=[40,40],spacing_row_col_mm=[1,1],image_hw=(81,81))
    meshes={};observations=[]
    angle=np.deg2rad(4)
    rotation=np.array([[np.cos(angle),-np.sin(angle),0],[np.sin(angle),np.cos(angle),0],[0,0,1]])
    # Independent literal ground-truth transform, not the implementation helper.
    pivot=np.array([0,0,40]);translation=np.array([2,-1.5,0])
    for name,centre in (('T3',[-7,-7,40]),('T5',[7,8,55])):
        mesh=trimesh.creation.box(extents=[8,10,6]);mesh.apply_translation(centre)
        meshes[name]=(mesh.vertices.copy(),mesh.faces.copy(),np.array(centre))
        target=(mesh.vertices-pivot)@rotation.T+pivot+translation
        mask=render_hard(target,mesh.faces,cam,image_hw=cam.image_hw)
        contour=max(find_contours(mask.astype(float),1-1e-6),key=len)[:,::-1]
        observations.append(record(name,xy=contour.tolist()))
    result=api().fit_inplane_pose(meshes,cam,observations,pivot,
                                 config={'resolutions':[81],'max_evaluations_per_level':220,
                                         'translation_bound_mm':4.,'rotation_bound_deg':8.})
    assert result['final_fit_loss_native_px'] < result['initial_fit_loss_native_px']*.4
    np.testing.assert_allclose(result['final_parameters'][:2],[2,-1.5],atol=.8)
    assert abs(result['final_parameters'][2]-4)<2
    assert result['holdout_consumed'] is False
    assert result['absolute_geometry_accepted'] is False
    assert all(abs(x)<=b for x,b in zip(result['final_parameters'],[4,4,8]))


def test_final_offscreen_training_meshes_are_failure_not_measured_pixel_error():
    meshes={}
    for name in ('T3','T5'):
        mesh=trimesh.creation.box(extents=[2,2,2]);mesh.apply_translation([1000,1000,40])
        meshes[name]=(mesh.vertices,mesh.faces,np.array([1000,1000,40]))
    result=api().fit_inplane_pose(meshes,camera(),[record(),record('T5')],[0,0,40],
                                 config={'resolutions':[16],'max_evaluations_per_level':8,
                                         'translation_bound_mm':1.,'rotation_bound_deg':1.})
    assert result['status']=='fit_projection_failure'
    assert result['final_fit_loss_native_px'] is None
    assert result['final_parameters'] is None
    assert result['failed_fit_structures']==['T3','T5']
    assert result['optimization_attempted'] is True


def test_heart_archive_geometry_is_bound_to_prior_digest_not_just_mask_tag(tmp_path):
    import json,hashlib
    probe=importlib.import_module('experiments.heart_mesh_dynamic.probes.reviewed_static_pose')
    mesh_path=tmp_path/'mesh.npz';report_path=tmp_path/'prior.json'
    np.savez_compressed(mesh_path,vertices_lps_mm=np.zeros((4,3)),faces=np.array([[0,1,2]]),source_sha256=np.asarray('a'*64))
    digest=hashlib.sha256(mesh_path.read_bytes()).hexdigest()
    report_path.write_text(json.dumps({'source_sha256':{str(mesh_path):digest}}),encoding='utf-8')
    # Retain mask identity but alter geometry: this must not silently become the same mesh.
    np.savez_compressed(mesh_path,vertices_lps_mm=np.ones((4,3)),faces=np.array([[0,1,2]]),source_sha256=np.asarray('a'*64))
    with pytest.raises(ValueError,match='SHA256'):
        probe.load_pinned_heart_mesh(mesh_path,report_path,'a'*64)


def test_evaluation_counts_only_declared_s2_scope_and_preserves_missing_holdout():
    probe=importlib.import_module('experiments.heart_mesh_dynamic.probes.reviewed_static_pose')
    pack={'schema_version':1,'case_id':'x','pack_id':'a'*64,'display_material_sha256':'b'*64,
          'series':[{'run_id':r,'frame_index':21,'image_hw':[61,81]} for r in ('series_01','series_02')],
          'structures':[{'id':f'T{i}','split':'fit' if i%2 else 'holdout'} for i in range(1,13)]}
    review={'schema_version':1,'case_id':'x','pack_id':'a'*64,'display_material_sha256':'b'*64,
            'reviewer':'test','observations':[record(),record('T5')]}
    scoped,observed=probe.evaluation_scope(pack,review)
    from experiments.heart_mesh_dynamic.probes.bone_pose_review import evaluate_review
    edges={('series_02',name):np.ones((61,81),bool) for name in ('T3','T5')}
    result=evaluate_review(observed,scoped,edges)
    assert result['expected_structure_count']==3
    assert result['reviewed_visible_structure_count']==2
    assert result['missing_structure_count']==1
