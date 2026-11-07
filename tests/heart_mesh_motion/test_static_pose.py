import importlib
import numpy as np
import torch


def test_static_target_never_consumes_holdout_frames():
    api=importlib.import_module('ctdr_pipeline.heart_mesh_motion.static_pose')
    cine=np.arange(6*12*12,dtype=np.uint16).reshape(6,12,12)%256
    a=api.support_background(cine,[0,1,4,5])
    cine[2:4]=999
    b=api.support_background(cine,[0,1,4,5])
    np.testing.assert_array_equal(a[0],b[0]);np.testing.assert_array_equal(a[1],b[1])


def test_static_calibration_recovers_known_inplane_translation_but_not_absolute_geometry():
    api=importlib.import_module('ctdr_pipeline.heart_mesh_motion.static_pose')
    from ctdr_pipeline.heart_mesh_motion.bone_projection import render_bone,BoneVolume
    from ctdr_pipeline.heart_mesh_motion.contracts import CameraData
    data=np.zeros((24,24,24),np.float32);data[4:15,7:10,5:9]=1;data[7:18,13:18,13:17]=.6
    affine=np.eye(4);affine[:3,3]=[-12,-12,55]
    volume=BoneVolume(data,affine,'0'*64,{})
    camera=CameraData([0,0,0],[1,0,0],[0,1,0],[0,0,1],[0,0,100],[15.5,15.5],[1,1],(32,32),{},False)
    target=render_bone(volume,camera,torch.tensor([4.,-4.]),image_hw=(32,32),samples=48).detach().numpy()
    report=api.fit_static_translation(volume,camera,target,np.ones_like(target),bound_mm=8,grid_step_mm=4,steps=5,resolution=32,samples=48)
    np.testing.assert_allclose(report['translation_uv_mm'],[4,-4],atol=.3)
    assert report['absolute_geometry_accepted'] is False
    assert report['optimized_dofs']==['translation_u_mm','translation_v_mm']
    np.testing.assert_allclose(np.asarray(report['ct_to_xa_lps_matrix'])[:3,3],[4,-4,0],atol=.3)


def test_empty_static_observations_are_insufficient_not_success():
    api=importlib.import_module('ctdr_pipeline.heart_mesh_motion.static_pose')
    from ctdr_pipeline.heart_mesh_motion.bone_projection import BoneVolume
    from ctdr_pipeline.heart_mesh_motion.contracts import CameraData
    volume=BoneVolume(np.ones((4,4,4),np.float32),np.eye(4),'0'*64,{})
    camera=CameraData([0,0,-10],[1,0,0],[0,1,0],[0,0,1],[0,0,10],[8,8],[1,1],(16,16),{},False)
    result=api.fit_static_translation(volume,camera,np.zeros((16,16)),np.ones((16,16)),steps=1,resolution=16)
    assert result['status']=='insufficient_observations'
    assert result['absolute_geometry_accepted'] is False


def test_bone_free_volume_does_not_report_a_registered_pose():
    api=importlib.import_module('ctdr_pipeline.heart_mesh_motion.static_pose')
    from ctdr_pipeline.heart_mesh_motion.bone_projection import BoneVolume
    from ctdr_pipeline.heart_mesh_motion.contracts import CameraData
    volume=BoneVolume(np.zeros((4,4,4),np.float32),np.eye(4),'0'*64,{})
    camera=CameraData([0,0,-10],[1,0,0],[0,1,0],[0,0,1],[0,0,10],[8,8],[1,1],(16,16),{},False)
    result=api.fit_static_translation(volume,camera,np.arange(256).reshape(16,16),np.ones((16,16)),steps=1,resolution=16)
    assert result['status']=='insufficient_projection' and result['ct_to_xa_lps_matrix'] is None


def test_nonempty_ct_wholly_outside_search_is_insufficient_not_nan_gradient():
    api=importlib.import_module('ctdr_pipeline.heart_mesh_motion.static_pose')
    from ctdr_pipeline.heart_mesh_motion.bone_projection import BoneVolume
    from ctdr_pipeline.heart_mesh_motion.contracts import CameraData
    affine=np.eye(4);affine[:3,3]=[10000,10000,10000]
    volume=BoneVolume(np.ones((4,4,4),np.float32),affine,'0'*64,{})
    camera=CameraData([0,0,-10],[1,0,0],[0,1,0],[0,0,1],[0,0,10],[8,8],[1,1],(16,16),{},False)
    result=api.fit_static_translation(volume,camera,np.arange(256).reshape(16,16),np.ones((16,16)),steps=1,resolution=16,bound_mm=4,grid_step_mm=4)
    assert result['status']=='insufficient_projection' and result['ct_to_xa_lps_matrix'] is None


def test_preflight_checks_lexical_authority_before_resolution(tmp_path,monkeypatch):
    from pathlib import Path
    from ctdr_pipeline.heart_mesh_motion import static_preflight
    import pytest
    # Simulate a resolving alias without needing Windows symlink creation rights.
    original=Path.resolve
    monkeypatch.setattr(Path,'resolve',lambda p,*a,**k:original(Path(str(p).replace('case_candidate','case')),*a,**k))
    with pytest.raises(ValueError,match='candidate'):
        static_preflight.run_static_preflight(tmp_path/'case_candidate',tmp_path/'assets',tmp_path/'audit/preflight.json',
            tmp_path/'ct/ct.nii.gz',tmp_path/'results/new')


def _weighted_observation_fixture():
    from ctdr_pipeline.heart_mesh_motion.bone_projection import BoneVolume
    from ctdr_pipeline.heart_mesh_motion.contracts import CameraData
    data=np.zeros((24,24,24),np.float32);data[4:15,7:10,5:9]=1;data[7:18,13:18,13:17]=.6
    affine=np.eye(4);affine[:3,3]=[-12,-12,55]
    volume=BoneVolume(data,affine,'0'*64,{})
    camera=CameraData([0,0,0],[1,0,0],[0,1,0],[0,0,1],[0,0,100],[15.5,15.5],[1,1],(32,32),{},False)
    return volume,camera


def test_resized_constant_target_returns_no_pose_even_when_native_is_nonconstant():
    from ctdr_pipeline.heart_mesh_motion.static_pose import support_background,fit_static_translation
    volume,camera=_weighted_observation_fixture()
    checker=(np.indices((32,32)).sum(0)%2*255).astype(np.uint8)
    target,weight=support_background(np.repeat(checker[None],6,axis=0),[0,1,4,5])
    result=fit_static_translation(volume,camera,target,weight,bound_mm=4,grid_step_mm=4,steps=0,resolution=16,samples=16)
    assert result['status']=='insufficient_observations'
    assert result['ct_to_xa_lps_matrix'] is None and result['translation_uv_mm'] is None


def test_projection_outside_weighted_region_returns_no_arbitrary_pose():
    from ctdr_pipeline.heart_mesh_motion.static_pose import fit_static_translation
    volume,camera=_weighted_observation_fixture()
    target=np.indices((32,32))[0].astype(np.float32)
    weight=np.zeros((32,32),np.float32);weight[4:28,26:30]=1
    result=fit_static_translation(volume,camera,target,weight,bound_mm=.1,grid_step_mm=.1,steps=1,resolution=32,samples=16)
    assert result['status']=='insufficient_projection'
    assert result['ct_to_xa_lps_matrix'] is None and result['translation_uv_mm'] is None
