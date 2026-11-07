import importlib
import numpy as np
import pytest


def test_depth_ambiguity_has_equal_projection_and_different_shape():
    module=importlib.import_module('ctdr_pipeline.heart_mesh_motion.synthetic_mesh')
    mesh,camera=module.digital_fixture()
    from ctdr_pipeline.heart_mesh_motion.geometry import project_lps
    other=camera.source_lps_mm+(mesh.vertices_lps_mm-camera.source_lps_mm)*1.1
    np.testing.assert_allclose(project_lps(mesh.vertices_lps_mm,camera)[0],project_lps(other,camera)[0],atol=1e-10)
    assert np.linalg.norm(other-mesh.vertices_lps_mm)>1


def test_no_backend_means_no_synthetic_fit_or_fake_success():
    module=importlib.import_module('ctdr_pipeline.heart_mesh_motion.synthetic_mesh')
    from ctdr_pipeline.heart_mesh_motion.contracts import FitConfig
    if importlib.util.find_spec('pytorch3d') is not None:pytest.skip('backend is present')
    with pytest.raises(RuntimeError,match='backend'):
        module.run_mesh_benchmark(FitConfig(resolutions=(32,),optimizer_budget=1))


def test_fitter_respects_configured_bounds_weight_and_holdout_independence(monkeypatch):
    # A substituted differentiable scalar tests fitter control, NOT renderer validity.
    import torch
    from ctdr_pipeline.heart_mesh_motion import synthetic_mesh as module,renderer
    from ctdr_pipeline.heart_mesh_motion.contracts import FitConfig
    monkeypatch.setattr(renderer,'render_soft',lambda v,f,c,*,image_hw:torch.sigmoid(v[:,0].mean()/10).expand(*image_hw))
    mesh,camera=module.digital_fixture();basis=module.build_ffd_basis(mesh,camera)
    targets=np.ones((6,16,16),np.float32);visible=np.ones_like(targets,bool)
    support=np.array([0,1,4,5]);times=np.arange(6.)
    bounds=dict(translation_mm=.01,rotation_deg=.01,scale_min=.999,scale_max=1.001,local_mm=.01)
    zero=FitConfig(resolutions=(16,),optimizer_budget=2,bounds=bounds,loss_weights=dict(data=0.,amplitude=0.,temporal=0.))
    p,_,history,_=module._fit(mesh,camera,basis,targets,visible,support,times,'rigid_inplane',zero,'cpu')
    np.testing.assert_array_equal(p,0);np.testing.assert_array_equal(history,0)
    active=FitConfig(resolutions=(16,),optimizer_budget=2,bounds=bounds)
    a=module._fit(mesh,camera,basis,targets,visible,support,times,'ffd_inplane_radial',active,'cpu')[0]
    assert np.abs(a[:,:2]).max()<=.010001
    assert np.abs(a[:,2]).max()<=np.deg2rad(.010001)
    assert a[:,3].min()>=-.001001 and a[:,3].max()<=.001001
    changed=targets.copy();changed[2:4]=0
    b=module._fit(mesh,camera,basis,changed,visible,support,times,'ffd_inplane_radial',active,'cpu')[0]
    np.testing.assert_array_equal(a,b)


def test_actual_renderer_inverse_fit_reduces_support_loss():
    pytest.importorskip('pytorch3d')
    from ctdr_pipeline.heart_mesh_motion import synthetic_mesh as module
    from ctdr_pipeline.heart_mesh_motion.contracts import FitConfig,MeshData
    mesh,camera=module.digital_fixture()
    initial=MeshData(mesh.vertices_lps_mm+[10,0,0],mesh.faces,mesh.source_sha256,mesh.report)
    basis=module.build_ffd_basis(initial,camera)
    target=module.render_hard(mesh.vertices_lps_mm,mesh.faces,camera,image_hw=(32,32))
    targets=np.repeat(target[None],6,axis=0).astype(np.float32)
    params,predicted,history,_=module._fit(initial,camera,basis,targets,np.ones_like(targets,bool),np.array([0,1,4,5]),np.arange(6.),
        'rigid_inplane',FitConfig(resolutions=(32,),optimizer_budget=12),'cpu')
    assert np.isfinite(predicted).all() and history[-1]<history[0]
    assert abs(params[:,0].mean()+10)<10
