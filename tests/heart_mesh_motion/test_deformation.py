import importlib
import numpy as np
import torch
import trimesh


def fixture():
    from ctdr_pipeline.heart_mesh_motion.contracts import MeshData,CameraData
    m=trimesh.creation.icosphere(subdivisions=1,radius=10);m.apply_translation([0,0,200])
    mesh=MeshData(m.vertices,m.faces,'0'*64,{'centroid_lps_mm':[0,0,200]})
    camera=CameraData([0,0,0],[1,0,0],[0,1,0],[0,0,1],[0,0,500],[31.5,31.5],[2,2],(64,64),{},False)
    return mesh,camera


def test_zero_parameters_preserve_vertices_and_basis_has_no_independent_depth():
    module=importlib.import_module('ctdr_pipeline.heart_mesh_motion.deformation')
    mesh,camera=fixture();basis=module.build_ffd_basis(mesh,camera)
    params=torch.zeros(4+basis['rank'],dtype=torch.float64)
    vertices=module.deform_vertices(mesh,camera,params,'ffd_inplane_radial',basis)
    np.testing.assert_array_equal(vertices.numpy(),mesh.vertices_lps_mm)
    assert basis['rank']<=51
    np.testing.assert_allclose(np.einsum('vdk,d->vk',basis['vectors_lps'],camera.n_lps),0,atol=1e-10)


def test_rigid_translation_and_radial_scale_are_distinct_and_differentiable():
    module=importlib.import_module('ctdr_pipeline.heart_mesh_motion.deformation')
    mesh,camera=fixture();basis=module.build_ffd_basis(mesh,camera)
    rigid=module.deform_vertices(mesh,camera,torch.tensor([2.,-3,0],dtype=torch.float64),'rigid_inplane',basis)
    np.testing.assert_allclose(rigid.numpy(),mesh.vertices_lps_mm+[2,-3,0],atol=1e-12)
    params=torch.zeros(4+basis['rank'],dtype=torch.float64);params[3]=.05;params.requires_grad_()
    deformed=module.deform_vertices(mesh,camera,params,'ffd_inplane_radial',basis)
    np.testing.assert_allclose(deformed.detach().numpy(),[0,0,200]+(mesh.vertices_lps_mm-[0,0,200])*1.05,atol=1e-12)
    deformed.square().sum().backward();assert torch.isfinite(params.grad).all()


def test_ffd_does_not_duplicate_global_translation_or_rotation():
    module=importlib.import_module('ctdr_pipeline.heart_mesh_motion.deformation')
    mesh,camera=fixture();basis=module.build_ffd_basis(mesh,camera)
    vectors=basis['vectors_lps'];reference=mesh.vertices_lps_mm-[0,0,200]
    assert np.max(np.abs(vectors.sum(axis=0)))<1e-8
    rigid_rotation=np.cross(camera.n_lps,reference)
    assert np.max(np.abs(np.einsum('vdk,vd->k',vectors,rigid_rotation)))<1e-8
