import importlib
import numpy as np
import pytest
import torch


def test_missing_renderer_raises_not_a_fake_soft_image():
    module=importlib.import_module('ctdr_pipeline.heart_mesh_motion.renderer')
    if importlib.util.find_spec('pytorch3d') is not None:pytest.skip('missing-backend scenario not active')
    from ctdr_pipeline.heart_mesh_motion.contracts import CameraData
    camera=CameraData([0,0,0],[1,0,0],[0,1,0],[0,0,1],[0,0,100],[31.5,31.5],[1,1],(64,64),{},False)
    with pytest.raises(RuntimeError,match='PyTorch3D'):
        module.render_soft(torch.tensor([[0.,0,50],[1,0,50],[0,1,50]]),torch.tensor([[0,1,2]]),camera,image_hw=(64,64))


def test_actual_soft_renderer_gradient_matches_pixel_orientation_and_finite_difference():
    module=importlib.import_module('ctdr_pipeline.heart_mesh_motion.renderer')
    pytest.importorskip('pytorch3d')
    from ctdr_pipeline.heart_mesh_motion.contracts import CameraData
    camera=CameraData([0,0,0],[1,0,0],[0,1,0],[0,0,1],[0,0,100],[31.5,31.5],[1,1],(64,64),{},False)
    points=torch.tensor([[-8.,-6,60],[7,-5,60],[3,8,60]])
    faces=torch.tensor([[0,1,2]])
    x=torch.arange(64).float()[None,:]
    def loss(shift):
        v=points+torch.stack([shift,shift*0,shift*0])[None,:]
        return (module.render_soft(v,faces,camera,image_hw=(64,64))*x).sum()
    shift=torch.tensor(.3,requires_grad=True);value=loss(shift);value.backward()
    gradient=float(shift.grad);numeric=float((loss(torch.tensor(.31))-loss(torch.tensor(.29)))/.02)
    assert np.isfinite(gradient) and gradient>0
    assert abs(gradient-numeric)/max(abs(numeric),1e-4)<1e-2


@pytest.mark.parametrize('image_hw,principal,spacing',[((64,64),[31.5,31.5],[4,4]),((80,96),[37.5,43.5],[3,5])])
def test_actual_soft_and_independent_hard_projection_share_pixel_centres(image_hw,principal,spacing):
    pytest.importorskip('pytorch3d')
    from dataclasses import replace
    from scipy import ndimage
    from ctdr_pipeline.heart_mesh_motion.synthetic_mesh import digital_fixture
    from ctdr_pipeline.heart_mesh_motion.reference_raster import render_hard
    from ctdr_pipeline.heart_mesh_motion.renderer import render_soft
    mesh,camera=digital_fixture()
    camera=replace(camera,image_hw=image_hw,principal_xy_px=principal,spacing_row_col_mm=spacing)
    hard=render_hard(mesh.vertices_lps_mm,mesh.faces,camera,image_hw=image_hw)
    soft=render_soft(torch.tensor(mesh.vertices_lps_mm,dtype=torch.float32),torch.tensor(mesh.faces),camera,image_hw=image_hw).detach().numpy()>.5
    # Soft alpha expands a boundary band; symmetry still fixes the exact centre.
    np.testing.assert_allclose(np.argwhere(soft).mean(0),np.argwhere(hard).mean(0),atol=.15)
    boundary=ndimage.binary_dilation(hard,iterations=2)^ndimage.binary_erosion(hard,iterations=2)
    np.testing.assert_array_equal(soft[~boundary],hard[~boundary])
