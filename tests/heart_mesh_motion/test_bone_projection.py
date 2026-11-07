import importlib
import numpy as np
import torch


def camera():
    from ctdr_pipeline.heart_mesh_motion.contracts import CameraData
    return CameraData([0,0,0],[1,0,0],[0,1,0],[0,0,1],[0,0,100],[20,20],[1,1],(41,41),{},False)


def test_bone_ray_integral_uses_lps_affine_and_millimetre_length():
    api=importlib.import_module('ctdr_pipeline.heart_mesh_motion.bone_projection')
    affine=np.diag([1.,1.,-1.,1.]);affine[:3,3]=[-5,-5,20]
    volume=api.BoneVolume(np.ones((11,11,11),np.float32),affine,'a'*64,{})
    image=api.render_bone(volume,camera(),torch.zeros(2),image_hw=(41,41),samples=48)
    assert abs(float(image[20,20])-10)<1e-4
    assert np.isfinite(image.detach().numpy()).all()


def phantom():
    api=importlib.import_module('ctdr_pipeline.heart_mesh_motion.bone_projection')
    data=np.zeros((31,31,31),np.float32)
    data[10:18,11:15,11:16]=1;data[13:21,17:21,17:23]=.7
    affine=np.eye(4);affine[:3,3]=[-15,-15,55]
    return api.BoneVolume(data,affine,'a'*64,{'source':'test_phantom_not_patient'})


def test_bone_projection_translation_is_physical_and_differentiable_without_flip():
    api=importlib.import_module('ctdr_pipeline.heart_mesh_motion.bone_projection')
    v=phantom();c=camera();p=torch.tensor([.37,.29],dtype=torch.float64,requires_grad=True)
    yy=torch.arange(41)[:,None].float()
    def objective(parameters):return (api.render_bone(v,c,parameters,image_hw=(41,41),samples=48)*yy).sum()
    value=objective(p);value.backward()
    assert torch.isfinite(p.grad).all() and p.grad[1]>0
    # .005 crosses interpolation cells; probe the stable local derivative in double.
    eps=.0001
    numeric=float((objective(p.detach()+torch.tensor([0.,eps],dtype=p.dtype))-objective(p.detach()-torch.tensor([0.,eps],dtype=p.dtype)))/(2*eps))
    assert abs(float(p.grad[1])-numeric)/max(abs(numeric),1)<.02

