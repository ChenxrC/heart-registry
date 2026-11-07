"""Low-dimensional mesh deformation with independent view depth locked."""
from __future__ import annotations

import numpy as np


def build_ffd_basis(mesh,camera,*,grid_shape=(3,3,3)) -> dict:
    if tuple(grid_shape)!=(3,3,3):raise ValueError('First version uses the frozen 3x3x3 grid')
    import trimesh
    solid=trimesh.Trimesh(mesh.vertices_lps_mm,mesh.faces,process=False)
    if not solid.is_watertight:raise ValueError('FFD reference must be closed')
    center=np.asarray(solid.center_mass)
    points=mesh.vertices_lps_mm-center
    axes=np.column_stack([camera.u_lps,camera.v_lps,camera.n_lps])
    local=points@axes
    low,high=local.min(0),local.max(0)
    if np.any(high-low<1e-8):raise ValueError('Reference has zero extent')
    coordinates=np.clip((local-low)/(high-low)*2,0,2)
    nodes=np.array(np.meshgrid(np.arange(3),np.arange(3),np.arange(3),indexing='ij')).reshape(3,-1).T
    weights=np.prod(np.maximum(1-np.abs(coordinates[:,None,:]-nodes[None,:,:]),0),axis=2)
    raw=np.zeros((len(points),3,54))
    raw[:,:,0::2]=weights[:,None,:]*camera.u_lps[None,:,None]
    raw[:,:,1::2]=weights[:,None,:]*camera.v_lps[None,:,None]
    flattened=raw.reshape(-1,54)
    rigid=np.stack([np.broadcast_to(camera.u_lps,points.shape),np.broadcast_to(camera.v_lps,points.shape),
        np.cross(camera.n_lps,points)],axis=-1).reshape(-1,3)
    q,_=np.linalg.qr(rigid)
    flattened-=q@(q.T@flattened)
    u,s,_=np.linalg.svd(flattened,full_matrices=False)
    rank=int(np.count_nonzero(s>s[0]*1e-8))
    vectors=u[:,:rank].reshape(len(points),3,rank)
    vectors/=np.linalg.norm(vectors,axis=1).max(axis=0)[None,None,:]
    return {'rank':rank,'vectors_lps':vectors,'center_lps_mm':center,'grid_shape':list(grid_shape),
        'singular_values':s[:rank],'independent_view_depth_optimized':False,'scale_depth_is_prior_coupling':True}


def deform_vertices(mesh,camera,parameters,model: str,basis: dict):
    import torch
    if model not in ('static','rigid_inplane','ffd_inplane_radial'):raise ValueError('Unknown model')
    expected={'static':0,'rigid_inplane':3,'ffd_inplane_radial':4+basis['rank']}[model]
    if parameters.ndim!=1 or len(parameters)!=expected or not bool(torch.isfinite(parameters).all()):
        raise ValueError('Parameter dimensions/nonfinite values invalid')
    tensor=lambda v:torch.as_tensor(v,dtype=parameters.dtype,device=parameters.device)
    vertices=tensor(mesh.vertices_lps_mm)
    if model=='static':return vertices
    centered=vertices-tensor(basis['center_lps_mm'])
    delta=torch.zeros_like(vertices)
    if model=='ffd_inplane_radial':
        delta=parameters[3]*centered+torch.einsum('vdk,k->vd',tensor(basis['vectors_lps']),parameters[4:])
    axis=tensor(camera.n_lps).expand_as(vertices)
    cross=torch.linalg.cross(axis,centered+delta)
    angle=parameters[2]
    rotated_delta=torch.sin(angle)*cross+(1-torch.cos(angle))*torch.linalg.cross(axis,cross)
    return vertices+delta+rotated_delta+parameters[0]*tensor(camera.u_lps)+parameters[1]*tensor(camera.v_lps)
