"""Independent hard targets and support-only inverse-rendering reference tests."""
from __future__ import annotations

from dataclasses import replace,asdict
import hashlib
import json
import time
import numpy as np
import trimesh

from .contracts import MeshData,CameraData,FitConfig
from .deformation import build_ffd_basis,deform_vertices
from .evaluation import make_split,interpolate_parameters
from .geometry import project_lps
from .reference_raster import render_hard
from .runtime import probe_runtime


def digital_fixture():
    source=trimesh.creation.icosphere(subdivisions=1,radius=1)
    vertices=source.vertices*[28,38,24]+[0,0,220]
    mesh=MeshData(vertices,source.faces,'0'*64,{'source':'digital_ellipsoid_not_patient','optimization_ready':True})
    camera=CameraData([0,0,0],[1,0,0],[0,1,0],[0,0,1],[0,0,500],[31.5,31.5],[4,4],(64,64),{'source':'synthetic'},False)
    return mesh,camera


def _truth(mesh,camera,basis,case,count):
    import torch
    trajectories=[]
    for phase in np.linspace(0,2*np.pi,count,endpoint=False):
        theta=np.zeros(4+basis['rank'],np.float32)
        if case in ('translation','mixed','occlusion','missing','wrong_camera','partial_fov'):
            theta[:2]=[3*np.sin(phase),2*np.sin(phase+.4)]
        if case in ('contraction','mixed'):theta[3]=-.05*(1-np.cos(phase))/2
        if case in ('local_ffd','mixed') and basis['rank']:theta[4]=1.5*np.sin(phase)
        vertices=deform_vertices(mesh,camera,torch.from_numpy(theta),'ffd_inplane_radial',basis).numpy()
        if case=='out_of_model_depth':vertices=vertices+np.array([0,0,8*np.sin(phase)])
        trajectories.append(vertices)
    return np.stack(trajectories)


def _fit(mesh,camera,basis,targets,visibility,support,times,model,config,device):
    import torch
    from .renderer import render_soft
    k={'static':0,'rigid_inplane':3,'ffd_inplane_radial':4+basis['rank']}[model]
    bounds=config.bounds
    scale_extent=max(1-bounds['scale_min'],bounds['scale_max']-1)
    scales=np.array([bounds['translation_mm']]*2+[np.deg2rad(bounds['rotation_deg']),scale_extent]+[bounds['local_mm']]*basis['rank'],np.float32)[:k]
    normalized=torch.zeros((len(support),k),device=device,requires_grad=True)
    optimizer=torch.optim.Adam([normalized],lr=.04) if k else None
    faces=torch.as_tensor(mesh.faces,device=device,dtype=torch.int64)
    target=torch.as_tensor(targets[support],device=device,dtype=torch.float32)
    valid=torch.as_tensor(visibility[support],device=device,dtype=torch.bool)
    scaling=torch.as_tensor(scales,device=device)
    dt=torch.as_tensor(np.diff(times[support]),device=device,dtype=torch.float32)
    history=[];started=time.perf_counter()
    if k:
        for _ in range(config.optimizer_budget):
            optimizer.zero_grad();terms=[]
            for j,index in enumerate(support):
                if not bool(valid[j].any()):continue
                vertices=deform_vertices(mesh,camera,normalized[j]*scaling,model,basis)
                alpha=render_soft(vertices,faces,camera,image_hw=targets.shape[1:])
                terms.append(torch.nn.functional.binary_cross_entropy(alpha[valid[j]].clamp(1e-5,1-1e-5),target[j][valid[j]]))
            if not terms:raise ValueError('No synthetic support observations')
            loss=config.loss_weights['data']*torch.stack(terms).mean()+config.loss_weights['amplitude']*normalized.square().mean()
            if len(support)>1:
                velocity=(normalized[1:]-normalized[:-1])/dt[:,None]
                loss=loss+config.loss_weights['temporal']*velocity.square().mean()
            if not bool(torch.isfinite(loss)):raise RuntimeError('Nonfinite synthetic objective')
            loss.backward();optimizer.step()
            with torch.no_grad():
                normalized.clamp_(-1,1)
                if model=='ffd_inplane_radial':
                    if scale_extent:
                        normalized[:,3].clamp_((bounds['scale_min']-1)/scale_extent,(bounds['scale_max']-1)/scale_extent)
                    vectors=torch.as_tensor(basis['vectors_lps'],dtype=torch.float32,device=device)
                    displacement=torch.einsum('vdk,sk->svd',vectors,normalized[:,4:]*scaling[4:])
                    maximum=displacement.norm(dim=2).amax(dim=1).clamp_min(max(bounds['local_mm'],1e-12))
                    normalized[:,4:]*=(bounds['local_mm']/maximum)[:,None]
            history.append(float(loss.detach()))
    parameters=(normalized.detach()*scaling).cpu().numpy() if k else np.empty((len(support),0))
    all_parameters=interpolate_parameters(times[support],parameters,times)
    vertices=np.stack([deform_vertices(mesh,camera,torch.as_tensor(p),model,basis).numpy() for p in all_parameters])
    return all_parameters,vertices,history,time.perf_counter()-started


def run_mesh_benchmark(config: FitConfig,reference_mesh: MeshData|None=None,*,device='cpu',smoke=False) -> dict:
    runtime=probe_runtime(device)
    if runtime['status']!='available':raise RuntimeError('Verified differentiable backend unavailable: '+runtime.get('reason','unknown'))
    mesh,camera=digital_fixture();transform=np.eye(4)
    if reference_mesh is not None:
        if not reference_mesh.report.get('optimization_ready',False):raise ValueError('Reference mesh failed optimization preparation')
        center=np.asarray(trimesh.Trimesh(reference_mesh.vertices_lps_mm,reference_mesh.faces,process=False).center_mass)
        extent=np.linalg.norm(reference_mesh.vertices_lps_mm-center,axis=1).max()
        transform[:3,3]=[0,0,max(220,6*extent)]-center
        mesh=MeshData(reference_mesh.vertices_lps_mm+transform[:3,3],reference_mesh.faces,reference_mesh.source_sha256,
            {'source':'CT_static_shape_in_synthetic_camera','optimization_ready':True})
    basis=build_ffd_basis(mesh,camera)
    count=6 if smoke else 12;times=np.arange(count,dtype=float)*.133333
    split=make_split(count,'synthetic');support=split['support_indices'];holdout=split['holdout_indices']
    size=config.resolutions[-1]
    cases=['translation'] if smoke else ['stationary','translation','contraction','local_ffd','mixed','out_of_model_depth','occlusion','missing','wrong_camera','partial_fov']
    perturbations=[(0,0),(5,0),(-5,0),(0,5),(0,-5)]
    uv,_=project_lps(mesh.vertices_lps_mm,camera)
    other=camera.source_lps_mm+1.1*(mesh.vertices_lps_mm-camera.source_lps_mm)
    ambiguity={'unique_depth':False,'same_projection_max_abs_px':float(np.abs(project_lps(other,camera)[0]-uv).max()),
        'different_3d_rms_mm':float(np.sqrt(np.mean(np.sum((other-mesh.vertices_lps_mm)**2,axis=1))))}
    results=[];example=None
    for case in cases:
        truth=_truth(mesh,camera,basis,case,count)
        target_camera=replace(camera,principal_xy_px=camera.principal_xy_px+[20,0]) if case=='partial_fov' else camera
        targets=np.stack([render_hard(v,mesh.faces,target_camera,image_hw=(size,size)) for v in truth]).astype(np.float32)
        visible=np.ones_like(targets,dtype=bool)
        if case=='occlusion':visible[:,:,size//3:2*size//3]=False
        if case=='missing':visible[1]=False
        fit_camera=replace(target_camera,principal_xy_px=target_camera.principal_xy_px+[5,0]) if case=='wrong_camera' else target_camera
        for shift_u,shift_v in perturbations:
            offset=shift_u*camera.u_lps+shift_v*camera.v_lps
            initial=MeshData(mesh.vertices_lps_mm+offset,mesh.faces,mesh.source_sha256,mesh.report)
            initial_basis={**basis,'center_lps_mm':basis['center_lps_mm']+offset}
            for model in ('static','rigid_inplane','ffd_inplane_radial'):
                parameters,predicted,history,seconds=_fit(initial,fit_camera,initial_basis,targets,visible,support,times,model,config,device)
                masks=np.stack([render_hard(v,mesh.faces,fit_camera,image_hw=(size,size)) for v in predicted])
                disagreement=[float(np.mean(np.not_equal(masks[i],targets[i].astype(bool))[visible[i]])) if visible[i].any() else None for i in range(count)]
                rms=np.sqrt(np.mean(np.sum((predicted-truth)**2,axis=2),axis=1))
                results.append({'case':case,'model':model,'initial_offset_uv_mm':[shift_u,shift_v],
                    'foreground_disagreement_by_frame':disagreement,'synthetic_vertex_rms_mm_by_frame':rms.tolist(),
                    'holdout_vertex_rms_mm':float(rms[holdout].mean()),'support_vertex_rms_mm':float(rms[support].mean()),
                    'optimization_seconds':seconds,'history':history,'parameters':parameters.tolist(),
                    'self_intersection_validated':False,'status':'synthetic_reference_result_not_clinical'})
                if case=='translation' and shift_u==shift_v==0 and model=='ffd_inplane_radial':
                    example={'source_vertices':mesh.vertices_lps_mm.tolist(),'truth_vertices':truth.tolist(),
                        'fitted_vertices':predicted.tolist(),'faces':mesh.faces.tolist(),'target_masks':targets.astype(int).tolist(),
                        'fitted_masks':masks.astype(int).tolist()}
    effective=asdict(config)
    config_hash=hashlib.sha256(json.dumps(effective,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
    return {'schema_version':1,'research_only':True,'runtime':runtime,'seed':config.seed,'smoke':smoke,
        'effective_config':effective,'config_sha256':config_hash,
        'frame_times_seconds':times.tolist(),'split':split,'resolution':size,'optimizer_budget':config.optimizer_budget,
        'shape_source':mesh.report['source'],'source_to_synthetic_transform':transform.tolist(),
        'camera':camera.to_dict(),'depth_ambiguity':ambiguity,'results':results,'example_motion':example,
        'real_patient_mesh_fit_executed':False,'patient_true_3d_tre_mm':None,'clinical_validation':False,
        'limitations':['Synthetic reference solver only; no actual XA edge fit or unique depth recovery.',
            'Smoke uses a reduced sequence/configuration and is not the full benchmark.',
            'Reference loss is full-alpha BCE, not the later partial-edge production objective.',
            'Surface self-intersection acceptance is not implemented; do not promote results to validated anatomy.']}
