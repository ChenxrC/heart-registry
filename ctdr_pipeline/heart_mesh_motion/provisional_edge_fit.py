"""Conditional unsigned-contour diagnostic, not the validated signed-band fitter.

Long manual arcs do not define a complete foreground mask or material-point
correspondences. Only finite local neighbourhoods of supplied open arcs enter
the image objective. Depth, camera, topology and original annotations stay fixed.
"""
from __future__ import annotations
import time
import numpy as np

from .deformation import build_ffd_basis,deform_vertices
from .renderer import render_soft


def sample_observation(observation,native_hw,*,spacing_px=2.):
    if observation.get('review_state')!='visible_confirmed':raise ValueError('Insufficient confirmed observation')
    if not np.isfinite(spacing_px) or spacing_px<=0:raise ValueError('Positive sampling interval required')
    samples=[]
    for arc in observation.get('arcs',[]):
        xy=np.asarray(arc['xy_px'],float)
        if xy.ndim!=2 or xy.shape[1]!=2 or not 2<=len(xy)<=4096 or not np.isfinite(xy).all():
            raise ValueError('Finite bounded open arc required')
        if np.any(xy<0) or np.any(xy>np.asarray(native_hw)[::-1]-1):raise ValueError('Arc outside native image')
        lengths=np.linalg.norm(np.diff(xy,axis=0),axis=1)
        for a,b,length in zip(xy[:-1],xy[1:],lengths):
            if length>1e-6:samples.extend(np.linspace(a,b,max(2,int(np.ceil(length/spacing_px))+1))[:-1])
        samples.append(xy[-1])
    points=np.asarray(samples,float).reshape(-1,2)
    if len(points):
        from skimage.measure import points_in_poly
        excluded=np.zeros(len(points),bool)
        for polygon in observation.get('excluded_polygons_xy_px',[]):
            polygon=np.asarray(polygon,float)
            if polygon.ndim!=2 or polygon.shape[1]!=2 or len(polygon)<3 or not np.isfinite(polygon).all():raise ValueError('Invalid exclusion')
            excluded|=points_in_poly(points,polygon)
        points=points[~excluded]
    if len(points)<2:raise ValueError('Insufficient non-excluded visible arc samples')
    return points


def unsigned_edge_loss(alpha,samples_xy,*,native_hw,blur_native_px):
    """Smoothed silhouette iso-.5 loss at actual observed arc coordinates only.

    Gaussian support is truncated at3sigma, so unknown pixels outside local
    windows have exactly no influence. Empty/filled silhouettes are not rewarded.
    This is a surrogate boundary objective, NOT Beer-Lambert intensity fitting.
    """
    import torch
    import torch.nn.functional as F
    if alpha.ndim!=2 or not bool(torch.isfinite(alpha).all()):raise ValueError('Finite alpha image required')
    xy=np.asarray(samples_xy,float)
    if xy.ndim!=2 or xy.shape[1]!=2 or len(xy)<2 or not np.isfinite(xy).all():raise ValueError('Nonempty finite arc samples required')
    native=np.asarray(native_hw,float)
    if native.shape!=(2,) or np.any(native<=0) or np.any(xy<0) or np.any(xy>native[::-1]-1):raise ValueError('Native coordinates required')
    if not np.isfinite(blur_native_px) or blur_native_px<=0:raise ValueError('Positive native blur scale required')
    image=alpha[None,None]
    # Separable finite Gaussian; replicate padding is not an anatomy observation.
    for dimension,scale in enumerate(np.asarray(alpha.shape)/native):
        sigma=max(.5,float(blur_native_px*scale));radius=int(np.ceil(3*sigma))
        x=torch.arange(-radius,radius+1,dtype=alpha.dtype,device=alpha.device)
        kernel=torch.exp(-x.square()/(2*sigma*sigma));kernel/=kernel.sum()
        if dimension==0:
            image=F.conv2d(F.pad(image,(0,0,radius,radius),mode='replicate'),kernel[None,None,:,None])
        else:image=F.conv2d(F.pad(image,(radius,radius,0,0),mode='replicate'),kernel[None,None,None,:])
    # Native integer pixel centres -> grid_sample align_corners=False exactly.
    normalized=2*(torch.as_tensor(xy,dtype=alpha.dtype,device=alpha.device)+.5)/torch.as_tensor(native[::-1].copy(),dtype=alpha.dtype,device=alpha.device)-1
    predicted=F.grid_sample(image,normalized[None,None],align_corners=False,padding_mode='border').reshape(-1)
    return F.smooth_l1_loss(predicted,torch.full_like(predicted,.5),beta=.1)


def temporal_penalty(parameters,times_seconds):
    import torch
    t=np.asarray(times_seconds,float)
    if t.ndim!=1 or len(t)!=len(parameters) or not np.isfinite(t).all() or np.any(np.diff(t)<=0):raise ValueError('Strict actual time sequence required')
    if len(t)<2 or parameters.shape[1]==0:return parameters.sum()*0
    dt=torch.as_tensor(np.diff(t),dtype=parameters.dtype,device=parameters.device)
    return ((parameters[1:]-parameters[:-1])/dt[:,None]).square().mean()


def mesh_step_valid(reference,candidate,faces):
    a=np.asarray(reference,float);b=np.asarray(candidate,float);f=np.asarray(faces)
    if a.shape!=b.shape or not np.isfinite(b).all():return dict(valid=False,reason='nonfinite_or_wrong_shape')
    first=np.cross(a[f[:,1]]-a[f[:,0]],a[f[:,2]]-a[f[:,0]])
    second=np.cross(b[f[:,1]]-b[f[:,0]],b[f[:,2]]-b[f[:,0]])
    area0=np.linalg.norm(first,axis=1);area1=np.linalg.norm(second,axis=1)
    ratio=area1/np.maximum(area0,1e-12)
    flip=np.sum(first*second,axis=1)<=0
    edges=np.concatenate([f[:,[0,1]],f[:,[1,2]],f[:,[2,0]]])
    old=np.linalg.norm(a[edges[:,1]]-a[edges[:,0]],axis=1)
    new=np.linalg.norm(b[edges[:,1]]-b[edges[:,0]],axis=1)
    edge_ratio=new/np.maximum(old,1e-12)
    good=bool(not flip.any() and ratio.min()>.15 and edge_ratio.min()>.5 and edge_ratio.max()<2.)
    return dict(valid=good,reason='ok' if good else 'face_flip_or_extreme_local_distortion',
        min_area_ratio=float(ratio.min()),min_edge_ratio=float(edge_ratio.min()),max_edge_ratio=float(edge_ratio.max()),
        flipped_faces=int(flip.sum()),self_intersection_checked=False)


def fit_provisional(mesh,camera,observations,config,*,steps_per_resolution=12,device='cpu',basis=None,progress=None,boundary_backend=None):
    import torch
    if config.mode=='diagnostic_conditional' and not config.allow_unaccepted_geometry:
        raise ValueError('Explicit unaccepted geometry permission required for patient fitting')
    if not observations:raise ValueError('Insufficient observations')
    if mesh is None or not mesh.report.get('optimization_ready',False):raise ValueError('Optimization-ready mesh required')
    ordered=sorted(observations,key=lambda o:o['frame_index']);indices=[o['frame_index'] for o in ordered]
    if len(set(indices))!=len(indices):raise ValueError('Duplicate frame')
    times=np.array([o['timestamp_ms'] for o in ordered],float)/1000
    if not np.isfinite(times).all() or np.any(np.diff(times)<=0):raise ValueError('Actual increasing times required')
    samples=[sample_observation(o,camera.image_hw) for o in ordered]
    if basis is None:basis=build_ffd_basis(mesh,camera)
    model=config.model;k={'static':0,'rigid_inplane':3,'ffd_inplane_radial':4+basis['rank']}[model]
    count=len(ordered);normalized=torch.zeros((count,k),device=device,requires_grad=bool(k))
    bounds=config.bounds;scale_extent=max(1-bounds['scale_min'],bounds['scale_max']-1)
    scales=np.array([bounds['translation_mm']]*2+[np.deg2rad(bounds['rotation_deg']),scale_extent]+[bounds['local_mm']]*basis['rank'],np.float32)[:k]
    scaling=torch.tensor(scales,device=device);faces=torch.as_tensor(mesh.faces,dtype=torch.int64,device=device)
    stage_steps=[steps_per_resolution]*len(config.resolutions) if type(steps_per_resolution) is int else list(steps_per_resolution)
    if len(stage_steps)!=len(config.resolutions) or any(type(n) is not int or n<1 for n in stage_steps):raise ValueError('Positive per-resolution epoch budget required')
    if any(n>config.optimizer_budget for n in stage_steps):raise ValueError('Epoch budget exceeds frozen configuration')
    blur_native=max(2.,min(camera.image_hw)/64)
    def vertices(j):return deform_vertices(mesh,camera,normalized[j]*scaling,model,basis)
    def data_at(j,size):
        if boundary_backend is not None:return boundary_backend(vertices(j),samples[j],size)
        alpha=render_soft(vertices(j),faces,camera,image_hw=(size,size))
        return unsigned_edge_loss(alpha,samples[j],native_hw=camera.image_hw,blur_native_px=blur_native)
    with torch.no_grad():before=float(torch.stack([data_at(j,config.resolutions[-1]) for j in range(count)]).mean())
    optimizer=torch.optim.Adam([normalized],lr=.04) if k else None
    history=[];rejected=0;started=time.perf_counter()
    if k:
        for size,epochs in zip(config.resolutions,stage_steps):
            for epoch in range(epochs):
                for j in range(count):
                    old=normalized.detach().clone();optimizer.zero_grad()
                    data=data_at(j,size)
                    loss=config.loss_weights['data']*data+config.loss_weights['amplitude']*normalized.square().mean()+config.loss_weights['temporal']*temporal_penalty(normalized,times)
                    if not bool(torch.isfinite(loss)):raise RuntimeError('Nonfinite diagnostic loss')
                    loss.backward()
                    if normalized.grad is None or not bool(torch.isfinite(normalized.grad).all()):raise RuntimeError('Missing/nonfinite differentiable gradient')
                    torch.nn.utils.clip_grad_norm_([normalized],10.)
                    optimizer.step()
                    with torch.no_grad():
                        normalized.clamp_(-1,1)
                        if model=='ffd_inplane_radial':
                            normalized[:,3].clamp_((bounds['scale_min']-1)/max(scale_extent,1e-12),(bounds['scale_max']-1)/max(scale_extent,1e-12))
                            vectors=torch.as_tensor(basis['vectors_lps'],dtype=torch.float32,device=device)
                            delta=torch.einsum('vdk,sk->svd',vectors,normalized[:,4:]*scaling[4:])
                            maximum=delta.norm(dim=2).amax(dim=1).clamp_min(max(bounds['local_mm'],1e-12))
                            normalized[:,4:]*=(bounds['local_mm']/maximum)[:,None]
                        valid=all(mesh_step_valid(mesh.vertices_lps_mm,vertices(t).cpu().numpy(),mesh.faces)['valid'] for t in range(count))
                        if not valid:
                            normalized.copy_(old);optimizer.state.clear();rejected+=1
                    history.append(dict(resolution=size,epoch=epoch,frame_index=indices[j],loss=float(loss.detach()),data_loss=float(data.detach()),step_rejected=not valid))
                if progress:progress(dict(model=model,resolution=size,epoch=epoch+1,epochs=epochs,rejected_steps=rejected))
    with torch.no_grad():
        after=float(torch.stack([data_at(j,config.resolutions[-1]) for j in range(count)]).mean())
        parameters=(normalized*scaling).cpu().numpy();predicted=np.stack([vertices(j).cpu().numpy() for j in range(count)])
    return dict(model=model,mode=config.mode,frame_indices=indices,times_seconds=times,parameters=parameters,
        vertices_lps_mm=predicted,faces=mesh.faces.copy(),history=history,data_loss_before=before,data_loss_after=after,
        optimization_seconds=time.perf_counter()-started,rejected_steps=rejected,
        boundary_loss=boundary_backend.name if boundary_backend is not None else 'provisional_unsigned_smoothed_alpha_iso_half_local_only',
        blur_native_px=blur_native if boundary_backend is None else None,
        samples_per_frame=[len(s) for s in samples],basis_rank=basis['rank'],camera_optimized=False,
        independent_depth_optimized=False,whole_foreground_supervised=False,all_frames_development_fit=True,
        independent_validation=False,clinical_validation=False,absolute_geometry_accepted=False,
        real_patient_mesh_fit_executed=config.mode=='diagnostic_conditional' and bool(k),heart_motion_ground_truth_validated=False,
        self_intersection_checked=False,patient_true_3d_tre_mm=None)
