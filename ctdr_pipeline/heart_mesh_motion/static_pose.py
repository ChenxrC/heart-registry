"""Support-only restricted rigid initialization; never accepts patient geometry."""
from __future__ import annotations

import numpy as np
from .bone_projection import render_bone


def support_background(cine,support):
    frames=np.asarray(cine)
    indices=np.asarray(support)
    if frames.ndim!=3 or indices.ndim!=1 or len(indices)<2 or indices.dtype.kind not in 'iu' or indices.min()<0 or indices.max()>=len(frames):
        raise ValueError('Invalid support frame indices')
    selected=frames[indices].astype(np.float32)
    if not np.isfinite(selected).all():raise ValueError('Nonfinite support cine')
    target=255.-np.median(selected,axis=0)
    variation=np.std(selected,axis=0)
    weight=1/(1+(variation/10.)**2)
    weight[:4]=0;weight[-4:]=0;weight[:,:4]=0;weight[:,-4:]=0
    return target.astype(np.float32),weight.astype(np.float32)


def fit_static_translation(volume,camera,target,weight,*,bound_mm=80.,grid_step_mm=20.,steps=30,resolution=64,samples=96,device='cpu'):
    import torch
    import torch.nn.functional as F
    from ctdr_pipeline.registration.objectives import gradient_magnitude
    if not np.isfinite([bound_mm,grid_step_mm]).all() or min(bound_mm,grid_step_mm)<=0 or grid_step_mm>bound_mm:raise ValueError('Invalid search bounds')
    if type(steps) is not int or steps<0 or type(resolution) is not int or resolution<16:raise ValueError('Invalid static fitting budget')
    a=np.asarray(target,dtype=np.float32);w=np.asarray(weight,dtype=np.float32)
    if a.shape!=camera.image_hw or w.shape!=a.shape or not np.isfinite(a).all() or not np.isfinite(w).all() or (w<0).any():
        raise ValueError('Target/weight/camera dimensions or values invalid')
    base={'absolute_geometry_accepted':False,'clinical_validation':False,'patient_true_3d_tre_mm':None,
        'optimized_dofs':['translation_u_mm','translation_v_mm'],'locked_dofs':['translation_n','rotation','intrinsics','heart_shape'],
        'objective':'0.3 intensity NCC + 0.7 gradient-magnitude NCC; proxy bone projection',
        'independent_bone_landmark_validation':False,'density_is_proxy':True}
    if w.sum()<16 or np.std(a[w>0])<1e-5:
        return {**base,'status':'insufficient_observations','ct_to_xa_lps_matrix':None,'translation_uv_mm':None}
    if not np.any(volume.density_zyx>0):
        return {**base,'status':'insufficient_projection','ct_to_xa_lps_matrix':None,'translation_uv_mm':None}
    tensor=lambda x:torch.as_tensor(x,dtype=torch.float32,device=device)
    target_t=F.interpolate(tensor(a)[None,None],size=(resolution,resolution),mode='bilinear',align_corners=False)
    weights=F.interpolate(tensor(w)[None,None],size=(resolution,resolution),mode='bilinear',align_corners=False)
    total_weight=weights.sum()
    def weighted_variance(image):
        mean=(image*weights).sum()/total_weight.clamp_min(1e-6)
        return (weights*(image-mean).square()).sum()/total_weight.clamp_min(1e-6)
    base['effective_observations']={'resolution':resolution,'weight_sum':float(total_weight),
        'target_weighted_variance':float(weighted_variance(target_t)),
        'minimum_weight_sum':16.,'minimum_weighted_variance':1e-8}
    if float(total_weight)<16 or float(weighted_variance(target_t))<=1e-8:
        return {**base,'status':'insufficient_observations','ct_to_xa_lps_matrix':None,'translation_uv_mm':None}
    target_gradient=gradient_magnitude(target_t)
    def safe_ncc(first,second):
        total=weights.sum().clamp_min(1e-6)
        aa=first-(first*weights).sum()/total;bb=second-(second*weights).sum()/total
        denominator=((weights*aa.square()).sum()*(weights*bb.square()).sum()).clamp_min(1e-12).sqrt()
        return (weights*aa*bb).sum()/denominator
    def score(parameters):
        projected=render_bone(volume,camera,parameters,image_hw=(resolution,resolution),samples=samples)[None,None]
        if float(weighted_variance(projected).detach())<=1e-8:return None
        return .3*safe_ncc(projected,target_t)+.7*safe_ncc(gradient_magnitude(projected),target_gradient)
    grid=np.unique(np.r_[np.arange(-bound_mm,bound_mm+.001,grid_step_mm),0.,bound_mm])
    candidates=[]
    with torch.no_grad():
        initial_value=score(tensor([0,0]));initial=float(initial_value) if initial_value is not None else None
        for u in grid:
            for v in grid:
                value=score(tensor([u,v]))
                if value is not None and np.isfinite(float(value)):candidates.append((float(value),float(u),float(v)))
    if not candidates:return {**base,'status':'insufficient_projection','ct_to_xa_lps_matrix':None,'translation_uv_mm':None}
    candidates.sort(reverse=True);best_value,u,v=candidates[0]
    best=tensor([u,v]);p=torch.nn.Parameter(best.clone());opt=torch.optim.Adam([p],lr=1.0)
    history=[]
    for index in range(steps):
        opt.zero_grad();value=score(p)
        if value is None:break
        if not bool(torch.isfinite(value)):raise RuntimeError('Nonfinite static objective')
        if float(value.detach())>best_value:best_value=float(value.detach());best=p.detach().clone()
        (-value).backward()
        if p.grad is None or not bool(torch.isfinite(p.grad).all()):raise RuntimeError('Invalid static gradient')
        history.append({'iteration':index,'score':float(value.detach()),'translation_uv_mm':p.detach().cpu().tolist()})
        opt.step()
        with torch.no_grad():p.clamp_(-bound_mm,bound_mm)
    with torch.no_grad():
        last=score(p)
        if last is not None and float(last)>best_value:best_value=float(last);best=p.detach().clone()
    result=best.cpu().numpy();matrix=np.eye(4);matrix[:3,3]=result[0]*camera.u_lps+result[1]*camera.v_lps
    at_bound=bool(np.any(np.abs(result)>=bound_mm-.5))
    return {**base,'status':'candidate_at_search_boundary' if at_bound else 'restricted_rigid_candidate_requires_review',
        'translation_uv_mm':result.tolist(),'ct_to_xa_lps_matrix':matrix.tolist(),'initial_score':initial,'final_score':best_value,
        'score_improvement':best_value-initial if initial is not None else None,'at_search_boundary':at_bound,'history':history,
        'coarse_top_candidates':[list(row) for row in candidates[:5]],
        'config':dict(bound_mm=bound_mm,grid_step_mm=grid_step_mm,steps=steps,resolution=resolution,samples=samples),
        'interpretation':'Image similarity candidate only; no unique depth/rotation, no independent anatomy acceptance.'}
