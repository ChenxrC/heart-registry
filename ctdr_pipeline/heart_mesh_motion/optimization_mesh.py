"""Explicit opt-in mesh preparation, separate from the unchanged display gate."""
from __future__ import annotations

import time
import numpy as np
import trimesh
from scipy import ndimage

from .contracts import MeshData
from .mesh_quality import audit_optimization_geometry
from .reference_raster import render_hard


def _edge(mask):
    edge=mask & ~ndimage.binary_erosion(mask,border_value=1)
    edge[[0,-1]]=False;edge[:,[0,-1]]=False
    return edge


def projection_fidelity(reference: MeshData,candidate: MeshData,cameras,*,reference_masks=None) -> dict:
    if not cameras:raise ValueError('At least one explicit camera required')
    reports=[]
    for index,camera in enumerate(cameras):
        first=render_hard(reference.vertices_lps_mm,reference.faces,camera,image_hw=camera.image_hw) if reference_masks is None else reference_masks[index]
        second=render_hard(candidate.vertices_lps_mm,candidate.faces,camera,image_hw=camera.image_hw)
        a,b=_edge(first),_edge(second)
        if min(a.sum(),b.sum())<8:
            reports.append({'accepted':False,'symmetric_edge_distance_p95_px':None,'reason':'insufficient_visible_model_edge'})
            continue
        distances=np.r_[ndimage.distance_transform_edt(~a)[b],ndimage.distance_transform_edt(~b)[a]]
        p95=float(np.percentile(distances,95))
        reports.append({'accepted':p95<=1.,'symmetric_edge_distance_p95_px':p95,
            'native_image_hw':list(camera.image_hw),'is_xa_annotation_error':False,
            'reference_edge_pixels':int(a.sum()),'candidate_edge_pixels':int(b.sum())})
    return {'accepted':all(v['accepted'] for v in reports),'threshold_native_px':1.,'views':reports,
        'interpretation':'Model simplification consistency under fixed cameras, NOT actual XA alignment.'}


def prepare_optimization_mesh(source: MeshData,cameras,*,max_faces=20000,device='cpu',reference_masks=None) -> MeshData:
    if type(max_faces) is not int or max_faces<4:raise ValueError('Invalid face budget')
    started=time.perf_counter()
    reference=trimesh.Trimesh(source.vertices_lps_mm,source.faces,process=False)
    candidate=reference.copy()
    if len(reference.faces)>max_faces:
        import pyvista as pv
        poly=pv.PolyData(reference.vertices,np.c_[np.full(len(reference.faces),3),reference.faces].ravel())
        reduced=poly.decimate(1-max_faces/len(reference.faces),volume_preservation=True)
        candidate=trimesh.Trimesh(np.asarray(reduced.points),np.asarray(reduced.faces).reshape(-1,4)[:,1:],process=False)
    quality=audit_optimization_geometry(reference,candidate)
    temporary=MeshData(candidate.vertices,candidate.faces,source.source_sha256,{})
    projection=projection_fidelity(source,temporary,cameras,reference_masks=reference_masks)
    backward={'verified':False,'device':device,'gradient_norm':None}
    try:
        import torch
        from .renderer import render_soft
        vertices=torch.tensor(candidate.vertices,dtype=torch.float32,device=device,requires_grad=True)
        faces=torch.tensor(candidate.faces,dtype=torch.int64,device=device)
        alpha=render_soft(vertices,faces,cameras[0],image_hw=(64,64))
        loss=(alpha*torch.arange(64,dtype=torch.float32,device=device)[None,:]).sum();loss.backward()
        good=vertices.grad is not None and bool(torch.isfinite(vertices.grad).all()) and float(vertices.grad.norm())>0
        backward.update(verified=good,gradient_norm=float(vertices.grad.norm()) if good else None)
    except (ImportError,RuntimeError,ValueError,OSError) as error:
        backward['reason']=f'{type(error).__name__}: {error}'
    ready=quality['geometric_accepted'] and projection['accepted'] and backward['verified'] and len(candidate.faces)<=max_faces
    report={'surface_kind':'registration_optimization','profile':'optimization_mesh_v1','optimization_ready':bool(ready),
        'geometry_quality':quality,'projection_fidelity':projection,'backward_probe':backward,
        'registration_faces':len(reference.faces),'optimization_faces':len(candidate.faces),
        'elapsed_seconds':time.perf_counter()-started,'coordinate_system':'DICOM_patient_LPS_mm',
        'clinical_validation':False,'absolute_geometry_accepted':False,'source_anatomy_verified':False,
        'old_display_rules_modified':False,'face_budget':max_faces}
    return MeshData(candidate.vertices,candidate.faces,source.source_sha256,report)
