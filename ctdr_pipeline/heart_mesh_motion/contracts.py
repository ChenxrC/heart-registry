"""Validated numerical contracts, independent of any differentiable backend."""
from __future__ import annotations

from dataclasses import dataclass, field
import re
import numpy as np


def array(value, shape, name):
    result=np.asarray(value,dtype=np.float64)
    if result.shape != shape or not np.isfinite(result).all():
        raise ValueError(f'{name} must be finite with shape {shape}')
    return result.copy()


@dataclass(frozen=True)
class MeshData:
    vertices_lps_mm: np.ndarray
    faces: np.ndarray
    source_sha256: str
    report: dict=field(default_factory=dict)

    def __post_init__(self):
        v=np.asarray(self.vertices_lps_mm,dtype=np.float64)
        f=np.asarray(self.faces)
        if v.ndim!=2 or v.shape[1]!=3 or len(v)<3 or not np.isfinite(v).all():
            raise ValueError('vertices must be finite Vx3')
        if f.ndim!=2 or f.shape[1]!=3 or len(f)<1 or f.dtype.kind not in 'iu' or f.min()<0 or f.max()>=len(v):
            raise ValueError('faces must be valid integer Fx3 indices')
        if np.any(np.diff(np.sort(f,axis=1),axis=1)==0):raise ValueError('Repeated triangle vertex')
        if not isinstance(self.source_sha256,str) or re.fullmatch('[0-9a-f]{64}',self.source_sha256) is None:
            raise ValueError('Mesh requires a SHA256 source identity')
        object.__setattr__(self,'vertices_lps_mm',v.copy())
        object.__setattr__(self,'faces',f.astype(np.int64,copy=True))


@dataclass(frozen=True)
class CameraData:
    source_lps_mm: np.ndarray
    u_lps: np.ndarray
    v_lps: np.ndarray
    n_lps: np.ndarray
    detector_center_lps_mm: np.ndarray
    principal_xy_px: np.ndarray
    spacing_row_col_mm: np.ndarray
    image_hw: tuple[int,int]
    provenance: dict
    geometry_accepted: bool=False

    def __post_init__(self):
        for name in ('source_lps_mm','u_lps','v_lps','n_lps','detector_center_lps_mm'):
            object.__setattr__(self,name,array(getattr(self,name),(3,),name))
        for name in ('principal_xy_px','spacing_row_col_mm'):
            object.__setattr__(self,name,array(getattr(self,name),(2,),name))
        if len(self.image_hw)!=2 or any(type(x) is not int or x<=0 for x in self.image_hw):
            raise ValueError('image_hw must contain exact positive integers')
        if type(self.geometry_accepted) is not bool:raise ValueError('geometry_accepted must be boolean')
        basis=np.column_stack([self.u_lps,self.v_lps,self.n_lps])
        if not np.allclose(basis.T@basis,np.eye(3),atol=1e-6) or np.linalg.det(basis)<0.999999:
            raise ValueError('camera basis must be orthonormal and right handed')
        if np.any(self.spacing_row_col_mm<=0) or np.dot(self.detector_center_lps_mm-self.source_lps_mm,self.n_lps)<=0:
            raise ValueError('camera spacing and detector depth must be positive')
        if np.linalg.norm(np.cross(self.detector_center_lps_mm-self.source_lps_mm,self.n_lps))>1e-4:
            raise ValueError('detector center must lie on its normal from source')
        object.__setattr__(self,'image_hw',tuple(self.image_hw))

    def to_dict(self):
        return {k:(v.tolist() if isinstance(v,np.ndarray) else v) for k,v in vars(self).items()}


@dataclass(frozen=True)
class FitConfig:
    model: str='ffd_inplane_radial'
    mode: str='synthetic'
    seed: int=20261017
    resolutions: tuple[int,...]=(128,256)
    optimizer_budget: int=80
    bounds: dict=field(default_factory=lambda:dict(translation_mm=20.,rotation_deg=10.,scale_min=.90,scale_max=1.10,local_mm=10.))
    loss_weights: dict=field(default_factory=lambda:dict(data=1.,amplitude=.001,temporal=.01))
    split_id: str='blocked_middle_third'
    allow_unaccepted_geometry: bool=False
    allow_unreviewed_edges: bool=False

    def __post_init__(self):
        if self.model not in ('static','rigid_inplane','ffd_inplane_radial'):raise ValueError('Unknown model')
        if self.mode not in ('synthetic','diagnostic_conditional'):raise ValueError('Unknown mode')
        if type(self.seed) is not int or type(self.optimizer_budget) is not int or self.optimizer_budget<1:raise ValueError('Invalid seed/budget')
        if not self.resolutions or any(type(i) is not int or i<16 for i in self.resolutions):raise ValueError('Invalid resolutions')
        required={'translation_mm','rotation_deg','scale_min','scale_max','local_mm'}
        if set(self.bounds)!=required or not all(type(v) in (int,float) and np.isfinite(v) for v in self.bounds.values()):
            raise ValueError('Invalid bounds')
        b=self.bounds
        if min(b['translation_mm'],b['rotation_deg'],b['local_mm'])<0 or not 0<b['scale_min']<=1<=b['scale_max']:
            raise ValueError('Invalid parameter bounds')
        if set(self.loss_weights)!={'data','amplitude','temporal'} or not all(type(v) in (int,float) and np.isfinite(v) and v>=0 for v in self.loss_weights.values()):
            raise ValueError('Invalid loss weights')
