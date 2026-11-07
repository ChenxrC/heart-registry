"""Explicit LPS ray marching of a bone-density proxy, not a diagnostic X-ray."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import tempfile
import numpy as np

from .geometry import resize_camera
from .inputs import read_bytes_snapshot,sha256


@dataclass(frozen=True)
class BoneVolume:
    density_zyx: np.ndarray
    voxel_to_lps: np.ndarray
    source_sha256: str
    report: dict

    def __post_init__(self):
        a=np.asarray(self.density_zyx,dtype=np.float32);m=np.asarray(self.voxel_to_lps,dtype=np.float64)
        if a.ndim!=3 or min(a.shape)<2 or not np.isfinite(a).all() or (a<0).any():raise ValueError('Invalid bone volume')
        if m.shape!=(4,4) or not np.isfinite(m).all() or not np.allclose(m[3],[0,0,0,1]) or abs(np.linalg.det(m[:3,:3]))<1e-9:
            raise ValueError('Invalid LPS affine')
        object.__setattr__(self,'density_zyx',a.copy());object.__setattr__(self,'voxel_to_lps',m.copy())


def load_bone_volume(path: Path,*,expected_sha256: str,sampling_mm=3.) -> BoneVolume:
    import SimpleITK as sitk
    if not np.isfinite(sampling_mm) or sampling_mm<=0:raise ValueError('Invalid CT sampling')
    content,digest=read_bytes_snapshot(path,expected_sha256=expected_sha256)
    scratch=Path(__file__).resolve().parents[2]/'tmp';scratch.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='heart_ct_snapshot_',dir=scratch) as folder:
        snapshot=Path(folder)/'ct.nii.gz';snapshot.write_bytes(content)
        image=sitk.ReadImage(str(snapshot),sitk.sitkFloat32)
    size=np.asarray(image.GetSize());spacing=np.asarray(image.GetSpacing())
    new_size=np.maximum(2,np.ceil((size-1)*spacing/sampling_mm).astype(int)+1)
    new_spacing=(size-1)*spacing/(new_size-1)
    sampled=sitk.Resample(image,[int(v) for v in new_size],sitk.Transform(),sitk.sitkLinear,
        image.GetOrigin(),new_spacing.tolist(),image.GetDirection(),-1024.,sitk.sitkFloat32)
    hu=sitk.GetArrayFromImage(sampled)
    density=np.clip((hu-150.)/1000.,0,3).astype(np.float32)
    matrix=np.eye(4);matrix[:3,:3]=np.array(sampled.GetDirection()).reshape(3,3)@np.diag(sampled.GetSpacing())
    matrix[:3,3]=sampled.GetOrigin()
    if sha256(path)!=digest:raise ValueError('CT source changed during extraction')
    return BoneVolume(density,matrix,digest,{'shape_zyx':list(density.shape),'spacing_xyz_mm':list(sampled.GetSpacing()),
        'density_definition':'clip((HU-150)/1000,0,3)','not_quantitative_attenuation':True,'coordinate_system':'DICOM_LPS_mm'})


def render_bone(volume: BoneVolume,camera,translation_uv,*,image_hw=(64,64),samples=96):
    import torch
    import torch.nn.functional as F
    if type(samples) is not int or samples<8:raise ValueError('At least8 ray samples required')
    p=translation_uv
    if p.shape!=(2,) or not bool(torch.isfinite(p).all()):raise ValueError('Two finite rigid translations required')
    c=resize_camera(camera,tuple(image_hw));dtype,device=p.dtype,p.device
    tensor=lambda x:torch.as_tensor(x,dtype=dtype,device=device)
    yy,xx=torch.meshgrid(torch.arange(image_hw[0],dtype=dtype,device=device),torch.arange(image_hw[1],dtype=dtype,device=device),indexing='ij')
    offset=p[0]*tensor(c.u_lps)+p[1]*tensor(c.v_lps)
    source=tensor(c.source_lps_mm)-offset
    detector=tensor(c.detector_center_lps_mm)+(xx[...,None]-c.principal_xy_px[0])*c.spacing_row_col_mm[1]*tensor(c.u_lps)
    detector=detector+(yy[...,None]-c.principal_xy_px[1])*c.spacing_row_col_mm[0]*tensor(c.v_lps)-offset
    rays=detector-source
    inverse=tensor(np.linalg.inv(volume.voxel_to_lps))
    start=source@inverse[:3,:3].T+inverse[:3,3]
    direction=rays@inverse[:3,:3].T
    sizes=tensor(np.array(volume.density_zyx.shape[::-1])-1)
    parallel=direction.abs()<1e-8
    safe=torch.where(parallel,torch.ones_like(direction),direction)
    aa=-start/safe;bb=(sizes-start)/safe
    lower=torch.where(parallel,torch.full_like(aa,-1e6),torch.minimum(aa,bb))
    upper=torch.where(parallel,torch.full_like(bb,1e6),torch.maximum(aa,bb))
    valid_parallel=~(parallel & ((start<0)|(start>sizes))).any(-1)
    near=lower.amax(-1).clamp_min(0);far=upper.amin(-1).clamp_max(1)
    span=(far-near).clamp_min(0)*valid_parallel.to(dtype)
    fractions=(torch.arange(samples,dtype=dtype,device=device)+.5)/samples
    t=near[...,None]+span[...,None]*fractions
    ijk=start+direction[...,None,:]*t[...,None]
    grid=(2*ijk/sizes-1)[None]
    density=tensor(volume.density_zyx)[None,None]
    values=F.grid_sample(density,grid,mode='bilinear',padding_mode='zeros',align_corners=True)[0,0]
    return values.mean(-1)*span*rays.norm(dim=-1)
