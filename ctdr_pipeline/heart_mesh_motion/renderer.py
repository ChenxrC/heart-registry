"""PyTorch3D adapter from LPS/OpenCV pixel camera; no hard-render fallback."""
from __future__ import annotations

import numpy as np
from .geometry import resize_camera


def render_soft(vertices_lps_mm,faces,camera,*,image_hw):
    try:
        import torch
        from pytorch3d.structures import Meshes
        from pytorch3d.utils import cameras_from_opencv_projection
        from pytorch3d.renderer import MeshRenderer,MeshRasterizer,SoftSilhouetteShader,RasterizationSettings,BlendParams
    except ImportError as error:
        raise RuntimeError('PyTorch3D compiled rasterizer is required; no fallback animation') from error
    if not isinstance(vertices_lps_mm,torch.Tensor) or vertices_lps_mm.ndim!=2 or vertices_lps_mm.shape[1]!=3:
        raise ValueError('vertices must be a Vx3 torch Tensor')
    if not torch.isfinite(vertices_lps_mm).all():raise ValueError('Nonfinite mesh')
    camera=resize_camera(camera,tuple(image_hw))
    device,dtype=vertices_lps_mm.device,vertices_lps_mm.dtype
    if dtype!=torch.float32:raise ValueError('PyTorch3D rasterization requires float32 vertices')
    tensor=lambda a:torch.as_tensor(a,dtype=dtype,device=device)
    rotation=tensor(np.stack([camera.u_lps,camera.v_lps,camera.n_lps]))
    translation=-(rotation@tensor(camera.source_lps_mm))
    depth=vertices_lps_mm@rotation[2]+translation[2]
    if bool((depth<=1e-6).any()):raise ValueError('Mesh on or behind camera source')
    distance=float((camera.detector_center_lps_mm-camera.source_lps_mm)@camera.n_lps)
    # Our DICOM/OpenCV coordinates put pixel centres at integer indices. The
    # PyTorch3D raster lattice samples screen-boundary coordinates at i+0.5.
    # Convert conventions here AFTER resizing; this is not an empirical pose shift.
    principal=camera.principal_xy_px+.5
    intrinsic=tensor([[distance/camera.spacing_row_col_mm[1],0,principal[0]],
        [0,distance/camera.spacing_row_col_mm[0],principal[1]],[0,0,1]])
    # Official conversion handles +left/+up PyTorch3D vs +right/+down OpenCV.
    cameras=cameras_from_opencv_projection(rotation[None],translation[None],intrinsic[None],tensor([image_hw]))
    sigma=.0005
    raster=MeshRasterizer(cameras=cameras,raster_settings=RasterizationSettings(
        image_size=tuple(image_hw),blur_radius=float(np.log(1/1e-4-1)*sigma),faces_per_pixel=8,cull_backfaces=False))
    render=MeshRenderer(raster,SoftSilhouetteShader(blend_params=BlendParams(sigma=sigma)))
    mesh=Meshes(verts=[vertices_lps_mm],faces=[faces.to(device=device,dtype=torch.int64)])
    return render(mesh)[0,...,3]
