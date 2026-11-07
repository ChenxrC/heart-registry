"""Actually exercise a small rasterizer backward, never install on import."""
from __future__ import annotations

import importlib.metadata
import platform


def probe_runtime(device: str) -> dict:
    report={'python':platform.python_version(),'platform':platform.system(),'device_requested':device,
        'renderer_backward_verified':False,'status':'unavailable','clinical_validation':False}
    try:
        import torch
        report.update(torch=torch.__version__,torch_cuda=torch.version.cuda,cuda_available=torch.cuda.is_available())
        if device.startswith('cuda') and not torch.cuda.is_available():raise RuntimeError('CUDA PyTorch runtime unavailable')
        from pytorch3d.structures import Meshes
        from pytorch3d.renderer import MeshRenderer,MeshRasterizer,SoftSilhouetteShader,RasterizationSettings,FoVPerspectiveCameras
        report['pytorch3d']=importlib.metadata.version('pytorch3d')
        vertices=torch.tensor([[-.5,-.4,2.],[.6,-.4,2.],[.1,.7,2.]],device=device,requires_grad=True)
        mesh=Meshes(verts=[vertices],faces=[torch.tensor([[0,1,2]],device=device)])
        render=MeshRenderer(MeshRasterizer(cameras=FoVPerspectiveCameras(device=device),
            raster_settings=RasterizationSettings(image_size=32,blur_radius=.01,faces_per_pixel=4)),SoftSilhouetteShader())
        alpha=render(mesh)[0,...,3]
        loss=(alpha*torch.linspace(0,1,32,device=device)[None,:]).sum();loss.backward()
        good=vertices.grad is not None and bool(torch.isfinite(vertices.grad).all()) and float(vertices.grad.norm())>0
        if not good:raise RuntimeError('Rasterizer backward produced no finite nonzero gradient')
        report.update(status='available',renderer_backward_verified=True,gradient_norm=float(vertices.grad.norm()),reason=None)
    except (ImportError,RuntimeError,OSError,ValueError) as error:
        report['reason']=f'{type(error).__name__}: {error}'
    return report
