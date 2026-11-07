"""Independent CPU binary triangle raster; integer pixels denote centres."""
from __future__ import annotations

import numpy as np
from .geometry import project_lps,resize_camera
from .contracts import MeshData


def render_hard(vertices_lps_mm,faces,camera,*,image_hw):
    # Validate faces before indexing; no hull approximation or shading.
    data=MeshData(vertices_lps_mm,faces,'0'*64,{})
    uv,_=project_lps(data.vertices_lps_mm,resize_camera(camera,tuple(image_hw)))
    height,width=image_hw
    image=np.zeros((height,width),dtype=bool)
    for triangle in uv[data.faces]:
        xmin=max(0,int(np.ceil(triangle[:,0].min())));xmax=min(width-1,int(np.floor(triangle[:,0].max())))
        ymin=max(0,int(np.ceil(triangle[:,1].min())));ymax=min(height-1,int(np.floor(triangle[:,1].max())))
        if xmin>xmax or ymin>ymax:continue
        a,b,c=triangle
        denominator=(b[1]-c[1])*(a[0]-c[0])+(c[0]-b[0])*(a[1]-c[1])
        if abs(denominator)<1e-12:continue
        x,y=np.meshgrid(np.arange(xmin,xmax+1),np.arange(ymin,ymax+1))
        w0=((b[1]-c[1])*(x-c[0])+(c[0]-b[0])*(y-c[1]))/denominator
        w1=((c[1]-a[1])*(x-c[0])+(a[0]-c[0])*(y-c[1]))/denominator
        image[ymin:ymax+1,xmin:xmax+1]|=(w0>=-1e-10)&(w1>=-1e-10)&((1-w0-w1)>=-1e-10)
    return image
