"""Piecewise-differentiable actual mesh contour; no dense soft-alpha surrogate.

Face-side and line-of-sight visibility are discrete, recomputed each iteration.
Selected contour samples remain torch expressions of actual LPS mesh vertices.
"""
from __future__ import annotations
import numpy as np
import trimesh


def visible_contour_points(vertices,faces,camera,*,topology=None,spacing_px=2.):
    import torch
    import vtk
    import pyvista as pv
    raw=vertices.detach().cpu().numpy().astype(float)
    if not np.isfinite(raw).all():raise ValueError('Nonfinite contour geometry')
    if topology is None:
        surface=trimesh.Trimesh(raw,faces,process=False)
        topology=(surface.face_adjacency.copy(),surface.face_adjacency_edges.copy())
    pairs,edges=topology;tri=raw[faces]
    normals=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0])
    side=np.sum(normals*(tri.mean(axis=1)-camera.source_lps_mm),axis=1)
    candidate=edges[side[pairs[:,0]]*side[pairs[:,1]]<=0]
    poly=pv.PolyData(raw,np.c_[np.full(len(faces),3),faces].ravel())
    locator=vtk.vtkStaticCellLocator();locator.SetDataSet(poly);locator.BuildLocator()
    source=np.asarray(camera.source_lps_mm);visible=[]
    from .geometry import project_lps
    raw_xy,_=project_lps(raw,camera)
    t=vtk.mutable(0.);sub=vtk.mutable(0);cell_id=vtk.mutable(0);cell=vtk.vtkGenericCell()
    for edge in candidate:
        screen_delta=raw_xy[edge[1]]-raw_xy[edge[0]];screen_length=np.linalg.norm(screen_delta)
        if screen_length<1e-8:continue
        normal=np.array([-screen_delta[1],screen_delta[0]])/screen_length
        mid_xy=raw_xy[edge].mean(0);occupied=[]
        # A front-visible ridge is NOT an outer silhouette if deeper tissue fills
        # both sides. Probe true ray occupancy on each side of its projected edge.
        for sign in (-1,1):
            q=mid_xy+sign*.5*normal
            detector=camera.detector_center_lps_mm+(q[0]-camera.principal_xy_px[0])*camera.spacing_row_col_mm[1]*camera.u_lps+(q[1]-camera.principal_xy_px[1])*camera.spacing_row_col_mm[0]*camera.v_lps
            occupied.append(bool(locator.IntersectWithLine(source,detector,1e-6,t,[0.,0.,0.],[0.,0.,0.],sub,cell_id,cell)))
        if occupied[0]==occupied[1]:continue
        midpoint=raw[edge].mean(0);direction=midpoint-source;distance=np.linalg.norm(direction)
        endpoint=midpoint+direction/distance*.25;hit=[0.,0.,0.];pcoord=[0.,0.,0.]
        found=locator.IntersectWithLine(source,endpoint,1e-6,t,hit,pcoord,sub,cell_id,cell)
        if not found or np.linalg.norm(np.asarray(hit)-source)>=distance-.25:visible.append(edge)
    if not visible:raise ValueError('Insufficient visible contour')
    tensor=lambda value:torch.as_tensor(value,dtype=vertices.dtype,device=vertices.device)
    def project(points):
        relative=points-tensor(camera.source_lps_mm);depth=relative@tensor(camera.n_lps)
        if bool((depth<=0).any()):raise ValueError('Contour behind projection source')
        distance=float((camera.detector_center_lps_mm-camera.source_lps_mm)@camera.n_lps)
        return torch.stack([camera.principal_xy_px[0]+distance*(relative@tensor(camera.u_lps))/depth/camera.spacing_row_col_mm[1],
                            camera.principal_xy_px[1]+distance*(relative@tensor(camera.v_lps))/depth/camera.spacing_row_col_mm[0]],dim=1)
    endpoints=project(vertices);xy_detached=endpoints.detach().cpu().numpy();parts=[]
    for a,b in visible:
        count=max(2,int(np.ceil(np.linalg.norm(xy_detached[b]-xy_detached[a])/spacing_px))+1)
        lam=torch.linspace(0,1,count,dtype=vertices.dtype,device=vertices.device)
        parts.append(project(vertices[a][None]*(1-lam[:,None])+vertices[b][None]*lam[:,None]))
    points=torch.cat(parts);detached=points.detach().cpu().numpy()
    keep=np.all(detached>=0,axis=1)&np.all(detached<=np.asarray(camera.image_hw)[::-1]-1,axis=1)
    points=points[torch.as_tensor(keep,device=vertices.device)]
    if len(points)<8:raise ValueError('Insufficient in-FOV model contour')
    return points,dict(visible_edges=np.asarray(visible).tolist(),visibility_discrete=True,
        visibility_method='front_back_edges_two_sided_ray_occupancy_and_first_hit',ray_tolerance_mm=.25,
        sample_count=len(points),independent_depth_recovered=False)


class VisibleContourObjective:
    name='visible_mesh_contour_distance_piecewise_differentiable'
    def __init__(self,mesh,camera):
        source=trimesh.Trimesh(mesh.vertices_lps_mm,mesh.faces,process=False)
        self.topology=(source.face_adjacency.copy(),source.face_adjacency_edges.copy())
        self.faces=mesh.faces;self.camera=camera

    def __call__(self,vertices,samples_xy,size):
        import torch
        import torch.nn.functional as F
        xy,_=visible_contour_points(vertices,self.faces,self.camera,topology=self.topology,
            spacing_px=max(1.,min(self.camera.image_hw)/size/2))
        observed=torch.as_tensor(samples_xy,dtype=vertices.dtype,device=vertices.device)
        distance=torch.cdist(observed[None],xy[None])[0]
        forward=distance.min(dim=1).values.clamp_max(32.)
        backward=distance.min(dim=0).values
        local=backward[backward.detach()<=32.]
        if not len(local):raise ValueError('No model contour near actual observed arcs')
        norm=max(1.,min(self.camera.image_hw)/64.)
        return (F.smooth_l1_loss(forward,torch.zeros_like(forward),beta=2.)+
                F.smooth_l1_loss(local,torch.zeros_like(local),beta=2.))/(2*norm)
