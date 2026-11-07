"""Explicit source/detector projection in native column/row coordinates."""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import numpy as np

from .contracts import CameraData
from .inputs import read_json_snapshot


def load_camera(calibration_path: Path, *, image_hw: tuple[int,int],expected_sha256: str|None=None) -> CameraData:
    report,_=read_json_snapshot(calibration_path,expected_sha256=expected_sha256)
    cal=report['calibration']
    detector=report['detector']
    if (detector['rows'],detector['columns'])!=tuple(image_hw):raise ValueError('Calibration/cine dimensions differ')
    # This work does not independently accept any inherited calibration.
    return CameraData(cal['source_lps_mm'],cal['u_lps'],cal['v_lps'],cal['n_lps'],
        cal['detector_center_lps_mm'],cal['principal_uv_px'],cal['pixel_spacing_row_col_mm'],image_hw,
        {'quality_status':report.get('quality_status'),'transferred_from':report.get('transferred_from'),
         'reported_quality_gate_passed':report.get('quality_gate_passed')},False)


def project_lps(vertices: np.ndarray,camera: CameraData) -> tuple[np.ndarray,np.ndarray]:
    points=np.asarray(vertices,dtype=np.float64)
    if points.ndim!=2 or points.shape[1]!=3 or not np.isfinite(points).all():raise ValueError('Finite Nx3 points required')
    rays=points-camera.source_lps_mm
    depth=rays@camera.n_lps
    if np.any(depth<=1e-8):raise ValueError('Point is on or behind the source plane')
    distance=float((camera.detector_center_lps_mm-camera.source_lps_mm)@camera.n_lps)
    xy=np.column_stack([distance*(rays@camera.u_lps)/depth/camera.spacing_row_col_mm[1]+camera.principal_xy_px[0],
        distance*(rays@camera.v_lps)/depth/camera.spacing_row_col_mm[0]+camera.principal_xy_px[1]])
    return xy,depth


def resize_camera(camera: CameraData,image_hw: tuple[int,int]) -> CameraData:
    if len(image_hw)!=2 or any(type(i) is not int or i<=0 for i in image_hw):raise ValueError('Invalid image dimensions')
    sy,sx=np.asarray(image_hw,dtype=float)/np.asarray(camera.image_hw)
    return replace(camera,principal_xy_px=(camera.principal_xy_px+.5)*[sx,sy]-.5,
        spacing_row_col_mm=camera.spacing_row_col_mm/[sy,sx],image_hw=tuple(image_hw))


def audit_projection(mesh,camera,cine,frame_indices: list[int]) -> dict:
    from .reference_raster import render_hard
    cine=np.asarray(cine)
    if cine.ndim!=3 or tuple(cine.shape[1:])!=camera.image_hw:raise ValueError('Cine/camera image dimensions differ')
    if any(type(i) is not int or not 0<=i<len(cine) for i in frame_indices):raise ValueError('Invalid audit frame')
    uv,depth=project_lps(mesh.vertices_lps_mm,camera)
    mask=render_hard(mesh.vertices_lps_mm,mesh.faces,camera,image_hw=camera.image_hw)
    h,w=camera.image_hw
    in_fov=(uv[:,0]>=0)&(uv[:,0]<w)&(uv[:,1]>=0)&(uv[:,1]<h)
    return {'frame_indices':frame_indices,'geometry_accepted':False,
        'measured_xa_heart_edge_error_px':None,'projected_foreground_pixels':int(mask.sum()),
        'projected_vertex_bounds_xy':[uv.min(0).tolist(),uv.max(0).tolist()],
        'fraction_vertices_in_fov':float(in_fov.mean()),'minimum_source_normal_depth_mm':float(depth.min()),
        'mask_touches_fov_border':bool(mask[0].any() or mask[-1].any() or mask[:,0].any() or mask[:,-1].any()),
        'interpretation':'CT model projection only; no measured XA heart-boundary agreement or geometry acceptance'}
