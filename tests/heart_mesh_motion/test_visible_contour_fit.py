"""True projected contour is independent of dense soft-alpha raster coverage."""
import importlib,json
from pathlib import Path
import numpy as np
import pytest
import torch
from scipy import ndimage
from scipy.spatial import cKDTree


def api():return importlib.import_module('ctdr_pipeline.heart_mesh_motion.visible_contour_fit')


def fixture():
    from ctdr_pipeline.heart_mesh_motion.synthetic_mesh import digital_fixture
    return digital_fixture()


def test_actual_projected_contour_matches_independent_digital_hard_boundary():
    from ctdr_pipeline.heart_mesh_motion.reference_raster import render_hard
    mesh,camera=fixture();v=torch.tensor(mesh.vertices_lps_mm,dtype=torch.float32)
    xy,detail=api().visible_contour_points(v,mesh.faces,camera)
    mask=render_hard(mesh.vertices_lps_mm,mesh.faces,camera,image_hw=camera.image_hw)
    edge=mask&~ndimage.binary_erosion(mask);y,x=np.where(edge)
    distance=cKDTree(np.c_[x,y]).query(xy.detach().numpy())[0]
    assert np.percentile(distance,95)<1.5
    assert detail['visibility_discrete'] is True
    assert len(xy)>8


def test_occluded_rear_surface_contour_is_not_an_observed_boundary():
    import trimesh
    from ctdr_pipeline.heart_mesh_motion.contracts import MeshData
    first=trimesh.creation.icosphere(subdivisions=1,radius=25);first.vertices+=[0,0,220]
    second=trimesh.creation.icosphere(subdivisions=1,radius=15);second.vertices+=[0,0,300]
    joined=trimesh.util.concatenate([first,second]);_,camera=fixture()
    xy,detail=api().visible_contour_points(torch.tensor(joined.vertices,dtype=torch.float32),joined.faces,camera)
    assert len(xy)>8
    assert np.all(np.asarray(detail['visible_edges'])<len(first.vertices))


def test_visible_front_ridge_inside_rear_silhouette_is_not_an_outer_boundary():
    import trimesh
    first=trimesh.creation.icosphere(subdivisions=1,radius=15);first.vertices+=[0,0,220]
    second=trimesh.creation.icosphere(subdivisions=1,radius=40);second.vertices+=[0,0,300]
    joined=trimesh.util.concatenate([first,second]);_,camera=fixture()
    _,detail=api().visible_contour_points(torch.tensor(joined.vertices,dtype=torch.float32),joined.faces,camera)
    assert np.all(np.asarray(detail['visible_edges'])>=len(first.vertices))


def test_contour_distance_backpropagates_correct_translation_direction():
    mesh,camera=fixture();backend=api().VisibleContourObjective(mesh,camera)
    truth=torch.tensor(mesh.vertices_lps_mm+np.array([4.,0,0]),dtype=torch.float32)
    target,_=api().visible_contour_points(truth,mesh.faces,camera)
    shift=torch.tensor(0.,requires_grad=True)
    v=torch.tensor(mesh.vertices_lps_mm,dtype=torch.float32)+torch.stack([shift,shift*0,shift*0])
    loss=backend(v,target.detach().numpy(),64);loss.backward()
    assert float(shift.grad)<0
    assert np.isfinite(float(shift.grad))


def test_full_density_patient_mesh_contour_matches_native_independent_projection():
    from ctdr_pipeline.heart_mesh_motion.contracts import CameraData
    from ctdr_pipeline.heart_mesh_motion.reference_raster import render_hard
    path=Path('artifacts/heart_mesh_dynamic_1234567/selected_heart_edges_s7_002')
    if not path.exists():pytest.skip('Local pinned hospital mesh not present')
    a=np.load(path/'mesh_reference.npz');d=json.loads((path/'selected_camera_unaccepted.json').read_text());d['image_hw']=tuple(d['image_hw']);camera=CameraData(**d)
    xy,_=api().visible_contour_points(torch.tensor(a['vertices_lps_mm'],dtype=torch.float32),a['faces'],camera)
    mask=render_hard(a['vertices_lps_mm'],a['faces'],camera,image_hw=(512,512));edge=mask&~ndimage.binary_erosion(mask);y,x=np.where(edge)
    distance=cKDTree(np.c_[x,y]).query(xy.detach().numpy())[0]
    assert np.percentile(distance,95)<2.
    reverse=cKDTree(xy.detach().numpy()).query(np.c_[x,y])[0]
    assert np.percentile(reverse,95)<2.


def test_existing_optimizer_can_consume_true_contour_objective_without_alpha_fallback():
    from ctdr_pipeline.heart_mesh_motion.provisional_edge_fit import fit_provisional
    from ctdr_pipeline.heart_mesh_motion.contracts import FitConfig
    mesh,camera=fixture();target,_=api().visible_contour_points(torch.tensor(mesh.vertices_lps_mm+[4.,0,0],dtype=torch.float32),mesh.faces,camera)
    # Small open arcs near the independent known translated geometry.
    xy=target.detach().numpy()[::4];arcs=[dict(xy_px=[p.tolist(),(p+[0,.2]).tolist()]) for p in xy]
    obs=[dict(frame_index=i,timestamp_ms=i*133.333,review_state='visible_confirmed',arcs=arcs,excluded_polygons_xy_px=[]) for i in range(3)]
    cfg=FitConfig(model='rigid_inplane',mode='synthetic',resolutions=(64,),optimizer_budget=8)
    fit=fit_provisional(mesh,camera,obs,cfg,steps_per_resolution=8,boundary_backend=api().VisibleContourObjective(mesh,camera))
    assert fit['data_loss_after']<fit['data_loss_before']
    assert fit['boundary_loss']=='visible_mesh_contour_distance_piecewise_differentiable'
    assert np.mean(fit['parameters'][:,0])>1.
