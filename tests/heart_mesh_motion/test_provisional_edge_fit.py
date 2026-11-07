"""Unsigned diagnostic contour fitting; not a patient-accuracy benchmark."""
import importlib
import numpy as np
import pytest
import torch


def api():return importlib.import_module('ctdr_pipeline.heart_mesh_motion.provisional_edge_fit')


def observation(points=None):
    return dict(frame_index=0,timestamp_ms=0.,review_state='visible_confirmed',
        arcs=[dict(xy_px=points or [[15.,8.],[15.,23.]],inward_side_point_xy_px=None)],
        excluded_polygons_xy_px=[])


def test_partial_arc_sampling_does_not_close_or_fill_the_heart():
    o=observation([[5.,5.],[5.,15.],[15.,15.]])
    samples=api().sample_observation(o,(32,32),spacing_px=2.)
    assert np.all((samples[:,0]==5)|(samples[:,1]==15))
    assert not np.any(np.all(samples==[10,10],axis=1))


def test_not_visible_or_fully_excluded_is_insufficient_not_a_zero_loss():
    o=observation();o['review_state']='not_visible'
    with pytest.raises(ValueError):api().sample_observation(o,(32,32))
    o=observation();o['excluded_polygons_xy_px']=[[[0,0],[31,0],[31,31],[0,31]]]
    with pytest.raises(ValueError):api().sample_observation(o,(32,32))


def test_unsigned_loss_ignores_unknown_pixels_outside_local_observation_support():
    alpha=torch.zeros((32,32));alpha[:,0:16]=1
    samples=api().sample_observation(observation(),(32,32))
    first=api().unsigned_edge_loss(alpha,samples,native_hw=(32,32),blur_native_px=1.)
    other=alpha.clone();other[:,:3]=.4;other[:,29:]=.9
    second=api().unsigned_edge_loss(other,samples,native_hw=(32,32),blur_native_px=1.)
    assert torch.allclose(first,second)
    assert first < api().unsigned_edge_loss(torch.zeros_like(alpha),samples,native_hw=(32,32),blur_native_px=1.)
    assert first < api().unsigned_edge_loss(torch.ones_like(alpha),samples,native_hw=(32,32),blur_native_px=1.)


def test_unsigned_loss_backpropagates_with_native_pixel_centre_scaling():
    y,x=torch.meshgrid(torch.arange(32),torch.arange(32),indexing='ij')
    shift=torch.tensor(1.,requires_grad=True)
    alpha=torch.sigmoid((16+shift-x.float())/.7)
    samples=np.array([[29.,15.],[29.,31.],[29.,45.]])
    loss=api().unsigned_edge_loss(alpha,samples,native_hw=(64,64),blur_native_px=2.)
    loss.backward()
    assert float(shift.grad)>0
    assert np.isfinite(float(shift.grad))


def test_temporal_penalty_uses_actual_time_not_frame_number_and_is_not_periodic():
    p=torch.tensor([[0.],[1.],[3.]])
    a=api().temporal_penalty(p,np.array([0.,1.,3.]))
    b=api().temporal_penalty(p,np.array([0.,2.,6.]))
    assert float(a)==pytest.approx(1.)
    assert float(b)==pytest.approx(.25)
    with pytest.raises(ValueError):api().temporal_penalty(p,np.array([0.,1.,1.]))


def test_mesh_step_guard_rejects_flipped_degenerate_or_nonfinite_vertices():
    v=np.array([[0.,0,0],[1,0,0],[0,1,0],[0,0,1]])
    f=np.array([[0,2,1],[0,1,3],[1,2,3],[2,0,3]])
    assert api().mesh_step_valid(v,v,f)['valid']
    moved=v.copy();moved[0]=v[1]
    assert not api().mesh_step_valid(v,moved,f)['valid']
    moved=v.copy();moved[0]=np.nan
    assert not api().mesh_step_valid(v,moved,f)['valid']


def test_missing_explicit_diagnostic_permission_prevents_patient_fit():
    from ctdr_pipeline.heart_mesh_motion.contracts import FitConfig
    config=FitConfig(model='rigid_inplane',mode='diagnostic_conditional')
    with pytest.raises(ValueError,match='geometry'):
        api().fit_provisional(None,None,[],config)


def test_real_differentiable_solver_reduces_independent_known_translation_error():
    pytest.importorskip('pytorch3d')
    from ctdr_pipeline.heart_mesh_motion.synthetic_mesh import digital_fixture
    from ctdr_pipeline.heart_mesh_motion.reference_raster import render_hard
    from ctdr_pipeline.heart_mesh_motion.contracts import FitConfig
    from scipy import ndimage
    mesh,camera=digital_fixture();obs=[]
    for i in range(3):
        mask=render_hard(mesh.vertices_lps_mm+np.array([4.,0,0]),mesh.faces,camera,image_hw=(64,64))
        edge=mask&~ndimage.binary_erosion(mask);yy,xx=np.where(edge)
        # Disjoint one-segment arcs, independent hard-raster target, no closed mask supervision.
        arcs=[dict(xy_px=[[float(x),float(y)],[float(x),float(y)+1]],inward_side_point_xy_px=None) for x,y in zip(xx[::4],yy[::4]) if y<62]
        obs.append(dict(frame_index=i,timestamp_ms=i*133.333,review_state='visible_confirmed',arcs=arcs,excluded_polygons_xy_px=[]))
    config=FitConfig(model='rigid_inplane',mode='synthetic',resolutions=(32,),optimizer_budget=18,
        loss_weights=dict(data=1.,amplitude=.001,temporal=.001))
    result=api().fit_provisional(mesh,camera,obs,config,steps_per_resolution=18)
    assert result['mode']=='synthetic'
    assert result['data_loss_after']<result['data_loss_before']
    assert np.mean(result['parameters'][:,0])>1
    assert result['real_patient_mesh_fit_executed'] is False
    assert result['camera_optimized'] is False
    assert result['independent_validation'] is False
