"""Actual partial heart observations must never become self-generated truth."""
from copy import deepcopy
import importlib

import numpy as np
import pytest


def api():return importlib.import_module('ctdr_pipeline.heart_mesh_motion.heart_edge_review')


def material():
    return dict(case_id='1234567',run_id='series_07',image_hw=[81,101],
        display_material_sha256='a'*64,selection_sha256='b'*64,selected_camera_sha256='c'*64,
        frames=[dict(frame_index=i,timestamp_ms=i*100.,frame_sha256=str(i)*64,
                     valid_row_intervals=[[4,96] for _ in range(81)]) for i in range(7)])


def arc():return dict(xy_px=[[20.,10.],[20.,20.]],inward_side_point_xy_px=[30.,15.])


def observation(index=0):
    return dict(frame_index=index,timestamp_ms=index*100.,frame_sha256=str(index)*64,
        review_state='visible_confirmed',visible_heart_edge_confirmed=True,arcs=[arc()],
        excluded_polygons_xy_px=[],note='synthetic test, not patient annotation')


def review():
    return dict(schema='human_visible_heart_edges_v1',case_id='1234567',run_id='series_07',reviewer='test',
        display_material_sha256='a'*64,selection_sha256='b'*64,selected_camera_sha256='c'*64,
        coordinate_convention='0-based native pixel centers; x=column/right, y=row/down',
        suggestion_assisted=True,independent_blinded_truth=False,automatic_edges_used_as_truth=False,
        anatomical_identity_verified=False,absolute_geometry_accepted=False,clinical_validation=False,
        real_patient_mesh_fit_executed=False,heart_motion_reconstructed=False,patient_true_3d_tre_mm=None,
        observations=[observation()])


def test_normal_direction_uses_actual_side_point_not_polyline_order():
    n=api().arc_inward_normals([[20,10],[20,20]],[30,15])
    reverse=api().arc_inward_normals([[20,20],[20,10]],[30,15])
    np.testing.assert_allclose(n,[[1,0]])
    np.testing.assert_allclose(reverse,[[1,0]])


def test_interior_side_collinear_with_edge_is_ambiguous_not_auto_flipped():
    with pytest.raises(ValueError):api().arc_inward_normals([[20,10],[20,20]],[20,15])


def test_partial_disjoint_arcs_keep_unknown_area_and_do_not_unlock_fitting():
    value=review();value['observations'][0]['arcs'].append(dict(xy_px=[[50.,40.],[60.,40.]],inward_side_point_xy_px=[55.,50.]))
    before=deepcopy(value);result=api().validate_heart_edge_review(value,material())
    assert value==before
    assert len(result['frames'][0]['arcs'])==2
    assert result['frames'][0]['arcs'][0]['xy_px']==before['observations'][0]['arcs'][0]['xy_px']
    assert result['summary']['unreviewed_frame_indices']==[1,2,3,4,5,6]
    assert result['summary']['confirmed_visible_frame_count']==1
    assert result['summary']['whole_heart_foreground_mask_available'] is False
    assert result['real_patient_mesh_fit_executed'] is False
    assert result['patient_true_3d_tre_mm'] is None


@pytest.mark.parametrize('change',[
    {'selected_camera_sha256':'d'*64},{'selection_sha256':'d'*64},{'display_material_sha256':'d'*64},
    {'case_id':'other'},{'run_id':'series_02'},{'reviewer':''},{'automatic_edges_used_as_truth':True},
    {'absolute_geometry_accepted':True},{'heart_motion_reconstructed':True},{'patient_true_3d_tre_mm':0.},
])
def test_material_or_acceptance_mismatch_is_rejected(change):
    value=review();value.update(change)
    with pytest.raises(ValueError):api().validate_heart_edge_review(value,material())


@pytest.mark.parametrize('change',[
    {'frame_index':7},{'timestamp_ms':20.},{'frame_sha256':'d'*64},
    {'arcs':[]},{'visible_heart_edge_confirmed':False},
    {'arcs':[dict(xy_px=[[20.,10.],[20.,20.]],inward_side_point_xy_px=None)]},
    {'arcs':[dict(xy_px=[[0.,10.],[0.,20.]],inward_side_point_xy_px=[30.,15.])]},
    {'arcs':[dict(xy_px=[[20.,10.],[20.,10.]],inward_side_point_xy_px=[30.,15.])]},
])
def test_bad_frame_or_unobserved_heart_boundary_cannot_be_confirmed(change):
    value=review();value['observations'][0].update(change)
    with pytest.raises(ValueError):api().validate_heart_edge_review(value,material())


def test_uncertain_and_not_visible_are_saved_without_zero_error_or_fabricated_normals():
    value=review();uncertain=observation(0);uncertain.update(review_state='uncertain',visible_heart_edge_confirmed=False)
    uncertain['arcs'][0]['inward_side_point_xy_px']=None
    missing=observation(1);missing.update(review_state='not_visible',visible_heart_edge_confirmed=False,arcs=[])
    value['observations']=[uncertain,missing]
    result=api().validate_heart_edge_review(value,material())
    assert result['status']=='insufficient_confirmed_heart_observations'
    assert result['summary']['confirmed_visible_frame_count']==0
    assert result['frames'][0]['arcs'][0]['inward_normal_xy'] is None


def test_duplicate_frames_and_not_visible_with_guessed_arcs_are_rejected():
    value=review();value['observations']*=2
    with pytest.raises(ValueError):api().validate_heart_edge_review(value,material())
    value=review();value['observations'][0].update(review_state='not_visible',visible_heart_edge_confirmed=False)
    with pytest.raises(ValueError):api().validate_heart_edge_review(value,material())


def test_flat_detector_padding_cannot_create_suggested_edges():
    image=np.zeros((81,101),dtype='uint8');image[10:70,12:88]=120
    length=np.zeros((81,101));length[20:60,25:75]=10
    result=api().propose_heart_edge_points(image,length)
    assert result['points_xy_px']==[]
    assert result['automatic_edges_are_truth'] is False


def test_candidates_require_raw_intensity_evidence_not_the_ct_outline():
    image=np.full((81,101),80.,dtype=float);image[:,55:]=180
    length=np.zeros((81,101));length[20:60,25:75]=10
    result=api().propose_heart_edge_points(image,length)
    points=np.asarray(result['points_xy_px'])
    assert len(points)>0
    assert np.all(np.abs(points[:,0]-54.5)<2)
    assert result['automatic_edges_are_truth'] is False


def test_out_of_native_coordinates_are_rejected_before_any_interpolation(monkeypatch):
    value=review();value['observations'][0]['arcs'][0]['xy_px'][1]=[1e9,20.]
    def forbidden_allocation(_):
        raise AssertionError('Out-of-native coordinates reached interpolation')
    monkeypatch.setattr(api(),'sample_curve',forbidden_allocation)
    with pytest.raises(ValueError,match='outside native'):
        api().validate_heart_edge_review(value,material())


def test_uncertain_arc_without_side_still_has_a_vertex_limit():
    value=review();obs=value['observations'][0]
    obs.update(review_state='uncertain',visible_heart_edge_confirmed=False)
    obs['arcs']=[dict(xy_px=[[20.,10.+i%2] for i in range(4097)],inward_side_point_xy_px=None)]
    with pytest.raises(ValueError,match='4096'):
        api().validate_heart_edge_review(value,material())
