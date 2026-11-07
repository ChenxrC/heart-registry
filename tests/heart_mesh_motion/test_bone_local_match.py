"""Catch unsafe hint imports and false boundary support, using real pixels."""
import copy

import numpy as np
import pytest

from ctdr_pipeline.heart_mesh_motion import bone_candidate_match as appearance


def review_fixture():
    return dict(schema='human_s7_r1_correspondence_suggestion_v1',case_id='1234501',
        display_material_sha256='a'*64,source_review_sha256='b'*64,reviewer='reader',
        review_state='correspondence_suggestion',source_region_id='R1',
        source_run='series_02',source_frame_index=21,target_run='series_07',target_frame_index=0,
        target_xy_px=[40.,50.],
        coordinate_convention='0-based native pixel centers; x=column/right, y=row/down',
        anatomical_level=None,anatomical_identity_verified=False,
        absolute_geometry_accepted=False,clinical_validation=False)


def test_review_import_is_only_a_location_hint_not_candidate_matrix_or_anatomical_identity():
    from ctdr_pipeline.heart_mesh_motion.bone_local_match import import_hint
    review=review_fixture();review['candidate_id']='previous_candidate_05'
    hint=import_hint(review,case_id='1234501',material_sha256='a'*64,source_review_sha256='b'*64,target_hw=(96,96))
    assert hint['target_xy_px']==[40.,50.]
    assert hint['exact_point_pair'] is False
    assert hint['anatomical_identity_verified'] is False
    assert 'source_to_target_affine' not in hint


@pytest.mark.parametrize('change',[
    {'display_material_sha256':'c'*64}, {'source_review_sha256':'c'*64},
    {'case_id':'wrong'}, {'source_frame_index':20}, {'target_frame_index':1},
    {'review_state':'uncertain'}, {'reviewer':' '}, {'target_xy_px':[np.nan,1.]},
    {'target_xy_px':[96.,1.]}, {'target_xy_px':[-1.,1.]},
    {'anatomical_level':'T4'}, {'absolute_geometry_accepted':True},
    {'anatomical_identity_verified':True}, {'clinical_validation':True},
    {'coordinate_convention':'one based row column'},
])
def test_review_import_rejects_wrong_lineage_bounds_or_promoted_geometry(change):
    from ctdr_pipeline.heart_mesh_motion.bone_local_match import import_hint
    review=copy.deepcopy(review_fixture());review.update(change)
    with pytest.raises(ValueError):
        import_hint(review,case_id='1234501',material_sha256='a'*64,source_review_sha256='b'*64,target_hw=(96,96))


def test_hint_circle_excludes_a_stronger_distant_match_before_spatial_nms():
    rng=np.random.default_rng(24)
    source=rng.uniform(40,180,(40,40));target=rng.uniform(40,180,(160,160))
    target[10:50,10:50]=source  # Stronger but outside the authorized local hint.
    target[90:130,100:140]=source+rng.normal(0,1,(40,40))
    curves=[np.array([[10.,10.],[30.,10.],[30.,30.]])]
    results=appearance.search_candidates(source,target,curves,[np.eye(2)],
        reference_xy=[20.,20.],center_roi_xyxy=[0,0,159,159],band_width_px=6,
        center_hint_xy=[120.,110.],center_hint_radius_px=5.,max_candidates=1)
    assert len(results)==1
    np.testing.assert_allclose(results[0]['reference_xy_px'],[120.,110.],atol=1.)


def test_local_hint_requires_finite_positive_radius_and_complete_pair():
    im=np.arange(4096.).reshape(64,64);curves=[np.array([[10.,10.],[30.,30.]])]
    for kwargs in (dict(center_hint_xy=[20.,20.]),dict(center_hint_radius_px=5.),
                   dict(center_hint_xy=[20.,20.],center_hint_radius_px=-1.),
                   dict(center_hint_xy=[np.nan,20.],center_hint_radius_px=5.)):
        with pytest.raises(ValueError):
            appearance.search_candidates(im,im,curves,[np.eye(2)],reference_xy=[20.,20.],
                center_roi_xyxy=[0,0,63,63],**kwargs)


def test_boundary_support_prefers_aligned_edge_and_rejects_crossing_edge():
    from ctdr_pipeline.heart_mesh_motion.bone_local_match import build_edge_field,boundary_support
    im=np.zeros((96,96));im[:,40:]=200.
    field=build_edge_field(im,np.ones(im.shape,bool))
    aligned=boundary_support(np.array([[39.5,20.],[39.5,75.]]),np.eye(3),field)
    crossing=boundary_support(np.array([[20.,45.],[75.,45.]]),np.eye(3),field)
    distant=boundary_support(np.array([[52.,20.],[52.,75.]]),np.eye(3),field)
    assert aligned['support_mean']>.85
    assert crossing['support_mean']<.05
    assert distant['support_mean']==0.
    assert aligned['valid_fraction']==1.


def test_unknown_arc_endpoints_are_not_closed_to_invent_an_edge():
    from ctdr_pipeline.heart_mesh_motion.bone_local_match import build_edge_field,boundary_support
    im=np.zeros((96,96));im[55:]=200.
    curve=np.array([[20.,20.],[75.,20.],[75.,55.]])
    # Only a fabricated last-to-first diagonal would gain appreciable support.
    result=boundary_support(curve,np.eye(3),build_edge_field(im,np.ones(im.shape,bool)))
    assert result['support_mean']<.01
    assert result['sample_count']==90  # 55+35 unit-length segment midpoints.


def test_flat_image_and_padding_boundary_do_not_supply_anatomical_edges():
    from ctdr_pipeline.heart_mesh_motion.bone_local_match import build_edge_field,boundary_support
    flat=np.full((96,96),100.)
    curve=np.array([[20.,20.],[75.,20.]])
    assert boundary_support(curve,np.eye(3),build_edge_field(flat,np.ones(flat.shape,bool)))['support_mean']==0.
    padded=np.zeros((96,96));padded[15:81,15:81]=100.
    result=boundary_support(np.array([[15.,20.],[15.,75.]]),np.eye(3),
        build_edge_field(padded,padded>0))
    assert result['support_mean']==0.
    assert result['valid_fraction']==0.


def test_clipped_arc_is_flagged_not_rewarded_by_missing_observations():
    from ctdr_pipeline.heart_mesh_motion.bone_local_match import build_edge_field,boundary_support
    im=np.zeros((96,96));im[:,40:]=200.
    result=boundary_support(np.array([[39.5,-20.],[39.5,75.]]),np.eye(3),
        build_edge_field(im,np.ones(im.shape,bool)))
    assert result['valid_fraction']<.9
    assert result['usable_for_ranking'] is False
    assert result['support_mean']<.85


def test_ranking_preserves_relative_regions_without_claiming_holdout_validation():
    from ctdr_pipeline.heart_mesh_motion.bone_local_match import build_edge_field,rank_candidates
    im=np.zeros((96,96));im[:,40:]=200.
    curves=[np.array([[39.5,y],[39.5,y+10.]]) for y in (20.,40.,60.)]
    wrong=np.eye(3);wrong[0,2]=12
    candidates=[dict(score=.9,reference_xy_px=[51.5,25.],source_to_target_affine=wrong.tolist()),
                dict(score=.8,reference_xy_px=[39.5,25.],source_to_target_affine=np.eye(3).tolist())]
    results=rank_candidates(candidates,curves,build_edge_field(im,np.ones(im.shape,bool)),
        scoring_regions=(0,1,2))
    assert results[0]['reference_xy_px']==[39.5,25.]
    assert results[0]['layout']['top_to_bottom_preserved'] is True
    assert results[0]['independent_validation'] is False
    assert results[0]['anatomical_identity_verified'] is False
    assert len(results[0]['region_boundary_evidence'])==3
    assert candidates[0]['score']==.9  # Original appearance proposals remain untouched.


def test_r1_sensitivity_does_not_use_r2_r3_to_rank():
    from ctdr_pipeline.heart_mesh_motion.bone_local_match import build_edge_field,rank_candidates
    im=np.zeros((96,96));im[:,40:]=200.
    curves=[np.array([[39.5,20.],[39.5,30.]]),np.array([[51.,40.],[51.,50.]]),
            np.array([[51.,60.],[51.,70.]])]
    candidate=dict(score=.8,reference_xy_px=[39.5,25.],source_to_target_affine=np.eye(3).tolist())
    field=build_edge_field(im,np.ones(im.shape,bool))
    single=rank_candidates([candidate],curves,field,scoring_regions=(0,))[0]
    joint=rank_candidates([candidate],curves,field,scoring_regions=(0,1,2))[0]
    assert single['combined_score']>joint['combined_score']+.2
    assert single['scoring_region_ids']==['R1']


def test_grid_diagnostics_distinguish_search_limit_from_small_center_disagreement():
    from ctdr_pipeline.heart_mesh_motion.bone_local_match import linear_grid_evidence
    grid=dict(scale_factors=[.55,1.,1.3],vertical_anisotropy=[.8,1.,1.2],rotation_deg=[-90.,0.,90.])
    at_lower=np.diag([.22,.165,1.])
    first=linear_grid_evidence(at_lower,[.4,.3],grid)
    assert first['scale_factor']==pytest.approx(.55)
    assert first['rotation_deg']==pytest.approx(0.)
    assert first['search_bounds_touched']==['scale_factor_lower']
    rotated=np.array([[0.,-.24,10.],[.4,0.,20.],[0.,0.,1.]])
    second=linear_grid_evidence(rotated,[.4,.3],grid)
    assert second['rotation_deg']==pytest.approx(90.)
    assert second['vertical_anisotropy']==pytest.approx(.8)
    assert second['search_bounds_touched']==['vertical_anisotropy_lower','rotation_deg_upper']


def test_r1_only_eligibility_does_not_silently_use_other_region_order():
    from ctdr_pipeline.heart_mesh_motion.bone_local_match import build_edge_field,rank_candidates
    im=np.zeros((96,96));im[:,40:]=200.
    curves=[np.array([[39.5,y],[39.5,y+10.]]) for y in (20.,60.,40.)]
    candidate=dict(score=.8,reference_xy_px=[39.5,25.],source_to_target_affine=np.eye(3).tolist())
    result=rank_candidates([candidate],curves,build_edge_field(im,np.ones(im.shape,bool)),scoring_regions=(0,))[0]
    assert result['eligible_for_ranking'] is True
    assert result['layout']['top_to_bottom_preserved'] is False
