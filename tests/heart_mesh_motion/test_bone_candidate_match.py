"""Candidate scores are appearance evidence, never anatomical acceptance."""
import numpy as np
import pytest

from ctdr_pipeline.heart_mesh_motion import bone_candidate_match as match


def test_native_affine_preserves_xy_and_only_adds_target_patch_translation():
    a = np.array([[.5,.1,-2.], [0.,.25,-3.], [0.,0.,1.]])
    result = match.native_affine(a, top_left_yx=(10,20))
    np.testing.assert_allclose(result, [[.5,.1,18.], [0.,.25,7.], [0.,0.,1.]])
    np.testing.assert_allclose(match.transform_xy([[4.,8.]], result), [[20.8,9.]])


def test_spacing_ratio_distinguishes_rows_and_columns():
    np.testing.assert_allclose(match.detector_sampling_ratio([.3,.2],[.6,.8]), [.25,.5])


def test_observed_arc_band_does_not_fill_or_automatically_close_unknown_region():
    mask = match.arc_band((64,64), [np.array([[10.,10.],[50.,10.],[50.,50.]])], width=5)
    assert mask[10,30] > 0
    assert mask[30,50] > 0
    assert mask[30,30] == 0  # Interior/closing diagonal was never observed.
    assert mask[50,30] == 0


def test_masked_ncc_finds_known_shift_despite_gain_offset_and_ignored_pixels():
    rng=np.random.default_rng(21)
    template=rng.normal(size=(7,11)); weights=np.ones_like(template)
    weights[:, :2]=0
    target=rng.normal(size=(48,60))*.2+30
    target[23:30,17:28]=template*4+30
    template[:, :2]=1e5  # Non-observed pixels must have no influence.
    scores=match.masked_ncc(template,weights,target)
    assert np.unravel_index(np.argmax(scores),scores.shape)==(23,17)
    assert scores[23,17] == pytest.approx(1.,abs=1e-7)


def test_flat_or_missing_appearance_is_not_a_perfect_candidate():
    assert not np.isfinite(match.masked_ncc(np.ones((5,5)),np.ones((5,5)),np.ones((16,16)))).any()
    with pytest.raises(ValueError,match='weights'):
        match.masked_ncc(np.eye(5),np.zeros((5,5)),np.eye(16))


def test_full_search_cannot_create_signal_by_zero_padding_flat_source():
    source=np.full((64,64),100.)
    target=np.zeros((100,100));target[18:82,18:82]=100.
    curves=[np.array([[1.,1.],[60.,1.],[60.,60.]])]
    result=match.search_candidates(source,target,curves,[np.eye(2)],reference_xy=[30,30],
        center_roi_xyxy=[0,0,99,99],band_width_px=12,target_valid=(target>0))
    assert result==[]


def test_peak_stage_preserves_distant_fifth_peak_before_global_spatial_suppression():
    scores=np.full((64,64),-np.inf)
    for y,x,value in [(10,10,1.),(10,18,.99),(18,10,.98),(18,18,.97),(50,50,.9)]:
        scores[y,x]=value
    proposals=match.peak_proposals(scores,np.eye(3),reference_xy=[0,0],template_band_weight=100.)
    selected=match.spatial_nms(proposals,max_candidates=5,radius_px=18.)
    assert [p['reference_xy_px'] for p in selected]==[[10.,10.],[50.,50.]]


def test_spatial_nms_keeps_alternative_levels_not_many_variants_of_one_location():
    candidates=[{'score':.9,'reference_xy_px':[100.,100.]},
                {'score':.89,'reference_xy_px':[102.,103.]},
                {'score':.8,'reference_xy_px':[100.,130.]}]
    selected=match.spatial_nms(candidates,max_candidates=3,radius_px=18.)
    assert [c['score'] for c in selected]==[.9,.8]


def test_search_recovers_synthetic_translation_without_promoting_identity():
    rng=np.random.default_rng(2)
    source=rng.uniform(40,160,size=(80,80))
    target=np.full((180,180),30.)
    target[40:120,55:135]=source
    curves=[np.array([[20.,20.],[60.,20.],[60.,50.]])]
    candidates=match.search_candidates(source,target,curves,[np.eye(2)],
        reference_xy=[40.,30.],band_width_px=8,center_roi_xyxy=[0,0,179,179],max_candidates=3)
    assert candidates
    first=candidates[0]
    np.testing.assert_allclose(match.transform_xy([[20.,20.]],first['source_to_target_affine']),[[75.,60.]],atol=1e-6)
    assert first['anatomical_identity_verified'] is False
    assert first['absolute_geometry_accepted'] is False


def test_search_rejects_reflection_and_nonfinite_coordinates():
    source=np.arange(4096).reshape(64,64)
    curve=[np.array([[10.,10.],[20.,20.]])]
    with pytest.raises(ValueError,match='orientation'):
        match.search_candidates(source,source,curve,[np.diag([-1.,1.])],
            reference_xy=[15,15],center_roi_xyxy=[0,0,63,63])
    with pytest.raises(ValueError,match='coordinates'):
        match.arc_band((64,64),[np.array([[np.nan,10.],[20.,20.]])])
