"""An S7 human arc is evidence, never automatically accepted geometry."""
from copy import deepcopy

import pytest


def material():
    return dict(case_id='1234567',display_material_sha256='a'*64,
        parent_local_material_sha256='b'*64,source_review_sha256='c'*64,
        source=dict(run_id='series_02',frame_index=21),
        target=dict(run_id='series_07',frame_index=0,image_hw=[512,512]),
        region_ids=['R1','R2','R3'],ct_levels=['C6','C7']+[f'T{i}' for i in range(1,13)]+['L1','L2'])


def observation(region='R1',level=None):
    return dict(source_region_id=region,review_state='same_region_visible',
        target_arcs_xy_px=[[[200.,180.],[210.,183.],[220.,181.]],[[201.,204.],[215.,205.]]],
        same_region_confirmed_by_reviewer=True,anatomical_level=level,
        level_confirmed_by_reviewer=level is not None,
        level_basis='Checked in CT/scout and XA context' if level else '',note='')


def review():
    return dict(schema='human_s7_bone_correspondence_edges_v1',case_id='1234567',
        display_material_sha256='a'*64,parent_local_material_sha256='b'*64,
        source_review_sha256='c'*64,reviewer='reader',
        source_run='series_02',source_frame_index=21,target_run='series_07',target_frame_index=0,
        coordinate_convention='0-based native pixel centers; x=column/right, y=row/down',
        suggestion_assisted=True,independent_blinded_truth=False,automatic_target_arcs_used=False,
        anatomical_identity_verified=False,absolute_geometry_accepted=False,clinical_validation=False,
        rigid_refit_allowed=False,rigid_refit_executed=False,heart_deformation_executed=False,
        patient_true_3d_tre_mm=None,observations=[observation()])


def test_human_open_arcs_keep_unknown_level_and_never_unlock_3d_fitting():
    from ctdr_pipeline.heart_mesh_motion.bone_correspondence_review import validate_correspondence_review
    value=review();before=deepcopy(value)
    result=validate_correspondence_review(value,material())
    assert value==before
    assert result['observations'][0]['target_arcs_xy_px']==before['observations'][0]['target_arcs_xy_px']
    assert result['observations'][0]['anatomical_level'] is None
    assert result['summary']['same_region_count']==1
    assert result['summary']['named_level_count']==0
    assert result['summary']['missing_regions']==['R2','R3']
    assert result['rigid_refit_allowed'] is False
    assert result['absolute_geometry_accepted'] is False
    assert result['patient_true_3d_tre_mm'] is None


@pytest.mark.parametrize('change',[
    {'display_material_sha256':'d'*64},{'parent_local_material_sha256':'d'*64},
    {'source_review_sha256':'d'*64},{'case_id':'other'}, {'target_frame_index':1},
    {'source_frame_index':20},{'reviewer':' '},{'observations':[]},
    {'independent_blinded_truth':True},{'suggestion_assisted':False},
    {'automatic_target_arcs_used':True},{'rigid_refit_allowed':True},
    {'anatomical_identity_verified':True},{'absolute_geometry_accepted':True},
    {'clinical_validation':True},{'heart_deformation_executed':True},
    {'patient_true_3d_tre_mm':0.},
])
def test_review_rejects_wrong_material_frames_or_promoted_acceptance(change):
    from ctdr_pipeline.heart_mesh_motion.bone_correspondence_review import validate_correspondence_review
    value=review();value.update(change)
    with pytest.raises(ValueError):validate_correspondence_review(value,material())


@pytest.mark.parametrize('change',[
    {'source_region_id':'R4'},{'same_region_confirmed_by_reviewer':False},
    {'target_arcs_xy_px':[]},{'target_arcs_xy_px':[[[1.,2.]]]},
    {'target_arcs_xy_px':[[[1.,2.],[1.,2.]]]},
    {'target_arcs_xy_px':[[[float('nan'),2.],[4.,5.]]]},
    {'target_arcs_xy_px':[[[512.,2.],[4.,5.]]]},
    {'target_arcs_xy_px':[[[-1.,2.],[4.,5.]]]},
    {'anatomical_level':'T4','level_confirmed_by_reviewer':False},
    {'anatomical_level':'T4','level_confirmed_by_reviewer':True,'level_basis':''},
    {'anatomical_level':'T99','level_confirmed_by_reviewer':True,'level_basis':'checked'},
])
def test_visible_region_requires_real_finite_arcs_and_explicit_human_evidence(change):
    from ctdr_pipeline.heart_mesh_motion.bone_correspondence_review import validate_correspondence_review
    value=review();value['observations'][0].update(change)
    with pytest.raises(ValueError):validate_correspondence_review(value,material())


def test_uncertain_correspondence_can_retain_visible_arcs_but_cannot_claim_a_level():
    from ctdr_pipeline.heart_mesh_motion.bone_correspondence_review import validate_correspondence_review
    value=review();obs=value['observations'][0]
    obs.update(review_state='identity_uncertain',same_region_confirmed_by_reviewer=False)
    result=validate_correspondence_review(value,material())
    assert result['summary']['same_region_count']==0
    assert result['summary']['uncertain_count']==1
    assert result['observations'][0]['target_arcs_xy_px']
    obs.update(anatomical_level='T4',level_confirmed_by_reviewer=True,level_basis='guess')
    with pytest.raises(ValueError):validate_correspondence_review(value,material())


def test_not_visible_has_no_fabricated_arc_or_level():
    from ctdr_pipeline.heart_mesh_motion.bone_correspondence_review import validate_correspondence_review
    value=review();value['observations'][0].update(review_state='not_visible',
        same_region_confirmed_by_reviewer=False,target_arcs_xy_px=[])
    result=validate_correspondence_review(value,material())
    assert result['summary']['not_visible_count']==1
    assert result['summary']['named_level_count']==0
    value['observations'][0]['target_arcs_xy_px']=[[[1.,2.],[3.,4.]]]
    with pytest.raises(ValueError):validate_correspondence_review(value,material())


def test_duplicate_regions_or_nonmonotone_named_levels_are_rejected_not_relabelled():
    from ctdr_pipeline.heart_mesh_motion.bone_correspondence_review import validate_correspondence_review
    for observations in ([observation('R1'),observation('R1')],
                         [observation('R1','T4'),observation('R2','T4')],
                         [observation('R1','T5'),observation('R2','T4')]):
        value=review();value['observations']=observations
        with pytest.raises(ValueError):validate_correspondence_review(value,material())


def test_explicit_named_levels_are_recorded_as_human_claims_not_verified_anatomy():
    from ctdr_pipeline.heart_mesh_motion.bone_correspondence_review import validate_correspondence_review
    value=review();value['observations']=[observation('R1','T4'),observation('R3','T6')]
    result=validate_correspondence_review(value,material())
    assert result['summary']['named_level_count']==2
    assert result['anatomical_identity_verified'] is False
    assert result['rigid_refit_allowed'] is False
