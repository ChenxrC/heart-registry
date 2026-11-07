"""Identity must not be inferred from a smaller registration residual."""
from copy import deepcopy

import numpy as np
import pytest

from ctdr_pipeline.heart_mesh_motion import bone_identity_audit as audit


def review_fixture():
    return {
        'schema_version': 1, 'case_id': 'example', 'pack_id': 'pack',
        'display_material_sha256': 'display', 'reviewer': 'doctor',
        'observations': [
            {'run_id': 's2', 'frame_index': 21, 'structure_id': label,
             'split': split, 'identity_confirmed': True,
             'review_state': 'confirmed_visible_edge',
             'source': 'human_review_on_raw_XA_not_model_generated',
             'xy_px': [[10., y], [30., y], [30., y+20], [10., y+20]]}
            for label, split, y in [('T6', 'holdout', 300), ('T4', 'holdout', 50), ('T5', 'fit', 180)]
        ]}


def pack_fixture():
    return {'case_id': 'example', 'pack_id': 'pack', 'display_material_sha256': 'display',
            'series': [{'run_id': 's2', 'frame_index': 21, 'image_hw': [512, 512]}],
            'structures': [{'id': 'T4', 'split': 'holdout'}, {'id': 'T5', 'split': 'fit'},
                           {'id': 'T6', 'split': 'holdout'}]}


def test_neutral_regions_keep_pixels_but_do_not_inherit_identity_or_split():
    review = review_fixture(); original = deepcopy(review)
    result = audit.neutral_regions(review, pack_fixture(), run_id='s2')
    assert [x['region_id'] for x in result] == ['R1', 'R2', 'R3']
    assert [x['provisional_user_label'] for x in result] == ['T4', 'T5', 'T6']
    assert result[1]['xy_px'] == [[10., 180], [30., 180], [30., 200], [10., 200]]
    assert result[1]['identity_state'] == 'unresolved'
    assert result[1]['split'] is None
    assert result[1]['source_identity_confirmed'] is True
    assert review == original


def test_neutral_regions_reject_wrong_display_material():
    review = review_fixture(); review['display_material_sha256'] = 'stale'
    with pytest.raises(ValueError, match='display material'):
        audit.neutral_regions(review, pack_fixture(), run_id='s2')


def test_neutral_regions_do_not_turn_unknown_edges_into_observations():
    review = review_fixture(); review['observations'][0].update(review_state='unknown', xy_px=[])
    regions = audit.neutral_regions(review, pack_fixture(), run_id='s2')
    assert len(regions) == 2
    assert all(x['provisional_user_label'] != 'T6' for x in regions)


def test_neutral_coordinates_are_not_accepted_by_confirmed_review_importer():
    from experiments.heart_mesh_dynamic.probes.bone_pose_review import validate_review
    regions = audit.neutral_regions(review_fixture(), pack_fixture(), run_id='s2')
    with pytest.raises(ValueError, match='does not identify'):
        validate_review({'schema': 'unresolved_bone_regions_v1', 'regions': regions}, pack_fixture())


def test_lps_to_plane_keeps_row_column_spacing_distinct_and_reports_depth():
    xy, depth = audit.lps_to_plane(np.array([[12., 5., -26.], [14., 7., -23.]]),
        origin=[10., 0., -20.], col_axis=[1., 0., 0.], row_axis=[0., 0., -1.],
        spacing_row_col=[3., 2.])
    np.testing.assert_allclose(xy, [[1., 2.], [2., 1.]])
    np.testing.assert_allclose(depth, [5., 7.])


def test_lps_to_plane_rejects_non_orthogonal_axes():
    with pytest.raises(ValueError, match='orthonormal'):
        audit.lps_to_plane(np.zeros((1,3)), origin=[0.,0.,0.], col_axis=[1.,0.,0.],
                           row_axis=[1.,0.,0.], spacing_row_col=[1.,1.])


def test_chain_audit_uses_superior_lps_z_not_filename_sort_or_array_direction():
    records = [{'label': 'T10', 'centroid_lps_mm': [0., 0., -30.]},
               {'label': 'T2', 'centroid_lps_mm': [0., 0., -10.]},
               {'label': 'T3', 'centroid_lps_mm': [0., 0., -20.]}]
    result = audit.chain_audit(records)
    assert result['cranial_to_caudal'] == ['T2', 'T3', 'T10']
    assert result['order_consistent'] is True
    assert result['anatomical_identity_verified'] is False


def test_chain_audit_flags_inverted_numbering_and_retains_missing_labels():
    records = [{'label': 'T1', 'centroid_lps_mm': [0., 0., -20.]},
               {'label': 'T2', 'centroid_lps_mm': [0., 0., -10.]},
               {'label': 'T3', 'centroid_lps_mm': None}]
    result = audit.chain_audit(records)
    assert result['order_consistent'] is False
    assert result['missing_labels'] == ['T3']


def test_lower_residuals_and_ct_only_anchors_cannot_unlock_patient_fit():
    regions = audit.neutral_regions(review_fixture(), pack_fixture(), run_id='s2')
    result = audit.identity_gate(regions, ct_chain_consistent=True,
                                xa_anchor_evidence=[])
    assert result['rigid_refit_allowed'] is False
    assert result['accepted_mapping'] is None
    assert result['unresolved_regions'] == ['R1', 'R2', 'R3']
    assert result['absolute_geometry_accepted'] is False


def test_audit_rejects_claimed_anchors_instead_of_silently_accepting_untested_evidence():
    with pytest.raises(ValueError, match='separately reviewed importer'):
        audit.identity_gate([], ct_chain_consistent=True,
                            xa_anchor_evidence=[{'source': 'model_selected_by_low_loss'}])

