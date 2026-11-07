"""Validate manual S7 open-arc correspondence evidence without accepting geometry."""
from __future__ import annotations

from copy import deepcopy
import json

import numpy as np


COORDINATES='0-based native pixel centers; x=column/right, y=row/down'
FALSE_FLAGS=('independent_blinded_truth','automatic_target_arcs_used',
    'anatomical_identity_verified','absolute_geometry_accepted','clinical_validation',
    'rigid_refit_allowed','rigid_refit_executed','heart_deformation_executed')


def validate_correspondence_review(review,material):
    """Human identity statements remain claims requiring a separate static-pose review."""
    expected=dict(schema='human_s7_bone_correspondence_edges_v1',case_id=material['case_id'],
        display_material_sha256=material['display_material_sha256'],
        parent_local_material_sha256=material['parent_local_material_sha256'],
        source_review_sha256=material['source_review_sha256'],
        source_run=material['source']['run_id'],source_frame_index=material['source']['frame_index'],
        target_run=material['target']['run_id'],target_frame_index=material['target']['frame_index'],
        coordinate_convention=COORDINATES)
    if not isinstance(review,dict) or any(review.get(k)!=v for k,v in expected.items()):
        raise ValueError('Review material, patient, frame or coordinate convention differs')
    json.dumps(review,allow_nan=False)
    if not isinstance(review.get('reviewer'),str) or not review['reviewer'].strip():
        raise ValueError('Reviewer required for visible, uncertain and missing observations')
    if (review.get('suggestion_assisted') is not True or any(review.get(k) is not False for k in FALSE_FLAGS)
            or review.get('patient_true_3d_tre_mm','missing') is not None):
        raise ValueError('Correspondence evidence cannot claim blinded truth, 3D or clinical acceptance')
    observations=review.get('observations')
    if not isinstance(observations,list) or not 1<=len(observations)<=len(material['region_ids']):
        raise ValueError('Save at least one observed or explicitly uncertain/missing region')
    seen=set();named=[];hw=np.asarray(material['target']['image_hw'],float)
    if hw.shape!=(2,) or not np.isfinite(hw).all() or np.any(hw<1) or np.any(hw!=np.floor(hw)):
        raise ValueError('Native target dimensions required')
    for obs in observations:
        if not isinstance(obs,dict):raise ValueError('Region object required')
        region=obs.get('source_region_id');state=obs.get('review_state')
        if region not in material['region_ids'] or region in seen:raise ValueError('Unknown or duplicate source region')
        seen.add(region)
        if state not in ('same_region_visible','identity_uncertain','not_visible'):
            raise ValueError('Explicit visible, uncertain or not-visible state required')
        if obs.get('same_region_confirmed_by_reviewer') is not (state=='same_region_visible'):
            raise ValueError('Same-region checkbox must match the recorded review state')
        arcs=obs.get('target_arcs_xy_px')
        if not isinstance(arcs,list) or len(arcs)>8:raise ValueError('At most eight separate open arcs per region')
        if state=='same_region_visible' and not arcs:raise ValueError('Visible correspondence needs a manual arc')
        if state=='not_visible' and arcs:raise ValueError('Not-visible region cannot contain fabricated edges')
        for arc in arcs:
            points=np.asarray(arc)
            if (points.ndim!=2 or points.shape[1]!=2 or not 2<=len(points)<=4096
                    or not np.issubdtype(points.dtype,np.number) or not np.isfinite(points).all()
                    or np.any(points<0) or np.any(points>hw[::-1]-1)
                    or np.linalg.norm(np.diff(points,axis=0),axis=1).sum()<1.):
                raise ValueError('Each manual open arc needs finite native coordinates and length >=1 pixel')
        level=obs.get('anatomical_level')
        if level is None:
            if obs.get('level_confirmed_by_reviewer') is not False:
                raise ValueError('Unknown anatomical level cannot be marked confirmed')
        else:
            if (state!='same_region_visible' or level not in material['ct_levels']
                    or obs.get('level_confirmed_by_reviewer') is not True
                    or not isinstance(obs.get('level_basis'),str) or not obs['level_basis'].strip()):
                raise ValueError('A named level needs same-region confirmation, a CT label and explicit supporting basis')
            named.append((material['region_ids'].index(region),material['ct_levels'].index(level)))
        if any(k in obs for k in ('source_to_target_affine','transform_4x4','camera')):
            raise ValueError('Geometry is not accepted through a manual edge review')
    ordered=[v for _,v in sorted(named)]
    if any(b<=a for a,b in zip(ordered[:-1],ordered[1:])):
        raise ValueError('Named levels conflict with source ordering; review them or retain unknown')
    result=deepcopy(review)
    result['summary']=dict(same_region_count=sum(o['review_state']=='same_region_visible' for o in observations),
        uncertain_count=sum(o['review_state']=='identity_uncertain' for o in observations),
        not_visible_count=sum(o['review_state']=='not_visible' for o in observations),
        named_level_count=len(named),missing_regions=[r for r in material['region_ids'] if r not in seen],
        anatomical_labels_are='explicit human claims, not automatically verified anatomy',
        interpretation='No centroid pairing, no closed-foreground assumption, no point/3D registration error')
    result['status']='manual_correspondence_evidence_requires_anatomical_and_pose_review'
    return result
