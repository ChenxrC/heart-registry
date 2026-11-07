"""Human-location-constrained 2D bone proposals, never accepted 3D geometry."""
from __future__ import annotations

import copy

import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree

from .bone_candidate_match import transform_xy


def import_hint(review,*,case_id,material_sha256,source_review_sha256,target_hw):
    """Bind the location suggestion to its actual material, without using candidate matrices."""
    if not isinstance(case_id,str) or not case_id.strip():
        raise ValueError('A caller-supplied case identity is required')
    expected=dict(schema='human_s7_r1_correspondence_suggestion_v1',case_id=case_id,
        display_material_sha256=material_sha256,source_review_sha256=source_review_sha256,
        review_state='correspondence_suggestion',source_region_id='R1',
        source_run='series_02',source_frame_index=21,target_run='series_07',target_frame_index=0,
        coordinate_convention='0-based native pixel centers; x=column/right, y=row/down')
    if not isinstance(review,dict) or any(review.get(k)!=v for k,v in expected.items()):
        raise ValueError('Review lineage, frame, state or coordinate convention differs')
    if not isinstance(review.get('reviewer'),str) or not review['reviewer'].strip():
        raise ValueError('Named reviewer required')
    if (review.get('anatomical_level') is not None or any(review.get(k) is not False for k in
            ('anatomical_identity_verified','absolute_geometry_accepted','clinical_validation'))):
        raise ValueError('Only unresolved location suggestions are accepted')
    xy=np.asarray(review.get('target_xy_px'),float)
    hw=np.asarray(target_hw)
    if (hw.shape!=(2,) or np.any(hw<1) or xy.shape!=(2,) or not np.isfinite(xy).all()
            or np.any(xy<0) or np.any(xy>hw[::-1]-1)):
        raise ValueError('Native pixel coordinate outside target bounds')
    return dict(target_xy_px=xy.tolist(),exact_point_pair=False,anatomical_identity_verified=False,
        anatomical_level=None,reviewer=review['reviewer'],
        source='suggestion-assisted human location; not independent blinded truth')


def build_edge_field(image,valid,*,quantile=.8):
    """Polarity-independent ridge gradients; invalid padding cannot manufacture an edge."""
    image=np.asarray(image,float);valid=np.asarray(valid,bool)
    if (image.ndim!=2 or valid.shape!=image.shape or not np.isfinite(image).all()
            or not valid.any() or not np.isfinite(quantile) or not 0<quantile<1):
        raise ValueError('Finite image, nonempty valid support and quantile in (0,1) required')
    # Continue nearest observed intensities across padding before differentiation.
    nearest=ndimage.distance_transform_edt(~valid,return_distances=False,return_indices=True)
    smooth=ndimage.gaussian_filter(image[tuple(nearest)],1.)
    gx=ndimage.sobel(smooth,1)/8.;gy=ndimage.sobel(smooth,0)/8.
    magnitude=np.hypot(gx,gy)
    safe=ndimage.binary_erosion(valid,iterations=4,border_value=0)
    directions=np.stack((gx,gy),-1)/np.maximum(magnitude[...,None],1e-12)
    yy,xx=np.indices(image.shape)
    ahead=ndimage.map_coordinates(magnitude,[yy+directions[...,1],xx+directions[...,0]],order=1,mode='nearest')
    behind=ndimage.map_coordinates(magnitude,[yy-directions[...,1],xx-directions[...,0]],order=1,mode='nearest')
    positive=magnitude[safe&(magnitude>1e-8)]
    threshold=float(np.quantile(positive,quantile)) if len(positive) else float('inf')
    edge=safe&(magnitude>=threshold)&(magnitude>1e-8)&(magnitude>=ahead-1e-10)&(magnitude>=behind-1e-10)
    yx=np.argwhere(edge);xy=yx[:,::-1].astype(float)
    return dict(tree=cKDTree(xy),xy=xy,directions=directions[edge],valid=safe,
        threshold=None if not np.isfinite(threshold) else threshold,edge_count=len(xy),
        image_hw=image.shape,quantile=float(quantile))


def boundary_support(curve,affine,field,*,radius_px=6.,sigma_px=2.):
    """Sample only observed segments. Nearby aligned image edges are evidence, not truth."""
    curve=np.asarray(curve,float)
    if (curve.ndim!=2 or curve.shape[1]!=2 or len(curve)<2 or not np.isfinite(curve).all()
            or not np.isfinite([radius_px,sigma_px]).all() or min(radius_px,sigma_px)<=0):
        raise ValueError('Finite observed arc and positive edge distances required')
    transformed=transform_xy(curve,affine);points=[];normals=[]
    for p,q in zip(transformed[:-1],transformed[1:]):
        delta=q-p;length=float(np.linalg.norm(delta))
        if length<1e-9:continue
        count=int(np.ceil(length));t=(np.arange(count)+.5)/count
        points.extend(p+t[:,None]*delta)
        normals.extend(np.tile([-delta[1]/length,delta[0]/length],(count,1)))
    if not points:raise ValueError('Observed arc has no length')
    points=np.asarray(points);normals=np.asarray(normals)
    h,w=field['image_hw'];inside=(points[:,0]>=0)&(points[:,0]<=w-1)&(points[:,1]>=0)&(points[:,1]<=h-1)
    observed=np.zeros(len(points),bool)
    observed[inside]=ndimage.map_coordinates(field['valid'].astype(float),points[inside,::-1].T,
        order=0,mode='constant',cval=0.)>.5
    scores=np.zeros(len(points));distances=np.full(len(points),radius_px)
    if field['edge_count']:
        indices=field['tree'].query_ball_point(points[observed],radius_px)
        for i,neighbors in zip(np.flatnonzero(observed),indices):
            if not neighbors:continue
            neighbors=np.asarray(neighbors);delta=field['xy'][neighbors]-points[i]
            distance=np.linalg.norm(delta,axis=1)
            alignment=np.abs(field['directions'][neighbors]@normals[i])**2
            support=np.exp(-.5*(distance/sigma_px)**2)*alignment
            scores[i]=float(support.max())
            compatible=alignment>=.5
            if compatible.any():distances[i]=float(distance[compatible].min())
    fraction=float(observed.mean())
    return dict(support_mean=float(scores.mean()),support_fraction_ge_half=float((scores>=.5).mean()),
        valid_fraction=fraction,usable_for_ranking=fraction>=.98,sample_count=len(points),
        truncated_oriented_edge_distance_mean_px=float(distances.mean()),
        metric_kind='directed observed-arc image-edge evidence; NOT registration error or anatomical truth')


def rank_candidates(candidates,curves,field,*,scoring_regions):
    """Rerank a finite appearance shortlist; report all region evidence separately."""
    if len(curves)!=3 or tuple(scoring_regions) not in ((0,),(0,1,2)):
        raise ValueError('Three original arcs and a declared R1-only/joint scoring subset required')
    ranked=[]
    for original in candidates:
        candidate=copy.deepcopy(original);a=np.asarray(candidate['source_to_target_affine'],float)
        if np.linalg.det(a[:2,:2])<=0:raise ValueError('Reflected or collapsed proposal')
        evidence=[dict(region_id=f'R{i+1}',**boundary_support(c,a,field)) for i,c in enumerate(curves)]
        centers=np.array([(np.min(c,0)+np.max(c,0))/2 for c in curves])
        moved=transform_xy(centers,a);ordered=bool(np.all(np.diff(moved[:,1])>0))
        # R2/R3 ordering is diagnostic only for the declared R1-only sensitivity arm.
        eligible=(ordered or tuple(scoring_regions)==(0,)) and all(evidence[i]['usable_for_ranking'] for i in scoring_regions)
        appearance_score=float(candidate['score'])
        if not np.isfinite(appearance_score):raise ValueError('Finite appearance score required')
        boundary_score=float(np.mean([evidence[i]['support_mean'] for i in scoring_regions]))
        candidate.update(appearance_score=appearance_score,boundary_score=boundary_score,
            combined_score=.5*float(np.clip(appearance_score,0,1))+.5*boundary_score,
            eligible_for_ranking=eligible,region_boundary_evidence=evidence,
            scoring_region_ids=[f'R{i+1}' for i in scoring_regions],
            layout=dict(transformed_reference_centers_xy=moved.tolist(),
                adjacent_displacements_xy=np.diff(moved,axis=0).tolist(),
                top_to_bottom_preserved=ordered,
                interpretation='Common-affine ordering and image-edge support are not independent identity validation'),
            independent_validation=False,anatomical_identity_verified=False,absolute_geometry_accepted=False)
        ranked.append(candidate)
    return sorted(ranked,key=lambda c:(not c['eligible_for_ranking'],-c['combined_score']))


def linear_grid_evidence(affine,ratio_xy,grid):
    """Expose linear-grid limits independently of the human-hint radius limit."""
    a=np.asarray(affine,float);transform_xy([[0.,0.]],a)
    ratio=np.asarray(ratio_xy,float);linear=a[:2,:2]
    if ratio.shape!=(2,) or not np.isfinite(ratio).all() or np.any(ratio<=0) or np.linalg.det(linear)<=0:
        raise ValueError('Positive detector sampling and orientation required')
    scale=float(np.linalg.norm(linear[:,0])/ratio[0])
    anisotropy=float(np.linalg.norm(linear[:,1])/(ratio[1]*scale))
    angle=float(np.rad2deg(np.arctan2(linear[1,0],linear[0,0])))
    values=dict(scale_factor=scale,vertical_anisotropy=anisotropy,rotation_deg=angle)
    names=dict(scale_factor='scale_factors',vertical_anisotropy='vertical_anisotropy',rotation_deg='rotation_deg')
    bounds=[]
    for name,key in names.items():
        choices=np.asarray(grid[key],float)
        if choices.ndim!=1 or not len(choices) or not np.isfinite(choices).all():raise ValueError('Finite search grid required')
        if not np.any(np.isclose(values[name],choices,rtol=1e-7,atol=1e-7)):
            raise ValueError('Candidate does not belong to the frozen linear grid')
        if np.isclose(values[name],choices.min()):bounds.append(name+'_lower')
        if np.isclose(values[name],choices.max()):bounds.append(name+'_upper')
    return dict(**values,search_bounds_touched=bounds,
        interpretation='Grid boundary is not a calibrated anatomical limit or accuracy measure')
