"""Actual partial XA-heart observations; no camera/shape fitting or acceptance."""
from __future__ import annotations
from copy import deepcopy
import json

import numpy as np
from scipy import ndimage

from .bone_local_match import build_edge_field
from .reviewed_static_pose import sample_curve

COORDINATES='0-based native pixel centers; x=column/right, y=row/down'
FALSE_FLAGS=('independent_blinded_truth','automatic_edges_used_as_truth','anatomical_identity_verified',
             'absolute_geometry_accepted','clinical_validation','real_patient_mesh_fit_executed','heart_motion_reconstructed')


def _xy(value,*,single=False):
    a=np.asarray(value)
    if (a.dtype.kind not in 'fiu' or not np.isfinite(a).all()
            or (a.shape!=(2,) if single else a.ndim!=2 or a.shape[1]!=2)):
        raise ValueError('Finite numeric native xy coordinates required')
    return a.astype(float)


def arc_inward_normals(points,side_point):
    xy=_xy(points);inside=_xy(side_point,single=True)
    if not 2<=len(xy)<=4096:raise ValueError('An observed arc needs2–4096 points')
    delta=np.diff(xy,axis=0);length=np.linalg.norm(delta,axis=1)
    if np.any(length<1e-6) or length.sum()<1:raise ValueError('Distinct consecutive arc points and length>=1px required')
    normals=np.column_stack([-delta[:,1],delta[:,0]])/length[:,None]
    dot=np.sum((inside-(xy[:-1]+xy[1:])/2)*normals,axis=1)
    if np.any(np.abs(dot)<.5):raise ValueError('Inward side ambiguous; choose a point clearly on the heart side of every segment')
    return normals*np.where(dot>0,1.,-1.)[:,None]


def _inside_native(points,hw):
    if np.any(points<0) or np.any(points>np.asarray(hw)[::-1]-1):
        raise ValueError('Observation lies outside native image')


def _inside_visible_detector(points,frame,hw):
    _inside_native(points,hw)
    intervals=np.asarray(frame['valid_row_intervals'])
    if intervals.shape!=(hw[0],2) or intervals.dtype.kind not in 'iu':
        raise ValueError('Native valid-row detector intervals required')
    rows=np.floor(points[:,1]+.5).astype(int)
    ranges=intervals[rows]
    if (np.any(ranges[:,0]>ranges[:,1]) or np.any(points[:,0]<ranges[:,0]) or np.any(points[:,0]>ranges[:,1])):
        raise ValueError('Confirmed heart arcs cannot follow unknown detector padding or collimation edges')


def validate_heart_edge_review(document,material):
    expected=dict(schema='human_visible_heart_edges_v1',case_id=material['case_id'],run_id=material['run_id'],
                  display_material_sha256=material['display_material_sha256'],selection_sha256=material['selection_sha256'],
                  selected_camera_sha256=material['selected_camera_sha256'],coordinate_convention=COORDINATES)
    if not isinstance(document,dict) or any(document.get(k)!=v for k,v in expected.items()):
        raise ValueError('Heart review material, case, run or selected camera differs')
    json.dumps(document,allow_nan=False)
    if not isinstance(document.get('reviewer'),str) or not document['reviewer'].strip():raise ValueError('Reviewer required')
    if (document.get('suggestion_assisted') is not True or any(document.get(k) is not False for k in FALSE_FLAGS)
            or document.get('patient_true_3d_tre_mm','missing') is not None):
        raise ValueError('Observations cannot claim fitted motion, independent truth or geometry/clinical acceptance')
    observations=document.get('observations');hw=tuple(material['image_hw'])
    known={f['frame_index']:f for f in material['frames']}
    if not isinstance(observations,list) or len(observations)>len(known):raise ValueError('Explicit frame observations required')
    seen=set();frames=[]
    for obs in observations:
        if not isinstance(obs,dict):raise ValueError('Frame observation object required')
        index=obs.get('frame_index')
        if type(index) is not int or index not in known or index in seen:raise ValueError('Unknown or duplicate frame')
        seen.add(index);source=known[index]
        if obs.get('frame_sha256')!=source['frame_sha256'] or obs.get('timestamp_ms')!=source['timestamp_ms']:
            raise ValueError('Actual frame bytes or acquisition time differs')
        state=obs.get('review_state');confirmed=state=='visible_confirmed'
        if state not in ('visible_confirmed','uncertain','not_visible'):
            raise ValueError('Explicit visibility review state required')
        if obs.get('visible_heart_edge_confirmed') is not confirmed:raise ValueError('Visibility checkbox differs from state')
        arcs=obs.get('arcs');polygons=obs.get('excluded_polygons_xy_px')
        if not isinstance(arcs,list) or len(arcs)>8 or (confirmed and not arcs) or (state=='not_visible' and arcs):
            raise ValueError('Visible arcs required; not-visible frames cannot retain guessed arcs')
        actual=[]
        for arc in arcs:
            xy=_xy(arc.get('xy_px'))
            if not 2<=len(xy)<=4096:raise ValueError('An observed arc needs2–4096 points')
            _inside_native(xy,hw)
            samples=sample_curve(xy)
            side=arc.get('inward_side_point_xy_px');normals=None
            if side is not None:
                side_array=_xy(side,single=True);_inside_native(side_array[None],hw)
                normals=arc_inward_normals(xy,side_array).tolist()
            if confirmed:
                if normals is None:raise ValueError('Confirmed arc requires an actual inward-side point')
                _inside_visible_detector(samples,source,hw)
                _inside_visible_detector(side_array[None],source,hw)
            actual.append(dict(xy_px=deepcopy(arc['xy_px']),inward_side_point_xy_px=deepcopy(side),
                               inward_normal_xy=normals,normal_scope='one unit inward normal per actual consecutive segment',
                               automatically_closed=False))
        if not isinstance(polygons,list) or len(polygons)>16:raise ValueError('At most16 explicit exclusion polygons')
        for polygon in polygons:
            xy=_xy(polygon)
            if not 3<=len(xy)<=256:raise ValueError('Exclusion polygon needs3–256 points')
            _inside_native(xy,hw)
        if not isinstance(obs.get('note'),str):raise ValueError('Observation note must be text')
        frames.append({**deepcopy(obs),'arcs':actual,'unknown_pixels_are_background':False,
                       'excluded_regions_are_background':False,'measured_heart_edge_error_px':None})
    visible=sum(f['review_state']=='visible_confirmed' for f in frames)
    return dict(schema='validated_visible_heart_evidence_v1',case_id=material['case_id'],run_id=material['run_id'],
                status='manual_partial_heart_evidence_requires_fit_review' if visible else 'insufficient_confirmed_heart_observations',
                frames=frames,summary=dict(confirmed_visible_frame_count=visible,
                    uncertain_frame_count=sum(f['review_state']=='uncertain' for f in frames),
                    not_visible_frame_count=sum(f['review_state']=='not_visible' for f in frames),
                    unreviewed_frame_indices=sorted(set(known)-seen),whole_heart_foreground_mask_available=False),
                anatomical_identity_verified=False,absolute_geometry_accepted=False,clinical_validation=False,
                real_patient_mesh_fit_executed=False,heart_motion_reconstructed=False,patient_true_3d_tre_mm=None)


def detector_intervals(image):
    a=np.asarray(image)
    valid=ndimage.binary_fill_holes(a>0)
    safe=ndimage.binary_erosion(valid,iterations=4,border_value=0)
    result=[]
    for row in safe:
        x=np.flatnonzero(row);result.append([int(x[0]),int(x[-1])] if len(x) else [1,0])
    return result


def propose_heart_edge_points(image,length_mm,*,radius_px=24.,quantile=.8):
    image=np.asarray(image);length=np.asarray(length_mm,float)
    if image.ndim!=2 or length.shape!=image.shape or not np.isfinite(length).all() or np.any(length<0):
        raise ValueError('Native raw image and fixed mesh path-length map required')
    if not np.isfinite(radius_px) or radius_px<=0:raise ValueError('Positive search-band radius required')
    valid=ndimage.binary_fill_holes(image>0)
    if not valid.any():return dict(points_xy_px=[],automatic_edges_are_truth=False,source='raw image gradients only')
    field=build_edge_field(image,valid,quantile=quantile)
    mask=length>1e-6;edge=mask&~ndimage.binary_erosion(mask,border_value=1)
    edge[[0,-1]]=False;edge[:,[0,-1]]=False
    points=field['xy']
    if edge.any() and len(points):
        distance=ndimage.distance_transform_edt(~edge)
        keep=distance[points[:,1].astype(int),points[:,0].astype(int)]<=radius_px
        points=points[keep]
    else:points=np.empty((0,2))
    return dict(points_xy_px=points.tolist(),automatic_edges_are_truth=False,
                source='raw image gradient ridges, restricted by fixed CT-outline search band; not cardiac segmentation',
                search_radius_px=float(radius_px),gradient_quantile=float(quantile))
