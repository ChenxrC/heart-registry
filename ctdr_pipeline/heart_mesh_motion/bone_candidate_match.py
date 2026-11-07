"""Unaccepted 2D appearance proposals between native XA frames, not 3D poses."""
from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage, signal


def detector_sampling_ratio(source_row_col, target_row_col):
    source, target = np.asarray(source_row_col, float), np.asarray(target_row_col, float)
    if (source.shape != (2,) or target.shape != (2,) or not np.isfinite([source,target]).all()
            or np.any(source <= 0) or np.any(target <= 0)):
        raise ValueError('Positive finite detector row/column sampling required')
    return (source / target)[::-1]  # x then y; sampling hint only, not patient depth.


def transform_xy(points, affine):
    p, a = np.asarray(points,float), np.asarray(affine,float)
    if (p.ndim != 2 or p.shape[1] != 2 or a.shape != (3,3)
            or not np.isfinite(p).all() or not np.isfinite(a).all()
            or not np.allclose(a[2],[0,0,1])):
        raise ValueError('Finite xy coordinates and 2D affine required')
    return p @ a[:2,:2].T + a[:2,2]


def native_affine(source_to_patch, *, top_left_yx):
    a = np.asarray(source_to_patch,float).copy()
    transform_xy([[0,0]],a)
    yx = np.asarray(top_left_yx,float)
    if yx.shape != (2,) or not np.isfinite(yx).all():
        raise ValueError('Finite target patch row/column required')
    a[:2,2] += yx[::-1]
    return a


def arc_band(image_hw, curves, *, width=32):
    if len(image_hw)!=2 or any(type(v) is not int or v<1 for v in image_hw):
        raise ValueError('Positive integer image shape required')
    if type(width) is not int or width<1:raise ValueError('Positive integer band width required')
    image=Image.new('L',(image_hw[1],image_hw[0]));draw=ImageDraw.Draw(image)
    for curve in curves:
        xy=np.asarray(curve,float)
        if (xy.ndim!=2 or xy.shape[1]!=2 or len(xy)<2 or not np.isfinite(xy).all()
                or np.any(xy<0) or np.any(xy[:,0]>image_hw[1]-1) or np.any(xy[:,1]>image_hw[0]-1)):
            raise ValueError('Observed finite native coordinates required')
        # No polygon fill and no inferred segment from the last point to the first.
        draw.line([tuple(p) for p in xy],fill=255,width=width)
    return np.asarray(image,dtype=float)/255


def masked_ncc(template, weights, target):
    t,w,im=(np.asarray(v,float) for v in (template,weights,target))
    if (t.ndim!=2 or w.shape!=t.shape or im.ndim!=2 or np.any(np.asarray(t.shape)>im.shape)
            or not all(np.isfinite(v).all() for v in (t,w,im))):
        raise ValueError('Finite 2D template/weights/target with valid dimensions required')
    total=float(w.sum())
    if total<1e-8 or np.any(w<0):raise ValueError('Nonnegative nonempty weights required')
    shape=tuple(np.asarray(im.shape)-np.asarray(t.shape)+1)
    var_t=float(np.sum(w*(t-np.sum(w*t)/total)**2))
    result=np.full(shape,-np.inf)
    if var_t <= 1e-10*total:return result
    corr=lambda a,b:signal.correlate(a,b,mode='valid',method='fft')
    s=corr(im,w);var_i=np.maximum(corr(im*im,w)-s*s/total,0)
    numerator=corr(im,w*(t-np.sum(w*t)/total))
    valid=var_i>1e-10*total
    result[valid]=np.clip(numerator[valid]/np.sqrt(var_t*var_i[valid]),-1,1)
    return result


def spatial_nms(candidates, *, max_candidates=5, radius_px=18.):
    if type(max_candidates) is not int or max_candidates<1 or not np.isfinite(radius_px) or radius_px<0:
        raise ValueError('Positive candidate budget and finite suppression radius required')
    selected=[]
    for candidate in sorted(candidates,key=lambda c:-c['score']):
        xy=np.asarray(candidate['reference_xy_px'],float)
        if xy.shape!=(2,) or not np.isfinite(xy).all() or not np.isfinite(candidate['score']):
            raise ValueError('Finite candidate score/position required')
        if all(np.linalg.norm(xy-np.asarray(s['reference_xy_px']))>=radius_px for s in selected):
            selected.append(candidate)
            if len(selected)==max_candidates:break
    return selected


def _features(image):
    image=np.asarray(image,float)/255.
    smooth=ndimage.gaussian_filter(image,1.)
    highpass=smooth-ndimage.gaussian_filter(image,3.)
    return highpass,np.hypot(ndimage.sobel(smooth,0),ndimage.sobel(smooth,1))


def _warp_template(source, band, linear):
    yx=np.argwhere(band>0)
    if len(yx)<16:return None
    lo=yx.min(0)[::-1];hi=yx.max(0)[::-1]
    corners=np.array([[lo[0],lo[1]],[hi[0],lo[1]],[lo[0],hi[1]],[hi[0],hi[1]]])@linear.T
    low=np.floor(corners.min(0)).astype(int)-12
    high=np.ceil(corners.max(0)).astype(int)+12
    hw=(high-low+1)[::-1]
    if np.any(hw<5):return None
    y,x=np.indices(hw);global_xy=np.stack([x+low[0],y+low[1]],-1)
    source_xy=global_xy@np.linalg.inv(linear).T
    rc=np.stack([source_xy[...,1],source_xy[...,0]])
    smallest=np.linalg.svd(linear,compute_uv=False).min()
    anti_alias=ndimage.gaussian_filter(source,max(0.,.5/smallest-.5))
    # Extend intensity continuously for feature filters. Zero weights, not a
    # fabricated black frame, represent unobserved source pixels.
    image=ndimage.map_coordinates(anti_alias,rc,order=1,mode='nearest')
    weights=ndimage.map_coordinates(band,rc,order=1,mode='constant',cval=0.)
    if weights.sum()<16:return None
    a=np.eye(3);a[:2,:2]=linear;a[:2,2]=-low
    return image,weights,a


def peak_proposals(scores,source_to_patch,*,reference_xy,template_band_weight):
    """Keep all separated local maxima until global spatial suppression."""
    scores=np.asarray(scores,float)
    maxima=np.isfinite(scores)&(scores>=.05)&(scores==ndimage.maximum_filter(scores,size=15,mode='constant',cval=-np.inf))
    proposals=[]
    for yx in np.argwhere(maxima):
        a=native_affine(source_to_patch,top_left_yx=yx)
        proposals.append(dict(score=float(scores[tuple(yx)]),reference_xy_px=transform_xy(np.asarray(reference_xy,float)[None],a)[0].tolist(),
            source_to_target_affine=a.tolist(),template_band_weight=float(template_band_weight),
            anatomical_identity_verified=False,absolute_geometry_accepted=False,
            matrix_kind='2D native-pixel affine, NOT a CT-to-XA 3D pose'))
    return proposals


def search_candidates(source,target,curves,linears,*,reference_xy,center_roi_xyxy,
                      band_width_px=32,max_candidates=5,nms_radius_px=18.,target_valid=None,
                      center_hint_xy=None,center_hint_radius_px=None):
    source,target=(np.asarray(im,float) for im in (source,target))
    if (source.ndim!=2 or target.ndim!=2 or not np.isfinite(source).all() or not np.isfinite(target).all()):
        raise ValueError('Finite 2D native images required')
    ref=np.asarray(reference_xy,float);roi=np.asarray(center_roi_xyxy,float)
    if (ref.shape!=(2,) or roi.shape!=(4,) or not np.isfinite(ref).all() or not np.isfinite(roi).all()
            or np.any(roi[2:]<roi[:2])):raise ValueError('Finite reference and search rectangle required')
    hint=None
    if center_hint_xy is not None or center_hint_radius_px is not None:
        hint=np.asarray(center_hint_xy,float)
        if (hint.shape!=(2,) or not np.isfinite(hint).all() or center_hint_radius_px is None
                or not np.isfinite(center_hint_radius_px) or center_hint_radius_px<=0):
            raise ValueError('Finite center hint and positive radius must be supplied together')
    valid=np.ones(target.shape,float) if target_valid is None else np.asarray(target_valid,float)
    if valid.shape!=target.shape or not np.isfinite(valid).all() or np.any((valid<0)|(valid>1)):
        raise ValueError('Target validity must lie in [0,1]')
    band=arc_band(tuple(source.shape),curves,width=band_width_px)
    target_features=_features(target);proposals=[]
    for linear in linears:
        linear=np.asarray(linear,float)
        if linear.shape!=(2,2) or not np.isfinite(linear).all() or np.linalg.det(linear)<=0:
            raise ValueError('Positive orientation-preserving linear map required')
        warped=_warp_template(source,band,linear)
        if warped is None:continue
        image,weights,source_to_patch=warped
        if np.any(np.asarray(image.shape)>target.shape):continue
        features=_features(image)
        maps=[masked_ncc(t,weights,im) for t,im in zip(features,target_features)]
        finite=np.isfinite(maps[0])&np.isfinite(maps[1])
        scores=np.full(maps[0].shape,-np.inf)
        scores[finite]=.65*maps[0][finite]+.35*maps[1][finite]
        coverage=signal.correlate(valid,weights,mode='valid',method='fft')/weights.sum()
        yy,xx=np.indices(scores.shape);anchor=transform_xy(ref[None],source_to_patch)[0]
        allowed=(coverage>=.98)&(xx+anchor[0]>=roi[0])&(xx+anchor[0]<=roi[2])&(yy+anchor[1]>=roi[1])&(yy+anchor[1]<=roi[3])
        if hint is not None:
            allowed &= (xx+anchor[0]-hint[0])**2+(yy+anchor[1]-hint[1])**2<=center_hint_radius_px**2
        scores[~allowed]=-np.inf
        proposals.extend(peak_proposals(scores,source_to_patch,reference_xy=ref,template_band_weight=weights.sum()))
    return spatial_nms(proposals,max_candidates=max_candidates,radius_px=nms_radius_px)
