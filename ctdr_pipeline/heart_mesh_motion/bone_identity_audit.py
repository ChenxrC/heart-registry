"""Conservative region identities and geometric references, not an auto-numberer."""
from __future__ import annotations

from copy import deepcopy
import re

import numpy as np


def neutral_regions(review, pack, *, run_id):
    # Keep the established display/frame/coordinate validation at the input boundary.
    from experiments.heart_mesh_dynamic.probes.bone_pose_review import validate_review
    observations = validate_review(review, pack)
    visible = [o for o in observations if o['run_id'] == run_id
               and o['review_state'] == 'confirmed_visible_edge']
    visible.sort(key=lambda o: float((np.min(o['xy_px'], axis=0)[1]
                                    + np.max(o['xy_px'], axis=0)[1]) / 2))
    return [dict(region_id=f'R{i+1}', run_id=run_id, frame_index=o['frame_index'],
                 provisional_user_label=o['structure_id'],
                 source_identity_confirmed=o['identity_confirmed'],
                 source_split=o['split'], split=None, identity_state='unresolved',
                 xy_px=deepcopy(o['xy_px']), source=o.get('source'),
                 interpretation='Visible arc retained; anatomical level not yet verified.')
            for i, o in enumerate(visible)]


def lps_to_plane(points, *, origin, col_axis, row_axis, spacing_row_col):
    """Orthogonal image-plane geometry, explicitly NOT cone-beam scout calibration."""
    points = np.asarray(points, dtype=float)
    origin, col, row = (np.asarray(x, dtype=float) for x in (origin, col_axis, row_axis))
    spacing = np.asarray(spacing_row_col, dtype=float)
    if (points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all()
            or any(v.shape != (3,) or not np.isfinite(v).all() for v in (origin, col, row))
            or spacing.shape != (2,) or not np.isfinite(spacing).all() or np.any(spacing <= 0)):
        raise ValueError('Finite LPS points and positive row/column spacing required')
    basis = np.column_stack([col, row])
    if not np.allclose(basis.T @ basis, np.eye(2), atol=1e-6):
        raise ValueError('Image-plane axes must be orthonormal')
    offset = points - origin
    return np.column_stack([offset @ col / spacing[1], offset @ row / spacing[0]]), offset @ np.cross(col, row)


def chain_audit(records):
    def ordinal(label):
        match = re.fullmatch(r'([CTL])(\d+)', label)
        if match is None:
            raise ValueError('Expected C/T/L vertebral label')
        prefix, number = match.group(1), int(match.group(2))
        if not 1 <= number <= {'C': 7, 'T': 12, 'L': 5}[prefix]:
            raise ValueError('Invalid vertebral level')
        return {'C': 0, 'T': 7, 'L': 19}[prefix] + number
    if len({r['label'] for r in records}) != len(records):
        raise ValueError('Duplicate vertebral label')
    for r in records:
        ordinal(r['label'])
        c = r['centroid_lps_mm']
        if c is not None and (np.asarray(c).shape != (3,) or not np.isfinite(c).all()):
            raise ValueError('Finite centroid or explicit missing value required')
    present = sorted((r for r in records if r['centroid_lps_mm'] is not None),
                     key=lambda r: -r['centroid_lps_mm'][2])
    levels = [ordinal(r['label']) for r in present]
    return dict(cranial_to_caudal=[r['label'] for r in present],
                order_consistent=bool(len(levels) >= 2 and np.all(np.diff(levels) > 0)),
                missing_labels=[r['label'] for r in records if r['centroid_lps_mm'] is None],
                anatomical_identity_verified=False,
                interpretation='Automated-label ordering check only, not independent patient numbering truth.')


def identity_gate(regions, *, ct_chain_consistent, xa_anchor_evidence):
    """This audit phase cannot promote an uncertain XA label to accepted identity."""
    if xa_anchor_evidence:
        raise ValueError('Independent XA anchor review needs a separately reviewed importer; this audit cannot accept it')
    return dict(status='pending_independent_XA_identity_review',
                ct_chain_consistent=bool(ct_chain_consistent),
                unresolved_regions=[r['region_id'] for r in regions],
                accepted_mapping=None, rigid_refit_allowed=False,
                absolute_geometry_accepted=False, clinical_validation=False,
                patient_true_3d_tre_mm=None,
                reason='CT-only anchors and fixed-camera residuals do not identify levels in cropped XA.')
