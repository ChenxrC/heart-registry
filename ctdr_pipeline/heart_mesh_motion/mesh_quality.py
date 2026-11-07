"""Versioned numerical optimization quality; never replaces display/anatomy QA."""
from __future__ import annotations

import numpy as np
import trimesh


def check_self_intersections(mesh: trimesh.Trimesh) -> dict:
    """Broad-phase cell bounds + VTK triangle test, with shared boundaries excluded.

    Shared-vertex pairs are inset towards each centroid by 1e-6 before testing:
    legitimate shared edges/points are not interior intersections. This is a
    finite-precision numerical check, not an exact-arithmetic certificate.
    """
    import vtk
    import pyvista as pv
    triangles=np.asarray(mesh.triangles,dtype=np.float64)
    if not np.isfinite(triangles).all():raise ValueError('Nonfinite triangles')
    poly=pv.PolyData(mesh.vertices,np.c_[np.full(len(mesh.faces),3),mesh.faces].ravel())
    locator=vtk.vtkStaticCellLocator();locator.SetDataSet(poly);locator.BuildLocator()
    ids=vtk.vtkIdList();count=0;pairs=[];tested=0
    for i,a in enumerate(triangles):
        lo=a.min(0)-1e-9;hi=a.max(0)+1e-9
        locator.FindCellsWithinBounds([lo[0],hi[0],lo[1],hi[1],lo[2],hi[2]],ids)
        for item in range(ids.GetNumberOfIds()):
            j=ids.GetId(item)
            if j<=i:continue
            b=triangles[j]
            shared=bool(np.isin(mesh.faces[i],mesh.faces[j]).any())
            aa=a+(a.mean(0)-a)*1e-6 if shared else a
            bb=b+(b.mean(0)-b)*1e-6 if shared else b
            tested+=1
            if vtk.vtkTriangle.TrianglesIntersect(*aa,*bb):
                count+=1
                if len(pairs)<20:pairs.append([i,j])
    return {'method':'VTK_static_cell_locator_and_TrianglesIntersect','arithmetic':'float64',
        'shared_pair_inset_fraction':1e-6,'broad_phase_padding_mm':1e-9,
        'tested_pairs':tested,'interior_intersections':count,'first_pairs':pairs,
        'exact_predicate_certificate':False}


def audit_optimization_geometry(reference: trimesh.Trimesh,candidate: trimesh.Trimesh) -> dict:
    from ctdr_pipeline.visualization.surface_mesh import audit_surface_pair
    legacy=audit_surface_pair(reference,candidate,'heart').to_dict()
    tri=candidate.triangles
    squared=np.square(np.diff(np.concatenate([tri,tri[:,:1]],axis=1),axis=1)).sum(axis=(1,2))
    quality=4*np.sqrt(3)*candidate.area_faces/np.maximum(squared,1e-20)
    intersections=check_self_intersections(candidate)
    duplicate=len(candidate.faces)-len(np.unique(np.sort(candidate.faces,axis=1),axis=0))
    checks={
        'closed':bool(candidate.is_watertight),'winding':bool(candidate.is_winding_consistent),
        'topology_preserved':candidate.euler_number==reference.euler_number and candidate.body_count==reference.body_count,
        'centroid':legacy['centroid_shift_mm']<=.5,
        'volume':.97<=legacy['absolute_volume_ratio']<=1.03,
        'surface_distance':legacy['bidirectional_surface_distance_p95_mm']<=1.5,
        'manifold':legacy['non_manifold_edge_delta']<=0,
        'nondegenerate':bool(np.all(candidate.area_faces>1e-10)),
        'unique_faces':duplicate==0,
        'triangle_quality':bool(quality.min()>=.01 and np.percentile(quality,1)>=.05),
        'self_intersection':intersections['interior_intersections']==0,
    }
    checks={k:bool(v) for k,v in checks.items()}
    return {'profile':'optimization_geometry_v1','geometric_accepted':all(checks.values()),
        'failed_checks':[k for k,v in checks.items() if not v],'checks':checks,
        'legacy_display_audit':legacy,'dihedral_angle_p95_deg':legacy['adjacency_angle_p95_deg'],
        'dihedral_is_optimization_hard_gate':False,'triangle_quality_min':float(quality.min()),
        'triangle_quality_p01':float(np.percentile(quality,1)),
        'reference_euler_number':int(reference.euler_number),'candidate_euler_number':int(candidate.euler_number),
        'reference_components':int(reference.body_count),'candidate_components':int(candidate.body_count),
        'self_intersection':intersections,'clinical_validation':False,
        'source_anatomical_topology_independently_verified':False,
        'thresholds':dict(centroid_mm=.5,volume_ratio=[.97,1.03],surface_distance_p95_mm=1.5,
            triangle_quality_min=.01,triangle_quality_p01=.05,degenerate_area_mm2=1e-10),
        'interpretation':'Engineering numerical usability only; anatomical source and calibration require independent evidence.'}
