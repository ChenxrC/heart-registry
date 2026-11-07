"""Use registration anatomy, not the beautified display surface, for fitting."""
from __future__ import annotations

from pathlib import Path
import numpy as np
import trimesh
import tempfile

from .contracts import MeshData
from .inputs import sha256,read_bytes_snapshot


class MeshPreparationError(ValueError):
    """Failed fitting mesh, with untouched full source for diagnostic display only."""
    def __init__(self,message,diagnostic_mesh):
        super().__init__(message)
        self.diagnostic_mesh=diagnostic_mesh


def load_reference_mesh(mask_path: Path,*,max_faces: int=10000,expected_sha256: str|None=None) -> MeshData:
    if type(max_faces) is not int or max_faces<4:raise ValueError('max_faces must be an integer >=4')
    content,before=read_bytes_snapshot(mask_path,expected_sha256=expected_sha256)
    from ctdr_pipeline.visualization.surface_mesh import extract_registration_surface,audit_surface_pair
    # SimpleITK's filename API reads a private same-byte snapshot, not a reopened source.
    scratch=Path(__file__).resolve().parents[2]/'tmp'
    scratch.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='heart_mesh_snapshot_',dir=scratch) as folder:
        snapshot=Path(folder)/'heart.nii.gz';snapshot.write_bytes(content)
        registration=extract_registration_surface(snapshot,'heart').as_trimesh()
    if not registration.is_watertight or not registration.is_winding_consistent or abs(registration.volume)<1e-8:
        raise ValueError('Reference heart surface is not a closed consistent volume')
    mesh=registration.copy();audit=None
    if len(mesh.faces)>max_faces:
        import pyvista as pv
        surface=pv.PolyData(mesh.vertices,np.column_stack([np.full(len(mesh.faces),3),mesh.faces]).ravel())
        reduced=surface.decimate_pro(1-max_faces/len(mesh.faces),preserve_topology=True,boundary_vertex_deletion=False)
        mesh=trimesh.Trimesh(vertices=np.asarray(reduced.points),faces=np.asarray(reduced.faces).reshape(-1,4)[:,1:],process=False)
        audit=audit_surface_pair(registration,mesh,'heart').to_dict()
        if not audit['within_tolerance'] or len(mesh.faces)>max_faces:
            if sha256(mask_path)!=before:raise ValueError('Input mask changed during extraction')
            diagnostic=MeshData(registration.vertices,registration.faces,before,{
                'surface_kind':'registration_full_diagnostic_only','optimization_ready':False,
                'centroid_lps_mm':registration.center_mass.tolist(),'volume_mm3':abs(float(registration.volume)),
                'watertight':bool(registration.is_watertight),'simplification_audit':audit,
                'registration_faces':len(registration.faces),'rejected_candidate_faces':len(mesh.faces)})
            raise MeshPreparationError(f'Optimization mesh simplification failed acceptance: {audit}',diagnostic)
    if sha256(mask_path)!=before:raise ValueError('Input mask changed during extraction')
    if not mesh.is_watertight or not mesh.is_winding_consistent:raise ValueError('Simplified mesh topology is invalid')
    # Mapping is provenance, not a claim of new anatomical correspondences.
    from scipy.spatial import cKDTree
    distance,index=cKDTree(registration.vertices).query(mesh.vertices)
    report={'surface_kind':'registration','optimization_ready':True,'watertight':bool(mesh.is_watertight),'winding_consistent':bool(mesh.is_winding_consistent),
        'centroid_lps_mm':mesh.center_mass.tolist(),'volume_mm3':abs(float(mesh.volume)),
        'registration_faces':len(registration.faces),'optimization_faces':len(mesh.faces),'simplification_audit':audit,
        'nearest_source_vertex_index':index.tolist(),'nearest_source_vertex_distance_mm_max':float(distance.max()),
        'coordinate_system':'DICOM_patient_LPS_mm','clinical_validation':False}
    return MeshData(mesh.vertices,mesh.faces,before,report)
