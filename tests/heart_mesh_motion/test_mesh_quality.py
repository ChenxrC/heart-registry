import importlib
import numpy as np
import trimesh
import pytest


def test_optimization_quality_does_not_confuse_coarse_sphere_angle_with_shape_error():
    api=importlib.import_module('ctdr_pipeline.heart_mesh_motion.mesh_quality')
    sphere=trimesh.creation.icosphere(subdivisions=2,radius=50)
    result=api.audit_optimization_geometry(sphere,sphere.copy())
    assert result['dihedral_angle_p95_deg']>4
    assert result['geometric_accepted'] is True
    assert result['clinical_validation'] is False
    assert result['self_intersection']['interior_intersections']==0


def test_shape_change_is_not_accepted_as_smoothing():
    api=importlib.import_module('ctdr_pipeline.heart_mesh_motion.mesh_quality')
    sphere=trimesh.creation.icosphere(subdivisions=1,radius=30)
    changed=sphere.copy();changed.vertices*=1.2
    result=api.audit_optimization_geometry(sphere,changed)
    assert result['geometric_accepted'] is False
    assert 'volume' in result['failed_checks']


@pytest.mark.parametrize('points,faces,expected',[
    ([[0,0,0],[2,0,0],[0,2,0],[.5,.5,-1],[.5,.5,1],[1.5,.5,0]],[[0,1,2],[3,4,5]],1),
    ([[0,0,0],[2,0,0],[0,2,0],[2,2,0]],[[0,1,2],[1,3,2]],0),
    ([[0,0,0],[2,0,0],[0,2,0],[1,0,0],[0,1,0]],[[0,1,2],[0,3,4]],1),
])
def test_numerical_triangle_intersection_distinguishes_shared_boundary_from_interior(points,faces,expected):
    api=importlib.import_module('ctdr_pipeline.heart_mesh_motion.mesh_quality')
    report=api.check_self_intersections(trimesh.Trimesh(np.array(points,float),np.array(faces),process=False))
    assert report['interior_intersections']==expected


def test_projection_fidelity_uses_independent_masks_and_rejects_shifted_shape():
    api=importlib.import_module('ctdr_pipeline.heart_mesh_motion.optimization_mesh')
    from ctdr_pipeline.heart_mesh_motion.synthetic_mesh import digital_fixture
    from ctdr_pipeline.heart_mesh_motion.contracts import MeshData
    mesh,camera=digital_fixture()
    same=api.projection_fidelity(mesh,mesh,[camera])
    shifted=MeshData(mesh.vertices_lps_mm+[10,0,0],mesh.faces,mesh.source_sha256,{})
    wrong=api.projection_fidelity(mesh,shifted,[camera])
    assert same['accepted'] is True and same['views'][0]['symmetric_edge_distance_p95_px']==0
    assert wrong['accepted'] is False


def test_optimization_ready_requires_actual_backward_not_only_geometric_quality():
    api=importlib.import_module('ctdr_pipeline.heart_mesh_motion.optimization_mesh')
    from ctdr_pipeline.heart_mesh_motion.synthetic_mesh import digital_fixture
    mesh,camera=digital_fixture()
    prepared=api.prepare_optimization_mesh(mesh,[camera],max_faces=200)
    if importlib.util.find_spec('pytorch3d') is None:
        assert prepared.report['optimization_ready'] is False
    else:
        assert prepared.report['optimization_ready'] is True
        assert prepared.report['backward_probe']['gradient_norm']>0
    assert prepared.report['clinical_validation'] is False

