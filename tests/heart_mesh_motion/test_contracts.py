import importlib

import numpy as np
import pytest


def api():
    return importlib.import_module('ctdr_pipeline.heart_mesh_motion.contracts')


def test_mesh_rejects_bad_faces_and_nonfinite_vertices():
    module = api()
    v = np.array([[0., 0, 10], [1, 0, 10], [0, 1, 10]])
    with pytest.raises(ValueError):
        module.MeshData(v, np.array([[0, 1, 3]]), 'a' * 64, {})
    with pytest.raises(ValueError):
        module.MeshData(v * float('nan'), np.array([[0, 1, 2]]), 'a' * 64, {})


def test_camera_requires_orthonormal_right_handed_basis_and_positive_distance():
    module = api()
    values = dict(source_lps_mm=[0,0,0], detector_center_lps_mm=[0,0,1000],
        u_lps=[1,0,0],v_lps=[0,1,0],n_lps=[0,0,1],principal_xy_px=[31.5,31.5],
        spacing_row_col_mm=[1,2],image_hw=(64,64),provenance={},geometry_accepted=False)
    camera = module.CameraData(**values)
    assert camera.geometry_accepted is False
    with pytest.raises(ValueError):
        module.CameraData(**{**values,'n_lps':[0,0,-1]})
    with pytest.raises(ValueError):
        module.CameraData(**{**values,'spacing_row_col_mm':[0,2]})


@pytest.mark.parametrize('value', [True, 0, -1, 2.0])
def test_invalid_image_dimensions_rejected(value):
    module=api()
    with pytest.raises(ValueError):
        module.CameraData([0,0,0],[1,0,0],[0,1,0],[0,0,1],[0,0,1000],[0,0],[1,1],(value,64),{},False)
