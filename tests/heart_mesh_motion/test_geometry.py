import importlib
import numpy as np
import pytest


def camera():
    cls=importlib.import_module('ctdr_pipeline.heart_mesh_motion.contracts').CameraData
    return cls([0,0,0],[1,0,0],[0,1,0],[0,0,1],[0,0,1000],[31.5,31.5],[1,2],(64,64),{},False)


def test_analytic_projection_anisotropic_spacing_and_pixel_center_scaling():
    api=importlib.import_module('ctdr_pipeline.heart_mesh_motion.geometry')
    uv,depth=api.project_lps(np.array([[5.,10,500],[-10,-5,1000]]),camera())
    np.testing.assert_allclose(uv,[[36.5,51.5],[26.5,26.5]],atol=1e-10)
    np.testing.assert_allclose(depth,[500,1000])
    small=api.resize_camera(camera(),(32,32))
    np.testing.assert_allclose(api.project_lps(np.array([[5.,10,500]]),small)[0],[[18.,25.5]])


@pytest.mark.parametrize('z',[0,-1,float('nan')])
def test_behind_source_never_projects_as_valid(z):
    api=importlib.import_module('ctdr_pipeline.heart_mesh_motion.geometry')
    with pytest.raises(ValueError):api.project_lps(np.array([[0.,0,z]]),camera())
