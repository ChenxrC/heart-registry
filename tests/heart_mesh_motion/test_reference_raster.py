import importlib
import numpy as np
import pytest


def test_triangle_raster_is_not_a_convex_hull_and_uses_integer_pixel_centers():
    c=importlib.import_module('ctdr_pipeline.heart_mesh_motion.contracts').CameraData
    camera=c([0,0,0],[1,0,0],[0,1,0],[0,0,1],[0,0,10],[0,0],[1,1],(12,12),{},False)
    raster=importlib.import_module('ctdr_pipeline.heart_mesh_motion.reference_raster').render_hard
    vertices=np.array([[1,1,10],[4,1,10],[1,4,10],[7,7,10],[10,7,10],[10,10,10]],float)
    image=raster(vertices,np.array([[0,1,2],[3,4,5]]),camera,image_hw=(12,12))
    assert image.shape==(12,12)
    assert image[1,1] and image[2,2] and not image[4,4]
    assert image[7,9] and not image[6,6]
    assert not image[:,0].any()


def test_offscreen_triangle_does_not_become_a_border_edge():
    c=importlib.import_module('ctdr_pipeline.heart_mesh_motion.contracts').CameraData
    camera=c([0,0,0],[1,0,0],[0,1,0],[0,0,1],[0,0,10],[0,0],[1,1],(12,12),{},False)
    raster=importlib.import_module('ctdr_pipeline.heart_mesh_motion.reference_raster').render_hard
    image=raster(np.array([[-20,-20,10],[-10,-20,10],[-20,-10,10]]),np.array([[0,1,2]]),camera,image_hw=(12,12))
    assert image.sum()==0
