import importlib
import hashlib
import numpy as np
import SimpleITK as sitk
import pytest


def test_reference_mesh_preserves_physical_coordinates_and_source(tmp_path):
    api=importlib.import_module('ctdr_pipeline.heart_mesh_motion.mesh')
    data=np.zeros((20,20,20),np.uint8);data[4:12,5:13,6:14]=1
    image=sitk.GetImageFromArray(data);image.SetSpacing([2,3,4]);image.SetOrigin([100,200,-500])
    path=tmp_path/'heart.nii.gz';sitk.WriteImage(image,str(path));before=hashlib.sha256(path.read_bytes()).hexdigest()
    mesh=api.load_reference_mesh(path,max_faces=10000)
    assert len(mesh.faces)<=10000
    np.testing.assert_allclose(mesh.report['centroid_lps_mm'],[119,225.5,-470],atol=1)
    assert mesh.source_sha256==before==hashlib.sha256(path.read_bytes()).hexdigest()
    assert mesh.report['watertight'] is True


def test_mask_consumption_rejects_unfrozen_bytes_before_extraction(tmp_path):
    api=importlib.import_module('ctdr_pipeline.heart_mesh_motion.mesh')
    path=tmp_path/'heart.nii.gz';path.write_bytes(b'changed mask')
    with pytest.raises(ValueError,match='SHA256|consumed'):
        api.load_reference_mesh(path,expected_sha256='a'*64)
