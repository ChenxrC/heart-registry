import importlib
import numpy as np
import hashlib
import json
from pathlib import Path
import pytest


def test_projection_audit_is_geometry_only_not_xa_matching():
    contracts=importlib.import_module('ctdr_pipeline.heart_mesh_motion.contracts')
    cam=contracts.CameraData([0,0,0],[1,0,0],[0,1,0],[0,0,1],[0,0,10],[0,0],[1,1],(12,12),{},False)
    mesh=contracts.MeshData(np.array([[1,1,10],[4,1,10],[1,4,10]],float),np.array([[0,1,2]]),'a'*64,{})
    api=importlib.import_module('ctdr_pipeline.heart_mesh_motion.geometry')
    result=api.audit_projection(mesh,cam,np.zeros((3,12,12),dtype=np.uint8),[0,2])
    assert result['frame_indices']==[0,2]
    assert result['geometry_accepted'] is False
    assert result['measured_xa_heart_edge_error_px'] is None
    assert result['projected_foreground_pixels']>0


def test_audit_rejects_consumed_cine_that_disagrees_with_frozen_digest(tmp_path,monkeypatch):
    from ctdr_pipeline.heart_mesh_motion import audit,inputs
    from ctdr_pipeline.heart_mesh_motion.synthetic_mesh import digital_fixture
    case=tmp_path/'case';assets=tmp_path/'assets';case.mkdir();assets.mkdir()
    preflight=tmp_path/'prior/preflight.json';preflight.parent.mkdir();preflight.write_text('{}')
    series=assets/'all_series/series_01';series.mkdir(parents=True)
    manifest=series/'series_manifest.json';manifest.write_text(json.dumps({'shape_frame_row_column':[31,8,8]}))
    cine=series/'cine_u8.raw';cine.write_bytes(bytes(31*8*8))
    mask=assets/'thorax_masks/heart.nii.gz';mask.parent.mkdir();mask.write_bytes(b'fixture')
    frozen={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (manifest,cine,mask)}
    monkeypatch.setattr(inputs,'OUTPUT_ROOT',tmp_path/'results')
    monkeypatch.setattr(audit,'freeze_inputs',lambda *_:{'sha256':frozen})
    monkeypatch.setattr(audit,'load_reference_mesh',lambda *a,**kw:digital_fixture()[0])
    original=Path.read_bytes
    monkeypatch.setattr(Path,'read_bytes',lambda p:bytes([1])*(31*8*8) if p==cine else original(p))
    output=tmp_path/'results/fresh'
    with pytest.raises(ValueError,match='Consumed input SHA256'):
        audit.run_projection_audit(case,assets,preflight,output)
    assert not output.exists()
