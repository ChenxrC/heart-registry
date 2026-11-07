import importlib
import json
from pathlib import Path

import pytest


def api():
    return importlib.import_module('ctdr_pipeline.heart_mesh_motion.inputs')


def test_snapshot_parsed_content_and_digest_are_the_same_bytes(tmp_path,monkeypatch):
    path=tmp_path/'a.json';path.write_text('{"value":1}',encoding='utf-8')
    original=Path.read_bytes
    def replace(p):
        content=original(p)
        if p==path:p.write_text('{"value":2}',encoding='utf-8')
        return content
    monkeypatch.setattr(Path,'read_bytes',replace)
    data,digest=api().read_json_snapshot(path)
    assert data=={'value':1}
    import hashlib
    assert digest==hashlib.sha256(b'{"value":1}').hexdigest()
    with pytest.raises(ValueError,match='changed'):
        api().assert_unchanged({path:digest})


def test_output_existing_or_overlapping_inputs_rejected_before_write(tmp_path,monkeypatch):
    monkeypatch.setattr(api(),'OUTPUT_ROOT',tmp_path/'results',raising=False)
    source=tmp_path/'source';source.mkdir()
    with pytest.raises(ValueError):api().reserve_output([source],source/'inside')
    with pytest.raises(FileExistsError):api().reserve_output([source],source)
    target=tmp_path/'results/new';assert api().reserve_output([source],target)==target.resolve()
    assert target.is_dir()


@pytest.mark.parametrize('relative',['1234567/fresh','artifacts/old_run/fresh','artifacts/review_candidate/fresh','results/old/fresh'])
def test_all_outputs_reject_protected_or_nested_destinations_before_mkdir(tmp_path,monkeypatch,relative):
    from unittest.mock import patch
    monkeypatch.setattr(api(),'OUTPUT_ROOT',tmp_path/'results',raising=False)
    with patch.object(Path,'mkdir') as create:
        with pytest.raises(ValueError):api().reserve_output([],tmp_path/relative)
    create.assert_not_called()


def test_consumed_bytes_must_equal_frozen_hash_even_if_path_hash_is_restored(tmp_path,monkeypatch):
    import hashlib
    path=tmp_path/'cine.raw';path.write_bytes(b'A')
    frozen=hashlib.sha256(b'A').hexdigest()
    original=Path.read_bytes
    monkeypatch.setattr(Path,'read_bytes',lambda p:b'B' if p==path else original(p))
    with pytest.raises(ValueError,match='consumed|SHA256'):
        api().read_bytes_snapshot(path,expected_sha256=frozen)


def test_candidate_json_not_read(tmp_path):
    p=tmp_path/'cache_candidate';p.mkdir();f=p/'data.json';f.write_text('{}')
    with pytest.raises(ValueError):api().read_json_snapshot(f)


def test_nonfinite_json_rejected(tmp_path):
    p=tmp_path/'bad.json';p.write_text('{"x":NaN}')
    with pytest.raises(ValueError):api().read_json_snapshot(p)


def test_freeze_inputs_binds_source_and_all_consumed_assets(tmp_path):
    import hashlib
    case,assets=tmp_path/'case',tmp_path/'assets';case.mkdir();assets.mkdir()
    records={};links={}
    mask=assets/'thorax_masks/heart.nii.gz';mask.parent.mkdir();mask.write_bytes(b'mask')
    records['thorax_masks/heart.nii.gz']={'read_status':'read','sha256':hashlib.sha256(b'mask').hexdigest()}
    for rid in ('series_01','series_02'):
        dcm=case/(rid+'.dcm');dcm.write_bytes(b'fixture dcm')
        links[rid]={'status':'verified','source_relative_path':dcm.name,'actual_sha256':hashlib.sha256(dcm.read_bytes()).hexdigest()}
        series=assets/'all_series'/rid;series.mkdir(parents=True)
        for name in ('series_manifest.json','cine_u8.raw',rid+'_calibration.json'):
            p=series/name;p.write_bytes(b'fixture')
            records[p.relative_to(assets).as_posix()]={'read_status':'read','sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
    audit=tmp_path/'preflight.json';audit.write_text(json.dumps({'case_id':'case','assets':records,'source_links':links,'gates':{'absolute_geometry_accepted':False}}))
    result=api().freeze_inputs(case,assets,audit)
    assert len(result['sha256'])==10
    assert result['gates']['absolute_geometry_accepted'] is False
    (assets/'all_series/series_01/cine_u8.raw').write_bytes(b'changed')
    with pytest.raises(ValueError,match='SHA256'):api().freeze_inputs(case,assets,audit)
