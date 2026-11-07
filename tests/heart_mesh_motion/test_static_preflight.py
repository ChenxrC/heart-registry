import hashlib
import json
from pathlib import Path
import numpy as np
import SimpleITK as sitk
import pytest


@pytest.mark.parametrize('blank_cine',[False,True])
def test_preflight_freezes_sources_and_exports_consistent_transform_and_camera(tmp_path,monkeypatch,blank_cine):
    from ctdr_pipeline.heart_mesh_motion import static_preflight,inputs
    from ctdr_pipeline.heart_mesh_motion.contracts import CameraData
    from ctdr_pipeline.heart_mesh_motion.geometry import project_lps
    case=tmp_path/'case';assets=tmp_path/'assets';case.mkdir();assets.mkdir()
    cache=tmp_path/'ct/ct.nii.gz';cache.parent.mkdir()
    hu=np.full((20,20,20),-1000,np.float32);hu[3:17,7:10,6:10]=1000;hu[5:15,12:16,12:16]=700
    image=sitk.GetImageFromArray(hu);image.SetOrigin([-10,-10,55]);sitk.WriteImage(image,str(cache))
    mask=assets/'thorax_masks/heart.nii.gz';mask.parent.mkdir()
    a=np.zeros((20,20,20),np.uint8);a[5:15,5:15,5:15]=1
    image=sitk.GetImageFromArray(a);image.SetOrigin([-10,-10,55]);sitk.WriteImage(image,str(mask))
    digest=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    records={'thorax_masks/heart.nii.gz':{'read_status':'read','sha256':digest(mask)}};links={}
    camera=dict(source_lps_mm=[0,0,0],u_lps=[1,0,0],v_lps=[0,1,0],n_lps=[0,0,1],
        detector_center_lps_mm=[0,0,100],principal_uv_px=[11.5,11.5],pixel_spacing_row_col_mm=[1,1])
    for rid,count in [('series_01',31),('series_02',32)]:
        source=case/(rid+'.dcm');source.write_bytes(b'SYNTHETIC IO FIXTURE, NOT PATIENT')
        links[rid]={'status':'verified','source_relative_path':source.name,'actual_sha256':digest(source)}
        folder=assets/'all_series'/rid;folder.mkdir(parents=True)
        yy,xx=np.mgrid[:24,:24];frame=(120+40*np.sin(xx*.5)+30*np.cos(yy*.7)).astype(np.uint8)
        if blank_cine:frame[:]=127
        (folder/'cine_u8.raw').write_bytes(np.repeat(frame[None],count,axis=0).tobytes())
        (folder/'series_manifest.json').write_text(json.dumps({'shape_frame_row_column':[count,24,24],
            'source_record':{'photometric_interpretation':'MONOCHROME2'}}))
        (folder/f'{rid}_calibration.json').write_text(json.dumps({'calibration':camera,'detector':{'rows':24,'columns':24}}))
        for name in ('cine_u8.raw','series_manifest.json',f'{rid}_calibration.json'):
            p=folder/name;records[p.relative_to(assets).as_posix()]={'read_status':'read','sha256':digest(p)}
    preflight=tmp_path/'audits/preflight.json';preflight.parent.mkdir()
    preflight.write_text(json.dumps({'case_id':'case','source_links':links,'assets':records,'gates':{'absolute_geometry_accepted':False}}))
    before={p:digest(p) for p in tmp_path.rglob('*') if p.is_file()}
    monkeypatch.setattr(inputs,'OUTPUT_ROOT',tmp_path/'results')
    output=tmp_path/'results/new'
    report=static_preflight.run_static_preflight(case,assets,preflight,cache,output,steps=0,resolution=16,max_faces=5000)
    assert report['source_integrity_unchanged'] is True and report['absolute_geometry_accepted'] is False
    assert report['real_patient_mesh_deformation_fit_executed'] is False
    assert report['real_patient_rigid_registration_executed'] is not blank_cine
    assert all(digest(p)==h for p,h in before.items())
    for row in report['series']:
        if blank_cine:
            assert row['ct_to_xa_lps_matrix'] is None
            assert not (output/f"{row['run_id']}_camera_in_ct.json").exists()
            continue
        assert row['ct_to_xa_lps_matrix'] is not None
        matrix=np.asarray(row['ct_to_xa_lps_matrix']);inverse=np.asarray(row['xa_to_ct_lps_matrix'])
        np.testing.assert_allclose(matrix@inverse,np.eye(4),atol=1e-9)
        corrected=json.loads((output/f"{row['run_id']}_camera_in_ct.json").read_text())
        second=CameraData(**{**corrected,'image_hw':tuple(corrected['image_hw'])})
        first=CameraData([0,0,0],[1,0,0],[0,1,0],[0,0,1],[0,0,100],[11.5,11.5],[1,1],(24,24),{},False)
        point=np.array([[1.,2.,65.]])
        a=project_lps(point+matrix[:3,3],first)[0];b=project_lps(point,second)[0]
        np.testing.assert_allclose(a,b,atol=1e-8)
        assert (output/f"{row['run_id']}_before_after.png").exists()
