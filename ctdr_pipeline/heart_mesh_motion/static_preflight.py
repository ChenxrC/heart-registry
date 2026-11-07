"""New-output-only static mesh/geometry experiments; no dynamic patient fitting."""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import io
import time
import numpy as np
from PIL import Image,ImageDraw
from scipy import ndimage

from .contracts import MeshData
from .inputs import freeze_inputs,read_json_snapshot,read_bytes_snapshot,sha256,assert_unchanged,validate_output,reserve_output,write_json
from .geometry import load_camera
from .mesh import load_reference_mesh
from .optimization_mesh import prepare_optimization_mesh
from .reference_raster import render_hard
from .bone_projection import load_bone_volume,render_bone
from .static_pose import support_background,fit_static_translation
from .evaluation import make_split


def _preview(cine_frame,before_mask,after_mask,target,before_drr,after_drr,run_id):
    sheet=Image.new('RGB',(1024,824),'#112131');draw=ImageDraw.Draw(sheet)
    for column,(mask,title) in enumerate(((before_mask,'BEFORE: inherited S7 pose'),(after_mask,'AFTER: UNVALIDATED 2DOF rigid candidate'))):
        rgb=np.repeat(cine_frame[:,:,None],3,axis=2)
        edge=mask & ~ndimage.binary_erosion(mask,border_value=1)
        edge[[0,-1]]=False;edge[:,[0,-1]]=False;rgb[edge]=[255,210,40]
        sheet.paste(Image.fromarray(rgb),(column*512,30));draw.text((column*512+6,8),title,fill='white')
    for column,(array,title) in enumerate(((target,'XA support median (inverted)'),(before_drr,'CT bone proxy BEFORE'),(after_drr,'CT bone proxy AFTER'),(np.abs(after_drr-before_drr),'Change (not an error map)'))):
        values=np.asarray(array,dtype=float);low,high=np.percentile(values,[1,99])
        gray=np.clip((values-low)/max(high-low,1e-8)*255,0,255).astype(np.uint8)
        sheet.paste(Image.fromarray(gray).resize((256,256)),(column*256,568));draw.text((column*256+5,548),title,fill='white')
    buffer=io.BytesIO();sheet.save(buffer,format='PNG');return buffer.getvalue()


def run_static_preflight(case_dir,assets_root,audit_path,ct_volume_path,output_dir,*,device='cpu',steps=30,resolution=64,max_faces=20000):
    import torch
    for name,value,minimum in (('steps',steps,0),('resolution',resolution,16),('max_faces',max_faces,4)):
        if type(value) is not int or value<minimum:raise ValueError(f'Invalid {name}')
    case_dir,assets_root,audit_path,ct_volume_path=map(Path,(case_dir,assets_root,audit_path,ct_volume_path))
    from ctdr_pipeline.motion_validation.input_policy import require_eligible_input
    for p in (case_dir,assets_root,audit_path,ct_volume_path):require_eligible_input(p)
    case_dir,assets_root,audit_path,ct_volume_path=map(lambda p:p.resolve(),(case_dir,assets_root,audit_path,ct_volume_path))
    validate_output([case_dir,assets_root,audit_path.parent,ct_volume_path.parent],output_dir)
    manifest=freeze_inputs(case_dir,assets_root,audit_path)
    snapshots={Path(p):h for p,h in manifest['sha256'].items()}
    snapshots[ct_volume_path]=sha256(ct_volume_path)
    volume=load_bone_volume(ct_volume_path,expected_sha256=snapshots[ct_volume_path])
    mask=assets_root/'thorax_masks/heart.nii.gz'
    loaded=load_reference_mesh(mask,max_faces=10000000,expected_sha256=snapshots[mask])
    source=MeshData(loaded.vertices_lps_mm,loaded.faces,loaded.source_sha256,{'optimization_ready':False,'kind':'full_registration_reference_only'})
    voxel=np.c_[source.vertices_lps_mm,np.ones(len(source.vertices_lps_mm))]@np.linalg.inv(volume.voxel_to_lps).T
    inside=((voxel[:,:3]>=-.5)&(voxel[:,:3]<=np.array(volume.density_zyx.shape[::-1])-.5)).all(1)
    if inside.mean()<.99:raise ValueError('CT cache and heart mask physical bounds disagree')
    cameras=[];run_inputs=[]
    for rid in ('series_01','series_02'):
        folder=assets_root/'all_series'/rid
        metadata_path=folder/'series_manifest.json';metadata,_=read_json_snapshot(metadata_path,expected_sha256=snapshots[metadata_path])
        shape=metadata['shape_frame_row_column']
        if not isinstance(shape,list) or len(shape)!=3 or any(type(v) is not int or v<1 for v in shape):raise ValueError('Invalid cine dimensions')
        content,_=read_bytes_snapshot(folder/'cine_u8.raw',expected_sha256=snapshots[folder/'cine_u8.raw'])
        if len(content)!=int(np.prod(shape)):raise ValueError('Cine byte count mismatch')
        cine=np.frombuffer(content,dtype=np.uint8).reshape(shape)
        if metadata['source_record']['photometric_interpretation']!='MONOCHROME2':raise ValueError('Only audited MONOCHROME2 cine is supported')
        camera_path=folder/f'{rid}_calibration.json'
        camera=load_camera(camera_path,image_hw=tuple(shape[1:]),expected_sha256=snapshots[camera_path])
        split=make_split(len(cine),rid);target,weight=support_background(cine,split['support_indices'])
        cameras.append(camera);run_inputs.append((rid,cine,split,target,weight))
    print('Static preflight: independent mesh quality and native silhouette checks...',flush=True)
    reference_masks=[render_hard(source.vertices_lps_mm,source.faces,c,image_hw=c.image_hw) for c in cameras]
    candidate=prepare_optimization_mesh(source,cameras,max_faces=max_faces,device=device,reference_masks=reference_masks)
    reports=[];images={};camera_outputs={}
    for camera,(rid,cine,split,target,weight),before_mask in zip(cameras,run_inputs,reference_masks):
        print(f'Static preflight: {rid}, support-only restricted rigid search...',flush=True)
        start=time.perf_counter()
        result=fit_static_translation(volume,camera,target,weight,steps=steps,resolution=resolution,device=device)
        result.update(run_id=rid,split=split,elapsed_seconds=time.perf_counter()-start,
            target_type='temporally_weighted_XA_background_surrogate_not_bone_ground_truth',
            ct_cache_case_association='explicit_selected_cache_with_compatible_physical_mask_bounds_not_independent_voxel_validation')
        parameters=result['translation_uv_mm']
        if parameters is not None:
            offset=np.asarray(result['ct_to_xa_lps_matrix'])[:3,3]
            after_mask=render_hard(source.vertices_lps_mm+offset,source.faces,camera,image_hw=camera.image_hw)
            with torch.no_grad():
                before=render_bone(volume,camera,torch.zeros(2,device=device),image_hw=(128,128)).cpu().numpy()
                after=render_bone(volume,camera,torch.tensor(parameters,device=device),image_hw=(128,128)).cpu().numpy()
            images[rid]=_preview(cine[split['support_indices'][0]],before_mask,after_mask,target,before,after,rid)
            corrected=replace(camera,source_lps_mm=camera.source_lps_mm-offset,
                detector_center_lps_mm=camera.detector_center_lps_mm-offset,
                provenance={**camera.provenance,'static_translation_fit':result['status'],'run_id':rid},geometry_accepted=False)
            camera_outputs[rid]=corrected.to_dict()
            result['xa_to_ct_lps_matrix']=np.linalg.inv(result['ct_to_xa_lps_matrix']).tolist()
        reports.append(result)
    assert_unchanged(snapshots)
    output=reserve_output([case_dir,assets_root,audit_path.parent,ct_volume_path.parent],output_dir)
    np.savez_compressed(output/'mesh_optimization.npz',vertices_lps_mm=candidate.vertices_lps_mm,faces=candidate.faces,
        source_sha256=np.asarray(candidate.source_sha256),optimization_ready=np.asarray(candidate.report['optimization_ready']))
    write_json(output/'mesh_optimization_report.json',candidate.report)
    for rid,content in images.items():(output/f'{rid}_before_after.png').write_bytes(content)
    for rid,camera in camera_outputs.items():write_json(output/f'{rid}_camera_in_ct.json',camera)
    report={'schema_version':1,'status':'static_preflight_research_candidates_only','case_id':case_dir.name,
        'optimization_mesh_ready':candidate.report['optimization_ready'],'series':reports,'ct_bone_volume':volume.report,
        'source_sha256':{str(p):h for p,h in snapshots.items()},'source_integrity_unchanged':True,
        'static_rigid_search_attempted':True,
        'real_patient_rigid_registration_executed':any(r['ct_to_xa_lps_matrix'] is not None for r in reports),
        'real_patient_mesh_deformation_fit_executed':False,
        'real_patient_mesh_fit_executed':False,'absolute_geometry_accepted':False,
        'clinical_validation':False,'patient_true_3d_tre_mm':None,'old_artifacts_overwritten':False,
        'limitations':['Only detector-plane translation fitted; rotations/depth/intrinsics fixed.',
            'Background similarity is not independently verified bone/heart correspondence.',
            'Full heart need not lie inside the coronary field of view.']}
    write_json(output/'static_preflight.json',report)
    return report
