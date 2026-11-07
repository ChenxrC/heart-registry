"""Read-only patient CT-heart projection audit, never an anatomical fit."""
from __future__ import annotations

from pathlib import Path
import io
import numpy as np
from PIL import Image,ImageDraw
from scipy import ndimage

from .inputs import freeze_inputs,read_json_snapshot,read_bytes_snapshot,assert_unchanged,reserve_output,validate_output,write_json
from .mesh import load_reference_mesh,MeshPreparationError
from .geometry import load_camera,audit_projection
from .reference_raster import render_hard


def run_projection_audit(case_dir: Path,assets_root: Path,audit_path: Path,output_dir: Path) -> dict:
    # Validate output without creating it until input reads and rendering finish.
    from ctdr_pipeline.motion_validation.runner import validate_output_paths
    case_dir,assets_root,output_dir=validate_output_paths(case_dir,assets_root,output_dir)
    validate_output([case_dir,assets_root,Path(audit_path).parent],output_dir)
    audit_path=Path(audit_path)
    for protected in (audit_path,audit_path.parent):
        if output_dir.is_relative_to(protected.resolve()) or protected.resolve().is_relative_to(output_dir):
            raise ValueError('Output overlaps frozen audit input')
    manifest=freeze_inputs(case_dir,assets_root,audit_path)
    snapshots={Path(p):h for p,h in manifest['sha256'].items()}
    mask_path=assets_root/'thorax_masks/heart.nii.gz'
    reason=None
    try:mesh=load_reference_mesh(mask_path,expected_sha256=snapshots[mask_path])
    except MeshPreparationError as error:
        mesh=error.diagnostic_mesh;reason=str(error)
    frames={'series_01':[0,5,10,15,20,25,30],'series_02':[0,5,10,16,21,26,31]}
    reports=[];images={};cameras={}
    for rid,indices in frames.items():
        series=assets_root/'all_series'/rid
        metadata_path=series/'series_manifest.json'
        metadata,_=read_json_snapshot(metadata_path,expected_sha256=snapshots[metadata_path])
        shape=metadata.get('shape_frame_row_column')
        if not isinstance(shape,list) or len(shape)!=3 or any(type(v) is not int or v<=0 for v in shape):raise ValueError('Invalid cine shape')
        cine_path=series/'cine_u8.raw'
        content,_=read_bytes_snapshot(cine_path,expected_sha256=snapshots[cine_path])
        raw=np.frombuffer(content,dtype=np.uint8)
        if raw.size!=int(np.prod(shape)):raise ValueError('Cine byte count mismatch')
        cine=raw.reshape(shape)
        camera_path=series/f'{rid}_calibration.json'
        camera=load_camera(camera_path,image_hw=tuple(shape[1:]),expected_sha256=snapshots[camera_path])
        report=audit_projection(mesh,camera,cine,indices);report['run_id']=rid
        reports.append(report);cameras[rid]=camera.to_dict()
        mask=render_hard(mesh.vertices_lps_mm,mesh.faces,camera,image_hw=camera.image_hw)
        edge=mask & ~ndimage.binary_erosion(mask,border_value=1)
        edge[[0,-1],:]=False;edge[:,[0,-1]]=False
        sheet=Image.new('RGB',(1024,568),'#102030')
        for j,index in enumerate(indices):
            rgb=np.repeat(cine[index,:,:,None],3,axis=2);rgb[edge]=[255,210,40]
            tile=Image.fromarray(rgb).resize((256,256),Image.Resampling.LANCZOS)
            x,y=(j%4)*256,(j//4)*284
            sheet.paste(tile,(x,y+28));ImageDraw.Draw(sheet).text((x+4,y+5),f'{rid} frame {index} | CT ONLY',fill='white')
        buffer=io.BytesIO();sheet.save(buffer,format='PNG');images[rid]=buffer.getvalue()
    assert_unchanged(snapshots)
    output=reserve_output([case_dir,assets_root,audit_path.parent],output_dir)
    (output/'projection_audit').mkdir()
    for rid,content in images.items():(output/'projection_audit'/f'{rid}.png').write_bytes(content)
    for rid,camera in cameras.items():write_json(output/f'camera_{rid}.json',camera)
    np.savez_compressed(output/'mesh_reference.npz',vertices_lps_mm=mesh.vertices_lps_mm,faces=mesh.faces,
        source_sha256=np.asarray(mesh.source_sha256),optimization_ready=np.asarray(mesh.report.get('optimization_ready',False)))
    write_json(output/'input_manifest.json',manifest);write_json(output/'mesh_report.json',mesh.report)
    report={'status':'static_projection_audit_only','research_only':True,'real_patient_mesh_fit_executed':False,
        'clinical_validation':False,'patient_true_3d_tre_mm':None,'absolute_geometry_accepted':False,
        'optimization_mesh_ready':mesh.report.get('optimization_ready',False),'mesh_rejection_reason':reason,
        'series':reports,'source_integrity_unchanged':True}
    write_json(output/'projection_audit.json',report)
    (output/'summary.md').write_text('# CT 心脏静态投影审计\n\n黄色轮廓是 CT 模型投影，不是 XA 分割结果。未执行患者运动拟合。\n\n'
        +f"优化网格通过：{report['optimization_mesh_ready']}；标定接受：false；真实三维 TRE：null。\n\n"
        +(f'简化失败，保留完整注册表面仅供静态诊断：{reason}\n' if reason else '')
        +'\n两段序列独立显示，不代表相位同步。未测量 XA 心缘误差。\n',encoding='utf-8')
    return report
