"""S2 reviewed-bone 3DOF rigid diagnostic; new output only, never auto-accept."""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import base64
import io
import json
import sys
import numpy as np

from ctdr_pipeline.heart_mesh_motion.contracts import CameraData
from ctdr_pipeline.heart_mesh_motion.geometry import project_lps
from ctdr_pipeline.heart_mesh_motion.inputs import (
    read_json_snapshot, read_bytes_snapshot, validate_output, reserve_output,
    write_json, sha256, assert_unchanged,
)
from ctdr_pipeline.heart_mesh_motion.reviewed_static_pose import (
    DEFAULT_CONFIG, training_observations, fit_inplane_pose, camera_in_original_ct,
)
from .bone_pose_review import (
    verify_materials, validate_review, evaluate_review, _snapshot_image, _mask_mesh,
    check_mask_geometry, _outline_map,
)
from .beer_lambert_visualization import mesh_path_lengths, visible_boundary
from ._paths import research_roots


def load_pinned_heart_mesh(mesh_path, provenance_path, expected_mask_sha256):
    provenance,provenance_hash=read_json_snapshot(provenance_path)
    expected=provenance.get('source_sha256',{}).get(str(mesh_path))
    if not isinstance(expected,str):raise ValueError('Prior heart archive SHA256 is missing')
    content,digest=read_bytes_snapshot(mesh_path,expected_sha256=expected)
    with np.load(io.BytesIO(content),allow_pickle=False) as archive:
        if str(archive['source_sha256'].item())!=expected_mask_sha256:
            raise ValueError('Heart archive mask identity differs from frozen source')
        vertices,faces=archive['vertices_lps_mm'].copy(),archive['faces'].copy()
    return vertices,faces,digest,provenance_hash


def evaluation_scope(pack,review):
    validate_review(review,pack)
    scoped_pack=deepcopy(pack);scoped_review=deepcopy(review)
    scoped_pack['series']=[r for r in pack['series'] if r['run_id']=='series_02']
    scoped_pack['structures']=[s for s in pack['structures'] if s['id'] in ('T3','T4','T5')]
    if len(scoped_pack['series'])!=1 or len(scoped_pack['structures'])!=3:
        raise ValueError('Declared S2/T3-T5 evaluation scope is missing from review pack')
    scoped_pack['evaluation_scope']='derived S2/T3-T5 scope; original material identity retained'
    scoped_review['observations']=[o for o in review['observations'] if o['run_id']=='series_02' and o['structure_id'] in ('T3','T4','T5')]
    return scoped_pack,scoped_review


def _bone_comparison(xa, observations, edges_before, edges_after, evaluations, output):
    from PIL import Image, ImageDraw, ImageFont
    font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',16)
    small=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',14)
    canvas=Image.new('RGB',(1536,672),'#12212d');draw=ImageDraw.Draw(canvas)
    draw.text((12,8),'S2 / frame21 · 3DOF三维刚性修正：T3/T5拟合，T4留出；不改变尺度/深度/心脏形状',font=font,fill='white')
    draw.text((12,37),'金色=用户拟合骨缘；粉色=用户留出骨缘；青色=CT整椎骨模型投影。几何未验收，不是临床精度。',font=small,fill='#ffcf7a')
    for col,(name,edges) in enumerate((('原始XA + 用户骨缘',{}),('修正前：此前平移候选',edges_before),('修正后：新3DOF研究候选',edges_after))):
        array=np.repeat(xa[...,None],3,axis=-1).copy()
        for edge in edges.values():array[edge]=[40,235,225]
        image=Image.fromarray(array);painter=ImageDraw.Draw(image)
        for item in observations:
            if item['review_state']!='confirmed_visible_edge':continue
            points=[tuple(p) for p in item['xy_px']]
            color='#ffd264' if item['split']=='fit' else '#ff82cf'
            painter.line(points,fill=color,width=2)
            painter.text(points[0],item['structure_id'],font=font,fill=color,stroke_width=1,stroke_fill='#15202a')
        draw.text((col*512+10,75),name,font=font,fill='white');canvas.paste(image,(col*512,104))
        if col:
            result=evaluations['before' if col==1 else 'after']
            for row,split in enumerate(('fit','holdout')):
                value=result['split_summary'][split]['mean_per_structure_px']
                text=f'{split}: {value:.2f}px' if value is not None else f'{split}: unavailable'
                draw.text((col*512+10,625+row*21),text,font=small,fill='white')
    canvas.save(output/'bone_before_after.png')


def _heart_comparison(vertices,faces,cameras,xa,output):
    from PIL import Image, ImageDraw, ImageFont
    font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',15)
    sheet=Image.new('RGB',(1024,1172),'#12212d');draw=ImageDraw.Draw(sheet)
    draw.text((12,7),'静态CT心脏：相同刚性修正的条件性投影；不是已重建的心跳，不是心缘误差图',font=font,fill='#ffcf7a')
    records=[]
    for row,(name,camera) in enumerate(cameras):
        length=mesh_path_lengths(vertices,faces,camera)['length_mm']
        transmission=np.rint(np.exp(-.010*length)*255).astype('uint8')
        overlay=np.repeat(xa[...,None],3,axis=-1).copy();overlay[visible_boundary(length)]=[45,235,225]
        top=64+row*552
        draw.text((12,top-23),name+' · 左：均匀包络透射率 μ=0.010/mm（未标定） · 右：模型边缘叠加',font=font,fill='white')
        sheet.paste(Image.fromarray(transmission).convert('RGB'),(0,top));sheet.paste(Image.fromarray(overlay),(512,top))
        np.savez_compressed(output/f'heart_{row}_projection.npz',length_mm=length.astype('float32'),mu_per_mm=np.asarray(.010))
        records.append({'camera':name,'coverage_fraction':float((length>1e-6).mean()),
                        'maximum_path_length_mm':float(length.max()),'measured_xa_heart_edge_error_px':None})
    sheet.save(output/'heart_static_before_after.png')
    return records


def run_probe(output_dir):
    from PIL import Image
    root,case_id=research_roots()
    pack_dir=root/'artifacts/heart_mesh_dynamic'/case_id/'bone_pose_review'
    reviewed=root/'artifacts/heart_mesh_dynamic'/case_id/'bone_pose_review_reviewed'
    review_path=reviewed/'review_observations_original.json'
    depth=root/'artifacts/annotation_registration'/case_id/'totalsegmentator_depth'
    previous=root/'artifacts/heart_mesh_dynamic'/case_id/'static_preflight'
    heart_provenance=root/'artifacts/heart_mesh_dynamic'/case_id/'beer_lambert_visualization/report.json'
    inputs=[pack_dir,reviewed,depth,previous,heart_provenance]
    validate_output(inputs,output_dir)
    snapshots={}
    def read(path,expected=None):
        value,digest=read_json_snapshot(path,expected_sha256=expected);snapshots[path]=digest;return value
    pack=read(pack_dir/'pack.json');manifest=read(pack_dir/'manifest.json')
    archive_bytes,digest=read_bytes_snapshot(pack_dir/'model_edges.npz');snapshots[pack_dir/'model_edges.npz']=digest
    verify_materials(pack,manifest,archive_bytes)
    old_check=read(reviewed/'checkpoint.json')
    review=read(review_path,old_check['original_review_snapshot_sha256'])
    validate_review(review,pack)
    frozen_sources={Path(p):h for p,h in manifest['source_sha256'].items()}
    assert_unchanged(frozen_sources)
    snapshots.update(frozen_sources)
    mesh_path=previous/'mesh_optimization.npz'
    heart_path=root/'artifacts/xa_dynamic_vessel'/case_id/'thorax_masks/heart.nii.gz'
    heart_vertices,heart_faces,mesh_hash,provenance_hash=load_pinned_heart_mesh(mesh_path,heart_provenance,snapshots[heart_path])
    snapshots[mesh_path]=mesh_hash;snapshots[heart_provenance]=provenance_hash
    import importlib.metadata
    runtime_versions={'python':sys.version,**{name:importlib.metadata.version(name) for name in
                      ('numpy','scipy','Pillow','SimpleITK','scikit-image','trimesh')}}
    project_root=Path(__file__).resolve().parents[3]
    code_paths=[Path(__file__),Path(__file__).with_name('reviewed_static_pose_viewer.html'),
                project_root/'ctdr_pipeline/heart_mesh_motion/reviewed_static_pose.py',
                project_root/'experiments/heart_mesh_dynamic/probes/bone_pose_review.py',
                project_root/'experiments/heart_mesh_dynamic/probes/beer_lambert_visualization.py']
    code_paths += [project_root/'ctdr_pipeline/heart_mesh_motion'/name for name in
                   ('geometry.py','reference_raster.py','contracts.py','inputs.py','bone_projection.py')]
    code_snapshots={p:sha256(p) for p in code_paths if p.is_file()}
    lock_snapshots={project_root/'experiments/heart_mesh_dynamic'/name:sha256(project_root/'experiments/heart_mesh_dynamic'/name)
                    for name in ('pyproject.toml','uv.lock')}
    # Starting pose was frozen before these annotations; no T4-dependent choice.
    camera_index=next(i for i,p in enumerate(pack['projections']) if p['run_id']=='series_02' and p['camera_id']=='static_candidate')
    camera_dict=deepcopy(manifest['camera_parameters'][camera_index]);camera_dict['image_hw']=tuple(camera_dict['image_hw'])
    camera=CameraData(**camera_dict)
    if camera.geometry_accepted:raise ValueError('This probe requires explicitly unaccepted research geometry')
    cal_path=root/'artifacts/xa_dynamic_vessel'/case_id/'all_series/series_02/series_02_calibration.json'
    calibration=read(cal_path,snapshots[cal_path])
    sod=float(calibration['calibration']['sod_mm'])
    if not np.isfinite(sod) or sod<=0:raise ValueError('Finite positive SOD required for frozen axial pivot')
    pivot=camera.source_lps_mm+sod*camera.n_lps
    fit_records=training_observations(review['observations'],run_id='series_02')
    if [o['structure_id'] for o in fit_records]!=['T3','T5']:
        raise ValueError('S2 requires the two predeclared confirmed fit structures T3/T5')
    if any(o['frame_index']!=21 for o in fit_records):raise ValueError('Expected fixed S2 anchor frame21')
    output=reserve_output(inputs,output_dir)
    config={'scope':'S2 frame21 only','fit_structures':['T3','T5'],'holdout_structures':['T4'],
            'starting_camera':'static_candidate from review pack003', 'optimizer':deepcopy(DEFAULT_CONFIG),
            'pivot_lps_mm':pivot.tolist(),'geometry_accepted':False,
            'holdout_policy':'No T4 coordinates/mesh passed to solver; no selection after holdout evaluation',
            'label_scope_warning':'reviewed visible edges may not equal CT whole-vertebra envelope; no automatic anatomical acceptance'}
    write_json(output/'config_frozen.json',config)
    config_hash=sha256(output/'config_frozen.json')
    write_json(output/'execution_frozen.json',{'runtime_versions':runtime_versions,
               'code_sha256':{str(p):h for p,h in code_snapshots.items()},
               'lock_sha256':{str(p):h for p,h in lock_snapshots.items()},
               'source_sha256':{str(p):h for p,h in snapshots.items()}})
    execution_hash=sha256(output/'execution_frozen.json')
    ct_path=depth/'ct_series4_native_geometry.nii.gz';expected_ct=snapshots[ct_path]
    ct=_snapshot_image(ct_path,snapshots)
    if snapshots[ct_path]!=expected_ct:raise ValueError('CT changed')
    meshes={}
    def load_structure(name):
        path=depth/f'vertebrae_masks/vertebrae_{name}.nii.gz';expected=snapshots[path]
        image=_snapshot_image(path,snapshots)
        if snapshots[path]!=expected:raise ValueError('Segmentation source changed')
        check_mask_geometry(ct,image)
        return _mask_mesh(image)
    # Only training anatomy is consumed before the fit is irrevocably frozen.
    for name in ('T3','T5'):meshes[name]=load_structure(name)
    result=fit_inplane_pose(meshes,camera,fit_records,pivot)
    result.update(config_frozen_sha256=config_hash,case_id=pack['case_id'],run_id='series_02',reference_frame_index=21,
                  real_patient_rigid_fit_executed=bool(result.get('optimization_attempted')),
                  rigid_candidate_produced=result['final_parameters'] is not None,
                  real_patient_mesh_deformation_fit_executed=False)
    if result.get('config') is not None and result['config']!=config['optimizer']:
        raise ValueError('Solver configuration differs from frozen configuration')
    write_json(output/'fit_result_frozen.json',result)
    fit_hash=sha256(output/'fit_result_frozen.json')
    if result['final_parameters'] is None:
        assert_unchanged(snapshots);assert_unchanged(code_snapshots);assert_unchanged(lock_snapshots)
        write_json(output/'report.json',result)
        print(json.dumps({'output':str(output),'status':result['status']}));return result
    matrix=np.asarray(result['ct_to_base_camera_frame_lps_matrix'])
    corrected=camera_in_original_ct(camera,matrix)
    # Only now load the held-out anatomy and construct its measured distance.
    meshes['T4']=load_structure('T4')
    edges_before,_=_outline_map(meshes,camera)
    with np.load(io.BytesIO(archive_bytes),allow_pickle=False) as archive:
        for name,edge in edges_before.items():
            if not np.array_equal(edge,archive[f'series_02__static_candidate__{name}']):
                raise ValueError('Recomputed baseline differs from frozen review material')
    edges_after,projection_reports=_outline_map(meshes,corrected)
    scoped_pack,scoped_review=evaluation_scope(pack,review)
    evaluations={'before':evaluate_review(scoped_review,scoped_pack,{('series_02',n):e for n,e in edges_before.items()}),
                 'after':evaluate_review(scoped_review,scoped_pack,{('series_02',n):e for n,e in edges_after.items()})}
    for stage,evaluation in evaluations.items():
        evaluation['fit_executed']=stage=='after'
        evaluation['interpretation']='Conditional distance on 3 reviewed S2 structures; no full silhouette/TRE/acceptance'
    equivalence=[]
    for surface in meshes.values():
        if surface is None:continue
        vertices=surface[0]
        moved=np.c_[vertices,np.ones(len(vertices))]@matrix.T
        a,_=project_lps(moved[:,:3],camera);b,_=project_lps(vertices,corrected)
        equivalence.append(float(np.max(np.abs(a-b))))
    if max(equivalence)>1e-6:raise ValueError('Rigid camera/CT transform equivalence failed')
    write_json(output/'camera_before.json',camera.to_dict());write_json(output/'camera_after_unaccepted.json',corrected.to_dict())
    write_json(output/'rigid_transform_lps.json',{'convention':'column-vector LPS millimetres; relative to frozen S2 starting camera frame',
               'matrix_role':'relative correction in the same LPS basis; NOT full absolute CT-to-XA calibration or camera-local coordinates',
               'ct_to_base_camera_frame':matrix.tolist(),'inverse':np.linalg.inv(matrix).tolist(),
               'usage':'Move CT by matrix with base camera OR keep original CT with camera_after; NEVER both.',
               'absolute_geometry_accepted':False,'clinical_validation':False})
    write_json(output/'evaluation.json',evaluations)
    np.savez_compressed(output/'model_edges_after.npz',**edges_after)
    raw,_=read_bytes_snapshot(review_path,expected_sha256=snapshots[review_path]);(output/'review_original.json').write_bytes(raw)
    series=next(s for s in pack['series'] if s['run_id']=='series_02')
    xa=np.asarray(Image.open(io.BytesIO(base64.b64decode(series['xa'].split(',',1)[1]))).convert('L'))
    _bone_comparison(xa,scoped_review['observations'],edges_before,edges_after,evaluations,output)
    heart=_heart_comparison(heart_vertices,heart_faces,[('修正前',camera),('修正后（未验收）',corrected)],xa,output)
    assert_unchanged(snapshots);assert_unchanged(code_snapshots);assert_unchanged(lock_snapshots)
    if sha256(output/'fit_result_frozen.json')!=fit_hash or sha256(output/'config_frozen.json')!=config_hash:
        raise ValueError('Fit/config changed after held-out evaluation')
    if sha256(output/'execution_frozen.json')!=execution_hash:raise ValueError('Frozen code/runtime changed')
    report={'case_id':pack['case_id'],'status':result['status'],'run_id':'series_02',
            'real_patient_rigid_fit_executed':True,'real_patient_mesh_deformation_fit_executed':False,
            'fit_result_frozen_sha256':fit_hash,'config_frozen_sha256':config_hash,
            'execution_frozen_sha256':execution_hash,'runtime_versions':runtime_versions,
            'final_parameters':result['final_parameters'],'at_search_boundary':result['at_search_boundary'],
            'all_optimizer_stages_converged':result['all_optimizer_stages_converged'],
            'evaluations':evaluations,'after_projection_reports':projection_reports,
            'rigid_camera_equivalence_max_px':max(equivalence),'heart_conditional_projection':heart,
            'source_sha256':{str(p):h for p,h in snapshots.items()},'source_integrity_unchanged':True,
            'holdout_used_for_optimization_or_selection':False,'old_artifacts_overwritten':False,
            'absolute_geometry_accepted':False,'clinical_validation':False,'patient_true_3d_tre_mm':None,
            'limitations':['Only u/v translation and rotation about n, not complete 6DOF.',
                           'Whole vertebrae include posterior elements; observed boundary definitions may differ.',
                           'T4 is structure holdout, not an independent patient or blinded clinical test.',
                           'S1 is not fitted or updated. CT-heart model remains static.',
                           'Uniform heart envelope attenuation is illustrative, not calibrated radiography.']}
    write_json(output/'report.json',report)
    payload={'parameters':result['final_parameters'],'at_bound':result['at_search_boundary'],
             'converged':result['all_optimizer_stages_converged'],
             'before':evaluations['before']['per_structure'],'after':evaluations['after']['per_structure']}
    (output/'data.js').write_text('window.STATIC_POSE_RESULT='+json.dumps(payload,ensure_ascii=False,allow_nan=False)+';\n',encoding='utf-8')
    template=Path(__file__).with_name('reviewed_static_pose_viewer.html');(output/'index.html').write_bytes(template.read_bytes())
    write_json(output/'checkpoint.json',{'runtime_versions':runtime_versions,'source_sha256':report['source_sha256'],
               'code_sha256':{str(p):h for p,h in code_snapshots.items()},
               'lock_sha256':{str(p):h for p,h in lock_snapshots.items()},
               'fit_result_frozen_sha256':fit_hash,'config_frozen_sha256':config_hash,
               'execution_frozen_sha256':execution_hash,
               'git':'not a Git repository; no commit or worktree created'})
    print(json.dumps({'output':str(output),'parameters':result['final_parameters'],
                     'fit_before':evaluations['before']['split_summary']['fit'],
                     'fit_after':evaluations['after']['split_summary']['fit'],
                     'holdout_after':evaluations['after']['split_summary']['holdout'],
                     'geometry_accepted':False},ensure_ascii=False))
    return report


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,required=True)
    run_probe(parser.parse_args().output_dir)
