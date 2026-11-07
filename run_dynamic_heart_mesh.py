"""Independent CT-heart mesh experiments. Real XA motion fitting is not enabled."""
import argparse
import json
from pathlib import Path


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='command',required=True)
    env=sub.add_parser('env');env.add_argument('--device',default='cpu');env.add_argument('--output-dir',type=Path)
    audit=sub.add_parser('audit')
    for option in ('case-dir','assets-root','audit-path','output-dir'):audit.add_argument('--'+option,type=Path,required=True)
    synthetic=sub.add_parser('synthetic')
    synthetic.add_argument('--reference-mesh',type=Path);synthetic.add_argument('--output-dir',type=Path,required=True)
    synthetic.add_argument('--device',default='cpu');synthetic.add_argument('--seed',type=int,default=20261017)
    synthetic.add_argument('--steps',type=int,default=80);synthetic.add_argument('--resolution',type=int,default=128)
    synthetic.add_argument('--smoke',action='store_true')
    static=sub.add_parser('static-preflight')
    for option in ('case-dir','assets-root','audit-path','ct-volume','output-dir'):
        static.add_argument('--'+option,type=Path,required=True)
    static.add_argument('--device',default='cpu');static.add_argument('--steps',type=int,default=30)
    static.add_argument('--resolution',type=int,default=64);static.add_argument('--max-faces',type=int,default=20000)
    args=parser.parse_args()
    if getattr(args,'output_dir',None):
        from ctdr_pipeline.heart_mesh_motion.inputs import validate_output
        validate_output([],args.output_dir)
    if args.command=='static-preflight':
        from ctdr_pipeline.heart_mesh_motion.static_preflight import run_static_preflight
        result=run_static_preflight(args.case_dir,args.assets_root,args.audit_path,args.ct_volume,args.output_dir,
            device=args.device,steps=args.steps,resolution=args.resolution,max_faces=args.max_faces)
        print(json.dumps(result,ensure_ascii=False,allow_nan=False,indent=2))
        return 0
    if args.command=='env':
        from ctdr_pipeline.heart_mesh_motion.runtime import probe_runtime
        report=probe_runtime(args.device)
        if args.output_dir:
            from ctdr_pipeline.heart_mesh_motion.inputs import reserve_output,write_json
            target=reserve_output([],args.output_dir);write_json(target/'environment.json',report)
        print(json.dumps(report,ensure_ascii=False,allow_nan=False,indent=2))
        return 0 if report['status']=='available' else 2
    if args.command=='audit':
        from ctdr_pipeline.heart_mesh_motion.audit import run_projection_audit
        report=run_projection_audit(args.case_dir,args.assets_root,args.audit_path,args.output_dir)
        print(json.dumps(report,ensure_ascii=False,allow_nan=False,indent=2))
        return 0
    if args.command=='synthetic':
        import os
        import io
        import hashlib
        import numpy as np
        from ctdr_pipeline.heart_mesh_motion.contracts import MeshData,FitConfig
        from ctdr_pipeline.heart_mesh_motion.inputs import reserve_output,write_json,assert_unchanged
        from ctdr_pipeline.motion_validation.input_policy import require_eligible_input
        from ctdr_pipeline.heart_mesh_motion.runtime import probe_runtime
        if os.path.lexists(args.output_dir):raise FileExistsError('Output must not exist')
        reference=None;snapshots={}
        if args.reference_mesh:
            require_eligible_input(args.reference_mesh)
            content=args.reference_mesh.read_bytes();snapshots[args.reference_mesh]=hashlib.sha256(content).hexdigest()
            with np.load(io.BytesIO(content),allow_pickle=False) as data:
                reference=MeshData(data['vertices_lps_mm'],data['faces'],str(data['source_sha256'].item()),
                    {'optimization_ready':bool(data['optimization_ready'].item())})
        runtime=probe_runtime(args.device)
        if runtime['status']!='available':
            target=reserve_output([p.parent for p in snapshots],args.output_dir)
            write_json(target/'benchmark_status.json',{'status':'backend_unavailable','runtime':runtime,
                'real_patient_mesh_fit_executed':False,'clinical_validation':False,'patient_true_3d_tre_mm':None})
            print('Differentiable backend unavailable; no synthetic fit performed.');return 2
        from ctdr_pipeline.heart_mesh_motion.synthetic_mesh import run_mesh_benchmark
        config=FitConfig(seed=args.seed,resolutions=(args.resolution,),optimizer_budget=args.steps)
        result=run_mesh_benchmark(config,reference,device=args.device,smoke=args.smoke)
        assert_unchanged(snapshots)
        target=reserve_output([p.parent for p in snapshots],args.output_dir)
        write_json(target/'synthetic_benchmark.json',result)
        example=result.get('example_motion')
        if example:
            from PIL import Image,ImageDraw
            frames=[]
            for truth,fitted in zip(example['target_masks'],example['fitted_masks']):
                a=np.asarray(truth,bool);b=np.asarray(fitted,bool);rgb=np.zeros((*a.shape,3),np.uint8)
                rgb[a]=[255,210,50];rgb[b]=[50,220,255];rgb[a&b]=[240,240,240]
                frame=Image.fromarray(rgb).resize((384,384));ImageDraw.Draw(frame).text((8,8),'SYNTHETIC: target yellow / fitted cyan',fill='white');frames.append(frame)
            frames[0].save(target/'synthetic_inverse_fit.gif',save_all=True,append_images=frames[1:],duration=180,loop=0)
            frames[0].save(target/'synthetic_inverse_fit.png')
        print(f'Synthetic-only benchmark written: {target}');return 0


if __name__=='__main__':raise SystemExit(main())
