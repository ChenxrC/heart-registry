"""Explicit input authority and content snapshots; no recursive data discovery."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

from ctdr_pipeline.motion_validation.input_policy import require_eligible_input,is_linked_path

OUTPUT_ROOT=Path(os.environ.get(
    'HEART_REGISTRY_OUTPUT_ROOT',
    str(Path(__file__).resolve().parents[2]/'artifacts/heart_mesh_dynamic'),
)).expanduser().resolve()


def sha256(path: Path) -> str:
    require_eligible_input(path)
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):digest.update(block)
    return digest.hexdigest()


def read_bytes_snapshot(path: Path,*,expected_sha256: str|None=None) -> tuple[bytes,str]:
    require_eligible_input(path)
    content=Path(path).read_bytes()
    require_eligible_input(path)
    digest=hashlib.sha256(content).hexdigest()
    if expected_sha256 is not None and digest!=expected_sha256:
        raise ValueError(f'Consumed input SHA256 differs from frozen bytes: {path}')
    return content,digest


def read_json_snapshot(path: Path,*,expected_sha256: str|None=None) -> tuple[dict,str]:
    content,digest=read_bytes_snapshot(path,expected_sha256=expected_sha256)
    data=json.loads(content.decode('utf-8'))
    if not isinstance(data,dict):raise ValueError('JSON object required')
    json.dumps(data,allow_nan=False)
    return data,digest


def assert_unchanged(snapshot: dict[Path,str]) -> None:
    if any(sha256(p)!=value for p,value in snapshot.items()):raise ValueError('Inputs changed during operation')


def validate_output(inputs: list[Path], output_dir: Path) -> Path:
    output=Path(output_dir)
    if os.path.lexists(output):raise FileExistsError('Output must not exist')
    require_eligible_input(output)
    for p in (*reversed(output.absolute().parents),output.absolute()):
        if is_linked_path(p):raise ValueError('Output traverses a link/reparse point')
    output=output.resolve()
    if output.parent!=OUTPUT_ROOT.resolve():
        raise ValueError(f'Output must be a new immediate child of {OUTPUT_ROOT}')
    for original in inputs:
        require_eligible_input(original)
        source=Path(original).resolve()
        if output.is_relative_to(source) or source.is_relative_to(output):raise ValueError('Output overlaps input')
    return output


def reserve_output(inputs: list[Path], output_dir: Path) -> Path:
    output=validate_output(inputs,output_dir)
    output.mkdir(parents=True,exist_ok=False)
    return output


def freeze_inputs(case_dir: Path, assets_root: Path, audit_path: Path) -> dict:
    for p in (case_dir,assets_root,audit_path):require_eligible_input(p)
    case_dir,assets_root,audit_path=Path(case_dir).resolve(),Path(assets_root).resolve(),Path(audit_path).resolve()
    audit,digest=read_json_snapshot(audit_path)
    if audit.get('case_id')!=case_dir.name:raise ValueError('Audit/case identity mismatch')
    snapshots={audit_path:digest};runs=[]
    names=['thorax_masks/heart.nii.gz']
    for run_id in ('series_01','series_02'):
        link=audit.get('source_links',{}).get(run_id,{})
        if link.get('status')!='verified':raise ValueError('Source DICOM identity is not verified')
        rel=link.get('source_relative_path')
        if not isinstance(rel,str) or Path(rel).is_absolute():raise ValueError('Invalid DICOM relative path')
        source=case_dir/rel
        require_eligible_input(source)
        if not source.resolve().is_relative_to(case_dir):raise ValueError('DICOM escapes case root')
        actual=sha256(source)
        if actual!=link.get('actual_sha256'):raise ValueError('Source DICOM SHA256 differs from audit')
        snapshots[source]=actual
        names.extend(f'all_series/{run_id}/{name}' for name in ('series_manifest.json','cine_u8.raw',f'{run_id}_calibration.json'))
        runs.append({'run_id':run_id,'source_link':link})
    for relative in names:
        path=assets_root/relative
        actual=sha256(path);entry=audit.get('assets',{}).get(relative,{})
        if entry.get('read_status')!='read' or actual!=entry.get('sha256'):raise ValueError(f'Asset SHA256 mismatch: {relative}')
        snapshots[path]=actual
    assert_unchanged(snapshots)
    return {'schema_version':1,'case_id':case_dir.name,'case_dir':str(case_dir),'assets_root':str(assets_root),
        'audit_path':str(audit_path),'runs':runs,'sha256':{str(p):h for p,h in snapshots.items()},
        'gates':audit.get('gates',{}),'clinical_validation':False,'patient_true_3d_tre_mm':None}


def write_json(path: Path, value: dict) -> None:
    Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
