import json
from pathlib import Path
import subprocess
import sys
import pytest


def test_cli_unavailable_device_writes_failure_not_synthetic_success(tmp_path,monkeypatch):
    import run_dynamic_heart_mesh
    from ctdr_pipeline.heart_mesh_motion import inputs
    monkeypatch.setattr(inputs,'OUTPUT_ROOT',tmp_path/'results',raising=False)
    output=tmp_path/'results/failed'
    monkeypatch.setattr(sys,'argv',['run_dynamic_heart_mesh.py','synthetic','--smoke','--steps','1',
        '--device','cuda:999','--output-dir',str(output)])
    assert run_dynamic_heart_mesh.main()==2
    result=json.loads((output/'benchmark_status.json').read_text(encoding='utf-8'))
    assert result['status']=='backend_unavailable'
    assert result['real_patient_mesh_fit_executed'] is False
    assert not (output/'synthetic_benchmark.json').exists()


@pytest.mark.parametrize('command',['env','synthetic'])
def test_cli_rejects_protected_output_before_backend_or_write(tmp_path,monkeypatch,command):
    import run_dynamic_heart_mesh
    from ctdr_pipeline.heart_mesh_motion import inputs,runtime
    monkeypatch.setattr(inputs,'OUTPUT_ROOT',tmp_path/'results')
    monkeypatch.setattr(runtime,'probe_runtime',lambda *_:pytest.fail('Backend must not run for invalid output'))
    output=tmp_path/'1234567/forbidden'
    monkeypatch.setattr(sys,'argv',['run_dynamic_heart_mesh.py',command,'--output-dir',str(output)])
    with pytest.raises(ValueError):run_dynamic_heart_mesh.main()
    assert not output.exists()


def test_static_preflight_checks_output_boundary_before_opening_inputs(tmp_path,monkeypatch):
    import run_dynamic_heart_mesh
    from ctdr_pipeline.heart_mesh_motion import inputs
    monkeypatch.setattr(inputs,'OUTPUT_ROOT',tmp_path/'results')
    monkeypatch.setattr(sys,'argv',['run_dynamic_heart_mesh.py','static-preflight','--case-dir','missing-case',
        '--assets-root','missing-assets','--audit-path','missing-audit','--ct-volume','missing-ct',
        '--output-dir',str(tmp_path/'1234567/forbidden')])
    with pytest.raises(ValueError):run_dynamic_heart_mesh.main()
    assert not (tmp_path/'1234567/forbidden').exists()
