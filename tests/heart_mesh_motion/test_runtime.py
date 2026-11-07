import importlib
import json


def test_runtime_probe_reports_actual_backward_not_import_only():
    result=importlib.import_module('ctdr_pipeline.heart_mesh_motion.runtime').probe_runtime('cpu')
    json.dumps(result,allow_nan=False)
    assert result['device_requested']=='cpu'
    assert result['renderer_backward_verified'] is False or result['status']=='available'
    assert result['clinical_validation'] is False
    if result['status']!='available':assert result['reason']
