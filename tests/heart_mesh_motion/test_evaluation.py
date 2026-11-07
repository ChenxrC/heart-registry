import importlib
import numpy as np
import pytest


def test_splits_keep_new_anchor_outside_holdout():
    module=importlib.import_module('ctdr_pipeline.heart_mesh_motion.evaluation')
    a=module.make_split(31,'series_01');b=module.make_split(32,'series_02')
    assert a['holdout_indices']==list(range(10,20))
    assert b['holdout_indices']==list(range(10,21))
    assert 20 in a['support_indices'] and 21 in b['support_indices']


def test_interpolation_only_uses_given_support_and_actual_time():
    module=importlib.import_module('ctdr_pipeline.heart_mesh_motion.evaluation')
    result=module.interpolate_parameters([0.,2.,5.],np.array([[0.,0],[2,4],[8,10]]),[1.,3.5])
    np.testing.assert_allclose(result,[[1,2],[5,7]])
    with pytest.raises(ValueError):module.interpolate_parameters([0,0],np.zeros((2,1)),[0])
