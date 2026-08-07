import ml_dtypes
import numpy as np

from microbench.inf2_nki.common.inputs import make_input


def test_bfloat16_input_module_import_does_not_depend_on_triton_viz_side_effects():
    value = make_input((2, 3), "bfloat16", seed=7)
    assert value.shape == (2, 3)
    assert value.dtype == np.dtype(ml_dtypes.bfloat16)
