from triton_viz.transformers.nki_extract_slice import transform_code


def test_transform_code_rewrites_load():
    source_code = """
import numpy as np
sbuf_value = nl.load(x[2:4, None, ..., nl.arange(128)[None, :]], mask=mask)
"""
    expected_transformed_code = """\
import numpy as np
sbuf_value = nl.masked_load(x, (slice(2, 4, None), None, ..., nl.arange(128)[None, :]), mask=mask)\
"""
    assert transform_code(source_code) == expected_transformed_code


def test_transform_code_rewrites_store():
    source_code = """
import numpy as np
nl.store(x[2:4, None, ..., nl.arange(128)[None, :]], sbuf_value, mask=mask)
"""
    expected_transformed_code = """\
import numpy as np
nl.masked_store(x, (slice(2, 4, None), None, ..., nl.arange(128)[None, :]), sbuf_value, mask=mask)\
"""
    assert transform_code(source_code) == expected_transformed_code


def test_transform_code_rewrites_load_transpose2d():
    source_code = """
import numpy as np
sbuf_value_t = nl.load_transpose2d(x[2:4, None, ..., nl.arange(128)[None, :]], mask=mask)
"""
    expected_transformed_code = """\
import numpy as np
sbuf_value_t = nl.load_transpose2d(x, (slice(2, 4, None), None, ..., nl.arange(128)[None, :]), mask=mask)\
"""
    assert transform_code(source_code) == expected_transformed_code
