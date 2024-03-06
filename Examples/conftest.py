import pytest
from triton_viz.interpreter import record_builder

@pytest.fixture(autouse=True, scope='function')
def clear_cache():
    yield
    record_builder.reset()
