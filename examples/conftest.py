import pytest
from triton_viz import clear


@pytest.fixture(autouse=True, scope="function")
def clear_cache():
    yield
    clear()
