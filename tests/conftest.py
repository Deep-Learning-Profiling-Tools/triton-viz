import os

import pytest

# tritonbench_kernels/ contains standalone scripts run via triton-sanitizer,
# not test modules. Exclude them from pytest collection.
collect_ignore_glob = [
    os.path.join(os.path.dirname(__file__), "end_to_end", "tritonbench_kernels", "*"),
]


@pytest.fixture(scope="session", params=["cpu"])
def device(request):
    return request.param
