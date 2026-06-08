import pytest


def pytest_addoption(parser):
    group = parser.getgroup("triton-viz")
    group.addoption(
        "--triton-kernels-device",
        choices=("auto", "cuda", "cpu"),
        default="auto",
        help=(
            "Device mode for tests/end_to_end/test_triton_kernels.py. "
            "'cpu' runs CPU fake-tensor checks for CI without CUDA."
        ),
    )


@pytest.fixture(scope="session", params=["cpu"])
def device(request):
    return request.param
