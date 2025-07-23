import pytest


@pytest.fixture(scope="session", params=["cpu"])
def device(request):
    return request.param
