from __future__ import annotations

from pathlib import Path

import pytest

TESTS = Path(__file__).resolve().parent

# ─────────── IR mode on a Triton outside the tested window (D29) ───────────
#
# IR mode (tilelens.ir, Sanitizer(compile=True), the host compile) runs only on
# the Triton releases in tilelens.core.config.TESTED_TRITON_VERSIONS unless
# TILELENS_IR_ALLOW_UNTESTED_TRITON=1 says otherwise (D10b), and so do its
# tests: on any other release every test marked IR_MODE is skipped, with a
# reason naming the installed Triton and the window. Running them with the
# override is how a release joins the window. The modules below are marked
# here; a test module elsewhere opts in with ``pytestmark = pytest.mark.ir_mode``.
# The tests that the gate itself refuses correctly live in
# tests/unit/test_ir_version_gate.py, which is not marked and runs on every
# release; the reader conformance suite (tests/conformance/) is not marked
# either: it runs everywhere, as a non-strict xfail outside the window.
IR_MODE = "ir_mode"
# Paths relative to tests/: an entry ending in "/" covers every module below
# that directory, one ending in "*" every module whose path it prefixes.
IR_MODE_MODULES: tuple[str, ...] = (
    "unit/ir/",
    "unit/sanitizer_compiled/",
    "unit/test_ir_lifecycle.py",
    "end_to_end/test_ir_*",
    "end_to_end/test_compiled_sanitizer.py",
    "end_to_end/test_host_compile.py",
)


def pytest_addoption(parser):
    group = parser.getgroup("tilelens")
    group.addoption(
        "--triton-kernels-device",
        choices=("auto", "cuda", "cpu"),
        default="auto",
        help=(
            "Device mode for tests/end_to_end/test_triton_kernels.py. "
            "'cpu' runs CPU fake-tensor checks for CI without CUDA."
        ),
    )


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        f"{IR_MODE}: a test of IR mode, skipped on a Triton release outside "
        "tilelens.core.config.TESTED_TRITON_VERSIONS unless "
        "TILELENS_IR_ALLOW_UNTESTED_TRITON=1 (D29)",
    )


def is_ir_mode_module(path: Path) -> bool:
    """Whether the test module at ``path`` is one of IR_MODE_MODULES."""
    try:
        relative = Path(path).resolve().relative_to(TESTS).as_posix()
    except ValueError:  # not under tests/
        return False
    for entry in IR_MODE_MODULES:
        if entry.endswith(("/", "*")):
            if relative.startswith(entry.rstrip("*")):
                return True
        elif relative == entry:
            return True
    return False


def ir_mode_skip_reason() -> str | None:
    """Why the IR-mode tests skip on the installed Triton; None when they
    run (a release in the window, or the override set)."""
    from tilelens.core.config import TESTED_TRITON_VERSIONS, untested_triton_version

    version = untested_triton_version()
    if version is None:
        return None
    window = ", ".join(f"{release}.x" for release in TESTED_TRITON_VERSIONS)
    return (
        f"IR mode is not tested on the installed Triton {version}: the tested "
        "window (tilelens.core.config.TESTED_TRITON_VERSIONS) is Triton "
        f"{window}; set TILELENS_IR_ALLOW_UNTESTED_TRITON=1 to run the IR-mode "
        "tests anyway (D29)"
    )


def pytest_collection_modifyitems(config, items):
    marked = []
    for item in items:
        if is_ir_mode_module(item.path):
            item.add_marker(IR_MODE)
        if item.get_closest_marker(IR_MODE) is not None:
            marked.append(item)
    reason = ir_mode_skip_reason() if marked else None
    if reason is not None:
        # First of the item's skipif marks, so its reason is the one given
        # even where another (e.g. "needs a CUDA GPU") holds as well.
        skip = pytest.mark.skipif(True, reason=reason)
        for item in marked:
            item.add_marker(skip, append=False)


@pytest.fixture
def unreachable_driver(monkeypatch):
    """``unreachable_driver(message)`` makes Triton's active driver
    unreachable, as on a machine without a GPU: any question to it raises
    ``AssertionError(message)``. One call reaches the real driver: unloading
    a module an earlier test loaded on a real GPU, which Triton 3.8's
    CompiledKernel.__del__ does through the driver whenever that kernel is
    collected (e.g. when tilelens.clear() drops the launch holding it)."""
    from triton.runtime.driver import driver

    owner = type(driver)
    real = owner.__dict__["active"]

    def refuse(message: str) -> None:
        class Utils:
            def unload_module(self, module):
                return real.__get__(driver, owner).utils.unload_module(module)

            def __getattr__(self, name):
                raise AssertionError(message)

        class Active:
            utils = Utils()

            def __getattr__(self, name):
                raise AssertionError(message)

        stand_in = Active()
        monkeypatch.setattr(owner, "active", property(lambda self: stand_in))

    return refuse


@pytest.fixture(scope="session", params=["cpu"])
def device(request):
    return request.param
