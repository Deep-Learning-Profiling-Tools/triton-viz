import pytest, os
os.environ["TRITON_SANITIZER_BACKEND"] = "off"
import triton_viz.core.config as cfg


def test_switch_backend():
    """Switch back and forth at runtime."""
    original = cfg.sanitizer_backend

    cfg.sanitizer_backend = "symexec"
    assert cfg.sanitizer_backend == "symexec"

    cfg.sanitizer_backend = "off"
    assert cfg.sanitizer_backend == "off"

    cfg.sanitizer_backend = original


def test_invalid_backend_raises():
    """Setting an unknown backend should raise ValueError."""
    with pytest.raises(ValueError):
        cfg.sanitizer_backend = "does_not_exist"
