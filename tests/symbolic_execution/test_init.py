from triton_viz.core.client import Client
from triton_viz.clients import Sanitizer
from triton_viz.clients.sanitizer.sanitizer import (
    SanitizerBruteForce,
    NullSanitizer,
    SanitizerSymbolicExecution,
)
from triton_viz import config as cfg


def test_brute_force():
    cfg.sanitizer_backend = "brute_force"
    s1 = Sanitizer(abort_on_error=True)
    assert isinstance(s1, SanitizerBruteForce) and s1.abort_on_error is True


def test_null_sanitizer():
    cfg.sanitizer_backend = "off"
    s2 = Sanitizer(abort_on_error=True)
    assert isinstance(s2, NullSanitizer)


def test_symbolic_execution():
    cfg.sanitizer_backend = "symexec"
    s3 = Sanitizer(abort_on_error=True)
    assert isinstance(s3, SanitizerSymbolicExecution) and s3.abort_on_error is True


def test_default_sanitizer():
    s = Sanitizer()
    assert isinstance(s, Client)
