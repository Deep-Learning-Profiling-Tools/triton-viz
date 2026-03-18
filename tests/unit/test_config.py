import os

import pytest

from triton_viz.core.config import Config


class TestVirtualMemoryEnv:
    def test_env_0_disables(self, monkeypatch):
        monkeypatch.setenv("SANITIZER_ENABLE_FAKE_TENSOR", "0")
        c = Config()
        assert c.virtual_memory is False

    def test_env_1_enables(self, monkeypatch):
        monkeypatch.setenv("SANITIZER_ENABLE_FAKE_TENSOR", "1")
        c = Config()
        assert c.virtual_memory is True

    def test_env_unset_defaults_true(self, monkeypatch):
        monkeypatch.delenv("SANITIZER_ENABLE_FAKE_TENSOR", raising=False)
        c = Config()
        assert c.virtual_memory is True

    def test_env_arbitrary_enables(self, monkeypatch):
        monkeypatch.setenv("SANITIZER_ENABLE_FAKE_TENSOR", "auto")
        c = Config()
        assert c.virtual_memory is True
