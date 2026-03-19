import warnings

import pytest

from triton_viz.core.config import Config, TensorMode


class TensorModeEnvTest:
    def test_env_0_is_force_real(self, monkeypatch):
        monkeypatch.setenv("SANITIZER_ENABLE_FAKE_TENSOR", "0")
        c = Config()
        assert c.tensor_mode is TensorMode.FORCE_REAL

    def test_env_1_is_force_fake(self, monkeypatch):
        monkeypatch.setenv("SANITIZER_ENABLE_FAKE_TENSOR", "1")
        c = Config()
        assert c.tensor_mode is TensorMode.FORCE_FAKE

    def test_env_unset_is_lazy_auto(self, monkeypatch):
        monkeypatch.delenv("SANITIZER_ENABLE_FAKE_TENSOR", raising=False)
        c = Config()
        assert c.tensor_mode is TensorMode.LAZY_AUTO

    def test_env_auto_is_lazy_auto(self, monkeypatch):
        monkeypatch.setenv("SANITIZER_ENABLE_FAKE_TENSOR", "auto")
        c = Config()
        assert c.tensor_mode is TensorMode.LAZY_AUTO

    def test_env_AUTO_case_insensitive(self, monkeypatch):
        monkeypatch.setenv("SANITIZER_ENABLE_FAKE_TENSOR", "AUTO")
        c = Config()
        assert c.tensor_mode is TensorMode.LAZY_AUTO

    def test_env_unrecognised_warns_and_defaults_lazy_auto(self, monkeypatch):
        monkeypatch.setenv("SANITIZER_ENABLE_FAKE_TENSOR", "bogus")
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            c = Config()
        assert c.tensor_mode is TensorMode.LAZY_AUTO
        assert len(w) == 1
        assert "bogus" in str(w[0].message)
