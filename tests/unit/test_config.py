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

    def test_env_unset_is_force_real(self, monkeypatch):
        monkeypatch.delenv("SANITIZER_ENABLE_FAKE_TENSOR", raising=False)
        c = Config()
        assert c.tensor_mode is TensorMode.FORCE_REAL

    def test_env_auto_is_lazy_auto(self, monkeypatch):
        monkeypatch.setenv("SANITIZER_ENABLE_FAKE_TENSOR", "auto")
        c = Config()
        assert c.tensor_mode is TensorMode.LAZY_AUTO

    def test_env_AUTO_case_insensitive(self, monkeypatch):
        monkeypatch.setenv("SANITIZER_ENABLE_FAKE_TENSOR", "AUTO")
        c = Config()
        assert c.tensor_mode is TensorMode.LAZY_AUTO

    def test_env_unrecognised_warns_and_defaults_force_real(self, monkeypatch):
        monkeypatch.setenv("SANITIZER_ENABLE_FAKE_TENSOR", "bogus")
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            c = Config()
        assert c.tensor_mode is TensorMode.FORCE_REAL
        assert len(w) == 1
        assert "bogus" in str(w[0].message)


class VirtualMemoryDeprecationTest:
    """config.virtual_memory is a deprecated shim over tensor_mode."""

    def test_read_maps_force_real_to_false(self, monkeypatch):
        monkeypatch.setenv("SANITIZER_ENABLE_FAKE_TENSOR", "0")
        c = Config()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            assert c.virtual_memory is False
        assert any(issubclass(x.category, DeprecationWarning) for x in w)

    def test_read_maps_lazy_auto_to_true(self, monkeypatch):
        monkeypatch.setenv("SANITIZER_ENABLE_FAKE_TENSOR", "auto")
        c = Config()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            assert c.virtual_memory is True
        assert any(issubclass(x.category, DeprecationWarning) for x in w)

    def test_read_maps_force_fake_to_true(self, monkeypatch):
        monkeypatch.setenv("SANITIZER_ENABLE_FAKE_TENSOR", "1")
        c = Config()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            assert c.virtual_memory is True
        assert any(issubclass(x.category, DeprecationWarning) for x in w)

    def test_write_true_sets_lazy_auto(self, monkeypatch):
        monkeypatch.delenv("SANITIZER_ENABLE_FAKE_TENSOR", raising=False)
        c = Config()
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            c.virtual_memory = True
        assert c.tensor_mode is TensorMode.LAZY_AUTO

    def test_write_false_sets_force_real(self, monkeypatch):
        monkeypatch.setenv("SANITIZER_ENABLE_FAKE_TENSOR", "1")
        c = Config()
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            c.virtual_memory = False
        assert c.tensor_mode is TensorMode.FORCE_REAL
