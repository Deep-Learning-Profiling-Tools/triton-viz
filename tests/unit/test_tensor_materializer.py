import numpy as np
import pytest
import torch

from triton_viz.core.config import TensorMode, config
from triton_viz.core.patch import TensorMaterializer


def _make_materializer(*tensors):
    """Create a TensorMaterializer and register the given tensors."""
    m = TensorMaterializer()
    for t in tensors:
        m.register(t)
    return m


# ---- Basic rebase (CPU no-op, offset=0) ----


class RebasePointersSingleStorageTest:
    def test_basic(self):
        t = torch.arange(8, dtype=torch.float32)
        m = _make_materializer(t)
        base = t.untyped_storage().data_ptr()
        ptrs = np.array([base, base + 8, base + 16], dtype=np.int64)
        result = m.rebase_pointers(ptrs)
        np.testing.assert_array_equal(result, ptrs)


class RebasePointersMixedStorageTest:
    def test_two_storages_interleaved(self):
        t1 = torch.arange(4, dtype=torch.float32)
        t2 = torch.arange(4, dtype=torch.float32)
        m = _make_materializer(t1, t2)
        base1 = t1.untyped_storage().data_ptr()
        base2 = t2.untyped_storage().data_ptr()
        ptrs = np.array([base1, base2, base1 + 4, base2 + 4], dtype=np.int64)
        result = m.rebase_pointers(ptrs)
        np.testing.assert_array_equal(result, ptrs)


class RebasePointersMaskedGarbageTest:
    def test_garbage_in_masked_lanes(self):
        t = torch.arange(4, dtype=torch.float32)
        m = _make_materializer(t)
        base = t.untyped_storage().data_ptr()
        garbage = 0xDEADBEEF
        ptrs = np.array([base, garbage, base + 4, garbage], dtype=np.int64)
        mask = np.array([True, False, True, False])
        result = m.rebase_pointers(ptrs, mask=mask)
        assert result[0] == base
        assert result[1] == garbage
        assert result[2] == base + 4
        assert result[3] == garbage


class RebasePointersEmptyTest:
    def test_zero_element(self):
        m = TensorMaterializer()
        ptrs = np.array([], dtype=np.int64)
        result = m.rebase_pointers(ptrs)
        assert result.size == 0


class RebasePointersAllMaskedTest:
    def test_all_masked_garbage(self):
        m = TensorMaterializer()
        garbage = 0xDEADBEEF
        ptrs = np.array([garbage, garbage, garbage], dtype=np.int64)
        mask = np.array([False, False, False])
        result = m.rebase_pointers(ptrs, mask=mask)
        np.testing.assert_array_equal(result, ptrs)


class RebasePointersSingleElementMaskedOutTest:
    def test_single_masked_out(self):
        m = TensorMaterializer()
        garbage = 0xDEADBEEF
        ptrs = np.array([garbage], dtype=np.int64)
        mask = np.array([False])
        result = m.rebase_pointers(ptrs, mask=mask)
        assert result[0] == garbage


# ---- Non-zero offset tests (monkeypatch _cpu_offset) ----


class RebasePointersNonZeroOffsetTest:
    def test_fast_path_applies_offset(self, monkeypatch):
        t = torch.arange(4, dtype=torch.float32)
        m = _make_materializer(t)
        base = t.untyped_storage().data_ptr()
        monkeypatch.setattr(m, "_cpu_offset", lambda b: 1000)
        ptrs = np.array([base, base + 4, base + 8, base + 12], dtype=np.int64)
        result = m.rebase_pointers(ptrs)
        np.testing.assert_array_equal(result, ptrs + 1000)

    def test_slow_path_per_storage_offset(self, monkeypatch):
        t1 = torch.arange(4, dtype=torch.float32)
        t2 = torch.arange(4, dtype=torch.float32)
        m = _make_materializer(t1, t2)
        base1 = t1.untyped_storage().data_ptr()
        base2 = t2.untyped_storage().data_ptr()
        monkeypatch.setattr(m, "_cpu_offset", lambda b: 100 if b == base1 else 200)
        ptrs = np.array([base1, base2, base1 + 4, base2 + 4], dtype=np.int64)
        result = m.rebase_pointers(ptrs)
        expected = np.array(
            [base1 + 100, base2 + 200, base1 + 4 + 100, base2 + 4 + 200],
            dtype=np.int64,
        )
        np.testing.assert_array_equal(result, expected)

    def test_masked_lanes_not_offset(self, monkeypatch):
        t = torch.arange(4, dtype=torch.float32)
        m = _make_materializer(t)
        base = t.untyped_storage().data_ptr()
        monkeypatch.setattr(m, "_cpu_offset", lambda b: 1000)
        garbage = 0xDEADBEEF
        ptrs = np.array([base, garbage, base + 4, garbage], dtype=np.int64)
        mask = np.array([True, False, True, False])
        result = m.rebase_pointers(ptrs, mask=mask)
        assert result[0] == base + 1000
        assert result[1] == garbage
        assert result[2] == base + 4 + 1000
        assert result[3] == garbage


# ---- Broadcast mask tests ----


class RebasePointersBroadcastMaskTest:
    def test_scalar_true_mask(self, monkeypatch):
        t = torch.arange(4, dtype=torch.float32)
        m = _make_materializer(t)
        base = t.untyped_storage().data_ptr()
        monkeypatch.setattr(m, "_cpu_offset", lambda b: 500)
        ptrs = np.array([base, base + 4, base + 8, base + 12], dtype=np.int64)
        result = m.rebase_pointers(ptrs, mask=np.bool_(True))
        np.testing.assert_array_equal(result, ptrs + 500)

    def test_scalar_false_mask(self, monkeypatch):
        t = torch.arange(4, dtype=torch.float32)
        m = _make_materializer(t)
        base = t.untyped_storage().data_ptr()
        monkeypatch.setattr(m, "_cpu_offset", lambda b: 500)
        ptrs = np.array([base, base + 4, base + 8, base + 12], dtype=np.int64)
        result = m.rebase_pointers(ptrs, mask=np.bool_(False))
        # All masked — no offset applied
        np.testing.assert_array_equal(result, ptrs)

    def test_row_broadcast_mask(self, monkeypatch):
        t = torch.arange(8, dtype=torch.float32)
        m = _make_materializer(t)
        base = t.untyped_storage().data_ptr()
        monkeypatch.setattr(m, "_cpu_offset", lambda b: 500)
        ptrs = np.array([[base, base + 4], [base + 8, base + 12]], dtype=np.int64)
        # mask (1,2) broadcasts to (2,2): col 0 valid, col 1 masked
        mask = np.array([[True, False]])
        result = m.rebase_pointers(ptrs, mask=mask)
        assert result[0, 0] == base + 500
        assert result[0, 1] == base + 4       # masked
        assert result[1, 0] == base + 8 + 500
        assert result[1, 1] == base + 12      # masked

    def test_column_broadcast_mask(self, monkeypatch):
        t = torch.arange(8, dtype=torch.float32)
        m = _make_materializer(t)
        base = t.untyped_storage().data_ptr()
        monkeypatch.setattr(m, "_cpu_offset", lambda b: 500)
        ptrs = np.array([[base, base + 4], [base + 8, base + 12]], dtype=np.int64)
        # mask (2,1) broadcasts to (2,2): row 0 valid, row 1 masked
        mask = np.array([[True], [False]])
        result = m.rebase_pointers(ptrs, mask=mask)
        assert result[0, 0] == base + 500
        assert result[0, 1] == base + 4 + 500
        assert result[1, 0] == base + 8       # masked
        assert result[1, 1] == base + 12      # masked


# ---- Failure-path tests: unmappable pointers ----


class RebasePointersFailurePathTest:
    def test_unmappable_ptr_force_fake_raises(self, monkeypatch):
        """In FORCE_FAKE mode, unregistered pointers must raise RuntimeError."""
        monkeypatch.setattr(config, "tensor_mode", TensorMode.FORCE_FAKE)
        t = torch.arange(4, dtype=torch.float32)
        m = _make_materializer(t)
        bad_addr = 0xCAFEBABE
        ptrs = np.array([bad_addr], dtype=np.int64)
        with pytest.raises(RuntimeError, match="No registered storage"):
            m.rebase_pointers(ptrs)

    def test_unmappable_ptr_lazy_auto_fallback(self, monkeypatch):
        """In LAZY_AUTO mode, unmappable pointers trigger eager fallback.

        After the fallback materialises all storages, the retry still cannot
        map an address that genuinely doesn't belong to any storage, so we
        expect RuntimeError even in LAZY_AUTO for truly foreign addresses.
        The fallback is useful when the *storage is registered but not yet
        materialised to CPU* — here we test that _eager_materialise_all runs.
        """
        monkeypatch.setattr(config, "tensor_mode", TensorMode.LAZY_AUTO)
        t = torch.arange(4, dtype=torch.float32)
        m = _make_materializer(t)
        bad_addr = 0xCAFEBABE
        ptrs = np.array([bad_addr], dtype=np.int64)
        # Even after fallback, a truly unmappable addr still raises
        with pytest.raises(RuntimeError, match="No registered storage"):
            m.rebase_pointers(ptrs)
        # But the fallback did materialise all storages
        base = t.untyped_storage().data_ptr()
        assert base in m._cpu_cache

    def test_lazy_auto_fallback_materialises_all(self, monkeypatch):
        """Verify eager fallback copies every registered storage to CPU."""
        monkeypatch.setattr(config, "tensor_mode", TensorMode.LAZY_AUTO)
        t1 = torch.arange(4, dtype=torch.float32)
        t2 = torch.arange(4, dtype=torch.float32)
        m = _make_materializer(t1, t2)
        bad_addr = 0xCAFEBABE
        ptrs = np.array([bad_addr], dtype=np.int64)
        with pytest.raises(RuntimeError):
            m.rebase_pointers(ptrs)
        base1 = t1.untyped_storage().data_ptr()
        base2 = t2.untyped_storage().data_ptr()
        assert base1 in m._cpu_cache
        assert base2 in m._cpu_cache
