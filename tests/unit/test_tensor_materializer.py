import numpy as np
import pytest
import torch

from triton_viz.core.patch import TensorMaterializer


def _make_materializer(*tensors):
    """Create a TensorMaterializer and register the given tensors."""
    m = TensorMaterializer()
    for t in tensors:
        m.register(t)
    return m


class RebasePointersSingleStorageTest:
    def test_basic(self):
        t = torch.arange(8, dtype=torch.float32)
        m = _make_materializer(t)
        base = t.untyped_storage().data_ptr()
        # Pointers to elements 0, 2, 4
        ptrs = np.array([base, base + 8, base + 16], dtype=np.int64)
        result = m.rebase_pointers(ptrs)
        # On CPU, cpu() is a no-op so offset is 0 — addresses should be unchanged
        np.testing.assert_array_equal(result, ptrs)


class RebasePointersMixedStorageTest:
    def test_two_storages_interleaved(self):
        t1 = torch.arange(4, dtype=torch.float32)
        t2 = torch.arange(4, dtype=torch.float32)
        m = _make_materializer(t1, t2)
        base1 = t1.untyped_storage().data_ptr()
        base2 = t2.untyped_storage().data_ptr()
        # Interleave pointers from two storages
        ptrs = np.array([base1, base2, base1 + 4, base2 + 4], dtype=np.int64)
        result = m.rebase_pointers(ptrs)
        # CPU no-op: all addresses stay the same
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
        # Valid lanes rebased (no-op on CPU), masked lanes untouched
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


# --- Non-zero offset tests (monkeypatch _cpu_offset) ---


class RebasePointersNonZeroOffsetTest:
    def test_fast_path_applies_offset(self, monkeypatch):
        t = torch.arange(4, dtype=torch.float32)
        m = _make_materializer(t)
        base = t.untyped_storage().data_ptr()
        monkeypatch.setattr(m, "_cpu_offset", lambda b: 1000)
        ptrs = np.array([base, base + 4, base + 8, base + 12], dtype=np.int64)
        result = m.rebase_pointers(ptrs)
        expected = ptrs + 1000
        np.testing.assert_array_equal(result, expected)

    def test_slow_path_per_storage_offset(self, monkeypatch):
        t1 = torch.arange(4, dtype=torch.float32)
        t2 = torch.arange(4, dtype=torch.float32)
        m = _make_materializer(t1, t2)
        base1 = t1.untyped_storage().data_ptr()
        base2 = t2.untyped_storage().data_ptr()
        monkeypatch.setattr(m, "_cpu_offset", lambda b: 100 if b == base1 else 200)
        ptrs = np.array([base1, base2, base1 + 4, base2 + 4], dtype=np.int64)
        result = m.rebase_pointers(ptrs)
        expected = np.array([base1 + 100, base2 + 200, base1 + 4 + 100, base2 + 4 + 200], dtype=np.int64)
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


# --- Broadcast mask tests (validates broadcast_to fix) ---


class RebasePointersBroadcastMaskTest:
    def test_scalar_true_mask(self, monkeypatch):
        t = torch.arange(4, dtype=torch.float32)
        m = _make_materializer(t)
        base = t.untyped_storage().data_ptr()
        monkeypatch.setattr(m, "_cpu_offset", lambda b: 500)
        ptrs = np.array([base, base + 4, base + 8, base + 12], dtype=np.int64)
        result = m.rebase_pointers(ptrs, mask=np.bool_(True))
        expected = ptrs + 500
        np.testing.assert_array_equal(result, expected)

    def test_row_broadcast_mask(self, monkeypatch):
        t = torch.arange(8, dtype=torch.float32)
        m = _make_materializer(t)
        base = t.untyped_storage().data_ptr()
        monkeypatch.setattr(m, "_cpu_offset", lambda b: 500)
        # 2x2 pointer array
        ptrs = np.array([[base, base + 4], [base + 8, base + 12]], dtype=np.int64)
        # mask (1,2): first column True, second column False — broadcasts to (2,2)
        mask = np.array([[True, False]])
        result = m.rebase_pointers(ptrs, mask=mask)
        # Column 0 (valid): rebased; Column 1 (masked): untouched
        assert result[0, 0] == base + 500
        assert result[0, 1] == base + 4  # masked — no offset
        assert result[1, 0] == base + 8 + 500
        assert result[1, 1] == base + 12  # masked — no offset

    def test_column_broadcast_mask(self, monkeypatch):
        t = torch.arange(8, dtype=torch.float32)
        m = _make_materializer(t)
        base = t.untyped_storage().data_ptr()
        monkeypatch.setattr(m, "_cpu_offset", lambda b: 500)
        # 2x2 pointer array
        ptrs = np.array([[base, base + 4], [base + 8, base + 12]], dtype=np.int64)
        # mask (2,1): first row True, second row False — broadcasts to (2,2)
        mask = np.array([[True], [False]])
        result = m.rebase_pointers(ptrs, mask=mask)
        # Row 0 (valid): rebased; Row 1 (masked): untouched
        assert result[0, 0] == base + 500
        assert result[0, 1] == base + 4 + 500
        assert result[1, 0] == base + 8  # masked — no offset
        assert result[1, 1] == base + 12  # masked — no offset
