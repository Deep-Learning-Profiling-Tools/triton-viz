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


class TestRebasePointersSingleStorage:
    def test_basic(self):
        t = torch.arange(8, dtype=torch.float32)
        m = _make_materializer(t)
        base = t.untyped_storage().data_ptr()
        # Pointers to elements 0, 2, 4
        ptrs = np.array([base, base + 8, base + 16], dtype=np.int64)
        result = m.rebase_pointers(ptrs)
        # On CPU, cpu() is a no-op so offset is 0 — addresses should be unchanged
        np.testing.assert_array_equal(result, ptrs)


class TestRebasePointersMixedStorage:
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


class TestRebasePointersMaskedGarbage:
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


class TestRebasePointersEmpty:
    def test_zero_element(self):
        m = TensorMaterializer()
        ptrs = np.array([], dtype=np.int64)
        result = m.rebase_pointers(ptrs)
        assert result.size == 0


class TestRebasePointersAllMasked:
    def test_all_masked_garbage(self):
        m = TensorMaterializer()
        garbage = 0xDEADBEEF
        ptrs = np.array([garbage, garbage, garbage], dtype=np.int64)
        mask = np.array([False, False, False])
        result = m.rebase_pointers(ptrs, mask=mask)
        np.testing.assert_array_equal(result, ptrs)


class TestRebasePointersSingleElementMaskedOut:
    def test_single_masked_out(self):
        m = TensorMaterializer()
        garbage = 0xDEADBEEF
        ptrs = np.array([garbage], dtype=np.int64)
        mask = np.array([False])
        result = m.rebase_pointers(ptrs, mask=mask)
        assert result[0] == garbage
