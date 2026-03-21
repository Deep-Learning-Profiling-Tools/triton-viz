import numpy as np
import pytest
import threading
import torch

from triton_viz.core.config import TensorMode, config
from triton_viz.core.patch import TensorMaterializer, UnmappablePointerError


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
        np.testing.assert_array_equal(result, ptrs)

    def test_row_broadcast_mask(self, monkeypatch):
        t = torch.arange(8, dtype=torch.float32)
        m = _make_materializer(t)
        base = t.untyped_storage().data_ptr()
        monkeypatch.setattr(m, "_cpu_offset", lambda b: 500)
        ptrs = np.array([[base, base + 4], [base + 8, base + 12]], dtype=np.int64)
        mask = np.array([[True, False]])
        result = m.rebase_pointers(ptrs, mask=mask)
        assert result[0, 0] == base + 500
        assert result[0, 1] == base + 4
        assert result[1, 0] == base + 8 + 500
        assert result[1, 1] == base + 12

    def test_column_broadcast_mask(self, monkeypatch):
        t = torch.arange(8, dtype=torch.float32)
        m = _make_materializer(t)
        base = t.untyped_storage().data_ptr()
        monkeypatch.setattr(m, "_cpu_offset", lambda b: 500)
        ptrs = np.array([[base, base + 4], [base + 8, base + 12]], dtype=np.int64)
        mask = np.array([[True], [False]])
        result = m.rebase_pointers(ptrs, mask=mask)
        assert result[0, 0] == base + 500
        assert result[0, 1] == base + 4 + 500
        assert result[1, 0] == base + 8
        assert result[1, 1] == base + 12


# ---- Failure-path tests: unmappable pointers ----


class RebasePointersFailurePathTest:
    def test_unmappable_ptr_force_fake_raises(self, monkeypatch):
        """In FORCE_FAKE mode, unregistered pointers raise UnmappablePointerError."""
        monkeypatch.setattr(config, "tensor_mode", TensorMode.FORCE_FAKE)
        t = torch.arange(4, dtype=torch.float32)
        m = _make_materializer(t)
        bad_addr = 0xCAFEBABE
        ptrs = np.array([bad_addr], dtype=np.int64)
        with pytest.raises(UnmappablePointerError, match="No registered storage"):
            m.rebase_pointers(ptrs)

    def test_unmappable_ptr_lazy_auto_prematerialises(self, monkeypatch):
        """LAZY_AUTO pre-materialises all storages before retrying.

        A truly foreign address still raises after the retry, but the
        pre-materialisation has populated the CPU cache.
        """
        monkeypatch.setattr(config, "tensor_mode", TensorMode.LAZY_AUTO)
        t = torch.arange(4, dtype=torch.float32)
        m = _make_materializer(t)
        bad_addr = 0xCAFEBABE
        ptrs = np.array([bad_addr], dtype=np.int64)
        with pytest.raises(UnmappablePointerError, match="No registered storage"):
            m.rebase_pointers(ptrs)
        base = t.untyped_storage().data_ptr()
        assert base in m._cpu_cache

    def test_lazy_auto_prematerialises_all_storages(self, monkeypatch):
        """Verify pre-materialisation covers every registered storage."""
        monkeypatch.setattr(config, "tensor_mode", TensorMode.LAZY_AUTO)
        t1 = torch.arange(4, dtype=torch.float32)
        t2 = torch.arange(4, dtype=torch.float32)
        m = _make_materializer(t1, t2)
        bad_addr = 0xCAFEBABE
        ptrs = np.array([bad_addr], dtype=np.int64)
        with pytest.raises(UnmappablePointerError):
            m.rebase_pointers(ptrs)
        base1 = t1.untyped_storage().data_ptr()
        base2 = t2.untyped_storage().data_ptr()
        assert base1 in m._cpu_cache
        assert base2 in m._cpu_cache


# ---- Thread-safety tests ----


class CpuCacheThreadSafetyTest:
    def test_concurrent_materialise_same_storage(self):
        """Two threads hitting _cpu_offset for the same storage must see the
        same CPU copy — i.e. the same data_ptr."""
        t = torch.arange(16, dtype=torch.float32)
        m = _make_materializer(t)
        base = t.untyped_storage().data_ptr()

        results: dict[int, int] = {}
        barrier = threading.Barrier(2)

        def worker(tid):
            barrier.wait()
            offset = m._cpu_offset(base)
            results[tid] = offset

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(2)]
        for th in threads:
            th.start()
        for th in threads:
            th.join()

        assert results[0] == results[1], (
            "Concurrent _cpu_offset calls returned different offsets — "
            "two CPU copies were created for the same GPU storage"
        )

    def test_concurrent_rebase_returns_consistent_offsets(self):
        """Multiple threads calling rebase_pointers concurrently must all
        apply the same offset."""
        t = torch.arange(8, dtype=torch.float32)
        m = _make_materializer(t)
        base = t.untyped_storage().data_ptr()
        ptrs = np.array([base, base + 4], dtype=np.int64)

        results: dict[int, np.ndarray] = {}
        barrier = threading.Barrier(4)

        def worker(tid):
            barrier.wait()
            results[tid] = m.rebase_pointers(ptrs.copy())

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(4)]
        for th in threads:
            th.start()
        for th in threads:
            th.join()

        for tid in range(1, 4):
            np.testing.assert_array_equal(
                results[0], results[tid],
                err_msg=f"Thread 0 and thread {tid} got different rebase results",
            )
