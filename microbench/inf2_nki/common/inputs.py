"""Host-side input generation for Inf2 NKI microbenchmarks."""

from __future__ import annotations

import numpy as np

DTYPE_TO_NUMPY = {
    "float32": np.float32,
    "float16": np.float16,
    "bfloat16": np.dtype("bfloat16"),
    "int8": np.int8,
    "int32": np.int32,
    "uint32": np.uint32,
}


def make_input(shape: tuple[int, ...], dtype_name: str, seed: int = 0) -> np.ndarray:
    """Create deterministic host input arrays with stable shapes/dtypes."""
    rng = np.random.default_rng(seed)
    np_dtype = DTYPE_TO_NUMPY.get(dtype_name)
    if np_dtype is None:
        raise ValueError(f"Unsupported dtype {dtype_name!r}")
    if np.issubdtype(np_dtype, np.integer):
        if np_dtype == np.uint32:
            return rng.integers(0, 1024, size=shape, dtype=np_dtype)
        return rng.integers(-7, 7, size=shape, dtype=np_dtype)
    return rng.normal(loc=0.0, scale=0.1, size=shape).astype(np_dtype)


def make_pointer_ring(length: int, stride: int = 1) -> np.ndarray:
    """Create a deterministic pointer-chasing ring of uint32 indices.

    The ring is returned with shape ``(1, length)`` because NKI ``nl.load``
    limits the partition (first) dimension to 128; a flat ``(length,)`` array
    would be interpreted as ``length`` partitions and rejected for length > 128.
    Keeping one partition and ``length`` elements in the free dimension lets the
    pointer-chase kernel index the free dimension with a data-dependent value.

    ``stride`` should be coprime with ``length`` for a full-length ring. The
    ring is host data so that the measured kernel depends on HBM-loaded next
    indices rather than compiler-known constants.
    """
    if length <= 0:
        raise ValueError("length must be positive")
    if stride <= 0:
        raise ValueError("stride must be positive")
    ring = np.empty((1, length), dtype=np.uint32)
    for i in range(length):
        ring[0, i] = (i + stride) % length
    return ring


def pointer_ring_walk(ring: np.ndarray, repeat: int) -> int:
    """Return the index reached after ``repeat`` dependent hops from the seed.

    This is the CPU reference used to validate the pointer-chase kernel. The
    seed load reads ``ring[0, 0]``; each subsequent hop reads ``ring[0, idx]``.
    """
    idx = int(ring[0, 0])
    for _ in range(repeat):
        idx = int(ring[0, idx])
    return idx
