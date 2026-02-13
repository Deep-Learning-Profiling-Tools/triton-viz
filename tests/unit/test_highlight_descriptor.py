import numpy as np

from triton_viz.visualizer.interface import _build_highlight_descriptor


def test_build_highlight_descriptor_strided_grid():
    """Returns a descriptor when coords form a dense strided Cartesian selection."""
    coords = np.array(
        [
            [0, 0, 0],
            [2, 0, 0],
            [0, 3, 0],
            [2, 3, 0],
        ]
    )
    descriptor = _build_highlight_descriptor(coords)
    assert descriptor == {
        "start": [0, 0, 0],
        "shape": [2, 2, 1],
        "stride": [2, 3, 1],
    }


def test_build_highlight_descriptor_sparse_returns_none():
    """Returns None when coords cannot be represented by shape+stride."""
    coords = np.array(
        [
            [0, 0, 0],
            [2, 0, 0],
            [1, 3, 0],
        ]
    )
    assert _build_highlight_descriptor(coords) is None


def test_build_highlight_descriptor_4d_strided_grid():
    """Supports arbitrary rank inputs when coords form a dense strided grid."""
    a = np.array([0, 2])
    b = np.array([3])
    c = np.array([10, 20])
    d = np.array([5, 8, 11])
    coords = np.stack(np.meshgrid(a, b, c, d, indexing="ij"), axis=-1).reshape(-1, 4)
    descriptor = _build_highlight_descriptor(coords)
    assert descriptor == {
        "start": [0, 3, 10, 5],
        "shape": [2, 1, 2, 3],
        "stride": [2, 1, 10, 3],
    }


def test_build_highlight_descriptor_4d_sparse_returns_none():
    """Rejects higher-rank coords that are not a full Cartesian product."""
    coords = np.array(
        [
            [0, 0, 0, 0],
            [1, 0, 0, 0],
            [0, 1, 0, 0],
        ]
    )
    assert _build_highlight_descriptor(coords) is None
