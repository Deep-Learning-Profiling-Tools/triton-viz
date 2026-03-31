import numpy as np
import pytest

import triton_viz.visualizer.interface as viz_interface
from triton_viz.visualizer.interface import (
    _build_highlight_descriptor,
    _collect_load_store_program_subsets,
    _coords_from_offsets,
)


@pytest.fixture
def _isolate_viz_interface_state():
    """Save and restore viz_interface global state around a test."""
    saved = (
        viz_interface.global_data,
        viz_interface.raw_tensor_data,
        viz_interface.load_overall_maps,
        viz_interface.store_overall_maps,
        viz_interface.transfer_overall_maps,
    )
    yield
    (
        viz_interface.global_data,
        viz_interface.raw_tensor_data,
        viz_interface.load_overall_maps,
        viz_interface.store_overall_maps,
        viz_interface.transfer_overall_maps,
    ) = saved


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


def test_coords_from_offsets_supports_nd_shapes():
    """Converts masked byte offsets into N-d coordinates."""
    shape = (2, 3, 4, 5)
    coords = np.array(
        [
            [1, 2, 3, 4],
            [0, 1, 0, 2],
            [1, 0, 2, 1],
        ],
        dtype=np.int64,
    )
    linear = np.ravel_multi_index(coords.T, shape)
    offsets = linear * 4
    masks = np.array([True, True, True], dtype=bool)
    result = _coords_from_offsets(shape, offsets, masks, "torch.float32")
    np.testing.assert_array_equal(result, coords)


def test_coords_from_offsets_filters_masked_and_oob():
    """Drops masked-out entries and out-of-bounds accesses."""
    shape = (2, 2, 2)
    valid = np.ravel_multi_index(np.array([[1], [0], [1]]), shape)[0]
    offsets = np.array([valid * 4, 999999, 0], dtype=np.int64)
    masks = np.array([True, True, False], dtype=bool)
    result = _coords_from_offsets(shape, offsets, masks, "torch.float32")
    np.testing.assert_array_equal(result, np.array([[1, 0, 1]], dtype=np.int64))


def test_collect_load_store_program_subsets_supports_nd_coords(
    _isolate_viz_interface_state,
):
    """Aggregates per-coordinate program sets for N-d global coords."""
    viz_interface.global_data = {
        "ops": {
            "visualization_data": {
                "(0,0,0)": [
                    {
                        "type": "Load",
                        "overall_key": "LOAD:ptr",
                        "time_idx": 0,
                        "op_index": 0,
                        "uuid": "u0",
                    }
                ],
                "(1,0,0)": [
                    {
                        "type": "Load",
                        "overall_key": "LOAD:ptr",
                        "time_idx": 0,
                        "op_index": 0,
                        "uuid": "u1",
                    }
                ],
            }
        }
    }
    viz_interface.load_overall_maps = {
        "LOAD:ptr": {
            "shape": [2, 2, 2, 2, 2],
            "tiles": [
                {
                    "uuid": "u0",
                    "global_coords": [[0, 1, 0, 1, 0], [1, 1, 0, 1, 0]],
                },
                {
                    "uuid": "u1",
                    "global_coords": [[1, 1, 0, 1, 0]],
                },
            ],
        }
    }
    viz_interface.store_overall_maps = {}
    viz_interface.transfer_overall_maps = {}

    payload = _collect_load_store_program_subsets("Load", "LOAD:ptr", 0, 0)
    coord_to_count = {tuple(entry[:-1]): int(entry[-1]) for entry in payload["counts"]}
    assert coord_to_count[(0, 1, 0, 1, 0)] == 1
    assert coord_to_count[(1, 1, 0, 1, 0)] == 2
    assert payload["subset_count"] == 2


def test_collect_transfer_program_subsets(
    _isolate_viz_interface_state,
):
    """Aggregates Transfer subsets through the same endpoint path."""
    viz_interface.global_data = {
        "ops": {
            "visualization_data": {
                "(0,0,0)": [
                    {
                        "type": "Transfer",
                        "overall_key": "TRANSFER:ptr",
                        "time_idx": 0,
                        "op_index": 0,
                        "uuid": "u0",
                    }
                ],
                "(1,0,0)": [
                    {
                        "type": "Transfer",
                        "overall_key": "TRANSFER:ptr",
                        "time_idx": 0,
                        "op_index": 0,
                        "uuid": "u1",
                    }
                ],
            }
        }
    }
    viz_interface.load_overall_maps = {}
    viz_interface.store_overall_maps = {}
    viz_interface.transfer_overall_maps = {
        "TRANSFER:ptr": {
            "shape": [4, 4],
            "tiles": [
                {"uuid": "u0", "global_coords": [[0, 0], [0, 1]]},
                {"uuid": "u1", "global_coords": [[0, 1], [1, 1]]},
            ],
        }
    }

    payload = _collect_load_store_program_subsets("Transfer", "TRANSFER:ptr", 0, 0)
    coord_to_count = {tuple(entry[:-1]): int(entry[-1]) for entry in payload["counts"]}
    assert coord_to_count[(0, 0)] == 1
    assert coord_to_count[(0, 1)] == 2
    assert coord_to_count[(1, 1)] == 1


def test_get_load_store_all_programs_accepts_transfer(
    _isolate_viz_interface_state,
):
    viz_interface.global_data = {
        "ops": {
            "visualization_data": {
                "(0,0,0)": [
                    {
                        "type": "Transfer",
                        "overall_key": "TRANSFER:ptr",
                        "time_idx": 0,
                        "op_index": 0,
                        "uuid": "u0",
                    }
                ]
            }
        }
    }
    viz_interface.raw_tensor_data = {}
    viz_interface.load_overall_maps = {}
    viz_interface.store_overall_maps = {}
    viz_interface.transfer_overall_maps = {
        "TRANSFER:ptr": {
            "shape": [2, 2],
            "tiles": [{"uuid": "u0", "global_coords": [[0, 0], [1, 1]]}],
        }
    }

    with viz_interface.app.test_client() as client:
        response = client.post(
            "/api/getLoadStoreAllPrograms",
            json={"type": "Transfer", "overall_key": "TRANSFER:ptr", "time_idx": 0},
        )

    assert response.status_code == 200
    payload = response.get_json()
    coord_to_count = {tuple(entry[:-1]): int(entry[-1]) for entry in payload["counts"]}
    assert coord_to_count[(0, 0)] == 1
    assert coord_to_count[(1, 1)] == 1
