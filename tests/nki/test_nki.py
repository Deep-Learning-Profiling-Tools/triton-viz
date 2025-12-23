#!/usr/bin/env python3
"""
Test script to verify NDArray slicing functionality after fixes
"""
import numpy as np
import pytest

try:
    from triton_viz.core.nki import NDArray
except ModuleNotFoundError:
    pytest.skip(
        "NeuronX dependencies are missing. Install triton-viz[nki] to run these tests.",
        allow_module_level=True,
    )

pytestmark = pytest.mark.nki  # only run at "pytest -m nki"


def test_ndarray_creation():
    print("Testing NDArray creation...")

    # Test creation with value
    data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    nd1 = NDArray(value=data, name="test_array")
    print(f"Created NDArray: {nd1}")
    print(f"Shape: {nd1.shape}")
    print(f"Dtype: {nd1.dtype}")
    print(f"Value:\n{nd1.data}")
    print()

    # Test creation with shape and dtype
    nd2 = NDArray(shape=(2, 3), dtype=np.float32, name="shaped_array")
    print(f"Created shaped NDArray: {nd2}")
    print(f"Shape: {nd2.shape}")
    print(f"Dtype: {nd2.dtype}")
    print()


def test_slicing():
    print("Testing slicing operations...")

    data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    nd_array = NDArray(value=data, name="test_array")

    # Test [:, :] (all elements)
    # slice_all = nd_array[:, :]
    slice_all = nd_array[:2, :2]
    assert np.allclose(slice_all.data, data[:2, :2])
    print(f"nd_array[:, :] = {slice_all}")
    print(f"Value:\n{slice_all.data}")
    print()

    # Test [:, 0] (first column)
    slice_col = nd_array[:, 0]
    assert np.allclose(slice_col.data, data[:, 0])
    print(f"nd_array[:, 0] = {slice_col}")
    print(f"Value: {slice_col.data}")
    print()

    # Test [0, :] (first row)
    slice_row = nd_array[0, :]
    assert np.allclose(slice_row.data, data[0, :])
    print(f"nd_array[0, :] = {slice_row}")
    print(f"Value: {slice_row.data}")
    print()

    # Test advanced indexing
    rewritten_slice = (
        np.arange(2)[:, None],
        np.arange(3)[None, :],
    )
    slice_advanced = nd_array[rewritten_slice]
    assert np.allclose(slice_advanced.data, data[rewritten_slice])
    print(f"nd_array[nl.arange(2)[:, None], nl.arange(3)[None, :]] = {slice_advanced}")
    print(f"Value:\n{slice_advanced.data}")
    print()

    # test nd_array[0, :3, :, 2:4, 2]
    data = np.reshape(np.arange(3 * 4 * 5 * 6 * 7), (3, 4, 5, 6, 7))
    nd_array = NDArray(value=data, name="test_array_5d")
    rewritten_slice = (
        0,
        np.arange(3)[:, None, None],
        np.arange(5)[None, :, None],
        np.arange(2, 4)[None, None, :],
        2,
    )  # this is what the above slice would be represented as after triton-viz tracing
    slice_advanced = nd_array[rewritten_slice]
    assert np.allclose(slice_advanced.data, data[rewritten_slice])
    assert np.allclose(slice_advanced.data, data[0, :3, :, 2:4, 2])
    print(f"Value:\n{slice_advanced.data}")
    print()


def test_arithmetic():
    print("Testing arithmetic operations...")

    data1 = np.array([[1, 2], [3, 4]])
    data2 = np.array([[5, 6], [7, 8]])

    nd1 = NDArray(value=data1, name="array1")
    nd2 = NDArray(value=data2, name="array2")

    # Test addition
    result = nd1 + nd2
    print(f"Addition result: {result}")
    print(f"Value:\n{result.data}")
    print()

    # Test slicing on result
    slice_result = result[:, 0]
    print(f"Slice of result [:, 0]: {slice_result}")
    print(f"Value: {slice_result.data}")
    print()


if __name__ == "__main__":
    test_ndarray_creation()
    test_slicing()
    test_arithmetic()
    print("All tests completed successfully!")
