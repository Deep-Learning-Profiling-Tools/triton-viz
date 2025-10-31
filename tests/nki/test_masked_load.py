import numpy as np
import pytest
from triton_viz.core.nki_masked_load import masked_load, masked_store


def print_op_details(
    test_name,
    operation_type,
    input_data,
    keys,
    values=None,
    mask=None,
    output=None,
    error_expected=None,
):
    """Print detailed information about a masked load/store operation."""
    print(f"{test_name}:")
    print(f"Operation type: {operation_type}")
    print("---------------------------------")
    print(f"Input:\n{input_data}")
    print("---------------------------------")
    print(f"Values:\n{values}")
    print("---------------------------------")
    print(f"Keys:\n{keys}")
    print("---------------------------------")
    print(f"Mask:\n{mask}")
    print("---------------------------------")
    print(f"Result {operation_type}:\n{output}")
    if error_expected:
        print(f"Expecting {error_expected}...")
    print()


def test_masked_load_none_case():
    """Test masked_load with mask=None (direct indexing)."""
    arr = np.array([1, 2, 3, 4, 5])
    result = masked_load(arr, (slice(1, 4),), mask=None)
    expected = arr[1:4]

    print_op_details(
        "Test 1: mask=None case", "load", arr, (slice(1, 4),), mask=None, output=result
    )

    assert np.array_equal(result, expected)


def test_masked_load_in_bounds():
    """Test masked_load with in-bounds indexing and mask."""
    arr = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    mask = np.array([[True, True, False], [True, False, True], [False, True, True]])
    result = masked_load(arr, (slice(0, 3), slice(0, 3)), mask=mask)
    result[
        [0, 1, 2], [2, 1, 0]
    ] = 0  # manually set masked out positions to 0 for testing
    expected = np.array([[1, 2, 0], [4, 0, 6], [0, 8, 9]])

    print_op_details(
        "Test 2: In-bounds indexing with mask",
        "load",
        arr,
        (slice(0, 3), slice(0, 3)),
        mask=mask,
        output=result,
    )

    assert np.array_equal(result, expected)


def test_masked_load_out_of_bounds():
    """Test masked_load with out-of-bounds indexing and mask."""
    arr = np.array([[1, 2], [3, 4]])
    mask = np.array(
        [
            [True, False, False],
            [False, True, False],
            [False, False, False],
            [False, False, False],
            [False, False, False],
        ]
    )  # note that arr.shape != mask.shape, this is intended

    # normally this arr[slice] would OOB but the mask at the OOB idxs are false so they're not loaded
    result = masked_load(arr, (slice(0, 5), slice(0, 3)), mask=mask)

    print_op_details(
        "Test 3: Out-of-bounds indexing with mask",
        "load",
        arr,
        (slice(0, 5), slice(0, 3)),
        mask=mask,
        output=result,
    )

    assert result.shape == (5, 3)


def test_masked_load_integer_oob():
    """Test masked_load with integer out-of-bounds indexing."""
    arr = np.array([10, 20, 30])
    mask = np.array([False])
    result = masked_load(arr, ([5],), mask=mask)  # idx 5 is OOB

    print_op_details(
        "Test 4: Integer indexing out of bounds",
        "load",
        arr,
        ([5],),
        mask=mask,
        output=result,
    )

    assert isinstance(result, np.ndarray)


def test_masked_load_array_indexing():
    """Test masked_load with array indexing."""
    arr = np.array([100, 200, 300, 400, 500])
    indices = np.array([0, 2, 4])
    mask = np.array([True, True, True])
    result = masked_load(arr, (indices,), mask=mask)
    expected = arr[indices]

    print_op_details(
        "Test 5: Array indexing", "load", arr, (indices,), mask=mask, output=result
    )

    assert np.array_equal(result, expected)


def test_masked_load_mixed_indexing():
    """Test masked_load with mixed indexing types."""
    arr = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
    mask = np.array([True, False, True, False])
    result = masked_load(arr, (slice(0, 4), 2), mask=mask)  # set col 2

    print_op_details(
        "Test 6: Mixed indexing types",
        "load",
        arr,
        (slice(0, 4), 2),
        mask=mask,
        output=result,
    )

    assert result.shape == (4,)


def test_masked_load_complex_indexing():
    """Test masked_load with complex multi-dimensional indexing."""
    arr = np.array(
        [
            [
                [1, 2, 3],
                [4, 5, 6],
                [7, 8, 9],
            ],
            [
                [11, 12, 13],
                [14, 15, 16],
                [17, 18, 19],
            ],
            [
                [21, 22, 23],
                [24, 25, 26],
                [27, 28, 29],
            ],
        ]
    )
    arr_slice = (
        slice(None, None, None),
        None,
        np.arange(20)[:, None],
        np.arange(3)[None, :],
        None,
        None,
    )
    mask = np.mgrid[:3, :1, :20, :3, :1, :1]
    mask = (mask[0] > 0) & (mask[2] < 3) & (mask[3] < 2)
    result = masked_load(arr, arr_slice, mask=mask)
    UD = np.iinfo(arr.dtype).max  # UD = undefined (out of bounds)
    expected = np.array(
        [
            [
                [UD, UD, UD],
                [UD, UD, UD],
                [UD, UD, UD],
                [UD, UD, UD],
                [UD, UD, UD],
                [UD, UD, UD],
                [UD, UD, UD],
                [UD, UD, UD],
                [UD, UD, UD],
                [UD, UD, UD],
                [UD, UD, UD],
                [UD, UD, UD],
                [UD, UD, UD],
                [UD, UD, UD],
                [UD, UD, UD],
                [UD, UD, UD],
                [UD, UD, UD],
                [UD, UD, UD],
                [UD, UD, UD],
                [UD, UD, UD],
            ],
            [
                [11, 12, UD],
                [14, 15, UD],
                [17, 18, UD],
                [UD, UD, UD],
                [UD, UD, UD],
                [UD, UD, UD],
                [UD, UD, UD],
                [UD, UD, UD],
                [UD, UD, UD],
                [UD, UD, UD],
                [UD, UD, UD],
                [UD, UD, UD],
                [UD, UD, UD],
                [UD, UD, UD],
                [UD, UD, UD],
                [UD, UD, UD],
                [UD, UD, UD],
                [UD, UD, UD],
                [UD, UD, UD],
                [UD, UD, UD],
            ],
            [
                [21, 22, UD],
                [24, 25, UD],
                [27, 28, UD],
                [UD, UD, UD],
                [UD, UD, UD],
                [UD, UD, UD],
                [UD, UD, UD],
                [UD, UD, UD],
                [UD, UD, UD],
                [UD, UD, UD],
                [UD, UD, UD],
                [UD, UD, UD],
                [UD, UD, UD],
                [UD, UD, UD],
                [UD, UD, UD],
                [UD, UD, UD],
                [UD, UD, UD],
                [UD, UD, UD],
                [UD, UD, UD],
                [UD, UD, UD],
            ],
        ]
    )[:, None, :, :, None, None]

    print(
        "Load Test 7: Complex multi-dimensional indexing (omitted details, check code to see inputs/outputs)"
    )

    assert np.allclose(result, expected)


def test_masked_load_incomplete_slices():
    """Test masked_load with incomplete slices."""
    arr = np.array(
        [
            [
                [1, 2, 3],
                [4, 5, 6],
                [7, 8, 9],
            ],
            [
                [11, 12, 13],
                [14, 15, 16],
                [17, 18, 19],
            ],
            [
                [21, 22, 23],
                [24, 25, 26],
                [27, 28, 29],
            ],
        ]
    )
    arr_slice = (slice(None, None, None),)
    mask = np.mgrid[:3, :3, :3]
    mask = (0 < mask[0]) & (mask[0] < 2) & (mask[1] < 2) & (mask[2] != 1)
    result = masked_load(arr, arr_slice, mask=mask)
    UD = np.iinfo(arr.dtype).max  # UD = undefined (out of bounds)
    expected = np.array(
        [
            [
                [UD, UD, UD],
                [UD, UD, UD],
                [UD, UD, UD],
            ],
            [
                [11, UD, 13],
                [14, UD, 16],
                [UD, UD, UD],
            ],
            [
                [UD, UD, UD],
                [UD, UD, UD],
                [UD, UD, UD],
            ],
        ]
    )

    print_op_details(
        "Test 8: Incomplete slices", "load", arr, arr_slice, mask=mask, output=result
    )

    assert np.allclose(result, expected)


def test_masked_load_index_error_with_true_mask():
    """Test that IndexError is raised when mask=True at out-of-bounds location."""
    arr = np.array(
        [
            [
                [1, 2, 3],
                [4, 5, 6],
                [7, 8, 9],
            ],
            [
                [11, 12, 13],
                [14, 15, 16],
                [17, 18, 19],
            ],
            [
                [21, 22, 23],
                [24, 25, 26],
                [27, 28, 29],
            ],
        ]
    )
    arr_slice = (slice(None, None, None), slice(0, 4))
    mask = np.mgrid[:3, :4, :3]
    mask = (mask[0] < 3) & (mask[1] < 3) & (mask[2] < 3)
    mask[-1, -1, -1] = True  # index error

    print_op_details(
        "Test 9: Index error should still happen when mask=True at OOB location",
        "load",
        arr,
        arr_slice,
        mask=mask,
        error_expected="IndexError",
    )

    with pytest.raises(IndexError):
        masked_load(arr, arr_slice, mask=mask)
    print("Success: Correctly raised IndexError\n")


def test_masked_load_mask_shape_mismatch():
    """Test that AssertionError is raised when mask shape doesn't match array slice."""
    arr = np.array(
        [
            [
                [1, 2, 3],
                [4, 5, 6],
                [7, 8, 9],
            ],
            [
                [11, 12, 13],
                [14, 15, 16],
                [17, 18, 19],
            ],
            [
                [21, 22, 23],
                [24, 25, 26],
                [27, 28, 29],
            ],
        ]
    )
    arr_slice = (slice(None, None, None),)
    mask = np.mgrid[:3, :4, :3]
    mask = (mask[0] < 3) & (mask[1] < 3) & (mask[2] < 3)

    print_op_details(
        "Test 10: Error if mask shape wrong",
        "load",
        arr,
        arr_slice,
        mask=mask,
        error_expected="AssertionError",
    )

    with pytest.raises(AssertionError):
        masked_load(arr, arr_slice, mask=mask)
    print("Success: Correctly raised AssertionError\n")


def test_masked_load_correct_mask_shape():
    """Test masked_load with correctly shaped mask."""
    arr = np.array(
        [
            [
                [1, 2, 3],
                [4, 5, 6],
                [7, 8, 9],
            ],
            [
                [11, 12, 13],
                [14, 15, 16],
                [17, 18, 19],
            ],
            [
                [21, 22, 23],
                [24, 25, 26],
                [27, 28, 29],
            ],
        ]
    )
    arr_slice = (slice(None, None, None),)
    mask = np.arange(27).reshape(3, 3, 3) % 2 == 0
    result = masked_load(arr, arr_slice, mask=mask)

    print_op_details(
        "Test 11: Correct mask shape", "load", arr, arr_slice, mask=mask, output=result
    )

    assert result.shape == mask.shape


def test_masked_store_none_case():
    """Test masked_store with mask=None (direct indexing)."""
    arr = np.array([1, 2, 3, 4, 5])
    values = np.array([10, 20, 30])
    arr_copy = arr.copy()
    masked_store(arr_copy, (slice(1, 4),), values, mask=None)
    expected = arr.copy()
    expected[1:4] = values

    print_op_details(
        "Store Test 1: mask=None case",
        "store",
        arr,
        (slice(1, 4),),
        values=values,
        mask=None,
        output=arr_copy,
    )

    assert np.array_equal(arr_copy, expected)


def test_masked_store_in_bounds():
    """Test masked_store with in-bounds indexing and mask."""
    arr = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    values = np.array([[100, 200, 0], [400, 0, 600], [0, 800, 900]])
    mask = np.array([[True, True, False], [True, False, True], [False, True, True]])
    arr_copy = arr.copy()
    masked_store(arr_copy, (slice(0, 3), slice(0, 3)), values, mask=mask)
    expected = np.array([[100, 200, 3], [400, 5, 600], [7, 800, 900]])

    print_op_details(
        "Store Test 2: In-bounds indexing with mask",
        "store",
        arr,
        (slice(0, 3), slice(0, 3)),
        values=values,
        mask=mask,
        output=arr_copy,
    )

    assert np.array_equal(arr_copy, expected)


def test_masked_store_out_of_bounds():
    """Test masked_store with out-of-bounds indexing and mask."""
    arr = np.array([[1, 2], [3, 4]])
    values = np.array(
        [
            [100, 0, 0],
            [0, 200, 0],
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
        ]
    )
    mask = np.array(
        [
            [True, False, False],
            [False, True, False],
            [False, False, False],
            [False, False, False],
            [False, False, False],
        ]
    )
    arr_copy = arr.copy()
    masked_store(arr_copy, (slice(0, 5), slice(0, 3)), values, mask=mask)
    expected = np.array([[100, 2], [3, 200]])

    print_op_details(
        "Store Test 3: Out-of-bounds indexing with mask",
        "store",
        arr,
        (slice(0, 5), slice(0, 3)),
        values=values,
        mask=mask,
        output=arr_copy,
    )

    assert np.array_equal(arr_copy, expected)


def test_masked_store_integer_oob():
    """Test masked_store with integer out-of-bounds indexing."""
    arr = np.array([10, 20, 30])
    values = np.array([999])
    mask = np.array([False])  # mask=False at OOB index
    arr_copy = arr.copy()
    masked_store(arr_copy, ([5],), values, mask=mask)  # Index 5 is OOB

    print_op_details(
        "Store Test 4: Integer indexing out of bounds",
        "store",
        arr,
        ([5],),
        values=values,
        mask=mask,
        output=arr_copy,
    )

    assert np.array_equal(arr_copy, arr)


def test_masked_store_array_indexing():
    """Test masked_store with array indexing."""
    arr = np.array([100, 200, 300, 400, 500])
    indices = np.array([0, 2, 4])
    values = np.array([111, 333, 555])
    mask = np.array([True, True, True])
    arr_copy = arr.copy()
    masked_store(arr_copy, (indices,), values, mask=mask)
    expected = np.array([111, 200, 333, 400, 555])

    print_op_details(
        "Store Test 5: Array indexing",
        "store",
        arr,
        (indices,),
        values=values,
        mask=mask,
        output=arr_copy,
    )

    assert np.array_equal(arr_copy, expected)


def test_masked_store_mixed_indexing():
    """Test masked_store with mixed indexing types."""
    arr = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
    values = np.array([100, 200, 300, 400])
    mask = np.array([True, False, True, False])
    arr_copy = arr.copy()
    masked_store(arr_copy, (slice(0, 4), 2), values, mask=mask)  # set col 2
    expected = np.array([[1, 2, 100, 4], [5, 6, 7, 8], [9, 10, 300, 12]])

    print_op_details(
        "Store Test 6: Mixed indexing types",
        "store",
        arr,
        (slice(0, 4), 2),
        values=values,
        mask=mask,
        output=arr_copy,
    )

    assert np.array_equal(arr_copy, expected)


def test_masked_store_complex_indexing():
    """Test masked_store with complex multi-dimensional indexing."""
    arr = np.array(
        [
            [
                [1, 2, 3],
                [4, 5, 6],
                [7, 8, 9],
            ],
            [
                [11, 12, 13],
                [14, 15, 16],
                [17, 18, 19],
            ],
            [
                [21, 22, 23],
                [24, 25, 26],
                [27, 28, 29],
            ],
        ]
    )
    arr_slice = (
        slice(None, None, None),
        None,
        np.arange(20)[:, None],
        np.arange(3)[None, :],
        None,
        None,
    )
    mask = np.mgrid[:3, :1, :20, :3, :1, :1]
    mask = (mask[0] > 0) & (mask[2] < 3) & (mask[3] < 2)
    values = np.array(
        [
            [
                [99, 99, 99],
                [99, 99, 99],
                [99, 99, 99],
                [99, 99, 99],
                [99, 99, 99],
                [99, 99, 99],
                [99, 99, 99],
                [99, 99, 99],
                [99, 99, 99],
                [99, 99, 99],
                [99, 99, 99],
                [99, 99, 99],
                [99, 99, 99],
                [99, 99, 99],
                [99, 99, 99],
                [99, 99, 99],
                [99, 99, 99],
                [99, 99, 99],
                [99, 99, 99],
                [99, 99, 99],
            ],
            [
                [31, 32, 99],
                [34, 35, 99],
                [37, 38, 99],
                [99, 99, 99],
                [99, 99, 99],
                [99, 99, 99],
                [99, 99, 99],
                [99, 99, 99],
                [99, 99, 99],
                [99, 99, 99],
                [99, 99, 99],
                [99, 99, 99],
                [99, 99, 99],
                [99, 99, 99],
                [99, 99, 99],
                [99, 99, 99],
                [99, 99, 99],
                [99, 99, 99],
                [99, 99, 99],
                [99, 99, 99],
            ],
            [
                [41, 42, 99],
                [44, 45, 99],
                [47, 48, 99],
                [99, 99, 99],
                [99, 99, 99],
                [99, 99, 99],
                [99, 99, 99],
                [99, 99, 99],
                [99, 99, 99],
                [99, 99, 99],
                [99, 99, 99],
                [99, 99, 99],
                [99, 99, 99],
                [99, 99, 99],
                [99, 99, 99],
                [99, 99, 99],
                [99, 99, 99],
                [99, 99, 99],
                [99, 99, 99],
                [99, 99, 99],
            ],
        ]
    )[:, None, :, :, None, None]
    arr_copy = arr.copy()
    masked_store(arr_copy, arr_slice, values, mask=mask)
    expected_modified = np.array(
        [
            [
                [1, 2, 3],
                [4, 5, 6],
                [7, 8, 9],
            ],
            [
                [31, 32, 13],
                [34, 35, 16],
                [37, 38, 19],
            ],
            [
                [41, 42, 23],
                [44, 45, 26],
                [47, 48, 29],
            ],
        ]
    )

    print(
        "Store Test 7: Complex multi-dimensional indexing (omitted details, check code to see inputs/outputs)"
    )

    assert np.array_equal(arr_copy, expected_modified)


def test_masked_store_oob_with_true_mask():
    """Test that IndexError is raised when mask=True at out-of-bounds location."""
    arr = np.array([[1, 2], [3, 4]])
    values = np.array([[100, 200], [300, 400], [500, 600]])
    mask = np.array(
        [
            [True, True],
            [True, True],
            [True, False],  # This position is OOB and mask=True
        ]
    )
    arr_copy = arr.copy()

    print_op_details(
        "Store Test 8: OOB with mask=True should raise IndexError",
        "store",
        arr,
        (slice(0, 3), slice(0, 2)),
        values=values,
        mask=mask,
        error_expected="IndexError",
    )

    with pytest.raises(IndexError):
        masked_store(arr_copy, (slice(0, 3), slice(0, 2)), values, mask=mask)
    print("Success: Correctly raised IndexError\n")


def test_masked_store_values_mask_shape_mismatch():
    """Test that AssertionError is raised when values and mask shapes don't match."""
    arr = np.array([1, 2, 3])
    values = np.array([10, 20, 30])  # Shape (3,)
    mask = np.array([True, False])  # Shape (2,) - different from values
    arr_copy = arr.copy()

    print_op_details(
        "Store Test 9: Values and mask shape mismatch should raise AssertionError",
        "store",
        arr,
        (slice(0, 3),),
        values=values,
        mask=mask,
        error_expected="AssertionError",
    )

    with pytest.raises(AssertionError):
        masked_store(arr_copy, (slice(0, 3),), values, mask=mask)
    print("Success: Correctly raised AssertionError\n")


def test_masked_store_values_shape_mismatch():
    """Test that IndexError is raised when values shape is incompatible."""
    arr = np.array([1, 2, 3])
    values = np.array([10, 20, 30, 40])  # Wrong shape
    mask = np.array([True, False, True, False])
    arr_copy = arr.copy()

    print_op_details(
        "Store Test 10: Values shape mismatch",
        "store",
        arr,
        (slice(0, 3),),
        values=values,
        mask=mask,
        error_expected="IndexError",
    )

    with pytest.raises(IndexError):
        masked_store(arr_copy, (slice(0, 3),), values, mask=mask)
    print("Success: Correctly raised IndexError\n")
