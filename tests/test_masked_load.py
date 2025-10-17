import numpy as np
from triton_viz.core.nki_masked_load import masked_load, masked_store

def test_masked_load():
    """Test the masked_load function with various scenarios."""

    print("=== Testing masked_load function ===\n")

    # Test 1: mask=None case (direct indexing)
    print("Test 1: mask=None case")
    arr = np.array([1, 2, 3, 4, 5])
    result = masked_load(arr, (slice(1, 4),), mask=None)
    expected = arr[1:4]
    print(f"Input: {arr}")
    print(f"Keys: slice(1, 4)")
    print(f"Result: {result}")
    print(f"Expected: {expected}")
    print(f"Match: {np.array_equal(result, expected)}\n")

    # Test 2: In-bounds indexing with mask
    print("Test 2: In-bounds indexing with mask")
    arr = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    mask = np.array([[True, True, False], [True, False, True], [False, True, True]])
    result = masked_load(arr, (slice(0, 3), slice(0, 3)), mask=mask)
    result[[0,1,2],[2,1,0]] = 0
    expected = np.array([[1, 2, 0], [4, 0, 6], [0, 8, 9]])
    print(f"Input array:\n{arr}")
    print(f"Mask:\n{mask}")
    print(f"Keys: (slice(0, 2), slice(0, 2))")
    print(f"Result:\n{result}")
    print(f"Expected:\n{expected}")
    print(f"Match: {np.array_equal(result, expected)}\n")

    # Test 3: Out-of-bounds indexing with mask
    print("Test 3: Out-of-bounds indexing with mask")
    arr = np.array([[1, 2], [3, 4]])
    mask = np.array([
        [True , False, False],
        [False, True , False],
        [False, False, False],
        [False, False, False],
        [False, False, False],
    ])  # Shape doesn't match exactly, but demonstrates concept
    # This would normally cause IndexError for arr[5], but with mask should work
    result = masked_load(arr, (slice(0, 5), slice(0, 3)), mask=mask)
    print(f"Input array:\n{arr}")
    print(f"Keys: (slice(0, 5), slice(0, 3)) - exceeds array bounds")
    print(f"Result shape: {result.shape}")
    print(f"Result:\n{result}")
    print("Success: Handled OOB indexing\n")

    # Test 4: Integer indexing OOB
    print("Test 4: Integer indexing out of bounds")
    arr = np.array([10, 20, 30])
    mask = np.array([False])  # Extra mask element
    result = masked_load(arr, ([5],), mask=mask)  # Index 5 is OOB
    print(f"Input: {arr}")
    print(f"Keys: (5,) - OOB index")
    print(f"Result: {result}")
    print(f"Result type: {type(result)}")
    print("Success: Handled OOB integer indexing\n")

    # Test 5: Array indexing
    print("Test 5: Array indexing")
    arr = np.array([100, 200, 300, 400, 500])
    indices = np.array([0, 2, 4])  # Valid indices
    mask = np.array([True, True, True])
    result = masked_load(arr, (indices,), mask=mask)
    expected = arr[indices]
    print(f"Input: {arr}")
    print(f"Indices: {indices}")
    print(f"Result: {result}")
    print(f"Expected: {expected}")
    print(f"Match: {np.array_equal(result, expected)}\n")

    # Test 6: Mixed indexing types
    print("Test 6: Mixed indexing types")
    arr = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
    mask = np.array([True, False, True, False])
    # Mix of slice and integer
    result = masked_load(arr, (slice(0, 4), 2), mask=mask)  # Get column 2, allow extending rows
    print(f"Input array:\n{arr}")
    print(f"Keys: (slice(0, 4), 2) - slice extends beyond array rows")
    print(f"Result: {result}")
    print(f"Result shape: {result.shape}")
    print("Success: Handled mixed indexing\n")

    # Test 7: Mixed indexing types
    print("Test 7: Mixed indexing types")
    arr = np.array([
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
    ])
    arr_slice = (
        slice(None, None, None),
        None,
        np.arange(20)[:, None],
        np.arange(3)[None, :],
        None,
        None
    )
    mask = np.mgrid[:3, :1, :20, :3, :1, :1]
    mask = (mask[0] > 0) & (mask[2] < 3) & (mask[3] < 2)
    result = masked_load(arr, arr_slice, mask=mask)
    expected = np.array([
        [
            [ 6,  6,  6],
            [ 6,  6,  6],
            [ 6,  6,  6],
            [ 6,  6,  6],
            [ 6,  6,  6],
            [ 6,  6,  6],
            [ 6,  6,  6],
            [ 6,  6,  6],
            [ 6,  6,  6],
            [ 6,  6,  6],
            [ 6,  6,  6],
            [ 6,  6,  6],
            [ 6,  6,  6],
            [ 6,  6,  6],
            [ 6,  6,  6],
            [ 6,  6,  6],
            [ 6,  6,  6],
            [ 6,  6,  6],
            [ 6,  6,  6],
            [ 6,  6,  6],
        ],
        [
            [11, 12,  6],
            [14, 15,  6],
            [17, 18,  6],
            [ 6,  6,  6],
            [ 6,  6,  6],
            [ 6,  6,  6],
            [ 6,  6,  6],
            [ 6,  6,  6],
            [ 6,  6,  6],
            [ 6,  6,  6],
            [ 6,  6,  6],
            [ 6,  6,  6],
            [ 6,  6,  6],
            [ 6,  6,  6],
            [ 6,  6,  6],
            [ 6,  6,  6],
            [ 6,  6,  6],
            [ 6,  6,  6],
            [ 6,  6,  6],
            [ 6,  6,  6],
        ],
        [
            [21, 22,  6],
            [24, 25,  6],
            [27, 28,  6],
            [ 6,  6,  6],
            [ 6,  6,  6],
            [ 6,  6,  6],
            [ 6,  6,  6],
            [ 6,  6,  6],
            [ 6,  6,  6],
            [ 6,  6,  6],
            [ 6,  6,  6],
            [ 6,  6,  6],
            [ 6,  6,  6],
            [ 6,  6,  6],
            [ 6,  6,  6],
            [ 6,  6,  6],
            [ 6,  6,  6],
            [ 6,  6,  6],
            [ 6,  6,  6],
            [ 6,  6,  6],
            ],
        ]
    )[:, None, :, :, None, None]
    assert np.allclose(result, expected)

    # Test 8: incomplete slices
    print("Test 8: incomplete slices")
    arr = np.array([
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
    ])
    arr_slice = (
        slice(None, None, None),
    )
    mask = np.mgrid[:3, :3, :3]
    mask = (0 < mask[0]) & (mask[0] < 2) & (mask[1] < 2) & (mask[2] != 1)
    result = masked_load(arr, arr_slice, mask=mask)
    expected = np.array([
        [
            [ 6,  6,  6],
            [ 6,  6,  6],
            [ 6,  6,  6],
        ],
        [
            [11,  6, 13],
            [14,  6, 16],
            [ 6,  6,  6],
        ],
        [
            [ 6,  6,  6],
            [ 6,  6,  6],
            [ 6,  6,  6],
        ],
    ])
    assert np.allclose(result, expected)

    print("Test 9: Make sure index error still happen when mask=True")
    arr = np.array([
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
    ])
    arr_slice = (
        slice(None, None, None),
        slice(0, 4)
    )
    mask = np.mgrid[:3, :4, :3]
    mask = (mask[0] < 3) & (mask[1] < 3) & (mask[2] < 3)
    mask[-1, -1, -1] = True # index error
    try:
        result = masked_load(arr, arr_slice, mask=mask)
        raise RuntimeError("Should've raised an index error but did not")
    except IndexError:
        pass

    print("Test 10: Make sure error if mask shape wrong")
    arr = np.array([
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
    ])
    arr_slice = (slice(None, None, None),)
    mask = np.mgrid[:3, :4, :3]
    mask = (mask[0] < 3) & (mask[1] < 3) & (mask[2] < 3)
    try:
        result = masked_load(arr, arr_slice, mask=mask)
        raise RuntimeError("Should've raised an assertion error (arr[arr_slice].shape != mask.shape) but did not")
    except AssertionError:
        pass


    print("Test 10: Make sure error if mask shape wrong")
    arr = np.array([
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
    ])
    arr_slice = (slice(None, None, None),)
    mask = np.arange(27).reshape(3, 3, 3) % 2 == 0
    result = masked_load(arr, arr_slice, mask=mask)
    print(result)

def test_masked_store():
    """Test the masked_store function with various scenarios."""

    print("=== Testing masked_store function ===\n")

    # Test 1: mask=None case (direct indexing)
    print("Test 1: mask=None case")
    arr = np.array([1, 2, 3, 4, 5])
    values = np.array([10, 20, 30])
    arr_copy = arr.copy()
    masked_store(arr_copy, (slice(1, 4),), values, mask=None)
    expected = arr.copy()
    expected[1:4] = values
    print(f"Original: {arr}")
    print(f"Values: {values}")
    print(f"Result: {arr_copy}")
    print(f"Expected: {expected}")
    assert np.array_equal(arr_copy, expected)

    # Test 2: In-bounds indexing with mask
    print("Test 2: In-bounds indexing with mask")
    arr = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    values = np.array([[100, 200, 0], [400, 0, 600], [0, 800, 900]])
    mask = np.array([[True, True, False], [True, False, True], [False, True, True]])
    arr_copy = arr.copy()
    masked_store(arr_copy, (slice(0, 3), slice(0, 3)), values, mask=mask)
    expected = np.array([[100, 200, 3], [400, 5, 600], [7, 800, 900]])
    print(f"Original array:\n{arr}")
    print(f"Values:\n{values}")
    print(f"Mask:\n{mask}")
    print(f"Result:\n{arr_copy}")
    print(f"Expected:\n{expected}")
    assert np.array_equal(arr_copy, expected)

    # Test 3: Out-of-bounds indexing with mask (should succeed where mask=False)
    print("Test 3: Out-of-bounds indexing with mask")
    arr = np.array([[1, 2], [3, 4]])
    values = np.array([
        [100, 0, 0],
        [0, 200, 0],
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0],
    ])
    mask = np.array([
        [True , False, False],
        [False, True , False],
        [False, False, False],
        [False, False, False],
        [False, False, False],
    ])
    arr_copy = arr.copy()
    masked_store(arr_copy, (slice(0, 5), slice(0, 3)), values, mask=mask)
    expected = np.array([[100, 2], [3, 200]])
    print(f"Original array:\n{arr}")
    print(f"Keys: (slice(0, 5), slice(0, 3)) - exceeds array bounds")
    print(f"Result:\n{arr_copy}")
    print(f"Expected:\n{expected}")
    assert np.array_equal(arr_copy, expected)

    # Test 4: Integer indexing OOB
    print("Test 4: Integer indexing out of bounds")
    arr = np.array([10, 20, 30])
    values = np.array([999])
    mask = np.array([False])  # mask=False at OOB index
    arr_copy = arr.copy()
    masked_store(arr_copy, ([5],), values, mask=mask)  # Index 5 is OOB
    print(f"Original: {arr}")
    print(f"Keys: (5,) - OOB index")
    print(f"Result: {arr_copy}")
    print(f"Expected: {arr} (unchanged)")
    assert np.array_equal(arr_copy, arr)

    # Test 5: Array indexing
    print("Test 5: Array indexing")
    arr = np.array([100, 200, 300, 400, 500])
    indices = np.array([0, 2, 4])
    values = np.array([111, 333, 555])
    mask = np.array([True, True, True])
    arr_copy = arr.copy()
    masked_store(arr_copy, (indices,), values, mask=mask)
    expected = np.array([111, 200, 333, 400, 555])
    print(f"Original: {arr}")
    print(f"Indices: {indices}")
    print(f"Values: {values}")
    print(f"Result: {arr_copy}")
    print(f"Expected: {expected}")
    assert np.array_equal(arr_copy, expected)

    # Test 6: Mixed indexing types
    print("Test 6: Mixed indexing types")
    arr = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
    values = np.array([100, 200, 300, 400])
    mask = np.array([True, False, True, False])
    arr_copy = arr.copy()
    masked_store(arr_copy, (slice(0, 4), 2), values, mask=mask)  # Set column 2
    expected = np.array([[1, 2, 100, 4], [5, 6, 7, 8], [9, 10, 300, 12]])
    print(f"Original array:\n{arr}")
    print(f"Values: {values}")
    print(f"Keys: (slice(0, 4), 2) - slice extends beyond array rows")
    print(f"Result:\n{arr_copy}")
    print(f"Expected:\n{expected}")
    assert np.array_equal(arr_copy, expected)

    # Test 7: Complex 3D indexing (simplified)
    print("Test 7: Complex 3D indexing")
    arr = np.array([
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
    ])
    ## Simpler test case that uses basic slicing with mask
    #arr_slice = (slice(0, 3), slice(0, 3))  # Just use 2D slicing for simplicity
    #mask = np.array([
    #    [True, True, False],
    #    [False, True, True],
    #    [True, False, False]
    #])
    #values = np.array([
    #    [100, 200, 0],
    #    [0, 300, 400],
    #    [500, 0, 0]
    #])
    #arr_copy = arr[:, :, 0]  # Take first slice along last dimension
    #arr_copy = arr_copy.copy()
    #masked_store(arr_copy, arr_slice, values, mask=mask)

    #expected_modified = np.array([
    #    [100, 200, 7],   # Original row 0: [1, 4, 7], mask: [True, True, False] -> [100, 200, 7]
    #    [11, 300, 400],  # Original row 1: [11, 14, 17], mask: [False, True, True] -> [11, 300, 400]
    #    [500, 24, 27]    # Original row 2: [21, 24, 27], mask: [True, False, False] -> [500, 24, 27]
    #])

    arr_slice = (
        slice(None, None, None),
        None,
        np.arange(20)[:, None],
        np.arange(3)[None, :],
        None,
        None
    )
    mask = np.mgrid[:3, :1, :20, :3, :1, :1]
    mask = (mask[0] > 0) & (mask[2] < 3) & (mask[3] < 2)
    result = masked_load(arr, arr_slice, mask=mask)
    values = np.array([
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

    expected_modified = np.array([
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
    ])

    print(f"Input array shape: {arr_copy.shape}")
    print(f"Keys: 2D slicing")
    print(f"Values shape: {values.shape}")
    print(f"Mask shape: {mask.shape}")
    assert np.array_equal(arr_copy, expected_modified)

    # Test 8: Error case - OOB with mask=True should raise IndexError
    print("Test 8: OOB with mask=True should raise IndexError")
    arr = np.array([[1, 2], [3, 4]])
    values = np.array([[100, 200], [300, 400], [500, 600]])
    mask = np.array([
        [True, True],
        [True, True],
        [True, False],  # This position is OOB and mask=True
    ])
    arr_copy = arr.copy()
    try:
        masked_store(arr_copy, (slice(0, 3), slice(0, 2)), values, mask=mask)
        raise RuntimeError("Should've raised an index error but did not")
    except IndexError:
        print("Success: Correctly raised IndexError for OOB with mask=True\n")

    # Test 9: Error case - values and mask shape mismatch
    print("Test 9: Values and mask shape mismatch should raise AssertionError")
    arr = np.array([1, 2, 3])
    values = np.array([10, 20, 30])  # Shape (3,)
    mask = np.array([True, False])   # Shape (2,) - different from values
    arr_copy = arr.copy()
    try:
        masked_store(arr_copy, (slice(0, 3),), values, mask=mask)
        print(f"ERROR: Should have raised AssertionError but didn't")
        print(f"values.shape: {values.shape}")
        print(f"mask.shape: {mask.shape}")
        raise RuntimeError("Should've raised an assertion error but did not")
    except AssertionError:
        print("Success: Correctly raised AssertionError for values/mask shape mismatch\n")

    # Test 10: Values shape mismatch
    print("Test 10: Values shape mismatch should raise AssertionError")
    arr = np.array([1, 2, 3])
    values = np.array([10, 20, 30, 40])  # Wrong shape
    mask = np.array([True, False, True, False])
    arr_copy = arr.copy()
    try:
        masked_store(arr_copy, (slice(0, 3),), values, mask=mask)
        raise RuntimeError("Should've raised an assertion error but did not")
    except AssertionError:
        print("Success: Correctly raised AssertionError for values shape mismatch\n")

    print("All masked_store tests completed!")

if __name__ == "__main__":
    test_masked_load()
    print("\n" + "="*50 + "\n")
    test_masked_store()
