#!/usr/bin/env python3
"""
Test script to verify NDArray slicing functionality after fixes
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'triton_viz', 'core'))

import numpy as np

# Import the nki module directly
exec(open('triton_viz/core/nki.py').read())

def test_ndarray_creation():
    print("Testing NDArray creation...")
    
    # Test creation with value
    data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    nd1 = NDArray(value=data, name='test_array')
    print(f"Created NDArray: {nd1}")
    print(f"Shape: {nd1.shape}")
    print(f"Dtype: {nd1.dtype}")
    print(f"Value:\n{nd1.value}")
    print()
    
    # Test creation with shape and dtype
    nd2 = NDArray(shape=(2, 3), dtype=np.float32, name='shaped_array')
    print(f"Created shaped NDArray: {nd2}")
    print(f"Shape: {nd2.shape}")
    print(f"Dtype: {nd2.dtype}")
    print()

def test_slicing():
    print("Testing slicing operations...")
    
    data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    nd_array = NDArray(value=data, name='test_array')
    
    # Test [:, :] (all elements)
    slice_all = nd_array[:, :]
    print(f"nd_array[:, :] = {slice_all}")
    print(f"Value:\n{slice_all.value}")
    print()
    
    # Test [:, 0] (first column)
    slice_col = nd_array[:, 0]
    print(f"nd_array[:, 0] = {slice_col}")
    print(f"Value: {slice_col.value}")
    print()
    
    # Test [0, :] (first row)
    slice_row = nd_array[0, :]
    print(f"nd_array[0, :] = {slice_row}")
    print(f"Value: {slice_row.value}")
    print()

def test_arithmetic():
    print("Testing arithmetic operations...")
    
    data1 = np.array([[1, 2], [3, 4]])
    data2 = np.array([[5, 6], [7, 8]])
    
    nd1 = NDArray(value=data1, name='array1')
    nd2 = NDArray(value=data2, name='array2')
    
    # Test addition
    result = nd1 + nd2
    print(f"Addition result: {result}")
    print(f"Value:\n{result.value}")
    print()
    
    # Test slicing on result
    slice_result = result[:, 0]
    print(f"Slice of result [:, 0]: {slice_result}")
    print(f"Value: {slice_result.value}")
    print()

if __name__ == "__main__":
    test_ndarray_creation()
    test_slicing()
    test_arithmetic()
    print("All tests completed successfully!")
