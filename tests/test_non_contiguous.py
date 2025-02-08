import torch
from triton_viz.clients.utils import check_storage_contiguous, get_physical_addr_from_tensor_slice


def test_transpose():
    a = torch.arange(25).view(5, 5)
    b = a.t()
    print(b)
    print([(0, b.numel() - 1)])

def test_2d_slice():
    a = torch.arange(25).view(5, 5)
    b = a[1:4, 1:4]
    print('b:', b)
    print("is_contiguous:", check_storage_contiguous(b))
    segments = get_physical_addr_from_tensor_slice(b)
    for start, end in segments:
        print(f"[{(start - b.data_ptr()) / b.element_size()}, {(end - b.data_ptr()) / b.element_size()}]")

def test_3d_slice():
    a = torch.arange(125).view(5, 5, 5)
    b = a[1:4, 1:4, 1:4]
    print('b:', b)
    print("is_contiguous:", check_storage_contiguous(b))
    segments = get_physical_addr_from_tensor_slice(b)
    for start, end in segments:
        print(f"[{(start - b.data_ptr()) / b.element_size()}, {(end - b.data_ptr()) / b.element_size()}]")