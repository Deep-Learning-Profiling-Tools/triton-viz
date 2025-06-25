import numpy as np
import triton.language as tl

from triton_viz.core.data import Load, RawLoad
from triton_viz.clients.sanitizer.sanitizer import SanitizerSymbolicExecution

def _sum_offsets_from_addptr(expr):
    """
    Traverse an addptr SymbolicExpr and sum all constant offsets.
    If any offset is not constant, return None.
    """
    offsets = []
    non_const_offset = None

    cur = expr
    while cur.op == "addptr":
        off = cur.offsets
        if off.op != "const": # If any offset is not constant, we cannot sum it.
            non_const_offset = off
            break
        offsets.append(np.asarray(off.to_py(), dtype=np.int64))
        cur = cur.ptr

    if non_const_offset:
        raise ValueError(f"Some non-constant offsets found ({non_const_offset}) in the addptr chain.")
    return np.sum(offsets, axis=0)

class CaptureSanitizer(SanitizerSymbolicExecution):
    """
    Record all offsets, then union into a set.
    """

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self._offset_lists: list[list[int]] = list()

    @property
    def observed_offsets(self):
        return self._offset_lists

    def register_op_callback(self, op_type):
        pre, post, orig_overrider = super().register_op_callback(op_type)
        if op_type not in (Load, RawLoad) or orig_overrider is None:
            return pre, post, orig_overrider
        def new_load_overrider(ptr, *a, **k):
            # exec original overrider
            load_expr = orig_overrider(ptr, *a, **k)

            # Important: We only record pointers accessing fp32!
            # This is because fp32 is usually the outermost dtype of a "load of a load" chain.
            # This is the case in all unittests.
            p = load_expr.ptr
            if (
                hasattr(p, "dtype_tt")
                and isinstance(p.dtype_tt, tl.pointer_type)
                and p.dtype_tt.element_ty is tl.float32
            ): # filtering fp32 pointers
                offs = _sum_offsets_from_addptr(p)
                if offs is not None:
                    self._offset_lists.append(offs.tolist())
            return load_expr
        return pre, post, new_load_overrider
