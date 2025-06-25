from triton_viz.clients.sanitizer.sanitizer import SymbolicExpr
import triton.language as tl


def test_addptr_expr_eval():
    base = SymbolicExpr.from_value(1000)  # Synthetic base address
    base.dtype_tt = tl.pointer_type(
        tl.int32
    )  # Simulate a pointer type, int32 = 4 bytes
    offset = SymbolicExpr.from_value(3)
    expr = SymbolicExpr("addptr", base, offset)
    assert expr.eval()[0] == 1000 + 3 * 4


# def test_addptr_overrider():
#     # 通过 sanitizer 的 overrider 走一遍
#     ptr_dtype = tl.pointer_type(tl.int32)
#     ptr_th    = TensorHandle(1000, ptr_dtype)
#     sanitizer = SanitizerSymbolicExecution(abort_on_error=False)
#     _, _, addptr_fn = sanitizer.register_op_callback(AddPtr)
#     expr = addptr_fn(ptr_th, 3)                   # 3 是 offset
#     assert expr.op == "addptr"
#     assert expr.eval() == 1000 + 3 * 4            # element_bytewidth = 4
