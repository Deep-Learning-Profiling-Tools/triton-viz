from collections.abc import Callable
from typing import Any

import numpy as np
import triton.language as tl
import triton.experimental.gluon.language as gluon_lang  # type: ignore
from triton.experimental.gluon.language import _core as gluon_core  # type: ignore
from triton.experimental.gluon.language import _math as gluon_math  # type: ignore
from triton.experimental.gluon.language import _semantic as gluon_semantic  # type: ignore
from triton.experimental.gluon.language.amd import cdna3 as gluon_amd_cdna3  # type: ignore
from triton.experimental.gluon.language.amd import cdna4 as gluon_amd_cdna4  # type: ignore
from triton.experimental.gluon.language.amd import gfx1250 as gluon_amd_gfx1250  # type: ignore
from triton.experimental.gluon.language.amd import rdna3 as gluon_amd_rdna3  # type: ignore
from triton.experimental.gluon.language.amd import rdna4 as gluon_amd_rdna4  # type: ignore
from triton.experimental.gluon.language.amd.cdna4 import (  # type: ignore
    async_copy as gluon_amd_cdna4_async_copy,
)
from triton.experimental.gluon.language.amd.gfx1250 import (  # type: ignore
    async_copy as gluon_amd_async_copy,
)
from triton.experimental.gluon.language.amd.gfx1250 import (  # type: ignore
    tdm as gluon_amd_tdm,
)
from triton.experimental.gluon.language.nvidia.blackwell import (  # type: ignore
    tma as gluon_blackwell_tma,
)
from triton.experimental.gluon.language.nvidia.hopper import (  # type: ignore
    tma as gluon_hopper_tma,
)

from ..data import (
    AddPtr,
    Allocate,
    BinaryOp,
    Broadcast,
    Dot,
    ExpandDims,
    Fma,
    Join,
    Load,
    MakeRange,
    Op,
    ProgramId,
    Reshape,
    Rsqrt,
    Splat,
    Store,
    TernaryOp,
    UnaryOp,
)
from ..symbolic_metadata import (
    INT32,
    SymbolicTensorValue,
    SymbolicTypeSpec,
    TensorDescriptorAccess,
)

from .base import AdapterResult, Frontend, _LangPatchScope, register_frontend
from .triton import TritonFrontend

_WARP_SPECIALIZE_SCHEDULER: Any = None
_MISSING = object()


def _set_warp_specialize_scheduler(scheduler: Any) -> Any:
    global _WARP_SPECIALIZE_SCHEDULER
    previous = _WARP_SPECIALIZE_SCHEDULER
    _WARP_SPECIALIZE_SCHEDULER = scheduler
    return previous


def _maybe_yield_warp_specialize() -> None:
    if _WARP_SPECIALIZE_SCHEDULER is not None:
        _WARP_SPECIALIZE_SCHEDULER.yield_point()


def _gluon_load_adapter(
    ptr: Any,
    *_args: Any,
    mask: Any = None,
    other: Any = None,
    **_kwargs: Any,
) -> AdapterResult:
    if _is_simulated_tensor_descriptor_like(ptr) and _args:
        ptr_handle, mask_handle = _simulated_descriptor_pointer_args(ptr, _args[0])
        return AdapterResult(ptr_handle, mask_handle, None)
    if mask is None and _args:
        mask = _args[0]
    ptr = _gluon_tensor_handle(ptr)
    mask = _gluon_tensor_handle(mask)
    ptr_data = getattr(ptr, "data", _MISSING)
    if mask is None and ptr_data is not _MISSING:
        from triton.runtime.interpreter import TensorHandle

        mask = TensorHandle(np.ones_like(ptr_data, dtype=bool), tl.int1)
    return AdapterResult(ptr, mask, None)


def _gluon_descriptor_load_adapter(
    pred_arg_index: int,
    *,
    descriptor_kwarg: str,
    coords_kwarg: str,
) -> Callable[..., AdapterResult]:
    def adapter(
        *args: Any,
        mask: Any = None,
        pred: Any = None,
        **kwargs: Any,
    ) -> AdapterResult:
        if mask is None:
            mask = pred
        load_args = args[1:] if args else ()
        if mask is None and len(load_args) > pred_arg_index:
            mask = load_args[pred_arg_index]
        descriptor = args[0] if args else kwargs[descriptor_kwarg]
        coords = load_args[0] if load_args else kwargs[coords_kwarg]
        return AdapterResult(
            TensorDescriptorAccess(descriptor, coords, mask), mask, None
        )

    return adapter


def _gluon_async_copy_load_adapter(
    smem: Any,
    pointer: Any,
    mask: Any = None,
    *_args: Any,
    **_kwargs: Any,
) -> AdapterResult:
    return _gluon_load_adapter(pointer, mask=mask)


def _gluon_async_copy_store_adapter(
    pointer: Any,
    smem: Any,
    mask: Any = None,
    *_args: Any,
    **_kwargs: Any,
) -> AdapterResult:
    return _gluon_store_adapter(pointer, mask=mask)


def _gluon_buffer_load_to_shared_adapter(
    smem: Any,
    ptr: Any,
    offsets: Any,
    *args: Any,
    **kwargs: Any,
) -> AdapterResult:
    del smem
    return _gluon_load_adapter(
        ptr + offsets,
        mask=args[0] if args else kwargs.get("mask"),
    )


def _gluon_store_adapter(
    ptr: Any,
    value: Any = None,
    *_args: Any,
    mask: Any = None,
    pred: Any = None,
    **_kwargs: Any,
) -> AdapterResult:
    if _is_simulated_tensor_descriptor_like(ptr) and _args:
        ptr_handle, mask_handle = _simulated_descriptor_pointer_args(ptr, value)
        return AdapterResult(ptr_handle, mask_handle, None)
    if mask is None:
        mask = pred
    if _is_global_tensor_descriptor_like(ptr) and value is not None:
        return AdapterResult(TensorDescriptorAccess(ptr, value, mask), mask, None)
    if (
        mask is None
        and not _is_global_tensor_descriptor_like(ptr)
        and not _is_shared_memory_descriptor_like(ptr)
    ):
        if _args:
            mask = _args[0]
    ptr = _gluon_tensor_handle(ptr)
    mask = _gluon_tensor_handle(mask)
    ptr_data = getattr(ptr, "data", _MISSING)
    if mask is None and ptr_data is not _MISSING:
        from triton.runtime.interpreter import TensorHandle

        mask = TensorHandle(np.ones_like(ptr_data, dtype=bool), tl.int1)
    return AdapterResult(ptr, mask, None)


def _is_shared_memory_descriptor_like(value: Any) -> bool:
    return type(value).__name__ == "shared_memory_descriptor"


def _is_global_tensor_descriptor_like(value: Any) -> bool:
    type_name = type(value).__name__
    return type_name in {
        "tensor_descriptor",
        "tensor_descriptor_im2col",
        "TensorDescriptor",
        "TensorDescriptorIm2Col",
    }


def _is_simulated_tensor_descriptor_like(value: Any) -> bool:
    return type(value).__name__ == "SimulatedTensorDescriptor"


def _simulated_descriptor_scalar(value: Any) -> int:
    if type(value).__name__ == "constexpr":
        value = getattr(value, "value", value)
    if isinstance(value, gluon_core.tensor):
        value = value.handle
    tensor_data = getattr(value, "data", _MISSING)
    if tensor_data is not _MISSING:
        data = np.asarray(tensor_data)
        if data.size == 1:
            return int(data.reshape(-1)[0])
    return int(value)


def _simulated_descriptor_pointer_args(descriptor: Any, coord: Any) -> tuple[Any, Any]:
    from triton.runtime.interpreter import TensorHandle

    coord = coord if isinstance(coord, (list, tuple)) else (coord,)
    coord_values = tuple(_simulated_descriptor_scalar(item) for item in coord)
    block_shape = tuple(int(dim) for dim in descriptor.block_shape)
    strides = tuple(int(stride) for stride in descriptor.strides)
    base = descriptor.base
    data_ptr = getattr(base, "data_ptr", None)
    base_ptr = int(data_ptr()) if callable(data_ptr) else 0
    dtype = getattr(descriptor, "dtype", None)
    if dtype is None and getattr(base, "dtype", _MISSING) is not _MISSING:
        import triton

        dtype = tl.str_to_ty(triton.runtime.jit.mangle_type(base), None)
        if isinstance(dtype, tl.pointer_type):
            dtype = dtype.element_ty
    element_size_fn = getattr(base, "element_size", None)
    element_size = int(element_size_fn()) if callable(element_size_fn) else 1
    offsets = np.zeros(block_shape, dtype=np.uint64)
    mask = np.ones(block_shape, dtype=bool)
    for block_idx in np.ndindex(block_shape):
        element_offset = 0
        for dim, idx in enumerate(block_idx):
            element_offset += (coord_values[dim] + idx) * strides[dim]
        offsets[block_idx] = base_ptr + element_offset * element_size
    return TensorHandle(offsets, tl.pointer_type(dtype or tl.int8)), TensorHandle(
        mask,
        tl.int1,
    )


def _simulated_descriptor_overrider_args(
    op_type: type[Op],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> AdapterResult | None:
    if not args or not _is_simulated_tensor_descriptor_like(args[0]):
        return None
    descriptor = args[0]
    coord = args[1] if len(args) > 1 else kwargs.get("coord")
    if coord is None:
        return None
    if op_type is Load:
        pred = kwargs.get("pred")
        if pred is None and len(args) > 4:
            pred = args[4]
        return AdapterResult(
            TensorDescriptorAccess(descriptor, coord, pred), None, None
        )
    if op_type is Store:
        return AdapterResult(TensorDescriptorAccess(descriptor, coord, None), 0, None)
    return None


def _gluon_tensor_handle(value: Any) -> Any:
    if isinstance(value, gluon_core.tensor):
        return value.handle
    return value


def _gluon_binary_adapter(
    lhs: Any,
    rhs: Any,
    *_args: Any,
    **_kwargs: Any,
) -> AdapterResult:
    return AdapterResult(lhs, rhs)


def _gluon_unary_adapter(arg: Any, *_args: Any, **_kwargs: Any) -> AdapterResult:
    return AdapterResult(arg)


def _gluon_binary_op_adapter(op: Callable) -> Callable[..., AdapterResult]:
    def adapter(lhs: Any, rhs: Any, *_args: Any, **_kwargs: Any) -> AdapterResult:
        return AdapterResult(lhs, rhs, op)

    return adapter


def _gluon_unary_op_adapter(op: Callable) -> Callable[..., AdapterResult]:
    def adapter(arg: Any, *_args: Any, **_kwargs: Any) -> AdapterResult:
        return AdapterResult(arg, op)

    return adapter


def _gluon_where_adapter(
    condition: Any,
    x: Any,
    y: Any,
    *_args: Any,
    **_kwargs: Any,
) -> AdapterResult:
    return AdapterResult(condition, x, y, np.where)


def _gluon_dot_adapter(a: Any, b: Any, *_args: Any, **_kwargs: Any) -> AdapterResult:
    return AdapterResult(_gluon_tensor_handle(a), _gluon_tensor_handle(b))


def _gluon_allocate_adapter(*args: Any, **_kwargs: Any) -> AdapterResult:
    return AdapterResult(*args[:4])


def _gluon_make_range_adapter(start: Any, end: Any, *_args: Any, **_kwargs: Any):
    start_value = int(start)
    end_value = int(end)
    return AdapterResult(
        SymbolicTypeSpec(INT32, (end_value - start_value,)), start_value, end_value
    )


def _gluon_semantic_addptr_adapter(
    _semantic: Any,
    lhs: Any,
    rhs: Any,
    *_args: Any,
    **_kwargs: Any,
) -> AdapterResult:
    return AdapterResult(lhs, rhs)


def _gluon_semantic_binary_op_adapter(op: Callable) -> Callable[..., AdapterResult]:
    def adapter(
        _semantic: Any,
        lhs: Any,
        rhs: Any,
        *_args: Any,
        **_kwargs: Any,
    ) -> AdapterResult:
        return AdapterResult(lhs, rhs, op)

    return adapter


def _existing_ops(namespace: Any, attrs: dict[str, type[Op]]) -> dict[str, type[Op]]:
    if namespace is None:
        return {}
    namespace_attrs = vars(namespace)
    return {attr: op_type for attr, op_type in attrs.items() if attr in namespace_attrs}


GLUON_CORE_OPS: dict[str, type[Op]] = {
    "program_id": ProgramId,
    "load": Load,
    "store": Store,
    "arange": MakeRange,
    "full": Splat,
    "dot_fma": Dot,
    "expand_dims": ExpandDims,
    "reshape": Reshape,
    "broadcast": Broadcast,
    "join": Join,
    "add": BinaryOp,
    "sub": BinaryOp,
    "mul": BinaryOp,
    "where": TernaryOp,
}


GLUON_MATH_OPS: dict[str, type[Op]] = {
    "exp": UnaryOp,
    "exp2": UnaryOp,
    "fma": Fma,
    "log": UnaryOp,
    "log2": UnaryOp,
    "cos": UnaryOp,
    "rsqrt": Rsqrt,
    "sin": UnaryOp,
    "sqrt": UnaryOp,
    "abs": UnaryOp,
    "floor": UnaryOp,
    "ceil": UnaryOp,
}


GLUON_NAMESPACES: dict[Any, dict[str, type[Op]]] = {
    gluon_core: dict(GLUON_CORE_OPS),
    gluon_lang: dict(GLUON_CORE_OPS),
    gluon_math: GLUON_MATH_OPS,
    gluon_semantic.GluonSemantic: {
        "add": AddPtr,
        "less_than": BinaryOp,
        "less_equal": BinaryOp,
        "greater_than": BinaryOp,
        "greater_equal": BinaryOp,
    },
}
_TMA_NAMESPACES: tuple[tuple[Any, dict[str, type[Op]]], ...] = (
    (
        gluon_hopper_tma,
        {
            "make_tensor_descriptor": Allocate,
            "async_load": Load,
            "async_load_im2col": Load,
            "async_store": Store,
            "async_copy_global_to_shared": Load,
            "async_copy_global_to_shared_im2col": Load,
            "async_copy_shared_to_global": Store,
            "async_atomic_add": Store,
            "async_atomic_min": Store,
            "async_atomic_max": Store,
            "async_atomic_and": Store,
            "async_atomic_or": Store,
            "async_atomic_xor": Store,
        },
    ),
    (
        gluon_blackwell_tma,
        {
            "make_tensor_descriptor": Allocate,
            "async_load": Load,
            "async_load_im2col": Load,
            "async_store": Store,
            "async_copy_global_to_shared": Load,
            "async_copy_global_to_shared_im2col": Load,
            "async_copy_shared_to_global": Store,
            "async_atomic_add": Store,
            "async_atomic_min": Store,
            "async_atomic_max": Store,
            "async_atomic_and": Store,
            "async_atomic_or": Store,
            "async_atomic_xor": Store,
        },
    ),
    (
        gluon_amd_cdna3,
        {
            "mfma": Dot,
            "buffer_load": Load,
            "buffer_store": Store,
            "buffer_atomic_add": Store,
            "buffer_atomic_max": Store,
            "buffer_atomic_min": Store,
            "buffer_atomic_and": Store,
            "buffer_atomic_or": Store,
            "buffer_atomic_xor": Store,
            "buffer_atomic_xchg": Store,
        },
    ),
    (
        gluon_amd_cdna4,
        {
            "mfma": Dot,
            "buffer_load": Load,
            "buffer_store": Store,
            "buffer_atomic_add": Store,
            "buffer_atomic_max": Store,
            "buffer_atomic_min": Store,
            "buffer_atomic_and": Store,
            "buffer_atomic_or": Store,
            "buffer_atomic_xor": Store,
            "buffer_atomic_xchg": Store,
        },
    ),
    (
        gluon_amd_rdna3,
        {
            "wmma": Dot,
        },
    ),
    (
        gluon_amd_rdna4,
        {
            "wmma": Dot,
        },
    ),
    (
        gluon_amd_cdna4_async_copy,
        {
            "global_load_to_shared": Load,
            "buffer_load_to_shared": Load,
        },
    ),
    (
        gluon_amd_gfx1250,
        {
            "wmma": Dot,
            "buffer_load": Load,
            "buffer_store": Store,
        },
    ),
    (
        gluon_amd_tdm,
        {
            "make_tensor_descriptor": Allocate,
            "update_tensor_descriptor": Allocate,
            "async_load": Load,
            "async_store": Store,
        },
    ),
    (
        gluon_amd_async_copy,
        {
            "global_to_shared": Load,
            "shared_to_global": Store,
        },
    ),
)
for namespace, attrs in _TMA_NAMESPACES:
    existing = _existing_ops(namespace, attrs)
    if existing:
        GLUON_NAMESPACES[namespace] = existing

_GLUON_BINARY_NUMPY_OPS: dict[str, Callable] = {
    "add": np.add,
    "sub": np.subtract,
    "mul": np.multiply,
}
_GLUON_UNARY_NUMPY_OPS: dict[str, Callable] = {
    "abs": np.abs,
    "ceil": np.ceil,
    "cos": np.cos,
    "exp": np.exp,
    "exp2": np.exp2,
    "floor": np.floor,
    "log": np.log,
    "log2": np.log2,
    "sin": np.sin,
    "sqrt": np.sqrt,
}
_TMA_LOAD_PRED_ARG_INDICES: dict[str, int] = {
    "async_load": 3,
    "async_load_im2col": 4,
    "async_copy_global_to_shared": 3,
    "async_copy_global_to_shared_im2col": 4,
}
GLUON_CALLABLE_ADAPTERS: dict[Callable, Callable[..., AdapterResult]] = {}
for namespace, attrs in GLUON_NAMESPACES.items():
    for attr, op_type in attrs.items():
        original = getattr(namespace, attr)
        if op_type is BinaryOp and attr in _GLUON_BINARY_NUMPY_OPS:
            GLUON_CALLABLE_ADAPTERS[original] = _gluon_binary_op_adapter(
                _GLUON_BINARY_NUMPY_OPS[attr]
            )
        elif op_type is UnaryOp and attr in _GLUON_UNARY_NUMPY_OPS:
            GLUON_CALLABLE_ADAPTERS[original] = _gluon_unary_op_adapter(
                _GLUON_UNARY_NUMPY_OPS[attr]
            )
        if namespace is gluon_amd_tdm and attr == "async_load":
            GLUON_CALLABLE_ADAPTERS[original] = _gluon_descriptor_load_adapter(
                2,
                descriptor_kwarg="src",
                coords_kwarg="offsets",
            )
        elif namespace is gluon_amd_async_copy and attr == "global_to_shared":
            GLUON_CALLABLE_ADAPTERS[original] = _gluon_async_copy_load_adapter
        elif namespace is gluon_amd_async_copy and attr == "shared_to_global":
            GLUON_CALLABLE_ADAPTERS[original] = _gluon_async_copy_store_adapter
        elif (
            namespace is gluon_amd_cdna4_async_copy and attr == "global_load_to_shared"
        ):
            GLUON_CALLABLE_ADAPTERS[original] = _gluon_async_copy_load_adapter
        elif (
            namespace is gluon_amd_cdna4_async_copy and attr == "buffer_load_to_shared"
        ):
            GLUON_CALLABLE_ADAPTERS[original] = _gluon_buffer_load_to_shared_adapter
        elif attr in _TMA_LOAD_PRED_ARG_INDICES:
            GLUON_CALLABLE_ADAPTERS[original] = _gluon_descriptor_load_adapter(
                _TMA_LOAD_PRED_ARG_INDICES[attr],
                descriptor_kwarg="tensor_desc",
                coords_kwarg="coord",
            )

GLUON_ADAPTERS: dict[type[Op], Callable[..., AdapterResult]] = {
    ProgramId: lambda axis, *_args, **_kwargs: AdapterResult(axis),
    Load: _gluon_load_adapter,
    Store: _gluon_store_adapter,
    MakeRange: _gluon_make_range_adapter,
    Splat: lambda shape, value, *_args, **_kwargs: AdapterResult(shape, value),
    Allocate: _gluon_allocate_adapter,
    Dot: _gluon_dot_adapter,
    BinaryOp: _gluon_binary_adapter,
    AddPtr: _gluon_binary_adapter,
    UnaryOp: _gluon_unary_adapter,
    TernaryOp: _gluon_where_adapter,
    Fma: lambda x, y, z, *_args, **_kwargs: AdapterResult(x, y, z),
    Rsqrt: lambda arg, *_args, **_kwargs: AdapterResult(arg),
}
GLUON_CALLABLE_ADAPTERS[
    getattr(gluon_semantic.GluonSemantic, "add")
] = _gluon_semantic_addptr_adapter
_GLUON_SEMANTIC_BINARY_NUMPY_OPS: dict[str, Callable] = {
    "less_than": np.less,
    "less_equal": np.less_equal,
    "greater_than": np.greater,
    "greater_equal": np.greater_equal,
}
for attr, op in _GLUON_SEMANTIC_BINARY_NUMPY_OPS.items():
    GLUON_CALLABLE_ADAPTERS[
        getattr(gluon_semantic.GluonSemantic, attr)
    ] = _gluon_semantic_binary_op_adapter(op)


class GluonFrontend(Frontend):
    def __init__(self):
        definition = Frontend.from_namespaces(
            name="gluon",
            builder=None,
            namespaces=GLUON_NAMESPACES,
            adapters=GLUON_ADAPTERS,
        )
        super().__init__(
            name=definition.name,
            builder=definition.builder,
            original_ops=definition.original_ops,
            adapters=definition.adapters,
            namespaces=definition.namespaces,
        )

    @staticmethod
    def symbolic_ops_for_op_type(op_type: type[Op]) -> tuple[str, ...]:
        return TritonFrontend.symbolic_ops_for_op_type(op_type)

    def maybe_yield_for_multism(self) -> None:
        _maybe_yield_warp_specialize()

    def run_op_overrider(
        self,
        op: Callable,
        op_type: type[Op],
        op_overrider: Callable,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ):
        adapter_key = getattr(op, "__triton_viz_original__", op)
        adapter = GLUON_CALLABLE_ADAPTERS.get(adapter_key, self.adapters[op_type])
        adapter_result = adapter(*args, **kwargs)
        if getattr(op, "__triton_viz_simulated__", False):
            descriptor_args = _simulated_descriptor_overrider_args(
                op_type,
                args,
                kwargs,
            )
            # Symbolic clients may not understand every concrete TensorHandle
            # shape yet; keep simulation moving after best-effort callbacks.
            try:
                if descriptor_args is not None:
                    op_overrider(*descriptor_args.args, **descriptor_args.kwargs)
                else:
                    op_overrider(*adapter_result.args, **adapter_result.kwargs)
            except (NotImplementedError, TypeError, ValueError):
                pass
            return op(*args, **kwargs)
        ret = op_overrider(*adapter_result.args, **adapter_result.kwargs)
        return ret

    def normalize_symbolic_value(self, value: Any) -> Any:
        if type(value).__name__ == "constexpr":
            constexpr_value = getattr(value, "value", _MISSING)
            if constexpr_value is not _MISSING:
                return constexpr_value
        if isinstance(value, gluon_core.tensor):
            value = value.handle
        try:
            from triton.runtime.interpreter import TensorHandle
        except ImportError:  # pragma: no cover - Triton import already required here
            TensorHandle = ()  # type: ignore[assignment]
        if isinstance(value, TensorHandle):
            dtype_or_spec = TritonFrontend._normalize_triton_type(value.dtype)
            dtype = (
                dtype_or_spec.dtype
                if isinstance(dtype_or_spec, SymbolicTypeSpec)
                else dtype_or_spec
            )
            return SymbolicTensorValue(
                np.asarray(value.data),
                dtype,
                dict(getattr(value, "attr", {})),
            )
        if isinstance(value, (tl.block_type, tl.pointer_type, tl.core.dtype)):
            return TritonFrontend._normalize_triton_type(value)
        if isinstance(value, SymbolicTensorValue):
            return value
        return super().normalize_symbolic_value(value)

    def to_frontend_symbolic_value(self, value: Any) -> Any:
        if isinstance(value, tuple):
            return tuple(self.to_frontend_symbolic_value(item) for item in value)
        if isinstance(value, list):
            return [self.to_frontend_symbolic_value(item) for item in value]
        return value

    def from_frontend_symbolic_value(self, value: Any) -> Any:
        if isinstance(value, gluon_core.tensor):
            value = value.handle
        if isinstance(value, tuple):
            return tuple(self.from_frontend_symbolic_value(item) for item in value)
        if isinstance(value, list):
            return [self.from_frontend_symbolic_value(item) for item in value]
        return value

    def wrap_symbolic_concrete_fn(self, concrete_fn: Callable) -> Callable:
        def wrapped(*args: Any, **kwargs: Any) -> Any:
            frontend_args = tuple(self.to_frontend_symbolic_value(arg) for arg in args)
            frontend_kwargs = {
                key: self.to_frontend_symbolic_value(value)
                for key, value in kwargs.items()
            }
            ret = concrete_fn(*frontend_args, **frontend_kwargs)
            return self.from_frontend_symbolic_value(ret)

        return wrapped

    def patch_lang(self, fn, client_manager: Any = None) -> _LangPatchScope:
        scope = _LangPatchScope()
        globals_dict = getattr(fn, "__globals__", {})
        for namespace, attrs in self.namespaces.items():
            for attr in attrs:
                original = self.original_ops[namespace][attr]
                patched = getattr(namespace, attr)
                for name, value in list(globals_dict.items()):
                    if value is original:
                        scope.set_item(globals_dict, name, patched)
        return scope


frontend = register_frontend(GluonFrontend())
