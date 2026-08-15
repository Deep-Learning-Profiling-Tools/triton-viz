from collections.abc import Callable
from typing import Any

from triton_viz.core.data import (
    Allocate,
    Dot,
    Load,
    MakeRange,
    Op,
    ProgramId,
    ReduceSum,
    Store,
)
from triton_viz.core.data import NkiCompute

from .base import AdapterResult, Frontend, _LangPatchScope, register_frontend


HAS_NKI = False
nki_builder = None
try:
    from triton_viz.core.simulation.nki import NDArray, nki_builder  # type: ignore

    HAS_NKI = True
except ModuleNotFoundError:
    pass


def _nki_dot_adapter(x: Any, y: Any, *_args: Any, **_kwargs: Any) -> AdapterResult:
    assert HAS_NKI
    # Tilebench TensorE kernels use
    # ``nisa.nc_matmul(stationary=..., moving=...)``, which is lowered onto
    # ``nki_builder.nc_matmul``. That operation already carries the transpose
    # in its definition, so flag it for the tracer.
    stationary = _kwargs.get("stationary")
    moving = _kwargs.get("moving")
    if stationary is not None and moving is not None:
        return AdapterResult(stationary, moving, transpose_input=True)
    # Preserve the original operand object (and thus its ``data_ptr()``) instead
    # of materializing a transposed copy, so the recorded ``Dot.input_ptrs`` can
    # be matched against the producing transfer's ``dst_ptr``. The transpose is
    # applied downstream for shape/value rendering only.
    if _kwargs.get("transpose_x", False):
        return AdapterResult(x, y, transpose_input=True)
    return AdapterResult(x, y)


def _nki_reduce_sum_adapter(
    input_tensor: Any, *args: Any, mask: Any = None, **kwargs: Any
) -> AdapterResult:
    axis = args[0] if args else kwargs.get("axis")
    keep_dims = kwargs.get("keep_dims", kwargs.get("keepdims", False))
    return AdapterResult(input_tensor, axis, keep_dims)


NKI_ADAPTERS: dict[type[Op], Callable[..., AdapterResult]] = {}
NKI_NAMESPACES: dict[Any, dict[str, type[Op]]] = {}
if HAS_NKI:
    assert nki_builder is not None

    NKI_NAMESPACES = {
        nki_builder: {
            "program_id": ProgramId,
            "ndarray": Allocate,
            "load": Load,
            "store": Store,
            "matmul": Dot,
            "nc_matmul": Dot,
            "sum": ReduceSum,
            "arange": MakeRange,
            # Elementwise/reduction/activation compute APIs -> one general record.
            # (These are the events the old frontend silently dropped, so real
            # nl.* kernels such as softmax/rmsnorm now trace their compute ops.)
            "exp": NkiCompute,
            "relu": NkiCompute,
            "sigmoid": NkiCompute,
            "tanh": NkiCompute,
            "silu": NkiCompute,
            "gelu": NkiCompute,
            "sqrt": NkiCompute,
            "abs": NkiCompute,
            "log": NkiCompute,
            "pow": NkiCompute,
            "reciprocal": NkiCompute,
            "square": NkiCompute,
            "rsqrt": NkiCompute,
            "copy": NkiCompute,
            "multiply": NkiCompute,
            "add": NkiCompute,
            "subtract": NkiCompute,
            "divide": NkiCompute,
            "maximum": NkiCompute,
            "minimum": NkiCompute,
            "greater": NkiCompute,
            "where": NkiCompute,
            "max": NkiCompute,
            "min": NkiCompute,
            "mean": NkiCompute,
        }
    }

    NKI_ADAPTERS = {
        ProgramId: lambda axis, *_args, **_kwargs: AdapterResult(axis),
        Allocate: lambda *_args, **_kwargs: AdapterResult(),
        Load: lambda src, keys, *, mask=None, **_kwargs: AdapterResult(
            src,
            mask,
            keys,
        ),
        Store: lambda dst, keys, value, *, mask=None, **_kwargs: AdapterResult(
            dst,
            mask,
            keys,
            value,
        ),
        Dot: _nki_dot_adapter,
        ReduceSum: _nki_reduce_sum_adapter,
        # NkiCompute uses the default passthrough adapter: the tracer reconstructs
        # operands/engine from metadata attached to the result NDArray, so it does
        # not depend on each op's positional signature.
    }


class NKIFrontend(Frontend):
    def __init__(self):
        definition = Frontend.from_namespaces(
            name="nki",
            builder=nki_builder,
            namespaces=NKI_NAMESPACES,
            adapters=NKI_ADAPTERS,
        )
        super().__init__(
            name=definition.name,
            builder=definition.builder,
            original_ops=definition.original_ops,
            adapters=definition.adapters,
            namespaces=definition.namespaces,
        )

    def patch_lang(self, fn, client_manager: Any = None) -> _LangPatchScope:
        from triton_viz.core.simulation.nki import nki_patch_lang

        scope = _LangPatchScope()
        nki_patch_lang(scope)
        return scope


frontend = register_frontend(NKIFrontend())
