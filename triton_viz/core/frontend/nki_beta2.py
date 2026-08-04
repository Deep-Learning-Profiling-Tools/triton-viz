from collections.abc import Callable
from typing import Any

from triton_viz.core.data import (
    Allocate,
    BinaryOp,
    DmaTranspose,
    Dot,
    Op,
    ProgramId,
    Transfer,
)

from .base import AdapterResult, Frontend, _LangPatchScope, register_frontend


HAS_NKI_BETA2 = False
nisa = None
nl = None
NDArray = None
nki_builder = None
try:
    from triton_viz.core.simulation import nki_beta2 as _nki_beta2  # type: ignore

    NDArray = _nki_beta2.NDArray
    nki_builder = _nki_beta2.nki_builder
    nisa = _nki_beta2.nisa
    nl = _nki_beta2.nl

    HAS_NKI_BETA2 = True
except ModuleNotFoundError:
    pass


def _nki_beta2_dot_adapter(
    dst: Any, stationary: Any, moving: Any, *_args: Any, **kwargs: Any
) -> AdapterResult:
    assert NDArray is not None
    # Do NOT materialize a transposed copy here: building ``NDArray(value=...T)``
    # allocates fresh NumPy storage, so the recorded stationary ``data_ptr()``
    # would differ from the SBUF address the producing DMA wrote, breaking
    # ``Dot.input_ptrs`` <-> ``Transfer.dst_ptr`` dependency matching. Pass the
    # original ``stationary`` object through and flag that the tracer/consumer
    # should treat it as transposed, so the pointer is preserved.
    return AdapterResult(stationary, moving, transpose_input=True)


def _nki_beta2_binary_adapter(
    dst: Any, data1: Any, data2: Any, op: Any, *_args: Any, **kwargs: Any
) -> AdapterResult:
    return AdapterResult(data1, data2, op, dst)


NKI_BETA2_ADAPTERS: dict[type[Op], Callable[..., AdapterResult]] = {}
NKI_BETA2_NAMESPACES: dict[Any, dict[str, type[Op]]] = {}
if HAS_NKI_BETA2:
    assert nki_builder is not None

    NKI_BETA2_NAMESPACES = {
        _nki_beta2: {
            "program_id": ProgramId,
            "ndarray": Allocate,
            "nc_matmul": Dot,
            "dma_copy": Transfer,
            "dma_transpose": DmaTranspose,
            "tensor_copy": Transfer,
            "tensor_tensor": BinaryOp,
        },
    }

    NKI_BETA2_ADAPTERS = {
        ProgramId: lambda axis, *_args, **_kwargs: AdapterResult(axis),
        Allocate: lambda *_args, **_kwargs: AdapterResult(),
        Dot: _nki_beta2_dot_adapter,
        Transfer: lambda dst, src, *_args, **_kwargs: AdapterResult(
            src,
            dst,
            src.buffer,
            dst.buffer,
            "copy",
        ),
        BinaryOp: _nki_beta2_binary_adapter,
        DmaTranspose: lambda dst, src, *_args, **_kwargs: AdapterResult(
            src,
            dst,
            src.buffer,
            dst.buffer,
            "transpose",
        ),
    }


class NKIBeta2Frontend(Frontend):
    def __init__(self):
        definition = Frontend.from_namespaces(
            name="nki_beta2",
            builder=nki_builder,
            namespaces=NKI_BETA2_NAMESPACES,
            adapters=NKI_BETA2_ADAPTERS,
        )
        super().__init__(
            name=definition.name,
            builder=definition.builder,
            original_ops=definition.original_ops,
            adapters=definition.adapters,
            namespaces=definition.namespaces,
        )

    def patch_lang(self, fn, client_manager: Any = None) -> _LangPatchScope:
        from triton_viz.core.simulation.nki_beta2 import nki_patch_lang

        scope = _LangPatchScope()
        nki_patch_lang(scope)
        return scope


frontend = register_frontend(NKIBeta2Frontend())
