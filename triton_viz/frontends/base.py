from dataclasses import dataclass
from typing import Any, Callable
from triton_viz.core.data import Op


@dataclass
class AdapterResult:
    """
    For each backend, ops may have slightly different function signatures
    which we run through (backend, function)-specific adapters to return
    standardized args/kwargs for client callbacks.
    """

    args: tuple[Any, ...]
    kwargs: dict[str, Any]

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.args = args
        self.kwargs = kwargs


def passthrough_adapter(*args: Any, **kwargs: Any) -> AdapterResult:
    """Return arguments unchanged for clients that expect the original signature."""
    return AdapterResult(*args, **kwargs)


@dataclass(frozen=True)
class Frontend:
    """
    Defines a DSL frontend's (Triton, NKI, etc.) patchable ops and adapter metadata.

    builder: backend builder used for interpreter execution.
    original_ops: mapping from namespaces and attributes to unpatched callables.
    adapters: mapping from op types to adapter functions.
        Adapters transform DSL function signatures into DSL-independent signatures for general analysis.
    namespaces: mapping of namespaces (e.g. triton.language, triton.language.math, etc.)
        to attribute names and their op types for patching.

    Example:
        original_ops = {
            tl: {"sum": <function tl.sum>, "max": <function tl.max>},
            tl.math: {"umulhi": <function tl.math.umulhi>},
        }
        adapters = {
            ReduceSum: <function reduce_adapter>,
            ReduceMax: <function reduce_adapter>,
            Umulhi: <function passthrough_adapter>,
        }
        namespaces = {
            tl: {"sum": ReduceSum, "max": ReduceMax},
            tl.math: {"umulhi": Umulhi},
        }
    """

    builder: Any
    original_ops: dict[Any, dict[str, Callable]]
    adapters: dict[type[Op], Callable[..., AdapterResult]]
    namespaces: dict[Any, dict[str, type[Op]]]

    @classmethod
    def from_namespaces(
        cls,
        *,
        builder: Any,
        namespaces: dict[Any, dict[str, type[Op]]],
        adapters: dict[type[Op], Callable[..., AdapterResult]],
    ) -> "Frontend":
        """
        Builds a Frontend by deriving original_ops.

        Fills in any missing adapters using passthrough_adapter.
        """
        # save original op definitions
        original_ops: dict[Any, dict[str, Callable]] = {
            namespace: {} for namespace in namespaces
        }
        for namespace, attrs in namespaces.items():
            for attr, op in attrs.items():
                original_ops[namespace][attr] = getattr(namespace, attr)

        # set adapter to passthrough for each op in each namespace if not otherwise specified
        for namespace, attrs in namespaces.items():
            for op_type in attrs.values():
                adapters.setdefault(op_type, passthrough_adapter)

        return cls(
            builder=builder,
            original_ops=original_ops,
            adapters=adapters,
            namespaces=namespaces,
        )


OPERATION_REGISTRY: dict[str, Frontend] = {}
