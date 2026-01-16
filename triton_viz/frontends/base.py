from dataclasses import dataclass
from typing import Any, Callable


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


def program_id_adapter(axis: Any, *_args: Any, **_kwargs: Any) -> AdapterResult:
    return AdapterResult(axis)


@dataclass(frozen=True)
class Frontend:
    """
    Defines a DSL frontend's (Triton, NKI, etc.) patchable ops and adapter metadata.

    builder: backend builder used for interpreter execution.
    op_list: all op types supported by this frontend.
    original_ops: mapping from op types to unpatched callables.
    op_attr_names: mapping from op types to attribute names on the primary namespace.
    adapters: mapping from op types to adapter functions.
        Adapters transform DSL function signatures into DSL-independent signatures for general analysis.
    namespaces: mapping of namespaces (e.g. triton.language, triton.language.math, etc.)
        to op attribute names for patching.
    """

    builder: Any
    op_list: set[type[Any]]
    original_ops: dict[type[Any], Any]
    op_attr_names: dict[type[Any], str]
    adapters: dict[type[Any], Callable[..., AdapterResult]]
    namespaces: dict[Any, dict[type[Any], str]]

    @classmethod
    def from_namespaces(
        cls,
        *,
        builder: Any,
        namespaces: dict[Any, dict[type[Any], str]],
        adapters: dict[type[Any], Callable[..., AdapterResult]],
        primary_namespace: Any,
        default_adapter: Callable[..., AdapterResult] = passthrough_adapter,
    ) -> "Frontend":
        """
        Builds a Frontend by deriving op_list, original_ops, and op_attr_names.

        Also fills in any missing adapters using default_adapter.
        """
        op_list, original_ops, op_attr_names = build_frontend_ops(
            namespaces, primary_namespace
        )
        ensure_adapters(adapters, op_list, default_adapter)
        return cls(
            builder=builder,
            op_list=op_list,
            original_ops=original_ops,
            op_attr_names=op_attr_names,
            adapters=adapters,
            namespaces=namespaces,
        )


OPERATION_REGISTRY: dict[str, Frontend] = {}


def register_frontend(name: str, frontend: Frontend) -> None:
    OPERATION_REGISTRY[name] = frontend


def build_frontend_ops(
    namespaces: dict[Any, dict[type[Any], str]], primary_namespace: Any
) -> tuple[set[type[Any]], dict[type[Any], Any], dict[type[Any], str]]:
    """
    Returns (op_list, original_ops, op_attr_names) derived from namespaces.

    op_list: all op types across namespaces.
    original_ops: mapping from op types to unpatched callables.
    op_attr_names: mapping from op types to attribute names on the primary namespace.
    """
    op_list: set[type[Any]] = set()
    original_ops: dict[type[Any], Any] = {}
    for namespace, attrs in namespaces.items():
        op_list |= attrs.keys()
        for op, attr in attrs.items():
            if op not in original_ops:
                original_ops[op] = getattr(namespace, attr)
    op_attr_names = namespaces.get(primary_namespace, {})
    return op_list, original_ops, op_attr_names


def ensure_adapters(
    adapters: dict[type[Any], Callable[..., AdapterResult]],
    op_list: set[type[Any]],
    default: Callable[..., AdapterResult],
) -> None:
    for op_type in op_list:
        adapters.setdefault(op_type, default)
