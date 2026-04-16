from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import torch

from ...core.data import Op


AccessMode = Literal["read", "write"]


# Operations supported by ``apply_rmw`` / recognized as valid ``atomic_op``
# values on RMW events. ``"cas"`` is an atomic_op value too (on CAS events)
# but is deliberately NOT part of this set — the semantics are different.
VALID_RMW_OPS: frozenset[str] = frozenset(
    {"add", "xchg", "and", "or", "xor", "max", "min"}
)


_MEM_SEM_ENUM_NAMES: dict[str, str] = {
    "ACQUIRE": "acquire",
    "RELEASE": "release",
    "RELAXED": "relaxed",
    "ACQUIRE_RELEASE": "acq_rel",
}


@dataclass
class AccessEventRecord:
    """Effect-aware memory access event.

    The same dataclass serves Step 1's symbolic load/store path and the
    concrete atomic / plain load-store path. Symbolic events populate
    ``symbolic_expr`` / ``addr_expr`` / ``premises`` / ``local_constraints``
    / ``access_mode``; concrete events populate ``lane_addrs`` /
    ``active_mask`` / ``elem_size`` / ``read_mask`` / ``write_mask`` /
    ``atomic_*`` / ``written_value``.
    """

    # identity / source
    event_id: int
    op_type: type[Op]
    source_location: tuple[str, int, str] | None = None
    grid_idx: tuple[int, ...] | None = None

    # tensor metadata
    tensor: torch.Tensor | None = None
    tensor_name: str | None = None

    # Step 1 symbolic fields (load / store / tensor_pointer_*)
    access_mode: AccessMode | None = None
    symbolic_expr: Any | None = None
    addr_expr: Any | None = None
    premises: tuple[Any, ...] = field(default_factory=tuple)
    local_constraints: tuple[Any, ...] = field(default_factory=tuple)

    # Concrete lane-level fields (populated for atomics + plain concrete events)
    lane_addrs: np.ndarray | None = None
    active_mask: np.ndarray | None = None
    elem_size: int | None = None

    # Effect model
    read_mask: np.ndarray | None = None
    write_mask: np.ndarray | None = None

    # Atomic metadata
    atomic_op: str | None = None
    atomic_sem: str | None = None
    atomic_scope: str | None = None
    atomic_cmp: np.ndarray | None = None
    atomic_val: np.ndarray | None = None
    atomic_old: np.ndarray | None = None
    written_value: np.ndarray | None = None

    # Reserved for Step 4/5 HB-exclusion; inert this PR.
    legacy_atomic: bool = False


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def maybe_concretize(x: Any) -> Any:
    """Best-effort reduce a possibly-symbolic value to its concrete form.

    Handles two input shapes this package actually sees:
      * ``SymbolicExpr`` (and its subclasses) — exposes ``concretize()`` that
        walks the expression tree, runs the original ops, and returns a
        ``TensorHandle`` (or scalar/tuple for const nodes).
      * ``tl.core.tensor`` whose ``.handle`` is a ``SymbolicExpr`` — the
        race detector's atomic overrider sees these when an upstream
        symbolic overrider ran first, wrapping its result in ``tl.core.tensor``.

    Anything already concrete (``TensorHandle``, numpy array, torch tensor,
    Python scalar) passes through unchanged.
    """
    if x is None:
        return None
    handle = getattr(x, "handle", None)
    if handle is not None and hasattr(handle, "concretize"):
        # tl.core.tensor wrapping a SymbolicExpr
        return handle.concretize()
    if hasattr(x, "concretize"):
        return x.concretize()
    return x


def normalize_sem_scope(sem: Any, scope: Any) -> tuple[str, str]:
    """Triton defaults: ``sem='acq_rel'``, ``scope='gpu'``.

    Accepts both string inputs (``"acquire"``, ``"cta"``) and interpreter
    enum values (``MEM_SEMANTIC.ACQUIRE``). Enum names are mapped to the
    short vocabulary used by the HB solver (``"acquire"``, ``"release"``,
    ``"acq_rel"``, ``"relaxed"``); unknown enum names fall back to the
    enum's lowercased name.
    """
    return _canonical_sem(sem), _canonical_scope(scope)


def _canonical_sem(sem: Any) -> str:
    if sem is None:
        return "acq_rel"
    name = getattr(sem, "name", None)
    if isinstance(name, str):
        return _MEM_SEM_ENUM_NAMES.get(name, name.lower())
    return str(sem).lower()


def _canonical_scope(scope: Any) -> str:
    if scope is None:
        return "gpu"
    name = getattr(scope, "name", None)
    if isinstance(name, str):
        return name.lower()
    return str(scope).lower()


def flatten_np(x: Any) -> np.ndarray:
    """Coerce ``x`` (tl.tensor / TensorHandle / ndarray / torch.Tensor /
    python scalar / SymbolicExpr / SymbolicExpr-wrapped tl.tensor) into a
    1-D ndarray. Symbolic inputs are concretized first via
    :func:`maybe_concretize`."""
    x = maybe_concretize(x)
    inner: Any = x
    for attr in ("handle", "data"):
        while hasattr(inner, attr):
            nxt = getattr(inner, attr)
            if nxt is inner:
                break
            inner = nxt
    if isinstance(inner, torch.Tensor):
        arr = inner.detach().cpu().numpy()
    else:
        arr = np.asarray(inner)
    return arr.ravel()


def active_mask_for(mask: Any, nlanes: int) -> np.ndarray:
    """Canonicalize ``mask`` into a bool array of length ``nlanes``.

    Rules:
      - None         → ``np.ones(nlanes, dtype=bool)``
      - scalar bool  → ``np.full(nlanes, bool(mask), dtype=bool)``
      - block/tensor → ``flatten_np(mask).astype(bool)``; length MUST
                       equal ``nlanes`` or ``ValueError`` (no silent broadcast).

    Symbolic inputs are concretized first via :func:`maybe_concretize`.
    """
    mask = maybe_concretize(mask)
    if mask is None:
        return np.ones(nlanes, dtype=bool)

    if isinstance(mask, (bool, np.bool_)):
        return np.full(nlanes, bool(mask), dtype=bool)

    flat = flatten_np(mask)
    if flat.ndim == 0 or flat.size == 1:
        return np.full(nlanes, bool(flat.item()), dtype=bool)

    if flat.shape[0] != nlanes:
        raise ValueError(
            f"active_mask_for: mask length {flat.shape[0]} does not match "
            f"nlanes {nlanes}; explicit broadcast is disallowed."
        )
    return flat.astype(bool, copy=False)


def broadcast_lane_operand(x: Any, nlanes: int) -> np.ndarray:
    """Canonicalize an atomic cmp/val operand into a 1-D ndarray of
    length ``nlanes``. Same shape contract as ``active_mask_for`` so callback
    code never relies on NumPy's implicit broadcasting.

    Rules:
      - python scalar / 0-d array → broadcast to ``nlanes``
      - 1-D array of length ``nlanes`` → pass through
      - anything else → ``ValueError`` with the offending shape

    Symbolic inputs are concretized first via :func:`maybe_concretize`.
    """
    x = maybe_concretize(x)
    if isinstance(x, (int, float, bool, np.integer, np.floating, np.bool_)):
        scalar = np.asarray(x)
        return np.full(nlanes, scalar, dtype=scalar.dtype)

    flat = flatten_np(x)
    if flat.ndim == 0 or flat.size == 1:
        return np.full(nlanes, flat.item(), dtype=flat.dtype)

    if flat.shape[0] != nlanes:
        raise ValueError(
            f"broadcast_lane_operand: operand length {flat.shape[0]} does not "
            f"match nlanes {nlanes}; explicit broadcast is disallowed."
        )
    return flat


def _pointer_elem_size(ptr: Any) -> int | None:
    """Best-effort: pull the element byte-size out of a pointer value.

    Looks for the usual interpreter pointer-element-type chains without
    over-fitting to one runtime shape. Returns ``None`` if nothing authoritative
    is reachable.
    """
    ptr = maybe_concretize(ptr)
    candidate: Any = ptr
    for attr in ("handle", "data"):
        while hasattr(candidate, attr):
            nxt = getattr(candidate, attr)
            if nxt is candidate:
                break
            candidate = nxt

    type_obj = getattr(ptr, "type", None)
    element_ty = getattr(type_obj, "element_ty", None) if type_obj is not None else None
    if element_ty is not None:
        bitwidth = getattr(element_ty, "primitive_bitwidth", None)
        if isinstance(bitwidth, int) and bitwidth > 0 and bitwidth % 8 == 0:
            return bitwidth // 8

    # numpy / torch pointer arrays don't carry element_ty; bail out.
    return None


def infer_elem_size(val: Any, ptr: Any) -> int:
    """Byte size of a single atomic element. POINTER FIRST.

    Order:
      1. pointer element dtype (authoritative — matches the hardware store width).
      2. value dtype from a non-scalar ``val`` (``flatten_np(val).dtype.itemsize``)
         — only taken if the value isn't a bare Python int/float/bool (those
         get boxed to int64/float64 and would mis-classify int32 atomics).
      3. else ``ValueError``.

    Symbolic inputs are concretized first via :func:`maybe_concretize`.
    """
    val = maybe_concretize(val)
    ptr = maybe_concretize(ptr)

    ptr_size = _pointer_elem_size(ptr)
    if ptr_size is not None:
        return ptr_size

    if val is None or isinstance(val, (int, float, bool)):
        raise ValueError(
            "infer_elem_size: pointer has no element type and value is a bare "
            "Python scalar or absent; refusing to guess element size."
        )

    flat = flatten_np(val)
    itemsize = flat.dtype.itemsize
    if itemsize <= 0:
        raise ValueError(
            f"infer_elem_size: value dtype {flat.dtype} has non-positive itemsize."
        )
    return int(itemsize)


def resolve_tensor_from_pointer(
    ptr: Any,
    active_mask: np.ndarray,
    elem_size: int,
    tensor_addrs: list[tuple[int, int, torch.Tensor]],
) -> torch.Tensor | None:
    """Best-effort tensor lookup by concrete address interval.

    1. ``lane_addrs = flatten_np(ptr).astype(int64)``; filter to active lanes.
    2. If no active lanes → ``None``.
    3. ``lo = min(active_addrs)``, ``hi = max(active_addrs) + elem_size - 1``.
    4. Find registered tensor intervals that fully cover ``[lo, hi]``.
    5. Exactly one match → return it; zero or >1 → ``None``.

    Symbolic pointers are concretized first via :func:`maybe_concretize`.
    """
    lane_addrs = flatten_np(ptr).astype(np.int64, copy=False)
    if lane_addrs.shape[0] != active_mask.shape[0]:
        return None
    active_addrs = lane_addrs[active_mask]
    if active_addrs.size == 0:
        return None
    lo = int(active_addrs.min())
    hi = int(active_addrs.max()) + elem_size - 1

    matches: list[torch.Tensor] = []
    for start, end, tensor in tensor_addrs:
        if start <= lo and hi <= end:
            matches.append(tensor)

    if len(matches) == 1:
        return matches[0]
    return None


def effects_at_addr(event: AccessEventRecord, addr: int) -> tuple[bool, bool]:
    """Return ``(reads, writes)`` for byte address ``addr``. Aggregates with
    ``np.any`` across EVERY lane whose element-byte range covers ``addr`` —
    single-lane-only would reintroduce upstream #344's bug."""
    if event.lane_addrs is None or event.elem_size is None:
        raise ValueError("effects_at_addr requires concrete lane_addrs/elem_size")
    active = (
        event.active_mask
        if event.active_mask is not None
        else np.ones_like(event.lane_addrs, dtype=bool)
    )
    hits = (
        active
        & (event.lane_addrs <= addr)
        & (addr < event.lane_addrs + event.elem_size)
    )
    if not np.any(hits):
        return (False, False)

    if event.read_mask is not None:
        reads = bool(np.any(event.read_mask[hits]))
    else:
        reads = event.access_mode == "read"

    if event.write_mask is not None:
        writes = bool(np.any(event.write_mask[hits]))
    else:
        writes = event.access_mode == "write"

    return reads, writes


def apply_rmw(op: str, old: np.ndarray, val: np.ndarray) -> np.ndarray:
    """Compute the new value an RMW writes. Result dtype == ``old.dtype``
    (guards against NumPy's silent promotion)."""
    if op not in VALID_RMW_OPS:
        raise NotImplementedError(f"Unsupported atomic_rmw op: {op!r}")
    if op == "add":
        result = old + val
    elif op == "xchg":
        result = np.asarray(val)
    elif op == "and":
        result = np.bitwise_and(old, val)
    elif op == "or":
        result = np.bitwise_or(old, val)
    elif op == "xor":
        result = np.bitwise_xor(old, val)
    elif op == "max":
        result = np.maximum(old, val)
    else:  # "min"
        result = np.minimum(old, val)
    return np.asarray(result).astype(old.dtype, copy=False)
