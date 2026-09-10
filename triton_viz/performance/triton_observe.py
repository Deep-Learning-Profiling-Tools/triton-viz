"""Observe Triton through CPU interpretation, without target compilation.

The trace is concrete: use representative inputs for data-dependent branches.
Every launched program is observed; there is no implicit grid extrapolation.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from triton_viz.core.callbacks import ForLoopCallbacks, OpCallbacks
from triton_viz.core.client import Client


class PerformanceTrace(Client):
    NAME = "performance_trace"

    def __init__(self):
        super().__init__()
        self.events: list[dict[str, Any]] = []
        self.grid: tuple[int, ...] = ()
        self._values: dict[int, int] = {}
        self._keepalive: list[Any] = []

    def pre_run_callback(self, fn):
        return True

    def post_run_callback(self, fn):
        return True

    def pre_warmup_callback(self, jit_fn, *args, **kwargs):
        return False

    def post_warmup_callback(self, jit_fn, ret):
        pass

    def arg_callback(self, name, arg, arg_cvt):
        pass

    def grid_callback(self, grid):
        self.grid = tuple(int(v) for v in grid)

    def grid_idx_callback(self, grid_idx):
        self.grid_idx = tuple(int(v) for v in grid_idx)

    def register_for_loop_callback(self):
        return ForLoopCallbacks()

    def finalize(self):
        self._keepalive.clear()
        return self.events

    @staticmethod
    def _array(value):
        value = getattr(value, "handle", value)
        data = getattr(value, "data", None)
        return data if isinstance(data, np.ndarray) else None

    def register_op_callback(self, op_type):
        name = op_type.name

        def before(*args, **kwargs):
            stack = self._get_thread_local("raw_memory_stack", [])
            stack.append(False)
            self._set_thread_local("raw_memory_stack", stack)

        @self.lock_fn
        def after(ret, *args, **kwargs):
            stack = self._get_thread_local("raw_memory_stack", [])
            if name in {"raw_load", "raw_store"}:
                # Triton's raw operations delegate to masked operations in
                # some versions. Count the underlying transfer exactly once,
                # while still supporting versions with standalone raw ops.
                if stack.pop():
                    return
            elif name in {"load", "store"} and stack:
                stack[-1] = True
            output = self._array(ret)
            arrays = [(arg, self._array(arg)) for arg in args]
            arrays = [(arg, arr) for arg, arr in arrays if arr is not None]
            seq = len(self.events)
            deps = sorted(
                {self._values[id(arr)] for _, arr in arrays if id(arr) in self._values}
            )
            event = {
                "seq": seq,
                "op": name,
                "program": list(self.grid_idx or ()),
                "dependencies": deps,
                "shape": list(output.shape) if output is not None else [],
                "dtype": str(
                    getattr(ret, "dtype", output.dtype if output is not None else "")
                ),
                "input_shapes": [list(arr.shape) for _, arr in arrays],
                "elements": int(output.size) if output is not None else 0,
            }
            if name in {"binary_op", "unary_op"}:
                function = next((arg for arg in args if callable(arg)), None)
                event["primitive"] = getattr(function, "__name__", "unknown")
            if name in {"load", "raw_load", "store", "raw_store"}:
                ptr = self._array(args[0])
                # Memory width comes from the pointer, including masked stores
                # whose frontend adapter does not expose the stored value.
                mask = (
                    self._array(args[1])
                    if name in {"load", "store"} and len(args) > 1
                    else None
                )
                active = (
                    np.ones(ptr.shape, dtype=bool)
                    if mask is None
                    else np.broadcast_to(mask, ptr.shape)
                )
                dtype = getattr(getattr(args[0], "dtype", None), "element_ty", None)
                # BF16 interpreter storage may be float32; semantic width wins.
                width = getattr(dtype, "primitive_bitwidth", None)
                if not width:
                    raise ValueError("Cannot determine semantic memory element width")
                item_bytes = max(1, int(width // 8))
                addresses = ptr[active].astype(np.uint64)
                sectors = np.unique(addresses // 32).size
                if item_bytes > 1 and addresses.size:
                    sectors = np.unique(
                        np.concatenate(
                            (addresses // 32, (addresses + item_bytes - 1) // 32)
                        )
                    ).size
                event.update(
                    bytes=int(active.sum()) * item_bytes,
                    sectors=int(sectors),
                    item_bytes=item_bytes,
                    elements=int(ptr.size),
                    masked=not bool(np.all(active)),
                )
            self.events.append(event)
            if output is not None:
                self._values[id(output)] = seq
                self._keepalive.append(output)

        return OpCallbacks(
            before_callback=before if name in {"raw_load", "raw_store"} else None,
            after_callback=after,
        )


def observe(kernel, grid, *args, num_warps=4, num_stages=2, **kwargs):
    """Interpret a fixed Triton launch on CPU and return portable source facts.

    Pass CPU tensors. Autotuners must be resolved to a fixed configuration first.
    Source interpretation executes stores into those CPU tensors.
    """
    import triton_viz

    if any(getattr(arg, "is_cuda", False) for arg in (*args, *kwargs.values())):
        raise ValueError("Observe requires CPU inputs, independent of target execution")
    if hasattr(kernel, "configs"):
        raise ValueError("Resolve autotuning before pre-compile prediction")
    trace = PerformanceTrace()
    triton_viz.trace(trace)(kernel)[grid](
        *args, num_warps=num_warps, num_stages=num_stages, **kwargs
    )
    return {
        "schema": "triton-viz.gpu-source.v1",
        "grid": list(trace.grid),
        "program_count": math.prod(trace.grid),
        "num_warps": num_warps,
        "num_stages": num_stages,
        "events": trace.events,
    }
