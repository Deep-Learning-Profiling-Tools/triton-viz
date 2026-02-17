from collections.abc import Callable
from typing import Any

import triton.language as tl

from .data import Flip


def patch_flip(scope: Any, get_trace_runtime: Callable[[], Any]) -> None:
    # Wrap tl.flip to emit a Flip record after computing result.
    try:
        _orig_flip = getattr(tl, "flip", None)
        if _orig_flip is None or getattr(_orig_flip, "__triton_viz_wrapped__", False):
            return

        def _viz_flip(x, *args, **kwargs):
            # Call original flip implementation
            ret = _orig_flip(x, *args, **kwargs)
            # Best-effort extract dim
            dim = None
            if args:
                dim = args[0]
            if "dim" in kwargs:
                dim = kwargs.get("dim")
            # Best-effort shapes
            in_shape = None
            out_shape = None
            x_arr = None
            r_arr = None
            try:
                # interpreter tensors may expose .data or .handle.data
                x_data = getattr(x, "data", None)
                if x_data is None and hasattr(x, "handle"):
                    x_data = getattr(x.handle, "data", None)
                if x_data is not None:
                    in_shape = tuple(x_data.shape)
                    x_arr = x_data
            except Exception:
                pass
            try:
                r_data = getattr(ret, "data", None)
                if r_data is None and hasattr(ret, "handle"):
                    r_data = getattr(ret.handle, "data", None)
                if r_data is not None:
                    out_shape = tuple(r_data.shape)
                    r_arr = r_data
            except Exception:
                pass

            # Emit a Flip record to the active tracer, if available
            try:
                cm = get_trace_runtime()
                if cm is not None and hasattr(cm, "get_client"):
                    tracer = cm.get_client("tracer")
                    if tracer is not None:
                        input_payload = None
                        output_payload = None
                        try:
                            # Avoid huge payloads: cap to 64k elements
                            def _maybe_list(arr):
                                import numpy as _np

                                if arr is None:
                                    return None
                                try:
                                    if arr.size <= 65536:
                                        return _np.asarray(arr).tolist()
                                except Exception:
                                    pass
                                return None

                            input_payload = _maybe_list(x_arr)
                            output_payload = _maybe_list(r_arr)
                        except Exception:
                            pass

                        rec = Flip(
                            input_shape=in_shape or tuple(),
                            output_shape=out_shape or (in_shape or tuple()),
                            dim=int(dim) if dim is not None else 0,
                            input_data=input_payload,
                            output_data=output_payload,
                        )
                        # attach call path already handled by Flip.__post_init__
                        tracer.records.append(rec)
            except Exception:
                # Never fail kernel execution due to viz
                pass
            return ret

        # mark wrapper to avoid double-wrapping on subsequent patch_lang calls
        setattr(_viz_flip, "__triton_viz_wrapped__", True)
        if hasattr(scope, "set_attr"):
            scope.set_attr(tl, "flip", _viz_flip)
        else:
            tl.flip = _viz_flip  # type: ignore[assignment]
    except Exception:
        # If wrapping fails, continue without Flip records
        pass
