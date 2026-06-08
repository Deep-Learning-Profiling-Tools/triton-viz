from collections.abc import Callable
from contextlib import contextmanager
from typing import Any

from .frontend.base import AdapterResult, LANG_PATCH_SCOPES
from .frontend.base import get_frontend
from .callbacks import OpCallbacks
from .config import config as cfg
from .data import Op


def _frontend(frontend_name: str):
    """
    Purpose:
        Load a frontend object only when that frontend is used.

    Args:
        frontend_name: Frontend name to resolve.

    Returns:
        Frontend object implementing the registry and runtime patching interface.
    """
    return get_frontend(frontend_name)


class PatchOp:
    def __init__(
        self,
        op: Callable,
        op_type: type[Op],
        callbacks: OpCallbacks,
        adapter: Callable[..., AdapterResult],
        run_op_overrider: Callable,
        maybe_yield_for_multism: Callable[[], None] | None = None,
    ):
        self.op = op
        self.op_type = op_type
        self.before_callback = callbacks.before_callback
        self.after_callback = callbacks.after_callback
        self.op_overrider = callbacks.op_overrider
        self.adapter = adapter
        self.maybe_yield_for_multism = maybe_yield_for_multism
        self.run_op_overrider = run_op_overrider

    def __getattr__(self, name: str) -> Any:
        return getattr(self.op, name)

    def __call__(self, *args, **kwargs):
        if self.maybe_yield_for_multism is not None:
            self.maybe_yield_for_multism()

        if not self.before_callback and not self.after_callback:
            if self.op_overrider:
                return self.run_op_overrider(
                    self.op,
                    self.op_type,
                    self.op_overrider,
                    args,
                    kwargs,
                )
            return self.op(*args, **kwargs)

        if self.before_callback:
            before_args = self.adapter(*args, **kwargs)
            self.before_callback(*before_args.args, **before_args.kwargs)

        if self.op_overrider:
            ret = self.run_op_overrider(
                self.op,
                self.op_type,
                self.op_overrider,
                args,
                kwargs,
            )
        else:
            ret = self.op(*args, **kwargs)

        if self.after_callback:
            # Pass ret so that we don't have to derive output shape from args.
            after_args = self.adapter(*args, **kwargs)
            self.after_callback(ret, *after_args.args, **after_args.kwargs)
        return ret


def patch_op(namespace: Any, attr: str, callbacks: OpCallbacks, frontend_name: str):
    """
    Purpose:
        Replace one frontend operator with a PatchOp wrapper that invokes callbacks.

    Args:
        namespace: Namespace object that owns the operator.
        attr: Attribute name for the operator on the namespace.
        callbacks: Callback bundle for before, after, and override behavior.
        frontend_name: Frontend name that owns the operator registry entry.

    Returns:
        None.

    Raises:
        ValueError: If frontend_name is not registered.
    """
    frontend = _frontend(frontend_name)
    op_type = frontend.namespaces[namespace][attr]
    original_op = frontend.original_ops[namespace][attr]
    adapter = frontend.adapters[op_type]
    frontend.prepare_patched_op(namespace, op_type, original_op)
    maybe_yield_for_multism = (
        frontend.maybe_yield_for_multism if cfg.num_sms > 1 else None
    )
    patched_op = PatchOp(
        original_op,
        op_type,
        callbacks,
        adapter,
        run_op_overrider=frontend.run_op_overrider,
        maybe_yield_for_multism=maybe_yield_for_multism,
    )
    setattr(namespace, attr, patched_op)


def unpatch_op(namespace: Any, attr: str, frontend_name: str):
    """
    Purpose:
        Restore one previously patched frontend operator to its original function.

    Args:
        namespace: Namespace object that owns the operator.
        attr: Attribute name for the operator on the namespace.
        frontend_name: Frontend name that owns the operator registry entry.

    Returns:
        None.
    """
    frontend = _frontend(frontend_name)
    original_op = frontend.original_ops[namespace][attr]
    setattr(namespace, attr, original_op)


class LoopIter:
    """
    Purpose:
        Wrap an iterable so registered loop hooks run around each iteration.

    Args:
        hooks: Object that owns range_type, before_loop, loop_iter, and after_loop hooks.
        iterable: Iterable produced by the patched loop expression.
        lineno: Source line number for the loop.
        range_type: Frontend-specific classification of the loop iterable.

    Returns:
        Iterator that yields possibly overridden loop indices.
    """

    def __init__(self, hooks, iterable, lineno, range_type):
        self._it = iter(iterable)
        self._lineno = lineno
        self._hooks = hooks
        # triggering range_type
        self._hooks.range_type(self._lineno, range_type)
        # triggering before_loop
        if self._hooks.before_loop:
            self._hooks.before_loop(self._lineno, iterable)

    def __iter__(self):
        return self

    def __next__(self):
        idx = None
        try:
            idx = next(self._it)
        except StopIteration:
            # Exiting the loop and triggering after_loop
            if self._hooks.after_loop:
                self._hooks.after_loop(self._lineno)
            raise

        # trigger loop overriders and loop listeners
        idx = self._hooks.loop_iter(self._lineno, idx)
        return idx


def patch_for_loop(frontend_name: str = "triton"):
    """
    Purpose:
        Enable frontend-specific for-loop interception when supported.

    Args:
        frontend_name: Frontend name to patch.

    Returns:
        None.
    """
    _frontend(frontend_name).patch_for_loop()


def unpatch_for_loop(frontend_name: str = "triton"):
    """
    Purpose:
        Disable frontend-specific for-loop interception if it was enabled.

    Args:
        frontend_name: Frontend name to unpatch.

    Returns:
        None.
    """
    _frontend(frontend_name).unpatch_for_loop()


def patch_lang(fn, frontend_name, client_manager=None):
    """
    Purpose:
        Patch frontend language/runtime symbols for one traced function scope.

    Args:
        fn: Function whose execution scope needs frontend language patching.
        frontend_name: Frontend name to patch.
        client_manager: Optional active ClientManager exposed to patched helpers.

    Returns:
        None.

    Raises:
        ValueError: If frontend_name is unsupported.
    """
    scope = _frontend(frontend_name).patch_lang(
        fn,
        client_manager=client_manager,
    )

    LANG_PATCH_SCOPES.setdefault(frontend_name, []).append(scope)


def unpatch_lang(frontend_name):
    """
    Purpose:
        Restore the most recent language/runtime patch scope for a frontend.

    Args:
        frontend_name: Frontend name whose latest patch scope should be restored.

    Returns:
        None.
    """
    scopes = LANG_PATCH_SCOPES.get(frontend_name)
    scope = scopes.pop() if scopes else None
    if scope is not None and hasattr(scope, "restore"):
        scope.restore()


@contextmanager
def patch_calls(frontend_name):
    """
    Purpose:
        Patch frontend call entry points while a traced launch is active.

    Args:
        frontend_name: Frontend name whose call entry points should be patched.

    Returns:
        Context manager that restores call entry points when the scope exits.

    Yields:
        None. Control returns to the caller while frontend call patches are active.
    """
    with _frontend(frontend_name).patch_calls():
        yield
