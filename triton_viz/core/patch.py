from collections.abc import Callable
from contextlib import contextmanager
from functools import wraps
from typing import Any

from .frontend.base import AdapterResult, LANG_PATCH_SCOPES
from .frontend.base import get_frontend
from .callbacks import OpCallbacks
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
        maybe_yield_for_multism: Callable[[], None],
    ):
        self.op = op
        self.op_type = op_type
        # Copy callback fields once at patch time; patched ops are called for
        # every frontend operation, so avoid repeated bundle attribute lookups.
        self.before_callback = callbacks.before_callback
        self.after_callback = callbacks.after_callback
        self.op_overrider = callbacks.op_overrider
        self.adapter = adapter
        # Frontend hooks are resolved before installing the wrapper so __call__
        # stays independent of frontend registry lookups.
        self.maybe_yield_for_multism = maybe_yield_for_multism
        self.run_op_overrider = run_op_overrider

    def __getattr__(self, name: str) -> Any:
        # Module-level patched ops are PatchOp instances; preserve access to
        # attributes exposed by the original callable.
        return getattr(self.op, name)

    def __get__(self, instance: Any, owner: type | None = None) -> Any:
        # PatchOp can be installed on classes, e.g. Gluon shared-memory
        # descriptor methods. Preserve normal method binding so self is passed.
        if instance is None:
            return self

        @wraps(self.op)
        def bound(*args: Any, **kwargs: Any) -> Any:
            return self(instance, *args, **kwargs)

        return bound

    def __call__(self, *args, **kwargs):
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
    patched_op = PatchOp(
        original_op,
        op_type,
        callbacks,
        adapter,
        run_op_overrider=frontend.run_op_overrider,
        maybe_yield_for_multism=frontend.maybe_yield_for_multism,
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
        hooks: Object that owns range_type, before_loop, loop_iter,
            after_loop, and abandoned_loop hooks.
        iterable: Iterable produced by the patched loop expression.
        lineno: Source line number for the loop.
        range_type: Frontend-specific classification of the loop iterable.

    Returns:
        Iterator that yields possibly overridden loop indices.

    The for-loop rewrite executes the loop inside a ``with`` block over this
    object, giving early exits (break / return / an exception in the loop
    body) a deterministic teardown point: ``abandoned_loop`` fires exactly
    when the iterable was NOT exhausted. The hook owner decides policy.
    """

    def __init__(self, hooks, iterable, lineno, range_type):
        self._it = iter(iterable)
        self._lineno = lineno
        self._hooks = hooks
        self._exhausted = False
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
            self._exhausted = True
            if self._hooks.after_loop:
                self._hooks.after_loop(self._lineno)
            raise

        # trigger loop overriders and loop listeners
        idx = self._hooks.loop_iter(self._lineno, idx)
        return idx

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if not self._exhausted and self._hooks.abandoned_loop:
            self._hooks.abandoned_loop(self._lineno, exc_type)
        return False


class PassthroughLoopIter:
    """Loop wrapper used when no client manager is active.

    Preserves the original iterable's behavior while still satisfying the
    ``with`` protocol emitted by the for-loop rewrite.
    """

    __slots__ = ("_iterable",)

    def __init__(self, iterable):
        self._iterable = iterable

    def __iter__(self):
        return iter(self._iterable)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False


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
