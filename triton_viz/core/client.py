from contextlib import contextmanager, nullcontext

from abc import ABC, abstractmethod
from typing import ClassVar, Any
from collections.abc import Callable
import threading

from .data import Op, Launch
from .patch import (
    patch_op,
    unpatch_op,
    patch_for_loop,
    unpatch_for_loop,
    patch_if_else,
    unpatch_if_else,
    patch_calls,
    LoopIter,
)
from functools import wraps
from .callbacks import OpCallbacks, ForLoopCallbacks, IfElseCallbacks
from .patch import patch_lang, unpatch_lang, OPERATION_REGISTRY
from .config import config as cfg


class Client(ABC):
    NAME: ClassVar[str]

    def __init__(self) -> None:
        # Whether this client needs ASM information from kernel warmup
        self.collect_asm: bool = False
        # Storage for ASM information if collected
        self.asm_info: dict | None = None
        # Thread-local scratch space for per-thread callback state
        self._thread_local = threading.local()
        # Lock for serializing shared state where needed
        self._lock = threading.RLock()

    def _lock_context(self):
        if cfg.num_sms > 1:
            return self._lock
        return nullcontext()

    def lock_fn(self, fn: Callable) -> Callable:
        """Forces serial execution of the given function."""

        @wraps(fn)
        def wrapped(*args, **kwargs):
            with self._lock_context():
                return fn(*args, **kwargs)

        return wrapped

    @abstractmethod
    def pre_run_callback(self, fn: Callable) -> bool:
        """
        Returns True if the function should continue running, False if it should be skipped.
        """
        ...

    @abstractmethod
    def post_run_callback(self, fn: Callable) -> bool:
        """
        Returns True if the function should continue running, False if it should be skipped.
        """
        ...

    @abstractmethod
    def arg_callback(self, name, arg, arg_cvt):
        ...

    @abstractmethod
    def grid_callback(self, grid: tuple[int, ...]):
        ...

    @abstractmethod
    def grid_idx_callback(self, grid_idx: tuple[int, ...]):
        ...

    @abstractmethod
    def register_op_callback(
        self, op_type: type[Op], *args: Any, **kwargs: Any
    ) -> OpCallbacks:
        ...

    @abstractmethod
    def register_for_loop_callback(self) -> ForLoopCallbacks:
        ...

    def register_if_else_callback(self) -> IfElseCallbacks:
        return IfElseCallbacks()

    @abstractmethod
    def finalize(self) -> list:
        ...

    @abstractmethod
    def pre_warmup_callback(self, jit_fn: Callable, *args, **kwargs) -> bool:
        """
        Returns True if the warmup should proceed, False to skip warmup.
        """
        ...

    @abstractmethod
    def post_warmup_callback(self, jit_fn: Callable, ret: Any) -> None:
        ...

    def _set_thread_local(self, key: str, value: Any) -> None:
        setattr(self._thread_local, key, value)

    def _get_thread_local(self, key: str, default: Any = None) -> Any:
        return getattr(self._thread_local, key, default)

    @property
    def grid_idx(self) -> tuple[int, ...] | None:
        return self._get_thread_local("grid_idx", None)

    @grid_idx.setter
    def grid_idx(self, value: tuple[int, ...] | None) -> None:
        self._set_thread_local("grid_idx", value)


class ClientManager:
    def __init__(self, clients: list[Client] | None = None):
        self.clients: dict[str, Client] = {}
        if clients:
            self.add_clients(clients)
        self.launch = Launch()
        self._lock = threading.Lock()
        self._clear_loop_hooks()
        self._clear_if_hooks()
        self._if_patching_active: bool = False

    def _lock_context(self):
        if cfg.num_sms > 1:
            return self._lock
        return nullcontext()

    def get_client(self, name: str) -> Client | None:
        return self.clients.get(name)

    def add_clients(self, new_clients_list: list[Client]) -> None:
        for new_client in new_clients_list:
            duplicate = any(
                isinstance(existing_client, new_client.__class__)
                for existing_client in self.clients.values()
            )
            if not duplicate:
                self.clients[new_client.NAME] = new_client

    @contextmanager
    def patch_warmup(self, jit_fn):
        if not hasattr(jit_fn, "warmup"):
            yield
            return

        def patcher(fn):
            @wraps(fn)
            def wrapped(*args, **kwargs):
                if all(
                    not client.pre_warmup_callback(jit_fn, *args, **kwargs)
                    for client in self.clients.values()
                ):
                    return None
                kwargs.pop("warmup", None)
                ret = fn(*args, **kwargs)
                for client in self.clients.values():
                    client.post_warmup_callback(jit_fn, ret)
                return ret

            return wrapped

        jit_fn.warmup = patcher(jit_fn.warmup)
        try:
            yield
        finally:
            jit_fn.warmup = jit_fn.warmup.__wrapped__

    @contextmanager
    def patch_run(self, fn, backend: str):
        namespaces = OPERATION_REGISTRY[backend].namespaces
        with patch_calls(backend):
            # Collect all for-loop callbacks from clients
            all_loop_callbacks = []
            all_if_callbacks = []
            for client in self.clients.values():
                for namespace, attrs in namespaces.items():  # patch ops
                    for attr, op in attrs.items():
                        callbacks = client.register_op_callback(op)
                        patch_op(namespace, attr, callbacks, backend=backend)
                all_loop_callbacks.append(client.register_for_loop_callback())
                all_if_callbacks.append(client.register_if_else_callback())

            self._populate_loop_hooks(all_loop_callbacks)
            patch_for_loop()

            # Only enable if/else patching if at least one client provides
            # an eval_condition_callback. Without a provider, the AST rewrite
            # would have no way to determine the condition's truth value.
            self._populate_if_hooks(all_if_callbacks)
            self._if_patching_active = self._eval_condition_callback is not None
            if self._if_patching_active:
                patch_if_else()

            patch_lang(fn, backend, client_manager=self)
            try:
                yield
            finally:
                for namespace, attrs in namespaces.items():
                    for attr, op in attrs.items():
                        unpatch_op(namespace, attr, backend)
                unpatch_for_loop()
                self._clear_loop_hooks()
                if self._if_patching_active:
                    unpatch_if_else()
                    self._if_patching_active = False
                self._clear_if_hooks()
                unpatch_lang(backend)

    def pre_run_callback(self, fn: Callable) -> bool:
        with self._lock_context():
            rets = []
            for client in self.clients.values():
                rets.append(client.pre_run_callback(fn))
            return all(rets) if rets else True

    def post_run_callback(self, fn: Callable) -> bool:
        with self._lock_context():
            rets = []
            for client in self.clients.values():
                rets.append(client.post_run_callback(fn))
            return any(rets)

    def finalize(self) -> None:
        with self._lock_context():
            self.launch.records = []
            for client in self.clients.values():
                self.launch.records += client.finalize()

    def arg_callback(self, name, arg, arg_cvt):
        with self._lock_context():
            if hasattr(arg, "data_ptr"):
                self.launch.tensors.add(arg)
            for client in self.clients.values():
                client.arg_callback(name, arg, arg_cvt)

    def grid_callback(self, grid: tuple[int]):
        with self._lock_context():
            self.launch.grid = grid
            for client in self.clients.values():
                client.grid_callback(grid)

    def grid_idx_callback(self, grid_idx: tuple[int, ...]):
        with self._lock_context():
            for client in self.clients.values():
                client.grid_idx_callback(grid_idx)

    # --- For-loop callback management ---

    def _clear_loop_hooks(self) -> None:
        self._range_type_hooks: list[Callable] = []
        self._before: list[Callable] = []
        self._iter_listeners: list[Callable] = []
        self._iter_overrider: Callable | None = None
        self._range_wrapper_factory: Callable | None = None
        self._after: list[Callable] = []

    def _populate_loop_hooks(self, callbacks_list: list[ForLoopCallbacks]) -> None:
        self._clear_loop_hooks()
        for cb in callbacks_list:
            if cb.range_type_callback is not None:
                self._range_type_hooks.append(cb.range_type_callback)
            if cb.before_loop_callback is not None:
                self._before.append(cb.before_loop_callback)
            if cb.loop_iter_listener is not None:
                self._iter_listeners.append(cb.loop_iter_listener)
            if cb.loop_iter_overrider is not None:
                if self._iter_overrider is not None:
                    raise RuntimeError("Only one loop_iter overrider allowed")
                self._iter_overrider = cb.loop_iter_overrider
            if cb.range_wrapper_factory is not None:
                if self._range_wrapper_factory is not None:
                    raise RuntimeError("Only one range_wrapper_factory allowed")
                self._range_wrapper_factory = cb.range_wrapper_factory
            if cb.after_loop_callback is not None:
                self._after.append(cb.after_loop_callback)

    def range_type(self, lineno: int, range_type: str) -> None:
        for hook in self._range_type_hooks:
            hook(lineno, range_type)

    def before_loop(self, lineno: int, iterable: Any) -> None:
        for hook in self._before:
            hook(lineno, iterable)

    def loop_iter(self, lineno: int, idx: Any) -> Any:
        if self._iter_overrider is not None:
            new_idx = self._iter_overrider(lineno, idx)
            if new_idx is not None:
                idx = new_idx

        for hook in self._iter_listeners:
            hook(lineno, idx)

        return idx

    def after_loop(self, lineno: int) -> None:
        for hook in self._after:
            hook(lineno)

    def loop_iter_wrapper(
        self,
        iterable_callable: Callable,
        iter_args,
        iter_kwargs,
        lineno: int,
        range_type: str,
    ) -> "LoopIter":
        args = tuple(iter_args) if iter_args is not None else ()
        kwargs = dict(iter_kwargs) if iter_kwargs is not None else {}

        if self._range_wrapper_factory is not None:
            wrapped = self._range_wrapper_factory(
                None, lineno, range_type, args, kwargs, iterable_callable
            )
            if wrapped is not None:
                iterable = wrapped
            else:
                iterable = iterable_callable(*args, **kwargs)
        else:
            iterable = iterable_callable(*args, **kwargs)
        return LoopIter(self, iterable, lineno, range_type)

    # --- If/else callback management ---

    def _clear_if_hooks(self) -> None:
        self._pre_if_hooks: list[Callable] = []
        self._eval_condition_callback: Callable | None = None
        self._flip_condition_hooks: list[Callable] = []
        self._post_if_hooks: list[Callable] = []

    def _populate_if_hooks(self, callbacks_list: list[IfElseCallbacks]) -> None:
        self._clear_if_hooks()
        for cb in callbacks_list:
            if cb.pre_if_callback is not None:
                self._pre_if_hooks.append(cb.pre_if_callback)
            if cb.eval_condition_callback is not None:
                if self._eval_condition_callback is not None:
                    raise RuntimeError("Only one eval_condition_callback allowed")
                self._eval_condition_callback = cb.eval_condition_callback
            if cb.flip_condition_callback is not None:
                self._flip_condition_hooks.append(cb.flip_condition_callback)
            if cb.post_if_callback is not None:
                self._post_if_hooks.append(cb.post_if_callback)

    def pre_if(self, condition: Any, lineno: int) -> None:
        for hook in self._pre_if_hooks:
            hook(condition, lineno)

    def eval_condition(self, lineno: int) -> bool:
        if self._eval_condition_callback is not None:
            return self._eval_condition_callback(lineno)
        return True

    def flip_condition(self, lineno: int) -> None:
        for hook in self._flip_condition_hooks:
            hook(lineno)

    def post_if(self, lineno: int) -> None:
        for hook in self._post_if_hooks:
            hook(lineno)
