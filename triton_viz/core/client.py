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
    patch_calls,
)
from functools import wraps
from .callbacks import OpCallbacks, ForLoopCallbacks
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
            for client in self.clients.values():
                for namespace, attrs in namespaces.items():  # patch ops
                    for attr, op in attrs.items():
                        callbacks = client.register_op_callback(op)
                        patch_op(namespace, attr, callbacks, backend=backend)

                # patch for loops
                loop_callbacks = client.register_for_loop_callback()
                patch_for_loop(loop_callbacks)
            # Remaps core language functions to interpreted ones
            patch_lang(fn, backend)
            try:
                yield
            finally:
                for namespace, attrs in namespaces.items():
                    for attr, op in attrs.items():
                        unpatch_op(namespace, attr, backend)
                unpatch_for_loop()
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
