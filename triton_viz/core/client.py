from contextlib import contextmanager

from abc import ABC, abstractmethod
from typing import ClassVar, Optional, Any
from collections.abc import Callable

from .data import Op, Launch
from .patch import (
    patch_op,
    unpatch_op,
    patch_for_loop,
    unpatch_for_loop,
    op_list,
    patch_calls,
)
from functools import wraps
from .callbacks import OpCallbacks, ForLoopCallbacks
from .patch import patch_lang, unpatch_lang


class Client(ABC):
    NAME: ClassVar[str]

    def __init__(self):
        # Whether this client needs ASM information from kernel warmup
        self.collect_asm: bool = False
        # Storage for ASM information if collected
        self.asm_info: Optional[dict] = None

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
    def register_op_callback(self, op_type: type[Op]) -> OpCallbacks:
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


class ClientManager:
    def __init__(self, clients: Optional[list[Client]] = None):
        self.clients: dict[str, Client] = {}
        if clients:
            self.add_clients(clients)
        self.launch = Launch()

    def get_client(self, name: str) -> Optional[Client]:
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
    def patch_run(self, fn):
        with patch_calls():
            for client in self.clients.values():
                for op in op_list:
                    # patch ops

                    callbacks = client.register_op_callback(op)
                    patch_op(op, callbacks)

                # patch for loops
                loop_callbacks = client.register_for_loop_callback()
                patch_for_loop(loop_callbacks)
                # Remaps core language functions to interpreted ones
                patch_lang(fn)
            try:
                yield
            finally:
                for op in op_list:
                    unpatch_op(op)
                unpatch_for_loop()
                unpatch_lang()

    def pre_run_callback(self, fn: Callable) -> bool:
        rets = []
        for client in self.clients.values():
            rets.append(client.pre_run_callback(fn))
        return all(rets) if rets else True

    def post_run_callback(self, fn: Callable) -> bool:
        rets = []
        for client in self.clients.values():
            rets.append(client.post_run_callback(fn))
        return any(rets)

    def finalize(self) -> None:
        self.launch.records = []
        for client in self.clients.values():
            self.launch.records += client.finalize()

    def arg_callback(self, name, arg, arg_cvt):
        if hasattr(arg, "data_ptr"):
            self.launch.tensors.add(arg)
        for client in self.clients.values():
            client.arg_callback(name, arg, arg_cvt)

    def grid_callback(self, grid: tuple[int]):
        self.launch.grid = grid
        for client in self.clients.values():
            client.grid_callback(grid)

    def grid_idx_callback(self, grid_idx: tuple[int, ...]):
        for client in self.clients.values():
            client.grid_idx_callback(grid_idx)
