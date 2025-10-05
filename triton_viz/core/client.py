from contextlib import contextmanager

from abc import ABC, abstractmethod
from typing import ClassVar, Optional
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
    def register_op_callback(self, op: type[Op]) -> OpCallbacks:
        ...

    @abstractmethod
    def register_for_loop_callback(self) -> ForLoopCallbacks:
        ...

    @abstractmethod
    def finalize(self) -> list:
        ...

    def set_asm_info(self, asm_info: dict):
        """
        Receive and store ASM information from kernel compilation.

        :param asm_info: Dictionary containing ASM code for different architectures
                        (e.g., {'amdgcn': '...', 'ptx': '...'})
        """
        self.asm_info = asm_info


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
    def patch(self, fn):
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
        return any(rets)

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

    def needs_asm(self) -> bool:
        """
        Check if any registered client needs ASM information.

        :return: True if at least one client has collect_asm=True
        """
        return any(client.collect_asm for client in self.clients.values())

    def distribute_asm_info(self, asm_info: dict):
        """
        Distribute ASM information to clients that need it.

        :param asm_info: Dictionary containing ASM code for different architectures
        """
        for client in self.clients.values():
            if client.collect_asm and hasattr(client, "set_asm_info"):
                client.set_asm_info(asm_info)
