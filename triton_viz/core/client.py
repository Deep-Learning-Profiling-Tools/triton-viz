from contextlib import contextmanager

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import ClassVar

from .data import Op, Launch
from .patch import patch_op, unpatch_op, op_list, patch_calls


class Client(ABC):
    NAME: ClassVar[str]

    @abstractmethod
    def arg_callback(self, arg, arg_cvt):
        ...

    @abstractmethod
    def grid_callback(self, grid: tuple[int]):
        ...

    @abstractmethod
    def grid_idx_callback(self, grid_idx: tuple[int]):
        ...

    @abstractmethod
    def register_op_callback(
        self, op: type[Op]
    ) -> tuple[Callable | None, Callable | None, Callable | None]:
        ...

    @abstractmethod
    def finalize(self) -> list:
        ...


class ClientManager:
    def __init__(self, clients: list[Client] | None = None):
        self.clients: dict[str, Client] = {}
        if clients:
            self.add_clients(clients)
        self.launch = Launch()

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
    def patch(self):
        with patch_calls():
            for client in self.clients.values():
                for op in op_list:
                    (
                        before_callback,
                        after_callback,
                        op_overrider,
                    ) = client.register_op_callback(op)
                    patch_op(op, before_callback, after_callback, op_overrider)
            try:
                yield
            finally:
                for op in op_list:
                    unpatch_op(op)

    def finalize(self) -> None:
        self.launch.records = []
        for client in self.clients.values():
            self.launch.records += client.finalize()

    def arg_callback(self, arg, arg_cvt):
        if hasattr(arg, "data_ptr"):
            self.launch.tensors.append(arg)
        for client in self.clients.values():
            client.arg_callback(arg, arg_cvt)

    def grid_callback(self, grid: tuple[int]):
        self.launch.grid = grid
        for client in self.clients.values():
            client.grid_callback(grid)

    def grid_idx_callback(self, grid_idx: tuple[int]):
        for client in self.clients.values():
            client.grid_idx_callback(grid_idx)
