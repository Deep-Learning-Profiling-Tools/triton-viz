from contextlib import contextmanager

from abc import ABC, abstractmethod

from .data import Op, Launch
from .patch import patch_op, unpatch_op, op_list, patch_calls
from typing import Tuple, Callable, Type, Optional


class Client(ABC):
    name: str

    @abstractmethod
    def arg_callback(self, arg, arg_cvt):
        pass

    @abstractmethod
    def grid_callback(self, grid: Tuple[int]):
        pass

    @abstractmethod
    def grid_idx_callback(self, grid_idx: Tuple[int]):
        pass

    @abstractmethod
    def register_op_callback(self, op: Type[Op]) -> Tuple[Optional[Callable], Optional[Callable]]:
        pass

    @abstractmethod
    def finalize(self) -> list:
        pass


class ClientManager:
    def __init__(self, clients: Optional[list[Client]] = None):
        self.clients = clients if clients is not None else []
        self.launch = Launch()

    def add_clients(self, new_clients: Tuple[Client]) -> None:
        for new_client in new_clients:
            duplicate = any(
                isinstance(existing_client, new_client.__class__)
                for existing_client in self.clients
            )
            if not duplicate:
                self.clients.append(new_client)

    @contextmanager
    def patch(self):
        with patch_calls():
            for client in self.clients:
                for op in op_list:
                    before_callback, after_callback, op_overrider = client.register_op_callback(op)
                    patch_op(op, before_callback, after_callback, op_overrider)
            try:
                yield
            finally:
                for op in op_list:
                    unpatch_op(op)

    def finalize(self) -> None:
        self.launch.records = []
        for client in self.clients:
            self.launch.records += client.finalize()

    def arg_callback(self, arg, arg_cvt):
        if hasattr(arg, "data_ptr"):
            self.launch.tensors.append(arg)
        for client in self.clients:
            client.arg_callback(arg, arg_cvt)

    def grid_callback(self, grid: Tuple[int]):
        self.launch.grid = grid
        for client in self.clients:
            client.grid_callback(grid)

    def grid_idx_callback(self, grid_idx: Tuple[int]):
        for client in self.clients:
            client.grid_idx_callback(grid_idx)
