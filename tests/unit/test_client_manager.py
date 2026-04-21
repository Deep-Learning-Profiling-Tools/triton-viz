from triton_viz.core.callbacks import ForLoopCallbacks, OpCallbacks
from triton_viz.core.client import Client, ClientManager


class _DummyClient(Client):
    """Dummy client whose `self.tensors` comes from user input."""

    NAME = "dummy"

    def __init__(self, tensors=None, records=None):
        super().__init__()
        self.tensors = [] if tensors is None else list(tensors)
        self._records = [] if records is None else list(records)

    def pre_run_callback(self, fn):
        return True

    def post_run_callback(self, fn):
        return True

    def arg_callback(self, name, arg, arg_cvt):
        return None

    def grid_callback(self, grid):
        return None

    def grid_idx_callback(self, grid_idx):
        return None

    def register_op_callback(self, op_type, *args, **kwargs):
        return OpCallbacks()

    def register_for_loop_callback(self):
        return ForLoopCallbacks()

    def finalize(self):
        return list(self._records)

    def pre_warmup_callback(self, jit_fn, *args, **kwargs):
        return False

    def post_warmup_callback(self, jit_fn, ret):
        return None


def test_client_manager_finalize_collects_client_tensors():
    """Test that the client manager collects any internal tensors that a client stores tensors.
    Important for the tracer + visualizer as the tracer stores tensors allocated in kernel scope,
    which the visualizer then displays.
    """
    tensor = object()
    record = object()
    manager = ClientManager([_DummyClient(tensors=[tensor], records=[record])])

    manager.finalize()

    assert tensor in manager.launch.tensors
    assert manager.launch.records == [record]
