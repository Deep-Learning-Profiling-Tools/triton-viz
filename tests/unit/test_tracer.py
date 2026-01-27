from unittest.mock import MagicMock

from triton_viz.clients.tracer.tracer import Tracer, _convert_grid_idx


# ======== _convert_grid_idx Tests ===========


def test_convert_grid_idx_none():
    assert _convert_grid_idx(None) is None


def test_convert_grid_idx_int():
    assert _convert_grid_idx(1) == (1, 0, 0)


def test_convert_grid_idx_tuple_1d():
    assert _convert_grid_idx((2,)) == (2, 0, 0)


def test_convert_grid_idx_tuple_2d():
    assert _convert_grid_idx((2, 3)) == (2, 3, 0)


def test_convert_grid_idx_tuple_3d():
    assert _convert_grid_idx((2, 3, 4)) == (2, 3, 4)


# ======== Tracer Init Tests ===========


def test_tracer_init_defaults():
    tracer = Tracer()
    assert tracer.callpath is True
    assert tracer.grid_idx is None
    assert tracer.records == []
    assert tracer.tensors == []
    assert tracer.sample is True


def test_tracer_init_with_grid_idx_int():
    tracer = Tracer(grid_idx=1)
    assert tracer.grid_idx == (1, 0, 0)


def test_tracer_init_with_grid_idx_tuple():
    tracer = Tracer(grid_idx=(2, 3))
    assert tracer.grid_idx == (2, 3, 0)


# ======== Tracer Callback Tests ===========


def test_tracer_pre_run_callback():
    tracer = Tracer()
    assert tracer.pre_run_callback(lambda: None) is True


def test_tracer_post_run_callback():
    tracer = Tracer()
    assert tracer.post_run_callback(lambda: None) is True


def test_tracer_pre_warmup_callback():
    tracer = Tracer()
    assert tracer.pre_warmup_callback(None) is False


def test_tracer_arg_callback_with_tensor():
    tracer = Tracer()
    mock_tensor = MagicMock()
    mock_tensor.data_ptr = MagicMock(return_value=12345)
    tracer.arg_callback("x", mock_tensor, None)
    assert mock_tensor in tracer.tensors


def test_tracer_arg_callback_without_tensor():
    tracer = Tracer()
    non_tensor = 42
    tracer.arg_callback("x", non_tensor, None)
    assert tracer.tensors == []


def test_tracer_grid_idx_callback_sampling_match():
    tracer = Tracer(grid_idx=(1, 0, 0))
    tracer.grid_idx_callback((1, 0, 0))
    assert tracer.sample is True


def test_tracer_grid_idx_callback_sampling_no_match():
    tracer = Tracer(grid_idx=(1, 0, 0))
    tracer.grid_idx_callback((2, 0, 0))
    assert tracer.sample is False


def test_tracer_grid_idx_callback_no_filter():
    tracer = Tracer()
    tracer.grid_idx_callback((0, 0, 0))
    assert tracer.sample is True
    tracer.grid_idx_callback((5, 3, 2))
    assert tracer.sample is True


# ======== Tracer _get_tensor Tests ===========


def test_tracer_get_tensor():
    tracer = Tracer()

    mock_tensor1 = MagicMock()
    mock_tensor1.data_ptr = MagicMock(return_value=1000)
    mock_tensor2 = MagicMock()
    mock_tensor2.data_ptr = MagicMock(return_value=2000)
    mock_tensor3 = MagicMock()
    mock_tensor3.data_ptr = MagicMock(return_value=3000)

    tracer.tensors = [mock_tensor1, mock_tensor2, mock_tensor3]

    assert tracer._get_tensor(1500) == mock_tensor1
    assert tracer._get_tensor(2000) == mock_tensor2
    assert tracer._get_tensor(2500) == mock_tensor2
    assert tracer._get_tensor(3500) == mock_tensor3
