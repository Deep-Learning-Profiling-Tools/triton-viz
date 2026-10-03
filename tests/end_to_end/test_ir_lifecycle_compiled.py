"""End-to-end tests of the core IR lifecycle on real kernels: IR clients receive
kernels compiled on the host (D25) through ClientManager.ir_capture, with or
without the real launch, across plain, @heuristics and @autotune∘@heuristics
kernels. Counterparts on a fake compile live in tests/unit/test_ir_lifecycle.py.

Only what needs a device runs on one: a real launch (an IR client declaring
LAUNCH="run"), device-memory accounting, and an interpreting client's voted
warmup (the JIT's own compile). Everything else runs on CPU tensors with
Triton's driver unreachable (``_no_driver``), as on a machine without a GPU.
"""

import importlib
import types

import pytest
import torch
import triton
import triton.language as tl
from triton.backends.compiler import GPUTarget
from triton.compiler import CompiledKernel
from triton.compiler.errors import CompilationError, CompileTimeAssertionFailure

import tilelens
from tilelens.core.callbacks import ForLoopCallbacks, OpCallbacks
from tilelens.core.client import Client
from tilelens.core.config import DEFAULT_IR_TARGET
from tilelens.core.data import Store
from tilelens.core.host_compile import HostCompiler, HostKernel

# `tilelens.core.trace` the attribute is the trace() decorator; the module
# holds the `launches` list.
trace_module = importlib.import_module("tilelens.core.trace")
config_module = importlib.import_module("tilelens.core.config")


def _real_compiles_available() -> bool:
    # Triton imported under TRITON_INTERPRET=1 builds its own standard library
    # as InterpretedFunctions, so nothing can compile for real in-process.
    import triton.language.standard as tl_standard
    from triton.runtime.jit import JITFunction

    return isinstance(tl_standard.cdiv, JITFunction)


pytestmark = pytest.mark.skipif(
    not _real_compiles_available(),
    reason="Triton was imported under TRITON_INTERPRET=1: nothing compiles in-process",
)

GPU_REASON = "a real launch (or the JIT's own compile) needs a CUDA GPU"
# Marks what needs a device; everything else runs without Triton's driver.
needs_gpu = pytest.mark.skipif(not torch.cuda.is_available(), reason=GPU_REASON)
# The default IR target (D26, amended), and one passed explicitly.
CUDA89 = GPUTarget("cuda", 89, 32)
CUDA80 = GPUTarget("cuda", 80, 32)


@pytest.fixture(autouse=True)
def _no_driver(request, unreachable_driver):
    """Unless the test is marked needs_gpu: Triton's driver is unreachable,
    as on a machine without a GPU, so any driver query on the IR path fails
    the test (D25)."""
    if any(
        mark.kwargs.get("reason") == GPU_REASON
        for mark in request.node.iter_markers("skipif")
    ):
        return
    unreachable_driver("the IR path queried Triton's driver")


@pytest.fixture(autouse=True)
def _default_ir_target(monkeypatch):
    """The default IR target (D26), whatever TILELENS_IR_TARGET the caller
    set: in the process config, and in any Config read from the environment."""
    for name in ("TILELENS_IR_TARGET", "TRITON_VIZ_IR_TARGET"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(config_module.config, "ir_target", DEFAULT_IR_TARGET)


@pytest.fixture(autouse=True)
def _real_jit(monkeypatch):
    # tests/unit/test_multithreading.py sets TRITON_INTERPRET=1 at import time,
    # and a traced launch's patch scope restores knobs.runtime.interpret as an
    # explicit override. These tests need @triton.jit to build real
    # JITFunctions, so pin the knob off and put back exactly what was there.
    from triton import knobs

    monkeypatch.delenv("TRITON_INTERPRET", raising=False)
    missing = object()
    previous = knobs.runtime.__dict__.get("interpret", missing)
    knobs.runtime.__dict__["interpret"] = False
    yield
    if previous is missing:
        knobs.runtime.__dict__.pop("interpret", None)
    else:
        knobs.runtime.__dict__["interpret"] = previous


class _IRClient(Client):
    NEEDS_INTERPRETER = False
    IR_STAGES = frozenset({"ttir"})

    def __init__(self):
        super().__init__()
        self.log: list = []
        self.events: list = []
        self.failures: list = []
        self.finalized: list = []

    def begin_launch(self, call):
        self.log.append("begin")
        self.events = []
        self.failures = []

    def abort_launch(self, exc):
        self.log.append("abort")

    def before_launch(self, event):
        self.log.append("before")
        self.events.append(event)

    def after_launch(self, event):
        self.log.append("after")

    def compile_failed(self, event):
        self.log.append("compile_failed")
        self.failures.append(event)

    def finalize(self):
        self.log.append("finalize")
        self.finalized.append(list(self.events))
        return []

    def pre_warmup_callback(self, jit_fn, *args, **kwargs):
        return False

    def post_warmup_callback(self, jit_fn, ret):
        pass

    def _unreachable(self, *args, **kwargs):
        raise AssertionError(f"interpreter hook reached IR client {self.NAME}")

    pre_run_callback = _unreachable
    post_run_callback = _unreachable
    arg_callback = _unreachable
    grid_callback = _unreachable
    grid_idx_callback = _unreachable
    register_op_callback = _unreachable
    register_for_loop_callback = _unreachable


class _SkipIRClient(_IRClient):
    NAME = "ir_skip"
    LAUNCH = "skip"


class _RunIRClient(_IRClient):
    NAME = "ir_run"
    LAUNCH = "run"


class _EagerCounter(Client):
    """Interpreting client counting stores; optionally votes for a warmup."""

    NAME = "eager_counter"

    def __init__(self, warmup_vote=False):
        super().__init__()
        self.stores = 0
        self.warmup_vote = warmup_vote
        self.warmups: list = []

    def _on_store(self, *args, **kwargs):
        self.stores += 1

    def pre_run_callback(self, fn):
        return True

    def post_run_callback(self, fn):
        return True

    def arg_callback(self, name, arg, arg_cvt):
        pass

    def grid_callback(self, grid):
        pass

    def grid_idx_callback(self, grid_idx):
        pass

    def register_op_callback(self, op_type, *args, **kwargs):
        if op_type is Store:
            return OpCallbacks(before_callback=self._on_store)
        return OpCallbacks()

    def register_for_loop_callback(self):
        return ForLoopCallbacks()

    def finalize(self):
        return []

    def pre_warmup_callback(self, jit_fn, *args, **kwargs):
        return self.warmup_vote

    def post_warmup_callback(self, jit_fn, ret):
        self.warmups.append(ret)


def _make_add_one():
    @triton.jit
    def add_one(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
        offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n
        tl.store(out_ptr + offs, tl.load(x_ptr + offs, mask=mask) + 1, mask=mask)

    return add_one


def _make_autotuned(**autotune_kwargs):
    @triton.autotune(
        configs=[
            triton.Config({"BLOCK": 16}, num_warps=1),
            triton.Config({"BLOCK": 32}, num_warps=2),
        ],
        key=["n"],
        **autotune_kwargs,
    )
    @triton.heuristics({"EVEN": lambda args: args["n"] % args["BLOCK"] == 0})
    @triton.jit
    def add_one_tuned(x_ptr, out_ptr, n, BLOCK: tl.constexpr, EVEN: tl.constexpr):
        offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        if EVEN:
            tl.store(out_ptr + offs, tl.load(x_ptr + offs) + 1)
        else:
            mask = offs < n
            tl.store(out_ptr + offs, tl.load(x_ptr + offs, mask=mask) + 1, mask=mask)

    return add_one_tuned


def _grid(meta):
    return (triton.cdiv(meta["n"], meta["BLOCK"]),)


def _grid64(meta):
    # The interpreter hands grid callables tensor-converted runtime args, so
    # interpreted launches may only read constexprs here.
    return (triton.cdiv(64, meta["BLOCK"]),)


def _device(ir_cls) -> str:
    # A real launch needs device tensors; a compile does not.
    return "cuda" if ir_cls.LAUNCH == "run" else "cpu"


def _inputs(n=64, device="cpu"):
    x = torch.arange(n, dtype=torch.float32, device=device)
    return x, torch.zeros_like(x)


def _synchronize():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


# Each IR client class, the one that launches for real only with a GPU.
IR_CLASSES = [_SkipIRClient, pytest.param(_RunIRClient, marks=needs_gpu)]


def test_ir_only_skip_compiles_on_the_host_without_launching():
    ir = _SkipIRClient()
    kernel = _make_add_one()
    hooks = []
    kernel.add_pre_run_hook(lambda *args, **kwargs: hooks.append(1))
    traced = tilelens.trace(ir)(kernel)
    x, out = _inputs()

    ret = traced[(4,)](x, out, 64, BLOCK=16)

    assert torch.equal(out, torch.zeros_like(x))
    assert ir.log == ["begin", "before", "after", "finalize"]
    (events,) = ir.finalized
    (event,) = events
    assert event.launched is False
    # Compiled on the host for the default target, through TTIR only.
    assert isinstance(event.kernel, HostKernel)
    assert event.target == event.kernel.target == CUDA89
    assert list(event.kernel.asm) == ["ttir"]
    assert "tt.func" in event.kernel.asm["ttir"]
    assert event.resolved_grid == (4, 1, 1)
    assert event.specialization == event.kernel.hash
    assert "run" not in vars(traced.jit_fn)
    # JITFunction.run was never entered: no pre_run_hook fired.
    assert hooks == []
    # One config: the launch returns its kernel, as the untraced launch
    # would, and the Launch carries the grid, but no tensor (D23).
    assert ret is event.kernel
    launch = trace_module.launches[-1]
    assert launch.grid == (4, 1, 1)
    assert not launch.tensors


def test_a_relaunch_reuses_the_traces_host_compile(monkeypatch):
    compiles = []
    compile = HostCompiler._compile_source

    def counting(*args, **kwargs):
        compiles.append(1)
        return compile(*args, **kwargs)

    monkeypatch.setattr(HostCompiler, "_compile_source", staticmethod(counting))
    ir = _SkipIRClient()
    traced = tilelens.trace(ir)(_make_add_one())
    x, out = _inputs()

    first = traced[(4,)](x, out, 64, BLOCK=16)
    again = traced[(4,)](torch.zeros(64), torch.zeros(64), 64, BLOCK=16)
    other = traced[(4,)](x, out, 63, BLOCK=16)  # 63: not divisible by 16

    assert again is first and other is not first
    assert len(compiles) == 2
    assert [len(events) for events in ir.finalized] == [1, 1, 1]
    assert ir.finalized[0][0].specialization == ir.finalized[1][0].specialization


@needs_gpu
def test_ir_only_run_launches_the_real_kernel():
    ir = _RunIRClient()
    kernel = _make_add_one()
    hooks = []
    kernel.add_pre_run_hook(lambda *args, **kwargs: hooks.append(1))
    traced = tilelens.trace(ir)(kernel)
    x, out = _inputs(device="cuda")

    ret = traced[(4,)](x, out, 64, BLOCK=16)
    torch.cuda.synchronize()

    torch.testing.assert_close(out, x + 1)
    (events,) = ir.finalized
    # Compiled first (launched=False), then seen again before the launch;
    # both carry the host compile, the launch compiled its own device kernel.
    assert [e.launched for e in events] == [False, True]
    assert events[0].kernel is events[1].kernel
    assert events[0].kernel.target == CUDA89
    assert isinstance(ret, CompiledKernel) and ret.module is not None
    # Only the real launch entered JITFunction.run.
    assert hooks == [1]


@pytest.mark.parametrize("ir_cls", IR_CLASSES)
def test_autotune_over_heuristics_reports_every_config(ir_cls):
    user = _make_autotuned()
    ir = ir_cls()
    traced = tilelens.trace(ir)(user)
    x, out = _inputs(device=_device(ir_cls))

    rets = [traced[_grid](x, out, 64), traced[_grid](x, out, 64)]
    grids = [launch.grid for launch in trace_module.launches[-2:]]
    _synchronize()

    # Every launch reports every config compile-only, whatever the autotune
    # cache holds or the benchmark picked.
    for events in ir.finalized:
        compiled = [e for e in events if not e.launched]
        assert [e.kwargs["BLOCK"] for e in compiled] == [16, 32]
        assert {e.kwargs["EVEN"] for e in compiled} == {True}
        assert len({e.specialization for e in compiled}) == 2
    first, second = ir.finalized
    if ir_cls is _SkipIRClient:
        assert len(first) == len(second) == 2
        assert torch.equal(out, torch.zeros_like(x))
        # No config was picked: nothing to return, and the configs' grids
        # differ, so the Launch has none.
        assert rets == [None, None] and grids == [None, None]
    else:
        # Benchmarking launches each config (once reported, however many
        # benchmark calls); the cached second launch only the winner.
        assert sorted(e.kwargs["BLOCK"] for e in first if e.launched) == [16, 32]
        assert len([e for e in second if e.launched]) == 1
        torch.testing.assert_close(out, x + 1)
        # The winner's kernel and grid, on the benchmarking launch too.
        winner = traced.ir_runner.best_config.kwargs["BLOCK"]
        assert all(ret.hash == rets[1].hash for ret in rets)
        assert grids == [(64 // winner, 1, 1)] * 2
    assert user.cache == {}


def _make_runtime_stride_configs():
    # S is a runtime int that 2 and 3 specialize alike: both configs compile
    # to one kernel, yet cover other elements (and here grids).
    @triton.autotune(
        configs=[
            triton.Config({"S": 2}, num_warps=1),
            triton.Config({"S": 3}, num_warps=1),
        ],
        key=["n"],
    )
    @triton.jit
    def strided_copy(x_ptr, out_ptr, n, S, BLOCK: tl.constexpr):
        offs = tl.program_id(0) * BLOCK * S + tl.arange(0, BLOCK)
        mask = offs < n
        tl.store(out_ptr + offs, tl.load(x_ptr + offs, mask=mask), mask=mask)

    return strided_copy


def _grid_per_stride(meta):
    return (64 // (16 * meta["S"]),)  # S=2: 2 programs, S=3: 1


@pytest.mark.parametrize("ir_cls", IR_CLASSES)
def test_configs_compiling_to_one_kernel_each_get_an_event(ir_cls):
    """D22: the dedup key holds the binding, so the second config is not
    lost behind the first one's (specialization, launched)."""
    ir = ir_cls()
    traced = tilelens.trace(ir)(_make_runtime_stride_configs())
    x, out = _inputs(device=_device(ir_cls))

    traced[_grid_per_stride](x, out, 64, BLOCK=16)
    _synchronize()

    (events,) = ir.finalized
    compiled = [e for e in events if not e.launched]
    assert [(e.kwargs["S"], e.resolved_grid) for e in compiled] == [
        (2, (2, 1, 1)),
        (3, (1, 1, 1)),
    ]
    assert len({e.specialization for e in events}) == 1
    if ir_cls is _RunIRClient:
        # Each config benchmarked for real, seen once more as launched.
        assert sorted(e.kwargs["S"] for e in events if e.launched) == [2, 3]


@needs_gpu
def test_benchmark_repetitions_share_one_event_per_config():
    """An autotuned "run" launch benchmarks every config with many real
    calls; each config's calls share one binding, so the event count stays
    two per config (compile-only, then launched)."""
    user = _make_autotuned()
    calls = []
    user.fn.fn.add_pre_run_hook(lambda *args, **kwargs: calls.append(1))
    ir = _RunIRClient()
    traced = tilelens.trace(ir)(user)
    x, out = _inputs(device="cuda")

    traced[_grid](x, out, 64)
    torch.cuda.synchronize()

    (events,) = ir.finalized
    assert sorted((e.launched, e.kwargs["BLOCK"]) for e in events) == [
        (False, 16),
        (False, 32),
        (True, 16),
        (True, 32),
    ]
    # One run() entry per benchmark repetition and for the final launch (the
    # host compiles enter none): far more calls than events.
    assert len(calls) > 10 * len(events)
    torch.testing.assert_close(out, x + 1)


@needs_gpu
def test_a_fresh_constexpr_object_per_call_adds_no_event():
    """A heuristic building its tl.dtype per call hands every benchmark
    repetition an equal but distinct constexpr object. Triton hashes it into
    the kernel, so it adds no binding: two events per config, as above."""

    @triton.autotune(
        configs=[
            triton.Config({"BLOCK": 16}, num_warps=1),
            triton.Config({"BLOCK": 32}, num_warps=2),
        ],
        key=["n"],
    )
    @triton.heuristics({"DT": lambda args: tl.dtype("fp32")})
    @triton.jit
    def cast_add_one(x_ptr, out_ptr, n, DT: tl.constexpr, BLOCK: tl.constexpr):
        offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n
        value = tl.load(x_ptr + offs, mask=mask).to(DT) + 1
        tl.store(out_ptr + offs, value, mask=mask)

    ir = _RunIRClient()
    traced = tilelens.trace(ir)(cast_add_one)
    x, out = _inputs(device="cuda")

    traced[_grid](x, out, 64)
    torch.cuda.synchronize()

    (events,) = ir.finalized
    assert sorted((e.launched, e.kwargs["BLOCK"]) for e in events) == [
        (False, 16),
        (False, 32),
        (True, 16),
        (True, 32),
    ]
    torch.testing.assert_close(out, x + 1)


class _ForgetfulRunIRClient(_RunIRClient):
    """Keeps nothing of a launch, as a harness-side client would."""

    NAME = "ir_run_forgetful"

    def before_launch(self, event):
        self.log.append("before")

    def finalize(self):
        self.log.append("finalize")
        return []


def _compiled_sanitizer():
    from tilelens.clients import Sanitizer

    return Sanitizer(compile=True, abort_on_error=False)


def test_ir_only_launches_retain_no_tensor():
    """D23, without a GPU: a harness launching an IR-only trace in a loop
    with fresh tensors, calling tilelens.clear() after each launch, holds on
    to none of them (the grid callable closes over them, too); neither does
    the trace's host-compile cache."""
    import gc
    import weakref

    traced = tilelens.trace(_compiled_sanitizer())(_make_add_one())
    refs = []

    def launch():
        x = torch.empty(4096)
        out = torch.empty_like(x)

        def grid(meta):
            return (triton.cdiv(x.numel(), meta["BLOCK"]),)

        traced[grid](x, out, 4096, BLOCK=1024)
        assert not trace_module.launches[-1].tensors
        tilelens.clear()
        refs.extend((weakref.ref(x), weakref.ref(out)))

    for _ in range(3):
        launch()
    gc.collect()

    assert [ref() for ref in refs] == [None] * 6


@needs_gpu
@pytest.mark.parametrize(
    "make_client, make_kernel",
    [
        (_compiled_sanitizer, _make_add_one),
        (_ForgetfulRunIRClient, _make_add_one),
        (_ForgetfulRunIRClient, _make_autotuned),
    ],
    ids=["compiled-sanitizer", "run", "run-autotuned"],
)
def test_ir_only_launches_retain_no_device_memory(make_client, make_kernel):
    """D23: a harness launching an IR-only trace in a loop with fresh
    device tensors, calling tilelens.clear() after each launch, holds on to
    none of them (the grid callable closes over them, too)."""
    traced = tilelens.trace(make_client())(make_kernel())
    n = 16 * 2**20  # 64 MiB of float32
    # The autotuned kernel's configs set BLOCK themselves.
    kwargs = {"BLOCK": 1024} if make_kernel is _make_add_one else {}

    def launch():
        x = torch.empty(n, device="cuda")
        out = torch.empty_like(x)

        def grid(meta):
            return (triton.cdiv(x.numel(), meta["BLOCK"]),)

        traced[grid](x, out, n, **kwargs)
        assert not trace_module.launches[-1].tensors
        tilelens.clear()

    # Measured from before the first launch: the manager's Launch is
    # replaced per launch, so tensors it held would be the last launch's.
    torch.cuda.synchronize()
    base = torch.cuda.memory_allocated()
    for _ in range(10):
        launch()
    torch.cuda.synchronize()

    assert torch.cuda.memory_allocated() - base < 2**20


class _InterruptingRunIRClient(_ForgetfulRunIRClient):
    """The user hits Ctrl+C in a launch's first autotune benchmark call."""

    NAME = "ir_run_interrupting"

    def before_launch(self, event):
        super().before_launch(event)
        if event.launched:
            raise KeyboardInterrupt


@needs_gpu
def test_an_interrupted_benchmark_retains_no_device_memory():
    """D23: Triton's _bench skips the post_hook that drops a benchmark
    call's restore_value clones for a KeyboardInterrupt; the aborted launch
    keeps neither those device clones nor the caller's tensors."""
    traced = tilelens.trace(_InterruptingRunIRClient())(
        _make_autotuned(restore_value=["x_ptr"])
    )
    n = 16 * 2**20  # 64 MiB of float32

    def launch():
        x = torch.empty(n, device="cuda")
        out = torch.empty_like(x)
        with pytest.raises(KeyboardInterrupt):
            traced[_grid](x, out, n)
        tilelens.clear()

    torch.cuda.synchronize()
    base = torch.cuda.memory_allocated()
    for _ in range(3):
        launch()
    torch.cuda.synchronize()

    assert torch.cuda.memory_allocated() - base < 2**20


@pytest.mark.parametrize("ir_cls", IR_CLASSES)
def test_plain_heuristics_kernel_fires_events(ir_cls):
    @triton.heuristics({"BLOCK": lambda args: 16})
    @triton.jit
    def heur_add_one(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
        offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n
        tl.store(out_ptr + offs, tl.load(x_ptr + offs, mask=mask) + 1, mask=mask)

    ir = ir_cls()
    traced = tilelens.trace(ir)(heur_add_one)
    x, out = _inputs(device=_device(ir_cls))

    traced[_grid](x, out, 64)
    _synchronize()

    (events,) = ir.finalized
    assert [e.kwargs["BLOCK"] for e in events if not e.launched] == [16]
    assert events[0].resolved_grid == (4, 1, 1)
    if ir_cls is _RunIRClient:
        torch.testing.assert_close(out, x + 1)
    else:
        assert torch.equal(out, torch.zeros_like(x))


@pytest.mark.parametrize("ir_cls", IR_CLASSES)
def test_a_compile_error_is_data_and_the_next_launch_is_clean(ir_cls):
    @triton.jit
    def bounded(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
        tl.static_assert(BLOCK <= 32)
        offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n
        tl.store(out_ptr + offs, tl.load(x_ptr + offs, mask=mask) + 1, mask=mask)

    ir = ir_cls()
    traced = tilelens.trace(ir)(bounded)
    x, out = _inputs(device=_device(ir_cls))

    # The only config failed to host-compile: reported as data (D27). A
    # skipped launch then ends normally; a real launch compiles for the
    # device, which fails as the untraced launch does.
    if ir_cls is _RunIRClient:
        with pytest.raises(CompileTimeAssertionFailure):
            traced[(1,)](x, out, 64, BLOCK=64)
        assert ir.log == ["begin", "compile_failed", "abort"]
    else:
        assert traced[(1,)](x, out, 64, BLOCK=64) is None
        assert ir.log == ["begin", "compile_failed", "finalize"]
        assert ir.finalized == [[]]
    (failure,) = ir.failures
    assert isinstance(failure.error, CompileTimeAssertionFailure)
    assert failure.target == CUDA89
    assert "run" not in vars(traced.jit_fn)

    ir.log.clear()
    traced[(4,)](x, out, 64, BLOCK=16)
    _synchronize()

    if ir_cls is _RunIRClient:
        assert ir.log == ["begin"] + ["before", "after"] * 2 + ["finalize"]
        assert [len(events) for events in ir.finalized] == [2]
        torch.testing.assert_close(out, x + 1)
    else:
        assert ir.log == ["begin", "before", "after", "finalize"]
        # The failed launch finalized with nothing compiled.
        assert [len(events) for events in ir.finalized] == [0, 1]


def _make_bounded_autotuned():
    @triton.autotune(
        configs=[
            triton.Config({"BLOCK": 16}, num_warps=1),
            # Fails to compile (static_assert) ...
            triton.Config({"BLOCK": 64}, num_warps=1),
            # ... compiles, but needs more threads than a block can have.
            triton.Config({"BLOCK": 16}, num_warps=64),
        ],
        key=["n"],
    )
    @triton.jit
    def bounded(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
        tl.static_assert(BLOCK <= 32)
        offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n
        tl.store(out_ptr + offs, tl.load(x_ptr + offs, mask=mask) + 1, mask=mask)

    return bounded


def test_a_config_that_fails_to_compile_is_reported_and_one_too_big_is_analyzed():
    """D25: nothing is loaded, so a config some device could not run
    (num_warps=64: more threads than a block can have) is analyzed like any
    other; it can only add findings, never hide one."""
    ir = _SkipIRClient()
    traced = tilelens.trace(ir)(_make_bounded_autotuned())
    x, out = _inputs()

    traced[_grid](x, out, 64)

    (events,) = ir.finalized
    assert [(e.kwargs["BLOCK"], e.kwargs["num_warps"]) for e in events] == [
        (16, 1),
        (16, 64),
    ]
    ((config, error),) = [
        ((f.kwargs["BLOCK"], f.kwargs["num_warps"]), f.error) for f in ir.failures
    ]
    assert config == (64, 1) and isinstance(error, CompileTimeAssertionFailure)
    assert torch.equal(out, torch.zeros_like(x))


@needs_gpu
def test_the_real_launch_drops_what_the_device_cannot_run():
    """Under "run" the device decides: the JIT's own load of the num_warps=64
    config raises OutOfResources, which the autotuner's benchmark absorbs
    after the host-compiled config was delivered."""
    ir = _RunIRClient()
    traced = tilelens.trace(ir)(_make_bounded_autotuned())
    x, out = _inputs(device="cuda")

    traced[_grid](x, out, 64)
    torch.cuda.synchronize()

    (events,) = ir.finalized
    compiled = {(e.kwargs["BLOCK"], e.kwargs["num_warps"]) for e in events}
    assert compiled == {(16, 1), (16, 64)}
    # The failing config is reported once, although benchmarked again.
    assert [(f.kwargs["BLOCK"], f.kwargs["num_warps"]) for f in ir.failures] == [
        (64, 1)
    ]
    # Only the winner reached after_launch as a real launch; (16, 64) raised.
    assert ir.log.count("after") == len(events) - 1
    torch.testing.assert_close(out, x + 1)


def _times_two():
    # `other=` exercises the semantic._load_legacy path (`other.handle if
    # other else None`) that a leaked tensor.__bool__ would break.
    @triton.jit
    def times_two(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
        offs = tl.arange(0, BLOCK)
        mask = offs < n
        loaded = tl.load(x_ptr + offs, mask=mask, other=-1.0)
        tl.store(out_ptr + offs, loaded * 2, mask=mask)

    return times_two


def test_mixed_trace_compiles_for_ir_and_interprets_for_eager():
    kernel = _make_add_one()
    ir, eager = _SkipIRClient(), _EagerCounter()
    traced = tilelens.trace(eager)(tilelens.trace(ir)(kernel))
    x, out = _inputs()

    traced[(4,)](x, out, 64, BLOCK=16)

    (events,) = ir.finalized
    assert [e.launched for e in events] == [False]
    assert list(events[0].kernel.asm) == ["ttir"]
    assert eager.stores == 4
    torch.testing.assert_close(out, x + 1)
    # Launch.tensors: the interpreter's copies only, not the caller's
    # tensors on top.
    tensors = trace_module.launches[-1].tensors
    assert len(tensors) == 2 and not {id(t) for t in tensors} & {id(x), id(out)}

    # D4b regression: interpreter patches must not leak into later compiles,
    # of another kernel or of this one (a new BLOCK forces a recompile).
    compiler = HostCompiler()
    for jit_fn, block in ((_times_two(), 64), (kernel, 32)):
        compiled = compiler.compile(
            jit_fn, (x, out, 64), {"BLOCK": block}, target=CUDA80, stages={"ttir"}
        )
        assert "tt.store" in compiled.asm["ttir"]


@needs_gpu
def test_interpreter_patches_do_not_leak_into_later_real_launches():
    kernel = _make_add_one()
    traced = tilelens.trace(_EagerCounter())(tilelens.trace(_SkipIRClient())(kernel))
    x, out = _inputs(device="cuda")
    traced[(4,)](x, out, 64, BLOCK=16)

    doubled = torch.zeros_like(x)
    _times_two()[(1,)](x, doubled, 64, BLOCK=64)
    again = torch.zeros_like(x)
    kernel[(2,)](x, again, 64, BLOCK=32)
    torch.cuda.synchronize()
    torch.testing.assert_close(doubled, x * 2)
    torch.testing.assert_close(again, x + 1)


@pytest.mark.parametrize(
    "client_cls",
    [_SkipIRClient, _EagerCounter, pytest.param(_RunIRClient, marks=needs_gpu)],
)
def test_trace_leaves_the_users_autotuner_unchanged(client_cls):
    user = _make_autotuned()
    heuristics, jit_fn = user.fn, user.fn.fn
    before = dict(vars(user))
    before_heuristics = dict(vars(heuristics))
    traced = tilelens.trace(client_cls())(user)
    x, out = _inputs(device=_device(client_cls))

    traced[_grid64](x, out, 64)
    _synchronize()

    assert vars(user).keys() == before.keys()
    assert all(vars(user)[k] is v for k, v in before.items())
    assert vars(heuristics).keys() == before_heuristics.keys()
    assert all(vars(heuristics)[k] is v for k, v in before_heuristics.items())
    assert user.fn is heuristics and heuristics.fn is jit_fn
    assert user.cache == {}


@needs_gpu
def test_the_users_autotuner_still_autotunes_for_real_after_a_trace():
    user = _make_autotuned()
    x, out = _inputs(device="cuda")
    tilelens.trace(_RunIRClient())(user)[_grid64](x, out, 64)

    untraced = torch.zeros_like(x)
    user[_grid64](x, untraced, 64)
    torch.cuda.synchronize()
    torch.testing.assert_close(untraced, x + 1)
    assert len(user.cache) == 1
    assert user.best_config in user.configs


def _add_one_helper(x):
    return x + 1


def _kernel_with_helper(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    tl.store(
        out_ptr + offs,
        _traced_add_one_helper(tl.load(x_ptr + offs, mask=mask)),  # noqa: F821
        mask=mask,
    )


@pytest.fixture
def helper_kernel():
    """A kernel whose device function is a module global wrapped by
    tilelens.trace, as the CLI wrappers do for every @triton.jit function.
    The real code generator only accepts JITFunctions, so real compiles must
    see the unwrapped binding. Built here, not at import, so the jits are real
    even when TRITON_INTERPRET was set during collection."""
    module_globals = globals()
    helper = tilelens.trace(_EagerCounter())(triton.jit(_add_one_helper))
    module_globals["_traced_add_one_helper"] = helper
    try:
        yield triton.jit(_kernel_with_helper), helper
    finally:
        module_globals.pop("_traced_add_one_helper", None)


@pytest.mark.parametrize("ir_cls", IR_CLASSES)
def test_traced_device_function_compiles_under_ir_capture(helper_kernel, ir_cls):
    kernel, helper = helper_kernel
    ir = ir_cls()
    traced = tilelens.trace(ir)(kernel)
    x, out = _inputs(device=_device(ir_cls))

    traced[(4,)](x, out, 64, BLOCK=16)
    _synchronize()

    (events,) = ir.finalized
    if ir_cls is _RunIRClient:
        torch.testing.assert_close(out, x + 1)
        assert [e.launched for e in events] == [False, True]
    else:
        assert [e.launched for e in events] == [False]
    assert "tt.func" in events[0].kernel.asm["ttir"]
    assert globals()["_traced_add_one_helper"] is helper


def _kernel_with_package_helper(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    tl.store(
        out_ptr + offs,
        _helper_pkg.api.add_one(tl.load(x_ptr + offs, mask=mask)),  # noqa: F821
        mask=mask,
    )


@pytest.mark.parametrize("ir_cls", IR_CLASSES)
def test_traced_device_function_behind_a_package_path_compiles(ir_cls):
    # `pkg.api.add_one`, where `api` re-exports the traced helper: Triton
    # resolves it through two module attributes.
    module_globals = globals()
    helper = tilelens.trace(_EagerCounter())(triton.jit(_add_one_helper))
    pkg = types.ModuleType("tilelens_test_helper_pkg")
    pkg.api = types.ModuleType("tilelens_test_helper_pkg.api")
    pkg.api.add_one = helper
    module_globals["_helper_pkg"] = pkg
    try:
        ir = ir_cls()
        traced = tilelens.trace(ir)(triton.jit(_kernel_with_package_helper))
        x, out = _inputs(device=_device(ir_cls))

        traced[(4,)](x, out, 64, BLOCK=16)
        _synchronize()

        (events,) = ir.finalized
        assert "tt.func" in events[0].kernel.asm["ttir"]
        assert pkg.api.add_one is helper
        if ir_cls is _RunIRClient:
            torch.testing.assert_close(out, x + 1)
        else:
            assert torch.equal(out, torch.zeros_like(x))
    finally:
        module_globals.pop("_helper_pkg", None)


@needs_gpu
def test_traced_device_function_compiles_in_the_interpreted_warmup(helper_kernel):
    # Interpreting clients that vote for a warmup compile for real, through
    # the JIT (e.g. the profiler); the traced helper must resolve there too.
    kernel, helper = helper_kernel
    eager = _EagerCounter(warmup_vote=True)
    traced = tilelens.trace(eager)(kernel)
    x, out = _inputs(device="cuda")

    traced[(4,)](x, out, 64, BLOCK=16)
    torch.cuda.synchronize()

    assert len(eager.warmups) == 1 and "ttir" in eager.warmups[0].asm
    assert eager.stores == 4
    torch.testing.assert_close(out, x + 1)
    assert globals()["_traced_add_one_helper"] is helper


def _make_apply_fn():
    @triton.jit
    def apply_fn(x_ptr, out_ptr, n, FN: tl.constexpr, BLOCK: tl.constexpr):
        offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n
        tl.store(out_ptr + offs, FN(tl.load(x_ptr + offs, mask=mask)), mask=mask)

    return apply_fn


def _make_apply_default():
    # Triton's code generator evaluates parameter defaults in the kernel's
    # globals, so the default is a module global (bound by the caller).
    @triton.jit
    def apply_default(
        x_ptr,
        out_ptr,
        n,
        BLOCK: tl.constexpr,
        FN: tl.constexpr = _traced_default_helper,  # noqa: F821
    ):
        offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n
        tl.store(out_ptr + offs, FN(tl.load(x_ptr + offs, mask=mask)), mask=mask)

    return apply_default


def _make_apply_first():
    @triton.jit
    def apply_first(x_ptr, out_ptr, n, FNS: tl.constexpr, BLOCK: tl.constexpr):
        offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n
        tl.store(out_ptr + offs, FNS[0](tl.load(x_ptr + offs, mask=mask)), mask=mask)

    return apply_first


def _assert_real_compiles_still_work(monkeypatch, *, launch: bool):
    # D4b: the interpreter's triton.language patches must not have leaked. A
    # never-seen kernel, host-compiled (never disk-cached); with ``launch``
    # (a needs_gpu test) also compiled and launched by the JIT (no disk-cache
    # shortcut).
    monkeypatch.setenv("TRITON_ALWAYS_COMPILE", "1")

    @triton.jit
    def times_two(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
        offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n
        tl.store(out_ptr + offs, tl.load(x_ptr + offs, mask=mask) * 2, mask=mask)

    x, out = _inputs()
    compiled = HostCompiler().compile(
        times_two, (x, out, 64), {"BLOCK": 16}, target=CUDA80, stages={"ttir"}
    )
    assert "tt.store" in compiled.asm["ttir"]
    if not launch:
        return
    x, out = _inputs(device="cuda")
    times_two[(4,)](x, out, 64, BLOCK=16)
    torch.cuda.synchronize()
    torch.testing.assert_close(out, x * 2)


@pytest.mark.parametrize("ir_cls", IR_CLASSES)
@pytest.mark.parametrize("passing", ["keyword", "default", "tuple"])
def test_traced_helper_passed_as_an_argument_compiles(ir_cls, passing, monkeypatch):
    # The CLI shape (every @triton.jit is a TritonTrace), with the helper
    # reaching the real compile as a constexpr argument, which the globals
    # unwrap cannot see.
    helper = tilelens.trace(_EagerCounter())(triton.jit(_add_one_helper))
    if passing == "keyword":
        kernel, extra = _make_apply_fn(), {"FN": helper}
    elif passing == "default":
        monkeypatch.setitem(globals(), "_traced_default_helper", helper)
        kernel, extra = _make_apply_default(), {}
    else:
        kernel, extra = _make_apply_first(), {"FNS": (helper,)}
    ir = ir_cls()
    x, out = _inputs(device=_device(ir_cls))

    tilelens.trace(ir)(kernel)[(4,)](x, out, 64, BLOCK=16, **extra)
    _synchronize()

    assert ir.failures == []
    (events,) = ir.finalized
    assert "tt.func" in events[0].kernel.asm["ttir"]
    if ir_cls is _RunIRClient:
        torch.testing.assert_close(out, x + 1)
    _assert_real_compiles_still_work(monkeypatch, launch=ir_cls is _RunIRClient)


@needs_gpu
def test_voted_warmup_compiles_with_a_traced_helper_argument(monkeypatch):
    # Interpreting clients that vote for a warmup compile for real, through
    # the JIT.
    helper = tilelens.trace(_EagerCounter())(triton.jit(_add_one_helper))
    eager = _EagerCounter(warmup_vote=True)
    x, out = _inputs(device="cuda")

    tilelens.trace(eager)(_make_apply_fn())[(4,)](x, out, 64, BLOCK=16, FN=helper)
    torch.cuda.synchronize()

    assert len(eager.warmups) == 1 and "ttir" in eager.warmups[0].asm
    torch.testing.assert_close(out, x + 1)
    _assert_real_compiles_still_work(monkeypatch, launch=True)


def test_a_traced_callee_the_compile_still_reaches_fails_it_cleanly(monkeypatch):
    # Should a compile reach a TritonTrace anyway (here: with the argument
    # mapping switched off), the trace refuses to interpret there. The
    # compile fails; triton.language is left alone for later compiles.
    monkeypatch.setattr(
        trace_module, "_untraced_call_args", lambda jit_fn, args, kwargs: (args, kwargs)
    )
    helper = tilelens.trace(_EagerCounter())(triton.jit(_add_one_helper))
    ir = _SkipIRClient()
    x, out = _inputs()

    # A compile failure like any other (D27): data, and the skipped launch
    # ends normally.
    tilelens.trace(ir)(_make_apply_fn())[(4,)](x, out, 64, BLOCK=16, FN=helper)

    assert ir.log == ["begin", "compile_failed", "finalize"]
    (failure,) = ir.failures
    assert isinstance(failure.error, CompilationError)
    assert "outside a traced launch" in str(failure.error)
    _assert_real_compiles_still_work(monkeypatch, launch=False)
