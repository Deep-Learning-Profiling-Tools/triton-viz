"""CPU-only tests of the core IR lifecycle: client declarations, ClientManager
dispatch rules, the ``ir_capture`` run wrapper and TritonTrace's runner handling.

The core compiles IR kernels on the host (tilelens.core.host_compile, D25).
These tests pin call sequences, so a fake stands in for that compile: a
``_FakeJit`` compiles through its ``fake_compile`` and launches through its
``run``, and a real JITFunction gets both from ``fake_compile(jit_fn)``
(``_install_fake_run``); a host compile nothing faked fails the test. The
real host compile is tested in tests/unit/ir/test_host_compile.py (against
the JIT's own compile in tests/end_to_end/test_host_compile.py) and end to
end in tests/end_to_end/test_ir_lifecycle_compiled.py.
"""

import ast
import gc
import importlib
import inspect
import threading
import types
import weakref
from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch
import triton
import triton.language as tl
from triton.compiler.errors import CompileTimeAssertionFailure
from triton.runtime import Autotuner
from triton.runtime.autotuner import Heuristics
from triton.runtime.interpreter import InterpretedFunction

import tilelens
from tilelens.clients import Sanitizer, Tracer
from tilelens.core.callbacks import ForLoopCallbacks, OpCallbacks
from tilelens.core.client import (
    Client,
    ClientManager,
    LanguagePatchedError,
    LaunchCall,
    LaunchEvent,
    _resolve_grid,
)
from tilelens.core.config import DEFAULT_IR_TARGET, config as tilelens_config
from tilelens.core.data import Store
from tilelens.core.frontend.base import LANG_PATCH_SCOPES, get_frontend
from tilelens.core.host_compile import HostCompiler, default_ir_target
from tilelens.core.trace import (
    GluonTrace,
    KernelTraceSupport,
    NKITrace,
    TraceInterface,
    TritonTrace,
    _untraced_call_args,
    _unwrapped_trace_globals,
)

# `tilelens.core.trace` the attribute is the trace() decorator; the module
# holds the `launches` list.
trace_module = importlib.import_module("tilelens.core.trace")


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


@pytest.fixture(autouse=True)
def _default_ir_target(monkeypatch):
    # Whatever TILELENS_IR_TARGET the caller has set.
    monkeypatch.setattr(tilelens_config, "ir_target", DEFAULT_IR_TARGET)


@pytest.fixture(autouse=True)
def _fake_host_compile(monkeypatch):
    """Route the core's host compile to the jit_fn's ``fake_compile`` (see
    the module docstring); every compile is recorded in ``compiles`` as
    (jit_fn, target, stages)."""
    compiles: list[tuple] = []

    def compile(self, jit_fn, args, kwargs, *, target, stages=()):
        fake = getattr(jit_fn, "fake_compile", None)
        assert fake is not None, f"unexpected host compile of {jit_fn!r}"
        compiles.append((jit_fn, target, frozenset(stages)))
        return fake(*args, **kwargs)

    monkeypatch.setattr(HostCompiler, "compile", compile)
    return compiles


# ======== Fake clients =========


class _EagerClient(Client):
    """Interpreting client that records every callback it receives."""

    NAME = "eager"

    def __init__(self, *, warmup_vote=False, loop_overrider=None, records=()):
        super().__init__()
        self.calls: list = []
        self.stores = 0
        self.warmup_vote = warmup_vote
        self.loop_overrider = loop_overrider
        self.records = list(records)
        self.on_store = self._on_store

    def _on_store(self, *args, **kwargs):
        self.stores += 1

    def pre_run_callback(self, fn):
        self.calls.append("pre_run")
        return True

    def post_run_callback(self, fn):
        self.calls.append("post_run")
        return True

    def arg_callback(self, name, arg, arg_cvt):
        self.calls.append(("arg", name))

    def grid_callback(self, grid):
        self.calls.append(("grid", grid))

    def grid_idx_callback(self, grid_idx):
        self.calls.append("grid_idx")

    def register_op_callback(self, op_type, *args, **kwargs):
        if op_type is Store:
            return OpCallbacks(before_callback=self.on_store)
        return OpCallbacks()

    def register_for_loop_callback(self):
        return ForLoopCallbacks(loop_iter_overrider=self.loop_overrider)

    def finalize(self):
        self.calls.append("finalize")
        return list(self.records)

    def pre_warmup_callback(self, jit_fn, *args, **kwargs):
        self.calls.append("pre_warmup")
        return self.warmup_vote

    def post_warmup_callback(self, jit_fn, ret):
        self.calls.append(("post_warmup", ret))

    def begin_launch(self, call):
        self.calls.append("begin")

    def abort_launch(self, exc):
        self.calls.append(("abort", type(exc)))

    def before_launch(self, event):
        self.calls.append("before_launch")


class _OtherEagerClient(_EagerClient):
    NAME = "other_eager"


class _SiblingEagerClient(Client):
    """A second interpreting client class, unrelated to _EagerClient."""

    NAME = "sibling_eager"

    def __init__(self, loop_overrider=None):
        super().__init__()
        self.loop_overrider = loop_overrider

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
        return OpCallbacks()

    def register_for_loop_callback(self):
        return ForLoopCallbacks(loop_iter_overrider=self.loop_overrider)

    def finalize(self):
        return []

    def pre_warmup_callback(self, jit_fn, *args, **kwargs):
        return False

    def post_warmup_callback(self, jit_fn, ret):
        pass


class _IRClient(Client):
    """IR client: records lifecycle hooks; interpreter hooks must never fire."""

    NEEDS_INTERPRETER = False
    IR_STAGES = frozenset({"ttir"})

    def __init__(self, log=None, *, raise_in_before=None, records=()):
        super().__init__()
        self.log = [] if log is None else log
        self.events: list[LaunchEvent] = []
        self.failures: list[LaunchEvent] = []
        self.finalized: list[list[LaunchEvent]] = []
        self.launch_calls: list[LaunchCall] = []
        self.raise_in_before = raise_in_before
        self.records = list(records)

    def begin_launch(self, call):
        self.log.append("begin")
        self.launch_calls.append(call)
        self.events = []
        self.failures = []

    def abort_launch(self, exc):
        self.log.append(("abort", type(exc)))

    def before_launch(self, event):
        self.log.append("before")
        if self.raise_in_before is not None:
            raise self.raise_in_before
        self.events.append(event)

    def after_launch(self, event):
        self.log.append("after")

    def compile_failed(self, event):
        self.log.append(("compile_failed", type(event.error)))
        self.failures.append(event)

    def finalize(self):
        self.log.append("finalize")
        self.finalized.append(list(self.events))
        return list(self.records)

    def pre_warmup_callback(self, jit_fn, *args, **kwargs):
        self.log.append("pre_warmup")
        return False

    def post_warmup_callback(self, jit_fn, ret):
        self.log.append("post_warmup")

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


class _IndifferentIRClient(_IRClient):
    NAME = "ir_indifferent"


# ======== Fake compile =========


class _FakeKernel:
    def __init__(self, key):
        self.hash = f"hash-{key}"
        self.asm = {"ttir": f"// ttir {key}"}

    def _init_handles(self):
        # A CompiledKernel loads its binary here; IR mode never does (D25).
        raise AssertionError("IR mode loaded a kernel")


def _fake_kernel_signature(x_ptr, n, BLOCK=4):
    pass


class _FakeJit:
    """Stands in for a JITFunction: the host compile calls fake_compile,
    a real launch run()."""

    signature = inspect.signature(_fake_kernel_signature)

    def __init__(self, log=None, *, compile_error=None):
        self.log = [] if log is None else log
        self.compile_error = compile_error

    def fake_compile(self, *args, **kwargs):
        self.log.append("compile")
        if self.compile_error is not None:
            raise self.compile_error
        return _FakeKernel(kwargs.get("BLOCK", 4))

    def run(self, *args, grid, warmup, **kwargs):
        self.log.append("compile" if warmup else "launch")
        return None if warmup else "launched"


def _install_fake_run(monkeypatch, jit_fn, run):
    """Install ``run(*args, grid, warmup, **kwargs)`` on a real JITFunction
    as both its launch (``run``, warmup=False) and its fake host compile
    (warmup=True, grid=None: a host compile needs no grid)."""
    monkeypatch.setattr(jit_fn, "run", run, raising=False)
    monkeypatch.setattr(
        jit_fn,
        "fake_compile",
        lambda *args, **kwargs: run(*args, grid=None, warmup=True, **kwargs),
        raising=False,
    )


@pytest.fixture
def fake_compile(monkeypatch):
    """Record a real JITFunction's host compiles (``warmup=True``) and
    launches (``warmup=False``), in order, instead of running them.

    ``compile_error(kwargs)`` may return an exception for a compile to
    raise, e.g. per config.
    """

    def install(jit_fn, *, fail_first=False, compile_error=None):
        calls: list[SimpleNamespace] = []

        def run(*args, grid, warmup, **kwargs):
            calls.append(
                SimpleNamespace(args=args, grid=grid, warmup=warmup, kwargs=kwargs)
            )
            if fail_first and len(calls) == 1:
                raise RuntimeError("compile failed")
            if warmup and compile_error is not None:
                error = compile_error(kwargs)
                if error is not None:
                    raise error
            return _FakeKernel(tuple(sorted(kwargs.items())))

        _install_fake_run(monkeypatch, jit_fn, run)
        return calls

    return install


def _fake_bench(kernel_call, quantiles):
    # An Autotuner do_bench that needs no GPU: every config ties.
    kernel_call()
    return [1.0, 1.0, 1.0]


def _static_assert_failure():
    return CompileTimeAssertionFailure(None, ast.Pass(), "static_assert failed")


def _call(**overrides):
    fields: dict = dict(jit_fn=None, args=(), kwargs={}, grid=(1,), capture=False)
    fields.update(overrides)
    return LaunchCall(**fields)


def _make_plain_kernel():
    @triton.jit
    def add_one(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
        offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n
        tl.store(out_ptr + offs, tl.load(x_ptr + offs, mask=mask) + 1, mask=mask)

    return add_one


def _make_autotuned_kernel(**autotune_kwargs):
    @triton.autotune(
        configs=[triton.Config({"BLOCK": 4}), triton.Config({"BLOCK": 8})],
        key=["n"],
        **autotune_kwargs,
    )
    @triton.heuristics({"EVEN": lambda args: args["n"] % args["BLOCK"] == 0})
    @triton.jit
    def add_one_tuned(x_ptr, out_ptr, n, BLOCK: tl.constexpr, EVEN: tl.constexpr):
        offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n
        tl.store(out_ptr + offs, tl.load(x_ptr + offs, mask=mask) + 1, mask=mask)

    return add_one_tuned


def _grid(meta):
    return (triton.cdiv(meta["n"], meta["BLOCK"]),)


def _grid8(meta):
    # The interpreter hands grid callables tensor-converted runtime args, so
    # interpreted launches may only read constexprs here.
    return (triton.cdiv(8, meta["BLOCK"]),)


def _dummy_lang_fn():
    """Provides tl globals for patch_lang in patch_run tests."""
    return tl.arange(0, 1)


def _in_thread(fn, *args):
    """Run ``fn(*args)`` on another host thread; its result or exception."""
    outcome: dict = {}

    def target():
        try:
            outcome["result"] = fn(*args)
        except BaseException as exc:
            outcome["error"] = exc

    worker = threading.Thread(target=target)
    worker.start()
    worker.join(30)
    assert not worker.is_alive()
    return outcome


# ======== 1 / 2a-2b: declarations and composition =========


def test_client_declaration_defaults():
    client = _EagerClient()
    assert client.NEEDS_INTERPRETER is True
    assert client.IR_STAGES == frozenset()
    assert client.LAUNCH == "indifferent"
    assert not hasattr(client, "collect_asm")
    assert not hasattr(client, "asm_info")
    for existing in (Sanitizer(), Tracer()):
        assert existing.NEEDS_INTERPRETER is True
        assert existing.LAUNCH == "indifferent"


def test_add_clients_rejects_skip_run_conflict_before_inserting():
    manager = ClientManager([_SkipIRClient(), _EagerClient()])

    with pytest.raises(RuntimeError, match="Trace the kernel twice"):
        manager.add_clients([_IndifferentIRClient(), _RunIRClient()])

    # Nothing from the rejected batch was inserted.
    assert list(manager.clients) == ["ir_skip", "eager"]

    with pytest.raises(RuntimeError, match="LAUNCH='skip'"):
        ClientManager([_RunIRClient(), _SkipIRClient()])


def test_launch_conflict_check_sees_the_resulting_ir_clients():
    # A same-NAME client replaces the one it would otherwise conflict with.
    class _RunInSkipSlot(_IRClient):
        NAME = "ir_skip"
        LAUNCH = "run"

    manager = ClientManager([_SkipIRClient()])
    manager.add_clients([_RunInSkipSlot()])
    assert [type(c) for c in manager.clients.values()] == [_RunInSkipSlot]
    assert manager.launch_policy() == "run"

    # An interpreting client's LAUNCH takes no part in the vote.
    class _EagerSkip(_EagerClient):
        NAME = "eager_skip"
        LAUNCH = "skip"

    manager = ClientManager([_EagerSkip()])
    manager.add_clients([_RunIRClient()])
    assert manager.launch_policy() == "run"


def test_add_clients_keeps_duplicate_rule_and_accepts_indifferent():
    first = _SkipIRClient()
    manager = ClientManager([first, _IndifferentIRClient(), _EagerClient()])
    manager.add_clients([_SkipIRClient()])

    assert list(manager.clients) == ["ir_skip", "ir_indifferent", "eager"]
    assert manager.clients["ir_skip"] is first


def test_add_clients_rejects_unknown_launch_value():
    class _BadIRClient(_IRClient):
        NAME = "ir_bad"
        LAUNCH = "maybe"

    with pytest.raises(ValueError, match="LAUNCH must be one of"):
        ClientManager([_BadIRClient()])


def test_trace_decorator_rejects_conflicting_launch_preferences():
    traced = tilelens.trace(_SkipIRClient())(_make_plain_kernel())

    with pytest.raises(RuntimeError, match="cannot share one trace"):
        tilelens.trace(_RunIRClient())(traced)

    assert list(traced.client_manager.clients) == ["ir_skip"]


def test_client_partition_and_launch_policy():
    eager, ir = _EagerClient(), _IndifferentIRClient()
    manager = ClientManager([eager, ir])

    assert manager.interpreting_clients() == [eager]
    assert manager.ir_clients() == [ir]
    assert manager.launch_policy() == "run"

    manager.add_clients([_SkipIRClient()])
    assert manager.launch_policy() == "skip"

    # Only IR clients' declarations drive the policy.
    assert ClientManager([_EagerClient()]).launch_policy() == "run"


# ======== 2c: patch_warmup =========


class _FakeWarmupJit:
    def __init__(self):
        self.warmups: list[dict] = []

    def warmup(self, *args, **kwargs):
        self.warmups.append(kwargs)
        return "compiled"


def test_patch_warmup_polls_every_client_and_compiles_on_any_vote():
    voter, abstainer = _EagerClient(warmup_vote=True), _OtherEagerClient()
    ir = _IndifferentIRClient()
    manager = ClientManager([voter, abstainer, ir])
    jit_fn = _FakeWarmupJit()

    with manager.patch_warmup(jit_fn):
        ret = jit_fn.warmup(1, grid=(1,), warmup=False)

    assert ret == "compiled"
    assert jit_fn.warmups == [{"grid": (1,)}]
    # No short-circuit after the first True vote; every client votes and
    # every client sees the result.
    assert voter.calls == ["pre_warmup", ("post_warmup", "compiled")]
    assert abstainer.calls == ["pre_warmup", ("post_warmup", "compiled")]
    assert ir.log == ["pre_warmup", "post_warmup"]
    assert "warmup" not in vars(jit_fn)


def test_patch_warmup_skips_compile_without_votes():
    client = _EagerClient()
    manager = ClientManager([client, _OtherEagerClient()])
    jit_fn = _FakeWarmupJit()

    with manager.patch_warmup(jit_fn):
        assert jit_fn.warmup(1, grid=(1,)) is None

    assert jit_fn.warmups == []
    assert client.calls == ["pre_warmup"]


def test_patch_warmup_enters_the_compile_context_only_for_a_real_compile():
    entered: list = []

    @contextmanager
    def compile_context():
        entered.append("enter")
        yield
        entered.append("exit")

    jit_fn = _FakeWarmupJit()
    abstaining = ClientManager([_EagerClient()])
    with abstaining.patch_warmup(jit_fn, compile_context=compile_context):
        assert jit_fn.warmup(1, grid=(1,)) is None
    assert entered == []

    voting = ClientManager([_EagerClient(warmup_vote=True)])
    with voting.patch_warmup(jit_fn, compile_context=compile_context):
        assert jit_fn.warmup(1, grid=(1,)) == "compiled"
    assert entered == ["enter", "exit"]


def test_patch_warmup_scopes_on_two_threads_vote_apart_and_leave_no_gate():
    # Two traces sharing a JITFunction warm up on two host threads, their
    # scopes interleaved: A opens, B opens, A closes, B closes.
    a_client = _EagerClient()
    b_client = _OtherEagerClient(warmup_vote=True)
    a_manager, b_manager = ClientManager([a_client]), ClientManager([b_client])
    jit_fn = _FakeWarmupJit()
    a_in, b_in, a_out = threading.Event(), threading.Event(), threading.Event()
    results: dict = {}

    def thread_a():
        with a_manager.patch_warmup(jit_fn):
            a_in.set()
            b_in.wait(10)
            results["a"] = jit_fn.warmup("a", grid=(1,))
        a_out.set()

    def thread_b():
        a_in.wait(10)
        with b_manager.patch_warmup(jit_fn):
            b_in.set()
            a_out.wait(10)
            results["b"] = jit_fn.warmup("b", grid=(1,))
            # A thread with no scope of its own is not gated.
            results["other"] = _in_thread(jit_fn.warmup, "other")["result"]

    threads = [threading.Thread(target=thread_a), threading.Thread(target=thread_b)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(30)

    # Each call was voted on by its own thread's trace only.
    assert results == {"a": None, "b": "compiled", "other": "compiled"}
    assert a_client.calls == ["pre_warmup"]
    assert b_client.calls == ["pre_warmup", ("post_warmup", "compiled")]
    # The last scope to close removed the gate: an untraced warmup compiles
    # and polls nobody.
    assert "warmup" not in vars(jit_fn)
    assert jit_fn.warmup("untraced", grid=(1,)) == "compiled"
    assert a_client.calls == ["pre_warmup"] and len(b_client.calls) == 2


def test_patch_warmup_shares_one_gate_and_puts_back_what_was_there():
    jit_fn = _FakeWarmupJit()

    def users_warmup(*args, **kwargs):
        return "users"

    jit_fn.warmup = users_warmup
    manager = ClientManager([_EagerClient(warmup_vote=True)])

    with manager.patch_warmup(jit_fn):
        gate = jit_fn.warmup
        with ClientManager([_OtherEagerClient()]).patch_warmup(jit_fn):
            # Nested on one thread: the same gate, the inner scope votes.
            assert jit_fn.warmup is gate
            assert jit_fn.warmup(1, grid=(1,)) is None
        assert jit_fn.warmup is gate
        assert jit_fn.warmup(1, grid=(1,)) == "users"

    assert vars(jit_fn)["warmup"] is users_warmup


def test_patch_warmup_compiles_on_the_real_arguments():
    client = _EagerClient(warmup_vote=True)
    manager = ClientManager([client])
    jit_fn = _FakeWarmupJit()
    mapped: list = []

    def real_args(fn, args, kwargs):
        mapped.append((fn, args, dict(kwargs)))
        return args, {**kwargs, "FN": "untraced"}

    with manager.patch_warmup(jit_fn, real_args=real_args):
        jit_fn.warmup("x", grid=(1,), FN="traced", warmup=False)

    # The votes saw the call as made; only the compile got the mapping.
    assert client.calls[0] == "pre_warmup"
    assert mapped == [(jit_fn, ("x",), {"grid": (1,), "FN": "traced"})]
    assert jit_fn.warmups == [{"grid": (1,), "FN": "untraced"}]


# ======== 2d: patch_run =========


def _first_op():
    frontend = get_frontend("triton")
    namespace, attrs = next(iter(frontend.namespaces.items()))
    attr = next(iter(attrs))
    return frontend, namespace, attr


def test_patch_run_registers_ops_only_for_interpreting_clients():
    eager = _EagerClient()
    # The IR client would raise if asked for op or loop callbacks.
    manager = ClientManager([eager, _IndifferentIRClient()])
    frontend, namespace, attr = _first_op()
    original = frontend.original_ops[namespace][attr]
    store_patches = [
        (ns, name)
        for ns, attrs in frontend.namespaces.items()
        for name, op_type in attrs.items()
        if op_type is Store
    ]
    assert store_patches

    with manager.patch_run(_dummy_lang_fn, frontend_name="triton"):
        for ns, name in store_patches:
            assert getattr(ns, name).before_callback is eager.on_store

    assert getattr(namespace, attr) is original


def test_patch_run_loop_hook_conflict_leaves_nothing_patched():
    manager = ClientManager(
        [
            _EagerClient(loop_overrider=lambda site, idx: idx),
            _SiblingEagerClient(loop_overrider=lambda site, idx: idx),
        ]
    )
    frontend, namespace, attr = _first_op()
    original = frontend.original_ops[namespace][attr]
    scopes_before = len(LANG_PATCH_SCOPES.get("triton", []))

    with pytest.raises(RuntimeError, match="Only one loop_iter overrider"):
        with manager.patch_run(_dummy_lang_fn, frontend_name="triton"):
            pass

    assert getattr(namespace, attr) is original
    assert frontend._patch_calls_scope == 0
    assert not frontend._loop_ast_patched
    assert len(LANG_PATCH_SCOPES.get("triton", [])) == scopes_before
    assert manager._iter_overrider is None


# ======== 2e: interpreter callbacks =========


def test_interpreter_callbacks_reach_only_interpreting_clients():
    eager = _EagerClient()
    manager = ClientManager([eager, _IndifferentIRClient()])
    tensor = torch.zeros(1)

    assert manager.pre_run_callback(_dummy_lang_fn) is True
    assert manager.post_run_callback(_dummy_lang_fn) is True
    manager.arg_callback("x_ptr", tensor, tensor)
    manager.grid_callback((2, 1, 1))
    manager.grid_idx_callback((0, 0, 0))

    assert eager.calls == [
        "pre_run",
        "post_run",
        ("arg", "x_ptr"),
        ("grid", (2, 1, 1)),
        "grid_idx",
    ]
    assert tensor in manager.launch.tensors
    assert manager.launch.grid == (2, 1, 1)


def test_run_votes_without_interpreting_clients_keep_the_grid_running():
    manager = ClientManager([_IndifferentIRClient()])

    assert manager.pre_run_callback(_dummy_lang_fn) is True
    assert manager.post_run_callback(_dummy_lang_fn) is True


# ======== 2f-2g: finalize, begin/abort =========


def test_finalize_runs_every_client_and_reraises_first_exception():
    class _Exiting(_EagerClient):
        NAME = "exiting"

        def finalize(self):
            super().finalize()
            raise SystemExit(3)

    class _Failing(_SiblingEagerClient):
        NAME = "failing"

        def finalize(self):
            self.finalized = True
            raise ValueError("second failure")

    record = object()
    exiting, healthy, failing = (
        _Exiting(),
        _OtherEagerClient(records=[record]),
        _Failing(),
    )
    manager = ClientManager([exiting, healthy, failing])

    with pytest.raises(SystemExit) as info:
        manager.finalize()

    assert info.value.code == 3
    assert "finalize" in healthy.calls
    assert failing.finalized
    assert manager.launch.records == [record]


def test_begin_and_abort_fan_out_to_every_client():
    log: list = []
    eager, ir = _EagerClient(), _IndifferentIRClient(log)
    manager = ClientManager([eager, ir])
    call = _call()

    manager.begin_launch(call)
    manager.abort_launch(KeyError("x"))

    assert eager.calls == ["begin", ("abort", KeyError)]
    assert log == ["begin", ("abort", KeyError)]
    assert ir.launch_calls == [call]


def test_each_launch_gets_its_own_launch_record():
    manager = ClientManager([_EagerClient(records=["record"])])
    manager.begin_launch(_call())
    first = manager.launch
    manager.arg_callback("x_ptr", torch.zeros(1), None)
    manager.finalize()

    manager.begin_launch(_call())

    assert manager.launch is not first
    assert manager.launch.records == [] and not manager.launch.tensors
    assert first.records == ["record"] and len(first.tensors) == 1


def test_abort_hook_failure_never_masks_the_launch_exception():
    class _BrokenAbort(_EagerClient):
        NAME = "broken_abort"

        def abort_launch(self, exc):
            raise RuntimeError("abort hook failed")

    log: list = []
    manager = ClientManager([_BrokenAbort(), _IndifferentIRClient(log)])
    launch_exc = ValueError("launch failed")

    if hasattr(launch_exc, "add_note"):
        manager.abort_launch(launch_exc)
        assert any("abort hook failed" in n for n in launch_exc.__notes__)
    else:
        with pytest.warns(RuntimeWarning, match="abort hook failed"):
            manager.abort_launch(launch_exc)
    # Every client still got the abort.
    assert log == [("abort", ValueError)]


def test_abort_hook_interrupt_propagates_after_every_client():
    class _Interrupting(_EagerClient):
        NAME = "interrupting"

        def abort_launch(self, exc):
            raise KeyboardInterrupt

    log: list = []
    manager = ClientManager([_Interrupting(), _IndifferentIRClient(log)])
    launch_exc = ValueError("launch failed")

    with pytest.raises(KeyboardInterrupt) as info:
        manager.abort_launch(launch_exc)

    assert info.value.__cause__ is launch_exc
    assert log == [("abort", ValueError)]


def test_no_abort_after_finalize_started():
    log: list = []
    manager = ClientManager([_IndifferentIRClient(log)])
    manager.begin_launch(_call())
    manager.finalize()

    manager.abort_launch(SystemExit(3))

    assert log == ["begin", "finalize"]


def test_begin_failure_aborts_exactly_the_clients_that_began():
    class _BrokenBegin(_IndifferentIRClient):
        fail = True

        def begin_launch(self, call):
            super().begin_launch(call)
            if self.fail:
                raise KeyError("begin failed")

    log: list = []
    first, broken, last = _SkipIRClient(log), _BrokenBegin(log), _EagerClient()
    manager = ClientManager([first, broken, last])

    with pytest.raises(KeyError):
        manager.begin_launch(_call())

    # The failing client began (and may hold partial state); `last` never did.
    assert log == ["begin", "begin", ("abort", KeyError), ("abort", KeyError)]
    assert last.calls == []
    # No launch was left open, so another host thread may begin one.
    broken.fail = False
    assert "error" not in _in_thread(manager.begin_launch, _call())


def test_begin_launch_refuses_another_threads_launch_without_touching_it():
    log: list = []
    manager = ClientManager([_SkipIRClient(log)])
    manager.begin_launch(_call())
    launch = manager.launch

    refused = _in_thread(manager.begin_launch, _call())["error"]
    # A stray abort from that thread does not reach this launch either.
    _in_thread(manager.abort_launch, refused)

    assert isinstance(refused, RuntimeError)
    assert "another host thread" in str(refused)
    assert manager.launch is launch
    assert log == ["begin"]

    # Once this launch ends, the other thread may begin the next one.
    manager.finalize()
    assert "error" not in _in_thread(manager.begin_launch, _call())
    assert log == ["begin", "finalize", "begin"]


# ======== 2h: ir_capture =========


def test_ir_capture_skip_compiles_without_launching(_fake_host_compile):
    log: list = []
    ir, eager = _SkipIRClient(log), _EagerClient()
    manager = ClientManager([ir, eager])
    jit_fn = _FakeJit(log)
    x = torch.zeros(10)

    with manager.ir_capture(jit_fn):
        assert "run" in vars(jit_fn)
        ret = jit_fn.run(x, 10, grid=_grid, warmup=False, BLOCK=4, num_warps=2)

    assert "run" not in vars(jit_fn)
    assert log == ["compile", "before", "after"]
    # One host compile, for the default target (D26), through the stages
    # the IR client declares.
    target = default_ir_target()
    assert _fake_host_compile == [(jit_fn, target, frozenset({"ttir"}))]
    (event,) = ir.events
    assert event.target == target
    assert ret is event.kernel
    assert event.jit_fn is jit_fn
    assert event.args == (x, 10)
    assert dict(event.kwargs) == {"BLOCK": 4, "num_warps": 2}
    assert dict(event.bound_args) == {"x_ptr": x, "n": 10, "BLOCK": 4}
    assert event.grid is _grid
    assert event.resolved_grid == (3, 1, 1)
    assert event.launched is False
    assert event.specialization == "hash-4"
    assert "ttir" in event.kernel.asm
    # D5 surface, and no IR event for the interpreting peer. The binding
    # never adds tensors (D23): an interpreted run's arg_callback records
    # its own.
    assert not manager.launch.tensors
    assert manager.launch.grid == (3, 1, 1)
    assert "before_launch" not in eager.calls


def test_ir_capture_run_policy_launches_after_before_launch():
    log: list = []
    ir = _RunIRClient(log)
    manager = ClientManager([ir])
    jit_fn = _FakeJit(log)

    with manager.ir_capture(jit_fn):
        ret = jit_fn.run(torch.zeros(4), 4, grid=(1,), warmup=False)

    assert ret == "launched"
    assert log == ["compile", "before", "launch", "after"]
    assert ir.events[0].launched is True
    assert ir.events[0].resolved_grid == (1, 1, 1)
    assert dict(ir.events[0].bound_args)["BLOCK"] == 4  # default applied


def test_ir_capture_warmup_call_never_launches():
    log: list = []
    manager = ClientManager([_RunIRClient(log)])
    jit_fn = _FakeJit(log)

    with manager.ir_capture(jit_fn):
        jit_fn.run(torch.zeros(4), 4, grid=None, warmup=True)

    assert log == ["compile", "before", "after"]
    assert manager.clients["ir_run"].events[0].launched is False
    assert manager.clients["ir_run"].events[0].resolved_grid is None


def test_ir_capture_restores_on_error_and_does_not_double_wrap():
    log: list = []
    manager = ClientManager([_RunIRClient(log, raise_in_before=ValueError("stop"))])
    jit_fn = _FakeJit(log)

    with pytest.raises(ValueError, match="stop"):
        with manager.ir_capture(jit_fn):
            wrapper = jit_fn.run
            with manager.ir_capture(jit_fn):
                assert jit_fn.run is wrapper
            assert jit_fn.run is wrapper
            jit_fn.run(torch.zeros(4), 4, grid=(1,), warmup=False)

    # before_launch raised: no launch, no after_launch, wrapper removed.
    assert log == ["compile", "before"]
    assert "run" not in vars(jit_fn)


def test_a_failing_host_compile_never_stops_a_real_launch():
    """The host compile is not the device's: a launching call still
    launches, and the JIT's own compile decides (D25)."""
    log: list = []
    skip = ClientManager([_SkipIRClient(log)])
    jit_fn = _FakeJit(log, compile_error=_static_assert_failure())
    with skip.ir_capture(jit_fn):
        assert jit_fn.run(torch.zeros(4), 4, grid=(1,), warmup=False) is None
    assert log == ["compile", ("compile_failed", CompileTimeAssertionFailure)]

    log.clear()
    run = ClientManager([_RunIRClient(log)])
    jit_fn = _FakeJit(log, compile_error=_static_assert_failure())
    with run.ir_capture(jit_fn):
        assert jit_fn.run(torch.zeros(4), 4, grid=(1,), warmup=False) == "launched"
    assert log == [
        "compile",
        ("compile_failed", CompileTimeAssertionFailure),
        "launch",
    ]


def test_ir_capture_delivers_each_specialization_once_per_launch_mode():
    log: list = []
    ir = _RunIRClient(log)
    manager = ClientManager([ir])
    jit_fn = _FakeJit(log)
    x = torch.zeros(4)

    with manager.ir_capture(jit_fn, compile_only=True):
        jit_fn.run(x, 4, grid=(1,), warmup=True, BLOCK=4)
    with manager.ir_capture(jit_fn):
        # e.g. an autotuner benchmarking two configs, then launching one.
        for block in (4, 4, 8, 4):
            jit_fn.run(x, 4, grid=(1,), warmup=False, BLOCK=block)

    assert [(e.specialization, e.launched) for e in ir.events] == [
        ("hash-4", False),
        ("hash-4", True),
        ("hash-8", True),
    ]
    assert log.count("launch") == 4  # every call still launched

    # The next traced launch reports its specializations again.
    manager.begin_launch(_call(capture=True))
    with manager.ir_capture(jit_fn):
        jit_fn.run(x, 4, grid=(1,), warmup=False, BLOCK=4)
    assert [(e.specialization, e.launched) for e in ir.events] == [("hash-4", True)]


class _Opaque:
    """A value the binding fingerprint knows nothing about (weakref-able)."""


def test_ir_capture_delivers_each_binding_of_a_specialization():
    """D22: calls compiling to one kernel ("hash-4") are told apart by their
    binding fingerprint, never by tensor data."""
    ir = _SkipIRClient()
    manager = ClientManager([ir])
    jit_fn = _FakeJit()
    x, y = torch.zeros(8), torch.zeros(8)
    opaque, items = _Opaque(), [1]

    def delivered(*args, grid=(1,), **kwargs) -> bool:
        before = len(ir.events)
        jit_fn.run(*args, grid=grid, warmup=True, **kwargs)
        return len(ir.events) > before

    with manager.ir_capture(jit_fn, compile_only=True):
        assert delivered(x, 4)
        assert not delivered(x, 4)  # the same call again
        assert not delivered(x, 4, grid=_grid)  # a callable grid: (1, 1, 1)
        assert delivered(x, 5)  # a scalar's value
        assert delivered(x, 4.0)  # ... and type
        assert delivered(x, 4, grid=(2,))  # the grid
        assert delivered(x, 4, num_warps=8)  # a kwarg (compile option)
        assert delivered(y, 4)  # a tensor's data_ptr
        assert delivered(x[:4], 4)  # ... shape
        assert delivered(x[::2], 4)  # ... strides
        assert delivered(x.view(torch.int32), 4)  # ... dtype
        x.add_(1)
        assert not delivered(x, 4)  # never its data
        assert delivered(x, (4, 5))  # a tuple, item by item
        assert not delivered(x, (4, 5))
        assert delivered(x, opaque)  # anything else by identity
        assert not delivered(x, opaque)
        assert delivered(x, items)
        assert delivered(x, [1])  # equal, but another object

    assert {e.specialization for e in ir.events} == {"hash-4"}
    assert [e.resolved_grid for e in ir.events][:4] == [(1, 1, 1)] * 3 + [(2, 1, 1)]


class _ConstexprJit(_FakeJit):
    """A _FakeJit whose BLOCK is a tl.constexpr parameter."""

    params = [
        SimpleNamespace(name=name, is_constexpr=name == "BLOCK")
        for name in _FakeJit.signature.parameters
    ]


class _FreshDType:
    """Equal to every other instance in all but identity, as a tl.dtype a
    heuristic builds per call is (the fake kernel's hash holds the repr)."""

    def __repr__(self):
        return "fp32"


def test_a_constexpr_argument_counts_only_through_the_specialization():
    """Triton hashes a constexpr argument into the kernel, so the binding
    fingerprint leaves it out: an equal but fresh constexpr object per call
    adds no binding, passed by keyword or positionally; a non-constexpr
    argument still counts by identity."""
    ir = _SkipIRClient()
    manager = ClientManager([ir])
    jit_fn = _ConstexprJit()
    x = torch.zeros(8)

    def delivered(*args, **kwargs) -> bool:
        before = len(ir.events)
        jit_fn.run(*args, grid=(1,), warmup=True, **kwargs)
        return len(ir.events) > before

    with manager.ir_capture(jit_fn, compile_only=True):
        assert delivered(x, 4, BLOCK=_FreshDType())  # compiles "hash-fp32"
        assert not delivered(x, 4, BLOCK=_FreshDType())
        assert delivered(x, 5, BLOCK=_FreshDType())  # a runtime argument
        assert delivered(x, 4, _FreshDType())  # "hash-4": BLOCK is no kwarg
        assert not delivered(x, 4, _FreshDType())
        opaque = _Opaque()
        assert delivered(x, opaque, BLOCK=_FreshDType())
        assert delivered(x, _Opaque(), BLOCK=_FreshDType())

    specializations = [e.specialization for e in ir.events]
    assert specializations == ["hash-fp32"] * 2 + ["hash-4"] + ["hash-fp32"] * 2
    # Each event still carries the call's own constexpr object.
    assert all(isinstance(e.bound_args["BLOCK"], _FreshDType) for e in ir.events)


def test_a_pinned_value_outlives_its_call_only_until_the_launch_ends():
    """An unknown value's identity token keeps the object alive for the
    launch, so a fresh object per call never reuses a delivered id; once
    the launch ends (finalize or abort), nothing holds it."""

    class _Forgetful(_SkipIRClient):
        def before_launch(self, event):
            self.log.append(event.bound_args["n"].__class__.__name__)

    for end in ("finalize", "abort"):
        ir = _Forgetful()
        manager = ClientManager([ir])
        jit_fn = _FakeJit()
        manager.begin_launch(_call(capture=True))
        refs = []
        with manager.ir_capture(jit_fn, compile_only=True):
            for _ in range(3):
                value = _Opaque()
                refs.append(weakref.ref(value))
                jit_fn.run(torch.zeros(4), value, grid=(1,), warmup=True)
                del value
        # Three distinct objects, three events: none was freed mid-launch.
        assert ir.log.count("_Opaque") == 3
        assert all(ref() is not None for ref in refs)
        if end == "finalize":
            manager.finalize()
        else:
            manager.abort_launch(RuntimeError("launch failed"))
        gc.collect()
        assert all(ref() is None for ref in refs)


def test_compile_only_window_reports_failures_as_data():
    log: list = []
    ir = _RunIRClient(log)
    manager = ClientManager([ir])
    x = torch.zeros(4)
    error = RuntimeError("the front end failed")
    broken = _FakeJit(log, compile_error=error)

    with manager.ir_capture(broken, compile_only=True) as window:
        assert broken.run(x, 4, grid=(1,), warmup=True) is None
    assert (window.compiled, window.failures) == (0, [error])

    assert ir.events == []
    (failure,) = ir.failures
    assert failure.error is error and failure.kernel is None
    assert failure.specialization is None and failure.launched is False
    assert failure.target == default_ir_target()
    assert dict(failure.bound_args) == {"x_ptr": x, "n": 4, "BLOCK": 4}


def test_a_kernel_the_device_could_not_load_is_still_delivered():
    """D25: nothing is loaded, so a config too big for some device (the
    fake's _init_handles would raise) is analyzed like any other."""
    ir = _SkipIRClient()
    manager = ClientManager([ir])
    jit_fn = _FakeJit()
    with manager.ir_capture(jit_fn, compile_only=True) as window:
        kernel = jit_fn.run(torch.zeros(4), 4, grid=(1,), warmup=True)
    assert window.compiled == 1 and ir.failures == []
    (event,) = ir.events
    assert event.kernel is kernel and event.specialization == "hash-4"


def test_a_failing_config_is_reported_once_per_launch():
    """A launch window's benchmark call of a config the compile-only pass
    already reported is no news; another call, or the next launch, is."""
    log: list = []
    ir = _RunIRClient(log)
    manager = ClientManager([ir])
    broken = _FakeJit(log, compile_error=_static_assert_failure())
    x = torch.zeros(4)

    with manager.ir_capture(broken, compile_only=True) as window:
        broken.run(x, 4, grid=(1,), warmup=True, BLOCK=8)
    with manager.ir_capture(broken):
        for block in (8, 8, 16):
            assert broken.run(x, 4, grid=(1,), warmup=False, BLOCK=block) == "launched"

    assert [dict(f.kwargs)["BLOCK"] for f in ir.failures] == [8, 16]
    assert len(window.failures) == 1
    assert log.count("launch") == 3
    manager.begin_launch(_call(capture=True))
    with manager.ir_capture(broken, compile_only=True):
        broken.run(x, 4, grid=(1,), warmup=True, BLOCK=8)
    assert [dict(f.kwargs)["BLOCK"] for f in ir.failures] == [8]


class _TTGIRClient(_SkipIRClient):
    NAME = "ir_ttgir"
    IR_STAGES = frozenset({"ttgir"})


class _HipIRClient(_SkipIRClient):
    NAME = "ir_hip"

    def __init__(self, log=None):
        super().__init__(log)
        self.ir_target = "hip:gfx942"


def test_each_target_compiles_once_and_reaches_only_its_clients(_fake_host_compile):
    """D26: one host compile per distinct target, through the latest stage
    its clients declare; each client sees only its own target's events."""
    from triton.backends.compiler import GPUTarget

    cuda89, gfx942 = GPUTarget("cuda", 89, 32), GPUTarget("hip", "gfx942", 64)
    ttir, hip, ttgir = _SkipIRClient(), _HipIRClient(), _TTGIRClient()
    manager = ClientManager([ttir, hip, ttgir])
    jit_fn = _FakeJit(compile_error=None)

    with manager.ir_capture(jit_fn, compile_only=True):
        jit_fn.run(torch.zeros(4), 4, grid=(1,), warmup=True)

    assert _fake_host_compile == [
        (jit_fn, cuda89, frozenset({"ttir", "ttgir"})),
        (jit_fn, gfx942, frozenset({"ttir"})),
    ]
    assert [e.target for e in ttir.events] == [cuda89]
    assert [e.target for e in ttgir.events] == [cuda89]
    assert ttir.events[0] is ttgir.events[0]
    assert [e.target for e in hip.events] == [gfx942]

    # A target set explicitly to the default's value shares its compile.
    _fake_host_compile.clear()
    hip.ir_target = cuda89
    manager.begin_launch(_call(capture=True))
    with manager.ir_capture(jit_fn, compile_only=True):
        jit_fn.run(torch.zeros(4), 4, grid=(1,), warmup=True)
    assert [t for _, t, _ in _fake_host_compile] == [cuda89]
    assert [e.target for e in hip.events] == [cuda89]


def test_the_configured_target_is_the_default(monkeypatch, _fake_host_compile):
    from triton.backends.compiler import GPUTarget

    monkeypatch.setattr(tilelens_config, "ir_target", "cuda:90")
    ir = _SkipIRClient()
    manager = ClientManager([ir])
    jit_fn = _FakeJit()
    with manager.ir_capture(jit_fn, compile_only=True):
        jit_fn.run(torch.zeros(4), 4, grid=(1,), warmup=True)
    assert ir.events[0].target == GPUTarget("cuda", 90, 32)

    # A spec naming no target is an error before anything compiles, not
    # a compile failure.
    monkeypatch.setattr(tilelens_config, "ir_target", "cuda:sm90")
    with pytest.raises(ValueError, match="TILELENS_IR_TARGET.*'cuda:sm90'"):
        with manager.ir_capture(jit_fn, compile_only=True):
            pass
    assert len(_fake_host_compile) == 1 and ir.failures == []


class _MisspelledStageClient(_SkipIRClient):
    NAME = "ir_misspelled"
    IR_STAGES = frozenset({"TTIR"})


def test_an_ir_stage_no_kernel_holds_is_refused_before_compiling(
    _fake_host_compile,
):
    """An IR_STAGES name the target's kernels never hold is the client's
    bug: a ValueError before anything compiles, never a silent compile of
    the whole pipeline or a compile failure."""
    from triton.backends.compiler import GPUTarget

    manager = ClientManager([_MisspelledStageClient()])
    jit_fn = _FakeJit()
    with pytest.raises(
        ValueError, match=r"_MisspelledStageClient.IR_STAGES: .*\['TTIR'\]"
    ):
        with manager.ir_capture(jit_fn, compile_only=True):
            pass
    assert _fake_host_compile == []
    # Names are checked against the client's own target: "sass" is CUDA's.
    manager.compiler.check_stages(GPUTarget("cuda", 80, 32), {"sass"})
    with pytest.raises(ValueError, match=r"\['sass'\] are no stage .*hip:gfx942"):
        manager.compiler.check_stages(GPUTarget("hip", "gfx942", 64), {"sass"})


def test_an_invalid_configured_target_fails_the_launch(fake_compile, monkeypatch):
    monkeypatch.setattr(tilelens_config, "ir_target", "sm80")
    log: list = []
    traced = tilelens.trace(_SkipIRClient(log))(_make_plain_kernel())
    calls = fake_compile(traced.jit_fn)

    with pytest.raises(ValueError, match="no IR target"):
        traced[(1,)](torch.zeros(4), torch.zeros(4), 4, BLOCK=4)
    assert calls == [] and log == ["begin", ("abort", ValueError)]


def test_ir_capture_refuses_another_owner_and_ignores_other_threads():
    log: list = []
    first = ClientManager([_SkipIRClient(log)])
    second = ClientManager([_RunIRClient()])
    jit_fn = _FakeJit(log)
    results: list = []

    with first.ir_capture(jit_fn):
        with pytest.raises(RuntimeError, match="already being captured"):
            with second.ir_capture(jit_fn):
                pass
        worker = threading.Thread(
            target=lambda: results.append(
                jit_fn.run(torch.zeros(4), 4, grid=(1,), warmup=False)
            )
        )
        worker.start()
        worker.join()

    # The other thread's call went straight to the original run.
    assert results == ["launched"]
    assert log == ["launch"]


def test_ir_capture_compiles_and_launches_on_the_real_arguments():
    ir = _RunIRClient()
    manager = ClientManager([ir])
    received: list = []

    class _RecordingJit(_FakeJit):
        def run(self, *args, grid, warmup, **kwargs):
            received.append((warmup, args, dict(kwargs)))
            return super().run(*args, grid=grid, warmup=warmup, **kwargs)

        def fake_compile(self, *args, **kwargs):
            received.append((True, args, dict(kwargs)))
            return super().fake_compile(*args, **kwargs)

    jit_fn = _RecordingJit()
    x = torch.zeros(4)

    def real_args(fn, args, kwargs):
        assert fn is jit_fn
        return (args[0], 8), {**kwargs, "BLOCK": 8}

    with manager.ir_capture(jit_fn, real_args=real_args):
        jit_fn.run(x, "traced", grid=(1,), warmup=False)

    assert [(w, a[1], k) for w, a, k in received] == [
        (True, 8, {"BLOCK": 8}),
        (False, 8, {"BLOCK": 8}),
    ]
    # The event describes the call as made; the kernel is the compiled one.
    (event,) = ir.events
    assert event.args == (x, "traced") and dict(event.kwargs) == {}
    assert event.specialization == "hash-8"


@pytest.fixture
def patched_language():
    """An interpreted traced launch's language patch, as if active on
    another host thread."""
    scopes = LANG_PATCH_SCOPES.setdefault("triton", [])
    scope = object()
    scopes.append(scope)
    yield
    scopes.remove(scope)


def test_real_compiles_refuse_while_the_language_is_patched(patched_language):
    log: list = []
    ir = _RunIRClient(log)
    manager = ClientManager([ir])
    jit_fn = _FakeJit(log)
    x = torch.zeros(4)

    # A compile reports the refusal as data (no compile ran: its own type,
    # a RuntimeError) ...
    with manager.ir_capture(jit_fn, compile_only=True) as window:
        assert jit_fn.run(x, 4, grid=(1,), warmup=True) is None
    (failure,) = ir.failures
    assert failure.error is window.failures[0]
    assert isinstance(failure.error, LanguagePatchedError)
    assert isinstance(failure.error, RuntimeError)
    assert "language patched" in str(failure.error)
    # ... a real launch raises it (the same call's compile failure is not
    # reported again), and so does a voted warmup.
    with manager.ir_capture(jit_fn):
        with pytest.raises(RuntimeError, match="language patched"):
            jit_fn.run(x, 4, grid=(1,), warmup=False)
    assert len(ir.failures) == 1
    warmup_jit = _FakeWarmupJit()
    with ClientManager([_EagerClient(warmup_vote=True)]).patch_warmup(warmup_jit):
        with pytest.raises(RuntimeError, match="language patched"):
            warmup_jit.warmup(x, grid=(1,))

    # Nothing was compiled or launched.
    assert log == [("compile_failed", LanguagePatchedError)]
    assert warmup_jit.warmups == []


def test_an_ir_only_launch_raises_the_patched_language_refusal(
    fake_compile, patched_language
):
    """No compile's outcome, so not a compile failure D27 lets a launch
    survive: the IR-only launch fails, as it did before D27 (concurrent
    traced launches that mix interpretation and real compiles are
    unsupported). The IR client saw it as data first."""
    log: list = []
    ir = _SkipIRClient(log)
    traced = tilelens.trace(ir)(_make_plain_kernel())
    calls = fake_compile(traced.jit_fn)

    with pytest.raises(LanguagePatchedError, match="language patched"):
        traced[(2,)](torch.zeros(8), torch.zeros(8), 8, BLOCK=4)
    assert log == [
        "begin",
        ("compile_failed", LanguagePatchedError),
        ("abort", LanguagePatchedError),
    ]
    assert calls == []


def test_resolve_grid():
    assert _resolve_grid((2,), {}) == (2, 1, 1)
    assert _resolve_grid((2, 3, 4), {}) == (2, 3, 4)
    assert _resolve_grid(lambda meta: (meta["n"], 2), {"n": 5}) == (5, 2, 1)
    assert _resolve_grid(None, {}) is None
    assert _resolve_grid(lambda meta: (meta["missing"],), {}) is None
    assert _resolve_grid((1, 1, 1, 1), {}) is None


# ======== 3a: runner chain rebuild =========


def test_trace_does_not_mutate_the_users_autotuner_chain():
    user = _make_autotuned_kernel(restore_value=["out_ptr"])
    heuristics, jit_fn = user.fn, user.fn.fn
    before = dict(vars(user))
    before_heuristics = dict(vars(heuristics))

    traced = tilelens.trace(_EagerClient())(user)

    assert vars(user).keys() == before.keys()
    assert all(vars(user)[k] is v for k, v in before.items())
    assert all(vars(heuristics)[k] is v for k, v in before_heuristics.items())
    assert vars(heuristics).keys() == before_heuristics.keys()

    # Interpreter chain: copies of both layers over the InterpretedFunction.
    runner = traced.runner
    assert isinstance(runner, Autotuner) and runner is not user
    assert isinstance(runner.fn, Heuristics) and runner.fn is not heuristics
    assert isinstance(runner.fn.fn, InterpretedFunction)
    assert runner.fn.fn is traced.interpreted_fn
    assert runner._do_bench is KernelTraceSupport.dummy_benchmarker

    # Real chain: copies of both layers over the user's JITFunction.
    real = traced.warmup_runner
    assert isinstance(real, Autotuner) and real is not user and real is not runner
    assert isinstance(real.fn, Heuristics) and real.fn is not heuristics
    assert real.fn.fn is jit_fn is traced.jit_fn
    assert real._do_bench is user._do_bench

    # Per-run state is private to each copy.
    assert len({id(user.cache), id(runner.cache), id(real.cache)}) == 3
    assert runner.cache_results is False and real.cache_results is False


def test_rebuilt_autotuner_restore_hooks_bind_to_the_copy():
    user = _make_autotuned_kernel(restore_value=["out_ptr"])
    traced = tilelens.trace(_EagerClient())(user)
    out = torch.ones(2)
    nargs = {"out_ptr": out}

    traced.runner.pre_hook(nargs)
    out.zero_()
    traced.runner.post_hook(nargs, exception=None)

    assert torch.equal(out, torch.ones(2))
    assert "restore_copies" in vars(traced.runner)
    assert "restore_copies" not in vars(user)


def test_trace_refuses_an_autotuner_whose_default_hooks_it_cannot_isolate():
    user = _make_autotuned_kernel(restore_value=["out_ptr"])
    # As if Triton's default hook were a bound method of the Autotuner: the
    # closure rebinding cannot point it at the traced copy.
    user.pre_hook = types.MethodType(lambda self, kwargs, reset_only=False: 0, user)

    with pytest.raises(RuntimeError, match="could not be isolated"):
        tilelens.trace(_EagerClient())(user)


def test_interpreter_copy_drops_a_benchmarker_cached_on_the_user_autotuner():
    user = _make_autotuned_kernel()
    sentinel = object()
    user.__dict__["do_bench"] = sentinel

    traced = tilelens.trace(_EagerClient())(user)

    assert traced.runner.do_bench is KernelTraceSupport.dummy_benchmarker
    assert user.do_bench is sentinel


def test_autotune_over_heuristics_interprets_every_layer():
    # The interpreter chain used to drop the Heuristics layer under an
    # Autotuner, so the heuristic constexpr never reached the kernel.
    user = _make_autotuned_kernel()
    traced = tilelens.trace(_EagerClient())(user)
    x = torch.arange(8, dtype=torch.float32)
    out = torch.zeros(8)

    traced[_grid8](x, out, 8)

    torch.testing.assert_close(out, x + 1)
    assert user.cache == {}


def test_heuristics_warmup_reaches_the_warmup_votes(fake_compile):
    @triton.heuristics({"BLOCK": lambda args: 4})
    @triton.jit
    def heur_kernel(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
        offs = tl.arange(0, BLOCK)
        tl.store(out_ptr + offs, tl.load(x_ptr + offs))

    client = _EagerClient(warmup_vote=True)
    traced = tilelens.trace(client)(heur_kernel)
    calls = fake_compile(traced.jit_fn)

    ret = traced.warmup(torch.zeros(4), torch.zeros(4), 4, grid=(1,))

    assert isinstance(ret, _FakeKernel)
    assert [c.warmup for c in calls] == [True]
    assert calls[0].kwargs["BLOCK"] == 4
    assert client.calls[0] == "pre_warmup"
    assert client.calls[1][0] == "post_warmup"


# ======== 3b: TritonTrace.run lifecycle =========


def test_ir_only_skip_compiles_every_config_without_running(fake_compile):
    log: list = []
    ir = _SkipIRClient(log)
    traced = tilelens.trace(ir)(_make_autotuned_kernel())
    calls = fake_compile(traced.jit_fn)
    x = torch.arange(8, dtype=torch.float32)
    out = torch.zeros(8)

    traced[_grid](x, out, 8)

    assert torch.equal(out, torch.zeros(8))
    assert [c.warmup for c in calls] == [True, True]
    assert log == ["begin", "before", "after", "before", "after", "finalize"]
    (events,) = ir.finalized
    assert [e.kwargs["BLOCK"] for e in events] == [4, 8]
    assert [e.kwargs["EVEN"] for e in events] == [True, True]
    assert [e.resolved_grid for e in events] == [(2, 1, 1), (1, 1, 1)]
    assert len({e.specialization for e in events}) == 2
    assert not any(e.launched for e in events)
    (call,) = ir.launch_calls
    assert call.jit_fn is traced.jit_fn and call.capture is True
    assert call.args == (x, out, 8) and dict(call.kwargs) == {} and call.grid is _grid
    # The fake compile is back in place; the capture wrapper is gone.
    assert "run" in vars(traced.jit_fn)
    assert not getattr(traced.jit_fn.run, "_tilelens_ir_capture", False)


def test_ir_only_run_launches_through_the_real_runner(fake_compile):
    ir = _RunIRClient()
    traced = tilelens.trace(ir)(_make_plain_kernel())
    calls = fake_compile(traced.jit_fn)

    ret = traced[(2,)](torch.zeros(8), torch.zeros(8), 8, BLOCK=4)

    assert isinstance(ret, _FakeKernel)
    # Compile-only pass, then the launch window's host compile and the real
    # launch.
    assert [c.warmup for c in calls] == [True, True, False]
    (events,) = ir.finalized
    assert [e.launched for e in events] == [False, True]
    assert {e.specialization for e in events} == {ret.hash}
    assert events[1].resolved_grid == (2, 1, 1)
    assert events[1].grid == (2,)


def test_run_policy_reports_every_config_on_every_launch(fake_compile):
    ir = _RunIRClient()
    user = _make_autotuned_kernel(do_bench=_fake_bench)
    traced = tilelens.trace(ir)(user)
    fake_compile(traced.jit_fn)
    x, out = torch.zeros(8), torch.zeros(8)

    traced[_grid](x, out, 8)
    traced[_grid](x, out, 8)

    first, second = ir.finalized
    # Independent of the autotune cache and of benchmark timing (D3).
    for events in (first, second):
        assert [e.kwargs["BLOCK"] for e in events if not e.launched] == [4, 8]
    # Real launches: benchmarking launches each config once; the second,
    # cached launch only the winner.
    assert sorted(e.kwargs["BLOCK"] for e in first if e.launched) == [4, 8]
    assert [e.kwargs["BLOCK"] for e in second if e.launched] == [4]
    assert user.cache == {}


@pytest.mark.parametrize(
    "failing",
    [
        # A config Autotuner._bench drops when it fails like this,
        {8: _static_assert_failure},
        # every config,
        {4: _static_assert_failure, 8: _static_assert_failure},
        # an error the autotuner does not tolerate.
        {8: lambda: ValueError("bad config")},
    ],
    ids=["one-config", "every-config", "untolerated-error"],
)
def test_ir_only_compile_failures_never_fail_the_launch(fake_compile, failing):
    """D27: a config that fails to compile for the IR target is data for
    the IR clients, whatever the error and even when no config compiled; the
    skipped launch returns as it does when every config compiles (None for
    an autotuned kernel)."""
    log: list = []
    ir = _SkipIRClient(log)
    traced = tilelens.trace(ir)(_make_autotuned_kernel())
    fake_compile(
        traced.jit_fn,
        compile_error=lambda kw: failing[kw["BLOCK"]]()
        if kw["BLOCK"] in failing
        else None,
    )
    x, out = torch.zeros(8), torch.zeros(8)

    assert traced[_grid](x, out, 8) is None

    assert log[0] == "begin" and log[-1] == "finalize"
    assert not [e for e in log if isinstance(e, tuple) and e[0] == "abort"]
    (events,) = ir.finalized
    assert [e.kwargs["BLOCK"] for e in events] == [
        b for b in (4, 8) if b not in failing
    ]
    # Every failing config reached the IR client as data, once.
    assert sorted(f.kwargs["BLOCK"] for f in ir.failures) == sorted(failing)


def test_a_failed_host_compile_ends_the_launch_normally(fake_compile):
    """D27 for a plain kernel: its only config failed, so the skipped launch
    returns None (the host-compiled kernel it returns otherwise), and the
    next launch compiles as if nothing had happened."""
    log: list = []
    ir = _SkipIRClient(log)
    traced = tilelens.trace(ir)(_make_plain_kernel())
    calls = fake_compile(traced.jit_fn, fail_first=True)
    args = (torch.zeros(8), torch.zeros(8), 8)

    assert traced[(2,)](*args, BLOCK=4) is None
    assert log == ["begin", ("compile_failed", RuntimeError), "finalize"]
    assert not getattr(traced.jit_fn.run, "_tilelens_ir_capture", False)

    log.clear()
    assert isinstance(traced[(2,)](*args, BLOCK=4), _FakeKernel)

    assert log == ["begin", "before", "after", "finalize"]
    assert [len(events) for events in ir.finalized] == [0, 1]
    assert len(calls) == 2


def test_under_run_the_device_decides_after_a_failed_host_compile(fake_compile):
    """Under "run" a failed host compile does not stop the real launch,
    which compiles its own kernel for the device (here the fake device's
    compile succeeds); the failure is reported once."""
    ir = _RunIRClient()
    traced = tilelens.trace(ir)(_make_plain_kernel())
    calls = fake_compile(
        traced.jit_fn, compile_error=lambda kw: ValueError("not for this target")
    )

    ret = traced[(2,)](torch.zeros(8), torch.zeros(8), 8, BLOCK=4)

    assert isinstance(ret, _FakeKernel)
    # Compile-only pass, the launch window's host compile, the real launch.
    assert [c.warmup for c in calls] == [True, True, False]
    assert ir.finalized == [[]]
    (failure,) = ir.failures
    assert isinstance(failure.error, ValueError) and not failure.launched


def test_mixed_trace_compiles_for_ir_and_interprets_for_eager(fake_compile):
    log: list = []
    ir, eager = _SkipIRClient(log), _EagerClient()
    traced = tilelens.trace(ir)(_make_plain_kernel())
    traced = tilelens.trace(eager)(traced)
    calls = fake_compile(traced.jit_fn)
    x = torch.arange(8, dtype=torch.float32)
    out = torch.zeros(8)

    traced[(2,)](x, out, 8, BLOCK=4)

    # IR client: one compile-only event, no real launch.
    assert [c.warmup for c in calls] == [True]
    (events,) = ir.finalized
    assert [e.launched for e in events] == [False]
    # Interpreting client: the full interpreted run, which wrote the output.
    assert eager.stores == 2
    assert eager.calls.count("pre_run") == 2
    assert "finalize" in eager.calls
    torch.testing.assert_close(out, x + 1)
    # Both clients vote on the legacy warmup.
    assert "pre_warmup" in eager.calls and "pre_warmup" in log


def test_mixed_trace_survives_ir_compile_failures(fake_compile):
    # E.g. a kernel the host compile rejects, which the interpreter runs.
    log: list = []
    ir, eager = _SkipIRClient(log), _EagerClient()
    traced = tilelens.trace(eager)(tilelens.trace(ir)(_make_plain_kernel()))
    fake_compile(
        traced.jit_fn,
        compile_error=lambda kw: RuntimeError("the host compile failed"),
    )
    x = torch.arange(8, dtype=torch.float32)
    out = torch.zeros(8)

    traced[(2,)](x, out, 8, BLOCK=4)

    torch.testing.assert_close(out, x + 1)
    assert eager.stores == 2 and "finalize" in eager.calls
    assert ir.finalized == [[]]
    assert [type(f.error) for f in ir.failures] == [RuntimeError]
    assert not any(isinstance(e, tuple) and e[0] == "abort" for e in log)


# ======== D28: a call that does not bind raises, as untraced =========


def _bind_failure():
    """The TypeError the host compile raises for a call missing ``n``,
    marked as a bind failure (tilelens.core.host_compile.bind_failed)."""
    from tilelens.core import host_compile

    exc = TypeError("dynamic_func() missing 1 required positional argument: 'n'")
    host_compile._mark_bind_failed(exc)
    return exc


@pytest.mark.parametrize("compile_only", [True, False])
@pytest.mark.parametrize("ir_cls", [_SkipIRClient, _RunIRClient])
def test_ir_capture_raises_a_call_that_does_not_bind(ir_cls, compile_only):
    """A bind failure is the call's own error, which JITFunction.run raises
    on any device: ir_capture raises that very exception, compile-only or
    not, under either launch policy. No compile_failed event, no real
    launch, and the capture is removed."""
    log: list = []
    manager = ClientManager([ir_cls(log)])
    unbound = _bind_failure()
    jit_fn = _FakeJit(log, compile_error=unbound)

    with manager.ir_capture(jit_fn, compile_only=compile_only) as window:
        with pytest.raises(TypeError) as raised:
            jit_fn.run(torch.zeros(4), grid=(1,), warmup=compile_only)

    assert raised.value is unbound
    assert log == ["compile"]
    assert (window.compiled, window.failures) == (0, [])
    assert "run" not in vars(jit_fn)


@pytest.mark.parametrize("ir_cls", [_SkipIRClient, _RunIRClient])
@pytest.mark.parametrize(
    "make, launch",
    [
        (_make_plain_kernel, lambda k, x, out: k[(2,)](x, out, 8, BLOCK=4)),
        (
            lambda: _make_autotuned_kernel(do_bench=_fake_bench),
            lambda k, x, out: k[_grid](x, out, 8),
        ),
    ],
    ids=["plain", "autotuned"],
)
def test_a_traced_call_that_does_not_bind_raises_as_untraced(
    fake_compile, ir_cls, make, launch
):
    """D28: an IR-only launch raises the bind failure (D27's "the program
    goes on" is for kernel compile failures only), from the first compile
    of the compile-only pass: the IR client's launch is aborted, never
    finalized, nothing launches or is recorded, and the next launch runs as
    if nothing had happened."""
    log: list = []
    ir = ir_cls(log)
    traced = tilelens.trace(ir)(make())
    unbound = _bind_failure()
    failing = [unbound]
    calls = fake_compile(
        traced.jit_fn, compile_error=lambda kw: failing.pop() if failing else None
    )
    x, out = torch.zeros(8), torch.zeros(8)
    launches = len(trace_module.launches)

    with pytest.raises(TypeError) as raised:
        launch(traced, x, out)

    assert raised.value is unbound
    assert log == ["begin", ("abort", TypeError)]
    assert [c.warmup for c in calls] == [True]
    assert len(trace_module.launches) == launches
    assert not getattr(traced.jit_fn.run, "_tilelens_ir_capture", False)

    log.clear()
    launch(traced, x, out)
    assert log[0] == "begin" and log[-1] == "finalize"
    assert len(trace_module.launches) == launches + 1


def test_a_mixed_trace_raises_a_call_that_does_not_bind_before_interpreting(
    fake_compile,
):
    """D28 in a mixed trace (D4b): the IR clients' compile pass raises the
    bind failure before the interpreter runs (which would fail on the same
    call too), so the untraced JIT's error is the one raised; every
    client's launch is aborted."""
    log: list = []
    ir, eager = _SkipIRClient(log), _EagerClient()
    traced = tilelens.trace(eager)(tilelens.trace(ir)(_make_plain_kernel()))
    unbound = _bind_failure()
    fake_compile(traced.jit_fn, compile_error=lambda kw: unbound)
    x, out = torch.arange(8, dtype=torch.float32), torch.zeros(8)

    with pytest.raises(TypeError) as raised:
        traced[(2,)](x, out, 8, BLOCK=4)

    assert raised.value is unbound
    assert log == ["begin", ("abort", TypeError)]
    assert eager.calls == ["begin", ("abort", TypeError)]
    assert eager.stores == 0 and torch.equal(out, torch.zeros(8))


def test_a_launch_failing_in_finalize_is_not_aborted(fake_compile):
    class _ExitingIR(_SkipIRClient):
        def finalize(self):
            super().finalize()
            raise SystemExit(3)

    log: list = []
    traced = TritonTrace(_make_plain_kernel(), _ExitingIR(log))
    traced.add_client(_IndifferentIRClient(log))
    fake_compile(traced.jit_fn)

    with pytest.raises(SystemExit):
        traced[(1,)](torch.zeros(4), torch.zeros(4), 4, BLOCK=4)

    # Each launch ends in finalize or abort, never both.
    assert log.count("finalize") == 2
    assert not any(isinstance(e, tuple) and e[0] == "abort" for e in log)


def test_each_traced_launch_is_recorded_separately(fake_compile):
    class _VerdictIR(_SkipIRClient):
        def finalize(self):
            super().finalize()
            return [f"verdict-{len(self.finalized)}"]

    traced = tilelens.trace(_VerdictIR())(_make_plain_kernel())
    fake_compile(traced.jit_fn)
    before = len(trace_module.launches)
    a, b = torch.zeros(8), torch.zeros(16)

    traced[(2,)](a, a, 8, BLOCK=4)
    traced[(4,)](b, b, 16, BLOCK=4)

    first, second = trace_module.launches[before:]
    assert first is not second
    assert first.records == ["verdict-1"] and second.records == ["verdict-2"]
    # Launch.grid from the IR binding, per launch; an IR-only launch
    # records no tensors (D23).
    assert not first.tensors and not second.tensors
    assert (first.grid, second.grid) == ((2, 1, 1), (4, 1, 1))


def test_cli_shape_traces_the_autotuner_over_an_inner_trace(fake_compile):
    # The CLI wrappers turn every @triton.jit into a TritonTrace and wrap the
    # Autotuner built on it again: TritonTrace(Autotuner(TritonTrace(JIT))).
    inner_ir, outer_ir = _SkipIRClient(), _SkipIRClient()
    jit_fn = _make_plain_kernel()
    inner = TritonTrace(jit_fn, inner_ir)
    user = triton.autotune(
        configs=[triton.Config({"BLOCK": 4}), triton.Config({"BLOCK": 8})],
        key=["n"],
    )(inner)
    before = dict(vars(user))
    outer = TritonTrace(user, outer_ir)
    fake_compile(outer.jit_fn)
    x, out = torch.zeros(8), torch.zeros(8)

    outer[_grid](x, out, 8)

    assert outer.jit_fn is inner.jit_fn is jit_fn
    (events,) = outer_ir.finalized
    assert [e.kwargs["BLOCK"] for e in events] == [4, 8]
    assert inner_ir.log == []
    assert torch.equal(out, torch.zeros(8))
    assert vars(user).keys() == before.keys()
    assert all(vars(user)[k] is v for k, v in before.items())


def test_a_skipped_launch_returns_the_kernel_only_without_an_autotuner(fake_compile):
    @triton.heuristics({"BLOCK": lambda args: 4})
    @triton.jit
    def heur_kernel(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
        offs = tl.arange(0, BLOCK)
        tl.store(out_ptr + offs, tl.load(x_ptr + offs))

    args = (torch.zeros(8), torch.zeros(8), 8)
    plain = tilelens.trace(_SkipIRClient())(_make_plain_kernel())
    fake_compile(plain.jit_fn)
    heur = tilelens.trace(_SkipIRClient())(heur_kernel)
    fake_compile(heur.jit_fn)
    tuned = tilelens.trace(_SkipIRClient())(_make_autotuned_kernel())
    fake_compile(tuned.jit_fn)

    # One config: the kernel the untraced launch would return.
    assert isinstance(plain[(2,)](*args, BLOCK=4), _FakeKernel)
    assert isinstance(heur[(2,)](*args), _FakeKernel)
    # Autotuned: no config was picked.
    assert tuned[_grid](*args) is None


def test_launch_grid_is_the_grid_the_launch_ran_with(fake_compile):
    args = (torch.zeros(8), torch.zeros(8), 8)
    # Run policy: benchmarking launches every config, then the winner (ties
    # pick the first); Launch.grid is the winner's grid.
    run = tilelens.trace(_RunIRClient())(_make_autotuned_kernel(do_bench=_fake_bench))
    calls = fake_compile(run.jit_fn)
    run[_grid](*args)
    assert [c.kwargs["BLOCK"] for c in calls if not c.warmup] == [4, 8, 4]
    assert trace_module.launches[-1].grid == (2, 1, 1)

    # Skip policy: nothing launched; configs that disagree on the grid
    # leave it open, a grid they share is the launch's.
    skip = tilelens.trace(_SkipIRClient())(_make_autotuned_kernel())
    fake_compile(skip.jit_fn)
    skip[_grid](*args)
    assert trace_module.launches[-1].grid is None
    skip[(3,)](*args)
    assert trace_module.launches[-1].grid == (3, 1, 1)


def test_launch_tensors_have_one_representation_per_launch(fake_compile):
    x = torch.arange(8, dtype=torch.float32)
    out = torch.zeros(8)

    ir_only = tilelens.trace(_SkipIRClient())(_make_plain_kernel())
    fake_compile(ir_only.jit_fn)
    ir_only[(2,)](x, out, 8, BLOCK=4)
    # None (D23): holding the caller's device tensors would keep them alive
    # after the launch; the IR clients' records hold the facts they need.
    assert not trace_module.launches[-1].tensors
    assert trace_module.launches[-1].grid == (2, 1, 1)

    mixed = tilelens.trace(_EagerClient())(
        tilelens.trace(_SkipIRClient())(_make_plain_kernel())
    )
    fake_compile(mixed.jit_fn)
    mixed[(2,)](x, out, 8, BLOCK=4)
    # Only the interpreter's host copies, whose addresses the eager
    # clients' records use; not the caller's tensors on top.
    tensors = trace_module.launches[-1].tensors
    assert len(tensors) == 2
    assert not {id(t) for t in tensors} & {id(x), id(out)}


class _ForgetfulSkipIR(_SkipIRClient):
    """Keeps nothing of a launch past its hooks; ``fail`` makes after_launch
    raise (under "run" only after a real launch)."""

    NAME = "forgetful_skip"

    def __init__(self, fail=False):
        super().__init__()
        self.fail = fail

    def begin_launch(self, call):
        self.log.append("begin")

    def before_launch(self, event):
        self.log.append("before")

    def after_launch(self, event):
        if self.fail and (event.launched or self.LAUNCH == "skip"):
            raise RuntimeError("after_launch failed")

    def finalize(self):
        self.log.append("finalize")
        return []


class _ForgetfulRunIR(_ForgetfulSkipIR):
    NAME = "forgetful_run"
    LAUNCH = "run"


@pytest.mark.parametrize("outcome", ["finalized", "aborted"])
@pytest.mark.parametrize("ir_cls", [_ForgetfulSkipIR, _ForgetfulRunIR])
@pytest.mark.parametrize(
    "make",
    [_make_plain_kernel, lambda: _make_autotuned_kernel(do_bench=_fake_bench)],
    ids=["plain", "autotuned"],
)
def test_an_ir_only_launch_keeps_no_caller_tensor(make, ir_cls, outcome, monkeypatch):
    """D23: once an IR-only launch has ended and tilelens.clear() ran,
    nothing of the trace refers to the caller's tensors or to a grid
    callable closing over them: not the Launch the manager keeps, its dedup
    keys or last-launch grid, nor the trace's copy of the autotuner."""
    ir = ir_cls(fail=outcome == "aborted")
    traced = tilelens.trace(ir)(make())

    def run(*args, grid, warmup, **kwargs):
        return _FakeKernel(kwargs.get("BLOCK", 4))  # keeps no argument

    _install_fake_run(monkeypatch, traced.jit_fn, run)
    kwargs = {"BLOCK": 4} if make is _make_plain_kernel else {}

    def launch() -> list[weakref.ref]:
        # The caller's references end with this frame.
        x, out = torch.zeros(8), torch.zeros(8)

        def grid(meta):
            return (triton.cdiv(x.numel(), meta["BLOCK"]),)

        if outcome == "aborted":
            with pytest.raises(RuntimeError, match="after_launch failed"):
                traced[grid](x, out, 8, **kwargs)
        else:
            traced[grid](x, out, 8, **kwargs)
            assert not trace_module.launches[-1].tensors
            assert ir.log[-1] == "finalize"
        return [weakref.ref(x), weakref.ref(out)]

    refs = launch()
    tilelens.clear()
    gc.collect()

    assert [ref() for ref in refs] == [None, None]


def test_an_interrupted_benchmark_keeps_no_restore_value_clone(monkeypatch):
    """D23: Autotuner._bench runs the post_hook that drops a benchmark
    call's restore_value clones only for an Exception, so a Ctrl+C in the
    call skips it; the trace's autotuner copy keeps neither the clones nor
    the caller's tensors once the launch has ended."""
    clones: list[weakref.ref] = []

    class _Interrupting(_ForgetfulRunIR):
        def before_launch(self, event):
            super().before_launch(event)
            if event.launched:  # a benchmark call: the user hits Ctrl+C
                copies = traced.ir_runner.restore_copies
                clones.extend(weakref.ref(clone) for clone in copies.values())
                raise KeyboardInterrupt

    traced = tilelens.trace(_Interrupting())(
        _make_autotuned_kernel(do_bench=_fake_bench, restore_value=["out_ptr"])
    )

    def run(*args, grid, warmup, **kwargs):
        return _FakeKernel(kwargs.get("BLOCK", 4))  # keeps no argument

    _install_fake_run(monkeypatch, traced.jit_fn, run)

    def launch() -> list[weakref.ref]:
        x, out = torch.zeros(8), torch.zeros(8)
        with pytest.raises(KeyboardInterrupt):
            traced[_grid](x, out, 8)
        return [weakref.ref(x), weakref.ref(out)]

    refs = launch()
    tilelens.clear()
    gc.collect()

    assert len(clones) == 1  # the pre_hook's clone of out_ptr
    assert [ref() for ref in refs + clones] == [None, None, None]


@pytest.mark.parametrize("mixed", [False, True], ids=["eager", "mixed"])
def test_an_interpreted_launch_that_raises_keeps_no_caller_tensor(mixed, monkeypatch):
    """Once an interpreted autotuned launch has raised (here in a config
    pre_hook), the trace's autotuner copies keep none of the caller's
    tensors: a mixed launch (D4b) is held to the IR-only guarantee (D23),
    and an eager one to the same."""

    def boom(nargs):
        raise RuntimeError("pre_hook boom")

    @triton.autotune(configs=[triton.Config({"BLOCK": 4}, pre_hook=boom)], key=["n"])
    @triton.jit
    def add_one_hooked(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
        offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n
        tl.store(out_ptr + offs, tl.load(x_ptr + offs, mask=mask) + 1, mask=mask)

    kernel: object = add_one_hooked
    if mixed:
        kernel = tilelens.trace(_ForgetfulSkipIR())(kernel)
    traced = tilelens.trace(_EagerClient())(kernel)

    def run(*args, grid, warmup, **kwargs):
        return _FakeKernel(kwargs.get("BLOCK", 4))  # keeps no argument

    _install_fake_run(monkeypatch, traced.jit_fn, run)

    def launch() -> list[weakref.ref]:
        x, out = torch.zeros(8), torch.zeros(8)
        with pytest.raises(RuntimeError, match="pre_hook boom"):
            traced[_grid8](x, out, 8)
        return [weakref.ref(x), weakref.ref(out)]

    refs = launch()
    tilelens.clear()
    gc.collect()

    assert traced.runner.nargs is None
    assert [ref() for ref in refs] == [None, None]


def test_a_concurrent_launch_of_one_trace_is_refused_before_it_begins(monkeypatch):
    log: list = []
    ir = _SkipIRClient(log)
    traced = tilelens.trace(ir)(_make_autotuned_kernel())
    second: list = []

    def run(*args, grid, warmup, **kwargs):
        if kwargs["BLOCK"] == 8 and not second:
            # Between this launch's two configs, launch again from another
            # host thread.
            second.append(_in_thread(traced[_grid], *args))
        return _FakeKernel(kwargs["BLOCK"])

    _install_fake_run(monkeypatch, traced.jit_fn, run)
    traced[_grid](torch.zeros(8), torch.zeros(8), 8)

    (outcome,) = second
    assert isinstance(outcome["error"], RuntimeError)
    assert "another host thread" in str(outcome["error"])
    # The first launch saw neither a second begin nor an abort, and kept
    # every config.
    assert log == ["begin", "before", "after", "before", "after", "finalize"]
    (events,) = ir.finalized
    assert [e.kwargs["BLOCK"] for e in events] == [4, 8]


def test_ir_compiles_are_not_gated_by_an_instance_warmup_patch(
    fake_compile, monkeypatch
):
    # E.g. a vote gate someone left on the JITFunction, declining every
    # compile: IR compiles go through JITFunction's own warmup.
    def declining_warmup(*args, **kwargs):
        return None

    args = (torch.zeros(8), torch.zeros(8), 8)
    for traced, grid, kwargs in (
        (tilelens.trace(_SkipIRClient())(_make_plain_kernel()), (2,), {"BLOCK": 4}),
        (tilelens.trace(_SkipIRClient())(_make_autotuned_kernel()), _grid, {}),
    ):
        calls = fake_compile(traced.jit_fn)
        monkeypatch.setattr(traced.jit_fn, "warmup", declining_warmup, raising=False)
        traced[grid](*args, **kwargs)
        (ir,) = traced.client_manager.ir_clients()
        assert ir.finalized[-1] and calls


def test_ir_launch_compiles_on_untraced_arguments(fake_compile):
    helper = tilelens.trace(_SiblingEagerClient())(triton.jit(_unwrap_leaf))

    @triton.jit
    def apply(x_ptr, out_ptr, n, FN: tl.constexpr, BLOCK: tl.constexpr):
        offs = tl.arange(0, BLOCK)
        tl.store(out_ptr + offs, tl.load(x_ptr + offs))

    ir = _RunIRClient()
    traced = tilelens.trace(ir)(apply)
    calls = fake_compile(traced.jit_fn)
    traced[(1,)](torch.zeros(4), torch.zeros(4), 4, FN=helper, BLOCK=4)

    # Compile-only pass, the launch window's compile, the launch: all on the
    # JITFunction.
    assert [c.warmup for c in calls] == [True, True, False]
    assert all(c.kwargs["FN"] is helper.jit_fn for c in calls)
    # Events describe the call as made.
    assert all(e.kwargs["FN"] is helper for e in ir.finalized[0])

    # The interpreted launches' voted warmup compiles on them too.
    voter = _EagerClient(warmup_vote=True)
    traced = tilelens.trace(voter)(apply)
    calls = fake_compile(traced.jit_fn)
    traced[(1,)](torch.zeros(4), torch.zeros(4), 4, FN=helper, BLOCK=4)
    assert [c.kwargs["FN"] for c in calls] == [helper.jit_fn]
    assert voter.calls[:2] == ["begin", "pre_warmup"]
    assert isinstance(voter.calls[2][1], _FakeKernel)  # post_warmup


@pytest.mark.parametrize(
    "ir_cls, expect_written", [(_SkipIRClient, False), (_RunIRClient, True)]
)
def test_ir_clients_without_a_jit_function(ir_cls, expect_written):
    ir = ir_cls()
    traced = TritonTrace(InterpretedFunction(_make_plain_kernel().fn), ir)
    assert traced.jit_fn is None
    x = torch.arange(8, dtype=torch.float32)
    out = torch.zeros(8)

    traced[(2,)](x, out, 8, BLOCK=4)

    # No compiled kernel, so no events; "run" interprets the whole grid.
    assert ir.finalized == [[]]
    (call,) = ir.launch_calls
    assert call.jit_fn is None and call.capture is False
    if expect_written:
        torch.testing.assert_close(out, x + 1)
    else:
        assert torch.equal(out, torch.zeros(8))


@pytest.mark.parametrize("trace_cls", [GluonTrace, NKITrace])
def test_gluon_and_nki_traces_run_the_launch_lifecycle(trace_cls):
    # Built without __init__: the Gluon simulation and NKI are not importable
    # everywhere, and a skip-only trace never reaches them.
    class _BrokenBegin(_SkipIRClient):
        def begin_launch(self, call):
            super().begin_launch(call)
            raise KeyError("begin failed")

    log: list = []
    traced = trace_cls.__new__(trace_cls)
    TraceInterface.__init__(traced, _SkipIRClient(log))

    # Only IR clients, one of them skipping: nothing is interpreted.
    assert traced[(2,)](torch.zeros(4)) is None
    assert log == ["begin", "finalize"]
    (call,) = traced.client_manager.clients["ir_skip"].launch_calls
    assert call.jit_fn is None and call.capture is False and call.grid == (2,)

    log = []
    traced = trace_cls.__new__(trace_cls)
    TraceInterface.__init__(traced, _BrokenBegin(log))
    with pytest.raises(KeyError):
        traced[(2,)](torch.zeros(4))
    assert log == ["begin", ("abort", KeyError)]


# ======== 3c: unwrapped trace globals =========


def _unwrap_leaf(x):
    return x + 1


def _unwrap_helper(x):
    return _unwrap_traced_leaf(x)  # noqa: F821


def test_unwrapped_trace_globals_swaps_only_what_the_kernel_reaches():
    module_globals = globals()
    leaf = tilelens.trace(_SiblingEagerClient())(triton.jit(_unwrap_leaf))
    helper = tilelens.trace(_SiblingEagerClient())(triton.jit(_unwrap_helper))
    unrelated = tilelens.trace(_SiblingEagerClient())(_make_plain_kernel())
    # A package whose `api` re-exports `impl`'s binding.
    pkg = types.ModuleType("tilelens_test_pkg")
    pkg.api = types.ModuleType("tilelens_test_pkg.api")
    pkg.impl = types.ModuleType("tilelens_test_pkg.impl")
    pkg.api.helper = pkg.impl.helper = helper
    kernel_globals = {"helper": helper, "pkg": pkg, "unrelated": unrelated, "keep": 1}
    exec("def kernel_fn():\n    return helper, pkg.api.helper\n", kernel_globals)
    module_globals["_unwrap_traced_leaf"] = leaf
    module_globals["_unwrap_traced_unrelated"] = unrelated

    try:
        with pytest.raises(KeyError):
            with _unwrapped_trace_globals(kernel_globals["kernel_fn"]):
                # Direct, through a two-level module path, and transitively
                # through the traced helper's own globals.
                assert kernel_globals["helper"] is helper.jit_fn
                assert pkg.api.helper is helper.jit_fn
                assert module_globals["_unwrap_traced_leaf"] is leaf.jit_fn
                # Not reachable from the kernel's code: left alone.
                assert pkg.impl.helper is helper
                assert kernel_globals["unrelated"] is unrelated
                assert module_globals["_unwrap_traced_unrelated"] is unrelated
                assert kernel_globals["keep"] == 1
                raise KeyError("restore on error")

        assert kernel_globals["helper"] is helper
        assert pkg.api.helper is helper
        assert module_globals["_unwrap_traced_leaf"] is leaf
    finally:
        module_globals.pop("_unwrap_traced_leaf", None)
        module_globals.pop("_unwrap_traced_unrelated", None)


def test_unwrapped_trace_globals_covers_names_bound_to_traced_defaults():
    # Triton's dependency walker resolves a parameter default expression
    # (``FN=helper``) in the kernel's globals; the name is not in co_names.
    helper = tilelens.trace(_SiblingEagerClient())(triton.jit(_unwrap_leaf))
    kernel_globals = {"helper": helper, "alias": helper, "other": helper.jit_fn}
    exec("def kernel_fn(FN=helper):\n    return FN\n", kernel_globals)

    with _unwrapped_trace_globals(kernel_globals["kernel_fn"]):
        assert kernel_globals["helper"] is helper.jit_fn
        assert kernel_globals["alias"] is helper.jit_fn
    assert kernel_globals["helper"] is kernel_globals["alias"] is helper


def test_untraced_call_args_unwraps_arguments_tuples_and_defaults():
    helper = tilelens.trace(_SiblingEagerClient())(triton.jit(_unwrap_leaf))
    raw = triton.jit(_unwrap_leaf)
    # A trace without a JITFunction has nothing to unwrap to.
    no_jit = TritonTrace(InterpretedFunction(_unwrap_leaf), _SiblingEagerClient())

    @triton.jit
    def kernel(
        x_ptr,
        FN: tl.constexpr,
        FNS: tl.constexpr,
        ACT: tl.constexpr = helper,
        N: tl.constexpr = 1,
    ):
        pass

    x = torch.zeros(1)
    args, kwargs = _untraced_call_args(
        kernel, (x, helper), {"FNS": (raw, helper), "num_warps": 4}
    )
    assert args[0] is x and args[1] is helper.jit_fn
    assert kwargs["FNS"][0] is raw and kwargs["FNS"][1] is helper.jit_fn
    # The traced default is passed explicitly; plain defaults are left alone.
    assert kwargs["ACT"] is helper.jit_fn
    assert "N" not in kwargs and kwargs["num_warps"] == 4

    fns = (raw, no_jit)
    args, kwargs = _untraced_call_args(kernel, (x, no_jit, fns, raw), {})
    assert args[1] is no_jit and args[2] is fns and args[3] is raw
    assert kwargs == {}


def test_a_traced_function_refuses_to_run_outside_an_interpreted_launch():
    helper = tilelens.trace(_SiblingEagerClient())(triton.jit(_unwrap_leaf))
    program_id = tl.program_id

    # E.g. a real compile reaching the trace as a plain Python callee.
    with pytest.raises(TypeError, match="outside a traced launch's interpreter"):
        helper(1)

    # The interpreter never ran, so triton.language was never patched.
    assert tl.program_id is program_id


def _kernel_calling_nested_leaf(x_ptr, out_ptr, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    tl.store(out_ptr + offs, _nested_traced_leaf(tl.load(x_ptr + offs)))  # noqa: F821


def test_nested_traced_calls_compare_only_interpreting_clients(fake_compile):
    # The CLI shape: the helper is traced with the eager client only, the
    # kernel with it and an IR client, which takes no part in the
    # interpreted run (D4b).
    module_globals = globals()
    module_globals["_nested_traced_leaf"] = tilelens.trace(_EagerClient())(
        triton.jit(_unwrap_leaf)
    )
    try:
        traced = tilelens.trace(_EagerClient())(
            tilelens.trace(_SkipIRClient())(triton.jit(_kernel_calling_nested_leaf))
        )
        fake_compile(traced.jit_fn)
        x = torch.arange(4, dtype=torch.float32)
        out = torch.zeros(4)

        traced[(1,)](x, out, BLOCK=4)

        torch.testing.assert_close(out, x + 1)
    finally:
        module_globals.pop("_nested_traced_leaf", None)
