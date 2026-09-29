"""Smoke of the IR layers end to end: a toy IRClient under tilelens.trace
parses every specialization's TTIR, compiled on the host (D25: CPU tensors,
no GPU), through the real ParseCache and tilelens.ir.ttir_reader.parse_ttir,
and puts its IRVerdict into Launch.records. Each kernel is launched twice;
the second launch must find its texts in the parse cache. Per-layer tests
live in tests/unit/ir/ and tests/end_to_end/test_ir_client.py.
"""

from __future__ import annotations

import importlib
import inspect

import pytest
import torch
import triton
import triton.language as tl

import tilelens
from tilelens.ir import ConfigVerdict, IRClient, IRVerdict, ParseCache, Refusal
from tilelens.ir import _mlir_walk, ttir_reader
from tilelens.ir.ttir_reader import Const, IterArgOffset, Param

trace_module = importlib.import_module("tilelens.core.trace")


def _real_compiles_available() -> bool:
    # Triton imported under TRITON_INTERPRET=1 builds its own standard library
    # as InterpretedFunctions, so nothing can compile for real in-process. No
    # GPU is needed: IR mode compiles on the host (D25).
    import triton.language.standard as tl_standard
    from triton.runtime.jit import JITFunction

    return isinstance(tl_standard.cdiv, JITFunction)


pytestmark = pytest.mark.skipif(
    not _real_compiles_available(),
    reason="Triton was imported under TRITON_INTERPRET=1: nothing compiles in-process",
)


@pytest.fixture(autouse=True)
def _no_driver(unreachable_driver):
    """IR mode needs no GPU (D25): Triton's driver is unreachable here, as on
    a machine without one (where it raises "0 active drivers")."""
    unreachable_driver("IR mode queried Triton's driver")


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


@pytest.fixture
def parsed_texts(monkeypatch):
    """Every text the real reader parses. ParseCache resolves its default
    reader at each lookup, so the spy stands in for it for the whole test."""
    texts: list[str] = []
    parse_ttir = ttir_reader.parse_ttir

    def spy(text):
        texts.append(text)
        return parse_ttir(text)

    monkeypatch.setattr(ttir_reader, "parse_ttir", spy)
    return texts


class _ParsingIR(IRClient):
    """Parses each specialization's TTIR; a refused config makes the launch
    "unsupported" with the first refusal."""

    NAME = "parsing_ir"
    LAUNCH = "skip"
    IR_STAGES = frozenset({"ttir"})

    def __init__(self):
        super().__init__()
        self.parses = ParseCache()
        # Per finalized launch: (specializations, parse outcomes).
        self.launches: list[tuple] = []

    def analyze_launch(self, log):
        assert log.failures == (), log.failures
        specs = log.specializations
        outcomes = [self.parses.get(spec.artifacts.stages["ttir"]) for spec in specs]
        self.launches.append((specs, outcomes))
        per_config = []
        for spec, outcome in zip(specs, outcomes):
            assert outcome.error is None, outcome.error
            if outcome.refusal is None:
                per_config.append(
                    ConfigVerdict(spec.specialization, spec.config, "parsed")
                )
            else:
                refusal = Refusal.from_exception(outcome.refusal)
                per_config.append(
                    ConfigVerdict(spec.specialization, spec.config, "refused", refusal)
                )
        refusals = [c.refusal for c in per_config if c.refusal is not None]
        if refusals:
            return [], IRVerdict(
                self.NAME, "unsupported", refusal=refusals[0], per_config=per_config
            )
        return [], IRVerdict(self.NAME, "parsed", per_config=per_config)

    def on_analysis_error(self, exc):
        return IRVerdict(self.NAME, "error", notes=[f"{type(exc).__name__}: {exc}"])

    def on_refusal(self, refusal):
        return IRVerdict(self.NAME, "unsupported", refusal=refusal)


def _launch_twice(kernel, grid, *args, **kwargs):
    """Trace ``kernel`` with a fresh _ParsingIR and launch it twice; checks
    what every launch must hold and returns the client and the first
    launch's graphs (None for a refused text), in specialization order."""
    ir = _ParsingIR()
    traced = tilelens.trace(ir)(kernel)
    verdicts = []
    for _ in range(2):
        traced[grid](*args, **kwargs)
        verdicts.append(ir.last_verdict)

    for verdict in verdicts:
        assert verdict.status != "error", verdict.notes
    first, second = verdicts
    assert [launch.records for launch in trace_module.launches[-2:]] == [
        [first],
        [second],
    ]
    assert second == first
    (specs, outcomes), (specs_again, outcomes_again) = ir.launches
    assert [s.specialization for s in specs_again] == [s.specialization for s in specs]
    # The second launch is a parse-cache hit: the same outcome objects, a
    # refusal's kind included.
    assert all(a is b for a, b in zip(outcomes_again, outcomes, strict=True))
    return ir, [outcome.graph for outcome in outcomes]


def _line_of(jit_fn, needle: str) -> int:
    """The source line of ``jit_fn`` that contains ``needle``."""
    lines, start = inspect.getsourcelines(jit_fn.fn)
    (offset,) = [i for i, line in enumerate(lines) if needle in line]
    return start + offset


def _vector(n=64, dtype=torch.float32):
    return torch.arange(n, dtype=dtype)


def _make_plain():
    @triton.jit
    def plain(x_ptr, out_ptr, BLOCK: tl.constexpr):
        offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        tl.store(out_ptr + offs, tl.load(x_ptr + offs) + 1)

    return plain


def _make_masked():
    @triton.jit
    def masked(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
        offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n
        tl.store(out_ptr + offs, tl.load(x_ptr + offs, mask=mask) + 1, mask=mask)

    return masked


def _make_row_sum():
    @triton.jit
    def row_sum(x_ptr, out_ptr, n_cols, BLOCK: tl.constexpr):
        row = tl.program_id(0)
        ptrs = x_ptr + row * n_cols + tl.arange(0, BLOCK)
        acc = tl.zeros((BLOCK,), dtype=tl.float32)
        for _ in range(0, n_cols, BLOCK):
            acc += tl.load(ptrs)
            ptrs += BLOCK
        tl.store(out_ptr + row, tl.sum(acc))

    return row_sum


def _make_autotuned():
    @triton.autotune(
        configs=[triton.Config({"BLOCK": b}, num_warps=1) for b in (16, 32)],
        key=["n"],
    )
    @triton.jit
    def masked_tuned(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
        offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n
        tl.store(out_ptr + offs, tl.load(x_ptr + offs, mask=mask) + 1, mask=mask)

    return masked_tuned


def _make_gather():
    @triton.jit
    def gather(x_ptr, idx_ptr, out_ptr, n, BLOCK: tl.constexpr):
        offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n
        idx = tl.load(idx_ptr + offs, mask=mask, other=0)
        tl.store(out_ptr + offs, tl.load(x_ptr + idx, mask=mask), mask=mask)

    return gather


def _make_calls():
    # Triton 3.6 passes only scalars to a noinline function.
    @triton.jit(noinline=True)
    def store_one(out_ptr, i):
        tl.store(out_ptr + i, 1.0)

    @triton.jit
    def calls(out_ptr):
        store_one(out_ptr, tl.program_id(0))

    return calls


def _assert_skipped(out):
    # LAUNCH="skip": compiled and analyzed, never launched.
    assert torch.equal(out, torch.zeros_like(out))


def test_plain_kernel():
    kernel = _make_plain()
    x, out = _vector(), torch.zeros(64)

    ir, (graph,) = _launch_twice(kernel, (4,), x, out, BLOCK=16)

    _assert_skipped(out)
    (config,) = ir.last_verdict.per_config
    assert (config.status, config.config) == ("parsed", {})
    assert trace_module.launches[-1].grid == (4, 1, 1)
    assert graph.kernel_name == "plain"
    assert [(a.kind, a.base_param, a.mask) for a in graph.accesses] == [
        ("load", "x_ptr", None),
        ("store", "out_ptr", None),
    ]
    line = _line_of(kernel, "tl.store(")
    source = (kernel.fn.__code__.co_filename, line)
    assert [(a.loc.file, a.loc.line) for a in graph.accesses] == [source, source]
    assert graph.loop is None and graph.pid_axes == {0}


def test_masked_kernel():
    x, out = _vector(), torch.zeros(64)

    ir, (graph,) = _launch_twice(_make_masked(), (4,), x, out, 64, BLOCK=16)

    _assert_skipped(out)
    assert ir.last_verdict.status == "parsed"
    assert [(a.kind, a.base_param) for a in graph.accesses] == [
        ("load", "x_ptr"),
        ("store", "out_ptr"),
    ]
    assert all(a.mask is not None for a in graph.accesses)
    assert graph.arg("n").int_bits == 32 and graph.loop is None


def test_loop_with_a_pointer_iter_arg():
    x, out = _vector(4 * 64), torch.zeros(4)

    ir, (graph,) = _launch_twice(_make_row_sum(), (4,), x, out, 64, BLOCK=16)

    _assert_skipped(out)
    assert ir.last_verdict.status == "parsed"
    loop = graph.loop
    assert (loop.lower, loop.upper, loop.step) == (Const(0), Param("n_cols"), Const(16))
    (iter_arg,) = graph.iter_args
    assert (iter_arg.base_param, iter_arg.delta) == ("x_ptr", Const(16))
    load, store = graph.accesses
    assert (load.kind, load.in_loop, load.offset) == ("load", True, IterArgOffset(0))
    assert (store.kind, store.in_loop) == ("store", False)
    ((spec,), _), _ = ir.launches
    (binding,) = spec.bindings
    assert dict(binding.params) == {"n_cols": 64}


def test_autotune_parses_both_configs(parsed_texts):
    x, out = _vector(), torch.zeros(64)

    ir, graphs = _launch_twice(
        _make_autotuned(), lambda meta: (triton.cdiv(64, meta["BLOCK"]),), x, out, 64
    )

    _assert_skipped(out)
    per_config = ir.last_verdict.per_config
    assert [(c.status, c.config["BLOCK"]) for c in per_config] == [
        ("parsed", 16),
        ("parsed", 32),
    ]
    assert len({c.specialization for c in per_config}) == 2
    # One parse per config, none on the second launch.
    assert len(parsed_texts) == 2
    assert [g.kernel_name for g in graphs] == ["masked_tuned"] * 2
    # A skipped autotuned launch picks no config, and the configs' grids differ.
    assert trace_module.launches[-1].grid is None


def test_gather_refuses_as_indirect_address(parsed_texts):
    kernel = _make_gather()
    x, out = _vector(), torch.zeros(64)
    idx = torch.arange(64, dtype=torch.int32)

    ir, graphs = _launch_twice(kernel, (4,), x, idx, out, 64, BLOCK=16)

    _assert_skipped(out)
    verdict = ir.last_verdict
    assert (verdict.status, verdict.refusal.kind) == ("unsupported", "indirect-address")
    assert verdict.per_config[0].refusal == verdict.refusal
    assert verdict.refusal.loc.line == _line_of(kernel, "x_ptr + idx")
    assert graphs == [None] and len(parsed_texts) == 1


def test_noinline_call_refuses_as_call(parsed_texts):
    kernel = _make_calls()
    out = torch.zeros(4)

    ir, graphs = _launch_twice(kernel, (4,), out)

    _assert_skipped(out)
    verdict = ir.last_verdict
    assert (verdict.status, verdict.refusal.kind) == ("unsupported", "call")
    assert "store_one" in verdict.refusal.message
    assert verdict.refusal.loc.line == _line_of(kernel, "store_one(")
    assert graphs == [None] and len(parsed_texts) == 1


def _release() -> str:
    return _mlir_walk.triton_release()[0]


def _make_barrier():
    @triton.jit
    def barrier(x_ptr, out_ptr, BLOCK: tl.constexpr):
        offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        v = tl.load(x_ptr + offs)
        tl.debug_barrier()
        tl.store(out_ptr + offs, v)

    return barrier


# The op tl.debug_barrier() prints, per Triton release; each release's
# reader holds its own barrier inert.
_BARRIER_OP = {"3.6": "gpu.barrier", "3.8": "ttg.barrier all"}


def test_debug_barrier_is_inert(parsed_texts):
    x, out = _vector(), torch.zeros(64)

    ir, (graph,) = _launch_twice(_make_barrier(), (4,), x, out, BLOCK=16)

    _assert_skipped(out)
    assert ir.last_verdict.status == "parsed"
    assert [(a.kind, a.base_param) for a in graph.accesses] == [
        ("load", "x_ptr"),
        ("store", "out_ptr"),
    ]
    (text,) = parsed_texts
    assert _BARRIER_OP[_release()] in text


def _make_tuple_args():
    @triton.jit
    def pair_copy(ptrs, n, BLOCK: tl.constexpr):
        offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n
        tl.store(ptrs[1] + offs, tl.load(ptrs[0] + offs, mask=mask), mask=mask)

    return pair_copy


# The names of a tuple parameter's flattened TTIR arguments, per Triton
# release: 3.6 names every leaf by the parameter (the reader refuses two
# parameters of one name), 3.8 by its path.
_TUPLE_LEAVES = {"3.6": None, "3.8": ("ptrs.0", "ptrs.1")}


def test_tuple_parameter_leaves(parsed_texts):
    x, out = _vector(), torch.zeros(64)

    ir, (graph,) = _launch_twice(_make_tuple_args(), (4,), (x, out), 64, BLOCK=16)

    _assert_skipped(out)
    leaves = _TUPLE_LEAVES[_release()]
    verdict = ir.last_verdict
    if leaves is None:
        assert (verdict.status, verdict.refusal.kind) == ("unsupported", "other")
        assert "two parameters named 'ptrs'" in verdict.refusal.message
        assert graph is None
        return
    assert verdict.status == "parsed"
    assert [a.name for a in graph.func_args] == [*leaves, "n"]
    assert [(a.kind, a.base_param) for a in graph.accesses] == [
        ("load", leaves[0]),
        ("store", leaves[1]),
    ]
