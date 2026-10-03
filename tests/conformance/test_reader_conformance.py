"""D10b: the TTIR reader's static footprint equals what Triton's interpreter touches.

For every (kernel, launch) of the corpus (``_corpus.py``):

1. the launch is compiled to TTIR in a clean subprocess (``_ttir_capture``)
   the way IR mode compiles it: on the host, by
   ``tilelens.core.host_compile``, for the default target
   ``GPUTarget("cuda", 89, 32)`` (D25, D26), CPU only or not;
2. :func:`tilelens.ir.ttir_reader.parse_ttir` reads it, and the concrete
   evaluator (``_static_footprint``) enumerates every program id x arange
   lane x loop iteration of the AccessGraph with unbounded integers, the
   reader's width obligations checked concretely;
3. the same launch runs under ``TRITON_INTERPRET=1`` in another subprocess
   (``_interp_footprint``), its loads / stores / atomics instrumented;
4. per access site ``(base argument, kind, source line)`` the
   ``(pid_0, pid_1, pid_2, element offset)`` points must be EQUAL: each
   program's own footprint, no subset slack, either way. The site's
   ``AccessEvent.elem_bits`` must equal the element width of the pointers
   the interpreter accessed there (the byte model is offset x width), and
   ``graph.pid_axes`` the axes of the TTIR's ``tt.get_program_id`` /
   ``tt.get_num_programs`` ops.

Accesses the model over-approximates by design (``mask_dropped``,
``guarded``) are skipped, sites that read an atomic observation or whose
launch breaks a width obligation are excluded; each is reported and pinned
by the ``EXPECTED`` table below, as are the reader's refusals: a refusal, an
exclusion or a case that exercises no access, anywhere the table does not
list it, fails, and so does a reader crash (on its own case). For the audit
regression kernels (``r_*``) and the representational limits (``j_*``) the
requirement is "the reader refuses with the listed kind OR the footprints
conform", so reverting a reader fix makes this suite fail; a case the
interpreter cannot run (inline asm) must refuse.

The capture child also compiles each launch with the JIT itself, for the
same target (a stand-in driver reports a cuda:89 device, so no GPU is
needed), and test_host_ttir_is_the_jit_ttir checks explicitly that the
host compile is the JIT's: the same hash and the same TTIR text, so the
host compile's binder and stage cut cannot change what the reader sees.

Version policy: this suite runs on the INSTALLED Triton. A Triton minor
release may be added to ``tilelens.core.config.TESTED_TRITON_VERSIONS``
(the IR-mode gate, D10b) only when this suite passes on it (the
host-vs-JIT check included; neither needs a GPU), run with
``TILELENS_IR_ALLOW_UNTESTED_TRITON=1``. Without that override, on
a release outside the table the cases still run but are expected to fail
(``xfail``, not strict): a CI job that installs the newest Triton stays
green and still shows how far it conforms. The capture uses tilelens's own
host compile (the private Triton API it leans on is feature-checked there),
so this suite validates that path on each release too; the oracle leans on
the interpreter builder's memory methods and ``grid_idx``; adapting it to a
new release is part of adding it.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
import traceback
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Mapping

import pytest

from tilelens.core.config import TESTED_TRITON_VERSIONS, untested_triton_version
from tilelens.ir import _mlir_walk
from tilelens.ir import ttir_reader
from tilelens.ir.ttir_reader import TTIRKind, UnsupportedTTIR, parse_ttir

from . import _corpus
from . import _interp_footprint as interp_side
from . import _static_footprint as S
from . import _ttir_capture as capture_side

NAMES = [c.name for c in _corpus.CASES]


@dataclass(frozen=True)
class Expect:
    # The reader may refuse with this kind; otherwise the case must conform.
    refusal: TTIRKind | None = None
    # The reader MUST refuse (with ``refusal``): the interpreter cannot run
    # the kernel, so accepting it can never conform.
    must_refuse: bool = False
    # Accepted: exclusion reason (joined by "+" when one site has several) ->
    # number of excluded sites.
    excluded: Mapping[str, int] = field(default_factory=dict)
    # Accepted: at least one compared site with a non-empty footprint. False
    # only where every site is excluded by design.
    compared: bool = True
    why: str = ""


_OBSERVED_2 = {S.OBSERVED: 2}
EXPECTED: dict[str, Expect] = {
    # ── audit regressions (ir_mode_audit/probes/): refuse with this kind OR conform ──
    "r_p1_variant_delta": Expect(TTIRKind.LOOP_VARIANT_ADVANCE, why="p1: p += k"),
    "r_p2_swap": Expect(TTIRKind.LOOP_VARIANT_ADVANCE, why="p2: p, q = q, p"),
    "r_p3_call_guarded": Expect(TTIRKind.CALL, why="p3: noinline call"),
    "r_p3_call_offset": Expect(TTIRKind.CALL, why="p3: noinline call"),
    "r_p3_call_formals": Expect(TTIRKind.CALL, why="p3: noinline call"),
    # Triton's interpreter cannot run inline asm: accepting these is a
    # reader regression, never a conformance question
    "r_inline_asm_store": Expect(
        TTIRKind.INLINE_ASM, must_refuse=True, why="an impure asm st.global"
    ),
    "r_pure_asm_int_addr": Expect(
        TTIRKind.INLINE_ASM, must_refuse=True, why="a pure asm handed an address"
    ),
    "r_loop_observed_advance": Expect(
        TTIRKind.LOOP_VARIANT_ADVANCE, why="advance by an in-loop observation"
    ),
    "r_int_iterarg_offset": Expect(
        TTIRKind.LOOP_VARIANT_ADVANCE, why="an integer offset carried by the loop"
    ),
    "r_observed_lanes": Expect(
        TTIRKind.INDIRECT_ADDRESS, why="two lanes of one tensor observation"
    ),
    # p4: the reader represents observations by design; its graph-aware
    # mentions_observed must flag the loads / stores (the evaluator checks it
    # against its own walk), the atomic itself is compared.
    "r_p4_observed_direct": Expect(excluded=_OBSERVED_2),
    "r_p4_observed_loop": Expect(excluded=_OBSERVED_2),
    "r_p4_observed_delta": Expect(excluded=_OBSERVED_2),
    # D9: a launch where a width obligation fails is reported, not compared
    # (the *_small / *_pid0 launches of the same kernels are compared).
    "r_trunci_alias_wrap": Expect(
        excluded={S.OBLIGATION: 1}, compared=False, why="trunci(pid * 2**32)"
    ),
    "r_i32_wrap_wrap": Expect(
        excluded={S.OBLIGATION: 1}, compared=False, why="(pid * S) * S wraps"
    ),
    "r_iv_wrap_wrap": Expect(
        excluded={S.OBLIGATION: 1}, compared=False, why="the loop increment wraps"
    ),
    "b_unsigned_cmp_highbit": Expect(
        excluded={S.OBLIGATION: 1},
        compared=False,
        why="cmpi ult reads n = -1 as 2**32 - 1: the unsigned-operand obligation fails",
    ),
    # ── representational limits of the reader: refuse with this kind OR conform ──
    "a_copy_bool": Expect(
        TTIRKind.OTHER,
        why="a *i1 argument is accessed through a tt.bitcast to *i8: the reader reads 1 vs 8 bits as a "
        "width change, though both are one byte in memory",
    ),
    "c_block_ptr_loop": Expect(
        TTIRKind.LOOP_VARIANT_ADVANCE,
        why="an advanced block pointer's offsets are integer iter_args (3.6: rewrite_tensor_pointer; "
        "3.8: the frontend lowers tl.make_block_ptr to pointer arithmetic)",
    ),
    "j_two_loops": Expect(TTIRKind.NESTED_LOOP, why="two sequential scf.for"),
    "j_nested_loops": Expect(TTIRKind.NESTED_LOOP, why="an scf.for in an scf.for"),
    "j_loop_under_if": Expect(TTIRKind.CONTROL_FLOW, why="an scf.for under an scf.if"),
    "j_while": Expect(TTIRKind.CONTROL_FLOW, why="an scf.while"),
    "j_csr_bound": Expect(
        TTIRKind.DATA_DEPENDENT_BOUND, why="loop bounds loaded from memory"
    ),
    "j_gather": Expect(TTIRKind.INDIRECT_ADDRESS, why="an index loaded from memory"),
    # ── accesses the model over-approximates or cannot evaluate concretely ──
    "i_datadep_mask": Expect(excluded={S.MASK_DROPPED: 1}),
    "i_guarded_branch": Expect(excluded={S.GUARDED: 1}),
    "i_atomic_max_float": Expect(
        excluded={S.MASK_DROPPED: 1}, why="two atomics masked by the value's sign"
    ),
    "i_observed_mask_path": Expect(excluded=_OBSERVED_2),
}


def test_expectation_table_names_corpus_cases():
    assert not set(EXPECTED) - set(NAMES)
    assert len(NAMES) == len(set(NAMES))
    assert all(e.refusal is not None for e in EXPECTED.values() if e.must_refuse)


# ─────────────────────────── running the corpus ───────────────────────────


@dataclass
class Outcome:
    record: dict[str, Any]  # the capture child's record
    refusal: UnsupportedTTIR | None = None
    reader_error: str | None = None  # parse_ttir raised something else
    pid_axes: frozenset[int] = frozenset()
    static: S.StaticFootprint | None = None
    static_error: str | None = None
    interp: dict[str, Any] | None = None


@dataclass
class Run:
    outcomes: dict[str, Outcome]
    seconds: dict[str, float]
    # the children's results, as they came (shared between xdist workers)
    records: dict[str, Any]
    interp: dict[str, Any]


def _selected(request) -> list[str]:
    names = []
    for item in request.session.items:
        callspec = getattr(item, "callspec", None)
        if (
            getattr(item, "module", None) is request.module
            and callspec is not None
            and "name" in callspec.params
        ):
            names.append(callspec.params["name"])
    return list(dict.fromkeys(names)) or NAMES


def _run(
    names: list[str],
    records: dict[str, Any] | None = None,
    interp: dict[str, Any] | None = None,
) -> Run:
    """Capture (unless ``records`` are given), read, evaluate and interpret
    (the accepted cases ``interp`` lacks) every case in ``names``."""
    t0 = time.monotonic()
    if records is None:
        records = capture_side.capture(names)
    t1 = time.monotonic()
    outcomes: dict[str, Outcome] = {}
    for name in names:
        out = outcomes[name] = Outcome(records[name])
        if out.record.get("error"):
            continue
        try:
            graph = parse_ttir(out.record["ttir"])
        except UnsupportedTTIR as e:
            out.refusal = e
            continue
        except Exception as e:  # noqa: BLE001 - reported by the case's test
            out.reader_error = (
                f"{type(e).__name__}: {e}\n{traceback.format_exc()[-3000:]}"
            )
            continue
        out.pid_axes = graph.pid_axes
        try:
            out.static = S.static_footprint(
                graph, out.record["params"], out.record["grid"]
            )
        except Exception as e:  # noqa: BLE001 - reported by the case's test
            out.static_error = f"{type(e).__name__}: {e}"
    t2 = time.monotonic()
    accepted = [
        n
        for n, o in outcomes.items()
        if o.static is not None and not EXPECTED.get(n, Expect()).must_refuse
    ]
    interp = dict(interp or {})
    missing = [n for n in accepted if n not in interp]
    if missing:
        interp.update(interp_side.run_interpreter(missing))
    for name in accepted:
        outcomes[name].interp = interp[name]
    t3 = time.monotonic()
    seconds = {"capture": t1 - t0, "static": t2 - t1, "interpreter": t3 - t2}
    return Run(outcomes, seconds, records, interp)


def _shared_result(tmp_path_factory, names: list[str]) -> str | None:
    """Where pytest-xdist workers share the run: each worker collects the
    whole module, so without it every worker that runs any case would
    capture, read and interpret the whole corpus. None outside xdist."""
    if os.environ.get("PYTEST_XDIST_WORKER") is None:
        return None
    digest = hashlib.sha1("\0".join(names).encode()).hexdigest()[:12]
    # the base temp directory's parent is common to one run's workers
    root = tmp_path_factory.getbasetemp().parent
    return str(root / f"tilelens-conformance-{digest}.json")


@pytest.fixture(scope="module")
def conformance(request, tmp_path_factory) -> Run:
    names = _selected(request)
    shared = _shared_result(tmp_path_factory, names)
    if shared is None:
        return _run(names)
    try:
        from filelock import FileLock  # a torch dependency
    except ImportError:
        return _run(names)
    with FileLock(shared + ".lock"):
        if os.path.exists(shared):
            with open(shared, encoding="utf-8") as f:
                done = json.load(f)
            return _run(names, done["records"], done["interp"])
        run = _run(names)
        with open(shared, "w", encoding="utf-8") as f:
            json.dump({"records": run.records, "interp": run.interp}, f)
        return run


def _reasons(static: S.StaticFootprint) -> Counter:
    return Counter(
        "+".join(sorted({e.reason for e in why})) for why in static.excluded.values()
    )


def _describe(key, s: set[S.Point], d: set[S.Point]) -> str:
    only_s, only_d = sorted(s - d), sorted(d - s)
    return (
        f"{key}: (pid_0, pid_1, pid_2, offset) static-only {only_s[:8]} ({len(only_s)}), "
        f"interpreter-only {only_d[:8]} ({len(only_d)}); |static|={len(s)} |interpreter|={len(d)}"
    )


# tt.get_program_id / tt.get_num_programs, custom (``x``) or generic
# (``"() <{axis = 0 : i32}>``) form
_RE_PID_OP = re.compile(
    r"\btt\.get_(?:program_id|num_programs)(?:\s+([xyz])\b|\"\(\)\s*<\{axis\s*=\s*(\d+))"
)


def _ttir_pid_axes(text: str) -> frozenset[int]:
    return frozenset(
        "xyz".index(word) if word else int(num)
        for word, num in _RE_PID_OP.findall(text)
    )


# D10b version policy (see the module docstring)
_UNTESTED = untested_triton_version()


@pytest.mark.xfail(
    _UNTESTED is not None,
    reason=f"Triton {_UNTESTED} is outside TESTED_TRITON_VERSIONS {TESTED_TRITON_VERSIONS}; "
    "set TILELENS_IR_ALLOW_UNTESTED_TRITON=1 to require conformance",
    strict=False,
)
@pytest.mark.parametrize("name", NAMES)
def test_reader_conformance(name, conformance, record_property):
    out = conformance.outcomes[name]
    exp = EXPECTED.get(name, Expect())
    rec = out.record
    props: dict[str, Any] = {
        "source": rec.get("source"),
        "seconds": conformance.seconds,
    }
    record_property("conformance", props)
    assert not rec.get(
        "error"
    ), f"TTIR capture failed: {rec.get('error')}\n{rec.get('traceback', '')}"
    assert out.reader_error is None, f"the reader crashed: {out.reader_error}"

    if out.refusal is not None:
        e = out.refusal
        props["outcome"] = f"refused:{e.kind}"
        assert (
            exp.refusal is not None
        ), f"unexpected refusal ({e.kind}): {e.message} (TTIR line {e.line_no})"
        assert (
            e.kind is exp.refusal
        ), f"refused as {e.kind}, expected {exp.refusal}: {e.message}"
        return

    props["outcome"] = "error"
    assert (
        not exp.must_refuse
    ), f"the reader accepted a kernel it must refuse as {exp.refusal}: {exp.why}"
    axes = _ttir_pid_axes(rec["ttir"])
    assert (
        out.pid_axes == axes
    ), f"graph.pid_axes is {sorted(out.pid_axes)}; the TTIR's program-id ops read axes {sorted(axes)}"
    assert out.static_error is None, f"static evaluation failed: {out.static_error}"
    static, run = out.static, out.interp
    assert static is not None and run is not None
    assert not run["errors"] and not run["wild"], (
        "interpreter run failed:\n" + "\n".join(run["errors"])
    )

    # sites are keyed by line: every one must sit in the kernel's own file
    kernel_file = rec["file"]
    for key, files in static.files.items():
        assert {os.path.realpath(f) for f in files} == {
            kernel_file
        }, f"{key} sits in {files}, not {kernel_file}"
    dynamic: dict[S.SiteKey, set[S.Point]] = {}
    widths: dict[S.SiteKey, set[int]] = {}
    for site in run["sites"]:
        arg, kind, line = site["arg"], site["kind"], site["line"]
        assert (
            site["file"] == kernel_file
        ), f"the interpreter's {kind} of {arg!r} sits in {site['file']}:{line}, not {kernel_file}"
        dynamic.setdefault((arg, kind, line), set()).update(
            tuple(p) for p in site["points"]
        )
        widths.setdefault((arg, kind, line), set()).update(site["bits"])

    compared = (set(static.sites) | set(dynamic)) - set(static.excluded)
    mismatches = [
        _describe(k, static.sites.get(k, set()), dynamic.get(k, set()))
        for k in sorted(compared)
        if static.sites.get(k, set()) != dynamic.get(k, set())
    ]
    exercised = [k for k in compared if dynamic.get(k)]
    reasons = _reasons(static)
    if mismatches:
        outcome = "mismatch"
    elif exercised:
        outcome = "conform"
    else:
        outcome = (
            "not-compared"  # every site excluded (the table says so, or the test fails)
        )
    props.update(
        outcome=outcome,
        compared_sites=len(compared),
        exercised_sites=len(exercised),
        points=sum(len(dynamic.get(k, ())) for k in compared),
        excluded=dict(reasons),
        skipped_accesses=sum(
            e.reason in S.SKIPPED for why in static.excluded.values() for e in why
        ),
        obligation_failures=len(static.obligation_failures),
    )
    assert not mismatches, "footprints differ:\n" + "\n".join(mismatches)
    bad_widths = [
        f"{k}: the reader's elem_bits {sorted(static.bits.get(k, ()))}, "
        f"the interpreter's pointers {sorted(widths[k])}"
        for k in sorted(compared)
        if k in widths and static.bits.get(k) != widths[k]
    ]
    assert not bad_widths, "element widths differ:\n" + "\n".join(bad_widths)
    assert reasons == Counter(exp.excluded), (
        f"excluded sites {dict(reasons)}, expected {dict(exp.excluded)}: "
        + "; ".join(
            f"{k}: {[(e.reason, e.detail) for e in v]}"
            for k, v in static.excluded.items()
        )
    )
    if exp.compared:
        assert (
            exercised
        ), f"no compared site touched memory (compared: {sorted(compared)})"
    else:
        assert (
            not compared
        ), f"expected every site excluded, compared {sorted(compared)}"


# The op tl.debug_barrier() prints, per Triton release: a_debug_ops must
# read it, and the release's reader vocabulary must hold it as inert.
_BARRIER_OP = {"3.6": "gpu.barrier", "3.8": "ttg.barrier"}


@pytest.mark.xfail(
    _UNTESTED is not None,
    reason=f"Triton {_UNTESTED} is outside TESTED_TRITON_VERSIONS {TESTED_TRITON_VERSIONS}; "
    "set TILELENS_IR_ALLOW_UNTESTED_TRITON=1 to require conformance",
    strict=False,
)
def test_the_debug_barrier_case_reads_the_release_barrier(conformance):
    rec = conformance.outcomes["a_debug_ops"].record
    assert not rec.get("error"), rec.get("error")
    release = _mlir_walk.triton_release()[0]
    op = _BARRIER_OP[release]
    assert re.search(rf"^\s*{re.escape(op)}\b", rec["ttir"], re.M), op
    assert op in ttir_reader._VOCABULARIES[release].inert
    assert conformance.outcomes["a_debug_ops"].refusal is None


@pytest.mark.xfail(
    _UNTESTED is not None,
    reason=f"Triton {_UNTESTED} is outside TESTED_TRITON_VERSIONS {TESTED_TRITON_VERSIONS}; "
    "set TILELENS_IR_ALLOW_UNTESTED_TRITON=1 to require conformance",
    strict=False,
)
@pytest.mark.parametrize("name", NAMES)
def test_host_ttir_is_the_jit_ttir(name, conformance, record_property):
    """D25: IR mode reads host-compiled TTIR. The JIT's own compile of the
    same launch for the same target (cuda:89, a stand-in driver's) must be
    the same kernel: the same hash and the same TTIR text."""
    rec = conformance.outcomes[name].record
    if rec.get("error"):
        pytest.skip("the capture failed (test_reader_conformance reports it)")
    same = rec["ttir"] == rec["jit_ttir"] and rec["hash"] == rec["jit_hash"]
    record_property("host_vs_jit", {"same_text": same})
    assert rec["jit_target"] == ["cuda", 89, 32]
    assert rec["hash"] == rec["jit_hash"]
    assert rec["ttir"] == rec["jit_ttir"]
