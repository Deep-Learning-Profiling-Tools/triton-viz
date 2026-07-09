"""End-to-end tests for the T1 global-memory race track: the TTIR reader's
AccessGraph lowered (under one launch's concrete params) into
TwoCopySymbolicHBSolver records with symbolic pid/grid/arange/loop.

"proved@T1" here means: race-free for this input, on EVERY grid along the
pid axes the kernel reads (unread axes are pinned to 1)."""

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from triton_viz.clients.common.ttir_reader import UnsupportedTTIR, parse_ttir
from triton_viz.clients.race_detector.compiled.client import CompiledRaceDetector
from triton_viz.clients.race_detector.compiled.global_records import (
    GlobalTensor,
    encode_graph,
    encode_graph_t0,
    symbolic_grid,
    t0_linearity_gate,
)
from triton_viz.clients.race_detector.two_copy_symbolic_hb_solver import (
    TwoCopySymbolicHBSolver,
)

GOLDEN = Path(__file__).resolve().parents[1] / "golden" / "ttgir"


def _read(name):
    return (GOLDEN / name).read_text()


def _t(ptr, numel=4096, elem=4):
    return GlobalTensor(data_ptr=ptr, elem_size=elem, numel=numel)


def _solve(graph, params, tensors):
    # T1-style call: a 1-D launch sizes the unread axes of atomic-bearing
    # graphs (read axes stay symbolic; non-atomic graphs are unaffected).
    enc = encode_graph(graph, params, tensors)
    solver = TwoCopySymbolicHBSolver(
        enc.records,
        grid=symbolic_grid(enc, (4, 1, 1)),
        arange_dict=enc.arange_dict,
    )
    return enc, solver.find_races()


def _mini(*body_lines):
    body = "\n    ".join(body_lines)
    return (
        "module {\n"
        "  tt.func public @k(%x_ptr: !tt.ptr<f32>, %out_ptr: !tt.ptr<f32>)"
        " attributes {noinline = false} {\n"
        f"    {body}\n"
        "    tt.return\n"
        "  }\n"
        "}\n"
    )


ADD_TENSORS = {"x_ptr": _t(0x1000), "y_ptr": _t(0x11000), "out_ptr": _t(0x21000)}


# ─────────────────────── S3 exit criteria ───────────────────────


def test_add_proved_t1():
    """Stock elementwise kernel: per-pid footprints are disjoint — race-free
    for this input on EVERY 1-D grid (a claim the dynamic mode cannot make)."""
    enc, reports = _solve(
        parse_ttir(_read("add_sm80.ttir")), {"n_elements": 4096}, ADD_TENSORS
    )
    assert enc.used_pid_axes == {0}
    assert reports == []


def test_pid_stride_mutation_races():
    """Block stride 1024 → 512 under BLOCK=1024 makes adjacent blocks
    overlap: a definite WAW on the output with a cross-block witness."""
    g = parse_ttir(
        _read("add_sm80.ttir").replace(
            "arith.constant 1024 : i32", "arith.constant 512 : i32"
        )
    )
    _, reports = _solve(g, {"n_elements": 4096}, ADD_TENSORS)
    ww = [
        r
        for r in reports
        if r.first_record.tensor_name == "out_ptr"
        and r.second_record.tensor_name == "out_ptr"
    ]
    assert ww
    assert ww[0].witness_grid_a != ww[0].witness_grid_b


def test_tile2d_masked_2d_proved_t1():
    """Masked 2-D kernel with two pid axes: both grid dims stay symbolic."""
    enc, reports = _solve(
        parse_ttir(_read("tile2d_sm80.ttir")),
        {"M": 64, "N": 64, "stride_m": 64, "stride_n": 1},
        {"in_ptr": _t(0x1000), "out_ptr": _t(0x11000)},
    )
    assert enc.used_pid_axes == {0, 1}
    assert reports == []


def test_matmul_loop_proved_t1():
    """The K-loop rides the full machinery: iter-arg pointers advance by a
    symbolic iteration (copy-local var, range premise), and per-block C
    tiles stay disjoint under the in-bounds premise."""
    g = parse_ttir(_read("matmul_s3_sm80.ttir"))
    params = {
        "M": 128, "N": 128, "K": 64,
        "stride_am": 64, "stride_bk": 128, "stride_cm": 128,
    }  # fmt: skip
    tensors = {
        "a_ptr": _t(0x10000, numel=128 * 64, elem=2),
        "b_ptr": _t(0x20000, numel=64 * 128, elem=2),
        "c_ptr": _t(0x30000, numel=128 * 128, elem=2),
    }
    enc = encode_graph(g, params, tensors)
    assert any(r.copy_local_vars for r in enc.records)  # the loop var
    assert any(r.premises for r in enc.records)  # its range
    solver = TwoCopySymbolicHBSolver(
        enc.records, grid=symbolic_grid(enc), arange_dict=enc.arange_dict
    )
    assert solver.find_races() == []


# ─────────────────────── atomics ───────────────────────


def test_atomic_rmw_mutual_atomicity():
    """fadd and exch hit the same out region from every block: device-scope
    same-width atomics at the same address are mutually atomic — no race.
    The plain load/store pair keeps its disjoint per-pid footprint."""
    _, reports = _solve(
        parse_ttir(_read("atomic_sm80.ttir")),
        {"n_elements": 1024},
        {"x_ptr": _t(0x1000), "out_ptr": _t(0x11000)},
    )
    assert reports == []


def test_atomic_vs_plain_store_races():
    """An atomic RMW and a PLAIN store to the same location are not
    mutually atomic — that must surface as a race."""
    g = parse_ttir(
        _mini(
            "%pid = tt.get_program_id x : i32",
            "%true = arith.constant dense<true> : tensor<64xi1>",
            "%v = arith.constant dense<0> : tensor<64xi32>",
            "%r = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32>",
            "%c64 = arith.constant 64 : i32",
            "%b = arith.muli %pid, %c64 : i32",
            "%bs = tt.splat %b : i32 -> tensor<64xi32>",
            "%off = arith.addi %bs, %r : tensor<64xi32>",
            "%sx = tt.splat %x_ptr : !tt.ptr<f32> -> tensor<64x!tt.ptr<f32>>",
            "%px = tt.addptr %sx, %off : tensor<64x!tt.ptr<f32>>, tensor<64xi32>",
            "%l = tt.load %px : tensor<64x!tt.ptr<f32>>",
            "%p = tt.splat %out_ptr : !tt.ptr<f32> -> tensor<64x!tt.ptr<f32>>",
            "%q = tt.addptr %p, %r : tensor<64x!tt.ptr<f32>>, tensor<64xi32>",
            "%a = tt.atomic_rmw fadd, acq_rel, gpu, %q, %l, %true"
            " : (tensor<64x!tt.ptr<f32>>, tensor<64xf32>, tensor<64xi1>)"
            " -> tensor<64xf32>",
            "tt.store %q, %l : tensor<64x!tt.ptr<f32>>",
        )
    )
    _, reports = _solve(g, {}, {"x_ptr": _t(0x1000), "out_ptr": _t(0x11000)})
    pairs = {
        frozenset((r.first_record.atomic_kind, r.second_record.atomic_kind))
        for r in reports
    }
    assert frozenset(("rmw", "none")) in pairs


def test_cas_routes_to_interpreter():
    with pytest.raises(UnsupportedTTIR) as ei:
        encode_graph(
            parse_ttir(_read("cas_sm80.ttir")),
            {},
            {"lock_ptr": _t(0x1000), "out_ptr": _t(0x2000)},
        )
    assert ei.value.kind == "cas-synchronization"


# ─────────────────────── path conditions ───────────────────────

# pid_branch mutated so the guarded store writes x[0..255] (a FIXED range):
# with the pid==0 path modeled, only block 0 stores and only block 0's load
# overlaps it — different_blocks makes that unsatisfiable. Without the
# branch, every block stores the range block 0 loads: a real race.
_GUARDED_FIXED_STORE = (
    lambda: _read("pid_branch_sm80.ttir")
    .replace("%1 = tt.splat %out_ptr", "%1 = tt.splat %x_ptr")
    .replace("%2 = tt.addptr %1, %offs_2", "%2 = tt.addptr %1, %offs_0")
)


def test_path_condition_proves_race_freedom():
    _, reports = _solve(
        parse_ttir(_GUARDED_FIXED_STORE()),
        {"n_elements": 1024},
        {"x_ptr": _t(0x1000), "out_ptr": _t(0x11000)},
    )
    assert reports == []


def test_without_the_branch_it_races():
    text = (
        _GUARDED_FIXED_STORE()
        .replace("scf.if %0 {", "")
        .replace("    } loc(#loc10)", "")
    )
    _, reports = _solve(
        parse_ttir(text),
        {"n_elements": 1024},
        {"x_ptr": _t(0x1000), "out_ptr": _t(0x11000)},
    )
    assert reports  # the scf.if path was the only thing preventing the race


# ─────────────────────── uncertainty discipline ───────────────────────

# pid used (symbolic grid) + a store to a FIXED range behind a
# data-dependent mask: cross-block WAW is SAT, but only under the widened
# mask — never a certifiable witness.
_WIDENED_OVERLAP = _mini(
    "%pid = tt.get_program_id x : i32",
    "%c64 = arith.constant 64 : i32",
    "%b = arith.muli %pid, %c64 : i32",
    "%r = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32>",
    "%bs = tt.splat %b : i32 -> tensor<64xi32>",
    "%off = arith.addi %bs, %r : tensor<64xi32>",
    "%sx = tt.splat %x_ptr : !tt.ptr<f32> -> tensor<64x!tt.ptr<f32>>",
    "%px = tt.addptr %sx, %off : tensor<64x!tt.ptr<f32>>, tensor<64xi32>",
    "%l = tt.load %px : tensor<64x!tt.ptr<f32>>",
    "%z = arith.constant dense<0> : tensor<64xi32>",
    "%m = arith.cmpi sgt, %l, %z : tensor<64xi32>",
    "%p = tt.splat %out_ptr : !tt.ptr<f32> -> tensor<64x!tt.ptr<f32>>",
    "%q = tt.addptr %p, %r : tensor<64x!tt.ptr<f32>>, tensor<64xi32>",
    "tt.store %q, %l, %m : tensor<64x!tt.ptr<f32>>",
)


def test_widened_mask_reports_are_flagged_uncertain():
    g = parse_ttir(_WIDENED_OVERLAP)
    enc, reports = _solve(g, {}, {"x_ptr": _t(0x1000), "out_ptr": _t(0x11000)})
    assert enc.uncertain_event_ids  # the dropped-mask store
    assert reports
    assert all(
        {r.first.event_id, r.second.event_id} & enc.uncertain_event_ids for r in reports
    )


# ───────────── grid-pinning soundness (adversarial repros) ─────────────


def test_pid_in_stored_value_is_not_pinned():
    """store(out + r, pid): the pid never enters address math, but blocks
    write DIFFERENT values to the same elements — a definite WAW. The pid
    axis is recorded at PARSE time (AccessGraph.pid_axes), so the grid must
    stay symbolic and the race must surface."""
    g = parse_ttir(
        _mini(
            "%pid = tt.get_program_id x : i32",
            "%r = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32>",
            "%v = tt.splat %pid : i32 -> tensor<64xi32>",
            "%p = tt.splat %out_ptr : !tt.ptr<f32> -> tensor<64x!tt.ptr<f32>>",
            "%q = tt.addptr %p, %r : tensor<64x!tt.ptr<f32>>, tensor<64xi32>",
            "tt.store %q, %v : tensor<64x!tt.ptr<f32>>",
        )
    )
    assert g.pid_axes == {0}
    enc, reports = _solve(g, {}, {"x_ptr": _t(0x1000), "out_ptr": _t(0x11000)})
    assert enc.used_pid_axes == {0}
    assert reports  # definite cross-block WAW — previously a false 'ok' proof


def test_pid_only_in_dropped_mask_is_never_certified():
    """The pid read sits inside a data-dependent (dropped) mask: its Pid
    leaves are swallowed into DataDep before evaluation, but the PARSE-time
    axis set keeps the grid symbolic, so the widened WAW is SAT and lands in
    the uncertain channel — at worst 'unsupported', never a race-freedom
    proof."""
    g = parse_ttir(
        _mini(
            "%pid = tt.get_program_id x : i32",
            "%r = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32>",
            "%s = tt.splat %x_ptr : !tt.ptr<f32> -> tensor<64x!tt.ptr<f32>>",
            "%l = tt.load %s : tensor<64x!tt.ptr<f32>>",
            "%ps = tt.splat %pid : i32 -> tensor<64xi32>",
            "%m = arith.cmpi sgt, %l, %ps : tensor<64xi32>",
            "%p = tt.splat %out_ptr : !tt.ptr<f32> -> tensor<64x!tt.ptr<f32>>",
            "%q = tt.addptr %p, %r : tensor<64x!tt.ptr<f32>>, tensor<64xi32>",
            "tt.store %q, %l, %m : tensor<64x!tt.ptr<f32>>",
        )
    )
    assert g.pid_axes == {0}
    enc, reports = _solve(g, {}, {"x_ptr": _t(0x1000), "out_ptr": _t(0x11000)})
    assert reports  # SAT under the widening — not silently 'ok'
    assert all(
        {r.first.event_id, r.second.event_id} & enc.uncertain_event_ids for r in reports
    )


_ZERO_TRIP = _mini(
    "%c0 = arith.constant 0 : i32",
    "%c1 = arith.constant 1 : i32",
    "%pid = tt.get_program_id x : i32",
    "%r = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32>",
    "%v = arith.constant dense<0> : tensor<64xi32>",
    "%p = tt.splat %out_ptr : !tt.ptr<f32> -> tensor<64x!tt.ptr<f32>>",
    "%q = tt.addptr %p, %r : tensor<64x!tt.ptr<f32>>, tensor<64xi32>",
    "scf.for %i = %c0 to %K step %c1 : i32 {",
    "  tt.store %q, %v : tensor<64x!tt.ptr<f32>>",
    "}",
)


def test_zero_trip_loop_has_no_footprint():
    """K=0: the store never executes — no reports (a phantom-iteration model
    previously produced a DEFINITE witness for a race that cannot happen).
    K=1 sanity-checks that the same store does race when it runs."""
    g = parse_ttir(
        _ZERO_TRIP.replace(
            "%x_ptr: !tt.ptr<f32>, %out_ptr: !tt.ptr<f32>",
            "%x_ptr: !tt.ptr<f32>, %out_ptr: !tt.ptr<f32>, %K: i32",
        )
    )
    tensors = {"x_ptr": _t(0x1000), "out_ptr": _t(0x11000)}
    _, reports0 = _solve(g, {"K": 0}, tensors)
    assert reports0 == []
    _, reports1 = _solve(g, {"K": 1}, tensors)
    assert reports1  # fixed-range store from every block: real WAW


def test_non_contiguous_tensor_fails_closed():
    g = parse_ttir(_read("add_sm80.ttir"))
    tensors = dict(ADD_TENSORS)
    tensors["out_ptr"] = GlobalTensor(
        data_ptr=0x21000, elem_size=4, numel=4096, contiguous=False
    )
    with pytest.raises(UnsupportedTTIR, match="non-contiguous"):
        encode_graph(g, {"n_elements": 4096}, tensors)


def test_numpy_grid_dims_still_coerce():
    np = pytest.importorskip("numpy")
    assert TwoCopySymbolicHBSolver._normalize_grid((np.int64(4),)) == (4, 1, 1)


# ─────────────────────── tier selector (S4) ───────────────────────


def test_add_passes_the_gate_and_proves_t0():
    """The add kernel's stride is a folded constant, so the encoding stays
    linear with symbolic params: T0 proves race-freedom for ANY n_elements,
    any 1-D grid — a strictly stronger claim than T1."""
    g = parse_ttir(_read("add_sm80.ttir"))
    assert t0_linearity_gate(g)
    groups = dict(encode_graph_t0(g))
    assert set(groups) == {"out_ptr"}  # read-only x/y groups are skipped
    enc = groups["out_ptr"]
    solver = TwoCopySymbolicHBSolver(
        enc.records, grid=symbolic_grid(enc), arange_dict=enc.arange_dict
    )
    assert solver.find_races() == []


def test_param_stride_kernels_fail_the_gate():
    """tile2d multiplies an arange by a Param stride, matmul advances its
    iter-arg pointers by Param deltas: both are symbolic×symbolic at T0."""
    assert not t0_linearity_gate(parse_ttir(_read("tile2d_sm80.ttir")))
    assert not t0_linearity_gate(parse_ttir(_read("matmul_s3_sm80.ttir")))


# A store to out[0..63] masked by r < n, plus a pid-dependent load keeping
# the grid symbolic. At T0 (n symbolic) the cross-block WAW is SAT — but a
# T0 witness picks its own n, so the selector must fall to T1 and judge
# THIS launch's n.
_INPUT_DEPENDENT_RACE = _mini(
    "%pid = tt.get_program_id x : i32",
    "%c64 = arith.constant 64 : i32",
    "%b = arith.muli %pid, %c64 : i32",
    "%r = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32>",
    "%bs = tt.splat %b : i32 -> tensor<64xi32>",
    "%off = arith.addi %bs, %r : tensor<64xi32>",
    "%sx = tt.splat %x_ptr : !tt.ptr<f32> -> tensor<64x!tt.ptr<f32>>",
    "%px = tt.addptr %sx, %off : tensor<64x!tt.ptr<f32>>, tensor<64xi32>",
    "%l = tt.load %px : tensor<64x!tt.ptr<f32>>",
    "%nb = tt.splat %n : i32 -> tensor<64xi32>",
    "%m = arith.cmpi slt, %r, %nb : tensor<64xi32>",
    "%p = tt.splat %out_ptr : !tt.ptr<f32> -> tensor<64x!tt.ptr<f32>>",
    "%q = tt.addptr %p, %r : tensor<64x!tt.ptr<f32>>, tensor<64xi32>",
    "tt.store %q, %l, %m : tensor<64x!tt.ptr<f32>>",
).replace(
    "%x_ptr: !tt.ptr<f32>, %out_ptr: !tt.ptr<f32>",
    "%x_ptr: !tt.ptr<f32>, %out_ptr: !tt.ptr<f32>, %n: i32",
)


def test_t0_sat_falls_to_t1_and_judges_this_launch():
    g = parse_ttir(_INPUT_DEPENDENT_RACE)
    assert t0_linearity_gate(g)  # linear: r < n is a symbolic COMPARISON
    args = (
        torch.zeros(4096, dtype=torch.float32),
        torch.zeros(4096, dtype=torch.float32),
    )
    # n=0: the mask kills every lane — this launch is race-free, but only
    # at T1 (T0's symbolic n admits a witness). Rung must be T1.
    det = CompiledRaceDetector()
    _launch(
        det,
        ["x_ptr", "out_ptr", "n"],
        (*args, 0),
        {"grid": (4,)},
        _INPUT_DEPENDENT_RACE,
    )
    assert det.last_global_status == "ok"
    assert det.last_global_provenance == "proved@T1"
    # n=5: the same kernel really races on this launch.
    det2 = CompiledRaceDetector()
    _launch(
        det2,
        ["x_ptr", "out_ptr", "n"],
        (*args, 5),
        {"grid": (4,)},
        _INPUT_DEPENDENT_RACE,
    )
    assert det2.last_global_status == "races"
    assert det2.last_global_provenance is None


# Shift-by-one-block kernel: block p loads [64p, 64p+64) of x and stores
# [64p+64, 64p+128) of out. Non-aliased: provable at T0 (store footprints
# disjoint per pid; the load-only group is skipped). Aliased in-place
# (x_ptr is out_ptr): block p's store overlaps block p+1's load — a real
# cross-block RAW that only T1's real bases can see.
_SHIFT_KERNEL = _mini(
    "%pid = tt.get_program_id x : i32",
    "%c64 = arith.constant 64 : i32",
    "%b = arith.muli %pid, %c64 : i32",
    "%r = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32>",
    "%bs = tt.splat %b : i32 -> tensor<64xi32>",
    "%off = arith.addi %bs, %r : tensor<64xi32>",
    "%sx = tt.splat %x_ptr : !tt.ptr<f32> -> tensor<64x!tt.ptr<f32>>",
    "%px = tt.addptr %sx, %off : tensor<64x!tt.ptr<f32>>, tensor<64xi32>",
    "%l = tt.load %px : tensor<64x!tt.ptr<f32>>",
    "%c64t = tt.splat %c64 : i32 -> tensor<64xi32>",
    "%off2 = arith.addi %off, %c64t : tensor<64xi32>",
    "%p = tt.splat %out_ptr : !tt.ptr<f32> -> tensor<64x!tt.ptr<f32>>",
    "%q = tt.addptr %p, %off2 : tensor<64x!tt.ptr<f32>>, tensor<64xi32>",
    "tt.store %q, %l : tensor<64x!tt.ptr<f32>>",
)


def test_aliased_launch_never_proves_t0():
    """The T0 partition assumes non-aliased args; the selector may accept a
    T0 proof only when THIS launch's captured intervals are disjoint. An
    in-place launch must fall to T1 and report the cross-block RAW."""
    shared = torch.zeros(4096, dtype=torch.float32)
    det = CompiledRaceDetector()
    _launch(det, ["x_ptr", "out_ptr"], (shared, shared), {"grid": (4,)}, _SHIFT_KERNEL)
    assert det.last_global_status == "races"
    assert det.last_global_provenance is None
    # Non-aliased: same kernel proves at T0.
    det2 = CompiledRaceDetector()
    _launch(
        det2,
        ["x_ptr", "out_ptr"],
        (
            torch.zeros(4096, dtype=torch.float32),
            torch.zeros(4096, dtype=torch.float32),
        ),
        {"grid": (4,)},
        _SHIFT_KERNEL,
    )
    assert det2.last_global_status == "ok"
    assert det2.last_global_provenance == "proved@T0"


def test_unverifiable_capture_blocks_t0():
    """A pointer passed as a raw int leaves no tensor metadata: the
    non-aliasing premise is unverifiable, so T0 must not stand in — T1
    fails closed instead of a blind 'proved@T0'."""
    det = CompiledRaceDetector()
    _launch(
        det,
        ["x_ptr", "out_ptr"],
        (0x1000, 0x11000),  # raw addresses, no metadata
        {"grid": (4,)},
        _SHIFT_KERNEL,
    )
    assert det.last_global_status == "unsupported"
    assert "missing tensor metadata" in (det.last_global_reason or "")


def test_deep_term_chain_never_escapes_finalize():
    """A legal TTIR with a very deep offset chain exhausts recursion in the
    gate/eval walks; that must degrade to 'unsupported', never crash the
    user's launch teardown."""
    lines = [
        "%pid = tt.get_program_id x : i32",
        "%r = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32>",
        "%c1 = arith.constant dense<1> : tensor<64xi32>",
        "%v0 = arith.addi %r, %c1 : tensor<64xi32>",
    ]
    for i in range(1500):
        lines.append(f"%v{i + 1} = arith.addi %v{i}, %c1 : tensor<64xi32>")
    lines += [
        "%p = tt.splat %out_ptr : !tt.ptr<f32> -> tensor<64x!tt.ptr<f32>>",
        "%q = tt.addptr %p, %v1500 : tensor<64x!tt.ptr<f32>>, tensor<64xi32>",
        "tt.store %q, %c1 : tensor<64x!tt.ptr<f32>>",
    ]
    det = CompiledRaceDetector()
    _launch(
        det,
        ["x_ptr", "out_ptr"],
        (torch.zeros(64, dtype=torch.float32), torch.zeros(4096, dtype=torch.float32)),
        {"grid": (2,)},
        _mini(*lines),
    )  # must not raise
    assert det.last_global_status == "unsupported"


# ─────────────────────── client end-to-end ───────────────────────


def _fake_jit(names):
    return SimpleNamespace(arg_names=list(names))


def _launch(det, jit_names, args, kwargs, ttir_text):
    det.pre_warmup_callback(_fake_jit(jit_names), *args, **kwargs)
    det.post_warmup_callback(None, SimpleNamespace(asm={"ttir": ttir_text}))
    det.finalize()


ADD_NAMES = ["x_ptr", "y_ptr", "out_ptr", "n_elements", "BLOCK_SIZE"]


def _add_args():
    return (
        torch.zeros(4096, dtype=torch.float32),
        torch.zeros(4096, dtype=torch.float32),
        torch.zeros(4096, dtype=torch.float32),
        4096,
    )


def test_client_t1_proof_end_to_end():
    det = CompiledRaceDetector()
    _launch(
        det,
        ADD_NAMES,
        _add_args(),
        {"grid": (4,), "BLOCK_SIZE": 1024},
        _read("add_sm80.ttir"),
    )
    assert det.last_global_status == "ok"
    assert det.last_global_reports == []
    # add's stride is a folded constant → the selector reaches T0.
    assert det.last_global_provenance == "proved@T0"
    assert det.last_status == "no_ttgir"  # TTGIR verdict independent


def test_client_reports_definite_races():
    det = CompiledRaceDetector()
    mutated = _read("add_sm80.ttir").replace(
        "arith.constant 1024 : i32", "arith.constant 512 : i32"
    )
    _launch(det, ADD_NAMES, _add_args(), {"grid": (8,), "BLOCK_SIZE": 1024}, mutated)
    assert det.last_global_status == "races"
    assert det.last_global_reports


def test_client_downgrades_widened_reports():
    det = CompiledRaceDetector()
    x = torch.zeros(64, dtype=torch.float32)
    out = torch.zeros(64, dtype=torch.float32)
    _launch(det, ["x_ptr", "out_ptr"], (x, out), {"grid": (2,)}, _WIDENED_OVERLAP)
    assert det.last_global_status == "unsupported"
    assert "over-approximation" in (det.last_global_reason or "")
    assert det.last_global_reports == []


def test_client_multi_warmup_abstains():
    det = CompiledRaceDetector()
    det.pre_warmup_callback(_fake_jit(ADD_NAMES), *_add_args(), grid=(4,))
    det.pre_warmup_callback(_fake_jit(ADD_NAMES), *_add_args(), grid=(4,))
    det.post_warmup_callback(
        None, SimpleNamespace(asm={"ttir": _read("add_sm80.ttir")})
    )
    det.finalize()
    assert det.last_global_status == "unsupported"
    assert "ambiguous" in (det.last_global_reason or "")
    # and the launch capture reset: the next launch is clean again
    _launch(
        det,
        ADD_NAMES,
        _add_args(),
        {"grid": (4,), "BLOCK_SIZE": 1024},
        _read("add_sm80.ttir"),
    )
    assert det.last_global_status == "ok"


def test_client_unparseable_ttir_is_unsupported_globally():
    det = CompiledRaceDetector()
    _launch(
        det,
        ["idx_ptr", "src_ptr", "out_ptr", "n_elements", "BLOCK"],
        (
            torch.zeros(64, dtype=torch.int32),
            torch.zeros(64, dtype=torch.float32),
            torch.zeros(64, dtype=torch.float32),
            64,
        ),
        {"grid": (1,)},
        _read("gather_sm80.ttir"),
    )
    assert det.last_global_status == "unsupported"
    assert (det.last_global_reason or "").startswith("indirect-address")
