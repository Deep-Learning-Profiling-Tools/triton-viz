"""End-to-end tests for the hybrid's information channels (plan §I.4):

C2 — witness replay: SAT reports are replayed under the interpreter on
pre-launch tensor clones; definite reports classify confirmed/unconfirmed,
and a CONFIRMED widened (dropped-mask) report graduates to a definite race
— the abstention becomes a verdict.

C3 — differential cross-check: the static model's concretely enumerated
footprint must match the interpreter's, per program instance.

These run the REAL kernels (host-compiled TTIR + CPU interpreter replay)."""

from pathlib import Path
from types import SimpleNamespace

import torch
import triton
import triton.language as tl
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource

from triton_viz.clients.common.ttir_reader import parse_ttir
from triton_viz.clients.race_detector.compiled.client import CompiledRaceDetector
from triton_viz.clients.race_detector.compiled.global_records import GlobalTensor
from triton_viz.clients.race_detector.compiled.replay import (
    confirm_witness,
    cross_check,
)

GOLDEN = Path(__file__).resolve().parents[1] / "golden" / "ttgir"


# ─────────────────────── kernels under test ───────────────────────


@triton.jit
def waw_kernel(x_ptr, out_ptr, BLOCK: tl.constexpr):
    """Every block stores the same fixed range (while reading a
    pid-dependent slice of x): a definite cross-block WAW."""
    pid = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    v = tl.load(x_ptr + pid * BLOCK + offs)
    tl.store(out_ptr + offs, v)


@triton.jit
def dd_mask_kernel(flag_ptr, x_ptr, out_ptr, BLOCK: tl.constexpr):
    """Fixed-range store behind a DATA-DEPENDENT mask: the static model
    widens the mask (uncertain SAT); only the replay can tell whether this
    launch's flag data makes the WAW real."""
    pid = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    keep = tl.load(flag_ptr + offs) > 0
    v = tl.load(x_ptr + pid * BLOCK + offs)
    tl.store(out_ptr + offs, v, mask=keep)


@triton.jit
def add_kernel(x_ptr, y_ptr, out_ptr, n_elements, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elements
    x = tl.load(x_ptr + offs, mask=mask)
    y = tl.load(y_ptr + offs, mask=mask)
    tl.store(out_ptr + offs, x + y, mask=mask)


def _ttir_of(fn, signature, constexprs):
    # Robust under TRITON_INTERPRET pollution: another test module
    # (test_multithreading) sets the env var at IMPORT time, so this
    # module's @triton.jit kernels become InterpretedFunction whenever it
    # is imported later in the same process (alphabetical collection,
    # sequential or xdist). triton >= 3.7 ASTSource.hash() requires
    # fn.cache_key, which InterpretedFunction lacks — rebuild the real
    # JITFunction from the raw callable for host compilation.
    if not hasattr(fn, "cache_key") and hasattr(fn, "fn"):
        fn = triton.runtime.jit.JITFunction(fn.fn)
    src = ASTSource(fn=fn, signature=signature, constexprs=constexprs)
    return triton.compile(src, target=GPUTarget("cuda", 80, 32)).asm["ttir"]


def _store_line(kernel) -> int:
    """The kernel's tl.store source line — per-site foci key replay
    buckets by (base, kind, user line)."""
    import inspect

    fn = kernel.fn if hasattr(kernel, "fn") else kernel
    lines, start = inspect.getsourcelines(fn)
    for i, line in enumerate(lines):
        if "tl.store(out_ptr" in line:
            return start + i
    raise AssertionError("no store line found")


def _launch(det, jit_fn, args, kwargs, ttir_text):
    det.pre_warmup_callback(jit_fn, *args, **kwargs)
    det.post_warmup_callback(None, SimpleNamespace(asm={"ttir": ttir_text}))
    det.finalize()


# ─────────────────────── C2: definite reports ───────────────────────


def test_c2_confirms_a_real_waw():
    ttir = _ttir_of(
        waw_kernel,
        {"x_ptr": "*fp32", "out_ptr": "*fp32", "BLOCK": "constexpr"},
        {"BLOCK": 64},
    )
    det = CompiledRaceDetector()
    x, out = torch.randn(256), torch.zeros(64)
    _launch(det, waw_kernel, (x, out), {"grid": (4,), "BLOCK": 64}, ttir)
    assert det.last_status == "no_ttgir"  # TTGIR track untouched
    assert det.last_global_status == "races"
    assert det.last_global_confirmation == "confirmed"


def test_c2_off_keeps_the_old_surface():
    ttir = _ttir_of(
        waw_kernel,
        {"x_ptr": "*fp32", "out_ptr": "*fp32", "BLOCK": "constexpr"},
        {"BLOCK": 64},
    )
    det = CompiledRaceDetector(confirm_races=False)
    _launch(
        det, waw_kernel, (torch.randn(256), torch.zeros(64)),
        {"grid": (4,), "BLOCK": 64}, ttir,
    )  # fmt: skip
    assert det.last_global_status == "races"
    assert det.last_global_confirmation is None


# ─────────────────────── C2: widened reports ───────────────────────

_DD_SIG = {
    "flag_ptr": "*i32",
    "x_ptr": "*fp32",
    "out_ptr": "*fp32",
    "BLOCK": "constexpr",
}


def _dd_launch(flag_value: int, **det_kwargs):
    ttir = _ttir_of(dd_mask_kernel, _DD_SIG, {"BLOCK": 64})
    det = CompiledRaceDetector(**det_kwargs)
    flags = torch.full((64,), flag_value, dtype=torch.int32)
    x = torch.randn(256)
    out = torch.zeros(64)
    _launch(det, dd_mask_kernel, (flags, x, out), {"grid": (4,), "BLOCK": 64}, ttir)
    return det


def _dd_launch_no_replay(flag_value: int):
    # replay channel OFF: the widened SAT lands the GENERIC abstention
    # (confirmation never ran), the capped/unavailable-demotion shape
    return _dd_launch(flag_value, confirm_races=False)


def test_c2_upgrades_a_confirmed_widened_race():
    """flags all positive: the real mask is true everywhere, the WAW is
    real — the widened report graduates from abstention to a definite,
    replay-confirmed race."""
    det = _dd_launch(flag_value=1)
    assert det.last_global_status == "races"
    assert det.last_global_confirmation == "confirmed"
    assert det.last_global_reports


def test_c2_classifies_race_unconfirmed():
    """flags all zero: the real mask kills every lane; the widened SAT does
    not reproduce — the race-unconfirmed terminal state (potential, never a
    definite report). §3n: the faithfully-refuted hazard is retained as
    content-fragility EVIDENCE (for the composed dispatcher's
    proof-plus-attribute upgrade), and the client-side attribute stays
    False — only the dispatcher, which sees the dynamic track, may
    stamp it."""
    det = _dd_launch(flag_value=0)
    assert det.last_global_status == "unsupported"
    assert "race-unconfirmed" in (det.last_global_reason or "")
    assert det.last_global_reports == []
    assert det.last_content_hazard, "refuted hazard must be carried as evidence"
    assert det.last_global_verdict["content_fragile"] is False


def test_capped_demotion_carries_no_content_hazard():
    """the generic over-approximation abstention (replay unavailable /
    unclassifiable) must NOT populate the content-fragility evidence —
    §3n guardrail (i): only the faithful all-refuted demotion earns the
    upgrade path."""
    det = _dd_launch_no_replay(flag_value=0)
    assert det.last_global_status == "unsupported"
    assert "race-unconfirmed" not in (det.last_global_reason or "")
    assert det.last_content_hazard == []


def test_c2_witness_replay_direct():
    """The primitive itself: same kernel, verdict flips with the data.
    The overlap check is focused on the report's access pair (here: the
    out store on both sides)."""
    flags1 = torch.ones(64, dtype=torch.int32)
    flags0 = torch.zeros(64, dtype=torch.int32)
    x, out = torch.randn(256), torch.zeros(64)
    focus = (int(out.data_ptr()), "store", _store_line(dd_mask_kernel))
    v1, _ = confirm_witness(
        dd_mask_kernel, (flags1, x, out), {"BLOCK": 64}, (0, 0, 0), (1, 0, 0),
        (4,), focus_a=focus, focus_b=focus,
    )  # fmt: skip
    v0, _ = confirm_witness(
        dd_mask_kernel, (flags0, x, out), {"BLOCK": 64}, (0, 0, 0), (1, 0, 0),
        (4,), focus_a=focus, focus_b=focus,
    )  # fmt: skip
    assert (v1, v0) == ("confirmed", "unconfirmed")


def test_c2_unfocused_or_intra_instance_is_unavailable():
    """No foci → unavailable (a whole-block check can fabricate
    confirmations); same pid twice → unavailable (duplicate lanes collapse
    in an address set); witness pids outside the launch grid, an unknown
    (callable) grid, and rmw∩rmw foci → unavailable."""
    flags = torch.ones(64, dtype=torch.int32)
    x, out = torch.randn(256), torch.zeros(64)
    v, _ = confirm_witness(
        dd_mask_kernel, (flags, x, out), {"BLOCK": 64}, (0, 0, 0), (1, 0, 0), (4,)
    )
    assert v == "unavailable"
    focus = (int(out.data_ptr()), "store", _store_line(dd_mask_kernel))
    v, _ = confirm_witness(
        dd_mask_kernel, (flags, x, out), {"BLOCK": 64}, (1, 0, 0), (1, 0, 0),
        (4,), focus_a=focus, focus_b=focus,
    )  # fmt: skip
    assert v == "unavailable"
    # witness block does not exist on this launch's grid
    v, why = confirm_witness(
        dd_mask_kernel, (flags, x, out), {"BLOCK": 64}, (0, 0, 0), (1, 0, 0),
        (1,), focus_a=focus, focus_b=focus,
    )  # fmt: skip
    assert v == "unavailable" and "do not exist" in (why or "")
    # callable grid cannot parameterize a faithful replay
    v, _ = confirm_witness(
        dd_mask_kernel, (flags, x, out), {"BLOCK": 64}, (0, 0, 0), (1, 0, 0),
        lambda meta: (4,), focus_a=focus, focus_b=focus,
    )  # fmt: skip
    assert v == "unavailable"
    # rmw∩rmw: scope/width live outside the footprint
    rmw_focus = (int(out.data_ptr()), "atomic_rmw", _store_line(dd_mask_kernel))
    v, _ = confirm_witness(
        dd_mask_kernel, (flags, x, out), {"BLOCK": 64}, (0, 0, 0), (1, 0, 0),
        (4,), focus_a=rmw_focus, focus_b=rmw_focus,
    )  # fmt: skip
    assert v == "unavailable"


@triton.jit
def mixed_kernel(m_ptr, x_ptr, out_ptr, aux_ptr, BLOCK: tl.constexpr):
    """An exact WAW on out (fixed range, every block) NEXT TO a widened
    store on aux whose real mask is dead. The replay must classify each
    report by ITS OWN access pair — the aux report must not ride the out
    conflict to a fabricated definite race."""
    pid = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    v = tl.load(x_ptr + pid * BLOCK + offs)
    tl.store(out_ptr + offs, v)
    keep = tl.load(m_ptr + offs) > 0
    tl.store(aux_ptr + offs, v, mask=keep)


def test_c2_focus_blocks_fabricated_upgrade():
    ttir = _ttir_of(
        mixed_kernel,
        {
            "m_ptr": "*i32",
            "x_ptr": "*fp32",
            "out_ptr": "*fp32",
            "aux_ptr": "*fp32",
            "BLOCK": "constexpr",
        },  # fmt: skip
        {"BLOCK": 64},
    )
    det = CompiledRaceDetector()
    m0 = torch.zeros(64, dtype=torch.int32)  # aux's real mask is dead
    _launch(
        det,
        mixed_kernel,
        (m0, torch.randn(256), torch.zeros(64), torch.zeros(64)),
        {"grid": (4,), "BLOCK": 64},
        ttir,
    )
    assert det.last_global_status == "races"  # the out WAW is real
    names = {
        (r.first_record.tensor_name, r.second_record.tensor_name)
        for r in det.last_global_reports
    }
    assert all("aux_ptr" not in pair for pair in names), names
    # the withheld aux possibility is noted, not reported
    assert "withheld" in (det.last_global_reason or "")


# ────────────── adversarial regressions (2nd verification round) ──────────────


@triton.jit
def np_mask_kernel(x_ptr, out_ptr, BLOCK: tl.constexpr):
    """The mask observes the GRID via tl.num_programs. At the real launch
    grid (4,) the mask is dead; a synthetic max(pid)+1 replay grid would
    flip it alive and fabricate a confirmed race. Since NumPrograms became
    a modeled term (spec part B wiring) the static side reports a DEFINITE
    race — the store is live on every grid other than 4, and the T1 claim
    covers every grid — but the anti-fabrication property under test is
    unchanged: the replay must run at the LAUNCH grid, where the mask is
    dead, and must therefore never say "confirmed"."""
    pid = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    v = tl.load(x_ptr + pid * BLOCK + offs)
    limit = tl.where(tl.num_programs(0) == 4, 0, BLOCK)
    keep = offs < limit
    tl.store(out_ptr + offs, v, mask=keep)


def test_out_of_extent_exact_sat_lands_launch_scoped_proof():
    """np_mask_kernel's store is live on every grid EXCEPT the launch's
    (4,) — the §3c shape. Historically this pinned the anti-fabrication
    property (C2 must replay at the LAUNCH grid, where the mask is dead,
    and never confirm); the launch-scoped rung now retires the scenario
    one step earlier: the pinned re-query is UNSAT at extent 4, so the
    verdict is a launch-scoped PROOF and the any-grid evidence rides the
    independent grid-fragile attribute — C2 never runs, so there is
    nothing left to fabricate."""
    ttir = _ttir_of(
        np_mask_kernel,
        {"x_ptr": "*fp32", "out_ptr": "*fp32", "BLOCK": "constexpr"},
        {"BLOCK": 64},
    )
    det = CompiledRaceDetector()
    x, out = torch.randn(256), torch.zeros(64)
    _launch(det, np_mask_kernel, (x, out), {"grid": (4,), "BLOCK": 64}, ttir)
    assert det.last_global_status == "ok"
    assert det.last_global_provenance == "proved@T1-launch"
    assert det.last_global_reports == []
    assert det.last_global_confirmation is None  # C2 never engaged
    assert det.last_grid_fragile, "any-grid evidence must be carried"
    assert all(
        r.model.get("grid_0") != "4" for r in det.last_grid_fragile
    ), "the fragility witness lives on a grid where the mask is alive"
    v = det.last_global_verdict
    assert v["verdict"] == "race-free"
    assert v["proved_scope"] == "this-params-this-grid"
    assert v["grid_fragile"] is True


@triton.jit
def same_tensor_kernel(m_ptr, x_ptr, out_ptr, BLOCK: tl.constexpr):
    """An exact WAW on out[0:64] and a dead widened store on out[64:128]:
    SAME tensor, SAME kind — the two sites share one footprint bucket, so
    the widened report is unclassifiable (ambiguous), never confirmed on
    the strength of the exact store's overlap."""
    pid = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    v = tl.load(x_ptr + pid * BLOCK + offs)
    tl.store(out_ptr + offs, v)
    keep = tl.load(m_ptr + offs) > 0
    tl.store(out_ptr + BLOCK + offs, v, mask=keep)


def test_c2_same_tensor_sites_are_classified_separately():
    ttir = _ttir_of(
        same_tensor_kernel,
        {
            "m_ptr": "*i32",
            "x_ptr": "*fp32",
            "out_ptr": "*fp32",
            "BLOCK": "constexpr",
        },  # fmt: skip
        {"BLOCK": 64},
    )
    det = CompiledRaceDetector()
    m0 = torch.zeros(64, dtype=torch.int32)  # the widened store never runs
    _launch(
        det,
        same_tensor_kernel,
        (m0, torch.randn(256), torch.zeros(128)),
        {"grid": (4,), "BLOCK": 64},
        ttir,
    )
    assert det.last_global_status == "races"  # the exact WAW is real
    # exactly ONE definite report — the dead widened store must not ride
    # the exact store's overlap into a fabricated second race. Per-SITE
    # footprint keying (base, kind, line) now CLASSIFIES the widened
    # report instead of declining it as ambiguous: its own site's
    # footprint is empty (mask dead), so it stays an unconfirmed
    # withheld abstention while the exact site confirms.
    assert len(det.last_global_reports) == 1
    assert det.last_global_confirmation == "partial"
    assert "withheld" in (det.last_global_reason or "")


@triton.jit
def unrolled_store_kernel(out_ptr, BLOCK: tl.constexpr, N_BLKS: tl.constexpr):
    """The aiter#3091 shape: every program writes the full output with no
    pid partitioning, and the store is UNROLLED by tl.static_range onto a
    single source line. The two unrolled iterations collapse to one
    (out_ptr, store, line) footprint bucket — count > 1, so the bucket is
    'ambiguous'. But this is an EXACT cross-block WAW whose access is live
    by construction; the ambiguous gate (which exists to stop dropped-mask
    WIDENED reports riding an unrelated same-line overlap) must NOT decline
    it. Regression pin for the aiter races-unclassified→race-confirmed fix."""
    offs = tl.arange(0, BLOCK)
    for i in tl.static_range(N_BLKS):
        tl.store(out_ptr + i * BLOCK + offs, offs)


def test_c2_confirms_exact_waw_at_unrolled_ambiguous_site():
    ttir = _ttir_of(
        unrolled_store_kernel,
        {"out_ptr": "*i32", "BLOCK": "constexpr", "N_BLKS": "constexpr"},
        {"BLOCK": 32, "N_BLKS": 2},
    )
    det = CompiledRaceDetector()
    _launch(
        det,
        unrolled_store_kernel,
        (torch.zeros(64, dtype=torch.int32),),
        {"grid": (4,), "BLOCK": 32, "N_BLKS": 2},
        ttir,
    )
    assert det.last_global_status == "races"  # cross-block WAW is real
    # the exact report confirms despite the unrolled same-line bucket —
    # before the fix this stayed None (=> races-unclassified terminal)
    assert det.last_global_confirmation == "confirmed"


def test_c2_same_tensor_live_widened_site_graduates():
    """The recovery the per-site keying exists for: with the mask DATA
    live, the widened store's OWN site overlaps across blocks and the
    report graduates to a definite race — previously unclassifiable
    because both stores shared the (tensor, kind) bucket."""
    ttir = _ttir_of(
        same_tensor_kernel,
        {
            "m_ptr": "*i32",
            "x_ptr": "*fp32",
            "out_ptr": "*fp32",
            "BLOCK": "constexpr",
        },  # fmt: skip
        {"BLOCK": 64},
    )
    det = CompiledRaceDetector()
    m1 = torch.ones(64, dtype=torch.int32)  # the widened store RUNS
    _launch(
        det,
        same_tensor_kernel,
        (m1, torch.randn(256), torch.zeros(128)),
        {"grid": (4,), "BLOCK": 64},
        ttir,
    )
    assert det.last_global_status == "races"
    assert len(det.last_global_reports) == 2  # exact WAW + graduated widened
    assert det.last_global_confirmation == "confirmed"


def test_widened_out_of_extent_sat_lands_launch_scoped_proof():
    """grid=(1,): a single program instance cannot cross-block race, so
    the widened any-grid SAT has no witness on this launch. Previously a
    withheld abstention; the §3c rung now proves it AS LAUNCHED — sound
    even from widened evidence, because widening only ENLARGES
    footprints: extent-UNSAT of the over-approximation implies
    extent-UNSAT of the real footprints. The widened evidence rides the
    grid-fragile attribute, never a graduation claim."""
    ttir = _ttir_of(dd_mask_kernel, _DD_SIG, {"BLOCK": 64})
    det = CompiledRaceDetector()
    flags = torch.ones(64, dtype=torch.int32)
    _launch(
        det,
        dd_mask_kernel,
        (flags, torch.randn(64), torch.zeros(64)),
        {"grid": (1,), "BLOCK": 64},
        ttir,
    )
    assert det.last_global_status == "ok"
    assert det.last_global_provenance == "proved@T1-launch"
    assert det.last_global_reports == []
    assert det.last_grid_fragile
    # NOT the race-unconfirmed claim: the replay never classified anything
    assert "race-unconfirmed" not in (det.last_global_reason or "")


def test_c3_exact_sibling_of_widened_access_not_diffed():
    """Symmetric exclusion: a tensor with BOTH an exact and a widened store
    must not produce a fabricated static-only divergence (the widened
    access has no static footprint, but its exact sibling does — one-sided
    deletion made C3 cry lowering-divergence on a correct kernel)."""
    ttir = _ttir_of(
        same_tensor_kernel,
        {
            "m_ptr": "*i32",
            "x_ptr": "*fp32",
            "out_ptr": "*fp32",
            "BLOCK": "constexpr",
        },  # fmt: skip
        {"BLOCK": 64},
    )
    det = CompiledRaceDetector(confirm_races=False, differential_check=True)
    m1 = torch.ones(64, dtype=torch.int32)
    _launch(
        det,
        same_tensor_kernel,
        (m1, torch.randn(256), torch.zeros(128)),
        {"grid": (4,), "BLOCK": 64},
        ttir,
    )
    assert det.last_differential == [], det.last_differential


# ─────────────────────── C3: differential cross-check ───────────────────────


def test_c3_static_and_interpreter_agree_on_add():
    n = 2500
    ttir = _ttir_of(
        add_kernel,
        {
            "x_ptr": "*fp32",
            "y_ptr": "*fp32",
            "out_ptr": "*fp32",
            "n_elements": "i32",
            "BLOCK": "constexpr",
        },
        {"BLOCK": 1024},
    )
    graph = parse_ttir(ttir)
    x, y, out = torch.randn(n), torch.randn(n), torch.zeros(n)
    tensors = {
        "x_ptr": GlobalTensor(int(x.data_ptr()), 4, n),
        "y_ptr": GlobalTensor(int(y.data_ptr()), 4, n),
        "out_ptr": GlobalTensor(int(out.data_ptr()), 4, n),
    }
    issues = cross_check(
        graph,
        {"n_elements": n},
        tensors,
        add_kernel,
        (x, y, out, n),
        {"BLOCK": 1024},
        pids=[(0, 0, 0), (2, 0, 0)],  # pid 2 exercises the ragged masked tail
        grid=(3,),
    )
    assert issues == [], issues


def test_c3_client_integration():
    """differential_check=True: the client runs the diff on its own snapshot
    and publishes [] when the lowering and the interpreter agree."""
    n = 2500
    ttir = _ttir_of(
        add_kernel,
        {
            "x_ptr": "*fp32",
            "y_ptr": "*fp32",
            "out_ptr": "*fp32",
            "n_elements": "i32",
            "BLOCK": "constexpr",
        },
        {"BLOCK": 1024},
    )
    det = CompiledRaceDetector(differential_check=True)
    x, y, out = torch.randn(n), torch.randn(n), torch.zeros(n)
    _launch(det, add_kernel, (x, y, out, n), {"grid": (3,), "BLOCK": 1024}, ttir)
    assert det.last_global_status == "ok"
    assert det.last_differential == [], det.last_differential


def test_c3_detects_a_divergence():
    """Feed the static side the WRONG parameter: the footprints must
    disagree — the diff is load-bearing, not vacuously empty."""
    n = 2500
    ttir = _ttir_of(
        add_kernel,
        {
            "x_ptr": "*fp32",
            "y_ptr": "*fp32",
            "out_ptr": "*fp32",
            "n_elements": "i32",
            "BLOCK": "constexpr",
        },
        {"BLOCK": 1024},
    )
    graph = parse_ttir(ttir)
    x, y, out = torch.randn(n), torch.randn(n), torch.zeros(n)
    tensors = {
        "x_ptr": GlobalTensor(int(x.data_ptr()), 4, n),
        "y_ptr": GlobalTensor(int(y.data_ptr()), 4, n),
        "out_ptr": GlobalTensor(int(out.data_ptr()), 4, n),
    }
    issues = cross_check(
        graph,
        {"n_elements": n + 100},  # static believes a different mask bound
        tensors,
        add_kernel,
        (x, y, out, n),
        {"BLOCK": 1024},
        pids=[(2, 0, 0)],
        grid=(3,),
    )
    assert issues, "the differential must flag the divergence"


# ─────────────────────── C3 through the client ───────────────────────

_ADD_SIG = {
    "x_ptr": "*fp32",
    "y_ptr": "*fp32",
    "out_ptr": "*fp32",
    "n_elements": "i32",
    "BLOCK": "constexpr",
}


def test_c3_client_catches_a_lowering_mismatch():
    """The client is fed a TTIR whose block stride differs from what the
    kernel source actually does — as if the compiler had lowered the kernel
    differently than the reader believes. The static footprint (from the
    TTIR) diverges from the interpreter footprint (from the source):
    exactly the lowering/semantics oracle the plan assigns to C3."""
    n = 2500
    ttir = _ttir_of(add_kernel, _ADD_SIG, {"BLOCK": 1024}).replace(
        "arith.constant 1024 : i32", "arith.constant 512 : i32"
    )
    det = CompiledRaceDetector(differential_check=True)
    x, y, out = torch.randn(n), torch.randn(n), torch.zeros(n)
    _launch(det, add_kernel, (x, y, out, n), {"grid": (3,), "BLOCK": 1024}, ttir)
    assert det.last_differential  # pid footprints disagree


def test_c3_client_off_by_default():
    n = 2500
    ttir = _ttir_of(add_kernel, _ADD_SIG, {"BLOCK": 1024})
    det = CompiledRaceDetector()
    x, y, out = torch.randn(n), torch.randn(n), torch.zeros(n)
    _launch(det, add_kernel, (x, y, out, n), {"grid": (3,), "BLOCK": 1024}, ttir)
    assert det.last_differential is None
