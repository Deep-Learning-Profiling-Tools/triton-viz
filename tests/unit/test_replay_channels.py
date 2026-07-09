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
    src = ASTSource(fn=fn, signature=signature, constexprs=constexprs)
    return triton.compile(src, target=GPUTarget("cuda", 80, 32)).asm["ttir"]


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


def _dd_launch(flag_value: int):
    ttir = _ttir_of(dd_mask_kernel, _DD_SIG, {"BLOCK": 64})
    det = CompiledRaceDetector()
    flags = torch.full((64,), flag_value, dtype=torch.int32)
    x = torch.randn(256)
    out = torch.zeros(64)
    _launch(det, dd_mask_kernel, (flags, x, out), {"grid": (4,), "BLOCK": 64}, ttir)
    return det


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
    definite report)."""
    det = _dd_launch(flag_value=0)
    assert det.last_global_status == "unsupported"
    assert "race-unconfirmed" in (det.last_global_reason or "")
    assert det.last_global_reports == []


def test_c2_witness_replay_direct():
    """The primitive itself: same kernel, verdict flips with the data.
    The overlap check is focused on the report's access pair (here: the
    out store on both sides)."""
    flags1 = torch.ones(64, dtype=torch.int32)
    flags0 = torch.zeros(64, dtype=torch.int32)
    x, out = torch.randn(256), torch.zeros(64)
    focus = (int(out.data_ptr()), "store")
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
    focus = (int(out.data_ptr()), "store")
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
    rmw_focus = (int(out.data_ptr()), "atomic_rmw")
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


def test_c2_replays_at_the_launch_grid():
    ttir = _ttir_of(
        np_mask_kernel,
        {"x_ptr": "*fp32", "out_ptr": "*fp32", "BLOCK": "constexpr"},
        {"BLOCK": 64},
    )
    det = CompiledRaceDetector()
    x, out = torch.randn(256), torch.zeros(64)
    _launch(det, np_mask_kernel, (x, out), {"grid": (4,), "BLOCK": 64}, ttir)
    # A real race on every grid BUT the launch's: reported (universal-grid
    # claim), with a witness grid other than 4...
    assert det.last_global_status == "races"
    assert det.last_global_reports
    assert all(
        r.model.get("grid_0") != "4" for r in det.last_global_reports
    ), "the witness must live on a grid where the mask is alive"
    # ...and the launch-grid replay (mask dead at grid 4) must never
    # fabricate a confirmation.
    assert det.last_global_confirmation != "confirmed"


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


def test_c2_same_tensor_bucket_is_ambiguous():
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
    # the exact store's shared bucket into a fabricated second race
    assert len(det.last_global_reports) == 1
    assert "withheld" in (det.last_global_reason or "")


def test_c2_no_graduation_outside_the_launch_grid():
    """grid=(1,): a single program instance cannot cross-block race. The
    solver's witnesses (grid-generic by design) do not exist on this
    launch, so the widened report must stay a withheld abstention — the
    'on this launch's data' graduation claim would be false."""
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
    assert det.last_global_status == "unsupported"
    assert det.last_global_reports == []
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
