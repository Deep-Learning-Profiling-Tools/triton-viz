"""Regression pins for the adversarially-confirmed tranche-2 findings.

An agent workflow attacked the TMA/mbarrier model's soundness claims and
confirmed 12 findings (11 soundness, 1 precision) by executing crafted
TTGIR through analyze_ttgir; each test here replays one attack and pins
the FIXED verdict. Finding numbers reference the 2026-07-10 verification
run. Finding 9 (prologue-arming drain bound must use the wait-chain base:
c ≥ (s - b_w) mod S, not c ≥ s) has no crafted-IR pin — building a
consistent b_w≠0 protocol needs a full hand-written module — and is
covered by the formula in hb.validate_reuse_drain plus review.
"""

from pathlib import Path

from triton_viz.clients.race_detector.compiled import analyze_ttgir
from triton_viz.clients.race_detector.data import RaceType

GOLDEN = Path(__file__).resolve().parents[1] / "golden" / "ttgir"


def _read(name: str) -> str:
    return (GOLDEN / name).read_text()


def test_f1_waw_double_tma_copy_one_shot_exact_bytes_is_reported():
    """Finding 1: two TMA copies into the same buffer under one exact-byte
    one-shot arming used to get a clean proof — both individually covered
    by the wait, but mutually unordered (nondeterministic byte mixture)."""
    stock = _read("matmul_tma_s1_sm90.ttgir")
    copy_line = (
        "ttng.async_tma_copy_global_to_local %a_desc_2[%a, %a_8] %a_9, " "%a_10, %true"
    )
    dup_line = (
        "ttng.async_tma_copy_global_to_local %a_desc_2[%b, %a_8] %a_9, " "%a_10, %true"
    )
    assert copy_line in stock
    mutated = stock.replace(
        copy_line, copy_line + " : X\n        " + dup_line, 1
    ).replace(" : X\n", " :\n".replace(" :\n", " "), 0)
    # keep the original type suffix by duplicating the whole line instead
    lines = stock.splitlines()
    out = []
    for line in lines:
        out.append(line)
        if copy_line in line:
            out.append(line.replace("[%a, %a_8]", "[%b, %a_8]"))
        if "ttng.barrier_expect %a_10, 4096" in line:
            out[-1] = line.replace("4096", "8192")
    mutated = "\n".join(out)
    r = analyze_ttgir(mutated)
    assert r.status == "ok", r.unsupported_reason
    waw = [rep for rep in r.reports if rep.race_type == RaceType.WAW]
    assert waw, [rep.message[:80] for rep in r.reports]
    assert all(rep.alloc == "%a_9" for rep in waw)


def test_f1_waw_stock_pipelines_still_prove():
    """The WAW query must not break the stock proofs: every same-slot
    writer pair is retired by the wait in effect before the later write."""
    for name in (
        "matmul_s3_sm80.ttgir",
        "matmul_s3_sm90.ttgir",
        "matmul_tma_s3_sm90.ttgir",
        "matmul_tma_s1_sm90.ttgir",
    ):
        r = analyze_ttgir(_read(name))
        assert r.status == "ok" and r.reports == [], name


def test_f2_one_shot_init_after_wait_is_unsupported():
    """Finding 2: init_barrier moved after the wait (iteration-0 wait on
    uninitialized mbarrier storage) used to keep the proof."""
    stock = _read("matmul_tma_s1_sm90.ttgir")
    init_line = "ttng.init_barrier %a_10, 1"
    wait_line = "ttng.wait_barrier %a_10, %c0_i32"
    lines = stock.splitlines()
    init_full = next(ln for ln in lines if init_line in ln)
    out = []
    for line in lines:
        if init_line in line:
            continue
        out.append(line)
        if wait_line in line:
            out.append(init_full)
    r = analyze_ttgir("\n".join(out))
    assert r.status == "unsupported"
    assert "before its init_barrier" in (r.unsupported_reason or "")


def test_f3_one_shot_loop_carried_zero_phase_is_accepted():
    """Finding 3 (precision): a loop-carried phase that provably stays 0
    (iter_arg init 0, yielded unchanged) used to produce false RAW
    reports; the simulation now accepts it."""
    stock = _read("matmul_tma_s1_sm90.ttgir")
    mutated = stock.replace(
        "iter_args(%arg7 = %cst)", "iter_args(%arg7 = %cst, %ph = %c0_i32)"
    )
    # scf.yield gains the unchanged phase arg
    mutated = mutated.replace("scf.yield %acc_14#0 :", "scf.yield %acc_14#0, %ph :")
    mutated = mutated.replace(
        "ttng.wait_barrier %a_10, %c0_i32", "ttng.wait_barrier %a_10, %ph"
    )
    r = analyze_ttgir(mutated)
    assert r.status == "ok", r.unsupported_reason
    assert r.reports == []


def _inject_identity_noise(stock: str, target_line_marker: str, ssa: str) -> str:
    """Route ``ssa`` through +50/-50 (value unchanged) right before its
    use on the marked line, and add the %c50_i32 constant."""
    lines = stock.splitlines()
    out = []
    for line in lines:
        if "%c3_i32 = arith.constant 3 : i32" in line:
            out.append(line)
            out.append("    %c50_i32 = arith.constant 50 : i32 loc(#loc1)")
            continue
        if target_line_marker in line:
            indent = line[: len(line) - len(line.lstrip())]
            out.append(f"{indent}{ssa}_n1 = arith.addi {ssa}, %c50_i32 : i32")
            out.append(f"{indent}{ssa}_n2 = arith.subi {ssa}_n1, %c50_i32 : i32")
            out.append(
                line.replace(f"{ssa}]", f"{ssa}_n2]")
                .replace(f"{ssa} ", f"{ssa}_n2 ")
                .replace(f"{ssa},", f"{ssa}_n2,")
            )
            continue
        out.append(line)
    return "\n".join(out)


def test_f4_phase_chain_constant_beyond_window_is_unsupported():
    """Finding 4: a phase chain that matches the canonical parity inside
    the simulation window can diverge beyond it only by carrying an
    out-of-window constant — such chains are now rejected even when the
    value is unchanged (here: phase routed through +50/-50)."""
    stock = _read("matmul_tma_s3_sm90.ttgir")
    mutated = _inject_identity_noise(
        stock, "ttng.wait_barrier %acc_29, %acc_28", "%acc_28"
    )
    r = analyze_ttgir(mutated)
    assert r.status == "unsupported"
    assert "beyond the simulation window" in (r.unsupported_reason or "")


def test_f5_slot_chain_constant_beyond_window_is_unsupported():
    """Finding 5: same finite-window hole in resolve_slot — an
    out-of-window constant in a slot-index chain is now rejected."""
    stock = _read("matmul_tma_s3_sm90.ttgir")
    mutated = _inject_identity_noise(
        stock, "ttg.memdesc_index %acc[%acc_26]", "%acc_26"
    )
    r = analyze_ttgir(mutated)
    assert r.status == "unsupported"
    assert "beyond the simulation window" in (r.unsupported_reason or "")


def test_f6_unrelated_unevaluable_iter_arg_no_longer_poisons():
    """Finding 6 (honest over-abstention): an iter_arg with a constant
    init but an unevaluable yield used to poison the phase simulation of
    an unrelated canonical chain; only dependent args are advanced now."""
    stock = _read("matmul_tma_s3_sm90.ttgir")
    mutated = stock.replace(
        "iter_args(%arg7 = %cst,", "iter_args(%junk = %c0_i32, %arg7 = %cst,"
    )
    # %junk's yield is an op outside the parsed def set (muli) — it can
    # never be simulated, but the phase chain does not depend on it.
    mutated = mutated.replace("scf.yield %acc_33#0,", "scf.yield %junkmul, %acc_33#0,")
    mutated = mutated.replace(
        "%acc_22 = arith.subi %1, %c2_i32 : i32",
        "%junkmul = arith.muli %junk, %c2_i32 : i32\n"
        "      %acc_22 = arith.subi %1, %c2_i32 : i32",
    )
    r = analyze_ttgir(mutated)
    assert r.status == "ok", r.unsupported_reason
    assert r.reports == []


def test_f7_drain_requires_canonical_loop_bounds():
    """Finding 7: the drain predicate arithmetic reads the induction
    variable as the iteration index — a step≠1 loop must not pass it."""
    stock = _read("matmul_tma_s3_sm90.ttgir")
    mutated = stock.replace("step %c1_i32", "step %c2_i32")
    r = analyze_ttgir(mutated)
    assert r.status == "unsupported"
    assert "lower=0/step=1" in (r.unsupported_reason or "")


def test_f8_reuse_after_tma_store_requires_store_wait_drain():
    """Finding 8: a local_alloc reusing freed storage after a TMA
    local→global store needs an async_tma_store_wait {pendings=0} between
    them — the store is an async READ still in flight otherwise."""
    stock = _read("matmul_tma_s3_sm90.ttgir")
    # a reuse alloc squeezed between the l2g store and its store_wait
    mutated = stock.replace(
        "ttng.async_tma_store_wait {pendings = 0 : i32}",
        "%reuse2 = ttg.local_alloc : () -> "
        "!ttg.memdesc<64x64xf16, #shared1, #smem, mutable> loc(#loc18)\n"
        "    ttng.async_tma_store_wait {pendings = 0 : i32}",
    )
    r = analyze_ttgir(mutated)
    assert r.status == "unsupported"
    assert "async_tma_store_wait" in (r.unsupported_reason or "")

    # control: reuse AFTER the store_wait drains — still a proof
    control = stock.replace(
        "ttng.async_tma_store_wait {pendings = 0 : i32} loc(#loc18)",
        "ttng.async_tma_store_wait {pendings = 0 : i32} loc(#loc18)\n"
        "    %reuse2 = ttg.local_alloc : () -> "
        "!ttg.memdesc<64x64xf16, #shared1, #smem, mutable> loc(#loc18)",
    )
    r2 = analyze_ttgir(control)
    assert r2.status == "ok", r2.unsupported_reason


def test_f10_loop_fence_does_not_order_prologue_store_vs_epilogue_read():
    """Finding 10: a loop-segment fence executes only if the loop runs —
    it must not be credited with ordering a prologue store before an
    epilogue TMA read (trip count 0 skips it)."""
    stock = _read("matmul_tma_s3_sm90.ttgir")
    lines = stock.splitlines()
    store_line = next(ln for ln in lines if "%3 = ttg.local_alloc %2" in ln)
    fence_line = next(ln for ln in lines if "fence_async_shared" in ln)
    out = []
    for line in lines:
        if line is store_line or line is fence_line:
            continue
        if "%a_7 = ttg.local_alloc" in line:
            # plant the (SSA-dangling, parser-tolerated) store in the
            # prologue and the fence inside the loop
            out.append(store_line)
        if "ttng.warp_group_dot_wait" in line and "pendings = 1" in line:
            out.append(line)
            out.append(fence_line)
            continue
        out.append(line)
    r = analyze_ttgir("\n".join(out))
    assert r.status == "ok", r.unsupported_reason
    fence_reports = [rep for rep in r.reports if "fence_async_shared" in rep.message]
    assert fence_reports, [rep.message[:80] for rep in r.reports]


def test_f11_never_initialized_barrier_is_unsupported():
    """Finding 11: the full TMA protocol on a barrier allocation that is
    never init_barrier'd used to prove ok — mbarrier ops on uninitialized
    storage are UB."""
    stock = _read("matmul_tma_s3_sm90.ttgir")
    mutated = "\n".join(
        line for line in stock.splitlines() if "ttng.init_barrier" not in line
    )
    r = analyze_ttgir(mutated)
    assert r.status == "unsupported"
    reason = r.unsupported_reason or ""
    assert "never-initialized" in reason or "never init_barrier'd" in reason


def test_f12_protocol_ops_on_data_allocation_are_unsupported():
    """Finding 12: the sync/data partition must be two-directional —
    mbarrier protocol ops anchored on a wgmma-read DATA allocation (which
    has no init_barrier) must not validate."""
    stock = _read("matmul_tma_s3_sm90.ttgir")
    # anchor the loop arming on %a_13 — a prologue view of the wgmma-read
    # DATA allocation %a_7, defined well before the expect
    mutated = stock.replace(
        "ttng.barrier_expect %acc_39, 8192, %acc_23 : "
        "!ttg.memdesc<1xi64, #shared2, #smem, mutable>",
        "ttng.barrier_expect %a_13, 8192, %acc_23 : "
        "!ttg.memdesc<64x32xf16, #shared, #smem, mutable>",
    )
    r = analyze_ttgir(mutated)
    assert r.status == "unsupported"
    reason = r.unsupported_reason or ""
    assert (
        "never-initialized" in reason
        or "never init_barrier'd" in reason
        or "data access" in reason
    )
