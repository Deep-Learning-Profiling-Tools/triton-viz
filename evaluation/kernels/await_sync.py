"""C1.4 litmus corpus: the await abstraction (spin-loop synchronization).

Fence-ordered model (paper design-fence-order.md, stage 4): every kernel
fences (tl.debug_barrier) between its data accesses and its atomics, on
both twins, so the twins still differ by the one detail each names.

Three DRB-style groups — producer/consumer wait, CAS mutex, decoupled
look-back chain — each race-free version proved through the awaited-read
encoding (exit predicate as a termination premise + rf/sw machinery), each
racy twin breaking exactly one link (release, acquire, scope, the RMW
unlock).

Verdicts here are CONDITIONAL ON TERMINATION (assumes_termination): the
static provenance carries "+assumes-termination", C2 replay is classified
unavailable before any execution, and C3 is excluded symmetrically. The
dynamic column is the comparison datum: the interpreter fail-stops on the
host-level spin (per-instance value in control flow).
"""

from typing import Any, Literal

import torch
import triton
import triton.language as tl

from evaluation.spec import Corpus, LaunchSpec

_Expected = Literal["race", "race-free"]

CORPUS = Corpus("await_sync")

BLOCK = 64


# ── producer_consumer_wait ───────────────────────────────────────


@triton.jit
def pc_wait_kernel(flag_ptr, data_ptr, out_ptr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    if pid == 0:
        offs = tl.arange(0, BLOCK)
        tl.store(data_ptr + offs, offs)
        tl.debug_barrier()
        tl.atomic_xchg(flag_ptr, 1, sem="release")
    else:
        while tl.atomic_add(flag_ptr, 0, sem="acquire") != 1:
            pass
        tl.debug_barrier()
        offs = tl.arange(0, BLOCK)
        v = tl.load(data_ptr + offs)
        tl.store(out_ptr + pid * BLOCK + offs, v)


@triton.jit
def pc_wait_relaxed_writer_kernel(flag_ptr, data_ptr, out_ptr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    if pid == 0:
        offs = tl.arange(0, BLOCK)
        tl.store(data_ptr + offs, offs)
        tl.debug_barrier()
        tl.atomic_xchg(flag_ptr, 1, sem="relaxed")
    else:
        while tl.atomic_add(flag_ptr, 0, sem="acquire") != 1:
            pass
        tl.debug_barrier()
        offs = tl.arange(0, BLOCK)
        v = tl.load(data_ptr + offs)
        tl.store(out_ptr + pid * BLOCK + offs, v)


@triton.jit
def pc_wait_relaxed_spin_kernel(flag_ptr, data_ptr, out_ptr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    if pid == 0:
        offs = tl.arange(0, BLOCK)
        tl.store(data_ptr + offs, offs)
        tl.debug_barrier()
        tl.atomic_xchg(flag_ptr, 1, sem="release")
    else:
        while tl.atomic_add(flag_ptr, 0, sem="relaxed") != 1:
            pass
        tl.debug_barrier()
        offs = tl.arange(0, BLOCK)
        v = tl.load(data_ptr + offs)
        tl.store(out_ptr + pid * BLOCK + offs, v)


@triton.jit
def pc_wait_cta_scope_kernel(flag_ptr, data_ptr, out_ptr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    if pid == 0:
        offs = tl.arange(0, BLOCK)
        tl.store(data_ptr + offs, offs)
        tl.debug_barrier()
        tl.atomic_xchg(flag_ptr, 1, sem="release", scope="cta")
    else:
        while tl.atomic_add(flag_ptr, 0, sem="acquire", scope="cta") != 1:
            pass
        tl.debug_barrier()
        offs = tl.arange(0, BLOCK)
        v = tl.load(data_ptr + offs)
        tl.store(out_ptr + pid * BLOCK + offs, v)


@triton.jit
def pc_wait_or_poll_kernel(flag_ptr, data_ptr, out_ptr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    if pid == 0:
        offs = tl.arange(0, BLOCK)
        tl.store(data_ptr + offs, offs)
        tl.debug_barrier()
        tl.atomic_xchg(flag_ptr, 1, sem="release")
    else:
        while tl.atomic_or(flag_ptr, 0, sem="acquire") != 1:
            pass
        tl.debug_barrier()
        offs = tl.arange(0, BLOCK)
        v = tl.load(data_ptr + offs)
        tl.store(out_ptr + pid * BLOCK + offs, v)


@triton.jit
def pc_wait_xor_poll_kernel(flag_ptr, data_ptr, out_ptr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    if pid == 0:
        offs = tl.arange(0, BLOCK)
        tl.store(data_ptr + offs, offs)
        tl.debug_barrier()
        tl.atomic_xchg(flag_ptr, 1, sem="release")
    else:
        while tl.atomic_xor(flag_ptr, 0, sem="acquire") != 1:
            pass
        tl.debug_barrier()
        offs = tl.arange(0, BLOCK)
        v = tl.load(data_ptr + offs)
        tl.store(out_ptr + pid * BLOCK + offs, v)


def _pc_args(seed: int) -> tuple:
    return (
        torch.zeros(1, dtype=torch.int32),
        torch.zeros(BLOCK, dtype=torch.int32),
        torch.zeros(2 * BLOCK, dtype=torch.int32),
    )


_PC_SIG = {
    "flag_ptr": "*i32",
    "data_ptr": "*i32",
    "out_ptr": "*i32",
    "BLOCK": "constexpr",
}
_PC_PAIR = ("tl.store(data_ptr + offs, offs)", "v = tl.load(data_ptr + offs)")

_PC_SPECS: tuple[tuple[str, Any, _Expected, str], ...] = (
    (
        "pc_wait_no",
        pc_wait_kernel,
        "race-free",
        "release publish + acquire spin: proof conditional on termination",
    ),
    (
        "pc_wait_relaxed_writer_yes",
        pc_wait_relaxed_writer_kernel,
        "race",
        "relaxed publisher heads no release sequence",
    ),
    (
        "pc_wait_relaxed_spin_yes",
        pc_wait_relaxed_spin_kernel,
        "race",
        "relaxed spinner acquires nothing",
    ),
    (
        "pc_wait_cta_scope_yes",
        pc_wait_cta_scope_kernel,
        "race",
        "cta scope does not cover the peer CTA",
    ),
    (
        "pc_wait_or_poll_no",
        pc_wait_or_poll_kernel,
        "race-free",
        "identity atomic_or(0) poll: the write-back republishes the "
        "observation, same proof as the add(0) poll",
    ),
    (
        "pc_wait_xor_poll_no",
        pc_wait_xor_poll_kernel,
        "race-free",
        "identity atomic_xor(0) poll: the write-back republishes the "
        "observation, same proof as the add(0) poll",
    ),
)

for name, fn, expected, note in _PC_SPECS:
    CORPUS.add(
        LaunchSpec(
            name=name,
            kernel_fn=fn,
            signature=_PC_SIG,
            constexprs={"BLOCK": BLOCK},
            make_args=_pc_args,
            grid=(2,),
            expected=expected,
            race_pair=None if expected == "race-free" else _PC_PAIR,
            pattern="producer-consumer-wait",
            params_note=note,
        )
    )


# ── flag resets / reads before the publish (pre-exit representative) ──
#
# The await collapse drops the spin's failed iterations; the pre-exit
# representative record carries their footprint. These rows pin the two
# corners that used to prove silently — a value-modeled weak atomic write
# (scope-mismatched reset) and a plain read of the awaited flag — plus the
# morally-strong reset twin that must keep proving.


@triton.jit
def pc_wait_cta_reset_kernel(flag_ptr, data_ptr, out_ptr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    if pid == 0:
        offs = tl.arange(0, BLOCK)
        tl.store(data_ptr + offs, offs)
        tl.debug_barrier()
        tl.atomic_xchg(flag_ptr, 0, sem="relaxed", scope="cta")
        tl.atomic_xchg(flag_ptr, 1, sem="release")
    else:
        while tl.atomic_add(flag_ptr, 0, sem="acquire") != 1:
            pass
        tl.debug_barrier()
        offs = tl.arange(0, BLOCK)
        v = tl.load(data_ptr + offs)
        tl.store(out_ptr + pid * BLOCK + offs, v)


@triton.jit
def pc_wait_atomic_reset_kernel(flag_ptr, data_ptr, out_ptr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    if pid == 0:
        offs = tl.arange(0, BLOCK)
        tl.store(data_ptr + offs, offs)
        tl.debug_barrier()
        tl.atomic_xchg(flag_ptr, 0, sem="relaxed")
        tl.atomic_xchg(flag_ptr, 1, sem="release")
    else:
        while tl.atomic_add(flag_ptr, 0, sem="acquire") != 1:
            pass
        tl.debug_barrier()
        offs = tl.arange(0, BLOCK)
        v = tl.load(data_ptr + offs)
        tl.store(out_ptr + pid * BLOCK + offs, v)


@triton.jit
def pc_wait_flag_read_kernel(flag_ptr, data_ptr, out_ptr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    if pid == 0:
        offs = tl.arange(0, BLOCK)
        tl.store(data_ptr + offs, offs)
        tl.debug_barrier()
        fv = tl.load(flag_ptr)
        tl.store(out_ptr, fv)
        tl.atomic_xchg(flag_ptr, 1, sem="release")
    else:
        while tl.atomic_add(flag_ptr, 0, sem="acquire") != 1:
            pass
        tl.debug_barrier()
        offs = tl.arange(0, BLOCK)
        v = tl.load(data_ptr + offs)
        tl.store(out_ptr + pid * BLOCK + offs, v)


_PC_POLL_NEEDLE = 'while tl.atomic_add(flag_ptr, 0, sem="acquire") != 1:'

CORPUS.add(
    LaunchSpec(
        name="pc_wait_cta_reset_yes",
        kernel_fn=pc_wait_cta_reset_kernel,
        signature=_PC_SIG,
        constexprs={"BLOCK": BLOCK},
        make_args=_pc_args,
        grid=(2,),
        expected="race",
        # The representative's designed demonstration: the reset is a
        # VALUE-MODELED weak atomic write (equal width, mismatched scope)
        # — it opens no rf_unknown escape and the termination premise pins
        # the poll to the publisher, so the kernel silently proved before
        # the pre-exit representative. The WAW lands on the reset line vs
        # the poll line (the rep carries the poll's source location).
        race_pair=(
            'tl.atomic_xchg(flag_ptr, 0, sem="relaxed", scope="cta")',
            _PC_POLL_NEEDLE,
        ),
        pattern="producer-consumer-wait",
        params_note="cta-scoped relaxed reset po-before the gpu publish "
        "races the spin's failed iterations",
    )
)
CORPUS.add(
    LaunchSpec(
        name="pc_wait_atomic_reset_no",
        kernel_fn=pc_wait_atomic_reset_kernel,
        signature=_PC_SIG,
        constexprs={"BLOCK": BLOCK},
        make_args=_pc_args,
        grid=(2,),
        expected="race-free",
        pattern="producer-consumer-wait",
        params_note="gpu-scoped relaxed reset is morally strong with the "
        "poll and its representative: mutually atomic, no conflict",
    )
)
CORPUS.add(
    LaunchSpec(
        name="pc_wait_flag_read_yes",
        kernel_fn=pc_wait_flag_read_kernel,
        signature=_PC_SIG,
        constexprs={"BLOCK": BLOCK},
        make_args=_pc_args,
        grid=(2,),
        expected="race",
        # Corner B: the plain read races the identity write-backs of the
        # spin's failed iterations — the WAR rides the rep's write half.
        race_pair=(
            "fv = tl.load(flag_ptr)",
            _PC_POLL_NEEDLE,
        ),
        pattern="producer-consumer-wait",
        params_note="plain read of the awaited flag po-before the publish "
        "races the failed iterations' write-backs",
    )
)


# ── mutex via CAS loop ───────────────────────────────────────────


@triton.jit
def mutex_kernel(lock_ptr, x_ptr, out_ptr):
    pid = tl.program_id(0)
    while tl.atomic_cas(lock_ptr, 0, 1, sem="acquire") != 0:
        pass
    tl.debug_barrier()
    v = tl.load(x_ptr)
    tl.store(x_ptr, v + 1)
    tl.debug_barrier()
    tl.atomic_xchg(lock_ptr, 0, sem="release")
    tl.store(out_ptr + pid, 1)


@triton.jit
def mutex_plain_unlock_kernel(lock_ptr, x_ptr, out_ptr):
    pid = tl.program_id(0)
    while tl.atomic_cas(lock_ptr, 0, 1, sem="acquire") != 0:
        pass
    tl.debug_barrier()
    v = tl.load(x_ptr)
    tl.store(x_ptr, v + 1)
    tl.debug_barrier()
    tl.store(lock_ptr, 0)
    tl.store(out_ptr + pid, 1)


@triton.jit
def mutex_relaxed_cas_kernel(lock_ptr, x_ptr, out_ptr):
    pid = tl.program_id(0)
    while tl.atomic_cas(lock_ptr, 0, 1, sem="relaxed") != 0:
        pass
    tl.debug_barrier()
    v = tl.load(x_ptr)
    tl.store(x_ptr, v + 1)
    tl.debug_barrier()
    tl.atomic_xchg(lock_ptr, 0, sem="release")
    tl.store(out_ptr + pid, 1)


def _mutex_args(seed: int) -> tuple:
    return (
        torch.zeros(1, dtype=torch.int32),
        torch.zeros(1, dtype=torch.int32),
        torch.zeros(4, dtype=torch.int32),
    )


_MUTEX_SIG = {"lock_ptr": "*i32", "x_ptr": "*i32", "out_ptr": "*i32"}
_MUTEX_PAIR = ("v = tl.load(x_ptr)", "tl.store(x_ptr, v + 1)")

_MUTEX_SPECS: tuple[tuple[str, Any, _Expected, str], ...] = (
    (
        "mutex_cas_no",
        mutex_kernel,
        "race-free",
        "CAS lock (acquire) + xchg unlock (release): needs RMW immediacy — "
        "two acquisitions of the same 0 are unsat",
    ),
    (
        "mutex_plain_unlock_yes",
        mutex_plain_unlock_kernel,
        "race",
        "plain-store unlock breaks the release chain (and the closed world)",
    ),
    (
        "mutex_relaxed_cas_yes",
        mutex_relaxed_cas_kernel,
        "race",
        "relaxed CAS acquires nothing",
    ),
)

for name, fn, expected, note in _MUTEX_SPECS:
    CORPUS.add(
        LaunchSpec(
            name=name,
            kernel_fn=fn,
            signature=_MUTEX_SIG,
            constexprs={},
            make_args=_mutex_args,
            grid=(2,),
            expected=expected,
            race_pair=None if expected == "race-free" else _MUTEX_PAIR,
            pattern="mutex-cas",
            params_note=note,
        )
    )


# ── decoupled look-back chain ────────────────────────────────────


@triton.jit
def lookback_kernel(flag_ptr, out_ptr):
    pid = tl.program_id(0)
    if pid > 0:
        while tl.atomic_add(flag_ptr + pid - 1, 0, sem="acquire") == 0:
            pass
        tl.debug_barrier()
        prev = tl.load(out_ptr + pid - 1)
        tl.store(out_ptr + pid, prev + 1)
    else:
        tl.store(out_ptr + pid, 1)
    tl.debug_barrier()
    tl.atomic_xchg(flag_ptr + pid, 1, sem="release")


@triton.jit
def lookback_cta_scope_kernel(flag_ptr, out_ptr):
    pid = tl.program_id(0)
    if pid > 0:
        while tl.atomic_add(flag_ptr + pid - 1, 0, sem="acquire", scope="cta") == 0:
            pass
        tl.debug_barrier()
        prev = tl.load(out_ptr + pid - 1)
        tl.store(out_ptr + pid, prev + 1)
    else:
        tl.store(out_ptr + pid, 1)
    tl.debug_barrier()
    tl.atomic_xchg(flag_ptr + pid, 1, sem="release", scope="cta")


def _lookback_args(seed: int) -> tuple:
    return (
        torch.zeros(4, dtype=torch.int32),
        torch.zeros(4, dtype=torch.int32),
    )


_LB_SIG = {"flag_ptr": "*i32", "out_ptr": "*i32"}

CORPUS.add(
    LaunchSpec(
        name="lookback_chain_no",
        kernel_fn=lookback_kernel,
        signature=_LB_SIG,
        constexprs={},
        make_args=_lookback_args,
        grid=(4,),
        expected="race-free",
        pattern="lookback-chain",
        params_note="pid i spins on flag[i-1] (pid-dependent loop-invariant "
        "address), publishes flag[i] with release",
    )
)
CORPUS.add(
    LaunchSpec(
        name="lookback_cta_scope_yes",
        kernel_fn=lookback_cta_scope_kernel,
        signature=_LB_SIG,
        constexprs={},
        make_args=_lookback_args,
        grid=(4,),
        expected="race",
        # Acceptable endpoints (witness matching is subset-based): the
        # look-back read races the PREDECESSOR's publish — the pid-0
        # predecessor stores in the else branch, pid>0 predecessors in the
        # then branch. The two-copy closed world can only source the
        # adjacent-to-pid-0 variant (a pid>=2 chain needs a third
        # instance), so the reported pair uses the else-branch store.
        race_pair=(
            "prev = tl.load(out_ptr + pid - 1)",
            "tl.store(out_ptr + pid, prev + 1)",
            "tl.store(out_ptr + pid, 1)",
        ),
        pattern="lookback-chain",
        params_note="cta scope cannot order cross-CTA neighbors",
    )
)
