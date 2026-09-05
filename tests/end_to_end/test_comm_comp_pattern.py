"""Category 8a e2e pins: the communication-kernel pattern, single-GPU half.

DeepSeek-V3-style SM partition (advisor positioning 2026-07-11): the pid
range splits into a COMM role (publish a global-memory payload, arrive
on a semaphore with a release xchg) and a COMP role (acquire-poll the
semaphore, then read the payload). The guarded producer/consumer family
with a role split on pid instead of pid parity — decided by the STATIC
track (the await abstraction models the spin; the interpreter cannot
execute pid-divergent while loops).

The arrive is a release XCHG: a release ADD-arrive plus an add(0)
acquire poll puts two value-interacting RMW records on the semaphore —
the S6 ticket-lock boundary — and the sw edge cannot be derived today
(the corpus comment in tritonracebench.py records the probe).

Corpus twins live in evaluation/kernels/tritonracebench.py (trb025);
these pins are self-contained copies so the test suite does not import
the evaluation package.
"""

from types import SimpleNamespace

import torch
import triton
import triton.language as tl
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource

from triton_viz.clients.race_detector.compiled.client import CompiledRaceDetector

N_COMM = 1
BLOCK = 16
GRID = (3,)  # 1 comm pid + 2 comp pids


@triton.jit
def _comm_comp_kernel(
    sem_ptr, payload_ptr, out_ptr, N_COMM: tl.constexpr, BLOCK: tl.constexpr
):
    pid = tl.program_id(0)
    if pid < N_COMM:
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        tl.store(payload_ptr + offs, (offs + 1).to(tl.float32))
        tl.atomic_xchg(sem_ptr, 1, sem="release")
    else:
        while tl.atomic_add(sem_ptr, 0, sem="acquire") != N_COMM:
            pass
        offs = tl.arange(0, BLOCK)
        v = tl.load(payload_ptr + offs)
        tl.store(out_ptr + (pid - N_COMM) * BLOCK + tl.arange(0, BLOCK), v)


@triton.jit
def _relaxed_poll_kernel(
    sem_ptr, payload_ptr, out_ptr, N_COMM: tl.constexpr, BLOCK: tl.constexpr
):
    pid = tl.program_id(0)
    if pid < N_COMM:
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        tl.store(payload_ptr + offs, (offs + 1).to(tl.float32))
        tl.atomic_xchg(sem_ptr, 1, sem="release")
    else:
        while tl.atomic_add(sem_ptr, 0, sem="relaxed") != N_COMM:
            pass
        offs = tl.arange(0, BLOCK)
        v = tl.load(payload_ptr + offs)
        tl.store(out_ptr + (pid - N_COMM) * BLOCK + tl.arange(0, BLOCK), v)


@triton.jit
def _poll_initial_kernel(
    sem_ptr, payload_ptr, out_ptr, N_COMM: tl.constexpr, BLOCK: tl.constexpr
):
    pid = tl.program_id(0)
    if pid < N_COMM:
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        tl.store(payload_ptr + offs, (offs + 1).to(tl.float32))
        tl.atomic_xchg(sem_ptr, 1, sem="release")
    else:
        while tl.atomic_add(sem_ptr, 0, sem="acquire") != 0:
            pass
        offs = tl.arange(0, BLOCK)
        v = tl.load(payload_ptr + offs)
        tl.store(out_ptr + (pid - N_COMM) * BLOCK + tl.arange(0, BLOCK), v)


@triton.jit
def _role_skip_kernel(
    sem_ptr, payload_ptr, out_ptr, N_COMM: tl.constexpr, BLOCK: tl.constexpr
):
    pid = tl.program_id(0)
    if pid < N_COMM:
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        tl.store(payload_ptr + offs, (offs + 1).to(tl.float32))
        tl.atomic_xchg(sem_ptr, 1, sem="release")
    else:
        if pid == N_COMM:
            while tl.atomic_add(sem_ptr, 0, sem="acquire") != N_COMM:
                pass
        offs = tl.arange(0, BLOCK)
        v = tl.load(payload_ptr + offs)
        tl.store(out_ptr + (pid - N_COMM) * BLOCK + tl.arange(0, BLOCK), v)


_SIG = {
    "sem_ptr": "*i32", "payload_ptr": "*fp32", "out_ptr": "*fp32",
    "N_COMM": "constexpr", "BLOCK": "constexpr",
}  # fmt: skip


def _run_static(kernel):
    src = ASTSource(
        fn=kernel, signature=_SIG, constexprs={"N_COMM": N_COMM, "BLOCK": BLOCK}
    )
    ttir = triton.compile(src, target=GPUTarget("cuda", 80, 32)).asm["ttir"]
    det = CompiledRaceDetector(confirm_races=False, differential_check=False)
    args = (
        torch.zeros(1, dtype=torch.int32),
        torch.zeros(BLOCK, dtype=torch.float32),
        torch.zeros(2 * BLOCK, dtype=torch.float32),
    )
    jit_fn = SimpleNamespace(arg_names=list(_SIG))
    det.pre_warmup_callback(jit_fn, *args, grid=GRID)
    det.post_warmup_callback(jit_fn, SimpleNamespace(asm={"ttir": ttir}))
    det.finalize()
    return det


def _payload_reports(det):
    return [
        r
        for r in det.last_global_reports
        if {r.first_record.tensor_name, r.second_record.tensor_name} == {"payload_ptr"}
    ]


def test_comm_comp_control_proves_with_termination_premise():
    """release-xchg arrive + acquire poll: every comp read of the payload
    is ordered after the comm publish; the await abstraction makes the
    proof conditional on termination."""
    det = _run_static(_comm_comp_kernel)
    assert det.last_global_status == "ok"
    assert det.last_global_assumes_termination
    assert det.last_global_provenance.endswith("+assumes-termination")


def test_comm_comp_relaxed_poll_races_on_payload():
    """Twin (a): the poll carries the arrival value at relaxed — no
    ordering, so the payload read races with the comm publish."""
    det = _run_static(_relaxed_poll_kernel)
    assert det.last_global_status == "races"
    assert _payload_reports(det)


def test_comm_comp_poll_initial_value_races_on_payload():
    """Twin (b): polling the WRONG counter value (the initial 0) exits the
    spin without ever acquiring the release arrival — no sw edge."""
    det = _run_static(_poll_initial_kernel)
    assert det.last_global_status == "races"
    assert _payload_reports(det)


def test_comm_comp_role_branch_skipping_poll_races_on_payload():
    """Twin (c): only the first comp pid polls; the role split's other
    branch reads the payload with no synchronization at all."""
    det = _run_static(_role_skip_kernel)
    assert det.last_global_status == "races"
    assert _payload_reports(det)
