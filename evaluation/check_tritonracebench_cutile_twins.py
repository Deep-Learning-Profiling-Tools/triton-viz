"""Validate the cuTile repair twins: catalog pairing and concrete outputs.

Run from the detector checkout; ``--gpu`` launches each new row repeatedly and
checks the outputs its Triton twin is specified to produce. GPU success checks
execution and outputs; the race-free arguments are the twins' ports of the
Triton arguments in evaluation/TRITONRACEBENCH_REPAIRS.md and do not follow
from the absence of observed output errors.
"""

import argparse
import itertools
import json
from pathlib import Path

import torch

from evaluation.kernels.tritonracebench import CORPUS as TRITON
from evaluation.kernels.tritonracebench_cutile import CORPUS as CUTILE, ROWS

# The seven repair rows ported here; the two remaining Triton-only rows are
# fence-dropped by construction and have no cuTile spelling (see the module
# docstring of evaluation/kernels/tritonracebench_cutile.py).
NEW = (
    "trb013_batch_ticket_no",
    "trb016_atomic_flag_observation_no",
    "trb017_cas_unlock_no",
    "trb021_role_specific_order_no",
    "trb025_both_consumer_branches_no",
    "trb025_failed_cas_arrival_no",
    "trb026_fenced_tile_handoff_no",
)
UNPORTABLE = ("trb026_reread_unfenced_yes", "trb026_guarded_no_producer_fence_yes")


def check_catalog():
    triton = {s.name: s for s in TRITON.specs}
    cutile = {s.name: s for s in CUTILE.specs}
    assert len(ROWS) == 69, len(ROWS)
    assert len(cutile) == 69, len(cutile)
    # every cuTile row is a name twin of a Triton row, with the same label,
    # pattern and grid; the corpus is a subset of the Triton roster.
    for name, spec in cutile.items():
        twin = triton[name]
        assert spec.expected == twin.expected, (name, spec.expected, twin.expected)
        assert spec.pattern == twin.pattern, (name, spec.pattern, twin.pattern)
        assert tuple(spec.grid) == tuple(twin.grid), (name, spec.grid, twin.grid)
    missing = sorted(set(triton) - set(cutile))
    assert missing == sorted(UNPORTABLE), missing
    for name in NEW:
        assert name in cutile and cutile[name].expected == "race-free", name
    return triton, cutile


# ── the token chains the race-free labels rest on ────────────────
# cuda.tile has no fence. The compiler's token pass gives an instance
# two kinds of edge (see the corpus module docstring): same-array
# chaining, and join_tokens into and out of a RELEASE/ACQ_REL atomic.
# A RELAXED or ACQUIRE-only atomic receives no join, so these chains are
# a property of the captured IR and not of the language in general. A
# recapture that lost one would silently invert a label, so each row's
# chain is asserted here, independently of any detector verdict.

# row -> (a release/acq_rel atomic must carry a preceding store,
#         a later load must be gated on an atomic's result)
TOKEN_CHAINS = {
    "trb013_batch_ticket_no": (False, False),  # RMW indivisibility only
    "trb016_atomic_flag_observation_no": (True, True),
    "trb017_cas_unlock_no": (True, True),
    "trb021_role_specific_order_no": (True, True),
    "trb025_both_consumer_branches_no": (True, True),
    "trb025_failed_cas_arrival_no": (True, True),
    "trb026_fenced_tile_handoff_no": (True, True),
}
_RELEASING = ("MemoryOrder.RELEASE", "MemoryOrder.ACQ_REL")


def _parse_tokens(ir):
    """token id -> (op, {input token ids}); accesses by kind.

    Handles the three token producers of the captured IR (memory access,
    join_tokens, loop) plus the loop's break/continue operands, which are
    what carry a spin loop's acquiring atomic out of its body.
    """
    produces, kinds = (
        {},
        {"store": set(), "load": set(), "atomic": set(), "release": set()},
    )
    loops = []  # (indent, [result token names]) innermost last
    for raw in ir.splitlines():
        line, indent = raw.strip(), len(raw) - len(raw.lstrip())
        # A region (loop or if) stays open until a line at its own indent
        # that is not one of its header continuations; its break/continue/
        # yield operands are always deeper than the header.
        while (
            loops
            and indent <= loops[-1][0]
            and not line.startswith(("do", "then", "else", "("))
        ):
            loops.pop()
        if "= loop" in line or line.startswith("if(") or "= if(" in line:
            results = (
                [
                    r.split(":")[0].strip()
                    for r in line.split("=", 1)[0].split(",")
                    if "Token" in r
                ]
                if "=" in line.split("(", 1)[0]
                else []
            )
            for r in results:
                produces.setdefault(r, ("region", set()))
            loops.append((indent, results))
            continue
        if line.startswith(("break", "continue", "yield")) and loops:
            operands = (
                [o.strip() for o in line.split(None, 1)[1].split(",")]
                if len(line.split(None, 1)) > 1
                else []
            )
            for result, operand in zip(loops[-1][1], operands):
                produces[result][1].add(operand)
            continue
        if "= join_tokens(" in line:
            name = line.split(":", 1)[0].strip()
            inside = line.split("tokens=(", 1)[1].rsplit(")", 2)[0]
            produces[name] = (
                "join",
                {t.strip() for t in inside.split(",") if t.strip()},
            )
            continue
        for op in ("store_pointer", "load_pointer", "tile_atomic_"):
            if f"= {op}" in line or (op == "tile_atomic_" and "= tile_atomic_" in line):
                results = [p.strip() for p in line.split("=", 1)[0].split(",")]
                token_out = next(
                    (r.split(":")[0].strip() for r in results if "Token" in r), None
                )
                token_in = None
                if "token=" in line:
                    token_in = line.split("token=", 1)[1].split(",")[0].strip()
                if token_out is None:
                    break
                short = (
                    "store"
                    if op == "store_pointer"
                    else "load"
                    if op == "load_pointer"
                    else "atomic"
                )
                produces[token_out] = (short, {token_in} if token_in else set())
                kinds[short].add(token_out)
                if short == "atomic" and any(o in line for o in _RELEASING):
                    kinds["release"].add(token_out)
                break
    return produces, kinds


def _reaches(produces, start, targets):
    seen, stack = set(), [start]
    while stack:
        cur = stack.pop()
        if cur in seen:
            continue
        seen.add(cur)
        if cur in targets:
            return True
        stack.extend(produces.get(cur, ("", set()))[1])
    return False


# Racy rows whose label comes from the ABSENCE of the publication chain:
# they must have no RELEASE/ACQ_REL atomic at all, which keeps the
# assertion above from passing vacuously.
NO_PUBLICATION = ("trb021_acquire_only_yes", "trb016_pc_wait_relaxed_writer_yes")


def check_token_chains(specs):
    for name in NO_PUBLICATION:
        _, kinds = _parse_tokens(specs[name]["ir"])
        assert not kinds["release"], (
            f"{name}: this racy row is expected to have no RELEASE/ACQ_REL "
            "atomic; the publication assertion would be vacuous"
        )
    for name, (needs_release, needs_gate) in TOKEN_CHAINS.items():
        ir = specs[name]["ir"]
        produces, kinds = _parse_tokens(ir)
        assert kinds["atomic"], name
        if needs_release:
            assert kinds["release"], f"{name}: no RELEASE/ACQ_REL atomic in the IR"
            published = [
                tok
                for tok in kinds["release"]
                if _reaches(produces, next(iter(produces[tok][1]), ""), kinds["store"])
            ]
            assert published, (
                f"{name}: no RELEASE/ACQ_REL atomic carries a preceding store; "
                "the publication no longer orders the payload"
            )
        if needs_gate:
            gated = [
                tok
                for tok in kinds["load"]
                if _reaches(produces, next(iter(produces[tok][1]), ""), kinds["atomic"])
            ]
            assert gated, (
                f"{name}: no load is gated on an atomic result; the consumer read "
                "is no longer ordered after the synchronizing atomic"
            )


def _launch(name, args):
    import cuda.tile as ct

    row = ROWS[name]
    grid3 = tuple(row["grid"]) + (1,) * (3 - len(row["grid"]))
    ct.launch(torch.cuda.current_stream(), grid3, row["kernel"], args)
    torch.cuda.synchronize()


def check_gpu(repetitions):
    assert torch.cuda.is_available()
    for _ in range(repetitions):
        for name in NEW:
            args = tuple(
                a.cuda() if torch.is_tensor(a) else a
                for a in ROWS[name]["make_args"](0)
            ) + tuple(ROWS[name]["consts"])
            _launch(name, args)
            got = [a.cpu() for a in args if torch.is_tensor(a)]
            if name == "trb013_batch_ticket_no":
                # four fetch-add(2) tickets reserve four disjoint two-slot batches
                assert got[0].item() == 8, got[0]
                pairs = got[1][:8].view(4, 2)
                assert torch.equal(pairs[:, 0], pairs[:, 1]), pairs
                assert sorted(pairs[:, 0].tolist()) == list(range(4)), pairs
                assert not got[1][8:].any()
            elif name == "trb016_atomic_flag_observation_no":
                assert got[0].item() == 1
                assert got[1].tolist() == list(range(64))
                assert got[2][0].item() in (0, 1)  # the producer's flag observation
                assert got[2][64:128].tolist() == list(range(64))
            elif name == "trb017_cas_unlock_no":
                assert got[0].item() == 0  # unlocked by the release CAS
                assert got[1].item() == 2  # both critical sections ran
                assert got[2][:2].tolist() == [1, 1]
            elif name == "trb021_role_specific_order_no":
                assert got[0].item() == 1
                assert got[1].item() == 1
                assert got[2][0].item() == 0  # the producer never writes out[0]
                # the consumer publishes iff it acquired, and then it read data
                assert got[2][1].item() in (0, 1)
                if got[2][1].item() != 0:
                    assert got[1].item() == 1
            elif name in (
                "trb025_failed_cas_arrival_no",
                "trb025_both_consumer_branches_no",
            ):
                assert got[0].item() == 1
                payload = [float(i + 1) for i in range(16)]
                assert got[1].tolist() == payload
                assert got[2][:16].tolist() == payload
                assert got[2][16:32].tolist() == payload
            elif name == "trb026_fenced_tile_handoff_no":
                assert got[0].item() == 1
                assert got[1].tolist() == list(range(1, 17))
                assert got[2].tolist() in (list(range(1, 17)), [0] * 16)
            else:  # pragma: no cover - the tuple above is closed
                raise AssertionError(name)
    # The ticket argument holds for every arrival order, independently of any
    # run: the k-th arriving instance takes base 2k, so no two instances share
    # a slot whichever order they arrive in.
    for order in itertools.permutations(range(4)):
        owner = {}
        for rank, instance in enumerate(order):
            for slot in (2 * rank, 2 * rank + 1):
                assert slot not in owner, (order, slot)
                owner[slot] = instance
        assert sorted(owner) == list(range(8)), order
        assert len(set(owner.values())) == 4, order


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", action="store_true", help="launch each new row")
    ap.add_argument("--repetitions", type=int, default=10)
    ns = ap.parse_args()
    check_catalog()
    print(
        f"catalog: 69 cuTile rows, {len(NEW)} new twins, "
        f"{len(UNPORTABLE)} Triton-only rows accounted for"
    )
    specs = json.loads(
        (
            Path(__file__).parent / "kernels" / "tritonracebench_cutile_specs.json"
        ).read_text()
    )["rows"]
    check_token_chains(specs)
    print(
        f"token chains: {len(TOKEN_CHAINS)} rows, publication and gating edges present"
    )
    if ns.gpu:
        check_gpu(ns.repetitions)
        print(f"gpu: {len(NEW)} rows x {ns.repetitions} launches, outputs as specified")


if __name__ == "__main__":
    main()
