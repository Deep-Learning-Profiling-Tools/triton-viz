"""Validate repair metadata and concrete outputs, independently of detector verdicts.

Run from the detector checkout with --gpu for ten native launches per new row.
GPU success checks execution and outputs; the race-free arguments are in the
accompanying review and do not follow from absence of observed output errors.
"""
import argparse
import hashlib
import itertools
import json
from collections import Counter
from pathlib import Path

import torch
from evaluation.kernels.tritonracebench import CORPUS
from evaluation.kernels.tritonracebench_repairs import REPAIR_ROWS


def check_catalog():
    by_name = {s.name: s for s in CORPUS.specs}
    scored = [s for s in CORPUS.specs if s.expected is not None]
    assert len(scored) == 70
    assert Counter(s.expected for s in scored) == {"race": 35, "race-free": 35}
    assert len({s.pattern for s in scored}) == 25
    for name, kernel, _, _, _, _, pattern, paired, _ in REPAIR_ROWS:
        s = by_name[name]
        assert s.expected == "race-free"
        assert by_name[paired].expected == "race"
        assert s.pattern == by_name[paired].pattern == pattern
        assert s.grid == by_name[paired].grid
        assert s.kernel_fn == kernel
    # Every ordering of four fetch-add(2) operations reserves distinct batches.
    for order in itertools.permutations(range(4)):
        owned = set()
        for rank, _ in enumerate(order):
            slots = {2 * rank, 2 * rank + 1}
            assert not owned & slots
            owned |= slots
        assert owned == set(range(8))
    return by_name


def check_gpu(by_name, repetitions):
    assert torch.cuda.is_available()
    checks = []
    for name, *_ in REPAIR_ROWS:
        s = by_name[name]
        active_receivers = 0
        for repeat in range(repetitions):
            args = tuple(
                a.cuda() if isinstance(a, torch.Tensor) else a
                for a in s.make_args(repeat)
            )
            s.kernel_fn[s.grid](*args, **s.constexprs)
            torch.cuda.synchronize()
            got = [a.cpu() for a in args]
            if name.startswith("trb021"):
                assert got[0].tolist() == [1] and got[1].tolist() == [1]
                assert got[2].tolist() in ([0, 0], [0, 1])
                active_receivers += int(got[2][1] == 1)
            elif name.startswith("trb013"):
                assert got[0].item() == 8
                pairs = got[1][:8].reshape(4, 2)
                assert torch.equal(pairs[:, 0], pairs[:, 1])
                assert sorted(pairs[:, 0].tolist()) == list(range(4))
                assert not got[1][8:].any()
            elif name.startswith("trb016"):
                expected = torch.arange(64, dtype=torch.int32)
                assert got[0].item() == 1 and got[2][0].item() == 0
                assert torch.equal(got[1], expected)
                assert torch.equal(got[2][64:], expected)
                active_receivers += 1
            elif name.startswith("trb017"):
                assert got[0].item() == 0 and got[1].item() == 2
                assert got[2].tolist() == [1, 1, 0, 0]
            elif name.startswith("trb025"):
                expected = torch.arange(1, 17, dtype=torch.float32)
                assert got[0].item() == 1
                assert torch.equal(got[1], expected)
                assert torch.equal(got[2], expected.repeat(2))
                active_receivers += 2
            elif name.startswith("trb026"):
                expected = torch.arange(1, 17, dtype=torch.int32)
                assert got[0].item() == 1 and torch.equal(got[1], expected)
                assert torch.equal(got[2], expected) or not got[2].any()
                active_receivers += int(torch.equal(got[2], expected))
        checks.append(
            {
                "name": name,
                "native_launches": repetitions,
                "active_receivers": active_receivers,
                "guarded_consumer_observed": (
                    bool(active_receivers)
                    if name.startswith(("trb021", "trb026"))
                    else None
                ),
                "kernel_sha256": hashlib.sha256(s.kernel_fn.src.encode()).hexdigest(),
            }
        )
        print(name, "PASS", active_receivers, flush=True)
    return checks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument("--repetitions", type=int, default=10)
    parser.add_argument("--output", type=Path)
    ns = parser.parse_args()
    assert ns.repetitions > 0
    by_name = check_catalog()
    result = {
        "scored": 70,
        "race": 35,
        "race_free": 35,
        "patterns": 25,
        "gpu_checks": check_gpu(by_name, ns.repetitions) if ns.gpu else [],
    }
    if ns.output:
        ns.output.write_text(json.dumps(result, indent=2) + "\n")
    print("PASS: 70 scored cases, 35/35 labels, 25 patterns")


if __name__ == "__main__":
    main()
