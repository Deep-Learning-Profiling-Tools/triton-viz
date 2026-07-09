"""LaunchSpec: one evaluation row = one kernel under one concrete launch.

Ground-truth labels attach to the (kernel, launch-params) pair — NOT the
kernel (plan S5, departure 4): a kernel listed with several differently
labeled launches derives the kernel-level "∃ racy input" truth that audits
the claim ladder (a premise-compatible proved@T0 against a yes-launch is
ladder-unsound).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Literal


@dataclass(frozen=True)
class LaunchSpec:
    # identity
    name: str  # unique within the corpus; DRB-style suffix _yes/_no
    kernel_fn: Any  # the @triton.jit function (NOT autotuner-wrapped)
    # compilation (host-only): triton signature dict incl. constexpr entries
    signature: dict[str, str]
    constexprs: dict[str, int]
    # launch
    make_args: Callable[[int], tuple]  # seed -> positional args (CPU tensors + scalars)
    grid: tuple[int, ...]
    # ground truth, scoped to THIS launch
    expected: Literal["race", "race-free"] | None = None
    # the planted racing access pair as source-line NEEDLES (substrings of
    # the kernel's source lines, resolved to line numbers at scoring time —
    # robust against edits shifting absolute numbers), when expected=="race"
    race_pair: tuple[str, ...] | None = None
    # race-pattern taxonomy bucket (DRB-style)
    pattern: str = ""
    # free-form note (e.g. which parameter makes this launch racy)
    params_note: str = ""
    # True when make_args aliases pointer arguments (e.g. in-place). Such a
    # yes-launch violates the T0 non-aliasing premise, so the ladder audit
    # must NOT count it against a proved@T0 of the same specialization.
    aliased: bool = False

    def spec_id(self) -> str:
        return self.name


@dataclass
class Corpus:
    name: str
    specs: list[LaunchSpec] = field(default_factory=list)

    def add(self, spec: LaunchSpec) -> LaunchSpec:
        assert spec.name not in {s.name for s in self.specs}, spec.name
        self.specs.append(spec)
        return spec
