"""Unit tests for the A2 gate (atomic-ordering barrier coverage over PTX).

Golden inputs in tests/golden/a2gate/ were generated under the corpus
pin triton 3.6.0, which predates triton PR #10816: their PTX carries no
ordering barriers around non-relaxed atomics, so the gate must report
violations on them (the A2-class defect, live in the pin). The
simulated post-fix variants are built here by inserting ``bar.sync``
lines per the #10816 rule, and the mutation matrix deletes them again
one side at a time.
"""

import os
import re

import pytest

from triton_viz.clients.race_detector.compiled.ptx_gate import (
    GateUnsupported,
    check_gate,
    parse_ptx,
    ttir_obligations,
)

GOLDEN = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "golden", "a2gate"
)


def _read(name: str) -> str:
    with open(os.path.join(GOLDEN, name)) as f:
        return f.read()


def _insert_barriers(ptx: str, *, before: list[str], after: list[str]) -> str:
    """Simulate the #10816 fix: bar.sync adjacent to the named atomics.

    ``before``/``after`` are substrings identifying atomic instruction
    lines (e.g. ``".release.add"``); every matching line gets a
    ``bar.sync 0;`` inserted on that side.
    """
    out = []
    for line in ptx.splitlines():
        pre = [s for s in before if s in line]
        post = [s for s in after if s in line]
        if pre:
            out.append("\tbar.sync \t0;")
        out.append(line)
        if post:
            out.append("\tbar.sync \t0;")
    return "\n".join(out) + "\n"


# ── obligations ───────────────────────────────────────────────────────


def test_obligations_extraction_sems():
    obs = ttir_obligations(_read("a2_sems.ttir"))
    assert [(o.kind, o.sem, o.scope) for o in obs] == [
        ("rmw", "relaxed", "gpu"),
        ("rmw", "acquire", "gpu"),
        ("rmw", "release", "gpu"),
        ("rmw", "acq_rel", "gpu"),
    ]
    # Locations resolve through the loc table (consecutive source lines).
    lines = [o.loc[1] for o in obs]
    assert lines == sorted(lines) and len(set(lines)) == 4


def test_obligations_extraction_cas_named_loc_alias():
    # The CAS result is named ("old"), so its loc is a named alias
    # (#locN = loc("old"(#locM))) that must resolve transitively.
    obs = ttir_obligations(_read("a2_cas.ttir"))
    assert len(obs) == 1
    assert obs[0].kind == "cas" and obs[0].sem == "acq_rel"
    assert obs[0].loc is not None


def test_atomic_poll_refuses():
    with pytest.raises(GateUnsupported) as ei:
        ttir_obligations("%0 = tt.atomic_poll acquire, gpu, %p loc(#loc1)")
    assert ei.value.kind == "atomic-poll"


# ── the pre-fix goldens: violations ───────────────────────────────────


def test_prefix_sems_violations():
    r = check_gate(_read("a2_sems.ttir"), _read("a2_sems.ptx"), "a2_sems")
    assert r.status == "violation"
    sides = sorted(
        (rep.split(": ")[1].split(" at")[0], "before" in rep) for rep in r.reports
    )
    # acquire: after missing; release: before missing; acq_rel: both.
    assert sides == [
        ("acq_rel rmw", False),
        ("acq_rel rmw", True),
        ("acquire rmw", False),
        ("release rmw", True),
    ]
    assert r.obligations == 4


def test_prefix_cas_staging_discharges_after_side():
    # The CAS result is used, so the pre-fix PTX already carries the
    # result-staging st.shared / bar.sync / ld.shared sequence: the
    # acquire half of acq_rel is discharged, only the release half
    # (barrier BEFORE) is missing.
    r = check_gate(_read("a2_cas.ttir"), _read("a2_cas.ptx"), "a2_cas")
    assert r.status == "violation"
    assert len(r.reports) == 1
    assert "before" in r.reports[0]


# ── the simulated post-fix: verified, then the mutation matrix ────────


def _fixed_sems_ptx() -> str:
    return _insert_barriers(
        _read("a2_sems.ptx"),
        before=[".release.add", ".acq_rel.add"],
        after=[".acquire.add", ".acq_rel.add"],
    )


def test_simulated_postfix_verified():
    r = check_gate(_read("a2_sems.ttir"), _fixed_sems_ptx(), "a2_sems")
    assert r.status == "verified", r.reports
    assert r.obligations == 4


def test_simulated_postfix_cas_verified():
    fixed = _insert_barriers(_read("a2_cas.ptx"), before=[".cas.b32"], after=[])
    r = check_gate(_read("a2_cas.ttir"), fixed, "a2_cas")
    assert r.status == "verified", r.reports


def test_adjacent_atomics_can_share_barriers():
    # In a2_sems the four atomics are back-to-back, so a neighbor's
    # post-barrier legitimately covers the next atomic's pre-side (a
    # bar.sync is a bar.sync). Dropping ONLY the release atomic's own
    # pre-barrier therefore still verifies: the acquire's post-barrier
    # sits immediately before it. This is by design, and it is why the
    # per-side mutation matrix below uses single-atomic snippets.
    mutated = _insert_barriers(
        _read("a2_sems.ptx"),
        before=[".acq_rel.add"],
        after=[".acquire.add", ".acq_rel.add"],
    )
    r = check_gate(_read("a2_sems.ttir"), mutated, "a2_sems")
    assert r.status == "verified", r.reports


_SYNTH_TTIR = """\
#loc = loc("k.py":1:0)
tt.func public @k(%p: !tt.ptr<i32>) {{
  %c = arith.constant 1 : i32 loc(#loc1)
  %0 = tt.atomic_rmw add, {sem}, gpu, %p, %c, %true : (!tt.ptr<i32>, i32, i1) -> i32 loc(#loc1)
  tt.return loc(#loc2)
}}
#loc1 = loc("k.py":5:0)
#loc2 = loc("k.py":6:0)
"""

_SYNTH_PTX = """\
.visible .entry k(
.param .u64 k_param_0
)
{{
$L__func_begin0:
\t.loc\t1 5 0
{pre}\t@%p1 atom.global.gpu.{sem}.add.u32 %r1, [ %rd1 + 0 ], %r2;
{post}\tret;
$L__func_end0:
}}
\t.file\t1 "k.py"
"""

_BAR = "\tbar.sync \t0;\n"


@pytest.mark.parametrize(
    "sem, pre, post, status, sides",
    [
        ("relaxed", "", "", "verified", []),
        ("release", _BAR, "", "verified", []),
        ("release", "", "", "violation", ["before"]),
        ("acquire", "", _BAR, "verified", []),
        ("acquire", "", "", "violation", ["after"]),
        ("acq_rel", _BAR, _BAR, "verified", []),
        ("acq_rel", "", _BAR, "violation", ["before"]),
        ("acq_rel", _BAR, "", "violation", ["after"]),
        ("acq_rel", "", "", "violation", ["before", "after"]),
    ],
)
def test_mutation_matrix_single_atomic(sem, pre, post, status, sides):
    ttir = _SYNTH_TTIR.format(sem=sem)
    ptx = _SYNTH_PTX.format(sem=sem, pre=pre, post=post)
    r = check_gate(ttir, ptx, "k")
    assert r.status == status, r.reports
    assert sorted(
        "before" if "before" in rep else "after" for rep in r.reports
    ) == sorted(sides)


# ── refusals and edge cases ───────────────────────────────────────────


def test_cluster_barrier_refuses():
    ptx = _read("a2_sems.ptx").replace(
        "$L__func_end0:", "\tbarrier.cluster.arrive;\n$L__func_end0:"
    )
    r = check_gate(_read("a2_sems.ttir"), ptx, "a2_sems")
    assert r.status == "unsupported"
    assert r.reason.startswith("cluster-barrier")


def test_missing_line_info_refuses():
    ptx = re.sub(r"^\s*\.loc.*$", "", _read("a2_sems.ptx"), flags=re.M)
    r = check_gate(_read("a2_sems.ttir"), ptx, "a2_sems")
    assert r.status == "unsupported"
    assert r.reason.startswith("obligation-unmatched")


def test_ttir_without_loc_refuses():
    ttir = re.sub(r" loc\(#loc\d*\)", "", _read("a2_sems.ttir"))
    r = check_gate(ttir, _read("a2_sems.ptx"), "a2_sems")
    assert r.status == "unsupported"
    assert r.reason.startswith("no-loc")


def test_no_atomics_is_vacuously_verified():
    ttir = "tt.func public @k() {\n  tt.return loc(#loc1)\n}\n"
    r = check_gate(ttir, _read("a2_sems.ptx"), "k")
    assert r.status == "verified"
    assert r.obligations == 0
    assert "no atomic ordering obligations" in r.reason


def test_predicated_barrier_not_credited():
    fixed = _fixed_sems_ptx().replace("\tbar.sync \t0;", "\t@%p1 bar.sync \t0;", 1)
    r = check_gate(_read("a2_sems.ttir"), fixed, "a2_sems")
    assert r.status == "violation"


_PEEPHOLE_PTX = """\
.visible .entry k(
.param .u64 k_param_0
)
{{
$L__func_begin0:
\t.loc\t1 5 0
\tbar.sync \t0;
\tmov.u32 %r6, 0x0;
\t@%p2 ld.global.gpu.acquire.b32 %r6, [ %rd3 + 0 ];
\t@%p2 st.shared.b32 [ %r7 + 0 ], %r6;
{post}\tld.shared.b32 \t%r8, [global_smem];
\tret;
$L__func_end0:
}}
\t.file\t1 "k.py"
"""


def test_zero_rmw_acquire_peephole_with_rendezvous_verified():
    # tl.atomic_add(p, 0, sem="acquire") lowers to a sem-qualified plain
    # load plus a broadcast staging sequence with rendezvous barriers
    # (no atom instruction at all); the ldst site vocabulary matches it
    # against the rmw obligation and the staging bar.sync covers the
    # after side.
    ttir = _SYNTH_TTIR.format(sem="acquire")
    ptx = _PEEPHOLE_PTX.format(post="\tbar.sync \t0;\n")
    r = check_gate(ttir, ptx, "k")
    assert r.status == "verified", r.reports


def test_zero_rmw_acquire_peephole_without_post_barrier_violates():
    ttir = _SYNTH_TTIR.format(sem="acquire")
    ptx = _PEEPHOLE_PTX.format(post="")
    r = check_gate(ttir, ptx, "k")
    assert r.status == "violation"
    assert len(r.reports) == 1 and "after" in r.reports[0]


def test_parse_ptx_finds_sites_and_qualifier_orders():
    # RMW prints scope-before-sem; CAS prints sem-before-scope.
    sems = parse_ptx(_read("a2_sems.ptx"))
    assert [(s.kind, s.sem, s.scope) for s in sems.sites] == [
        ("rmw", "relaxed", "gpu"),
        ("rmw", "acquire", "gpu"),
        ("rmw", "release", "gpu"),
        ("rmw", "acq_rel", "gpu"),
    ]
    cas = parse_ptx(_read("a2_cas.ptx"))
    assert [(s.kind, s.sem, s.scope) for s in cas.sites] == [("cas", "acq_rel", "gpu")]
