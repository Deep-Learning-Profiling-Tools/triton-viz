"""Terminal summary of the reader conformance suite (D10b): the counts a
TESTED_TRITON_VERSIONS decision reads, aggregated from the per-case
``conformance`` user property of test_reader_conformance.py."""

from __future__ import annotations

from collections import Counter


def pytest_terminal_summary(terminalreporter) -> None:
    cases: list[tuple[str, dict]] = []
    for reports in terminalreporter.stats.values():
        for report in reports:
            if getattr(report, "when", None) != "call":
                continue
            for key, props in getattr(report, "user_properties", ()):
                if key == "conformance":
                    cases.append((report.outcome, props))
    if not cases:
        return
    import triton

    from tilelens.core.config import untested_triton_version

    outcomes = Counter(p.get("outcome", "error") for _, p in cases)
    refused = Counter(
        o.split(":", 1)[1] for o in outcomes.elements() if o.startswith("refused:")
    )
    accepted = [p for _, p in cases if not p.get("outcome", "").startswith("refused:")]
    excluded: Counter = Counter()
    for p in accepted:
        excluded.update(p.get("excluded", {}))
    seconds = max((p["seconds"] for _, p in cases), key=lambda s: sum(s.values()))
    w = terminalreporter.write_line
    terminalreporter.section("TTIR reader conformance (D10b)")
    untested = (
        " (outside TESTED_TRITON_VERSIONS: failures expected, xfail)"
        if untested_triton_version()
        else ""
    )
    w(
        f"triton {triton.__version__}{untested}; "
        f"TTIR source: {', '.join(sorted({str(p.get('source')) for _, p in cases}))}"
    )
    compared = sum(1 for p in accepted if p.get("exercised_sites"))
    w(
        f"cases {len(cases)}: compared {compared} (conforming {outcomes['conform']}, mismatching "
        f"{outcomes['mismatch']}), accepted but not compared {outcomes['not-compared']}, "
        f"errors {outcomes['error']}; failed tests {sum(1 for o, _ in cases if o != 'passed')}"
    )
    w(f"refused {sum(refused.values())}: {dict(sorted(refused.items()))}")
    w(
        f"sites compared {sum(p.get('compared_sites', 0) for p in accepted)} "
        f"({sum(p.get('points', 0) for p in accepted)} (program, offset) points); "
        f"skipped accesses {sum(p.get('skipped_accesses', 0) for p in accepted)}; "
        f"excluded sites {dict(sorted(excluded.items()))}"
    )
    w(
        f"obligation-violating launches {sum(1 for p in accepted if p.get('obligation_failures'))}"
    )
    w(
        "runtime: "
        + ", ".join(f"{k} {v:.1f}s" for k, v in seconds.items())
        + f" (total {sum(seconds.values()):.1f}s)"
    )
    compared_jit = [
        (report.outcome, props)
        for reports in terminalreporter.stats.values()
        for report in reports
        if getattr(report, "when", None) == "call"
        for key, props in getattr(report, "user_properties", ())
        if key == "host_vs_jit"
    ]
    if compared_jit:
        w(
            f"host vs JIT compile (cuda:89, stand-in driver): {len(compared_jit)} "
            f"compared, {sum(1 for _, p in compared_jit if p['same_text'])} the same "
            "kernel (hash and TTIR text)"
        )
