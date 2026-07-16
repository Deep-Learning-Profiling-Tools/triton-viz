"""The 2-D concretization map (plan §I.2) from the results JSONLs.

Axis 1 (x): what is concretized — cumulative left to right:
nothing (T0) → scalar params (T1, IR front-end) → memory contents →
paths (interpreter front-end; the two arrive together — memory
concretization requires executing load semantics, which forces one
path). The (memory-without-paths) cell is UNREACHABLE by construction:
that asymmetry is §I.2's point, and the map shows it.

Axis 2 (y): what stays symbolic — params, pid, grid, loop trip at T0;
pid, grid (launch-contract floored), trip at T1; only the thread
interleaving (pid, alpha-renamed) in the interpreter's two-copy solve.

Every benchmark row's terminal state determines its point; abstentions
(unsupported / compile-error) have no point and land in the residual
table. The script stays out of the harness proper (plan §III):

Usage:  uv run python -m evaluation.concretization_map [results/*.jsonl]
Writes results/CONCRETIZATION_MAP.md, .csv and .svg.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

RESULTS_DIR = Path(__file__).parent / "results"

# x, y are ordinal cell indices into AXIS_X / AXIS_Y below.
AXIS_X = ("nothing", "scalar params", "memory contents", "+ paths")
AXIS_Y = (
    "pid (interleaving)",
    "pid + trip (grid = launch)",
    "pid + grid≥launch + trip",
    "params + pid + grid + trip",
)

# terminal state → (x, y, class). Conditional proofs share the proof
# point; the marker records the premise.
POINTS: dict[str, tuple[int, int, str]] = {
    "proved@T0": (0, 3, "proof"),
    "proved@T0+assumes-termination": (0, 3, "conditional proof"),
    "proved@T1": (1, 2, "proof"),
    "proved@T1+assumes-termination": (1, 2, "conditional proof"),
    # The §3c launch-scoped rung: params concretized AND the grid pinned
    # to the launch extent — one step more concrete than T1 on the y
    # axis, still on the IR front-end. Its grid-fragile attribute is
    # per-row metadata, not a separate point.
    "proved@T1-launch": (1, 1, "launch-scoped proof"),
    "proved@T1-launch+assumes-termination": (1, 1, "conditional proof"),
    # A static-track race verdict is decided on the IR front-end at T1.
    "races-unclassified": (1, 2, "report"),
    # Confirmation/refutation happen on the interpreter front-end, where
    # memory contents and paths are concretized together.
    "race-confirmed": (3, 0, "confirmed race"),
    "race-unconfirmed": (3, 0, "unconfirmed report"),
    # Composed-dispatcher decisions on static-abstained rows: the
    # interpreter front-end's own verdicts (per-launch scope, optionally
    # + contents-snapshot). A proof can now live on the interpreter
    # point too.
    "race@interp": (3, 0, "report"),
    "proved@interp": (3, 0, "proof"),
}
RESIDUAL = ("unsupported", "compile-error", "crash", "timeout")


def load_rows(paths: list[Path]) -> list[dict]:
    rows = []
    for p in paths:
        for line in p.read_text().splitlines():
            row = json.loads(line)
            if row.get("header"):
                continue
            row["_corpus"] = row.get("corpus", p.stem)
            rows.append(row)
    return rows


def build(rows: list[dict]) -> tuple[Counter, Counter, Counter]:
    """(per-(x,y,class) counts, per-cell-and-corpus counts, residual)."""
    points: Counter = Counter()
    by_corpus: Counter = Counter()
    residual: Counter = Counter()
    for row in rows:
        t = row.get("terminal")
        if t in POINTS:
            x, y, cls = POINTS[t]
            points[(x, y, cls)] += 1
            by_corpus[(x, y, cls, row["_corpus"])] += 1
        elif t in RESIDUAL:
            residual[(t, row["_corpus"])] += 1
        else:
            residual[(f"UNMAPPED:{t}", row["_corpus"])] += 1
    return points, by_corpus, residual


def to_markdown(points: Counter, by_corpus: Counter, residual: Counter) -> str:
    lines = [
        "# 2-D concretization map (plan §I.2)",
        "",
        "x: what is concretized (cumulative). y: what stays symbolic.",
        "The (memory-without-paths) column is unreachable by construction —",
        "concretizing memory means executing load semantics, which forces",
        "one path (§I.2); the interpreter front-end owns both at once.",
        "",
        "| x (concretized) | y (symbolic) | class | rows | corpora |",
        "|---|---|---|---|---|",
    ]
    for (x, y, cls), n in sorted(points.items()):
        corp = ", ".join(
            f"{c.rsplit('/', 1)[-1]}:{m}"
            for (px, py, pc, c), m in sorted(by_corpus.items())
            if (px, py, pc) == (x, y, cls)
        )
        lines.append(f"| {AXIS_X[x]} | {AXIS_Y[y]} | {cls} | {n} | {corp} |")
    lines += [
        "",
        "## Residual (no point on the map)",
        "",
        "| terminal | corpus | rows |",
        "|---|---|---|",
    ]
    for (t, c), n in sorted(residual.items()):
        lines.append(f"| {t} | {c} | {n} |")
    lines.append("")
    return "\n".join(lines)


def to_csv(points: Counter) -> str:
    out = ["x,y,x_label,y_label,class,count"]
    for (x, y, cls), n in sorted(points.items()):
        out.append(f'{x},{y},"{AXIS_X[x]}","{AXIS_Y[y]}","{cls}",{n}')
    out.append("")
    return "\n".join(out)


_CLASS_STYLE = {
    # (fill, stroke, shape) — shapes: circle / diamond / square
    "proof": ("#2e7d32", "#1b5e20", "circle"),
    "conditional proof": ("#9ccc65", "#558b2f", "circle"),
    "launch-scoped proof": ("#00838f", "#006064", "circle"),
    "report": ("#ef6c00", "#e65100", "diamond"),
    "confirmed race": ("#c62828", "#8e0000", "square"),
    "unconfirmed report": ("#757575", "#424242", "diamond"),
}


def to_svg(points: Counter) -> str:
    """Dependency-free scatter: cell grid, marker area ∝ row count,
    hatched band on the unreachable column."""
    cw, ch, mx, my = 190, 95, 250, 60  # cell size, margins
    width = mx + cw * len(AXIS_X) + 40
    height = my + ch * len(AXIS_Y) + 110
    e: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
        f'height="{height}" font-family="Helvetica,Arial,sans-serif">',
        '<rect width="100%" height="100%" fill="white"/>',
        "<defs><pattern id='hatch' width='8' height='8' "
        "patternTransform='rotate(45)' patternUnits='userSpaceOnUse'>"
        "<line x1='0' y1='0' x2='0' y2='8' stroke='#d0d0d0' stroke-width='2'/>"
        "</pattern></defs>",
    ]

    def cx(x: int) -> float:
        return mx + cw * (x + 0.5)

    def cy(y: int) -> float:
        return my + ch * (len(AXIS_Y) - 1 - y + 0.5)

    # grid + axis labels
    for i in range(len(AXIS_X) + 1):
        e.append(
            f'<line x1="{mx + cw * i}" y1="{my}" x2="{mx + cw * i}" '
            f'y2="{my + ch * len(AXIS_Y)}" stroke="#cccccc"/>'
        )
    for j in range(len(AXIS_Y) + 1):
        e.append(
            f'<line x1="{mx}" y1="{my + ch * j}" x2="{mx + cw * len(AXIS_X)}" '
            f'y2="{my + ch * j}" stroke="#cccccc"/>'
        )
    # unreachable column: memory-without-paths
    e.append(
        f'<rect x="{mx + cw * 2}" y="{my}" width="{cw}" '
        f'height="{ch * len(AXIS_Y)}" fill="url(#hatch)"/>'
    )
    e.append(
        f'<text x="{cx(2)}" y="{my + ch * len(AXIS_Y) / 2}" font-size="12" '
        f'fill="#888888" text-anchor="middle" '
        f'transform="rotate(-90 {cx(2)} {my + ch * len(AXIS_Y) / 2})">'
        "unreachable: memory ⇒ paths (§I.2)</text>"
    )
    for i, lab in enumerate(AXIS_X):
        e.append(
            f'<text x="{cx(i)}" y="{my + ch * len(AXIS_Y) + 24}" '
            f'font-size="13" text-anchor="middle">{lab}</text>'
        )
    for j, lab in enumerate(AXIS_Y):
        e.append(
            f'<text x="{mx - 12}" y="{cy(j) + 4}" font-size="13" '
            f'text-anchor="end">{lab}</text>'
        )
    e.append(
        f'<text x="{mx + cw * len(AXIS_X) / 2}" '
        f'y="{my + ch * len(AXIS_Y) + 52}" font-size="14" '
        f'text-anchor="middle" font-weight="bold">concretized</text>'
    )
    e.append(
        f'<text x="40" y="{my + ch * len(AXIS_Y) / 2}" font-size="14" '
        f'text-anchor="middle" font-weight="bold" '
        f'transform="rotate(-90 40 {my + ch * len(AXIS_Y) / 2})">'
        "stays symbolic</text>"
    )

    # markers — offset within the cell per class so they don't overlap
    offsets = {
        "proof": (-38, 0),
        "conditional proof": (14, 0),
        "launch-scoped proof": (-38, 0),
        "report": (52, 0),
        "confirmed race": (-20, 0),
        "unconfirmed report": (30, 0),
    }
    for (x, y, cls), n in sorted(points.items()):
        fill, stroke, shape = _CLASS_STYLE[cls]
        r = max(9.0, min(26.0, 5.5 * (n**0.5)))
        px = cx(min(x, len(AXIS_X) - 1)) + offsets[cls][0]
        py = cy(y) + offsets[cls][1]
        if shape == "circle":
            e.append(
                f'<circle cx="{px}" cy="{py}" r="{r}" fill="{fill}" '
                f'stroke="{stroke}" stroke-width="1.5"/>'
            )
        elif shape == "square":
            e.append(
                f'<rect x="{px - r}" y="{py - r}" width="{2 * r}" '
                f'height="{2 * r}" fill="{fill}" stroke="{stroke}" '
                f'stroke-width="1.5"/>'
            )
        else:  # diamond
            e.append(
                f'<polygon points="{px},{py - r} {px + r},{py} {px},{py + r} '
                f'{px - r},{py}" fill="{fill}" stroke="{stroke}" '
                f'stroke-width="1.5"/>'
            )
        e.append(
            f'<text x="{px}" y="{py + 4}" font-size="12" fill="white" '
            f'text-anchor="middle" font-weight="bold">{n}</text>'
        )

    # legend
    ly = height - 28
    lx = mx
    for cls, (fill, stroke, shape) in _CLASS_STYLE.items():
        if shape == "circle":
            e.append(
                f'<circle cx="{lx}" cy="{ly}" r="7" fill="{fill}" '
                f'stroke="{stroke}"/>'
            )
        elif shape == "square":
            e.append(
                f'<rect x="{lx - 7}" y="{ly - 7}" width="14" height="14" '
                f'fill="{fill}" stroke="{stroke}"/>'
            )
        else:
            e.append(
                f'<polygon points="{lx},{ly - 8} {lx + 8},{ly} {lx},{ly + 8} '
                f'{lx - 8},{ly}" fill="{fill}" stroke="{stroke}"/>'
            )
        e.append(f'<text x="{lx + 14}" y="{ly + 4}" font-size="12">{cls}</text>')
        lx += 30 + 8 * len(cls)
    e.append("</svg>")
    return "\n".join(e)


def main() -> None:
    args = [Path(a) for a in sys.argv[1:]]
    paths = args or sorted(RESULTS_DIR.glob("*.jsonl"))
    rows = load_rows(paths)
    points, by_corpus, residual = build(rows)
    RESULTS_DIR.mkdir(exist_ok=True)
    md = to_markdown(points, by_corpus, residual)
    (RESULTS_DIR / "CONCRETIZATION_MAP.md").write_text(md)
    (RESULTS_DIR / "CONCRETIZATION_MAP.csv").write_text(to_csv(points))
    (RESULTS_DIR / "CONCRETIZATION_MAP.svg").write_text(to_svg(points))
    print(md)
    print(f"[{len(rows)} rows from {len(paths)} corpora]")


if __name__ == "__main__":
    main()
