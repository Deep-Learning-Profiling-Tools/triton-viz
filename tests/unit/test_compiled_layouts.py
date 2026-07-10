"""Differential tests: layout closed forms vs. basis constructions/oracles.

The closed forms in layouts.py and the basis construction in
``swizzled_offset_bases`` are independent transcriptions of the triton C++
sources, so agreement between them (and, where available, with triton's own
LinearLayout) is strong evidence of correctness.
"""

import pytest

from triton_viz.clients.race_detector.compiled.layouts import (
    BlockedLayout,
    NVMMASharedLayout,
    SwizzledSharedLayout,
    nvmma_offset_bases,
    parse_shared_layout,
    swizzled_offset_bases,
    xor_linear_apply,
)

# The shared layouts observed in the golden dumps plus stress variants
# (including vec*maxPhase > numCols, which exercises the % numCols clip).
SWIZZLED_CASES = [
    # (vec, perPhase, maxPhase, order, shape)
    (8, 2, 4, (1, 0), (64, 32)),  # matmul A, sm80
    (8, 1, 8, (1, 0), (32, 64)),  # matmul B, sm80
    (1, 1, 1, (1, 0), (16, 16)),  # no swizzle
    (4, 1, 4, (1, 0), (8, 8)),  # vec*maxPhase = 16 > numCols = 8
    (2, 2, 2, (0, 1), (16, 32)),  # transposed order
]


@pytest.mark.parametrize("vec,per_phase,max_phase,order,shape", SWIZZLED_CASES)
def test_swizzled_closed_form_matches_basis_construction(
    vec, per_phase, max_phase, order, shape
):
    layout = SwizzledSharedLayout(vec, per_phase, max_phase, tuple(order))
    bases = swizzled_offset_bases(layout, shape)

    col_dim, row_dim = order[0], order[1]
    num_cols, num_rows = shape[col_dim], shape[row_dim]

    seen = set()
    for row in range(num_rows):
        for col in range(num_cols):
            coords = [0, 0]
            coords[col_dim], coords[row_dim] = col, row
            got = layout.element_offset(tuple(coords), shape)

            # Basis construction: input bits are col bits then row bits.
            x = col | (row << (num_cols.bit_length() - 1))
            expect = xor_linear_apply(bases, x, 1)[0]
            assert got == expect, (row, col, got, expect)
            seen.add(got)
    # The swizzle is a bijection over one stage buffer.
    assert seen == set(range(num_rows * num_cols))


# (swizzlingByteWidth, elementBitWidth, transposed, shape) — the two golden
# sm90 layouts plus stress variants: 32B swizzle, 8-bit elements, a shape
# wider than one swizzle tile (col-repetition bits), and transposed order.
NVMMA_CASES = [
    (64, 16, False, (64, 32)),  # matmul A, sm90 golden
    (128, 16, False, (32, 64)),  # matmul B, sm90 golden
    (32, 16, False, (16, 16)),
    (128, 8, False, (16, 128)),
    (128, 16, False, (16, 128)),  # numCols 128 > tileCols 64: col-rep bits
    (64, 16, True, (32, 64)),  # transposed: dim0 is the inner dim
    (0, 16, False, (16, 32)),  # no swizzle: plain row-major
]


@pytest.mark.parametrize("w,e,transposed,shape", NVMMA_CASES)
def test_nvmma_closed_form_matches_basis_construction(w, e, transposed, shape):
    """The closed form maps (row, col) → offset; the basis construction
    (transcribed from nvmmaSharedToLinearLayout) maps offset bits → (row,
    col). Composing them must be the identity, and the map must be a
    bijection over one stage buffer. The upstream LinearLayout binding
    still aborts on shared encodings (LinearEncodingAttr is
    distributed-only in the 3.7.1 wheel), so two independent
    transcriptions cross-check each other, as for swizzled_shared."""
    layout = NVMMASharedLayout(w, transposed, e)
    total = shape[0] * shape[1]

    seen = set()
    for r0 in range(shape[0]):
        for c1 in range(shape[1]):
            off = layout.element_offset((r0, c1), shape)
            assert 0 <= off < total, ((r0, c1), off)
            seen.add(off)
    assert seen == set(range(total)), "not a bijection over the stage buffer"

    if w == 0:
        return  # no bases construction for the unswizzled form
    bases = nvmma_offset_bases(layout, shape)
    assert len(bases) == total.bit_length() - 1
    col_dim = 0 if transposed else 1
    for off in range(total):
        row, col = xor_linear_apply(bases, off, 2)
        coords = [0, 0]
        coords[1 - col_dim], coords[col_dim] = row, col
        assert layout.element_offset(tuple(coords), shape) == off, (off, row, col)


def test_nvmma_parse_and_dispatch():
    attr = (
        "#ttg.nvmma_shared<{swizzlingByteWidth = 64, transposed = false, "
        "elementBitWidth = 16}>"
    )
    layout = parse_shared_layout(attr)
    assert isinstance(layout, NVMMASharedLayout)
    assert (layout.swizzle_byte_width, layout.elem_bits, layout.transposed) == (
        64,
        16,
        False,
    )
    swz = parse_shared_layout(
        "#ttg.swizzled_shared<{vec = 8, perPhase = 2, maxPhase = 4, order = [1, 0]}>"
    )
    assert isinstance(swz, SwizzledSharedLayout)


def test_nvmma_fp4_padded_is_unsupported():
    from triton_viz.clients.race_detector.compiled.ttgir_reader import (
        UnsupportedTTGIR,
    )

    with pytest.raises(UnsupportedTTGIR):
        NVMMASharedLayout.parse(
            "#ttg.nvmma_shared<{swizzlingByteWidth = 64, transposed = false, "
            "elementBitWidth = 16, fp4Padded = true}>"
        )


BLOCKED_CASES = [
    # (spt, tpw, wpc, order, shape) — golden-dump layouts + variants
    ((1, 8), (4, 8), (4, 1), (1, 0), (64, 32)),
    ((1, 8), (8, 4), (4, 1), (1, 0), (32, 64)),
    ((4,), (32,), (4,), (0,), (512,)),
    ((2, 2), (8, 4), (2, 2), (0, 1), (32, 32)),
]


@pytest.mark.parametrize("spt,tpw,wpc,order,shape", BLOCKED_CASES)
def test_blocked_owner_coords_partition_the_tensor(spt, tpw, wpc, order, shape):
    """Every tensor element is owned by at least one (thread, register), and
    coords stay in range — the layout covers the tensor (broadcast layouts
    may multiply-own elements; coverage is the safety-relevant property for
    write footprints)."""
    layout = BlockedLayout(spt, tpw, wpc, order)
    lanes = 1
    for t in tpw:
        lanes *= t
    warps = 1
    for w in wpc:
        warps *= w
    n_threads = lanes * warps
    n_regs = layout.regs_per_thread(shape)

    covered = set()
    for tid in range(n_threads):
        for reg in range(n_regs):
            coords = layout.owner_coords(tid, reg, shape)
            for d, c in enumerate(coords):
                assert 0 <= c < shape[d], (tid, reg, coords)
            covered.add(coords)
    total = 1
    for s in shape:
        total *= s
    assert len(covered) == total


def test_blocked_matches_gluon_linear_layout_oracle():
    """Cross-check the blocked closed form against triton's own
    LinearLayout via the gluon builder (distributed layouts only — shared
    layouts SIGABRT in the 3.6.0 wheel and must never be passed there)."""
    pytest.importorskip("triton")
    try:
        from triton._C.libtriton import gluon_ir as gi
        from triton._C import libtriton

        ctx = libtriton.ir.context()
        libtriton.ir.load_dialects(ctx)
        builder = gi.GluonOpBuilder(ctx)
    except Exception:
        pytest.skip("gluon builder unavailable")

    shape = [64, 32]
    layout = BlockedLayout((1, 8), (4, 8), (4, 1), (1, 0))
    try:
        attr = builder.get_blocked_layout(
            ctx,
            list(layout.size_per_thread),
            list(layout.threads_per_warp),
            list(layout.warps_per_cta),
            list(layout.order),
            [1, 1],
            [1, 1],
            [0, 1],
        )
        ll = builder.to_linear_layout(attr, shape)
    except TypeError:
        pytest.skip("gluon builder signature differs on this triton version")
    except Exception:
        pytest.skip("gluon to_linear_layout unavailable")

    reg_bases = ll.reg_bases
    lane_bases = ll.lane_bases
    warp_bases = ll.warp_bases

    def oracle_coords(tid: int, reg: int) -> tuple[int, ...]:
        lane, warp = tid % 32, tid // 32
        out = [0, 0]
        for bits, bases in ((reg, reg_bases), (lane, lane_bases), (warp, warp_bases)):
            i = 0
            while bits:
                if bits & 1:
                    out[0] ^= bases[i][0]
                    out[1] ^= bases[i][1]
                bits >>= 1
                i += 1
        return tuple(out)

    n_regs = layout.regs_per_thread(tuple(shape))
    for tid in range(128):
        for reg in range(n_regs):
            assert layout.owner_coords(tid, reg, tuple(shape)) == oracle_coords(
                tid, reg
            ), (tid, reg)
