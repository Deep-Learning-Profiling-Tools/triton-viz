"""Layout → address closed forms for the compiled-mode race detector.

Triton maps tensor elements to threads and shared-memory offsets via
XOR-linear maps over GF(2) (LinearLayout). The closed forms below were
transcribed from the triton 3.6.x C++ sources and verified exhaustively
against the ``LinearLayout`` ground truth during the design recon (see
``race_detector_static_hybrid_plan.md`` Part II §4):

  * blocked (distributed): which tensor element does (warp, lane, register)
    own — affine decomposition along ``order`` with repetition bits when the
    tensor is larger than one layout tile;
  * swizzled shared: element coords → shared-memory element offset with the
    XOR phase swizzle.

Python integer versions are used both for witness enrichment in reports and
as the implementation under test; ``triton.tools.LinearLayout`` serves as
the differential oracle in unit tests (never call GluonOpBuilder's
``to_linear_layout`` on shared encodings — it SIGABRTs in the 3.6.0 wheel).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .ttgir_reader import UnsupportedTTGIR

_RE_BLOCKED = re.compile(
    r"#ttg\.blocked<\{sizePerThread = \[([\d, ]+)\], "
    r"threadsPerWarp = \[([\d, ]+)\], warpsPerCTA = \[([\d, ]+)\], "
    r"order = \[([\d, ]+)\]\}>"
)
_RE_SWIZZLED = re.compile(
    r"#ttg\.swizzled_shared<\{vec = (\d+), perPhase = (\d+), "
    r"maxPhase = (\d+), order = \[([\d, ]+)\]\}>"
)
_RE_NVMMA = re.compile(
    r"#ttg\.nvmma_shared<\{swizzlingByteWidth = (\d+), "
    r"transposed = (false|true), elementBitWidth = (\d+)"
    r"(?:, fp4Padded = (false|true))?\}>"
)


def _ints(s: str) -> tuple[int, ...]:
    return tuple(int(x) for x in s.replace(" ", "").split(",") if x)


@dataclass(frozen=True)
class BlockedLayout:
    size_per_thread: tuple[int, ...]
    threads_per_warp: tuple[int, ...]
    warps_per_cta: tuple[int, ...]
    order: tuple[int, ...]

    @staticmethod
    def parse(attr: str) -> "BlockedLayout":
        m = _RE_BLOCKED.search(attr)
        if not m:
            raise UnsupportedTTGIR(f"unparsable blocked layout: {attr!r}")
        return BlockedLayout(
            _ints(m.group(1)), _ints(m.group(2)), _ints(m.group(3)), _ints(m.group(4))
        )

    def owner_coords(
        self, tid: int, reg: int, shape: tuple[int, ...]
    ) -> tuple[int, ...]:
        """Tensor coords owned by thread ``tid``'s register ``reg``.

        Verified equivalent to the C++ LinearLayout construction: decompose
        lane/warp/register ids along dims with ``order[0]`` fastest;
        ``coord[d] = rep[d]*tile[d] + warp[d]*tpw[d]*spt[d] + lane[d]*spt[d]
        + regInTile[d]`` where ``tile[d] = spt*tpw*wpc``. Register bits
        beyond one tile enumerate tile repetitions (order-fastest). Sizes
        must be powers of two (asserted by TTGIR construction).
        """
        rank = len(shape)
        spt, tpw, wpc = self.size_per_thread, self.threads_per_warp, self.warps_per_cta
        lane_total = 1
        for t in tpw:
            lane_total *= t
        lane = tid % lane_total
        warp = tid // lane_total

        lane_idx = [0] * rank
        warp_idx = [0] * rank
        reg_in_tile = [0] * rank
        rep_idx = [0] * rank

        lane_rest, warp_rest = lane, warp
        for d in self.order:
            lane_idx[d] = lane_rest % tpw[d]
            lane_rest //= tpw[d]
            warp_idx[d] = warp_rest % wpc[d]
            warp_rest //= wpc[d]

        r = reg
        for d in self.order:
            reg_in_tile[d] = r % spt[d]
            r //= spt[d]
        # Remaining register bits enumerate tile repetitions, order-fastest.
        for d in self.order:
            tile_d = spt[d] * tpw[d] * wpc[d]
            reps_d = max(1, shape[d] // tile_d)
            rep_idx[d] = r % reps_d
            r //= reps_d

        # When one layout tile exceeds the tensor (tile[d] > shape[d]) the
        # C++ construction emits ZERO bases for the surplus bits — a
        # broadcast where several threads own the same element. With pow2
        # sizes the contributions occupy disjoint bit fields, so dropping
        # the surplus bits is exactly a modulo by shape[d].
        return tuple(
            (
                rep_idx[d] * spt[d] * tpw[d] * wpc[d]
                + warp_idx[d] * tpw[d] * spt[d]
                + lane_idx[d] * spt[d]
                + reg_in_tile[d]
            )
            % shape[d]
            for d in range(rank)
        )

    def regs_per_thread(self, shape: tuple[int, ...]) -> int:
        total = 1
        for d, s in enumerate(shape):
            tile = (
                self.size_per_thread[d]
                * self.threads_per_warp[d]
                * self.warps_per_cta[d]
            )
            total *= self.size_per_thread[d] * max(1, s // tile)
        return total


@dataclass(frozen=True)
class SwizzledSharedLayout:
    vec: int
    per_phase: int
    max_phase: int
    order: tuple[int, ...]

    @staticmethod
    def parse(attr: str) -> "SwizzledSharedLayout":
        m = _RE_SWIZZLED.search(attr)
        if not m:
            raise UnsupportedTTGIR(f"unparsable swizzled_shared layout: {attr!r}")
        return SwizzledSharedLayout(
            int(m.group(1)), int(m.group(2)), int(m.group(3)), _ints(m.group(4))
        )

    def element_offset(self, coords: tuple[int, ...], shape: tuple[int, ...]) -> int:
        """Element offset inside one stage buffer for tensor ``coords``.

        Closed form (verified against swizzledSharedToLinearLayout's basis
        construction, including the ``% numCols`` clip when
        vec*maxPhase > numCols):

            phase   = (row / perPhase) % maxPhase
            off     = row*numCols
                      + (((col/vec) XOR phase) * vec) % numCols
                      + col % vec
        """
        if len(shape) == 1:
            return coords[0]
        if len(shape) != 2:
            raise UnsupportedTTGIR(f"rank-{len(shape)} shared layouts unsupported")
        col_dim = self.order[0]
        row_dim = self.order[1]
        col, row = coords[col_dim], coords[row_dim]
        num_cols = shape[col_dim]
        phase = (row // self.per_phase) % self.max_phase
        swizzled = (((col // self.vec) ^ phase) * self.vec) % num_cols
        return row * num_cols + swizzled + col % self.vec


@dataclass(frozen=True)
class NVMMASharedLayout:
    """``#ttg.nvmma_shared`` (sm90 wgmma operands) → element offset.

    Closed form from the design recon (plan Part II §4), the same XOR
    scheme as swizzled_shared applied to an 8-row × (8·W/E)-element core
    tile with ``vec = 128/E``, ``perPhase = 128/W``, ``maxPhase = W/16``
    (W = swizzlingByteWidth bytes, E = elementBitWidth bits; W=0 means no
    swizzle, plain row-major). ``vec·maxPhase`` equals the tile width, so
    the ``% numCols`` clip of the legacy form never engages. Tiles repeat
    along the inner dimension first, then the outer. The upstream
    LinearLayout binding still aborts on shared encodings (3.7.1 wraps the
    result in a distributed-only LinearEncodingAttr), so the differential
    test cross-checks this closed form against the independent basis
    construction in :func:`nvmma_offset_bases` instead.
    """

    swizzle_byte_width: int
    transposed: bool
    elem_bits: int

    @staticmethod
    def parse(attr: str) -> "NVMMASharedLayout":
        m = _RE_NVMMA.search(attr)
        if not m:
            raise UnsupportedTTGIR(f"unparsable nvmma_shared layout: {attr!r}")
        if m.group(4) == "true":
            raise UnsupportedTTGIR("fp4Padded nvmma_shared layouts unsupported")
        return NVMMASharedLayout(
            swizzle_byte_width=int(m.group(1)),
            transposed=m.group(2) == "true",
            elem_bits=int(m.group(3)),
        )

    def _geometry(self, shape: tuple[int, ...]) -> tuple[int, int, int, int, int, int]:
        """(col_dim, row_dim, tile_cols, vec, per_phase, max_phase)."""
        if len(shape) != 2:
            raise UnsupportedTTGIR(
                f"rank-{len(shape)} nvmma_shared layouts unsupported"
            )
        col_dim = 0 if self.transposed else 1
        row_dim = 1 - col_dim
        w = self.swizzle_byte_width
        tile_cols = 8 * w // self.elem_bits  # W bytes per row, in elements
        vec = 128 // self.elem_bits
        per_phase = 128 // w
        max_phase = w // 16
        return col_dim, row_dim, tile_cols, vec, per_phase, max_phase

    def element_offset(self, coords: tuple[int, ...], shape: tuple[int, ...]) -> int:
        if self.swizzle_byte_width == 0:
            # No swizzle: row-major with the inner dim given by `transposed`.
            if len(shape) != 2:
                raise UnsupportedTTGIR(
                    f"rank-{len(shape)} nvmma_shared layouts unsupported"
                )
            col_dim = 0 if self.transposed else 1
            row_dim = 1 - col_dim
            return coords[row_dim] * shape[col_dim] + coords[col_dim]

        col_dim, row_dim, tile_cols, vec, per_phase, max_phase = self._geometry(shape)
        num_cols, num_rows = shape[col_dim], shape[row_dim]
        if num_cols % tile_cols or num_rows % 8:
            raise UnsupportedTTGIR(
                f"nvmma_shared shape {shape} does not tile the "
                f"8x{tile_cols} swizzle tile"
            )
        col, row = coords[col_dim], coords[row_dim]
        r, rr = row % 8, row // 8
        c, cr = col % tile_cols, col // tile_cols
        phase = (r // per_phase) % max_phase
        in_tile = r * tile_cols + ((c // vec) ^ phase) * vec + c % vec
        return 8 * num_cols * rr + 8 * tile_cols * cr + in_tile


SharedLayout = SwizzledSharedLayout | NVMMASharedLayout


def parse_shared_layout(attr: str) -> SharedLayout:
    if "nvmma_shared" in attr:
        return NVMMASharedLayout.parse(attr)
    return SwizzledSharedLayout.parse(attr)


def xor_linear_apply(bases: list[list[int]], x: int, out_rank: int) -> tuple[int, ...]:
    """Generic XOR-linear map: out = XOR of bases[i] for each set bit of x.

    Matches ``LinearLayout.apply`` for a single input dimension; used by the
    unit tests as a cross-check between closed forms and basis
    constructions.
    """
    out = [0] * out_rank
    i = 0
    while x:
        if x & 1:
            for d in range(out_rank):
                out[d] ^= bases[i][d]
        x >>= 1
        i += 1
    return tuple(out)


def nvmma_offset_bases(
    layout: NVMMASharedLayout, shape: tuple[int, ...]
) -> list[list[int]]:
    """Bases of the offset → (row, col) map exactly as
    ``nvmmaSharedToLinearLayout`` builds them (the INVERSE direction of the
    closed form, which makes the two transcriptions independent):

    tile col bits c = 1,2,..<tileCols: basis (0, c);
    tile row bits r = 1,2,4: basis (r, vec·((r/perPhase) % maxPhase));
    then col-repetition bits (0, tileCols·2^k), then row-repetition bits
    (8·2^m, 0). Two-output bases as (row, col) pairs, offset bits
    minor-first.
    """
    col_dim, row_dim, tile_cols, vec, per_phase, max_phase = layout._geometry(shape)
    num_cols, num_rows = shape[col_dim], shape[row_dim]
    bases: list[list[int]] = []
    c = 1
    while c < tile_cols:
        bases.append([0, c])
        c *= 2
    r = 1
    while r < 8:
        bases.append([r, vec * ((r // per_phase) % max_phase)])
        r *= 2
    cr = tile_cols
    while cr < num_cols:
        bases.append([0, cr])
        cr *= 2
    rr = 8
    while rr < num_rows:
        bases.append([rr, 0])
        rr *= 2
    return bases


def swizzled_offset_bases(
    layout: SwizzledSharedLayout, shape: tuple[int, ...]
) -> list[list[int]]:
    """Offset bases exactly as swizzledSharedToLinearLayout builds them.

    For col bits c = 1,2,4,..<numCols: basis {0, c} (offset contribution c).
    For row bits r = 1,2,4,..<numRows: basis
    {r*numCols + (vec*((r/perPhase) % maxPhase)) % numCols}.
    Returned as single-output-dim bases (element offset), input bits ordered
    col bits first then row bits.
    """
    col_dim, row_dim = layout.order[0], layout.order[1]
    num_cols, num_rows = shape[col_dim], shape[row_dim]
    bases: list[list[int]] = []
    c = 1
    while c < num_cols:
        bases.append([c])
        c *= 2
    r = 1
    while r < num_rows:
        swz = (layout.vec * ((r // layout.per_phase) % layout.max_phase)) % num_cols
        bases.append([r * num_cols + swz])
        r *= 2
    return bases
