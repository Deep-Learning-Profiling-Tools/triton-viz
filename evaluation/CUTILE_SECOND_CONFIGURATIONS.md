# Seven second configurations for the cuTile real-operator corpus

Date: 2026-09-10. Takes `tilebench_cutile` from 61 to 68 measured
configurations.

## Why

The real-operator cuTile experiment measured 61 configurations. Each was
TileBench operator at case 0 of its own benchmark case grid, because the
capture ran `run_benchmark_suite(op, case_indices=[0])`. Every operator's
grid holds twenty or more cases, so a second configuration of an operator
is an ordinary, already-authored input shape, not a new kernel and not a
fabricated one.

## Which seven, and why those

The selection is principled rather than convenient: take every
SINGLE-KERNEL operator whose case 1 changes a SHAPE or a structural
parameter rather than only the element dtype. Single-kernel keeps the
arithmetic honest, one operator contributing exactly one configuration; a
shape change makes the configuration genuinely new rather than a
re-typing of the same launch.

Eight operators qualify. `quantize_global` is left out because it is the
same change (an elementwise kernel at twice the element count, fp32) on
the same shape as `fused_activation`, so it would add a duplicate rather
than a new configuration.

| configuration | case 0 | case 1 | new verdict |
|---|---|---|---|
| `flash_attention_case1` | seq_len 1024 | seq_len 2048 | proved@T1 |
| `flash_decode_case1` | seq_len 2048 | seq_len 4096 | proved@T1 |
| `block_sparse_attention_case1` | M 512 | M 1024 | proved@T1+content |
| `dequantize_rowwise_case1` | cols 512 | cols 1024 | proved@T1 |
| `kl_divergence_case1` | cols 1024 | cols 2048 | proved@T1 |
| `matmul_int8_case1` | K 1024 | K 2048 | proved@T1 |
| `fused_activation_case1` | n 1048576 | n 2097152 | proved@T0 |

Four of the seven compile to DIFFERENT CuTile IR text (`flash_decode`,
`block_sparse_attention`, `dequantize_rowwise`, `matmul_int8`). The other
three compile to the same text but launch differently: `flash_attention`
and `fused_activation` change the grid, and `kl_divergence` changes the
array extents the kernel's flattened shape parameters carry. All seven
therefore encode differently, which is what a configuration is.

## How they were captured and merged

`evaluation/tilebench_cutile_capture` gained a `--case-index` flag. A
non-zero index stores the record under the case name `<op>_case<N>` and
records `case_index` and `case_params`, so the configuration can be read
off the corpus without the TileBench checkout. All seven were captured on
the pinned checkout (224ec81f) with cuda.tile 1.5.0 on the RTX 4090, zero
errors, one kernel record each.

The merge refuses on anything but an addition: every pre-existing case
entry must stay byte-identical, no new record may duplicate an existing
one under the capture's own fingerprint, and the total must land on 68.
It reported 7 rows added, 68 total, zero problems, and a byte-identical
check over the 61 pre-existing entries.

The corpus module builds these as `ctb_<op>_case1__<kernel>`. They have
no same-operator Triton twin (the `tilebench` corpus is at case 0), so
they sit outside the cross-DSL differential; the module docstring says so.

## Measured

The corpus was rerun at L2: **68 rows, 64 proofs, 4 abstentions**. All
seven new configurations prove, and none of the 61 pre-existing rows
changed verdict or proof rung.

The four abstentions are the same four the abstention-closure work left
open: `histogram_partial` (a 262144-element index source against the
16384-element address-snapshot bound), both `radix_sort` rows (an address
through `tile_scan`), and `streamk_matmul first_wave_kernel` (a genuinely
data-dependent walk).

The frozen pinned roster goes from 1256 to 1263 rows
(`evaluation/pinned_manifest.py`).

## Adoption note

The pinned evaluation numbers are unchanged by this commit. The paper's
real-operator group may be restated as 68 configurations with 64 proofs
and 4 abstentions only after a pinned run adopts it; until then it stands
at 61 with 47 proofs and 14 abstentions.
