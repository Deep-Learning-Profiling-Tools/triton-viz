# Fence-order probes (2026-09-04)

Evidence behind the paper's option-A decision (paper repo
`design-fence-order.md`): does program order between two tile
operations of one instance order their memory accesses?

- `fig1_ptx.py`: compiles the paper's Figure 1 kernel (store the whole
  tile, then load own slot) for N in {32,128,1024} and num_warps in
  {1,4}, dumps TTIR/TTGIR/LLIR/PTX next to itself, and prints the
  memory/barrier instruction order. Finding: no bar.sync, membar, or
  fence between st.global and ld.global; `tl.debug_barrier` lowers to
  `bar.sync 0` exactly between them.
- `fig1_stale_reads.py [iters]`: launches the same kernel repeatedly
  with hist pre-filled with -1 and counts launches where the phase-2
  load returned the stale value. Finding (RTX 4090, triton 3.6):
  559/3000 (N=128, w4), 377/3000 (N=1024, w4), 1446/3000 (N=1024, w8);
  0/3000 with the barrier; single-CTA p=1000: 3000/3000 stale, p=0: 0.
- `cross_warp_stress.py [iters]`: independent cross-warp variant.
  Finding: 7207/153600 stale (grid 512), 0 with the barrier.
- `cutile_tokens.py`: compiles cuda.tile kernels and prints the full
  final IR to trace token wiring. Finding: the `token_order` pass
  makes same-parameter RAW/WAW/WAR token-ordered automatically and
  widens chains at release/acquire; cross-parameter aliasing is not
  ordered; the DSL has no fence primitive.

Run from the repo root with the project venv (GPU required):
`.venv/bin/python evaluation/probes/fence_order/<script>.py`.
