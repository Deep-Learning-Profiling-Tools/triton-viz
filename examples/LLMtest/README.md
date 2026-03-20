# LLM test

This folder contains intentionally buggy Triton kernels for evaluating
LLM-based trace analysis in Triton-Viz.

## How to run

From repo root:

- `python "examples/LLM test/buggy_vector_add_shift.py"`
- `python "examples/LLM test/buggy_matmul_missing_k.py"`
- `python "examples/LLM test/buggy_softmax_no_stability.py"`

Each script:

- runs a buggy kernel with `@triton_viz.trace(client=Tracer())`
- prints a quick numerical mismatch signal
- launches Triton-Viz (`triton_viz.launch(share=True)`)

## Intended bug patterns

- `buggy_vector_add_shift.py`: wrong load index for `y` (`offset + 1`)
- `buggy_matmul_missing_k.py`: missing last K tile accumulation
- `buggy_softmax_no_stability.py`: skips max-subtraction, prone to overflow
