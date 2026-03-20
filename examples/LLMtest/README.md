# LLM test

Intentionally buggy Triton kernels for trying the visualizer’s LLM assistant.

## Run

From repo root:

```bash
python examples/LLMtest/buggy_vector_add_shift.py
python examples/LLMtest/buggy_matmul_missing_k.py
python examples/LLMtest/buggy_softmax_no_stability.py
```

Each script has two optional constants at the top:

- `_LL_CONFIG_PATH` — path to a JSON file (same keys as `triton_viz/visualizer/llm_config.example.json`)
- `_LL_API_KEY` — API key string if it is not in that file

Fill either or both, then run. If both are empty, LLM still follows `llm_config.local.json` / env vars / defaults like the rest of the app.

## Bug patterns

- `buggy_vector_add_shift.py`: wrong load index for `y` (`offset + 1`)
- `buggy_matmul_missing_k.py`: missing last K tile accumulation
- `buggy_softmax_no_stability.py`: no max-subtraction, prone to overflow
