import os


def _is_one(env: str, default: str = "0") -> bool:
    return os.getenv(env, default) == "1"


def _get_int_env(env: str, default: int, minimum: int | None = None) -> int:
    try:
        value = int(os.getenv(env, str(default)))
    except ValueError:
        value = default
    return max(minimum, value) if minimum is not None else value


class Config:
    """
    Runtime configuration loaded from environment variables.

    Fields map to environment variables with string values where "1" enables a flag:
    - verbose: TRITON_VIZ_VERBOSE, enables verbose logging.
    - num_sms: TRITON_VIZ_NUM_SMS, emulated concurrent SMs in the CPU interpreter
      (min 1).
    - enable_sanitizer: ENABLE_SANITIZER, toggles the sanitizer pipeline.
    - enable_profiler: ENABLE_PROFILER, toggles the profiler pipeline.
    - enable_timing: ENABLE_TIMING, collects timing info during execution.
    - report_grid_execution_progress: REPORT_GRID_EXECUTION_PROGRESS, logs per
      program block progress in the interpreter.
    - virtual_memory: SANITIZER_ENABLE_FAKE_TENSOR, uses a fake tensor backend in
      sanitizer runs to avoid real memory reads.
    - profiler_enable_load_store_skipping: PROFILER_ENABLE_LOAD_STORE_SKIPPING,
      skips redundant load/store checks to speed profiling.
    - profiler_enable_block_sampling: PROFILER_ENABLE_BLOCK_SAMPLING, samples a
      subset of blocks to reduce profiling overhead.
    - profiler_disable_buffer_load_check: PROFILER_DISABLE_BUFFER_LOAD_CHECK,
      disables buffer load checks in the profiler.
    """

    def __init__(self) -> None:
        self.cli_active: bool = False
        self.reset()

    def reset(self) -> None:
        """Reload configuration from environment variables and apply defaults."""
        self.verbose: bool = _is_one("TRITON_VIZ_VERBOSE")
        self.num_sms: int = _get_int_env("TRITON_VIZ_NUM_SMS", 1, minimum=1)
        self.enable_sanitizer: bool = _is_one("ENABLE_SANITIZER", "1")
        self.enable_profiler: bool = _is_one("ENABLE_PROFILER", "1")
        self.enable_timing: bool = _is_one("ENABLE_TIMING")
        self.report_grid_execution_progress: bool = _is_one(
            "REPORT_GRID_EXECUTION_PROGRESS"
        )
        self.virtual_memory: bool = _is_one("SANITIZER_ENABLE_FAKE_TENSOR")
        self.profiler_enable_load_store_skipping: bool = _is_one(
            "PROFILER_ENABLE_LOAD_STORE_SKIPPING", "1"
        )
        self.profiler_enable_block_sampling: bool = _is_one(
            "PROFILER_ENABLE_BLOCK_SAMPLING", "1"
        )
        self.profiler_disable_buffer_load_check: bool = _is_one(
            "PROFILER_DISABLE_BUFFER_LOAD_CHECK"
        )


config = Config()
