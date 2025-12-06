import os


class Config:
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        # --- Verbose mode flag ---
        self.verbose = os.getenv("TRITON_VIZ_VERBOSE", "0") == "1"

        # --- Sanitizer activation flag ---
        self.sanitizer_activated = False

        # --- Sanitizer disable flag ---
        self._disable_sanitizer = os.getenv("DISABLE_SANITIZER", "0") == "1"

        # --- Profiler disable flag ---
        self._disable_profiler = os.getenv("DISABLE_PROFILER", "0") == "1"

        # --- Timing enable flag ---
        self._enable_timing = os.getenv("ENABLE_TIMING", "0") == "1"

        # --- Grid execution progress flag ---
        self.report_grid_execution_progress = (
            os.getenv("REPORT_GRID_EXECUTION_PROGRESS", "0") == "1"
        )  # verify using setter

        # --- Sanitizer Cache Ablation Study ---
        # Cache 1: SymExpr Node Cache - caches Z3 expressions in SymExpr._z3
        self.enable_symbol_cache = (
            os.getenv("SANITIZER_ENABLE_SYMBOL_CACHE", "1") == "1"
        )

        # Cache 2: Loop Iterator Cache - deduplicates memory patterns in loops
        self.enable_loop_cache = os.getenv("SANITIZER_ENABLE_LOOP_CACHE", "1") == "1"

        # Cache 3: Grid Cache - shared solver per kernel launch (incremental SMT)
        self.enable_grid_cache = os.getenv("SANITIZER_ENABLE_GRID_CACHE", "1") == "1"

        # Cache 4: Kernel Cache - skip re-analysis of identical kernel launches
        self.enable_kernel_cache = (
            os.getenv("SANITIZER_ENABLE_KERNEL_CACHE", "1") == "1"
        )

        # --- Fake tensor flag ---
        self._virtual_memory = os.getenv("SANITIZER_ENABLE_FAKE_TENSOR", "0") == "1"

        # --- Profiler Optimization Ablation Study ---
        # Optimization 1: enable load/store/dot skipping
        self._profiler_enable_load_store_skipping = (
            os.getenv("PROFILER_ENABLE_LOAD_STORE_SKIPPING", "1") == "1"
        )

        # Optimization 2: Profiler enable block sampling
        self._profiler_enable_block_sampling = (
            os.getenv("PROFILER_ENABLE_BLOCK_SAMPLING", "1") == "1"
        )

        # --- Profiler Performance Issues Detection ---
        # AMD Buffer Load Check: detects buffer_load instruction usage in kernel ASM
        self._profiler_disable_buffer_load_check = (
            os.getenv("PROFILER_DISABLE_BUFFER_LOAD_CHECK", "0") == "1"
        )

        # --- Profiler Backend Selection ---
        # Use symbolic profiler (SymbolicProfiler) instead of concrete profiler (ProfilerConcrete)
        self._profiler_use_symbolic = (
            os.getenv("PROFILER_USE_SYMBOLIC", "0") == "1"
        )

    # ---------- disable_sanitizer ----------
    @property
    def disable_sanitizer(self) -> bool:
        return self._disable_sanitizer

    @disable_sanitizer.setter
    def disable_sanitizer(self, value: bool) -> None:
        if not isinstance(value, bool):
            raise TypeError("disable_sanitizer expects a bool.")

        previous = getattr(self, "_disable_sanitizer", None)
        self._disable_sanitizer = value

        # User-friendly status messages
        if value and not previous:
            print("Triton Sanitizer disabled.")
        elif not value and previous:
            print("Triton Sanitizer enabled.")

    # ---------- disable_profiler ----------
    @property
    def disable_profiler(self) -> bool:
        return self._disable_profiler

    @disable_profiler.setter
    def disable_profiler(self, value: bool) -> None:
        if not isinstance(value, bool):
            raise TypeError("disable_profiler expects a bool.")

        previous = getattr(self, "_disable_profiler", None)
        self._disable_profiler = value

        # User-friendly status messages
        if value and not previous:
            print("Triton Profiler disabled.")
        elif not value and previous:
            print("Triton Profiler enabled.")

    # ---------- enable_timing ----------
    @property
    def enable_timing(self) -> bool:
        return self._enable_timing

    @enable_timing.setter
    def enable_timing(self, value: bool) -> None:
        if not isinstance(value, bool):
            raise TypeError("enable_timing expects a bool.")
        self._enable_timing = value

        previous = getattr(self, "_enable_timing", None)
        self._enable_timing = value

        # User-friendly status messages
        if value and not previous:
            print("Triton timing enabled.")
        elif not value and previous:
            print("Triton timing disabled.")

    # ---------- report_grid_execution_progress ----------
    @property
    def report_grid_execution_progress(self) -> bool:
        return self._report_grid_execution_progress

    @report_grid_execution_progress.setter
    def report_grid_execution_progress(self, flag: bool) -> None:
        if not isinstance(flag, bool):
            raise TypeError("report_grid_execution_progress expects a bool.")
        self._report_grid_execution_progress = flag
        if flag:
            print("Grid-progress reporting is now ON.")

    # ---------- profiler_enable_load_store_skipping ----------
    @property
    def profiler_enable_load_store_skipping(self) -> bool:
        return self._profiler_enable_load_store_skipping

    @profiler_enable_load_store_skipping.setter
    def profiler_enable_load_store_skipping(self, value: bool) -> None:
        if not isinstance(value, bool):
            raise TypeError("profiler_enable_load_store_skipping expects a bool.")

        previous = getattr(self, "_profiler_enable_load_store_skipping", None)
        self._profiler_enable_load_store_skipping = value

        # User-friendly status messages
        if value and not previous:
            print("Profiler load/store/dot skipping enabled.")
        elif not value and previous:
            print("Profiler load/store/dot skipping disabled.")

    # ---------- profiler_enable_block_sampling ----------
    @property
    def profiler_enable_block_sampling(self) -> bool:
        return self._profiler_enable_block_sampling

    @profiler_enable_block_sampling.setter
    def profiler_enable_block_sampling(self, value: bool) -> None:
        if not isinstance(value, bool):
            raise TypeError("profiler_enable_block_sampling expects a bool.")

        previous = getattr(self, "_profiler_enable_block_sampling", None)
        self._profiler_enable_block_sampling = value

        # User-friendly status messages
        if value and not previous:
            print("Profiler block sampling enabled.")
        elif not value and previous:
            print("Profiler block sampling disabled.")

    # ---------- virtual memory ----------
    @property
    def virtual_memory(self) -> bool:
        return self._virtual_memory

    # ---------- profiler_disable_buffer_load_check ----------
    @property
    def profiler_disable_buffer_load_check(self) -> bool:
        return self._profiler_disable_buffer_load_check

    @profiler_disable_buffer_load_check.setter
    def profiler_disable_buffer_load_check(self, value: bool) -> None:
        if not isinstance(value, bool):
            raise TypeError("profiler_disable_buffer_load_check expects a bool.")

        previous = getattr(self, "_profiler_disable_buffer_load_check", None)
        self._profiler_disable_buffer_load_check = value

        # User-friendly status messages
        if value and not previous:
            print("Profiler buffer load check disabled.")
        elif not value and previous:
            print("Profiler buffer load check enabled.")

    # ---------- profiler_use_symbolic ----------
    @property
    def profiler_use_symbolic(self) -> bool:
        return self._profiler_use_symbolic

    @profiler_use_symbolic.setter
    def profiler_use_symbolic(self, value: bool) -> None:
        if not isinstance(value, bool):
            raise TypeError("profiler_use_symbolic expects a bool.")

        previous = getattr(self, "_profiler_use_symbolic", None)
        self._profiler_use_symbolic = value

        # User-friendly status messages
        if value and not previous:
            print("Profiler using symbolic backend (SymbolicProfiler).")
        elif not value and previous:
            print("Profiler using concrete backend (ConcreteProfiler).")


config = Config()
