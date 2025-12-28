import os


class Config:
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        # --- Verbose mode flag ---
        self.verbose = os.getenv("TRITON_VIZ_VERBOSE", "0") == "1"

        # --- Interpreter concurrency (CPU) ---
        num_sms_env = os.getenv("TRITON_VIZ_NUM_SMS", "1")
        try:
            self._num_sms = max(1, int(num_sms_env))
        except ValueError:
            self._num_sms = 1

        # --- Sanitizer enable flag ---
        enable_sanitizer_env = os.getenv("ENABLE_SANITIZER")
        if enable_sanitizer_env is None:
            self._enable_sanitizer = not (
                os.getenv("DISABLE_SANITIZER", "0") == "1"
            )
        else:
            self._enable_sanitizer = enable_sanitizer_env == "1"

        # --- Profiler enable flag ---
        enable_profiler_env = os.getenv("ENABLE_PROFILER")
        if enable_profiler_env is None:
            self._enable_profiler = not (os.getenv("DISABLE_PROFILER", "0") == "1")
        else:
            self._enable_profiler = enable_profiler_env == "1"

        # --- Timing enable flag ---
        self._enable_timing = os.getenv("ENABLE_TIMING", "0") == "1"

        # --- Grid execution progress flag ---
        self._report_grid_execution_progress = (
            os.getenv("REPORT_GRID_EXECUTION_PROGRESS", "0") == "1"
        )  # verify using setter

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

    # ---------- enable_sanitizer ----------
    @property
    def enable_sanitizer(self) -> bool:
        return self._enable_sanitizer

    @enable_sanitizer.setter
    def enable_sanitizer(self, value: bool) -> None:
        if not isinstance(value, bool):
            raise TypeError("enable_sanitizer expects a bool.")

        previous = getattr(self, "_enable_sanitizer", None)
        self._enable_sanitizer = value

        # User-friendly status messages
        if value and not previous:
            print("Triton Sanitizer enabled.")
        elif not value and previous:
            print("Triton Sanitizer disabled.")

    # ---------- enable_profiler ----------
    @property
    def enable_profiler(self) -> bool:
        return self._enable_profiler

    @enable_profiler.setter
    def enable_profiler(self, value: bool) -> None:
        if not isinstance(value, bool):
            raise TypeError("enable_profiler expects a bool.")

        previous = getattr(self, "_enable_profiler", None)
        self._enable_profiler = value

        # User-friendly status messages
        if value and not previous:
            print("Triton Profiler enabled.")
        elif not value and previous:
            print("Triton Profiler disabled.")

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

    # ---------- num_sms ----------
    @property
    def num_sms(self) -> int:
        return self._num_sms

    @num_sms.setter
    def num_sms(self, value: int) -> None:
        if not isinstance(value, int):
            raise TypeError("num_sms expects an int.")
        if value < 1:
            raise ValueError("num_sms must be >= 1.")
        self._num_sms = value

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


config = Config()
