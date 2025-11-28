import os


class Config:
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        # --- Verbose mode flag ---
        self.verbose = os.getenv("TRITON_VIZ_VERBOSE", "0") == "1"

        # --- Sanitizer activation flag ---
        self.sanitizer_activated = False

        # --- Interpreter concurrency (CPU) ---
        num_sms_env = os.getenv("TRITON_VIZ_NUM_SMS", "1")
        try:
            self._num_sms = max(1, int(num_sms_env))
        except ValueError:
            self._num_sms = 1

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


config = Config()
