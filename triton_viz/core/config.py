import os
import sys
import types
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    verbose: bool
    sanitizer_activated: bool
    virtual_memory: bool
    disable_sanitizer: bool
    report_grid_execution_progress: bool

    def reset() -> None:
        ...


class Config(types.ModuleType):
    def __init__(self, name: str) -> None:
        super().__init__(name)
        self.reset()

    def reset(self) -> None:
        # --- Verbose mode flag ---
        self.verbose = os.getenv("TRITON_VIZ_VERBOSE", "0") == "1"

        # --- Sanitizer activation flag ---
        self.sanitizer_activated = False

        # --- Sanitizer disable flag ---
        self._disable_sanitizer = os.getenv("DISABLE_SANITIZER", "0") == "1"

        # --- Grid execution progress flag ---
        self.report_grid_execution_progress = (
            os.getenv("REPORT_GRID_EXECUTION_PROGRESS", "0") == "1"
        )  # verify using setter

        # --- Virtual memory flag ---
        self._virtual_memory = os.getenv("TRITON_VIZ_VIRTUAL_MEMORY", "0") == "1"

    @property
    def virtual_memory(self) -> bool:
        return self._virtual_memory

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


# Replace the current module object with a live Config instance
sys.modules[__name__] = Config(__name__)
