import os
import sys
import types
from typing import TYPE_CHECKING, Literal


if TYPE_CHECKING:
    verbose: bool
    sanitizer_activated: bool
    sanitizer_backend: Literal["off", "brute_force", "symexec"]
    report_grid_execution_progress: bool
    available_backends: tuple[str, ...]

    def reset() -> None:
        ...


# Back-end options recognised by the sanitizer
AVAILABLE_SANITIZER_BACKENDS = ("off", "brute_force", "symexec")


class Config(types.ModuleType):
    def __init__(self, name: str) -> None:
        super().__init__(name)
        self.reset()

    def reset(self) -> None:
        # --- Verbose mode flag ---
        self.verbose = os.getenv("TRITON_VIZ_VERBOSE", "0") == "1"

        # --- Sanitizer activation flag ---
        self.sanitizer_activated = False

        # --- Sanitizer backend ---
        self._sanitizer_backend = os.getenv("TRITON_SANITIZER_BACKEND", "") or None

        # --- Grid execution progress flag ---
        self.report_grid_execution_progress = (
            os.getenv("REPORT_GRID_EXECUTION_PROGRESS", "0") == "1"
        )  # verify using setter

    # ---------- sanitizer_backend ----------
    @property
    def sanitizer_backend(self) -> str:
        if not self.sanitizer_activated:
            return "not_activated"
        if self._sanitizer_backend is None:
            raise RuntimeError(
                f"TRITON_SANITIZER_BACKEND is not set!"
                f"Available backends are: {AVAILABLE_SANITIZER_BACKENDS}"
            )
        return self._sanitizer_backend

    @sanitizer_backend.setter
    def sanitizer_backend(self, value: str) -> None:
        if value not in AVAILABLE_SANITIZER_BACKENDS:
            raise ValueError(
                f"Invalid sanitizer_backend: {value!r}. "
                f"Valid choices: {AVAILABLE_SANITIZER_BACKENDS}"
            )

        previous = getattr(self, "_sanitizer_backend", None)
        self._sanitizer_backend = value

        # User-friendly status messages
        if value == "off" and previous != "off":
            print("Triton Sanitizer disabled.")
        elif value != "off" and (previous == "off" or previous is None):
            print(f"Triton Sanitizer enabled with backend: {value!r}")

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

    # ---------- read-only helpers ----------
    @property
    def available_backends(self):
        return AVAILABLE_SANITIZER_BACKENDS


# Replace the current module object with a live Config instance
sys.modules[__name__] = Config(__name__)
