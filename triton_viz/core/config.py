import os, sys, types


# Back-end options recognised by the sanitizer
AVAILABLE_SANITIZER_BACKENDS = ("off", "brute_force", "symexec")

class Config(types.ModuleType):
    def __init__(self, name: str) -> None:
        super().__init__(name)

        # --- Sanitizer backend ---
        env_backend = os.getenv("TRITON_SANITIZER_BACKEND", "")
        if env_backend:
            self.sanitizer_backend = env_backend  # verify using setter
        else:
            print(
                f"TRITON_SANITIZER_BACKEND not set. "
                f"Available backends are: {AVAILABLE_SANITIZER_BACKENDS}. Defaulting to 'off'."
            )
            self._sanitizer_backend = "off"

        # --- Grid execution progress flag ---
        env_flag = os.getenv("REPORT_GRID_EXECUTION_PROGRESS", "0")
        self.report_grid_execution_progress = env_flag == "1"  # verify using setter

    # ---------- sanitizer_backend ----------
    @property
    def sanitizer_backend(self) -> str:
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
