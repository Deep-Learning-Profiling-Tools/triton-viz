import os


# global configs
report_grid_execution_progress = os.getenv('REPORT_GRID_EXECUTION_PROGRESS', '0') == '1'

# sanitizer configs
sanitizer_backend = os.getenv("TRITON_SANITIZER_BACKEND", "")
available_backends = ["off", "brute_force", "z3", "symexec"]
if sanitizer_backend == "":
    print(f"TRITON_SANITIZER_BACKEND not set. Available backends are: {available_backends}. Defaulting to 'off'.")
    sanitizer_backend = "off"
if sanitizer_backend == "off":
    print("Triton Sanitizer is disabled since TRITON_SANITIZER_BACKEND=off.")
if sanitizer_backend not in available_backends:
    raise ValueError(f"Invalid TRITON_SANITIZER_BACKEND: {sanitizer_backend}. Available backends are: {available_backends}")

