import os


# global configs
trace_enabled = os.getenv('ENABLE_TRITON_SANITIZER', '0') == '1'
report_grid_execution_progress = os.getenv('REPORT_GRID_EXECUTION_PROGRESS', '0') == '1'

# sanitizer configs
sanitizer_backend = os.getenv("TRITON_SANITIZER_BACKEND", "")

# global states
global_warning_toggled = dict()
global_warning_toggled['sanitizer'] = False
global_warning_toggled['trace'] = False

