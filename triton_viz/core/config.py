import os


# configs
trace_enabled = os.getenv('ENABLE_TRITON_SANITIZER', '0') == '1'
report_grid_execution_progress = os.getenv('REPORT_GRID_EXECUTION_PROGRESS', '0') == '1'

# global states
trace_warning_toggled = False

