import os


# configs
trace_enabled = os.getenv('ENABLE_TRITON_SANITIZER', '0') == '1'

# global states
trace_warning_toggled = False