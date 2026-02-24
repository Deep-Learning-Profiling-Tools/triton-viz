import os
import shutil
import runpy
import sys
import pytest
import triton
import triton_viz
from triton_viz.clients import Sanitizer, Profiler
from triton_viz.core.config import config as cfg


# Command names
SANITIZER_COMMAND = "triton-sanitizer"
PROFILER_COMMAND = "triton-profiler"

# store the original triton.jit
_original_jit = triton.jit
_original_autotune = triton.autotune


def sanitizer_wrapper(kernel):
    abort_on_error = True
    tracer = triton_viz.trace(client=Sanitizer(abort_on_error=abort_on_error))
    return tracer(kernel)


def profiler_wrapper(kernel):
    tracer = triton_viz.trace(client=Profiler())
    return tracer(kernel)


def create_patched_jit(wrapper_func):
    def _patched_jit(fn=None, **jit_kw):
        if fn is None:  # @triton.jit(**opts)

            def _decorator(f):
                k = _original_jit(**jit_kw)(f)
                return wrapper_func(k)

            return _decorator
        else:  # @triton.jit
            k = _original_jit(fn)
            return wrapper_func(k)

    return _patched_jit


def create_patched_autotune(wrapper_func):
    def _patched_autotune(fn=None, **autotune_kw):
        if fn is None:

            def _decorator(f):
                k = _original_autotune(**autotune_kw)(f)
                return wrapper_func(k)

            return _decorator
        else:
            k = _original_autotune(fn)
            return wrapper_func(k)

    return _patched_autotune


def _apply_wrapper(wrapper_func, command_name, usage_msg):
    """
    Generic function to apply a wrapper to triton.jit and run the user script.
    """
    cfg.cli_active = True

    # Create patched functions with the specific wrapper
    _patched_jit = create_patched_jit(wrapper_func)
    _patched_autotune = create_patched_autotune(wrapper_func)

    # patching the original triton.jit
    triton.jit = _patched_jit
    triton.language.jit = _patched_jit
    import triton.runtime.interpreter as _interp

    _interp.jit = _patched_jit

    # patching triton.autotune
    triton.autotune = _patched_autotune

    # run user script
    if len(sys.argv) < 2:
        print(usage_msg)
        sys.exit(1)

    script = sys.argv[1]

    if os.path.isfile(script):
        # It's a Python script file, run it directly
        sys.argv = sys.argv[1:]
        runpy.run_path(script, run_name="__main__")
    else:
        # It might be a command like 'pytest', run it as a subprocess
        cmd = sys.argv[1:]

        # Check if it's an executable command
        if shutil.which(cmd[0]):
            if cmd[0] == "pytest":
                sys.exit(pytest.main(cmd[1:]))
            elif cmd[0] == "python":
                sys.argv = cmd[1:]
                try:
                    runpy.run_path(cmd[1], run_name="__main__")
                except SystemExit as e:
                    sys.exit(e.code)
                sys.exit(0)
        print(f"Error: '{script}' is neither a valid file nor a command")
        sys.exit(1)


def apply_sanitizer():
    """
    Apply the sanitizer wrapper to triton.jit and run the user script.
    """
    _apply_wrapper(
        sanitizer_wrapper,
        SANITIZER_COMMAND,
        f"Usage: {SANITIZER_COMMAND} <script.py> [args...]",
    )


def apply_profiler():
    """
    Apply the profiler wrapper to triton.jit and run the user script.
    """
    _apply_wrapper(
        profiler_wrapper,
        PROFILER_COMMAND,
        f"Usage: {PROFILER_COMMAND} <script.py> [args...]",
    )
