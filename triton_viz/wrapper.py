import runpy, sys
import triton
import triton_viz
from triton_viz.clients import Sanitizer


# store the original triton.jit
_original_jit = triton.jit


def sanitizer_wrapper(kernel):
    abort_on_error = True
    tracer = triton_viz.trace(clients=Sanitizer(abort_on_error=abort_on_error))
    return tracer(kernel)


def _patched_jit(fn=None, **jit_kw):
    if fn is None:  # @triton.jit(**opts)

        def _decorator(f):
            k = _original_jit(**jit_kw)(f)
            return sanitizer_wrapper(k)

        return _decorator
    else:  # @triton.jit
        k = _original_jit(fn)
        return sanitizer_wrapper(k)


def apply():
    """
    Apply the sanitizer wrapper to triton.jit and run the user script.
    """
    # patching the original triton.jit
    triton.jit = _patched_jit
    triton.language.jit = _patched_jit
    import triton.runtime.interpreter as _interp

    _interp.jit = _patched_jit

    # run user script
    # argv is like: ['triton-sanitizer', 'user_script.py', 'arg1', 'arg2', ...]
    if len(sys.argv) < 2:
        print("Usage: triton-sanitizer <script.py> [args...]")
        sys.exit(1)
    script = sys.argv[1]
    sys.argv = sys.argv[1:]
    runpy.run_path(script, run_name="__main__")
