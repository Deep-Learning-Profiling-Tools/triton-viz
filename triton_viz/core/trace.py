import weakref
from contextlib import contextmanager
from copy import deepcopy
from collections.abc import Callable
from typing import Any
from ..utils.traceback_utils import CODE_KEYS, get_code_key

from .config import config as cfg
from ..clients import Sanitizer, Profiler, RaceDetector, Tracer
from ..clients.race_detector.race_detector import NullRaceDetector
from .client import ClientManager, Client
from .data import Launch
import types


launches: list[Launch] = []

# Every live TritonTrace, so real-compile windows can present the underlying
# JITFunction to Triton's code generator (see _unwrapped_jit_globals).
_all_traces: "weakref.WeakSet[Any]" = weakref.WeakSet()


@contextmanager
def _unwrapped_jit_globals():
    """Temporarily swap module globals holding a TritonTrace back to the
    wrapped JITFunction.

    Under the CLI wrappers every ``@triton.jit`` function — including DEVICE
    functions — is wrapped into a TritonTrace. Triton's real code generator
    resolves a callee through the caller's ``__globals__`` and only accepts
    JITFunctions there ("Unsupported function referenced" otherwise). The
    interpreter path tolerates the wrapper via ``TritonTrace.__call__``; a
    real compile (the warmup-only path) does not, so for the duration of a
    real-compile window each trace's global binding is unwound to its
    ``jit_fn`` and restored afterwards.
    """
    swapped: list[tuple[dict, str, Any]] = []
    for trace in list(_all_traces):
        jit_fn = trace.jit_fn
        base_fn = trace.base_fn
        if jit_fn is None or base_fn is None:
            continue
        module_globals = getattr(base_fn, "__globals__", None)
        if module_globals is None:
            continue
        for name, value in list(module_globals.items()):
            if value is trace:
                module_globals[name] = jit_fn
                swapped.append((module_globals, name, trace))
    try:
        yield
    finally:
        for module_globals, name, trace in swapped:
            module_globals[name] = trace


class TraceInterface:
    def __init__(self, client: str | Client) -> None:
        self.client_manager = ClientManager()
        self.add_client(client)

    @staticmethod
    def _normalize_client(client: str | Client) -> Client:
        if isinstance(client, str):
            name = client.lower()
            if name == "sanitizer":
                return Sanitizer()
            if name == "profiler":
                return Profiler()
            if name == "race_detector":
                return RaceDetector()
            if name == "tracer":
                return Tracer()
            raise ValueError(f"Unknown client: {client}")
        elif isinstance(client, Client):
            return client
        else:
            raise TypeError(f"Expected str or Client, got {type(client)}")

    def add_client(self, new_client: str | Client) -> None:
        self.client_manager.add_clients([self._normalize_client(new_client)])

    def finalize(self):
        self.client_manager.finalize()
        launches.append(self.client_manager.launch)


class LaunchInterface:
    def warmup(self, *args, grid, **kwargs):
        from triton.runtime.jit import MockTensor

        return self.run(
            grid=grid,
            warmup=True,
            *map(MockTensor.wrap_dtype, args),
            **kwargs,
        )

    def __getitem__(self, grid):
        return lambda *args, **kwargs: self.run(
            grid=grid,
            warmup=False,
            *args,
            **kwargs,
        )


class KernelTraceSupport:
    @staticmethod
    def _is_autotuner(runner: Any) -> bool:
        from triton.runtime import Autotuner

        return isinstance(runner, Autotuner)

    @staticmethod
    def _is_heuristics(runner: Any) -> bool:
        from triton.runtime.autotuner import Heuristics

        return isinstance(runner, Heuristics)

    @staticmethod
    def dummy_benchmarker(fn, quantiles):
        fn()
        return (1.0, 1.0, 1.0)

    def _interpreter_runner(self, runner: Any, interpreted_fn: Any) -> Any:
        if self._is_autotuner(runner):
            runner.fn = interpreted_fn
            # Kernel Cache: replace the benchmark with a dummy to skip performance testing.
            runner._do_bench = self.dummy_benchmarker
            return runner
        if self._is_heuristics(runner):
            runner.fn = interpreted_fn
            return runner
        return interpreted_fn

    def _warmup_runner(self, runner: Any, jit_fn: Any | None) -> Any | None:
        if not (self._is_autotuner(runner) or self._is_heuristics(runner)):
            return jit_fn
        if jit_fn is None:
            return None
        warmup_runner = deepcopy(runner)
        warmup_runner.fn = jit_fn
        return warmup_runner

    def _copy_callable_attrs(
        self,
        runner: Any,
        base_fn: Callable | None = None,
        *,
        src_fallback: Any = None,
    ) -> None:
        for attr in ("__name__", "__module__", "__doc__", "__qualname__"):
            if hasattr(runner, attr):
                setattr(self, attr, getattr(runner, attr))
            elif base_fn is not None and hasattr(base_fn, attr):
                setattr(self, attr, getattr(base_fn, attr))

        if not hasattr(self, "__name__"):
            self.__name__ = "<unknown>"

        if hasattr(runner, "src"):
            self.src = runner.src
        elif src_fallback is not None and hasattr(src_fallback, "src"):
            self.src = src_fallback.src


class TritonTrace(LaunchInterface, TraceInterface, KernelTraceSupport):
    def __init__(
        self,
        runner: Any,
        client: str | Client,
    ) -> None:
        from triton import JITFunction
        from triton.runtime.autotuner import Heuristics
        from triton.runtime.interpreter import InterpretedFunction

        self.jit_fn: Any | None = None
        self.base_fn: Callable | None = None
        self.interpreted_fn: Any | None = None

        def unpack_kernel(
            source: Any,
        ) -> tuple[Any | None, Callable | None, Any | None]:
            if isinstance(source, TritonTrace):
                return source.jit_fn, source.base_fn, source.interpreted_fn
            if isinstance(source, JITFunction):
                base_fn = source.fn
                return source, base_fn, InterpretedFunction(base_fn)
            if isinstance(source, InterpretedFunction):
                return None, source.fn, source
            if isinstance(source, Heuristics):
                # Heuristics wraps another kernel, recursively unpack it
                return unpack_kernel(source.fn)
            raise TypeError(f"Unsupported runner type: {type(source)}")

        if self._is_autotuner(runner):
            self.jit_fn, self.base_fn, self.interpreted_fn = unpack_kernel(runner.fn)
        elif self._is_heuristics(runner):
            self.jit_fn, self.base_fn, self.interpreted_fn = unpack_kernel(runner.fn)
        else:
            self.jit_fn, self.base_fn, self.interpreted_fn = unpack_kernel(runner)
        self.runner = self._interpreter_runner(runner, self.interpreted_fn)
        self.warmup_runner = self._warmup_runner(runner, self.jit_fn)

        self.arg_names = runner.arg_names

        self.fn = runner

        TraceInterface.__init__(self, client)

        self._copy_callable_attrs(runner, self.base_fn, src_fallback=self.jit_fn)

        # Register for _unwrapped_jit_globals: real-compile windows unwind
        # this trace's module-global binding to the raw JITFunction so the
        # code generator can resolve it as a device-function callee.
        _all_traces.add(self)

    def run(self, *args, **kwargs):
        clients = self.client_manager.clients
        warmup_only = (
            bool(clients)
            and self.warmup_runner is not None
            and all(getattr(c, "WARMUP_ONLY", False) for c in clients.values())
        )

        if warmup_only:
            # Warmup-only clients (e.g. the compiled-mode race detector)
            # consume nothing from the interpreted run: their whole analysis
            # input is the warmup compilation artifact. Skip the interpreter
            # entirely — no language patching, no grid loop — and execute the
            # REAL kernel instead, so the host script keeps its true semantics
            # (outputs, asserts, autotuning). This is load-bearing, not just
            # an optimization: Triton's interpreter patches tl.core.tensor
            # dunders in place and the snapshot/restore around it does not
            # survive a traced launch followed by a REAL compile of a second
            # kernel in the same process (the leaked interpreter __bool__
            # breaks semantic._load_legacy's `other.handle if other else
            # None`). Warmup-only clients never need that machinery, so they
            # never engage it. NOTE: the real handle is warmup_runner —
            # self.runner is the InterpretedFunction (or an Autotuner whose
            # fn was swapped to it); warmup_runner is the raw JITFunction, or
            # for Autotuner/Heuristics a deepcopy still bound to the real
            # kernel. The whole window (warmup compile + real launch, which
            # may compile further specializations) runs with TritonTrace
            # globals unwound so device-function callees resolve to real
            # JITFunctions in the code generator.
            with _unwrapped_jit_globals():
                with self.client_manager.patch_warmup(self.jit_fn):
                    if self.warmup_runner:
                        self.warmup_runner.warmup(*args, **kwargs)
                try:
                    ret = self.warmup_runner.run(*args, **kwargs)
                finally:
                    self.finalize()
                return ret

        with self.client_manager.patch_warmup(self.jit_fn):
            if self.warmup_runner:
                # Real warmup compilation walks the kernel AST to hash and
                # inline referenced device functions. When a kernel calls a
                # `@triton.jit` helper that we have also wrapped in trace()
                # (the CLI / "wrap every jit" pattern), that helper global is a
                # TritonTrace, which triton's dependency walker rejects
                # ("Unsupported function referenced"). Only compiled-mode
                # clients trigger this warmup; eager traces skip it. Present the
                # raw jit_fns to the compiler, then restore the wrappers.
                try:
                    with _unwrap_traced_globals(self.base_fn):
                        self.warmup_runner.warmup(*args, **kwargs)
                except RuntimeError as exc:
                    # Driverless host (CPU-only CI): triton's warmup resolves
                    # driver.active and dies with "0 active drivers". Warmup
                    # is OPTIONAL for the eager clients (under
                    # TRITON_INTERPRET it never ran at all — jit_fn is None
                    # and this branch is skipped); compiled-mode clients that
                    # NEED the artifacts surface their own honest
                    # no_ttgir/no_ttir verdicts downstream. Anything other
                    # than the driver-discovery failure still raises.
                    if "active driver" not in str(exc):
                        raise

        with self.client_manager.patch_run(self.base_fn, frontend_name="triton"):
            kwargs.update({"client_manager": self.client_manager})
            kwargs.update({"jit_fn": self.jit_fn})
            ret = self.runner.run(*args, **kwargs)
            self.finalize()
            return ret

    def __call__(self, *args, **kwargs):
        # When a traced JIT function is called from within another JIT function,
        # we need to execute the underlying function directly

        # check that client sets match for calling and called functions
        from .frontend import triton as triton_frontend

        outer_client_manager = triton_frontend.frontend.current_client_manager()
        if outer_client_manager is not None:
            outer_clients = set(outer_client_manager.clients)
            inner_clients = set(self.client_manager.clients)
            if outer_clients != inner_clients:
                raise RuntimeError(
                    "nested traced calls require matching clients; "
                    f"outer={outer_clients}, inner={inner_clients}"
                )

        return self.interpreted_fn(*args, **kwargs)

    def warmup(self, *args, **kwargs):
        with self.client_manager.patch_warmup(self.jit_fn):
            if self.warmup_runner:
                self.warmup_runner.warmup(*args, **kwargs)


@contextmanager
def _unwrap_traced_globals(base_fn: Callable | None):
    """Temporarily replace any TritonTrace in the kernel's reachable globals
    with its underlying jit_fn, for the duration of a real warmup compilation.

    A device function (``@triton.jit`` helper called from inside another
    kernel) wrapped in trace() appears as a TritonTrace. triton's dependency
    walker and codegen only accept JITCallables, so they raise "Unsupported
    function referenced" on the wrapper. Kernel and same-module helpers share
    one module dict, so swapping every TritonTrace there resolves the common
    case; a helper imported from another module is reached as ``mod.helper``,
    so we also descend one level into module objects in the kernel's globals
    (only TritonTrace entries are touched, and everything is restored).
    """
    g = getattr(base_fn, "__globals__", None)
    if g is None:
        yield
        return
    saved: list[tuple[dict, str, Any]] = []

    def unwrap(container: dict) -> None:
        for name, val in list(container.items()):
            if isinstance(val, TritonTrace) and val.jit_fn is not None:
                saved.append((container, name, val))
                container[name] = val.jit_fn

    unwrap(g)
    for val in list(g.values()):
        if isinstance(val, types.ModuleType):
            mod_dict = getattr(val, "__dict__", None)
            if isinstance(mod_dict, dict):
                unwrap(mod_dict)
    try:
        yield
    finally:
        for container, name, val in saved:
            container[name] = val


class NKITrace(LaunchInterface, TraceInterface):
    def __init__(self, kernel, client: str | Client, beta2: bool = True) -> None:
        nki_fn_cls: object = None
        if beta2:
            from .simulation.nki_beta2 import NKIBeta2InterpretedFunction

            nki_fn_cls = NKIBeta2InterpretedFunction
        else:
            from .simulation.nki import NKIInterpretedFunction

            nki_fn_cls = NKIInterpretedFunction

        self.frontend_name = "nki_beta2" if beta2 else "nki"
        if isinstance(kernel, nki_fn_cls):
            assert hasattr(kernel, "fn")
            self.interpreter_fn = kernel
            self.func = kernel.fn
        else:
            self.interpreter_fn = nki_fn_cls(kernel)
            self.func = kernel

        TraceInterface.__init__(self, client)

    def __getattr__(self, name):
        # Forward any missing attributes to the underlying runner
        # This allows Trace to transparently proxy attributes like 'src', 'hash', etc.
        # Use object.__getattribute__ to avoid infinite recursion
        try:
            fn = object.__getattribute__(self, "fn")
            if hasattr(fn, name):
                return getattr(fn, name)
        except AttributeError:
            pass

        try:
            jit_fn = object.__getattribute__(self, "jit_fn")
            if hasattr(jit_fn, name):
                return getattr(jit_fn, name)
        except AttributeError:
            pass

        try:
            base_fn = object.__getattribute__(self, "base_fn")
            if hasattr(base_fn, name):
                return getattr(base_fn, name)
        except AttributeError:
            pass

        raise AttributeError(
            f"'{type(self).__name__}' object has no attribute '{name}'"
        )

    def __getitem__(self, *grid):
        return LaunchInterface.__getitem__(self, tuple(*grid))

    def __call__(self, *args, **kwargs):
        return self[(1,)](*args, **kwargs)

    def run(self, *args, pre_trace=True, platform_target="trn1", **kwargs):
        """
        pre_trace: determines whether to do an initial NKI Beta 2 trace to capture some semantic errors.
            pre_trace=False has fewer guarantees on interpreter parity with NKI compiler but must be set
            if you want full python flexibility inside kernels (e.g. importing modules inside a kernel).
            Does nothing if self.frontend_name == 'nki'.
        """
        if self.frontend_name == "nki_beta2" and pre_trace:
            import nki

            kwargs.pop("warmup", None)
            grid = kwargs.pop("grid", None)
            nki.trace(self.func, grid=grid, platform_target=platform_target).specialize(
                *args, **kwargs
            )
            kwargs["grid"] = grid
        with self.client_manager.patch_run(
            self.func,
            frontend_name=self.frontend_name,
        ):
            kwargs.update({"client_manager": self.client_manager})
            ret = self.interpreter_fn.run(*args, **kwargs)
            self.finalize()
            return ret


class GluonTrace(LaunchInterface, TraceInterface, KernelTraceSupport):
    def __init__(self, runner: Any, client: str | Client) -> None:
        from triton.experimental import gluon
        from triton.runtime.autotuner import Heuristics
        from .simulation.gluon import GluonInterpretedFunction

        # Gluon has exposed different JIT class names across Triton versions.
        # Treat an object that already has the JIT runner protocol as compiled,
        # and only call gluon.jit for raw Python functions.
        if not all(hasattr(runner, attr) for attr in ("fn", "run", "arg_names")):
            runner = gluon.jit(runner)

        def unpack_kernel(source: Any) -> tuple[Callable, Any]:
            if isinstance(source, GluonTrace):
                return source.base_fn, source.interpreted_fn
            if isinstance(source, Heuristics):
                return unpack_kernel(source.fn)
            if all(hasattr(source, attr) for attr in ("fn", "arg_names")):
                base_fn = source.fn
                return base_fn, GluonInterpretedFunction(base_fn, source.arg_names)
            raise TypeError(f"Unsupported runner type: {type(source)}")

        self.base_fn, self.interpreted_fn = unpack_kernel(
            runner.fn if self._is_autotuner(runner) else runner
        )
        self.fn = runner
        self.arg_names = runner.arg_names
        self.runner = self._interpreter_runner(runner, self.interpreted_fn)

        TraceInterface.__init__(self, client)
        self._copy_callable_attrs(runner, self.base_fn)

    def run(self, *args, **kwargs):
        grid = kwargs.get("grid")
        if grid is None:
            raise TypeError(
                "GluonTrace.run() missing required keyword argument: 'grid'"
            )

        with self.client_manager.patch_run(self.base_fn, frontend_name="gluon"):
            try:
                ret = self.runner.run(
                    *args,
                    **kwargs,
                    client_manager=self.client_manager,
                )
            finally:
                self.client_manager.post_run_callback(self.base_fn)
            self.finalize()
            return ret

    def __call__(self, *args, **kwargs):
        return self.fn(*args, **kwargs)

    def warmup(self, *args, **kwargs):
        return self.run(*args, warmup=True, **kwargs)


def trace_source(kernel):
    """
    Add the kernel code to be traceable within stack traces for clients
    (e.g. to capture source code to display with visualizer/client).
    You can also use this to decorate other functions that a kernel calls.
    """
    base_fn = kernel
    while not isinstance(base_fn, types.FunctionType):
        # base_fn may be a raw function but also a JITFunction, Autotuner, InterpretedFunction, ...
        # we want to strip away the wrappers until we get to the python function
        base_fn = base_fn.fn
    CODE_KEYS.add(get_code_key(base_fn))
    return kernel


def trace(client: str | Client | None = None, frontend: str = "triton"):
    """
    Create a trace object that can be used to run a kernel with instrumentation client(s).

    :param kernel: The kernel to run.
    :param client: A client to run with the kernel. Defaults to Tracer() if not specified.
    """
    if client is None:
        client = Tracer()

    if not isinstance(client, (str, Client)):
        raise TypeError(f"Expected str or Client, got {type(client)}")

    def _is_sanitizer_client(selected: str | Client) -> bool:
        if isinstance(selected, str):
            return selected.lower() == "sanitizer"
        return isinstance(selected, Sanitizer)

    def _is_race_detector_client(selected: str | Client) -> bool:
        if isinstance(selected, str):
            return selected.lower() == "race_detector"
        # A NullRaceDetector instance means the public RaceDetector(...) factory
        # was called while the flag was off — equivalent to the string-dispatch
        # flag-off case, so take the same fast path. Explicit SymbolicRaceDetector()
        # instances bypass the factory's __new__ and reflect a deliberate opt-in;
        # they are intentionally NOT matched here, preserving the "explicit
        # detector wins over flag" semantic guarded by
        # test_flag_off_does_not_swallow_explicit_instance.
        return isinstance(selected, NullRaceDetector)

    def decorator(kernel) -> TritonTrace | NKITrace | GluonTrace | Any:
        if cfg.cli_active and isinstance(kernel, TraceInterface):
            raise RuntimeError(
                "@triton_viz.trace() decorator cannot be used together with "
                "CLI wrapper (e.g., triton-sanitizer / triton-profiler). "
                "Please remove the @triton_viz.trace() decorator from your code "
                "when using CLI tools."
            )

        if _is_sanitizer_client(client) and not cfg.enable_sanitizer:
            # when dry-running triton-sanitizer CLI (i.e. wrap kernels with sanitizer
            # tracing but don't actually sanitize), don't actually trace the kernel
            return kernel

        if _is_race_detector_client(client) and not cfg.enable_race_detector:
            # Flag-off escape hatch: leave the kernel untraced so ENABLE_RACE_DETECTOR=0
            # truly has zero runtime impact on code that opts into race_detector tracing.
            return kernel

        # If the object is already initialized as a TraceInterface, just append the new client(s)
        if isinstance(kernel, TraceInterface):
            trace = kernel
            trace.add_client(client)
            return trace

        trace_source(kernel)

        # First-time wrapping
        if frontend in ("nki", "nki_beta2"):
            return NKITrace(kernel, client, beta2=("beta2" in frontend))
        elif frontend == "gluon":
            return GluonTrace(kernel, client)
        elif frontend == "triton":
            return TritonTrace(kernel, client)
        else:
            raise ValueError(f"Unknown frontend: {frontend}")

    return decorator


def clear() -> None:
    """
    Clear all traces.
    """
    launches.clear()
