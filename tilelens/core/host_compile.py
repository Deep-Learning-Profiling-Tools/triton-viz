"""Host compile: one JITFunction call compiled for a GPUTarget without a GPU
(D25, D26).

IR mode reads the kernels Triton compiles, but the analysis is CPU work, so
the compile is too: :class:`HostCompiler` binds a call with the JIT's own
binder (``create_function_from_signature`` for the target's backend: the
signature types, i32 / i64 / u64 integers by value, the equal-to-1 and
divisibility specializations, tuples, tensor descriptors, constexprs,
``do_not_specialize``), packs it with ``JITFunction._pack_args`` and
compiles an ``ASTSource`` for the target, exactly as ``JITFunction.run``
would on a device of that target. No driver is queried and nothing is
loaded or launched: no ``get_current_device``, no stream, no
``_init_handles``. The JIT runtime's own hooks are not called either (the
function's ``pre_run_hooks``, ``knobs.runtime.jit_cache_hook`` /
``jit_post_compile_hook``, async compile mode): the host compile is no JIT
run.

The target is the caller's, whatever the machine has. While a thread host
compiles, Triton's ``driver.active`` answers that thread's target query
(``get_current_target``: what ``tl.target_info.is_cuda()`` /
``cuda_capability_geq()`` / ``is_hip()`` and the front end's own target
checks read) with the compile's target, and refuses any device query with
:class:`HostCompileUnavailable`; other threads, and this one outside its
compile, see Triton's own driver. The compile options name the target's
arch, so ``TRITON_OVERRIDE_ARCH`` does not reach a host compile either. A
compile that raises after its front end asked the driver anything (the
target, or a device query it refused and the kernel's code may have caught),
or after an earlier compile of the same kernel for the target did, says so
(:func:`target_queried`): what failed may be the target's answer. A call
that does not bind the kernel's parameters says that instead
(:func:`bind_failed`): the JIT raises the same for it on any device.

The pipeline stops at the latest stage the caller asks for: a TTIR-only
request runs the front end and the backend's ``ttir`` passes (the same
passes ``triton.compile`` runs), never ``ttgir`` / ``llir`` / the binary;
``"source"`` (the front end's module before any pass) needs no pass. Such a
truncated compile is kept in memory only (a ``HostKernel``), never in
Triton's on-disk cache, whose entries must hold the whole pipeline. A
request for the pipeline's last stage or for ``"sass"`` (disassembled from
the binary), and any request under a knob that rewrites or dumps stages
(``TRITON_KERNEL_OVERRIDE``, ``TRITON_KERNEL_DUMP``, ``USE_IR_LOC``, an
``ir_override`` option), is compiled by ``triton.compile`` itself, which
returns an (unloaded) ``CompiledKernel``; no device is involved either.
:meth:`HostCompiler.check_stages` rejects a stage name no kernel compiled
for the target holds.

Either artifact has ``.asm`` (stage -> text, or bytes for a binary),
``.metadata`` (a namedtuple: ``target``, ``name``, the compile options, ...)
and ``.hash``, the specialization: what ``triton.compile`` names the kernel
for that target, whatever stage the compile stopped at.

The APIs used are private to Triton. :func:`triton_api` checks that they
exist and that the front end's target queries can be scoped, and the first
compile for each target host-compiles a small built-in kernel first, so a
changed API fails as :class:`HostCompileUnavailable` naming it rather than
as an error blamed on the user's kernel (IR mode's version gate, D10b,
still bounds the Triton releases this runs on). Where releases' JIT
runtimes differ in what the host compile mirrors, ``_RELEASE_RUNTIMES``
says per Triton minor release what each does, and triton_api checks the
installed release's row against its code: a custom pipeline
(``knobs.runtime.add_stages_inspection_hook``), which Triton 3.8's JIT and
``triton.compile`` also key a kernel by, and a ``CompiledKernel.__del__``
that unloads through the driver (3.8), which may run in the middle of a host
compile. A release with no row, or one whose code does not match its row,
fails closed: a host compile under a custom pipeline is refused, and the
driver's ``utils`` are refused like any device query.

Importing this module does not import Triton.
"""

from __future__ import annotations

import functools
import hashlib
import inspect
import linecache
import re
import threading
from collections import namedtuple
from collections.abc import Hashable, Iterable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from types import CodeType, MappingProxyType, SimpleNamespace
from typing import Any

from . import config as config_module
from .config import DEFAULT_IR_TARGET


class HostCompileUnavailable(RuntimeError):
    """The host compile cannot run: the installed Triton lacks (or changed)
    an API it uses, or the compile asked for something only a device has.
    Never a kernel's own compile error."""


# The attributes a host compile's exception carries when the front end had
# asked the driver before it was raised (see target_queried), and when the
# call did not bind the kernel's parameters (see bind_failed).
_TARGET_QUERIED = "_tilelens_target_queried"
_BIND_FAILED = "_tilelens_bind_failed"
# The keyword arguments no option of the target's backend names (see
# unknown_options).
_UNKNOWN_OPTIONS = "_tilelens_unknown_options"


def target_queried(exc: BaseException | None) -> bool:
    """Whether ``exc`` was raised by a host compile whose front end had
    asked Triton's driver anything before it failed, so the failure may
    follow from the target's answer: the target
    (``driver.active.get_current_target()``: ``tl.target_info``, the
    tensor-descriptor lowering's native-TMA check, a constexpr function
    asking the driver), or anything else, which the host compile refuses
    (a device query the kernel's code caught, falling back to an answer of
    its own, is still a question about the device). Also true when an
    earlier compile of the same kernel for the same target by the same
    HostCompiler had asked: the kernel's code may keep the answer (a memo)
    and not ask again. False for any other exception, a bind failure
    (bind_failed) included. What the compile options derive from the target
    (e.g. its fp8 types, or whether ``num_ctas > 1`` is allowed) is no
    query. An answer the kernel's code keeps from a compile this
    HostCompiler did not run (another trace's, another target's, the
    untraced program's), and never asks for again, cannot be seen."""
    return exc is not None and getattr(exc, _TARGET_QUERIED, False) is True


def bind_failed(exc: BaseException | None) -> bool:
    """Whether ``exc`` was raised by a host compile while binding the call
    to the kernel's parameters (the JIT's binder: a missing or unexpected
    argument, an argument of a type Triton cannot pass) or keying it
    (``compute_cache_key``: e.g. an unhashable constexpr value): no target,
    and no compile, decides it, so ``JITFunction.run`` raises the same for
    the call on any device, and a traced launch raises it as is (D28, see
    tilelens.core.client.ClientManager.ir_capture). False for any other
    exception, e.g. the KeyError for a keyword that names neither a
    parameter nor an option of the target's backend (another backend may
    know it, see unknown_options), and for a HostCompileUnavailable."""
    return exc is not None and getattr(exc, _BIND_FAILED, False) is True


def _mark(exc: BaseException, attr: str) -> None:
    try:
        setattr(exc, attr, True)
    except Exception:  # an exception type that takes no attribute
        pass


def _mark_target_queried(exc: BaseException) -> None:
    _mark(exc, _TARGET_QUERIED)


def _mark_bind_failed(exc: BaseException) -> None:
    _mark(exc, _BIND_FAILED)


def unknown_options(exc: BaseException | None) -> tuple[str, ...]:
    """The call's keyword arguments that name neither a parameter of the
    kernel nor a compile option of the target's backend, when ``exc`` is
    the KeyError the JIT raises for them (``JITFunction._pack_args``); ()
    for any other exception. Such a call fails on every device whose
    backend does not know them (a misspelled option: on every GPU), yet
    another backend may know them (e.g. HIP's ``waves_per_eu``), so the
    compile failure is the target's, not the call's (not bind_failed)."""
    names = getattr(exc, _UNKNOWN_OPTIONS, ()) if exc is not None else ()
    return names if isinstance(names, tuple) else ()


def _mark_unknown_options(
    exc: BaseException, jit_fn: Any, backend: Any, kwargs: Mapping[str, Any]
) -> None:
    # JITFunction._pack_args's own check: a keyword in neither the parsed
    # options nor the signature. Parsing again is how the JIT reads the
    # options; if parsing is what failed, nothing is marked.
    try:
        known = vars(backend.parse_options(dict(kwargs)))
    except Exception:
        return
    params = {param.name for param in jit_fn.params}
    names = tuple(k for k in kwargs if k not in known and k not in params)
    if names:
        try:
            setattr(exc, _UNKNOWN_OPTIONS, names)
        except Exception:
            pass


def _mark_call_error(exc: BaseException) -> None:
    """Mark ``exc``, raised while binding or keying the call, as the call's
    own error (bind_failed), unless the host compile could not run."""
    if host_compile_unavailable(exc) is None:
        _mark_bind_failed(exc)


def host_compile_unavailable(exc: BaseException) -> HostCompileUnavailable | None:
    """The HostCompileUnavailable behind ``exc``: ``exc`` itself, or one it
    was raised from or while handling (Triton's code generator re-raises
    what a kernel's code raised as a CompilationError from it); None if
    there is none, i.e. ``exc`` is the kernel's own compile error."""
    seen: set[int] = set()
    link: BaseException | None = exc
    while link is not None and id(link) not in seen:
        if isinstance(link, HostCompileUnavailable):
            return link
        seen.add(id(link))
        # The chain a traceback shows: the cause, else the unsuppressed context.
        if link.__cause__ is not None:
            link = link.__cause__
        else:
            link = None if link.__suppress_context__ else link.__context__
    return None


# ─────────────────────────── targets (D26) ───────────────────────────

_TARGET_FORMS = (
    "'cuda:<compute capability>' (e.g. 'cuda:80', 'cuda:90'), "
    "'hip:<gfx arch>' (e.g. 'hip:gfx942'), either optionally followed by "
    "':<warp size>', or a triton.backends.compiler.GPUTarget"
)
_RE_CUDA = re.compile(r"cuda:(\d+)(?::(\d+))?")
# gfx<major><minor><stepping>: gfx90a, gfx942, gfx1100, ...
_RE_GFX = r"gfx\d{1,2}[0-9a-z]{2}"
_RE_HIP = re.compile(rf"hip:({_RE_GFX})(?::(\d+))?")
# Volta: no Triton release targets an older NVIDIA GPU.
_MIN_CUDA_CAPABILITY = 70


def _is_int(value: Any) -> bool:
    # A bool is an int, but no capability or warp size.
    return isinstance(value, int) and not isinstance(value, bool)


def _checked_target(target: Any, spec: Any) -> Any:
    backend, arch, warp_size = target.backend, target.arch, target.warp_size
    valid = (
        backend == "cuda"
        and _is_int(arch)
        and arch >= _MIN_CUDA_CAPABILITY
        or backend == "hip"
        and isinstance(arch, str)
        and re.fullmatch(_RE_GFX, arch) is not None
    )
    if not valid or not _is_int(warp_size) or warp_size <= 0:
        raise ValueError(
            f"invalid IR target {spec!r}: expected {_TARGET_FORMS}; a CUDA "
            f"compute capability is at least {_MIN_CUDA_CAPABILITY}, a warp "
            "size positive"
        )
    return target


@functools.lru_cache(maxsize=64)
def _parse_target_spec(spec: str) -> Any:
    from triton.backends.compiler import GPUTarget

    text = spec.strip().lower()
    if match := _RE_CUDA.fullmatch(text):
        capability, warp_size = match.group(1), match.group(2)
        target = GPUTarget("cuda", int(capability), int(warp_size) if warp_size else 32)
    elif match := _RE_HIP.fullmatch(text):
        gfx, warp_size = match.group(1), match.group(2)
        # CDNA (gfx9*) runs 64-wide wavefronts, RDNA 32-wide.
        default = 64 if gfx.startswith("gfx9") else 32
        target = GPUTarget("hip", gfx, int(warp_size) if warp_size else default)
    else:
        raise ValueError(f"invalid IR target {spec!r}: expected {_TARGET_FORMS}")
    return _checked_target(target, spec)


def parse_ir_target(spec: Any) -> Any:
    """The ``GPUTarget`` an IR target spec names: a ``GPUTarget`` itself, or
    a string such as ``"cuda:89"``, ``"cuda:90"``, ``"hip:gfx942"`` or
    ``"hip:gfx1100:32"``. Raises ValueError for anything else, a CUDA
    compute capability below 70 or a warp size that is not positive
    included."""
    from triton.backends.compiler import GPUTarget

    if isinstance(spec, GPUTarget):
        return _checked_target(spec, spec)
    if isinstance(spec, str):
        return _parse_target_spec(spec)
    raise ValueError(f"invalid IR target {spec!r}: expected {_TARGET_FORMS}")


def format_ir_target(target: Any) -> str:
    """A GPUTarget as the spec parse_ir_target reads back, e.g.
    ``"cuda:89"``; the warp size only where it is not the default."""
    backend = getattr(target, "backend", None)
    arch = getattr(target, "arch", None)
    warp_size = getattr(target, "warp_size", None)
    if backend not in ("cuda", "hip"):
        return repr(target)
    spec = f"{backend}:{arch}"
    try:
        default = _parse_target_spec(spec).warp_size
    except ValueError:
        return repr(target)
    return spec if warp_size == default else f"{spec}:{warp_size}"


def resolve_ir_target(requested: Any = None) -> Any:
    """The ``GPUTarget`` for a client's ``ir_target``: ``requested`` when it
    is set, else the configured default (``tilelens.config.ir_target``, from
    ``TILELENS_IR_TARGET``, else ``"cuda:89"``)."""
    if requested is not None:
        return parse_ir_target(requested)
    spec = config_module.config.ir_target
    try:
        return parse_ir_target(spec)
    except ValueError as exc:
        raise ValueError(
            f"tilelens.config.ir_target (TILELENS_IR_TARGET) is {spec!r}, which "
            f"is no IR target: {exc}"
        ) from None


def default_ir_target() -> Any:
    """``GPUTarget("cuda", 89, 32)`` (D26, amended: sm89 is the first
    capability Triton compiles fp8e4nv for)."""
    return parse_ir_target(DEFAULT_IR_TARGET)


def _target_arch(target: Any) -> str | None:
    """``target``'s ``arch`` compile option, as its backend's
    parse_options derives it unless TRITON_OVERRIDE_ARCH says otherwise;
    None for a backend this module does not know."""
    if target.backend == "cuda":
        return f"sm{target.arch}"
    if target.backend == "hip":
        return str(target.arch)
    return None


# ─────────────────── the target Triton's front end sees ───────────────────

_MISSING = object()


class _TargetDriver:
    """``triton.runtime.driver.active`` on a thread while it host-compiles:
    it answers the target query with the compile's target, so Triton's
    front end (``tl.target_info``, its own target checks, a user's
    constexpr function) sees the target the kernel is compiled for, never
    the machine's device. The host has no device, stream or device
    property to give, so anything else is refused, with one exception on a
    release whose ``CompiledKernel.__del__`` unloads its module through the
    driver (``_ReleaseRuntime.unloads_on_del``): see _UnloadOnlyUtils."""

    def __init__(self, target: Any, set_aside: Any = None) -> None:
        self._target = target
        # A context manager factory that sets this thread's scope aside
        # (_ScopedActiveDriver.set_aside), when the driver's ``utils`` are
        # to unload modules (see _UnloadOnlyUtils); None refuses them.
        self._set_aside = set_aside
        # Whether the driver was asked anything (see target_queried): the
        # target, or a question it refuses, which the kernel's code may
        # catch and answer itself (e.g. "no big shared memory"), so a
        # failure after it may be the device's all the same.
        self.queried = False

    def get_current_target(self) -> Any:
        self.queried = True
        return self._target

    @property
    def utils(self) -> Any:
        if self._set_aside is None:
            self._refuse("utils")
        return _UnloadOnlyUtils(self, self._set_aside)

    def _refuse(self, name: str) -> Any:
        self.queried = True
        raise HostCompileUnavailable(
            f"compiling for {format_ir_target(self._target)} on the host, "
            f"Triton asked its driver for {name!r}: a host compile has no "
            "device to ask and answers only the target query"
        )

    def __getattr__(self, name: str) -> Any:
        if name.startswith("__"):
            raise AttributeError(name)
        return self._refuse(name)


class _UnloadOnlyUtils:
    """``driver.active.utils`` on a thread while it host-compiles, on a
    Triton release whose ``CompiledKernel.__del__`` unloads a loaded module
    through it (``_ReleaseRuntime.unloads_on_del``, Triton 3.8): a kernel a
    real launch loaded can be collected on any thread, in the middle of a
    host compile too. ``unload_module`` releases the module through the
    driver the thread has outside its compile, which loaded it; it asks
    nothing about the device, so the compile is not marked as having asked
    (see target_queried). Anything else is refused as any device query."""

    def __init__(self, scoped: _TargetDriver, set_aside: Any) -> None:
        self._scoped = scoped
        self._set_aside = set_aside

    def unload_module(self, module: Any) -> Any:
        from triton.runtime.driver import driver

        with self._set_aside():
            return driver.active.utils.unload_module(module)

    def __getattr__(self, name: str) -> Any:
        if name.startswith("__"):
            raise AttributeError(name)
        return self._scoped._refuse(f"utils.{name}")


class _ScopedActiveDriver:
    """Thread-scoped ``driver.active`` (see _TargetDriver).

    While any thread host-compiles, the DriverConfig class's ``active``
    property is wrapped: a compiling thread gets its _TargetDriver, every
    other thread (and the compiling one outside its compile) whatever
    ``active`` was before, i.e. Triton's own driver or a test's stand-in.
    The last compile to end puts the class attribute back, unless someone
    replaced the wrapper in the meantime. Replacing the process-wide active
    driver instead would hand the target driver to another thread's real
    launch.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._local = threading.local()
        self._depth = 0
        self._owner: Any = None
        self._previous: Any = _MISSING
        self._wrapper: Any = None

    @contextmanager
    def targeting(
        self, config_cls: type, target: Any, *, unloads: bool = False
    ) -> Iterator[_TargetDriver]:
        """Scope ``driver.active`` on this thread to a _TargetDriver for
        ``target``, which is yielded. ``unloads``: its ``utils`` unload
        modules through the driver outside the scope (see
        _UnloadOnlyUtils) instead of being refused."""
        with self._lock:
            if self._depth == 0:
                self._install(config_cls)
            self._depth += 1
        saved = getattr(self._local, "driver", None)
        scoped = self._local.driver = _TargetDriver(
            target, self.set_aside if unloads else None
        )
        try:
            yield scoped
        finally:
            self._local.driver = saved
            with self._lock:
                self._depth -= 1
                if self._depth == 0:
                    self._uninstall()

    @contextmanager
    def set_aside(self) -> Iterator[None]:
        """This thread's scope set aside: ``driver.active`` is what it is
        outside every host compile of the thread."""
        saved = getattr(self._local, "driver", None)
        self._local.driver = None
        try:
            yield
        finally:
            self._local.driver = saved

    def _install(self, config_cls: type) -> None:
        fallback = inspect.getattr_static(config_cls, "active")
        local = self._local

        def active(config: Any) -> Any:
            scoped = getattr(local, "driver", None)
            if scoped is not None:
                return scoped
            return fallback.__get__(config, type(config))

        self._owner = config_cls
        self._previous = config_cls.__dict__.get("active", _MISSING)
        self._wrapper = property(active)
        setattr(config_cls, "active", self._wrapper)

    def _uninstall(self) -> None:
        owner, wrapper = self._owner, self._wrapper
        if owner is not None and owner.__dict__.get("active") is wrapper:
            if self._previous is _MISSING:
                delattr(owner, "active")
            else:
                setattr(owner, "active", self._previous)
        self._owner, self._previous, self._wrapper = None, _MISSING, None


_SCOPED_DRIVER = _ScopedActiveDriver()


# ─────────────────── what Triton releases differ in ───────────────────


@dataclass(frozen=True)
class _ReleaseRuntime:
    """What a Triton minor release's JIT runtime does, where releases
    differ, that the host compile mirrors (``stages_hook_keys``) or answers
    (``unloads_on_del``). Each field is checked against the installed
    Triton's code (_detected_runtime) before it is relied on."""

    # JITFunction.run and triton.compile call
    # knobs.runtime.add_stages_inspection_hook with no arguments for a
    # (key, hash) pair: the JIT appends the string
    # '("custom_pipeline", <hash>)' to the call's specialization (so to its
    # cache key and to the ASTSource's attributes), triton.compile appends
    # <key> to the kernel's cache key (so to its hash). False: only a
    # backend's add_stages calls the hook, as it does in the host compile's
    # own pipeline too.
    stages_hook_keys: bool
    # CompiledKernel.__del__ unloads a loaded module through
    # driver.active.utils.unload_module: a kernel a real launch loaded may be
    # collected while a thread host-compiles (see _UnloadOnlyUtils).
    unloads_on_del: bool


# Keyed by Triton minor release. On a release with no row here, or one whose
# installed code does not do what its row says (a changed private API), the
# runtime is unknown and fails closed: a host compile while
# knobs.runtime.add_stages_inspection_hook is set raises
# HostCompileUnavailable (its kernel's hash might not be the JIT's), and
# ``driver.active.utils`` is refused inside a host compile like any device
# query.
_RELEASE_RUNTIMES: Mapping[str, _ReleaseRuntime] = MappingProxyType(
    {
        "3.6": _ReleaseRuntime(stages_hook_keys=False, unloads_on_del=False),
        "3.8": _ReleaseRuntime(stages_hook_keys=True, unloads_on_del=True),
    }
)
_STAGES_HOOK = "add_stages_inspection_hook"


def _code_names(fn: Any) -> frozenset[str] | None:
    """The names ``fn``'s code (and the code nested in it) reads; None when
    ``fn`` is no Python function."""
    code = getattr(fn, "__code__", None)
    if code is None:
        return None
    names: set[str] = set()
    pending = [code]
    while pending:
        current = pending.pop()
        names.update(current.co_names)
        pending.extend(c for c in current.co_consts if isinstance(c, CodeType))
    return frozenset(names)


def _detected_runtime(
    compile_fn: Any, jit_function: type, compiled_kernel: type
) -> dict[str, bool | None]:
    """What the installed Triton's code does of each _ReleaseRuntime field:
    True or False, or None where it cannot tell (e.g. only one of
    triton.compile and JITFunction.run reads the stages-inspection hook)."""
    readers = [
        _code_names(compile_fn),
        _code_names(inspect.getattr_static(jit_function, "run", None)),
    ]
    reads_hook = {names is not None and _STAGES_HOOK in names for names in readers}
    finalizer = _code_names(inspect.getattr_static(compiled_kernel, "__del__", None))
    return {
        "stages_hook_keys": reads_hook.pop() if len(reads_hook) == 1 else None,
        "unloads_on_del": finalizer is not None and "unload_module" in finalizer,
    }


def _release_runtime(
    version: str, detected: Mapping[str, bool | None]
) -> tuple[_ReleaseRuntime | None, str]:
    """The installed release's row of _RELEASE_RUNTIMES when its code does
    what the row says, else None; with why not, for the error that names
    it."""
    release = ".".join(version.split(".")[:2])
    row = _RELEASE_RUNTIMES.get(release)
    if row is None:
        return None, (
            f"Triton {release} has no row in "
            "tilelens.core.host_compile._RELEASE_RUNTIMES"
        )
    differ = sorted(
        name for name, value in detected.items() if getattr(row, name) != value
    )
    if differ:
        found = ", ".join(f"{name}={detected[name]}" for name in differ)
        return None, (
            f"the installed Triton {version} does not do what the Triton "
            f"{release} row of tilelens.core.host_compile._RELEASE_RUNTIMES "
            f"says (its code shows {found})"
        )
    return row, ""


# ─────────────────────────── Triton's API ───────────────────────────


def _unavailable(version: str, what: str) -> HostCompileUnavailable:
    return HostCompileUnavailable(
        f"IR mode compiles kernels on the host with Triton's private compile "
        f"API, and on Triton {version} {what} (IR mode is tested on the "
        "releases in tilelens.core.config.TESTED_TRITON_VERSIONS)"
    )


@functools.lru_cache(maxsize=1)
def triton_api() -> SimpleNamespace:
    """The Triton internals the host compile uses, checked for presence,
    and the front end's target queries checked to answer a scoped target
    (see _TargetDriver). Raises HostCompileUnavailable naming what is
    missing or does not behave so; a failure is not cached."""
    import triton

    def missing(what: str) -> HostCompileUnavailable:
        return _unavailable(triton.__version__, f"it lacks {what}")

    try:
        from triton import knobs
        from triton._C.libtriton import get_cache_invalidating_env_vars, ir
        from triton.backends.compiler import GPUTarget, Language
        from triton.compiler import ASTSource, compile, get_cache_key, make_backend
        from triton.compiler.compiler import CompiledKernel
        from triton.runtime.driver import driver
        from triton.runtime.jit import (
            JITFunction,
            compute_cache_key,
            create_function_from_signature,
        )
    except ImportError as exc:
        raise missing(str(exc)) from exc
    for owner, name, attr in (
        (ir, "triton._C.libtriton.ir", "context"),
        (ir, "triton._C.libtriton.ir", "load_dialects"),
        (ASTSource, "ASTSource", "make_ir"),
        (knobs.runtime, "knobs.runtime", "debug"),
        (knobs.compilation, "knobs.compilation", "instrumentation_mode"),
        (Language, "triton.backends.compiler.Language", "TRITON"),
    ):
        if not hasattr(owner, attr):
            raise missing(f"{name}.{attr}")
    if not isinstance(inspect.getattr_static(type(driver), "active", None), property):
        raise missing(
            "triton.runtime.driver.driver.active as a property of its class, "
            "which the host compile scopes to answer the target query"
        )
    try:
        from triton.compiler.compiler import filter_traceback
    except ImportError:  # only trims a front-end error's traceback

        def filter_traceback(e: BaseException) -> None:  # type: ignore[misc]
            pass

    runtime, runtime_unknown = _release_runtime(
        triton.__version__, _detected_runtime(compile, JITFunction, CompiledKernel)
    )
    unloads = runtime is not None and runtime.unloads_on_del
    for target in (GPUTarget("cuda", 80, 32), GPUTarget("cuda", 90, 32)):
        try:
            with _SCOPED_DRIVER.targeting(type(driver), target, unloads=unloads):
                wrong = _unscoped_target_queries(target)
        except Exception as exc:
            raise _unavailable(
                triton.__version__,
                "its front end's target queries could not be asked "
                f"({type(exc).__name__}: {exc})",
            ) from exc
        if wrong:
            raise _unavailable(
                triton.__version__,
                f"its front end's target queries answer {wrong} while compiling "
                f"for {format_ir_target(target)} on the host",
            )
    return SimpleNamespace(
        version=triton.__version__,
        knobs=knobs,
        ir=ir,
        get_cache_invalidating_env_vars=get_cache_invalidating_env_vars,
        GPUTarget=GPUTarget,
        Language=Language,
        ASTSource=ASTSource,
        compile=compile,
        get_cache_key=get_cache_key,
        make_backend=make_backend,
        compute_cache_key=compute_cache_key,
        create_function_from_signature=create_function_from_signature,
        filter_traceback=filter_traceback,
        driver_config=type(driver),
        JITFunction=JITFunction,
        # The installed release's _RELEASE_RUNTIMES row, None when unknown
        # (then runtime_unknown says why).
        runtime=runtime,
        runtime_unknown=runtime_unknown,
        unloads=unloads,
    )


def _unscoped_target_queries(target: Any) -> dict[str, Any]:
    """The front end's target queries (where the installed Triton has them)
    that do not answer ``target`` under its scope, with what they answer."""
    wrong: dict[str, Any] = {}
    try:
        from triton.language import target_info
    except ImportError:
        target_info = None
    current_target = getattr(target_info, "current_target", None)
    if current_target is not None and (got := current_target()) != target:
        wrong["tl.target_info.current_target()"] = got
    try:
        from triton.language.semantic import TritonSemantic
    except ImportError:
        TritonSemantic = None
    has_native_tma = getattr(TritonSemantic, "_has_native_tma", None)
    if has_native_tma is not None:
        # It reads nothing of the semantic, only the driver's target.
        native = has_native_tma(None)
        if native != (target.backend == "cuda" and target.arch >= 90):
            wrong["TritonSemantic._has_native_tma()"] = native
    return wrong


# Host-compiled before the first compile for each target: scalars only (it
# needs no tensor, and no name from triton.language); ``one`` takes the
# equal-to-1 constexpr specialization. Its source is registered with
# linecache under a name of its own, so the JIT reads it from there and not
# from this file, which may have changed on disk since it was imported.
_SELF_TEST_SOURCE = """\
def _self_test_kernel(n, flag, one):
    if flag:
        n = n * one
"""
_SELF_TEST_FILE = "<tilelens host-compile self-test>"


def _self_test_jit_function(api: SimpleNamespace) -> Any:
    lines = _SELF_TEST_SOURCE.splitlines(keepends=True)
    linecache.cache[_SELF_TEST_FILE] = (
        len(_SELF_TEST_SOURCE),
        None,
        lines,
        _SELF_TEST_FILE,
    )
    namespace: dict[str, Any] = {"__name__": __name__}
    exec(compile(_SELF_TEST_SOURCE, _SELF_TEST_FILE, "exec"), namespace)
    return api.JITFunction(namespace["_self_test_kernel"])


@functools.lru_cache(maxsize=None)
def _self_test_target(target: Any) -> None:
    """Host-compile the built-in _self_test_kernel for ``target`` through its TTIR;
    raise HostCompileUnavailable if that fails. Only a success is cached."""
    api = triton_api()
    try:
        kernel = HostCompiler().compile(
            _self_test_jit_function(api),
            (5, True, 1),
            {},
            target=target,
            stages={"ttir"},
            _self_test=True,
        )
        text = kernel.asm["ttir"]
    except Exception as exc:
        raise _unavailable(
            api.version,
            f"a built-in test kernel failed to host-compile for "
            f"{format_ir_target(target)} ({type(exc).__name__}: {exc})",
        ) from exc
    if "tt.func" not in text or "_self_test_kernel" not in text:
        raise _unavailable(
            api.version,
            f"a built-in test kernel host-compiled for {format_ir_target(target)} "
            "to no TTIR function",
        )


# ─────────────────────────── the artifact ───────────────────────────


@dataclass(frozen=True, eq=False)
class HostKernel:
    """A kernel compiled on the host up to a stage (see the module
    docstring): what a ``CompiledKernel`` holds of it, never loaded."""

    # The specialization: what triton.compile names this kernel.
    hash: str
    name: str
    # Stage -> text (bytes for a binary stage), every stage compiled, in
    # pipeline order, "source" first when asked for.
    asm: Mapping[str, str | bytes] = field(repr=False)
    # A namedtuple, as CompiledKernel.metadata: "target", "name", "hash",
    # the compile options and whatever the compiled stages added.
    metadata: Any = field(repr=False)

    @property
    def target(self) -> Any:
        return self.metadata.target


# Stages a compiled kernel holds besides its backend's pipeline: the front
# end's own module (what triton.compile keeps as "source"), and the CUDA
# binary's disassembly (CompiledKernel.asm derives "sass" from "cubin").
_SOURCE_STAGE = "source"
_DERIVED_STAGES = {"sass": "cubin"}


def _full_pipeline_forced(api: SimpleNamespace, options: Any) -> bool:
    # Knobs that make triton.compile rewrite or dump stages, which only it
    # implements.
    compilation = api.knobs.compilation
    return bool(
        getattr(compilation, "override", False)
        or getattr(compilation, "dump_ir", False)
        or getattr(compilation, "use_ir_loc", None)
        or getattr(options, "ir_override", None)
    )


def _keying_stages_hook(api: SimpleNamespace) -> Any:
    """``knobs.runtime.add_stages_inspection_hook`` where the installed
    release keys kernels by it (``_ReleaseRuntime.stages_hook_keys``); None
    where no hook is set or the release does not (its backends'
    ``add_stages`` still call it, in a host compile's pipeline too). Raises
    HostCompileUnavailable for a hook set on a release whose runtime is
    unknown: the host compile's kernel might not be the one the JIT names."""
    hook = getattr(api.knobs.runtime, _STAGES_HOOK, None)
    if hook is None:
        return None
    if api.runtime is None:
        raise HostCompileUnavailable(
            f"knobs.runtime.{_STAGES_HOOK} is set, and how Triton "
            f"{api.version}'s JIT keys a kernel by it is not known "
            f"({api.runtime_unknown}), so a host compile would not name its "
            "kernel as the JIT does"
        )
    return hook if api.runtime.stages_hook_keys else None


def _check_used_globals(jit_fn: Any) -> None:
    # JITFunction.run's check, for every kernel handed out: a kernel
    # compiled before a global it reads changed is stale.
    not_present = object()
    for (name, _), (value, globals_dict) in jit_fn.used_global_vals.items():
        if (new := globals_dict.get(name, not_present)) != value:
            raise RuntimeError(
                f"Global variable {name} has changed since we compiled this "
                f"kernel, from {value} to {new}"
            )


class HostCompiler:
    """Host compiles with an in-process cache (one per trace: the
    ClientManager's ``compiler``), keyed by the JIT's own specialization
    key (``compute_cache_key``: the bound specialization and the call's
    compile options), the target and the requested stages."""

    def __init__(self) -> None:
        # (id(jit_fn), target) -> (jit_fn, backend, binder, key cache).
        self._binders: dict[Hashable, tuple[Any, Any, Any, dict]] = {}
        # (id(jit_fn), target) -> jit_fn, for each kernel a compile of which
        # for the target asked the driver (see target_queried).
        self._asked: dict[Hashable, Any] = {}
        # (id(jit_fn), specialization key, target, stages) -> (jit_fn, kernel).
        self._kernels: dict[Hashable, tuple[Any, Any]] = {}
        # target -> the stages a kernel compiled for it can hold.
        self._stage_names: dict[Hashable, frozenset[str]] = {}

    def check_stages(self, target: Any, stages: Iterable[str]) -> None:
        """Raise ValueError naming each of ``stages`` no kernel compiled for
        ``target`` holds: a stage of the backend's pipeline (as its
        ``add_stages`` builds it for a Triton kernel), ``"source"``, or one
        derived from a pipeline stage (``"sass"``) is fine. Raises
        HostCompileUnavailable if the target's stages cannot be listed (as
        its compiles would)."""
        known = self._stage_names_of(target)
        unknown = set(stages) - known
        if unknown:
            raise ValueError(
                f"IR stages {sorted(unknown)} are no stage of a kernel compiled "
                f"for {format_ir_target(target)}, which holds "
                f"{', '.join(sorted(known))}"
            )

    def _stage_names_of(self, target: Any) -> frozenset[str]:
        names = self._stage_names.get(target)
        if names is None:
            api = triton_api()
            arch = _target_arch(target)
            pipeline: dict[str, Any] = {}
            try:
                backend = api.make_backend(target)
                with _SCOPED_DRIVER.targeting(
                    api.driver_config, target, unloads=api.unloads
                ):
                    options = backend.parse_options(
                        {} if arch is None else {"arch": arch}
                    )
                    backend.add_stages(pipeline, options, api.Language.TRITON)
            except Exception as exc:
                raise _unavailable(
                    api.version,
                    f"the stages of a kernel compiled for {format_ir_target(target)} "
                    f"cannot be listed ({type(exc).__name__}: {exc})",
                ) from exc
            derived = {
                name for name, base in _DERIVED_STAGES.items() if base in pipeline
            }
            names = self._stage_names[target] = frozenset(
                {*pipeline, _SOURCE_STAGE, *derived}
            )
        return names

    def compile(
        self,
        jit_fn: Any,
        args: tuple,
        kwargs: Mapping[str, Any],
        *,
        target: Any,
        stages: Iterable[str] = (),
        _self_test: bool = False,
    ) -> Any:
        """Compile the call ``jit_fn.run(*args, **kwargs)`` would compile on
        a device of ``target`` (a GPUTarget), through the latest of
        ``stages`` (nothing requested: the first stage). Raises what the
        JIT's bind, pack or compile raises (a bind failure marked as such,
        see bind_failed; any other error marked when this compile, or an
        earlier one of ``jit_fn`` for ``target``, had asked the driver, see
        target_queried), or HostCompileUnavailable. (``_self_test``: the
        built-in test compile, which always stops at the requested stage.)"""
        api = triton_api()
        for attr in ("signature", "params", "_pack_args", "used_global_vals"):
            if not hasattr(jit_fn, attr):
                raise HostCompileUnavailable(
                    f"cannot host-compile {jit_fn!r}: it has no {attr!r} "
                    f"(a JITFunction of Triton {api.version} has)"
                )
        if not _self_test:
            _self_test_target(target)
        asked_key = (id(jit_fn), target)
        with _SCOPED_DRIVER.targeting(
            api.driver_config, target, unloads=api.unloads
        ) as scoped:
            try:
                return self._compile(
                    api, jit_fn, args, kwargs, target, frozenset(stages), _self_test
                )
            except Exception as exc:
                asked_before = self._asked.get(asked_key) is jit_fn
                if not bind_failed(exc) and (scoped.queried or asked_before):
                    _mark_target_queried(exc)
                raise
            finally:
                if scoped.queried:
                    self._asked[asked_key] = jit_fn

    def _compile(
        self,
        api: SimpleNamespace,
        jit_fn: Any,
        args: tuple,
        kwargs: Mapping[str, Any],
        target: Any,
        stages: frozenset[str],
        truncate: bool,
    ) -> Any:
        backend, binder, key_cache = self._binder(api, jit_fn, target)
        # What JITFunction.run adds to every call's options.
        kwargs = dict(kwargs)
        kwargs["debug"] = (
            kwargs.get("debug", getattr(jit_fn, "debug", None))
            or api.knobs.runtime.debug
        )
        kwargs["instrumentation_mode"] = api.knobs.compilation.instrumentation_mode
        # The target's arch as a compile option, which the backend's
        # parse_options takes over TRITON_OVERRIDE_ARCH: a host compile is
        # for the target it was asked for (D26). Not where "arch" is the
        # call's own (a launch option, or a kernel parameter).
        arch = _target_arch(target)
        if (
            arch is not None
            and "arch" not in kwargs
            and all(param.name != "arch" for param in jit_fn.params)
        ):
            kwargs["arch"] = arch
        try:
            bound_args, specialization, options = binder(*args, **kwargs)
        except Exception as exc:
            # The call's own error (see bind_failed); the backend only adds
            # its tensor-alignment flags to the specialization.
            _mark_call_error(exc)
            raise
        stages_hook = _keying_stages_hook(api)
        if stages_hook is not None:
            # JITFunction.run's field for a custom pipeline, as it spells it.
            _, pipeline_hash = stages_hook()
            specialization.append(f'("custom_pipeline", {pipeline_hash})')
        try:
            cache_key = api.compute_cache_key(key_cache, specialization, options)
        except Exception as exc:
            # The call's own error too (e.g. an unhashable constexpr value):
            # the key is the bound specialization and the call's options,
            # which JITFunction.run keys the call by right after its binder,
            # on any device.
            _mark_call_error(exc)
            raise
        key = (id(jit_fn), cache_key, target, stages)
        cached = self._kernels.get(key)
        if cached is not None and cached[0] is jit_fn:
            kernel = cached[1]
        else:
            try:
                options, signature, constexprs, attrs = jit_fn._pack_args(
                    backend, kwargs, bound_args, specialization, options
                )
            except KeyError as exc:
                _mark_unknown_options(exc, jit_fn, backend, kwargs)
                raise
            compiled_arch = getattr(options, "arch", arch)
            if compiled_arch != arch:
                raise HostCompileUnavailable(
                    f"the call's compile options name arch {compiled_arch!r}, "
                    f"not {arch!r} of the IR target {format_ir_target(target)} "
                    "(an 'arch' launch option, or TRITON_OVERRIDE_ARCH with a "
                    "kernel parameter named 'arch'), so its host compile would "
                    "not be for the target"
                )
            source = api.ASTSource(jit_fn, signature, constexprs, attrs)
            kernel = self._compile_source(
                api, source, backend, target, options, stages, truncate, stages_hook
            )
            self._kernels[key] = (jit_fn, kernel)
        _check_used_globals(jit_fn)
        return kernel

    def _binder(self, api: SimpleNamespace, jit_fn: Any, target: Any) -> tuple:
        entry = self._binders.get((id(jit_fn), target))
        if entry is None or entry[0] is not jit_fn:
            backend = api.make_backend(target)
            binder = api.create_function_from_signature(
                jit_fn.signature, jit_fn.params, backend
            )
            entry = self._binders[(id(jit_fn), target)] = (jit_fn, backend, binder, {})
        return entry[1:]

    @staticmethod
    def _compile_source(
        api: SimpleNamespace,
        source: Any,
        backend: Any,
        target: Any,
        options: Any,
        stages: frozenset[str],
        truncate: bool,
        stages_hook: Any = None,
    ) -> Any:
        pipeline: dict[str, Any] = {}
        backend.add_stages(pipeline, options, source.language)
        names = list(pipeline)
        # Nothing requested: the first stage; "source" alone: no pass.
        wanted = stages - {_SOURCE_STAGE} if stages else frozenset(names[:1])
        full = not truncate and (
            not wanted <= set(names)  # "sass"
            or max(map(names.index, wanted), default=-1) == len(names) - 1
            or _full_pipeline_forced(api, options)
        )
        if full:
            return api.compile(source, target=target, options=options.__dict__)
        last = max(map(names.index, wanted), default=-1)
        # triton.compile's front half, stopped after ``names[last]``.
        env_vars = api.get_cache_invalidating_env_vars()
        key = api.get_cache_key(source, backend, options, env_vars)
        if stages_hook is not None:
            # What triton.compile appends for a custom pipeline (it asks the
            # hook again, as here).
            key += stages_hook()[0]
        digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
        metadata = {
            "hash": digest,
            "target": target,
            **options.__dict__,
            **env_vars,
            "triton_version": api.version,
        }
        # Keep the context referenced until every module of it is gone.
        context = api.ir.context()
        api.ir.load_dialects(context)
        backend.load_dialects(context)
        codegen_fns = backend.get_codegen_implementation(options)
        module_map = backend.get_module_map()
        try:
            module = source.make_ir(target, options, codegen_fns, module_map, context)
        except Exception as exc:
            api.filter_traceback(exc)
            raise
        asm: dict[str, str | bytes] = {}
        if _SOURCE_STAGE in stages:
            asm[_SOURCE_STAGE] = str(module)
        for name in names[: last + 1]:
            module = pipeline[name](module, metadata)
            asm[name] = module if isinstance(module, (str, bytes)) else str(module)
        del module
        # A later stage names the entry point; up to here it is the kernel's.
        metadata.setdefault("name", source.name)
        kernel_metadata = namedtuple(  # type: ignore[misc]
            "KernelMetadata", sorted(metadata)
        )(**metadata)
        del context
        return HostKernel(
            hash=digest,
            name=metadata["name"],
            asm=MappingProxyType(asm),
            metadata=kernel_metadata,
        )
