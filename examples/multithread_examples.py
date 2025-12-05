import argparse
import time
import triton
import triton.language as tl
import torch


def make_kernels(trace_decorator=None):
    def wrap(fn):
        fn = triton.jit(fn)
        if trace_decorator:
            fn = trace_decorator(fn)
        return fn

    @wrap
    def producer_consumer(x, out, N: tl.constexpr):
        pid = tl.program_id(0)
        if pid == 0:
            counter = 0
            # Wait until PID 1 finishes, then write a marker.
            max_spin = 10_000
            while tl.atomic_cas(out, 0.0, 0.0) == 0.0 and counter < max_spin:
                counter += 1
            if counter >= max_spin:
                # Prevent infinite hangs when concurrency is missing.
                tl.store(out, -2.0)
                return
            tl.store(out, counter)
        elif pid == 1:
            arange = tl.arange(0, 128)
            s = 0.0
            for i in range(N // 128):
                s += tl.sum(tl.load(x + i * 128 + arange))
            tl.atomic_add(out, s + 1)

    @wrap
    def racing_threads(x, out, N: tl.constexpr, SLOW_ITERS: tl.constexpr):
        pid = tl.program_id(0)

        if pid == 0:
            arange = tl.arange(0, 32)
            s = 0.0
            for i in range(SLOW_ITERS):
                s += tl.sum(tl.load(x + i * 32 + arange))
            tl.atomic_cas(out, 0.0, SLOW_ITERS + s)
        elif pid == 1:
            arange = tl.arange(0, 32)
            s = 0.0
            for i in range(1):
                s += tl.sum(tl.load(x + i * 32 + arange))
            tl.atomic_cas(out, 0.0, 1 + s)

    return producer_consumer, racing_threads


def main():
    parser = argparse.ArgumentParser(
        description="Producer/consumer and racing-thread examples."
    )
    parser.add_argument(
        "--viz",
        action="store_true",
        help="Enable triton-viz tracing (uses tracer client).",
    )
    parser.add_argument(
        "--num-sms",
        type=int,
        default=2,
        help="Concurrent blocks to emulate in the Triton interpreter (when --viz is enabled).",
    )
    args = parser.parse_args()
    print(args)

    trace_decorator = None
    if args.viz:
        import triton_viz

        triton_viz.config.num_sms = max(1, args.num_sms)
        trace_decorator = triton_viz.trace("tracer")

    producer_consumer, racing_threads = make_kernels(trace_decorator)

    device = (
        "cpu" if args.viz else "cuda"
    )  # triton-viz only supports CPU, triton only supports CUDA (TRITON_INTERPRET=1 runs on CPU but hangs)

    # Use smaller workloads on CPU to keep interpreter runs snappy.
    pc_N = 16384 * 16 if device == "cuda" else 2048
    slow_iters = 16384 if device == "cuda" else 256
    rt_N = slow_iters * 32

    x = torch.zeros((pc_N,), device=device)
    out = torch.zeros((), device=device)
    pc_grid = producer_consumer[(2, 1, 1)]
    print("x.shape:", x.shape)
    tic = time.perf_counter()
    pc_grid(x, out, pc_N)
    toc = time.perf_counter()
    print("duration:", toc - tic)
    print("producer_consumer output:", out.item())  # should not be -1

    x = torch.zeros((rt_N,), device=device)
    out = torch.zeros((), device=device)
    racing_threads[(2, 1, 1)](x, out, rt_N, slow_iters)
    print(
        "racing_threads output:", out.item()
    )  # should be 1 as this is returned in the branch that finishes first


if __name__ == "__main__":
    main()
