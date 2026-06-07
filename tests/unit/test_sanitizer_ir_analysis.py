import triton
import triton.language as tl

from triton_viz.clients.sanitizer.ir_analysis import summarize_loop_structure


@triton.jit
def _jit_loop_kernel(x):
    for i in tl.range(0, 4):
        x += i


def _wrapped_loop_kernel(x):
    for i in range(2):
        x += i


class _TraceLikeKernel:
    base_fn = staticmethod(_wrapped_loop_kernel)


def test_summarize_constant_loop_ranges():
    summaries = summarize_loop_structure(
        """
        def kernel(x):
            for i in range(4):
                x += i
            for j in tl.range(2, 10, 2):
                x += j
            for k in tl.static_range(start=8, end=0, step=-2):
                x += k
        """
    )

    assert [
        (summary.target, summary.range_type, summary.start, summary.stop, summary.step)
        for summary in summaries
    ] == [
        ("i", "python_range", 0, 4, 1),
        ("j", "tl_range", 2, 10, 2),
        ("k", "tl_static_range", 8, 0, -2),
    ]
    assert [summary.trip_count for summary in summaries] == [4, 4, 4]


def test_summarize_symbolic_loop_bounds_without_condensing():
    summaries = summarize_loop_structure(
        """
        def kernel(x, n, block):
            for i in tl.range(0, n, block):
                x += i
        """
    )

    assert len(summaries) == 1
    summary = summaries[0]
    assert summary.range_type == "tl_range"
    assert summary.start == 0
    assert summary.stop is None
    assert summary.step is None
    assert not summary.is_constant_range
    assert summary.trip_count is None


def test_summarize_triton_jit_function():
    summaries = summarize_loop_structure(_jit_loop_kernel)

    assert len(summaries) == 1
    summary = summaries[0]
    assert summary.target == "i"
    assert summary.range_type == "tl_range"
    assert summary.start == 0
    assert summary.stop == 4
    assert summary.step == 1


def test_summarize_trace_like_kernel_wrapper():
    summaries = summarize_loop_structure(_TraceLikeKernel())

    assert len(summaries) == 1
    summary = summaries[0]
    assert summary.target == "i"
    assert summary.range_type == "python_range"
    assert summary.start == 0
    assert summary.stop == 2
    assert summary.step == 1


def test_summarize_unknown_iterable_as_non_constant():
    summaries = summarize_loop_structure(
        """
        def kernel(values):
            for value in values:
                pass
        """
    )

    assert len(summaries) == 1
    summary = summaries[0]
    assert summary.target == "value"
    assert summary.range_type == "unknown"
    assert not summary.is_constant_range
