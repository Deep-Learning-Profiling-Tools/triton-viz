"""Conditional values cannot borrow an inactive arm's positional anchor."""

from types import SimpleNamespace

import pytest

from triton_viz.clients.race_detector.race_detector import SymbolicRaceDetector


def _node(op, **children):
    return SimpleNamespace(op=op, children=children)


def _detector(*loads):
    detector = object.__new__(SymbolicRaceDetector)
    detector._dep_anchor_ids = {id(load): i for i, load in enumerate(loads)}
    return detector


@pytest.mark.parametrize("other", ["trans", "const"])
@pytest.mark.parametrize("direct_first", [False, True])
def test_where_requires_positional_path_in_both_value_arms(other, direct_first):
    value = _node("load")
    alternative = _node(other, operand=value) if other == "trans" else _node("const")
    lhs, rhs = (value, alternative) if direct_first else (alternative, value)
    selected = _node("where", cond=_node("const"), lhs=lhs, rhs=rhs)
    assert _detector(value)._dep_loads_of(_node("store", value=selected)) == ()


def test_where_retains_condition_and_only_the_common_value_dependencies():
    cond, common, left_only, right_only = (_node("load") for _ in range(4))
    selected = _node(
        "where",
        cond=cond,
        lhs=_node("add", lhs=common, rhs=left_only),
        rhs=_node("sub", lhs=common, rhs=right_only),
    )
    assert _detector(cond, common, left_only, right_only)._dep_loads_of(
        _node("store", value=selected)
    ) == (0, 1)


def test_loaded_condition_keeps_its_positional_path_with_permuted_value_arm():
    value = _node("load")
    selected = _node(
        "where",
        cond=_node("greater", lhs=value, rhs=_node("const")),
        lhs=_node("trans", operand=value),
        rhs=_node("const"),
    )
    assert _detector(value)._dep_loads_of(_node("store", value=selected)) == (0,)


def test_simultaneous_direct_path_survives_a_conditional_or_permuted_path():
    value = _node("load")
    selected = _node("where", cond=_node("const"), lhs=value, rhs=_node("const"))
    output = _node(
        "add",
        lhs=value,
        rhs=_node("add", lhs=selected, rhs=_node("trans", operand=value)),
    )
    assert _detector(value)._dep_loads_of(_node("store", value=output)) == (0,)


def test_store_pointer_is_not_promoted_into_a_value_dependency():
    value = _node("load")
    assert (
        _detector(value)._dep_loads_of(_node("store", ptr=value, value=_node("const")))
        == ()
    )


def test_unknown_where_shape_fails_closed():
    value = _node("load")
    selected = _node("where", condition=value, true_value=value, false_value=value)
    assert _detector(value)._dep_loads_of(_node("store", value=selected)) == ()


def test_shared_deep_dag_does_not_require_python_recursion():
    value = _node("load")
    output = value
    for _ in range(2000):
        output = _node("add", lhs=output, rhs=value)
    assert _detector(value)._dep_loads_of(_node("store", value=output)) == (0,)


@pytest.mark.parametrize("entry", ["create_select", "ternary_op"])
@pytest.mark.parametrize("keywords", [False, True])
def test_enum_select_delegation_preserves_taint_but_intersects_positional_arms(
    entry, keywords
):
    import numpy as np
    import triton.language as tl
    from triton.runtime.interpreter import TensorHandle, interpreter_builder
    from triton_viz.clients.race_detector.concrete_enum import (
        ConcreteFootprintRecorder,
        _DEP_ATTR,
        _set_positional_deps,
        _tag,
        _taint_of,
    )

    cond = TensorHandle(np.array([False, True]), tl.int1)
    lhs = TensorHandle(np.array([1, 2], dtype=np.int32), tl.int32)
    rhs = TensorHandle(np.array([3, 4], dtype=np.int32), tl.int32)
    for value, anchors in ((cond, {3}), (lhs, {0, 1}), (rhs, {0, 2})):
        _tag(value, frozenset(anchors))
        _set_positional_deps(value, frozenset(anchors))
    recorder = ConcreteFootprintRecorder(fence_order=True)
    try:
        recorder._wrap_attr(interpreter_builder, "ternary_op", None)
        recorder._wrap_attr(interpreter_builder, "create_select", None)
        if entry == "create_select":
            if keywords:
                result = interpreter_builder.create_select(cond=cond, lhs=lhs, rhs=rhs)
            else:
                result = interpreter_builder.create_select(cond, lhs, rhs)
        elif keywords:
            result = interpreter_builder.ternary_op(
                lhs=cond, rhs=lhs, other=rhs, op=np.where
            )
        else:
            result = interpreter_builder.ternary_op(cond, lhs, rhs, np.where)
        assert result.data.tolist() == [3, 2]
        assert result.attr[_DEP_ATTR] == frozenset((0, 3))
        assert _taint_of(result) == frozenset((0, 1, 2, 3))
    finally:
        recorder.cleanup()


def test_enum_generic_ternary_callable_cannot_invent_a_dependency():
    import numpy as np
    import triton.language as tl
    from triton.runtime.interpreter import TensorHandle, interpreter_builder
    from triton_viz.clients.race_detector.concrete_enum import (
        ConcreteFootprintRecorder,
        _DEP_ATTR,
        _set_positional_deps,
        _tag,
        _taint_of,
    )

    handles = [TensorHandle(np.array([n], dtype=np.int32), tl.int32) for n in range(3)]
    for index, handle in enumerate(handles):
        _tag(handle, frozenset((index,)))
        _set_positional_deps(handle, frozenset((index,)))
    recorder = ConcreteFootprintRecorder(fence_order=True)
    try:
        recorder._wrap_attr(interpreter_builder, "ternary_op", None)
        output = interpreter_builder.ternary_op(
            *handles, lambda cond, lhs, rhs: np.where(cond, lhs, rhs)
        )
        assert output.attr[_DEP_ATTR] == frozenset()
        assert _taint_of(output) == frozenset((0, 1, 2))
        clamp = interpreter_builder.ternary_op(*handles, np.clip)
        assert clamp.attr[_DEP_ATTR] == frozenset((0, 1, 2))
    finally:
        recorder.cleanup()
