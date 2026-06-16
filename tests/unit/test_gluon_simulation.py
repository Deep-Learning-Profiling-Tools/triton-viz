import numpy as np

from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language import _core as gluon_core
from triton.experimental.gluon.language import _layouts as layouts

from triton_viz.core.simulation import gluon


def _dummy_kernel():
    pass


def test_gluon_simulation_uses_dedicated_builder():
    assert hasattr(gluon.gluon_builder, "create_local_scatter")
    assert not hasattr(gluon.interpreter_builder, "create_local_scatter")


def test_gluon_patch_lang_restores_module_and_class_builtins():
    original_full = gl.full
    original_scatter = gluon_core.shared_memory_descriptor.scatter

    scope = gluon.patch_lang(_dummy_kernel)
    try:
        assert gl.full is not original_full
        assert gluon_core.shared_memory_descriptor.scatter is not original_scatter
    finally:
        scope.restore()

    assert gl.full is original_full
    assert gluon_core.shared_memory_descriptor.scatter is original_scatter


def test_gluon_clc_result_uses_interpreter_tensors():
    result = gluon.CLCResult()

    assert result.is_canceled().handle.data.tolist() == [False]
    assert result.program_id(0).handle.data.tolist() == [0]


def test_gluon_shared_scatter_routes_through_builder():
    scope = gluon.patch_lang(_dummy_kernel)
    calls = []
    builder_cls = gluon.gluon_builder.__class__
    original_scatter = builder_cls.create_local_scatter

    def create_local_scatter(builder, mem_desc, values, indices, axis):
        calls.append(axis)
        return original_scatter(builder, mem_desc, values, indices, axis)

    try:
        scope.set_attr(builder_cls, "create_local_scatter", create_local_scatter)

        shared_layout = layouts.SharedLinearLayout([[1, 0], [0, 1]])
        smem = gl.allocate_shared_memory(gl.int32, [2, 4], shared_layout)
        smem.store(gl.full([2, 4], 0, gl.int32, layouts.AutoLayout()))

        values = gl.full([2, 4], 7, gl.int32, layouts.AutoLayout())
        indices = gl.full([2, 4], 1, gl.int32, layouts.AutoLayout())
        smem.scatter(values, indices, 1)

        loaded = smem.load(layouts.AutoLayout())
        np.testing.assert_array_equal(
            loaded.handle.data,
            np.array([[0, 7, 0, 0], [0, 7, 0, 0]], dtype=np.int32),
        )
        assert calls == [1]
    finally:
        scope.restore()
