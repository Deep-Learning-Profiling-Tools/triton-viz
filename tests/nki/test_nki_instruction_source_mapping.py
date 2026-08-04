import json


def test_source_region_id_is_stable_and_separates_regions():
    from triton_viz.tools.nki_trace_dump import _annotate_fusion_signature
    events = [
        {"op": "compute", "api_op": "add", "grid_idx": [0], "output_shape": [128, 64], "output_dtype": "float32"},
        {"op": "load", "grid_idx": [0]},
        {"op": "compute", "api_op": "multiply", "grid_idx": [0], "output_shape": [128, 64], "output_dtype": "float32"},
    ]
    copy = json.loads(json.dumps(events))
    _annotate_fusion_signature(events); _annotate_fusion_signature(copy)
    assert events[0]["source_region_id"] == copy[0]["source_region_id"]
    assert events[0]["source_region_id"] != events[2]["source_region_id"]


def test_parse_penguin_and_transfer_bounded_region_assignment(tmp_path):
    from triton_viz.tools.nki_instruction_source_mapping import assign_penguin_regions, parse_penguin
    penguin = tmp_path / "penguin.py"
    penguin.write_text('\n'.join([
        'v1 = m8.SBAtomLoad(parent=v1, id=1, dl=m2.DebugLocation(tensor_op_name="x", kernel="k"))',
        'v2 = m8.TensorTensorOp(op=m11.ALUOpcode(op=np.multiply), parent=v1, id=2, dl=m2.DebugLocation(tensor_op_name="sq", kernel="k"))',
        'v3 = m8.TensorReduceOp(op=m11.ALUOpcode(op=np.add), parent=v1, id=3, dl=m2.DebugLocation(tensor_op_name="sum", kernel="k"))',
        'v4 = m8.SBAtomLoad(parent=v1, id=4, dl=m2.DebugLocation(tensor_op_name="w", kernel="k"))',
        'v5 = m8.TensorTensorOp(op=m11.ALUOpcode(op=np.multiply), parent=v1, id=5, dl=m2.DebugLocation(tensor_op_name="out", kernel="k"))',
    ]), encoding="utf-8")
    ops = parse_penguin(penguin)
    assert assign_penguin_regions(ops, [{"tokens": ["multiply", "reduce_sum"]}, {"tokens": ["multiply"]}]) == {2: 0, 3: 0, 5: 1}
