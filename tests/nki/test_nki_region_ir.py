from triton_viz.tools.nki_region_ir import build_region_ir, compositional_features, structural_family


def test_region_ir_encodes_structure_dag_shape_dtype_and_tail():
    members = [
        {"op": "compute", "api_op": "multiply", "input_ptrs": [1, 2], "output_ptr": 3,
         "input_shape": [128, 1000], "output_shape": [128, 1000], "output_dtype": "float32"},
        {"op": "reduce_sum", "input_ptrs": [3], "output_ptr": 4,
         "input_shape": [128, 1000], "output_shape": [128, 1]},
        {"op": "compute", "api_op": "rsqrt", "input_ptrs": [4], "output_ptr": 5,
         "input_shape": [128, 1], "output_shape": [128, 1], "output_dtype": "float32"},
    ]
    context = [{"op": "load", "active_lanes": 128 * 900, "partition_count": 128,
                "offsets_shape": [128, 1000], "mem_src": "HBM", "mem_dst": "SBUF"}]
    ir = build_region_ir(members, context)
    assert ir["reduction_count"] == 1
    assert ir["two_input_elementwise_count"] == 1
    assert ir["transcendental_count"] == 1
    assert ir["dag_edges"] == [[0, 1], [1, 2]]
    assert ir["logical_free_dim"] == 900 and ir["has_mask_or_tail"]
    assert ir["uses_sbuf"] and not ir["uses_psum"]
    assert compositional_features(ir)["rsqrt_newton_interaction"] == 0


def test_region_ir_key_ignores_pointer_values():
    a = [{"op": "compute", "api_op": "add", "input_ptrs": [1, 2], "output_ptr": 3,
          "input_shape": [128, 64], "output_shape": [128, 64], "output_dtype": "float32"}]
    b = [{**a[0], "input_ptrs": [100, 200], "output_ptr": 300}]
    assert build_region_ir(a)["structural_key"] == build_region_ir(b)["structural_key"]


def test_structural_family_distinguishes_multiply_chain_from_add_chain():
    def event(op, inputs, output):
        return {"op": "compute", "api_op": op, "input_ptrs": inputs, "output_ptr": output,
                "input_shape": [128, 64], "output_shape": [128, 64], "output_dtype": "float32"}
    multiply = build_region_ir([event("multiply", [1, 2], 3), event("multiply", [3, 4], 5)])
    add = build_region_ir([event("add", [1, 2], 3), event("add", [3, 4], 5)])
    assert structural_family(multiply) == "elementwise_multiply_n2"
    assert structural_family(add) == "elementwise_two_n2"
