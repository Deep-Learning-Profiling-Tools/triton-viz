import csv
import json
from pathlib import Path

import numpy as np

from microbench.inf2_nki.harness.generate_sweep_config import expand_sweeps
from microbench.inf2_nki.harness.run_microbench import expand_config, run_one
from microbench.inf2_nki.tests.bandwidth_dma.kernels import (
    transpose_pipeline_work_bytes,
    transpose_work_bytes,
    work_bytes,
)
from microbench.inf2_nki.profile_parser.summarize_profile import summarize_run
from microbench.inf2_nki.profile_parser.export_csv import export_csv
from microbench.inf2_nki.profile_parser.plot_dma_free_dimension import load_points
from microbench.inf2_nki.profile_parser.plot_dma_partition_surface import load_surface
from microbench.inf2_nki.tests.static_dma.kernels import work_units as static_dma_work_units
from microbench.inf2_nki.common.inputs import make_pointer_ring, pointer_ring_walk


def test_quick_config_expands_modes_and_microbench_classes():
    config = json.loads(Path("microbench/inf2_nki/configs/quick.json").read_text())
    specs = expand_config(config)
    assert len(specs) == 15
    assert {spec["kind"] for spec in specs} >= {
        "pointer_chase",
        "dma_roundtrip_latency",
        "dma_bandwidth",
        "vector_add",
        "scalar_exp",
        "tensor_matmul",
        "tensor_dma_overlap",
        "program_mapping",
    }
    assert all("mode" in spec and "modes" not in spec for spec in specs)
    ptr = next(spec for spec in specs if spec["kind"] == "pointer_chase")
    assert ptr["mode"] == "hbm_index_chain"
    bw_modes = {spec["mode"] for spec in specs if spec["kind"] == "dma_bandwidth"}
    assert bw_modes == {"hbm_to_sbuf_stream", "sbuf_to_hbm_stream", "roundtrip_stream"}


def test_sweep_template_expands_to_explicit_benchmarks():
    config = {
        "suite": "tiny",
        "sweeps": {
            "dma_bandwidth": {
                "dtype": ["float32"],
                "p": [128],
                "f": [256, 512],
                "repeat": [1],
                "mode": ["hbm_to_sbuf_stream", "roundtrip_stream"],
            }
        },
    }
    expanded = expand_sweeps(config)
    assert len(expanded["benchmarks"]) == 4
    assert expanded["benchmarks"][0]["kind"] == "dma_bandwidth"
    assert expanded["benchmarks"][0]["modes"] == ["hbm_to_sbuf_stream"]


def test_matrix_config_expands_cartesian_product_and_modes():
    config = {"benchmarks": [{
        "name": "surface", "kind": "dma_bandwidth", "dtype": "float32", "repeat": 1,
        "modes": ["hbm_to_sbuf_stream", "roundtrip_stream"],
        "matrix": {"p": [1, 8], "f": [32, 64]},
    }]}
    specs = expand_config(config)
    assert len(specs) == 8
    assert {(spec["p"], spec["f"]) for spec in specs} == {(1, 32), (1, 64), (8, 32), (8, 64)}
    assert all("matrix" not in spec for spec in specs)


def test_runtime_overhead_config_excludes_compiler_invalid_empty_kernel():
    config = json.loads(
        Path("microbench/inf2_nki/configs/runtime_overhead.json").read_text()
    )
    specs = expand_config(config)
    assert specs
    assert all(spec["mode"] != "empty" for spec in specs)


def test_run_one_uses_per_test_folder_and_work_metadata(tmp_path):
    spec = {
        "name": "ptr_chase",
        "kind": "pointer_chase",
        "dtype": "uint32",
        "ring_length": 1024,
        "stride": 1,
        "repeat": 2,
        "mode": "hbm_index_chain",
    }
    # Monkeypatch by skipping actual Neuron execution: use skip-existing path.
    bench_dir = tmp_path / "latency_pointer_chase" / "ptr_chase__hbm_index_chain__dtypeuint32__ring_length1024__stride1__repeat2"
    bench_dir.mkdir(parents=True)
    (bench_dir / "manifest.json").write_text("{}")
    row = run_one(spec, tmp_path, warmup=1, iters=1, profile_export="none", explorer_timeout_s=1, skip_existing=True)
    assert row["status"] == "skipped_existing"
    assert "latency_pointer_chase" in row["dir"]


def test_skip_existing_counts_as_success_for_resumable_runs(tmp_path, monkeypatch):
    from microbench.inf2_nki.harness import run_microbench

    config = tmp_path / "config.json"
    config.write_text(
        json.dumps(
            {
                "suite": "resume",
                "benchmarks": [
                    {
                        "name": "ptr",
                        "kind": "pointer_chase",
                        "dtype": "uint32",
                        "ring_length": 16,
                        "stride": 1,
                        "repeat": 1,
                        "modes": ["hbm_index_chain"],
                    }
                ],
            }
        )
    )
    output = tmp_path / "results"
    run_id = "resume"
    bench_dir = (
        output
        / run_id
        / "latency_pointer_chase"
        / "ptr__hbm_index_chain__dtypeuint32__ring_length16__stride1__repeat1"
    )
    bench_dir.mkdir(parents=True)
    (bench_dir / "manifest.json").write_text("{}")
    monkeypatch.setattr(run_microbench, "_collect_versions", lambda: {})
    assert (
        run_microbench.main(
            [
                "--config",
                str(config),
                "--output-root",
                str(output),
                "--run-id",
                run_id,
                "--skip-existing",
            ]
        )
        == 0
    )
    manifest = json.loads((output / run_id / "run_manifest.json").read_text())
    assert manifest["num_ok"] == manifest["num_benchmarks"] == 1


def test_dma_work_metadata_counts_observable_keepalive_stores():
    work = work_bytes(p=128, f=2048, repeat=4, mode="hbm_to_sbuf_stream")
    assert work["hbm_read_bytes"] == 4 * 128 * 2048 * 4
    assert work["hbm_write_bytes"] == 4 * 128 * 4
    assert work["total_hbm_bytes"] == 4 * 128 * 2048 * 4 + 4 * 128 * 4
    assert work["free_bytes_per_partition"] == 8192
    assert work["dma_engines_expected"] == 16
    assert work["partitions_per_dma_engine"] == 8


def test_dma_work_metadata_scales_with_programs():
    one = work_bytes(p=128, f=8192, repeat=4, mode="roundtrip_stream", programs=1)
    two = work_bytes(p=128, f=8192, repeat=4, mode="roundtrip_stream", programs=2)
    assert two["hbm_read_bytes"] == 2 * one["hbm_read_bytes"]
    assert two["hbm_write_bytes"] == 2 * one["hbm_write_bytes"]
    assert two["total_hbm_bytes"] == 2 * one["total_hbm_bytes"]


def test_dma_transpose_work_metadata_describes_hbm_and_sbuf_layouts():
    work = transpose_work_bytes(p=16, f=512, dtype_name="float32")
    assert work["partition_count"] == 16
    assert work["hbm_minor_dimension_elements"] == 16
    assert work["free_bytes_per_partition"] == 2048
    assert work["hbm_read_bytes"] == 16 * 512 * 4
    assert work["transpose"] is True


def test_dma_transpose_pipeline_work_metadata_separates_controls():
    transpose = transpose_pipeline_work_bytes(
        p=128, f=1024, mode="transpose_only", dtype_name="float32"
    )
    store = transpose_pipeline_work_bytes(
        p=128, f=1024, mode="store_only", dtype_name="float32"
    )
    dependent = transpose_pipeline_work_bytes(
        p=128, f=1024, mode="transpose_then_store", dtype_name="float32"
    )
    full = 128 * 1024 * 4
    assert transpose["hbm_read_bytes"] == full
    assert transpose["hbm_write_bytes"] == 128 * 4
    assert store["hbm_read_bytes"] == 0
    assert store["hbm_write_bytes"] == full
    assert dependent["total_hbm_bytes"] == 2 * full
    assert dependent["dependent_transpose_store"] is True


def test_static_dma_work_metadata_separates_scatter_from_hbm_traffic():
    scatter = static_dma_work_units(
        p=8, x=4, y=8, mode="sbuf_transpose_scatter", dtype_name="float32"
    )
    baseline = static_dma_work_units(
        p=8, x=4, y=8, mode="hbm_roundtrip_baseline", dtype_name="float32"
    )
    assert scatter["static_dma_transfer_count"] == 32
    assert scatter["static_dma_bytes"] == 8 * 32 * 4
    assert baseline["static_dma_bytes"] == 0
    assert scatter["total_hbm_bytes"] == baseline["total_hbm_bytes"]


def test_profile_summary_handles_missing_parquet(tmp_path):
    bench_dir = tmp_path / "run" / "bandwidth_dma" / "bench"
    bench_dir.mkdir(parents=True)
    (bench_dir / "manifest.json").write_text(
        json.dumps({"id": "bench", "status": "ok", "spec": {"kind": "dma_bandwidth"}})
    )
    summary = summarize_run(tmp_path / "run")
    assert summary["benchmarks"]["bench"]["status"] == "ok"
    assert summary["benchmarks"]["bench"]["parquet_tables"] == []


def test_latency_fit_uses_repeat_slope():
    from microbench.inf2_nki.profile_parser.fit_latency import fit_results

    rows = []
    for repeat, latency in [(1, 11.0), (2, 13.0), (4, 17.0)]:
        rows.append(
            {
                "status": "ok",
                "spec": {"kind": "pointer_chase", "name": "ptr", "mode": "hbm_index_chain", "ring_length": 1024, "stride": 1, "repeat": repeat},
                "work": {"dependent_hbm_loads": repeat + 1},
                "latency_percentiles": {"nc_latency": {"p50_us": latency}},
            }
        )
    result = fit_results(rows)
    assert len(result["fits"]) == 1
    assert result["fits"][0]["slope_us"] > 0


def test_csv_export_is_generic_and_includes_derived_metrics(tmp_path):
    run = tmp_path / "sample_run"
    run.mkdir()
    (run / "run_manifest.json").write_text(json.dumps({"run_id": "sample_run"}))
    for repeat, latency in ((1, 10.0), (3, 14.0)):
        bench = run / "latency_pointer_chase" / f"pointer_{repeat}"
        bench.mkdir(parents=True)
        (bench / "manifest.json").write_text(json.dumps({
            "id": f"pointer_{repeat}", "status": "ok", "microbench_class": "latency",
            "spec": {"kind": "pointer_chase", "name": "pointer", "mode": "hbm_index_chain",
                     "repeat": repeat, "new_parameter": "automatically_exported"},
            "work": {"dependent_hbm_loads": repeat},
            "latency_percentiles": {"nc_latency": {"p50_us": latency}},
        }))

    bench = run / "bandwidth_dma" / "dma"
    bench.mkdir(parents=True)
    (bench / "manifest.json").write_text(json.dumps({
        "id": "dma", "status": "ok", "microbench_class": "bandwidth",
        "spec": {"kind": "dma_bandwidth", "mode": "roundtrip_stream"},
        "work": {"hbm_read_bytes": 1000, "hbm_write_bytes": 1000},
    }))
    (bench / "explorer_summary.json").write_text(json.dumps({
        "model": {"hbm_read_bytes": 1000, "hbm_write_bytes": 1000, "dma_active_time": 1e-6}
    }))

    output = tmp_path / "all.csv"
    exported = export_csv(run, output)
    with output.open(newline="") as file:
        rows = list(csv.DictReader(file))
    assert len(exported) == len(rows) == 4
    dma = next(row for row in rows if row["id"] == "dma")
    assert float(dma["derived.hbm_gbps_dma_active"]) == 2.0
    assert dma["derived.read_byte_count_match"] == "True"
    pointer = next(row for row in rows if row["id"] == "pointer_1")
    assert pointer["spec.new_parameter"] == "automatically_exported"
    fit = next(row for row in rows if row["row_type"] == "latency_fit")
    assert float(fit["fit.slope_ns"]) == 2000.0


def test_dma_free_dimension_plot_loader_selects_and_scales_rows(tmp_path):
    path = tmp_path / "all_results.csv"
    fieldnames = ["row_type", "status", "spec.name", "mode", "run_id",
                  "work.free_bytes_per_partition", "derived.read_gbps_dma_active"]
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow({
            "row_type": "benchmark", "status": "ok", "spec.name": "dma_free_dimension",
            "mode": "hbm_to_sbuf_stream", "run_id": "run", "work.free_bytes_per_partition": 4096,
            "derived.read_gbps_dma_active": 272,
        })
        writer.writerow({"row_type": "benchmark", "status": "ok", "spec.name": "other"})
    points = load_points(path)
    assert len(points) == 1
    assert points[0]["per_engine_bytes_per_ns"] == 17.0


def test_dma_partition_surface_loader_computes_engine_utilization(tmp_path):
    path = tmp_path / "all_results.csv"
    fieldnames = ["row_type", "status", "spec.name", "mode", "run_id", "work.partition_count",
                  "work.free_bytes_per_partition", "work.dma_engines_expected", "derived.read_gbps_dma_active"]
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow({
            "row_type": "benchmark", "status": "ok", "spec.name": "dma_partition_surface",
            "mode": "hbm_to_sbuf_stream", "run_id": "run", "work.partition_count": 16,
            "work.free_bytes_per_partition": 2048, "work.dma_engines_expected": 2,
            "derived.read_gbps_dma_active": 136,
        })
    points = load_surface(path)
    assert len(points) == 1
    assert points[0]["engine_utilization"] == 0.5


def test_pointer_ring_is_2d_and_within_partition_limit():
    ring = make_pointer_ring(1024, stride=17)
    # Must be (1, N): NKI load caps the partition dim at 128, so the ring lives
    # in the free dimension. A flat (N,) ring would be rejected for N > 128.
    assert ring.shape == (1, 1024)
    assert ring.dtype == np.uint32
    # Deterministic ring: element i points to (i + stride) % N.
    assert int(ring[0, 0]) == 17
    assert int(ring[0, 1023]) == (1023 + 17) % 1024


def test_pointer_ring_walk_matches_manual_walk():
    ring = make_pointer_ring(256, stride=17)
    # seed = ring[0,0]; then two dependent hops.
    idx = int(ring[0, 0])
    idx = int(ring[0, idx])
    idx = int(ring[0, idx])
    assert pointer_ring_walk(ring, 2) == idx


def test_pointer_chase_kernel_is_correct_in_simulator():
    # This is the key regression guard: the kernel must actually walk the ring,
    # not silently return a constant. Uses nki.simulate_kernel (real inputs on
    # CPU) because nki.benchmark runs on zeroed inputs and cannot verify logic.
    pytest = __import__("pytest")
    try:
        from microbench.inf2_nki.harness.validate_kernels import validate_pointer_chase
    except ModuleNotFoundError:  # pragma: no cover - neuron deps missing
        pytest.skip("NeuronX dependencies are missing.")

    report = validate_pointer_chase(ring_length=256, stride=17, repeats=(1, 2, 4, 8))
    assert report["ok"], report
    assert all(case["match"] for case in report["cases"])
