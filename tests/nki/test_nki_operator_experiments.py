"""Holdout measurement protocol: median over independent compilations."""

import pytest
# --- median over independent compilations -------------------------------------


def test_median_hardware_trial_keeps_the_median_run_whole(tmp_path):
    """The retained NC-p50 and its profile must come from the same execution."""
    from triton_viz.tools.nki_operator_experiments import _median_hardware_trial

    # Trial 2 lands in the slow compilation mode.
    plan = {1: (10.0, "fast_a"), 2: (38.0, "slow"), 3: (11.0, "fast_b")}
    calls = []

    def runner(op, inputs, artifact_dir, warmup, iters):
        index = int(str(artifact_dir.name).replace("hardware_trial", ""))
        artifact_dir.mkdir(parents=True, exist_ok=True)
        calls.append(index)
        nc, tag = plan[index]
        return nc, {"tag": tag, "total_active_time": nc / 1e6}

    art = tmp_path / "case"
    art.mkdir()
    nc_p50, profile = _median_hardware_trial(runner, "relu", [], art, 1, 1, 3)
    assert calls == [1, 2, 3]
    # Median of {10, 38, 11} is 11 -- and the profile is that run's, not a
    # per-field median that would mix executions.
    assert nc_p50 == pytest.approx(11.0)
    assert profile["tag"] == "fast_b"
    # hardware/ points at the median trial so downstream readers are unaware.
    assert (art / "hardware").is_symlink()
    assert (art / "hardware").resolve().name == "hardware_trial3"


def test_median_hardware_trial_falls_back_when_every_trial_fails(tmp_path):
    from triton_viz.tools.nki_operator_experiments import _median_hardware_trial

    seen = []

    def runner(op, inputs, artifact_dir, warmup, iters):
        artifact_dir.mkdir(parents=True, exist_ok=True)
        seen.append(artifact_dir.name)
        return (None, {}) if artifact_dir.name.startswith("hardware_trial") else (7.0, {"tag": "plain"})

    art = tmp_path / "case"
    art.mkdir()
    nc_p50, profile = _median_hardware_trial(runner, "relu", [], art, 1, 1, 2)
    # Rather than reporting nothing, it falls back to one ordinary measurement.
    assert nc_p50 == pytest.approx(7.0)
    assert profile["tag"] == "plain"
    assert seen[-1] == "hardware"
