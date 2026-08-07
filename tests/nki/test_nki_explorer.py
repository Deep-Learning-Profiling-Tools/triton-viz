import pytest

from triton_viz.tools import nki_explorer
from triton_viz.tools.nki_explorer import export_parquet


def test_export_parquet_reuses_only_complete_nonempty_required_tables(
    tmp_path, monkeypatch
):
    parquet = tmp_path / "explorer_parquet"
    parquet.mkdir()
    (parquet / "Instruction.parquet").write_bytes(b"instruction")
    (parquet / "ActiveTime.parquet").write_bytes(b"active")

    def unexpected_popen(*args, **kwargs):
        raise AssertionError("complete parquet should not launch Explorer")

    monkeypatch.setattr(nki_explorer.subprocess, "Popen", unexpected_popen)
    assert export_parquet(tmp_path) == parquet


def test_export_parquet_reports_early_explorer_failure(tmp_path, monkeypatch):
    class FailedProcess:
        def poll(self):
            return 9

        def communicate(self):
            return "out", "bad schema"

    monkeypatch.setattr(
        nki_explorer.subprocess, "Popen", lambda *args, **kwargs: FailedProcess()
    )
    with pytest.raises(RuntimeError, match="bad schema"):
        export_parquet(tmp_path)
