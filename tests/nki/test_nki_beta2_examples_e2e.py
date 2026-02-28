import importlib.util
from pathlib import Path

import pytest

try:
    import nki  # noqa: F401
except ModuleNotFoundError:
    pytest.skip(
        "NeuronX dependencies are missing. Install triton-viz[nki] to run these tests.",
        allow_module_level=True,
    )

pytestmark = pytest.mark.nki

_EXAMPLE_DIR = Path(__file__).resolve().parents[2] / "examples" / "nki"
_EXAMPLE_PATHS = (
    _EXAMPLE_DIR / "matmul_beta2.py",
    _EXAMPLE_DIR / "rmsnorm_beta2.py",
    _EXAMPLE_DIR / "rope_beta2.py",
    _EXAMPLE_DIR / "softmax_beta2.py",
    _EXAMPLE_DIR / "tiled_attention_beta2.py",
)


def _load_module(module_name, module_path):
    """Load one beta2 example module from file path."""
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("example_path", _EXAMPLE_PATHS, ids=lambda p: p.stem)
def test_beta2_example_run_demo_prints_success(example_path, monkeypatch, capsys):
    """Each beta2 example demo should report matching actual and expected outputs."""
    module_name = f"nki_beta2_example_{example_path.stem}"
    module = _load_module(module_name, example_path)

    monkeypatch.setattr(module.triton_viz, "launch", lambda *args, **kwargs: None)
    monkeypatch.setattr(module, "TRITON_VIZ_ENABLED", True)

    module.triton_viz.clear()
    module._run_demo()
    assert "actual equals expected" in capsys.readouterr().out
