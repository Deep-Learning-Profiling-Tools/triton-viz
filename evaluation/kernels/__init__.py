"""Evaluation corpora. Each module exposes CORPUS: evaluation.spec.Corpus."""

CORPORA = (
    "golden_smoke",
    "rmw_sync",
    "await_sync",
    "tritonracebench",
    "tutorials",
    "liger",
    "tritonbench_g",
    "fla",
    "flagattn",
    "flaggems",
    "torchao",
    "tritonbench_meta",
    "aiter_originals",
    "tilebench",
    "tilebench_cutile",
)


def load(name: str):
    import importlib

    mod = importlib.import_module(f"evaluation.kernels.{name}")
    return mod.CORPUS
