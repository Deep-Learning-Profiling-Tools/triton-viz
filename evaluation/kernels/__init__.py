"""Evaluation corpora. Each module exposes CORPUS: evaluation.spec.Corpus."""

CORPORA = ("golden_smoke",)


def load(name: str):
    import importlib

    mod = importlib.import_module(f"evaluation.kernels.{name}")
    return mod.CORPUS
