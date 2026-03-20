import argparse
from pathlib import Path
from typing import Any


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser for loading a saved trace into the visualizer."""
    parser = argparse.ArgumentParser(
        prog="triton-visualizer",
        description="Launch the Triton-Viz visualizer from a saved trace archive.",
    )
    parser.add_argument("trace_file", type=Path, help="Path to a .tvz trace archive.")
    parser.add_argument("--port", type=int, default=None, help="Port to bind locally.")
    parser.add_argument(
        "--share",
        action="store_true",
        help="Enable the Cloudflare share link and serve through a public URL.",
    )
    parser.add_argument(
        "--no-block",
        action="store_true",
        help="Return immediately instead of blocking on the visualizer server.",
    )
    parser.add_argument(
        "--llm-api-key",
        default=None,
        help="Optional API key for the visualizer LLM assistant (calls setup_llm).",
    )
    parser.add_argument(
        "--llm-base-url",
        default=None,
        help="Optional OpenAI-compatible API base URL override.",
    )
    return parser


def main(argv: list[str] | None = None):
    """Parse CLI args and launch the visualizer with a saved trace."""
    args = build_parser().parse_args(argv)
    import triton_viz

    triton_viz.load(args.trace_file)
    llm_kw: dict[str, Any] = {}
    if args.llm_api_key is not None:
        llm_kw["api_key"] = args.llm_api_key
    if args.llm_base_url is not None:
        llm_kw["base_url"] = args.llm_base_url
    if llm_kw:
        triton_viz.setup_llm(**llm_kw)
    return triton_viz.launch(
        share=args.share,
        port=args.port,
        block=not args.no_block,
    )
