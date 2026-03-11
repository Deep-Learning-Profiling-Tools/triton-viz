import argparse
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser for loading a saved trace into the visualizer."""
    parser = argparse.ArgumentParser(
        prog="triton-visualizer",
        description="Launch the Triton-Viz visualizer from a saved trace archive.",
    )
    parser.add_argument("trace_file", type=Path, help="Path to a .tvz trace archive.")
    parser.add_argument("--port", type=int, default=None, help="Port to bind locally.")
    parser.add_argument(
        "--no-share",
        action="store_true",
        help="Disable the Cloudflare share link and serve locally only.",
    )
    parser.add_argument(
        "--no-block",
        action="store_true",
        help="Return immediately instead of blocking on the visualizer server.",
    )
    return parser


def main(argv: list[str] | None = None):
    """Parse CLI args and launch the visualizer with a saved trace."""
    args = build_parser().parse_args(argv)
    import triton_viz

    triton_viz.load(args.trace_file)
    return triton_viz.launch(
        share=not args.no_share,
        port=args.port,
        block=not args.no_block,
    )
