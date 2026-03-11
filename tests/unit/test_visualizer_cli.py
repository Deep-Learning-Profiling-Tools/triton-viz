from pathlib import Path
from unittest.mock import patch

from triton_viz.visualizer_cli import main


def test_visualizer_cli_forwards_trace_file_and_flags():
    trace_file = Path("trace.tvz")
    with patch("triton_viz.load") as mock_load, patch(
        "triton_viz.launch"
    ) as mock_launch:
        main([str(trace_file), "--port", "9000", "--no-share", "--no-block"])

    mock_load.assert_called_once_with(trace_file)
    mock_launch.assert_called_once_with(
        share=False,
        port=9000,
        block=False,
    )
