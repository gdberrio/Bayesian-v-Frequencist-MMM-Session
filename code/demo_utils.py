from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

CODE_DIR = Path(__file__).resolve().parent


def finalize_figure(
    filename: str,
    *,
    fig=None,
    dpi: int = 150,
    bbox_inches: str = "tight",
) -> Path:
    """Save a figure into code/ regardless of the caller's working directory."""
    figure = fig or plt.gcf()
    output_path = CODE_DIR / filename
    figure.savefig(output_path, dpi=dpi, bbox_inches=bbox_inches)

    if "agg" in plt.get_backend().lower():
        plt.close(figure)
    else:
        plt.show()

    return output_path
