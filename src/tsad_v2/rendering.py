from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict

import numpy as np


def render_series(
    series: np.ndarray,
    output_path: Path,
    start_index: int,
    render_config: Dict[str, Any],
) -> None:
    cache_dir = output_path.parent / ".mpl-cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_dir))
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    values = np.asarray(series, dtype=np.float64).reshape(-1)
    if not np.isfinite(values).all():
        raise ValueError("series contains NaN or infinity")
    width = int(render_config.get("width", 1200))
    height = int(render_config.get("height", 360))
    dpi = int(render_config.get("dpi", 120))
    fig, axis = plt.subplots(figsize=(width / dpi, height / dpi), dpi=dpi)
    x_axis = np.arange(start_index, start_index + len(values))
    axis.plot(x_axis, values, color="#1f5f8b", linewidth=float(render_config.get("line_width", 1.2)))
    axis.set_xlim(start_index, start_index + len(values) - 1)
    axis.set_xlabel("time index")
    axis.set_ylabel("value")
    axis.grid(alpha=0.18, linewidth=0.6)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, format="png")
    plt.close(fig)

