from __future__ import annotations

from typing import Any, Dict

import numpy as np

from .data.common import record_values
from .prompts import interval_prompt

MODALITIES = ("text", "vision")


def validate_modality(modality: str) -> str:
    if modality not in MODALITIES:
        raise ValueError(f"modality must be one of {MODALITIES}, got {modality!r}")
    return modality


def serialize_indexed_series(values: np.ndarray, start_index: int, precision: int = 5) -> str:
    flattened = np.asarray(values, dtype=np.float64).reshape(-1)
    if not np.isfinite(flattened).all():
        raise ValueError("series contains NaN or infinity")
    pairs = [
        f"({start_index + offset},{value:.{precision}g})"
        for offset, value in enumerate(flattened)
    ]
    return "[" + ",".join(pairs) + "]"


def text_prompt(record: Dict[str, Any], precision: int = 5) -> str:
    start = int(record.get("window_start", 0))
    end = int(record.get("window_end", start + int(record["length"])))
    values = record_values(record)
    if len(values) != end - start:
        raise ValueError(f"record length mismatch: {len(values)} vs [{start}, {end})")
    return (
        interval_prompt(start, end, input_name="indexed time series")
        + " Indexed values: "
        + serialize_indexed_series(values, start, precision)
    )
