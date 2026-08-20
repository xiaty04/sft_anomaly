from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np

from .data.common import record_bounds, record_values
from .prompts import interval_prompt

MODALITIES = ("text", "vision")


def validate_modality(modality: str) -> str:
    if modality not in MODALITIES:
        raise ValueError(f"modality must be one of {MODALITIES}, got {modality!r}")
    return modality


def serialize_indexed_series(values: np.ndarray, start_index: int, precision: int = 5) -> str:
    flattened = np.asarray(values, dtype=np.float64).reshape(-1)
    indices = np.arange(start_index, start_index + len(flattened), dtype=np.int64)
    return serialize_indexed_points(indices, flattened, precision)


def serialize_indexed_points(
    indices: np.ndarray, values: np.ndarray, precision: int = 5
) -> str:
    flattened = np.asarray(values, dtype=np.float64).reshape(-1)
    coordinates = np.asarray(indices, dtype=np.int64).reshape(-1)
    if len(coordinates) != len(flattened):
        raise ValueError("indices and values must have the same length")
    if not np.isfinite(flattened).all():
        raise ValueError("series contains NaN or infinity")
    pairs = [
        f"({int(index)},{value:.{precision}g})"
        for index, value in zip(coordinates, flattened)
    ]
    return "[" + ",".join(pairs) + "]"


def extrema_summary(
    values: np.ndarray, start_index: int, max_points: int
) -> Tuple[np.ndarray, np.ndarray]:
    flattened = np.asarray(values, dtype=np.float64).reshape(-1)
    if max_points <= 0:
        raise ValueError("max_points must be positive")
    if len(flattened) <= max_points:
        indices = np.arange(start_index, start_index + len(flattened), dtype=np.int64)
        return indices, flattened
    if max_points < 4:
        raise ValueError("max_points must be at least 4 when summarization is required")

    bin_count = (max_points - 2) // 2
    edges = np.linspace(1, len(flattened) - 1, bin_count + 1, dtype=np.int64)
    selected = {0, len(flattened) - 1}
    for left, right in zip(edges[:-1], edges[1:]):
        if right <= left:
            continue
        segment = flattened[left:right]
        selected.add(int(left + np.argmin(segment)))
        selected.add(int(left + np.argmax(segment)))
    offsets = np.asarray(sorted(selected), dtype=np.int64)
    return offsets + start_index, flattened[offsets]


def text_prompt(record: Dict[str, Any], precision: int = 5) -> str:
    start, end = record_bounds(record)
    values = record_values(record)
    if len(values) != end - start:
        raise ValueError(f"input length mismatch: {len(values)} vs [{start}, {end})")
    max_points = int(record.get("text_max_points", len(values)))
    indices, selected = extrema_summary(values, start, max_points)
    if len(selected) < len(values):
        input_name = "extrema-preserving indexed summary of the complete time series"
        summary_note = (
            f" The complete range has {len(values)} points; {len(selected)} representative "
            "points are listed with their original global coordinates."
        )
    else:
        input_name = "indexed time series"
        summary_note = ""
    return (
        interval_prompt(start, end, input_name=input_name)
        + summary_note
        + " Indexed values: "
        + serialize_indexed_points(indices, selected, precision)
    )
