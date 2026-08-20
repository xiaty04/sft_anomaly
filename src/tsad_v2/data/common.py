from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

from ..intervals import canonicalize
from ..io import read_jsonl


def load_series(path: Path) -> np.ndarray:
    if path.suffix == ".npy":
        values = np.load(path)
    else:
        try:
            values = np.loadtxt(path, dtype=np.float64)
        except ValueError:
            values = np.loadtxt(path, dtype=np.float64, delimiter=",")
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if not len(values) or not np.isfinite(values).all():
        raise ValueError(f"series is empty or non-finite: {path}")
    return values


def record_bounds(record: Dict[str, Any]) -> Tuple[int, int]:
    start = int(record.get("input_start", record.get("window_start", 0)))
    end = int(
        record.get(
            "input_end",
            record.get("window_end", start + int(record["length"])),
        )
    )
    return start, end


def record_values(record: Dict[str, Any]) -> np.ndarray:
    values = load_series(Path(record["series_path"]))
    start, end = record_bounds(record)
    if not (0 <= start < end <= len(values)):
        raise ValueError(f"invalid input range [{start}, {end}) for series of length {len(values)}")
    selected = values[start:end]
    if len(selected) != int(record["length"]):
        raise ValueError(
            f"record length mismatch: {len(selected)} vs declared {record['length']}"
        )
    return selected


def validate_training_records(records: Sequence[Dict[str, Any]]) -> None:
    if not records:
        raise ValueError("training manifest is empty")
    ids = set()
    for record in records:
        sample_id = record.get("sample_id")
        if not sample_id or sample_id in ids:
            raise ValueError(f"missing or duplicate sample_id: {sample_id!r}")
        ids.add(sample_id)
        if record.get("source") != "synthetic":
            raise ValueError(f"training only accepts synthetic data: {sample_id}")
        if record.get("split") not in {"train", "val"}:
            raise ValueError(f"training record has forbidden split: {sample_id}")
        image_path = Path(record["image_path"])
        if not image_path.exists():
            raise FileNotFoundError(f"missing image for {sample_id}: {image_path}")
        series_path = Path(record["series_path"])
        if not series_path.exists():
            raise FileNotFoundError(f"missing series for {sample_id}: {series_path}")
        canonicalize(record.get("intervals", []), lower=0, upper=int(record["length"]))


def load_training_manifest(path: Path) -> List[Dict[str, Any]]:
    records = read_jsonl(path)
    validate_training_records(records)
    return records
