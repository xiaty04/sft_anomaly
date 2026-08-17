from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

from ..intervals import canonicalize
from ..io import read_jsonl


def window_starts(start: int, end: int, size: int, stride: int) -> List[int]:
    if size <= 0 or stride <= 0:
        raise ValueError("window size and stride must be positive")
    if end <= start:
        return []
    if end - start <= size:
        return [start]
    starts = list(range(start, end - size + 1, stride))
    final_start = end - size
    if starts[-1] != final_start:
        starts.append(final_start)
    return starts


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
        canonicalize(record.get("intervals", []), lower=0, upper=int(record["length"]))


def load_training_manifest(path: Path) -> List[Dict[str, Any]]:
    records = read_jsonl(path)
    validate_training_records(records)
    return records

