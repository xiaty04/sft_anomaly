from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

from ..intervals import clip, to_jsonable
from ..io import write_jsonl
from ..rendering import render_series
from .common import window_starts

UCR_PATTERN = re.compile(
    r"^(?P<id>\d+)_UCR_Anomaly_(?P<name>.+)_(?P<train>\d+)_(?P<start>\d+)_(?P<end>\d+)\.txt$"
)


def parse_ucr_filename(path: Path) -> Dict[str, Any]:
    match = UCR_PATTERN.match(path.name)
    if not match:
        raise ValueError(f"not a recognized UCR anomaly filename: {path.name}")
    values = match.groupdict()
    return {
        "archive_id": values["id"],
        "name": values["name"],
        "train_end_raw": int(values["train"]),
        "anomaly_start_raw": int(values["start"]),
        "anomaly_end_raw": int(values["end"]),
    }


def _load_series(path: Path) -> np.ndarray:
    try:
        values = np.loadtxt(path, dtype=np.float64)
    except ValueError:
        values = np.loadtxt(path, dtype=np.float64, delimiter=",")
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if not len(values) or not np.isfinite(values).all():
        raise ValueError(f"UCR series is empty or non-finite: {path}")
    return values


def _convert_metadata(metadata: Dict[str, Any], config: Dict[str, Any], length: int) -> Tuple[int, List[Tuple[int, int]]]:
    index_base = int(config.get("filename_index_base", 0))
    start = metadata["anomaly_start_raw"] - index_base
    end = metadata["anomaly_end_raw"] - index_base
    if bool(config.get("anomaly_end_inclusive", True)):
        end += 1
    if bool(config.get("train_end_is_count", True)):
        test_start = metadata["train_end_raw"]
    else:
        test_start = metadata["train_end_raw"] - index_base + 1
    if not (0 <= start < end <= length):
        raise ValueError(f"converted UCR interval is outside series: {(start, end)} vs {length}")
    if not (0 <= test_start < length):
        raise ValueError(f"converted UCR train boundary is outside series: {test_start} vs {length}")
    return test_start, [(start, end)]


def prepare_ucr(config: Dict[str, Any], render_config: Dict[str, Any]) -> Tuple[Path, Path]:
    raw_dir = Path(config["raw_dir"])
    output_dir = Path(config["output_dir"])
    files = sorted(raw_dir.glob("*.txt"))
    if not files:
        raise FileNotFoundError(f"no UCR .txt files found in {raw_dir}")
    image_dir = output_dir / "images"
    series_records, window_records = [], []
    for path in files:
        metadata = parse_ucr_filename(path)
        values = _load_series(path)
        test_start, intervals = _convert_metadata(metadata, config, len(values))
        series_id = f"ucr_{metadata['archive_id']}"
        series_records.append(
            {
                "sample_id": series_id,
                "series_id": series_id,
                "source": "ucr",
                "split": "test",
                "name": metadata["name"],
                "series_path": str(path.resolve()),
                "length": len(values),
                "eval_start": test_start,
                "eval_end": len(values),
                "intervals": to_jsonable(intervals),
                "ucr_metadata": metadata,
            }
        )
        starts = window_starts(
            test_start,
            len(values),
            int(config.get("window_size", 1000)),
            int(config.get("stride", 500)),
        )
        for window_index, window_start in enumerate(starts):
            window_end = min(window_start + int(config.get("window_size", 1000)), len(values))
            image_path = image_dir / f"{series_id}_{window_index:04d}.png"
            render_series(values[window_start:window_end], image_path, window_start, render_config)
            window_records.append(
                {
                    "sample_id": f"{series_id}_w{window_index:04d}",
                    "series_id": series_id,
                    "source": "ucr",
                    "split": "test",
                    "image_path": str(image_path.resolve()),
                    "window_start": window_start,
                    "window_end": window_end,
                    "length": window_end - window_start,
                    "intervals": to_jsonable(clip(intervals, window_start, window_end)),
                }
            )
    series_path = output_dir / "series.jsonl"
    windows_path = output_dir / "windows.jsonl"
    write_jsonl(series_path, series_records)
    write_jsonl(windows_path, window_records)
    return series_path, windows_path

