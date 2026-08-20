from __future__ import annotations

import re
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

from ..intervals import to_jsonable
from ..io import write_jsonl
from ..rendering import render_series
from .common import load_series

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


def prepare_ucr(config: Dict[str, Any], render_config: Dict[str, Any]) -> Path:
    started = time.perf_counter()
    raw_dir = Path(config["raw_dir"])
    output_dir = Path(config["output_dir"])
    files = sorted(raw_dir.glob("*.txt"))
    if not files:
        raise FileNotFoundError(f"no UCR .txt files found in {raw_dir}")
    available = len(files)
    max_series = config.get("max_series")
    if max_series is not None:
        files = files[: int(max_series)]
    image_dir = output_dir / "images"
    total = len(files)
    print(
        f"[prepare-ucr] selected {total}/{available} UCR files from {raw_dir}; "
        f"output={output_dir}",
        flush=True,
    )
    series_records = []
    text_max_points = int(config.get("text_max_points", 1024))
    if text_max_points < 4:
        raise ValueError("ucr.text_max_points must be at least 4")
    for index, path in enumerate(files, start=1):
        series_started = time.perf_counter()
        print(f"[prepare-ucr] series {index}/{total} loading {path.name}", flush=True)
        metadata = parse_ucr_filename(path)
        values = load_series(path)
        test_start, intervals = _convert_metadata(metadata, config, len(values))
        series_id = f"ucr_{metadata['archive_id']}"
        image_path = image_dir / f"{series_id}.png"
        reused = bool(config.get("reuse_images", True)) and image_path.exists()
        if not reused:
            render_series(values[test_start:], image_path, test_start, render_config)
        series_records.append(
            {
                "sample_id": series_id,
                "series_id": series_id,
                "source": "ucr",
                "split": "test",
                "name": metadata["name"],
                "series_path": str(path.resolve()),
                "image_path": str(image_path.resolve()),
                "input_unit": "series",
                "input_start": test_start,
                "input_end": len(values),
                "length": len(values) - test_start,
                "series_length": len(values),
                "eval_start": test_start,
                "eval_end": len(values),
                "text_max_points": text_max_points,
                "intervals": to_jsonable(intervals),
                "ucr_metadata": metadata,
            }
        )
        print(
            f"[prepare-ucr] series {index}/{total} length={len(values)} "
            f"eval=[{test_start},{len(values)}) image={'reused' if reused else 'rendered'} "
            f"elapsed={time.perf_counter() - series_started:.1f}s",
            flush=True,
        )
    series_path = output_dir / "series.jsonl"
    write_jsonl(series_path, series_records)
    print(f"[prepare-ucr] wrote {len(series_records)} series samples -> {series_path}", flush=True)
    print(f"[prepare-ucr] prepared {len(series_records)} full-range images -> {image_dir}", flush=True)
    print(f"[prepare-ucr] total elapsed={time.perf_counter() - started:.1f}s", flush=True)
    return series_path
