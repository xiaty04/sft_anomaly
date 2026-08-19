from __future__ import annotations

import re
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

from ..intervals import clip, to_jsonable
from ..io import write_jsonl
from ..rendering import render_series
from .common import load_series, window_starts

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


def prepare_ucr(config: Dict[str, Any], render_config: Dict[str, Any]) -> Tuple[Path, Path]:
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
    series_records, window_records = [], []
    for index, path in enumerate(files, start=1):
        series_started = time.perf_counter()
        print(f"[prepare-ucr] series {index}/{total} loading {path.name}", flush=True)
        metadata = parse_ucr_filename(path)
        values = load_series(path)
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
        max_windows = config.get("max_windows_per_series")
        if max_windows is not None:
            starts = starts[: int(max_windows)]
        window_total = len(starts)
        progress_step = max(1, min(25, window_total // 10 or 1))
        rendered = 0
        reused = 0
        print(
            f"[prepare-ucr] series {index}/{total} length={len(values)} "
            f"eval=[{test_start},{len(values)}) windows={window_total}",
            flush=True,
        )
        for window_index, window_start in enumerate(starts):
            window_end = min(window_start + int(config.get("window_size", 1000)), len(values))
            image_path = image_dir / f"{series_id}_{window_index:04d}.png"
            if bool(config.get("reuse_images", True)) and image_path.exists():
                reused += 1
            else:
                render_series(values[window_start:window_end], image_path, window_start, render_config)
                rendered += 1
            window_records.append(
                {
                    "sample_id": f"{series_id}_w{window_index:04d}",
                    "series_id": series_id,
                    "source": "ucr",
                    "split": "test",
                    "series_path": str(path.resolve()),
                    "image_path": str(image_path.resolve()),
                    "window_start": window_start,
                    "window_end": window_end,
                    "length": window_end - window_start,
                    "intervals": to_jsonable(clip(intervals, window_start, window_end)),
                }
            )
            completed = window_index + 1
            if completed == 1 or completed % progress_step == 0 or completed == window_total:
                elapsed = time.perf_counter() - series_started
                print(
                    f"[prepare-ucr] series {index}/{total} windows "
                    f"{completed}/{window_total} rendered={rendered} reused={reused} "
                    f"elapsed={elapsed:.1f}s",
                    flush=True,
                )
        print(
            f"[prepare-ucr] series {index}/{total} done in "
            f"{time.perf_counter() - series_started:.1f}s",
            flush=True,
        )
    series_path = output_dir / "series.jsonl"
    windows_path = output_dir / "windows.jsonl"
    write_jsonl(series_path, series_records)
    write_jsonl(windows_path, window_records)
    print(f"[prepare-ucr] wrote {len(series_records)} series -> {series_path}", flush=True)
    print(f"[prepare-ucr] wrote {len(window_records)} windows -> {windows_path}", flush=True)
    print(f"[prepare-ucr] wrote {len(window_records)} images -> {image_dir}", flush=True)
    print(f"[prepare-ucr] total elapsed={time.perf_counter() - started:.1f}s", flush=True)
    return series_path, windows_path
