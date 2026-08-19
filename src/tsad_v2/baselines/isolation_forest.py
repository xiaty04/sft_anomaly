from __future__ import annotations

import platform
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from ..config import save_effective_config, save_run_metadata
from ..data.common import load_series
from ..intervals import to_jsonable
from ..io import append_jsonl, read_jsonl


def _rolling(values: np.ndarray, window: int) -> tuple[np.ndarray, np.ndarray]:
    window = min(window, len(values))
    if window <= 1:
        return values.copy(), np.zeros_like(values)
    kernel = np.ones(window, dtype=np.float64) / window
    mean = np.convolve(values, kernel, mode="same")
    squared_mean = np.convolve(values * values, kernel, mode="same")
    return mean, np.sqrt(np.maximum(0.0, squared_mean - mean * mean))


def extract_features(values: np.ndarray, windows: List[int]) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    columns = [values, np.diff(values, prepend=values[0])]
    for window in windows:
        mean, std = _rolling(values, int(window))
        columns.extend([mean, std, values - mean])
    return np.column_stack(columns)


def mask_to_intervals(mask: np.ndarray, start_index: int, min_length: int = 1) -> List[tuple[int, int]]:
    intervals: List[tuple[int, int]] = []
    start = None
    for offset, anomalous in enumerate(np.asarray(mask, dtype=bool)):
        if anomalous and start is None:
            start = offset
        if start is not None and (not anomalous or offset == len(mask) - 1):
            end = offset + 1 if anomalous and offset == len(mask) - 1 else offset
            if end - start >= min_length:
                intervals.append((start_index + start, start_index + end))
            start = None
    return intervals


def run_isolation_forest(
    config: Dict[str, Any], manifest_path: Path, output_path: Path, limit: Optional[int] = None
) -> None:
    try:
        from sklearn.ensemble import IsolationForest
    except ImportError as exc:
        raise RuntimeError("Install project dependencies before running Isolation Forest") from exc

    records = read_jsonl(manifest_path)
    if limit is not None:
        records = records[:limit]
    completed = set()
    if output_path.exists():
        completed = {record["sample_id"] for record in read_jsonl(output_path)}
    baseline = config["isolation_forest"]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_effective_config(config, output_path.parent)
    save_run_metadata(
        {"method": "isolation_forest", "manifest": str(manifest_path), "python": platform.python_version()},
        output_path.parent,
    )
    windows = [int(value) for value in baseline.get("feature_windows", [5, 21, 51])]
    quantile = float(baseline.get("train_score_quantile", 0.995))
    min_length = int(baseline.get("min_event_length", 1))
    seed = int(config["project"].get("seed", 3407))
    for index, record in enumerate(records, start=1):
        if record["sample_id"] in completed:
            continue
        started = time.perf_counter()
        values = load_series(Path(record["series_path"]))
        test_start = int(record["eval_start"])
        train_features = extract_features(values[:test_start], windows)
        if len(train_features) < 8:
            raise ValueError(f"normal training prefix is too short: {record['sample_id']}")
        model = IsolationForest(
            n_estimators=int(baseline.get("n_estimators", 200)),
            contamination="auto",
            random_state=seed,
            n_jobs=int(baseline.get("n_jobs", -1)),
        )
        model.fit(train_features)
        train_scores = -model.score_samples(train_features)
        threshold = float(np.quantile(train_scores, quantile))
        test_features = extract_features(values, windows)[test_start:]
        test_scores = -model.score_samples(test_features)
        intervals = mask_to_intervals(test_scores >= threshold, test_start, min_length)
        append_jsonl(
            output_path,
            {
                "sample_id": record["sample_id"],
                "series_id": record["series_id"],
                "window_start": test_start,
                "window_end": len(values),
                "raw_output": None,
                "intervals": to_jsonable(intervals),
                "parse_valid": True,
                "parse_error": None,
                "latency_seconds": time.perf_counter() - started,
                "model": "isolation_forest",
                "threshold": threshold,
            },
        )
        print(f"[isolation-forest] {index}/{len(records)} {record['sample_id']}", flush=True)
