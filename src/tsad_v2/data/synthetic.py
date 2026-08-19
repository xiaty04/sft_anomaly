from __future__ import annotations

import time as time_module
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..intervals import Interval, to_jsonable
from ..io import write_jsonl
from ..rendering import render_series
from .common import load_series
from .ucr import parse_ucr_filename


def _background(length: int, rng: np.random.Generator) -> np.ndarray:
    time = np.arange(length)
    first_period = rng.uniform(45.0, 150.0)
    second_period = rng.uniform(18.0, 80.0)
    phase = rng.uniform(0, 2 * np.pi)
    signal = np.sin(2 * np.pi * time / first_period + phase)
    signal += rng.uniform(0.15, 0.45) * np.sin(2 * np.pi * time / second_period)
    noise = rng.normal(0, rng.uniform(0.03, 0.10), length)
    ar = np.zeros(length)
    coefficient = rng.uniform(0.2, 0.75)
    for index in range(1, length):
        ar[index] = coefficient * ar[index - 1] + noise[index]
    signal += ar
    signal += rng.uniform(-0.001, 0.001) * time
    return signal.astype(np.float32)


def _choose_intervals(length: int, count: int, rng: np.random.Generator) -> List[Interval]:
    minimum_gap = max(4, length // 50)
    candidates: List[Interval] = []
    for _ in range(500):
        duration = int(rng.integers(max(1, length // 100), max(3, length // 5)))
        start = int(rng.integers(minimum_gap, max(minimum_gap + 1, length - duration - minimum_gap)))
        proposed = (start, start + duration)
        if all(proposed[1] + minimum_gap <= old[0] or old[1] + minimum_gap <= proposed[0] for old in candidates):
            candidates.append(proposed)
            if len(candidates) == count:
                return sorted(candidates)
    raise RuntimeError("could not place non-overlapping anomaly intervals")


def _inject(
    series: np.ndarray,
    interval: Interval,
    anomaly_type: str,
    rng: np.random.Generator,
) -> Dict[str, float]:
    start, end = interval
    duration = end - start
    local_scale = max(float(np.std(series)), 0.1)
    if anomaly_type == "point":
        point_end = min(start + int(rng.integers(1, min(5, duration + 1))), end)
        interval = (start, point_end)
        magnitude = float(rng.choice([-1.0, 1.0]) * rng.uniform(3.0, 6.0) * local_scale)
        series[start:point_end] += magnitude
        return {"magnitude": magnitude, "effective_end": point_end}
    if anomaly_type == "range":
        shift = float(rng.choice([-1.0, 1.0]) * rng.uniform(1.5, 3.5) * local_scale)
        series[start:end] += shift + rng.normal(0, 0.25 * local_scale, duration)
        return {"shift": shift}
    if anomaly_type == "frequency":
        cycles = float(rng.uniform(4.0, 12.0))
        phase = np.linspace(0, cycles * 2 * np.pi, duration, endpoint=False)
        amplitude = float(rng.uniform(0.8, 1.8) * local_scale)
        series[start:end] += amplitude * np.sin(phase)
        return {"cycles": cycles, "amplitude": amplitude}
    if anomaly_type == "trend":
        slope = float(rng.choice([-1.0, 1.0]) * rng.uniform(1.5, 4.0) * local_scale / duration)
        series[start:end] += slope * np.arange(duration)
        return {"slope": slope}
    raise ValueError(f"unknown anomaly type: {anomaly_type}")


def generate_sample(
    length: int,
    anomaly_types: Sequence[str],
    normal_probability: float,
    max_intervals: int,
    seed: int,
    forced_type: str,
    background: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, List[Interval], List[str], List[Dict[str, float]]]:
    rng = np.random.default_rng(seed)
    if background is None:
        series = _background(length, rng)
    else:
        series = np.asarray(background, dtype=np.float32).reshape(-1).copy()
        if len(series) != length or not np.isfinite(series).all():
            raise ValueError("background must be finite and match the requested length")
    if rng.random() < normal_probability:
        return series, [], [], []
    count = int(rng.integers(1, max_intervals + 1))
    proposed = _choose_intervals(length, count, rng)
    intervals, used_types, parameters = [], [], []
    for index, interval in enumerate(proposed):
        anomaly_type = forced_type if index == 0 else str(rng.choice(anomaly_types))
        params = _inject(series, interval, anomaly_type, rng)
        if anomaly_type == "point":
            interval = (interval[0], int(params.pop("effective_end")))
        intervals.append(interval)
        used_types.append(anomaly_type)
        parameters.append(params)
    order = np.argsort([item[0] for item in intervals])
    return (
        series,
        [intervals[index] for index in order],
        [used_types[index] for index in order],
        [parameters[index] for index in order],
    )


def generate_split(
    split: str,
    count: int,
    config: Dict[str, Any],
    render_config: Dict[str, Any],
    base_seed: int,
    backgrounds: Sequence[Tuple[str, np.ndarray]],
) -> Path:
    if split not in {"train", "val"}:
        raise ValueError("synthetic split must be train or val")
    output_dir = Path(config["output_dir"])
    image_dir, series_dir = output_dir / "images" / split, output_dir / "series" / split
    image_dir.mkdir(parents=True, exist_ok=True)
    series_dir.mkdir(parents=True, exist_ok=True)
    anomaly_types = list(config.get("anomaly_types", ["point", "range", "frequency", "trend"]))
    split_offset = 0 if split == "train" else 10_000_000
    records = []
    started = time_module.perf_counter()
    print(f"[generate-synthetic] generating {split} split: {count} samples", flush=True)
    progress_step = 1 if count <= 20 else 10
    for index in range(count):
        seed = base_seed + split_offset + index
        background_id, source_values = backgrounds[index % len(backgrounds)]
        background_rng = np.random.default_rng(seed + 50_000_000)
        target_length = int(config.get("length", 1000))
        if len(source_values) >= target_length:
            maximum_start = len(source_values) - target_length
            source_start = int(background_rng.integers(0, maximum_start + 1))
            background = source_values[source_start : source_start + target_length].copy()
        else:
            source_start = 0
            old_axis = np.linspace(0.0, 1.0, len(source_values))
            new_axis = np.linspace(0.0, 1.0, target_length)
            background = np.interp(new_axis, old_axis, source_values)
        median = float(np.median(background))
        scale = float(np.median(np.abs(background - median)))
        if scale < 1e-6:
            scale = max(float(np.std(background)), 1.0)
        background = ((background - median) / scale).astype(np.float32)
        background *= float(background_rng.uniform(0.8, 1.2))
        background += background_rng.normal(0.0, 0.01, target_length).astype(np.float32)
        forced_type = anomaly_types[index % len(anomaly_types)]
        series, intervals, used_types, parameters = generate_sample(
            length=target_length,
            anomaly_types=anomaly_types,
            normal_probability=float(config.get("normal_probability", 0.2)),
            max_intervals=int(config.get("max_intervals", 3)),
            seed=seed,
            forced_type=forced_type,
            background=background,
        )
        sample_id = f"syn_{split}_{index:05d}"
        series_path = series_dir / f"{sample_id}.npy"
        image_path = image_dir / f"{sample_id}.png"
        np.save(series_path, series)
        render_series(series, image_path, 0, render_config)
        records.append(
            {
                "sample_id": sample_id,
                "series_id": sample_id,
                "source": "synthetic",
                "split": split,
                "length": len(series),
                "window_start": 0,
                "window_end": len(series),
                "series_path": str(series_path.resolve()),
                "image_path": str(image_path.resolve()),
                "intervals": to_jsonable(intervals),
                "anomaly_types": used_types,
                "generation_seed": seed,
                "background_source": "ucr_normal_prefix",
                "background_series_id": background_id,
                "background_start": source_start,
                "generation_parameters": parameters,
            }
        )
        if index == 0 or (index + 1) % progress_step == 0 or index + 1 == count:
            completed = index + 1
            elapsed = time_module.perf_counter() - started
            average = elapsed / completed
            eta = average * (count - completed)
            print(
                f"[generate-synthetic] {split} {completed}/{count} {sample_id} "
                f"elapsed={elapsed:.1f}s eta={eta:.1f}s",
                flush=True,
            )
    manifest_path = output_dir / f"{split}.jsonl"
    write_jsonl(manifest_path, records)
    print(
        f"[generate-synthetic] {split} done: {len(records)} records -> {manifest_path}; "
        f"elapsed={time_module.perf_counter() - started:.1f}s",
        flush=True,
    )
    return manifest_path


def _ucr_backgrounds(ucr_config: Dict[str, Any]) -> List[Tuple[str, np.ndarray]]:
    raw_dir = Path(ucr_config["raw_dir"])
    files = sorted(raw_dir.glob("*.txt"))
    max_series = ucr_config.get("max_series")
    if max_series is not None:
        files = files[: int(max_series)]
    backgrounds: List[Tuple[str, np.ndarray]] = []
    index_base = int(ucr_config.get("filename_index_base", 0))
    for path in files:
        metadata = parse_ucr_filename(path)
        values = load_series(path)
        if bool(ucr_config.get("train_end_is_count", True)):
            train_end = int(metadata["train_end_raw"])
        else:
            train_end = int(metadata["train_end_raw"]) - index_base + 1
        if not (8 <= train_end <= len(values)):
            raise ValueError(f"invalid UCR normal prefix for {path.name}: {train_end}")
        backgrounds.append((f"ucr_{metadata['archive_id']}", values[:train_end]))
    if len(backgrounds) < 2:
        raise ValueError("at least two UCR normal prefixes are required for train/val synthesis")
    return backgrounds


def split_backgrounds(
    backgrounds: Sequence[Tuple[str, np.ndarray]], seed: int
) -> Tuple[List[Tuple[str, np.ndarray]], List[Tuple[str, np.ndarray]]]:
    if len(backgrounds) < 2:
        raise ValueError("at least two backgrounds are required")
    order = np.random.default_rng(seed).permutation(len(backgrounds))
    shuffled = [backgrounds[int(index)] for index in order]
    split_index = max(1, min(len(shuffled) - 1, int(round(len(shuffled) * 0.8))))
    return shuffled[:split_index], shuffled[split_index:]


def generate_dataset(
    config: Dict[str, Any], render_config: Dict[str, Any], seed: int, ucr_config: Dict[str, Any]
) -> List[Path]:
    if config.get("background_source") != "ucr_normal_prefix":
        raise ValueError("synthetic.background_source must be ucr_normal_prefix")
    backgrounds = _ucr_backgrounds(ucr_config)
    train_backgrounds, val_backgrounds = split_backgrounds(backgrounds, seed)
    return [
        generate_split(
            "train", int(config["train_samples"]), config, render_config, seed, train_backgrounds
        ),
        generate_split(
            "val", int(config["val_samples"]), config, render_config, seed, val_backgrounds
        ),
    ]
