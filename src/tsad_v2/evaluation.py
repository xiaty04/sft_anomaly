from __future__ import annotations

import csv
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

from .intervals import canonicalize, to_jsonable
from .io import read_jsonl, write_jsonl
from .metrics import SampleCounts, aggregate, evaluate_sample


def evaluate_predictions(manifest_path: Path, predictions_path: Path, output_dir: Path) -> Dict[str, Any]:
    started = time.perf_counter()
    records = read_jsonl(manifest_path)
    predictions = read_jsonl(predictions_path)
    print(
        f"[evaluate] manifest={manifest_path} series={len(records)} "
        f"predictions={len(predictions)} output={output_dir}",
        flush=True,
    )
    expected_ids = {record["sample_id"] for record in records}
    if len(expected_ids) != len(records):
        raise ValueError("evaluation manifest contains duplicate sample_id values")
    prediction_by_id: Dict[str, Dict[str, Any]] = {}
    for prediction in predictions:
        sample_id = prediction["sample_id"]
        if sample_id not in expected_ids:
            raise ValueError(
                f"prediction sample_id is not present in the series manifest: {sample_id}"
            )
        if sample_id in prediction_by_id:
            raise ValueError(f"duplicate prediction for series sample: {sample_id}")
        prediction_by_id[sample_id] = prediction
    rows: List[Tuple[Dict[str, Any], SampleCounts]] = []
    details = []
    progress_step = max(1, min(25, len(records) // 10 or 1))
    for index, record in enumerate(records, start=1):
        sample_id = record["sample_id"]
        prediction = prediction_by_id.get(sample_id)
        parse_valid = bool(prediction) and bool(prediction.get("parse_valid", False))
        predicted = prediction.get("intervals", []) if prediction else []
        start = int(record.get("eval_start", record.get("input_start", 0)))
        end = int(record.get("eval_end", record.get("input_end", record["length"])))
        predicted_intervals = canonicalize(predicted, lower=start, upper=end)
        metrics, counts = evaluate_sample(
            predicted_intervals,
            record.get("intervals", []),
            start,
            end,
            parse_valid=parse_valid,
        )
        rows.append((metrics, counts))
        details.append(
            {
                "sample_id": sample_id,
                "prediction_present": prediction is not None,
                "predicted_intervals": to_jsonable(predicted_intervals),
                "target_intervals": record.get("intervals", []),
                **metrics,
            }
        )
        if index == 1 or index % progress_step == 0 or index == len(records):
            print(f"[evaluate] series {index}/{len(records)} {sample_id}", flush=True)
    summary = aggregate(rows)
    summary.update(
        {
            "manifest": str(manifest_path),
            "predictions": str(predictions_path),
            "missing_samples": sum(int(not item["prediction_present"]) for item in details),
        }
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_dir / "per_sample.jsonl", details)
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    print(
        f"[evaluate] complete missing={summary['missing_samples']} "
        f"parse_rate={summary.get('parse_rate')} point_f1={summary.get('point_f1')} "
        f"event_f1={summary.get('event_f1')} elapsed={time.perf_counter() - started:.1f}s",
        flush=True,
    )
    return summary


def compare_reports(report_paths: Sequence[Path], output_path: Path) -> None:
    fields = [
        "method",
        "parse_rate",
        "point_precision",
        "point_recall",
        "point_f1",
        "event_precision",
        "event_recall",
        "event_f1",
        "mean_matched_iou",
        "boundary_mae",
        "normal_accuracy",
    ]
    rows = []
    for path in report_paths:
        with path.open("r", encoding="utf-8") as handle:
            report = json.load(handle)
        rows.append({"method": path.parent.name, **{key: report.get(key) for key in fields[1:]}})
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
