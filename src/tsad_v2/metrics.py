from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from .intervals import Interval, canonicalize, intersection_length, length


def _safe_div(numerator: float, denominator: float, empty_value: float = 0.0) -> float:
    return numerator / denominator if denominator else empty_value


def _f1(precision: float, recall: float) -> float:
    return _safe_div(2.0 * precision * recall, precision + recall)


def interval_iou(left: Interval, right: Interval) -> float:
    intersection = max(0, min(left[1], right[1]) - max(left[0], right[0]))
    union = (left[1] - left[0]) + (right[1] - right[0]) - intersection
    return _safe_div(intersection, union)


def match_intervals(
    predicted: Sequence[Interval], target: Sequence[Interval]
) -> List[Tuple[int, int, float]]:
    candidates = []
    for pred_index, pred in enumerate(predicted):
        for target_index, truth in enumerate(target):
            score = interval_iou(pred, truth)
            if score > 0:
                candidates.append((score, pred_index, target_index))
    candidates.sort(reverse=True)
    used_pred, used_target, matches = set(), set(), []
    for score, pred_index, target_index in candidates:
        if pred_index in used_pred or target_index in used_target:
            continue
        used_pred.add(pred_index)
        used_target.add(target_index)
        matches.append((pred_index, target_index, score))
    return matches


@dataclass
class SampleCounts:
    point_tp: int
    point_fp: int
    point_fn: int
    event_tp: int
    event_fp: int
    event_fn: int
    matched_iou_sum: float
    matched_boundary_error_sum: float
    matched_count: int


def evaluate_sample(
    predicted: Iterable[Any],
    target: Iterable[Any],
    series_start: int,
    series_end: int,
    parse_valid: bool = True,
) -> Tuple[Dict[str, Any], SampleCounts]:
    predicted_intervals = canonicalize(predicted, lower=series_start, upper=series_end)
    target_intervals = canonicalize(target, lower=series_start, upper=series_end)
    tp = intersection_length(predicted_intervals, target_intervals)
    pred_points, target_points = length(predicted_intervals), length(target_intervals)
    fp, fn = pred_points - tp, target_points - tp
    point_precision = _safe_div(tp, tp + fp, 1.0 if target_points == 0 else 0.0)
    point_recall = _safe_div(tp, tp + fn, 1.0 if pred_points == 0 else 0.0)
    point_f1 = _f1(point_precision, point_recall)
    if not predicted_intervals and not target_intervals:
        point_f1 = 1.0

    matches = match_intervals(predicted_intervals, target_intervals)
    event_tp = len(matches)
    event_fp = len(predicted_intervals) - event_tp
    event_fn = len(target_intervals) - event_tp
    event_precision = _safe_div(event_tp, event_tp + event_fp, 1.0 if not target_intervals else 0.0)
    event_recall = _safe_div(event_tp, event_tp + event_fn, 1.0 if not predicted_intervals else 0.0)
    event_f1 = _f1(event_precision, event_recall)
    if not predicted_intervals and not target_intervals:
        event_f1 = 1.0

    iou_sum = sum(match[2] for match in matches)
    boundary_sum = sum(
        abs(predicted_intervals[pred_index][0] - target_intervals[target_index][0])
        + abs(predicted_intervals[pred_index][1] - target_intervals[target_index][1])
        for pred_index, target_index, _ in matches
    )
    mean_iou = _safe_div(iou_sum, len(matches), 1.0 if not target_intervals and not predicted_intervals else 0.0)
    boundary_mae = _safe_div(boundary_sum, 2 * len(matches)) if matches else None
    metrics = {
        "parse_valid": bool(parse_valid),
        "point_precision": point_precision,
        "point_recall": point_recall,
        "point_f1": point_f1,
        "event_precision": event_precision,
        "event_recall": event_recall,
        "event_f1": event_f1,
        "mean_matched_iou": mean_iou,
        "boundary_mae": boundary_mae,
        "normal_exact": bool(not predicted_intervals and not target_intervals),
        "predicted_event_count": len(predicted_intervals),
        "target_event_count": len(target_intervals),
    }
    counts = SampleCounts(
        point_tp=tp,
        point_fp=fp,
        point_fn=fn,
        event_tp=event_tp,
        event_fp=event_fp,
        event_fn=event_fn,
        matched_iou_sum=iou_sum,
        matched_boundary_error_sum=boundary_sum,
        matched_count=len(matches),
    )
    return metrics, counts


def aggregate(rows: Sequence[Tuple[Dict[str, Any], SampleCounts]]) -> Dict[str, Any]:
    if not rows:
        raise ValueError("cannot aggregate an empty evaluation")
    counts = [item[1] for item in rows]
    point_tp = sum(item.point_tp for item in counts)
    point_fp = sum(item.point_fp for item in counts)
    point_fn = sum(item.point_fn for item in counts)
    event_tp = sum(item.event_tp for item in counts)
    event_fp = sum(item.event_fp for item in counts)
    event_fn = sum(item.event_fn for item in counts)
    point_precision = _safe_div(point_tp, point_tp + point_fp, 1.0 if point_fn == 0 else 0.0)
    point_recall = _safe_div(point_tp, point_tp + point_fn, 1.0 if point_fp == 0 else 0.0)
    event_precision = _safe_div(event_tp, event_tp + event_fp, 1.0 if event_fn == 0 else 0.0)
    event_recall = _safe_div(event_tp, event_tp + event_fn, 1.0 if event_fp == 0 else 0.0)
    matched_count = sum(item.matched_count for item in counts)
    normal_rows = [metrics for metrics, _ in rows if metrics["target_event_count"] == 0]
    return {
        "samples": len(rows),
        "parse_rate": sum(int(metrics["parse_valid"]) for metrics, _ in rows) / len(rows),
        "point_precision": point_precision,
        "point_recall": point_recall,
        "point_f1": _f1(point_precision, point_recall),
        "event_precision": event_precision,
        "event_recall": event_recall,
        "event_f1": _f1(event_precision, event_recall),
        "mean_matched_iou": _safe_div(
            sum(item.matched_iou_sum for item in counts), matched_count
        ),
        "boundary_mae": _safe_div(
            sum(item.matched_boundary_error_sum for item in counts), 2 * matched_count
        )
        if matched_count
        else None,
        "normal_accuracy": (
            sum(int(metrics["normal_exact"]) for metrics in normal_rows) / len(normal_rows)
            if normal_rows
            else None
        ),
        "point_tp": point_tp,
        "point_fp": point_fp,
        "point_fn": point_fn,
        "event_tp": event_tp,
        "event_fp": event_fp,
        "event_fn": event_fn,
    }

