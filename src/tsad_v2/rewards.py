from __future__ import annotations

from typing import Any, Dict, List, Sequence

from .intervals import canonicalize, parse_interval_output
from .metrics import evaluate_sample


def completion_text(completion: Any) -> str:
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list) and completion:
        first = completion[0]
        if isinstance(first, dict):
            content = first.get("content", "")
            if isinstance(content, str):
                return content
    if isinstance(completion, dict):
        content = completion.get("content", "")
        if isinstance(content, str):
            return content
    return str(completion)


def interval_quality_reward(
    output: str,
    ground_truth: Sequence[Any],
    lower: int,
    upper: int,
) -> float:
    parsed = parse_interval_output(output, lower=lower, upper=upper)
    if not parsed.valid:
        return 0.0
    truth = canonicalize(ground_truth, lower=lower, upper=upper)
    metrics, _ = evaluate_sample(parsed.intervals, truth, lower, upper, parse_valid=True)
    count_score = 1.0 - min(
        1.0,
        abs(len(parsed.intervals) - len(truth)) / max(1, len(parsed.intervals), len(truth)),
    )
    if not parsed.intervals and not truth:
        boundary_score = 1.0
    elif metrics["boundary_mae"] is None:
        boundary_score = 0.0
    else:
        boundary_score = max(0.0, 1.0 - metrics["boundary_mae"] / max(1, upper - lower))
    return float(
        0.10
        + 0.40 * metrics["point_f1"]
        + 0.25 * metrics["event_f1"]
        + 0.15 * metrics["mean_matched_iou"]
        + 0.05 * count_score
        + 0.05 * boundary_score
    )


def format_reward_func(completions: Sequence[Any], **_: Any) -> List[float]:
    return [1.0 if parse_interval_output(completion_text(item)).valid else 0.0 for item in completions]


def quality_reward_func(
    completions: Sequence[Any],
    ground_truth: Sequence[Sequence[Dict[str, int]]],
    window_start: Sequence[int],
    window_end: Sequence[int],
    **_: Any,
) -> List[float]:
    return [
        interval_quality_reward(completion_text(completion), truth, int(start), int(end))
        for completion, truth, start, end in zip(
            completions, ground_truth, window_start, window_end
        )
    ]

