from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Any, Iterable, List, Optional, Sequence, Tuple

Interval = Tuple[int, int]


@dataclass(frozen=True)
class ParseResult:
    intervals: List[Interval]
    valid: bool
    error: Optional[str] = None


def _balanced_array(text: str) -> Optional[str]:
    start = text.find("[")
    if start < 0:
        return None
    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "[":
            depth += 1
        elif char == "]":
            depth -= 1
            if depth == 0:
                return text[start : index + 1]
    return None


def _as_int(value: Any) -> int:
    if isinstance(value, bool):
        raise ValueError("boolean is not an index")
    if isinstance(value, int):
        return value
    if isinstance(value, float) and math.isfinite(value) and value.is_integer():
        return int(value)
    raise ValueError(f"index must be an integer, got {value!r}")


def canonicalize(
    intervals: Iterable[Any],
    lower: Optional[int] = None,
    upper: Optional[int] = None,
    merge_touching: bool = True,
) -> List[Interval]:
    parsed: List[Interval] = []
    for item in intervals:
        if isinstance(item, dict):
            if set(item) != {"start", "end"}:
                raise ValueError("interval objects must contain only start and end")
            start, end = _as_int(item["start"]), _as_int(item["end"])
        elif isinstance(item, (list, tuple)) and len(item) == 2:
            start, end = _as_int(item[0]), _as_int(item[1])
        else:
            raise ValueError(f"invalid interval: {item!r}")
        if lower is not None:
            start, end = max(start, lower), max(end, lower)
        if upper is not None:
            start, end = min(start, upper), min(end, upper)
        if start >= end:
            raise ValueError(f"interval must satisfy start < end: {(start, end)}")
        parsed.append((start, end))

    parsed.sort()
    merged: List[Interval] = []
    for start, end in parsed:
        if merged and (start <= merged[-1][1] if merge_touching else start < merged[-1][1]):
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def parse_interval_output(
    output: str,
    lower: Optional[int] = None,
    upper: Optional[int] = None,
) -> ParseResult:
    if not isinstance(output, str):
        return ParseResult([], False, "output is not a string")
    candidate = _balanced_array(output.strip())
    if candidate is None:
        return ParseResult([], False, "no JSON array found")
    try:
        value = json.loads(candidate)
        if not isinstance(value, list):
            raise ValueError("output must be a JSON list")
        result = canonicalize(value, lower=lower, upper=upper)
    except (json.JSONDecodeError, ValueError, TypeError) as exc:
        return ParseResult([], False, str(exc))
    return ParseResult(result, True, None)


def to_jsonable(intervals: Sequence[Interval]) -> List[dict]:
    return [{"start": int(start), "end": int(end)} for start, end in intervals]


def to_json(intervals: Sequence[Interval]) -> str:
    return json.dumps(to_jsonable(intervals), ensure_ascii=False, separators=(",", ":"))


def length(intervals: Sequence[Interval]) -> int:
    return sum(end - start for start, end in intervals)


def intersection_length(left: Sequence[Interval], right: Sequence[Interval]) -> int:
    i = j = total = 0
    while i < len(left) and j < len(right):
        start = max(left[i][0], right[j][0])
        end = min(left[i][1], right[j][1])
        if start < end:
            total += end - start
        if left[i][1] <= right[j][1]:
            i += 1
        else:
            j += 1
    return total


def clip(intervals: Sequence[Interval], lower: int, upper: int) -> List[Interval]:
    clipped = []
    for start, end in intervals:
        start, end = max(start, lower), min(end, upper)
        if start < end:
            clipped.append((start, end))
    return canonicalize(clipped, lower=lower, upper=upper)

