from __future__ import annotations


def interval_prompt(start: int, end: int) -> str:
    return (
        "Detect every anomalous interval in the time-series plot. "
        f"The visible x-axis coordinates run from {start} to {end - 1}. "
        "Use half-open intervals [start, end), so a single anomalous point i is "
        "represented as {\"start\": i, \"end\": i+1}. "
        "Return only a JSON list ordered by time, with no explanation. "
        "Use [] when there is no anomaly. Example: "
        '[{"start":120,"end":145}]'
    )

