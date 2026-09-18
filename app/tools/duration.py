"""
app/tools/duration.py — Practice Duration Calculator Tool

Deterministic tool: sums the duration_minutes of each PoseItem in a sequence
and returns the total practice duration in minutes.

Tool contract:
    calculate_practice_duration(sequence: list[dict]) -> int
"""
from __future__ import annotations


def calculate_practice_duration(sequence: list[dict]) -> int:
    """
    Calculate the total duration of a practice sequence.

    Args:
        sequence: List of pose/activity dicts, each containing a
                  ``duration_minutes`` key (int or float).

    Returns:
        Total duration in whole minutes (sum of all pose durations).
        Returns 0 for an empty or invalid sequence.
    """
    if not sequence:
        return 0

    total = 0
    for item in sequence:
        try:
            total += int(item.get("duration_minutes", 0))
        except (TypeError, ValueError):
            # Skip items with non-numeric duration rather than crashing
            continue

    return total
