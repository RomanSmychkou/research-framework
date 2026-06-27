"""Reusable horizon calculations around ordered time boundaries.

The helpers here intentionally know nothing about train/validation/test or
global/local chunks. Callers pass a sorted timeline, a boundary index, and the
horizon direction they need. The returned index can then be used to materialize
embargo/purge ranges in whichever higher-level split markup is active.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Literal, Sequence

BoundarySide = Literal["left", "right"]


@dataclass(frozen=True)
class TrimmedChunk:
    """Chunk time window after applying purge/embargo trimming."""

    start: datetime
    end: datetime
    original_start: datetime
    original_end: datetime
    applied_horizon: timedelta
    extra_gap: timedelta
    operation: Literal["purge", "embargo"]

    @property
    def is_empty(self) -> bool:
        return self.start >= self.end

    def to_json(self) -> dict[str, object]:
        return {
            "start": self.start.isoformat(),
            "end": self.end.isoformat(),
            "original_start": self.original_start.isoformat(),
            "original_end": self.original_end.isoformat(),
            "applied_horizon_seconds": self.applied_horizon.total_seconds(),
            "extra_gap_seconds": self.extra_gap.total_seconds(),
            "operation": self.operation,
            "is_empty": self.is_empty,
        }


def _utc_naive(dt: datetime) -> datetime:
    if dt.tzinfo is not None:
        return dt.astimezone(timezone.utc).replace(tzinfo=None)
    return dt

def apply_embargo(
    chunk_start: datetime,
    chunk_end: datetime,
    embargo: timedelta,
    *,
    extra_gap: timedelta = timedelta(seconds=1),
) -> TrimmedChunk:
    """Trim chunk start forward by embargo horizon plus extra gap."""
    if chunk_start > chunk_end:
        raise ValueError("chunk_start must be <= chunk_end.")
    if embargo.total_seconds() < 0:
        raise ValueError("embargo must be non-negative.")
    if extra_gap.total_seconds() < 0:
        raise ValueError("extra_gap must be non-negative.")

    original_start = chunk_start
    original_end = chunk_end
    shift = embargo + extra_gap
    trimmed_start = chunk_start + shift
    if trimmed_start > chunk_end:
        trimmed_start = chunk_end
    return TrimmedChunk(
        start=trimmed_start,
        end=chunk_end,
        original_start=original_start,
        original_end=original_end,
        applied_horizon=embargo,
        extra_gap=extra_gap,
        operation="embargo",
    )


def apply_purge(
    chunk_start: datetime,
    chunk_end: datetime,
    purge: timedelta,
    *,
    extra_gap: timedelta = timedelta(seconds=1),
) -> TrimmedChunk:
    """Trim chunk end backward by purge horizon plus extra gap."""
    if chunk_start > chunk_end:
        raise ValueError("chunk_start must be <= chunk_end.")
    if purge.total_seconds() < 0:
        raise ValueError("purge must be non-negative.")
    if extra_gap.total_seconds() < 0:
        raise ValueError("extra_gap must be non-negative.")

    original_start = chunk_start
    original_end = chunk_end
    shift = purge + extra_gap
    trimmed_end = chunk_end - shift
    if trimmed_end < chunk_start:
        trimmed_end = chunk_start
    return TrimmedChunk(
        start=chunk_start,
        end=trimmed_end,
        original_start=original_start,
        original_end=original_end,
        applied_horizon=purge,
        extra_gap=extra_gap,
        operation="purge",
    )
