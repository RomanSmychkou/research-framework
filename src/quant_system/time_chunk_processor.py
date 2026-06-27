"""Time-bounded processing: split ranges into fixed ClickHouse compute chunks."""

from __future__ import annotations

import re
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

_DURATION_RE = re.compile(r"^(\d+(?:\.\d+)?)([dhms])$", re.IGNORECASE)


def parse_duration(value: str | None) -> timedelta | None:
    """Parse duration like 7d, 12h, 30m, 90s. none/off/0/all -> None."""
    v = (value or "").strip().lower()
    if not v or v in ("none", "off", "0", "false", "all"):
        return None
    m = _DURATION_RE.match(v)
    if not m:
        raise ValueError(
            f"Invalid duration {value!r}: expected none or like 7d, 12h, 30m, 120s"
        )
    num = float(m.group(1))
    if num <= 0:
        raise ValueError("duration must be positive")
    unit = m.group(2).lower()
    if unit == "d":
        return timedelta(days=num)
    if unit == "h":
        return timedelta(hours=num)
    if unit == "m":
        return timedelta(minutes=num)
    return timedelta(seconds=num)


def format_timedelta(td: timedelta) -> str:
    total_s = int(td.total_seconds())
    if total_s >= 86400 and total_s % 86400 == 0:
        return f"{total_s // 86400}d"
    if total_s >= 3600 and total_s % 3600 == 0:
        return f"{total_s // 3600}h"
    if total_s >= 60 and total_s % 60 == 0:
        return f"{total_s // 60}m"
    return f"{total_s}s"


def _as_utc_naive(dt: datetime) -> datetime:
    if dt.tzinfo is not None:
        return dt.astimezone(timezone.utc).replace(tzinfo=None)
    return dt


def _dt_ch_param(dt: datetime) -> str:
    """Stable datetime string for ClickHouse params (avoid tz-aware coercion quirks)."""
    return _as_utc_naive(dt).strftime("%Y-%m-%d %H:%M:%S.%f")


@dataclass(frozen=True)
class RowCountTimeBatchSpec:
    """One compute batch constrained by max source rows and represented as [start, end)."""

    index: int
    start: datetime
    end: datetime
    row_count: int
    is_last: bool

    @property
    def label(self) -> str:
        return f"row-batch {self.index + 1}"


def ensure_aware_utc(dt: datetime | None) -> datetime | None:
    if dt is None:
        return None
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def exchange_ts_for_ch(dt: datetime | None) -> datetime | None:
    """UTC naive ``exchange_ts`` for ClickHouse params and batching (no tz mix in min/max)."""
    if dt is None:
        return None
    return _as_utc_naive(ensure_aware_utc(dt))


def subtract_timedelta(dt: datetime, delta: timedelta) -> datetime:
    out = _as_utc_naive(dt) - delta
    return out.replace(tzinfo=timezone.utc) if dt.tzinfo else out


def add_timedelta(dt: datetime, delta: timedelta) -> datetime:
    out = _as_utc_naive(dt) + delta
    return out.replace(tzinfo=timezone.utc) if dt.tzinfo else out


def load_bounds_sql(
    client: Any,
    database: str,
    table: str,
    symbol: str,
    *,
    start: datetime | None = None,
    end: datetime | None = None,
    ts_col: str = "exchange_ts",
) -> tuple[datetime | None, datetime | None]:
    from .clickhouse_io import load_table_bounds

    return load_table_bounds(
        client,
        database,
        table,
        symbol,
        start=start,
        end=end,
        ts_col=ts_col,
    )


def plan_row_count_time_batches(
    client: Any,
    *,
    database: str,
    table: str,
    symbol: str,
    max_rows_batch: int,
    start: datetime | None = None,
    end: datetime | None = None,
    ts_col: str = "exchange_ts",
) -> list[RowCountTimeBatchSpec]:
    """
    Build [start, end) compute windows so each window covers at most `max_rows_batch`
    rows when possible (never splits a single timestamp across batches).
    """
    if max_rows_batch <= 0:
        raise ValueError("max_rows_batch must be positive")

    out = list(
        _iter_row_count_time_batches(
            client,
            database=database,
            table=table,
            symbol=symbol,
            max_rows_batch=max_rows_batch,
            start=start,
            end=end,
            ts_col=ts_col,
        )
    )
    for idx, batch in enumerate(out):
        out[idx] = RowCountTimeBatchSpec(
            index=idx,
            start=batch.start,
            end=batch.end,
            row_count=batch.row_count,
            is_last=idx == len(out) - 1,
        )
    return out


def _iter_row_count_time_batches(
    client: Any,
    *,
    database: str,
    table: str,
    symbol: str,
    max_rows_batch: int,
    start: datetime | None = None,
    end: datetime | None = None,
    ts_col: str = "exchange_ts",
) -> Iterator[RowCountTimeBatchSpec]:
    """Yield row-count bounded [start, end) windows iteratively from current cursor."""
    if max_rows_batch <= 0:
        raise ValueError("max_rows_batch must be positive")

    start = exchange_ts_for_ch(start)
    end = exchange_ts_for_ch(end)
    bounds_sql = f"""
    SELECT min({ts_col}), max({ts_col})
    FROM {database}.{table}
    WHERE symbol = %(symbol)s
    """
    bounds_params: dict[str, Any] = {"symbol": symbol}
    if start is not None:
        bounds_sql += f"\n  AND {ts_col} >= %(start_ts)s"
        bounds_params["start_ts"] = _dt_ch_param(start)
    if end is not None:
        bounds_sql += f"\n  AND {ts_col} < %(end_ts)s"
        bounds_params["end_ts"] = _dt_ch_param(end)
    bounds_row = client.execute(bounds_sql, bounds_params)
    if not bounds_row or bounds_row[0][0] is None:
        return
    t_min = exchange_ts_for_ch(bounds_row[0][0])
    t_max = exchange_ts_for_ch(bounds_row[0][1])
    if t_min is None or t_max is None:
        return

    effective_start = max(t_min, start) if start is not None else t_min
    # ClickHouse DateTime64 precision can truncate microseconds in params.
    # Keep a safe exclusive cap beyond max timestamp to include all rows at t_max.
    data_end_exclusive = add_timedelta(t_max, timedelta(seconds=1))
    effective_end = min(data_end_exclusive, end) if end is not None else data_end_exclusive
    if effective_start >= effective_end:
        return

    offset = max_rows_batch - 1
    cur = effective_start
    idx = 0
    while cur < effective_end:
        where = f"symbol = %(symbol)s AND {ts_col} >= %(start_ts)s AND {ts_col} < %(end_ts)s"
        params: dict[str, Any] = {
            "symbol": symbol,
            "start_ts": _dt_ch_param(cur),
            "end_ts": _dt_ch_param(effective_end),
        }
        boundary_sql = f"""
        SELECT {ts_col}
        FROM {database}.{table}
        WHERE {where}
        ORDER BY {ts_col}
        LIMIT 1 OFFSET %(row_offset)s
        """
        boundary_params = dict(params)
        boundary_params["row_offset"] = offset
        boundary_row = client.execute(boundary_sql, boundary_params)

        if boundary_row:
            boundary_ts = exchange_ts_for_ch(boundary_row[0][0])
            next_sql = f"""
            SELECT min({ts_col})
            FROM {database}.{table}
            WHERE symbol = %(symbol)s
              AND {ts_col} > %(boundary_ts)s
              AND {ts_col} < %(upper_bound)s
            """
            next_params: dict[str, Any] = {
                "symbol": symbol,
                "boundary_ts": _dt_ch_param(boundary_ts),
                "upper_bound": _dt_ch_param(effective_end),
            }
            next_row = client.execute(next_sql, next_params)
            next_ts = (
                exchange_ts_for_ch(next_row[0][0])
                if next_row and next_row[0][0] is not None
                else None
            )
            batch_end = next_ts if next_ts is not None else effective_end
        else:
            batch_end = effective_end

        count_sql = f"""
        SELECT count()
        FROM {database}.{table}
        WHERE symbol = %(symbol)s
          AND {ts_col} >= %(start_ts)s
          AND {ts_col} < %(end_ts)s
        """
        count_params: dict[str, Any] = {
            "symbol": symbol,
            "start_ts": _dt_ch_param(cur),
            "end_ts": _dt_ch_param(batch_end),
        }
        row_count = int(client.execute(count_sql, count_params)[0][0])
        if row_count <= 0:
            break

        yield RowCountTimeBatchSpec(
            index=idx,
            start=cur,
            end=batch_end,
            row_count=row_count,
            is_last=False,
        )
        idx += 1
        cur = batch_end


def run_row_count_time_batched(
    client: Any,
    *,
    database: str,
    table: str,
    symbol: str,
    max_rows_batch: int,
    handler: Callable[[RowCountTimeBatchSpec], None],
    start: datetime | None = None,
    end: datetime | None = None,
    ts_col: str = "exchange_ts",
    log: Callable[[str], None] | None = None,
) -> None:
    """Run handler for each row-count bounded time batch."""
    batches_iter = _iter_row_count_time_batches(
        client,
        database=database,
        table=table,
        symbol=symbol,
        max_rows_batch=max_rows_batch,
        start=start,
        end=end,
        ts_col=ts_col,
    )
    log_fn = log or (lambda _msg: None)
    prev: RowCountTimeBatchSpec | None = None
    for raw_batch in batches_iter:
        if prev is not None:
            batch = RowCountTimeBatchSpec(
                index=prev.index,
                start=prev.start,
                end=prev.end,
                row_count=prev.row_count,
                is_last=False,
            )
            log_fn(
                f"{batch.label} rows={batch.row_count} "
                f"emit=[{batch.start.isoformat()}, {batch.end.isoformat()})"
            )
            handler(batch)
        prev = raw_batch
    if prev is None:
        return
    last_batch = RowCountTimeBatchSpec(
        index=prev.index,
        start=prev.start,
        end=prev.end,
        row_count=prev.row_count,
        is_last=True,
    )
    log_fn(
        f"{last_batch.label} rows={last_batch.row_count} "
        f"emit=[{last_batch.start.isoformat()}, {last_batch.end.isoformat()})"
    )
    handler(last_batch)
