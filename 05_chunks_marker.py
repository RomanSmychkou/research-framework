"""Mark train/validation/test chunks and persist session boundaries to ClickHouse."""

from __future__ import annotations

import argparse
import bisect
import hashlib
import subprocess
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from statistics import mean, median, stdev
from typing import Any

import bootstrap_path

bootstrap_path.ensure_src_on_path(Path(__file__))

from quant_system import quant_env
from quant_system.clickhouse_insert import clickhouse_insert_rows
from quant_system.clickhouse_io import build_clickhouse_client, parse_cli_datetime
from quant_system.time_chunk_processor import exchange_ts_for_ch, parse_duration


@dataclass(frozen=True)
class TimeRange:
    start: datetime
    end: datetime


@dataclass(frozen=True)
class ChunkBoundary:
    start: datetime
    end: datetime


@dataclass(frozen=True)
class ChunkInstance:
    source_table: str
    chunk_start: datetime
    chunk_end: datetime
    chunk_type: str
    chunk_signature_hash: str


@dataclass(frozen=True)
class EffectiveAxis:
    segment_starts: list[datetime]
    segment_real_us: list[int]
    segment_effective_us: list[int]
    effective_prefix_end_us: list[int]
    total_real_us: int
    total_effective_us: int


def _hash_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _iso_utc(dt: datetime) -> str:
    if dt.tzinfo is not None:
        dt = dt.astimezone(timezone.utc).replace(tzinfo=None)
    return dt.isoformat(timespec="milliseconds") + "Z"


def _format_days_seconds_from_seconds(total_seconds: float) -> str:
    total_seconds_int = int(round(total_seconds))
    sign = "-" if total_seconds_int < 0 else ""
    abs_seconds = abs(total_seconds_int)
    days = abs_seconds // 86_400
    remainder = abs_seconds % 86_400
    hours = remainder // 3_600
    remainder %= 3_600
    minutes = remainder // 60
    seconds = remainder % 60
    return f"{sign}{days}d {hours:02d}:{minutes:02d}:{seconds:02d}"


def _format_days_seconds(delta: timedelta) -> str:
    return _format_days_seconds_from_seconds(delta.total_seconds())


def _parse_csv_list(raw: str) -> list[str]:
    return [p.strip() for p in raw.split(",") if p.strip()]


def _table_exists(client: Any, database: str, table: str) -> bool:
    sql = """
SELECT count()
FROM system.tables
WHERE database = %(database)s
  AND name = %(table)s
"""
    return int(client.execute(sql, {"database": database, "table": table})[0][0]) > 0


def _require_tables(client: Any, database: str, tables: list[str]) -> None:
    missing = [t for t in tables if not _table_exists(client, database, t)]
    if missing:
        joined = ", ".join(missing)
        raise SystemExit(
            f"Missing source table(s) in {database}: {joined}. "
            "Pass existing --tables-to-mark list."
        )
    if not _table_exists(client, database, "chunks_instances"):
        raise SystemExit(
            f"Missing table {database}.chunks_instances. "
            "Apply containers/clickhouse_db/.../01-schema.sql."
        )
    if not _table_exists(client, database, "chunks_marking_results"):
        raise SystemExit(
            f"Missing table {database}.chunks_marking_results. "
            "Apply containers/clickhouse_db/.../01-schema.sql."
        )
    if not _table_exists(client, database, "manual_marked_chunks"):
        raise SystemExit(
            f"Missing table {database}.manual_marked_chunks. "
            "Apply containers/clickhouse_db/.../01-schema.sql."
        )


def _load_effective_bounds(
    client: Any,
    database: str,
    table: str,
    symbol: str,
    start: datetime,
    end: datetime,
) -> TimeRange:
    sql = f"""
SELECT
    minIf(exchange_ts, exchange_ts >= %(start)s AND exchange_ts <= %(end)s),
    maxIf(exchange_ts, exchange_ts >= %(start)s AND exchange_ts <= %(end)s)
FROM {database}.{table}
WHERE symbol = %(symbol)s
"""
    row = client.execute(
        sql,
        {"symbol": symbol, "start": exchange_ts_for_ch(start), "end": exchange_ts_for_ch(end)},
    )
    min_ts, max_ts = row[0]
    if min_ts is None or max_ts is None:
        raise SystemExit(
            f"No rows in [{_iso_utc(start)}, {_iso_utc(end)}] "
            f"for {database}.{table}, symbol={symbol}."
        )
    return TimeRange(start=exchange_ts_for_ch(min_ts), end=exchange_ts_for_ch(max_ts))


def _load_anchor_timeline(
    client: Any,
    database: str,
    table: str,
    symbol: str,
    start: datetime,
    end: datetime,
) -> list[datetime]:
    sql = f"""
SELECT DISTINCT exchange_ts
FROM {database}.{table}
WHERE symbol = %(symbol)s
  AND exchange_ts >= %(start)s
  AND exchange_ts <= %(end)s
ORDER BY exchange_ts
"""
    rows = client.execute(
        sql,
        {"symbol": symbol, "start": exchange_ts_for_ch(start), "end": exchange_ts_for_ch(end)},
    )
    timeline = [exchange_ts_for_ch(row[0]) for row in rows if row and row[0] is not None]
    if len(timeline) < 2:
        raise SystemExit(
            "Anchor timeline has less than 2 distinct timestamps. "
            "Cannot split by time. Expand range or choose another source table."
        )
    return timeline


def _effective_sql_expr(
    *,
    real_expr: str,
    strip_gap_threshold_us: int | None,
) -> str:
    if strip_gap_threshold_us is None:
        return real_expr
    return f"if({real_expr} > %(strip_gap_threshold_us)s, toInt64(0), {real_expr})"


def _load_effective_span_stats_sql(
    client: Any,
    database: str,
    table: str,
    symbol: str,
    start: datetime,
    end: datetime,
    strip_gap_threshold_us: int | None,
) -> tuple[int, int]:
    real_expr = (
        "if(prev_ts IS NULL, toInt64(0), "
        "greatest(toInt64(dateDiff('microsecond', prev_ts, exchange_ts)), toInt64(0)))"
    )
    eff_expr = _effective_sql_expr(real_expr=real_expr, strip_gap_threshold_us=strip_gap_threshold_us)
    sql = f"""
WITH timeline AS (
    SELECT DISTINCT exchange_ts
    FROM {database}.{table}
    WHERE symbol = %(symbol)s
      AND exchange_ts >= %(start)s
      AND exchange_ts <= %(end)s
),
ordered AS (
    SELECT
        exchange_ts,
        lagInFrame(exchange_ts)
            OVER (ORDER BY exchange_ts ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS prev_ts
    FROM timeline
),
segments AS (
    SELECT
        {real_expr} AS real_delta_us,
        {eff_expr} AS eff_delta_us
    FROM ordered
)
SELECT
    toInt64(sum(real_delta_us)) AS total_real_us,
    toInt64(sum(eff_delta_us)) AS total_effective_us
FROM segments
"""
    params: dict[str, Any] = {
        "symbol": symbol,
        "start": exchange_ts_for_ch(start),
        "end": exchange_ts_for_ch(end),
    }
    if strip_gap_threshold_us is not None:
        params["strip_gap_threshold_us"] = int(strip_gap_threshold_us)
    row = client.execute(sql, params)
    if not row:
        return 0, 0
    return int(row[0][0] or 0), int(row[0][1] or 0)


def _load_effective_boundaries_sql(
    client: Any,
    database: str,
    table: str,
    symbol: str,
    start: datetime,
    end: datetime,
    n_chunks: int,
    strip_gap_threshold_us: int | None,
) -> list[datetime]:
    if n_chunks <= 1:
        return []
    real_expr = (
        "if(prev_ts IS NULL, toInt64(0), "
        "greatest(toInt64(dateDiff('microsecond', prev_ts, exchange_ts)), toInt64(0)))"
    )
    eff_expr = _effective_sql_expr(real_expr=real_expr, strip_gap_threshold_us=strip_gap_threshold_us)
    sql = f"""
WITH timeline AS (
    SELECT DISTINCT exchange_ts
    FROM {database}.{table}
    WHERE symbol = %(symbol)s
      AND exchange_ts >= %(start)s
      AND exchange_ts <= %(end)s
),
ordered AS (
    SELECT
        exchange_ts,
        lagInFrame(exchange_ts)
            OVER (ORDER BY exchange_ts ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS prev_ts
    FROM timeline
),
segments AS (
    SELECT
        prev_ts AS seg_start,
        exchange_ts AS seg_end,
        {eff_expr} AS eff_delta_us
    FROM ordered
),
cum AS (
    SELECT
        seg_start,
        seg_end,
        eff_delta_us,
        sum(eff_delta_us)
            OVER (ORDER BY seg_end ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS cum_eff_us,
        sum(eff_delta_us)
            OVER (ORDER BY seg_end ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) - eff_delta_us
            AS prev_cum_eff_us
    FROM segments
),
totals AS (
    SELECT toInt64(max(cum_eff_us)) AS total_eff_us
    FROM cum
),
targets AS (
    SELECT
        arrayJoin(range(1, %(n_chunks)s)) AS chunk_idx,
        intDiv(total_eff_us * chunk_idx, %(n_chunks)s) AS target_eff_us
    FROM totals
)
SELECT
    t.chunk_idx,
    minIf(
        c.seg_start + toIntervalMicrosecond(t.target_eff_us - c.prev_cum_eff_us),
        c.eff_delta_us > 0
        AND t.target_eff_us > c.prev_cum_eff_us
        AND t.target_eff_us <= c.cum_eff_us
    ) AS boundary_ts
FROM targets AS t
CROSS JOIN cum AS c
GROUP BY t.chunk_idx, t.target_eff_us
ORDER BY t.chunk_idx
"""
    params: dict[str, Any] = {
        "symbol": symbol,
        "start": exchange_ts_for_ch(start),
        "end": exchange_ts_for_ch(end),
        "n_chunks": n_chunks,
    }
    if strip_gap_threshold_us is not None:
        params["strip_gap_threshold_us"] = int(strip_gap_threshold_us)
    rows = client.execute(sql, params)
    out: list[datetime] = []
    for row in rows:
        boundary = row[1]
        if boundary is None:
            raise SystemExit(
                "Failed to map effective-time boundary to real timestamp. "
                "Try smaller --n-chunks-united or disable --strip-internal-gaps."
            )
        out.append(exchange_ts_for_ch(boundary))
    return out


def _effective_segment_us(delta_us: int, strip_gap_threshold_us: int | None) -> int:
    if strip_gap_threshold_us is None:
        return delta_us
    if delta_us > strip_gap_threshold_us:
        return 0
    return delta_us


def _map_effective_offset_to_real_ts(
    axis: EffectiveAxis,
    target_effective_us: int,
) -> datetime:
    if target_effective_us <= 0:
        return axis.segment_starts[0]
    idx = bisect.bisect_left(axis.effective_prefix_end_us, target_effective_us)
    if idx >= len(axis.effective_prefix_end_us):
        idx = len(axis.effective_prefix_end_us) - 1
    prev_eff_us = axis.effective_prefix_end_us[idx - 1] if idx > 0 else 0
    inside_eff_us = target_effective_us - prev_eff_us
    seg_eff_us = axis.segment_effective_us[idx]
    seg_real_us = axis.segment_real_us[idx]
    seg_start = axis.segment_starts[idx]
    offset_us = (seg_real_us * inside_eff_us) // seg_eff_us
    return seg_start + timedelta(microseconds=offset_us)


def _build_effective_axis(
    timeline: list[datetime],
    strip_gap_threshold_us: int | None,
) -> EffectiveAxis:
    total_real_us = 0
    total_effective_us = 0
    segment_starts: list[datetime] = []
    segment_real_us: list[int] = []
    segment_effective_us: list[int] = []
    effective_prefix_end_us: list[int] = []
    for idx in range(len(timeline) - 1):
        seg_start = timeline[idx]
        seg_us = int((timeline[idx + 1] - seg_start).total_seconds() * 1_000_000)
        eff_us = _effective_segment_us(seg_us, strip_gap_threshold_us)
        total_real_us += seg_us
        total_effective_us += eff_us
        if eff_us <= 0:
            continue
        segment_starts.append(seg_start)
        segment_real_us.append(seg_us)
        segment_effective_us.append(eff_us)
        effective_prefix_end_us.append(total_effective_us)
    return EffectiveAxis(
        segment_starts=segment_starts,
        segment_real_us=segment_real_us,
        segment_effective_us=segment_effective_us,
        effective_prefix_end_us=effective_prefix_end_us,
        total_real_us=total_real_us,
        total_effective_us=total_effective_us,
    )


def _split_main_chunks(
    bounds: TimeRange,
    n_chunks: int,
    *,
    timeline: list[datetime] | None = None,
    strip_gap_threshold: timedelta | None = None,
    effective_axis: EffectiveAxis | None = None,
) -> list[ChunkBoundary]:
    if n_chunks < 1:
        raise SystemExit("--n-chunks-united must be >= 1.")
    if bounds.start >= bounds.end:
        raise SystemExit("Start and end bounds are equal; cannot split into chunks.")

    boundaries: list[datetime] = [bounds.start]
    if timeline is None:
        span_us = int((bounds.end - bounds.start).total_seconds() * 1_000_000)
        if span_us < n_chunks:
            raise SystemExit(
                f"Range is too small for {n_chunks} chunks: "
                f"{_iso_utc(bounds.start)} .. {_iso_utc(bounds.end)}."
            )
        for idx in range(1, n_chunks):
            offset_us = (span_us * idx) // n_chunks
            boundaries.append(bounds.start + timedelta(microseconds=offset_us))
    else:
        axis = effective_axis
        if axis is None:
            strip_gap_threshold_us = (
                int(strip_gap_threshold.total_seconds() * 1_000_000)
                if strip_gap_threshold is not None
                else None
            )
            axis = _build_effective_axis(timeline, strip_gap_threshold_us)
        if axis.total_real_us <= 0:
            raise SystemExit("Anchor timeline has non-positive span.")
        if axis.total_effective_us <= 0:
            raise SystemExit(
                "Effective timeline span is zero after --strip-internal-gaps filtering. "
                "Use wider range or disable filtering via --strip-internal-gaps none."
            )
        for idx in range(1, n_chunks):
            target_eff_us = (axis.total_effective_us * idx) // n_chunks
            boundaries.append(
                _map_effective_offset_to_real_ts(axis, target_eff_us)
            )
    boundaries.append(bounds.end)

    chunks: list[ChunkBoundary] = []
    for idx in range(n_chunks):
        chunk_start = boundaries[idx]
        if idx < n_chunks - 1:
            next_start = boundaries[idx + 1]
            chunk_end = next_start - timedelta(seconds=1)
            if chunk_end < chunk_start:
                raise SystemExit(
                    "Chunk range collapsed due to 1-second anti-overlap shift. "
                    "Use smaller --n-chunks-united or wider date range."
                )
        else:
            chunk_end = bounds.end
        chunks.append(ChunkBoundary(start=chunk_start, end=chunk_end))
    return chunks


def _chunks_from_boundary_points(boundaries: list[datetime], n_chunks: int) -> list[ChunkBoundary]:
    if len(boundaries) != n_chunks + 1:
        raise SystemExit(
            f"Invalid boundaries count: expected {n_chunks + 1}, got {len(boundaries)}."
        )
    chunks: list[ChunkBoundary] = []
    for idx in range(n_chunks):
        chunk_start = boundaries[idx]
        if idx < n_chunks - 1:
            next_start = boundaries[idx + 1]
            chunk_end = next_start - timedelta(seconds=1)
            if chunk_end < chunk_start:
                raise SystemExit(
                    "Chunk range collapsed due to 1-second anti-overlap shift. "
                    "Use smaller --n-chunks-united or wider date range."
                )
        else:
            chunk_end = boundaries[-1]
        chunks.append(ChunkBoundary(start=chunk_start, end=chunk_end))
    return chunks


def _split_train_val_test(chunk: ChunkBoundary) -> list[tuple[str, ChunkBoundary]]:
    total_us = int((chunk.end - chunk.start).total_seconds() * 1_000_000)
    if total_us < 3_000_000:
        raise SystemExit(
            "Each global chunk must be at least 3 seconds long to create "
            "non-overlapping train/validation/test ranges with 1-second shift."
        )

    b1 = chunk.start + timedelta(microseconds=(total_us * 1) // 3)
    b2 = chunk.start + timedelta(microseconds=(total_us * 2) // 3)
    train_end = b1 - timedelta(seconds=1)
    val_start = b1
    val_end = b2 - timedelta(seconds=1)
    test_start = b2
    if train_end < chunk.start or val_end < val_start or chunk.end < test_start:
        raise SystemExit("Failed to split local train/validation/test ranges.")

    return [
        ("train", ChunkBoundary(start=chunk.start, end=train_end)),
        ("validation", ChunkBoundary(start=val_start, end=val_end)),
        ("test", ChunkBoundary(start=test_start, end=chunk.end)),
    ]


def _split_train_val_test_effective(
    client: Any,
    database: str,
    anchor_table: str,
    symbol: str,
    chunk: ChunkBoundary,
    strip_gap_threshold_us: int | None,
) -> list[tuple[str, ChunkBoundary]]:
    boundaries = _load_effective_boundaries_sql(
        client=client,
        database=database,
        table=anchor_table,
        symbol=symbol,
        start=chunk.start,
        end=chunk.end,
        n_chunks=3,
        strip_gap_threshold_us=strip_gap_threshold_us,
    )
    if len(boundaries) != 2:
        raise SystemExit(
            "Failed to build local effective TVT boundaries "
            f"for chunk [{_iso_utc(chunk.start)}, {_iso_utc(chunk.end)}]."
        )
    local_chunks = _chunks_from_boundary_points([chunk.start, boundaries[0], boundaries[1], chunk.end], 3)
    return [
        ("train", local_chunks[0]),
        ("validation", local_chunks[1]),
        ("test", local_chunks[2]),
    ]


def _rows_in_range(
    client: Any,
    database: str,
    table: str,
    symbol: str,
    chunk: ChunkBoundary,
) -> int:
    sql = f"""
SELECT count()
FROM {database}.{table}
WHERE symbol = %(symbol)s
  AND exchange_ts >= %(start)s
  AND exchange_ts <= %(end)s
"""
    return int(
        client.execute(
            sql,
            {
                "symbol": symbol,
                "start": exchange_ts_for_ch(chunk.start),
                "end": exchange_ts_for_ch(chunk.end),
            },
        )[0][0]
    )


def _rows_in_many_ranges(
    client: Any,
    database: str,
    table: str,
    symbol: str,
    chunks: list[ChunkBoundary],
) -> list[int]:
    if not chunks:
        return []
    selects: list[str] = []
    params: dict[str, Any] = {"symbol": symbol}
    for idx, chunk in enumerate(chunks):
        start_key = f"start_{idx}"
        end_key = f"end_{idx}"
        selects.append(
            f"sum(if(exchange_ts >= %({start_key})s AND exchange_ts <= %({end_key})s, 1, 0)) AS c_{idx}"
        )
        params[start_key] = exchange_ts_for_ch(chunk.start)
        params[end_key] = exchange_ts_for_ch(chunk.end)
    sql = f"""
SELECT
    {", ".join(selects)}
FROM {database}.{table}
WHERE symbol = %(symbol)s
"""
    row = client.execute(sql, params)
    if not row:
        return [0 for _ in chunks]
    return [int(value) for value in row[0]]


def _filter_non_empty_global_chunks(
    client: Any,
    database: str,
    anchor_table: str,
    symbol: str,
    chunks: list[ChunkBoundary],
) -> tuple[list[ChunkBoundary], int]:
    if not chunks:
        return [], 0
    counts = _rows_in_many_ranges(
        client=client,
        database=database,
        table=anchor_table,
        symbol=symbol,
        chunks=chunks,
    )
    kept: list[ChunkBoundary] = []
    dropped = 0
    for chunk, rows_n in zip(chunks, counts):
        if rows_n > 0:
            kept.append(chunk)
        else:
            dropped += 1
    return kept, dropped


def _show_manuals_history(
    client: Any,
    database: str,
    symbol: str,
    limit_n: int,
) -> None:
    sql = f"""
SELECT
    symbol,
    source_table,
    chunk_description,
    chunk_effective_start,
    chunk_effective_end,
    created_at,
    chunk_hash
FROM {database}.manual_marked_chunks
WHERE symbol = %(symbol)s
ORDER BY created_at DESC
LIMIT %(limit_n)s
"""
    rows = client.execute(sql, {"symbol": symbol, "limit_n": int(limit_n)})
    print(f"\nManual chunks history (symbol={symbol}, limit={limit_n}):")
    if not rows:
        print("- no manual chunks found")
        return
    for row in rows:
        (
            row_symbol,
            source_table,
            description,
            start_dt,
            end_dt,
            created_at,
            chunk_hash,
        ) = row
        print(
            f"- symbol={row_symbol} table={source_table} created_at={_iso_utc(created_at)} "
            f"start={_iso_utc(start_dt)} end={_iso_utc(end_dt)} "
            f"desc={description!r} hash={chunk_hash}"
        )


def _insert_manual_chunks(
    client: Any,
    database: str,
    symbol: str,
    tables_to_mark: list[str],
    description: str,
    chunk_start: datetime,
    chunk_end: datetime,
    create_time: datetime,
) -> int:
    rows: list[tuple[Any, ...]] = []
    for table in tables_to_mark:
        chunk_hash = _hash_text(
            "|".join(
                (
                    symbol,
                    description,
                    table,
                    _iso_utc(chunk_start),
                    _iso_utc(chunk_end),
                )
            )
        )
        rows.append(
            (
                symbol,
                table,
                description,
                exchange_ts_for_ch(chunk_start),
                exchange_ts_for_ch(chunk_end),
                exchange_ts_for_ch(create_time),
                chunk_hash,
            )
        )
    clickhouse_insert_rows(
        client,
        f"""
INSERT INTO {database}.manual_marked_chunks (
    symbol,
    source_table,
    chunk_description,
    chunk_effective_start,
    chunk_effective_end,
    created_at,
    chunk_hash
) VALUES
""",
        rows,
    )
    return len(rows)


def _dist_stats(values: list[int]) -> dict[str, float]:
    if not values:
        return {"min": 0.0, "max": 0.0, "mean": 0.0, "median": 0.0, "std": 0.0}
    return {
        "min": float(min(values)),
        "max": float(max(values)),
        "mean": float(mean(values)),
        "median": float(median(values)),
        "std": float(stdev(values)) if len(values) >= 2 else 0.0,
    }


def _git_hash(repo_root: Path) -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo_root),
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return out.strip()
    except Exception:
        return "unknown"


def _confirm(submit_ranges: bool) -> None:
    if submit_ranges:
        print("submit-ranges=True -> confirmation skipped.")
        return
    answer = input("Accept current split and insert rows? [y/N]: ").strip().lower()
    if answer not in {"y", "yes"}:
        raise SystemExit("Cancelled by user.")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Build unified TVT chunk markup over [start-date, end-date] (inclusive) "
            "and persist chunk instances/results to ClickHouse."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--symbol", required=True, help="Instrument symbol, e.g. BTCUSDT.")
    p.add_argument("--start-date", required=True, type=str, help="Inclusive ISO-8601 UTC start.")
    p.add_argument("--end-date", required=True, type=str, help="Inclusive ISO-8601 UTC end.")
    p.add_argument(
        "--tables-to-mark",
        type=str,
        default="spot_pt",
        help="Comma-separated source tables to evaluate and mark.",
    )
    p.add_argument(
        "--submit-ranges",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Skip interactive accept prompt when True.",
    )
    p.add_argument(
        "--n-chunks-united",
        type=int,
        default=30,
        help="Number of global chunks inside [start-date, end-date].",
    )
    p.add_argument(
        "--debug",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Print detailed rows distribution per local chunk when enabled.",
    )
    p.add_argument(
        "--strip-internal-gaps",
        type=str,
        default="1d",
        help=(
            "Strip internal anchor-table gaps longer than this duration from effective split time. "
            "Use 'none' to disable."
        ),
    )
    p.add_argument(
        "--manual-mode",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Mark one continuous chunk over globally stripped range and write to "
            "manual_marked_chunks."
        ),
    )
    p.add_argument(
        "--manual-chunk-description",
        type=str,
        default="",
        help="Description saved into manual_marked_chunks in --manual-mode.",
    )
    p.add_argument(
        "--show-manuals-history",
        type=int,
        default=0,
        metavar="N",
        help="Show last N manual_marked_chunks records (created_at DESC) for symbol.",
    )
    p.add_argument("--host", default=quant_env.CLICKHOUSE_HOST)
    p.add_argument("--port", type=int, default=quant_env.CLICKHOUSE_PORT)
    p.add_argument("--user", default=quant_env.CLICKHOUSE_USER)
    p.add_argument("--password", default=quant_env.CLICKHOUSE_PASSWORD)
    p.add_argument("--database", default=quant_env.CLICKHOUSE_DATABASE)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    start = parse_cli_datetime(args.start_date)
    end = parse_cli_datetime(args.end_date)
    if start is None or end is None:
        raise SystemExit("--start-date and --end-date are required ISO-8601 values.")
    start = exchange_ts_for_ch(start)
    end = exchange_ts_for_ch(end)
    if start > end:
        raise SystemExit("--start-date must be <= --end-date.")
    strip_internal_gaps = parse_duration(args.strip_internal_gaps)

    tables_to_mark = _parse_csv_list(args.tables_to_mark)
    if not tables_to_mark:
        raise SystemExit("At least one table is required in --tables-to-mark.")

    client = build_clickhouse_client(
        host=args.host,
        port=args.port,
        user=args.user,
        password=args.password,
    )
    _require_tables(client, args.database, tables_to_mark)
    if args.show_manuals_history > 0:
        _show_manuals_history(client, args.database, args.symbol, args.show_manuals_history)
        if not args.manual_mode:
            return

    table_bounds: dict[str, TimeRange] = {}
    for table in tables_to_mark:
        table_bounds[table] = _load_effective_bounds(
            client=client,
            database=args.database,
            table=table,
            symbol=args.symbol,
            start=start,
            end=end,
        )

    anchor_table = tables_to_mark[0]
    bounds = table_bounds[anchor_table]
    print(f"Anchor table for split: {anchor_table}")
    print(f"Requested range: {_iso_utc(start)} .. {_iso_utc(end)}")
    print(f"Effective range: {_iso_utc(bounds.start)} .. {_iso_utc(bounds.end)}")
    requested_span = end - start
    effective_requested_span = bounds.end - bounds.start
    gap_left = bounds.start - start
    gap_right = end - bounds.end
    print(
        "Requested/effective span: "
        f"requested={_format_days_seconds(requested_span)}, "
        f"effective={_format_days_seconds(effective_requested_span)}"
    )
    print(
        "Gap[start->first_row]: "
        f"{_format_days_seconds(gap_left)}; "
        "Gap[last_row->end]: "
        f"{_format_days_seconds(gap_right)}"
    )
    print("Per-table first/last rows inside requested range:")
    for table, tb in table_bounds.items():
        print(
            f"- {table}: first={_iso_utc(tb.start)}, last={_iso_utc(tb.end)}, "
            f"gap_left={_format_days_seconds(tb.start - start)}, "
            f"gap_right={_format_days_seconds(end - tb.end)}"
        )
    if args.manual_mode:
        print("\nWARNING: --manual-mode is enabled.")
        print(
            "Manual mode writes one continuous chunk per source_table into "
            "manual_marked_chunks over globally stripped range (effective bounds only), "
            "without global/local TVT split logic."
        )
        repo_root = bootstrap_path.ensure_src_on_path(Path(__file__))
        _ = _git_hash(repo_root)  # Keep session parity with non-manual mode.
        create_time = datetime.now(timezone.utc).replace(tzinfo=None)
        inserted = _insert_manual_chunks(
            client=client,
            database=args.database,
            symbol=args.symbol,
            tables_to_mark=tables_to_mark,
            description=args.manual_chunk_description,
            chunk_start=bounds.start,
            chunk_end=bounds.end,
            create_time=create_time,
        )
        print(
            f"Inserted manual chunks: {inserted} rows -> {args.database}.manual_marked_chunks."
        )
        return

    strip_gap_threshold_us = (
        int(strip_internal_gaps.total_seconds() * 1_000_000)
        if strip_internal_gaps is not None
        else None
    )
    real_us, effective_us = _load_effective_span_stats_sql(
        client=client,
        database=args.database,
        table=anchor_table,
        symbol=args.symbol,
        start=bounds.start,
        end=bounds.end,
        strip_gap_threshold_us=strip_gap_threshold_us,
    )
    if real_us <= 0:
        raise SystemExit("Anchor timeline has non-positive span.")
    if effective_us <= 0:
        raise SystemExit(
            "Effective timeline span is zero after --strip-internal-gaps filtering. "
            "Use wider range or disable filtering via --strip-internal-gaps none."
        )
    stripped_us = max(0, real_us - effective_us)
    print(
        "Effective split mode: "
        f"strip_internal_gaps={args.strip_internal_gaps!r}, "
        f"real_span={_format_days_seconds_from_seconds(real_us / 1_000_000)}, "
        f"effective_span={_format_days_seconds_from_seconds(effective_us / 1_000_000)}, "
        f"stripped={_format_days_seconds_from_seconds(stripped_us / 1_000_000)}"
    )

    interior_boundaries = _load_effective_boundaries_sql(
        client=client,
        database=args.database,
        table=anchor_table,
        symbol=args.symbol,
        start=bounds.start,
        end=bounds.end,
        n_chunks=args.n_chunks_united,
        strip_gap_threshold_us=strip_gap_threshold_us,
    )
    boundaries = [bounds.start, *interior_boundaries, bounds.end]
    main_chunks = _chunks_from_boundary_points(boundaries, args.n_chunks_united)
    main_chunks, dropped_chunks = _filter_non_empty_global_chunks(
        client=client,
        database=args.database,
        anchor_table=anchor_table,
        symbol=args.symbol,
        chunks=main_chunks,
    )
    if dropped_chunks > 0:
        print(
            f"Filtered empty global chunks on anchor table: dropped={dropped_chunks}, "
            f"kept={len(main_chunks)}"
        )
    if not main_chunks:
        raise SystemExit(
            "All global chunks are empty on anchor table after boundary mapping. "
            "Adjust --n-chunks-united or --strip-internal-gaps."
        )
    all_instances: list[ChunkInstance] = []
    table_chunk_counts: dict[str, list[int]] = {t: [] for t in tables_to_mark}
    table_chunk_debug: dict[str, list[tuple[int, str, int, ChunkBoundary]]] = {
        t: [] for t in tables_to_mark
    }

    repo_root = bootstrap_path.ensure_src_on_path(Path(__file__))
    git_hash = _git_hash(repo_root)
    create_time = datetime.now(timezone.utc).replace(tzinfo=None)

    local_specs: list[tuple[int, str, ChunkBoundary, str]] = []
    for chunk_idx, chunk in enumerate(main_chunks):
        if strip_gap_threshold_us is None:
            local_tvt = _split_train_val_test(chunk)
        else:
            local_tvt = _split_train_val_test_effective(
                client=client,
                database=args.database,
                anchor_table=anchor_table,
                symbol=args.symbol,
                chunk=chunk,
                strip_gap_threshold_us=strip_gap_threshold_us,
            )
        for chunk_type, local_chunk in local_tvt:
            signature = _hash_text(
                "|".join(
                    (
                        args.symbol,
                        _iso_utc(local_chunk.start),
                        _iso_utc(local_chunk.end),
                        chunk_type,
                        git_hash,
                    )
                )
            )
            local_specs.append((chunk_idx, chunk_type, local_chunk, signature))

    for table in tables_to_mark:
        local_chunks_only = [spec[2] for spec in local_specs]
        counts = _rows_in_many_ranges(
            client=client,
            database=args.database,
            table=table,
            symbol=args.symbol,
            chunks=local_chunks_only,
        )
        for idx, (chunk_idx, chunk_type, local_chunk, signature) in enumerate(local_specs):
            rows_n = counts[idx]
            all_instances.append(
                ChunkInstance(
                    source_table=table,
                    chunk_start=local_chunk.start,
                    chunk_end=local_chunk.end,
                    chunk_type=chunk_type,
                    chunk_signature_hash=signature,
                )
            )
            table_chunk_counts[table].append(rows_n)
            table_chunk_debug[table].append((chunk_idx, chunk_type, rows_n, local_chunk))

    print("\nRows distribution by table (min/max/mean/median/std):")
    for table in tables_to_mark:
        stats = _dist_stats(table_chunk_counts[table])
        print(
            f"- {table}: min={stats['min']:.0f} max={stats['max']:.0f} "
            f"mean={stats['mean']:.2f} median={stats['median']:.2f} std={stats['std']:.2f}"
        )
        if min(table_chunk_counts[table], default=0) == 0:
            print(f"  warning: {table} has empty local chunks in current split.")
    if args.debug:
        print("\nDetailed rows distribution by local chunks:")
        for table in tables_to_mark:
            print(f"- {table}:")
            for chunk_idx, chunk_type, rows_n, local_chunk in table_chunk_debug[table]:
                print(
                    f"  chunk={chunk_idx:03d} type={chunk_type:<10} rows={rows_n:<8d} "
                    f"start={_iso_utc(local_chunk.start)} end={_iso_utc(local_chunk.end)}"
                )

    _confirm(args.submit_ranges)

    instances_rows = [
        (
            args.symbol,
            instance.source_table,
            exchange_ts_for_ch(instance.chunk_start),
            exchange_ts_for_ch(instance.chunk_end),
            instance.chunk_type,
            instance.chunk_signature_hash,
            exchange_ts_for_ch(create_time),
            git_hash,
        )
        for instance in all_instances
    ]
    signatures = [item.chunk_signature_hash for item in all_instances]
    chunks_list_hash = _hash_text("|".join(signatures))
    results_row = (
        args.symbol,
        exchange_ts_for_ch(create_time),
        signatures,
        git_hash,
        chunks_list_hash,
    )

    clickhouse_insert_rows(
        client,
        f"""
INSERT INTO {args.database}.chunks_instances (
    symbol,
    source_table,
    chunk_start,
    chunk_end,
    chunk_type,
    chunk_signature_hash,
    create_time,
    git_hash
) VALUES
""",
        instances_rows,
    )
    clickhouse_insert_rows(
        client,
        f"""
INSERT INTO {args.database}.chunks_marking_results (
    symbol,
    created_at,
    chunks_list,
    git_hash,
    chunks_lish_hash
) VALUES
""",
        [results_row],
    )
    print(
        "Inserted chunk markup: "
        f"{len(instances_rows)} rows -> chunks_instances, 1 row -> chunks_marking_results."
    )


if __name__ == "__main__":
    main()
