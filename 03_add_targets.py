"""Compute trade-time targets from `spot_trades` into `features_kv`.

Contracts:
- `price_delta_4s`: continuous delta with 0.45% deadband to zero.
- `price_direction_4s`: ternary classes {-1,0,1} with 0.55% threshold.
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import bootstrap_path

bootstrap_path.ensure_src_on_path(Path(__file__))

from pipeline.features import (  # noqa: E402
    PRICE_DELTA_4S_HORIZON,
    build_feature_bundle,
    build_named_target_feature,
    render_feature_sql,
)
from quant_system.clickhouse_io import (  # noqa: E402
    add_connection_args,
    client_from_args,
    delete_features_kv_range,
    load_table_bounds,
    parse_cli_datetime,
    symbol_time_where,
)
from quant_system.time_chunk_processor import (  # noqa: E402
    RowCountTimeBatchSpec,
    add_timedelta,
    exchange_ts_for_ch,
    run_row_count_time_batched,
    subtract_timedelta,
)
from tqdm.auto import tqdm  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Compute trade-time target feature "
            "(delta or direction contract over t+4s horizon) "
            "from spot_trades and write to features_kv."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    add_connection_args(
        p,
        include_symbol=True,
        include_replace_range=True,
        include_io_batch=False,
    )
    p.set_defaults(database="crypto_db")
    p.add_argument(
        "--source-table",
        default="spot_trades",
        help="Source table name inside --database.",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Alias for --replace-range.",
    )
    p.add_argument(
        "--max-batch-rows",
        "--max-rows-batch",
        dest="max_batch_rows",
        type=int,
        default=None,
        help=(
            "Max source rows per compute batch for target INSERT SELECT. "
            "When set, splits effective emit range into multiple row-bounded time windows."
        ),
    )
    return p.parse_args()


def _dt_param(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%d %H:%M:%S.%f")


def _insert_sql(*, database: str, rendered_feature_sql: str) -> str:
    return f"""
INSERT INTO {database}.features_kv (symbol, exchange_ts, feature_name, feature_value)
SELECT
    symbol,
    bucket_time AS exchange_ts,
    %(feature_name)s AS feature_name,
    toFloat64(feature_value) AS feature_value
FROM (
{rendered_feature_sql}
) AS target_rows
WHERE isFinite(feature_value)
"""


def _feature_count_sql(database: str, feature_name: str, start: Any, end: Any) -> str:
    sql = f"""
SELECT count()
FROM {database}.features_kv
WHERE symbol = %(symbol)s
  AND feature_name = %(feature_name)s
"""
    if start is not None:
        sql += "\n  AND exchange_ts >= %(start)s"
    if end is not None:
        sql += "\n  AND exchange_ts < %(end)s"
    return sql


def _table_exists(client: Any, database: str, table: str) -> bool:
    sql = """
SELECT count()
FROM system.tables
WHERE database = %(database)s
  AND name = %(table)s
"""
    return int(client.execute(sql, {"database": database, "table": table})[0][0]) > 0


def _source_rows_count_sql(database: str, table: str, start: Any, end: Any) -> str:
    sql = f"""
SELECT count()
FROM {database}.{table}
WHERE symbol = %(symbol)s
"""
    if start is not None:
        sql += "\n  AND exchange_ts >= %(start_ts)s"
    if end is not None:
        sql += "\n  AND exchange_ts < %(end_ts)s"
    return sql


def _configured_target(
    *,
    database: str,
    source_table: str,
    where_clause: str,
    horizon_seconds: int,
    target_name: str,
):
    return build_named_target_feature(
        target_name=target_name,
        database=database,
        source_table=source_table,
        where_clause=where_clause,
        horizon_seconds=horizon_seconds,
    )


def main() -> None:
    args = parse_args()
    if args.force:
        args.replace_range = True
    if args.max_batch_rows is not None and args.max_batch_rows <= 0:
        raise SystemExit("--max-batch-rows must be positive")

    start = exchange_ts_for_ch(parse_cli_datetime(args.start))
    end = exchange_ts_for_ch(parse_cli_datetime(args.end))
    horizon = PRICE_DELTA_4S_HORIZON
    target_contracts = [target.name for target in build_feature_bundle(include_non_targets=False).targets]
    if not target_contracts:
        raise SystemExit("No active target contracts in declarations.")
    vprint = (lambda *a, **kw: print(*a, **kw, flush=True)) if not args.quiet else (lambda *_a, **_kw: None)

    client = client_from_args(args)
    if not _table_exists(client, args.database, args.source_table):
        raise SystemExit(
            f"Source table {args.database}.{args.source_table} does not exist.\n"
            "Target contracts expect trade rows with columns including "
            "exchange_ts/symbol/price (e.g. spot_trades).\n"
            "Pass an existing source table via --source-table."
        )
    src_min, src_max = load_table_bounds(
        client,
        args.database,
        args.source_table,
        args.symbol,
        start=start,
        end=end,
    )
    if src_min is None or src_max is None:
        vprint("No source rows in requested bounds.")
        return

    emit_start = src_min if start is None else max(src_min, start)
    max_emit_ts = subtract_timedelta(src_max, horizon)
    emit_end_from_data = add_timedelta(max_emit_ts, timedelta(milliseconds=1))
    emit_end = emit_end_from_data if end is None else min(end, emit_end_from_data)
    if emit_start >= emit_end:
        vprint("No target rows to compute for this interval (not enough forward horizon).")
        return

    count_params: dict[str, Any] = {"symbol": args.symbol}
    if emit_start is not None:
        count_params["start_ts"] = _dt_param(emit_start)
    if emit_end is not None:
        count_params["end_ts"] = _dt_param(emit_end)
    total_source_rows = int(
        client.execute(
            _source_rows_count_sql(args.database, args.source_table, emit_start, emit_end),
            count_params,
        )[0][0]
    )
    if not args.quiet:
        active_list = ", ".join(target_contracts)
        vprint(f"Active target contracts: {active_list}")
        vprint(f"Total source rows in emit range: {total_source_rows:,}")
        if args.max_batch_rows is not None:
            vprint(
                f"Row-batched target compute enabled: <= {args.max_batch_rows:,} rows per batch."
            )

    for feature_name in target_contracts:
        if not args.quiet:
            vprint(
                f"Target={feature_name} horizon={int(horizon.total_seconds())}s "
                f"emit=[{emit_start.isoformat()}, {emit_end.isoformat()})"
            )

        if args.replace_range:
            delete_features_kv_range(
                client=client,
                database=args.database,
                symbol=args.symbol,
                start=emit_start,
                end=emit_end,
                feature_names=[feature_name],
            )
            vprint(f"Deleted old rows for target={feature_name} in effective emit range.")

        def execute_batch(batch_start: datetime, batch_end: datetime) -> None:
            where_parts, query_params = symbol_time_where(
                symbol=args.symbol,
                start=batch_start,
                end=batch_end,
                ts_col="t.exchange_ts",
            )
            configured_target = _configured_target(
                database=args.database,
                source_table=args.source_table,
                where_clause=" AND ".join(where_parts),
                horizon_seconds=int(horizon.total_seconds()),
                target_name=feature_name,
            )
            sql = _insert_sql(
                database=args.database,
                rendered_feature_sql=render_feature_sql(configured_target),
            )
            client.execute(
                sql,
                {
                    **query_params,
                    "feature_name": configured_target.name,
                },
            )

        with tqdm(
            total=total_source_rows,
            desc=f"target insert {feature_name}",
            unit="rows",
            file=sys.stdout,
            ascii=True,
            leave=True,
            mininterval=0.1,
            disable=args.quiet,
            dynamic_ncols=True,
        ) as pbar:
            if args.max_batch_rows is None:
                execute_batch(emit_start, emit_end)
                if total_source_rows > 0:
                    pbar.update(total_source_rows)
            else:
                if not args.quiet:
                    vprint(
                        f"Compute target={feature_name} by row batches <= {args.max_batch_rows:,} rows "
                        f"from {args.database}.{args.source_table}"
                    )

                def _on_batch(batch: RowCountTimeBatchSpec) -> None:
                    execute_batch(batch.start, batch.end)
                    pbar.update(int(batch.row_count))

                run_row_count_time_batched(
                    client,
                    database=args.database,
                    table=args.source_table,
                    symbol=args.symbol,
                    max_rows_batch=args.max_batch_rows,
                    start=emit_start,
                    end=emit_end,
                    log=vprint,
                    handler=_on_batch,
                )

        if args.quiet:
            continue
        cnt_sql = _feature_count_sql(args.database, feature_name, emit_start, emit_end)
        count = int(
            client.execute(
                cnt_sql,
                {
                    "symbol": args.symbol,
                    "feature_name": feature_name,
                    "start": _dt_param(emit_start),
                    "end": _dt_param(emit_end),
                },
            )[0][0]
        )
        vprint(f"Inserted target rows for {feature_name}: {count}")


if __name__ == "__main__":
    main()
