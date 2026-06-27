"""
Compute rolling spot_trades features via the new FeatureInstance/Template contract and
write them into crypto_db.features_kv.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import bootstrap_path

bootstrap_path.ensure_src_on_path(Path(__file__))


from pipeline.features import (  # noqa: E402
    build_spot_trades_rolling_bundle,
    render_feature_sql,
)
from quant_system.clickhouse_io import (  # noqa: E402
    add_connection_args,
    client_from_args,
    delete_features_kv_range,
    parse_cli_datetime,
    symbol_time_where,
)
from quant_system.time_chunk_processor import (  # noqa: E402
    RowCountTimeBatchSpec,
    exchange_ts_for_ch,
    run_row_count_time_batched,
)
from tqdm.auto import tqdm  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute rolling spot_trades features (full-causal, exclude current row) "
            "and write to features_kv."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    add_connection_args(
        parser,
        include_symbol=True,
        include_replace_range=True,
        include_io_batch=False,
    )
    parser.add_argument(
        "--source-table",
        default="spot_trades",
        help="Source table name inside --database.",
    )
    parser.set_defaults(database="crypto_db")
    parser.add_argument(
        "--window-ms",
        type=int,
        default=300_000,
        help="Rolling lookback window in milliseconds.",
    )
    parser.add_argument(
        "--max-rows-batch",
        type=int,
        default=None,
        help=(
            "Max source rows per compute batch. "
            "When set, splits [start, end) into multiple time windows for INSERT SELECT."
        ),
    )
    return parser.parse_args()


def build_feature_insert_sql(
    *,
    database: str,
    rendered_feature_sql: str,
) -> str:
    return f"""
INSERT INTO {database}.features_kv (symbol, exchange_ts, feature_name, feature_value)
SELECT
    symbol,
    bucket_time AS exchange_ts,
    %(feature_name)s AS feature_name,
    toFloat64(feature_value) AS feature_value
FROM (
{rendered_feature_sql}
) AS feature_rows
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


def _dt_ch_param(dt: Any) -> str:
    return dt.strftime("%Y-%m-%d %H:%M:%S.%f")


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


def _configured_features(where_expr: str, *, args: argparse.Namespace) -> tuple[Any, ...]:
    bundle = build_spot_trades_rolling_bundle(
        database=args.database,
        source_table=args.source_table,
        where_clause=where_expr,
        window_ms=args.window_ms,
    )
    return bundle.features_in_compute_order


def main() -> None:
    args = parse_args()
    if args.max_rows_batch is not None and args.max_rows_batch <= 0:
        raise SystemExit("--max-rows-batch must be positive")
    vprint = (lambda *a, **kw: print(*a, **kw, flush=True)) if not args.quiet else (lambda *_a, **_kw: None)

    start = exchange_ts_for_ch(parse_cli_datetime(args.start))
    end = exchange_ts_for_ch(parse_cli_datetime(args.end))
    where_parts, base_params = symbol_time_where(args.symbol, start, end, ts_col="exchange_ts")
    configured_features = _configured_features(" AND ".join(where_parts), args=args)
    feature_names = [feature.name for feature in configured_features]

    vprint(
        f"Start compute: symbol={args.symbol} source={args.database}.{args.source_table} "
        f"start={start} end={end} max_rows_batch={args.max_rows_batch}"
    )
    vprint("Connecting to ClickHouse...")
    client = client_from_args(args)
    vprint("Connected.")
    if args.replace_range:
        vprint("Deleting existing features in requested range (--replace-range)...")
        delete_features_kv_range(
            client=client,
            database=args.database,
            symbol=args.symbol,
            start=start,
            end=end,
            feature_names=feature_names,
        )
        vprint("Delete finished.")

    vprint("Counting source rows in requested time range...")
    source_count_params: dict[str, Any] = {"symbol": args.symbol}
    if start is not None:
        source_count_params["start_ts"] = _dt_ch_param(start)
    if end is not None:
        source_count_params["end_ts"] = _dt_ch_param(end)
    total_source_rows = int(
        client.execute(
            _source_rows_count_sql(args.database, args.source_table, start, end),
            source_count_params,
        )[0][0]
    )
    vprint(f"Total source rows: {total_source_rows:,}")
    if args.max_rows_batch is not None:
        vprint(f"Row-batched compute enabled: <= {args.max_rows_batch:,} source rows per batch.")

    with tqdm(
        total=total_source_rows,
        desc="features insert",
        unit="rows",
        file=sys.stdout,
        ascii=True,
        leave=True,
        mininterval=0.1,
        disable=args.quiet,
        dynamic_ncols=True,
    ) as pbar:

        def execute_batch(batch_start: Any, batch_end: Any) -> None:
            batch_where_parts, batch_params = symbol_time_where(
                symbol=args.symbol,
                start=batch_start,
                end=batch_end,
                ts_col="exchange_ts",
            )
            batch_features = _configured_features(" AND ".join(batch_where_parts), args=args)
            for feature in batch_features:
                insert_sql = build_feature_insert_sql(
                    database=args.database,
                    rendered_feature_sql=render_feature_sql(feature),
                )
                query_params: dict[str, Any] = dict(batch_params)
                query_params["feature_name"] = feature.name
                client.execute(insert_sql, query_params)

        def execute_row_batch(batch: RowCountTimeBatchSpec) -> None:
            execute_batch(batch.start, batch.end)
            pbar.update(int(batch.row_count))

        if args.max_rows_batch is None:
            execute_batch(start, end)
            if total_source_rows > 0:
                pbar.update(total_source_rows)
        else:
            if not args.quiet:
                print(
                    f"Compute by row batches <= {args.max_rows_batch:,} rows "
                    f"from {args.database}.{args.source_table}",
                    flush=True,
                )
            run_row_count_time_batched(
                client,
                database=args.database,
                table=args.source_table,
                symbol=args.symbol,
                max_rows_batch=args.max_rows_batch,
                start=start,
                end=end,
                log=vprint,
                handler=execute_row_batch,
            )

    if args.quiet:
        return

    print("Inserted/available rows per feature:", flush=True)
    for feature_name in feature_names:
        query_params = dict(base_params)
        query_params["feature_name"] = feature_name
        count_sql = _feature_count_sql(args.database, feature_name, start, end)
        row_count = int(client.execute(count_sql, params=query_params)[0][0])
        print(f"  {feature_name}: {row_count}", flush=True)


if __name__ == "__main__":
    main()
