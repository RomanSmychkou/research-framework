"""ClickHouse connection, symbol/time filters, reads, and lightweight mutations."""

from __future__ import annotations

import argparse
from datetime import datetime
from typing import TYPE_CHECKING, Any

from clickhouse_driver.errors import ServerException

from . import quant_env
from .time_chunk_processor import exchange_ts_for_ch

if TYPE_CHECKING:
    from clickhouse_driver import Client

def _empty_trades_schema() -> dict[str, Any]:
    import polars as pl

    return {
        "exchange_ts": pl.Datetime("ms"),
        "side": pl.Utf8,
        "price": pl.Float64,
        "quantity": pl.Float64,
    }


def parse_cli_datetime(s: str | None) -> datetime | None:
    """Parse ISO-8601 CLI bound; empty string → None."""
    if s is None:
        return None
    v = s.strip()
    if not v:
        return None
    if v.endswith("Z"):
        v = v[:-1] + "+00:00"
    return datetime.fromisoformat(v)


def symbol_time_where(
    symbol: str,
    start: datetime | None = None,
    end: datetime | None = None,
    *,
    ts_col: str = "exchange_ts",
    end_inclusive: bool = False,
) -> tuple[list[str], dict[str, Any]]:
    """Build ``WHERE`` fragments and params for symbol + optional time range."""
    start = exchange_ts_for_ch(start)
    end = exchange_ts_for_ch(end)
    parts = ["symbol = %(symbol)s"]
    params: dict[str, Any] = {"symbol": symbol}
    if start is not None:
        parts.append(f"{ts_col} >= %(start)s")
        params["start"] = start
    if end is not None:
        op = "<=" if end_inclusive else "<"
        parts.append(f"{ts_col} {op} %(end)s")
        params["end"] = end
    return parts, params


def build_clickhouse_client(
    *,
    host: str | None = None,
    port: int | None = None,
    user: str | None = None,
    password: str | None = None,
    receive_timeout_sec: int | None = None,
) -> Client:
    from clickhouse_driver import Client

    kw: dict[str, Any] = {
        "host": host if host is not None else quant_env.CLICKHOUSE_HOST,
        "port": port if port is not None else quant_env.CLICKHOUSE_PORT,
        "user": user if user is not None else quant_env.CLICKHOUSE_USER,
        "password": password if password is not None else quant_env.CLICKHOUSE_PASSWORD,
    }
    if receive_timeout_sec is not None and receive_timeout_sec > 0:
        kw["send_receive_timeout"] = int(receive_timeout_sec)
    return Client(**kw)


def client_from_args(
    args: argparse.Namespace,
    *,
    receive_timeout_attr: str | None = "receive_timeout_sec",
) -> Client:
    """Native client from stage CLI namespace (``host``, ``port``, ``user``, ``password``)."""
    timeout = (
        getattr(args, receive_timeout_attr, None)
        if receive_timeout_attr
        else None
    )
    return build_clickhouse_client(
        host=args.host,
        port=args.port,
        user=args.user,
        password=args.password,
        receive_timeout_sec=timeout,
    )


def add_connection_args(
    p: argparse.ArgumentParser,
    *,
    include_symbol: bool = True,
    include_replace_range: bool = False,
    include_insert_batch: bool = False,
    include_io_batch: bool = False,
    include_quiet: bool = True,
    start_flags: tuple[str, ...] = ("--start",),
    end_flags: tuple[str, ...] = ("--end",),
    start_help: str = "Inclusive UTC lower bound (ISO-8601).",
    end_help: str = "Exclusive UTC upper bound (ISO-8601).",
) -> None:
    if include_symbol:
        p.add_argument(
            "--symbol",
            default=quant_env.QUANT_SYMBOL,
            help="Instrument symbol (or set QUANT_SYMBOL in .env).",
        )
    for flag in start_flags:
        p.add_argument(
            flag,
            type=str,
            default=None,
            help=start_help,
        )
    for flag in end_flags:
        p.add_argument(
            flag,
            type=str,
            default=None,
            help=end_help,
        )
    p.add_argument("--host", default=quant_env.CLICKHOUSE_HOST)
    p.add_argument("--port", type=int, default=quant_env.CLICKHOUSE_PORT)
    p.add_argument("--user", default=quant_env.CLICKHOUSE_USER)
    p.add_argument("--password", default=quant_env.CLICKHOUSE_PASSWORD)
    p.add_argument(
        "--receive-timeout-sec",
        type=int,
        default=quant_env.RECEIVE_TIMEOUT_SEC,
        metavar="SEC",
        help=(
            "Native client socket send/receive timeout (seconds). "
            "Driver default is 300; increase for huge scans, e.g. 7200."
        ),
    )
    p.add_argument("--database", default=quant_env.CLICKHOUSE_DATABASE)
    if include_replace_range:
        p.add_argument(
            "--replace-range",
            action=argparse.BooleanOptionalAction,
            default=quant_env.STAGE01_REPLACE_RANGE,
            help=(
                "ALTER DELETE features_kv for this symbol before insert; combine with "
                "time bounds to limit scope. Requires ALTER DELETE grant on features_kv."
            ),
        )
    if include_insert_batch:
        p.add_argument(
            "--insert-batch-rows",
            "--batch-rows",
            dest="insert_batch_rows",
            type=int,
            default=None,
            help=(
                "Rows per INSERT batch (overrides CLICKHOUSE_INSERT_BATCH_ROWS). "
                f"Default: {quant_env.DEFAULT_INSERT_BATCH_ROWS:,}."
            ),
        )
    if include_quiet:
        p.add_argument("--quiet", action="store_true", help="Minimal stdout.")
    if include_io_batch:
        io = p.add_argument_group("ClickHouse compute chunking (emit windows along exchange_ts)")
        io.add_argument(
            "--io-batch-span",
            "--initial-budget-timedelta",
            dest="io_batch_span",
            type=str,
            default=quant_env.IO_BATCH_SPAN,
            metavar="DURATION",
            help=(
                "Emit window per ClickHouse INSERT SELECT (env IO_BATCH_SPAN); "
                "stage 01 also loads lookback/forward padding; use none for one pass."
            ),
        )


def require_insert_batch_rows(row_override: int | None) -> None:
    if row_override is not None and row_override < quant_env.INSERT_MIN_ROWS:
        raise SystemExit(
            f"--insert-batch-rows must be >= {quant_env.INSERT_MIN_ROWS} "
            "for stable ClickHouse inserts."
        )


def load_table_bounds(
    client: Client,
    database: str,
    table: str,
    symbol: str,
    *,
    start: datetime | None = None,
    end: datetime | None = None,
    ts_col: str = "exchange_ts",
    end_inclusive: bool = False,
) -> tuple[datetime | None, datetime | None]:
    where, params = symbol_time_where(
        symbol, start, end, ts_col=ts_col, end_inclusive=end_inclusive
    )
    sql = f"""
    SELECT min({ts_col}), max({ts_col})
    FROM {database}.{table}
    WHERE {' AND '.join(where)}
    """
    row = client.execute(sql, params)
    if not row or row[0][0] is None:
        return None, None
    return exchange_ts_for_ch(row[0][0]), exchange_ts_for_ch(row[0][1])


def _empty_trades_agg_1s_schema() -> dict[str, Any]:
    import polars as pl

    return {
        "exchange_ts": pl.Datetime("ms"),
        "trade_count": pl.UInt64,
        "trade_count_buy": pl.UInt64,
        "trade_count_sell": pl.UInt64,
        "volume_buy": pl.Float64,
        "volume_sell": pl.Float64,
        "volume_total": pl.Float64,
        "notional_buy": pl.Float64,
        "notional_sell": pl.Float64,
        "notional_total": pl.Float64,
        "price_last": pl.Float64,
        "price_first": pl.Float64,
        "price_min": pl.Float64,
        "price_max": pl.Float64,
        "price_mean": pl.Float64,
        "price_std": pl.Float64,
        "price_q25": pl.Float64,
        "price_q75": pl.Float64,
        "q01": pl.Float64,
        "q99": pl.Float64,
    }


def load_trades_agg_1s_from_clickhouse(
    client: Client,
    database: str,
    symbol: str,
    start: datetime | None,
    end: datetime | None,
    *,
    table: str = "trades_agg_1s",
) -> Any:
    """Columnar SELECT of 1s trade aggregates for one symbol and optional window."""
    import polars as pl

    cols = ", ".join(
        (
            "exchange_ts",
            "trade_count",
            "trade_count_buy",
            "trade_count_sell",
            "volume_buy",
            "volume_sell",
            "volume_total",
            "notional_buy",
            "notional_sell",
            "notional_total",
            "price_last",
            "price_first",
            "price_min",
            "price_max",
            "price_mean",
            "price_std",
            "price_q25",
            "price_q75",
            "q01",
            "q99",
        )
    )
    where, params = symbol_time_where(symbol, start, end)
    sql = f"""
    SELECT {cols}
    FROM {database}.{table}
    WHERE {' AND '.join(where)}
    ORDER BY exchange_ts
    """
    data, columns_meta = client.execute(
        sql, params, columnar=True, with_column_types=True
    )
    schema = _empty_trades_agg_1s_schema()
    if not data or not data[0]:
        return pl.DataFrame(schema=schema)
    col_names = [c[0] for c in columns_meta]
    return pl.DataFrame(dict(zip(col_names, data)), schema=schema)


def load_trades_from_clickhouse(
    client: Client,
    database: str,
    symbol: str,
    start: datetime | None,
    end: datetime | None,
    *,
    table: str = "trades",
) -> Any:
    """Columnar SELECT of trades for one symbol and optional ``[start, end)`` window."""
    import polars as pl

    where, params = symbol_time_where(symbol, start, end)
    sql = f"""
    SELECT exchange_ts, side, price, quantity
    FROM {database}.{table}
    WHERE {' AND '.join(where)}
    """
    data, columns_meta = client.execute(
        sql, params, columnar=True, with_column_types=True
    )
    schema = _empty_trades_schema()
    if not data or not data[0]:
        return pl.DataFrame(schema=schema)
    col_names = [c[0] for c in columns_meta]
    return pl.DataFrame(dict(zip(col_names, data)), schema=schema)


def delete_features_kv_range(
    client: Client,
    database: str,
    symbol: str,
    start: datetime | None,
    end: datetime | None,
    *,
    feature_names: list[str] | None = None,
) -> None:
    where, params = symbol_time_where(symbol, start, end)
    if feature_names:
        where.append("feature_name IN %(feature_names)s")
        params["feature_names"] = tuple(feature_names)
    sql = f"ALTER TABLE {database}.features_kv DELETE WHERE {' AND '.join(where)}"
    try:
        client.execute(sql, params)
    except ServerException as e:
        if e.code == 497:
            raise SystemExit(
                "ClickHouse refused ALTER DELETE on features_kv (need ALTER DELETE grant).\n"
                "Run as admin, e.g.:\n"
                f"  GRANT ALTER DELETE ON {database}.features_kv TO collector;\n"
                "Updated default grants: containers/bybit-collector/clickhouse/init.sql"
            ) from e
        raise


def count_rows(
    client: Client,
    database: str,
    table: str,
    symbol: str,
    *,
    start: datetime | None = None,
    end: datetime | None = None,
    end_inclusive: bool = False,
) -> int:
    where, params = symbol_time_where(
        symbol, start, end, end_inclusive=end_inclusive
    )
    sql = f"SELECT count() FROM {database}.{table} WHERE {' AND '.join(where)}"
    return int(client.execute(sql, params)[0][0])
