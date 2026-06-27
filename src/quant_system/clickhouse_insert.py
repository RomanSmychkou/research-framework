"""Fixed-size batched ClickHouse INSERTs with split-and-retry on retriable errors."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Sequence

if TYPE_CHECKING:
    from clickhouse_driver import Client

INSERT_MIN_ROWS = 1_000
INSERT_MAX_RETRY_SPLITS = 8

_RETRIABLE_INSERT_CODES = frozenset({62, 159, 241, 252, 318})


def upload_csv_buffer_flush_rows(insert_chunk_rows: int) -> int:
    """Rows to read from CSV before flushing; capped by insert chunk size."""
    from . import quant_env

    cap = quant_env.CLICKHOUSE_CSV_BUFFER_ROWS
    return max(INSERT_MIN_ROWS, min(cap, insert_chunk_rows))


def is_retriable_insert_error(exc: BaseException) -> bool:
    from clickhouse_driver.errors import ServerException

    if not isinstance(exc, ServerException):
        return False
    code = getattr(exc, "code", None)
    if code in _RETRIABLE_INSERT_CODES:
        return True
    msg = str(exc).lower()
    needles = (
        "max query size",
        "too large",
        "memory limit",
        "timeout",
        "max insert block",
    )
    return any(n in msg for n in needles)


def _insert_rows_with_retry(
    client: Client,
    sql: str,
    rows: list[tuple[Any, ...]],
    *,
    min_rows: int = INSERT_MIN_ROWS,
    depth: int = 0,
) -> None:
    if not rows:
        return
    from clickhouse_driver.errors import ServerException

    try:
        client.execute(sql, rows)
    except ServerException as exc:
        if (
            depth >= INSERT_MAX_RETRY_SPLITS
            or len(rows) <= min_rows
            or not is_retriable_insert_error(exc)
        ):
            raise
        mid = max(min_rows, len(rows) // 2)
        if mid >= len(rows):
            raise
        _insert_rows_with_retry(
            client, sql, rows[:mid], min_rows=min_rows, depth=depth + 1
        )
        _insert_rows_with_retry(
            client, sql, rows[mid:], min_rows=min_rows, depth=depth + 1
        )


def _insert_columnar_with_retry(
    client: Client,
    sql: str,
    columns: Sequence[Sequence[Any]],
    *,
    min_rows: int = INSERT_MIN_ROWS,
    depth: int = 0,
) -> None:
    if not columns or not columns[0]:
        return
    n = len(columns[0])
    from clickhouse_driver.errors import ServerException

    try:
        client.execute(sql, columns, columnar=True)
    except ServerException as exc:
        if (
            depth >= INSERT_MAX_RETRY_SPLITS
            or n <= min_rows
            or not is_retriable_insert_error(exc)
        ):
            raise
        mid = max(min_rows, n // 2)
        if mid >= n:
            raise
        _insert_columnar_with_retry(
            client,
            sql,
            [col[:mid] for col in columns],
            min_rows=min_rows,
            depth=depth + 1,
        )
        _insert_columnar_with_retry(
            client,
            sql,
            [col[mid:] for col in columns],
            min_rows=min_rows,
            depth=depth + 1,
        )


def clickhouse_insert_trades_columnar(
    client: Client,
    sql: str,
    columns: Sequence[Sequence[Any]],
    *,
    batch_rows: int,
    progress: Callable[[str], None] | None = None,
) -> None:
    """Columnar INSERT for trades in fixed-size chunks."""
    if not columns or not columns[0]:
        return
    n = len(columns[0])
    chunk_rows = max(INSERT_MIN_ROWS, batch_rows)
    n_batches = (n + chunk_rows - 1) // chunk_rows
    done = 0
    for bi in range(n_batches):
        end = min(done + chunk_rows, n)
        chunk = [col[done:end] for col in columns]
        if progress is not None and n_batches > 1:
            progress(
                f"  INSERT batch {bi + 1}/{n_batches} "
                f"(rows {done + 1}…{end} / {n})"
            )
        _insert_columnar_with_retry(client, sql, chunk)
        done = end


def clickhouse_insert_rows(
    client: Client,
    sql: str,
    rows: list[tuple[Any, ...]],
    *,
    batch_rows: int | None = None,
    row_override: int | None = None,
    progress: Callable[[str], None] | None = None,
) -> None:
    """Insert tuples in fixed batches; halve and retry on retriable server errors."""
    if not rows:
        return
    from .quant_env import insert_batch_rows as _batch_rows

    chunk_rows = batch_rows if batch_rows is not None else _batch_rows(row_override)
    total = len(rows)
    n_batches = (total + chunk_rows - 1) // chunk_rows
    done = 0
    for bi in range(n_batches):
        chunk = rows[done : done + chunk_rows]
        if progress is not None and n_batches > 1:
            end = done + len(chunk)
            progress(
                f"  INSERT batch {bi + 1}/{n_batches} "
                f"(rows {done + 1}…{end} / {total})"
            )
        _insert_rows_with_retry(client, sql, chunk)
        done += len(chunk)


def clickhouse_insert_polars_kv(
    client: Client,
    rows: Any,
    database: str,
    *,
    batch_rows: int | None = None,
    row_override: int | None = None,
    progress: Callable[[str], None] | None = None,
) -> None:
    """Insert long features_kv layout from a Polars DataFrame."""
    import polars as pl

    optional_cols = [
        col for col in ("feature_version", "contract_hash") if col in rows.columns
    ]
    insert_cols = ["symbol", "exchange_ts", "feature_name", *optional_cols, "feature_value"]
    sql = f"INSERT INTO {database}.features_kv ({', '.join(insert_cols)}) VALUES"
    required_expr = (
        pl.col("feature_value").is_finite()
        & pl.col("exchange_ts").is_not_null()
        & pl.col("feature_name").is_not_null()
    )
    if "feature_version" in optional_cols:
        required_expr &= pl.col("feature_version").is_not_null()
    if "contract_hash" in optional_cols:
        required_expr &= pl.col("contract_hash").is_not_null()
    mat = (
        rows.filter(required_expr)
        .select(insert_cols)
    )
    if mat.height == 0:
        return
    tuples = mat.rows()
    clickhouse_insert_rows(
        client,
        sql,
        tuples,
        batch_rows=batch_rows,
        row_override=row_override,
        progress=progress,
    )


def build_clickhouse_client(*args: Any, **kwargs: Any) -> Client:
    from .clickhouse_io import build_clickhouse_client as _build

    return _build(*args, **kwargs)
