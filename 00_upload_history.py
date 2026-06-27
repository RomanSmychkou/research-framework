"""
Upload historical trades from *.csv.gz files into ClickHouse (pce_raw.trades).

CSV rows: id, timestamp (ms UTC), price, volume, side[, extra columns ignored].
Filenames: {SYMBOL}-{YYYY-MM}.csv.gz or {SYMBOL}_{YYYY-MM-DD}.csv.gz.

Hot path: binary gzip + columnar INSERT; background inserter overlaps parse with ClickHouse.
"""

from __future__ import annotations

import argparse
import gzip
import io
import json
import os
import queue
import re
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

import bootstrap_path

bootstrap_path.ensure_src_on_path(Path(__file__))

from scripts.project_snapshot import do_project_snapshot  # noqa: E402
from scripts.run_context_logger import finish_run, start_run  # noqa: E402

do_project_snapshot()

if TYPE_CHECKING:
    from clickhouse_driver import Client

from quant_system import quant_env  # noqa: E402, F401 — load .env before defaults
from quant_system.clickhouse_io import build_clickhouse_client  # noqa: E402

FILENAME_RE_MONTH = re.compile(r"^(.+)-(\d{4})-(\d{2})\.csv\.gz$")
FILENAME_RE_DAY = re.compile(r"^(.+)_(\d{4})-(\d{2})-(\d{2})\.csv\.gz$")
DEFAULT_CONFIG_NAME = quant_env.UPLOAD_CONFIG_NAME
INSERT_SQL = """
INSERT INTO {database}.{table} (
    exchange_ts, symbol, trade_id, price, quantity, side
) VALUES
"""

# Progress line every N rows read (stderr).
READ_PROGRESS_EVERY_ROWS = 500_000
_GZIP_IO_BUFFER = 1 << 20  # 1 MiB compressed read buffer
_MS_TO_DT = 1.0 / 1000.0
_STAGE_LOG_LOCK = threading.Lock()


def ms_to_ch_datetime(ms: int) -> datetime:
    return datetime.fromtimestamp(ms * _MS_TO_DT, tz=timezone.utc).replace(tzinfo=None)


def iso_range_utc(min_ms: int | None, max_ms: int | None) -> tuple[str | None, str | None]:
    if min_ms is None or max_ms is None:
        return None, None
    start = datetime.fromtimestamp(min_ms * _MS_TO_DT, tz=timezone.utc)
    end = datetime.fromtimestamp(max_ms * _MS_TO_DT, tz=timezone.utc)
    return (
        start.isoformat().replace("+00:00", "Z"),
        end.isoformat().replace("+00:00", "Z"),
    )


def parse_symbol(path: Path) -> str:
    name = path.name
    m = FILENAME_RE_MONTH.match(name)
    if m:
        return m.group(1)
    m = FILENAME_RE_DAY.match(name)
    if m:
        return m.group(1)
    raise ValueError(
        "File name must match SYMBOL-YYYY-MM.csv.gz or "
        f"SYMBOL_YYYY-MM-DD.csv.gz, got: {name!r}"
    )


def load_upload_config(config_path: Path) -> dict[str, Any]:
    if not config_path.is_file():
        return {"files": {}}
    with config_path.open(encoding="utf-8") as fp:
        raw = json.load(fp)
    if "files" not in raw:
        raw["files"] = {}
    return raw


def save_upload_config(config_path: Path, data: dict[str, Any]) -> None:
    config_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = config_path.with_suffix(config_path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="\n") as fp:
        json.dump(data, fp, indent=2, ensure_ascii=False)
        fp.write("\n")
    tmp.replace(config_path)


def _parse_ts_ms(raw: bytes) -> int:
    try:
        return int(raw)
    except ValueError:
        return int(float(raw))


def _parse_side(raw: bytes) -> str:
    if not raw:
        return "buy"
    lead = raw[0] | 0x20
    if lead == ord("b"):
        return "buy"
    if lead == ord("s"):
        return "sell"
    text = raw.decode("utf-8", "replace").strip().lower()
    if text in ("buy", "b"):
        return "buy"
    if text in ("sell", "s"):
        return "sell"
    return text or "buy"


class TradeColumnBatch:
    """Column buffers for one flush (columnar INSERT)."""

    __slots__ = (
        "symbol",
        "_dt_cache",
        "exchange_ts",
        "trade_id",
        "price",
        "quantity",
        "side",
    )

    def __init__(
        self,
        symbol: str,
        *,
        dt_cache: dict[int, datetime] | None = None,
    ) -> None:
        self.symbol = symbol
        self._dt_cache = dt_cache if dt_cache is not None else {}
        self.exchange_ts: list[datetime] = []
        self.trade_id: list[str] = []
        self.price: list[float] = []
        self.quantity: list[float] = []
        self.side: list[str] = []

    def _ms_to_dt(self, ms: int) -> datetime:
        cached = self._dt_cache.get(ms)
        if cached is not None:
            return cached
        dt = ms_to_ch_datetime(ms)
        self._dt_cache[ms] = dt
        return dt

    def append_line(self, line: bytes, start_ms: int | None) -> int | None:
        """Append row; return ts_ms or None if skipped."""
        parts = line.split(b",")
        if len(parts) < 5:
            return None
        ts_ms = _parse_ts_ms(parts[1])
        if start_ms is not None and ts_ms < start_ms:
            return None
        ex_dt = self._ms_to_dt(ts_ms)
        self.exchange_ts.append(ex_dt)
        self.trade_id.append(parts[0].decode("utf-8", "replace").strip())
        self.price.append(float(parts[2]))
        self.quantity.append(float(parts[3]))
        self.side.append(_parse_side(parts[4].strip()))
        return ts_ms

    def __len__(self) -> int:
        return len(self.exchange_ts)

    def columns(self) -> list[list[Any]]:
        n = len(self.exchange_ts)
        sym = self.symbol
        return [
            self.exchange_ts,
            [sym] * n,
            self.trade_id,
            self.price,
            self.quantity,
            self.side,
        ]


def row_from_csv_line(symbol: str, line: bytes, start_ms: int | None) -> tuple[Any, ...] | None:
    """Parse one CSV line into row tuple (tests)."""
    batch = TradeColumnBatch(symbol)
    ts_ms = batch.append_line(line, start_ms)
    if ts_ms is None:
        return None
    row = (
        batch.exchange_ts[0],
        symbol,
        batch.trade_id[0],
        batch.price[0],
        batch.quantity[0],
        batch.side[0],
        ts_ms,
    )
    return row


def row_to_tuple(symbol: str, row: list[str], ts_ms: int) -> tuple:
    """Build INSERT row (tests / legacy). Caller must ensure len(row) >= 5."""
    trade_id = str(row[0]).strip()
    price = float(row[2])
    qty = float(row[3])
    side = _parse_side(str(row[4]).strip().encode("utf-8"))
    ex_dt = ms_to_ch_datetime(ts_ms)
    return (ex_dt, symbol, trade_id, price, qty, side)


def _iter_data_lines(gz_path: Path):
    """Yield stripped UTF-8 data lines from gzip CSV (skip header)."""
    with gzip.open(gz_path, "rb") as gz_raw:
        stream = io.BufferedReader(gz_raw, buffer_size=_GZIP_IO_BUFFER)
        first = True
        for raw in stream:
            if first:
                first = False
                if raw.startswith(b"\xef\xbb\xbf"):
                    raw = raw[3:]
                continue
            if raw.endswith(b"\r\n"):
                raw = raw[:-2]
            elif raw.endswith(b"\n"):
                raw = raw[:-1]
            elif raw.endswith(b"\r"):
                raw = raw[:-1]
            if raw:
                yield raw


def parse_start_ms(s: str | None) -> int | None:
    """Inclusive lower bound on CSV trade timestamp (ms UTC)."""
    if s is None:
        return None
    v = s.strip()
    if not v:
        return None
    if v.endswith("Z"):
        v = v[:-1] + "+00:00"
    dt = datetime.fromisoformat(v)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    else:
        dt = dt.astimezone(timezone.utc)
    return int(dt.timestamp() * 1000)


def process_gz_file(
    client: Client,
    gz_path: Path,
    symbol: str,
    database: str,
    table: str,
    insert_batch_rows: int,
    start_ms: int | None = None,
    *,
    on_progress: Callable[[str], None] | None = None,
    log_each_flush: bool = False,
    file_t0: float | None = None,
) -> tuple[int, int | None, int | None]:
    """Returns (row_count, min_ts_ms, max_ts_ms)."""
    min_ms: int | None = None
    max_ms: int | None = None
    total = 0
    flush_idx = 0
    insert_sql = INSERT_SQL.format(database=database, table=table)
    last_milestone = 0
    t0_file = file_t0 if file_t0 is not None else time.perf_counter()
    dt_cache: dict[int, datetime] = {}

    insert_chunk_rows = insert_batch_rows
    buffer_flush_at = quant_env.upload_csv_buffer_flush_rows(insert_chunk_rows)

    batch = TradeColumnBatch(symbol, dt_cache=dt_cache)
    insert_errors: list[BaseException] = []
    insert_queue: queue.Queue[TradeColumnBatch | None] = queue.Queue(maxsize=1)

    def _raise_if_insert_failed() -> None:
        if insert_errors:
            raise insert_errors[0]

    def _inserter() -> None:
        try:
            while True:
                item = insert_queue.get()
                if item is None:
                    break
                quant_env.clickhouse_insert_trades_columnar(
                    client,
                    insert_sql,
                    item.columns(),
                    batch_rows=insert_chunk_rows,
                )
        except BaseException as exc:
            insert_errors.append(exc)

    worker = threading.Thread(
        target=_inserter,
        name=f"upload_history-insert-{symbol[:16]}",
        daemon=True,
    )
    worker.start()

    def _enqueue_flush() -> None:
        nonlocal batch, flush_idx
        _raise_if_insert_failed()
        if not batch:
            return
        flush_idx += 1
        if log_each_flush and on_progress is not None:
            on_progress(f"  → flush {flush_idx}: {len(batch):,} rows → CH")
        # Do not block forever when inserter thread died on network/CH error.
        while True:
            _raise_if_insert_failed()
            try:
                insert_queue.put(batch, timeout=0.2)
                break
            except queue.Full:
                if not worker.is_alive():
                    _raise_if_insert_failed()
                    raise RuntimeError("upload_history inserter thread stopped unexpectedly")
        batch = TradeColumnBatch(symbol, dt_cache=dt_cache)

    try:
        for line in _iter_data_lines(gz_path):
            _raise_if_insert_failed()
            ts_ms = batch.append_line(line, start_ms)
            if ts_ms is None:
                continue
            if min_ms is None or ts_ms < min_ms:
                min_ms = ts_ms
            if max_ms is None or ts_ms > max_ms:
                max_ms = ts_ms
            total += 1
            if (
                on_progress is not None
                and total - last_milestone >= READ_PROGRESS_EVERY_ROWS
            ):
                elapsed = max(time.perf_counter() - t0_file, 1e-6)
                rate = total / elapsed
                on_progress(
                    f"{gz_path.name}: {total:,} rows, {flush_idx} CH batches "
                    f"({rate:,.0f} rows/s)"
                )
                last_milestone = total
            if len(batch) >= buffer_flush_at:
                _enqueue_flush()
        _enqueue_flush()
    finally:
        if worker.is_alive():
            while True:
                _raise_if_insert_failed()
                try:
                    insert_queue.put(None, timeout=0.2)
                    break
                except queue.Full:
                    if not worker.is_alive():
                        break
        worker.join()

    if insert_errors:
        raise insert_errors[0]

    return total, min_ms, max_ms


def _vprint(args: argparse.Namespace, *a: Any, **kw: Any) -> None:
    if (not args.quiet) and bool(getattr(args, "verbose", False)):
        print(*a, **kw)


def _stage_log(args: argparse.Namespace, msg: str, *, t0: float | None = None) -> None:
    if args.quiet:
        return
    prefix = "[upload_history"
    if t0 is not None:
        prefix += f" +{time.perf_counter() - t0:.1f}s"
    prefix += "] "
    line = f"{prefix}{msg}\n"
    with _STAGE_LOG_LOCK:
        sys.stderr.write(line)
        sys.stderr.flush()


@dataclass(frozen=True)
class _PendingUpload:
    gz_path: Path
    symbol: str
    rel_name: str


@dataclass
class _UploadFileOutcome:
    rel_name: str
    row_count: int = 0
    min_ms: int | None = None
    max_ms: int | None = None
    elapsed_s: float = 0.0
    error: BaseException | None = None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Load historical *.csv.gz trades into ClickHouse pce_raw.trades.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "folder",
        type=Path,
        nargs="?",
        default=None,
        help=(
            "Directory with SYMBOL-YYYY-MM.csv.gz (default: HISTORICAL_UPLOAD_DIR from .env, "
            "relative to QUANT_PROJECT_ROOT)."
        ),
    )
    p.add_argument(
        "--config-name",
        default=DEFAULT_CONFIG_NAME,
        help=(
            "Upload manifest filename inside the data folder "
            "(env UPLOAD_CONFIG_NAME)."
        ),
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Insert every file; ignore prior manifest entries.",
    )
    p.add_argument(
        "--host",
        default=quant_env.CLICKHOUSE_HOST,
        help="ClickHouse host (env CLICKHOUSE_HOST, then hardcoded default).",
    )
    p.add_argument(
        "--port",
        type=int,
        default=quant_env.CLICKHOUSE_PORT,
        help="ClickHouse native TCP port (env CLICKHOUSE_PORT).",
    )
    p.add_argument(
        "--user",
        default=quant_env.CLICKHOUSE_USER,
        help="ClickHouse user (env CLICKHOUSE_USER).",
    )
    p.add_argument(
        "--password",
        default=quant_env.CLICKHOUSE_PASSWORD,
        help="ClickHouse password (env CLICKHOUSE_PASSWORD).",
    )
    p.add_argument(
        "--database",
        default=quant_env.CLICKHOUSE_DATABASE,
        help="ClickHouse database (env CLICKHOUSE_DATABASE).",
    )
    p.add_argument(
        "--table",
        default=quant_env.UPLOAD_TABLE,
        help="ClickHouse destination table (env UPLOAD_TABLE).",
    )
    p.add_argument(
        "--insert-batch-rows",
        "--batch-rows",
        "--batch-size",
        dest="insert_batch_rows",
        type=int,
        default=None,
        help=(
            "Rows per INSERT batch (overrides CLICKHOUSE_INSERT_BATCH_ROWS env). "
            f"Default: {quant_env.DEFAULT_INSERT_BATCH_ROWS:,}."
        ),
    )
    p.add_argument(
        "--start",
        type=str,
        default=None,
        help="Optional inclusive UTC lower bound (ISO-8601); skip CSV rows with earlier timestamps.",
    )
    p.add_argument(
        "--only-symbols",
        nargs="+",
        default=None,
        metavar="SYMBOL",
        help=(
            "Only process *.csv.gz for these symbols. "
            "Omit the flag to process every file in the folder."
        ),
    )
    p.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="Suppress per-file progress and skip messages (errors go to stderr).",
    )
    p.add_argument(
        "--verbose",
        action="store_true",
        help="Verbose per-file progress logs.",
    )
    return p.parse_args()


def _upload_one_file(
    args: argparse.Namespace,
    job: _PendingUpload,
    insert_batch_rows: int,
    start_ms: int | None,
    session_t0: float,
) -> _UploadFileOutcome:
    """Upload a single gzip CSV; owns ClickHouse client for this call."""
    outcome = _UploadFileOutcome(rel_name=job.rel_name)
    client = build_clickhouse_client(
        host=args.host,
        port=args.port,
        user=args.user,
        password=args.password,
    )
    file_t0 = time.perf_counter()
    try:
        if not args.quiet:
            _stage_log(args, f"▸ {job.rel_name} ({job.symbol})", t0=session_t0)

        def _on_progress(msg: str) -> None:
            _stage_log(args, f"{job.rel_name}: {msg}", t0=session_t0)

        n, min_ms, max_ms = process_gz_file(
            client,
            job.gz_path,
            job.symbol,
            args.database,
            args.table,
            insert_batch_rows,
            start_ms,
            on_progress=None if args.quiet else _on_progress,
            log_each_flush=args.verbose,
            file_t0=file_t0,
        )
        outcome.row_count = int(n)
        outcome.min_ms = min_ms
        outcome.max_ms = max_ms
        outcome.elapsed_s = time.perf_counter() - file_t0
        if n == 0 and not args.quiet:
            _stage_log(args, f"  {job.rel_name}: skip manifest (0 rows)", t0=session_t0)
        elif not args.quiet:
            start_iso, end_iso = iso_range_utc(min_ms, max_ms)
            rate_f = n / outcome.elapsed_s if outcome.elapsed_s > 0 else 0.0
            _stage_log(
                args,
                f"  ✓ {job.rel_name}: {n:,} rows in {outcome.elapsed_s:.1f}s "
                f"({rate_f:,.0f} rows/s) {start_iso} … {end_iso}",
                t0=session_t0,
            )
    except BaseException as exc:
        outcome.error = exc
    finally:
        try:
            client.disconnect()
        except Exception:
            pass
    return outcome


def main() -> None:
    args = parse_args()
    t0 = time.perf_counter()
    if (
        args.insert_batch_rows is not None
        and args.insert_batch_rows < quant_env.INSERT_MIN_ROWS
    ):
        raise SystemExit(
            f"--insert-batch-rows must be >= {quant_env.INSERT_MIN_ROWS} "
            "for stable ClickHouse inserts."
        )
    folder = (
        args.folder.expanduser().resolve()
        if args.folder is not None
        else quant_env.historical_upload_dir()
    )
    if not folder.is_dir():
        raise SystemExit(f"Not a directory: {folder}")

    config_path = folder / args.config_name
    manifest = load_upload_config(config_path)
    uploaded: dict[str, Any] = manifest.setdefault("files", {})

    gz_names = sorted(n for n in os.listdir(folder) if n.endswith(".csv.gz"))
    if not gz_names:
        print(f"No *.csv.gz files in {folder}", file=sys.stderr)
        return

    insert_batch_rows = quant_env.insert_batch_rows(args.insert_batch_rows)
    start_ms = parse_start_ms(args.start)
    if not args.quiet:
        ch = f"{args.host}:{args.port}/{args.database}.{args.table}"
        _stage_log(
            args,
            f"{len(gz_names)} files → {ch} | insert_batch_rows={insert_batch_rows:,}",
            t0=t0,
        )
    only_syms: frozenset[str] | None = None
    if args.only_symbols:
        only_syms = frozenset(s.strip() for s in args.only_symbols if s.strip())
    scanned = 0
    uploaded_files = 0
    inserted_rows = 0
    pending: list[_PendingUpload] = []

    for name in gz_names:
        gz_path = folder / name
        scanned += 1
        try:
            symbol = parse_symbol(gz_path)
        except ValueError as e:
            _vprint(args, f"Skip {gz_path.name}: {e}")
            continue

        if only_syms is not None and symbol not in only_syms:
            _vprint(args, f"Skip (--only-symbols): {gz_path.name} ({symbol})")
            continue

        rel_name = gz_path.name
        if not args.force and rel_name in uploaded:
            _vprint(args, f"Skip (already in manifest): {rel_name}")
            continue

        pending.append(_PendingUpload(gz_path=gz_path, symbol=symbol, rel_name=rel_name))

    if pending:
        for job in pending:
            outcome = _upload_one_file(args, job, insert_batch_rows, start_ms, t0)
            if outcome.error is not None:
                raise outcome.error
            if outcome.row_count <= 0:
                continue
            start_iso, end_iso = iso_range_utc(outcome.min_ms, outcome.max_ms)
            uploaded[outcome.rel_name] = {
                "start_date": start_iso,
                "end_date": end_iso,
            }
            save_upload_config(config_path, manifest)
            uploaded_files += 1
            inserted_rows += outcome.row_count

    if not args.quiet:
        print(
            f"Stage 00 OK: scanned={scanned} uploaded_files={uploaded_files} inserted_rows={inserted_rows}",
            flush=True,
        )


if __name__ == "__main__":
    uid = start_run("00_upload_history.py", sys.argv[1:], stage="upload_history-upload-history")
    try:
        main()
        finish_run(uid, state="success")
    except Exception as exc:  # noqa: BLE001
        finish_run(uid, state="error", extras={"error": str(exc)})
        raise
