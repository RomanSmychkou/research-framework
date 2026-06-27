"""
Load repo-root `.env` / `.env.local` and expose typed defaults for pipeline scripts.

Priority when resolving each setting:
  1. Variable already in the process environment (shell export)
  2. `.env.local` (overrides `.env` for the same key)
  3. `.env`
  4. Hardcoded fallback in this module / imported module constants

Not configured here (pass per run on CLI): time ranges, run-specific symbols if you
prefer CLI-only, secrets you do not want in a shared file.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import TypeVar

_T = TypeVar("_T")


def _strip_quotes(value: str) -> str:
    v = value.strip()
    if len(v) >= 2 and v[0] == v[-1] and v[0] in "\"'":
        return v[1:-1]
    return v


def _load_env_file(path: Path, *, override: bool) -> None:
    if not path.is_file():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].strip()
        if "=" not in line:
            continue
        key, _, val = line.partition("=")
        key = key.strip()
        if not key:
            continue
        val = _strip_quotes(val.strip())
        if override or key not in os.environ:
            os.environ[key] = val


def _discover_repo_root() -> Path:
    """Walk up from this package for pyproject.toml (repo root)."""
    here = Path(__file__).resolve().parent
    for candidate in (here, *here.parents):
        if (candidate / "pyproject.toml").is_file():
            return candidate
    if here.name == "quant_system" and here.parent.name == "src":
        return here.parent.parent
    return here


def _bootstrap_env_files() -> None:
    root = _discover_repo_root()
    _load_env_file(root / ".env", override=False)
    _load_env_file(root / ".env.local", override=True)


_bootstrap_env_files()


def env_str(key: str, default: str) -> str:
    raw = os.environ.get(key)
    if raw is None:
        return default
    stripped = raw.strip()
    return default if stripped == "" else stripped


def env_optional_str(key: str) -> str | None:
    raw = os.environ.get(key)
    if raw is None:
        return None
    stripped = raw.strip()
    return None if stripped == "" else stripped


def env_int(key: str, default: int) -> int:
    raw = os.environ.get(key)
    if raw is None or not str(raw).strip():
        return default
    return int(str(raw).strip())


def env_optional_int(key: str) -> int | None:
    raw = os.environ.get(key)
    if raw is None or not str(raw).strip():
        return None
    return int(str(raw).strip())


def env_float(key: str, default: float) -> float:
    raw = os.environ.get(key)
    if raw is None or not str(raw).strip():
        return default
    return float(str(raw).strip())


def env_optional_float(key: str) -> float | None:
    raw = os.environ.get(key)
    if raw is None or not str(raw).strip():
        return None
    return float(str(raw).strip())


def env_bool(key: str, default: bool) -> bool:
    raw = os.environ.get(key)
    if raw is None or not str(raw).strip():
        return default
    return str(raw).strip().lower() in ("1", "true", "yes", "on")


def env_str_list(key: str, default: list[str]) -> list[str]:
    raw = os.environ.get(key)
    if raw is None or not raw.strip():
        return list(default)
    return [part for part in re.split(r"[\s,]+", raw.strip()) if part]


def env_insert_row_override() -> int | None:
    """Optional explicit INSERT batch size (legacy env key names)."""
    for key in (
        "CLICKHOUSE_INSERT_BATCH_ROWS",
        "FEATURES_KV_BATCH_ROWS",
        "UPLOAD_BATCH_SIZE",
    ):
        value = env_optional_int(key)
        if value is not None:
            return value
    return None


def project_root() -> Path:
    """Repository root: QUANT_PROJECT_ROOT or directory with pyproject.toml."""
    override = env_optional_str("QUANT_PROJECT_ROOT")
    if override:
        p = Path(override)
        return p.resolve() if p.is_absolute() else (Path.cwd() / p).resolve()
    return _discover_repo_root()


def resolve_project_path(spec: str) -> Path:
    """Absolute path: use as-is; otherwise resolve under project_root()."""
    p = Path(spec)
    if p.is_absolute():
        return p.resolve()
    return (project_root() / p).resolve()


PROJECT_ROOT = project_root()

# --- Paths (relative to PROJECT_ROOT unless absolute) ---

COINS_DIR = env_str("COINS_DIR", "coins")
HISTORICAL_UPLOAD_DIR = env_str("HISTORICAL_UPLOAD_DIR", "coins/_historical_data/spot_pt")
# Per-symbol artifacts: features JSON, future stage-03 outputs (default: COINS_DIR/<symbol>/).
SYMBOL_ARTIFACTS_DIR = env_str("SYMBOL_ARTIFACTS_DIR", COINS_DIR)


def coins_root() -> Path:
    return resolve_project_path(COINS_DIR)


def historical_upload_dir() -> Path:
    return resolve_project_path(HISTORICAL_UPLOAD_DIR)


def symbol_artifacts_dir(symbol: str) -> Path:
    """Directory for one symbol under SYMBOL_ARTIFACTS_DIR (default coins/<symbol>)."""
    base = resolve_project_path(SYMBOL_ARTIFACTS_DIR)
    return (base / symbol).resolve()


def upload_manifest_path(data_dir: Path | None = None) -> Path:
    """Upload manifest JSON inside the historical data folder (or override dir)."""
    root = data_dir if data_dir is not None else historical_upload_dir()
    return (root / UPLOAD_CONFIG_NAME).resolve()


# --- ClickHouse (shared) ---

CLICKHOUSE_HOST = env_str("CLICKHOUSE_HOST", "127.0.0.1")
CLICKHOUSE_PORT = env_int("CLICKHOUSE_PORT", 9000)
CLICKHOUSE_USER = env_str("CLICKHOUSE_USER", "default")
CLICKHOUSE_PASSWORD = env_str("CLICKHOUSE_PASSWORD", "")
CLICKHOUSE_DATABASE = env_str("CLICKHOUSE_DATABASE", "crypto_db")

# --- 00_upload_history ---

UPLOAD_CONFIG_NAME = env_str("UPLOAD_CONFIG_NAME", "upload_history_config.json")
UPLOAD_TABLE = env_str("UPLOAD_TABLE", "spot_trades")

# Stage 00: rows to accumulate from CSV before flush (capped per insert chunk).
CLICKHOUSE_CSV_BUFFER_ROWS = env_int("CLICKHOUSE_CSV_BUFFER_ROWS", 1_000_000)
DEFAULT_INSERT_BATCH_ROWS = 50_000
INSERT_MIN_ROWS = 1_000


def insert_batch_rows(row_override: int | None = None) -> int:
    """Rows per INSERT batch (CLI override, then env, then default)."""
    if row_override is not None:
        return max(INSERT_MIN_ROWS, row_override)
    override = env_insert_row_override()
    if override is not None:
        return max(INSERT_MIN_ROWS, override)
    return max(
        INSERT_MIN_ROWS,
        env_int("CLICKHOUSE_INSERT_BATCH_ROWS", DEFAULT_INSERT_BATCH_ROWS),
    )

# --- 01_add_features / 01_add_target ---

QUANT_SYMBOL = env_optional_str("QUANT_SYMBOL")

FEATURE_WINDOWS_DEFAULT = env_str_list(
    "FEATURE_WINDOWS",
    ["3s", "10s", "30s", "75s", "200s", "320s", "16m", "62m", "183m", "365m", "3605m", "7210m"],
)
# Single label horizon for stage 01 (column pnl_after_<PNL_WINDOW>).
PNL_WINDOW = env_str("PNL_WINDOW", "3m")
# Optional full target column or bare horizon; else pnl_after_<PNL_WINDOW>.
TARGET = env_optional_str("TARGET")

ENTRY_FEE = env_float("ENTRY_FEE", 0.001)
EXIT_FEE = env_float("EXIT_FEE", 0.001)
SLIPPAGE_ENTRY = env_float("SLIPPAGE_ENTRY", 0.0005)
SLIPPAGE_EXIT = env_float("SLIPPAGE_EXIT", 0.0005)

IO_BATCH_SPAN = env_str("IO_BATCH_SPAN", "30d")
RECEIVE_TIMEOUT_SEC = env_optional_int("RECEIVE_TIMEOUT_SEC")

STAGE01_REPLACE_RANGE = env_bool("STAGE01_REPLACE_RANGE", True)

# --- 02 output paths ---

EXPORT_JSON_NAME = env_str("EXPORT_JSON_NAME", "actual_features.json")
# Optional full override; else SYMBOL_ARTIFACTS_DIR/<symbol>/<EXPORT_JSON_NAME>
STAGE02_EXPORT_DIR = env_optional_str("STAGE02_EXPORT_DIR")

# --- ClickHouse INSERT (stages 00/01; lazy import avoids quant_env ↔ clickhouse_insert cycle) ---

_IO_EXPORTS = frozenset(
    {
        "build_clickhouse_client",
        "client_from_args",
        "parse_cli_datetime",
        "symbol_time_where",
    }
)

_INSERT_EXPORTS = frozenset(
    {
        "clickhouse_insert_polars_kv",
        "clickhouse_insert_rows",
        "clickhouse_insert_trades_columnar",
        "is_retriable_insert_error",
        "upload_csv_buffer_flush_rows",
    }
)


def __getattr__(name: str) -> object:
    if name in _IO_EXPORTS:
        from . import clickhouse_io as _io

        return getattr(_io, name)
    if name in _INSERT_EXPORTS:
        from . import clickhouse_insert as _ins

        return getattr(_ins, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
