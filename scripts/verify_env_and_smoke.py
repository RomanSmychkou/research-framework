#!/usr/bin/env python3
"""
Verify .env.example ↔ quant_env parity and optional ClickHouse smoke (narrow window).

Usage (from repo root):
  python scripts/verify_env_and_smoke.py
  python scripts/verify_env_and_smoke.py --symbol AIXBTUSDT --start 2026-01-01 --end 2026-01-03

Exits 0 if env checks pass; CH stages are skipped when the server is unreachable.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import bootstrap_path  # noqa: E402

ROOT = bootstrap_path.ensure_src_on_path(Path(__file__))

from quant_system import quant_env as _qe  # noqa: E402

# Keys in .env.example (active assignments) → quant_env attribute + how to compare.
ENV_CHECKS: list[tuple[str, str, str]] = [
    ("COINS_DIR", "COINS_DIR", "str"),
    ("HISTORICAL_UPLOAD_DIR", "HISTORICAL_UPLOAD_DIR", "str"),
    ("SYMBOL_ARTIFACTS_DIR", "SYMBOL_ARTIFACTS_DIR", "str"),
    ("CLICKHOUSE_HOST", "CLICKHOUSE_HOST", "str"),
    ("CLICKHOUSE_PORT", "CLICKHOUSE_PORT", "int"),
    ("CLICKHOUSE_USER", "CLICKHOUSE_USER", "str"),
    ("CLICKHOUSE_PASSWORD", "CLICKHOUSE_PASSWORD", "str"),
    ("CLICKHOUSE_DATABASE", "CLICKHOUSE_DATABASE", "str"),
    ("UPLOAD_CONFIG_NAME", "UPLOAD_CONFIG_NAME", "str"),
    ("UPLOAD_TABLE", "UPLOAD_TABLE", "str"),
    ("CLICKHOUSE_INSERT_BATCH_ROWS", "DEFAULT_INSERT_BATCH_ROWS", "int"),
    ("CLICKHOUSE_CSV_BUFFER_ROWS", "CLICKHOUSE_CSV_BUFFER_ROWS", "int"),
    ("IO_BATCH_SPAN", "IO_BATCH_SPAN", "str"),
    ("STAGE01_REPLACE_RANGE", "STAGE01_REPLACE_RANGE", "bool"),
    ("EXPORT_JSON_NAME", "EXPORT_JSON_NAME", "str"),
]


def parse_dotenv_assignments(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
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
        if key:
            out[key] = val.strip()
    return out


def _child_dump_quant_env() -> dict[str, object]:
    code = """
import json
from quant_system import quant_env as q
def p(path):
    return str(path).replace("\\\\", "/")
out = {
    "PROJECT_ROOT": p(q.project_root()),
    "coins_root": p(q.coins_root()),
    "historical_upload_dir": p(q.historical_upload_dir()),
    "symbol_artifacts_AIXBT": p(q.symbol_artifacts_dir("AIXBTUSDT")),
}
for env_key, attr, kind in json.loads(__import__("os").environ["_ENV_CHECKS_JSON"]):
    v = getattr(q, attr)
    if kind == "list":
        out[attr] = list(v)
    elif kind == "bool":
        out[attr] = bool(v)
    else:
        out[attr] = v
print(json.dumps(out))
"""
    expected = parse_dotenv_assignments(ROOT / ".env.example")
    env = os.environ.copy()
    env.update(expected)
    env["PYTHONPATH"] = str(ROOT / "src")
    env["_ENV_CHECKS_JSON"] = json.dumps(ENV_CHECKS)
    proc = subprocess.run(
        [sys.executable, "-c", code],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"quant_env child failed ({proc.returncode}):\n{proc.stderr}\n{proc.stdout}"
        )
    return json.loads(proc.stdout.strip().splitlines()[-1])


def verify_env_example() -> None:
    expected = parse_dotenv_assignments(ROOT / ".env.example")
    got = _child_dump_quant_env()
    errors: list[str] = []
    for env_key, attr, kind in ENV_CHECKS:
        if env_key not in expected:
            errors.append(f"missing key in .env.example: {env_key}")
            continue
        raw = expected[env_key]
        val = got.get(attr)
        if kind == "list":
            import re

            exp_list = [p for p in re.split(r"[\s,]+", raw) if p]
            if list(val) != exp_list:
                errors.append(f"{env_key}: expected {exp_list!r}, got {val!r}")
        elif kind == "bool":
            exp_b = raw.lower() in ("1", "true", "yes", "on")
            if bool(val) != exp_b:
                errors.append(f"{env_key}: expected {exp_b}, got {val}")
        elif kind == "int":
            if int(val) != int(raw):
                errors.append(f"{env_key}: expected {raw}, got {val}")
        elif kind == "float":
            if abs(float(val) - float(raw)) > 1e-12:
                errors.append(f"{env_key}: expected {raw}, got {val}")
        else:
            if str(val) != raw:
                errors.append(f"{env_key}: expected {raw!r}, got {val!r}")

    root = Path(got["PROJECT_ROOT"])
    hist = Path(got["historical_upload_dir"])
    expected_hist = (root / expected["HISTORICAL_UPLOAD_DIR"]).resolve()
    if hist != expected_hist:
        errors.append(f"HISTORICAL_UPLOAD_DIR path mismatch: {hist}")

    sym_dir = Path(got["symbol_artifacts_AIXBT"])
    expected_sym_dir = (root / expected["SYMBOL_ARTIFACTS_DIR"] / "AIXBTUSDT").resolve()
    if sym_dir != expected_sym_dir:
        errors.append(f"symbol_artifacts_dir mismatch: {sym_dir}")

    if errors:
        raise SystemExit("env.example parity FAILED:\n  " + "\n  ".join(errors))
    print("OK env.example == quant_env (", len(ENV_CHECKS), "keys + path helpers)")


def try_clickhouse_smoke(symbol: str, start: str | None, end: str | None) -> None:
    from quant_system.clickhouse_io import (
        build_clickhouse_client,
        count_rows,
        load_table_bounds,
        parse_cli_datetime,
    )

    client = build_clickhouse_client()
    db = _qe.CLICKHOUSE_DATABASE
    start_dt = parse_cli_datetime(start)
    end_dt = parse_cli_datetime(end)

    n_trades = count_rows(client, db, "trades", symbol, start=start_dt, end=end_dt)
    n_kv = count_rows(client, db, "features_kv", symbol, start=start_dt, end=end_dt)
    t_min, t_max = load_table_bounds(
        client, db, "trades", symbol, start=start_dt, end=end_dt
    )
    ts = (t_min, t_max)
    print(f"OK ClickHouse {symbol}: trades={n_trades} features_kv_rows={n_kv} ts_range={ts}")

    if n_trades == 0:
        print("  skip stage 01/02 (no trades in window)")
        return

    py = sys.executable
    if n_kv == 0:
        cmd01 = [
            py,
            str(ROOT / "01_add_features.py"),
            "--symbol",
            symbol,
            "--quiet",
        ]
        if start:
            cmd01 += ["--start", start]
        if end:
            cmd01 += ["--end", end]
        print("  run:", " ".join(cmd01))
        subprocess.run(cmd01, cwd=ROOT, check=True)
    else:
        print("  skip 01 smoke (features_kv already has rows)")

    print("  stage 05 metrics smoke removed from this repo; skip")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--symbol", default="AIXBTUSDT")
    ap.add_argument("--start", default=None, help="ISO UTC lower bound for CH smoke")
    ap.add_argument("--end", default=None, help="ISO UTC upper bound for CH smoke")
    ap.add_argument("--skip-ch", action="store_true")
    args = ap.parse_args()

    os.chdir(ROOT)
    verify_env_example()

    if args.skip_ch:
        print("skip ClickHouse (--skip-ch)")
        return

    try:
        try_clickhouse_smoke(args.symbol, args.start, args.end)
    except Exception as e:
        err = str(e).lower()
        if "connection" in err or "210" in err or "refused" in err:
            print("SKIP ClickHouse smoke: server not reachable at env CLICKHOUSE_HOST:PORT")
            print("  Start: cd containers/bybit-collector && docker compose up -d clickhouse")
            return
        raise


if __name__ == "__main__":
    main()
