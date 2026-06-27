"""SQLi safety checks for project ClickHouse client usage patterns."""

from __future__ import annotations

import ast
from pathlib import Path
import unittest
from datetime import datetime, timezone
import re
from typing import Any

from quant_system.clickhouse_insert import clickhouse_insert_rows
from quant_system.clickhouse_io import (
    count_rows,
    delete_features_kv_range,
    load_table_bounds,
    load_trades_from_clickhouse,
    symbol_time_where,
)


HARMLESS_SQLI_PAYLOADS = (
    "john' -- harmless comment style",
    "' OR '1'='1",
    "x\\'; SELECT 1; --",
    "\" OR \"1\"=\"1",
    "abc') UNION SELECT 1,2,3 --",
    "'/**/OR/**/'a'='a",
    "name';#",
)
_FORBIDDEN_KEYWORDS = re.compile(r"\b(drop|truncate|alter)\b", re.IGNORECASE)
_SQL_METHOD_NAMES = {
    "execute",
    "executemany",
    "execute_iter",
    "query",
    "raw_query",
    "raw",
}
_SQL_ARG_KEYWORDS = {"sql", "query", "statement", "command"}


class _RecordingClient:
    def __init__(self, responses: list[Any] | None = None) -> None:
        self.calls: list[dict[str, Any]] = []
        self.responses = list(responses or [])

    def execute(self, sql: str, payload: Any | None = None, **kwargs: Any) -> Any:
        self.calls.append(
            {
                "sql": sql,
                "payload": payload,
                "kwargs": kwargs,
            }
        )
        if self.responses:
            return self.responses.pop(0)
        if kwargs.get("columnar") and kwargs.get("with_column_types"):
            return ([], [("exchange_ts", "DateTime64(3)"), ("side", "String")])
        return []


class TestClickhouseClientSqliSafety(unittest.TestCase):
    @staticmethod
    def _project_root() -> Path:
        return Path(__file__).resolve().parents[2]

    @staticmethod
    def _is_dangerous_sql_arg(expr: ast.AST) -> bool:
        if isinstance(expr, ast.JoinedStr):
            return True
        if isinstance(expr, ast.BinOp) and isinstance(expr.op, (ast.Add, ast.Mod)):
            return True
        if isinstance(expr, ast.Call) and isinstance(expr.func, ast.Attribute):
            return expr.func.attr == "format"
        return False

    def test_project_has_no_dangerous_sql_call_templates(self) -> None:
        violations: list[str] = []
        root = self._project_root()
        ignored_dirs = {
            ".git",
            ".venv",
            "venv",
            "__pycache__",
            ".pytest_cache",
            "backups",
            "legacy",
        }

        for py_file in root.rglob("*.py"):
            if any(part in ignored_dirs for part in py_file.parts):
                continue

            source = py_file.read_text(encoding="utf-8")
            module = ast.parse(source, filename=str(py_file))
            source_lines = source.splitlines()

            for node in ast.walk(module):
                if not isinstance(node, ast.Call):
                    continue

                if isinstance(node.func, ast.Attribute):
                    method_name = node.func.attr
                elif isinstance(node.func, ast.Name):
                    method_name = node.func.id
                else:
                    continue

                if method_name not in _SQL_METHOD_NAMES:
                    continue

                sql_arg: ast.AST | None = None
                if node.args:
                    sql_arg = node.args[0]
                else:
                    for kw in node.keywords:
                        if kw.arg in _SQL_ARG_KEYWORDS:
                            sql_arg = kw.value
                            break

                if sql_arg is None or not self._is_dangerous_sql_arg(sql_arg):
                    continue

                rel_path = py_file.relative_to(root).as_posix()
                line = source_lines[node.lineno - 1].strip() if node.lineno <= len(source_lines) else ""
                violations.append(f"{rel_path}:{node.lineno}: {line}")

        self.assertFalse(
            violations,
            "Dangerous SQL templates detected (use parameterized queries):\n"
            + "\n".join(violations),
        )

    def test_payload_fixture_is_explicitly_non_destructive(self) -> None:
        self.assertTrue(HARMLESS_SQLI_PAYLOADS)
        for payload in HARMLESS_SQLI_PAYLOADS:
            self.assertIsNone(_FORBIDDEN_KEYWORDS.search(payload), payload)

    def test_symbol_time_where_keeps_payload_in_params(self) -> None:
        symbol = HARMLESS_SQLI_PAYLOADS[0]
        start = datetime(2026, 1, 1, tzinfo=timezone.utc)
        end = datetime(2026, 1, 2, tzinfo=timezone.utc)

        where, params = symbol_time_where(symbol, start, end)

        self.assertEqual(where[0], "symbol = %(symbol)s")
        self.assertIn("exchange_ts >= %(start)s", where)
        self.assertIn("exchange_ts < %(end)s", where)
        self.assertEqual(params["symbol"], symbol)
        self.assertNotIn(symbol, " ".join(where))

    def test_load_trades_uses_execute_params_not_string_concat_across_payloads(self) -> None:
        client = _RecordingClient()
        start = datetime(2026, 1, 1, tzinfo=timezone.utc)
        end = datetime(2026, 1, 2, tzinfo=timezone.utc)

        for payload_symbol in HARMLESS_SQLI_PAYLOADS:
            _ = load_trades_from_clickhouse(
                client,  # type: ignore[arg-type]
                "crypto_db",
                payload_symbol,
                start,
                end,
            )

        self.assertEqual(len(client.calls), len(HARMLESS_SQLI_PAYLOADS))
        for i, payload in enumerate(HARMLESS_SQLI_PAYLOADS):
            sql = client.calls[i]["sql"]
            params = client.calls[i]["payload"]
            self.assertIn("WHERE symbol = %(symbol)s", sql)
            self.assertEqual(params["symbol"], payload)
            self.assertNotIn(payload, sql)

    def test_count_rows_uses_parameterized_symbol(self) -> None:
        payload_symbol = HARMLESS_SQLI_PAYLOADS[4]
        client = _RecordingClient(responses=[[(42,)]])

        out = count_rows(
            client,  # type: ignore[arg-type]
            "crypto_db",
            "trades",
            payload_symbol,
        )

        self.assertEqual(out, 42)
        self.assertEqual(len(client.calls), 1)
        sql = client.calls[0]["sql"]
        params = client.calls[0]["payload"]
        self.assertIn("symbol = %(symbol)s", sql)
        self.assertEqual(params["symbol"], payload_symbol)
        self.assertNotIn(payload_symbol, sql)

    def test_load_table_bounds_uses_parameterized_symbol(self) -> None:
        payload_symbol = HARMLESS_SQLI_PAYLOADS[5]
        client = _RecordingClient(
            responses=[[(datetime(2026, 1, 1), datetime(2026, 1, 2))]]
        )

        lo, hi = load_table_bounds(
            client,  # type: ignore[arg-type]
            "crypto_db",
            "trades",
            payload_symbol,
        )

        self.assertIsNotNone(lo)
        self.assertIsNotNone(hi)
        sql = client.calls[0]["sql"]
        params = client.calls[0]["payload"]
        self.assertIn("WHERE symbol = %(symbol)s", sql)
        self.assertEqual(params["symbol"], payload_symbol)
        self.assertNotIn(payload_symbol, sql)

    def test_delete_features_kv_range_parameterizes_symbol_and_in_values(self) -> None:
        payload_symbol = HARMLESS_SQLI_PAYLOADS[6]
        feature_payloads = [
            "feat_1",
            "f' OR '1'='1",
            "metric/**/union/**/select",
        ]
        client = _RecordingClient()

        delete_features_kv_range(
            client,  # type: ignore[arg-type]
            "crypto_db",
            payload_symbol,
            None,
            None,
            feature_names=feature_payloads,
        )

        self.assertEqual(len(client.calls), 1)
        sql = client.calls[0]["sql"]
        params = client.calls[0]["payload"]
        self.assertIn("symbol = %(symbol)s", sql)
        self.assertIn("feature_name IN %(feature_names)s", sql)
        self.assertEqual(params["symbol"], payload_symbol)
        self.assertEqual(params["feature_names"], tuple(feature_payloads))
        self.assertNotIn(payload_symbol, sql)
        for feature in feature_payloads:
            self.assertNotIn(feature, sql)

    def test_clickhouse_insert_rows_passes_payload_as_data_rows(self) -> None:
        client = _RecordingClient()
        rows = [
            (1, HARMLESS_SQLI_PAYLOADS[2], '{"city":"Brest"}'),
            (2, HARMLESS_SQLI_PAYLOADS[3], '{"city":"Minsk"}'),
            (3, HARMLESS_SQLI_PAYLOADS[4], '{"city":"Vilnius"}'),
            (4, HARMLESS_SQLI_PAYLOADS[5], '{"city":"Warsaw"}'),
        ]

        clickhouse_insert_rows(
            client,  # type: ignore[arg-type]
            "INSERT INTO tmp_test (id, name, metadata) VALUES",
            rows,
            batch_rows=1000,
        )

        self.assertEqual(len(client.calls), 1)
        sql = client.calls[0]["sql"]
        inserted_rows = client.calls[0]["payload"]
        self.assertEqual(sql, "INSERT INTO tmp_test (id, name, metadata) VALUES")
        self.assertEqual(inserted_rows, rows)
        for payload in HARMLESS_SQLI_PAYLOADS:
            self.assertNotIn(payload, sql)

    def test_clickhouse_insert_rows_batches_without_touching_payload_content(self) -> None:
        client = _RecordingClient()
        rows = [
            (i, HARMLESS_SQLI_PAYLOADS[i % len(HARMLESS_SQLI_PAYLOADS)], "{}")
            for i in range(1, 7)
        ]

        clickhouse_insert_rows(
            client,  # type: ignore[arg-type]
            "INSERT INTO tmp_test (id, name, metadata) VALUES",
            rows,
            batch_rows=2,
        )

        self.assertEqual(len(client.calls), 3)
        reconstructed: list[tuple[Any, ...]] = []
        for call in client.calls:
            reconstructed.extend(call["payload"])
            self.assertEqual(
                call["sql"], "INSERT INTO tmp_test (id, name, metadata) VALUES"
            )
        self.assertEqual(reconstructed, rows)


if __name__ == "__main__":
    unittest.main()
