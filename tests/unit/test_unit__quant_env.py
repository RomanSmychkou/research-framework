"""Tests for quant_env helpers and .env.example parity."""

from __future__ import annotations

import os
import unittest
from pathlib import Path
from unittest.mock import patch

from quant_system.quant_env import (
    CLICKHOUSE_DATABASE,
    DEFAULT_INSERT_BATCH_ROWS,
    IO_BATCH_SPAN,
    env_bool,
    env_insert_row_override,
    env_int,
    env_optional_int,
    env_str,
    env_str_list,
    insert_batch_rows,
    project_root,
    resolve_project_path,
    symbol_artifacts_dir,
)


class TestInsertRowOverride(unittest.TestCase):
    def test_insert_batch_rows_env_override(self) -> None:
        with patch.dict(
            os.environ,
            {"CLICKHOUSE_INSERT_BATCH_ROWS": "75000"},
            clear=True,
        ):
            self.assertEqual(env_insert_row_override(), 75_000)


class TestEnvHelpers(unittest.TestCase):
    def test_env_str_uses_os_environ(self) -> None:
        with patch.dict(os.environ, {"TEST_QUANT_STR": "from_env"}):
            self.assertEqual(env_str("TEST_QUANT_STR", "fallback"), "from_env")
            self.assertEqual(env_str("TEST_QUANT_MISSING", "fallback"), "fallback")

    def test_env_int(self) -> None:
        with patch.dict(os.environ, {"TEST_QUANT_INT": "42"}):
            self.assertEqual(env_int("TEST_QUANT_INT", 1), 42)

    def test_env_optional_int_empty(self) -> None:
        with patch.dict(os.environ, {"TEST_QUANT_OPT": ""}, clear=False):
            self.assertIsNone(env_optional_int("TEST_QUANT_OPT"))

    def test_env_bool(self) -> None:
        with patch.dict(os.environ, {"TEST_QUANT_BOOL": "yes"}):
            self.assertTrue(env_bool("TEST_QUANT_BOOL", False))

    def test_env_str_list(self) -> None:
        with patch.dict(os.environ, {"TEST_QUANT_LIST": "a, b  c"}):
            self.assertEqual(env_str_list("TEST_QUANT_LIST", []), ["a", "b", "c"])


class TestDefaultsAlignWithModules(unittest.TestCase):
    def test_io_span_matches_time_chunk_processor(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(
                env_str("IO_BATCH_SPAN", "30d"),
                "30d",
            )

    def test_example_file_documents_core_keys(self) -> None:
        root = Path(__file__).resolve().parents[2]
        example = (root / ".env.example").read_text(encoding="utf-8")
        for key in (
            "CLICKHOUSE_HOST",
            "COINS_DIR",
            "HISTORICAL_UPLOAD_DIR",
            "SYMBOL_ARTIFACTS_DIR",
            "UPLOAD_CONFIG_NAME",
            "CLICKHOUSE_CSV_BUFFER_ROWS",
            "CLICKHOUSE_INSERT_BATCH_ROWS",
            "IO_BATCH_SPAN",
            "EXPORT_JSON_NAME",
        ):
            self.assertIn(f"{key}=", example, msg=f"missing {key} in .env.example")


class TestImportedConstants(unittest.TestCase):
    def test_database_default(self) -> None:
        self.assertEqual(CLICKHOUSE_DATABASE, env_str("CLICKHOUSE_DATABASE", "crypto_db"))

    def test_span_and_insert_defaults(self) -> None:
        self.assertTrue(IO_BATCH_SPAN)
        self.assertEqual(insert_batch_rows(), DEFAULT_INSERT_BATCH_ROWS)


class TestProjectPaths(unittest.TestCase):
    def test_resolve_relative_under_root(self) -> None:
        root = project_root()
        p = resolve_project_path("coins/_historical_data")
        self.assertEqual(p, (root / "coins" / "_historical_data").resolve())

    def test_symbol_artifacts_dir(self) -> None:
        p = symbol_artifacts_dir("TESTUSDT")
        self.assertEqual(p.parent, resolve_project_path("coins"))
        self.assertEqual(p.name, "TESTUSDT")


if __name__ == "__main__":
    unittest.main()
