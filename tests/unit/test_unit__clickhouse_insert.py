"""Tests for fixed-size ClickHouse INSERT helpers."""

from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from quant_system.clickhouse_insert import (
    INSERT_MIN_ROWS,
    clickhouse_insert_rows,
    clickhouse_insert_trades_columnar,
    is_retriable_insert_error,
    upload_csv_buffer_flush_rows,
)
from quant_system.quant_env import DEFAULT_INSERT_BATCH_ROWS, insert_batch_rows


class TestInsertBatchRows(unittest.TestCase):
    def test_default_when_env_unset(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(insert_batch_rows(), DEFAULT_INSERT_BATCH_ROWS)

    def test_cli_override(self) -> None:
        self.assertEqual(insert_batch_rows(25_000), 25_000)

    def test_env_override(self) -> None:
        with patch.dict(
            os.environ,
            {"CLICKHOUSE_INSERT_BATCH_ROWS": "75000"},
            clear=True,
        ):
            self.assertEqual(insert_batch_rows(), 75_000)

    def test_floor_at_insert_min(self) -> None:
        self.assertEqual(insert_batch_rows(100), INSERT_MIN_ROWS)


class TestCsvBufferFlush(unittest.TestCase):
    def test_capped_by_insert_chunk(self) -> None:
        with patch("quant_system.quant_env.CLICKHOUSE_CSV_BUFFER_ROWS", 1_000_000):
            self.assertEqual(upload_csv_buffer_flush_rows(12_345), 12_345)

    def test_env_buffer_when_smaller(self) -> None:
        with patch("quant_system.quant_env.CLICKHOUSE_CSV_BUFFER_ROWS", 25_000):
            self.assertEqual(upload_csv_buffer_flush_rows(100_000), 25_000)


class TestInsertRetry(unittest.TestCase):
    def test_retriable_memory_code(self) -> None:
        from clickhouse_driver.errors import ServerException

        exc = ServerException("Memory limit exceeded")
        exc.code = 241
        self.assertTrue(is_retriable_insert_error(exc))

    def test_non_retriable(self) -> None:
        self.assertFalse(is_retriable_insert_error(ValueError("nope")))

    def test_columnar_halve_on_retriable(self) -> None:
        calls: list[int] = []

        class _FakeClient:
            def execute(
                self,
                _sql: str,
                columns: list,
                *,
                columnar: bool = False,
            ) -> None:
                assert columnar
                calls.append(len(columns[0]))
                if len(columns[0]) > 2000:
                    from clickhouse_driver.errors import ServerException

                    exc = ServerException("too large")
                    exc.code = 241
                    raise exc

        cols = [list(range(8000))]
        clickhouse_insert_trades_columnar(
            _FakeClient(),  # type: ignore[arg-type]
            "INSERT INTO t VALUES",
            cols,
            batch_rows=8000,
        )
        self.assertEqual(calls[0], 8000)
        self.assertGreater(len(calls), 1)
        self.assertTrue(all(c >= INSERT_MIN_ROWS for c in calls))

    def test_halve_on_retriable(self) -> None:
        calls: list[int] = []

        class _FakeClient:
            def execute(self, _sql: str, rows: list) -> None:
                calls.append(len(rows))
                if len(rows) > 2000:
                    from clickhouse_driver.errors import ServerException

                    exc = ServerException("too large")
                    exc.code = 241
                    raise exc

        big = [(1,)] * 8000
        clickhouse_insert_rows(
            _FakeClient(),  # type: ignore[arg-type]
            "INSERT INTO t VALUES",
            big,
            batch_rows=8000,
        )
        self.assertIn(8000, calls)
        self.assertTrue(any(n <= 4000 for n in calls))


if __name__ == "__main__":
    unittest.main()
