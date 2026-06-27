"""Tests for shared ClickHouse I/O helpers."""

from __future__ import annotations

import unittest
from datetime import datetime, timezone

from quant_system.clickhouse_io import parse_cli_datetime, symbol_time_where


class TestSymbolTimeWhere(unittest.TestCase):
    def test_exclusive_end_default(self) -> None:
        start = datetime(2026, 1, 1, tzinfo=timezone.utc)
        end = datetime(2026, 2, 1, tzinfo=timezone.utc)
        parts, params = symbol_time_where("BTCUSDT", start, end)
        self.assertIn("exchange_ts < %(end)s", parts)
        self.assertNotIn("<=", " ".join(parts))
        self.assertEqual(params["symbol"], "BTCUSDT")

    def test_inclusive_end_for_correlation(self) -> None:
        end = datetime(2026, 2, 1, tzinfo=timezone.utc)
        parts, _ = symbol_time_where("SYM", None, end, end_inclusive=True)
        self.assertIn("exchange_ts <= %(end)s", parts)


class TestParseCliDatetime(unittest.TestCase):
    def test_z_suffix(self) -> None:
        dt = parse_cli_datetime("2026-05-01T00:00:00Z")
        assert dt is not None
        self.assertEqual(dt.tzinfo, timezone.utc)

    def test_empty(self) -> None:
        self.assertIsNone(parse_cli_datetime(""))
        self.assertIsNone(parse_cli_datetime(None))


if __name__ == "__main__":
    unittest.main()
