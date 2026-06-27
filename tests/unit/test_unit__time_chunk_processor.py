"""Tests for time_chunk_processor helpers."""

from __future__ import annotations

import unittest
from datetime import datetime, timedelta

from quant_system.time_chunk_processor import (
    RowCountTimeBatchSpec,
    format_timedelta,
    parse_duration,
)


class TestParseDuration(unittest.TestCase):
    def test_parse_units(self) -> None:
        self.assertEqual(parse_duration("7d"), timedelta(days=7))
        self.assertEqual(parse_duration("12h"), timedelta(hours=12))

class TestRowCountTimeBatchSpec(unittest.TestCase):
    def test_label_is_human_readable(self) -> None:
        spec = RowCountTimeBatchSpec(
            index=0,
            start=datetime(2024, 1, 1, 0, 0, 0),
            end=datetime(2024, 1, 1, 0, 0, 1),
            row_count=10,
            is_last=False,
        )
        self.assertEqual(spec.label, "row-batch 1")


class TestFormatTimedelta(unittest.TestCase):
    def test_days(self) -> None:
        self.assertEqual(format_timedelta(timedelta(days=30)), "30d")


if __name__ == "__main__":
    unittest.main()
