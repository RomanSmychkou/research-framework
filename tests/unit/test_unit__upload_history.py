"""Tests for stage-00 upload_history helpers."""

from __future__ import annotations

import gzip
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock

from quant_system import quant_env
from tests.stage_import import load_stage

_uh = load_stage("upload_history", "00_upload_history.py")


class TestIsoRangeUtc(unittest.TestCase):
    def test_none_when_missing_bounds(self) -> None:
        self.assertEqual(_uh.iso_range_utc(None, None), (None, None))

    def test_formats_bounds(self) -> None:
        start, end = _uh.iso_range_utc(1_000, 2_000)
        self.assertTrue(start and start.endswith("Z"))
        self.assertTrue(end and end.endswith("Z"))


class TestRowFromCsvLine(unittest.TestCase):
    def test_parses_line(self) -> None:
        row = _uh.row_from_csv_line("SYM", b"42,5000,1.5,2.0,sell,1", None)
        self.assertIsNotNone(row)
        assert row is not None
        *data, ts_ms = row
        self.assertEqual(ts_ms, 5000)
        self.assertEqual(data[1], "SYM")
        self.assertEqual(data[2], "42")
        self.assertEqual(data[5], "sell")

    def test_skips_before_start_ms(self) -> None:
        self.assertIsNone(_uh.row_from_csv_line("SYM", b"1,999,1,1,buy", 1000))


class TestProcessGzTimestampRange(unittest.TestCase):
    def test_tracks_min_max_ms(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "SYM-2025-10.csv.gz"
            with gzip.open(path, "wt", encoding="utf-8", newline="") as f:
                f.write("id,timestamp,price,volume,side\n")
                f.write("1,1000,1.0,1.0,buy\n")
                f.write("2,5000,1.0,1.0,sell\n")
                f.write("3,3000,1.0,1.0,buy\n")
            client = MagicMock()
            n, min_ms, max_ms = _uh.process_gz_file(
                client,
                path,
                "SYM",
                "db",
                "trades",
                quant_env.insert_batch_rows(10_000),
                on_progress=None,
                log_each_flush=False,
            )
            self.assertEqual(n, 3)
            self.assertEqual(min_ms, 1000)
            self.assertEqual(max_ms, 5000)
            self.assertTrue(client.execute.called)


if __name__ == "__main__":
    unittest.main()
