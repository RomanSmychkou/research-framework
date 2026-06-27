"""Stage 01 CLI time bound clamping (timezone-safe)."""

from __future__ import annotations

import unittest
from datetime import datetime, timedelta, timezone

from quant_system.time_chunk_processor import ensure_aware_utc, exchange_ts_for_ch


class TestExchangeTsForCh(unittest.TestCase):
    def test_aware_cli_to_naive_utc(self) -> None:
        aware = datetime(2026, 1, 4, 0, 0, 0, tzinfo=timezone.utc)
        naive = exchange_ts_for_ch(aware)
        self.assertIsNone(naive.tzinfo)
        self.assertEqual(naive, datetime(2026, 1, 4, 0, 0, 0))

    def test_clamp_end_exclusive(self) -> None:
        t_min = datetime(2026, 1, 1, 0, 0, 0)
        t_max = datetime(2026, 1, 3, 12, 0, 0)
        end = exchange_ts_for_ch(datetime(2026, 1, 4, 0, 0, 0, tzinfo=timezone.utc))
        assert end is not None
        capped = min(t_max, end - timedelta(milliseconds=1))
        self.assertGreater(capped.year, 2020)
        self.assertLessEqual(capped, end)
        self.assertGreaterEqual(capped, t_min)

    def test_min_max_not_inverted_after_aware_start(self) -> None:
        t_min = exchange_ts_for_ch(datetime(2026, 1, 1, tzinfo=timezone.utc))
        t_max = exchange_ts_for_ch(datetime(2026, 1, 3, 12, 0, 0))
        start = exchange_ts_for_ch(datetime(2026, 1, 1, tzinfo=timezone.utc))
        end = exchange_ts_for_ch(datetime(2026, 1, 4, tzinfo=timezone.utc))
        assert t_min and t_max and start and end
        t_min = max(t_min, start)
        t_max = min(t_max, end - timedelta(milliseconds=1))
        self.assertLessEqual(t_min, t_max)
        self.assertNotEqual(ensure_aware_utc(t_max).year, 1970)


if __name__ == "__main__":
    unittest.main()
