from __future__ import annotations

import unittest
from datetime import datetime, timedelta

from quant_system.secure_horizons import apply_embargo, apply_purge


class TestApplyEmbargo(unittest.TestCase):
    def test_apply_embargo_adds_extra_one_second_gap(self) -> None:
        chunk_start = datetime(2026, 1, 1, 0, 0, 0)
        chunk_end = datetime(2026, 1, 1, 0, 1, 0)

        out = apply_embargo(chunk_start, chunk_end, timedelta(seconds=5))

        self.assertEqual(out.start, datetime(2026, 1, 1, 0, 0, 6))
        self.assertEqual(out.end, chunk_end)
        self.assertEqual(out.operation, "embargo")
        self.assertFalse(out.is_empty)

    def test_apply_embargo_clamps_to_chunk_end(self) -> None:
        chunk_start = datetime(2026, 1, 1, 0, 0, 0)
        chunk_end = datetime(2026, 1, 1, 0, 0, 5)

        out = apply_embargo(chunk_start, chunk_end, timedelta(seconds=10))

        self.assertEqual(out.start, chunk_end)
        self.assertTrue(out.is_empty)


class TestApplyPurge(unittest.TestCase):
    def test_apply_purge_adds_extra_one_second_gap(self) -> None:
        chunk_start = datetime(2026, 1, 1, 0, 0, 0)
        chunk_end = datetime(2026, 1, 1, 0, 1, 0)

        out = apply_purge(chunk_start, chunk_end, timedelta(seconds=7))

        self.assertEqual(out.start, chunk_start)
        self.assertEqual(out.end, datetime(2026, 1, 1, 0, 0, 52))
        self.assertEqual(out.operation, "purge")
        self.assertFalse(out.is_empty)

    def test_apply_purge_clamps_to_chunk_start(self) -> None:
        chunk_start = datetime(2026, 1, 1, 0, 0, 0)
        chunk_end = datetime(2026, 1, 1, 0, 0, 5)

        out = apply_purge(chunk_start, chunk_end, timedelta(seconds=10))

        self.assertEqual(out.end, chunk_start)
        self.assertTrue(out.is_empty)

    def test_trimmed_chunk_to_json(self) -> None:
        chunk_start = datetime(2026, 1, 1, 0, 0, 0)
        chunk_end = datetime(2026, 1, 1, 0, 1, 0)
        out = apply_purge(chunk_start, chunk_end, timedelta(seconds=7))

        payload = out.to_json()

        self.assertEqual(payload["start"], "2026-01-01T00:00:00")
        self.assertEqual(payload["end"], "2026-01-01T00:00:52")
        self.assertEqual(payload["original_start"], "2026-01-01T00:00:00")
        self.assertEqual(payload["original_end"], "2026-01-01T00:01:00")
        self.assertEqual(payload["applied_horizon_seconds"], 7.0)
        self.assertEqual(payload["extra_gap_seconds"], 1.0)
        self.assertEqual(payload["operation"], "purge")
        self.assertEqual(payload["is_empty"], False)


if __name__ == "__main__":
    unittest.main()
