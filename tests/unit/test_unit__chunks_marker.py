"""Unit tests for stage-05 chunk marker helpers."""

from __future__ import annotations

import unittest
from datetime import datetime, timedelta
from unittest.mock import patch

from tests.stage_import import load_stage

_cm = load_stage("chunks_marker_stage05", "05_chunks_marker.py")


class _FakeClient:
    def __init__(self, result: list[tuple[object, ...]]) -> None:
        self._result = result

    def execute(self, _sql: str, _params: dict[str, object]) -> list[tuple[object, ...]]:
        return self._result


class TestParseHelpers(unittest.TestCase):
    def test_parse_csv_list_trims_and_skips_empty(self) -> None:
        self.assertEqual(_cm._parse_csv_list(" spot_pt, , spot_trades ,,foo "), ["spot_pt", "spot_trades", "foo"])

    def test_dist_stats_empty(self) -> None:
        self.assertEqual(
            _cm._dist_stats([]),
            {"min": 0.0, "max": 0.0, "mean": 0.0, "median": 0.0, "std": 0.0},
        )

    def test_dist_stats_populated(self) -> None:
        stats = _cm._dist_stats([1, 2, 3, 4])
        self.assertEqual(stats["min"], 1.0)
        self.assertEqual(stats["max"], 4.0)
        self.assertEqual(stats["mean"], 2.5)
        self.assertEqual(stats["median"], 2.5)
        self.assertAlmostEqual(stats["std"], 1.2909944487, places=6)


class TestSplitStrategies(unittest.TestCase):
    def test_split_main_chunks_non_overlap_with_one_second_shift(self) -> None:
        bounds = _cm.TimeRange(
            start=datetime(2024, 1, 1, 0, 0, 0),
            end=datetime(2024, 1, 1, 0, 0, 20),
        )
        chunks = _cm._split_main_chunks(bounds, 2)
        self.assertEqual(len(chunks), 2)
        self.assertEqual(chunks[0].start, datetime(2024, 1, 1, 0, 0, 0))
        self.assertEqual(chunks[0].end, datetime(2024, 1, 1, 0, 0, 9))
        self.assertEqual(chunks[1].start, datetime(2024, 1, 1, 0, 0, 10))
        self.assertEqual(chunks[1].end, datetime(2024, 1, 1, 0, 0, 20))

    def test_split_main_chunks_rejects_zero_chunks(self) -> None:
        bounds = _cm.TimeRange(
            start=datetime(2024, 1, 1, 0, 0, 0),
            end=datetime(2024, 1, 1, 0, 0, 5),
        )
        with self.assertRaises(SystemExit):
            _cm._split_main_chunks(bounds, 0)

    def test_split_main_chunks_by_effective_time_strips_large_gap(self) -> None:
        bounds = _cm.TimeRange(
            start=datetime(2024, 1, 1, 0, 0, 0),
            end=datetime(2024, 1, 1, 0, 0, 40),
        )
        timeline = [
            datetime(2024, 1, 1, 0, 0, 0),
            datetime(2024, 1, 1, 0, 0, 5),
            datetime(2024, 1, 1, 0, 0, 10),
            datetime(2024, 1, 1, 0, 0, 35),  # stripped (gap 25s > 10s)
            datetime(2024, 1, 1, 0, 0, 40),
        ]
        chunks = _cm._split_main_chunks(
            bounds,
            3,
            timeline=timeline,
            strip_gap_threshold=timedelta(seconds=10),
        )
        self.assertEqual(len(chunks), 3)
        self.assertEqual(chunks[0].start, datetime(2024, 1, 1, 0, 0, 0))
        self.assertEqual(chunks[0].end, datetime(2024, 1, 1, 0, 0, 4))
        self.assertEqual(chunks[1].start, datetime(2024, 1, 1, 0, 0, 5))
        self.assertEqual(chunks[1].end, datetime(2024, 1, 1, 0, 0, 9))
        self.assertEqual(chunks[2].start, datetime(2024, 1, 1, 0, 0, 10))
        self.assertEqual(chunks[2].end, datetime(2024, 1, 1, 0, 0, 40))

    def test_split_train_val_test_time_based_tvt(self) -> None:
        chunk = _cm.ChunkBoundary(
            start=datetime(2024, 1, 1, 0, 0, 0),
            end=datetime(2024, 1, 1, 0, 0, 9),
        )
        local = _cm._split_train_val_test(chunk)
        self.assertEqual(local[0][0], "train")
        self.assertEqual(local[1][0], "validation")
        self.assertEqual(local[2][0], "test")
        self.assertEqual(local[0][1].start, datetime(2024, 1, 1, 0, 0, 0))
        self.assertEqual(local[0][1].end, datetime(2024, 1, 1, 0, 0, 2))
        self.assertEqual(local[1][1].start, datetime(2024, 1, 1, 0, 0, 3))
        self.assertEqual(local[1][1].end, datetime(2024, 1, 1, 0, 0, 5))
        self.assertEqual(local[2][1].start, datetime(2024, 1, 1, 0, 0, 6))
        self.assertEqual(local[2][1].end, datetime(2024, 1, 1, 0, 0, 9))

    def test_split_train_val_test_rejects_too_short_chunk(self) -> None:
        short = _cm.ChunkBoundary(
            start=datetime(2024, 1, 1, 0, 0, 0),
            end=datetime(2024, 1, 1, 0, 0, 2),
        )
        with self.assertRaises(SystemExit):
            _cm._split_train_val_test(short)


class TestDbRelatedHelpers(unittest.TestCase):
    def test_load_effective_bounds_reads_min_max(self) -> None:
        start = datetime(2024, 1, 1, 0, 0, 0)
        end = datetime(2024, 1, 1, 0, 0, 10)
        client = _FakeClient([(start + timedelta(seconds=1), end - timedelta(seconds=1))])
        got = _cm._load_effective_bounds(client, "crypto_db", "spot_pt", "BTCUSDT", start, end)
        self.assertEqual(got.start, datetime(2024, 1, 1, 0, 0, 1))
        self.assertEqual(got.end, datetime(2024, 1, 1, 0, 0, 9))

    def test_load_effective_bounds_raises_when_no_rows(self) -> None:
        start = datetime(2024, 1, 1, 0, 0, 0)
        end = datetime(2024, 1, 1, 0, 0, 10)
        client = _FakeClient([(None, None)])
        with self.assertRaises(SystemExit):
            _cm._load_effective_bounds(client, "crypto_db", "spot_pt", "BTCUSDT", start, end)

    def test_load_anchor_timeline_reads_distinct_ordered_timestamps(self) -> None:
        start = datetime(2024, 1, 1, 0, 0, 0)
        end = datetime(2024, 1, 1, 0, 1, 0)
        client = _FakeClient(
            [
                (datetime(2024, 1, 1, 0, 0, 1),),
                (datetime(2024, 1, 1, 0, 0, 2),),
                (datetime(2024, 1, 1, 0, 0, 5),),
            ]
        )
        got = _cm._load_anchor_timeline(client, "crypto_db", "spot_pt", "BTCUSDT", start, end)
        self.assertEqual(
            got,
            [
                datetime(2024, 1, 1, 0, 0, 1),
                datetime(2024, 1, 1, 0, 0, 2),
                datetime(2024, 1, 1, 0, 0, 5),
            ],
        )

    def test_require_tables_accepts_when_all_exist(self) -> None:
        with patch.object(_cm, "_table_exists", return_value=True):
            _cm._require_tables(client=object(), database="crypto_db", tables=["spot_pt", "spot_trades"])

    def test_require_tables_raises_for_missing_sources(self) -> None:
        existing = {"spot_pt": True, "spot_trades": False, "chunks_instances": True, "chunks_marking_results": True}

        def _exists(_client: object, _db: str, table: str) -> bool:
            return existing.get(table, False)

        with patch.object(_cm, "_table_exists", side_effect=_exists):
            with self.assertRaises(SystemExit):
                _cm._require_tables(client=object(), database="crypto_db", tables=["spot_pt", "spot_trades"])

    def test_rows_in_many_ranges_returns_vector(self) -> None:
        chunk_a = _cm.ChunkBoundary(
            start=datetime(2024, 1, 1, 0, 0, 0),
            end=datetime(2024, 1, 1, 0, 0, 1),
        )
        chunk_b = _cm.ChunkBoundary(
            start=datetime(2024, 1, 1, 0, 0, 2),
            end=datetime(2024, 1, 1, 0, 0, 3),
        )
        client = _FakeClient([(7, 11)])
        got = _cm._rows_in_many_ranges(client, "crypto_db", "spot_pt", "BTCUSDT", [chunk_a, chunk_b])
        self.assertEqual(got, [7, 11])

    def test_filter_non_empty_global_chunks_drops_zero_rows(self) -> None:
        chunks = [
            _cm.ChunkBoundary(
                start=datetime(2024, 1, 1, 0, 0, 0),
                end=datetime(2024, 1, 1, 0, 0, 9),
            ),
            _cm.ChunkBoundary(
                start=datetime(2024, 1, 1, 0, 0, 10),
                end=datetime(2024, 1, 1, 0, 0, 19),
            ),
        ]
        with patch.object(_cm, "_rows_in_many_ranges", return_value=[3, 0]):
            kept, dropped = _cm._filter_non_empty_global_chunks(
                client=object(),
                database="crypto_db",
                anchor_table="spot_pt",
                symbol="BTCUSDT",
                chunks=chunks,
            )
        self.assertEqual(len(kept), 1)
        self.assertEqual(dropped, 1)


if __name__ == "__main__":
    unittest.main()
