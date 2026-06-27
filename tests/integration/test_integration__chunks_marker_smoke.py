from __future__ import annotations

import unittest
from argparse import Namespace
from datetime import datetime, timedelta
from unittest.mock import patch

from tests.stage_import import load_stage

_cm = load_stage("chunks_marker_smoke", "05_chunks_marker.py")


class _FakeClient:
    def __init__(self) -> None:
        self.instances_rows: list[tuple[object, ...]] | None = None
        self.results_rows: list[tuple[object, ...]] | None = None

    def execute(self, sql: str, params: object = None) -> list[tuple[object, ...]]:
        text = " ".join(sql.split())
        if "FROM system.tables" in text:
            return [(1,)]
        if "minIf(exchange_ts" in text and "maxIf(exchange_ts" in text:
            return [(datetime(2024, 1, 1, 0, 0, 1), datetime(2024, 1, 1, 0, 0, 59))]
        if "SELECT toInt64(sum(real_delta_us)) AS total_real_us" in text:
            return [(58_000_000, 58_000_000)]
        if "arrayJoin(range(1, %(n_chunks)s)) AS chunk_idx" in text:
            if not isinstance(params, dict):
                raise AssertionError("Expected dict params for boundaries query")
            n_chunks = int(params["n_chunks"])
            start = params["start"]
            end = params["end"]
            span_sec = int((end - start).total_seconds())
            if n_chunks == 2:
                return [(1, start + timedelta(seconds=span_sec // 2))]
            if n_chunks == 3:
                return [
                    (1, start + timedelta(seconds=span_sec // 3)),
                    (2, start + timedelta(seconds=(2 * span_sec) // 3)),
                ]
            raise AssertionError(f"Unexpected n_chunks for boundaries query: {n_chunks}")
        if "sum(if(exchange_ts >=" in text and "FROM crypto_db.spot_pt" in text:
            if not isinstance(params, dict):
                raise AssertionError("Expected dict params for aggregate count query")
            ranges_n = len([k for k in params.keys() if str(k).startswith("start_")])
            return [tuple(10 for _ in range(ranges_n))]
        if "INSERT INTO crypto_db.chunks_instances" in text:
            assert isinstance(params, list)
            self.instances_rows = params
            return []
        if "INSERT INTO crypto_db.chunks_marking_results" in text:
            assert isinstance(params, list)
            self.results_rows = params
            return []
        raise AssertionError(f"Unexpected SQL in smoke: {text}")


class TestChunksMarkerSmoke(unittest.TestCase):
    def test_main_smoke_inserts_instances_and_session_result(self) -> None:
        fake = _FakeClient()
        args = Namespace(
            symbol="BTCUSDT",
            start_date="2024-01-01T00:00:00Z",
            end_date="2024-01-01T00:01:00Z",
            tables_to_mark="spot_pt",
            submit_ranges=True,
            n_chunks_united=2,
            host="127.0.0.1",
            port=9000,
            user="default",
            password="",
            database="crypto_db",
            debug=False,
            strip_internal_gaps="1d",
            manual_mode=False,
            manual_chunk_description="",
            show_manuals_history=0,
        )

        with (
            patch.object(_cm, "parse_args", return_value=args),
            patch.object(_cm, "build_clickhouse_client", return_value=fake),
            patch.object(_cm, "_git_hash", return_value="g123"),
        ):
            _cm.main()

        assert fake.instances_rows is not None
        assert fake.results_rows is not None

        self.assertEqual(len(fake.instances_rows), 6)  # 2 global chunks * 3 local TVT
        self.assertEqual(len(fake.results_rows), 1)  # one session row

        for row in fake.instances_rows:
            self.assertEqual(row[0], "BTCUSDT")
            self.assertEqual(row[1], "spot_pt")
            self.assertIn(row[4], ("train", "validation", "test"))
            self.assertEqual(row[7], "g123")

        session = fake.results_rows[0]
        self.assertEqual(session[0], "BTCUSDT")
        self.assertEqual(session[3], "g123")
        self.assertEqual(len(session[2]), 6)
        self.assertEqual(session[4], _cm._hash_text("|".join(session[2])))

    def test_main_filters_boundary_zero_chunk_with_internal_gaps(self) -> None:
        class _BoundaryGapClient:
            def __init__(self) -> None:
                self.instances_rows: list[tuple[object, ...]] | None = None
                self.results_rows: list[tuple[object, ...]] | None = None
                self._sum_if_calls = 0

            def execute(self, sql: str, params: object = None) -> list[tuple[object, ...]]:
                text = " ".join(sql.split())
                if "FROM system.tables" in text:
                    return [(1,)]
                if "minIf(exchange_ts" in text and "maxIf(exchange_ts" in text:
                    return [(datetime(2024, 1, 1, 0, 0, 1), datetime(2024, 1, 1, 0, 0, 59))]
                if "SELECT toInt64(sum(real_delta_us)) AS total_real_us" in text:
                    return [(58_000_000, 58_000_000)]
                if "arrayJoin(range(1, %(n_chunks)s)) AS chunk_idx" in text:
                    if not isinstance(params, dict):
                        raise AssertionError("Expected dict params for boundaries query")
                    n_chunks = int(params["n_chunks"])
                    start = params["start"]
                    end = params["end"]
                    span_sec = int((end - start).total_seconds())
                    if n_chunks == 3 and start == datetime(2024, 1, 1, 0, 0, 1) and end == datetime(2024, 1, 1, 0, 0, 59):
                        return [
                            (1, datetime(2024, 1, 1, 0, 0, 20)),
                            (2, datetime(2024, 1, 1, 0, 0, 21)),
                        ]
                    if n_chunks == 3:
                        return [
                            (1, start + timedelta(seconds=span_sec // 3)),
                            (2, start + timedelta(seconds=(2 * span_sec) // 3)),
                        ]
                    raise AssertionError(f"Unexpected n_chunks for boundaries query: {n_chunks}")
                if "sum(if(exchange_ts >=" in text and "FROM crypto_db.spot_pt" in text:
                    self._sum_if_calls += 1
                    if self._sum_if_calls == 1:
                        # Global chunks: [1..19], [20..20], [21..59] -> middle chunk empty.
                        return [(10, 0, 10)]
                    if not isinstance(params, dict):
                        raise AssertionError("Expected dict params for aggregate count query")
                    ranges_n = len([k for k in params.keys() if str(k).startswith("start_")])
                    return [tuple(5 for _ in range(ranges_n))]
                if "INSERT INTO crypto_db.chunks_instances" in text:
                    assert isinstance(params, list)
                    self.instances_rows = params
                    return []
                if "INSERT INTO crypto_db.chunks_marking_results" in text:
                    assert isinstance(params, list)
                    self.results_rows = params
                    return []
                raise AssertionError(f"Unexpected SQL in smoke: {text}")

        fake = _BoundaryGapClient()
        args = Namespace(
            symbol="BTCUSDT",
            start_date="2024-01-01T00:00:00Z",
            end_date="2024-01-01T00:01:00Z",
            tables_to_mark="spot_pt",
            submit_ranges=True,
            n_chunks_united=3,
            host="127.0.0.1",
            port=9000,
            user="default",
            password="",
            database="crypto_db",
            debug=False,
            strip_internal_gaps="1d",
            manual_mode=False,
            manual_chunk_description="",
            show_manuals_history=0,
        )

        with (
            patch.object(_cm, "parse_args", return_value=args),
            patch.object(_cm, "build_clickhouse_client", return_value=fake),
            patch.object(_cm, "_git_hash", return_value="g123"),
        ):
            _cm.main()

        assert fake.instances_rows is not None
        assert fake.results_rows is not None
        # One global zero chunk filtered out => 2 global chunks * 3 local TVT.
        self.assertEqual(len(fake.instances_rows), 6)
        self.assertEqual(len(fake.results_rows[0][2]), 6)

    def test_manual_mode_inserts_single_chunk_per_table(self) -> None:
        class _ManualClient:
            def __init__(self) -> None:
                self.manual_rows: list[tuple[object, ...]] | None = None

            def execute(self, sql: str, params: object = None) -> list[tuple[object, ...]]:
                text = " ".join(sql.split())
                if "FROM system.tables" in text:
                    return [(1,)]
                if "minIf(exchange_ts" in text and "maxIf(exchange_ts" in text:
                    return [(datetime(2024, 1, 1, 0, 0, 5), datetime(2024, 1, 1, 0, 0, 55))]
                if "INSERT INTO crypto_db.manual_marked_chunks" in text:
                    assert isinstance(params, list)
                    self.manual_rows = params
                    return []
                if "FROM crypto_db.manual_marked_chunks" in text and "ORDER BY created_at DESC" in text:
                    return []
                raise AssertionError(f"Unexpected SQL in manual smoke: {text}")

        fake = _ManualClient()
        args = Namespace(
            symbol="BTCUSDT",
            start_date="2024-01-01T00:00:00Z",
            end_date="2024-01-01T00:01:00Z",
            tables_to_mark="spot_pt,spot_trades",
            submit_ranges=False,
            n_chunks_united=30,
            host="127.0.0.1",
            port=9000,
            user="default",
            password="",
            database="crypto_db",
            debug=False,
            strip_internal_gaps="1d",
            manual_mode=True,
            manual_chunk_description="ops manual chunk",
            show_manuals_history=0,
        )

        with (
            patch.object(_cm, "parse_args", return_value=args),
            patch.object(_cm, "build_clickhouse_client", return_value=fake),
            patch.object(_cm, "_git_hash", return_value="g123"),
        ):
            _cm.main()

        assert fake.manual_rows is not None
        self.assertEqual(len(fake.manual_rows), 2)
        tables = [row[1] for row in fake.manual_rows]
        self.assertEqual(sorted(tables), ["spot_pt", "spot_trades"])
        for row in fake.manual_rows:
            self.assertEqual(row[0], "BTCUSDT")
            self.assertEqual(row[2], "ops manual chunk")
            expected_hash = _cm._hash_text(
                "|".join(
                    (
                        "BTCUSDT",
                        "ops manual chunk",
                        str(row[1]),
                        _cm._iso_utc(row[3]),
                        _cm._iso_utc(row[4]),
                    )
                )
            )
            self.assertEqual(row[6], expected_hash)


if __name__ == "__main__":
    unittest.main()
