from __future__ import annotations

import json
import unittest
from datetime import datetime, timezone

from quant_system.feature_contracts import (
    BatchBackfillContract,
    FeatureContractSnapshot,
    FeatureParentContract,
)


class TestFeatureContracts(unittest.TestCase):
    def test_feature_contract_serializes_dependencies_and_hash(self) -> None:
        contract = FeatureContractSnapshot(
            feature_name="cvd_30m",
            feature_version=1,
            defined_start_ts=datetime(2024, 1, 1, tzinfo=timezone.utc),
            defined_end_ts=datetime(2024, 2, 1, tzinfo=timezone.utc),
            feature_type="time_based",
            is_target=False,
            is_causal=True,
            own_lookback_ms=30 * 60 * 1000,
            parent_contracts=(FeatureParentContract("trades_ms", 2),),
            backfill=BatchBackfillContract(
                source_table="trades_ms",
                sql_template="SELECT ...",
                fill_policy="zero",
                fill_value=0.0,
                chunk_span="1d",
                resource_settings={"max_threads": 4},
            ),
        )
        payload = contract.to_json_dict()

        self.assertEqual(payload["feature_name"], "cvd_30m")
        self.assertEqual(payload["feature_version"], 1)
        self.assertEqual(payload["feature_type"], "time_based")
        self.assertFalse(payload["is_target"])
        self.assertTrue(payload["is_causal"])
        self.assertEqual(payload["own_lookback_ms"], 30 * 60 * 1000)
        self.assertEqual(payload["own_lookforward_ms"], 0)
        self.assertEqual(
            payload["parent_contracts"],
            [{"feature_name": "trades_ms", "feature_version": 2}],
        )
        self.assertEqual(payload["backfill"]["source_table"], "trades_ms")
        self.assertEqual(payload["backfill"]["fill_policy"], "zero")
        self.assertEqual(payload["backfill"]["fill_value"], 0.0)
        self.assertEqual(payload["backfill"]["resource_settings"], {"max_threads": 4})
        self.assertEqual(len(contract.contract_hash()), 64)

    def test_target_contract_uses_forward_horizon_for_purge(self) -> None:
        contract = FeatureContractSnapshot(
            feature_name="pnl_after_3m",
            feature_version=1,
            defined_start_ts=datetime(2024, 1, 1),
            defined_end_ts=datetime(2024, 2, 1),
            feature_type="target",
            is_target=True,
            is_causal=False,
            own_lookforward_ms=3 * 60 * 1000,
        )

        self.assertEqual(contract.own_lookback_ms, 0)
        self.assertEqual(contract.own_lookforward_ms, 3 * 60 * 1000)

    def test_non_target_cannot_have_forward_horizon(self) -> None:
        with self.assertRaises(ValueError):
            FeatureContractSnapshot(
                feature_name="bad_feature",
                feature_version=1,
                defined_start_ts=datetime(2024, 1, 1),
                defined_end_ts=datetime(2024, 2, 1),
                feature_type="time_based",
                is_target=False,
                is_causal=True,
                own_lookforward_ms=60_000,
            )

    def test_target_must_use_target_type(self) -> None:
        with self.assertRaises(ValueError):
            FeatureContractSnapshot(
                feature_name="pnl_after_3m",
                feature_version=1,
                defined_start_ts=datetime(2024, 1, 1),
                defined_end_ts=datetime(2024, 2, 1),
                feature_type="time_point",
                is_target=True,
                is_causal=False,
                own_lookforward_ms=3 * 60 * 1000,
            )

    def test_backfill_source_table_is_part_of_hash(self) -> None:
        base = dict(
            feature_name="cvd_30m",
            feature_version=1,
            defined_start_ts=datetime(2024, 1, 1),
            defined_end_ts=datetime(2024, 2, 1),
            feature_type="time_based",
            is_target=False,
            is_causal=True,
            own_lookback_ms=30 * 60 * 1000,
        )
        event_contract = FeatureContractSnapshot(
            **base,
            backfill=BatchBackfillContract(
                source_table="trades",
                sql_template="SELECT ...",
            ),
        )
        bucket_contract = FeatureContractSnapshot(
            **base,
            backfill=BatchBackfillContract(
                source_table="trades_1s",
                sql_template="SELECT ...",
            ),
        )

        self.assertNotEqual(event_contract.contract_hash(), bucket_contract.contract_hash())

    def test_backfill_sql_template_is_required(self) -> None:
        with self.assertRaises(ValueError):
            BatchBackfillContract(
                source_table="trades",
                sql_template="",
            )

    def test_backfill_fill_value_can_store_future_sentinels(self) -> None:
        contract = BatchBackfillContract(
            source_table="trades_agg_1s",
            sql_template="SELECT ...",
            fill_policy="causal_ffill",
            fill_value=-1,
        )

        self.assertEqual(json.loads(contract.to_json())["fill_value"], -1)


if __name__ == "__main__":
    unittest.main()
