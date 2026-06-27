from __future__ import annotations

import re
import unittest
from datetime import datetime

from pipeline.features.declarations import (
    ALL_FEATURE_DECLARATIONS,
    FeatureDeclaration,
    FIRST_LAST_MOMENTUM_DECLARATION,
    PRICE_DIRECTION_4S_TARGET_DECLARATION,
    PRICE_DELTA_10S_TARGET_DECLARATION,
    SELL_RISE_HIT_055_200MS_TARGET_DECLARATION,
    SPOT_TRADES_ROLLING_DECLARATIONS,
)
from pipeline.features.gate import (
    build_feature_bundle,
    build_named_target_feature,
    build_price_delta_target_feature,
    build_spot_trades_rolling_bundle,
)
from pipeline.features.feature_contract import FeatureInstance, FeatureTemplate
from pipeline.features.graph import topological_order, validate_declarations
from pipeline.features.sql_renderer import (
    render_feature_invalid_onehot_sql,
    render_feature_invalid_replace_sql,
    render_feature_sql,
    render_feature_sql_safe,
)
from pipeline.features.templates import FIRST_LAST_DELTA_TEMPLATE


class TestPipelineFeaturesSotGate(unittest.TestCase):
    _BANNED_SQL_PATTERNS: tuple[tuple[str, str], ...] = (
        (r"\bUNBOUNDED\s+FOLLOWING\b", "unbounded_forward_horizon"),
        (r"\bUNBOUNDED\s+PRECEDING\b", "unbounded_lookback_horizon"),
        (r"\bROWS\s+BETWEEN\b", "row_frame_semantics_can_cross_time_gaps"),
    )

    _BANNED_CAUSAL_PATTERNS: tuple[tuple[str, str], ...] = (
        (r"\bCURRENT\s+ROW\b", "current_row_in_causal_window"),
        (r"\b\d+\s+FOLLOWING\b", "future_lookahead_in_causal_window"),
        (r"\b\{[^}]*\}\s+FOLLOWING\b", "dynamic_future_lookahead_in_causal_window"),
        (r"\bUNBOUNDED\s+FOLLOWING\b", "unbounded_future_lookahead_in_causal_window"),
    )

    @staticmethod
    def _policy() -> dict[str, object]:
        return {"special_value": None, "onehot_encoding": False}

    @staticmethod
    def _non_causal_template() -> FeatureTemplate:
        return FeatureTemplate(
            name="non_causal_template_for_tests",
            description="non-causal template for summary lookforward tests",
            sql_query_shadow="SELECT 1",
            required_context_keys=("own_lookback_ms",),
            is_target=False,
            is_causal=False,
            base_context={"own_lookback_ms": 0},
        )

    @staticmethod
    def _target_template() -> FeatureTemplate:
        return FeatureTemplate(
            name="target_template_for_tests",
            description="target template for summary lookback tests",
            sql_query_shadow="SELECT 1",
            required_context_keys=("own_lookback_ms",),
            is_target=True,
            is_causal=False,
            base_context={"own_lookback_ms": 0},
        )

    def test_validate_declarations_rejects_duplicate_names(self) -> None:
        declarations = (
            FeatureDeclaration(
                name="dup",
                template=FIRST_LAST_DELTA_TEMPLATE,
                ml_fill_policy=self._policy(),
            ),
            FeatureDeclaration(
                name="dup",
                template=FIRST_LAST_DELTA_TEMPLATE,
                ml_fill_policy=self._policy(),
            ),
        )
        with self.assertRaises(ValueError):
            validate_declarations(declarations)

    def test_topological_order_rejects_cycle(self) -> None:
        declarations = (
            FeatureDeclaration(
                name="a",
                template=FIRST_LAST_DELTA_TEMPLATE,
                ml_fill_policy=self._policy(),
                parent_names=("b",),
            ),
            FeatureDeclaration(
                name="b",
                template=FIRST_LAST_DELTA_TEMPLATE,
                ml_fill_policy=self._policy(),
                parent_names=("a",),
            ),
        )
        with self.assertRaises(ValueError):
            topological_order(declarations)

    def test_topological_order_keeps_dependency_before_consumer(self) -> None:
        declarations = (
            FeatureDeclaration(
                name="z",
                template=FIRST_LAST_DELTA_TEMPLATE,
                ml_fill_policy=self._policy(),
            ),
            FeatureDeclaration(
                name="a",
                template=FIRST_LAST_DELTA_TEMPLATE,
                ml_fill_policy=self._policy(),
                parent_names=("z",),
            ),
        )
        ordered = topological_order(declarations)
        self.assertEqual([decl.name for decl in ordered], ["z", "a"])

    def test_build_bundle_resolves_dependency_objects(self) -> None:
        bundle = build_feature_bundle(
            include={"price_momentum_for_balance", "volume_balance"},
            include_targets=False,
        )
        self.assertEqual(
            [feature.name for feature in bundle.features_in_compute_order],
            ["price_momentum_for_balance", "volume_balance"],
        )
        volume = bundle.by_name["volume_balance"]
        self.assertEqual([parent.name for parent in volume.parents], ["price_momentum_for_balance"])
        self.assertEqual(volume.parent_names, ("price_momentum_for_balance",))

    def test_build_bundle_rejects_missing_dependency_in_selection(self) -> None:
        with self.assertRaises(ValueError):
            build_feature_bundle(
                include={"volume_balance"},
                include_targets=False,
            )

    def test_context_precedence_is_declaration_then_shared_then_per_feature(self) -> None:
        bundle = build_feature_bundle(
            include={FIRST_LAST_MOMENTUM_DECLARATION.name},
            include_targets=False,
            shared_context={
                "own_lookback_ms": 11_000,
                "where_clause": "symbol = 'SOLUSDT'",
            },
            per_feature_context={
                FIRST_LAST_MOMENTUM_DECLARATION.name: {
                    "own_lookback_ms": 22_000,
                    "where_clause": "symbol = 'DOGEUSDT'",
                }
            },
        )
        momentum = bundle.by_name[FIRST_LAST_MOMENTUM_DECLARATION.name]
        self.assertEqual(momentum.context["own_lookback_ms"], 300_000)
        self.assertEqual(momentum.context["where_clause"], "symbol = 'DOGEUSDT'")

    def test_materialized_feature_exposes_immutable_own_look_horizons(self) -> None:
        bundle = build_feature_bundle(
            include={FIRST_LAST_MOMENTUM_DECLARATION.name},
            include_targets=False,
        )
        momentum = bundle.by_name[FIRST_LAST_MOMENTUM_DECLARATION.name]
        self.assertEqual(momentum.own_lookback_ms, FIRST_LAST_MOMENTUM_DECLARATION.own_lookback_ms)
        self.assertEqual(momentum.own_lookforward_ms, FIRST_LAST_MOMENTUM_DECLARATION.own_lookforward_ms)

    def test_context_look_horizons_are_filled_from_declaration_attributes(self) -> None:
        feature = build_named_target_feature(
            target_name=SELL_RISE_HIT_055_200MS_TARGET_DECLARATION.name,
            database="crypto_db",
            source_table="spot_trades",
            where_clause="symbol = 'BTCUSDT'",
            horizon_seconds=4,
        )
        self.assertEqual(feature.context["own_lookback_ms"], 0)
        self.assertEqual(feature.context["own_lookforward_ms"], 200)
        self.assertNotIn("event_horizon_ms", feature.context)

    def test_build_spot_trades_rolling_bundle_returns_only_rolling_features(self) -> None:
        bundle = build_spot_trades_rolling_bundle(
            database="crypto_db",
            source_table="spot_trades",
            where_clause="symbol = 'BTCUSDT'",
            window_ms=123_000,
        )
        expected_names = {declaration.name for declaration in SPOT_TRADES_ROLLING_DECLARATIONS}
        actual_names = {feature.name for feature in bundle.features_in_compute_order}
        self.assertEqual(actual_names, expected_names)
        self.assertEqual(bundle.targets, ())
        self.assertEqual(len(bundle.features_in_compute_order), len(SPOT_TRADES_ROLLING_DECLARATIONS))

    def test_build_price_delta_target_feature_returns_target_feature(self) -> None:
        feature = build_price_delta_target_feature(
            database="crypto_db",
            source_table="spot_trades",
            where_clause="symbol = 'BTCUSDT'",
            horizon_seconds=77,
        )
        self.assertEqual(feature.name, PRICE_DELTA_10S_TARGET_DECLARATION.name)
        self.assertTrue(feature.is_target)
        self.assertEqual(feature.context["own_lookforward_ms"], 4_000)
        self.assertEqual(feature.context["table_name"], "crypto_db.spot_trades")

    def test_price_delta_target_sql_applies_deadband_zeroing(self) -> None:
        feature = build_price_delta_target_feature(
            database="crypto_db",
            source_table="spot_trades",
            where_clause="symbol = 'BTCUSDT'",
            horizon_seconds=4,
        )
        sql = render_feature_sql(feature)
        self.assertIn("if(", sql)
        self.assertIn("abs(", sql)
        self.assertIn("0.0", sql)
        self.assertIn("* 0.0045", sql)

    def test_build_named_direction_target_feature_returns_target_feature(self) -> None:
        feature = build_named_target_feature(
            target_name=PRICE_DIRECTION_4S_TARGET_DECLARATION.name,
            database="crypto_db",
            source_table="spot_trades",
            where_clause="symbol = 'BTCUSDT'",
            horizon_seconds=4,
        )
        self.assertEqual(feature.name, PRICE_DIRECTION_4S_TARGET_DECLARATION.name)
        self.assertTrue(feature.is_target)
        sql = render_feature_sql(feature)
        self.assertIn("multiIf(", sql)
        self.assertIn("1.0", sql)
        self.assertIn("-1.0", sql)
        self.assertIn("* 0.0055", sql)

    def test_build_named_sell_rise_hit_target_feature_returns_binary_target(self) -> None:
        feature = build_named_target_feature(
            target_name=SELL_RISE_HIT_055_200MS_TARGET_DECLARATION.name,
            database="crypto_db",
            source_table="spot_trades",
            where_clause="symbol = 'BTCUSDT'",
            horizon_seconds=4,
        )
        self.assertEqual(feature.name, SELL_RISE_HIT_055_200MS_TARGET_DECLARATION.name)
        self.assertTrue(feature.is_target)
        sql = render_feature_sql(feature)
        self.assertIn("maxIf(", sql)
        self.assertIn("upper(side) = 'SELL'", sql)
        self.assertIn("RANGE BETWEEN 1 FOLLOWING AND 200 FOLLOWING", sql)
        self.assertIn("* (1.0 + 0.0055)", sql)
        self.assertIn("1.0", sql)
        self.assertIn("0.0", sql)

    def test_sql_renderer_resolves_nested_metric_placeholders(self) -> None:
        bundle = build_spot_trades_rolling_bundle(
            database="crypto_db",
            source_table="spot_trades",
            where_clause="symbol = 'BTCUSDT'",
            window_ms=300_000,
        )
        feature = bundle.by_name["trade_count_buy"]
        sql = render_feature_sql(feature)
        self.assertIn("countIf(upper(side) = 'BUY') OVER w", sql)
        self.assertIn("FROM crypto_db.spot_trades", sql)
        self.assertNotIn("{side_column}", sql)

    def test_sql_renderer_requires_guard_bounds(self) -> None:
        feature = build_price_delta_target_feature(
            database="crypto_db",
            source_table="spot_trades",
            where_clause="symbol = 'BTCUSDT'",
            horizon_seconds=4,
        )
        with self.assertRaisesRegex(ValueError, "guard_start and guard_end are required"):
            render_feature_sql_safe(
                feature,
                guard_start=None,  # type: ignore[arg-type]
                guard_end=datetime(2026, 1, 2),
            )
        with self.assertRaisesRegex(ValueError, "guard_start and guard_end are required"):
            render_feature_sql_safe(
                feature,
                guard_start=datetime(2026, 1, 1),
                guard_end=None,  # type: ignore[arg-type]
            )

    def test_sql_renderer_rejects_non_increasing_guard_bounds(self) -> None:
        feature = build_price_delta_target_feature(
            database="crypto_db",
            source_table="spot_trades",
            where_clause="symbol = 'BTCUSDT'",
            horizon_seconds=4,
        )
        same = datetime(2026, 1, 1)
        with self.assertRaisesRegex(ValueError, "guard_start must be < guard_end"):
            render_feature_sql_safe(feature, guard_start=same, guard_end=same)

    def test_sql_renderer_includes_outer_guard_predicates(self) -> None:
        feature = build_price_delta_target_feature(
            database="crypto_db",
            source_table="spot_trades",
            where_clause="symbol = 'BTCUSDT'",
            horizon_seconds=4,
        )
        sql = render_feature_sql_safe(
            feature,
            guard_start=datetime(2026, 1, 1),
            guard_end=datetime(2026, 1, 2),
        )
        self.assertIn("WHERE bucket_time >= %(guard_start)s", sql)
        self.assertIn("AND bucket_time < %(guard_end)s", sql)

    def test_sql_renderer_builds_invalid_value_onehot_wrapper(self) -> None:
        bundle = build_spot_trades_rolling_bundle(
            database="crypto_db",
            source_table="spot_trades",
            where_clause="symbol = 'BTCUSDT'",
            window_ms=300_000,
        )
        feature = bundle.by_name["trade_count_buy"]
        sql = render_feature_invalid_onehot_sql(feature)
        self.assertIn("toFloat64OrNull(toString(feature_value))", sql)
        self.assertIn("isNull(parsed_value)", sql)
        self.assertIn("NOT isFinite(ifNull(parsed_value, 0.0))", sql)
        self.assertIn("AS base_feature", sql)

    def test_sql_renderer_builds_invalid_value_replace_wrapper(self) -> None:
        bundle = build_spot_trades_rolling_bundle(
            database="crypto_db",
            source_table="spot_trades",
            where_clause="symbol = 'BTCUSDT'",
            window_ms=300_000,
        )
        feature = bundle.by_name["trade_count_buy"]
        sql = render_feature_invalid_replace_sql(feature, special_value=-1.0)
        self.assertIn("toFloat64OrNull(toString(feature_value))", sql)
        self.assertIn("isNull(parsed_value)", sql)
        self.assertIn("NOT isFinite(ifNull(parsed_value, 0.0))", sql)
        self.assertIn("AS base_feature", sql)
        self.assertIn("-1.0", sql)

    def test_sql_renderer_invalid_replace_supports_string_as_literal_value(self) -> None:
        bundle = build_spot_trades_rolling_bundle(
            database="crypto_db",
            source_table="spot_trades",
            where_clause="symbol = 'BTCUSDT'",
            window_ms=300_000,
        )
        feature = bundle.by_name["trade_count_buy"]
        sql = render_feature_invalid_replace_sql(feature, special_value="1.25")
        self.assertIn("toFloat64OrZero('1.25')", sql)

    def test_sql_renderer_invalid_replace_treats_string_payload_as_literal(self) -> None:
        bundle = build_spot_trades_rolling_bundle(
            database="crypto_db",
            source_table="spot_trades",
            where_clause="symbol = 'BTCUSDT'",
            window_ms=300_000,
        )
        feature = bundle.by_name["trade_count_buy"]
        sql = render_feature_invalid_replace_sql(
            feature,
            special_value="1.0') OR 1=1 --",
        )
        self.assertIn("toFloat64OrZero('1.0\\') OR 1=1 --')", sql)

    def test_summary_look_horizons_follow_max_parent_branch_and_cache(self) -> None:
        def _instance(
            *,
            name: str,
            own_lookback_ms: int,
            own_lookforward_ms: int,
            parents: tuple[FeatureInstance, ...] = (),
        ) -> FeatureInstance:
            return FeatureInstance(
                template=FIRST_LAST_DELTA_TEMPLATE,
                name=name,
                ml_fill_policy=self._policy(),
                context={"own_lookback_ms": own_lookback_ms},
                parent_runtime_instances=parents,
                own_lookback_ms=own_lookback_ms,
                own_lookforward_ms=own_lookforward_ms,
            )

        branch_a = _instance(name="branch_a", own_lookback_ms=100, own_lookforward_ms=0)
        branch_b = _instance(name="branch_b", own_lookback_ms=60, own_lookforward_ms=0)
        branch_a_leaf = _instance(
            name="branch_a_leaf",
            own_lookback_ms=40,
            own_lookforward_ms=0,
            parents=(branch_a,),
        )
        branch_b_leaf = _instance(
            name="branch_b_leaf",
            own_lookback_ms=120,
            own_lookforward_ms=0,
            parents=(branch_b,),
        )
        sink = _instance(
            name="sink",
            own_lookback_ms=30,
            own_lookforward_ms=0,
            parents=(branch_a_leaf, branch_b_leaf),
        )

        self.assertFalse(hasattr(sink, "_summary_lookback_ms_cache"))
        self.assertTrue(hasattr(sink, "_summary_lookforward_ms_cache"))

        self.assertEqual(sink.summary_lookback_ms, 210)
        self.assertEqual(sink.summary_lookforward_ms, 0)
        self.assertTrue(hasattr(sink, "_summary_lookback_ms_cache"))
        self.assertTrue(hasattr(sink, "_summary_lookforward_ms_cache"))

    def test_summary_look_horizons_for_leaf_match_own_horizons(self) -> None:
        leaf = FeatureInstance(
            template=FIRST_LAST_DELTA_TEMPLATE,
            name="leaf_only",
            ml_fill_policy=self._policy(),
            context={"own_lookback_ms": 123},
            own_lookback_ms=123,
            own_lookforward_ms=0,
        )
        self.assertEqual(leaf.summary_lookback_ms, 123)
        self.assertEqual(leaf.summary_lookforward_ms, 0)

    def test_feature_instance_rejects_positive_summary_lookforward_for_causal(self) -> None:
        with self.assertRaises(ValueError):
            FeatureInstance(
                template=FIRST_LAST_DELTA_TEMPLATE,
                name="causal_with_forward",
                ml_fill_policy=self._policy(),
                context={"own_lookback_ms": 1},
                own_lookback_ms=0,
                own_lookforward_ms=1,
            )

    def test_feature_instance_rejects_positive_summary_lookback_for_target(self) -> None:
        with self.assertRaises(ValueError):
            FeatureInstance(
                template=self._target_template(),
                name="target_with_lookback",
                ml_fill_policy=self._policy(),
                context={"own_lookback_ms": 1},
                own_lookback_ms=1,
                own_lookforward_ms=0,
            )

    def test_feature_instance_rejects_non_feature_parent_for_summary_tree(self) -> None:
        with self.assertRaises(TypeError):
            FeatureInstance(
                template=FIRST_LAST_DELTA_TEMPLATE,
                name="bad_parent_type",
                ml_fill_policy=self._policy(),
                context={"own_lookback_ms": 1},
                parent_runtime_instances=("not_a_feature_instance",),  # type: ignore[arg-type]
                own_lookback_ms=1,
                own_lookforward_ms=0,
            )

    def test_feature_instance_rejects_duplicate_parent_names_for_summary_tree(self) -> None:
        parent = FeatureInstance(
            template=FIRST_LAST_DELTA_TEMPLATE,
            name="dup_parent",
            ml_fill_policy=self._policy(),
            context={"own_lookback_ms": 10},
            own_lookback_ms=10,
            own_lookforward_ms=0,
        )
        with self.assertRaises(ValueError):
            FeatureInstance(
                template=FIRST_LAST_DELTA_TEMPLATE,
                name="child_with_duplicate_parents",
                ml_fill_policy=self._policy(),
                context={"own_lookback_ms": 5},
                parent_runtime_instances=(parent, parent),
                own_lookback_ms=5,
                own_lookforward_ms=0,
            )

    def test_feature_instance_rejects_self_parent_for_summary_tree(self) -> None:
        with self.assertRaises(ValueError):
            feature = FeatureInstance(
                template=FIRST_LAST_DELTA_TEMPLATE,
                name="self_parent",
                ml_fill_policy=self._policy(),
                context={"own_lookback_ms": 3},
                own_lookback_ms=3,
                own_lookforward_ms=0,
            )
            feature.with_parents(feature)

    def test_summary_lookback_deep_tree_uses_max_branch_sum(self) -> None:
        def _instance(
            *,
            name: str,
            own_lookback_ms: int,
            own_lookforward_ms: int = 0,
            parents: tuple[FeatureInstance, ...] = (),
        ) -> FeatureInstance:
            return FeatureInstance(
                template=FIRST_LAST_DELTA_TEMPLATE,
                name=name,
                ml_fill_policy=self._policy(),
                context={"own_lookback_ms": own_lookback_ms},
                parent_runtime_instances=parents,
                own_lookback_ms=own_lookback_ms,
                own_lookforward_ms=own_lookforward_ms,
            )

        root_a = _instance(name="root_a", own_lookback_ms=20)
        root_b = _instance(name="root_b", own_lookback_ms=15)
        root_c = _instance(name="root_c", own_lookback_ms=12)
        a1 = _instance(name="a1", own_lookback_ms=40, parents=(root_a,))
        a2 = _instance(name="a2", own_lookback_ms=35, parents=(a1,))
        b1 = _instance(name="b1", own_lookback_ms=60, parents=(root_b,))
        b2 = _instance(name="b2", own_lookback_ms=10, parents=(b1,))
        c1 = _instance(name="c1", own_lookback_ms=25, parents=(root_c,))
        c2 = _instance(name="c2", own_lookback_ms=30, parents=(c1,))
        c3 = _instance(name="c3", own_lookback_ms=45, parents=(c2,))
        join_1 = _instance(name="join_1", own_lookback_ms=18, parents=(a2, b2))
        join_2 = _instance(name="join_2", own_lookback_ms=22, parents=(join_1, c3))
        sink = _instance(name="deep_sink_lb", own_lookback_ms=7, parents=(join_2,))

        # Max path is: root_a(20) -> a1(40) -> a2(35) -> join_1(18) -> join_2(22) -> sink(7)
        self.assertEqual(sink.summary_lookback_ms, 142)

    def test_summary_lookforward_deep_tree_uses_max_branch_sum(self) -> None:
        non_causal_template = self._non_causal_template()

        def _instance(
            *,
            name: str,
            own_lookforward_ms: int,
            own_lookback_ms: int = 0,
            parents: tuple[FeatureInstance, ...] = (),
        ) -> FeatureInstance:
            return FeatureInstance(
                template=non_causal_template,
                name=name,
                ml_fill_policy=self._policy(),
                context={"own_lookback_ms": own_lookback_ms},
                parent_runtime_instances=parents,
                own_lookback_ms=own_lookback_ms,
                own_lookforward_ms=own_lookforward_ms,
            )

        root_a = _instance(name="f_root_a", own_lookforward_ms=5)
        root_b = _instance(name="f_root_b", own_lookforward_ms=9)
        root_c = _instance(name="f_root_c", own_lookforward_ms=4)
        a1 = _instance(name="f_a1", own_lookforward_ms=14, parents=(root_a,))
        a2 = _instance(name="f_a2", own_lookforward_ms=6, parents=(a1,))
        b1 = _instance(name="f_b1", own_lookforward_ms=3, parents=(root_b,))
        b2 = _instance(name="f_b2", own_lookforward_ms=20, parents=(b1,))
        c1 = _instance(name="f_c1", own_lookforward_ms=11, parents=(root_c,))
        c2 = _instance(name="f_c2", own_lookforward_ms=12, parents=(c1,))
        c3 = _instance(name="f_c3", own_lookforward_ms=2, parents=(c2,))
        join_1 = _instance(name="f_join_1", own_lookforward_ms=8, parents=(a2, b2))
        join_2 = _instance(name="f_join_2", own_lookforward_ms=10, parents=(join_1, c3))
        sink = _instance(name="deep_sink_lf", own_lookforward_ms=1, parents=(join_2,))

        # Max path is: root_b(9) -> b1(3) -> b2(20) -> join_1(8) -> join_2(10) -> sink(1)
        self.assertEqual(sink.summary_lookforward_ms, 51)

    def test_all_declared_sql_templates_forbid_unbounded_following(self) -> None:
        banned = re.compile(r"\bUNBOUNDED\s+FOLLOWING\b", re.IGNORECASE)
        violations = [
            declaration.name
            for declaration in ALL_FEATURE_DECLARATIONS
            if banned.search(declaration.template.sql_query_shadow)
        ]
        self.assertFalse(
            violations,
            "UNBOUNDED FOLLOWING is forbidden in feature SQL templates. "
            f"Violating declarations: {', '.join(sorted(violations))}",
        )

    def test_default_registry_sql_templates_forbid_unbounded_following(self) -> None:
        banned = re.compile(r"\bUNBOUNDED\s+FOLLOWING\b", re.IGNORECASE)
        bundle = build_feature_bundle()
        violations = [
            feature.name
            for feature in bundle.features_in_compute_order
            if banned.search(feature.sql_template)
        ]
        self.assertFalse(
            violations,
            "UNBOUNDED FOLLOWING is forbidden in materialized SQL templates. "
            f"Violating features: {', '.join(sorted(violations))}",
        )

    def test_all_declared_sql_templates_forbid_leaky_expressions(self) -> None:
        violations: list[str] = []
        for declaration in ALL_FEATURE_DECLARATIONS:
            sql = declaration.template.sql_query_shadow
            for pattern, tag in self._BANNED_SQL_PATTERNS:
                if re.search(pattern, sql, flags=re.IGNORECASE):
                    violations.append(f"{declaration.name}:{tag}")
        self.assertFalse(
            violations,
            "Potentially leaky SQL constructs are forbidden in declarations. "
            f"Violations: {', '.join(sorted(violations))}",
        )

    def test_causal_declared_sql_templates_forbid_future_leakage(self) -> None:
        violations: list[str] = []
        for declaration in ALL_FEATURE_DECLARATIONS:
            if declaration.template.is_target or not declaration.template.is_causal:
                continue
            sql = declaration.template.sql_query_shadow
            for pattern, tag in self._BANNED_CAUSAL_PATTERNS:
                if re.search(pattern, sql, flags=re.IGNORECASE):
                    violations.append(f"{declaration.name}:{tag}")
        self.assertFalse(
            violations,
            "Causal declarations must not contain future or T+0 leakage patterns. "
            f"Violations: {', '.join(sorted(violations))}",
        )


if __name__ == "__main__":
    unittest.main()

