from __future__ import annotations

from dataclasses import dataclass, field
from datetime import timedelta
from typing import Any, Mapping

from .feature_contract import FeatureTemplate, MlFillPolicy
from .templates import (
    FIRST_LAST_DELTA_TEMPLATE,
    SIGNED_SIDE_AGGREGATE_TEMPLATE,
    SPOT_TRADES_ROLLING_METRIC_TEMPLATE,
)


@dataclass(frozen=True)
class FeatureDeclaration:
    name: str
    template: FeatureTemplate
    ml_fill_policy: MlFillPolicy  # TODO map_encoding
    description: str | None = None
    context_overrides: Mapping[str, Any] = field(default_factory=dict)
    parent_names: tuple[str, ...] = ()
    own_lookback_ms: int = 0
    own_lookforward_ms: int = 0
    enabled: bool = True


def _ml_fill_policy(
    *,
    special_value: Any | None = None,
    onehot_encoding: bool = False,
) -> MlFillPolicy:
    return {
        "special_value": special_value,
        "onehot_encoding": onehot_encoding,
    }


def _rolling_declaration(
    *,
    name: str,
    description: str,
    metric_expression: str,
) -> FeatureDeclaration:
    return FeatureDeclaration(
        name=name,
        description=description,
        template=SPOT_TRADES_ROLLING_METRIC_TEMPLATE,
        ml_fill_policy=_ml_fill_policy(),
        context_overrides={"metric_expression": metric_expression},
        own_lookback_ms=int(SPOT_TRADES_ROLLING_METRIC_TEMPLATE.base_context["own_lookback_ms"]),
        own_lookforward_ms=0,
    )


FIRST_LAST_MOMENTUM_DECLARATION = FeatureDeclaration(
    name="price_momentum",
    template=FIRST_LAST_DELTA_TEMPLATE,
    ml_fill_policy=_ml_fill_policy(),
    description="Разница между последним и первым ценовым тиком в пределах окна.",
    context_overrides={
        "where_clause": "symbol = 'BTCUSDT'",
    },
    own_lookback_ms=300_000,
    own_lookforward_ms=0,
)

FIRST_LAST_MOMENTUM_FOR_BALANCE_DECLARATION = FeatureDeclaration(
    name="price_momentum_for_balance",
    template=FIRST_LAST_DELTA_TEMPLATE,
    ml_fill_policy=_ml_fill_policy(),
    description="Momentum по spot_trades в том же окне, что и для balance-фичи.",
    context_overrides={
        "where_clause": "symbol = 'ETHUSDT'",
        "time_column": "exchange_ts",
        "price_column": "price",
        "table_name": "spot_trades",
    },
    own_lookback_ms=300_000,
    own_lookforward_ms=0,
)

SIDE_VOLUME_BALANCE_DECLARATION = FeatureDeclaration(
    name="volume_balance",
    template=SIGNED_SIDE_AGGREGATE_TEMPLATE,
    ml_fill_policy=_ml_fill_policy(),
    description="Сравнение объёмов покупок и продаж в скользящем окне.",
    context_overrides={
        "where_clause": "symbol = 'ETHUSDT'",
        "time_column": "exchange_ts",
        "volume_column": "quantity",
        "table_name": "spot_trades",
    },
    parent_names=("price_momentum_for_balance",),
    own_lookback_ms=300_000,
    own_lookforward_ms=0,
)

SPOT_TRADES_ROLLING_DECLARATIONS = (
    _rolling_declaration(
        name="trade_count_buy",
        description="Количество buy-трейдов в окне.",
        metric_expression="countIf(upper({side_column}) = 'BUY') OVER w",
    ),
    _rolling_declaration(
        name="trade_count_sell",
        description="Количество sell-трейдов в окне.",
        metric_expression="countIf(upper({side_column}) = 'SELL') OVER w",
    ),
    _rolling_declaration(
        name="cvd_buy",
        description="Суммарный buy-объем (CVD leg) в окне.",
        metric_expression="sumIf({quantity_column}, upper({side_column}) = 'BUY') OVER w",
    ),
    _rolling_declaration(
        name="cvd_sell",
        description="Суммарный sell-объем (CVD leg) в окне.",
        metric_expression="sumIf({quantity_column}, upper({side_column}) = 'SELL') OVER w",
    ),
    _rolling_declaration(
        name="mean_price_buy",
        description="Средняя цена buy-трейдов в окне.",
        metric_expression="avgIf({price_column}, upper({side_column}) = 'BUY') OVER w",
    ),
    _rolling_declaration(
        name="mean_price_sell",
        description="Средняя цена sell-трейдов в окне.",
        metric_expression="avgIf({price_column}, upper({side_column}) = 'SELL') OVER w",
    ),
    _rolling_declaration(
        name="mean_volume_buy",
        description="Средний объем buy-трейда в окне.",
        metric_expression="avgIf({quantity_column}, upper({side_column}) = 'BUY') OVER w",
    ),
    _rolling_declaration(
        name="mean_volume_sell",
        description="Средний объем sell-трейда в окне.",
        metric_expression="avgIf({quantity_column}, upper({side_column}) = 'SELL') OVER w",
    ),
    _rolling_declaration(
        name="trade_spread_signed",
        description="Signed spread по трейдам: mean_price_buy - mean_price_sell.",
        metric_expression=(
            "(avgIf({price_column}, upper({side_column}) = 'BUY') OVER w) - "
            "(avgIf({price_column}, upper({side_column}) = 'SELL') OVER w)"
        ),
    ),
)

PRICE_DELTA_TARGET_TEMPLATE_BINNED_1S = FeatureTemplate(
    name="price_delta_forward_binned_1s_template",
    description="Forward price delta target on 1-second binned timeline.",
    sql_query_shadow="""
WITH agg_1s AS (
    SELECT
        toStartOfSecond(exchange_ts) AS exchange_ts,
        symbol,
        argMax(price, exchange_ts) AS price_last
    FROM {table_name}
    WHERE symbol = %(symbol)s
      AND exchange_ts >= subtractMilliseconds(%(start)s, {own_lookforward_ms})
      AND exchange_ts < addMilliseconds(%(end)s, {own_lookforward_ms})
    GROUP BY symbol, exchange_ts
)
SELECT
    t.exchange_ts AS bucket_time,
    t.symbol AS symbol,
    if(
        abs(toFloat64(t_fwd.price_last) - toFloat64(t.price_last))
            < abs(toFloat64(t.price_last)) * {delta_deadband_ratio},
        0.0,
        toFloat64(t_fwd.price_last) - toFloat64(t.price_last)
    ) AS feature_value
FROM agg_1s AS t
INNER JOIN agg_1s AS t_fwd
    ON t_fwd.symbol = t.symbol
   AND t_fwd.exchange_ts = addMilliseconds(t.exchange_ts, {own_lookforward_ms})
WHERE {where_clause}
""",
    required_context_keys=("table_name", "where_clause", "own_lookforward_ms", "delta_deadband_ratio"),
    is_target=True,
    is_causal=False,
    base_context={
        "table_name": "spot_trades",
        "where_clause": "TRUE",
        "own_lookforward_ms": 4_000,
        "delta_deadband_ratio": 0.0045,
    },
)

PRICE_DELTA_TARGET_TEMPLATE_TRADE_TIME = FeatureTemplate(
    name="price_delta_forward_trade_time_template",
    description="Forward price delta target on trade-time timeline.",
    sql_query_shadow="""
SELECT
    t.exchange_ts AS bucket_time,
    t.symbol,
    if(
        abs(
            toFloat64(first_value(price) OVER w_fwd) - toFloat64(t.price)
        ) < abs(toFloat64(t.price)) * {delta_deadband_ratio},
        0.0,
        toFloat64(first_value(price) OVER w_fwd) - toFloat64(t.price)
    ) AS feature_value
FROM {table_name} AS t
WHERE {where_clause}
WINDOW w_fwd AS (
    PARTITION BY t.symbol
    ORDER BY toUnixTimestamp64Milli(t.exchange_ts)
    RANGE BETWEEN {own_lookforward_ms} FOLLOWING AND {own_lookforward_ms} FOLLOWING
)
""",
    required_context_keys=("table_name", "where_clause", "own_lookforward_ms", "delta_deadband_ratio"),
    is_target=True,
    is_causal=False,
    base_context={
        "table_name": "spot_trades",
        "where_clause": "TRUE",
        "own_lookforward_ms": 4_000,
        "delta_deadband_ratio": 0.0045,
    },
)

PRICE_DIRECTION_TARGET_TEMPLATE_TRADE_TIME = FeatureTemplate(
    name="price_direction_forward_trade_time_template",
    description="Forward directional target on trade-time timeline (-1/0/1 labels).",
    sql_query_shadow="""
SELECT
    t.exchange_ts AS bucket_time,
    t.symbol,
    multiIf(
        (
            toFloat64(first_value(price) OVER w_fwd) - toFloat64(t.price)
        ) > abs(toFloat64(t.price)) * {direction_threshold_ratio},
        1.0,
        (
            toFloat64(first_value(price) OVER w_fwd) - toFloat64(t.price)
        ) < -abs(toFloat64(t.price)) * {direction_threshold_ratio},
        -1.0,
        0.0
    ) AS feature_value
FROM {table_name} AS t
WHERE {where_clause}
WINDOW w_fwd AS (
    PARTITION BY t.symbol
    ORDER BY toUnixTimestamp64Milli(t.exchange_ts)
    RANGE BETWEEN {own_lookforward_ms} FOLLOWING AND {own_lookforward_ms} FOLLOWING
)
""",
    required_context_keys=("table_name", "where_clause", "own_lookforward_ms", "direction_threshold_ratio"),
    is_target=True,
    is_causal=False,
    base_context={
        "table_name": "spot_trades",
        "where_clause": "TRUE",
        "own_lookforward_ms": 4_000,
        "direction_threshold_ratio": 0.0055,
    },
)

SELL_RISE_HIT_055_TARGET_TEMPLATE_TRADE_TIME = FeatureTemplate(
    name="sell_rise_hit_055_forward_trade_time_template",
    description=(
        "Binary target: 1 if within future horizon there is at least one SELL trade "
        "with price above current by threshold."
    ),
    sql_query_shadow="""
SELECT
    t.exchange_ts AS bucket_time,
    t.symbol,
    if(
        ifNull(
            maxIf(
                toFloat64(price),
                upper(side) = 'SELL'
            ) OVER w_fwd_sell,
            toFloat64(-1)
        ) > toFloat64(t.price) * (1.0 + {rise_threshold_ratio}),
        1.0,
        0.0
    ) AS feature_value
FROM {table_name} AS t
WHERE {where_clause}
WINDOW w_fwd_sell AS (
    PARTITION BY t.symbol
    ORDER BY toUnixTimestamp64Milli(t.exchange_ts)
    RANGE BETWEEN 1 FOLLOWING AND {own_lookforward_ms} FOLLOWING
)
""",
    required_context_keys=("table_name", "where_clause", "own_lookforward_ms", "rise_threshold_ratio"),
    is_target=True,
    is_causal=False,
    base_context={
        "table_name": "spot_trades",
        "where_clause": "TRUE",
        "own_lookforward_ms": 200,
        "rise_threshold_ratio": 0.0055,
    },
)

PRICE_DELTA_4S_TARGET_DECLARATION_BINNED_1S = FeatureDeclaration(
    name="price_delta_4s",
    template=PRICE_DELTA_TARGET_TEMPLATE_BINNED_1S,
    ml_fill_policy=_ml_fill_policy(),
    description=(
        "Future price delta on 1s bins with 0.45% deadband: "
        "if |delta| < 0.45% * price_last(t), target is 0."
    ),
    own_lookback_ms=0,
    own_lookforward_ms=4_000,
)

PRICE_DELTA_4S_TARGET_DECLARATION = FeatureDeclaration(
    name="price_delta_4s",
    template=PRICE_DELTA_TARGET_TEMPLATE_TRADE_TIME,
    ml_fill_policy=_ml_fill_policy(),
    description=(
        "Future price delta on trade-time with 0.45% deadband: "
        "if |delta| < 0.45% * price(t), target is 0."
    ),
    own_lookback_ms=0,
    own_lookforward_ms=4_000,
    enabled=True,
)

PRICE_DIRECTION_4S_TARGET_DECLARATION = FeatureDeclaration(
    name="price_direction_4s",
    template=PRICE_DIRECTION_TARGET_TEMPLATE_TRADE_TIME,
    ml_fill_policy=_ml_fill_policy(),
    description=(
        "Future trade-time direction label with 0.55% threshold: "
        "-1 when delta < -0.55%*price(t), +1 when delta > +0.55%*price(t), else 0."
    ),
    own_lookback_ms=0,
    own_lookforward_ms=4_000,
    enabled=True,
)

SELL_RISE_HIT_055_200MS_TARGET_DECLARATION = FeatureDeclaration(
    name="sell_rise_hit_055_200ms",
    template=SELL_RISE_HIT_055_TARGET_TEMPLATE_TRADE_TIME,
    ml_fill_policy=_ml_fill_policy(),
    description=(
        "Binary label: 1 if in next 200ms exists >=1 SELL trade with "
        "price > current price by 0.55%, else 0."
    ),
    own_lookback_ms=0,
    own_lookforward_ms=200,
    enabled=True,
)

PRICE_DELTA_4S_HORIZON = timedelta(seconds=4)

# Backward-compatible aliases for older imports.
PRICE_DELTA_10S_TARGET_DECLARATION_BINNED_1S = PRICE_DELTA_4S_TARGET_DECLARATION_BINNED_1S
PRICE_DELTA_10S_TARGET_DECLARATION = PRICE_DELTA_4S_TARGET_DECLARATION
PRICE_DELTA_10S_HORIZON = PRICE_DELTA_4S_HORIZON

ALL_FEATURE_DECLARATIONS = (
    *SPOT_TRADES_ROLLING_DECLARATIONS,
    FIRST_LAST_MOMENTUM_DECLARATION,
    FIRST_LAST_MOMENTUM_FOR_BALANCE_DECLARATION,
    SIDE_VOLUME_BALANCE_DECLARATION,
    # PRICE_DELTA_4S_TARGET_DECLARATION_BINNED_1S,
    PRICE_DELTA_4S_TARGET_DECLARATION,
    PRICE_DIRECTION_4S_TARGET_DECLARATION,
    SELL_RISE_HIT_055_200MS_TARGET_DECLARATION,
)

