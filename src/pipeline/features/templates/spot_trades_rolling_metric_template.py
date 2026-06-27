from __future__ import annotations

from ..feature_contract import FeatureTemplate


SPOT_TRADES_ROLLING_METRIC_TEMPLATE = FeatureTemplate(
    name="spot_trades_rolling_metric_template",
    description=(
        "Универсальный шаблон оконной full-causal фичи по spot_trades: "
        "только прошлые строки, текущая запись исключается."
    ),
    sql_query_shadow="""
SELECT
    {time_column} AS bucket_time,
    {symbol_column},
    ({metric_expression}) AS feature_value
FROM {table_name}
WHERE {where_clause}
WINDOW w AS (
    PARTITION BY {symbol_column}
    ORDER BY toUnixTimestamp64Milli({time_column})
    RANGE BETWEEN {own_lookback_ms} PRECEDING AND 1 PRECEDING
)
""",
    required_context_keys=(
        "time_column",
        "symbol_column",
        "side_column",
        "price_column",
        "quantity_column",
        "table_name",
        "where_clause",
        "own_lookback_ms",
        "metric_expression",
    ),
    base_context={
        "time_column": "exchange_ts",
        "symbol_column": "symbol",
        "side_column": "side",
        "price_column": "price",
        "quantity_column": "quantity",
        "table_name": "spot_trades",
        "where_clause": "TRUE",
        "own_lookback_ms": 300_000,
        "metric_expression": "0.0",
    },
)
