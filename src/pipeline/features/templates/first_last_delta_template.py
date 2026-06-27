from __future__ import annotations

from ..default_contexts import PRICE_MOMENTUM_DEFAULT_CONTEXT
from ..feature_contract import FeatureTemplate


FIRST_LAST_DELTA_TEMPLATE = FeatureTemplate(
    name="first_last_delta_template",
    description="Универсальный шаблон разницы между первым и последним значением в окне.",
    sql_query_shadow="""
SELECT 
    {time_column} AS bucket_time,
    {symbol_column},
    (
        last_value({price_column}) OVER w -
        first_value({price_column}) OVER w
    ) AS feature_value
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
        "price_column",
        "table_name",
        "where_clause",
        "own_lookback_ms",
    ),
    base_context=PRICE_MOMENTUM_DEFAULT_CONTEXT,
)
