from __future__ import annotations

from ..default_contexts import VOLUME_BALANCE_DEFAULT_CONTEXT
from ..feature_contract import FeatureTemplate


SIGNED_SIDE_AGGREGATE_TEMPLATE = FeatureTemplate(
    name="signed_side_aggregate_template",
    description="Агрегация объёмов покупок и продаж с учётом знака SIDE.",
    sql_query_shadow="""
SELECT
    {time_column} AS bucket_time,
    {symbol_column},
    (
        sumIf({volume_column}, upper({side_column}) = 'BUY') OVER w -
        sumIf({volume_column}, upper({side_column}) = 'SELL') OVER w
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
        "volume_column",
        "side_column",
        "table_name",
        "where_clause",
        "own_lookback_ms",
    ),
    base_context=VOLUME_BALANCE_DEFAULT_CONTEXT,
)
