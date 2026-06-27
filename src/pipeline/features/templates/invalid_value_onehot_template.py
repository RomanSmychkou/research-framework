from __future__ import annotations

INVALID_VALUE_ONEHOT_SQL_TEMPLATE = """
SELECT
    bucket_time,
    symbol,
    toFloat64(
        if(
            isNull(parsed_value) OR NOT isFinite(ifNull(parsed_value, 0.0)),
            1,
            0
        )
    ) AS feature_value
FROM (
    SELECT
        bucket_time,
        symbol,
        toFloat64OrNull(toString(feature_value)) AS parsed_value
    FROM (
{base_feature_sql}
    ) AS base_feature
) AS normalized_feature
"""


def render_invalid_value_onehot_template(base_feature_sql: str) -> str:
    indented_base = "\n".join(f"        {line}" if line else "" for line in base_feature_sql.splitlines())
    return INVALID_VALUE_ONEHOT_SQL_TEMPLATE.format(base_feature_sql=indented_base)

