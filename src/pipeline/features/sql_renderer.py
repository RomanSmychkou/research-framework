from __future__ import annotations

import math
from datetime import datetime

from .feature_contract import FeatureInstance
from .templates import (
    render_invalid_value_onehot_template,
    render_invalid_value_replace_template,
)


def _sql_string_literal(value: str) -> str:
    escaped = value.replace("\\", "\\\\").replace("'", "\\'")
    return f"'{escaped}'"


def _render_feature_sql_base(feature: FeatureInstance) -> str:
    """
    Render feature SQL with nested placeholders support.

    First pass resolves template placeholders.
    Second pass resolves placeholders embedded in context expressions
    (for example metric_expression with {side_column}).
    """

    return feature.render_sql().format(**dict(feature.context))


def render_feature_sql(feature: FeatureInstance) -> str:
    """
    Render feature SQL without outer batch guards (unsafe mode).
    """

    return _render_feature_sql_base(feature)


def render_feature_sql_safe(
    feature: FeatureInstance,
    *,
    guard_start: datetime,
    guard_end: datetime,
) -> str:
    """
    Render feature SQL with explicit outer guard bounds.

    Guard bounds are mandatory to enforce chunk/batch limits at the final result
    shape (`bucket_time`, `symbol`, `feature_value`) even if inner template
    constraints are accidentally weakened.
    """

    if guard_start is None or guard_end is None:
        raise ValueError("guard_start and guard_end are required")
    assert guard_start is not None and guard_end is not None, (
        "guard_start and guard_end must be set before SQL rendering"
    )
    if guard_start >= guard_end:
        raise ValueError("guard_start must be < guard_end")
    assert guard_start < guard_end, "guard_start must be strictly less than guard_end"
    return f"""
SELECT *
FROM (
{_render_feature_sql_base(feature)}
) AS guarded_rows
WHERE bucket_time >= %(guard_start)s
  AND bucket_time < %(guard_end)s
"""


def render_feature_invalid_onehot_sql(feature: FeatureInstance) -> str:
    """
    Render SQL that emits binary feature_value=1 for invalid base values.

    Invalid means parsed numeric value is NULL/NaN/Inf (or any non-numeric token).
    """

    return render_invalid_value_onehot_template(_render_feature_sql_base(feature))


def render_feature_invalid_replace_sql(
    feature: FeatureInstance,
    *,
    special_value: int | float | str,
) -> str:
    """
    Render SQL that replaces invalid base values with special_value.

    Invalid means parsed numeric value is NULL/NaN/Inf (or any non-numeric token).
    """

    if isinstance(special_value, bool):
        raise TypeError("special_value must be int/float/string value, not bool")
    if isinstance(special_value, (int, float)):
        numeric = float(special_value)
        if not math.isfinite(numeric):
            raise ValueError("special_value must be finite")
        special_value_expression = repr(numeric)
    elif isinstance(special_value, str):
        special_value_expression = (
            f"toFloat64OrZero({_sql_string_literal(special_value)})"
        )
    else:
        raise TypeError("special_value must be int/float/string value")

    return render_invalid_value_replace_template(
        _render_feature_sql_base(feature),
        special_value_expression=special_value_expression,
    )

