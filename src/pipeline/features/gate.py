from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from .declarations import (
    FeatureDeclaration,
    PRICE_DELTA_4S_TARGET_DECLARATION,
    PRICE_DIRECTION_4S_TARGET_DECLARATION,
    SELL_RISE_HIT_055_200MS_TARGET_DECLARATION,
    SPOT_TRADES_ROLLING_DECLARATIONS,
)
from .factory import FeatureFactory
from .feature_contract import FeatureInstance
from .graph import topological_order
from .registry import FeatureRegistry, default_registry
_TARGET_DECLARATIONS_BY_NAME: dict[str, FeatureDeclaration] = {
    PRICE_DELTA_4S_TARGET_DECLARATION.name: PRICE_DELTA_4S_TARGET_DECLARATION,
    PRICE_DIRECTION_4S_TARGET_DECLARATION.name: PRICE_DIRECTION_4S_TARGET_DECLARATION,
    SELL_RISE_HIT_055_200MS_TARGET_DECLARATION.name: SELL_RISE_HIT_055_200MS_TARGET_DECLARATION,
}




@dataclass(frozen=True)
class FeatureBundle:
    features_in_compute_order: tuple[FeatureInstance, ...]
    by_name: Mapping[str, FeatureInstance]
    targets: tuple[FeatureInstance, ...]
    non_targets: tuple[FeatureInstance, ...]


def build_feature_bundle(
    *,
    shared_context: Mapping[str, Any] | None = None,
    per_feature_context: Mapping[str, Mapping[str, Any]] | None = None,
    include: set[str] | None = None,
    include_targets: bool = True,
    include_non_targets: bool = True,
    registry: FeatureRegistry | None = None,
) -> FeatureBundle:
    effective_registry = registry or default_registry()
    selected_declarations = effective_registry.select(
        include=include,
        include_targets=include_targets,
        include_non_targets=include_non_targets,
    )
    ordered_declarations = topological_order(selected_declarations)

    factory = FeatureFactory(
        shared_context=shared_context or {},
        per_feature_context=per_feature_context,
    )
    materialized = _materialize_ordered(ordered_declarations, factory)
    by_name = {feature.name: feature for feature in materialized}
    targets = tuple(feature for feature in materialized if feature.is_target)
    non_targets = tuple(feature for feature in materialized if not feature.is_target)
    return FeatureBundle(
        features_in_compute_order=materialized,
        by_name=FeatureFactory.freeze_mapping(by_name),
        targets=targets,
        non_targets=non_targets,
    )


def _materialize_ordered(
    declarations: tuple[FeatureDeclaration, ...],
    factory: FeatureFactory,
) -> tuple[FeatureInstance, ...]:
    by_name: dict[str, FeatureInstance] = {
        declaration.name: factory.materialize_declaration(declaration)
        for declaration in declarations
    }

    for declaration in declarations:
        parents = tuple(by_name[parent_name] for parent_name in declaration.parent_names)
        by_name[declaration.name] = by_name[declaration.name].with_parents(*parents)
    return tuple(by_name[declaration.name] for declaration in declarations)


def build_spot_trades_rolling_bundle(
    *,
    database: str,
    source_table: str,
    where_clause: str,
    window_ms: int,
    per_feature_context: Mapping[str, Mapping[str, Any]] | None = None,
    registry: FeatureRegistry | None = None,
) -> FeatureBundle:
    return build_feature_bundle(
        shared_context={
            "table_name": f"{database}.{source_table}",
            "where_clause": where_clause,
            "own_lookback_ms": window_ms,
        },
        per_feature_context=per_feature_context,
        include={declaration.name for declaration in SPOT_TRADES_ROLLING_DECLARATIONS},
        include_targets=False,
        registry=registry,
    )


def build_price_delta_target_feature(
    *,
    database: str,
    source_table: str,
    where_clause: str,
    horizon_seconds: int,
    per_feature_context: Mapping[str, Mapping[str, Any]] | None = None,
    registry: FeatureRegistry | None = None,
) -> FeatureInstance:
    return build_named_target_feature(
        target_name=PRICE_DELTA_4S_TARGET_DECLARATION.name,
        database=database,
        source_table=source_table,
        where_clause=where_clause,
        horizon_seconds=horizon_seconds,
        per_feature_context=per_feature_context,
        registry=registry,
    )


def build_named_target_feature(
    *,
    target_name: str,
    database: str,
    source_table: str,
    where_clause: str,
    horizon_seconds: int,
    per_feature_context: Mapping[str, Mapping[str, Any]] | None = None,
    registry: FeatureRegistry | None = None,
) -> FeatureInstance:
    declaration = _TARGET_DECLARATIONS_BY_NAME.get(target_name)
    if declaration is None:
        supported = ", ".join(sorted(_TARGET_DECLARATIONS_BY_NAME))
        raise ValueError(f"Unknown target feature {target_name!r}. Supported targets: {supported}")
    bundle = build_feature_bundle(
        shared_context={
            "table_name": f"{database}.{source_table}",
            "where_clause": where_clause,
            "own_lookforward_ms": int(horizon_seconds) * 1000,
        },
        per_feature_context=per_feature_context,
        include={declaration.name},
        include_non_targets=False,
        registry=registry,
    )
    if declaration.name in bundle.by_name:
        return bundle.by_name[declaration.name]
    return FeatureFactory(
        shared_context={
            "table_name": f"{database}.{source_table}",
            "where_clause": where_clause,
            "own_lookforward_ms": int(horizon_seconds) * 1000,
        },
        per_feature_context=per_feature_context,
    ).materialize_declaration(declaration)

