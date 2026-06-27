from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Iterable, Mapping

from .feature_contract import FeatureInstance

if TYPE_CHECKING:
    from .declarations import FeatureDeclaration


@dataclass(frozen=True)
class FeatureFactory:
    """
    Single place for runtime feature materialization.

    Factory applies shared and per-feature context overrides and returns
    immutable FeatureInstance instances ready for SQL rendering.
    """

    shared_context: Mapping[str, Any] = field(default_factory=dict)
    per_feature_context: Mapping[str, Mapping[str, Any]] | None = None

    def build(self, feature: FeatureInstance) -> FeatureInstance:
        per_feature = (
            dict(self.per_feature_context.get(feature.name, {}))
            if self.per_feature_context is not None
            else {}
        )
        return feature.with_context(**dict(self.shared_context), **per_feature)

    def build_many(self, features: Iterable[FeatureInstance]) -> tuple[FeatureInstance, ...]:
        return tuple(self.build(feature) for feature in features)

    def materialize_declaration(self, declaration: "FeatureDeclaration") -> FeatureInstance:
        per_feature = (
            dict(self.per_feature_context.get(declaration.name, {}))
            if self.per_feature_context is not None
            else {}
        )
        merged_context: dict[str, Any] = {
            **dict(declaration.context_overrides),
            **dict(self.shared_context),
            **per_feature,
        }

        # Single source of truth: context mirrors declaration-level horizons.
        merged_context["own_lookback_ms"] = declaration.own_lookback_ms
        merged_context["own_lookforward_ms"] = declaration.own_lookforward_ms

        return FeatureInstance(
            template=declaration.template,
            name=declaration.name,
            description=declaration.description,
            context=merged_context,
            ml_fill_policy=declaration.ml_fill_policy,
            parent_runtime_instances=(),
            own_lookback_ms=declaration.own_lookback_ms,
            own_lookforward_ms=declaration.own_lookforward_ms,
        )

    @staticmethod
    def freeze_mapping(mapping: Mapping[str, FeatureInstance]) -> Mapping[str, FeatureInstance]:
        return MappingProxyType(dict(mapping))

