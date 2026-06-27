from __future__ import annotations

from abc import ABC
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping, Optional, Tuple, TypedDict

FEATURE_BUCKET_GRANULARITY_KEY = "feature_bucket_granularity_ms"

#TODO не забыть сделать везде честный full-causal и работу с временными пропусками, 
# - nan/inf/None и их ml fill policy
#TODO понять как будет влиять causal поведение "исключая текущую строку" при расчёте 
# - одних фич из других — т.е. рассмотреть куммулятивный эффект такого накопления исключаемых данных


class MlFillPolicy(TypedDict):
    special_value: Any | None
    onehot_encoding: bool


@dataclass(frozen=True)
class BaseFeatureTemplate(ABC):
    """
    Minimal contract for every feature template.
    """

    name: str
    description: Optional[str]
    sql_query_shadow: str
    required_context_keys: Tuple[str, ...]
    is_target: bool
    is_causal: bool

    def __post_init__(self) -> None:
        if self.is_target and self.is_causal:
            raise ValueError("Target features can not be causal")
        if not self.sql_query_shadow.strip():
            raise ValueError("sql_query_shadow can not be empty")
        if not self.required_context_keys:
            raise ValueError("required_context_keys can not be empty")


@dataclass(frozen=True)
class FeatureTemplate(BaseFeatureTemplate):
    """
    Concrete template declaration without runtime feature-specific dependencies.
    """

    is_target: bool = field(default=False)
    is_causal: bool = field(default=True)
    base_context: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        super().__post_init__()
        missing = set(self.required_context_keys) - set(self.base_context.keys())
        if missing:
            raise ValueError(f"Template base_context must contain keys {missing}")

        object.__setattr__(self, "base_context", MappingProxyType(dict(self.base_context)))

    @property
    def sql_template(self) -> str:
        return self.sql_query_shadow


@dataclass(frozen=True)
class FeatureInstance:
    """
    Concrete feature instance:
    - gets metadata/SQL skeleton from template;
    - carries concrete semantic name/description;
    - carries final merged context (alias: context_overrides for declaration parity);
    - keeps links to concrete dependencies.
    """

    template: FeatureTemplate
    name: str
    ml_fill_policy: MlFillPolicy
    description: Optional[str] = None
    context: Mapping[str, Any] = field(default_factory=dict)
    parent_runtime_instances: Tuple["FeatureInstance", ...] = field(default_factory=tuple)
    own_lookback_ms: int = 0
    own_lookforward_ms: int = 0

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("FeatureInstance name can not be empty")
        merged_context = dict(self.template.base_context)
        merged_context.update(dict(self.context))

        missing = set(self.template.required_context_keys) - set(merged_context.keys())
        if missing:
            raise ValueError(f"Context must contain keys {missing}")
        
        if self.parent_runtime_instances and not all(
            isinstance(dep, FeatureInstance) for dep in self.parent_runtime_instances
        ):
            raise TypeError("parent_runtime_instances must contain only FeatureInstance instances")

        if not isinstance(self.own_lookback_ms, int):
            raise TypeError("own_lookback_ms must be int")
        if not isinstance(self.own_lookforward_ms, int):
            raise TypeError("own_lookforward_ms must be int")
        if self.own_lookback_ms < 0:
            raise ValueError("own_lookback_ms must be >= 0")
        if self.own_lookforward_ms < 0:
            raise ValueError("own_lookforward_ms must be >= 0")

        context_own_lookback_ms = merged_context.get("own_lookback_ms")
        if context_own_lookback_ms is not None and context_own_lookback_ms != self.own_lookback_ms:
            raise ValueError("Context own_lookback_ms must match FeatureInstance own_lookback_ms")
        context_own_lookforward_ms = merged_context.get("own_lookforward_ms")
        if context_own_lookforward_ms is not None and context_own_lookforward_ms != self.own_lookforward_ms:
            raise ValueError("Context own_lookforward_ms must match FeatureInstance own_lookforward_ms")

        dep_names = [dep.name for dep in self.parent_runtime_instances]
        if self.name in dep_names:
            raise ValueError("FeatureInstance can not depend on itself")
        if len(dep_names) != len(set(dep_names)):
            raise ValueError("parent_runtime_instances contains duplicate feature names")
        if self.is_target and self.is_causal:
            raise ValueError("FeatureInstance can not be both target and causal")
        if self.is_causal and self.summary_lookforward_ms > 0:
            raise ValueError(
                "Causal features must have summary_lookforward_ms == 0"
            )
        if self.is_target and self.summary_lookback_ms > 0:
            raise ValueError(
                "Target features must have summary_lookback_ms == 0"
            )

        policy = dict(self.ml_fill_policy)
        required_policy_keys = {"special_value", "onehot_encoding"}
        missing_policy_keys = required_policy_keys - set(policy.keys())
        if missing_policy_keys:
            raise ValueError(f"ml_fill_policy missing required keys: {sorted(missing_policy_keys)}")
        extra_policy_keys = set(policy.keys()) - required_policy_keys
        if extra_policy_keys:
            raise ValueError(f"ml_fill_policy contains unsupported keys: {sorted(extra_policy_keys)}")
        if not isinstance(policy["onehot_encoding"], bool):
            raise TypeError("ml_fill_policy['onehot_encoding'] must be bool")

        normalized_policy: MlFillPolicy = {
            "special_value": policy["special_value"],
            "onehot_encoding": policy["onehot_encoding"],
        }

        object.__setattr__(self, "context", MappingProxyType(merged_context))
        object.__setattr__(self, "ml_fill_policy", MappingProxyType(normalized_policy))

    @property
    def is_target(self) -> bool:
        return self.template.is_target

    @property
    def is_causal(self) -> bool:
        return self.template.is_causal

    @property
    def required_context_keys(self) -> Tuple[str, ...]:
        return self.template.required_context_keys

    @property
    def context_overrides(self) -> Mapping[str, Any]:
        """
        Declaration-parity alias.

        FeatureDeclaration uses ``context_overrides`` naming, while runtime instances
        store the merged immutable mapping in ``context``.
        """
        return self.context

    @property
    def sql_template(self) -> str:
        return self.template.sql_template

    @property
    def parents(self) -> Tuple["FeatureInstance", ...]:
        return self.parent_runtime_instances

    @property
    def parent_names(self) -> Tuple[str, ...]:
        return tuple(parent.name for parent in self.parents)

    @property
    def summary_lookback_ms(self) -> int:
        cache_attr = "_summary_lookback_ms_cache"
        if not hasattr(self, cache_attr):
            parent_max_lookback_ms = max(
                (parent.summary_lookback_ms for parent in self.parents),
                default=0,
            )
            object.__setattr__(
                self,
                cache_attr,
                self.own_lookback_ms + parent_max_lookback_ms,
            )
        return getattr(self, cache_attr)

    @property
    def summary_lookforward_ms(self) -> int:
        cache_attr = "_summary_lookforward_ms_cache"
        if not hasattr(self, cache_attr):
            parent_max_lookforward_ms = max(
                (parent.summary_lookforward_ms for parent in self.parents),
                default=0,
            )
            object.__setattr__(
                self,
                cache_attr,
                self.own_lookforward_ms + parent_max_lookforward_ms,
            )
        return getattr(self, cache_attr)

    def with_context(self, **context_overrides: Any) -> "FeatureInstance":
        return FeatureInstance(
            template=self.template,
            name=self.name,
            description=self.description,
            context={**dict(self.context), **context_overrides},
            ml_fill_policy=self.ml_fill_policy,
            parent_runtime_instances=self.parent_runtime_instances,
            own_lookback_ms=self.own_lookback_ms,
            own_lookforward_ms=self.own_lookforward_ms,
        )

    def with_parents(self, *parents: "FeatureInstance") -> "FeatureInstance":
        return FeatureInstance(
            template=self.template,
            name=self.name,
            description=self.description,
            context=self.context,
            ml_fill_policy=self.ml_fill_policy,
            parent_runtime_instances=tuple(parents),
            own_lookback_ms=self.own_lookback_ms,
            own_lookforward_ms=self.own_lookforward_ms,
        )

    def render_sql(self) -> str:
        return self.sql_template.format(**self.context)
