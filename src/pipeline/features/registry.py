from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

from .declarations import ALL_FEATURE_DECLARATIONS, FeatureDeclaration


@dataclass(frozen=True)
class FeatureRegistry:
    declarations: tuple[FeatureDeclaration, ...]

    def select(
        self,
        *,
        include: set[str] | None = None,
        include_targets: bool = True,
        include_non_targets: bool = True,
    ) -> tuple[FeatureDeclaration, ...]:
        selected: list[FeatureDeclaration] = []
        for declaration in self.declarations:
            if not declaration.enabled:
                continue
            if include is not None and declaration.name not in include:
                continue
            if declaration.template.is_target and not include_targets:
                continue
            if not declaration.template.is_target and not include_non_targets:
                continue
            selected.append(declaration)
        return tuple(selected)


@dataclass
class FeatureRegistryBuilder:
    _declarations: list[FeatureDeclaration] = field(default_factory=list)

    def register(self, declaration: FeatureDeclaration) -> "FeatureRegistryBuilder":
        self._declarations.append(declaration)
        return self

    def register_many(self, declarations: Iterable[FeatureDeclaration]) -> "FeatureRegistryBuilder":
        for declaration in declarations:
            self.register(declaration)
        return self

    def build(self) -> FeatureRegistry:
        return FeatureRegistry(declarations=tuple(self._declarations))


def default_registry() -> FeatureRegistry:
    builder = FeatureRegistryBuilder()
    for declaration in ALL_FEATURE_DECLARATIONS:
        builder.register(declaration)
    return builder.build()

