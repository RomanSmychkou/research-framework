from __future__ import annotations

from collections import defaultdict, deque

from .declarations import FeatureDeclaration


def validate_declarations(declarations: tuple[FeatureDeclaration, ...]) -> None:
    by_name = {decl.name: decl for decl in declarations}
    if len(by_name) != len(declarations):
        raise ValueError("Feature declarations contain duplicate names")
    for declaration in declarations:
        if declaration.name in declaration.parent_names:
            raise ValueError(f"Feature '{declaration.name}' can not depend on itself")
        missing = [dep for dep in declaration.parent_names if dep not in by_name]
        if missing:
            raise ValueError(
                f"Feature '{declaration.name}' has missing dependencies: {missing}"
            )


def topological_order(declarations: tuple[FeatureDeclaration, ...]) -> tuple[FeatureDeclaration, ...]:
    validate_declarations(declarations)
    by_name = {decl.name: decl for decl in declarations}
    outgoing: dict[str, list[str]] = defaultdict(list)
    indegree: dict[str, int] = {name: 0 for name in by_name}
    for declaration in declarations:
        for dependency in declaration.parent_names:
            outgoing[dependency].append(declaration.name)
            indegree[declaration.name] += 1

    queue = deque(sorted(name for name, degree in indegree.items() if degree == 0))
    ordered_names: list[str] = []
    while queue:
        current = queue.popleft()
        ordered_names.append(current)
        for follower in sorted(outgoing.get(current, ())):
            indegree[follower] -= 1
            if indegree[follower] == 0:
                queue.append(follower)

    if len(ordered_names) != len(by_name):
        raise ValueError("Feature declarations contain cyclic dependencies")
    return tuple(by_name[name] for name in ordered_names)

