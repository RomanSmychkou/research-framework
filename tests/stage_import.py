"""Load stage scripts from repo root for tests."""

from __future__ import annotations

from importlib.util import spec_from_file_location
import importlib.util
import sys
from pathlib import Path
from types import ModuleType

ROOT = Path(__file__).resolve().parents[1]

STAGE_ALIASES: dict[str, str] = {}

_CACHE: dict[str, ModuleType] = {}


def _stage_path(filename: str) -> Path:
    resolved = STAGE_ALIASES.get(filename, filename)
    return ROOT / resolved


def load_stage(module_name: str, filename: str) -> ModuleType:
    key = f"{module_name}:{filename}"
    if key in _CACHE:
        return _CACHE[key]

    import bootstrap_path

    path = _stage_path(filename)
    bootstrap_path.ensure_src_on_path(path)
    spec = spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load stage module from {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    _CACHE[key] = mod
    return mod
