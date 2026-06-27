"""Stdlib-only helper: put `src/` on sys.path for root entrypoints and scripts."""

from __future__ import annotations

import sys
from pathlib import Path


def ensure_src_on_path(anchor: Path | None = None) -> Path:
    """Insert `<repo>/src` and repo root on sys.path; return repository root."""
    start = (anchor or Path(__file__)).resolve()
    root = start.parent if start.is_file() else start
    for candidate in (root, *root.parents):
        if (candidate / "pyproject.toml").is_file():
            root = candidate
            break
    for sub in (root, root / "src"):
        if sub.is_dir():
            sub_s = str(sub)
            if sub_s not in sys.path:
                sys.path.insert(0, sub_s)
    return root


def run_project_snapshot(root: Path | None = None) -> str | None:
    """Snapshot tracked project files (see .dump_config.json). Returns hash or None."""
    try:
        from scripts.project_snapshot import do_project_snapshot

        repo = root or Path(__file__).resolve().parent
        return do_project_snapshot(
            config_path=repo / ".dump_config.json",
            backups_dir=repo / "backups",
        )
    except Exception:
        return None
