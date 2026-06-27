"""Project file snapshots: hash-tracked dumps per .dump_config.json."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Any, Iterator

CONFIG_NAME = ".dump_config.json"


def _repo_root(config_path: Path) -> Path:
    anchor = config_path.resolve().parent if config_path.exists() else Path.cwd()
    for candidate in (anchor, *anchor.parents):
        if (candidate / "pyproject.toml").is_file():
            return candidate
    return Path.cwd()


def _load_config(config_path: Path) -> dict[str, Any]:
    if not config_path.exists():
        return {
            "include": {
                "folders": ["."],
                "filetypes": [".py", ".sql", ".md", ".json", ".yaml", ".toml", ".jsonl"],
            },
            "exclude": {
                "folders": [".git", "__pycache__", ".venv", "backups", ".cursor"],
                "filetypes": [".pyc"],
            },
            "hashes": {},
            "project_snapshot_hash": "",
        }
    return json.loads(config_path.read_text(encoding="utf-8"))


def _save_config(config_path: Path, payload: dict[str, Any]) -> None:
    config_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _normalize_relpath(root: Path, path: Path) -> str:
    return str(path.resolve().relative_to(root.resolve())).replace("\\", "/")


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fp:
        for chunk in iter(lambda: fp.read(8192), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _path_has_excluded_folder(rel: str, exclude_folders: set[str]) -> bool:
    parts = rel.split("/")
    return any(part in exclude_folders for part in parts)


def _iter_tracked_files(
    root: Path,
    include: dict[str, Any],
    exclude: dict[str, Any],
) -> Iterator[Path]:
    include_folders: list[str] = include.get("folders") or ["."]
    exclude_folders: set[str] = set(exclude.get("folders") or [])
    include_exts: set[str] = set(include.get("filetypes") or [])
    exclude_exts: set[str] = set(exclude.get("filetypes") or [])
    seen: set[str] = set()

    for folder in include_folders:
        base = (root / folder).resolve()
        if not base.exists():
            continue
        if base.is_file():
            candidates = [base]
        else:
            candidates = []
            for dirpath, dirnames, filenames in os.walk(base, topdown=True):
                dirnames[:] = sorted(
                    d for d in dirnames if d not in exclude_folders
                )
                for name in sorted(filenames):
                    candidates.append(Path(dirpath) / name)

        for path in candidates:
            if not path.is_file():
                continue
            rel = _normalize_relpath(root, path)
            if rel in seen:
                continue
            seen.add(rel)
            if _path_has_excluded_folder(rel, exclude_folders):
                continue
            if path.suffix in exclude_exts:
                continue
            if include_exts and path.suffix not in include_exts:
                continue
            yield path


def do_project_snapshot(
    *,
    config_path: Path | None = None,
    backups_dir: Path | None = None,
    **_ignored: Any,
) -> str:
    cfg_path = config_path or Path(CONFIG_NAME)
    root = _repo_root(cfg_path)
    backup_root = backups_dir or (root / "backups")
    config = _load_config(cfg_path)
    include = config.get("include", {})
    exclude = config.get("exclude", {})
    hashes: dict[str, str] = dict(config.get("hashes", {}))

    for path in _iter_tracked_files(root, include, exclude):
        rel = _normalize_relpath(root, path)
        digest = _hash_file(path)
        if hashes.get(rel) == digest:
            continue
        target = backup_root / digest / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        hashes[rel] = digest

    snapshot = hashlib.sha256()
    for rel in sorted(hashes):
        snapshot.update(rel.encode("utf-8"))
        snapshot.update(b":")
        snapshot.update(hashes[rel].encode("utf-8"))

    config["hashes"] = hashes
    config["project_snapshot_hash"] = snapshot.hexdigest()
    _save_config(cfg_path, config)
    return config["project_snapshot_hash"]


def load_dump_config(config_path: Path = Path(CONFIG_NAME)) -> dict[str, Any]:
    return _load_config(config_path)


def cli() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Create project snapshots for auditing.")
    parser.add_argument("--config", default=CONFIG_NAME, help="Path to .dump_config.json")
    parser.add_argument("--backups", default="backups", help="Directory to store dumps.")
    args = parser.parse_args()

    project_hash = do_project_snapshot(
        config_path=Path(args.config),
        backups_dir=Path(args.backups),
    )
    print(f"Project snapshot hash: {project_hash}")


if __name__ == "__main__":
    cli()
