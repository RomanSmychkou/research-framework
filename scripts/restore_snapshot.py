from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Iterable

from scripts.project_snapshot import load_dump_config


def _collect_snapshots(config: dict[str, any]) -> Iterable[tuple[str, Path]]:
    hashes = config.get("hashes", {})
    for rel, digest in hashes.items():
        yield rel, Path("backups") / digest / rel


def restore_snapshot(
    *,
    target_dir: Path,
    files: Iterable[str] | None = None,
    overwrite: bool = False,
) -> None:
    config = load_dump_config()
    entries = list(_collect_snapshots(config))
    target_dir.mkdir(parents=True, exist_ok=True)
    for rel, src in entries:
        if files is not None and rel not in files:
            continue
        if not src.exists():
            continue
        dest = target_dir / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists() and not overwrite:
            continue
        shutil.copy2(src, dest)


def cli() -> None:
    parser = argparse.ArgumentParser(description="Restore project snapshot from backups.")
    parser.add_argument(
        "--target",
        "-t",
        type=Path,
        default=Path("restore"),
        help="Destination root for restored files (default ./restore).",
    )
    parser.add_argument(
        "--file",
        "-f",
        action="append",
        dest="files",
        help="Specific relative file(s) to restore (repeatable).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite files inside target if they exist.",
    )
    args = parser.parse_args()
    restore_snapshot(target_dir=args.target, files=args.files, overwrite=args.overwrite)


if __name__ == "__main__":
    cli()
