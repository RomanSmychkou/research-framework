from __future__ import annotations

import argparse
import json
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from scripts.project_snapshot import load_dump_config

LOG_PATH = Path(".run_context_log.jsonl")


def _iso_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _append(entry: dict[str, Any]) -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LOG_PATH.open("a", encoding="utf-8") as fp:
        fp.write(json.dumps(entry, ensure_ascii=False) + "\n")


def start_run(cmd: str, args: list[str], stage: str | None = None) -> str:
    config = load_dump_config(Path(".dump_config.json"))
    uid = f"{uuid.uuid4().hex}"
    entry: dict[str, Any] = {
        "uid": uid,
        "phase": "start",
        "command": cmd,
        "args": args,
        "stage": stage,
        "project_snapshot_hash": config.get("project_snapshot_hash", ""),
        "timestamp": _iso_now(),
    }
    _append(entry)
    return uid


def finish_run(uid: str, state: str, extras: dict[str, Any] | None = None) -> None:
    entry: dict[str, Any] = {
        "uid": uid,
        "phase": "finish",
        "state": state,
        "timestamp": _iso_now(),
    }
    if extras:
        entry["extras"] = extras
    _append(entry)


@contextmanager
def track(cmd: str, args: list[str], stage: str | None = None) -> Any:
    uid = start_run(cmd, args, stage=stage)
    try:
        yield uid
        finish_run(uid, state="success")
    except Exception as exc:  # noqa: BLE001
        finish_run(uid, state="error", extras={"error": str(exc)})
        raise


def cli() -> None:
    parser = argparse.ArgumentParser(description="Log run context for quant scripts.")
    parser.add_argument("command", help="Command being executed.")
    parser.add_argument("args", nargs=argparse.REMAINDER)
    parser.add_argument("--stage", help="Short stage tag (01, 02, etc.).")
    args = parser.parse_args()
    uid = start_run(args.command, args.args, stage=args.stage)
    finish_run(uid, state="success")


if __name__ == "__main__":
    cli()
