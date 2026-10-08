"""Append-only, hash-chained consent / usage log (one JSON file per project)."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

GENESIS = "0" * 64


def _digest(entry: dict[str, Any]) -> str:
    body = {k: v for k, v in entry.items() if k != "hash"}
    return hashlib.sha256(json.dumps(body, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


class ConsentLog:
    def __init__(self, path: Path):
        self.path = Path(path)

    def entries(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        return json.loads(self.path.read_text(encoding="utf-8"))["entries"]

    def append(self, event: str, **data: Any) -> dict[str, Any]:
        entries = self.entries()
        problems = self.verify(entries)
        if problems:
            raise RuntimeError(f"consent log is corrupted, refusing to append: {problems}")
        entry = {
            "seq": len(entries),
            "event": event,
            "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "prev_hash": entries[-1]["hash"] if entries else GENESIS,
            "data": data,
        }
        entry["hash"] = _digest(entry)
        entries.append(entry)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps({"version": 1, "entries": entries}, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(self.path)
        return entry

    def verify(self, entries: list[dict[str, Any]] | None = None) -> list[str]:
        entries = self.entries() if entries is None else entries
        problems, prev = [], GENESIS
        for i, e in enumerate(entries):
            if e.get("seq") != i:
                problems.append(f"entry {i}: bad sequence number")
            if e.get("prev_hash") != prev:
                problems.append(f"entry {i}: chain broken")
            if e.get("hash") != _digest(e):
                problems.append(f"entry {i}: content modified")
            prev = e.get("hash", "")
        return problems
