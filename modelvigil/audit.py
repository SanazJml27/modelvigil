"""Tamper-evident audit trail (hash-chained JSONL).

Supports the record-keeping (EU AI Act Art. 12) and post-market monitoring
(Art. 72) obligations: every signal, tool call, diagnosis, approval and model
change is appended with the hash of the previous entry, so any later edit
breaks the chain.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


def _default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, tuple):
        return list(o)
    return str(o)


class AuditLog:
    GENESIS = "0" * 64

    def __init__(self, path: str | Path | None = None):
        self.entries: list[dict] = []
        self.path = Path(path) if path else None
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text("")

    def append(self, week: int, actor: str, event: str, payload: dict | None = None) -> dict:
        prev = self.entries[-1]["hash"] if self.entries else self.GENESIS
        body = {"seq": len(self.entries), "week": week, "actor": actor, "event": event,
                "payload": payload or {}, "prev_hash": prev}
        canon = json.dumps(body, sort_keys=True, default=_default)
        body["hash"] = hashlib.sha256(canon.encode()).hexdigest()
        body = json.loads(json.dumps(body, default=_default))
        self.entries.append(body)
        if self.path:
            with self.path.open("a") as fh:
                fh.write(json.dumps(body, sort_keys=True) + "\n")
        return body

    def filter(self, event: str | None = None, actor: str | None = None) -> list[dict]:
        return [e for e in self.entries if (event is None or e["event"] == event)
                and (actor is None or e["actor"] == actor)]


def verify_chain(path: str | Path) -> tuple[bool, str]:
    prev = AuditLog.GENESIS
    for i, line in enumerate(Path(path).read_text().splitlines()):
        e = json.loads(line)
        h = e.pop("hash")
        if e["prev_hash"] != prev:
            return False, f"entry {i}: prev_hash mismatch"
        if hashlib.sha256(json.dumps(e, sort_keys=True).encode()).hexdigest() != h:
            return False, f"entry {i}: content hash mismatch (entry was modified)"
        prev = h
    return True, f"chain intact ({i + 1} entries)"
