"""Append-only JSONL log: the "perception receipts".

Every decision records the evidence it was based on, the action chosen and the reason.
The console reads this to show why the robot did what it did, and the same file is
proof for judges that decisions trace to perception. Gemini calls are logged here too
(model, prompt, config, output) with inline data stripped.
"""

import json
import os
import time
from typing import Any

from .config import LOG_DIR


class DecisionLog:
    def __init__(self, path: str | None = None):
        self.path = path or os.path.join(LOG_DIR, "decisions.jsonl")
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)

    def write(self, kind: str, **fields: Any) -> dict:
        """Append one event and return it."""
        event = {"t": time.time(), "kind": kind, **fields}
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(event, default=str) + "\n")
        return event

    def decision(self, *, evidence: dict, action: str, verdict: str, reason: str) -> dict:
        return self.write("decision", evidence=evidence, action=action, verdict=verdict, reason=reason)
