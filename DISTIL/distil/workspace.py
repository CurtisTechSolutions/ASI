"""Where state lives on disk.

One directory, `~/.distil` by default and overridable by `DISTIL_HOME`, holding
the memory, the policy, the tools the system wrote and the lineage of edits it
made to itself. Everything here is JSON or Python source -- readable, diffable,
and deletable by hand. A self-modifying system whose state is only inspectable
through the system itself is one you cannot audit after it goes wrong.
"""
from __future__ import annotations

import os
from pathlib import Path


class Workspace:
    def __init__(self, home: Path | str | None = None) -> None:
        self.home = Path(home or os.environ.get("DISTIL_HOME", Path.home() / ".distil"))
        self.home.mkdir(parents=True, exist_ok=True)
        self.workshop.mkdir(parents=True, exist_ok=True)

    @property
    def memory(self) -> Path: return self.home / "memory.jsonl"
    @property
    def policy(self) -> Path: return self.home / "policy.json"
    @property
    def workshop(self) -> Path: return self.home / "workshop"
    @property
    def regret(self) -> Path: return self.home / "regret.json"
    @property
    def journal(self) -> Path: return self.home / "journal.jsonl"

    def __str__(self) -> str:
        return str(self.home)
