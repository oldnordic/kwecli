"""
Conversation compaction with drift control.

Enforces structured snapshots on every compaction to prevent content/context/code/project drift.
Each snapshot must include these sections: Plan, TODOs, Decisions, Assumptions, Open Questions.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Any


REQUIRED_SECTIONS = [
    "plan",
    "todos",
    "decisions",
    "assumptions",
    "open_questions",
]


@dataclass
class CompactionSnapshot:
    timestamp: str
    plan: str
    todos: List[str]
    decisions: List[str]
    assumptions: List[str]
    open_questions: List[str]
    meta: Dict[str, Any] = field(default_factory=dict)

    def to_markdown(self) -> str:
        def section(title: str, items: list[str]) -> list[str]:
            out = [f"## {title}"]
            if items:
                out.extend([f"- {x}" for x in items])
            else:
                out.append("(none)")
            return out

        lines: list[str] = []
        lines.append(f"# Conversation Snapshot ({self.timestamp})")
        lines.append("## Plan")
        lines.append(self.plan or "(none)")
        lines.append("")
        lines.extend(section("TODOs", self.todos))
        lines.append("")
        lines.extend(section("Decisions", self.decisions))
        lines.append("")
        lines.extend(section("Assumptions", self.assumptions))
        lines.append("")
        lines.extend(section("Open Questions", self.open_questions))
        return "\n".join(lines)


def compact_conversation(history: List[Dict[str, Any]], context: Dict[str, Any] | None = None) -> CompactionSnapshot:
    """Create a drift-resistant snapshot from conversation history.

    Minimal heuristic: pull the latest plan/todos/etc. keys from messages/meta, else scaffold.
    """
    context = context or {}
    latest: Dict[str, Any] = {}

    # Walk from newest to oldest to find recent structured fields
    for msg in reversed(history or []):
        if not isinstance(msg, dict):
            continue
        for key in REQUIRED_SECTIONS:
            if key in latest:
                continue
            val = msg.get(key) or (msg.get("meta") or {}).get(key)
            if val:
                latest[key] = val
        if len(latest) == len(REQUIRED_SECTIONS):
            break

    # Defaults
    plan = str(latest.get("plan", context.get("plan", ""))).strip()
    todos = list(latest.get("todos", context.get("todos", [])) or [])
    decisions = list(latest.get("decisions", context.get("decisions", [])) or [])
    assumptions = list(latest.get("assumptions", context.get("assumptions", [])) or [])
    open_questions = list(latest.get("open_questions", context.get("open_questions", [])) or [])

    # Enforce presence: scaffold if missing
    if not plan:
        plan = "1) Decompose task 2) Implement 3) Test 4) Document"
    todos = todos or ["Add/Update tests", "Run lint + type-check", "Update docs"]

    snapshot = CompactionSnapshot(
        timestamp=datetime.utcnow().isoformat(),
        plan=plan,
        todos=todos,
        decisions=decisions,
        assumptions=assumptions,
        open_questions=open_questions,
        meta={"size": len(history or [])},
    )
    return snapshot
