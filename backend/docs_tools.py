#!/usr/bin/env python3
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Dict, Any, List


DOC_FILES = [
    "docs/PROJECT_GOAL.md",
    "docs/PLAN.md",
    "docs/TECH_STACK.md",
    "docs/TODO.md",
    "docs/STATUS.md",
]


def _sha1(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8", errors="ignore")).hexdigest()


def read_docs() -> Dict[str, Any]:
    """Read tracked docs and return {files: [{path, bytes, hash}], missing: []}."""
    files: List[Dict[str, Any]] = []
    missing: List[str] = []
    for p in DOC_FILES:
        path = Path(p)
        if not path.exists():
            missing.append(p)
            continue
        data = path.read_text(encoding="utf-8", errors="ignore")
        files.append({
            "path": str(path),
            "bytes": len(data.encode("utf-8", errors="ignore")),
            "hash": _sha1(data),
        })
    return {"files": files, "missing": missing}


def append_todos(lines: List[str]) -> Dict[str, Any]:
    path = Path("docs/TODO.md")
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# TODO\n\n", encoding="utf-8")
    text = path.read_text(encoding="utf-8", errors="ignore")
    before_hash = _sha1(text)
    to_add = [f"- [ ] {ln.strip()}" for ln in lines if ln.strip()]
    if to_add:
        if not text.endswith("\n"): text += "\n"
        text += "\n" + "\n".join(to_add) + "\n"
        path.write_text(text, encoding="utf-8")
    after_hash = _sha1(text)
    return {"path": str(path), "added": len(to_add), "hash_before": before_hash, "hash_after": after_hash}

