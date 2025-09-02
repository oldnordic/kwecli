#!/usr/bin/env python3
from __future__ import annotations

import difflib
from pathlib import Path
from typing import Any, Dict, List, Optional


def diff3_merge(base: str, current: str, new: str) -> Dict[str, Any]:
    """Perform a simple 3-way merge. If both current and new edit overlapping hunks vs base,
    insert conflict markers. Otherwise, apply both changes.
    Returns {merged: str, conflicts: int}.
    """
    # Compute opcodes from base->current and base->new
    sm_cur = difflib.SequenceMatcher(None, base, current)
    sm_new = difflib.SequenceMatcher(None, base, new)
    ops_cur = sm_cur.get_opcodes()
    ops_new = sm_new.get_opcodes()

    out: List[str] = []
    i_cur = 0
    i_new = 0
    pos = 0
    conflicts = 0
    while i_cur < len(ops_cur) or i_new < len(ops_new):
        op_c = ops_cur[i_cur] if i_cur < len(ops_cur) else ("equal", len(base), len(base), len(current), len(current))
        op_n = ops_new[i_new] if i_new < len(ops_new) else ("equal", len(base), len(base), len(new), len(new))
        # Choose the next base span to process
        next_base = min(op_c[1], op_n[1]) if pos == 0 else pos
        # If both equal and at end
        if i_cur >= len(ops_cur) and i_new >= len(ops_new):
            break

        # For simplicity: process in order of base indices
        if i_cur < len(ops_cur) and (i_new >= len(ops_new) or op_c[1] <= op_n[1]):
            tag, i1, i2, j1, j2 = op_c
            # copy unchanged base region preceding this op from current
            if pos < i1:
                out.append(current[j1 - (i2 - i1) if tag != 'insert' else j1: j1])
            if tag == 'equal':
                out.append(current[j1:j2])
            elif tag in ('replace', 'delete', 'insert'):
                # Check overlap with new's change region
                # If new also changed this base region, mark conflict
                changed_new = any(not (n_i2 <= i1 or n_i1 >= i2) and n_tag != 'equal' for (n_tag, n_i1, n_i2, _, _) in ops_new)
                if changed_new and tag != 'equal':
                    # Conflict
                    conflicts += 1
                    # Compose conflict block
                    cur_chunk = current[j1:j2]
                    # Approximate corresponding new chunk
                    # Map base span i1:i2 to new via op_n nearest
                    new_chunk = ''.join([new[k1:k2] for (t2, k_i1, k_i2, k1, k2) in ops_new if not (k_i2 <= i1 or k_i1 >= i2) and t2 != 'equal'])
                    out.append(''.join([
                        "<<<<<<< CURRENT\n", cur_chunk, "\n=======\n", new_chunk, "\n>>>>>>> NEW\n"]))
                else:
                    out.append(current[j1:j2])
            pos = i2
            i_cur += 1
        else:
            tag, i1, i2, j1, j2 = op_n
            if pos < i1:
                out.append(new[j1 - (i2 - i1) if tag != 'insert' else j1: j1])
            if tag == 'equal':
                out.append(new[j1:j2])
            elif tag in ('replace', 'delete', 'insert'):
                changed_cur = any(not (c_i2 <= i1 or c_i1 >= i2) and c_tag != 'equal' for (c_tag, c_i1, c_i2, _, _) in ops_cur)
                if changed_cur and tag != 'equal':
                    conflicts += 1
                    new_chunk = new[j1:j2]
                    cur_chunk = ''.join([current[k1:k2] for (t2, k_i1, k_i2, k1, k2) in ops_cur if not (k_i2 <= i1 or k_i1 >= i2) and t2 != 'equal'])
                    out.append(''.join([
                        "<<<<<<< CURRENT\n", cur_chunk, "\n=======\n", new_chunk, "\n>>>>>>> NEW\n"]))
                else:
                    out.append(new[j1:j2])
            pos = i2
            i_new += 1

    merged = ''.join(out)
    return {"merged": merged, "conflicts": conflicts}


def apply_batch(edits: List[Dict[str, Any]], dry_run: bool = False, allow_partial: bool = False) -> Dict[str, Any]:
    """Apply a batch of edits with optional 3-way merge support per edit.
    Each edit: {path, new_content, base_content?}. If base provided and file exists, use diff3_merge.
    Returns per-file results and overall success; writes atomically if not allow_partial.
    """
    results: List[Dict[str, Any]] = []
    to_write: List[tuple[Path, str]] = []
    for e in edits:
        path = e.get("path")
        new_content = e.get("new_content")
        base_content = e.get("base_content")
        if not path or new_content is None:
            results.append({"path": path, "success": False, "error": "invalid edit"})
            continue
        p = Path(path)
        current = p.read_text(encoding="utf-8", errors="ignore") if p.exists() else ""
        if base_content is not None and p.exists():
            m = diff3_merge(base_content, current, new_content)
            merged = m["merged"]
            conflicts = int(m.get("conflicts", 0))
            results.append({"path": str(p), "success": True, "conflicts": conflicts})
            to_write.append((p, merged))
        else:
            results.append({"path": str(p), "success": True, "conflicts": 0})
            to_write.append((p, new_content))

    success = all(r.get("success") for r in results)
    if dry_run:
        return {"success": success, "results": results}

    if not allow_partial and not success:
        return {"success": False, "results": results, "error": "batch contains invalid edits"}

    # write
    for p, content in to_write:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)

    return {"success": success, "results": results}


def _parse_unified_hunks(diff_text: str) -> List[Dict[str, Any]]:
    lines = diff_text.splitlines()
    hunks: List[Dict[str, Any]] = []
    i = 0
    # skip headers if present
    while i < len(lines) and not lines[i].startswith('@@'):
        i += 1
    while i < len(lines):
        if not lines[i].startswith('@@'):
            i += 1
            continue
        header = lines[i]
        # header like @@ -l,s +l2,s2 @@ optional text
        try:
            # Extract -l and +l2
            import re
            m = re.search(r"@@ -([0-9]+)(?:,[0-9]+)? \+([0-9]+)(?:,[0-9]+)? @@", header)
            if not m:
                break
            orig_start = int(m.group(1))
            # new_start = int(m.group(2))
        except Exception:
            break
        i += 1
        body: List[str] = []
        while i < len(lines) and not lines[i].startswith('@@'):
            body.append(lines[i])
            i += 1
        hunks.append({"orig_start": orig_start, "body": body})
    return hunks


def apply_unified_diff_to_text(original: str, diff_text: str) -> Dict[str, Any]:
    """Apply a simple unified diff to original text.
    Supports basic hunks with context, '-' removals, '+' additions. Returns {success, text?, error?}.
    """
    orig_lines = original.splitlines()
    new_lines: List[str] = []
    idx = 0  # current index in orig_lines
    hunks = _parse_unified_hunks(diff_text)
    for h in hunks:
        orig_start = max(1, int(h.get("orig_start", 1))) - 1
        # copy unchanged up to this hunk
        if orig_start > idx:
            new_lines.extend(orig_lines[idx:orig_start])
            idx = orig_start
        for ln in h.get("body", []):
            if not ln:
                continue
            tag = ln[0]
            content = ln[1:]
            if tag == ' ':
                # context: consume one from original; append original line if available else append content
                if idx < len(orig_lines):
                    new_lines.append(orig_lines[idx])
                    idx += 1
                else:
                    new_lines.append(content)
            elif tag == '-':
                # removal: consume one from original
                if idx < len(orig_lines):
                    idx += 1
            elif tag == '+':
                # addition: append content
                new_lines.append(content)
            else:
                # unknown marker; append as-is
                new_lines.append(ln)
    # append remainder
    if idx < len(orig_lines):
        new_lines.extend(orig_lines[idx:])
    return {"success": True, "text": "\n".join(new_lines)}


def apply_resolved(edits: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Apply resolved edits where each item is {path, content? or unified_diff?}.
    If unified_diff provided, patch against current file content; if content provided, write directly.
    """
    results: List[Dict[str, Any]] = []
    for e in edits:
        path = e.get("path")
        if not path:
            results.append({"success": False, "error": "missing path"})
            continue
        p = Path(path)
        if e.get("content") is not None:
            data = str(e.get("content"))
        else:
            diff_text = str(e.get("unified_diff") or "")
            current = p.read_text(encoding="utf-8", errors="ignore") if p.exists() else ""
            patched = apply_unified_diff_to_text(current, diff_text)
            if not patched.get("success"):
                results.append({"path": str(p), "success": False, "error": patched.get("error", "patch failed")})
                continue
            data = patched.get("text", "")
        try:
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(data)
            results.append({"path": str(p), "success": True})
        except Exception as ex:
            results.append({"path": str(p), "success": False, "error": str(ex)})
    return {"success": all(r.get("success") for r in results), "results": results}
