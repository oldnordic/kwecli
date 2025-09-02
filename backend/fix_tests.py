#!/usr/bin/env python3
from __future__ import annotations

import json
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from config.unified_config import get_config
from backend_core import KWECLIBackend


@dataclass
class TestRunResult:
    exit_code: int
    stdout: str
    stderr: str
    failing: List[str]


FAIL_RE = re.compile(r"^FAILED\s+([^\s:]+)(::[\w\[\]_:]+)?", re.M)


def parse_failures(output: str) -> List[str]:
    paths: List[str] = []
    for m in FAIL_RE.finditer(output):
        paths.append(m.group(1))
    # dedupe
    return sorted(set(paths))


def run_pytest(path: str = "tests") -> TestRunResult:
    proc = subprocess.run(
        ["python", "-m", "pytest", "-q", path], capture_output=True, text=True
    )
    failing = []
    if proc.returncode != 0:
        failing = parse_failures(proc.stdout + "\n" + proc.stderr)
    return TestRunResult(
        exit_code=proc.returncode, stdout=proc.stdout, stderr=proc.stderr, failing=failing
    )


def _apply_suggestions(backend: KWECLIBackend, suggestions: List[Dict[str, str]]) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    for s in suggestions:
        fp = s.get("file_path")
        content = s.get("new_content")
        if not fp or content is None:
            results.append({"file_path": fp, "success": False, "error": "invalid suggestion"})
            continue
        res = backend.agent_registry.get("write").run({"file_path": fp, "content": content, "create_dirs": True})
        results.append({"file_path": fp, **res})
    return results


def _suggest_patches_with_ollama(failing_files: List[str], context_note: str = "") -> List[Dict[str, str]]:
    cfg = get_config()
    if not cfg.use_ollama or not failing_files:
        return []
    try:
        import ollama  # type: ignore
    except Exception:
        return []
    prompt = (
        "You are fixing a Python project by making tests pass.\n"
        f"Failing files: {json.dumps(failing_files)}\n"
        "Return ONLY a compact JSON array of objects with exact keys: file_path, new_content.\n"
        "Do not include markdown or explanations. Provide full file content in new_content.\n"
    )
    if context_note:
        prompt += f"\nContext: {context_note}\n"
    try:
        res = ollama.generate(model=cfg.get_effective_model(), prompt=prompt)
        text = res.get("response") or res.get("output") or "[]"
        suggestions = json.loads(text)
        if isinstance(suggestions, list):
            ok: List[Dict[str, str]] = []
            for s in suggestions:
                if isinstance(s, dict) and "file_path" in s and "new_content" in s:
                    ok.append({"file_path": str(s["file_path"]), "new_content": str(s["new_content"])})
            return ok
    except Exception:
        return []
    return []


def iterate_fix_tests(backend: KWECLIBackend, path: str = "tests", max_iters: int = 3, context_note: str = "") -> Dict[str, Any]:
    iterations: List[Dict[str, Any]] = []
    cfg = get_config()
    for i in range(max_iters):
        run = run_pytest(path)
        iterations.append({
            "iter": i + 1,
            "exit_code": run.exit_code,
            "failing": run.failing,
            "stdout": run.stdout[-4000:],
        })
        if run.exit_code == 0:
            break
        # Try to suggest patches only if Ollama is enabled
        suggestions = _suggest_patches_with_ollama(run.failing, context_note=context_note)
        if not suggestions:
            break
        applied = _apply_suggestions(backend, suggestions)
        iterations[-1]["applied"] = applied
    return {"iterations": iterations, "success": iterations and iterations[-1].get("exit_code") == 0}

