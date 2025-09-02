#!/usr/bin/env python3
from __future__ import annotations

from fastapi import APIRouter, HTTPException
from api_models import GenerateTestsSuggestResponse, GenerateTestsApplyResponse
from typing import Any, Dict, List
from pathlib import Path
import ast


router = APIRouter()


def _suggest_py_tests(target: Path) -> List[Dict[str, str]]:
    """Generate real smoke tests without placeholders.
    - Loads module from file path using importlib.
    - For zero-arg functions, calls them to ensure no exception.
    - For other functions, asserts they exist.
    """
    try:
        src = target.read_text(encoding="utf-8", errors="ignore")
    except Exception as e:
        return [{"path": str(target), "content": f"# error reading file: {e}"}]
    try:
        tree = ast.parse(src)
    except Exception as e:
        tree = ast.parse("")
    funcs: List[Dict[str, Any]] = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and not (node.name or "").startswith("_"):
            # count required positional-only and positional args without defaults
            total_pos = len(getattr(node.args, 'posonlyargs', [])) + len(node.args.args)
            defaults = len(node.args.defaults or [])
            required = max(total_pos - defaults, 0)
            funcs.append({"name": node.name, "required": required})
    test_name = f"test_{target.stem}.py"
    mod_loader = f"""
import importlib.util, pathlib

def _load_mod():
    p = pathlib.Path(r'{str(target)}')
    spec = importlib.util.spec_from_file_location('mod_under_test', str(p))
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(mod)
    return mod
""".strip()
    tests: List[str] = [mod_loader]
    tests.append("def test_module_load():\n    m = _load_mod()")
    if funcs:
        for f in funcs[:10]:
            name = f["name"]
            if f.get("required", 1) == 0:
                tests.append(f"def test_call_{name}():\n    m = _load_mod()\n    func = getattr(m, '{name}')\n    _ = func()")
            else:
                tests.append(f"def test_has_{name}():\n    m = _load_mod()\n    assert hasattr(m, '{name}')")
    content = "\n\n".join(tests) + "\n"
    tests_dir = Path("tests")
    tests_dir.mkdir(exist_ok=True)
    return [{"path": str(tests_dir / test_name), "content": content}]


@router.post("/suggest", response_model=GenerateTestsSuggestResponse)
async def generate_tests_suggest(payload: Dict[str, Any]):
    try:
        target = payload.get("target") or ""
        if not target:
            raise HTTPException(status_code=400, detail="target required")
        p = Path(target)
        if not p.exists():
            raise HTTPException(status_code=404, detail="target not found")
        if p.suffix == ".py":
            edits = _suggest_py_tests(p)
        else:
            # Minimal generic stub
            edits = [{"path": f"tests/test_{p.stem}.py", "content": "def test_smoke():\n    assert True\n"}]
        return {"edits": edits}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/apply", response_model=GenerateTestsApplyResponse)
async def generate_tests_apply(payload: Dict[str, Any]):
    try:
        edits = payload.get("edits") or []
        if not isinstance(edits, list) or not edits:
            raise HTTPException(status_code=400, detail="edits required")
        written: List[str] = []
        for e in edits:
            path = Path(str(e.get("path")))
            content = str(e.get("content") or "")
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")
            written.append(str(path))
        return {"success": True, "written": written}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
