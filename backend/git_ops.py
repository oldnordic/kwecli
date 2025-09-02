#!/usr/bin/env python3
from __future__ import annotations

import subprocess
from pathlib import Path
from typing import List, Dict, Any, Optional


def _run(cmd: List[str], cwd: Optional[str] = None, timeout: int = 120) -> Dict[str, Any]:
    try:
        proc = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=timeout)
        return {
            "exit_code": proc.returncode,
            "stdout": proc.stdout,
            "stderr": proc.stderr,
        }
    except Exception as e:
        return {"exit_code": -1, "stdout": "", "stderr": str(e)}


def repo_root(start: str = ".") -> str:
    res = _run(["git", "rev-parse", "--show-toplevel"], cwd=start)
    if res["exit_code"] == 0:
        return res["stdout"].strip()
    return str(Path(start).resolve())


def status(cwd: str = ".") -> Dict[str, Any]:
    out = _run(["git", "status", "--porcelain"], cwd=cwd)
    if out["exit_code"] != 0:
        return {"success": False, "error": out["stderr"], "stdout": out["stdout"]}
    files: List[Dict[str, str]] = []
    for line in out["stdout"].splitlines():
        if not line:
            continue
        # Format: XY <path>
        st = line[:2]
        path = line[3:]
        files.append({"status": st.strip(), "path": path})
    return {"success": True, "files": files, "root": repo_root(cwd)}


def diff(cwd: str = ".", staged: bool = False) -> Dict[str, Any]:
    args = ["git", "diff"] + (["--cached"] if staged else [])
    out = _run(args, cwd=cwd)
    if out["exit_code"] != 0:
        return {"success": False, "error": out["stderr"]}
    return {"success": True, "diff": out["stdout"]}


def add(paths: List[str], cwd: str = ".") -> Dict[str, Any]:
    if not paths:
        return {"success": False, "error": "no paths provided"}
    out = _run(["git", "add"] + paths, cwd=cwd)
    if out["exit_code"] != 0:
        return {"success": False, "error": out["stderr"]}
    return {"success": True}


def commit(message: str, cwd: str = ".") -> Dict[str, Any]:
    if not message:
        return {"success": False, "error": "empty commit message"}
    out = _run(["git", "commit", "-m", message], cwd=cwd)
    if out["exit_code"] != 0:
        return {"success": False, "error": out["stderr"], "stdout": out["stdout"]}
    return {"success": True, "stdout": out["stdout"]}


def branches(cwd: str = ".") -> Dict[str, Any]:
    out = _run(["git", "branch", "--list"], cwd=cwd)
    if out["exit_code"] != 0:
        return {"success": False, "error": out["stderr"]}
    names: List[Dict[str, Any]] = []
    for line in out["stdout"].splitlines():
        current = line.startswith("*")
        name = line[2:] if current else line.strip()
        names.append({"name": name.strip(), "current": current})
    return {"success": True, "branches": names}


def checkout(name: str, cwd: str = ".", create: bool = False) -> Dict[str, Any]:
    if not name:
        return {"success": False, "error": "branch name required"}
    args = ["git", "checkout"] + (["-b"] if create else []) + [name]
    out = _run(args, cwd=cwd)
    if out["exit_code"] != 0:
        return {"success": False, "error": out["stderr"], "stdout": out["stdout"]}
    return {"success": True, "stdout": out["stdout"]}

