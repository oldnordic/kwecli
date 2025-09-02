#!/usr/bin/env python3
from __future__ import annotations

from fastapi import APIRouter, HTTPException
from typing import Any, Dict, Optional

from endpoint_tool_integration import get_endpoint_integration
from api_models import ShellSuggestResponse, ShellExplainResponse


router = APIRouter()


@router.post("/suggest", response_model=ShellSuggestResponse)
async def shell_suggest(payload: Dict[str, Any]):
    try:
        goal = (payload.get("goal") or "").strip()
        cwd = (payload.get("cwd") or ".").strip()
        if not goal:
            raise HTTPException(status_code=400, detail="goal required")
        prompt = (
            "You are a helpful CLI assistant. Suggest a single shell command to achieve the goal.\n"
            "- Output ONLY the command, no explanations.\n"
            f"- Goal: {goal}\n"
            f"- CWD: {cwd}\n"
        )
        integ = get_endpoint_integration()
        res = await integ.process_chat_with_tools(prompt, {"mode": "shell_suggest", "cwd": cwd})
        cmd = (res.get("message") or "").strip().splitlines()[0]
        return {"command": cmd}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/explain", response_model=ShellExplainResponse)
async def shell_explain(payload: Dict[str, Any]):
    try:
        cmd = (payload.get("command") or "").strip()
        if not cmd:
            raise HTTPException(status_code=400, detail="command required")
        prompt = (
            "Explain this shell command briefly and safely. Include flags meaning.\n"
            f"Command: {cmd}\n"
        )
        integ = get_endpoint_integration()
        res = await integ.process_chat_with_tools(prompt, {"mode": "shell_explain"})
        return {"explanation": res.get("message") or ""}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

