#!/usr/bin/env python3
from __future__ import annotations

from fastapi import APIRouter, HTTPException
from typing import Any, Dict, List

from .docs_tools import read_docs, append_todos
from api_models import DocsLoadResponse, DocsTodoAppendResponse


router = APIRouter()


@router.get("/load", response_model=DocsLoadResponse)
async def docs_load():
    try:
        return read_docs()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/todo/append", response_model=DocsTodoAppendResponse)
async def docs_todo_append(payload: Dict[str, Any]):
    try:
        lines = payload.get("lines") or []
        if not isinstance(lines, list) or not lines:
            raise HTTPException(status_code=400, detail="lines required")
        return append_todos([str(x) for x in lines])
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
