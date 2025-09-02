#!/usr/bin/env python3
from __future__ import annotations

from fastapi import APIRouter, HTTPException
from typing import Any, Dict

from backend_core import KWECLIBackend
from config.unified_config import get_config
from .fix_tests import iterate_fix_tests


router = APIRouter()


@router.post("/iterate")
async def fix_tests_iterate_endpoint(payload: Dict[str, Any]):
    try:
        path = payload.get("path") or "tests"
        max_iters = int(payload.get("max_iters", 3))
        context_note = payload.get("context_note") or ""
        backend = KWECLIBackend(get_config())
        result = iterate_fix_tests(backend, path=path, max_iters=max_iters, context_note=context_note)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
