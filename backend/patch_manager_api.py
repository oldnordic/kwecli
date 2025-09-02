#!/usr/bin/env python3
from __future__ import annotations

from fastapi import APIRouter, HTTPException
from typing import Any, Dict
from api_models import (
    PatchApplyBatchRequest,
    PatchApplyBatchResponse,
    PatchApplyResolvedRequest,
    PatchApplyResolvedResponse,
)

from .patch_manager import apply_batch, apply_resolved


router = APIRouter()


@router.post("/apply_batch", response_model=PatchApplyBatchResponse)
async def patch_apply_batch(payload: PatchApplyBatchRequest):
    try:
        edits = [e.dict() for e in payload.edits]
        dry_run = bool(payload.dry_run or False)
        allow_partial = bool(payload.allow_partial or False)
        if not isinstance(edits, list) or not edits:
            raise HTTPException(status_code=400, detail="edits required")
        return apply_batch(edits, dry_run=dry_run, allow_partial=allow_partial)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/apply_resolved", response_model=PatchApplyResolvedResponse)
async def patch_apply_resolved(payload: PatchApplyResolvedRequest):
    try:
        edits = [e.dict() for e in payload.edits]
        if not isinstance(edits, list) or not edits:
            raise HTTPException(status_code=400, detail="edits required")
        return apply_resolved(edits)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
