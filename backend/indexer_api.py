#!/usr/bin/env python3
from __future__ import annotations

from fastapi import APIRouter, HTTPException
from typing import Any, Dict, Optional
from pathlib import Path

from .indexer import CodeIndexer, DEFAULT_INDEX_FILE
from api_models import (
    IndexBuildResponse,
    IndexSearchResponse,
    IndexFileSymbolsResponse,
    IndexStatusResponse,
)
from .indexer_watcher import start_watch, stop_watch, watch_status


router = APIRouter()


def _mk_indexer(payload: Dict[str, Any]) -> CodeIndexer:
    root = payload.get("root") or "."
    store = payload.get("store_path")
    return CodeIndexer(Path(root), Path(store) if store else DEFAULT_INDEX_FILE)


@router.post("/build", response_model=IndexBuildResponse)
async def index_build(payload: Dict[str, Any]):
    try:
        idx = _mk_indexer(payload)
        return idx.build()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/search", response_model=IndexSearchResponse)
async def index_search(query: str, kind: Optional[str] = None, path_glob: Optional[str] = None, root: Optional[str] = None, store_path: Optional[str] = None):
    try:
        idx = CodeIndexer(Path(root or "."), Path(store_path) if store_path else DEFAULT_INDEX_FILE)
        return {"results": idx.search_symbols(query, kind=kind, path_glob=path_glob)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/file_symbols", response_model=IndexFileSymbolsResponse)
async def index_file_symbols(file_path: str, root: Optional[str] = None, store_path: Optional[str] = None):
    try:
        idx = CodeIndexer(Path(root or "."), Path(store_path) if store_path else DEFAULT_INDEX_FILE)
        return {"symbols": idx.file_symbols(file_path)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/status", response_model=IndexStatusResponse)
async def index_status(root: Optional[str] = None, store_path: Optional[str] = None):
    try:
        idx = CodeIndexer(Path(root or "."), Path(store_path) if store_path else DEFAULT_INDEX_FILE)
        data = idx.data or {}
        return {"root": data.get("root"), "symbols": len(data.get("symbols", [])), "files": len(data.get("files", []))}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Watcher controls (optional; requires watchdog)
@router.post("/watch/start")
async def index_watch_start(payload: Dict[str, Any]):
    try:
        root = payload.get("root") or "."
        store = payload.get("store_path")
        return start_watch(root, store)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/watch/stop")
async def index_watch_stop(payload: Dict[str, Any]):
    try:
        root = payload.get("root") or "."
        store = payload.get("store_path")
        return stop_watch(root, store)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/watch/status")
async def index_watch_status():
    try:
        return watch_status()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
