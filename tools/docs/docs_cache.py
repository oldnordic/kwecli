#!/usr/bin/env python3
"""
Local Docs Cache and Optional RAG Support.

Provides functions to ingest local documentation and query it using LlamaIndex
if available and enabled via config. No network access required at runtime.
"""

from __future__ import annotations

from typing import Optional
from pathlib import Path

from config.unified_config import get_config

_INDEX = None


def _rag_available() -> bool:
    cfg = get_config()
    if not cfg.rag_enabled:
        return False
    try:
        import llama_index  # noqa: F401
        return True
    except Exception:
        return False


def ingest_docs(path: str | Path) -> dict:
    """Ingest local docs into an in-memory index (if RAG enabled)."""
    cfg = get_config()
    if not cfg.rag_enabled:
        return {"success": False, "error": "RAG disabled in configuration", "status": 501}

    try:
        from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
    except Exception as e:
        return {"success": False, "error": f"llama-index not installed: {e}", "status": 501}

    docs_path = Path(path or cfg.docs_cache_path)
    if not docs_path.exists() or not docs_path.is_dir():
        return {"success": False, "error": f"docs path not found: {docs_path}", "status": 400}

    documents = SimpleDirectoryReader(str(docs_path)).load_data()
    global _INDEX
    _INDEX = VectorStoreIndex.from_documents(documents)
    return {"success": True, "documents": len(documents)}


def query_docs(question: str) -> dict:
    """Query the docs index; returns text answer if available."""
    cfg = get_config()
    if not cfg.rag_enabled:
        return {"success": False, "error": "RAG disabled in configuration", "status": 501}
    if not _INDEX:
        return {"success": False, "error": "Docs index not built. Call ingest first.", "status": 400}

    try:
        query_engine = _INDEX.as_query_engine()
        resp = query_engine.query(question)
        text = getattr(resp, "response", None) or str(resp)
        return {"success": True, "answer": text}
    except Exception as e:
        return {"success": False, "error": str(e), "status": 500}

