#!/usr/bin/env python3
from __future__ import annotations

import threading
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

from .indexer import CodeIndexer, DEFAULT_INDEX_FILE

_supported = True
try:
    from watchdog.observers import Observer  # type: ignore
    from watchdog.events import PatternMatchingEventHandler  # type: ignore
except Exception:
    _supported = False


class _IndexHandler(PatternMatchingEventHandler):
    def __init__(self, root: Path, store: Path):
        super().__init__(ignore_directories=False)
        self.root = root
        self.store = store
        self._lock = threading.Lock()

    def on_any_event(self, event):
        with self._lock:
            idx = CodeIndexer(self.root, self.store)
            idx.build()


_observers: Dict[Tuple[str, str], Observer] = {}


def start_watch(root: str = ".", store_path: Optional[str] = None) -> Dict[str, Any]:
    if not _supported:
        return {"success": False, "supported": False, "error": "watchdog not installed"}
    r = str(Path(root).resolve())
    s = str(Path(store_path).resolve()) if store_path else str(DEFAULT_INDEX_FILE.resolve())
    key = (r, s)
    if key in _observers:
        return {"success": True, "supported": True, "message": "already running"}
    obs = Observer()
    handler = _IndexHandler(Path(r), Path(s))
    obs.schedule(handler, r, recursive=True)
    obs.daemon = True
    obs.start()
    _observers[key] = obs
    return {"success": True, "supported": True, "running": True, "root": r, "store_path": s}


def stop_watch(root: str = ".", store_path: Optional[str] = None) -> Dict[str, Any]:
    if not _supported:
        return {"success": False, "supported": False, "error": "watchdog not installed"}
    r = str(Path(root).resolve())
    s = str(Path(store_path).resolve()) if store_path else str(DEFAULT_INDEX_FILE.resolve())
    key = (r, s)
    obs = _observers.pop(key, None)
    if obs is not None:
        obs.stop()
        obs.join(timeout=2)
        return {"success": True, "supported": True, "running": False}
    return {"success": True, "supported": True, "running": False, "message": "not running"}


def watch_status() -> Dict[str, Any]:
    running = [
        {"root": k[0], "store_path": k[1], "running": True}
        for k in list(_observers.keys())
    ]
    return {"supported": _supported, "watchers": running}

