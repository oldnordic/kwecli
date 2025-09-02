"""
KWECLI Unified Operations Module - LTMC Atomic Operations Integration

This module implements LTMC's atomic operations architecture in KWECLI,
ensuring synchronized multi-database operations across SQLite, FAISS, Neo4j, and Redis.

Files:
- store.py: Unified storage operations with atomic coordination
- retrieve.py: Unified retrieval operations with strategy-based routing  
- search.py: Unified search operations with multi-database integration

Architecture:
- Follows LTMC's unified operations pattern exactly
- Uses same database paths and configuration as LTMC
- Provides atomic coordination and Mind Graph tracking
- Eliminates KWECLI's custom storage implementations
"""

from .store import unified_store, store
from .retrieve import unified_retrieve, retrieve
from .search import unified_search, search

__all__ = [
    'unified_store', 'store',
    'unified_retrieve', 'retrieve', 
    'unified_search', 'search'
]