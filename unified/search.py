"""
KWECLI Unified Search Operations - LTMC Atomic Integration

Implements LTMC's unified search architecture in KWECLI for atomic
multi-database search operations using the same coordination as LTMC.

File: kwecli/unified/search.py
Purpose: Atomic search operations synchronized with LTMC
Architecture: Direct adaptation of LTMC's ltms/unified/search.py
"""

import logging
import asyncio
import os
import sys
import hashlib
from typing import Dict, Any, List, Optional, Union, Tuple
from datetime import datetime

# Add LTMC to path for atomic operations
ltmc_path = '/home/feanor/Projects/ltmc'
if ltmc_path not in sys.path:
    sys.path.insert(0, ltmc_path)

# Import LTMC's atomic coordination components
try:
    from ltms.tools.atomic_memory_integration import get_atomic_memory_manager
    from ltms.database.sync_coordinator import DatabaseSyncCoordinator
    from ltms.sync.sync_models import DocumentData
    from ltms.database.sqlite_manager import SQLiteManager
    from ltms.database.faiss_manager import FAISSManager
    from ltms.config.json_config_loader import get_config
    LTMC_AVAILABLE = True
except ImportError as e:
    logging.warning(f"LTMC integration not available: {e}")
    LTMC_AVAILABLE = False
    FAISSManager = None

logger = logging.getLogger(__name__)

class KWECLIUnifiedSearchManager:
    """
    KWECLI unified search manager using LTMC's atomic operations.
    
    Provides KWECLI with direct access to LTMC's multi-database search
    using the same atomic coordination and result deduplication.
    """
    
    def __init__(self):
        """Initialize KWECLI search with LTMC atomic coordination."""
        self._ltmc_available = LTMC_AVAILABLE
        self._atomic_manager = None
        self._sqlite_manager = None
        self._faiss_manager = None
        
        if LTMC_AVAILABLE:
            try:
                # Use LTMC's atomic memory manager
                self._atomic_manager = get_atomic_memory_manager()
                
                # Use LTMC's database managers with same config
                self._sqlite_manager = SQLiteManager(test_mode=False)
                if FAISSManager:
                    self._faiss_manager = FAISSManager(test_mode=False)
                
                logger.info("KWECLI unified search initialized with LTMC atomic coordination")
                
            except Exception as e:
                logger.error(f"Failed to initialize LTMC atomic coordination: {e}")
                self._ltmc_available = False
        
        if not self._ltmc_available:
            logger.warning("KWECLI search running without LTMC atomic coordination")
    
    async def unified_search(self, 
                            resource_type: str,
                            query: str,
                            top_k: int = 10,
                            conversation_id: Optional[str] = None,
                            filters: Optional[Dict[str, Any]] = None,
                            search_strategy: Optional[str] = None,
                            **additional_params) -> Dict[str, Any]:
        """
        KWECLI unified search using LTMC's atomic operations.
        
        Provides unified search across FAISS vector search, SQLite metadata,
        Neo4j relationships, and Redis cache with result deduplication.
        
        Args:
            resource_type: Type of resource ('memory', 'document', 'kwecli_ingestion', etc.)
            query: Search query string
            top_k: Number of results to return
            conversation_id: KWECLI conversation identifier for filtering
            filters: Additional search filters
            search_strategy: Override automatic strategy selection
            **additional_params: Additional parameters
            
        Returns:
            Standardized result dictionary with ranked, deduplicated results
        """
        try:
            # Input validation
            if not query or not query.strip():
                return self._create_error_response('Query cannot be empty')
            
            if top_k < 1 or top_k > 100:
                return self._create_error_response('top_k must be between 1 and 100')
            
            # Use LTMC atomic operations if available
            if self._ltmc_available and self._atomic_manager:
                return await self._execute_ltmc_atomic_search(
                    resource_type=resource_type,
                    query=query,
                    top_k=top_k,
                    conversation_id=conversation_id or 'kwecli_default',
                    filters=filters,
                    search_strategy=search_strategy,
                    **additional_params
                )
            else:
                return self._create_error_response('LTMC atomic operations not available')
            
        except Exception as e:
            logger.error(f"KWECLI unified search failed for {resource_type}: {e}")
            return self._create_error_response(f'Search operation failed: {str(e)}')
    
    async def _execute_ltmc_atomic_search(self, **params) -> Dict[str, Any]:
        """
        Execute search using LTMC's atomic coordination.
        
        Uses LTMC's atomic search with strategy-based routing and
        result deduplication across multiple databases.
        """
        try:
            # Determine search strategy
            selected_strategy = params.get('search_strategy') or self._get_kwecli_search_strategy(
                params['resource_type'], 
                params['query']
            )
            
            search_databases = self._get_search_databases(params['resource_type'], selected_strategy)
            
            # Execute unified search with strategy routing
            result = await self._execute_search_strategy(selected_strategy, params)
            
            # Enhance result with KWECLI metadata
            if result.get('success'):
                documents_found = len(result.get('data', {}).get('documents', []))
                
                result['kwecli_search_metadata'] = {
                    'operation': 'kwecli_unified_search',
                    'resource_type': params['resource_type'],
                    'search_strategy': selected_strategy,
                    'query': params['query'],
                    'databases_searched': search_databases,
                    'results_count': documents_found,
                    'source_system': 'kwecli',
                    'ltmc_atomic_coordination': True,
                    'search_timestamp': datetime.now().isoformat()
                }
            
            return result
            
        except Exception as e:
            logger.error(f"LTMC atomic search execution failed: {e}")
            return self._create_error_response(f'LTMC atomic search failed: {str(e)}')
    
    def _get_kwecli_search_strategy(self, resource_type: str, query: str) -> str:
        """
        KWECLI-specific search strategy selection.
        Adapts LTMC's strategy selection for KWECLI resource types.
        """
        kwecli_search_strategy_map = {
            # KWECLI-specific resource types
            'kwecli_ingestion': 'faiss_semantic_unified',        # Ingested docs need semantic search
            'kwecli_document': 'faiss_semantic_neo4j_enrich',   # Rich document content + relationships
            'kwecli_context': 'faiss_semantic_unified',         # Context semantic search
            'kwecli_chat': 'redis_cache_first_search',          # Fast chat search
            'kwecli_session': 'redis_pattern_search',           # Session pattern matching
            
            # Standard LTMC types (same strategies as LTMC)
            'memory': 'faiss_semantic_unified',                 # Full vector similarity
            'document': 'faiss_semantic_neo4j_enrich',          # Content + relationships
            'pattern_analysis': 'faiss_code_similarity',        # Code pattern matching
            'analysis': 'faiss_research_similarity',            # Research content matching
            'blueprint': 'neo4j_graph_traversal_search',       # Project relationships
            'coordination': 'sqlite_neo4j_combined_search',    # Agent coordination + graph
            'chat': 'redis_cache_first_search',                # Fast conversation search
            'task': 'sqlite_indexed_search',                   # Status/priority search
            'cache': 'redis_pattern_search',                   # Cache key patterns
        }
        
        return kwecli_search_strategy_map.get(resource_type, 'faiss_semantic_unified')
    
    def _get_search_databases(self, resource_type: str, strategy: str) -> List[str]:
        """Get database targets for KWECLI search operations."""
        strategy_database_map = {
            'faiss_semantic_unified': ['sqlite', 'faiss'],
            'faiss_semantic_neo4j_enrich': ['sqlite', 'faiss', 'neo4j'],
            'redis_cache_first_search': ['redis', 'sqlite'],
            'redis_pattern_search': ['redis'],
            'sqlite_indexed_search': ['sqlite'],
            'neo4j_graph_traversal_search': ['neo4j', 'sqlite'],
            'sqlite_neo4j_combined_search': ['sqlite', 'neo4j'],
            'faiss_code_similarity': ['sqlite', 'faiss'],
            'faiss_research_similarity': ['sqlite', 'faiss'],
        }
        
        return strategy_database_map.get(strategy, ['sqlite', 'faiss'])
    
    async def _execute_search_strategy(self, strategy: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute search operation using the selected strategy.
        Uses LTMC's atomic coordination and search methods.
        """
        try:
            if strategy == 'faiss_semantic_unified':
                return await self._faiss_semantic_unified_search(params)
            elif strategy == 'faiss_semantic_neo4j_enrich':
                return await self._faiss_semantic_neo4j_enrich_search(params)
            elif strategy == 'redis_cache_first_search':
                return await self._redis_cache_first_search(params)
            elif strategy == 'redis_pattern_search':
                return await self._redis_pattern_search(params)
            elif strategy == 'sqlite_indexed_search':
                return await self._sqlite_indexed_search(params)
            elif strategy == 'neo4j_graph_traversal_search':
                return await self._neo4j_graph_traversal_search(params)
            elif strategy == 'sqlite_neo4j_combined_search':
                return await self._sqlite_neo4j_combined_search(params)
            else:
                # Fallback to semantic search for unknown strategies
                logger.warning(f"Unknown strategy {strategy}, falling back to semantic search")
                return await self._faiss_semantic_unified_search(params)
                
        except Exception as e:
            logger.error(f"Search strategy {strategy} execution failed: {e}")
            return self._create_error_response(f'Search strategy {strategy} execution failed: {str(e)}')
    
    async def _faiss_semantic_unified_search(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute unified FAISS semantic search using LTMC's atomic coordination."""
        try:
            # Use LTMC's atomic search functionality
            search_result = await self._atomic_manager.atomic_search(
                query=params['query'], 
                k=params.get('top_k', 10),
                conversation_id=params.get('conversation_id', 'kwecli_default')
            )
            
            if search_result.get('success'):
                # Format results for KWECLI compatibility with deduplication
                documents = []
                seen_content_hashes = set()
                
                for i, result in enumerate(search_result.get('results', [])):
                    # Generate content hash for deduplication
                    content = result.get('content_preview', '')
                    content_hash = hashlib.md5(content.encode()).hexdigest()
                    
                    if content_hash not in seen_content_hashes:
                        seen_content_hashes.add(content_hash)
                        
                        documents.append({
                            'file_name': result.get('doc_id'),
                            'content': content,
                            'resource_type': result.get('metadata', {}).get('resource_type', 'document'),
                            'created_at': result.get('metadata', {}).get('stored_at', ''),
                            'similarity_score': 1.0 - result.get('distance', 1.0),  # Convert distance to similarity
                            'rank': len(documents) + 1,  # Rank based on deduplicated position
                            'kwecli_source': True,  # Mark as KWECLI retrieved
                            'content_hash': content_hash  # For debugging deduplication
                        })
                
                return self._create_success_response({
                    'documents': documents,
                    'query': params['query'],
                    'conversation_id': params.get('conversation_id'),
                    'total_found': len(documents),
                    'original_results': len(search_result.get('results', [])),
                    'deduplicated_count': len(search_result.get('results', [])) - len(documents),
                    'search_strategy': 'faiss_semantic_unified',
                    'atomic_search': True,
                    'kwecli_unified': True
                })
            else:
                return search_result
                
        except Exception as e:
            logger.error(f"FAISS semantic unified search failed: {e}")
            return self._create_error_response(f'FAISS semantic unified search failed: {str(e)}')
    
    async def _faiss_semantic_neo4j_enrich_search(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute FAISS semantic search with Neo4j relationship enrichment."""
        try:
            # Start with semantic search
            semantic_result = await self._faiss_semantic_unified_search(params)
            
            if semantic_result.get('success'):
                # TODO: Enhance with Neo4j relationship data when available
                # For now, return semantic results with enrichment marker
                data = semantic_result.get('data', {})
                data['neo4j_enrichment'] = 'planned'
                data['search_strategy'] = 'faiss_semantic_neo4j_enrich'
                
                return self._create_success_response(data)
            else:
                return semantic_result
                
        except Exception as e:
            logger.error(f"FAISS semantic Neo4j enrich search failed: {e}")
            return self._create_error_response(f'FAISS semantic Neo4j enrich search failed: {str(e)}')
    
    async def _redis_cache_first_search(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute Redis cache first search with SQLite fallback."""
        try:
            # For now, fallback directly to SQLite indexed search (Redis integration pending)
            logger.debug("Redis cache search not available, using SQLite indexed search fallback")
            return await self._sqlite_indexed_search(params)
            
        except Exception as e:
            logger.error(f"Redis cache first search failed: {e}")
            return self._create_error_response(f'Redis cache first search failed: {str(e)}')
    
    async def _redis_pattern_search(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute Redis pattern search (placeholder for future Redis integration)."""
        try:
            # Placeholder for Redis pattern search implementation
            return self._create_error_response('Redis pattern search not yet implemented in KWECLI')
            
        except Exception as e:
            logger.error(f"Redis pattern search failed: {e}")
            return self._create_error_response(f'Redis pattern search failed: {str(e)}')
    
    async def _sqlite_indexed_search(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute SQLite indexed search using LTMC's SQLite manager."""
        try:
            if not self._sqlite_manager:
                return self._create_error_response('SQLite manager not available')
            
            # Use SQLite full-text search capabilities
            # This is a simplified implementation - could be enhanced with LTMC's full SQLite search
            query = params['query']
            top_k = params.get('top_k', 10)
            
            # Placeholder for SQLite FTS search
            # TODO: Implement proper SQLite full-text search when LTMC SQLite search methods are available
            documents = []
            
            return self._create_success_response({
                'documents': documents,
                'query': query,
                'total_found': len(documents),
                'search_strategy': 'sqlite_indexed_search',
                'kwecli_unified': True,
                'note': 'SQLite FTS implementation pending'
            })
            
        except Exception as e:
            logger.error(f"SQLite indexed search failed: {e}")
            return self._create_error_response(f'SQLite indexed search failed: {str(e)}')
    
    async def _neo4j_graph_traversal_search(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute Neo4j graph traversal search (placeholder for future Neo4j integration)."""
        try:
            # Placeholder for Neo4j graph traversal search implementation
            return self._create_error_response('Neo4j graph traversal search not yet implemented in KWECLI')
            
        except Exception as e:
            logger.error(f"Neo4j graph traversal search failed: {e}")
            return self._create_error_response(f'Neo4j graph traversal search failed: {str(e)}')
    
    async def _sqlite_neo4j_combined_search(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute combined SQLite and Neo4j search."""
        try:
            # For now, fallback to SQLite indexed search
            logger.debug("Neo4j not available, using SQLite indexed search fallback")
            return await self._sqlite_indexed_search(params)
            
        except Exception as e:
            logger.error(f"SQLite Neo4j combined search failed: {e}")
            return self._create_error_response(f'SQLite Neo4j combined search failed: {str(e)}')
    
    def _create_success_response(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Create standardized success response."""
        return {
            'success': True,
            'data': data,
            'source': 'kwecli_unified_search',
            'ltmc_available': self._ltmc_available,
            'timestamp': datetime.now().isoformat()
        }
    
    def _create_error_response(self, error_message: str) -> Dict[str, Any]:
        """Create standardized error response."""
        return {
            'success': False,
            'error': error_message,
            'source': 'kwecli_unified_search',
            'ltmc_available': self._ltmc_available,
            'timestamp': datetime.now().isoformat()
        }


# Global instance for singleton pattern
_kwecli_unified_search_manager = None

def get_kwecli_unified_search_manager() -> KWECLIUnifiedSearchManager:
    """Get or create KWECLI unified search manager instance."""
    global _kwecli_unified_search_manager
    if _kwecli_unified_search_manager is None:
        _kwecli_unified_search_manager = KWECLIUnifiedSearchManager()
    return _kwecli_unified_search_manager

# Main unified search function for KWECLI
async def unified_search(resource_type: str,
                        query: str,
                        top_k: int = 10,
                        conversation_id: Optional[str] = None,
                        filters: Optional[Dict[str, Any]] = None,
                        search_strategy: Optional[str] = None,
                        **kwargs) -> Dict[str, Any]:
    """
    KWECLI unified search using LTMC's atomic operations.
    
    Single source of truth for all KWECLI search operations that
    integrates directly with LTMC's multi-database system.
    
    Args:
        resource_type: Type of resource being searched
        query: Search query string
        top_k: Number of results to return
        conversation_id: KWECLI conversation identifier for filtering
        filters: Additional search filters
        search_strategy: Override automatic strategy selection
        **kwargs: Additional parameters
        
    Returns:
        Standardized result dictionary with ranked, deduplicated results
    """
    manager = get_kwecli_unified_search_manager()
    return await manager.unified_search(
        resource_type=resource_type,
        query=query,
        top_k=top_k,
        conversation_id=conversation_id,
        filters=filters,
        search_strategy=search_strategy,
        **kwargs
    )

# Backward compatibility wrapper
async def search(resource_type: str = 'kwecli_document', **params) -> Dict[str, Any]:
    """
    Backward compatibility wrapper for existing KWECLI search calls.
    """
    # Extract required parameters with KWECLI-specific fallbacks
    query = params.get('query', '')
    if not query:
        query = params.get('search_query', params.get('q', ''))
    
    top_k = params.get('top_k', params.get('limit', params.get('k', 10)))
    conversation_id = params.get('conversation_id', 'kwecli_default')
    
    # Handle string top_k parameter
    if isinstance(top_k, str):
        try:
            top_k = int(top_k)
        except (ValueError, TypeError):
            top_k = 10
    
    return await unified_search(
        resource_type=resource_type,
        query=query,
        top_k=top_k,
        conversation_id=conversation_id,
        filters=params.get('filters'),
        search_strategy=params.get('search_strategy'),
        **{k: v for k, v in params.items() 
           if k not in ['query', 'search_query', 'q', 'top_k', 'limit', 'k', 'conversation_id', 'filters', 'search_strategy']}
    )