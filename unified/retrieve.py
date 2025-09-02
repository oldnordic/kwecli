"""
KWECLI Unified Retrieval Operations - LTMC Atomic Integration

Implements LTMC's unified retrieval architecture in KWECLI for atomic
multi-database retrieval operations using the same coordination as LTMC.

File: kwecli/unified/retrieve.py
Purpose: Atomic retrieval operations synchronized with LTMC
Architecture: Direct adaptation of LTMC's ltms/unified/retrieve.py
"""

import logging
import asyncio
import os
import sys
from typing import Dict, Any, List, Optional, Union
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
    from ltms.services.context_service import retrieve_by_type
    from ltms.config.json_config_loader import get_config
    LTMC_AVAILABLE = True
except ImportError as e:
    logging.warning(f"LTMC integration not available: {e}")
    LTMC_AVAILABLE = False

logger = logging.getLogger(__name__)

class KWECLIUnifiedRetrievalManager:
    """
    KWECLI unified retrieval manager using LTMC's atomic operations.
    
    Provides KWECLI with direct access to LTMC's multi-database retrieval
    using the same atomic coordination and strategy-based routing.
    """
    
    def __init__(self):
        """Initialize KWECLI retrieval with LTMC atomic coordination."""
        self._ltmc_available = LTMC_AVAILABLE
        self._atomic_manager = None
        self._sqlite_manager = None
        
        if LTMC_AVAILABLE:
            try:
                # Use LTMC's atomic memory manager
                self._atomic_manager = get_atomic_memory_manager()
                
                # Use LTMC's SQLite manager with same config
                self._sqlite_manager = SQLiteManager(test_mode=False)
                
                logger.info("KWECLI unified retrieval initialized with LTMC atomic coordination")
                
            except Exception as e:
                logger.error(f"Failed to initialize LTMC atomic coordination: {e}")
                self._ltmc_available = False
        
        if not self._ltmc_available:
            logger.warning("KWECLI retrieval running without LTMC atomic coordination")
    
    async def unified_retrieve(self, 
                              resource_type: str,
                              query: Optional[str] = None,
                              doc_id: Optional[str] = None,
                              conversation_id: Optional[str] = None,
                              top_k: int = 10,
                              filters: Optional[Dict[str, Any]] = None,
                              **additional_params) -> Dict[str, Any]:
        """
        KWECLI unified retrieval using LTMC's atomic operations.
        
        Retrieves documents using LTMC's multi-database coordination with
        strategy-based routing across SQLite, FAISS, Neo4j, and Redis.
        
        Args:
            resource_type: Type of resource ('memory', 'document', 'kwecli_ingestion', etc.)
            query: Search query (for semantic/vector search)
            doc_id: Specific document ID (for exact retrieval)
            conversation_id: KWECLI conversation identifier for filtering
            top_k: Number of results to return (for search operations)
            filters: Additional filters for complex queries
            **additional_params: Additional parameters
            
        Returns:
            Standardized result dictionary with documents and metadata
        """
        try:
            # Input validation and parameter normalization
            if not query and not doc_id:
                return self._create_error_response('Either query or doc_id must be provided')
            
            # Use LTMC atomic operations if available
            if self._ltmc_available and self._atomic_manager:
                return await self._execute_ltmc_atomic_retrieval(
                    resource_type=resource_type,
                    query=query,
                    doc_id=doc_id,
                    conversation_id=conversation_id,
                    top_k=top_k,
                    filters=filters,
                    **additional_params
                )
            else:
                return self._create_error_response('LTMC atomic operations not available')
            
        except Exception as e:
            logger.error(f"KWECLI unified retrieval failed for {resource_type}: {e}")
            return self._create_error_response(f'Retrieval operation failed: {str(e)}')
    
    async def _execute_ltmc_atomic_retrieval(self, **params) -> Dict[str, Any]:
        """
        Execute retrieval using LTMC's atomic coordination.
        
        Uses LTMC's atomic search and retrieval strategies for proper
        multi-database integration.
        """
        try:
            # Determine retrieval strategy
            retrieval_strategy = self._get_kwecli_retrieval_strategy(
                params['resource_type'], 
                params.get('query'), 
                params.get('doc_id')
            )
            
            # Execute retrieval based on strategy
            result = await self._execute_retrieval_strategy(retrieval_strategy, params)
            
            # Enhance result with KWECLI metadata
            if result.get('success'):
                documents_found = len(result.get('data', {}).get('documents', []))
                
                result['kwecli_retrieval_metadata'] = {
                    'operation': 'kwecli_unified_retrieve',
                    'resource_type': params['resource_type'],
                    'retrieval_strategy': retrieval_strategy,
                    'query': params.get('query'),
                    'doc_id': params.get('doc_id'),
                    'results_count': documents_found,
                    'source_system': 'kwecli',
                    'ltmc_atomic_coordination': True,
                    'retrieval_timestamp': datetime.now().isoformat()
                }
            
            return result
            
        except Exception as e:
            logger.error(f"LTMC atomic retrieval execution failed: {e}")
            return self._create_error_response(f'LTMC atomic retrieval failed: {str(e)}')
    
    def _get_kwecli_retrieval_strategy(self, resource_type: str, query: Optional[str], doc_id: Optional[str]) -> str:
        """
        KWECLI-specific retrieval strategy selection.
        Adapts LTMC's strategy selection for KWECLI resource types.
        """
        # Direct ID retrieval takes precedence
        if doc_id:
            if resource_type in ['kwecli_session', 'cache']:
                return 'redis_direct_lookup'
            else:
                return 'sqlite_exact_retrieval'
        
        # Query-based retrieval strategies for KWECLI resource types
        if query:
            kwecli_strategy_map = {
                # KWECLI-specific resource types
                'kwecli_ingestion': 'faiss_semantic_search',       # Ingested documents need semantic search
                'kwecli_document': 'faiss_semantic_neo4j_enrich',  # Rich document content + relationships
                'kwecli_context': 'faiss_semantic_search',         # Context semantic search
                'kwecli_chat': 'redis_cache_first_sqlite_fallback', # Fast chat retrieval
                'kwecli_session': 'redis_session_query',           # Session pattern matching
                
                # Standard LTMC types (same strategies as LTMC)
                'memory': 'faiss_semantic_search',
                'document': 'faiss_semantic_neo4j_enrich',
                'chat': 'redis_cache_first_sqlite_fallback',
                'task': 'sqlite_indexed_query',
                'blueprint': 'neo4j_graph_traversal',
                'pattern_analysis': 'faiss_code_similarity',
                'analysis': 'faiss_research_similarity',
            }
            
            return kwecli_strategy_map.get(resource_type, 'faiss_semantic_search')
        
        # Fallback for resource type without specific query/doc_id
        return 'sqlite_type_filtered_list'
    
    async def _execute_retrieval_strategy(self, strategy: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute retrieval operation using the selected strategy.
        Uses LTMC's atomic coordination and retrieval methods.
        """
        try:
            if strategy == 'faiss_semantic_search':
                return await self._faiss_semantic_search(params)
            elif strategy == 'sqlite_exact_retrieval':
                return await self._sqlite_exact_retrieval(params)
            elif strategy == 'redis_direct_lookup':
                return await self._redis_direct_lookup(params)
            elif strategy == 'faiss_semantic_neo4j_enrich':
                return await self._faiss_semantic_neo4j_enrich(params)
            elif strategy == 'sqlite_type_filtered_list':
                return await self._sqlite_type_filtered_list(params)
            elif strategy == 'redis_cache_first_sqlite_fallback':
                return await self._redis_cache_first_sqlite_fallback(params)
            else:
                # Fallback to semantic search for unknown strategies
                logger.warning(f"Unknown strategy {strategy}, falling back to semantic search")
                return await self._faiss_semantic_search(params)
                
        except Exception as e:
            logger.error(f"Retrieval strategy {strategy} execution failed: {e}")
            return self._create_error_response(f'Retrieval strategy {strategy} execution failed: {str(e)}')
    
    async def _faiss_semantic_search(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute FAISS semantic search using LTMC's atomic coordination."""
        try:
            # Use LTMC's atomic search functionality
            search_result = await self._atomic_manager.atomic_search(
                query=params['query'], 
                k=params.get('top_k', 10),
                conversation_id=params.get('conversation_id', 'kwecli_default')
            )
            
            if search_result.get('success'):
                # Format results for KWECLI compatibility
                documents = []
                for i, result in enumerate(search_result.get('results', [])):
                    documents.append({
                        'file_name': result.get('doc_id'),
                        'content': result.get('content_preview', ''),
                        'resource_type': result.get('metadata', {}).get('resource_type', 'document'),
                        'created_at': result.get('metadata', {}).get('stored_at', ''),
                        'similarity_score': 1.0 - result.get('distance', 1.0),  # Convert distance to similarity
                        'rank': i + 1,
                        'kwecli_source': True  # Mark as KWECLI retrieved
                    })
                
                return self._create_success_response({
                    'documents': documents,
                    'query': params['query'],
                    'conversation_id': params.get('conversation_id'),
                    'total_found': len(documents),
                    'retrieval_strategy': 'faiss_semantic_search',
                    'atomic_search': True,
                    'kwecli_unified': True
                })
            else:
                return search_result
                
        except Exception as e:
            logger.error(f"FAISS semantic search failed: {e}")
            return self._create_error_response(f'FAISS semantic search failed: {str(e)}')
    
    async def _sqlite_exact_retrieval(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute SQLite exact document retrieval by ID using LTMC's SQLite manager."""
        try:
            if not self._sqlite_manager:
                return self._create_error_response('SQLite manager not available')
            
            document = self._sqlite_manager.retrieve_document(params['doc_id'])
            
            if document:
                # Format as single document in list for compatibility
                documents = [{
                    'file_name': document.get('file_name', params['doc_id']),
                    'content': document.get('content', ''),
                    'resource_type': document.get('resource_type', params.get('resource_type', 'document')),
                    'created_at': document.get('created_at', ''),
                    'similarity_score': 1.0,  # Exact match
                    'rank': 1,
                    'kwecli_source': True  # Mark as KWECLI retrieved
                }]
                
                return self._create_success_response({
                    'documents': documents,
                    'doc_id': params['doc_id'],
                    'total_found': 1,
                    'retrieval_strategy': 'sqlite_exact_retrieval',
                    'exact_match': True,
                    'kwecli_unified': True
                })
            else:
                return self._create_success_response({
                    'documents': [],
                    'doc_id': params['doc_id'],
                    'total_found': 0,
                    'retrieval_strategy': 'sqlite_exact_retrieval',
                    'exact_match': False,
                    'kwecli_unified': True
                })
                
        except Exception as e:
            logger.error(f"SQLite exact retrieval failed: {e}")
            return self._create_error_response(f'SQLite exact retrieval failed: {str(e)}')
    
    async def _sqlite_type_filtered_list(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute SQLite type-filtered document listing using LTMC's context service."""
        try:
            if not LTMC_AVAILABLE:
                return self._create_error_response('Context service not available for type filtering')
            
            # Use LTMC's context service functionality
            result = retrieve_by_type(
                resource_type=params['resource_type'],
                project_id=params.get('project_id'),
                limit=params.get('top_k', 10),
                offset=params.get('offset', 0),
                date_range=params.get('date_range')
            )
            
            # Standardize response format for KWECLI
            if isinstance(result, dict) and result.get('success', True):
                documents = result.get('documents', [])
                
                # Mark documents as KWECLI retrieved
                for doc in documents:
                    doc['kwecli_source'] = True
                
                return self._create_success_response({
                    'documents': documents,
                    'resource_type': params['resource_type'],
                    'total_found': result.get('total_count', len(documents)),
                    'filtered_count': result.get('filtered_count', len(documents)),
                    'retrieval_strategy': 'sqlite_type_filtered_list',
                    'type_filtered': True,
                    'kwecli_unified': True
                })
            else:
                return self._create_error_response(result.get('error', 'Unknown error in retrieve_by_type'))
                
        except Exception as e:
            logger.error(f"SQLite type-filtered list failed: {e}")
            return self._create_error_response(f'SQLite type-filtered list failed: {str(e)}')
    
    async def _redis_direct_lookup(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute Redis direct key lookup (placeholder for future Redis integration)."""
        try:
            # Placeholder for Redis direct lookup implementation
            # TODO: Implement when Redis manager is available in KWECLI
            return self._create_error_response('Redis direct lookup not yet implemented in KWECLI')
            
        except Exception as e:
            logger.error(f"Redis direct lookup failed: {e}")
            return self._create_error_response(f'Redis direct lookup failed: {str(e)}')
    
    async def _faiss_semantic_neo4j_enrich(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute FAISS semantic search with Neo4j relationship enrichment."""
        try:
            # Start with semantic search
            semantic_result = await self._faiss_semantic_search(params)
            
            if semantic_result.get('success'):
                # TODO: Enhance with Neo4j relationship data when available
                # For now, return semantic results with enrichment marker
                data = semantic_result.get('data', {})
                data['neo4j_enrichment'] = 'planned'
                data['retrieval_strategy'] = 'faiss_semantic_neo4j_enrich'
                
                return self._create_success_response(data)
            else:
                return semantic_result
                
        except Exception as e:
            logger.error(f"FAISS semantic Neo4j enrich failed: {e}")
            return self._create_error_response(f'FAISS semantic Neo4j enrich failed: {str(e)}')
    
    async def _redis_cache_first_sqlite_fallback(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute Redis cache first with SQLite fallback."""
        try:
            # For now, fallback directly to SQLite (Redis integration pending)
            logger.debug("Redis cache not available, using SQLite fallback")
            return await self._sqlite_type_filtered_list(params)
            
        except Exception as e:
            logger.error(f"Redis cache first SQLite fallback failed: {e}")
            return self._create_error_response(f'Redis cache first SQLite fallback failed: {str(e)}')
    
    def _create_success_response(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Create standardized success response."""
        return {
            'success': True,
            'data': data,
            'source': 'kwecli_unified_retrieve',
            'ltmc_available': self._ltmc_available,
            'timestamp': datetime.now().isoformat()
        }
    
    def _create_error_response(self, error_message: str) -> Dict[str, Any]:
        """Create standardized error response."""
        return {
            'success': False,
            'error': error_message,
            'source': 'kwecli_unified_retrieve',
            'ltmc_available': self._ltmc_available,
            'timestamp': datetime.now().isoformat()
        }


# Global instance for singleton pattern
_kwecli_unified_retrieval_manager = None

def get_kwecli_unified_retrieval_manager() -> KWECLIUnifiedRetrievalManager:
    """Get or create KWECLI unified retrieval manager instance."""
    global _kwecli_unified_retrieval_manager
    if _kwecli_unified_retrieval_manager is None:
        _kwecli_unified_retrieval_manager = KWECLIUnifiedRetrievalManager()
    return _kwecli_unified_retrieval_manager

# Main unified retrieve function for KWECLI
async def unified_retrieve(resource_type: str,
                          query: Optional[str] = None,
                          doc_id: Optional[str] = None,
                          conversation_id: Optional[str] = None,
                          top_k: int = 10,
                          filters: Optional[Dict[str, Any]] = None,
                          **kwargs) -> Dict[str, Any]:
    """
    KWECLI unified retrieval using LTMC's atomic operations.
    
    Single source of truth for all KWECLI retrieval operations that
    integrates directly with LTMC's multi-database system.
    
    Args:
        resource_type: Type of resource being retrieved
        query: Search query for semantic/vector search
        doc_id: Specific document ID for exact retrieval
        conversation_id: KWECLI conversation identifier for filtering
        top_k: Number of results to return
        filters: Additional filters for complex queries
        **kwargs: Additional parameters
        
    Returns:
        Standardized result dictionary with documents and metadata
    """
    manager = get_kwecli_unified_retrieval_manager()
    return await manager.unified_retrieve(
        resource_type=resource_type,
        query=query,
        doc_id=doc_id,
        conversation_id=conversation_id,
        top_k=top_k,
        filters=filters,
        **kwargs
    )

# Backward compatibility wrapper
async def retrieve(resource_type: str = 'kwecli_document', **params) -> Dict[str, Any]:
    """
    Backward compatibility wrapper for existing KWECLI retrieve calls.
    """
    # Extract and normalize parameters with KWECLI-specific fallbacks
    query = params.get('query')
    doc_id = params.get('doc_id') or params.get('file_name')
    conversation_id = params.get('conversation_id', 'kwecli_default')
    top_k = params.get('top_k', params.get('k', 10))
    
    # Handle string top_k parameter
    if isinstance(top_k, str):
        try:
            top_k = int(top_k)
        except (ValueError, TypeError):
            top_k = 10
    
    return await unified_retrieve(
        resource_type=resource_type,
        query=query,
        doc_id=doc_id,
        conversation_id=conversation_id,
        top_k=top_k,
        filters=params.get('filters'),
        **{k: v for k, v in params.items() 
           if k not in ['query', 'doc_id', 'file_name', 'conversation_id', 'top_k', 'k', 'filters']}
    )