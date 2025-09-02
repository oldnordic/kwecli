"""
KWECLI Unified Storage Operations - LTMC Atomic Integration

Implements LTMC's unified storage architecture in KWECLI for atomic
multi-database operations. Uses same databases and coordination as LTMC.

File: kwecli/unified/store.py
Purpose: Atomic storage operations synchronized with LTMC
Architecture: Direct adaptation of LTMC's ltms/unified/store.py
"""

import logging
import asyncio
import os
import sys
from typing import Dict, Any, List, Optional
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
    from ltms.config.json_config_loader import get_config
    LTMC_AVAILABLE = True
except ImportError as e:
    logging.warning(f"LTMC integration not available: {e}")
    LTMC_AVAILABLE = False

logger = logging.getLogger(__name__)

class KWECLIUnifiedStorageManager:
    """
    KWECLI unified storage manager using LTMC's atomic operations.
    
    Provides KWECLI with direct access to LTMC's multi-database system
    using the same atomic coordination and database synchronization.
    """
    
    def __init__(self):
        """Initialize KWECLI storage with LTMC atomic coordination."""
        self._ltmc_available = LTMC_AVAILABLE
        self._atomic_manager = None
        self._sqlite_manager = None
        
        if LTMC_AVAILABLE:
            try:
                # Use LTMC's atomic memory manager
                self._atomic_manager = get_atomic_memory_manager()
                
                # Use LTMC's SQLite manager with same config
                self._sqlite_manager = SQLiteManager(test_mode=False)
                
                logger.info("KWECLI unified storage initialized with LTMC atomic coordination")
                
            except Exception as e:
                logger.error(f"Failed to initialize LTMC atomic coordination: {e}")
                self._ltmc_available = False
        
        if not self._ltmc_available:
            logger.warning("KWECLI storage running without LTMC atomic coordination")
    
    async def unified_store(self, 
                           resource_type: str,
                           content: str,
                           file_name: str,
                           metadata: Optional[Dict[str, Any]] = None,
                           conversation_id: str = 'kwecli_default',
                           tags: Optional[List[str]] = None,
                           **additional_params) -> Dict[str, Any]:
        """
        KWECLI unified storage using LTMC's atomic operations.
        
        Stores documents using LTMC's multi-database coordination with
        atomic transactions across SQLite, FAISS, Neo4j, and Redis.
        
        Args:
            resource_type: Type of resource ('memory', 'document', 'kwecli_ingestion', etc.)
            content: Document content to store
            file_name: Document filename/identifier
            metadata: Optional metadata dictionary
            conversation_id: KWECLI conversation identifier
            tags: Optional list of document tags
            **additional_params: Additional parameters
            
        Returns:
            Standardized result dictionary with success status and details
        """
        try:
            # Input validation
            if not content:
                return self._create_error_response('Content cannot be empty')
            if not file_name:
                return self._create_error_response('File name cannot be empty')
            
            # Use LTMC atomic operations if available
            if self._ltmc_available and self._atomic_manager:
                return await self._execute_ltmc_atomic_storage(
                    resource_type=resource_type,
                    content=content,
                    file_name=file_name,
                    metadata=metadata,
                    conversation_id=conversation_id,
                    tags=tags,
                    **additional_params
                )
            else:
                return self._create_error_response('LTMC atomic operations not available')
            
        except Exception as e:
            logger.error(f"KWECLI unified storage failed for '{file_name}': {e}")
            return self._create_error_response(f'Storage operation failed: {str(e)}')
    
    async def _execute_ltmc_atomic_storage(self, **params) -> Dict[str, Any]:
        """
        Execute storage using LTMC's atomic coordination.
        
        Uses LTMC's atomic_store_with_tiered_priority for proper
        multi-database synchronization.
        """
        try:
            # Prepare unified metadata with KWECLI context
            unified_metadata = {
                'resource_type': params['resource_type'],
                'conversation_id': params['conversation_id'],
                'storage_strategy': self._get_kwecli_storage_strategy(params['resource_type']),
                'kwecli_unified_version': '1.0',
                'source_system': 'kwecli',
                'ingestion_timestamp': datetime.now().isoformat(),
                **(params.get('metadata') or {}),
                **{k: v for k, v in params.items() 
                   if k not in ['resource_type', 'content', 'file_name', 'conversation_id', 'tags', 'metadata']}
            }
            
            # Normalize tags for KWECLI context
            normalized_tags = self._normalize_kwecli_tags(params.get('tags'), params['resource_type'])
            
            # Extract parameters that are already named arguments to avoid conflicts
            metadata_filtered = {k: v for k, v in unified_metadata.items() 
                                if k not in ['resource_type', 'conversation_id']}
            
            # Use LTMC's atomic storage with tiered priority
            result = await self._atomic_manager.atomic_store_with_tiered_priority(
                file_name=params['file_name'],
                content=params['content'],
                resource_type=params['resource_type'],
                tags=normalized_tags,
                conversation_id=params['conversation_id'],
                **metadata_filtered
            )
            
            # Enhance result with KWECLI metadata
            if result.get('success'):
                result['kwecli_storage_metadata'] = {
                    'operation': 'kwecli_unified_store',
                    'resource_type': params['resource_type'],
                    'storage_strategy': unified_metadata['storage_strategy'],
                    'source_system': 'kwecli',
                    'ltmc_atomic_coordination': True,
                    'database_targets': self._get_database_targets(params['resource_type']),
                    'ingestion_timestamp': unified_metadata['ingestion_timestamp']
                }
            
            return result
            
        except Exception as e:
            logger.error(f"LTMC atomic storage execution failed: {e}")
            return self._create_error_response(f'LTMC atomic storage failed: {str(e)}')
    
    def _get_kwecli_storage_strategy(self, resource_type: str) -> str:
        """Get optimal storage strategy for KWECLI resource types."""
        kwecli_strategy_map = {
            # KWECLI-specific resource types
            'kwecli_ingestion': 'full_semantic_search',  # Document ingestion needs full search
            'kwecli_chat': 'fast_retrieval_optimized',   # Chat needs fast access
            'kwecli_session': 'stateful_with_backup',    # Session state preservation
            'kwecli_document': 'content_relationship_enriched',  # Rich document storage
            'kwecli_context': 'full_semantic_search',    # Context needs semantic search
            
            # Standard LTMC types (same as LTMC)
            'memory': 'full_semantic_search',
            'document': 'content_relationship_enriched',
            'chat': 'fast_retrieval_optimized',
            'task': 'status_optimized',
            'blueprint': 'relationship_focused',
        }
        
        return kwecli_strategy_map.get(resource_type, 'balanced_full_integration')
    
    def _get_database_targets(self, resource_type: str) -> List[str]:
        """Resource-type-aware database routing for KWECLI operations."""
        kwecli_routing_map = {
            # KWECLI-specific routing for document ingestion
            'kwecli_ingestion': ['sqlite', 'faiss', 'neo4j'],  # Full integration for ingestion
            'kwecli_document': ['sqlite', 'faiss', 'neo4j'],   # Rich document storage
            'kwecli_context': ['sqlite', 'faiss'],             # Context with semantic search
            'kwecli_chat': ['sqlite', 'redis'],                # Fast chat retrieval
            'kwecli_session': ['redis', 'sqlite'],             # Session with backup
            
            # Standard LTMC routing (same as LTMC)
            'memory': ['sqlite', 'faiss', 'neo4j', 'redis'],
            'document': ['sqlite', 'faiss', 'neo4j'],
            'chat': ['sqlite', 'redis'],
            'task': ['sqlite', 'redis'],
            'blueprint': ['sqlite', 'neo4j'],
        }
        
        # Default to full integration for unknown types
        return kwecli_routing_map.get(resource_type, ['sqlite', 'faiss', 'neo4j', 'redis'])
    
    def _normalize_kwecli_tags(self, tags: Optional[Any], resource_type: str) -> List[str]:
        """Normalize tags for KWECLI context."""
        base_tags = ['kwecli']  # Always tag with KWECLI source
        
        # Add resource type tag
        base_tags.append(f"kwecli_{resource_type}")
        
        # Process input tags
        if tags:
            if isinstance(tags, str):
                base_tags.extend([tag.strip() for tag in tags.split(',') if tag.strip()])
            elif isinstance(tags, list):
                base_tags.extend([str(tag).strip() for tag in tags if tag])
        
        # Remove duplicates and return
        return list(set(base_tags))
    
    def _create_error_response(self, error_message: str) -> Dict[str, Any]:
        """Create standardized error response."""
        return {
            'success': False,
            'error': error_message,
            'source': 'kwecli_unified_store',
            'ltmc_available': self._ltmc_available,
            'timestamp': datetime.now().isoformat()
        }


# Global instance for singleton pattern
_kwecli_unified_storage_manager = None

def get_kwecli_unified_storage_manager() -> KWECLIUnifiedStorageManager:
    """Get or create KWECLI unified storage manager instance."""
    global _kwecli_unified_storage_manager
    if _kwecli_unified_storage_manager is None:
        _kwecli_unified_storage_manager = KWECLIUnifiedStorageManager()
    return _kwecli_unified_storage_manager

# Main unified store function for KWECLI
async def unified_store(resource_type: str,
                       content: str, 
                       file_name: str,
                       metadata: Optional[Dict[str, Any]] = None,
                       conversation_id: str = 'kwecli_default',
                       tags: Optional[List[str]] = None,
                       **kwargs) -> Dict[str, Any]:
    """
    KWECLI unified storage using LTMC's atomic operations.
    
    Single source of truth for all KWECLI storage operations that
    integrates directly with LTMC's multi-database system.
    
    Args:
        resource_type: Type of resource being stored
        content: Document content  
        file_name: Document filename/identifier
        metadata: Optional metadata dictionary
        conversation_id: KWECLI conversation identifier
        tags: Optional list of document tags
        **kwargs: Additional parameters
        
    Returns:
        Standardized result dictionary with success status and details
    """
    manager = get_kwecli_unified_storage_manager()
    return await manager.unified_store(
        resource_type=resource_type,
        content=content,
        file_name=file_name,
        metadata=metadata,
        conversation_id=conversation_id,
        tags=tags,
        **kwargs
    )

# Backward compatibility wrapper
async def store(resource_type: str = 'kwecli_document', **params) -> Dict[str, Any]:
    """
    Backward compatibility wrapper for existing KWECLI store calls.
    """
    # Extract required parameters with KWECLI-specific fallbacks
    content = params.get('content', '')
    file_name = params.get('file_name') or params.get('doc_id') or 'kwecli_unnamed_document'
    conversation_id = params.get('conversation_id', 'kwecli_default')
    
    return await unified_store(
        resource_type=resource_type,
        content=content,
        file_name=file_name,
        metadata=params.get('metadata'),
        conversation_id=conversation_id,
        tags=params.get('tags'),
        **{k: v for k, v in params.items() 
           if k not in ['content', 'file_name', 'doc_id', 'metadata', 'conversation_id', 'tags']}
    )