"""
KWECLI Unified Operations Manager - LTMC Integration Hub

Central manager that coordinates KWECLI's unified storage, retrieval, and search
operations using LTMC's atomic operations architecture.

File: kwecli/unified/manager.py
Purpose: Central coordination hub for KWECLI-LTMC integration
Architecture: Singleton pattern with atomic operations coordination
"""

import logging
import asyncio
import os
import sys
from typing import Dict, Any, List, Optional
from datetime import datetime

# Import KWECLI unified operations
from .store import get_kwecli_unified_storage_manager, unified_store
from .retrieve import get_kwecli_unified_retrieval_manager, unified_retrieve
from .search import get_kwecli_unified_search_manager, unified_search

# Add LTMC to path for status checking
ltmc_path = '/home/feanor/Projects/ltmc'
if ltmc_path not in sys.path:
    sys.path.insert(0, ltmc_path)

try:
    from ltms.config.json_config_loader import get_config
    from ltms.tools.atomic_memory_integration import get_atomic_memory_manager
    LTMC_AVAILABLE = True
except ImportError:
    LTMC_AVAILABLE = False

logger = logging.getLogger(__name__)

class KWECLIUnifiedOperationsManager:
    """
    Central manager for KWECLI's unified operations with LTMC integration.
    
    Provides a single interface for all KWECLI storage, retrieval, and search
    operations while ensuring atomic coordination with LTMC's database system.
    """
    
    def __init__(self):
        """Initialize unified operations manager."""
        self._ltmc_available = LTMC_AVAILABLE
        self._storage_manager = None
        self._retrieval_manager = None
        self._search_manager = None
        self._atomic_manager = None
        
        # Initialize component managers
        self._initialize_managers()
        
        logger.info(f"KWECLI Unified Operations Manager initialized (LTMC available: {self._ltmc_available})")
    
    def _initialize_managers(self):
        """Initialize all component managers."""
        try:
            # Initialize storage manager
            self._storage_manager = get_kwecli_unified_storage_manager()
            
            # Initialize retrieval manager
            self._retrieval_manager = get_kwecli_unified_retrieval_manager()
            
            # Initialize search manager
            self._search_manager = get_kwecli_unified_search_manager()
            
            # Initialize LTMC atomic manager if available
            if LTMC_AVAILABLE:
                self._atomic_manager = get_atomic_memory_manager()
            
        except Exception as e:
            logger.error(f"Failed to initialize component managers: {e}")
    
    async def store_document(self, 
                           content: str,
                           file_name: str,
                           resource_type: str = 'kwecli_document',
                           tags: Optional[List[str]] = None,
                           metadata: Optional[Dict[str, Any]] = None,
                           conversation_id: str = 'kwecli_default') -> Dict[str, Any]:
        """
        Store a document using KWECLI's unified storage operations.
        
        Args:
            content: Document content
            file_name: Document filename/identifier
            resource_type: Type of resource being stored
            tags: Optional document tags
            metadata: Optional metadata dictionary
            conversation_id: KWECLI conversation identifier
            
        Returns:
            Standardized result dictionary
        """
        try:
            if not self._storage_manager:
                return self._create_error_response('Storage manager not initialized')
            
            result = await self._storage_manager.unified_store(
                resource_type=resource_type,
                content=content,
                file_name=file_name,
                metadata=metadata,
                conversation_id=conversation_id,
                tags=tags
            )
            
            # Add manager metadata
            if result.get('success'):
                result['manager_metadata'] = {
                    'operation': 'unified_store_document',
                    'manager_version': '1.0',
                    'ltmc_integration': self._ltmc_available
                }
            
            return result
            
        except Exception as e:
            logger.error(f"Store document operation failed: {e}")
            return self._create_error_response(f'Store document failed: {str(e)}')
    
    async def retrieve_documents(self, 
                               query: Optional[str] = None,
                               doc_id: Optional[str] = None,
                               resource_type: str = 'kwecli_document',
                               top_k: int = 10,
                               conversation_id: Optional[str] = None,
                               filters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Retrieve documents using KWECLI's unified retrieval operations.
        
        Args:
            query: Search query for semantic retrieval
            doc_id: Specific document ID for exact retrieval
            resource_type: Type of resource being retrieved
            top_k: Number of results to return
            conversation_id: KWECLI conversation identifier
            filters: Additional retrieval filters
            
        Returns:
            Standardized result dictionary with documents
        """
        try:
            if not self._retrieval_manager:
                return self._create_error_response('Retrieval manager not initialized')
            
            result = await self._retrieval_manager.unified_retrieve(
                resource_type=resource_type,
                query=query,
                doc_id=doc_id,
                conversation_id=conversation_id,
                top_k=top_k,
                filters=filters
            )
            
            # Add manager metadata
            if result.get('success'):
                result['manager_metadata'] = {
                    'operation': 'unified_retrieve_documents',
                    'manager_version': '1.0',
                    'ltmc_integration': self._ltmc_available
                }
            
            return result
            
        except Exception as e:
            logger.error(f"Retrieve documents operation failed: {e}")
            return self._create_error_response(f'Retrieve documents failed: {str(e)}')
    
    async def search_documents(self, 
                             query: str,
                             resource_type: str = 'kwecli_document',
                             top_k: int = 10,
                             conversation_id: Optional[str] = None,
                             filters: Optional[Dict[str, Any]] = None,
                             search_strategy: Optional[str] = None) -> Dict[str, Any]:
        """
        Search documents using KWECLI's unified search operations.
        
        Args:
            query: Search query string
            resource_type: Type of resource being searched
            top_k: Number of results to return
            conversation_id: KWECLI conversation identifier
            filters: Additional search filters
            search_strategy: Override automatic strategy selection
            
        Returns:
            Standardized result dictionary with ranked, deduplicated results
        """
        try:
            if not self._search_manager:
                return self._create_error_response('Search manager not initialized')
            
            result = await self._search_manager.unified_search(
                resource_type=resource_type,
                query=query,
                top_k=top_k,
                conversation_id=conversation_id,
                filters=filters,
                search_strategy=search_strategy
            )
            
            # Add manager metadata
            if result.get('success'):
                result['manager_metadata'] = {
                    'operation': 'unified_search_documents',
                    'manager_version': '1.0',
                    'ltmc_integration': self._ltmc_available
                }
            
            return result
            
        except Exception as e:
            logger.error(f"Search documents operation failed: {e}")
            return self._create_error_response(f'Search documents failed: {str(e)}')
    
    async def ingest_documents_batch(self, 
                                   documents: List[Dict[str, Any]],
                                   resource_type: str = 'kwecli_ingestion',
                                   conversation_id: str = 'kwecli_ingestion') -> Dict[str, Any]:
        """
        Ingest multiple documents in batch using unified operations.
        
        Args:
            documents: List of document dictionaries with 'content', 'file_name', and optional 'metadata', 'tags'
            resource_type: Type of resource being ingested
            conversation_id: KWECLI conversation identifier
            
        Returns:
            Batch ingestion results with success/failure counts
        """
        try:
            if not self._storage_manager:
                return self._create_error_response('Storage manager not initialized')
            
            results = []
            successful_count = 0
            failed_count = 0
            
            for i, doc in enumerate(documents):
                try:
                    # Extract document information
                    content = doc.get('content', '')
                    file_name = doc.get('file_name', f'document_{i+1}')
                    metadata = doc.get('metadata', {})
                    tags = doc.get('tags', [])
                    
                    if not content:
                        failed_count += 1
                        results.append({
                            'file_name': file_name,
                            'success': False,
                            'error': 'Empty content'
                        })
                        continue
                    
                    # Add batch ingestion metadata
                    batch_metadata = {
                        **metadata,
                        'batch_ingestion': True,
                        'batch_index': i,
                        'batch_size': len(documents),
                        'batch_timestamp': datetime.now().isoformat()
                    }
                    
                    # Store document
                    result = await self._storage_manager.unified_store(
                        resource_type=resource_type,
                        content=content,
                        file_name=file_name,
                        metadata=batch_metadata,
                        conversation_id=conversation_id,
                        tags=tags
                    )
                    
                    if result.get('success'):
                        successful_count += 1
                    else:
                        failed_count += 1
                    
                    results.append({
                        'file_name': file_name,
                        'success': result.get('success', False),
                        'error': result.get('error') if not result.get('success') else None
                    })
                    
                except Exception as e:
                    failed_count += 1
                    results.append({
                        'file_name': doc.get('file_name', f'document_{i+1}'),
                        'success': False,
                        'error': str(e)
                    })
            
            return {
                'success': True,
                'batch_results': {
                    'total_documents': len(documents),
                    'successful_count': successful_count,
                    'failed_count': failed_count,
                    'success_rate': successful_count / len(documents) if documents else 0,
                    'results': results
                },
                'manager_metadata': {
                    'operation': 'batch_ingest_documents',
                    'manager_version': '1.0',
                    'ltmc_integration': self._ltmc_available,
                    'resource_type': resource_type,
                    'conversation_id': conversation_id
                },
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Batch document ingestion failed: {e}")
            return self._create_error_response(f'Batch ingestion failed: {str(e)}')
    
    async def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status for KWECLI unified operations."""
        try:
            status = {
                'kwecli_unified_operations': {
                    'available': True,
                    'version': '1.0',
                    'managers_initialized': {
                        'storage': self._storage_manager is not None,
                        'retrieval': self._retrieval_manager is not None,
                        'search': self._search_manager is not None,
                    }
                },
                'ltmc_integration': {
                    'available': self._ltmc_available,
                    'atomic_manager': self._atomic_manager is not None,
                },
                'database_status': {},
                'timestamp': datetime.now().isoformat()
            }
            
            # Get LTMC configuration status if available
            if LTMC_AVAILABLE:
                try:
                    config = get_config()
                    status['ltmc_configuration'] = {
                        'db_path': config.db_path,
                        'faiss_index_path': config.faiss_index_path,
                        'embedding_model': config.embedding_model,
                        'vector_dimension': config.vector_dimension,
                    }
                except Exception as e:
                    status['ltmc_configuration'] = {'error': str(e)}
            
            return {
                'success': True,
                'data': status,
                'source': 'kwecli_unified_operations_manager'
            }
            
        except Exception as e:
            logger.error(f"Get system status failed: {e}")
            return self._create_error_response(f'System status check failed: {str(e)}')
    
    def _create_error_response(self, error_message: str) -> Dict[str, Any]:
        """Create standardized error response."""
        return {
            'success': False,
            'error': error_message,
            'source': 'kwecli_unified_operations_manager',
            'ltmc_available': self._ltmc_available,
            'timestamp': datetime.now().isoformat()
        }


# Global singleton instance
_kwecli_unified_operations_manager = None

def get_kwecli_unified_operations_manager() -> KWECLIUnifiedOperationsManager:
    """Get or create KWECLI unified operations manager singleton instance."""
    global _kwecli_unified_operations_manager
    if _kwecli_unified_operations_manager is None:
        _kwecli_unified_operations_manager = KWECLIUnifiedOperationsManager()
    return _kwecli_unified_operations_manager

# Convenience functions for common operations
async def store_document(content: str, file_name: str, **kwargs) -> Dict[str, Any]:
    """Convenience function for document storage."""
    manager = get_kwecli_unified_operations_manager()
    return await manager.store_document(content=content, file_name=file_name, **kwargs)

async def retrieve_documents(query: Optional[str] = None, doc_id: Optional[str] = None, **kwargs) -> Dict[str, Any]:
    """Convenience function for document retrieval."""
    manager = get_kwecli_unified_operations_manager()
    return await manager.retrieve_documents(query=query, doc_id=doc_id, **kwargs)

async def search_documents(query: str, **kwargs) -> Dict[str, Any]:
    """Convenience function for document search."""
    manager = get_kwecli_unified_operations_manager()
    return await manager.search_documents(query=query, **kwargs)

async def ingest_documents_batch(documents: List[Dict[str, Any]], **kwargs) -> Dict[str, Any]:
    """Convenience function for batch document ingestion."""
    manager = get_kwecli_unified_operations_manager()
    return await manager.ingest_documents_batch(documents=documents, **kwargs)

async def get_system_status() -> Dict[str, Any]:
    """Convenience function for system status."""
    manager = get_kwecli_unified_operations_manager()
    return await manager.get_system_status()