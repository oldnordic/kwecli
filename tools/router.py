"""Router for multi-source memory: documents, chats, code, todos, etc."""

from typing import Dict, Any, List, Optional, Literal
from enum import Enum

from tools.retrieve import retrieve_by_type, retrieve_all_types
from tools.ask import ask_by_type, ask_with_context


class SourceType(Enum):
    """Enum for different source types."""
    DOCUMENT = "document"
    CODE = "code"
    CHAT = "chat"
    TODO = "todo"


class MemoryRouter:
    """Router for multi-source memory operations."""
    
    def __init__(self):
        self.sources = [source.value for source in SourceType]
    
    def route_query(self, query: str, source_types: Optional[List[str]] = None, 
                   max_tokens: int = 4000, top_k: int = 5) -> Dict[str, Any]:
        """Route query to appropriate sources."""
        if source_types is None:
            source_types = self.sources
        
        # Validate source types
        valid_sources = [source.value for source in SourceType]
        invalid_sources = [s for s in source_types if s not in valid_sources]
        if invalid_sources:
            return {
                "success": False,
                "error": f"Invalid source types: {invalid_sources}",
                "valid_sources": valid_sources
            }
        
        # If single source type, use direct routing
        if len(source_types) == 1:
            return ask_by_type(query, source_types[0], max_tokens, top_k)
        
        # Multi-source routing
        return self._route_multi_source(query, source_types, max_tokens, top_k)
    
    def _route_multi_source(self, query: str, source_types: List[str], 
                           max_tokens: int, top_k: int) -> Dict[str, Any]:
        """Route query across multiple source types."""
        all_results = {}
        total_documents = 0
        
        # Get results from each source type
        for source_type in source_types:
            try:
                result = ask_by_type(query, source_type, max_tokens, top_k)
                all_results[source_type] = result
                if result.get("success"):
                    total_documents += result.get("documents_used", 0)
            except Exception as e:
                all_results[source_type] = {
                    "success": False,
                    "error": str(e),
                    "source_type": source_type
                }
        
        # Combine results
        successful_results = {k: v for k, v in all_results.items() 
                           if v.get("success")}
        
        if not successful_results:
            return {
                "success": False,
                "error": "No relevant documents found in any source",
                "query": query,
                "results": all_results
            }
        
        # Create combined response
        combined_answer = self._combine_answers(successful_results)
        
        return {
            "success": True,
            "query": query,
            "source_types": source_types,
            "combined_answer": combined_answer,
            "total_documents": total_documents,
            "results_by_source": all_results,
            "successful_sources": list(successful_results.keys())
        }
    
    def _combine_answers(self, results: Dict[str, Any]) -> str:
        """Combine answers from multiple sources."""
        combined_parts = []
        
        for source_type, result in results.items():
            answer = result.get("answer", "")
            context_summary = result.get("context_summary", "")
            
            combined_parts.append(f"=== {source_type.upper()} ===")
            combined_parts.append(context_summary)
            combined_parts.append(answer)
            combined_parts.append("")
        
        return "\n".join(combined_parts).strip()
    
    def get_available_sources(self) -> List[str]:
        """Get list of available source types."""
        return self.sources
    
    def route_by_priority(self, query: str, priority_order: List[str], 
                         max_tokens: int = 4000, top_k: int = 5) -> Dict[str, Any]:
        """Route query with priority order for sources."""
        # Validate priority order
        valid_sources = [source.value for source in SourceType]
        invalid_priorities = [s for s in priority_order if s not in valid_sources]
        if invalid_priorities:
            return {
                "success": False,
                "error": f"Invalid priority sources: {invalid_priorities}",
                "valid_sources": valid_sources
            }
        
        # Try sources in priority order
        for source_type in priority_order:
            result = ask_by_type(query, source_type, max_tokens, top_k)
            if result.get("success") and result.get("documents_used", 0) > 0:
                result["priority_source"] = source_type
                return result
        
        # If no priority sources have results, try all sources
        return self.route_query(query, max_tokens=max_tokens, top_k=top_k)


# Convenience functions
def route_query(query: str, source_types: Optional[List[str]] = None, 
               max_tokens: int = 4000, top_k: int = 5) -> Dict[str, Any]:
    """Convenience function to route a query."""
    router = MemoryRouter()
    return router.route_query(query, source_types, max_tokens, top_k)


def route_by_priority(query: str, priority_order: List[str], 
                    max_tokens: int = 4000, top_k: int = 5) -> Dict[str, Any]:
    """Convenience function to route with priority."""
    router = MemoryRouter()
    return router.route_by_priority(query, priority_order, max_tokens, top_k)
