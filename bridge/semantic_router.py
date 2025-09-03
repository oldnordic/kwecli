#!/usr/bin/env python3
"""
KWECLI Semantic Query Router - Native LTMC Integration
======================================================

Main semantic router coordinator using modular architecture.
Routes queries to optimal database combinations using KWECLI's native bridge.

File: bridge/semantic_router.py  
Purpose: Main routing coordinator with native LTMC integration (≤300 lines)
"""

import time
import json
import logging
import hashlib
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

from .ltmc_native import get_ltmc_native, save_thought, log_artifact
from .semantic_router_core import SemanticAnalyzer, QueryType, DatabaseType, QueryContext

logger = logging.getLogger(__name__)


@dataclass  
class RoutingDecision:
    """Structured routing decision with performance tracking."""
    primary_db: DatabaseType
    secondary_dbs: List[DatabaseType]
    tool_sequence: List[str]
    confidence_score: float
    estimated_time_ms: int
    reasoning: str
    cache_key: Optional[str] = None


class SemanticRouter:
    """
    KWECLI Native Semantic Router - Routes queries using native LTMC integration.
    Uses modular architecture with SemanticAnalyzer for core analysis.
    """
    
    def __init__(self):
        """Initialize semantic router with native LTMC and modular analyzer."""
        self.ltmc = get_ltmc_native()
        self.analyzer = SemanticAnalyzer()
        self.sla_target_ms = 500
        
        # Database routing strategies based on query types
        self.db_strategies = {
            QueryType.CODE: [DatabaseType.SQLITE, DatabaseType.FAISS],
            QueryType.DOCUMENTATION: [DatabaseType.FAISS, DatabaseType.NEO4J],
            QueryType.PLANNING: [DatabaseType.SQLITE, DatabaseType.NEO4J, DatabaseType.REDIS],
            QueryType.RESEARCH: [DatabaseType.FAISS, DatabaseType.NEO4J],
            QueryType.DEBUGGING: [DatabaseType.SQLITE, DatabaseType.REDIS],
            QueryType.ARCHITECTURE: [DatabaseType.NEO4J, DatabaseType.FAISS],
            QueryType.MEMORY_RETRIEVAL: [DatabaseType.REDIS, DatabaseType.SQLITE],
            QueryType.PATTERN_ANALYSIS: [DatabaseType.FAISS, DatabaseType.NEO4J]
        }
        
        logger.info("KWECLI Semantic Router initialized with native LTMC integration")
    
    def route_query(self, query: str, context: Optional[Dict[str, Any]] = None) -> RoutingDecision:
        """
        Route query to optimal database combination using native KWECLI analysis.
        
        Args:
            query: Input query string
            context: Optional context dictionary
            
        Returns:
            RoutingDecision with optimal database routing
        """
        start_time = time.time()
        
        try:
            # Step 1: Analyze query using modular analyzer
            query_context = self.analyzer.analyze_query(query, context)
            
            # Step 2: Check cache
            cache_key = self._generate_cache_key(query, context)
            cached_decision = self._check_cache(cache_key)
            if cached_decision:
                logger.info(f"Cache hit: {cache_key[:8]}...")
                return cached_decision
            
            # Step 3: Make routing decision
            decision = self._make_routing_decision(query_context)
            decision.cache_key = cache_key
            
            # Step 4: Store and track
            self._store_decision(decision, query_context)
            elapsed_ms = int((time.time() - start_time) * 1000)
            self._track_performance(decision, elapsed_ms)
            
            logger.info(f"Routed in {elapsed_ms}ms: {decision.primary_db.value} + {len(decision.secondary_dbs)} secondary")
            return decision
            
        except Exception as e:
            logger.error(f"Routing failed: {e}")
            return self._create_fallback_decision(query, cache_key, str(e))
    
    def _make_routing_decision(self, query_context: QueryContext) -> RoutingDecision:
        """Make intelligent routing decision based on analysis."""
        # Get base strategy
        base_dbs = self.db_strategies.get(query_context.query_type, [DatabaseType.SQLITE])
        
        # Adapt based on complexity and confidence
        if query_context.complexity_score >= 7 and query_context.confidence >= 0.8:
            primary_db = base_dbs[0]
            secondary_dbs = base_dbs[1:] + [DatabaseType.REDIS]
        elif query_context.complexity_score >= 4:
            primary_db = base_dbs[0]
            secondary_dbs = base_dbs[1:2]
        else:
            primary_db = base_dbs[0]
            secondary_dbs = []
        
        # Generate tool sequence and estimate time
        tools = self._generate_tool_sequence(primary_db, secondary_dbs, query_context)
        estimated_time = self._estimate_time(tools, query_context.complexity_score)
        
        reasoning = f"Native: {query_context.query_type.value} (conf: {query_context.confidence:.2f}, complex: {query_context.complexity_score}/10)"
        
        return RoutingDecision(
            primary_db=primary_db,
            secondary_dbs=secondary_dbs,
            tool_sequence=tools,
            confidence_score=query_context.confidence,
            estimated_time_ms=estimated_time,
            reasoning=reasoning
        )
    
    def _generate_tool_sequence(self, primary_db: DatabaseType, secondary_dbs: List[DatabaseType], 
                              query_context: QueryContext) -> List[str]:
        """Generate native tool sequence."""
        sequence = ["ltmc_native"]
        
        # Map databases to native operations
        db_ops = {
            DatabaseType.SQLITE: "save_thought",
            DatabaseType.FAISS: "retrieve_similar", 
            DatabaseType.NEO4J: "log_artifact",
            DatabaseType.REDIS: "health_check"
        }
        
        # Add operations
        primary_op = db_ops.get(primary_db, "save_thought")
        if primary_op not in sequence:
            sequence.append(primary_op)
        
        for db in secondary_dbs:
            op = db_ops.get(db, "save_thought")
            if op not in sequence:
                sequence.append(op)
        
        # Add coordination for multi-DB
        if len(secondary_dbs) > 0:
            sequence.append("native_sync")
            
        return sequence
    
    def _estimate_time(self, tools: List[str], complexity: int) -> int:
        """Estimate execution time for native operations."""
        times = {
            "ltmc_native": 50, "save_thought": 100, "retrieve_similar": 150,
            "log_artifact": 75, "health_check": 25, "native_sync": 50
        }
        
        base = sum(times.get(tool, 100) for tool in tools)
        factor = 1 + (complexity * 0.15)
        return min(int(base * factor), self.sla_target_ms)
    
    def _generate_cache_key(self, query: str, context: Optional[Dict] = None) -> str:
        """Generate cache key."""
        cache_input = query
        if context:
            cache_input += json.dumps(context, sort_keys=True)
        return hashlib.md5(cache_input.encode()).hexdigest()
    
    def _check_cache(self, cache_key: str) -> Optional[RoutingDecision]:
        """Check native cache."""
        try:
            cached = self.ltmc.storage.retrieve_by_metadata({
                "cache_key": cache_key, "type": "routing_cache"
            })
            
            if cached and len(cached) > 0:
                data = cached[0]
                return RoutingDecision(
                    primary_db=DatabaseType(data.get("primary_db", "sqlite")),
                    secondary_dbs=[DatabaseType(db) for db in data.get("secondary_dbs", [])],
                    tool_sequence=data.get("tool_sequence", ["ltmc_native"]),
                    confidence_score=data.get("confidence_score", 0.7),
                    estimated_time_ms=data.get("estimated_time_ms", 300),
                    reasoning=data.get("reasoning", "Cached"),
                    cache_key=cache_key
                )
        except Exception:
            pass
        return None
    
    def _store_decision(self, decision: RoutingDecision, query_context: QueryContext):
        """Store decision using native storage."""
        try:
            save_thought(
                kind="routing_decision",
                content=f"Routed {query_context.query_type.value} to {decision.primary_db.value}",
                metadata={
                    "cache_key": decision.cache_key,
                    "primary_db": decision.primary_db.value,
                    "secondary_dbs": [db.value for db in decision.secondary_dbs],
                    "confidence_score": decision.confidence_score,
                    "query_type": query_context.query_type.value,
                    "timestamp": time.time()
                }
            )
        except Exception:
            pass
    
    def _track_performance(self, decision: RoutingDecision, actual_time_ms: int):
        """Track performance using native logging."""
        try:
            save_thought(
                kind="performance_metric",
                content=f"Routing: {actual_time_ms}ms",
                metadata={
                    "routing_performance": {
                        "primary_db": decision.primary_db.value,
                        "actual_ms": actual_time_ms,
                        "within_sla": actual_time_ms <= self.sla_target_ms
                    }
                }
            )
        except Exception:
            pass
    
    def _create_fallback_decision(self, query: str, cache_key: str, error: str) -> RoutingDecision:
        """Create safe fallback decision."""
        return RoutingDecision(
            primary_db=DatabaseType.SQLITE,
            secondary_dbs=[],
            tool_sequence=["ltmc_native", "save_thought"],
            confidence_score=0.3,
            estimated_time_ms=200,
            reasoning=f"Fallback: {error}",
            cache_key=cache_key
        )
    
    def learn_from_success(self, decision: RoutingDecision, success_metrics: Dict[str, Any]):
        """Learn from successful routing."""
        try:
            save_thought(
                kind="routing_success",
                content=f"Success: {decision.primary_db.value}",
                metadata={"routing_decision": decision.primary_db.value, "success_metrics": success_metrics}
            )
            log_artifact("semantic_router", "success_learning", f"routing_{int(time.time())}")
        except Exception:
            pass
    
    def get_routing_statistics(self) -> Dict[str, Any]:
        """Get routing statistics."""
        try:
            health = self.ltmc.health_check()
            return {
                "native_system_healthy": health.get("healthy", False),
                "available_databases": sum(1 for status in health.get("connections", {}).values() if status),
                "sqlite_available": health.get("connections", {}).get("sqlite", False),
                "faiss_available": health.get("connections", {}).get("faiss", False),
                "neo4j_available": health.get("connections", {}).get("neo4j", False),
                "redis_available": health.get("connections", {}).get("redis", False),
                "data_directory": health.get("data_dir"),
                "last_updated": time.time()
            }
        except Exception as e:
            return {"error": str(e), "fallback": True}


# Convenience function  
def route_query(query: str, context: Optional[Dict[str, Any]] = None) -> RoutingDecision:
    """Convenience function for routing queries."""
    router = SemanticRouter()
    return router.route_query(query, context)


if __name__ == "__main__":
    router = SemanticRouter()
    decision = router.route_query("Test query")
    print(f"✅ Routed to: {decision.primary_db.value}")