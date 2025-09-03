#!/usr/bin/env python3
"""
KWECLI Retrieval Refinement System - Phase 3 Enhancement
=========================================================

Advanced retrieval refinement coordinator using modular strategy engine.
Uses retrieval_strategies_core for CLAUDE.md compliance (≤300 lines).

File: bridge/retrieval_refiner.py
Purpose: Retrieval refinement coordinator (≤300 lines)
"""

import time
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

try:
    from .bridge_core import NativeLTMCBridge
    from .semantic_router import SemanticRouter, RoutingDecision
    from .quality_evaluator import QualityEvaluator
    from .retrieval_strategies_core import (
        RetrievalStrategiesEngine, RetrievalStrategy, RetrievalResult
    )
except ImportError:
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from bridge_core import NativeLTMCBridge
    from semantic_router import SemanticRouter, RoutingDecision
    from quality_evaluator import QualityEvaluator
    from retrieval_strategies_core import (
        RetrievalStrategiesEngine, RetrievalStrategy, RetrievalResult
    )

logger = logging.getLogger(__name__)


@dataclass
class RefinedResults:
    """Refined retrieval results with performance metrics."""
    query: str
    strategy: RetrievalStrategy
    results: List[RetrievalResult]
    total_score: float
    processing_time_ms: int
    fusion_weights: Dict[str, float]
    quality_metrics: Dict[str, Any]


class RetrievalRefiner:
    """
    KWECLI Retrieval Refinement System - optimizes context selection and relevance.
    Uses modular strategies engine for core implementations.
    """
    
    def __init__(self, project_path: str = None):
        """Initialize retrieval refiner with native LTMC integration."""
        self.ltmc_bridge = NativeLTMCBridge()
        if not self.ltmc_bridge.initialize():
            logger.error("❌ Failed to initialize LTMC bridge for retrieval refiner")
            raise RuntimeError("Bridge initialization failed")
        self.semantic_router = SemanticRouter()
        self.quality_evaluator = QualityEvaluator(project_path)
        self.project_path = Path(project_path) if project_path else Path.cwd()
        
        # Fusion weights for multi-source retrieval
        self.fusion_weights = {
            "faiss_vector": 0.4,
            "sqlite_keyword": 0.3,
            "neo4j_graph": 0.2,
            "redis_cache": 0.1
        }
        
        # Initialize strategies engine
        self.strategies_engine = RetrievalStrategiesEngine(self.ltmc_bridge, self.fusion_weights)
        
        # Performance SLA targets
        self.sla_targets = {"total_refinement_ms": 5000}
        
        # Quality thresholds for result filtering
        self.quality_thresholds = {
            "min_relevance_score": 0.1,  # Lowered to accept real LTMC results
            "max_results": 10,
            "diversity_threshold": 0.8,
            "context_overlap_limit": 0.7
        }
        
        logger.info(f"Retrieval Refiner initialized for: {self.project_path}")
    
    def refine_retrieval(self, query: str, routing_decision: RoutingDecision = None,
                        strategy: RetrievalStrategy = RetrievalStrategy.FUSION,
                        max_results: int = 5) -> RefinedResults:
        """
        Main retrieval refinement with multi-source fusion and intelligent reranking.
        
        Args:
            query: Input query for retrieval
            routing_decision: Optional semantic routing decision
            strategy: Retrieval refinement strategy to use
            max_results: Maximum number of refined results to return
            
        Returns:
            RefinedResults with optimized context and performance metrics
        """
        start_time = time.time()
        
        try:
            # Get routing decision if not provided
            if not routing_decision:
                routing_decision = self.semantic_router.route_query(query)
            
            # Apply retrieval strategy using modular engine
            if strategy == RetrievalStrategy.HYDE:
                raw_results = self.strategies_engine.hyde_retrieval(query, routing_decision)
            elif strategy == RetrievalStrategy.FUSION:
                raw_results = self.strategies_engine.fusion_retrieval(query, routing_decision)
            elif strategy == RetrievalStrategy.RERANK:
                raw_results = self.strategies_engine.reranking_retrieval(query, routing_decision)
            elif strategy == RetrievalStrategy.CONTEXTUAL:
                raw_results = self.strategies_engine.contextual_retrieval(query, routing_decision)
            else:
                raw_results = self.strategies_engine.fusion_retrieval(query, routing_decision)  # Default
            
            # Apply quality filtering and deduplication
            filtered_results = self._apply_quality_filtering(raw_results, query)
            
            # Final reranking and selection
            final_results = self._final_reranking(filtered_results, query, max_results)
            
            # Calculate performance metrics
            processing_time = int((time.time() - start_time) * 1000)
            quality_metrics = self._calculate_quality_metrics(final_results, query)
            
            # Create refined results
            refined = RefinedResults(
                query=query,
                strategy=strategy,
                results=final_results,
                total_score=sum(r.score for r in final_results) / len(final_results) if final_results else 0,
                processing_time_ms=processing_time,
                fusion_weights=self.fusion_weights.copy(),
                quality_metrics=quality_metrics
            )
            
            # Store refinement results for learning
            self._store_refinement_results(refined)
            
            logger.info(f"Retrieval refined in {processing_time}ms: {len(final_results)} results, avg score: {refined.total_score:.2f}")
            return refined
            
        except Exception as e:
            logger.error(f"Retrieval refinement failed: {e}")
            return self._create_fallback_results(query, strategy, str(e))
    
    def get_adaptive_strategy(self, query: str, routing_decision: RoutingDecision) -> RetrievalStrategy:
        """Select optimal retrieval strategy based on query analysis and routing."""
        try:
            query_words = len(query.split())
            has_tech_terms = any(term in query.lower() for term in ["implement", "code", "function", "class", "api"])
            
            if query_words > 10 and has_tech_terms:
                return RetrievalStrategy.HYDE
            elif len(routing_decision.secondary_dbs) >= 2:
                return RetrievalStrategy.FUSION
            elif routing_decision.confidence_score >= 0.8:
                return RetrievalStrategy.RERANK
            else:
                return RetrievalStrategy.CONTEXTUAL
        except Exception:
            return RetrievalStrategy.FUSION
    
    def batch_refine_retrieval(self, queries: List[str], 
                              strategy: RetrievalStrategy = None) -> List[RefinedResults]:
        results = []
        for query in queries:
            # Use adaptive strategy if none specified
            if strategy is None:
                routing_decision = self.semantic_router.route_query(query)
                adaptive_strategy = self.get_adaptive_strategy(query, routing_decision)
                refined = self.refine_retrieval(query, routing_decision, adaptive_strategy)
            else:
                refined = self.refine_retrieval(query, strategy=strategy)
            results.append(refined)
        return results
    
    def _apply_quality_filtering(self, results: List[RetrievalResult], query: str) -> List[RetrievalResult]:
        """Apply quality filtering and deduplication."""
        if not results:
            return []
        
        # Filter by minimum relevance score
        filtered = [r for r in results if r.score >= self.quality_thresholds["min_relevance_score"]]
        
        # Remove duplicates based on content similarity
        deduplicated = []
        for result in filtered:
            is_duplicate = False
            for existing in deduplicated:
                if self._calculate_similarity(result.content, existing.content) > self.quality_thresholds["context_overlap_limit"]:
                    is_duplicate = True
                    break
            if not is_duplicate:
                deduplicated.append(result)
        
        # Sort and limit results
        sorted_results = sorted(deduplicated, key=lambda x: x.score, reverse=True)
        return sorted_results[:self.quality_thresholds["max_results"]]
    
    def _calculate_similarity(self, content1: str, content2: str) -> float:
        """Calculate content similarity for deduplication."""
        words1 = set(content1.lower().split())
        words2 = set(content2.lower().split())
        if not words1 or not words2:
            return 0.0
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        return intersection / union if union > 0 else 0.0
    
    def _final_reranking(self, results: List[RetrievalResult], query: str, max_results: int) -> List[RetrievalResult]:
        """Apply final reranking and selection."""
        if not results:
            return []
        
        # Apply diversity scoring to avoid redundant results
        diversified_results = []
        for i, result in enumerate(results):
            diversity_penalty = i * 0.02  # Small penalty for later results
            result.score *= (1.0 - diversity_penalty)
            diversified_results.append(result)
        
        # Final selection
        return sorted(diversified_results, key=lambda x: x.score, reverse=True)[:max_results]
    
    def _calculate_quality_metrics(self, results: List[RetrievalResult], query: str) -> Dict[str, Any]:
        """Calculate quality metrics for refinement results."""
        if not results:
            return {"result_count": 0, "avg_score": 0, "source_diversity": 0, "quality_passed": False}
        
        return {
            "result_count": len(results),
            "avg_score": sum(r.score for r in results) / len(results),
            "source_diversity": len(set(r.source for r in results)),
            "top_score": max(r.score for r in results),
            "quality_passed": len(results) > 0 and max(r.score for r in results) > 0.5
        }
    
    def _store_refinement_results(self, refined: RefinedResults):
        """Store refinement results in LTMC for learning."""
        try:
            self.ltmc_bridge.memory_store(
                kind="retrieval_refinement",
                content=f"Refined: {refined.strategy.value} - {refined.total_score:.2f}",
                metadata={
                    "query": refined.query,
                    "strategy": refined.strategy.value,
                    "performance": refined.processing_time_ms,
                    "quality_metrics": refined.quality_metrics
                }
            )
        except Exception as e:
            logger.debug(f"Could not store refinement results: {e}")
    
    def _create_fallback_results(self, query: str, strategy: RetrievalStrategy, error: str) -> RefinedResults:
        """Create fallback results for errors."""
        return RefinedResults(
            query=query,
            strategy=strategy,
            results=[],
            total_score=0.0,
            processing_time_ms=0,
            fusion_weights={},
            quality_metrics={"error": error}
        )
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get retrieval refiner performance metrics."""
        return {"fusion_weights": self.fusion_weights, "strategies": [s.value for s in RetrievalStrategy]}


# Convenience function
def refine_query_retrieval(query: str, project_path: str = None,
                          strategy: RetrievalStrategy = RetrievalStrategy.FUSION,
                          max_results: int = 5) -> RefinedResults:
    """Refine query retrieval using KWECLI retrieval refinement system."""
    refiner = RetrievalRefiner(project_path)
    return refiner.refine_retrieval(query, strategy=strategy, max_results=max_results)


if __name__ == "__main__":
    print("🧪 Testing KWECLI Retrieval Refinement System...")
    refiner = RetrievalRefiner()
    test_queries = ["How to implement authentication in Python?", "Database optimization techniques"]
    
    for query in test_queries:
        print(f"📋 Testing: {query}")
        for strategy in [RetrievalStrategy.FUSION, RetrievalStrategy.HYDE]:
            results = refiner.refine_retrieval(query, strategy=strategy, max_results=3)
            print(f"   {strategy.value}: {len(results.results)} results, score: {results.total_score:.2f}")
    
    print("✅ Retrieval Refinement System test complete")