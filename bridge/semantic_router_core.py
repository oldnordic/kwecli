#!/usr/bin/env python3
"""
KWECLI Semantic Router Core Analysis Engine
============================================

Core analysis engine for semantic query classification and routing decisions.
Extracted from semantic_router.py for CLAUDE.md compliance (≤300 lines).

File: bridge/semantic_router_core.py
Purpose: Query analysis and classification engine
"""

import re
import time
from typing import Dict, List, Set, Optional
from dataclasses import dataclass
from enum import Enum

class QueryType(Enum):
    """Semantic query classification types."""
    CODE = "code"
    DOCUMENTATION = "documentation"
    PLANNING = "planning"
    RESEARCH = "research"
    DEBUGGING = "debugging"
    ARCHITECTURE = "architecture"
    MEMORY_RETRIEVAL = "memory_retrieval"
    PATTERN_ANALYSIS = "pattern_analysis"


class DatabaseType(Enum):
    """Available database systems in KWECLI."""
    SQLITE = "sqlite"
    FAISS = "faiss"
    NEO4J = "neo4j"
    REDIS = "redis"


@dataclass
class QueryContext:
    """Comprehensive query context analysis."""
    query: str
    query_type: QueryType
    confidence: float
    entities: List[str]
    keywords: Set[str]
    complexity_score: int


class SemanticAnalyzer:
    """
    Core semantic analysis engine for KWECLI queries.
    Handles query classification, entity extraction, and complexity analysis.
    """
    
    def __init__(self):
        """Initialize semantic analyzer."""
        pass
        
        # Query classification patterns for native analysis
        self.classification_patterns = {
            QueryType.CODE: [
                r'\b(function|class|method|variable|import|def|return)\b',
                r'\b(python|javascript|rust|go|java)\b',
                r'\.(py|js|rs|go|java|cpp|h)$',
                r'\b(bug|error|exception|trace|debug)\b'
            ],
            QueryType.DOCUMENTATION: [
                r'\b(document|readme|guide|manual|help)\b',
                r'\b(explain|describe|how to|tutorial)\b',
                r'\.(md|rst|txt|doc)$',
                r'\b(api|specification|protocol)\b'
            ],
            QueryType.PLANNING: [
                r'\b(plan|sprint|task|milestone|roadmap)\b',
                r'\b(schedule|timeline|deadline|priority)\b',
                r'\b(requirements|specification|scope)\b',
                r'\b(epic|story|feature)\b'
            ],
            QueryType.RESEARCH: [
                r'\b(research|study|analysis|investigate)\b',
                r'\b(benchmark|performance|comparison)\b',
                r'\b(survey|review|evaluation)\b',
                r'\b(literature|paper|article)\b'
            ],
            QueryType.DEBUGGING: [
                r'\b(debug|fix|issue|problem|error)\b',
                r'\b(trace|stack|exception|crash)\b',
                r'\b(reproduce|isolate|diagnose)\b',
                r'\b(broken|failing|bug)\b'
            ],
            QueryType.ARCHITECTURE: [
                r'\b(architecture|design|structure|pattern)\b',
                r'\b(component|module|service|system)\b',
                r'\b(interface|protocol|contract)\b',
                r'\b(scalability|performance|reliability)\b'
            ]
        }
    
    def analyze_query(self, query: str, context: Optional[Dict] = None) -> QueryContext:
        """Comprehensive query analysis using native KWECLI methods."""
        try:
            # Core analysis components
            query_type = self._classify_query_type(query)
            confidence = self._calculate_confidence(query, query_type)
            entities = self._extract_entities(query)
            keywords = self._extract_keywords(query)
            complexity = self._calculate_complexity(query, entities, keywords)
            
            return QueryContext(
                query=query,
                query_type=query_type,
                confidence=confidence,
                entities=entities,
                keywords=keywords,
                complexity_score=complexity
            )
            
        except Exception:
            # Fallback analysis
            return self._create_fallback_context(query)
    
    def _classify_query_type(self, query: str) -> QueryType:
        """Classify query type using native pattern matching."""
        scores = {}
        query_lower = query.lower()
        
        for query_type, patterns in self.classification_patterns.items():
            score = 0
            for pattern in patterns:
                matches = len(re.findall(pattern, query_lower, re.IGNORECASE))
                score += matches
            scores[query_type] = score
        
        # Return highest scoring type, default to RESEARCH
        if scores:
            best_type = max(scores, key=scores.get)
            if scores[best_type] > 0:
                return best_type
        
        return QueryType.RESEARCH
    
    def _calculate_confidence(self, query: str, query_type: QueryType) -> float:
        """Calculate confidence in classification."""
        patterns = self.classification_patterns[query_type]
        matches = sum(len(re.findall(pattern, query.lower(), re.IGNORECASE)) for pattern in patterns)
        
        # Base confidence from pattern matching
        confidence = min(0.95, 0.3 + (matches * 0.15))
        
        # Boost confidence for strong indicators
        strong_indicators = {
            QueryType.CODE: ['function', 'class', 'def', 'import'],
            QueryType.DOCUMENTATION: ['explain', 'how to', 'guide', 'manual'],
            QueryType.PLANNING: ['plan', 'sprint', 'task', 'roadmap'],
            QueryType.DEBUGGING: ['debug', 'error', 'fix', 'bug']
        }
        
        if query_type in strong_indicators:
            for indicator in strong_indicators[query_type]:
                if indicator in query.lower():
                    confidence += 0.1
        
        return min(confidence, 0.95)
    
    def _extract_entities(self, query: str) -> List[str]:
        """Extract entities using native pattern recognition."""
        entities = []
        
        # File name patterns
        file_patterns = [
            r'\b\w+\.(py|js|rs|go|java|cpp|h|md|txt|json|yaml|toml)\b',
            r'\b[A-Z][a-zA-Z]*\.py\b',
            r'\b[a-z_]+\.(py|js)\b'
        ]
        
        for pattern in file_patterns:
            entities.extend(re.findall(pattern, query, re.IGNORECASE))
        
        # Class/function name patterns
        entity_patterns = [
            r'\b[A-Z][a-z]+(?:[A-Z][a-z]+)*\b',  # PascalCase
            r'\b[a-z_]+_[a-z_]+\b',  # snake_case
            r'\b[A-Z_]+\b',  # CONSTANTS
        ]
        
        for pattern in entity_patterns:
            matches = re.findall(pattern, query)
            entities.extend([match for match in matches if len(match) > 2])
        
        return list(set(entities))
    
    def _extract_keywords(self, query: str) -> Set[str]:
        """Extract keywords using native processing."""
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have',
            'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should'
        }
        
        words = re.findall(r'\b\w+\b', query.lower())
        keywords = {word for word in words if len(word) > 2 and word not in stop_words}
        
        return keywords
    
    def _calculate_complexity(self, query: str, entities: List[str], keywords: Set[str]) -> int:
        """Calculate query complexity using native analysis."""
        complexity = 0
        
        # Base complexity factors
        complexity += min(len(query) // 50, 5)  # Length factor
        complexity += len(entities)  # Entity count
        complexity += len(keywords) // 3  # Keyword density
        
        # Pattern complexity indicators
        complexity_indicators = [
            (r'\b(all|every|complete|entire)\b', 3),
            (r'\b(similar|related|connected|associated)\b', 2),
            (r'\b(analyze|compare|evaluate|assess)\b', 2),
            (r'\b(find|search|locate|identify)\b', 1),
            (r'\b(complex|comprehensive|detailed)\b', 2)
        ]
        
        query_lower = query.lower()
        for pattern, weight in complexity_indicators:
            if re.search(pattern, query_lower):
                complexity += weight
        
        return min(complexity, 10)  # Cap at 10
    
    def _create_fallback_context(self, query: str) -> QueryContext:
        """Create fallback context for failed analysis."""
        # Simple keyword-based classification
        query_type = QueryType.RESEARCH
        if any(word in query.lower() for word in ['code', 'function', 'class']):
            query_type = QueryType.CODE
        elif any(word in query.lower() for word in ['doc', 'explain', 'guide']):
            query_type = QueryType.DOCUMENTATION
        elif any(word in query.lower() for word in ['plan', 'sprint', 'task']):
            query_type = QueryType.PLANNING
        
        return QueryContext(
            query=query,
            query_type=query_type,
            confidence=0.6,
            entities=[],
            keywords=self._extract_keywords(query),
            complexity_score=min(len(query.split()) // 3, 5)
        )


# Test functionality if run directly
if __name__ == "__main__":
    print("🧪 Testing KWECLI Semantic Analyzer Core...")
    
    analyzer = SemanticAnalyzer()
    
    test_queries = [
        "Find Python functions for database connections",
        "Explain API documentation standards", 
        "Create sprint plan for authentication",
        "Debug connection timeout errors"
    ]
    
    for query in test_queries:
        context = analyzer.analyze_query(query)
        print(f"\n📋 Query: {query}")
        print(f"   Type: {context.query_type.value}")
        print(f"   Confidence: {context.confidence:.2f}")
        print(f"   Complexity: {context.complexity_score}/10")
        print(f"   Entities: {len(context.entities)}")
        print(f"   Keywords: {len(context.keywords)}")
    
    print("\n✅ Semantic Analyzer Core test complete")