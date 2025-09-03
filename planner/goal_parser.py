#!/usr/bin/env python3
"""
Goal Parser - Natural Language Goal Processing
==============================================

Focused module for parsing natural language goals into structured data.
Handles goal classification, complexity assessment, and task estimation.

File: planner/goal_parser.py
Purpose: Goal parsing and classification
"""

import logging
import uuid
from typing import Dict, Any
from datetime import datetime

logger = logging.getLogger(__name__)


class GoalParser:
    """Focused goal parsing and classification."""
    
    def __init__(self):
        """Initialize goal parser."""
        self.goal_types = {
            "development": ["create", "build", "implement", "develop", "make", "add"],
            "maintenance": ["fix", "debug", "resolve", "repair", "correct", "patch"],
            "analysis": ["analyze", "review", "investigate", "study", "examine", "research"],
            "improvement": ["refactor", "improve", "optimize", "enhance", "upgrade", "modernize"]
        }
    
    def parse_goal(self, goal_text: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Parse natural language goal into structured components.
        
        Args:
            goal_text: Natural language description of goal
            context: Additional context (project type, constraints, etc.)
        
        Returns:
            Structured goal with classification and metadata
        """
        goal_id = f"goal_{uuid.uuid4().hex[:8]}"
        context = context or {}
        
        # Parse goal components
        goal_data = {
            "id": goal_id,
            "text": goal_text.strip(),
            "context": context,
            "created_at": datetime.now().isoformat(),
            "status": "parsed",
            "type": self._classify_goal_type(goal_text),
            "complexity": self._assess_complexity(goal_text),
            "estimated_tasks": self._estimate_task_count(goal_text),
            "keywords": self._extract_keywords(goal_text),
            "priority": self._determine_priority(goal_text, context)
        }
        
        return goal_data
    
    def _classify_goal_type(self, goal_text: str) -> str:
        """Classify goal type based on content analysis."""
        goal_lower = goal_text.lower()
        
        # Score each goal type
        type_scores = {}
        for goal_type, keywords in self.goal_types.items():
            score = sum(1 for keyword in keywords if keyword in goal_lower)
            if score > 0:
                type_scores[goal_type] = score
        
        # Return highest scoring type or default
        if type_scores:
            return max(type_scores, key=type_scores.get)
        else:
            return "general"
    
    def _assess_complexity(self, goal_text: str) -> str:
        """Assess goal complexity based on multiple factors."""
        word_count = len(goal_text.split())
        
        # Technical complexity indicators
        complex_indicators = [
            "system", "architecture", "integration", "database", 
            "api", "microservice", "distributed", "scalable"
        ]
        
        simple_indicators = [
            "fix", "update", "change", "add", "remove", "simple"
        ]
        
        goal_lower = goal_text.lower()
        complex_score = sum(1 for indicator in complex_indicators if indicator in goal_lower)
        simple_score = sum(1 for indicator in simple_indicators if indicator in goal_lower)
        
        # Combine word count and complexity indicators
        if word_count > 25 or complex_score > 2:
            return "high"
        elif word_count > 15 or (complex_score > 0 and simple_score == 0):
            return "medium"
        elif simple_score > 0 or word_count < 8:
            return "low"
        else:
            return "medium"
    
    def _estimate_task_count(self, goal_text: str) -> int:
        """Estimate number of tasks based on complexity and type."""
        complexity = self._assess_complexity(goal_text)
        goal_type = self._classify_goal_type(goal_text)
        
        # Base task counts by complexity
        base_counts = {
            "high": 8,
            "medium": 5,
            "low": 3
        }
        
        # Adjust by goal type
        type_multipliers = {
            "development": 1.0,
            "maintenance": 0.8,  # Usually fewer tasks
            "analysis": 0.6,     # Typically research-heavy
            "improvement": 1.2   # Often touches multiple areas
        }
        
        base_count = base_counts[complexity]
        multiplier = type_multipliers.get(goal_type, 1.0)
        
        return max(2, min(10, int(base_count * multiplier)))
    
    def _extract_keywords(self, goal_text: str) -> list:
        """Extract key technical terms from goal text."""
        # Common technical keywords
        tech_keywords = [
            "api", "database", "frontend", "backend", "server", "client",
            "authentication", "authorization", "security", "performance",
            "testing", "deployment", "docker", "kubernetes", "microservice",
            "rest", "graphql", "websocket", "cache", "queue", "storage"
        ]
        
        goal_lower = goal_text.lower()
        found_keywords = [kw for kw in tech_keywords if kw in goal_lower]
        
        # Add domain-specific words (words longer than 4 chars that aren't common)
        words = goal_text.lower().split()
        common_words = {"with", "from", "that", "this", "have", "will", "should", "would"}
        
        domain_words = [
            word.strip(".,!?;:") for word in words 
            if len(word) > 4 and word not in common_words
        ]
        
        return list(set(found_keywords + domain_words[:5]))  # Limit domain words
    
    def _determine_priority(self, goal_text: str, context: Dict[str, Any]) -> str:
        """Determine goal priority based on text and context."""
        # Check for explicit priority indicators
        high_priority_words = ["urgent", "critical", "asap", "immediately", "fix", "broken"]
        low_priority_words = ["nice", "future", "eventually", "consider", "maybe"]
        
        goal_lower = goal_text.lower()
        
        if any(word in goal_lower for word in high_priority_words):
            return "high"
        elif any(word in goal_lower for word in low_priority_words):
            return "low"
        elif context.get("deadline"):
            return "high"
        else:
            return "medium"


# Test functionality if run directly
if __name__ == "__main__":
    print("🧪 Testing GoalParser...")
    
    parser = GoalParser()
    
    test_goals = [
        "Create a REST API with FastAPI for user management",
        "Fix critical bug in authentication system",
        "Analyze database performance bottlenecks",
        "Refactor legacy code to use modern Python patterns"
    ]
    
    for goal_text in test_goals:
        result = parser.parse_goal(goal_text)
        print(f"\nGoal: {goal_text}")
        print(f"  Type: {result['type']}")
        print(f"  Complexity: {result['complexity']}")
        print(f"  Estimated tasks: {result['estimated_tasks']}")
        print(f"  Priority: {result['priority']}")
        print(f"  Keywords: {', '.join(result['keywords'])}")
    
    print("\n✅ GoalParser test complete")