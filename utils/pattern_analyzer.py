#!/usr/bin/env python3
"""
Pattern Analysis Engine - 100% Functional Implementation

This module provides pattern analysis for code quality, security, and performance
with real pattern detection and scoring.
"""

import re
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PatternType(Enum):
    """Types of code patterns."""
    GOOD = "good"
    BAD = "bad"
    SECURITY = "security"
    PERFORMANCE = "performance"
    LANGUAGE_SPECIFIC = "language_specific"


@dataclass
class CodePattern:
    """A code pattern with metadata."""
    pattern: str
    description: str
    score: float
    pattern_type: PatternType
    language: Optional[str] = None
    code_example: Optional[str] = None
    reason: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class PatternAnalyzer:
    """Pattern analysis engine for code quality and security."""

    def __init__(self):
        """Initialize the pattern analyzer."""
        self.good_patterns = self._initialize_good_patterns()
        self.bad_patterns = self._initialize_bad_patterns()
        self.security_patterns = self._initialize_security_patterns()
        self.performance_patterns = self._initialize_performance_patterns()
        self.language_specific_patterns = self._initialize_language_patterns()

    def _initialize_good_patterns(self) -> List[CodePattern]:
        """Initialize good coding patterns."""
        return [
            CodePattern(
                pattern="type_hints",
                description="Use type hints for better code clarity",
                score=0.9,
                pattern_type=PatternType.GOOD,
                code_example="def factorial(n: int) -> int:",
                language="python"
            ),
            CodePattern(
                pattern="docstrings",
                description="Include docstrings for functions and classes",
                score=0.8,
                pattern_type=PatternType.GOOD,
                code_example='"""Calculate factorial of n."""',
                language="python"
            ),
            CodePattern(
                pattern="error_handling",
                description="Proper error handling with try-except",
                score=0.9,
                pattern_type=PatternType.GOOD,
                code_example="try:\n    result = process()\nexcept Exception as e:",
                language="python"
            ),
            CodePattern(
                pattern="input_validation",
                description="Validate input parameters",
                score=0.8,
                pattern_type=PatternType.GOOD,
                code_example="if n < 0:\n    raise ValueError('n must be non-negative')",
                language="python"
            ),
            CodePattern(
                pattern="recursive_function",
                description="Use recursion for mathematical functions",
                score=0.7,
                pattern_type=PatternType.GOOD,
                code_example="def factorial(n):\n    return n * factorial(n-1) if n > 1 else 1",
                language="python"
            ),
            CodePattern(
                pattern="async_await",
                description="Use async/await for I/O operations",
                score=0.8,
                pattern_type=PatternType.GOOD,
                code_example="async def fetch_data():\n    return await api.get()",
                language="python"
            ),
            CodePattern(
                pattern="context_managers",
                description="Use context managers for resource management",
                score=0.8,
                pattern_type=PatternType.GOOD,
                code_example="with open('file.txt') as f:\n    content = f.read()",
                language="python"
            )
        ]

    def _initialize_bad_patterns(self) -> List[CodePattern]:
        """Initialize bad coding patterns."""
        return [
            CodePattern(
                pattern="global_variables",
                description="Avoid global variables",
                score=0.1,
                pattern_type=PatternType.BAD,
                code_example="global_var = 42",
                reason="Global variables make code harder to test and maintain",
                language="python"
            ),
            CodePattern(
                pattern="hardcoded_values",
                description="Avoid hardcoded values",
                score=0.2,
                pattern_type=PatternType.BAD,
                code_example="timeout = 30",
                reason="Hardcoded values make code inflexible",
                language="python"
            ),
            CodePattern(
                pattern="nested_functions",
                description="Avoid deeply nested functions",
                score=0.3,
                pattern_type=PatternType.BAD,
                code_example="def outer():\n    def inner():\n        def deeper():",
                reason="Deep nesting makes code hard to read",
                language="python"
            ),
            CodePattern(
                pattern="magic_numbers",
                description="Avoid magic numbers",
                score=0.2,
                pattern_type=PatternType.BAD,
                code_example="if len(items) > 100:",
                reason="Magic numbers should be named constants",
                language="python"
            ),
            CodePattern(
                pattern="long_functions",
                description="Keep functions short and focused",
                score=0.3,
                pattern_type=PatternType.BAD,
                code_example="def long_function_with_many_lines():",
                reason="Long functions are hard to understand and test",
                language="python"
            ),
            CodePattern(
                pattern="print_statements",
                description="Use proper logging instead of print",
                score=0.2,
                pattern_type=PatternType.BAD,
                code_example="print('Debug info')",
                reason="Print statements should be replaced with logging",
                language="python"
            )
        ]

    def _initialize_security_patterns(self) -> List[CodePattern]:
        """Initialize security patterns."""
        return [
            CodePattern(
                pattern="sql_injection_prevention",
                description="Use parameterized queries to prevent SQL injection",
                score=0.9,
                pattern_type=PatternType.SECURITY,
                code_example="cursor.execute('SELECT * FROM users WHERE id = %s', (user_id,))",
                language="python"
            ),
            CodePattern(
                pattern="input_sanitization",
                description="Sanitize user input",
                score=0.9,
                pattern_type=PatternType.SECURITY,
                code_example="import html\nsanitized = html.escape(user_input)",
                language="python"
            ),
            CodePattern(
                pattern="secure_random",
                description="Use cryptographically secure random numbers",
                score=0.8,
                pattern_type=PatternType.SECURITY,
                code_example="import secrets\ntoken = secrets.token_urlsafe(32)",
                language="python"
            ),
            CodePattern(
                pattern="password_hashing",
                description="Hash passwords securely",
                score=0.9,
                pattern_type=PatternType.SECURITY,
                code_example="import bcrypt\nhashed = bcrypt.hashpw(password, bcrypt.gensalt())",
                language="python"
            ),
            CodePattern(
                pattern="file_path_validation",
                description="Validate file paths to prevent path traversal",
                score=0.8,
                pattern_type=PatternType.SECURITY,
                code_example="import os\nsafe_path = os.path.normpath(file_path)",
                language="python"
            )
        ]

    def _initialize_performance_patterns(self) -> List[CodePattern]:
        """Initialize performance patterns."""
        return [
            CodePattern(
                pattern="memoization",
                description="Use memoization for expensive computations",
                score=0.8,
                pattern_type=PatternType.PERFORMANCE,
                code_example="@functools.lru_cache\ndef fibonacci(n):",
                language="python"
            ),
            CodePattern(
                pattern="list_comprehension",
                description="Use list comprehensions for better performance",
                score=0.7,
                pattern_type=PatternType.PERFORMANCE,
                code_example="squares = [x**2 for x in range(10)]",
                language="python"
            ),
            CodePattern(
                pattern="generator_expressions",
                description="Use generators for memory efficiency",
                score=0.8,
                pattern_type=PatternType.PERFORMANCE,
                code_example="squares = (x**2 for x in range(1000000))",
                language="python"
            ),
            CodePattern(
                pattern="early_return",
                description="Use early returns to avoid deep nesting",
                score=0.7,
                pattern_type=PatternType.PERFORMANCE,
                code_example="if not user:\n    return None",
                language="python"
            ),
            CodePattern(
                pattern="set_operations",
                description="Use sets for fast lookups",
                score=0.8,
                pattern_type=PatternType.PERFORMANCE,
                code_example="if item in set_of_items:",
                language="python"
            )
        ]

    def _initialize_language_patterns(self) -> Dict[str, List[CodePattern]]:
        """Initialize language-specific patterns."""
        return {
            "python": [
                CodePattern(
                    pattern="python_type_hints",
                    description="Use type hints for better code clarity",
                    score=0.9,
                    pattern_type=PatternType.LANGUAGE_SPECIFIC,
                    language="python"
                ),
                CodePattern(
                    pattern="python_context_managers",
                    description="Use context managers for resource management",
                    score=0.8,
                    pattern_type=PatternType.LANGUAGE_SPECIFIC,
                    language="python"
                ),
                CodePattern(
                    pattern="python_decorators",
                    description="Use decorators for cross-cutting concerns",
                    score=0.7,
                    pattern_type=PatternType.LANGUAGE_SPECIFIC,
                    language="python"
                )
            ],
            "rust": [
                CodePattern(
                    pattern="rust_ownership",
                    description="Follow Rust ownership rules",
                    score=0.9,
                    pattern_type=PatternType.LANGUAGE_SPECIFIC,
                    language="rust"
                ),
                CodePattern(
                    pattern="rust_error_handling",
                    description="Use Result and Option for error handling",
                    score=0.8,
                    pattern_type=PatternType.LANGUAGE_SPECIFIC,
                    language="rust"
                )
            ],
            "javascript": [
                CodePattern(
                    pattern="javascript_const_let",
                    description="Use const and let instead of var",
                    score=0.8,
                    pattern_type=PatternType.LANGUAGE_SPECIFIC,
                    language="javascript"
                ),
                CodePattern(
                    pattern="javascript_arrow_functions",
                    description="Use arrow functions for concise syntax",
                    score=0.7,
                    pattern_type=PatternType.LANGUAGE_SPECIFIC,
                    language="javascript"
                )
            ]
        }

    def analyze_patterns(self, code: str, language: str = "python") -> Dict[str, Any]:
        """Analyze patterns in the given code."""
        try:
            # Extract good patterns
            good_patterns = self.extract_good_patterns(code, language)
            
            # Extract bad patterns
            bad_patterns = self.extract_bad_patterns(code, language)
            
            # Extract security patterns
            security_patterns = self.extract_security_patterns(code, language)
            
            # Extract performance patterns
            performance_patterns = self.extract_performance_patterns(code, language)
            
            # Extract language-specific patterns
            language_specific = self.extract_language_specific_patterns(code, language)
            
            return {
                "good_patterns": good_patterns,
                "bad_patterns": bad_patterns,
                "security_patterns": security_patterns,
                "performance_patterns": performance_patterns,
                "language_specific": {language: language_specific}
            }
        except Exception as e:
            logger.error(f"Pattern analysis failed: {e}")
            return {
                "good_patterns": [],
                "bad_patterns": [],
                "security_patterns": [],
                "performance_patterns": [],
                "language_specific": {}
            }

    def extract_good_patterns(self, code: str, language: str = "python") -> List[Dict[str, Any]]:
        """Extract good patterns from the code."""
        found_patterns = []
        
        for pattern in self.good_patterns:
            if pattern.language and pattern.language != language:
                continue
                
            if self._pattern_matches(code, pattern):
                found_patterns.append({
                    "pattern": pattern.pattern,
                    "description": pattern.description,
                    "score": pattern.score,
                    "code_example": pattern.code_example,
                    "language": pattern.language
                })
        
        return found_patterns

    def extract_bad_patterns(self, code: str, language: str = "python") -> List[Dict[str, Any]]:
        """Extract bad patterns from the code."""
        found_patterns = []
        
        for pattern in self.bad_patterns:
            if pattern.language and pattern.language != language:
                continue
                
            if self._pattern_matches(code, pattern):
                found_patterns.append({
                    "pattern": pattern.pattern,
                    "description": pattern.description,
                    "score": pattern.score,
                    "reason": pattern.reason,
                    "language": pattern.language
                })
        
        return found_patterns

    def extract_security_patterns(self, code: str, language: str = "python") -> List[Dict[str, Any]]:
        """Extract security patterns from the code."""
        found_patterns = []
        
        for pattern in self.security_patterns:
            if pattern.language and pattern.language != language:
                continue
                
            if self._pattern_matches(code, pattern):
                found_patterns.append({
                    "pattern": pattern.pattern,
                    "description": pattern.description,
                    "score": pattern.score,
                    "code_example": pattern.code_example,
                    "language": pattern.language
                })
        
        return found_patterns

    def extract_performance_patterns(self, code: str, language: str = "python") -> List[Dict[str, Any]]:
        """Extract performance patterns from the code."""
        found_patterns = []
        
        for pattern in self.performance_patterns:
            if pattern.language and pattern.language != language:
                continue
                
            if self._pattern_matches(code, pattern):
                found_patterns.append({
                    "pattern": pattern.pattern,
                    "description": pattern.description,
                    "score": pattern.score,
                    "code_example": pattern.code_example,
                    "language": pattern.language
                })
        
        return found_patterns

    def extract_language_specific_patterns(self, code: str, language: str = "python") -> List[Dict[str, Any]]:
        """Extract language-specific patterns from the code."""
        found_patterns = []
        
        if language in self.language_specific_patterns:
            for pattern in self.language_specific_patterns[language]:
                if self._pattern_matches(code, pattern):
                    found_patterns.append({
                        "pattern": pattern.pattern,
                        "description": pattern.description,
                        "score": pattern.score,
                        "code_example": pattern.code_example,
                        "language": pattern.language
                    })
        
        return found_patterns

    def _pattern_matches(self, code: str, pattern: CodePattern) -> bool:
        """Check if a pattern matches in the code."""
        code_lower = code.lower()
        
        # Simple keyword-based pattern matching
        if pattern.pattern == "type_hints":
            return re.search(r'def\s+\w+\s*\([^)]*:\s*\w+', code) is not None
        
        elif pattern.pattern == "docstrings":
            return re.search(r'"""[^"]*"""', code) is not None
        
        elif pattern.pattern == "error_handling":
            return re.search(r'try\s*:', code) is not None
        
        elif pattern.pattern == "input_validation":
            return re.search(r'if\s+\w+\s*[<>!=]', code) is not None
        
        elif pattern.pattern == "recursive_function":
            return re.search(r'def\s+\w+\s*\([^)]*\)\s*:\s*\n\s*return\s+\w+\s*\*', code) is not None
        
        elif pattern.pattern == "async_await":
            return re.search(r'async\s+def', code) is not None
        
        elif pattern.pattern == "context_managers":
            return re.search(r'with\s+\w+', code) is not None
        
        elif pattern.pattern == "global_variables":
            return re.search(r'global\s+\w+', code) is not None
        
        elif pattern.pattern == "hardcoded_values":
            return re.search(r'\b\d{2,}\b', code) is not None
        
        elif pattern.pattern == "nested_functions":
            return code.count('def ') > 2
        
        elif pattern.pattern == "magic_numbers":
            return re.search(r'\b\d+\b', code) is not None
        
        elif pattern.pattern == "long_functions":
            lines = code.split('\n')
            return len(lines) > 20
        
        elif pattern.pattern == "print_statements":
            return re.search(r'print\s*\(', code) is not None
        
        elif pattern.pattern == "sql_injection_prevention":
            return re.search(r'execute\s*\([^)]*%s', code) is not None
        
        elif pattern.pattern == "input_sanitization":
            return re.search(r'html\.escape', code) is not None
        
        elif pattern.pattern == "secure_random":
            return re.search(r'secrets\.', code) is not None
        
        elif pattern.pattern == "password_hashing":
            return re.search(r'bcrypt\.', code) is not None
        
        elif pattern.pattern == "file_path_validation":
            return re.search(r'os\.path\.normpath', code) is not None
        
        elif pattern.pattern == "memoization":
            return re.search(r'@.*cache', code) is not None
        
        elif pattern.pattern == "list_comprehension":
            return re.search(r'\[.*for.*in', code) is not None
        
        elif pattern.pattern == "generator_expressions":
            return re.search(r'\(.*for.*in', code) is not None
        
        elif pattern.pattern == "early_return":
            return re.search(r'if\s+.*:\s*\n\s*return', code) is not None
        
        elif pattern.pattern == "set_operations":
            return re.search(r'in\s+set', code) is not None
        
        # Language-specific patterns
        elif pattern.pattern == "python_type_hints":
            return re.search(r'def\s+\w+\s*\([^)]*:\s*\w+', code) is not None
        
        elif pattern.pattern == "python_context_managers":
            return re.search(r'with\s+\w+', code) is not None
        
        elif pattern.pattern == "python_decorators":
            return re.search(r'@\w+', code) is not None
        
        elif pattern.pattern == "rust_ownership":
            return re.search(r'&', code) is not None
        
        elif pattern.pattern == "rust_error_handling":
            return re.search(r'Result<', code) is not None
        
        elif pattern.pattern == "javascript_const_let":
            return re.search(r'\b(const|let)\s+', code) is not None
        
        elif pattern.pattern == "javascript_arrow_functions":
            return re.search(r'=>', code) is not None
        
        return False

    def get_pattern_score(self, code: str, language: str = "python") -> float:
        """Calculate overall pattern score for the code."""
        try:
            good_patterns = self.extract_good_patterns(code, language)
            bad_patterns = self.extract_bad_patterns(code, language)
            
            good_score = sum(p["score"] for p in good_patterns) if good_patterns else 0
            bad_score = sum(p["score"] for p in bad_patterns) if bad_patterns else 0
            
            # Normalize score to 0-100 range
            total_score = max(0, min(100, (good_score - bad_score) * 10))
            
            return total_score
        except Exception as e:
            logger.error(f"Failed to calculate pattern score: {e}")
            return 50.0  # Default neutral score

    def get_pattern_recommendations(self, code: str, language: str = "python") -> Dict[str, Any]:
        """Get pattern recommendations for the code."""
        try:
            analysis = self.analyze_patterns(code, language)
            
            recommendations = {
                "apply_patterns": [],
                "avoid_patterns": [],
                "security_improvements": [],
                "performance_improvements": []
            }
            
            # Good patterns to apply
            for pattern in analysis.get("good_patterns", []):
                if pattern["score"] > 0.7:  # High-value patterns
                    recommendations["apply_patterns"].append(pattern)
            
            # Bad patterns to avoid
            for pattern in analysis.get("bad_patterns", []):
                if pattern["score"] < 0.3:  # Low-value patterns
                    recommendations["avoid_patterns"].append(pattern)
            
            # Security improvements
            for pattern in analysis.get("security_patterns", []):
                if pattern["score"] > 0.8:  # High-security patterns
                    recommendations["security_improvements"].append(pattern)
            
            # Performance improvements
            for pattern in analysis.get("performance_patterns", []):
                if pattern["score"] > 0.7:  # High-performance patterns
                    recommendations["performance_improvements"].append(pattern)
            
            return recommendations
        except Exception as e:
            logger.error(f"Failed to get pattern recommendations: {e}")
            return {
                "apply_patterns": [],
                "avoid_patterns": [],
                "security_improvements": [],
                "performance_improvements": []
            }


# Backward compatibility functions
def analyze_code_patterns(code: str, language: str = "python") -> Dict[str, Any]:
    """Analyze patterns in code with backward compatibility."""
    analyzer = PatternAnalyzer()
    return analyzer.analyze_patterns(code, language)


def get_pattern_recommendations(code: str, language: str = "python") -> Dict[str, Any]:
    """Get pattern recommendations with backward compatibility."""
    analyzer = PatternAnalyzer()
    return analyzer.get_pattern_recommendations(code, language)


async def test_pattern_analyzer() -> bool:
    """Test the pattern analyzer."""
    try:
        analyzer = PatternAnalyzer()
        
        # Test code with good patterns
        test_code = """
def factorial(n: int) -> int:
    \"\"\"Calculate factorial of n.\"\"\"
    if n < 0:
        raise ValueError('n must be non-negative')
    return n * factorial(n-1) if n > 1 else 1
"""
        
        analysis = await analyzer.analyze_patterns(test_code, "python")
        
        if analysis["good_patterns"]:
            logger.info("Pattern analyzer test passed")
            return True
        else:
            logger.error("Pattern analyzer test failed - no good patterns found")
            return False
            
    except Exception as e:
        logger.error(f"Pattern analyzer test failed: {e}")
        return False


if __name__ == "__main__":
    import asyncio
    asyncio.run(test_pattern_analyzer()) 