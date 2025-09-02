#!/usr/bin/env python3
"""
Code Review System

This module provides comprehensive code review capabilities including
syntax validation, style checking, security scanning, and quality scoring.
"""

import ast
import re
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class ReviewSeverity(Enum):
    """Severity levels for code review issues."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ReviewCategory(Enum):
    """Categories of code review issues."""
    SYNTAX = "syntax"
    STYLE = "style"
    SECURITY = "security"
    PERFORMANCE = "performance"
    MAINTAINABILITY = "maintainability"
    DOCUMENTATION = "documentation"
    BEST_PRACTICES = "best_practices"


@dataclass
class CodeIssue:
    """A code review issue."""
    line_number: int
    column: int
    severity: ReviewSeverity
    category: ReviewCategory
    message: str
    suggestion: Optional[str] = None
    code_snippet: Optional[str] = None


@dataclass
class CodeReviewResult:
    """Result of a code review."""
    issues: List[CodeIssue]
    score: float  # 0-100
    passed: bool
    summary: str
    metadata: Dict[str, Any]
    language: str = "python"  # Add language attribute


class CodeReviewer:
    """Comprehensive code review system."""
    
    def __init__(self):
        self.severity_weights = {
            ReviewSeverity.LOW: 1,
            ReviewSeverity.MEDIUM: 3,
            ReviewSeverity.HIGH: 7,
            ReviewSeverity.CRITICAL: 15
        }
    
    def review_code(self, code: str, language: str = "python") -> CodeReviewResult:
        """Perform comprehensive code review."""
        issues = []
        
        # Basic validation
        if not code.strip():
            return CodeReviewResult(
                issues=[],
                score=0.0,
                passed=False,
                summary="Code is empty",
                metadata={"language": language, "lines": 0},
                language=language
            )
        
        # Language-specific reviews
        if language.lower() in ["python", "py"]:
            issues.extend(self._review_python_code(code))
        elif language.lower() in ["javascript", "js"]:
            issues.extend(self._review_javascript_code(code))
        elif language.lower() in ["rust", "rs"]:
            issues.extend(self._review_rust_code(code))
        else:
            # Generic review for other languages
            issues.extend(self._review_generic_code(code, language))
        
        # Common security checks
        issues.extend(self._security_scan(code, language))
        
        # Performance checks
        issues.extend(self._performance_scan(code, language))
        
        # Documentation checks
        issues.extend(self._documentation_scan(code, language))
        
        # Calculate score
        score = self._calculate_score(issues)
        passed = score >= 70.0  # Pass threshold
        
        # Generate summary
        summary = self._generate_summary(issues, score)
        
        metadata = {
            "language": language,
            "lines": len(code.split('\n')),
            "characters": len(code),
            "issues_by_severity": self._count_issues_by_severity(issues),
            "issues_by_category": self._count_issues_by_category(issues)
        }
        
        return CodeReviewResult(
            issues=issues,
            score=score,
            passed=passed,
            summary=summary,
            metadata=metadata,
            language=language
        )
    
    def _review_python_code(self, code: str) -> List[CodeIssue]:
        """Review Python code specifically."""
        issues = []
        
        try:
            # Parse AST for syntax and structure analysis
            tree = ast.parse(code)
            
            # Check for common Python issues
            for node in ast.walk(tree):
                # Check for bare except clauses
                if isinstance(node, ast.ExceptHandler) and node.type is None:
                    issues.append(CodeIssue(
                        line_number=getattr(node, 'lineno', 0),
                        column=getattr(node, 'col_offset', 0),
                        severity=ReviewSeverity.HIGH,
                        category=ReviewCategory.BEST_PRACTICES,
                        message="Bare except clause - specify exception type",
                        suggestion="Use specific exception types like 'except ValueError:'"
                    ))
                
                # Check for unused imports
                if isinstance(node, ast.Import) or isinstance(node, ast.ImportFrom):
                    # This is a simplified check - in practice you'd need more sophisticated analysis
                    pass
                
                # Check for long functions
                if isinstance(node, ast.FunctionDef):
                    function_lines = self._count_function_lines(node, code)
                    if function_lines > 50:
                        issues.append(CodeIssue(
                            line_number=node.lineno,
                            column=node.col_offset,
                            severity=ReviewSeverity.MEDIUM,
                            category=ReviewCategory.MAINTAINABILITY,
                            message=f"Function is too long ({function_lines} lines)",
                            suggestion="Consider breaking the function into smaller functions"
                        ))
                
                # Check for missing docstrings
                if isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.Module)):
                    if not self._has_docstring(node):
                        if isinstance(node, ast.Module):
                            # Module docstring is optional
                            continue
                        issues.append(CodeIssue(
                            line_number=node.lineno,
                            column=node.col_offset,
                            severity=ReviewSeverity.LOW,
                            category=ReviewCategory.DOCUMENTATION,
                            message="Missing docstring",
                            suggestion="Add a docstring to describe the purpose"
                        ))
        
        except SyntaxError as e:
            issues.append(CodeIssue(
                line_number=e.lineno or 0,
                column=e.offset or 0,
                severity=ReviewSeverity.CRITICAL,
                category=ReviewCategory.SYNTAX,
                message=f"Syntax error: {e.msg}",
                suggestion="Fix the syntax error"
            ))
        
        except Exception as e:
            issues.append(CodeIssue(
                line_number=0,
                column=0,
                severity=ReviewSeverity.HIGH,
                category=ReviewCategory.SYNTAX,
                message=f"Error parsing code: {str(e)}",
                suggestion="Check the code structure"
            ))
        
        return issues
    
    def _review_javascript_code(self, code: str) -> List[CodeIssue]:
        """Review JavaScript code specifically."""
        issues = []
        
        # Check for common JavaScript issues
        lines = code.split('\n')
        
        for i, line in enumerate(lines, 1):
            # Check for console.log statements (often left in production code)
            if 'console.log(' in line:
                issues.append(CodeIssue(
                    line_number=i,
                    column=line.find('console.log('),
                    severity=ReviewSeverity.MEDIUM,
                    category=ReviewCategory.BEST_PRACTICES,
                    message="console.log statement found",
                    suggestion="Remove or replace with proper logging"
                ))
            
            # Check for var usage (prefer let/const)
            if re.search(r'\bvar\s+', line):
                issues.append(CodeIssue(
                    line_number=i,
                    column=line.find('var'),
                    severity=ReviewSeverity.LOW,
                    category=ReviewCategory.BEST_PRACTICES,
                    message="Using 'var' instead of 'let' or 'const'",
                    suggestion="Use 'let' or 'const' instead of 'var'"
                ))
            
            # Check for missing semicolons
            if (line.strip() and not line.strip().endswith(';') and 
                not line.strip().endswith('{') and not line.strip().endswith('}')):
                if not line.strip().startswith('//') and not line.strip().startswith('/*'):
                    issues.append(CodeIssue(
                        line_number=i,
                        column=len(line),
                        severity=ReviewSeverity.LOW,
                        category=ReviewCategory.STYLE,
                        message="Missing semicolon",
                        suggestion="Add semicolon at end of statement"
                    ))
        
        return issues
    
    def _review_rust_code(self, code: str) -> List[CodeIssue]:
        """Review Rust code specifically."""
        issues = []
        
        # Check for common Rust issues
        lines = code.split('\n')
        
        for i, line in enumerate(lines, 1):
            # Check for unwrap() usage (should use proper error handling)
            if '.unwrap()' in line:
                issues.append(CodeIssue(
                    line_number=i,
                    column=line.find('.unwrap()'),
                    severity=ReviewSeverity.MEDIUM,
                    category=ReviewCategory.BEST_PRACTICES,
                    message="Using unwrap() - consider proper error handling",
                    suggestion="Use match, if let, or ? operator for proper error handling"
                ))
            
            # Check for println! in library code
            if 'println!' in line:
                issues.append(CodeIssue(
                    line_number=i,
                    column=line.find('println!'),
                    severity=ReviewSeverity.LOW,
                    category=ReviewCategory.BEST_PRACTICES,
                    message="println! found - consider using a logging crate",
                    suggestion="Use a logging crate like log or tracing"
                ))
        
        return issues
    
    def _review_generic_code(self, code: str, language: str) -> List[CodeIssue]:
        """Generic code review for unsupported languages."""
        issues = []
        
        # Basic checks that apply to most languages
        lines = code.split('\n')
        
        # Check for very long lines
        for i, line in enumerate(lines, 1):
            if len(line) > 120:
                issues.append(CodeIssue(
                    line_number=i,
                    column=120,
                    severity=ReviewSeverity.LOW,
                    category=ReviewCategory.STYLE,
                    message="Line too long",
                    suggestion="Break long lines for better readability"
                ))
        
        # Check for trailing whitespace
        for i, line in enumerate(lines, 1):
            if line.rstrip() != line:
                issues.append(CodeIssue(
                    line_number=i,
                    column=len(line.rstrip()),
                    severity=ReviewSeverity.LOW,
                    category=ReviewCategory.STYLE,
                    message="Trailing whitespace",
                    suggestion="Remove trailing whitespace"
                ))
        
        return issues
    
    def _security_scan(self, code: str, language: str) -> List[CodeIssue]:
        """Perform security vulnerability scanning."""
        issues = []
        
        # Common security patterns to check
        security_patterns = [
            # SQL Injection - more comprehensive patterns
            (r'execute\s*\(\s*[\'"][^\'"]*\+', 'Potential SQL injection - use parameterized queries'),
            (r'[\'"][^\'"]*SELECT[^\'"]*\+', 'Potential SQL injection - use parameterized queries'),
            (r'[\'"][^\'"]*WHERE[^\'"]*\+', 'Potential SQL injection - use parameterized queries'),
            (r'[\'"][^\'"]*INSERT[^\'"]*\+', 'Potential SQL injection - use parameterized queries'),
            (r'[\'"][^\'"]*UPDATE[^\'"]*\+', 'Potential SQL injection - use parameterized queries'),
            (r'[\'"][^\'"]*DELETE[^\'"]*\+', 'Potential SQL injection - use parameterized queries'),
            # More general SQL injection patterns
            (r'[\'"][^\'"]*\+.*user_input', 'Potential SQL injection - use parameterized queries'),
            (r'[\'"][^\'"]*\+.*input', 'Potential SQL injection - use parameterized queries'),
            (r'[\'"][^\'"]*\+.*variable', 'Potential SQL injection - use parameterized queries'),
            # Command Injection
            (r'os\.system\s*\(', 'Command injection risk - use subprocess with proper escaping'),
            (r'subprocess\.run\s*\([^)]*shell\s*=\s*True', 'Command injection risk - avoid shell=True'),
            # Hardcoded credentials
            (r'password\s*=\s*[\'"][^\'"]+[\'"]', 'Hardcoded password detected'),
            (r'api_key\s*=\s*[\'"][^\'"]+[\'"]', 'Hardcoded API key detected'),
            # Weak random
            (r'random\.randint\s*\(', 'Weak random - use secrets module for cryptographic operations'),
            # Debug statements
            (r'debugger;', 'Debug statement found'),
            (r'pdb\.set_trace\s*\(', 'Debug statement found'),
        ]
        
        lines = code.split('\n')
        for i, line in enumerate(lines, 1):
            for pattern, message in security_patterns:
                if re.search(pattern, line, re.IGNORECASE):
                    issues.append(CodeIssue(
                        line_number=i,
                        column=0,
                        severity=ReviewSeverity.HIGH,
                        category=ReviewCategory.SECURITY,
                        message=message,
                        suggestion="Review and fix security vulnerability"
                    ))
        
        return issues
    
    def _performance_scan(self, code: str, language: str) -> List[CodeIssue]:
        """Perform performance analysis."""
        issues = []
        
        # Performance anti-patterns
        performance_patterns = [
            # Python specific
            (r'for\s+\w+\s+in\s+range\s*\(\s*len\s*\(', 'Inefficient iteration - use enumerate()'),
            (r'\.append\s*\(\s*\[\s*\]', 'Inefficient list creation'),
            # General
            (r'while\s+True:', 'Infinite loop risk'),
            (r'for\s+\w+\s+in\s+\w+\.keys\s*\(\):', 'Inefficient iteration - iterate directly'),
        ]
        
        lines = code.split('\n')
        for i, line in enumerate(lines, 1):
            for pattern, message in performance_patterns:
                if re.search(pattern, line, re.IGNORECASE):
                    issues.append(CodeIssue(
                        line_number=i,
                        column=0,
                        severity=ReviewSeverity.MEDIUM,
                        category=ReviewCategory.PERFORMANCE,
                        message=message,
                        suggestion="Optimize for better performance"
                    ))
        
        return issues
    
    def _documentation_scan(self, code: str, language: str) -> List[CodeIssue]:
        """Check documentation quality."""
        issues = []
        
        # Check for TODO comments
        lines = code.split('\n')
        for i, line in enumerate(lines, 1):
            if 'TODO' in line.upper() or 'FIXME' in line.upper():
                issues.append(CodeIssue(
                    line_number=i,
                    column=line.find('TODO') if 'TODO' in line.upper() else line.find('FIXME'),
                    severity=ReviewSeverity.LOW,
                    category=ReviewCategory.DOCUMENTATION,
                    message="TODO or FIXME comment found",
                    suggestion="Address the TODO/FIXME or remove the comment"
                ))
        
        return issues
    
    def _has_docstring(self, node) -> bool:
        """Check if a node has a docstring."""
        if not hasattr(node, 'body') or not node.body:
            return False
        
        first_stmt = node.body[0]
        return (isinstance(first_stmt, ast.Expr) and 
                isinstance(first_stmt.value, ast.Str) and 
                first_stmt.value.s.strip())
    
    def _count_function_lines(self, node, code: str) -> int:
        """Count the number of lines in a function."""
        lines = code.split('\n')
        start_line = node.lineno - 1
        end_line = start_line
        
        # Find the end of the function
        indent_level = len(lines[start_line]) - len(lines[start_line].lstrip())
        
        for i in range(start_line + 1, len(lines)):
            if lines[i].strip() == '':
                continue
            current_indent = len(lines[i]) - len(lines[i].lstrip())
            if current_indent <= indent_level and lines[i].strip():
                break
            end_line = i
        
        return end_line - start_line + 1
    
    def _calculate_score(self, issues: List[CodeIssue]) -> float:
        """Calculate a quality score based on issues."""
        if not issues:
            return 100.0
        
        total_weight = 0
        for issue in issues:
            total_weight += self.severity_weights[issue.severity]
        
        # Convert to score (0-100)
        max_possible_weight = len(issues) * self.severity_weights[ReviewSeverity.CRITICAL]
        score = max(0, 100 - (total_weight / max_possible_weight) * 100)
        
        return round(score, 1)
    
    def _generate_summary(self, issues: List[CodeIssue], score: float) -> str:
        """Generate a summary of the code review."""
        if not issues:
            return f"Code review passed with score {score}/100. No issues found."
        
        severity_counts = self._count_issues_by_severity(issues)
        category_counts = self._count_issues_by_category(issues)
        
        summary_parts = [f"Code review score: {score}/100"]
        
        if severity_counts:
            summary_parts.append("Issues by severity:")
            for severity, count in severity_counts.items():
                summary_parts.append(f"  {severity.value}: {count}")
        
        if category_counts:
            summary_parts.append("Issues by category:")
            for category, count in category_counts.items():
                summary_parts.append(f"  {category.value}: {count}")
        
        return "\n".join(summary_parts)
    
    def _count_issues_by_severity(self, issues: List[CodeIssue]) -> Dict[ReviewSeverity, int]:
        """Count issues by severity."""
        counts = {}
        for issue in issues:
            counts[issue.severity] = counts.get(issue.severity, 0) + 1
        return counts
    
    def _count_issues_by_category(self, issues: List[CodeIssue]) -> Dict[ReviewCategory, int]:
        """Count issues by category."""
        counts = {}
        for issue in issues:
            counts[issue.category] = counts.get(issue.category, 0) + 1
        return counts


# Global instance for easy use
code_reviewer = CodeReviewer()


def review_code(code: str, language: str = "python") -> CodeReviewResult:
    """Review code and return results."""
    return code_reviewer.review_code(code, language) 