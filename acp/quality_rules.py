#!/usr/bin/env python3
"""
Quality Rules Engine - Global CLAUDE.md Standards Enforcement

This module implements the hardcoded quality rules from Global CLAUDE.md and provides
comprehensive code analysis for enforcing quality standards on all agent outputs.

The rules are hardcoded and non-configurable to ensure consistent quality enforcement
across all development activities.
"""

import ast
import re
import time
import hashlib
from typing import Dict, List, Any, Optional, Set, Tuple
from dataclasses import dataclass, field
from pathlib import Path
from enum import Enum


class QualityViolationType(Enum):
    """Types of quality rule violations."""
    FORBIDDEN_PATTERN = "forbidden_pattern"
    STUB_OR_MOCK = "stub_or_mock" 
    PLACEHOLDER = "placeholder"
    TECHNICAL_DEBT = "technical_debt"
    FILE_SIZE = "file_size"
    MISSING_TESTS = "missing_tests"
    POOR_IMPLEMENTATION = "poor_implementation"
    SECURITY_ISSUE = "security_issue"
    PERFORMANCE_ISSUE = "performance_issue"


@dataclass
class QualityViolation:
    """Represents a specific quality rule violation."""
    violation_type: QualityViolationType
    message: str
    file_path: Optional[str] = None
    line_number: Optional[int] = None
    column_number: Optional[int] = None
    severity: str = "error"  # error, warning, info
    rule_id: str = ""
    suggestion: str = ""
    code_snippet: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class QualityReport:
    """Quality analysis report containing all violations and metrics."""
    violations: List[QualityViolation] = field(default_factory=list)
    files_analyzed: int = 0
    lines_analyzed: int = 0
    analysis_time: float = 0.0
    overall_score: float = 0.0
    compliance_percentage: float = 0.0
    timestamp: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Set timestamp if not provided."""
        if not self.timestamp:
            self.timestamp = time.strftime("%Y-%m-%dT%H:%M:%SZ")


class GlobalQualityRules:
    """
    Hardcoded Global CLAUDE.md quality rules - NON-CONFIGURABLE.
    
    These rules are enforced universally across all projects and agents
    to maintain consistent quality standards.
    """
    
    # Core quality requirements (hardcoded from Global CLAUDE.md)
    QUALITY_OVER_SPEED = True
    NO_STUBS_OR_MOCKS = True
    NO_PLACEHOLDERS = True
    COMPREHENSIVE_TESTING = True
    MODULAR_DESIGN_MAX_LINES = 300
    NO_TECHNICAL_DEBT_SHORTCUTS = True
    
    # Forbidden patterns that violate quality standards
    FORBIDDEN_PATTERNS = [
        # Placeholder patterns
        (r'\bTODO\b', "TODO comments indicate incomplete implementation"),
        (r'\bFIXME\b', "FIXME comments indicate broken code"),
        (r'\bHACK\b', "HACK comments indicate poor solutions"),
        (r'\bXXX\b', "XXX comments indicate problematic areas"),
        (r'pass\s*#.*(?:todo|fixme|implement|placeholder)', "Pass with placeholder comment"),
        
        # Mock and stub patterns
        (r'MagicMock', "MagicMock usage violates real implementation requirement"),
        (r'unittest\.mock', "unittest.mock violates real implementation requirement"),
        (r'@mock\.', "Mock decorators violate real implementation requirement"),
        (r'pytest\.mock', "pytest.mock violates real implementation requirement"),
        (r'mock\.patch', "mock.patch violates real implementation requirement"),
        
        # Technical debt patterns
        (r'file_proper', "Bandaid filename indicates technical debt"),
        (r'file_v2', "Version filename indicates technical debt"),
        (r'file_fixed', "Fixed filename indicates technical debt"),
        (r'file_backup', "Backup filename indicates technical debt"),
        (r'file_old', "Old filename indicates technical debt"),
        (r'file_new', "New filename indicates technical debt"),
        (r'_tmp', "Temporary file indicates technical debt"),
        (r'_temp', "Temporary file indicates technical debt"),
        
        # Poor implementation patterns
        (r'return\s+None\s*#.*(?:todo|implement)', "Returning None with TODO"),
        (r'raise\s+NotImplementedError', "NotImplementedError indicates stub"),
        (r'\.\.\.', "Ellipsis indicates incomplete implementation"),
        (r'print\s*\(.*debug', "Debug prints in production code"),
        
        # Security issues
        (r'allow_origins=\[\s*["\']?\*["\']?\s*\]', "CORS wildcard is security risk"),
        (r'bind.*0\.0\.0\.0', "Binding to 0.0.0.0 is security risk"),
        (r'password.*=.*["\'][^"\']+["\']', "Hardcoded password"),
        (r'secret.*=.*["\'][^"\']+["\']', "Hardcoded secret"),
        (r'api_key.*=.*["\'][^"\']+["\']', "Hardcoded API key"),
    ]
    
    # AST node patterns that indicate poor quality
    FORBIDDEN_AST_PATTERNS = [
        ("Pass", "Empty pass statement"),
        ("Ellipsis", "Ellipsis indicates incomplete implementation"),
    ]
    
    # Required patterns for quality code
    REQUIRED_PATTERNS = [
        # Type hints requirement
        (r'def\s+\w+\s*\([^)]*\)\s*->', "Function should have return type annotation"),
        # Docstring requirement for classes
        (r'class\s+\w+.*?:\s*"""', "Class should have docstring"),
        # Exception handling requirement  
        (r'try:\s*\n.*?except\s+Exception', "Generic exception handling should be specific"),
    ]


class CodeAnalyzer:
    """Advanced code analysis engine for quality rule enforcement."""
    
    def __init__(self):
        """Initialize the code analyzer."""
        self.rules = GlobalQualityRules()
        self._compiled_patterns = self._compile_patterns()
    
    def _compile_patterns(self) -> List[Tuple[re.Pattern, str]]:
        """Compile regex patterns for better performance."""
        compiled = []
        for pattern, message in self.rules.FORBIDDEN_PATTERNS:
            try:
                compiled.append((re.compile(pattern, re.IGNORECASE | re.MULTILINE), message))
            except re.error as e:
                # Log pattern compilation error but continue
                print(f"Warning: Failed to compile pattern '{pattern}': {e}")
        return compiled
    
    def analyze_code_content(self, content: str, file_path: str = "") -> List[QualityViolation]:
        """
        Analyze code content for quality violations.
        
        Args:
            content: The code content to analyze
            file_path: Optional file path for context
            
        Returns:
            List of quality violations found
        """
        violations = []
        
        # Pattern-based analysis
        violations.extend(self._analyze_patterns(content, file_path))
        
        # AST-based analysis for Python code
        if file_path.endswith('.py') or self._is_python_code(content):
            violations.extend(self._analyze_ast(content, file_path))
        
        # File size analysis
        violations.extend(self._analyze_file_size(content, file_path))
        
        # Line-by-line analysis
        violations.extend(self._analyze_lines(content, file_path))
        
        return violations
    
    def _analyze_patterns(self, content: str, file_path: str) -> List[QualityViolation]:
        """Analyze content using compiled regex patterns."""
        violations = []
        
        for pattern, message in self._compiled_patterns:
            for match in pattern.finditer(content):
                line_num = content[:match.start()].count('\n') + 1
                col_num = match.start() - content.rfind('\n', 0, match.start())
                
                # Extract code snippet around the match
                lines = content.split('\n')
                start_line = max(0, line_num - 2)
                end_line = min(len(lines), line_num + 1)
                snippet = '\n'.join(lines[start_line:end_line])
                
                violations.append(QualityViolation(
                    violation_type=self._get_violation_type(pattern.pattern),
                    message=message,
                    file_path=file_path,
                    line_number=line_num,
                    column_number=col_num,
                    severity="error",
                    rule_id=f"PATTERN_{hashlib.md5(pattern.pattern.encode()).hexdigest()[:8]}",
                    suggestion=self._get_pattern_suggestion(pattern.pattern),
                    code_snippet=snippet,
                    metadata={"matched_text": match.group()}
                ))
        
        return violations
    
    def _analyze_ast(self, content: str, file_path: str) -> List[QualityViolation]:
        """Analyze Python code using AST parsing."""
        violations = []
        
        try:
            tree = ast.parse(content)
        except SyntaxError as e:
            violations.append(QualityViolation(
                violation_type=QualityViolationType.POOR_IMPLEMENTATION,
                message=f"Syntax error: {e.msg}",
                file_path=file_path,
                line_number=e.lineno,
                column_number=e.offset,
                severity="error",
                rule_id="AST_SYNTAX_ERROR"
            ))
            return violations
        
        # AST visitor for analysis
        visitor = QualityASTVisitor(file_path)
        visitor.visit(tree)
        violations.extend(visitor.violations)
        
        return violations
    
    def _analyze_file_size(self, content: str, file_path: str) -> List[QualityViolation]:
        """Analyze file size against modular design rules."""
        violations = []
        
        lines = content.split('\n')
        line_count = len([line for line in lines if line.strip() and not line.strip().startswith('#')])
        
        if line_count > self.rules.MODULAR_DESIGN_MAX_LINES:
            violations.append(QualityViolation(
                violation_type=QualityViolationType.FILE_SIZE,
                message=f"File exceeds {self.rules.MODULAR_DESIGN_MAX_LINES} line limit: {line_count} lines",
                file_path=file_path,
                severity="error",
                rule_id="FILE_SIZE_LIMIT",
                suggestion=f"Break file into smaller modules (<{self.rules.MODULAR_DESIGN_MAX_LINES} lines each)",
                metadata={"line_count": line_count, "limit": self.rules.MODULAR_DESIGN_MAX_LINES}
            ))
        
        return violations
    
    def _analyze_lines(self, content: str, file_path: str) -> List[QualityViolation]:
        """Analyze individual lines for quality issues."""
        violations = []
        lines = content.split('\n')
        
        for i, line in enumerate(lines, 1):
            # Check for long lines
            if len(line) > 120:
                violations.append(QualityViolation(
                    violation_type=QualityViolationType.POOR_IMPLEMENTATION,
                    message=f"Line too long: {len(line)} characters (max 120)",
                    file_path=file_path,
                    line_number=i,
                    severity="warning",
                    rule_id="LINE_LENGTH",
                    code_snippet=line[:100] + "..." if len(line) > 100 else line
                ))
            
            # Check for trailing whitespace
            if line.rstrip() != line:
                violations.append(QualityViolation(
                    violation_type=QualityViolationType.POOR_IMPLEMENTATION,
                    message="Trailing whitespace",
                    file_path=file_path,
                    line_number=i,
                    severity="info",
                    rule_id="TRAILING_WHITESPACE"
                ))
        
        return violations
    
    def _is_python_code(self, content: str) -> bool:
        """Check if content appears to be Python code."""
        python_indicators = [
            'def ', 'class ', 'import ', 'from ', 'if __name__',
            'async def', 'await ', 'yield ', 'lambda '
        ]
        return any(indicator in content for indicator in python_indicators)
    
    def _get_violation_type(self, pattern: str) -> QualityViolationType:
        """Map regex pattern to violation type."""
        if any(p in pattern.lower() for p in ['todo', 'fixme', 'hack', 'xxx']):
            return QualityViolationType.PLACEHOLDER
        elif any(p in pattern.lower() for p in ['mock', 'patch']):
            return QualityViolationType.STUB_OR_MOCK
        elif any(p in pattern.lower() for p in ['file_', '_tmp', '_temp']):
            return QualityViolationType.TECHNICAL_DEBT
        elif any(p in pattern.lower() for p in ['password', 'secret', 'api_key']):
            return QualityViolationType.SECURITY_ISSUE
        else:
            return QualityViolationType.FORBIDDEN_PATTERN
    
    def _get_pattern_suggestion(self, pattern: str) -> str:
        """Get suggestion for fixing pattern violation."""
        suggestions = {
            r'\bTODO\b': "Complete the implementation instead of leaving TODO",
            r'\bFIXME\b': "Fix the issue instead of leaving FIXME comment",
            r'MagicMock': "Use real implementation with actual objects",
            r'unittest\.mock': "Replace mocks with real implementations and integration tests",
            r'file_proper': "Fix the original file instead of creating versions",
            r'allow_origins=\[\s*["\']?\*["\']?\s*\]': "Restrict CORS to specific origins",
        }
        
        for pat, suggestion in suggestions.items():
            if pat in pattern:
                return suggestion
        
        return "Follow Global CLAUDE.md quality standards"


class QualityASTVisitor(ast.NodeVisitor):
    """AST visitor for analyzing Python code quality."""
    
    def __init__(self, file_path: str):
        """Initialize the AST visitor."""
        self.file_path = file_path
        self.violations: List[QualityViolation] = []
        self.function_count = 0
        self.class_count = 0
    
    def visit_Pass(self, node: ast.Pass) -> None:
        """Visit pass statements to detect stubs."""
        self.violations.append(QualityViolation(
            violation_type=QualityViolationType.STUB_OR_MOCK,
            message="Pass statement indicates incomplete implementation",
            file_path=self.file_path,
            line_number=node.lineno,
            column_number=node.col_offset,
            severity="error",
            rule_id="AST_PASS_STATEMENT",
            suggestion="Implement actual functionality instead of pass"
        ))
        self.generic_visit(node)
    
    def visit_Ellipsis(self, node: ast.Ellipsis) -> None:
        """Visit ellipsis to detect placeholders."""
        self.violations.append(QualityViolation(
            violation_type=QualityViolationType.PLACEHOLDER,
            message="Ellipsis (...) indicates incomplete implementation",
            file_path=self.file_path,
            line_number=node.lineno,
            column_number=node.col_offset,
            severity="error",
            rule_id="AST_ELLIPSIS",
            suggestion="Replace ellipsis with actual implementation"
        ))
        self.generic_visit(node)
    
    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        """Visit function definitions to check quality."""
        self.function_count += 1
        
        # Check for missing return type annotation
        if node.returns is None and node.name != "__init__":
            self.violations.append(QualityViolation(
                violation_type=QualityViolationType.POOR_IMPLEMENTATION,
                message=f"Function '{node.name}' missing return type annotation",
                file_path=self.file_path,
                line_number=node.lineno,
                severity="warning",
                rule_id="AST_MISSING_RETURN_TYPE",
                suggestion="Add return type annotation (-> ReturnType)"
            ))
        
        # Check for missing docstring
        if not ast.get_docstring(node):
            self.violations.append(QualityViolation(
                violation_type=QualityViolationType.POOR_IMPLEMENTATION,
                message=f"Function '{node.name}' missing docstring",
                file_path=self.file_path,
                line_number=node.lineno,
                severity="info",
                rule_id="AST_MISSING_DOCSTRING",
                suggestion="Add descriptive docstring explaining function purpose"
            ))
        
        self.generic_visit(node)
    
    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        """Visit class definitions to check quality."""
        self.class_count += 1
        
        # Check for missing docstring
        if not ast.get_docstring(node):
            self.violations.append(QualityViolation(
                violation_type=QualityViolationType.POOR_IMPLEMENTATION,
                message=f"Class '{node.name}' missing docstring",
                file_path=self.file_path,
                line_number=node.lineno,
                severity="warning",
                rule_id="AST_CLASS_MISSING_DOCSTRING",
                suggestion="Add descriptive docstring explaining class purpose"
            ))
        
        self.generic_visit(node)
    
    def visit_Raise(self, node: ast.Raise) -> None:
        """Visit raise statements to check for stubs."""
        if isinstance(node.exc, ast.Call) and isinstance(node.exc.func, ast.Name):
            if node.exc.func.id == "NotImplementedError":
                self.violations.append(QualityViolation(
                    violation_type=QualityViolationType.STUB_OR_MOCK,
                    message="NotImplementedError indicates incomplete implementation",
                    file_path=self.file_path,
                    line_number=node.lineno,
                    severity="error",
                    rule_id="AST_NOT_IMPLEMENTED",
                    suggestion="Implement actual functionality instead of raising NotImplementedError"
                ))
        
        self.generic_visit(node)


class QualityRulesEngine:
    """
    Main quality rules engine that enforces Global CLAUDE.md standards.
    
    This engine provides comprehensive quality analysis and enforcement
    for all agent outputs and code generation activities.
    """
    
    def __init__(self):
        """Initialize the quality rules engine."""
        self.analyzer = CodeAnalyzer()
        self.rules = GlobalQualityRules()
        self._cache: Dict[str, QualityReport] = {}
    
    def analyze_content(
        self, 
        content: str, 
        file_path: str = "", 
        use_cache: bool = True
    ) -> QualityReport:
        """
        Analyze content for quality violations.
        
        Args:
            content: The content to analyze
            file_path: Optional file path for context
            use_cache: Whether to use cached results
            
        Returns:
            Quality report with violations and metrics
        """
        start_time = time.time()
        
        # Generate cache key
        cache_key = hashlib.sha256(f"{file_path}:{content}".encode()).hexdigest()
        
        if use_cache and cache_key in self._cache:
            return self._cache[cache_key]
        
        # Perform analysis
        violations = self.analyzer.analyze_code_content(content, file_path)
        
        # Calculate metrics
        lines_count = len(content.split('\n'))
        analysis_time = time.time() - start_time
        
        # Calculate compliance percentage
        total_checks = len(self.rules.FORBIDDEN_PATTERNS) + 10  # Base checks
        violation_count = len(violations)
        compliance = max(0.0, (total_checks - violation_count) / total_checks * 100.0)
        
        # Calculate overall score (0-100)
        score = self._calculate_quality_score(violations, lines_count)
        
        report = QualityReport(
            violations=violations,
            files_analyzed=1,
            lines_analyzed=lines_count,
            analysis_time=analysis_time,
            overall_score=score,
            compliance_percentage=compliance,
            metadata={
                "cache_key": cache_key,
                "rules_version": "global_claude_md_v1",
                "analyzer_version": "1.0.0"
            }
        )
        
        # Cache the report
        if use_cache:
            self._cache[cache_key] = report
        
        return report
    
    def analyze_file(self, file_path: str) -> QualityReport:
        """
        Analyze a file for quality violations.
        
        Args:
            file_path: Path to the file to analyze
            
        Returns:
            Quality report with violations and metrics
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            return self.analyze_content(content, file_path)
        except (IOError, UnicodeDecodeError) as e:
            # Return report with file reading error
            return QualityReport(
                violations=[QualityViolation(
                    violation_type=QualityViolationType.POOR_IMPLEMENTATION,
                    message=f"Cannot read file: {e}",
                    file_path=file_path,
                    severity="error",
                    rule_id="FILE_READ_ERROR"
                )],
                files_analyzed=0,
                metadata={"error": str(e)}
            )
    
    def analyze_directory(self, directory_path: str, extensions: Set[str] = None) -> QualityReport:
        """
        Analyze all files in a directory.
        
        Args:
            directory_path: Path to directory to analyze
            extensions: Set of file extensions to analyze (default: .py, .js, .ts, .go, .rs)
            
        Returns:
            Aggregated quality report
        """
        if extensions is None:
            extensions = {'.py', '.js', '.ts', '.go', '.rs', '.java', '.cpp', '.c', '.h'}
        
        start_time = time.time()
        all_violations = []
        files_analyzed = 0
        total_lines = 0
        
        directory = Path(directory_path)
        
        for file_path in directory.rglob('*'):
            if file_path.is_file() and file_path.suffix in extensions:
                report = self.analyze_file(str(file_path))
                all_violations.extend(report.violations)
                files_analyzed += 1
                total_lines += report.lines_analyzed
        
        analysis_time = time.time() - start_time
        
        # Calculate aggregated metrics
        if files_analyzed > 0:
            score = sum(self._calculate_quality_score([], 100) for _ in range(files_analyzed)) / files_analyzed
            if all_violations:
                score = max(0.0, score - len(all_violations) * 2.0)  # Penalty for violations
        else:
            score = 0.0
        
        total_checks = len(self.rules.FORBIDDEN_PATTERNS) * files_analyzed
        violation_count = len(all_violations)
        compliance = max(0.0, (total_checks - violation_count) / total_checks * 100.0) if total_checks > 0 else 100.0
        
        return QualityReport(
            violations=all_violations,
            files_analyzed=files_analyzed,
            lines_analyzed=total_lines,
            analysis_time=analysis_time,
            overall_score=score,
            compliance_percentage=compliance,
            metadata={
                "directory": directory_path,
                "extensions_analyzed": list(extensions),
                "total_files_in_directory": len(list(directory.rglob('*')))
            }
        )
    
    def is_compliant(self, content: str, file_path: str = "") -> bool:
        """
        Check if content is compliant with quality rules.
        
        Args:
            content: The content to check
            file_path: Optional file path for context
            
        Returns:
            True if compliant, False otherwise
        """
        report = self.analyze_content(content, file_path)
        
        # Check for any error-level violations
        error_violations = [v for v in report.violations if v.severity == "error"]
        
        return len(error_violations) == 0 and report.compliance_percentage >= 95.0
    
    def get_compliance_summary(self, report: QualityReport) -> Dict[str, Any]:
        """
        Get a summary of compliance status.
        
        Args:
            report: Quality report to summarize
            
        Returns:
            Compliance summary dictionary
        """
        violations_by_type = {}
        violations_by_severity = {}
        
        for violation in report.violations:
            # Group by type
            vtype = violation.violation_type.value
            violations_by_type[vtype] = violations_by_type.get(vtype, 0) + 1
            
            # Group by severity
            severity = violation.severity
            violations_by_severity[severity] = violations_by_severity.get(severity, 0) + 1
        
        return {
            "compliance_percentage": report.compliance_percentage,
            "overall_score": report.overall_score,
            "total_violations": len(report.violations),
            "violations_by_type": violations_by_type,
            "violations_by_severity": violations_by_severity,
            "files_analyzed": report.files_analyzed,
            "lines_analyzed": report.lines_analyzed,
            "analysis_time": report.analysis_time,
            "is_compliant": self._is_report_compliant(report),
            "critical_issues": [
                v.message for v in report.violations 
                if v.severity == "error" and v.violation_type in [
                    QualityViolationType.STUB_OR_MOCK,
                    QualityViolationType.PLACEHOLDER,
                    QualityViolationType.TECHNICAL_DEBT
                ]
            ]
        }
    
    def _calculate_quality_score(self, violations: List[QualityViolation], lines_count: int) -> float:
        """Calculate quality score (0-100) based on violations and code metrics."""
        base_score = 100.0
        
        # Penalty for violations by severity
        for violation in violations:
            if violation.severity == "error":
                base_score -= 10.0
            elif violation.severity == "warning":
                base_score -= 3.0
            elif violation.severity == "info":
                base_score -= 1.0
        
        # Bonus for good code metrics
        if lines_count > 0:
            # Penalty for very short files (likely stubs)
            if lines_count < 20:
                base_score -= 15.0
            # Penalty for very long files (violates modularity)
            elif lines_count > self.rules.MODULAR_DESIGN_MAX_LINES:
                base_score -= 20.0
        
        return max(0.0, min(100.0, base_score))
    
    def _is_report_compliant(self, report: QualityReport) -> bool:
        """Check if a quality report indicates compliance."""
        error_violations = [v for v in report.violations if v.severity == "error"]
        return len(error_violations) == 0 and report.compliance_percentage >= 95.0
    
    def clear_cache(self) -> None:
        """Clear the analysis cache."""
        self._cache.clear()
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            "cache_size": len(self._cache),
            "cache_keys": list(self._cache.keys())[:10]  # Show first 10 keys
        }


# Factory function for creating quality rules engine
def create_quality_rules_engine() -> QualityRulesEngine:
    """
    Create a new quality rules engine instance.
    
    Returns:
        Configured QualityRulesEngine instance
    """
    return QualityRulesEngine()


# Utility functions for common quality checks
def check_content_quality(content: str, file_path: str = "") -> bool:
    """
    Quick quality check for content compliance.
    
    Args:
        content: Content to check
        file_path: Optional file path
        
    Returns:
        True if content meets quality standards
    """
    engine = create_quality_rules_engine()
    return engine.is_compliant(content, file_path)


def get_quality_violations(content: str, file_path: str = "") -> List[QualityViolation]:
    """
    Get quality violations for content.
    
    Args:
        content: Content to analyze
        file_path: Optional file path
        
    Returns:
        List of quality violations
    """
    engine = create_quality_rules_engine()
    report = engine.analyze_content(content, file_path)
    return report.violations


def format_quality_report(report: QualityReport) -> str:
    """
    Format quality report as human-readable string.
    
    Args:
        report: Quality report to format
        
    Returns:
        Formatted report string
    """
    lines = []
    lines.append("=" * 60)
    lines.append("QUALITY ANALYSIS REPORT")
    lines.append("=" * 60)
    lines.append(f"Overall Score: {report.overall_score:.1f}/100")
    lines.append(f"Compliance: {report.compliance_percentage:.1f}%")
    lines.append(f"Files Analyzed: {report.files_analyzed}")
    lines.append(f"Lines Analyzed: {report.lines_analyzed}")
    lines.append(f"Analysis Time: {report.analysis_time:.2f}s")
    lines.append(f"Violations Found: {len(report.violations)}")
    lines.append("")
    
    if report.violations:
        lines.append("VIOLATIONS:")
        lines.append("-" * 40)
        for i, violation in enumerate(report.violations, 1):
            lines.append(f"{i}. [{violation.severity.upper()}] {violation.message}")
            if violation.file_path:
                lines.append(f"   File: {violation.file_path}")
            if violation.line_number:
                lines.append(f"   Line: {violation.line_number}")
            if violation.suggestion:
                lines.append(f"   Suggestion: {violation.suggestion}")
            lines.append("")
    else:
        lines.append("✅ No violations found - Code meets quality standards!")
    
    return "\n".join(lines)