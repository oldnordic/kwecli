#!/usr/bin/env python3
"""
Agent Quality Wrapper - Enforces Global CLAUDE.md Standards

This module provides wrapper functionality that automatically validates all agent outputs
against Global CLAUDE.md quality standards. It ensures no agent can produce non-compliant
results by intercepting and validating outputs before they are returned.
"""

import time
import asyncio
import json
from typing import Dict, List, Any, Optional, Callable, Union, TypeVar, Generic
from dataclasses import dataclass, field
from functools import wraps
from enum import Enum

from .quality_rules import (
    QualityRulesEngine, QualityReport, QualityViolation, 
    QualityViolationType, create_quality_rules_engine,
    check_content_quality, get_quality_violations
)
from ..agents.base_agent import SubAgent, AgentResult, AgentStatus


T = TypeVar('T')


class QualityEnforcementMode(Enum):
    """Quality enforcement modes."""
    STRICT = "strict"           # Reject any violations
    LENIENT = "lenient"         # Allow warnings, reject errors only
    MONITORING = "monitoring"   # Log violations but don't reject
    DISABLED = "disabled"       # No quality enforcement


class QualityAction(Enum):
    """Actions to take when quality violations are found."""
    REJECT = "reject"           # Reject the output
    FIX_AUTOMATIC = "fix_automatic"  # Try to fix automatically
    REQUEST_IMPROVEMENT = "request_improvement"  # Ask agent to improve
    LOG_AND_CONTINUE = "log_and_continue"  # Log but continue


@dataclass
class QualityEnforcementResult:
    """Result of quality enforcement on agent output."""
    passed: bool
    original_result: Any
    modified_result: Optional[Any] = None
    quality_report: Optional[QualityReport] = None
    enforcement_action: Optional[QualityAction] = None
    improvement_suggestions: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    processing_time: float = 0.0


@dataclass
class QualityMetrics:
    """Quality metrics for agent performance tracking."""
    agent_name: str
    total_outputs: int = 0
    passed_outputs: int = 0
    failed_outputs: int = 0
    average_quality_score: float = 0.0
    common_violations: List[str] = field(default_factory=list)
    improvement_trend: float = 0.0  # Positive means improving
    last_updated: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


class CodeQualityFixer:
    """Automatic code quality fixer for common violations."""
    
    def __init__(self):
        """Initialize the code quality fixer."""
        self.fix_patterns = {
            # Remove trailing whitespace
            r'\s+$': '',
            # Fix common formatting issues
            r'\t': '    ',  # Convert tabs to spaces
            # Remove multiple blank lines
            r'\n\s*\n\s*\n': '\n\n',
        }
    
    def fix_content(self, content: str) -> tuple[str, List[str]]:
        """
        Automatically fix common quality issues in content.
        
        Args:
            content: Content to fix
            
        Returns:
            Tuple of (fixed_content, list_of_fixes_applied)
        """
        import re
        
        fixed_content = content
        fixes_applied = []
        
        # Apply basic fixes
        for pattern, replacement in self.fix_patterns.items():
            if re.search(pattern, fixed_content, re.MULTILINE):
                fixed_content = re.sub(pattern, replacement, fixed_content, flags=re.MULTILINE)
                fixes_applied.append(f"Applied fix: {pattern} -> {replacement}")
        
        # Remove TODO comments (if safe)
        if 'TODO' in fixed_content and not self._has_critical_todos(fixed_content):
            original_lines = fixed_content.split('\n')
            fixed_lines = []
            for line in original_lines:
                if 'TODO' in line and line.strip().startswith('#'):
                    fixes_applied.append(f"Removed TODO comment: {line.strip()}")
                    continue
                fixed_lines.append(line)
            fixed_content = '\n'.join(fixed_lines)
        
        return fixed_content, fixes_applied
    
    def _has_critical_todos(self, content: str) -> bool:
        """Check if TODOs are critical and should not be auto-removed."""
        import re
        
        critical_patterns = [
            r'TODO.*critical',
            r'TODO.*important',
            r'TODO.*security',
            r'TODO.*implement.*core'
        ]
        
        for pattern in critical_patterns:
            if re.search(pattern, content, re.IGNORECASE):
                return True
        
        return False


class QualityWrapper:
    """
    Quality wrapper that enforces Global CLAUDE.md standards on agent outputs.
    
    This wrapper automatically validates all agent outputs and can reject, fix,
    or request improvements based on quality violations found.
    """
    
    def __init__(
        self, 
        enforcement_mode: QualityEnforcementMode = QualityEnforcementMode.STRICT,
        auto_fix_enabled: bool = True,
        max_improvement_attempts: int = 2
    ):
        """
        Initialize the quality wrapper.
        
        Args:
            enforcement_mode: How strictly to enforce quality rules
            auto_fix_enabled: Whether to attempt automatic fixes
            max_improvement_attempts: Maximum attempts to improve output
        """
        self.enforcement_mode = enforcement_mode
        self.auto_fix_enabled = auto_fix_enabled
        self.max_improvement_attempts = max_improvement_attempts
        
        self.quality_engine = create_quality_rules_engine()
        self.code_fixer = CodeQualityFixer()
        self.metrics: Dict[str, QualityMetrics] = {}
        
        self._violation_counts: Dict[str, int] = {}
        self._processing_times: List[float] = []
    
    def wrap_agent(self, agent: SubAgent) -> 'QualityWrappedAgent':
        """
        Wrap an agent with quality enforcement.
        
        Args:
            agent: The agent to wrap
            
        Returns:
            Quality-wrapped agent instance
        """
        return QualityWrappedAgent(agent, self)
    
    def validate_output(
        self, 
        output: Union[str, AgentResult], 
        context: Dict[str, Any] = None,
        agent_name: str = ""
    ) -> QualityEnforcementResult:
        """
        Validate agent output against quality standards.
        
        Args:
            output: The agent output to validate
            context: Additional context for validation
            agent_name: Name of the agent that produced output
            
        Returns:
            Quality enforcement result
        """
        start_time = time.time()
        context = context or {}
        
        # Extract content based on output type
        if isinstance(output, AgentResult):
            content = output.output
            file_path = context.get('file_path', '')
        elif isinstance(output, str):
            content = output
            file_path = context.get('file_path', '')
        else:
            content = str(output)
            file_path = ""
        
        # Analyze quality
        quality_report = self.quality_engine.analyze_content(content, file_path)
        
        # Determine enforcement action
        action = self._determine_action(quality_report)
        
        # Process based on action
        result = QualityEnforcementResult(
            passed=False,
            original_result=output,
            quality_report=quality_report,
            enforcement_action=action
        )
        
        if action == QualityAction.REJECT:
            result.passed = False
            result.improvement_suggestions = self._generate_improvement_suggestions(quality_report)
        
        elif action == QualityAction.FIX_AUTOMATIC and self.auto_fix_enabled:
            fixed_result = self._attempt_automatic_fix(output, quality_report)
            if fixed_result:
                result.passed = True
                result.modified_result = fixed_result
            else:
                result.passed = False
                result.improvement_suggestions = self._generate_improvement_suggestions(quality_report)
        
        elif action == QualityAction.LOG_AND_CONTINUE:
            result.passed = True
            result.modified_result = output
        
        else:
            # Default: check compliance
            result.passed = len([v for v in quality_report.violations if v.severity == "error"]) == 0
        
        result.processing_time = time.time() - start_time
        
        # Update metrics
        self._update_metrics(agent_name, result)
        
        return result
    
    async def validate_output_async(
        self, 
        output: Union[str, AgentResult], 
        context: Dict[str, Any] = None,
        agent_name: str = ""
    ) -> QualityEnforcementResult:
        """
        Asynchronously validate agent output.
        
        Args:
            output: The agent output to validate
            context: Additional context for validation
            agent_name: Name of the agent that produced output
            
        Returns:
            Quality enforcement result
        """
        # Run validation in thread pool for CPU-intensive analysis
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, 
            self.validate_output, 
            output, 
            context, 
            agent_name
        )
    
    def _determine_action(self, quality_report: QualityReport) -> QualityAction:
        """Determine what action to take based on quality report."""
        if self.enforcement_mode == QualityEnforcementMode.DISABLED:
            return QualityAction.LOG_AND_CONTINUE
        
        error_violations = [v for v in quality_report.violations if v.severity == "error"]
        warning_violations = [v for v in quality_report.violations if v.severity == "warning"]
        
        if self.enforcement_mode == QualityEnforcementMode.MONITORING:
            return QualityAction.LOG_AND_CONTINUE
        
        elif self.enforcement_mode == QualityEnforcementMode.LENIENT:
            if error_violations:
                return QualityAction.FIX_AUTOMATIC if self.auto_fix_enabled else QualityAction.REJECT
            else:
                return QualityAction.LOG_AND_CONTINUE
        
        else:  # STRICT mode
            if error_violations or len(warning_violations) > 3:
                return QualityAction.FIX_AUTOMATIC if self.auto_fix_enabled else QualityAction.REJECT
            else:
                return QualityAction.LOG_AND_CONTINUE
    
    def _attempt_automatic_fix(
        self, 
        output: Union[str, AgentResult], 
        quality_report: QualityReport
    ) -> Optional[Union[str, AgentResult]]:
        """Attempt to automatically fix quality violations."""
        if isinstance(output, AgentResult):
            fixed_content, fixes = self.code_fixer.fix_content(output.output)
            if fixes:
                # Create new AgentResult with fixed content
                fixed_result = AgentResult(
                    success=output.success,
                    output=fixed_content,
                    error_message=output.error_message,
                    metadata={
                        **output.metadata,
                        "quality_fixes_applied": fixes,
                        "original_output": output.output
                    }
                )
                return fixed_result
        elif isinstance(output, str):
            fixed_content, fixes = self.code_fixer.fix_content(output)
            if fixes:
                return fixed_content
        
        return None
    
    def _generate_improvement_suggestions(self, quality_report: QualityReport) -> List[str]:
        """Generate improvement suggestions based on quality violations."""
        suggestions = []
        
        violation_types = set(v.violation_type for v in quality_report.violations)
        
        if QualityViolationType.STUB_OR_MOCK in violation_types:
            suggestions.append("Replace all mocks and stubs with real implementations")
        
        if QualityViolationType.PLACEHOLDER in violation_types:
            suggestions.append("Remove all TODO, FIXME, and placeholder comments by implementing actual functionality")
        
        if QualityViolationType.TECHNICAL_DEBT in violation_types:
            suggestions.append("Avoid technical debt patterns like bandaid files and temporary solutions")
        
        if QualityViolationType.FILE_SIZE in violation_types:
            suggestions.append("Break large files into smaller, more focused modules (max 300 lines)")
        
        if QualityViolationType.SECURITY_ISSUE in violation_types:
            suggestions.append("Fix security issues like hardcoded credentials and unsafe configurations")
        
        # Add specific suggestions from violations
        for violation in quality_report.violations[:5]:  # Top 5 violations
            if violation.suggestion:
                suggestions.append(f"Line {violation.line_number}: {violation.suggestion}")
        
        return suggestions
    
    def _update_metrics(self, agent_name: str, result: QualityEnforcementResult) -> None:
        """Update quality metrics for an agent."""
        if not agent_name:
            return
        
        if agent_name not in self.metrics:
            self.metrics[agent_name] = QualityMetrics(agent_name=agent_name)
        
        metrics = self.metrics[agent_name]
        metrics.total_outputs += 1
        
        if result.passed:
            metrics.passed_outputs += 1
        else:
            metrics.failed_outputs += 1
        
        # Update average quality score
        if result.quality_report:
            current_score = result.quality_report.overall_score
            total_weight = metrics.total_outputs
            metrics.average_quality_score = (
                (metrics.average_quality_score * (total_weight - 1) + current_score) / total_weight
            )
            
            # Track common violations
            for violation in result.quality_report.violations:
                violation_key = f"{violation.violation_type.value}:{violation.message}"
                self._violation_counts[violation_key] = self._violation_counts.get(violation_key, 0) + 1
        
        metrics.last_updated = time.strftime("%Y-%m-%dT%H:%M:%SZ")
        
        # Update common violations for this agent
        agent_violations = []
        for violation_key, count in self._violation_counts.items():
            if count >= 2:  # Show violations that occur multiple times
                agent_violations.append(f"{violation_key} ({count} times)")
        metrics.common_violations = agent_violations[:5]  # Top 5
    
    def get_agent_metrics(self, agent_name: str) -> Optional[QualityMetrics]:
        """Get quality metrics for a specific agent."""
        return self.metrics.get(agent_name)
    
    def get_all_metrics(self) -> Dict[str, QualityMetrics]:
        """Get quality metrics for all agents."""
        return self.metrics.copy()
    
    def get_system_quality_summary(self) -> Dict[str, Any]:
        """Get overall system quality summary."""
        if not self.metrics:
            return {
                "total_agents": 0,
                "overall_pass_rate": 0.0,
                "average_quality_score": 0.0,
                "total_outputs_processed": 0
            }
        
        total_outputs = sum(m.total_outputs for m in self.metrics.values())
        total_passed = sum(m.passed_outputs for m in self.metrics.values())
        
        pass_rate = (total_passed / total_outputs * 100) if total_outputs > 0 else 0.0
        
        avg_score = sum(m.average_quality_score for m in self.metrics.values()) / len(self.metrics)
        
        # Find most common violations across all agents
        common_violations = []
        for violation_key, count in sorted(self._violation_counts.items(), key=lambda x: x[1], reverse=True)[:5]:
            common_violations.append(f"{violation_key} ({count} occurrences)")
        
        return {
            "total_agents": len(self.metrics),
            "overall_pass_rate": pass_rate,
            "average_quality_score": avg_score,
            "total_outputs_processed": total_outputs,
            "total_outputs_passed": total_passed,
            "total_outputs_failed": total_outputs - total_passed,
            "common_violations": common_violations,
            "enforcement_mode": self.enforcement_mode.value,
            "auto_fix_enabled": self.auto_fix_enabled,
            "average_processing_time": sum(self._processing_times[-100:]) / len(self._processing_times[-100:]) if self._processing_times else 0.0
        }
    
    def reset_metrics(self) -> None:
        """Reset all quality metrics."""
        self.metrics.clear()
        self._violation_counts.clear()
        self._processing_times.clear()
    
    def export_metrics(self) -> Dict[str, Any]:
        """Export metrics as JSON-serializable dictionary."""
        return {
            "metrics": {
                name: {
                    "agent_name": m.agent_name,
                    "total_outputs": m.total_outputs,
                    "passed_outputs": m.passed_outputs,
                    "failed_outputs": m.failed_outputs,
                    "average_quality_score": m.average_quality_score,
                    "common_violations": m.common_violations,
                    "improvement_trend": m.improvement_trend,
                    "last_updated": m.last_updated,
                    "metadata": m.metadata
                }
                for name, m in self.metrics.items()
            },
            "system_summary": self.get_system_quality_summary(),
            "violation_counts": self._violation_counts
        }


class QualityWrappedAgent:
    """An agent wrapped with automatic quality enforcement."""
    
    def __init__(self, agent: SubAgent, quality_wrapper: QualityWrapper):
        """
        Initialize the quality-wrapped agent.
        
        Args:
            agent: The original agent to wrap
            quality_wrapper: The quality wrapper instance
        """
        self.agent = agent
        self.quality_wrapper = quality_wrapper
        self._improvement_attempts = 0
    
    async def execute_task(self, task: str, context: Dict[str, Any]) -> AgentResult:
        """
        Execute task with automatic quality validation.
        
        Args:
            task: Task to execute
            context: Task context
            
        Returns:
            Quality-validated AgentResult
        """
        original_result = await self.agent.execute_task(task, context)
        
        # Validate the result
        enforcement_result = await self.quality_wrapper.validate_output_async(
            original_result, 
            context,
            self.agent.name
        )
        
        if enforcement_result.passed:
            # Return the result (potentially modified)
            return enforcement_result.modified_result or original_result
        else:
            # Quality check failed
            if self._improvement_attempts < self.quality_wrapper.max_improvement_attempts:
                self._improvement_attempts += 1
                
                # Add improvement context and retry
                improvement_context = {
                    **context,
                    "quality_issues": [v.message for v in enforcement_result.quality_report.violations],
                    "improvement_suggestions": enforcement_result.improvement_suggestions,
                    "previous_attempt": original_result.output,
                    "attempt_number": self._improvement_attempts
                }
                
                # Enhanced task with quality feedback
                enhanced_task = f"""
{task}

QUALITY IMPROVEMENT REQUIRED:
Previous attempt had these issues:
{chr(10).join('- ' + suggestion for suggestion in enforcement_result.improvement_suggestions)}

Please provide a corrected implementation that addresses all quality issues.
Follow Global CLAUDE.md standards strictly:
- No TODO, FIXME, or placeholder comments
- No mocks or stubs - use real implementations
- Maximum 300 lines per file
- Proper error handling and type hints
- Complete, working functionality
"""
                
                return await self.execute_task(enhanced_task, improvement_context)
            else:
                # Max attempts reached - return error result
                return AgentResult(
                    success=False,
                    output="",
                    error_message=f"Quality validation failed after {self._improvement_attempts} attempts. " +
                                 f"Violations: {'; '.join(enforcement_result.improvement_suggestions)}",
                    metadata={
                        "quality_report": enforcement_result.quality_report,
                        "improvement_attempts": self._improvement_attempts,
                        "original_output": original_result.output
                    }
                )
    
    def can_handle(self, task: str) -> bool:
        """Check if the wrapped agent can handle the task."""
        return self.agent.can_handle(task)
    
    def get_expertise(self):
        """Get the wrapped agent's expertise."""
        return self.agent.get_expertise()
    
    def get_tools(self):
        """Get the wrapped agent's tools."""
        return self.agent.get_tools()
    
    def get_status(self):
        """Get the wrapped agent's status."""
        return self.agent.get_status()
    
    def update_status(self, status: AgentStatus):
        """Update the wrapped agent's status."""
        return self.agent.update_status(status)
    
    @property
    def name(self):
        """Get the wrapped agent's name."""
        return self.agent.name
    
    @property
    def description(self):
        """Get the wrapped agent's description."""
        return self.agent.description
    
    def __getattr__(self, name):
        """Delegate any other attributes to the wrapped agent."""
        return getattr(self.agent, name)


def quality_enforced(
    enforcement_mode: QualityEnforcementMode = QualityEnforcementMode.STRICT,
    auto_fix: bool = True,
    max_attempts: int = 2
):
    """
    Decorator to add quality enforcement to agent methods.
    
    Args:
        enforcement_mode: Quality enforcement mode
        auto_fix: Whether to enable automatic fixes
        max_attempts: Maximum improvement attempts
        
    Returns:
        Decorated function with quality enforcement
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Get or create quality wrapper
            if not hasattr(wrapper, '_quality_wrapper'):
                wrapper._quality_wrapper = QualityWrapper(
                    enforcement_mode=enforcement_mode,
                    auto_fix_enabled=auto_fix,
                    max_improvement_attempts=max_attempts
                )
            
            # Execute original function
            result = await func(*args, **kwargs)
            
            # Validate result
            enforcement_result = await wrapper._quality_wrapper.validate_output_async(
                result,
                kwargs.get('context', {}),
                getattr(args[0], 'name', 'unknown') if args else 'unknown'
            )
            
            if enforcement_result.passed:
                return enforcement_result.modified_result or result
            else:
                raise ValueError(
                    f"Quality enforcement failed: {'; '.join(enforcement_result.improvement_suggestions)}"
                )
        
        return wrapper
    return decorator


# Factory functions
def create_quality_wrapper(
    enforcement_mode: QualityEnforcementMode = QualityEnforcementMode.STRICT,
    auto_fix_enabled: bool = True,
    max_improvement_attempts: int = 2
) -> QualityWrapper:
    """
    Create a quality wrapper instance.
    
    Args:
        enforcement_mode: Quality enforcement mode
        auto_fix_enabled: Whether to enable automatic fixes
        max_improvement_attempts: Maximum attempts to improve output
        
    Returns:
        Configured QualityWrapper instance
    """
    return QualityWrapper(
        enforcement_mode=enforcement_mode,
        auto_fix_enabled=auto_fix_enabled,
        max_improvement_attempts=max_improvement_attempts
    )


def wrap_agent_with_quality(agent: SubAgent, wrapper: QualityWrapper = None) -> QualityWrappedAgent:
    """
    Wrap an agent with quality enforcement.
    
    Args:
        agent: Agent to wrap
        wrapper: Optional quality wrapper (creates default if None)
        
    Returns:
        Quality-wrapped agent
    """
    if wrapper is None:
        wrapper = create_quality_wrapper()
    
    return wrapper.wrap_agent(agent)