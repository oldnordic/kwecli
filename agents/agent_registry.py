#!/usr/bin/env python3
"""
Agent Registry System

This module provides the agent registry system that manages sub-agents dynamically.
It handles agent registration, task assignment, load balancing, and persistence.
"""

import json
import os
import time
from typing import Dict, List, Any, Optional, Type
from dataclasses import dataclass
from pathlib import Path

from agents.base_agent import SubAgent, AgentResult, AgentStatus, AgentExpertise


class AgentRegistryError(Exception):
    """Base exception for agent registry errors."""
    pass


class AgentNotFoundError(AgentRegistryError):
    """Raised when an agent is not found in the registry."""
    pass


class AgentAlreadyRegisteredError(AgentRegistryError):
    """Raised when trying to register an agent with a duplicate name."""
    pass


class QualityEnforcementError(AgentRegistryError):
    """Raised when agent output fails quality enforcement."""
    pass


@dataclass
class TaskAssignmentResult:
    """Result of a task assignment operation."""
    success: bool
    assigned_agent: Optional[SubAgent] = None
    task_output: str = ""
    error_message: str = ""
    execution_time: float = 0.0
    metadata: Dict[str, Any] = None
    quality_passed: bool = True
    quality_score: float = 100.0
    quality_violations: List[str] = None
    improvement_attempts: int = 0

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
        if self.quality_violations is None:
            self.quality_violations = []


class AgentRegistry:
    """Registry for managing sub-agents dynamically with quality enforcement and discovery."""

    def discover_agents(self):
        """Dynamically discover and register all SubAgent subclasses in agents/ directory."""
        import importlib, pkgutil
        from agents.base_agent import SubAgent
        package = 'agents'
        for finder, name, ispkg in pkgutil.iter_modules(path=importlib.import_module(package).__path__):
            module = importlib.import_module(f'{package}.{name}')
            for attr in dir(module):
                obj = getattr(module, attr)
                if isinstance(obj, type) and issubclass(obj, SubAgent) and obj is not SubAgent:
                    try:
                        agent = obj()
                        self.register_agent(agent)
                    except Exception:
                        pass

    def __init__(self, enforce_quality: bool = True, max_improvement_attempts: int = 2):
        """Initialize the agent registry.
        
        Args:
            enforce_quality: Whether to enforce Global CLAUDE.md quality standards
            max_improvement_attempts: Maximum attempts to improve agent output
        """
        self._agents: Dict[str, SubAgent] = {}
        self.enforce_quality = enforce_quality
        self.max_improvement_attempts = max_improvement_attempts
        self._quality_metrics: Dict[str, Dict[str, Any]] = {}
        self._quality_engine = None
        
        # Discover and register built-in agents
        self.discover_agents()
        
        # Initialize quality engine if enforcement is enabled
        if self.enforce_quality:
            self._initialize_quality_engine()

    def register_agent(self, agent: SubAgent) -> None:
        """Register an agent with the registry.
        
        Args:
            agent: The agent to register
            
        Raises:
            AgentAlreadyRegisteredError: If an agent with the same name is already registered
        """
        if agent.name in self._agents:
            raise AgentAlreadyRegisteredError(
                f"Agent '{agent.name}' is already registered"
            )
        
        # Validate agent quality if enforcement is enabled
        if self.enforce_quality:
            self._validate_agent_quality(agent)
        
        self._agents[agent.name] = agent
        
        # Initialize quality metrics for this agent
        if self.enforce_quality:
            self._initialize_agent_metrics(agent.name)

    def unregister_agent(self, agent_name: str) -> None:
        """Unregister an agent from the registry.
        
        Args:
            agent_name: Name of the agent to unregister
            
        Raises:
            AgentNotFoundError: If the agent is not found
        """
        if agent_name not in self._agents:
            raise AgentNotFoundError(f"Agent '{agent_name}' not found")
        
        del self._agents[agent_name]

    def remove_agent(self, agent_name: str) -> None:
        """Remove an agent from the registry (alias for unregister_agent).
        
        Args:
            agent_name: Name of the agent to remove
            
        Raises:
            AgentNotFoundError: If the agent is not found
        """
        self.unregister_agent(agent_name)

    def get_agent(self, agent_name: str) -> SubAgent:
        """Get an agent by name.
        
        Args:
            agent_name: Name of the agent to retrieve
            
        Returns:
            The agent instance
            
        Raises:
            AgentNotFoundError: If the agent is not found
        """
        if agent_name not in self._agents:
            raise AgentNotFoundError(f"Agent '{agent_name}' not found")
        
        return self._agents[agent_name]

    def get_all_agents(self) -> List[SubAgent]:
        """Get all registered agents.
        
        Returns:
            List of all registered agents
        """
        return list(self._agents.values())

    def get_agent_names(self) -> List[str]:
        """Get names of all registered agents.
        
        Returns:
            List of agent names
        """
        return list(self._agents.keys())

    def get_agent_count(self) -> int:
        """Get the number of registered agents.
        
        Returns:
            Number of registered agents
        """
        return len(self._agents)

    def clear(self) -> None:
        """Clear all agents from the registry."""
        self._agents.clear()

    def find_agents_by_expertise(self, expertise: AgentExpertise) -> List[SubAgent]:
        """Find agents with a specific expertise.
        
        Args:
            expertise: The expertise to search for
            
        Returns:
            List of agents with the specified expertise
        """
        return [
            agent for agent in self._agents.values()
            if expertise in agent.get_expertise()
        ]

    def get_best_agent_for_task(self, task: str) -> Optional[SubAgent]:
        """Find the best agent for a given task.
        
        Args:
            task: Description of the task
            
        Returns:
            The best agent for the task, or None if no suitable agent found
        """
        available_agents = self.get_available_agents()
        
        if not available_agents:
            return None
        
        best_agent = None
        best_score = 0.0
        
        for agent in available_agents:
            if agent.can_handle(task):
                from agents.base_agent import get_agent_expertise_match
                score = get_agent_expertise_match(agent, task)
                
                if score > best_score:
                    best_score = score
                    best_agent = agent
        
        return best_agent

    def get_agent_by_expertise(self, expertise_name: str) -> List[SubAgent]:
        """Find agents by expertise name (alias for find_agents_by_expertise).
        
        Args:
            expertise_name: The expertise name to search for
            
        Returns:
            List of agents with the specified expertise
        """
        # Convert string to AgentExpertise enum
        try:
            expertise = AgentExpertise(expertise_name.lower())
            return self.find_agents_by_expertise(expertise)
        except ValueError:
            # If not a valid enum value, search by string matching
            return [
                agent for agent in self._agents.values()
                if any(exp.value.lower() == expertise_name.lower() 
                      for exp in agent.get_expertise())
            ]

    def find_agents_by_tool(self, tool: str) -> List[SubAgent]:
        """Find agents that use a specific tool.
        
        Args:
            tool: The tool to search for
            
        Returns:
            List of agents that use the specified tool
        """
        return [
            agent for agent in self._agents.values()
            if tool in agent.get_tools()
        ]

    def get_agent_by_tool(self, tool: str) -> List[SubAgent]:
        """Find agents by tool (alias for find_agents_by_tool).
        
        Args:
            tool: The tool to search for
            
        Returns:
            List of agents that use the specified tool
        """
        return self.find_agents_by_tool(tool)

    def find_agents_for_task(self, task: str) -> List[SubAgent]:
        """Find agents that can handle a specific task.
        
        Args:
            task: The task description
            
        Returns:
            List of agents that can handle the task
        """
        return [
            agent for agent in self._agents.values()
            if agent.can_handle(task)
        ]

    async def execute_task(self, task: str, context: Dict[str, Any]) -> List[AgentResult]:
        """Execute a task by finding all capable agents and delegating.
        
        Args:
            task: The task to execute
            context: Additional context for the task
            
        Returns:
            List of AgentResult from all capable agents
        """
        # Find agents that can handle this task
        capable_agents = self.find_agents_for_task(task)
        
        if not capable_agents:
            return []
        
        results = []
        for agent in capable_agents:
            try:
                # Execute the task with the agent
                result = await agent.execute_with_timing(task, context)
                results.append(result)
            except Exception as e:
                # Create error result for failed execution
                error_result = AgentResult(
                    success=False,
                    output="",
                    error_message=f"Task execution failed: {str(e)}",
                    metadata={"agent": agent.name, "error": str(e)}
                )
                results.append(error_result)
        
        return results

    def get_available_agents(self) -> List[SubAgent]:
        """Get all available (idle) agents.
        
        Returns:
            List of available agents
        """
        return [
            agent for agent in self._agents.values()
            if agent.get_status() == AgentStatus.IDLE
        ]

    def get_agent_status(self, agent_name: str) -> AgentStatus:
        """Get the status of a specific agent.
        
        Args:
            agent_name: Name of the agent
            
        Returns:
            Current status of the agent
            
        Raises:
            AgentNotFoundError: If the agent is not found
        """
        agent = self.get_agent(agent_name)
        return agent.get_status()

    async def assign_task_to_best_agent(
        self, task: str, context: Dict[str, Any]
    ) -> TaskAssignmentResult:
        """Assign a task to the best available agent.
        
        Args:
            task: Description of the task
            context: Additional context for the task
            
        Returns:
            TaskAssignmentResult with assignment details
        """
        import time
        
        start_time = time.time()
        
        # Get available agents
        available_agents = self.get_available_agents()
        
        if not available_agents:
            return TaskAssignmentResult(
                success=False,
                error_message="No available agents to handle the task"
            )
        
        # Find the best agent based on expertise match
        best_agent = None
        best_score = 0.0
        
        for agent in available_agents:
            # Check if agent can handle the task
            if agent.can_handle(task):
                # Calculate expertise match score
                from agents.base_agent import get_agent_expertise_match
                score = get_agent_expertise_match(agent, task)
                
                if score > best_score:
                    best_score = score
                    best_agent = agent
        
        if not best_agent:
            return TaskAssignmentResult(
                success=False,
                error_message="No agent can handle this task"
            )
        
        try:
            # Execute the task with quality enforcement
            result = await self._execute_with_quality_enforcement(
                best_agent, task, context
            )
            execution_time = time.time() - start_time
            
            # Perform quality check on result
            quality_passed, processed_output, violations, quality_score = self._enforce_output_quality(
                best_agent.name, result.output, context
            )
            
            return TaskAssignmentResult(
                success=result.success and quality_passed,
                assigned_agent=best_agent,
                task_output=processed_output,
                error_message=result.error_message or ("Quality enforcement failed" if not quality_passed else ""),
                execution_time=execution_time,
                metadata={
                    **result.metadata,
                    "quality_enforced": self.enforce_quality,
                    "quality_score": quality_score,
                    "quality_violations": len(violations)
                },
                quality_passed=quality_passed,
                quality_score=quality_score,
                quality_violations=violations[:5]  # Top 5 violations
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            return TaskAssignmentResult(
                success=False,
                assigned_agent=best_agent,
                error_message=f"Task execution failed: {str(e)}",
                execution_time=execution_time
            )

    async def assign_task_to_agent(
        self, agent_name: str, task: str, context: Dict[str, Any]
    ) -> TaskAssignmentResult:
        """Assign a task to a specific agent.
        
        Args:
            agent_name: Name of the agent to assign the task to
            task: Description of the task
            context: Additional context for the task
            
        Returns:
            TaskAssignmentResult with assignment details
        """
        import time
        
        start_time = time.time()
        
        try:
            agent = self.get_agent(agent_name)
        except AgentNotFoundError as e:
            return TaskAssignmentResult(
                success=False,
                error_message=f"Agent not found: {str(e)}"
            )
        
        # Check if agent is available
        if agent.get_status() != AgentStatus.IDLE:
            return TaskAssignmentResult(
                success=False,
                error_message=f"Agent '{agent_name}' is not available (status: {agent.get_status().value})"
            )
        
        try:
            # Execute the task with quality enforcement
            result = await self._execute_with_quality_enforcement(
                agent, task, context
            )
            execution_time = time.time() - start_time
            
            # Perform quality check on result
            quality_passed, processed_output, violations, quality_score = self._enforce_output_quality(
                agent.name, result.output, context
            )
            
            return TaskAssignmentResult(
                success=result.success and quality_passed,
                assigned_agent=agent,
                task_output=processed_output,
                error_message=result.error_message or ("Quality enforcement failed" if not quality_passed else ""),
                execution_time=execution_time,
                metadata={
                    **result.metadata,
                    "quality_enforced": self.enforce_quality,
                    "quality_score": quality_score,
                    "quality_violations": len(violations)
                },
                quality_passed=quality_passed,
                quality_score=quality_score,
                quality_violations=violations[:5]  # Top 5 violations
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            return TaskAssignmentResult(
                success=False,
                assigned_agent=agent,
                error_message=f"Task execution failed: {str(e)}",
                execution_time=execution_time
            )

    def to_dict(self) -> Dict[str, Any]:
        """Convert the registry to a dictionary for serialization.
        
        Returns:
            Dictionary representation of the registry
        """
        return {
            "agents": [
                agent.to_dict() for agent in self._agents.values()
            ]
        }

    @classmethod
    def from_dict(
        cls, data: Dict[str, Any], agent_class: Type[SubAgent]
    ) -> 'AgentRegistry':
        """Create a registry from a dictionary.
        
        Args:
            data: Dictionary containing registry data
            agent_class: Class to use for creating agents
            
        Returns:
            AgentRegistry instance
        """
        registry = cls()
        
        for agent_data in data.get("agents", []):
            agent = agent_class.from_dict(agent_data)
            registry.register_agent(agent)
        
        return registry

    def save_to_file(self, filepath: str) -> None:
        """Save the registry to a file.
        
        Args:
            filepath: Path to the file to save to
        """
        data = self.to_dict()
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load_from_file(
        cls, filepath: str, agent_class: Type[SubAgent]
    ) -> 'AgentRegistry':
        """Load a registry from a file.
        
        Args:
            filepath: Path to the file to load from
            agent_class: Class to use for creating agents
            
        Returns:
            AgentRegistry instance
        """
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        return cls.from_dict(data, agent_class)

    def get_metrics(self) -> Dict[str, Any]:
        """Get metrics about the registry.
        
        Returns:
            Dictionary containing registry metrics
        """
        total_agents = len(self._agents)
        available_agents = len(self.get_available_agents())
        busy_agents = sum(
            1 for agent in self._agents.values()
            if agent.get_status() == AgentStatus.BUSY
        )
        idle_agents = sum(
            1 for agent in self._agents.values()
            if agent.get_status() == AgentStatus.IDLE
        )
        error_agents = sum(
            1 for agent in self._agents.values()
            if agent.get_status() == AgentStatus.ERROR
        )
        
        return {
            "total_agents": total_agents,
            "available_agents": available_agents,
            "busy_agents": busy_agents,
            "idle_agents": idle_agents,
            "error_agents": error_agents,
            "utilization_rate": (busy_agents / total_agents) if total_agents > 0 else 0.0
        }

    def health_check(self) -> Dict[str, Any]:
        """Perform a health check on the registry.
        
        Returns:
            Dictionary containing health check results
        """
        metrics = self.get_metrics()
        issues = []
        
        # Check for agents in error state
        if metrics["error_agents"] > 0:
            issues.append(f"{metrics['error_agents']} agents in error state")
        
        # Check for low availability
        if metrics["available_agents"] == 0 and metrics["total_agents"] > 0:
            issues.append("No agents available")
        
        # Determine overall status
        if issues:
            status = "unhealthy"
        else:
            status = "healthy"
        
        return {
            "status": status,
            "total_agents": metrics["total_agents"],
            "available_agents": metrics["available_agents"],
            "issues": issues
        }

    def get_agent_summary(self) -> Dict[str, Any]:
        """Get a summary of all agents in the registry.
        
        Returns:
            Dictionary containing agent summaries
        """
        summaries = {}
        
        for agent in self._agents.values():
            # Handle expertise properly - it should be AgentExpertise enum values
            expertise_values = []
            for exp in agent.get_expertise():
                if hasattr(exp, 'value'):
                    expertise_values.append(exp.value)
                else:
                    expertise_values.append(str(exp))
            
            summaries[agent.name] = {
                "expertise": expertise_values,
                "tools": agent.get_tools(),
                "status": agent.get_status().value,
                "work_history_count": len(agent.work_history),
                "description": agent.description
            }
        
        return summaries

    def get_stats(self) -> Dict[str, Any]:
        """Get registry statistics (alias for get_metrics).
        
        Returns:
            Dictionary with registry statistics
        """
        metrics = self.get_metrics()
        metrics["agent_names"] = self.get_agent_names()
        return metrics

    def get_agent_performance_summary(self) -> Dict[str, Any]:
        """Get agent performance summary.
        
        Returns:
            Dictionary with agent performance information
        """
        return self.get_agent_summary()

    def _initialize_quality_engine(self) -> None:
        """Initialize the quality rules engine for enforcement."""
        try:
            from acp.quality_rules import create_quality_rules_engine
            self._quality_engine = create_quality_rules_engine()
        except ImportError as e:
            print(f"Warning: Could not initialize quality engine: {e}")
            self.enforce_quality = False

    def _validate_agent_quality(self, agent: SubAgent) -> None:
        """Validate that an agent meets quality standards."""
        if not self._quality_engine:
            return
            
        # Basic validation - check agent implementation
        agent_class = agent.__class__
        agent_source = ""
        
        try:
            import inspect
            agent_source = inspect.getsource(agent_class)
        except (OSError, TypeError):
            # Cannot get source - skip validation
            return
        
        # Analyze agent source for quality violations
        quality_report = self._quality_engine.analyze_content(
            agent_source, 
            f"agent_{agent.name}.py"
        )
        
        # Check for critical violations
        critical_violations = [
            v for v in quality_report.violations 
            if v.severity == "error" and v.violation_type.value in [
                "stub_or_mock", "placeholder", "technical_debt"
            ]
        ]
        
        if critical_violations:
            violation_messages = [v.message for v in critical_violations[:3]]
            raise QualityEnforcementError(
                f"Agent '{agent.name}' fails quality standards: {'; '.join(violation_messages)}"
            )

    def _initialize_agent_metrics(self, agent_name: str) -> None:
        """Initialize quality metrics tracking for an agent."""
        self._quality_metrics[agent_name] = {
            "total_tasks": 0,
            "successful_tasks": 0,
            "failed_quality": 0,
            "average_quality_score": 100.0,
            "improvement_attempts": 0,
            "last_quality_check": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "common_violations": []
        }

    def _enforce_output_quality(
        self, agent_name: str, output: str, context: Dict[str, Any], attempt: int = 1
    ) -> tuple[bool, str, List[str], float]:
        """
        Enforce quality standards on agent output.
        
        Args:
            agent_name: Name of the agent that produced the output
            output: The output to validate
            context: Task context
            attempt: Current improvement attempt number
            
        Returns:
            Tuple of (quality_passed, processed_output, violations, quality_score)
        """
        if not self._quality_engine or not self.enforce_quality:
            return True, output, [], 100.0
        
        # Analyze output quality
        file_path = context.get('file_path', f"agent_output_{agent_name}.py")
        quality_report = self._quality_engine.analyze_content(output, file_path)
        
        # Extract violations
        violations = [v.message for v in quality_report.violations]
        quality_score = quality_report.overall_score
        
        # Check for critical violations that require rejection
        critical_violations = [
            v for v in quality_report.violations 
            if v.severity == "error" and v.violation_type.value in [
                "stub_or_mock", "placeholder", "technical_debt"
            ]
        ]
        
        # Update agent metrics
        metrics = self._quality_metrics.get(agent_name, {})
        metrics["total_tasks"] = metrics.get("total_tasks", 0) + 1
        metrics["last_quality_check"] = time.strftime("%Y-%m-%dT%H:%M:%SZ")
        
        # Calculate new average quality score
        total_tasks = metrics["total_tasks"]
        current_avg = metrics.get("average_quality_score", 100.0)
        metrics["average_quality_score"] = (
            (current_avg * (total_tasks - 1) + quality_score) / total_tasks
        )
        
        if critical_violations:
            metrics["failed_quality"] = metrics.get("failed_quality", 0) + 1
            quality_passed = False
        else:
            metrics["successful_tasks"] = metrics.get("successful_tasks", 0) + 1
            quality_passed = True
        
        # Track common violations
        violation_types = [v.violation_type.value for v in quality_report.violations]
        for vtype in violation_types:
            common_violations = metrics.get("common_violations", [])
            existing = next((v for v in common_violations if v["type"] == vtype), None)
            if existing:
                existing["count"] += 1
            else:
                common_violations.append({"type": vtype, "count": 1})
            metrics["common_violations"] = sorted(
                common_violations, key=lambda x: x["count"], reverse=True
            )[:5]
        
        self._quality_metrics[agent_name] = metrics
        
        return quality_passed, output, violations, quality_score

    async def _execute_with_quality_enforcement(
        self, agent: SubAgent, task: str, context: Dict[str, Any]
    ) -> AgentResult:
        """
        Execute task with automatic quality enforcement and improvement attempts.
        
        Args:
            agent: The agent to execute the task
            task: Task to execute
            context: Task context
            
        Returns:
            Quality-validated AgentResult
        """
        if not self.enforce_quality:
            # No quality enforcement - execute directly
            return await agent.execute_with_timing(task, context)
        
        # Track improvement attempts in agent metrics
        agent_metrics = self._quality_metrics.get(agent.name, {})
        
        for attempt in range(1, self.max_improvement_attempts + 2):  # +1 for initial attempt
            # Execute the task
            result = await agent.execute_with_timing(task, context)
            
            if not result.success:
                # Task execution failed - return immediately
                return result
            
            # Check quality of the output
            quality_passed, processed_output, violations, quality_score = self._enforce_output_quality(
                agent.name, result.output, context, attempt
            )
            
            if quality_passed:
                # Quality check passed - return successful result
                return AgentResult(
                    success=True,
                    output=processed_output,
                    error_message=result.error_message,
                    metadata={
                        **result.metadata,
                        "quality_attempts": attempt,
                        "quality_score": quality_score,
                        "quality_passed": True
                    }
                )
            
            # Quality check failed
            if attempt <= self.max_improvement_attempts:
                # Attempt improvement
                agent_metrics["improvement_attempts"] = agent_metrics.get("improvement_attempts", 0) + 1
                self._quality_metrics[agent.name] = agent_metrics
                
                # Use the improvement attempt method
                result = await self._attempt_quality_improvement(
                    agent, task, context, result.output, violations, attempt
                )
                
                # Continue to next iteration with the improved result
            else:
                # Max attempts reached - return failure
                return AgentResult(
                    success=False,
                    output=processed_output,
                    error_message=f"Quality enforcement failed after {attempt-1} attempts. Violations: {'; '.join(violations[:3])}",
                    metadata={
                        **result.metadata,
                        "quality_attempts": attempt - 1,
                        "quality_score": quality_score,
                        "quality_passed": False,
                        "final_violations": violations
                    }
                )
        
        # Should never reach here, but return failed result as fallback
        return AgentResult(
            success=False,
            output="",
            error_message="Quality enforcement loop exceeded maximum iterations",
            metadata={"quality_enforcement_error": True}
        )

    async def _attempt_quality_improvement(
        self, agent: SubAgent, task: str, context: Dict[str, Any], 
        previous_output: str, violations: List[str], attempt: int
    ) -> AgentResult:
        """
        Attempt to improve agent output quality by providing specific feedback.
        
        Args:
            agent: The agent to retry with
            task: Original task
            context: Task context
            previous_output: Previous output that failed quality
            violations: List of quality violations found
            attempt: Current attempt number
            
        Returns:
            AgentResult from retry attempt
        """
        # Create enhanced task with quality feedback
        quality_feedback = "\n".join([
            "QUALITY IMPROVEMENT REQUIRED:",
            "Your previous output had these issues:",
            *[f"- {violation}" for violation in violations[:5]],
            "",
            "Follow Global CLAUDE.md standards:",
            "- No TODO, FIXME, or placeholder comments",
            "- No mocks or stubs - use real implementations", 
            "- Maximum 300 lines per file",
            "- Proper error handling and type hints",
            "- Complete, working functionality",
            "",
            f"Previous attempt (for reference only):",
            f"```",
            previous_output[:500] + ("..." if len(previous_output) > 500 else ""),
            f"```",
            "",
            "Provide a corrected implementation:"
        ])
        
        enhanced_task = f"{task}\n\n{quality_feedback}"
        
        # Add quality context
        quality_context = {
            **context,
            "quality_improvement_attempt": attempt,
            "previous_violations": violations,
            "enforce_quality": True
        }
        
        # Retry with enhanced context
        return await agent.execute_task(enhanced_task, quality_context)

    def get_quality_metrics(self, agent_name: str = None) -> Dict[str, Any]:
        """
        Get quality metrics for agents.
        
        Args:
            agent_name: Specific agent name, or None for all agents
            
        Returns:
            Quality metrics dictionary
        """
        if not self.enforce_quality:
            return {"quality_enforcement": "disabled"}
        
        if agent_name:
            return self._quality_metrics.get(agent_name, {})
        
        # Return system-wide quality summary
        total_tasks = sum(m.get("total_tasks", 0) for m in self._quality_metrics.values())
        total_successful = sum(m.get("successful_tasks", 0) for m in self._quality_metrics.values())
        total_failed_quality = sum(m.get("failed_quality", 0) for m in self._quality_metrics.values())
        
        if self._quality_metrics:
            avg_quality_score = sum(
                m.get("average_quality_score", 100.0) for m in self._quality_metrics.values()
            ) / len(self._quality_metrics)
        else:
            avg_quality_score = 100.0
        
        # Find most common violations across all agents
        all_violations = {}
        for metrics in self._quality_metrics.values():
            for violation in metrics.get("common_violations", []):
                vtype = violation["type"]
                all_violations[vtype] = all_violations.get(vtype, 0) + violation["count"]
        
        common_violations = sorted(
            [{"type": vtype, "count": count} for vtype, count in all_violations.items()],
            key=lambda x: x["count"], reverse=True
        )[:5]
        
        return {
            "quality_enforcement": "enabled",
            "total_agents_tracked": len(self._quality_metrics),
            "total_tasks_processed": total_tasks,
            "tasks_passed_quality": total_successful,
            "tasks_failed_quality": total_failed_quality,
            "quality_pass_rate": (total_successful / total_tasks * 100.0) if total_tasks > 0 else 100.0,
            "average_quality_score": avg_quality_score,
            "common_violations_system_wide": common_violations,
            "agent_metrics": self._quality_metrics
        }

    def set_quality_enforcement(self, enabled: bool) -> None:
        """
        Enable or disable quality enforcement.
        
        Args:
            enabled: Whether to enforce quality standards
        """
        self.enforce_quality = enabled
        if enabled and not self._quality_engine:
            self._initialize_quality_engine()


# Convenience functions for backward compatibility
def create_agent_registry() -> AgentRegistry:
    """Create a new agent registry.
    
    Returns:
        New AgentRegistry instance
    """
    return AgentRegistry()


def register_agent(registry: AgentRegistry, agent: SubAgent) -> None:
    """Register an agent with a registry.
    
    Args:
        registry: The registry to register with
        agent: The agent to register
    """
    registry.register_agent(agent)


def find_best_agent_for_task(
    registry: AgentRegistry, task: str
) -> Optional[SubAgent]:
    """Find the best agent for a given task.
    
    Args:
        registry: The registry to search
        task: Description of the task
        
    Returns:
        The best agent for the task, or None if no suitable agent found
    """
    available_agents = registry.get_available_agents()
    
    if not available_agents:
        return None
    
    best_agent = None
    best_score = 0.0
    
    for agent in available_agents:
        if agent.can_handle(task):
            from agents.base_agent import get_agent_expertise_match
            score = get_agent_expertise_match(agent, task)
            
            if score > best_score:
                best_score = score
                best_agent = agent
    
    return best_agent 