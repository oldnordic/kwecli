#!/usr/bin/env python3
"""
Core SubAgent Abstract Base Class

This module provides the abstract base class that all agents must inherit from.
Includes the SubAgent ABC with core methods and work history management.
"""

import time
from abc import ABC, abstractmethod
from typing import Dict, List, Any

from .agent_result import AgentResult, AgentStatus, AgentExpertise


class SubAgent(ABC):
    """Abstract base class for all sub-agents."""

    def __init__(
        self,
        name: str,
        expertise: List[AgentExpertise],
        tools: List[str],
        description: str = ""
    ):
        """Initialize a sub-agent.
        
        Args:
            name: Unique name for the agent
            expertise: List of areas this agent specializes in
            tools: List of tools/technologies this agent uses
            description: Human-readable description of the agent
        """
        self.name = name
        self.expertise = expertise
        self.tools = tools
        self.description = description
        self.status = AgentStatus.IDLE
        self.work_history: List[Dict[str, Any]] = []

    @abstractmethod
    async def execute_task(self, task: str, context: Dict[str, Any]) -> AgentResult:
        """Execute a task and return the result.
        
        Args:
            task: Description of the task to execute
            context: Additional context information
            
        Returns:
            AgentResult containing the execution result
        """
        pass

    @abstractmethod
    def can_handle(self, task: str) -> bool:
        """Check if this agent can handle the given task.
        
        Args:
            task: Description of the task
            
        Returns:
            True if the agent can handle this task, False otherwise
        """
        pass

    def get_expertise(self) -> List[AgentExpertise]:
        """Get the agent's areas of expertise.
        
        Returns:
            List of expertise areas
        """
        return self.expertise

    def get_tools(self) -> List[str]:
        """Get the agent's tools and technologies.
        
        Returns:
            List of tools/technologies
        """
        return self.tools

    def get_status(self) -> AgentStatus:
        """Get the current status of the agent.
        
        Returns:
            Current agent status
        """
        return self.status

    def update_status(self, status: AgentStatus) -> None:
        """Update the agent's status.
        
        Args:
            status: New status for the agent
        """
        self.status = status

    def add_work_history(self, task: str, result: AgentResult) -> None:
        """Add an entry to the agent's work history.
        
        Args:
            task: Description of the task that was executed
            result: Result of the task execution
        """
        entry = {
            "task": task,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "result": result
        }
        self.work_history.append(entry)

    def _record_work(self, task: str, result: AgentResult) -> None:
        """Record work in history (alias for add_work_history).
        
        Args:
            task: Description of the task that was executed
            result: Result of the task execution
        """
        self.add_work_history(task, result)

    def get_recent_work(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent work history entries.
        
        Args:
            limit: Maximum number of entries to return
            
        Returns:
            List of recent work history entries (most recent first)
        """
        return self.work_history[-limit:][::-1]

    def clear_work_history(self) -> None:
        """Clear the agent's work history."""
        self.work_history.clear()

    def get_success_rate(self) -> float:
        """Calculate the success rate based on work history.
        
        Returns:
            Success rate as a float between 0.0 and 1.0
        """
        if not self.work_history:
            return 0.0
        
        successful_tasks = sum(
            1 for entry in self.work_history 
            if entry["result"].success
        )
        
        return successful_tasks / len(self.work_history)

    def get_average_execution_time(self) -> float:
        """Calculate the average execution time based on work history.
        
        Returns:
            Average execution time in seconds
        """
        if not self.work_history:
            return 0.0
        
        total_time = sum(
            entry["result"].metadata.get("execution_time", 0.0)
            for entry in self.work_history
        )
        
        return total_time / len(self.work_history)

    def to_dict(self) -> Dict[str, Any]:
        """Convert the agent to a dictionary for serialization.
        
        Returns:
            Dictionary representation of the agent
        """
        from .agent_advanced import agent_to_dict
        return agent_to_dict(self)

    def __str__(self) -> str:
        """String representation of the agent.
        
        Returns:
            String representation showing agent name and status
        """
        return f"{self.name} ({self.status.value})"

    def __repr__(self) -> str:
        """Detailed string representation of the agent.
        
        Returns:
            Detailed string representation
        """
        expertise_str = ", ".join([exp.value for exp in self.expertise])
        tools_str = ", ".join(self.tools[:3])  # Show first 3 tools
        if len(self.tools) > 3:
            tools_str += f" (+{len(self.tools) - 3} more)"
        
        return (f"{self.name} - Expertise: {expertise_str}, "
                f"Tools: {tools_str}, Status: {self.status.value}")

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SubAgent':
        """Create an agent from a dictionary.
        
        Args:
            data: Dictionary containing agent data
            
        Returns:
            Agent instance
        """
        from .agent_advanced import agent_from_dict
        return agent_from_dict(cls, data)

    async def execute_with_timing(
        self, task: str, context: Dict[str, Any]
    ) -> AgentResult:
        """Execute a task with timing and history tracking.
        
        Args:
            task: Description of the task to execute
            context: Additional context information
            
        Returns:
            AgentResult containing the execution result
        """
        from .agent_advanced import execute_agent_with_timing
        return await execute_agent_with_timing(self, task, context)