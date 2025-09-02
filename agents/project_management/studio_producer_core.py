#!/usr/bin/env python3
"""
Studio Producer Agent Core Module

This module contains the core StudioProducer class definition, initialization,
task routing, and main execution logic.
"""

from typing import Dict, List, Any

from agents.base_agent import (
    SubAgent, AgentExpertise, AgentResult, create_agent_result
)
from .studio_producer_handlers import StudioProducerHandlers
from .studio_producer_utils import StudioProducerUtils


class StudioProducer(SubAgent):
    """Studio Producer agent for cross-team coordination and resource management."""

    def __init__(self):
        """Initialize the Studio Producer agent."""
        super().__init__(
            name="Studio Producer",
            expertise=[
                AgentExpertise.PRODUCT_MANAGEMENT,
                AgentExpertise.ANALYTICS
            ],
            tools=[
                "project_management",
                "resource_allocation",
                "workflow_optimization",
                "team_coordination",
                "process_improvement",
                "conflict_resolution",
                "sprint_planning",
                "communication_planning"
            ],
            description=(
                "Master studio orchestrator specializing in cross-team coordination, "
                "resource optimization, process design, and workflow automation. "
                "Ensures brilliant individuals work together as an even more brilliant "
                "team, maximizing output while maintaining rapid innovation culture."
            )
        )
        
        # Initialize handler and utility components
        self._handlers = StudioProducerHandlers()
        self._utils = StudioProducerUtils()

    def can_handle(self, task: str) -> bool:
        """Check if this agent can handle the given task.
        
        Args:
            task: Description of the task
            
        Returns:
            True if the agent can handle the task, False otherwise
        """
        if not task or not task.strip():
            return False
        
        task_lower = task.lower()
        
        # Coordination keywords
        coordination_keywords = [
            "coordinate", "coordination", "team", "teams", "cross-team",
            "handoff", "handoffs", "dependencies", "collaboration"
        ]
        
        # Resource management keywords
        resource_keywords = [
            "resource", "allocation", "capacity", "senior engineers",
            "designers", "qa", "team capacity", "workload"
        ]
        
        # Process optimization keywords
        process_keywords = [
            "workflow", "optimize", "optimization", "process", "bottleneck",
            "efficiency", "improvement", "streamline"
        ]
        
        # Sprint planning keywords
        sprint_keywords = [
            "sprint", "planning", "6-day", "cycle", "priorities",
            "timeline", "deadline"
        ]
        
        # Conflict resolution keywords
        conflict_keywords = [
            "conflict", "resolve", "mediation", "consensus", "disagreement",
            "blocker", "issue"
        ]
        
        # Team health keywords
        health_keywords = [
            "team health", "burnout", "monitoring", "wellbeing",
            "stress", "overload"
        ]
        
        # Communication keywords
        communication_keywords = [
            "communication", "meeting", "sync", "alignment",
            "coordination plan"
        ]
        
        all_keywords = (
            coordination_keywords + resource_keywords + process_keywords +
            sprint_keywords + conflict_keywords + health_keywords +
            communication_keywords
        )
        
        return any(keyword in task_lower for keyword in all_keywords)

    async def execute_task(self, task: str, context: Dict[str, Any]) -> AgentResult:
        """Execute a task and return the result.
        
        Args:
            task: Description of the task to execute
            context: Additional context information
            
        Returns:
            AgentResult containing the execution result
        """
        if not task or not task.strip():
            return create_agent_result(
                success=False,
                output="",
                error_message="Task cannot be empty",
                quality_score=0
            )
        
        try:
            task_lower = task.lower()
            
            # Determine task type and delegate to appropriate handler
            # Check for specific task types first (more specific matches)
            if any(word in task_lower for word in ["workflow", "optimize", "bottleneck"]):
                return await self._handlers.handle_workflow_optimization(task, context)
            elif any(word in task_lower for word in ["resource", "allocation", "capacity"]):
                return await self._handlers.handle_resource_allocation(task, context, self._utils)
            elif any(word in task_lower for word in ["conflict", "resolve", "mediation"]):
                return await self._handlers.handle_conflict_resolution(task, context)
            elif any(word in task_lower for word in ["team health", "burnout", "monitoring"]):
                return await self._handlers.handle_team_health_monitoring(task, context, self._utils)
            elif any(word in task_lower for word in ["process", "improvement", "efficiency"]):
                return await self._handlers.handle_process_improvement(task, context, self._utils)
            elif any(word in task_lower for word in ["communication", "meeting", "sync"]):
                return await self._handlers.handle_communication_planning(task, context)
            elif any(word in task_lower for word in ["sprint", "6-day"]):
                return await self._handlers.handle_sprint_planning(task, context)
            elif any(word in task_lower for word in ["coordinate", "coordination"]):
                return await self._handlers.handle_coordination_task(task, context)
            else:
                return await self._handlers.handle_general_coordination(task, context)
                
        except Exception as e:
            return create_agent_result(
                success=False,
                output="",
                error_message=f"Error executing task: {str(e)}",
                quality_score=0
            )