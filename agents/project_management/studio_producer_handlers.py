#!/usr/bin/env python3
"""
Studio Producer Agent Handlers Module

This module contains all task handler methods for the StudioProducer agent.
Each handler specializes in a specific type of coordination or management task.
"""

from typing import Dict, List, Any

from agents.base_agent import AgentResult, create_agent_result
from .studio_producer_handlers_primary import StudioProducerPrimaryHandlers
from .studio_producer_handlers_secondary import StudioProducerSecondaryHandlers


class StudioProducerHandlers:
    """Unified handler methods for Studio Producer agent tasks."""

    def __init__(self):
        """Initialize the handlers."""
        self._primary = StudioProducerPrimaryHandlers()
        self._secondary = StudioProducerSecondaryHandlers()

    # Primary handlers delegation
    async def handle_coordination_task(self, task: str, context: Dict[str, Any]) -> AgentResult:
        """Handle cross-team coordination tasks."""
        return await self._primary.handle_coordination_task(task, context)

    async def handle_resource_allocation(self, task: str, context: Dict[str, Any], utils) -> AgentResult:
        """Handle resource allocation tasks."""
        return await self._primary.handle_resource_allocation(task, context, utils)

    async def handle_workflow_optimization(self, task: str, context: Dict[str, Any]) -> AgentResult:
        """Handle workflow optimization tasks."""
        return await self._primary.handle_workflow_optimization(task, context)

    async def handle_sprint_planning(self, task: str, context: Dict[str, Any]) -> AgentResult:
        """Handle sprint planning tasks."""
        return await self._primary.handle_sprint_planning(task, context)

    # Secondary handlers delegation
    async def handle_conflict_resolution(self, task: str, context: Dict[str, Any]) -> AgentResult:
        """Handle conflict resolution tasks."""
        return await self._secondary.handle_conflict_resolution(task, context)

    async def handle_team_health_monitoring(self, task: str, context: Dict[str, Any], utils) -> AgentResult:
        """Handle team health monitoring tasks."""
        return await self._secondary.handle_team_health_monitoring(task, context, utils)

    async def handle_process_improvement(self, task: str, context: Dict[str, Any], utils) -> AgentResult:
        """Handle process improvement tasks."""
        return await self._secondary.handle_process_improvement(task, context, utils)

    async def handle_communication_planning(self, task: str, context: Dict[str, Any]) -> AgentResult:
        """Handle communication planning tasks."""
        return await self._secondary.handle_communication_planning(task, context)

    async def handle_general_coordination(self, task: str, context: Dict[str, Any]) -> AgentResult:
        """Handle general coordination tasks."""
        return await self._secondary.handle_general_coordination(task, context)