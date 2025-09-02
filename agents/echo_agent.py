#!/usr/bin/env python3
"""
Echo Agent for KWECLI: returns back the received task as a response.
"""
from typing import Dict, Any
from .agent_base import SubAgent
from .agent_result import AgentResult, AgentExpertise


class EchoAgent(SubAgent):
    """A simple agent that echoes the task input."""

    def __init__(self):
        super().__init__(
            name="echo",
            expertise=[AgentExpertise.GENERAL],
            tools=[],
            description="Echo agent that repeats the task text"
        )

    async def execute_task(self, task: str, context: Dict[str, Any]) -> AgentResult:
        # Return the task text as the result payload
        return AgentResult(success=True, data={"echo": task})

    def can_handle(self, task: str) -> bool:
        # This agent can handle any task
        return True
