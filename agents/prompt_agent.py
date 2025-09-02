#!/usr/bin/env python3
"""
PromptAgent Base and Concrete Subclasses

Defines a generic PromptAgent that sends formatted prompts to the LLM
and parses responses, plus two sample agents: MasterCoderAgent and
TestWriterAgent.
"""
from typing import Dict, Any
from .agent_base import SubAgent
from .agent_result import AgentResult, AgentExpertise
from endpoint_tool_integration import get_endpoint_integration


class PromptAgent(SubAgent):
    """Base class for prompt-driven agents using an LLM integration."""

    def __init__(self, name: str, prompt_template: str, expertise: AgentExpertise = AgentExpertise.GENERAL, description: str = ""):
        super().__init__(
            name=name,
            expertise=[expertise] if isinstance(expertise, AgentExpertise) else expertise,
            tools=[],
            description=description or f"Prompt-based agent: {name}"
        )
        self.prompt_template = prompt_template
        self.integration = get_endpoint_integration()

    async def execute_task(self, task: str, context: Dict[str, Any]) -> AgentResult:
        """Format the prompt and send it to the LLM, returning parsed result."""
        prompt = self.prompt_template.format(task=task, **context)
        # Use the chat integration to get a response
        result = await self.integration.process_chat_with_tools(prompt, context)
        # Expect 'message' in result
        if not result.get("success", True):
            return AgentResult(success=False, error=result.get("error", "Unknown error"))
        message = result.get("message") or result.get("response") or ""
        return AgentResult(success=True, data={"message": message})

    def can_handle(self, task: str) -> bool:
        """PromptAgent can handle any task by design."""
        return True


class MasterCoderAgent(PromptAgent):
    """Agent that responds as a master coder, focusing on code tasks."""

    def __init__(self):
        prompt = "You are a master coder. {task}"
        super().__init__(
            name="master_coder",
            prompt_template=prompt,
            expertise=AgentExpertise.GENERAL,
            description="Specialist in writing and reviewing code"
        )


class TestWriterAgent(PromptAgent):
    """Agent that writes pytest tests for given code or tasks."""

    def __init__(self):
        prompt = "You are an expert test engineer. Write pytest tests for: {task}"
        super().__init__(
            name="test_writer",
            prompt_template=prompt,
            expertise=AgentExpertise.GENERAL,
            description="Generates pytest tests for code tasks"
        )
