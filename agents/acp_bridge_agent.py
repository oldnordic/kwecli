#!/usr/bin/env python3
"""
ACP Bridge Agent

A minimal, fully-implemented SubAgent that exposes KWE capabilities to the ACP
integration. It focuses on code-generation requests and returns standardized
AgentResult objects. This agent is designed to be safely auto-registered at
startup so ACP can discover at least one working agent.
"""

from typing import Dict, Any, List
from agents.agent_base import SubAgent
from agents.agent_result import (
    AgentResult,
    AgentStatus,
    AgentExpertise,
    create_success_result,
    create_error_result,
)
from agents.qwen_agent import CodeGenerationAgent, CodeGenerationRequest, Language


class ACPBridgeAgent(SubAgent):
    """Minimal SubAgent bridging to CodeGenerationAgent for ACP tasks."""

    def __init__(self) -> None:
        super().__init__(
            name="acp-bridge-agent",
            expertise=[AgentExpertise.CODE_GENERATION, AgentExpertise.SYSTEM_INTEGRATION],
            tools=["code_generation", "review", "execute"],
            description="Bridges ACP tasks to KWE code generation and execution"
        )
        self._code_agent = CodeGenerationAgent()

    def can_handle(self, task: str) -> bool:
        t = (task or "").lower()
        return any(kw in t for kw in ("generate", "code", "implement", "create"))

    async def execute_task(self, task: str, context: Dict[str, Any]) -> AgentResult:
        self.update_status(AgentStatus.BUSY)
        try:
            description = task or context.get("description", "")
            language_str = context.get("language", "python").upper()
            try:
                language = Language[language_str]
            except Exception:
                language = Language.PYTHON

            req = CodeGenerationRequest(
                prompt=description,
                language=language,
                context=context.get("plan", ""),
                requirements=[
                    "Return complete, working code",
                    "Follow repository quality policies",
                ],
            )
            result = await self._code_agent.generate_code(req)
            if not result.success:
                self.update_status(AgentStatus.ERROR)
                return create_error_result(result.error_message or "generation_failed")

            self.update_status(AgentStatus.COMPLETED)
            return create_success_result(
                output=result.code,
                quality_score=100,
                metadata={"language": language.name.lower()}
            )
        except Exception as e:
            self.update_status(AgentStatus.ERROR)
            return create_error_result(str(e))

