#!/usr/bin/env python3
"""
Advanced Streaming Integration for Qwen Agent

This module provides streaming capabilities, markdown prompt formatting,
context management, and micro-agent orchestration for the Qwen Agent.
"""

import asyncio
import logging
import time
import re
from typing import AsyncGenerator, List, Dict, Any, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime

from agents.qwen_agent import (
    CodeGenerationRequest,
    CodeGenerationResult,
    Language
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class StreamingCodeGenerationRequest(CodeGenerationRequest):
    """Enhanced request for streaming code generation."""
    stream_callback: Optional[Callable[[str], None]] = None
    context_id: Optional[str] = None
    agent_id: Optional[str] = None


@dataclass
class StreamingCodeGenerationResult(CodeGenerationResult):
    """Enhanced result for streaming code generation."""
    streamed_chunks: List[str] = field(default_factory=list)
    final_code: str = ""
    streaming_metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ContextNode:
    """Represents a node in the context graph."""
    node_id: str
    content: Dict[str, Any]
    timestamp: datetime
    agent_id: str
    context_type: str
    priority: int
    dependencies: List[str] = field(default_factory=list)


@dataclass
class AgentContext:
    """Context for a specific agent."""
    agent_id: str
    project_id: str
    conversation_history: List[Dict[str, Any]] = field(default_factory=list)
    shared_knowledge: Dict[str, Any] = field(default_factory=dict)
    last_updated: datetime = field(default_factory=datetime.now)
    context_size: int = 0


@dataclass
class SharedContext:
    """Shared context across all agents in a project."""
    project_id: str
    agents: Dict[str, AgentContext] = field(default_factory=dict)
    global_knowledge: Dict[str, Any] = field(default_factory=dict)
    conversation_summary: str = ""
    context_window: int = 10000


class MarkdownPromptBuilder:
    """Builds markdown-formatted prompts to avoid JSON issues."""
    
    def build_system_prompt(self, language: Language) -> str:
        """Build system prompt for a specific language."""
        language_name = language.value
        
        return f"""# System Instructions

You are an expert {language_name} programmer. Generate high-quality, 
production-ready code based on the user's requirements.

## Requirements:
- Write clean, well-documented code
- Follow {language_name} best practices and conventions
- Include proper error handling
- Add appropriate comments and docstrings
- Ensure the code is functional and complete
- Use modern {language_name} features when appropriate
- Include type hints if supported by the language
- Add input validation where necessary

Please provide only the code without any explanations or markdown formatting."""

    def build_user_prompt(self, request: CodeGenerationRequest) -> str:
        """Build user prompt with markdown formatting."""
        system_prompt = self.build_system_prompt(request.language)
        
        # Build context section
        context_section = ""
        if request.context:
            context_section = f"\n## Context\n{request.context}"
        
        # Build requirements section
        requirements_section = ""
        if request.requirements:
            req_list = "\n".join(f"- {req}" for req in request.requirements)
            requirements_section = f"\n## Requirements\n{req_list}"
        
        # Build style guide section
        style_section = ""
        if request.style_guide:
            style_section = f"\n## Style Guide\n{request.style_guide}"
        
        # Build user request section
        user_section = f"\n## User Request\n{request.prompt}"
        
        # Build response format section
        response_section = (
            "\n## Response Format\n"
            "Please provide only the code without explanations or "
            "markdown formatting."
        )
        
        return (f"{system_prompt}{context_section}{requirements_section}"
                f"{style_section}{user_section}{response_section}")

    def format_context_as_markdown(self, context: Dict[str, Any]) -> str:
        """Format context dictionary as markdown."""
        if not context:
            return ""
        
        sections = []
        for key, value in context.items():
            if isinstance(value, list):
                items = "\n".join(f"- {item}" for item in value)
                sections.append(f"### {key.replace('_', ' ').title()}\n{items}")
            elif isinstance(value, dict):
                items = "\n".join(f"- **{k}**: {v}" for k, v in value.items())
                sections.append(f"### {key.replace('_', ' ').title()}\n{items}")
            else:
                sections.append(f"### {key.replace('_', ' ').title()}\n{value}")
        
        return "\n\n".join(sections)

    def format_requirements_as_markdown(self, requirements: List[str]) -> str:
        """Format requirements list as markdown."""
        if not requirements:
            return ""
        
        items = "\n".join(f"- {req}" for req in requirements)
        return f"## Requirements\n{items}"


class StreamingOllamaClient:
    """Handles streaming communication with Ollama."""
    
    def __init__(self, timeout: int = 120):
        self.timeout = timeout
    
    async def stream_from_ollama(self, prompt: str, model: str) -> AsyncGenerator[str, None]:
        """Stream responses from Ollama in real-time."""
        try:
            process = await asyncio.create_subprocess_exec(
                "ollama", "run", model, prompt,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            # Read from stdout asynchronously
            async for line in process.stdout:
                if line:
                    yield line.decode().strip()
            
            # Wait for process to complete
            await process.wait()
            
        except Exception as e:
            logger.error(f"Error streaming from Ollama: {e}")
            raise


class StreamingCodeGenerator:
    """Handles real-time streaming of code generation."""
    
    def __init__(self, default_model: str = "hf.co/unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF:Q4_K_M"):
        self.default_model = default_model
        self.client = StreamingOllamaClient()
        self.prompt_builder = MarkdownPromptBuilder()
    
    async def stream_code_generation(self, request: StreamingCodeGenerationRequest) -> AsyncGenerator[str, None]:
        """Stream code generation in real-time."""
        try:
            # Build markdown prompt
            markdown_prompt = self.format_markdown_prompt(request)
            
            # Stream from Ollama
            async for chunk in self.client.stream_from_ollama(
                markdown_prompt, request.model or self.default_model
            ):
                if chunk:
                    yield chunk
                    
                    # Call callback if provided
                    if request.stream_callback:
                        request.stream_callback(chunk)
                        
        except Exception as e:
            logger.error(f"Error in streaming code generation: {e}")
            raise
    
    async def stream_from_ollama(self, prompt: str, model: str) -> AsyncGenerator[str, None]:
        """Stream from Ollama - wrapper for client."""
        async for chunk in self.client.stream_from_ollama(prompt, model):
            yield chunk
    
    async def process_streaming_response(self, stream: AsyncGenerator[str, None]) -> StreamingCodeGenerationResult:
        """Process streaming response into final result."""
        chunks = []
        start_time = time.time()
        
        try:
            async for chunk in stream:
                chunks.append(chunk)
            
            # Combine chunks into final code
            full_response = "".join(chunks)
            
            # Extract code blocks
            code_blocks = self._extract_code_blocks(full_response)
            final_code = "\n\n".join(code_blocks) if code_blocks else full_response
            
            # Validate code
            is_valid, warnings = self._validate_code(final_code, Language.PYTHON)
            
            streaming_time = time.time() - start_time
            
            return StreamingCodeGenerationResult(
                code=final_code,
                language=Language.PYTHON,
                success=is_valid,
                warnings=warnings,
                streamed_chunks=chunks,
                final_code=final_code,
                streaming_metadata={
                    "total_chunks": len(chunks),
                    "streaming_time": streaming_time,
                    "response_length": len(full_response)
                }
            )
            
        except Exception as e:
            logger.error(f"Error processing streaming response: {e}")
            return StreamingCodeGenerationResult(
                code="",
                language=Language.PYTHON,
                success=False,
                error_message=f"Error processing stream: {str(e)}",
                streamed_chunks=chunks,
                streaming_metadata={"error": str(e)}
            )
    
    def format_markdown_prompt(self, request: CodeGenerationRequest) -> str:
        """Format request as markdown prompt."""
        # Handle StreamingCodeGenerationRequest with context_id and agent_id
        if hasattr(request, 'context_id') and hasattr(request, 'agent_id'):
            # For streaming requests, include additional context from context manager
            try:
                # Get context from the context manager if available
                if hasattr(self, 'context_manager') and self.context_manager:
                    # Since this is a synchronous method, we'll enhance the request
                    # with available context information without async calls
                    if not hasattr(request, 'metadata'):
                        request.metadata = {}
                    
                    # Add streaming context information
                    request.metadata['context_id'] = request.context_id
                    request.metadata['agent_id'] = request.agent_id
                    request.metadata['streaming_request'] = True
                    
                    # Log context enhancement
                    logger.info(f"Enhanced streaming request with context for agent {request.agent_id}")
            except Exception as e:
                # Log context enhancement failure but don't fail the request
                logger.warning(f"Failed to enhance request with context: {e}")
        
        return self.prompt_builder.build_user_prompt(request)
    
    def _extract_code_blocks(self, response: str) -> List[str]:
        """Extract code blocks from response."""
        # Look for code blocks with language specification
        code_block_pattern = (
            r"```(?:python|rust|javascript|typescript|go|cpp|java|csharp|php|ruby|"
            r"swift|kotlin|scala|r|matlab|shell|html|css|sql|yaml|json|markdown)?"
            r"\s*(.*?)\s*```"
        )
        matches = re.findall(code_block_pattern, response, re.DOTALL | re.IGNORECASE)
        
        if matches:
            return [match.strip() for match in matches]
        
        # If no code blocks found, return the entire response
        return [response.strip()]
    
    def _validate_code(self, code: str, language: Language) -> tuple[bool, List[str]]:
        """Validate generated code."""
        warnings = []
        
        if not code.strip():
            warnings.append("Generated code is empty")
            return False, warnings
        
        # Basic validation for Python
        if language == Language.PYTHON:
            try:
                compile(code, '<string>', 'exec')
            except SyntaxError as e:
                warnings.append(f"Python syntax error: {e}")
                return False, warnings
        
        return True, warnings


class IntelliLlamaContextManager:
    """Manages context across all micro-agents."""
    
    def __init__(self, project_id: str):
        self.project_id = project_id
        self.context_store: Dict[str, Dict[str, Any]] = {}
        self.context_nodes: List[ContextNode] = []
    
    async def add_context(self, agent_id: str, context: Dict[str, Any]) -> bool:
        """Add context for a specific agent."""
        try:
            if agent_id not in self.context_store:
                self.context_store[agent_id] = {}
            
            # Merge context
            self.context_store[agent_id].update(context)
            
            # Create context node
            node = ContextNode(
                node_id=f"{agent_id}-{int(time.time())}",
                content=context,
                timestamp=datetime.now(),
                agent_id=agent_id,
                context_type="agent_context",
                priority=1
            )
            self.context_nodes.append(node)
            
            logger.info(f"Added context for agent {agent_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error adding context for agent {agent_id}: {e}")
            return False
    
    async def get_shared_context(self) -> Dict[str, Any]:
        """Get shared context across all agents."""
        shared_context = {}
        
        for agent_id, context in self.context_store.items():
            shared_context[f"agent_{agent_id}"] = context
        
        return shared_context
    
    async def update_context(self, agent_id: str, updates: Dict[str, Any]) -> bool:
        """Update existing context for an agent."""
        try:
            if agent_id not in self.context_store:
                self.context_store[agent_id] = {}
            
            self.context_store[agent_id].update(updates)
            
            # Create update node
            node = ContextNode(
                node_id=f"{agent_id}-update-{int(time.time())}",
                content=updates,
                timestamp=datetime.now(),
                agent_id=agent_id,
                context_type="context_update",
                priority=2
            )
            self.context_nodes.append(node)
            
            logger.info(f"Updated context for agent {agent_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error updating context for agent {agent_id}: {e}")
            return False
    
    async def clear_context(self, agent_id: str) -> bool:
        """Clear context for a specific agent."""
        try:
            if agent_id in self.context_store:
                del self.context_store[agent_id]
            
            # Remove context nodes for this agent
            self.context_nodes = [node for node in self.context_nodes if node.agent_id != agent_id]
            
            logger.info(f"Cleared context for agent {agent_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error clearing context for agent {agent_id}: {e}")
            return False


class MicroAgentRegistry:
    """Manages all micro-agents with shared context."""
    
    def __init__(self, project_id: str):
        self.project_id = project_id
        self.agents: Dict[str, AgentContext] = {}
        self.context_manager = IntelliLlamaContextManager(project_id)
    
    async def register_agent(self, agent_id: str, agent_type: str) -> AgentContext:
        """Register a new agent."""
        agent_context = AgentContext(
            agent_id=agent_id,
            project_id=self.project_id,
            conversation_history=[],
            shared_knowledge={"agent_type": agent_type},
            last_updated=datetime.now(),
            context_size=0
        )
        
        self.agents[agent_id] = agent_context
        
        # Add initial context
        await self.context_manager.add_context(agent_id, {
            "agent_type": agent_type,
            "registered_at": datetime.now().isoformat()
        })
        
        logger.info(f"Registered agent {agent_id} of type {agent_type}")
        return agent_context
    
    async def get_agent_context(self, agent_id: str) -> AgentContext:
        """Get context for a specific agent."""
        if agent_id not in self.agents:
            raise ValueError(f"Agent {agent_id} not found")
        
        return self.agents[agent_id]
    
    async def update_agent_context(self, agent_id: str, updates: Dict[str, Any]) -> bool:
        """Update context for a specific agent."""
        if agent_id not in self.agents:
            return False
        
        # Update agent context
        self.agents[agent_id].shared_knowledge.update(updates)
        self.agents[agent_id].last_updated = datetime.now()
        
        # Update shared context
        await self.context_manager.update_context(agent_id, updates)
        
        return True
    
    async def get_shared_context(self) -> SharedContext:
        """Get shared context across all agents."""
        shared_knowledge = await self.context_manager.get_shared_context()
        
        return SharedContext(
            project_id=self.project_id,
            agents=self.agents.copy(),
            global_knowledge=shared_knowledge,
            conversation_summary="Active project with multiple agents",
            context_window=10000
        )
    
    async def broadcast_context_update(self, agent_id: str, update: Dict[str, Any]) -> bool:
        """Broadcast context update to all agents."""
        try:
            # Update the originating agent
            await self.update_agent_context(agent_id, update)
            
            # Add to global knowledge
            shared_context = await self.get_shared_context()
            shared_context.global_knowledge.update(update)
            
            logger.info(f"Broadcasted context update from agent {agent_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error broadcasting context update: {e}")
            return False


# Backward compatibility functions
async def stream_code_generation(prompt: str, language: Language = Language.PYTHON, 
                                model: str = None) -> AsyncGenerator[str, None]:
    """Stream code generation for backward compatibility."""
    request = StreamingCodeGenerationRequest(
        prompt=prompt,
        language=language,
        model=model
    )
    
    generator = StreamingCodeGenerator()
    async for chunk in generator.stream_code_generation(request):
        yield chunk


async def generate_code_streaming(prompt: str, language: Language = Language.PYTHON,
                                model: str = None) -> StreamingCodeGenerationResult:
    """Generate code with streaming and return full result."""
    request = StreamingCodeGenerationRequest(
        prompt=prompt,
        language=language,
        model=model
    )
    
    generator = StreamingCodeGenerator()
    chunks = []
    
    async for chunk in generator.stream_code_generation(request):
        chunks.append(chunk)
    
    # Process the chunks
    async def chunk_generator():
        for chunk in chunks:
            yield chunk
    
    return await generator.process_streaming_response(chunk_generator()) 