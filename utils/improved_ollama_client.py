#!/usr/bin/env python3
"""
Improved Ollama Client with HTTP API and Streaming Support

This module provides a modern HTTP-based Ollama client with:
- Structured JSON output support 
- Streaming capabilities
- Function calling support for Qwen3-Coder
- Proper timeout handling
- Error handling and fallbacks
"""

import asyncio
import json
import logging
import time
from typing import Dict, Any, List, Optional, AsyncGenerator, Union
from dataclasses import dataclass
from enum import Enum

import aiohttp

logger = logging.getLogger(__name__)


class TaskComplexity(Enum):
    """Task complexity levels for timeout determination."""
    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"


@dataclass
class OllamaRequest:
    """Request structure for Ollama API calls."""
    prompt: str
    model: str = "qwen3-coder"
    complexity: TaskComplexity = TaskComplexity.MODERATE
    context: Optional[str] = None
    language: Optional[str] = None
    requirements: Optional[List[str]] = None
    use_thinking: bool = True
    stream: bool = True
    structured_output: bool = False


@dataclass
class OllamaResponse:
    """Response structure from Ollama API."""
    content: str
    success: bool = True
    error: Optional[str] = None
    metadata: Dict[str, Any] = None
    tokens_used: Optional[int] = None
    response_time: Optional[float] = None
    parsed_data: Optional[Dict[str, str]] = None  # Parsed markdown content


class ImprovedOllamaClient:
    """Modern HTTP-based Ollama client with streaming and JSON support."""
    
    def __init__(self, base_url: str = "http://localhost:11434"):
        """
        Initialize the improved Ollama client.
        
        Args:
            base_url: Base URL for Ollama server
        """
        self.base_url = base_url
        self.session = None
        
        # Timeout configuration based on task complexity
        self.timeouts = {
            TaskComplexity.SIMPLE: 120,    # 2 minutes for simple tasks
            TaskComplexity.MODERATE: 240,  # 4 minutes for moderate tasks
            TaskComplexity.COMPLEX: 360    # 6 minutes for complex tasks
        }
        
        # Model configuration optimized for Qwen3-Coder
        self.model_options = {
            "num_ctx": 40960,        # Large context window
            "num_predict": 2048,     # Max tokens to generate
            "temperature": 0.7,      # Good balance of creativity and determinism
            "top_p": 0.8,           # Nucleus sampling
            "top_k": 20,            # Top-k sampling
            "repeat_penalty": 1.05,  # Reduce repetition
            "stop": ["\n\n\n", "```\n\n"],  # Stop on excessive whitespace
        }
    
    async def __aenter__(self):
        """Async context manager entry."""
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self.session:
            await self.session.close()
    
    def _build_structured_prompt(self, request: OllamaRequest) -> str:
        """Build a well-structured markdown prompt for Qwen3-Coder."""
        prompt_parts = []
        
        # Use markdown format for better structure and reliability
        prompt_parts.append("# Code Generation Request")
        prompt_parts.append("")
        
        # Main task
        if request.language:
            prompt_parts.append(f"## Task")
            prompt_parts.append(f"Generate {request.language} code for: {request.prompt}")
        else:
            prompt_parts.append(f"## Task")
            prompt_parts.append(request.prompt)
        
        prompt_parts.append("")
        
        # Context
        if request.context:
            prompt_parts.append("## Context")
            prompt_parts.append(request.context)
            prompt_parts.append("")
        
        # Requirements
        if request.requirements:
            prompt_parts.append("## Requirements")
            for req in request.requirements:
                prompt_parts.append(f"- {req}")
            prompt_parts.append("")
        
        # Output format instructions (markdown-based)
        if request.language:
            prompt_parts.append("## Output Format")
            prompt_parts.append("Please provide your response in the following markdown format:")
            prompt_parts.append("")
            prompt_parts.append("### Code")
            prompt_parts.append(f"```{request.language}")
            prompt_parts.append("// Your code here")
            prompt_parts.append("```")
            prompt_parts.append("")
            prompt_parts.append("### Explanation")
            prompt_parts.append("Brief explanation of the code and how it works.")
            prompt_parts.append("")
            if request.requirements:
                prompt_parts.append("### Dependencies")
                prompt_parts.append("List any required dependencies or imports.")
                prompt_parts.append("")
                prompt_parts.append("### Usage Example")
                prompt_parts.append(f"```{request.language}")
                prompt_parts.append("// Example usage")
                prompt_parts.append("```")
                prompt_parts.append("")
        
        # Instructions
        prompt_parts.append("## Instructions")
        prompt_parts.append("- Provide clean, well-documented code with proper error handling")
        prompt_parts.append("- Include clear comments explaining the logic")
        prompt_parts.append("- Follow best practices for the specified language")
        prompt_parts.append("- Ensure code is production-ready")
        
        # Build final prompt
        structured_prompt = "\n".join(prompt_parts)
        
        # Add thinking mode if requested (Qwen3-Coder specific)
        if request.use_thinking:
            structured_prompt = structured_prompt + "\n\n**Think through this step by step before generating the code.**"
        
        return structured_prompt
    
    def _parse_markdown_response(self, response_content: str) -> Dict[str, str]:
        """
        Parse markdown-formatted response to extract structured data.
        
        Args:
            response_content: Raw response content from Ollama
            
        Returns:
            Parsed response with code, explanation, dependencies, etc.
        """
        import re
        
        parsed = {
            "raw_content": response_content,
            "code": "",
            "explanation": "",
            "dependencies": "",
            "usage_example": ""
        }
        
        # Extract code blocks
        code_pattern = r'```(\w+)?\n(.*?)```'
        code_matches = re.findall(code_pattern, response_content, re.DOTALL)
        
        if code_matches:
            # First code block is usually the main code
            parsed["code"] = code_matches[0][1].strip()
            
            # Last code block might be usage example
            if len(code_matches) > 1:
                parsed["usage_example"] = code_matches[-1][1].strip()
        
        # Extract explanation (text between ### Explanation and next ###)
        explanation_pattern = r'### Explanation\s*\n(.*?)(?=###|\Z)'
        explanation_match = re.search(explanation_pattern, response_content, re.DOTALL)
        if explanation_match:
            parsed["explanation"] = explanation_match.group(1).strip()
        
        # Extract dependencies
        deps_pattern = r'### Dependencies\s*\n(.*?)(?=###|\Z)'
        deps_match = re.search(deps_pattern, response_content, re.DOTALL)
        if deps_match:
            parsed["dependencies"] = deps_match.group(1).strip()
        
        return parsed
    
    async def generate_response(self, request: OllamaRequest) -> OllamaResponse:
        """
        Generate response using HTTP API with streaming support.
        
        Args:
            request: Ollama request configuration
            
        Returns:
            Ollama response with content and metadata
        """
        start_time = time.time()
        
        try:
            # Build the request payload
            payload = {
                "model": request.model,
                "messages": [
                    {
                        "role": "user", 
                        "content": self._build_structured_prompt(request)
                    }
                ],
                "stream": request.stream,
                "options": self.model_options.copy()
            }
            
            # Note: Using markdown format instead of function calling for better reliability
            
            # Determine timeout
            timeout = self.timeouts.get(request.complexity, 240)
            
            # Make the request
            if not self.session:
                self.session = aiohttp.ClientSession()
            
            if request.stream:
                return await self._handle_streaming_response(payload, timeout, start_time, request)
            else:
                return await self._handle_non_streaming_response(payload, timeout, start_time, request)
                
        except Exception as e:
            logger.error(f"Ollama request failed: {e}")
            return OllamaResponse(
                content="",
                success=False,
                error=str(e),
                response_time=time.time() - start_time
            )
    
    async def _handle_streaming_response(self, payload: Dict[str, Any], timeout: int, start_time: float, request: OllamaRequest) -> OllamaResponse:
        """Handle streaming response from Ollama."""
        content_chunks = []
        tokens_used = 0
        
        try:
            async with self.session.post(
                f"{self.base_url}/api/chat",
                json=payload,
                timeout=aiohttp.ClientTimeout(total=timeout)
            ) as response:
                
                if response.status != 200:
                    error_text = await response.text()
                    return OllamaResponse(
                        content="",
                        success=False,
                        error=f"HTTP {response.status}: {error_text}",
                        response_time=time.time() - start_time
                    )
                
                async for line in response.content:
                    if line:
                        try:
                            chunk_data = json.loads(line.decode())
                            
                            # Extract content from different response formats
                            if "message" in chunk_data and "content" in chunk_data["message"]:
                                chunk_content = chunk_data["message"]["content"]
                                if chunk_content:
                                    content_chunks.append(chunk_content)
                            elif "response" in chunk_data:
                                chunk_content = chunk_data["response"]
                                if chunk_content:
                                    content_chunks.append(chunk_content)
                            
                            # Track token usage if available
                            if "eval_count" in chunk_data:
                                tokens_used = chunk_data["eval_count"]
                                
                        except json.JSONDecodeError:
                            # Skip malformed JSON lines
                            continue
                
                full_content = "".join(content_chunks)
                
                # Parse markdown if structured output was requested
                parsed_data = None
                if request.structured_output and full_content:
                    parsed_data = self._parse_markdown_response(full_content)
                
                return OllamaResponse(
                    content=full_content,
                    success=True,
                    metadata={
                        "chunks_received": len(content_chunks),
                        "tokens_used": tokens_used,
                        "streaming": True
                    },
                    tokens_used=tokens_used,
                    response_time=time.time() - start_time,
                    parsed_data=parsed_data
                )
                
        except asyncio.TimeoutError:
            return OllamaResponse(
                content="".join(content_chunks),
                success=False,
                error=f"Request timed out after {timeout} seconds",
                response_time=time.time() - start_time
            )
        except Exception as e:
            return OllamaResponse(
                content="".join(content_chunks),
                success=False,
                error=str(e),
                response_time=time.time() - start_time
            )
    
    async def _handle_non_streaming_response(self, payload: Dict[str, Any], timeout: int, start_time: float, request: OllamaRequest) -> OllamaResponse:
        """Handle non-streaming response from Ollama."""
        try:
            async with self.session.post(
                f"{self.base_url}/api/chat",
                json=payload,
                timeout=aiohttp.ClientTimeout(total=timeout)
            ) as response:
                
                if response.status != 200:
                    error_text = await response.text()
                    return OllamaResponse(
                        content="",
                        success=False,
                        error=f"HTTP {response.status}: {error_text}",
                        response_time=time.time() - start_time
                    )
                
                response_data = await response.json()
                
                # Extract content from response
                content = ""
                if "message" in response_data and "content" in response_data["message"]:
                    content = response_data["message"]["content"]
                elif "response" in response_data:
                    content = response_data["response"]
                
                tokens_used = response_data.get("eval_count", 0)
                
                # Parse markdown if structured output was requested
                parsed_data = None
                if request.structured_output and content:
                    parsed_data = self._parse_markdown_response(content)
                
                return OllamaResponse(
                    content=content,
                    success=True,
                    metadata={
                        "tokens_used": tokens_used,
                        "streaming": False
                    },
                    tokens_used=tokens_used,
                    response_time=time.time() - start_time,
                    parsed_data=parsed_data
                )
                
        except asyncio.TimeoutError:
            return OllamaResponse(
                content="",
                success=False,
                error=f"Request timed out after {timeout} seconds",
                response_time=time.time() - start_time
            )
        except Exception as e:
            return OllamaResponse(
                content="",
                success=False,
                error=str(e),
                response_time=time.time() - start_time
            )
    
    async def stream_response(self, request: OllamaRequest) -> AsyncGenerator[str, None]:
        """
        Stream response chunks in real-time.
        
        Args:
            request: Ollama request configuration
            
        Yields:
            Content chunks as they arrive
        """
        payload = {
            "model": request.model,
            "messages": [
                {
                    "role": "user", 
                    "content": self._build_structured_prompt(request)
                }
            ],
            "stream": True,
            "options": self.model_options.copy()
        }
        
        # Note: Using markdown format for structured output instead of function calling
        
        timeout = self.timeouts.get(request.complexity, 240)
        
        try:
            if not self.session:
                self.session = aiohttp.ClientSession()
            
            async with self.session.post(
                f"{self.base_url}/api/chat",
                json=payload,
                timeout=aiohttp.ClientTimeout(total=timeout)
            ) as response:
                
                if response.status != 200:
                    error_text = await response.text()
                    yield f"Error: HTTP {response.status}: {error_text}"
                    return
                
                async for line in response.content:
                    if line:
                        try:
                            chunk_data = json.loads(line.decode())
                            
                            # Extract and yield content
                            if "message" in chunk_data and "content" in chunk_data["message"]:
                                chunk_content = chunk_data["message"]["content"]
                                if chunk_content:
                                    yield chunk_content
                            elif "response" in chunk_data:
                                chunk_content = chunk_data["response"]
                                if chunk_content:
                                    yield chunk_content
                                    
                        except json.JSONDecodeError:
                            continue
                            
        except Exception as e:
            yield f"Error: {str(e)}"
    
    async def check_health(self) -> bool:
        """Check if Ollama server is accessible."""
        try:
            if not self.session:
                self.session = aiohttp.ClientSession()
            
            async with self.session.get(
                f"{self.base_url}/api/tags",
                timeout=aiohttp.ClientTimeout(total=10)
            ) as response:
                return response.status == 200
                
        except Exception:
            return False
    
    @staticmethod
    def determine_complexity(prompt: str, context: Optional[str] = None) -> TaskComplexity:
        """
        Determine task complexity based on prompt content.
        
        Args:
            prompt: The task prompt
            context: Optional context information
            
        Returns:
            Task complexity level
        """
        prompt_lower = prompt.lower()
        combined_text = f"{prompt} {context or ''}".lower()
        
        # Complex indicators
        complex_keywords = [
            "algorithm", "optimization", "performance", "concurrent", "parallel", 
            "distributed", "architecture", "framework", "system", "database",
            "api", "microservice", "docker", "kubernetes", "cloud", "ml", "ai",
            "neural", "model", "data structure", "graph", "tree", "sort", "search"
        ]
        
        # Simple indicators
        simple_keywords = [
            "hello world", "basic", "simple", "example", "demo", "test",
            "print", "echo", "fix typo", "syntax error", "variable", "function"
        ]
        
        # Count complexity indicators
        complex_count = sum(1 for keyword in complex_keywords if keyword in combined_text)
        simple_count = sum(1 for keyword in simple_keywords if keyword in combined_text)
        
        # Length-based complexity
        if len(prompt) > 500 or (context and len(context) > 200):
            complex_count += 1
        elif len(prompt) < 50:
            simple_count += 1
        
        # Determine complexity
        if complex_count >= 2 or "class" in prompt_lower and "method" in prompt_lower:
            return TaskComplexity.COMPLEX
        elif simple_count >= 2 and complex_count == 0:
            return TaskComplexity.SIMPLE
        else:
            return TaskComplexity.MODERATE


# Convenience functions for backward compatibility
async def generate_ollama_response(
    prompt: str,
    model: str = "qwen3-coder",
    context: Optional[str] = None,
    language: Optional[str] = None,
    requirements: Optional[List[str]] = None,
    use_streaming: bool = True
) -> OllamaResponse:
    """
    Generate response using improved Ollama client.
    
    Args:
        prompt: The task prompt
        model: Model to use
        context: Optional context
        language: Programming language for code generation
        requirements: List of requirements
        use_streaming: Whether to use streaming
        
    Returns:
        Ollama response
    """
    async with ImprovedOllamaClient() as client:
        request = OllamaRequest(
            prompt=prompt,
            model=model,
            complexity=ImprovedOllamaClient.determine_complexity(prompt, context),
            context=context,
            language=language,
            requirements=requirements,
            stream=use_streaming,
            structured_output=language is not None
        )
        
        return await client.generate_response(request)