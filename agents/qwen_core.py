#!/usr/bin/env python3
"""
Qwen Agent Core - Core types, exceptions, and Ollama integration.

This module provides the fundamental types, exceptions, and base functionality
for the Qwen code generation agent system.
"""

import subprocess
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Language(Enum):
    """Supported programming languages."""
    PYTHON = "python"
    RUST = "rust"
    JAVASCRIPT = "javascript"
    TYPESCRIPT = "typescript"
    GO = "go"
    CPP = "cpp"
    JAVA = "java"
    CSHARP = "csharp"
    PHP = "php"
    RUBY = "ruby"
    SWIFT = "swift"
    KOTLIN = "kotlin"
    SCALA = "scala"
    R = "r"
    MATLAB = "matlab"
    SHELL = "shell"
    HTML = "html"
    CSS = "css"
    SQL = "sql"
    YAML = "yaml"
    JSON = "json"
    MARKDOWN = "markdown"
    UNKNOWN = "unknown"


@dataclass
class CodeGenerationRequest:
    """Request for code generation."""
    prompt: str
    language: Language
    context: Optional[str] = None
    requirements: Optional[List[str]] = None
    style_guide: Optional[str] = None
    model: str = "hf.co/unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF:Q4_K_M"


@dataclass
class CodeGenerationResult:
    """Result of code generation."""
    code: str
    language: Language
    success: bool
    error_message: Optional[str] = None
    warnings: Optional[List[str]] = None
    metadata: Optional[Dict[str, Any]] = None


class CodeGenerationError(Exception):
    """Custom exception for code generation errors."""
    pass


class OllamaConnectionError(CodeGenerationError):
    """Raised when Ollama connection fails."""
    pass


class CodeExtractionError(CodeGenerationError):
    """Raised when code extraction fails."""
    pass


class OllamaIntegration:
    """Handles all Ollama service integration and validation."""
    
    def __init__(self):
        self.timeout = 5  # Default timeout for health checks
    
    def check_ollama_available(self) -> bool:
        """Check if Ollama is available and running."""
        try:
            result = subprocess.run(
                ["ollama", "list"],
                capture_output=True,
                text=True,
                timeout=self.timeout
            )
            return result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False
    
    def check_model_available(self, model: str) -> bool:
        """Check if the specified model is available."""
        try:
            result = subprocess.run(
                ["ollama", "list"],
                capture_output=True,
                text=True,
                timeout=self.timeout
            )
            if result.returncode == 0:
                # Check if the model name appears in the output
                return model in result.stdout
            return False
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False
    
    def run_ollama_generation(self, model: str, prompt: str, timeout: int = 120) -> subprocess.CompletedProcess:
        """Execute Ollama code generation with proper error handling."""
        try:
            result = subprocess.run(
                ["ollama", "run", model, prompt],
                capture_output=True,
                text=True,
                timeout=timeout
            )
            return result
        except subprocess.TimeoutExpired:
            raise CodeGenerationError(f"Ollama request timed out after {timeout} seconds")
        except Exception as e:
            raise CodeGenerationError(f"Ollama execution failed: {str(e)}")


class QwenAgentBase:
    """Base class for Qwen code generation agent."""
    
    def __init__(self, default_model: str = (
        "hf.co/unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF:Q4_K_M"
    )):
        self.default_model = default_model
        self.supported_languages = list(Language)
        self.ollama = OllamaIntegration()
    
    def validate_request(self, request: CodeGenerationRequest) -> tuple[bool, str]:
        """Validate a code generation request."""
        if not request.prompt.strip():
            return False, "Prompt cannot be empty"
        
        if not request.language:
            return False, "Language must be specified"
        
        if request.language not in self.supported_languages:
            return False, f"Unsupported language: {request.language}"
        
        return True, ""
    
    def create_error_result(
        self, 
        request: CodeGenerationRequest, 
        error_message: str, 
        metadata: Optional[Dict[str, Any]] = None
    ) -> CodeGenerationResult:
        """Create a standardized error result."""
        return CodeGenerationResult(
            code="",
            language=request.language,
            success=False,
            error_message=error_message,
            warnings=[],
            metadata=metadata or {}
        )
    
    def create_success_result(
        self, 
        request: CodeGenerationRequest, 
        code: str, 
        warnings: List[str],
        metadata: Optional[Dict[str, Any]] = None
    ) -> CodeGenerationResult:
        """Create a standardized success result."""
        return CodeGenerationResult(
            code=code,
            language=request.language,
            success=True,
            warnings=warnings,
            metadata=metadata or {}
        )
    
    def generate_code(self, request: CodeGenerationRequest) -> CodeGenerationResult:
        """Generate code using Ollama integration - REAL implementation, no shortcuts."""
        # Validate request first
        is_valid, error_msg = self.validate_request(request)
        if not is_valid:
            return self.create_error_result(request, error_msg)
        
        # Check if Ollama is available
        if not self.ollama.check_ollama_available():
            return self.create_error_result(
                request, 
                "Ollama service is not available. Please ensure Ollama is running."
            )
        
        # Use the default model or request-specific model
        model = request.model if hasattr(request, 'model') and request.model else self.default_model
        
        # Check if the model is available
        if not self.ollama.check_model_available(model):
            return self.create_error_result(
                request,
                f"Model '{model}' is not available. Please ensure it's installed in Ollama."
            )
        
        try:
            # Create a comprehensive prompt for code generation
            system_prompt = f"""You are an expert {request.language.value} programmer. Generate clean, efficient, and well-commented code based on the user's request.

Requirements:
- Generate only the requested code, no explanations or markdown
- Follow best practices for {request.language.value}
- Include appropriate error handling where needed
- Make the code production-ready
- Do not include any conversational text, just the code"""

            if request.context:
                system_prompt += f"\n\nAdditional context: {request.context}"
            
            if request.requirements:
                system_prompt += f"\n\nSpecific requirements: {', '.join(request.requirements)}"
            
            full_prompt = f"{system_prompt}\n\nUser request: {request.prompt}"
            
            # Generate code using Ollama
            logger.info(f"Generating {request.language.value} code using model {model}")
            result = self.ollama.run_ollama_generation(model, full_prompt, timeout=120)
            
            if result.returncode != 0:
                error_msg = f"Ollama generation failed: {result.stderr}"
                logger.error(error_msg)
                return self.create_error_result(request, error_msg)
            
            # Extract and clean the generated code
            generated_code = result.stdout.strip()
            
            if not generated_code:
                return self.create_error_result(request, "No code was generated")
            
            # Remove markdown formatting if present
            if generated_code.startswith('```'):
                lines = generated_code.split('\n')
                # Remove first line with ```language
                lines = lines[1:]
                # Remove last line with ``` if present
                if lines and lines[-1].strip() == '```':
                    lines = lines[:-1]
                generated_code = '\n'.join(lines).strip()
            
            # Basic validation - ensure we got actual code, not an error message
            # But be more specific about what constitutes an error
            if (generated_code.lower().startswith('error:') or 
                generated_code.lower().startswith('failed:') or
                'unknown agent type' in generated_code.lower()):
                return self.create_error_result(
                    request,
                    f"Generation returned error instead of code: {generated_code[:200]}"
                )
            
            # Create success result with metadata
            metadata = {
                "model_used": model,
                "generation_time": "actual_generation",  # Real timing could be added here
                "prompt_length": len(full_prompt),
                "code_length": len(generated_code)
            }
            
            logger.info(f"Successfully generated {len(generated_code)} characters of {request.language.value} code")
            
            return self.create_success_result(
                request=request,
                code=generated_code,
                warnings=[],
                metadata=metadata
            )
            
        except Exception as e:
            error_msg = f"Code generation failed with exception: {str(e)}"
            logger.error(error_msg)
            return self.create_error_result(request, error_msg)


# Backward compatibility functions
def test_ollama_connection() -> bool:
    """Test if Ollama is available and working."""
    ollama = OllamaIntegration()
    return ollama.check_ollama_available()


def check_model_available(model: str) -> bool:
    """Check if a specific model is available."""
    ollama = OllamaIntegration()
    return ollama.check_model_available(model)