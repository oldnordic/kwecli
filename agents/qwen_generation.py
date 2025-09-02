#!/usr/bin/env python3
"""
Qwen Agent Code Generation - Code generation logic and prompt building.

This module handles the core code generation functionality including
prompt building, language detection, and generation coordination.
"""

import logging
import time
from typing import Dict, List
from .qwen_core import (
    Language, CodeGenerationRequest, CodeGenerationResult,
    QwenAgentBase, CodeGenerationError
)

# Configure logging
logger = logging.getLogger(__name__)


class LanguageDetector:
    """Handles programming language detection from prompts and context."""
    
    def __init__(self):
        self.language_keywords = {
            Language.PYTHON: [
                "python", "def ", "import ", "from ", "class ",
                "async def", "print(", "if __name__"
            ],
            Language.RUST: [
                "rust", "fn ", "let ", "struct ", "impl ",
                "use ", "pub ", "mod "
            ],
            Language.JAVASCRIPT: [
                "javascript", "js", "function ", "const ",
                "let ", "var ", "console.log"
            ],
            Language.TYPESCRIPT: [
                "typescript", "ts", "interface ", "type ",
                "enum ", "export "
            ],
            Language.GO: [
                "go", "golang", "func ", "package ",
                "import ", "fmt."
            ],
            Language.CPP: [
                "cpp", "c++", "#include ", "class ",
                "namespace ", "std::"
            ],
            Language.JAVA: [
                "java", "public class ", "public static ",
                "import ", "System.out"
            ],
            Language.CSHARP: [
                "c#", "csharp", "using ", "namespace ",
                "public class ", "Console."
            ],
            Language.PHP: [
                "php", "<?php", "function ", "class ",
                "echo "
            ],
            Language.RUBY: [
                "ruby", "def ", "class ", "module ",
                "require ", "puts "
            ],
            Language.SWIFT: [
                "swift", "func ", "class ", "import ",
                "print("
            ],
            Language.KOTLIN: [
                "kotlin", "fun ", "class ", "import ",
                "println("
            ],
            Language.SCALA: [
                "scala", "def ", "class ", "import ",
                "println("
            ],
            Language.R: [
                "r language", "r programming", "function(",
                "<-", "print("
            ],
            Language.MATLAB: [
                "matlab", "function ", "end", "disp("
            ],
            Language.SHELL: [
                "bash", "shell", "#!/", "echo ",
                "if [", "for "
            ],
            Language.HTML: [
                "html", "<html", "<div", "<span",
                "<!DOCTYPE"
            ],
            Language.CSS: [
                "css", "{", "}", "color:",
                "background:", ".css"
            ],
            Language.SQL: [
                "sql", "select ", "insert ", "update ",
                "delete ", "create table"
            ],
            Language.YAML: [
                "yaml", "yml", "---", "key:",
                "version:"
            ],
            Language.JSON: [
                "json", "{", "}", '"key":',
                '"value"'
            ],
            Language.MARKDOWN: [
                "markdown", "md", "# ", "## ",
                "```"
            ]
        }
    
    def detect_language(self, prompt: str) -> Language:
        """Detect programming language from prompt or context."""
        prompt_lower = prompt.lower()

        # Count matches for each language with actual scoring
        language_scores = {}
        for lang, keywords in self.language_keywords.items():
            score = sum(1 for keyword in keywords if keyword in prompt_lower)
            if score > 0:
                language_scores[lang] = score

        # Return the language with the highest score, or Python as default
        if language_scores:
            return max(language_scores.items(), key=lambda x: x[1])[0]

        return Language.PYTHON  # Default to Python


class PromptBuilder:
    """Handles building comprehensive prompts for code generation."""
    
    def __init__(self):
        pass
    
    def build_prompt(self, request: CodeGenerationRequest) -> str:
        """Build a comprehensive prompt for code generation."""
        language_name = request.language.value

        # Real system prompt with actual instructions
        system_prompt = (
            f"You are an expert {language_name} programmer. "
            f"Generate high-quality, production-ready code based on "
            f"the user's requirements.\n\n"
            f"Requirements:\n"
            f"- Write clean, well-documented code\n"
            f"- Follow {language_name} best practices and conventions\n"
            f"- Include proper error handling\n"
            f"- Add appropriate comments and docstrings\n"
            f"- Ensure the code is functional and complete\n"
            f"- Use modern {language_name} features when appropriate\n"
            f"- Include type hints if supported by the language\n"
            f"- Add input validation where necessary\n\n"
            f"Please provide only the code without any explanations "
            f"or markdown formatting."
        )

        # Add context if provided
        if request.context:
            system_prompt += f"\n\nContext:\n{request.context}"

        # Add requirements if provided
        if request.requirements:
            system_prompt += (
                "\n\nSpecific Requirements:\n" + "\n".join(
                    "- " + req for req in request.requirements
                )
            )

        # Add style guide if provided
        if request.style_guide:
            system_prompt += f"\n\nStyle Guide:\n{request.style_guide}"

        # Add user prompt
        user_prompt = f"Generate {language_name} code for: {request.prompt}"

        return f"{system_prompt}\n\n{user_prompt}"


class CodeGenerationEngine:
    """Core code generation engine that coordinates all generation activities."""
    
    def __init__(self, agent_base: QwenAgentBase):
        self.agent_base = agent_base
        self.language_detector = LanguageDetector()
        self.prompt_builder = PromptBuilder()
    
    def prepare_request(self, request: CodeGenerationRequest) -> CodeGenerationRequest:
        """Prepare and validate a code generation request."""
        # Detect language if not specified
        if not request.language:
            request.language = self.language_detector.detect_language(request.prompt)
        
        return request
    
    async def execute_generation(self, request: CodeGenerationRequest) -> CodeGenerationResult:
        """Execute the complete code generation workflow."""
        start_time = time.time()
        
        try:
            # Validate request
            is_valid, error_msg = self.agent_base.validate_request(request)
            if not is_valid:
                return self.agent_base.create_error_result(
                    request, error_msg, {"error": "validation_failed"}
                )
            
            # Check if Ollama is available
            if not self.agent_base.ollama.check_ollama_available():
                return self.agent_base.create_error_result(
                    request,
                    "Ollama is not available. Please install and start Ollama first.",
                    {"error": "ollama_not_available"}
                )

            # Check if model is available
            if not self.agent_base.ollama.check_model_available(request.model):
                return self.agent_base.create_error_result(
                    request,
                    f"Model '{request.model}' is not available. "
                    f"Please pull it first with 'ollama pull {request.model}'",
                    {"error": "model_not_available", "model": request.model}
                )

            # Prepare request (language detection, etc.)
            prepared_request = self.prepare_request(request)

            # Build comprehensive prompt
            full_prompt = self.prompt_builder.build_prompt(prepared_request)

            logger.info(
                f"Generating {prepared_request.language.value} code with model "
                f"{prepared_request.model}"
            )

            # Execute Ollama generation
            result = self.agent_base.ollama.run_ollama_generation(
                prepared_request.model, full_prompt, timeout=120
            )

            if result.returncode != 0:
                return self.agent_base.create_error_result(
                    prepared_request,
                    f"Ollama failed with return code "
                    f"{result.returncode}: {result.stderr}",
                    {
                        "ollama_return_code": result.returncode,
                        "stderr": result.stderr
                    }
                )

            execution_time = time.time() - start_time
            
            # Return raw result for further processing by parsing module
            metadata = {
                "model": prepared_request.model,
                "language": prepared_request.language.value,
                "prompt_length": len(prepared_request.prompt),
                "response_length": len(result.stdout),
                "ollama_return_code": result.returncode,
                "execution_time": execution_time,
                "raw_response": result.stdout
            }
            
            # This will be processed by the parsing module
            return CodeGenerationResult(
                code=result.stdout,  # Raw response, to be parsed
                language=prepared_request.language,
                success=True,  # Initial success, validation comes later
                warnings=[],
                metadata=metadata
            )

        except CodeGenerationError as e:
            logger.error(f"Code generation error: {e}")
            return self.agent_base.create_error_result(
                request, str(e), {"error": "generation_error"}
            )
        except Exception as e:
            logger.error(f"Unexpected error during code generation: {e}")
            return self.agent_base.create_error_result(
                request, 
                f"Unexpected error: {str(e)}", 
                {"error": "unexpected", "exception": str(e)}
            )