#!/usr/bin/env python3
"""
Qwen Agent Parsing - Code extraction and response parsing.

This module handles parsing of Ollama responses, extracting code blocks,
and cleaning up the generated code for validation and use.
"""

import re
import logging
from typing import List
from .qwen_core import Language, CodeGenerationResult

# Configure logging
logger = logging.getLogger(__name__)


class ResponseCleaner:
    """Handles cleaning of raw LLM responses."""
    
    def __init__(self):
        self.artifacts_to_remove = [
            r'^python\s*$',  # Standalone "python" text
            r'^```python\s*$',  # Incomplete code block markers
            r'^```\s*$',  # Incomplete code block markers
            r'^\s*python\s*$',  # Standalone "python" with whitespace
            r'^\s*```python\s*$',  # Incomplete code block markers
            r'^\s*```\s*$',  # Incomplete code block markers
        ]
    
    def clean_response(self, response: str) -> str:
        """Clean the LLM response to remove common artifacts."""
        cleaned = response
        
        # Remove common LLM artifacts
        for pattern in self.artifacts_to_remove:
            cleaned = re.sub(
                pattern, '', cleaned, flags=re.MULTILINE | re.IGNORECASE
            )

        # Remove empty lines at the beginning and end
        lines = cleaned.split('\n')
        while lines and not lines[0].strip():
            lines.pop(0)
        while lines and not lines[-1].strip():
            lines.pop()

        return '\n'.join(lines)


class CodeBlockExtractor:
    """Extracts code blocks from LLM responses."""
    
    def __init__(self):
        self.response_cleaner = ResponseCleaner()
    
    def extract_code_blocks(self, response: str, language: Language) -> List[str]:
        """Extract code blocks from Ollama response with robust parsing."""
        language_name = language.value

        # Clean the response first
        cleaned_response = self.response_cleaner.clean_response(response)

        # Multiple patterns for code block extraction
        patterns = [
            rf"```{language_name}\s*(.*?)\s*```",  # ```python ... ```
            rf"```\s*(.*?)\s*```",  # ``` ... ```
            rf"`{language_name}\s*(.*?)\s*`",  # `python ... `
            rf"`\s*(.*?)\s*`",  # ` ... `
        ]

        extracted_blocks = []

        for pattern in patterns:
            matches = re.findall(
                pattern, cleaned_response, re.DOTALL | re.IGNORECASE
            )
            if matches:
                for match in matches:
                    cleaned_code = self._clean_code_block(match.strip(), language)
                    if cleaned_code:
                        extracted_blocks.append(cleaned_code)

        # If no code blocks found, try to extract the entire response
        if not extracted_blocks and cleaned_response.strip():
            # Check if the response looks like code
            if self._looks_like_code(cleaned_response, language):
                cleaned_code = self._clean_code_block(
                    cleaned_response.strip(), language
                )
                if cleaned_code:
                    extracted_blocks.append(cleaned_code)

        return extracted_blocks
    
    def _looks_like_code(self, response: str, language: Language) -> bool:
        """Check if the response looks like code based on language indicators."""
        code_indicators = {
            Language.PYTHON: ["def ", "class ", "import ", "from ", "print(", "if ", "for ", "while "],
            Language.RUST: ["fn ", "let ", "struct ", "impl ", "use ", "pub ", "mod "],
            Language.JAVASCRIPT: ["function ", "const ", "let ", "var ", "console.log"],
            Language.TYPESCRIPT: ["interface ", "type ", "enum ", "export "],
            Language.GO: ["func ", "package ", "import ", "fmt."],
            Language.CPP: ["#include ", "class ", "namespace ", "std::"],
            Language.JAVA: ["public class ", "public static ", "import ", "System.out"],
            Language.CSHARP: ["using ", "namespace ", "public class ", "Console."],
            Language.PHP: ["<?php", "function ", "class ", "echo "],
            Language.RUBY: ["def ", "class ", "module ", "require ", "puts "],
            Language.SWIFT: ["func ", "class ", "import ", "print("],
            Language.KOTLIN: ["fun ", "class ", "import ", "println("],
            Language.SCALA: ["def ", "class ", "import ", "println("],
            Language.R: ["function(", "<-", "print("],
            Language.MATLAB: ["function ", "end", "disp("],
            Language.SHELL: ["#!/", "echo ", "if [", "for "],
            Language.HTML: ["<html", "<div", "<span", "<!DOCTYPE"],
            Language.CSS: ["{", "}", "color:", "background:"],
            Language.SQL: ["select ", "insert ", "update ", "delete ", "create table"],
            Language.YAML: ["---", "key:", "version:"],
            Language.JSON: ["{", "}", '"key":', '"value"'],
            Language.MARKDOWN: ["# ", "## ", "```"]
        }
        
        indicators = code_indicators.get(language, [])
        return any(indicator in response for indicator in indicators)
    
    def _clean_code_block(self, code: str, language: Language) -> str:
        """Clean a code block to make it executable."""
        if not code.strip():
            return ""

        # Remove language identifiers that might be at the start
        language_patterns = [
            r'^python\s*$',  # Standalone "python"
            r'^```python\s*$',  # "```python"
            r'^```\s*$',  # "```"
            r'^`python\s*$',  # "`python"
            r'^`\s*$',  # "`"
        ]

        cleaned = code
        for pattern in language_patterns:
            cleaned = re.sub(
                pattern, '', cleaned, flags=re.MULTILINE | re.IGNORECASE
            )

        # Remove empty lines at the beginning and end
        lines = cleaned.split('\n')
        while lines and not lines[0].strip():
            lines.pop(0)
        while lines and not lines[-1].strip():
            lines.pop()

        cleaned = '\n'.join(lines)

        # Validate that we have actual code content
        if not cleaned.strip():
            return ""

        # Basic validation for Python
        if language == Language.PYTHON:
            # For Python, ensure we have at least some Python syntax
            python_keywords = ['def ', 'class ', 'import ', 'print(', 'if ', 'for ', 'while ']
            if not any(keyword in cleaned for keyword in python_keywords):
                return ""

        return cleaned


class ResponseParser:
    """Main parser that coordinates response processing."""
    
    def __init__(self):
        self.extractor = CodeBlockExtractor()
    
    def parse_response(self, result: CodeGenerationResult) -> CodeGenerationResult:
        """Parse a raw generation result and extract clean code."""
        if not result.success:
            return result
        
        # Extract raw response from metadata
        raw_response = result.metadata.get("raw_response", result.code)
        
        # Extract code blocks
        code_blocks = self.extractor.extract_code_blocks(raw_response, result.language)
        
        if not code_blocks:
            return CodeGenerationResult(
                code="",
                language=result.language,
                success=False,
                error_message="No code blocks found in Ollama response",
                warnings=[],
                metadata={**result.metadata, "response": raw_response[:500]}
            )
        
        # Combine all code blocks
        combined_code = "\n\n".join(code_blocks)
        
        # Update result with parsed code
        updated_metadata = result.metadata.copy()
        updated_metadata.update({
            "code_blocks_found": len(code_blocks),
            "parsed_code_length": len(combined_code)
        })
        
        return CodeGenerationResult(
            code=combined_code,
            language=result.language,
            success=True,  # Will be validated by validation module
            warnings=result.warnings,
            metadata=updated_metadata
        )