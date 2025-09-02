#!/usr/bin/env python3
"""
Qwen Agent Validation - Code validation and error handling.

This module handles validation of generated code, syntax checking,
and comprehensive error analysis for different programming languages.
"""

import re
import logging
from typing import Tuple, List
from .qwen_core import Language, CodeGenerationResult

# Configure logging
logger = logging.getLogger(__name__)


class CodeValidator:
    """Handles validation of generated code for different languages."""
    
    def __init__(self):
        pass
    
    def validate_code(self, code: str, language: Language) -> Tuple[bool, List[str]]:
        """Real code validation with actual checks."""
        warnings = []

        # Check for common issues
        if not code.strip():
            warnings.append("Generated code is empty")
            return False, warnings

        # Check for malformed code patterns
        if self._has_malformed_patterns(code):
            warnings.append("Code contains malformed patterns")
            return False, warnings

        # Language-specific validation
        if language == Language.PYTHON:
            return self._validate_python(code, warnings)
        elif language == Language.RUST:
            return self._validate_rust(code, warnings)
        elif language == Language.JAVASCRIPT:
            return self._validate_javascript(code, warnings)
        elif language == Language.GO:
            return self._validate_go(code, warnings)
        else:
            # Basic validation for other languages
            return self._validate_generic(code, warnings)
    
    def _has_malformed_patterns(self, code: str) -> bool:
        """Check for malformed code patterns."""
        malformed_patterns = [
            r'^\s*python\s*$',  # Just "python" on a line
            r'^\s*```python\s*$',  # Just "```python" on a line
            r'^\s*```\s*$',  # Just "```" on a line
        ]

        for pattern in malformed_patterns:
            if re.search(pattern, code, re.MULTILINE | re.IGNORECASE):
                return True
        return False
    
    def _validate_python(self, code: str, warnings: List[str]) -> Tuple[bool, List[str]]:
        """Validate Python code with syntax checking."""
        # Check for basic Python syntax patterns
        if "def " in code and ":" not in code:
            warnings.append("Function definition missing colon")
        if "import " in code and "from " not in code:
            warnings.append("Import statement may be incomplete")
        if "print(" in code and ")" not in code:
            warnings.append("Print statement incomplete")

        # Try to compile the Python code
        try:
            compile(code, '<string>', 'exec')
            return True, warnings
        except SyntaxError as e:
            warnings.append(f"Python syntax error: {e}")
            return False, warnings
    
    def _validate_rust(self, code: str, warnings: List[str]) -> Tuple[bool, List[str]]:
        """Validate Rust code patterns."""
        if "fn " in code and "{" not in code:
            warnings.append("Function definition missing braces")
        if "let " in code and ";" not in code:
            warnings.append("Variable declaration missing semicolon")
        if "use " in code and ";" not in code:
            warnings.append("Use statement missing semicolon")
        
        return True, warnings
    
    def _validate_javascript(self, code: str, warnings: List[str]) -> Tuple[bool, List[str]]:
        """Validate JavaScript code patterns."""
        if "function " in code and "{" not in code:
            warnings.append("Function definition missing braces")
        if "const " in code and "=" not in code:
            warnings.append("Constant declaration missing assignment")
        if "console.log(" in code and ")" not in code:
            warnings.append("Console.log statement incomplete")
        
        return True, warnings
    
    def _validate_go(self, code: str, warnings: List[str]) -> Tuple[bool, List[str]]:
        """Validate Go code patterns."""
        if "func " in code and "{" not in code:
            warnings.append("Function definition missing braces")
        if ("package " in code and "main" not in code and "fmt" not in code):
            warnings.append("Package declaration may be incomplete")
        
        return True, warnings
    
    def _validate_generic(self, code: str, warnings: List[str]) -> Tuple[bool, List[str]]:
        """Generic validation for unsupported languages."""
        # Basic checks for any language
        if len(code.strip()) < 10:
            warnings.append("Generated code seems too short")
        
        return True, warnings


class ResultValidator:
    """Validates and finalizes code generation results."""
    
    def __init__(self):
        self.code_validator = CodeValidator()
    
    def validate_result(self, result: CodeGenerationResult) -> CodeGenerationResult:
        """Validate a code generation result and finalize it."""
        if not result.success:
            return result
        
        # Validate the code
        is_valid, validation_warnings = self.code_validator.validate_code(
            result.code, result.language
        )
        
        # Combine existing warnings with validation warnings
        all_warnings = (result.warnings or []) + validation_warnings
        
        # Update metadata with validation information
        updated_metadata = result.metadata.copy() if result.metadata else {}
        updated_metadata.update({
            "validation_performed": True,
            "validation_warnings_count": len(validation_warnings),
            "final_code_length": len(result.code)
        })
        
        return CodeGenerationResult(
            code=result.code,
            language=result.language,
            success=is_valid,
            warnings=all_warnings,
            metadata=updated_metadata
        )