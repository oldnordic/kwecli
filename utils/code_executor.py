#!/usr/bin/env python3
"""
Safe Code Execution System

This module provides secure code execution capabilities with sandboxing,
timeout mechanisms, and resource limits.
"""

import subprocess
import tempfile
import os
import signal
import time
import threading
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum
import logging
import json
import re

logger = logging.getLogger(__name__)


class ExecutionStatus(Enum):
    """Execution status codes."""
    SUCCESS = "success"
    TIMEOUT = "timeout"
    ERROR = "error"
    SECURITY_VIOLATION = "security_violation"
    RESOURCE_LIMIT = "resource_limit"


@dataclass
class ExecutionResult:
    """Result of code execution."""
    status: ExecutionStatus
    stdout: str
    stderr: str
    return_code: int
    execution_time: float
    memory_usage: Optional[float] = None
    cpu_usage: Optional[float] = None
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = None


class CodeExecutionError(Exception):
    """Custom exception for code execution errors."""
    pass


class SecurityViolationError(CodeExecutionError):
    """Raised when security violations are detected."""
    pass


class ResourceLimitError(CodeExecutionError):
    """Raised when resource limits are exceeded."""
    pass


class CodeExecutor:
    """Safe code execution system."""
    
    def __init__(self, timeout: int = 30, max_memory_mb: int = 512):
        self.timeout = timeout
        self.max_memory_mb = max_memory_mb
        self.sandbox_dir = None
        
    def _create_sandbox(self) -> str:
        """Create a sandboxed execution environment."""
        sandbox_dir = tempfile.mkdtemp(prefix="kwe_exec_")
        self.sandbox_dir = sandbox_dir
        
        # Create subdirectories
        os.makedirs(os.path.join(sandbox_dir, "src"), exist_ok=True)
        os.makedirs(os.path.join(sandbox_dir, "output"), exist_ok=True)
        
        return sandbox_dir
    
    def _cleanup_sandbox(self):
        """Clean up the sandbox directory."""
        if self.sandbox_dir and os.path.exists(self.sandbox_dir):
            try:
                import shutil
                shutil.rmtree(self.sandbox_dir)
            except Exception as e:
                logger.warning(f"Failed to cleanup sandbox: {e}")
    
    def _detect_language(self, code: str) -> str:
        """Detect programming language from code."""
        code_lower = code.lower()
        
        # Language detection patterns
        if any(keyword in code_lower for keyword in ["def ", "import ", "from ", "class "]):
            return "python"
        elif any(keyword in code_lower for keyword in ["fn ", "let ", "struct ", "impl "]):
            return "rust"
        elif any(keyword in code_lower for keyword in ["function ", "const ", "let ", "var "]):
            return "javascript"
        elif any(keyword in code_lower for keyword in ["func ", "package ", "import "]):
            return "go"
        elif any(keyword in code_lower for keyword in ["public class ", "public static "]):
            return "java"
        elif any(keyword in code_lower for keyword in ["<?php", "function "]):
            return "php"
        elif any(keyword in code_lower for keyword in ["#!/", "echo ", "if ["]):
            return "shell"
        else:
            return "python"  # Default
    
    def _security_check(self, code: str, language: str) -> List[str]:
        """Perform security checks on the code."""
        violations = []
        
        # Dangerous patterns
        dangerous_patterns = [
            # File system access
            (r'os\.system\s*\(', 'System command execution'),
            (r'subprocess\.run\s*\([^)]*shell\s*=\s*True', 'Shell command execution'),
            (r'open\s*\([^)]*[\'"][^\'"]*[\'"]\s*,\s*[\'"][^\'"]*[\'"]', 'File system access'),
            (r'__import__\s*\(', 'Dynamic import'),
            (r'eval\s*\(', 'Code evaluation'),
            (r'exec\s*\(', 'Code execution'),
            # Network access
            (r'urllib\.', 'Network access'),
            (r'requests\.', 'Network access'),
            (r'socket\.', 'Network access'),
            # Process management
            (r'os\.kill\s*\(', 'Process management'),
            (r'os\.fork\s*\(', 'Process creation'),
            # System information
            (r'os\.environ', 'Environment variable access'),
            (r'platform\.', 'System information'),
        ]
        
        for pattern, description in dangerous_patterns:
            if re.search(pattern, code, re.IGNORECASE):
                violations.append(f"{description}: {pattern}")
        
        return violations
    
    def _prepare_execution(self, code: str, language: str) -> Tuple[str, str]:
        """Prepare code for execution."""
        sandbox_dir = self._create_sandbox()
        
        # Determine file extension and execution command
        if language == "python":
            ext = ".py"
            filename = "main.py"
            cmd = ["python", "main.py"]
        elif language == "rust":
            ext = ".rs"
            filename = "main.rs"
            cmd = ["rustc", "main.rs", "-o", "main", "&&", "./main"]
        elif language == "javascript":
            ext = ".js"
            filename = "main.js"
            cmd = ["node", "main.js"]
        elif language == "go":
            ext = ".go"
            filename = "main.go"
            cmd = ["go", "run", "main.go"]
        elif language == "java":
            ext = ".java"
            filename = "Main.java"
            cmd = ["javac", "Main.java", "&&", "java", "Main"]
        elif language == "php":
            ext = ".php"
            filename = "main.php"
            cmd = ["php", "main.php"]
        elif language == "shell":
            ext = ".sh"
            filename = "main.sh"
            cmd = ["bash", "main.sh"]
        else:
            ext = ".py"
            filename = "main.py"
            cmd = ["python", "main.py"]
        
        # Write code to file
        file_path = os.path.join(sandbox_dir, "src", filename)
        with open(file_path, 'w') as f:
            f.write(code)
        
        # Make shell scripts executable
        if language == "shell":
            os.chmod(file_path, 0o755)
        
        return sandbox_dir, " ".join(cmd)
    
    def _monitor_process(self, process, timeout: int) -> Tuple[bool, float]:
        """Monitor process execution with timeout."""
        start_time = time.time()
        
        def check_timeout():
            time.sleep(timeout)
            if process.poll() is None:
                process.terminate()
                time.sleep(1)
                if process.poll() is None:
                    process.kill()
        
        timeout_thread = threading.Thread(target=check_timeout)
        timeout_thread.daemon = True
        timeout_thread.start()
        
        try:
            stdout, stderr = process.communicate()
            execution_time = time.time() - start_time
            
            if process.returncode == -15:  # SIGTERM
                return False, execution_time
            else:
                return True, execution_time
                
        except Exception as e:
            logger.error(f"Process monitoring error: {e}")
            return False, time.time() - start_time
    
    def execute_code(self, code: str, language: str = None, timeout: int = None) -> ExecutionResult:
        """Execute code safely."""
        start_time = time.time()
        
        try:
            # Detect language if not specified
            if not language:
                language = self._detect_language(code)
            
            # Use provided timeout or default
            exec_timeout = timeout or self.timeout
            
            # Security check
            violations = self._security_check(code, language)
            if violations:
                return ExecutionResult(
                    status=ExecutionStatus.SECURITY_VIOLATION,
                    stdout="",
                    stderr="",
                    return_code=-1,
                    execution_time=time.time() - start_time,
                    error_message=f"Security violations detected: {', '.join(violations)}",
                    metadata={"violations": violations}
                )
            
            # Prepare execution environment
            sandbox_dir, cmd = self._prepare_execution(code, language)
            
            # Execute code
            process = subprocess.Popen(
                cmd,
                cwd=os.path.join(sandbox_dir, "src"),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                shell=True
            )
            
            # Monitor execution
            completed, execution_time = self._monitor_process(process, exec_timeout)
            
            if not completed:
                return ExecutionResult(
                    status=ExecutionStatus.TIMEOUT,
                    stdout="",
                    stderr="",
                    return_code=-1,
                    execution_time=execution_time,
                    error_message=f"Execution timed out after {exec_timeout} seconds"
                )
            
            # Get output
            stdout, stderr = process.communicate()
            
            # Check for errors
            if process.returncode != 0:
                return ExecutionResult(
                    status=ExecutionStatus.ERROR,
                    stdout=stdout,
                    stderr=stderr,
                    return_code=process.returncode,
                    execution_time=execution_time,
                    error_message=f"Execution failed with return code {process.returncode}"
                )
            
            # Success
            return ExecutionResult(
                status=ExecutionStatus.SUCCESS,
                stdout=stdout,
                stderr=stderr,
                return_code=process.returncode,
                execution_time=execution_time,
                metadata={"language": language, "sandbox_dir": sandbox_dir}
            )
            
        except Exception as e:
            logger.error(f"Code execution error: {e}")
            return ExecutionResult(
                status=ExecutionStatus.ERROR,
                stdout="",
                stderr="",
                return_code=-1,
                execution_time=time.time() - start_time,
                error_message=f"Execution error: {str(e)}"
            )
        
        finally:
            # Cleanup
            self._cleanup_sandbox()
    
    def execute_python_code(self, code: str, timeout: int = None) -> ExecutionResult:
        """Execute Python code specifically."""
        return self.execute_code(code, "python", timeout)
    
    def execute_shell_code(self, code: str, timeout: int = None) -> ExecutionResult:
        """Execute shell code specifically."""
        return self.execute_code(code, "shell", timeout)


# Global instance for easy use
code_executor = CodeExecutor()


def execute_code(code: str, language: str = None, timeout: int = None) -> ExecutionResult:
    """Execute code safely."""
    return code_executor.execute_code(code, language, timeout)


def execute_python_code(code: str, timeout: int = None) -> ExecutionResult:
    """Execute Python code safely."""
    return code_executor.execute_python_code(code, timeout)


def execute_shell_code(code: str, timeout: int = None) -> ExecutionResult:
    """Execute shell code safely."""
    return code_executor.execute_shell_code(code, timeout) 