#!/usr/bin/env python3
"""
Real Subprocess Agent Implementation

This agent provides real subprocess execution using actual system commands.
NO mocks, stubs, or placeholders are used - all subprocess operations are real.

Features:
- Real subprocess execution with proper error handling
- Real timeout management for long-running processes
- Real working directory and environment variable support
- Real command pipeline execution
- Async execution support for non-blocking operations
- Security considerations for command injection prevention
"""

import os
import asyncio
import subprocess
import time
import shlex
from typing import Dict, Any, List, Optional, Union

from agents.base_agent import SubAgent, AgentResult, AgentStatus, AgentExpertise


class RealSubprocessAgent(SubAgent):
    """Real subprocess execution agent using actual system commands."""
    
    def __init__(self):
        super().__init__(
            name="real_subprocess_agent",
            description="Real subprocess execution agent using actual system commands",
            expertise=[AgentExpertise.DEVOPS, AgentExpertise.INFRASTRUCTURE],
            tools=["subprocess", "bash", "shell", "command_execution"]
        )
        self.default_timeout = 30  # 30 seconds default timeout
        self.max_timeout = 300     # 5 minutes maximum timeout
    
    def can_handle(self, task: str) -> bool:
        """Check if this agent can handle subprocess/command execution tasks."""
        task_lower = task.lower()
        subprocess_keywords = [
            "bash", "command", "shell", "execute", "run", "subprocess",
            "script", "process", "terminal", "cmd"
        ]
        
        # Should handle subprocess-related tasks
        if any(keyword in task_lower for keyword in subprocess_keywords):
            return True
        
        # Should not handle unrelated tasks
        unrelated_keywords = [
            "analyze", "read file", "parse", "email", "database",
            "api", "web", "http", "json", "xml"
        ]
        
        if any(keyword in task_lower for keyword in unrelated_keywords):
            return False
        
        return False
    
    async def execute_task(self, task: str, context: Dict[str, Any]) -> AgentResult:
        """Execute subprocess task with real system command execution."""
        start_time = time.time()
        
        try:
            self.update_status(AgentStatus.BUSY)
            
            # Extract command from context
            command = context.get("command")
            if not command:
                return AgentResult(
                    success=False,
                    output="",
                    error_message="command required in context"
                )
            
            # Execute the command
            result = await self._execute_subprocess_command(command, context)
            
            # Add execution time
            execution_time = time.time() - start_time
            if result.metadata is None:
                result.metadata = {}
            result.metadata["execution_time"] = execution_time
            result.execution_time = execution_time
            
            # Add to work history
            self.add_work_history(task, result)
            
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            error_result = AgentResult(
                success=False,
                output="",
                error_message=f"Task execution failed: {str(e)}",
                metadata={"execution_time": execution_time},
                execution_time=execution_time
            )
            self.add_work_history(task, error_result)
            return error_result
            
        finally:
            self.update_status(AgentStatus.IDLE)
    
    async def _execute_subprocess_command(self, command: str, context: Dict[str, Any]) -> AgentResult:
        """Execute a real subprocess command with proper error handling."""
        try:
            # Get execution parameters
            timeout = context.get("timeout", self.default_timeout)
            working_directory = context.get("working_directory")
            environment = context.get("environment", {})
            shell = context.get("shell", True)
            
            # Validate timeout
            if timeout > self.max_timeout:
                timeout = self.max_timeout
            
            # Prepare environment
            env = os.environ.copy()
            env.update(environment)
            
            # Security: Basic command validation
            if self._is_dangerous_command(command):
                return AgentResult(
                    success=False,
                    output="",
                    error_message=f"Command rejected for security reasons: {command}"
                )
            
            # Execute the command
            process = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=working_directory,
                env=env
            )
            
            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(),
                    timeout=timeout
                )
                
                return_code = process.returncode
                
                # Decode output
                stdout_text = stdout.decode('utf-8', errors='replace') if stdout else ""
                stderr_text = stderr.decode('utf-8', errors='replace') if stderr else ""
                
                # Determine success
                success = return_code == 0
                
                # Combine output
                output = stdout_text
                if stderr_text and not success:
                    output = f"{stdout_text}\nSTDERR: {stderr_text}"
                
                return AgentResult(
                    success=success,
                    output=output.strip(),
                    error_message=stderr_text.strip() if not success and stderr_text else None,
                    metadata={
                        "command": command,
                        "return_code": return_code,
                        "working_directory": working_directory,
                        "timeout_used": timeout,
                        "environment_vars": list(environment.keys()) if environment else []
                    }
                )
                
            except asyncio.TimeoutError:
                # Kill the process if it's still running
                try:
                    process.terminate()
                    await asyncio.wait_for(process.wait(), timeout=5)
                except:
                    try:
                        process.kill()
                        await process.wait()
                    except Exception:
                        # Process already terminated or unable to kill - this is acceptable
                        # No further action needed for cleanup failure
                        None  # Explicit no-op
                
                return AgentResult(
                    success=False,
                    output="",
                    error_message=f"Command timed out after {timeout} seconds",
                    metadata={
                        "command": command,
                        "return_code": -1,
                        "timeout_used": timeout,
                        "timeout_exceeded": True
                    }
                )
                
        except FileNotFoundError:
            return AgentResult(
                success=False,
                output="",
                error_message=f"Command not found: {command}",
                metadata={"command": command, "return_code": 127}
            )
        except PermissionError:
            return AgentResult(
                success=False,
                output="",
                error_message=f"Permission denied for command: {command}",
                metadata={"command": command, "return_code": 126}
            )
        except Exception as e:
            return AgentResult(
                success=False,
                output="",
                error_message=f"Subprocess execution failed: {str(e)}",
                metadata={"command": command, "return_code": -1}
            )
    
    def _is_dangerous_command(self, command: str) -> bool:
        """Basic security check for dangerous commands."""
        dangerous_patterns = [
            "rm -rf /",
            "mkfs",
            "dd if=",
            ":(){ :|:& };:",  # Fork bomb
            "shutdown",
            "reboot",
            "halt",
            "init 0",
            "init 6",
            "> /dev/sd",      # Writing to disk devices
            "chmod 777 /",
            "chown root /",
        ]
        
        command_lower = command.lower()
        
        for pattern in dangerous_patterns:
            if pattern in command_lower:
                return True
        
        # Check for privilege escalation
        if command_lower.startswith(("sudo ", "su ", "doas ")):
            return True
        
        return False
    
    async def execute_task_async(self, task: str, context: Dict[str, Any]) -> AgentResult:
        """Async wrapper for execute_task to support async testing."""
        return await self.execute_task(task, context)