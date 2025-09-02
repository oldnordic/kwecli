"""
Secure Bash execution tool for KWE CLI.

Provides secure command execution with comprehensive safety measures including:
- Command allowlisting and security validation
- Timeout enforcement and resource monitoring
- Background execution with process management
- Real-time output streaming
- Security audit logging
"""

import os
import asyncio
import subprocess
import signal
import threading
import time
import psutil
import logging
import uuid
from pathlib import Path
from typing import Dict, Any, List, Optional, Callable, Union
from dataclasses import dataclass, asdict
from datetime import datetime

import sys
sys.path.append('/home/feanor/Projects/kwecli')

from ..core.tool_interface import BaseTool, ToolValidationMixin
from .security import CommandSecurityValidator, SecurityAuditLogger, SecurityViolationError


@dataclass
class BashExecutionResult:
    """Result of bash command execution."""
    success: bool
    command: str
    stdout: str
    stderr: str
    exit_code: int
    execution_time: float
    process_id: Optional[int] = None
    working_directory: Optional[str] = None
    resource_usage: Optional[Dict[str, Any]] = None
    status: str = "completed"  # running, completed, terminated, failed
    error: Optional[str] = None


@dataclass
class BackgroundProcess:
    """Background process tracking."""
    process_id: str
    command: str
    process: subprocess.Popen
    start_time: float
    working_directory: str
    monitor_thread: Optional[threading.Thread] = None
    resource_monitor: Optional['ResourceMonitor'] = None
    stdout_lines: List[str] = None
    stderr_lines: List[str] = None
    
    def __post_init__(self):
        if self.stdout_lines is None:
            self.stdout_lines = []
        if self.stderr_lines is None:
            self.stderr_lines = []


class ResourceMonitor:
    """Monitor resource usage of executing processes."""
    
    def __init__(self, pid: int):
        self.pid = pid
        self.start_time = time.time()
        self.memory_peak = 0
        self.cpu_time_start = 0
        self.disk_io_start = (0, 0)  # read, write
        self.network_io_start = (0, 0)  # sent, recv
        self.running = True
        self.samples_taken = 0
        self.total_cpu_time = 0
        
        try:
            self.process = psutil.Process(pid)
            
            # Take initial measurements
            self.cpu_time_start = sum(self.process.cpu_times())
            initial_memory = self.process.memory_info()
            self.memory_peak = initial_memory.rss
            
            # Get initial I/O stats if available
            try:
                io_counters = self.process.io_counters()
                self.disk_io_start = (io_counters.read_bytes, io_counters.write_bytes)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
            
            self.samples_taken = 1
            
        except psutil.NoSuchProcess:
            self.process = None
    
    def get_current_stats(self) -> Dict[str, Any]:
        """Get current resource usage statistics."""
        # Always return basic stats based on what we've collected
        base_stats = {
            "memory_peak": self.memory_peak,
            "cpu_time": self.total_cpu_time,
            "disk_io_read": 0,
            "disk_io_write": 0,
            "execution_time": time.time() - self.start_time,
            "samples_taken": self.samples_taken
        }
        
        if not self.process:
            return base_stats
        
        try:
            # Try to get current stats if process is still accessible
            if self.process.is_running():
                # Memory usage
                memory_info = self.process.memory_info()
                self.memory_peak = max(self.memory_peak, memory_info.rss)
                
                # CPU time
                current_cpu_time = sum(self.process.cpu_times())
                self.total_cpu_time = current_cpu_time - self.cpu_time_start
                
                # Disk I/O
                disk_read, disk_write = 0, 0
                try:
                    io_counters = self.process.io_counters()
                    disk_read = io_counters.read_bytes - self.disk_io_start[0]
                    disk_write = io_counters.write_bytes - self.disk_io_start[1]
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
                
                self.samples_taken += 1
                
                return {
                    "memory_current": memory_info.rss,
                    "memory_peak": self.memory_peak,
                    "cpu_time": self.total_cpu_time,
                    "disk_io_read": disk_read,
                    "disk_io_write": disk_write,
                    "execution_time": time.time() - self.start_time,
                    "samples_taken": self.samples_taken
                }
            else:
                # Process finished, return final stats
                return base_stats
                
        except psutil.NoSuchProcess:
            self.running = False
            return base_stats
    
    def stop(self):
        """Stop monitoring."""
        self.running = False


class BashTool(BaseTool, ToolValidationMixin):
    """
    Secure bash command execution tool.
    
    Provides comprehensive command execution with security validation,
    resource monitoring, and background process management.
    """
    
    def __init__(
        self,
        security_validator: Optional[CommandSecurityValidator] = None,
        audit_logger: Optional[SecurityAuditLogger] = None,
        max_concurrent_processes: int = 25,
        default_timeout: int = 120
    ):
        super().__init__()
        self.security_validator = security_validator or CommandSecurityValidator()
        self.audit_logger = audit_logger or SecurityAuditLogger()
        self.max_concurrent_processes = max_concurrent_processes
        self.default_timeout = default_timeout
        
        # Background process management
        self._background_processes: Dict[str, BackgroundProcess] = {}
        self._process_lock = threading.RLock()
        self._active_processes = 0
        
        self.logger = logging.getLogger(f"{self.__class__.__module__}.{self.__class__.__name__}")
    
    @property
    def name(self) -> str:
        return "bash"
    
    @property
    def category(self) -> str:
        return "system"
    
    @property
    def capabilities(self) -> List[str]:
        return [
            "command_execution", "bash", "shell", "system", 
            "background_execution", "resource_monitoring", 
            "output_streaming", "process_management"
        ]
    
    @property
    def description(self) -> str:
        return "Secure bash command execution with comprehensive safety measures"
    
    async def execute(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute bash command with security validation and monitoring."""
        # Validate parameters first (outside try-catch to let ValueError propagate)
        if not self.validate_parameters(parameters):
            # Check specifically for missing required parameters
            required = ["command"]
            if not self.validate_required_params(parameters, required):
                raise ValueError(f"Missing required parameters: {[p for p in required if p not in parameters]}")
            
            # Check for specific validation errors
            timeout = parameters.get("timeout", self.default_timeout)
            if timeout <= 0 or timeout > 3600:
                return {
                    "success": False,
                    "error": f"Invalid timeout value: {timeout}. Must be between 1 and 3600 seconds."
                }
                
            return {
                "success": False,
                "error": "Invalid parameters provided"
            }
        
        try:
            
            command = parameters["command"]
            working_directory = parameters.get("working_directory", os.getcwd())
            timeout = parameters.get("timeout", self.default_timeout)
            background = parameters.get("background", False)
            environment = parameters.get("environment", {})
            stream_output = parameters.get("stream_output", False)
            output_callback = parameters.get("output_callback")
            monitor_resources = parameters.get("monitor_resources", False)
            
            # Security validation
            validation_result = self.security_validator.validate_command(command, environment)
            if not validation_result["valid"]:
                error_msg = f"Security validation failed: {', '.join(validation_result['errors'])}"
                
                # Log security violation
                self.audit_logger.log_security_violation(
                    command=command,
                    violation_type="command_validation_failed",
                    user=os.getenv("USER", "unknown"),
                    details=validation_result
                )
                
                return {
                    "success": False,
                    "error": error_msg,
                    "command": command,
                    "security_errors": validation_result["errors"]
                }
            
            # Validate working directory
            dir_validation = self.security_validator.validate_working_directory(working_directory)
            if not dir_validation["valid"]:
                return {
                    "success": False,
                    "error": f"Working directory validation failed: {', '.join(dir_validation['errors'])}",
                    "command": command
                }
            
            # Validate environment variables
            if environment:
                env_validation = self.security_validator.validate_environment(environment)
                if not env_validation["valid"]:
                    return {
                        "success": False,
                        "error": f"Environment validation failed: {', '.join(env_validation['errors'])}",
                        "command": command
                    }
            
            # Validate timeout
            timeout_validation = self.security_validator.validate_timeout(timeout)
            if not timeout_validation["valid"]:
                return {
                    "success": False,
                    "error": f"Timeout validation failed: {', '.join(timeout_validation['errors'])}",
                    "command": command
                }
            
            # Check process limits
            if self._active_processes >= self.max_concurrent_processes:
                return {
                    "success": False,
                    "error": f"Maximum concurrent processes ({self.max_concurrent_processes}) reached",
                    "command": command
                }
            
            # Execute command
            if background:
                return await self._execute_background(
                    command, working_directory, environment, monitor_resources
                )
            else:
                return await self._execute_foreground(
                    command, working_directory, timeout, environment,
                    stream_output, output_callback, monitor_resources
                )
            
        except Exception as e:
            self.logger.error(f"Error executing command: {e}")
            return {
                "success": False,
                "error": str(e),
                "command": parameters.get("command", "unknown")
            }
    
    async def _execute_foreground(
        self,
        command: str,
        working_directory: str,
        timeout: int,
        environment: Dict[str, str],
        stream_output: bool,
        output_callback: Optional[Callable],
        monitor_resources: bool
    ) -> Dict[str, Any]:
        """Execute command in foreground with monitoring."""
        start_time = time.time()
        
        try:
            # Prepare environment
            exec_env = os.environ.copy()
            exec_env.update(environment)
            
            # Start process
            process = subprocess.Popen(
                command,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=working_directory,
                env=exec_env,
                text=True,
                preexec_fn=os.setsid if os.name != 'nt' else None
            )
            
            self._active_processes += 1
            
            # Setup resource monitoring
            resource_monitor = None
            if monitor_resources:
                resource_monitor = ResourceMonitor(process.pid)
            
            try:
                # Handle output streaming
                if stream_output and output_callback:
                    stdout_lines, stderr_lines = await self._stream_output(
                        process, timeout, output_callback
                    )
                else:
                    # Wait for completion with timeout
                    try:
                        stdout, stderr = await asyncio.wait_for(
                            asyncio.create_task(self._communicate_with_process(process)),
                            timeout=timeout
                        )
                        stdout_lines = [stdout] if stdout else []
                        stderr_lines = [stderr] if stderr else []
                    except asyncio.TimeoutError:
                        # Kill process group on timeout
                        try:
                            if os.name != 'nt':
                                os.killpg(os.getpgid(process.pid), signal.SIGTERM)
                            else:
                                process.terminate()
                            
                            # Wait a bit for graceful termination
                            await asyncio.sleep(1)
                            
                            if process.poll() is None:
                                # Force kill if still running
                                if os.name != 'nt':
                                    os.killpg(os.getpgid(process.pid), signal.SIGKILL)
                                else:
                                    process.kill()
                                    
                        except (ProcessLookupError, OSError):
                            pass
                        
                        execution_time = time.time() - start_time
                        
                        self.audit_logger.log_command_execution(
                            command=command,
                            user=os.getenv("USER", "unknown"),
                            working_directory=working_directory,
                            success=False,
                            exit_code=-1,
                            execution_time=execution_time
                        )
                        
                        return {
                            "success": False,
                            "error": f"Command timed out after {timeout} seconds",
                            "command": command,
                            "stdout": "",
                            "stderr": "",
                            "exit_code": -1,
                            "execution_time": execution_time
                        }
                
                exit_code = process.returncode
                execution_time = time.time() - start_time
                
                # Get resource usage
                resource_usage = {}
                if resource_monitor:
                    resource_usage = resource_monitor.get_current_stats()
                    resource_monitor.stop()
                
                # Combine output
                stdout = "\n".join(stdout_lines) if stdout_lines else ""
                stderr = "\n".join(stderr_lines) if stderr_lines else ""
                
                success = exit_code == 0
                
                # Log execution
                self.audit_logger.log_command_execution(
                    command=command,
                    user=os.getenv("USER", "unknown"),
                    working_directory=working_directory,
                    success=success,
                    exit_code=exit_code,
                    execution_time=execution_time,
                    pid=process.pid,
                    resource_usage=resource_usage
                )
                
                result = {
                    "success": success,
                    "command": command,
                    "stdout": stdout,
                    "stderr": stderr,
                    "exit_code": exit_code,
                    "execution_time": execution_time,
                    "working_directory": working_directory
                }
                
                if monitor_resources:
                    result["resource_usage"] = resource_usage
                
                if not success and stderr:
                    result["error"] = stderr
                
                return result
                
            finally:
                self._active_processes -= 1
                if resource_monitor:
                    resource_monitor.stop()
            
        except Exception as e:
            execution_time = time.time() - start_time
            self._active_processes -= 1
            
            self.logger.error(f"Error in foreground execution: {e}")
            
            return {
                "success": False,
                "error": str(e),
                "command": command,
                "stdout": "",
                "stderr": "",
                "exit_code": -1,
                "execution_time": execution_time
            }
    
    async def _communicate_with_process(self, process: subprocess.Popen) -> tuple:
        """Communicate with process asynchronously."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, process.communicate)
    
    async def _stream_output(
        self,
        process: subprocess.Popen,
        timeout: int,
        output_callback: Callable
    ) -> tuple:
        """Stream output from process with real-time callback."""
        stdout_lines = []
        stderr_lines = []
        
        async def read_stream(stream, lines_list, callback):
            """Read from stream and call callback for each line."""
            try:
                while True:
                    line = await asyncio.get_event_loop().run_in_executor(
                        None, stream.readline
                    )
                    if not line:
                        break
                    
                    line = line.strip()
                    lines_list.append(line)
                    if callback:
                        callback(line)
                        
            except Exception as e:
                self.logger.error(f"Error reading stream: {e}")
        
        # Start reading both streams
        tasks = [
            asyncio.create_task(read_stream(process.stdout, stdout_lines, output_callback)),
            asyncio.create_task(read_stream(process.stderr, stderr_lines, None))
        ]
        
        try:
            # Wait for process completion or timeout
            done, pending = await asyncio.wait_for(
                asyncio.wait(tasks + [asyncio.create_task(self._wait_for_process(process))]),
                timeout=timeout
            )
            
            # Cancel pending tasks
            for task in pending:
                task.cancel()
                
        except asyncio.TimeoutError:
            # Cancel all tasks and kill process
            for task in tasks:
                task.cancel()
            
            try:
                if os.name != 'nt':
                    os.killpg(os.getpgid(process.pid), signal.SIGTERM)
                else:
                    process.terminate()
            except (ProcessLookupError, OSError):
                pass
        
        return stdout_lines, stderr_lines
    
    async def _wait_for_process(self, process: subprocess.Popen):
        """Wait for process completion asynchronously."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, process.wait)
    
    async def _execute_background(
        self,
        command: str,
        working_directory: str,
        environment: Dict[str, str],
        monitor_resources: bool
    ) -> Dict[str, Any]:
        """Execute command in background."""
        try:
            # Generate unique process ID
            process_id = str(uuid.uuid4())
            
            # Prepare environment
            exec_env = os.environ.copy()
            exec_env.update(environment)
            
            # Start process
            process = subprocess.Popen(
                command,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=working_directory,
                env=exec_env,
                text=True,
                preexec_fn=os.setsid if os.name != 'nt' else None
            )
            
            # Setup resource monitoring
            resource_monitor = None
            if monitor_resources:
                resource_monitor = ResourceMonitor(process.pid)
            
            # Create background process tracking
            bg_process = BackgroundProcess(
                process_id=process_id,
                command=command,
                process=process,
                start_time=time.time(),
                working_directory=working_directory,
                resource_monitor=resource_monitor
            )
            
            # Start monitoring thread
            monitor_thread = threading.Thread(
                target=self._monitor_background_process,
                args=(bg_process,),
                daemon=True
            )
            bg_process.monitor_thread = monitor_thread
            monitor_thread.start()
            
            # Store process
            with self._process_lock:
                self._background_processes[process_id] = bg_process
            
            self._active_processes += 1
            
            return {
                "success": True,
                "process_id": process_id,
                "command": command,
                "status": "running",
                "pid": process.pid,
                "working_directory": working_directory
            }
            
        except Exception as e:
            self.logger.error(f"Error starting background process: {e}")
            return {
                "success": False,
                "error": str(e),
                "command": command
            }
    
    def _monitor_background_process(self, bg_process: BackgroundProcess):
        """Monitor background process in separate thread."""
        try:
            # Read output continuously
            while bg_process.process.poll() is None:
                try:
                    # Read stdout
                    if bg_process.process.stdout:
                        line = bg_process.process.stdout.readline()
                        if line:
                            bg_process.stdout_lines.append(line.strip())
                    
                    # Read stderr
                    if bg_process.process.stderr:
                        line = bg_process.process.stderr.readline()
                        if line:
                            bg_process.stderr_lines.append(line.strip())
                    
                    time.sleep(0.1)  # Small delay to avoid busy loop
                    
                except Exception as e:
                    self.logger.error(f"Error reading background process output: {e}")
                    break
            
            # Process completed, read remaining output
            if bg_process.process.stdout:
                remaining_stdout = bg_process.process.stdout.read()
                if remaining_stdout:
                    bg_process.stdout_lines.extend(remaining_stdout.strip().split('\n'))
            
            if bg_process.process.stderr:
                remaining_stderr = bg_process.process.stderr.read()
                if remaining_stderr:
                    bg_process.stderr_lines.extend(remaining_stderr.strip().split('\n'))
            
            # Stop resource monitoring
            if bg_process.resource_monitor:
                bg_process.resource_monitor.stop()
            
            # Log completion
            execution_time = time.time() - bg_process.start_time
            returncode = bg_process.process.returncode or 0
            
            # For audit logging, consider terminated processes as "not successful" but log the actual exit code
            success = returncode == 0
            
            self.audit_logger.log_command_execution(
                command=bg_process.command,
                user=os.getenv("USER", "unknown"),
                working_directory=bg_process.working_directory,
                success=success,
                exit_code=returncode,
                execution_time=execution_time,
                pid=bg_process.process.pid
            )
            
        except Exception as e:
            self.logger.error(f"Error monitoring background process: {e}")
        finally:
            self._active_processes -= 1
    
    async def get_background_status(self, process_id: str) -> Dict[str, Any]:
        """Get status of background process."""
        with self._process_lock:
            if process_id not in self._background_processes:
                return {
                    "success": False,
                    "error": f"Process {process_id} not found"
                }
            
            bg_process = self._background_processes[process_id]
            
            if bg_process.process.poll() is None:
                status = "running"
            else:
                # Properly classify process exit status
                returncode = bg_process.process.returncode
                if returncode == 0:
                    status = "completed"
                elif returncode < 0:
                    status = "terminated"  # Killed by signal (SIGTERM=-15, SIGKILL=-9, etc.)
                else:
                    status = "failed"  # Actual error condition
            
            result = {
                "success": True,
                "process_id": process_id,
                "status": status,
                "command": bg_process.command,
                "runtime": time.time() - bg_process.start_time
            }
            
            if bg_process.resource_monitor:
                result["resource_usage"] = bg_process.resource_monitor.get_current_stats()
            
            return result
    
    async def get_background_result(self, process_id: str) -> Dict[str, Any]:
        """Get final result of background process."""
        with self._process_lock:
            if process_id not in self._background_processes:
                return {
                    "success": False,
                    "error": f"Process {process_id} not found"
                }
            
            bg_process = self._background_processes[process_id]
            
            # Wait for process completion if still running
            if bg_process.process.poll() is None:
                return {
                    "success": False,
                    "error": f"Process {process_id} is still running"
                }
            
            # Get final results
            execution_time = time.time() - bg_process.start_time
            returncode = bg_process.process.returncode
            
            # Determine success and status consistently
            if returncode == 0:
                success = True
                status = "completed"
            elif returncode < 0:
                success = False  # Terminated processes are not "successful" but not "failed" either
                status = "terminated"
            else:
                success = False
                status = "failed"
            
            result = {
                "success": success,
                "process_id": process_id,
                "command": bg_process.command,
                "stdout": "\n".join(bg_process.stdout_lines),
                "stderr": "\n".join(bg_process.stderr_lines),
                "exit_code": returncode,
                "execution_time": execution_time,
                "working_directory": bg_process.working_directory,
                "status": status
            }
            
            if bg_process.resource_monitor:
                result["resource_usage"] = bg_process.resource_monitor.get_current_stats()
            
            # Clean up
            del self._background_processes[process_id]
            
            return result
    
    async def kill_background_process(self, process_id: str) -> Dict[str, Any]:
        """Kill background process."""
        with self._process_lock:
            if process_id not in self._background_processes:
                return {
                    "success": False,
                    "error": f"Process {process_id} not found"
                }
            
            bg_process = self._background_processes[process_id]
            
            try:
                if bg_process.process.poll() is None:
                    # Terminate gracefully first
                    if os.name != 'nt':
                        os.killpg(os.getpgid(bg_process.process.pid), signal.SIGTERM)
                    else:
                        bg_process.process.terminate()
                    
                    # Wait a bit for graceful termination
                    time.sleep(1)
                    
                    # Force kill if still running
                    if bg_process.process.poll() is None:
                        if os.name != 'nt':
                            os.killpg(os.getpgid(bg_process.process.pid), signal.SIGKILL)
                        else:
                            bg_process.process.kill()
                
                # Stop resource monitoring
                if bg_process.resource_monitor:
                    bg_process.resource_monitor.stop()
                
                return {
                    "success": True,
                    "process_id": process_id,
                    "message": "Process terminated"
                }
                
            except (ProcessLookupError, OSError) as e:
                return {
                    "success": True,
                    "process_id": process_id,
                    "message": f"Process already terminated: {e}"
                }
            except Exception as e:
                return {
                    "success": False,
                    "error": f"Failed to kill process: {e}"
                }
    
    def validate_parameters(self, parameters: Dict[str, Any]) -> bool:
        """Validate parameters for bash execution."""
        required = ["command"]
        if not self.validate_required_params(parameters, required):
            return False
        
        type_specs = {
            "command": str,
            "working_directory": str,
            "timeout": int,
            "background": bool,
            "environment": dict,
            "stream_output": bool,
            "monitor_resources": bool
        }
        if not self.validate_param_types(parameters, type_specs):
            return False
        
        # Validate timeout range
        timeout = parameters.get("timeout", self.default_timeout)
        if timeout <= 0 or timeout > 3600:  # Max 1 hour
            return False
        
        return True
    
    def get_schema(self) -> Dict[str, Any]:
        """Get parameter schema."""
        return {
            "name": self.name,
            "category": self.category,
            "parameters": {
                "command": {
                    "type": "string",
                    "description": "Command to execute",
                    "required": True
                },
                "working_directory": {
                    "type": "string",
                    "description": "Working directory for command execution",
                    "required": False
                },
                "timeout": {
                    "type": "integer",
                    "description": "Command timeout in seconds (max 3600)",
                    "required": False,
                    "default": self.default_timeout,
                    "minimum": 1,
                    "maximum": 3600
                },
                "background": {
                    "type": "boolean",
                    "description": "Execute command in background",
                    "required": False,
                    "default": False
                },
                "environment": {
                    "type": "object",
                    "description": "Environment variables for command",
                    "required": False
                },
                "stream_output": {
                    "type": "boolean",
                    "description": "Stream output in real-time",
                    "required": False,
                    "default": False
                },
                "monitor_resources": {
                    "type": "boolean",
                    "description": "Monitor resource usage during execution",
                    "required": False,
                    "default": False
                }
            }
        }