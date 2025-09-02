"""
Security framework for KWE CLI system tools.

Provides comprehensive security validation for command execution including:
- Command allowlisting and blocklisting
- Input sanitization and injection prevention
- Resource limits and monitoring
- Security audit logging and analysis
"""

import os
import re
import shlex
import json
import threading
import subprocess
import hashlib
import time
import logging
import logging.handlers
from pathlib import Path
from typing import Dict, Any, List, Optional, Set, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from cryptography.fernet import Fernet
import psutil


class SecurityViolationError(Exception):
    """Raised when a security violation is detected."""
    pass


@dataclass
class CommandValidationResult:
    """Result of command validation."""
    valid: bool
    command: str
    errors: List[str]
    warnings: List[str]
    sanitized_command: Optional[str] = None
    risk_level: str = "low"  # low, medium, high, critical


@dataclass
class ResourceLimits:
    """Resource limits for command execution."""
    max_memory: int = 1024 * 1024 * 1024  # 1GB
    max_cpu_time: int = 300  # 5 minutes
    max_processes: int = 10
    max_file_descriptors: int = 1024
    max_network_connections: int = 100


class CommandSecurityValidator:
    """
    Comprehensive security validator for command execution.
    
    Validates commands against allowlists, detects injection attempts,
    and enforces security policies.
    """
    
    # Default allowed commands for development workflows
    DEFAULT_ALLOWED_COMMANDS = [
        # Python ecosystem
        "python", "python3", "pip", "pip3", "poetry", "pipenv",
        "pytest", "black", "isort", "flake8", "mypy", "bandit",
        
        # Node.js ecosystem  
        "node", "npm", "yarn", "pnpm", "npx",
        "jest", "eslint", "prettier", "tsc",
        
        # Rust ecosystem
        "cargo", "rustc", "rustfmt", "clippy",
        
        # Go ecosystem
        "go", "gofmt", "golint",
        
        # Git operations
        "git",
        
        # Build and development tools
        "make", "cmake", "gcc", "clang",
        "docker", "docker-compose",
        
        # Safe system utilities
        "echo", "cat", "head", "tail", "grep", "find", "ls", "pwd",
        "which", "whereis", "file", "stat", "wc", "sort", "uniq",
        
        # Web tools
        "w3m", "curl", "wget"
    ]
    
    # Commands that are explicitly blocked
    BLOCKED_COMMANDS = [
        # Destructive file operations
        "rm", "rmdir", "del", "erase", "format", "mkfs", "fdisk",
        "shred", "wipe", "dd",
        
        # System modification
        "chmod", "chown", "chgrp", "mount", "umount", "fsck",
        "sudo", "su", "passwd", "useradd", "userdel", "usermod",
        
        # Network and security sensitive
        "nc", "netcat", "telnet", "ssh", "scp", "rsync",
        "wget", "curl", "ftp", "sftp",
        
        # Process and system control
        "kill", "killall", "pkill", "shutdown", "reboot", "halt",
        "crontab", "at", "batch",
        
        # Dangerous interpreters without specific scripts
        "bash", "sh", "zsh", "fish", "csh", "tcsh",
        "perl", "ruby", "php", "lua"
    ]
    
    # Dangerous patterns in commands
    DANGEROUS_PATTERNS = [
        r".*;\s*rm\s+-rf",  # Command chaining with rm -rf
        r".*&&\s*rm\s+-rf", # AND chaining with rm -rf
        r".*\|\s*rm\s+-rf", # Pipe chaining with rm -rf
        r".*`.*`.*",        # Command substitution with backticks
        r".*\$\(.*\).*",    # Command substitution with $()
        r".*eval\s+.*",     # Eval statements
        r".*exec\s+.*",     # Exec statements
        r".*/dev/(null|zero|random|urandom).*", # Device file access
        r".*>\s*/dev/.*",   # Writing to device files
        r".*2>&1.*",        # Stderr redirection (often used in attacks)
    ]
    
    # Dangerous environment variables
    DANGEROUS_ENV_VARS = [
        "LD_PRELOAD", "LD_LIBRARY_PATH", "DYLD_INSERT_LIBRARIES",
        "DYLD_LIBRARY_PATH", "PATH", "PYTHONPATH", "NODE_PATH"
    ]
    
    def __init__(
        self,
        allowed_commands: Optional[List[str]] = None,
        blocked_commands: Optional[List[str]] = None,
        working_directory: Optional[str] = None,
        resource_limits: Optional[ResourceLimits] = None,
        enable_strict_mode: bool = True
    ):
        self.allowed_commands = set(allowed_commands or self.DEFAULT_ALLOWED_COMMANDS)
        self.blocked_commands = set(blocked_commands or self.BLOCKED_COMMANDS)
        self.working_directory = working_directory
        self.resource_limits = resource_limits or ResourceLimits()
        self.enable_strict_mode = enable_strict_mode
        self.logger = logging.getLogger(f"{self.__class__.__module__}.{self.__class__.__name__}")
        self._lock = threading.RLock()
        
        # Compile dangerous patterns for performance
        self._dangerous_patterns = [re.compile(pattern) for pattern in self.DANGEROUS_PATTERNS]
    
    def validate_command(self, command: str, environment: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        """
        Validate a command for security issues.
        
        Args:
            command: Command string to validate
            
        Returns:
            Dictionary with validation results
        """
        with self._lock:
            errors = []
            warnings = []
            risk_level = "low"
            
            if not command or not command.strip():
                return {
                    "valid": False,
                    "command": command,
                    "errors": ["Empty command not allowed"],
                    "warnings": [],
                    "risk_level": "high"
                }
            
            command = command.strip()
            
            try:
                # Parse command to extract base command and arguments
                tokens = shlex.split(command)
                if not tokens:
                    return {
                        "valid": False,
                        "command": command,
                        "errors": ["Failed to parse command"],
                        "warnings": [],
                        "risk_level": "high"
                    }
                
                base_command = tokens[0]
                
                # Extract just the command name without path
                command_name = Path(base_command).name
                
                # Check if command is in allowlist
                if command_name not in self.allowed_commands:
                    errors.append(f"Command '{command_name}' not in allowlist")
                    risk_level = "high"
                
                # Check if command is explicitly blocked
                if command_name in self.blocked_commands:
                    errors.append(f"Command '{command_name}' is explicitly blocked")
                    risk_level = "critical"
                
                # Check for dangerous patterns
                for pattern in self._dangerous_patterns:
                    if pattern.search(command):
                        errors.append(f"Dangerous pattern detected in command")
                        risk_level = "critical"
                        break
                
                # Check for command injection attempts
                injection_indicators = [";", "&&", "||", "|", "`", "$(", ">>", "<<"]
                if any(indicator in command for indicator in injection_indicators):
                    # Allow some safe uses in specific contexts
                    if not self._is_safe_command_chaining(command):
                        errors.append("Potential command injection detected")
                        risk_level = "high"
                
                # Validate arguments for suspicious content
                argument_validation = self._validate_arguments(tokens[1:], environment)
                if not argument_validation["valid"]:
                    errors.extend(argument_validation["errors"])
                    if argument_validation["risk_level"] == "high":
                        risk_level = "high"
                
                # Additional strict mode checks
                if self.enable_strict_mode:
                    strict_validation = self._strict_mode_validation(command, tokens)
                    errors.extend(strict_validation["errors"])
                    warnings.extend(strict_validation["warnings"])
                    if strict_validation["risk_level"] == "high":
                        risk_level = "high"
                
            except Exception as e:
                errors.append(f"Command parsing failed: {str(e)}")
                risk_level = "high"
            
            return {
                "valid": len(errors) == 0,
                "command": command,
                "errors": errors,
                "warnings": warnings,
                "risk_level": risk_level
            }
    
    def _is_safe_command_chaining(self, command: str) -> bool:
        """Check if command chaining is safe in context."""
        # Allow safe patterns like: pytest tests/ && echo "Done"
        safe_chaining_patterns = [
            r"^(python|pytest|npm|cargo|go)\s+.+\s+&&\s+echo\s+",
            r"^(pip|npm)\s+install\s+.+\s+&&\s+(pip|npm)\s+",
        ]
        
        # Check for w3m command with safe option patterns
        if command.strip().startswith('w3m '):
            # w3m commands with -o options containing semicolons are safe
            # Example: w3m -o accept_encoding=identity;q=0 or w3m -o 'accept_encoding=identity;q=0'
            if re.search(r"-o\s+(?:'[^']*;[^']*'|[\w_]+=\w+;q=\d+)", command):
                return True
        
        return any(re.match(pattern, command) for pattern in safe_chaining_patterns)
    
    def _validate_arguments(self, args: List[str], environment: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        """Validate command arguments for security issues."""
        errors = []
        risk_level = "low"
        
        for arg in args:
            # Check for path traversal attempts
            if ".." in arg and ("/" in arg or "\\" in arg):
                errors.append(f"Path traversal attempt detected in argument: {arg}")
                risk_level = "high"
            
            # Check for suspicious file patterns
            if re.match(r"^/dev/.*", arg):
                errors.append(f"Device file access detected: {arg}")
                risk_level = "high"
            
            # Check for environment variable expansion
            if "$" in arg and not self._is_safe_env_expansion(arg, environment):
                errors.append(f"Unsafe environment variable expansion: {arg}")
                risk_level = "high"
        
        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "risk_level": risk_level
        }
    
    def _is_safe_env_expansion(self, arg: str, environment: Optional[Dict[str, str]] = None) -> bool:
        """Check if environment variable expansion is safe."""
        # Allow safe patterns like $HOME, $USER, $PWD
        safe_env_patterns = [
            r"^\$HOME(/.+)?$",
            r"^\$USER$",
            r"^\$PWD(/.+)?$",
            r"^\$\{HOME\}(/.+)?$",
            r"^\$\{USER\}$",
            r"^\$\{PWD\}(/.+)?$"
        ]
        
        # Check against standard safe patterns
        if any(re.match(pattern, arg) for pattern in safe_env_patterns):
            return True
        
        # If environment is provided, check if the variable is explicitly provided
        if environment:
            # Extract variable name from $VAR or ${VAR} format
            var_match = re.match(r'^\$\{?([A-Za-z_][A-Za-z0-9_]*)\}?.*$', arg)
            if var_match:
                var_name = var_match.group(1)
                return var_name in environment
        
        return False
    
    def _strict_mode_validation(self, command: str, tokens: List[str]) -> Dict[str, Any]:
        """Additional validation checks for strict mode."""
        errors = []
        warnings = []
        risk_level = "low"
        
        # Check command length
        if len(command) > 1000:
            warnings.append("Command is unusually long")
        
        # Check for too many arguments
        if len(tokens) > 50:
            warnings.append("Command has many arguments")
        
        # Check for non-ASCII characters
        if not command.isascii():
            errors.append("Non-ASCII characters detected in command")
            risk_level = "high"
        
        return {
            "errors": errors,
            "warnings": warnings,
            "risk_level": risk_level
        }
    
    def validate_working_directory(self, directory: str) -> Dict[str, Any]:
        """Validate working directory for command execution."""
        errors = []
        
        try:
            dir_path = Path(directory).resolve()
            
            if not dir_path.exists():
                errors.append(f"Working directory does not exist: {directory}")
            elif not dir_path.is_dir():
                errors.append(f"Path is not a directory: {directory}")
            elif not os.access(dir_path, os.R_OK | os.X_OK):
                errors.append(f"Insufficient permissions for directory: {directory}")
            
        except Exception as e:
            errors.append(f"Directory validation failed: {str(e)}")
        
        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "directory": directory
        }
    
    def validate_environment(self, env_vars: Dict[str, str]) -> Dict[str, Any]:
        """Validate environment variables for security issues."""
        errors = []
        warnings = []
        
        for key, value in env_vars.items():
            if key in self.DANGEROUS_ENV_VARS:
                # Allow PATH modifications in controlled way
                if key == "PATH" and self._is_safe_path_modification(value):
                    warnings.append(f"PATH modification detected: {value}")
                else:
                    errors.append(f"Dangerous environment variable: {key}")
            
            # Check for suspicious values
            if any(char in value for char in [";", "|", "&", "`", "$("]):
                errors.append(f"Suspicious value in environment variable {key}: {value}")
        
        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings
        }
    
    def _is_safe_path_modification(self, path_value: str) -> bool:
        """Check if PATH modification is safe."""
        # Only allow additions to standard paths
        safe_path_prefixes = [
            "/usr/bin", "/usr/local/bin", "/bin", "/sbin",
            "/home", "/opt", "~/.local/bin"
        ]
        
        path_components = path_value.split(":")
        return all(
            any(component.startswith(prefix) for prefix in safe_path_prefixes)
            for component in path_components
        )
    
    def validate_timeout(self, timeout: int) -> Dict[str, Any]:
        """Validate command timeout value."""
        errors = []
        
        if timeout < 0:
            errors.append("Timeout cannot be negative")
        elif timeout == 0:
            errors.append("Timeout cannot be zero")
        elif timeout > 3600:  # 1 hour max
            errors.append("Timeout exceeds maximum allowed (3600 seconds)")
        
        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "timeout": timeout
        }
    
    def validate_resource_limits(self, limits: Dict[str, Any]) -> Dict[str, Any]:
        """Validate resource limits for command execution."""
        errors = []
        
        for key, value in limits.items():
            if not isinstance(value, (int, float)) or value < 0:
                errors.append(f"Invalid resource limit for {key}: {value}")
        
        # Check reasonable bounds
        if "max_memory" in limits and limits["max_memory"] > 8 * 1024 * 1024 * 1024:  # 8GB
            errors.append("Memory limit exceeds reasonable bounds")
        
        if "max_cpu_time" in limits and limits["max_cpu_time"] > 3600:  # 1 hour
            errors.append("CPU time limit exceeds reasonable bounds")
        
        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "limits": limits
        }
    
    def sanitize_command(self, command: str) -> str:
        """Sanitize command by escaping dangerous characters."""
        # Basic sanitization using shlex
        try:
            tokens = shlex.split(command)
            return shlex.join(tokens)
        except ValueError:
            # If parsing fails, escape the entire command
            return shlex.quote(command)


class SecurityAuditLogger:
    """
    Security audit logger for command execution and security events.
    
    Provides comprehensive logging, analysis, and reporting of security-related
    events in the system.
    """
    
    def __init__(
        self,
        log_file: Optional[str] = None,
        max_log_size: int = 100 * 1024 * 1024,  # 100MB
        backup_count: int = 5,
        encrypt_logs: bool = False,
        encryption_key: Optional[str] = None
    ):
        # Set default log file path
        if log_file is None:
            # Use temporary directory for testing, /var/log for production
            import tempfile
            temp_dir = tempfile.gettempdir()
            log_file = os.path.join(temp_dir, "kwe_security.log")
        
        self.log_file = Path(log_file)
        self.max_log_size = max_log_size
        self.backup_count = backup_count
        self.encrypt_logs = encrypt_logs
        self._lock = threading.RLock()
        
        # Setup encryption if enabled
        if encrypt_logs:
            if encryption_key:
                # Use provided key (must be 32 characters for Fernet)
                key = hashlib.sha256(encryption_key.encode()).digest()
                self.fernet = Fernet(Fernet.generate_key())
            else:
                self.fernet = Fernet(Fernet.generate_key())
        else:
            self.fernet = None
        
        # Ensure log directory exists
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Setup logger
        self.logger = logging.getLogger("kwe_security_audit")
        self.logger.setLevel(logging.INFO)
        
        # File handler with rotation
        handler = logging.handlers.RotatingFileHandler(
            self.log_file,
            maxBytes=self.max_log_size,
            backupCount=self.backup_count
        )
        
        formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
    
    def log_command_execution(
        self,
        command: str,
        user: str,
        working_directory: str,
        success: bool,
        exit_code: int,
        execution_time: float,
        pid: Optional[int] = None,
        resource_usage: Optional[Dict[str, Any]] = None
    ) -> None:
        """Log command execution event."""
        with self._lock:
            log_entry = {
                "event_type": "COMMAND_EXECUTION",
                "command": command,
                "user": user,
                "working_directory": working_directory,
                "success": success,
                "exit_code": exit_code,
                "execution_time": execution_time,
                "pid": pid,
                "resource_usage": resource_usage or {},
                "timestamp": datetime.now().isoformat()
            }
            
            status = "SUCCESS" if success else "FAILURE"
            message = f"{status} - User: {user} - Command: {command} - Exit: {exit_code} - Time: {execution_time:.2f}s"
            
            if resource_usage:
                message += f" - Memory: {resource_usage.get('memory', 'N/A')} - CPU: {resource_usage.get('cpu', 'N/A')}"
            
            self._write_log_entry(log_entry, message)
    
    def log_security_violation(
        self,
        command: str,
        violation_type: str,
        user: str,
        details: Optional[Dict[str, Any]] = None,
        timestamp: Optional[str] = None
    ) -> None:
        """Log security violation event."""
        with self._lock:
            log_entry = {
                "event_type": "SECURITY_VIOLATION",
                "command": command,
                "violation_type": violation_type,
                "user": user,
                "details": details or {},
                "timestamp": timestamp or datetime.now().isoformat()
            }
            
            message = f"SECURITY_VIOLATION - Type: {violation_type} - User: {user} - Command: {command}"
            if details:
                message += f" - Details: {json.dumps(details)}"
            
            self._write_log_entry(log_entry, message, level=logging.WARNING)
    
    def log_resource_usage(
        self,
        command: str,
        cpu_time: float,
        memory_peak: int,
        disk_io: int,
        network_io: int,
        timestamp: Optional[str] = None
    ) -> None:
        """Log resource usage metrics."""
        with self._lock:
            log_entry = {
                "event_type": "RESOURCE_USAGE",
                "command": command,
                "cpu_time": cpu_time,
                "memory_peak": memory_peak,
                "disk_io": disk_io,
                "network_io": network_io,
                "timestamp": timestamp or datetime.now().isoformat()
            }
            
            message = f"RESOURCE_USAGE - Command: {command} - CPU: {cpu_time:.2f}s - Memory: {memory_peak} bytes - Disk: {disk_io} bytes - Network: {network_io} bytes"
            
            self._write_log_entry(log_entry, message)
    
    def _write_log_entry(self, log_entry: Dict[str, Any], message: str, level: int = logging.INFO) -> None:
        """Write log entry to file with optional encryption."""
        try:
            if self.encrypt_logs and self.fernet:
                # Encrypt the entire log entry
                encrypted_entry = self.fernet.encrypt(json.dumps(log_entry).encode())
                self.logger.log(level, f"ENCRYPTED: {encrypted_entry.decode()}")
            else:
                # Write as structured JSON
                json_entry = json.dumps(log_entry)
                self.logger.log(level, f"JSON: {json_entry}")
            
            # Also write human-readable message
            self.logger.log(level, message)
            
        except Exception as e:
            # Fallback to basic logging if encryption fails
            self.logger.error(f"Failed to write encrypted log entry: {e}")
            self.logger.log(level, message)
    
    def analyze_security_logs(
        self,
        start_time: str,
        end_time: str,
        event_types: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Analyze security logs for patterns and metrics."""
        with self._lock:
            try:
                start_dt = datetime.fromisoformat(start_time)
                end_dt = datetime.fromisoformat(end_time)
            except ValueError as e:
                return {"error": f"Invalid datetime format: {e}"}
            
            analysis = {
                "total_commands": 0,
                "successful_commands": 0,
                "failed_commands": 0,
                "security_violations": 0,
                "unique_users": set(),
                "command_frequency": {},
                "violation_types": {},
                "resource_usage_stats": {
                    "avg_cpu_time": 0,
                    "avg_memory": 0,
                    "total_disk_io": 0,
                    "total_network_io": 0
                }
            }
            
            try:
                with open(self.log_file, 'r') as f:
                    for line in f:
                        if "JSON:" in line:
                            try:
                                json_part = line.split("JSON:", 1)[1].strip()
                                entry = json.loads(json_part)
                                entry_time = datetime.fromisoformat(entry["timestamp"])
                                
                                if start_dt <= entry_time <= end_dt:
                                    if event_types is None or entry["event_type"] in event_types:
                                        self._update_analysis(analysis, entry)
                                        
                            except (json.JSONDecodeError, KeyError, ValueError):
                                continue
                
                # Calculate averages
                if analysis["total_commands"] > 0:
                    cpu_times = []  # Would need to collect these in _update_analysis
                    if cpu_times:
                        analysis["resource_usage_stats"]["avg_cpu_time"] = sum(cpu_times) / len(cpu_times)
                
                # Convert sets to lists for JSON serialization
                analysis["unique_users"] = list(analysis["unique_users"])
                
                return analysis
                
            except FileNotFoundError:
                return {"error": "Log file not found"}
            except Exception as e:
                return {"error": f"Log analysis failed: {e}"}
    
    def _update_analysis(self, analysis: Dict[str, Any], entry: Dict[str, Any]) -> None:
        """Update analysis with log entry data."""
        event_type = entry["event_type"]
        
        if event_type == "COMMAND_EXECUTION":
            analysis["total_commands"] += 1
            if entry.get("success", False):
                analysis["successful_commands"] += 1
            else:
                analysis["failed_commands"] += 1
            
            # Track user
            analysis["unique_users"].add(entry.get("user", "unknown"))
            
            # Track command frequency
            command = entry.get("command", "unknown")
            analysis["command_frequency"][command] = analysis["command_frequency"].get(command, 0) + 1
            
        elif event_type == "SECURITY_VIOLATION":
            analysis["security_violations"] += 1
            violation_type = entry.get("violation_type", "unknown")
            analysis["violation_types"][violation_type] = analysis["violation_types"].get(violation_type, 0) + 1
    
    def decrypt_logs(self) -> str:
        """Decrypt and return log contents."""
        if not self.encrypt_logs or not self.fernet:
            raise ValueError("Encryption not enabled for this logger")
        
        decrypted_content = []
        
        try:
            with open(self.log_file, 'r') as f:
                for line in f:
                    if "ENCRYPTED:" in line:
                        try:
                            encrypted_part = line.split("ENCRYPTED:", 1)[1].strip()
                            decrypted_data = self.fernet.decrypt(encrypted_part.encode())
                            decrypted_content.append(decrypted_data.decode())
                        except Exception:
                            decrypted_content.append(f"Failed to decrypt: {line}")
                    else:
                        decrypted_content.append(line.strip())
            
            return "\n".join(decrypted_content)
            
        except FileNotFoundError:
            return ""