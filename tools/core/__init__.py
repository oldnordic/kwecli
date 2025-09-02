"""
Core infrastructure for the KWE CLI Tools System.

This module contains the fundamental building blocks:
- Tool registration and discovery
- Tool execution engine with safety features
- Result handling and status tracking
- Permission management
- Base interfaces and contracts
"""

from .models import ToolResult, ToolInfo, ExecutionStatus
from .tool_interface import BaseTool
from .registry import ToolRegistry, ToolRegistrationError
from .executor import ToolExecutor, ExecutionTimeoutError, ExecutionError
from .permissions import PermissionManager

__all__ = [
    "ToolResult",
    "ToolInfo", 
    "ExecutionStatus", 
    "BaseTool",
    "ToolRegistry",
    "ToolRegistrationError",
    "ToolExecutor",
    "ExecutionTimeoutError",
    "ExecutionError",
    "PermissionManager"
]