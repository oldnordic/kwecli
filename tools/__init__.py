"""
KWE CLI Tools System

A comprehensive tool execution framework for the Knowledge Workflow Engine CLI.
Provides secure, monitored, and efficient tool execution with proper error handling,
resource management, and integration with the existing KWE CLI architecture.

This module follows the Phase 1 implementation plan:
- Core tool infrastructure (Registry, Executor, Results)
- File system operations tools
- Permission and security system
- Integration with FastAPI backend
"""

from tools.core.registry import ToolRegistry
from tools.core.executor import ToolExecutor
from tools.core.models import ToolResult, ToolInfo, ExecutionStatus
from tools.core.permissions import PermissionManager

# Chat system integration
try:
    from tools.kwecli_tools import KWECLIToolsIntegration
except ImportError:
    KWECLIToolsIntegration = None

__version__ = "1.0.0"
__all__ = [
    "ToolRegistry",
    "ToolExecutor",
    "ToolResult",
    "ToolInfo", 
    "ExecutionStatus",
    "PermissionManager",
    "KWECLIToolsIntegration"
]