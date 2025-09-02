"""
Core data models for the KWE CLI Tools System.

This module defines the fundamental data structures used throughout
the tool system for results, metadata, and status tracking.
"""

from typing import Dict, Any, Optional, List, Union
from enum import Enum
from datetime import datetime
import uuid


class ExecutionStatus(Enum):
    """Status of tool execution."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"


class ToolResult:
    """
    Result object returned by tool execution.
    
    Contains execution status, output data, timing information,
    and metadata for monitoring and debugging purposes.
    """
    
    def __init__(
        self,
        success: bool,
        output: str = "",
        data: Optional[Dict[str, Any]] = None,
        error_message: str = "",
        status: ExecutionStatus = ExecutionStatus.COMPLETED,
        execution_time: float = 0.0,
        execution_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        self.success = success
        self.output = output
        self.data = data or {}
        self.error_message = error_message
        self.status = status
        self.execution_time = execution_time
        self.execution_id = execution_id or str(uuid.uuid4())
        self.metadata = metadata or {}
        self.timestamp = datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary for serialization."""
        return {
            "success": self.success,
            "output": self.output,
            "data": self.data,
            "error_message": self.error_message,
            "status": self.status.value,
            "execution_time": self.execution_time,
            "execution_id": self.execution_id,
            "metadata": self.metadata,
            "timestamp": self.timestamp.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ToolResult":
        """Create result from dictionary."""
        return cls(
            success=data["success"],
            output=data.get("output", ""),
            data=data.get("data"),
            error_message=data.get("error_message", ""),
            status=ExecutionStatus(data.get("status", "completed")),
            execution_time=data.get("execution_time", 0.0),
            execution_id=data.get("execution_id"),
            metadata=data.get("metadata")
        )


class ToolInfo:
    """
    Information about a registered tool.
    
    Contains metadata about tool capabilities, configuration,
    and usage information for discovery and validation.
    """
    
    def __init__(
        self,
        name: str,
        category: str,
        description: str,
        capabilities: List[str],
        version: str = "1.0.0",
        author: str = "KWE CLI",
        parameters_schema: Optional[Dict[str, Any]] = None,
        permissions_required: Optional[List[str]] = None,
        resource_requirements: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        self.name = name
        self.category = category
        self.description = description
        self.capabilities = capabilities
        self.version = version
        self.author = author
        self.parameters_schema = parameters_schema or {}
        self.permissions_required = permissions_required or []
        self.resource_requirements = resource_requirements or {}
        self.metadata = metadata or {}
        self.created_at = datetime.now()
        self.last_used = None
        self.usage_count = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert tool info to dictionary for serialization."""
        return {
            "name": self.name,
            "category": self.category,
            "description": self.description,
            "capabilities": self.capabilities,
            "version": self.version,
            "author": self.author,
            "parameters_schema": self.parameters_schema,
            "permissions_required": self.permissions_required,
            "resource_requirements": self.resource_requirements,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat(),
            "last_used": self.last_used.isoformat() if self.last_used else None,
            "usage_count": self.usage_count
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ToolInfo":
        """Create tool info from dictionary."""
        info = cls(
            name=data["name"],
            category=data["category"],
            description=data["description"],
            capabilities=data["capabilities"],
            version=data.get("version", "1.0.0"),
            author=data.get("author", "KWE CLI"),
            parameters_schema=data.get("parameters_schema"),
            permissions_required=data.get("permissions_required"),
            resource_requirements=data.get("resource_requirements"),
            metadata=data.get("metadata")
        )
        
        if data.get("last_used"):
            info.last_used = datetime.fromisoformat(data["last_used"])
        
        info.usage_count = data.get("usage_count", 0)
        return info
    
    def update_usage(self):
        """Update usage statistics."""
        self.last_used = datetime.now()
        self.usage_count += 1


class ExecutionContext:
    """
    Context information for tool execution.
    
    Provides isolation and resource management for individual
    tool executions with proper cleanup handling.
    """
    
    def __init__(
        self,
        execution_id: str,
        tool_name: str,
        parameters: Dict[str, Any],
        user_context: Optional[Dict[str, Any]] = None,
        timeout: Optional[float] = None,
        max_retries: int = 0,
        resource_limits: Optional[Dict[str, Any]] = None
    ):
        self.execution_id = execution_id
        self.tool_name = tool_name
        self.parameters = parameters
        self.user_context = user_context or {}
        self.timeout = timeout
        self.max_retries = max_retries
        self.resource_limits = resource_limits or {}
        self.created_at = datetime.now()
        self.started_at = None
        self.completed_at = None
        self.attempt_count = 0
        self.progress = 0.0
        self.status = ExecutionStatus.PENDING
        self.temporary_resources = []
    
    def start_execution(self):
        """Mark execution as started."""
        self.started_at = datetime.now()
        self.status = ExecutionStatus.RUNNING
    
    def complete_execution(self, status: ExecutionStatus):
        """Mark execution as completed with given status."""
        self.completed_at = datetime.now()
        self.status = status
    
    def add_temporary_resource(self, resource: str):
        """Add temporary resource for cleanup."""
        self.temporary_resources.append(resource)
    
    def cleanup(self):
        """Cleanup temporary resources."""
        # Implementation will handle cleanup of temporary files, connections, etc.
        self.temporary_resources.clear()
    
    def get_execution_duration(self) -> Optional[float]:
        """Get execution duration in seconds."""
        if self.started_at and self.completed_at:
            return (self.completed_at - self.started_at).total_seconds()
        return None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert context to dictionary."""
        return {
            "execution_id": self.execution_id,
            "tool_name": self.tool_name,
            "parameters": self.parameters,
            "user_context": self.user_context,
            "timeout": self.timeout,
            "max_retries": self.max_retries,
            "resource_limits": self.resource_limits,
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "attempt_count": self.attempt_count,
            "progress": self.progress,
            "status": self.status.value,
            "temporary_resources": self.temporary_resources
        }