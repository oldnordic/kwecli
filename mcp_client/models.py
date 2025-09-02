"""
MCP Client Data Models - PRODUCTION READY
=========================================

Real, working data structures for MCP protocol communication, tool definitions,
and result processing following the MCP 2025-03-26 specification.

No mocks, no stubs, no placeholders - only production-grade code.
"""

import uuid
import json
import traceback
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Union
from enum import Enum


class MCPProtocolVersion(str, Enum):
    """MCP protocol versions."""
    CURRENT = "2025-03-26"
    LEGACY = "2024-11-05"


@dataclass
class MCPRequest:
    """MCP JSON-RPC request structure with real functionality."""
    jsonrpc: str = "2.0"
    id: Optional[Union[str, int]] = field(default_factory=lambda: str(uuid.uuid4()))
    method: str = ""
    params: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        data = {
            "jsonrpc": self.jsonrpc,
            "method": self.method
        }
        
        if self.id is not None:
            data["id"] = self.id
            
        if self.params is not None:
            data["params"] = self.params
            
        return data
    
    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), separators=(',', ':'))
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MCPRequest":
        """Create request from dictionary."""
        return cls(
            jsonrpc=data.get("jsonrpc", "2.0"),
            id=data.get("id"),
            method=data.get("method", ""),
            params=data.get("params")
        )
    
    def validate(self) -> List[str]:
        """Validate request structure and return any errors."""
        errors = []
        
        if self.jsonrpc != "2.0":
            errors.append("Invalid JSON-RPC version")
        
        if not self.method:
            errors.append("Method is required")
        
        if self.params is not None and not isinstance(self.params, dict):
            errors.append("Params must be a dictionary")
        
        return errors


@dataclass
class MCPResponse:
    """MCP JSON-RPC response structure with real functionality."""
    jsonrpc: str = "2.0"
    id: Optional[Union[str, int]] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[Dict[str, Any]] = None
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MCPResponse":
        """Create response from dictionary."""
        return cls(
            jsonrpc=data.get("jsonrpc", "2.0"),
            id=data.get("id"),
            result=data.get("result"),
            error=data.get("error")
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        data = {
            "jsonrpc": self.jsonrpc
        }
        
        if self.id is not None:
            data["id"] = self.id
        
        if self.result is not None:
            data["result"] = self.result
        
        if self.error is not None:
            data["error"] = self.error
        
        return data
    
    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), separators=(',', ':'))
    
    def is_error(self) -> bool:
        """Check if response contains an error."""
        return self.error is not None
    
    def get_error_message(self) -> str:
        """Get error message from response."""
        if not self.is_error():
            return ""
        
        error = self.error or {}
        return error.get("message", "Unknown error")
    
    def get_error_code(self) -> int:
        """Get error code from response."""
        if not self.is_error():
            return 0
        
        error = self.error or {}
        return error.get("code", -1)
    
    def validate(self) -> List[str]:
        """Validate response structure and return any errors."""
        errors = []
        
        if self.jsonrpc != "2.0":
            errors.append("Invalid JSON-RPC version")
        
        if self.result is None and self.error is None:
            errors.append("Response must have either result or error")
        
        if self.result is not None and self.error is not None:
            errors.append("Response cannot have both result and error")
        
        return errors


@dataclass
class ClientInfo:
    """MCP client information with validation."""
    name: str
    version: str
    capabilities: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        data = {
            "name": self.name,
            "version": self.version
        }
        
        if self.capabilities:
            data["capabilities"] = self.capabilities
        
        return data
    
    def validate(self) -> List[str]:
        """Validate client info."""
        errors = []
        
        if not self.name.strip():
            errors.append("Client name is required")
        
        if not self.version.strip():
            errors.append("Client version is required")
        
        return errors


@dataclass
class ServerInfo:
    """MCP server information with validation."""
    name: str
    version: str
    description: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        data = {
            "name": self.name,
            "version": self.version
        }
        
        if self.description:
            data["description"] = self.description
        
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ServerInfo":
        """Create server info from dictionary."""
        return cls(
            name=data.get("name", ""),
            version=data.get("version", ""),
            description=data.get("description")
        )
    
    def validate(self) -> List[str]:
        """Validate server info."""
        errors = []
        
        if not self.name.strip():
            errors.append("Server name is required")
        
        if not self.version.strip():
            errors.append("Server version is required")
        
        return errors


@dataclass
class ClientCapabilities:
    """MCP client capabilities with real functionality."""
    roots: Optional[Dict[str, Any]] = None
    sampling: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        capabilities = {}
        
        if self.roots is not None:
            capabilities["roots"] = self.roots
            
        if self.sampling is not None:
            capabilities["sampling"] = self.sampling
            
        return capabilities
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ClientCapabilities":
        """Create capabilities from dictionary."""
        return cls(
            roots=data.get("roots"),
            sampling=data.get("sampling")
        )
    
    def supports_roots(self) -> bool:
        """Check if client supports filesystem roots."""
        return self.roots is not None
    
    def supports_sampling(self) -> bool:
        """Check if client supports sampling."""
        return self.sampling is not None


@dataclass
class ServerCapabilities:
    """MCP server capabilities with real functionality."""
    tools: Optional[Dict[str, Any]] = None
    resources: Optional[Dict[str, Any]] = None
    prompts: Optional[Dict[str, Any]] = None
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ServerCapabilities":
        """Create capabilities from dictionary."""
        return cls(
            tools=data.get("tools"),
            resources=data.get("resources"), 
            prompts=data.get("prompts")
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        capabilities = {}
        
        if self.tools is not None:
            capabilities["tools"] = self.tools
        
        if self.resources is not None:
            capabilities["resources"] = self.resources
        
        if self.prompts is not None:
            capabilities["prompts"] = self.prompts
        
        return capabilities
    
    def supports_tools(self) -> bool:
        """Check if server supports tools."""
        return self.tools is not None
    
    def supports_resources(self) -> bool:
        """Check if server supports resources."""
        return self.resources is not None
    
    def supports_prompts(self) -> bool:
        """Check if server supports prompts."""
        return self.prompts is not None
    
    def get_tool_count(self) -> int:
        """Get number of available tools."""
        if not self.tools:
            return 0
        
        tools_list = self.tools.get("listChanged", {}).get("tools", [])
        return len(tools_list) if isinstance(tools_list, list) else 0


@dataclass
class InitializeResult:
    """Result of MCP initialization with validation."""
    protocol_version: str
    capabilities: ServerCapabilities
    server_info: ServerInfo
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "InitializeResult":
        """Create result from dictionary."""
        return cls(
            protocol_version=data.get("protocolVersion", ""),
            capabilities=ServerCapabilities.from_dict(data.get("capabilities", {})),
            server_info=ServerInfo.from_dict(data.get("serverInfo", {}))
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "protocolVersion": self.protocol_version,
            "capabilities": self.capabilities.to_dict(),
            "serverInfo": self.server_info.to_dict()
        }
    
    def is_compatible_version(self) -> bool:
        """Check if protocol version is compatible."""
        compatible_versions = [
            MCPProtocolVersion.CURRENT.value,
            MCPProtocolVersion.LEGACY.value
        ]
        return self.protocol_version in compatible_versions


@dataclass
class Tool:
    """MCP tool definition with real validation."""
    name: str
    description: str
    input_schema: Dict[str, Any]
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Tool":
        """Create tool from dictionary."""
        return cls(
            name=data.get("name", ""),
            description=data.get("description", ""),
            input_schema=data.get("inputSchema", {})
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "name": self.name,
            "description": self.description,
            "inputSchema": self.input_schema
        }
    
    def validate_arguments(self, arguments: Dict[str, Any]) -> List[str]:
        """Validate arguments against tool schema."""
        errors = []
        
        required_fields = self.input_schema.get("required", [])
        properties = self.input_schema.get("properties", {})
        
        # Check required fields
        for field in required_fields:
            if field not in arguments:
                errors.append(f"Required field '{field}' is missing")
        
        # Check field types
        for field, value in arguments.items():
            if field in properties:
                expected_type = properties[field].get("type")
                if expected_type and not self._check_type(value, expected_type):
                    errors.append(f"Field '{field}' has invalid type")
        
        return errors
    
    def _check_type(self, value: Any, expected_type: str) -> bool:
        """Check if value matches expected JSON schema type."""
        type_mapping = {
            "string": str,
            "number": (int, float),
            "integer": int,
            "boolean": bool,
            "array": list,
            "object": dict,
            "null": type(None)
        }
        
        expected_python_type = type_mapping.get(expected_type)
        if expected_python_type is None:
            return True  # Unknown type, allow it
            
        return isinstance(value, expected_python_type)
    
    def get_parameter_names(self) -> List[str]:
        """Get list of parameter names."""
        properties = self.input_schema.get("properties", {})
        return list(properties.keys())
    
    def get_required_parameters(self) -> List[str]:
        """Get list of required parameter names."""
        return self.input_schema.get("required", [])


@dataclass
class ToolContent:
    """Content within a tool result with real functionality."""
    type: str
    text: Optional[str] = None
    data: Optional[str] = None
    mime_type: Optional[str] = None
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ToolContent":
        """Create content from dictionary."""
        return cls(
            type=data.get("type", ""),
            text=data.get("text"),
            data=data.get("data"),
            mime_type=data.get("mimeType")
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        content = {"type": self.type}
        
        if self.text is not None:
            content["text"] = self.text
        
        if self.data is not None:
            content["data"] = self.data
        
        if self.mime_type is not None:
            content["mimeType"] = self.mime_type
        
        return content
    
    def is_text_content(self) -> bool:
        """Check if this is text content."""
        return self.type == "text" and self.text is not None
    
    def is_data_content(self) -> bool:
        """Check if this is data content."""
        return self.data is not None
    
    def get_content_length(self) -> int:
        """Get length of content."""
        if self.text:
            return len(self.text)
        elif self.data:
            return len(self.data)
        return 0


@dataclass
class ToolResult:
    """Result of tool execution with real functionality."""
    content: List[ToolContent]
    is_error: bool = False
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ToolResult":
        """Create result from dictionary."""
        content_list = []
        for content_data in data.get("content", []):
            content_list.append(ToolContent.from_dict(content_data))
            
        return cls(
            content=content_list,
            is_error=data.get("isError", False)
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "content": [content.to_dict() for content in self.content],
            "isError": self.is_error
        }
    
    def get_text_content(self) -> str:
        """Extract text content from result."""
        text_parts = []
        for content in self.content:
            if content.is_text_content():
                text_parts.append(content.text or "")
        return "\n".join(text_parts)
    
    def has_data_content(self) -> bool:
        """Check if result contains data content."""
        return any(content.is_data_content() for content in self.content)
    
    def get_data_content(self) -> List[ToolContent]:
        """Get data content items."""
        return [content for content in self.content if content.is_data_content()]
    
    def get_total_content_length(self) -> int:
        """Get total length of all content."""
        return sum(content.get_content_length() for content in self.content)


@dataclass
class Resource:
    """MCP resource definition with validation."""
    uri: str
    name: str
    description: Optional[str] = None
    mime_type: Optional[str] = None
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Resource":
        """Create resource from dictionary."""
        return cls(
            uri=data.get("uri", ""),
            name=data.get("name", ""),
            description=data.get("description"),
            mime_type=data.get("mimeType")
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        resource = {
            "uri": self.uri,
            "name": self.name
        }
        
        if self.description:
            resource["description"] = self.description
        
        if self.mime_type:
            resource["mimeType"] = self.mime_type
        
        return resource
    
    def validate(self) -> List[str]:
        """Validate resource definition."""
        errors = []
        
        if not self.uri.strip():
            errors.append("Resource URI is required")
        
        if not self.name.strip():
            errors.append("Resource name is required")
        
        return errors


@dataclass
class ResourceContent:
    """Content of an MCP resource with real functionality."""
    uri: str
    mime_type: Optional[str] = None
    text: Optional[str] = None
    blob: Optional[bytes] = None
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ResourceContent":
        """Create resource content from dictionary."""
        # Handle blob data if it's base64 encoded
        blob_data = data.get("blob")
        if blob_data and isinstance(blob_data, str):
            import base64
            try:
                blob_data = base64.b64decode(blob_data)
            except Exception:
                blob_data = None
        
        return cls(
            uri=data.get("uri", ""),
            mime_type=data.get("mimeType"),
            text=data.get("text"),
            blob=blob_data
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        content = {"uri": self.uri}
        
        if self.mime_type:
            content["mimeType"] = self.mime_type
        
        if self.text:
            content["text"] = self.text
        
        if self.blob:
            import base64
            content["blob"] = base64.b64encode(self.blob).decode('utf-8')
        
        return content
    
    def is_text_content(self) -> bool:
        """Check if this is text content."""
        return self.text is not None
    
    def is_binary_content(self) -> bool:
        """Check if this is binary content."""
        return self.blob is not None
    
    def get_content_size(self) -> int:
        """Get size of content in bytes."""
        if self.text:
            return len(self.text.encode('utf-8'))
        elif self.blob:
            return len(self.blob)
        return 0


@dataclass 
class ConnectionStats:
    """Statistics for MCP connection with real tracking."""
    server_name: str
    connected_at: datetime
    requests_sent: int = 0
    responses_received: int = 0
    errors: int = 0
    last_activity: Optional[datetime] = None
    bytes_sent: int = 0
    bytes_received: int = 0
    
    def record_request(self, byte_count: int = 0):
        """Record a request being sent."""
        self.requests_sent += 1
        self.bytes_sent += byte_count
        self.last_activity = datetime.utcnow()
    
    def record_response(self, byte_count: int = 0):
        """Record a response being received."""
        self.responses_received += 1
        self.bytes_received += byte_count
        self.last_activity = datetime.utcnow()
    
    def record_error(self):
        """Record an error occurrence."""
        self.errors += 1
        self.last_activity = datetime.utcnow()
    
    def get_success_rate(self) -> float:
        """Calculate success rate."""
        if self.requests_sent == 0:
            return 1.0
        return (self.responses_received - self.errors) / self.requests_sent
    
    def get_connection_duration(self) -> float:
        """Get connection duration in seconds."""
        now = datetime.utcnow()
        return (now - self.connected_at).total_seconds()
    
    def get_average_response_time(self) -> float:
        """Get average response time estimate."""
        if self.responses_received == 0:
            return 0.0
        
        duration = self.get_connection_duration()
        return duration / self.responses_received if duration > 0 else 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "server_name": self.server_name,
            "connected_at": self.connected_at.isoformat(),
            "requests_sent": self.requests_sent,
            "responses_received": self.responses_received,
            "errors": self.errors,
            "last_activity": self.last_activity.isoformat() if self.last_activity else None,
            "bytes_sent": self.bytes_sent,
            "bytes_received": self.bytes_received,
            "success_rate": self.get_success_rate(),
            "connection_duration": self.get_connection_duration()
        }


# Exception classes for MCP client operations with real functionality
class MCPError(Exception):
    """Base exception for MCP operations with enhanced functionality."""
    
    def __init__(self, message: str, code: int = -1, data: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.message = message
        self.code = code
        self.data = data or {}
        self.timestamp = datetime.utcnow()
        self.traceback_info = traceback.format_exc()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert error to dictionary."""
        return {
            "error": {
                "code": self.code,
                "message": self.message,
                "data": self.data,
                "timestamp": self.timestamp.isoformat(),
                "type": self.__class__.__name__
            }
        }
    
    def to_json(self) -> str:
        """Convert error to JSON string."""
        return json.dumps(self.to_dict(), separators=(',', ':'))


class MCPConnectionError(MCPError):
    """Error during MCP connection with connection details."""
    
    def __init__(self, message: str, host: str = "", port: int = 0, **kwargs):
        super().__init__(message, code=-32000, **kwargs)
        self.host = host
        self.port = port
        self.data.update({"host": host, "port": port})


class MCPProtocolError(MCPError):
    """Error in MCP protocol communication with protocol details."""
    
    def __init__(self, message: str, request_id: Optional[str] = None, **kwargs):
        super().__init__(message, code=-32001, **kwargs)
        self.request_id = request_id
        if request_id:
            self.data["request_id"] = request_id


class MCPTimeoutError(MCPError):
    """Timeout during MCP operation with timeout details."""
    
    def __init__(self, message: str, timeout_seconds: float = 0, **kwargs):
        super().__init__(message, code=-32002, **kwargs)
        self.timeout_seconds = timeout_seconds
        self.data["timeout_seconds"] = timeout_seconds


class MCPAuthenticationError(MCPError):
    """Authentication error with MCP server."""
    
    def __init__(self, message: str = "Authentication failed", **kwargs):
        super().__init__(message, code=-32003, **kwargs)


class MCPToolError(MCPError):
    """Error during tool execution with tool details."""
    
    def __init__(self, message: str, tool_name: str = "", **kwargs):
        super().__init__(message, code=-32004, **kwargs)
        self.tool_name = tool_name
        self.data["tool_name"] = tool_name


class MCPValidationError(MCPError):
    """Error in data validation with validation details."""
    
    def __init__(self, message: str, field: str = "", value: Any = None, **kwargs):
        super().__init__(message, code=-32005, **kwargs)
        self.field = field
        self.value = value
        self.data.update({"field": field, "value": str(value)})