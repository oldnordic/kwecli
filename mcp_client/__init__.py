"""
KWE CLI MCP (Model Context Protocol) Client.

Enterprise-grade MCP client that enables KWE CLI to connect to and
interact with MCP servers like Claude Code and other AI assistants.
"""

__version__ = "1.0.0"
__author__ = "KWE CLI Development Team"

from .client import KWEMCPClient, MCPServerConnection, create_mcp_client
from .config import MCPClientConfig, MCPServerConfig, TransportType, AuthType, load_default_config, create_example_config
from .transport import Transport, HTTPTransport, StdioTransport, WebSocketTransport, create_transport
from .models import (
    MCPRequest, MCPResponse, Tool, ToolResult, ServerCapabilities, ConnectionStats,
    MCPError, MCPConnectionError, MCPTimeoutError, MCPToolError, MCPProtocolError
)

__all__ = [
    "KWEMCPClient",
    "MCPServerConnection",
    "create_mcp_client",
    "MCPClientConfig", 
    "MCPServerConfig",
    "TransportType",
    "AuthType",
    "load_default_config",
    "create_example_config",
    "Transport",
    "HTTPTransport",
    "StdioTransport", 
    "WebSocketTransport",
    "create_transport",
    "MCPRequest",
    "MCPResponse", 
    "Tool",
    "ToolResult",
    "ServerCapabilities",
    "ConnectionStats",
    "MCPError",
    "MCPConnectionError",
    "MCPTimeoutError", 
    "MCPToolError",
    "MCPProtocolError"
]