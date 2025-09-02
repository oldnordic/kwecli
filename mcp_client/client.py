"""
KWE CLI MCP Client.

Enterprise-grade MCP client for connecting to and interacting with
MCP servers like Claude Code and other AI assistants.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Set, Tuple
from datetime import datetime, timedelta

from .models import (
    MCPRequest, MCPResponse, Tool, ToolResult, ServerCapabilities,
    InitializeResult, ConnectionStats, ClientCapabilities, ClientInfo,
    MCPError, MCPConnectionError, MCPTimeoutError, MCPToolError
)
from .config import MCPClientConfig, MCPServerConfig, TransportType
from .transport import Transport, create_transport


logger = logging.getLogger(__name__)


class MCPServerConnection:
    """Represents a connection to a single MCP server."""
    
    def __init__(self, server_config: MCPServerConfig):
        self.config = server_config
        self.transport = create_transport(server_config)
        self.capabilities: Optional[ServerCapabilities] = None
        self.tools: Dict[str, Tool] = {}
        self.stats = ConnectionStats(
            server_name=server_config.name,
            connected_at=datetime.utcnow()
        )
        self.last_health_check: Optional[datetime] = None
        self.health_check_task: Optional[asyncio.Task] = None
        
    async def connect(self) -> bool:
        """Connect to MCP server and discover capabilities."""
        try:
            # Establish transport connection
            if not await self.transport.connect():
                return False
            
            # Discover server capabilities
            await self._discover_capabilities()
            
            # Discover available tools
            await self._discover_tools()
            
            # Start health check task if enabled
            if self.config.health_check_enabled:
                self.health_check_task = asyncio.create_task(self._health_check_loop())
            
            logger.info(f"Connected to MCP server '{self.config.name}' with {len(self.tools)} tools")
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect to MCP server '{self.config.name}': {e}")
            await self.disconnect()
            return False
    
    async def disconnect(self):
        """Disconnect from MCP server."""
        # Stop health check task
        if self.health_check_task:
            self.health_check_task.cancel()
            try:
                await self.health_check_task
            except asyncio.CancelledError:
                pass
        
        # Close transport
        if self.transport:
            await self.transport.disconnect()
        
        logger.info(f"Disconnected from MCP server '{self.config.name}'")
    
    async def _discover_capabilities(self):
        """Discover server capabilities."""
        request = MCPRequest(
            method="initialize",
            params={
                "protocolVersion": "2025-03-26",
                "capabilities": ClientCapabilities().to_dict(),
                "clientInfo": ClientInfo(name="KWE CLI", version="1.0.0").__dict__
            }
        )
        
        self.stats.record_request()
        response = await self.transport.send_request(request)
        
        if response.is_error():
            self.stats.record_error()
            raise MCPConnectionError(f"Failed to initialize: {response.get_error_message()}")
        
        self.stats.record_response()
        
        # Parse initialization result
        if response.result:
            init_result = InitializeResult.from_dict(response.result)
            self.capabilities = init_result.capabilities
    
    async def _discover_tools(self):
        """Discover available tools from server."""
        if not self.capabilities or not self.capabilities.supports_tools():
            return
        
        request = MCPRequest(method="tools/list", params={})
        
        self.stats.record_request()
        response = await self.transport.send_request(request)
        
        if response.is_error():
            self.stats.record_error()
            logger.warning(f"Failed to discover tools from '{self.config.name}': {response.get_error_message()}")
            return
        
        self.stats.record_response()
        
        # Parse tools
        if response.result and "tools" in response.result:
            for tool_data in response.result["tools"]:
                tool = Tool.from_dict(tool_data)
                self.tools[tool.name] = tool
    
    async def execute_tool(self, tool_name: str, arguments: Dict[str, Any]) -> ToolResult:
        """Execute a tool on this server."""
        if tool_name not in self.tools:
            raise MCPToolError(f"Tool '{tool_name}' not found on server '{self.config.name}'")
        
        tool = self.tools[tool_name]
        
        # Validate arguments
        if not tool.validate_arguments(arguments):
            raise MCPToolError(f"Invalid arguments for tool '{tool_name}'")
        
        request = MCPRequest(
            method="tools/call",
            params={
                "name": tool_name,
                "arguments": arguments
            }
        )
        
        self.stats.record_request()
        
        try:
            response = await asyncio.wait_for(
                self.transport.send_request(request),
                timeout=self.config.tool_timeout
            )
            
            if response.is_error():
                self.stats.record_error()
                raise MCPToolError(f"Tool execution failed: {response.get_error_message()}")
            
            self.stats.record_response()
            
            # Parse tool result
            if response.result:
                return ToolResult.from_dict(response.result)
            else:
                return ToolResult(content=[], is_error=True)
                
        except asyncio.TimeoutError:
            self.stats.record_error()
            raise MCPTimeoutError(f"Tool execution timeout after {self.config.tool_timeout}s")
    
    async def _health_check_loop(self):
        """Background health check loop."""
        while self.transport.is_connected():
            try:
                await asyncio.sleep(self.config.health_check_interval)
                
                if await self.transport.health_check():
                    self.last_health_check = datetime.utcnow()
                else:
                    logger.warning(f"Health check failed for server '{self.config.name}'")
                    # Could implement reconnection logic here
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Health check error for server '{self.config.name}': {e}")
    
    def is_connected(self) -> bool:
        """Check if connection is active."""
        return self.transport.is_connected()
    
    def get_tool_names(self) -> List[str]:
        """Get list of available tool names."""
        return list(self.tools.keys())
    
    def get_tool(self, name: str) -> Optional[Tool]:
        """Get tool definition by name."""
        return self.tools.get(name)


class KWEMCPClient:
    """Main MCP client for KWE CLI."""
    
    def __init__(self, config: MCPClientConfig):
        self.config = config
        self.connections: Dict[str, MCPServerConnection] = {}
        self.connection_pool_semaphore = asyncio.Semaphore(config.connection_pool.max_connections)
        self.started = False
        
    async def start(self) -> bool:
        """Start MCP client and connect to all configured servers."""
        if self.started:
            return True
        
        success_count = 0
        total_servers = len(self.config.servers)
        
        # Connect to all servers
        connection_tasks = []
        for server_name, server_config in self.config.servers.items():
            task = asyncio.create_task(self._connect_server(server_name, server_config))
            connection_tasks.append(task)
        
        # Wait for all connections
        results = await asyncio.gather(*connection_tasks, return_exceptions=True)
        
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                server_name = list(self.config.servers.keys())[i]
                logger.error(f"Failed to connect to server '{server_name}': {result}")
            elif result:
                success_count += 1
        
        self.started = success_count > 0
        
        logger.info(f"MCP client started: {success_count}/{total_servers} servers connected")
        return self.started
    
    async def stop(self):
        """Stop MCP client and disconnect from all servers."""
        if not self.started:
            return
        
        # Disconnect all servers
        disconnect_tasks = []
        for connection in self.connections.values():
            task = asyncio.create_task(connection.disconnect())
            disconnect_tasks.append(task)
        
        await asyncio.gather(*disconnect_tasks, return_exceptions=True)
        
        self.connections.clear()
        self.started = False
        
        logger.info("MCP client stopped")
    
    async def _connect_server(self, server_name: str, server_config: MCPServerConfig) -> bool:
        """Connect to a specific server."""
        async with self.connection_pool_semaphore:
            try:
                connection = MCPServerConnection(server_config)
                if await connection.connect():
                    self.connections[server_name] = connection
                    return True
                return False
            except Exception as e:
                logger.error(f"Error connecting to server '{server_name}': {e}")
                return False
    
    def get_connected_servers(self) -> List[str]:
        """Get list of connected server names."""
        return [name for name, conn in self.connections.items() if conn.is_connected()]
    
    def get_server_connection(self, server_name: str) -> Optional[MCPServerConnection]:
        """Get connection for specific server."""
        return self.connections.get(server_name)
    
    def get_all_tools(self) -> Dict[str, Tuple[str, Tool]]:
        """Get all available tools across all servers."""
        all_tools = {}
        for server_name, connection in self.connections.items():
            if connection.is_connected():
                for tool_name, tool in connection.tools.items():
                    # Use server-prefixed tool name to avoid conflicts
                    prefixed_name = f"{server_name}.{tool_name}"
                    all_tools[prefixed_name] = (server_name, tool)
        return all_tools
    
    def find_tool(self, tool_name: str) -> Optional[Tuple[str, Tool]]:
        """Find a tool by name across all servers."""
        # First try exact match with server prefix
        all_tools = self.get_all_tools()
        if tool_name in all_tools:
            return all_tools[tool_name]
        
        # Then try without prefix (first match wins)
        for prefixed_name, (server_name, tool) in all_tools.items():
            if prefixed_name.endswith(f".{tool_name}"):
                return server_name, tool
        
        return None
    
    async def execute_tool(self, tool_name: str, arguments: Dict[str, Any], server_name: Optional[str] = None) -> ToolResult:
        """Execute a tool, optionally on a specific server."""
        if server_name:
            # Execute on specific server
            connection = self.connections.get(server_name)
            if not connection or not connection.is_connected():
                raise MCPConnectionError(f"Server '{server_name}' not connected")
            
            return await connection.execute_tool(tool_name, arguments)
        else:
            # Find and execute tool
            tool_info = self.find_tool(tool_name)
            if not tool_info:
                raise MCPToolError(f"Tool '{tool_name}' not found on any connected server")
            
            found_server_name, _ = tool_info
            connection = self.connections[found_server_name]
            return await connection.execute_tool(tool_name, arguments)
    
    def get_server_stats(self) -> Dict[str, ConnectionStats]:
        """Get connection statistics for all servers."""
        stats = {}
        for server_name, connection in self.connections.items():
            stats[server_name] = connection.stats
        return stats
    
    def get_client_info(self) -> Dict[str, Any]:
        """Get client information and status."""
        connected_servers = self.get_connected_servers()
        all_tools = self.get_all_tools()
        
        return {
            "client_name": self.config.client_name,
            "client_version": self.config.client_version,
            "started": self.started,
            "connected_servers": len(connected_servers),
            "total_servers": len(self.config.servers),
            "available_tools": len(all_tools),
            "servers": {
                name: {
                    "connected": conn.is_connected(),
                    "transport": conn.config.transport.value,
                    "tools": len(conn.tools),
                    "last_health_check": conn.last_health_check.isoformat() if conn.last_health_check else None
                }
                for name, conn in self.connections.items()
            }
        }
    
    async def refresh_server_tools(self, server_name: str) -> bool:
        """Refresh tools for a specific server."""
        connection = self.connections.get(server_name)
        if not connection or not connection.is_connected():
            return False
        
        try:
            await connection._discover_tools()
            logger.info(f"Refreshed tools for server '{server_name}': {len(connection.tools)} tools")
            return True
        except Exception as e:
            logger.error(f"Failed to refresh tools for server '{server_name}': {e}")
            return False
    
    async def test_server_connection(self, server_name: str) -> bool:
        """Test connection to a specific server."""
        connection = self.connections.get(server_name)
        if not connection:
            return False
        
        return await connection.transport.health_check()


# Factory function for easy client creation
async def create_mcp_client(config: Optional[MCPClientConfig] = None) -> KWEMCPClient:
    """Create and start MCP client with configuration."""
    if config is None:
        from .config import load_default_config
        config = load_default_config()
    
    client = KWEMCPClient(config)
    await client.start()
    return client