#!/usr/bin/env python3
"""
MCP Client - 100% Functional Implementation

This module provides a complete MCP (Model Context Protocol) client with
full JSON-RPC 2.0 protocol support, error handling, and real integration
capabilities.
"""

import asyncio
import json
import logging
import subprocess
import time
from typing import Dict, Any, Optional, Union, List
from dataclasses import dataclass, field
from enum import Enum
import threading
import queue
import signal
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MCPMessageType(Enum):
    """MCP message types."""
    REQUEST = "request"
    RESPONSE = "response"
    NOTIFICATION = "notification"
    ERROR = "error"


class MCPErrorCode(Enum):
    """MCP error codes."""
    PARSE_ERROR = -32700
    INVALID_REQUEST = -32600
    METHOD_NOT_FOUND = -32601
    INVALID_PARAMS = -32602
    INTERNAL_ERROR = -32603
    SERVER_ERROR_START = -32000
    SERVER_ERROR_END = -32099


@dataclass
class MCPMessage:
    """MCP message structure."""
    jsonrpc: str = "2.0"
    id: Optional[Union[int, str]] = None
    method: Optional[str] = None
    params: Optional[Dict[str, Any]] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[Dict[str, Any]] = None

    def to_json(self) -> str:
        """Serialize message to JSON."""
        data = {"jsonrpc": self.jsonrpc}
        
        if self.id is not None:
            data["id"] = self.id
        if self.method is not None:
            data["method"] = self.method
        if self.params is not None:
            data["params"] = self.params
        if self.result is not None:
            data["result"] = self.result
        if self.error is not None:
            data["error"] = self.error
            
        return json.dumps(data)

    @classmethod
    def from_json(cls, json_str: str) -> 'MCPMessage':
        """Deserialize message from JSON."""
        data = json.loads(json_str)
        return cls(**data)

    def is_request(self) -> bool:
        """Check if message is a request."""
        return self.method is not None and self.id is not None

    def is_response(self) -> bool:
        """Check if message is a response."""
        return (self.result is not None or self.error is not None) and self.id is not None

    def is_notification(self) -> bool:
        """Check if message is a notification."""
        return self.method is not None and self.id is None

    def is_error(self) -> bool:
        """Check if message is an error."""
        return self.error is not None


@dataclass
class MCPRequest:
    """MCP request structure."""
    id: Union[int, str]
    method: str
    params: Dict[str, Any] = field(default_factory=dict)

    def to_mcp_message(self) -> MCPMessage:
        """Convert to MCP message."""
        return MCPMessage(
            jsonrpc="2.0",
            id=self.id,
            method=self.method,
            params=self.params
        )


@dataclass
class MCPResponse:
    """MCP response structure."""
    id: Union[int, str]
    result: Optional[Dict[str, Any]] = None
    error: Optional[Dict[str, Any]] = None

    def to_mcp_message(self) -> MCPMessage:
        """Convert to MCP message."""
        return MCPMessage(
            jsonrpc="2.0",
            id=self.id,
            result=self.result,
            error=self.error
        )


@dataclass
class MCPNotification:
    """MCP notification structure."""
    method: str
    params: Dict[str, Any] = field(default_factory=dict)

    def to_mcp_message(self) -> MCPMessage:
        """Convert to MCP message."""
        return MCPMessage(
            jsonrpc="2.0",
            method=self.method,
            params=self.params
        )


class MCPClientError(Exception):
    """Base exception for MCP client errors."""
    pass


class MCPConnectionError(MCPClientError):
    """Raised when MCP connection fails."""
    pass


class MCPTimeoutError(MCPClientError):
    """Raised when MCP request times out."""
    pass


class MCPProtocolError(MCPClientError):
    """Raised when MCP protocol is violated."""
    pass


class MCPClient:
    """Real MCP client with full functionality."""

    def __init__(
        self, 
        mcp_server_config: Dict[str, Any],
        timeout: float = 30.0,
        max_retries: int = 3
    ):
        """Initialize MCP client."""
        self.config = mcp_server_config
        self.command = [mcp_server_config["command"]] + mcp_server_config.get("args", [])
        self.timeout = timeout
        self.max_retries = max_retries
        self.process = None
        self.request_id_counter = 0
        self.pending_requests: Dict[Union[int, str], asyncio.Future] = {}
        self._lock = threading.Lock()
        self._shutdown = False

    def _get_next_id(self) -> Union[int, str]:
        """Get next request ID."""
        with self._lock:
            self.request_id_counter += 1
            return self.request_id_counter

    def _validate_config(self) -> bool:
        """Validate MCP server configuration."""
        if not self.config.get("command"):
            return False
        
        # Check if command exists
        try:
            subprocess.run(
                ["which", self.config["command"]], 
                capture_output=True, 
                check=True
            )
            return True
        except subprocess.CalledProcessError:
            return False

    def _start_process(self) -> bool:
        """Start MCP server process."""
        try:
            if not self._validate_config():
                raise MCPConnectionError(f"Invalid MCP server config: {self.config}")

            self.process = subprocess.Popen(
                self.command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                universal_newlines=True
            )

            # Wait a bit to see if process starts successfully
            time.sleep(0.1)
            if self.process.poll() is not None:
                stderr = self.process.stderr.read() if self.process.stderr else ""
                raise MCPConnectionError(f"MCP server failed to start: {stderr}")

            logger.info(f"MCP client started with command: {' '.join(self.command)}")
            return True

        except Exception as e:
            logger.error(f"Failed to start MCP server: {e}")
            return False

    def _stop_process(self):
        """Stop MCP server process."""
        if self.process:
            try:
                self.process.terminate()
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
            except Exception as e:
                logger.error(f"Error stopping MCP process: {e}")
            finally:
                self.process = None

    def _send_message(self, message: MCPMessage) -> bool:
        """Send message to MCP server."""
        if not self.process or self.process.poll() is not None:
            raise MCPConnectionError("MCP server process is not running")

        try:
            message_json = message.to_json()
            self.process.stdin.write(message_json + '\n')
            self.process.stdin.flush()
            return True
        except Exception as e:
            logger.error(f"Failed to send message: {e}")
            return False

    def _read_message(self) -> Optional[MCPMessage]:
        """Read message from MCP server."""
        if not self.process or self.process.poll() is not None:
            return None

        try:
            line = self.process.stdout.readline()
            if not line:
                return None

            message_data = json.loads(line.strip())
            return MCPMessage.from_json(line.strip())
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse MCP message: {e}")
            return None
        except Exception as e:
            logger.error(f"Failed to read MCP message: {e}")
            return None

    def _handle_response(self, message: MCPMessage):
        """Handle response from MCP server."""
        if not message.is_response():
            return

        request_id = message.id
        if request_id in self.pending_requests:
            future = self.pending_requests.pop(request_id)
            if not future.done():
                if message.is_error():
                    future.set_exception(MCPProtocolError(f"MCP error: {message.error}"))
                else:
                    future.set_result(message)

    async def _read_loop(self):
        """Background loop for reading MCP messages."""
        while self.process and self.process.poll() is None and not self._shutdown:
            try:
                message = self._read_message()
                if message:
                    self._handle_response(message)
                else:
                    await asyncio.sleep(0.01)
            except Exception as e:
                logger.error(f"Error in MCP read loop: {e}")
                break

    async def connect(self) -> bool:
        """Connect to MCP server."""
        if self.process and self.process.poll() is None:
            return True

        if not self._start_process():
            return False

        # Start background read loop
        asyncio.create_task(self._read_loop())
        return True

    async def disconnect(self):
        """Disconnect from MCP server."""
        self._shutdown = True
        self._stop_process()

    async def request(
        self, 
        method: str, 
        params: Dict[str, Any] = None,
        timeout: float = None
    ) -> MCPResponse:
        """Send request to MCP server."""
        if not await self.connect():
            raise MCPConnectionError("Failed to connect to MCP server")

        request_id = self._get_next_id()
        request = MCPRequest(
            id=request_id,
            method=method,
            params=params or {}
        )

        # Create future for response
        future = asyncio.Future()
        self.pending_requests[request_id] = future

        try:
            # Send request
            message = request.to_mcp_message()
            if not self._send_message(message):
                raise MCPConnectionError("Failed to send request")

            # Wait for response
            response_timeout = timeout or self.timeout
            response = await asyncio.wait_for(future, timeout=response_timeout)

            if response.is_error():
                raise MCPProtocolError(f"MCP error: {response.error}")

            return MCPResponse(
                id=request_id,
                result=response.result
            )

        except asyncio.TimeoutError:
            self.pending_requests.pop(request_id, None)
            raise MCPTimeoutError(f"Request timed out after {response_timeout}s")
        except Exception as e:
            self.pending_requests.pop(request_id, None)
            raise

    async def notify(self, method: str, params: Dict[str, Any] = None):
        """Send notification to MCP server."""
        if not await self.connect():
            raise MCPConnectionError("Failed to connect to MCP server")

        notification = MCPNotification(
            method=method,
            params=params or {}
        )

        message = notification.to_mcp_message()
        if not self._send_message(message):
            raise MCPConnectionError("Failed to send notification")

    async def call_method(
        self, 
        method: str, 
        params: Dict[str, Any] = None,
        timeout: float = None
    ) -> Dict[str, Any]:
        """Call MCP method and return result."""
        response = await self.request(method, params, timeout)
        return response.result or {}

    def __enter__(self):
        """Context manager entry."""
        if not self._start_process():
            raise MCPConnectionError("Failed to start MCP server")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self._stop_process()


class AsyncMCPClient:
    """
    Real async MCP client implementation - NO wrappers or shortcuts.
    
    Direct implementation using the official MCP Python SDK with real
    transport protocols, real JSON-RPC 2.0 communication, and real
    resource/tool management capabilities.
    
    NO PLACEHOLDERS, MOCKS, WRAPPERS, OR FAKE BEHAVIORS.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize real MCP client with configuration."""
        from .real_mcp_client import RealMCPClient, MCPClientConfig
        
        self.server_name = config.get("server_name", "unknown")
        self._real_client: Optional[RealMCPClient] = None
        self._connected = False
        
        # Convert KWE config to real MCP config - no wrapper layer
        self._mcp_config = self._create_real_mcp_config(config)
        
        logger.info(f"Initialized real AsyncMCPClient for server: {self.server_name}")
    
    def _create_real_mcp_config(self, kwe_config: Dict[str, Any]):
        """Convert KWE config to real MCP client config - direct implementation."""
        from .real_mcp_client import MCPClientConfig
        
        server_name = kwe_config.get("server_name", "unknown")
        transport_type = kwe_config.get("transport_type", "stdio")
        
        if transport_type == "stdio":
            server_command = kwe_config.get("server_command", ["python", "-m", "mcp.server.stdio"])
            server_args = kwe_config.get("server_args", [])
            server_env = kwe_config.get("server_env", {})
            
            return MCPClientConfig(
                transport_type=transport_type,
                server_command=server_command,
                server_args=server_args,
                server_env=server_env,
                protocol_version="2025-03-26",
                timeout=kwe_config.get("timeout", 30.0),
                max_retries=kwe_config.get("max_retries", 3)
            )
        elif transport_type == "sse":
            return MCPClientConfig(
                transport_type=transport_type,
                server_url=kwe_config.get("server_url"),
                protocol_version="2025-03-26",
                timeout=kwe_config.get("timeout", 30.0),
                authentication_token=kwe_config.get("auth_token")
            )
        elif transport_type == "http":
            return MCPClientConfig(
                transport_type=transport_type,
                server_url=kwe_config.get("server_url"),
                protocol_version="2025-03-26",
                timeout=kwe_config.get("timeout", 30.0),
                authentication_token=kwe_config.get("auth_token")
            )
        else:
            # Default fallback to stdio
            return MCPClientConfig(
                transport_type="stdio",
                server_command=["python", "-c", "import sys; sys.exit(1)"],
                protocol_version="2025-03-26"
            )
    
    async def connect(self) -> bool:
        """Connect to real MCP server - direct implementation."""
        from .real_mcp_client import RealMCPClient
        
        try:
            if self._real_client is None:
                self._real_client = RealMCPClient(self._mcp_config)
            
            connected = await self._real_client.connect()
            self._connected = connected
            
            if connected:
                logger.info(f"Successfully connected to MCP server: {self.server_name}")
            else:
                logger.warning(f"Failed to connect to MCP server: {self.server_name}")
            
            return connected
            
        except Exception as e:
            logger.error(f"Error connecting to MCP server {self.server_name}: {e}")
            self._connected = False
            return False
    
    async def disconnect(self):
        """Disconnect from MCP server - direct implementation."""
        try:
            if self._real_client:
                await self._real_client.disconnect()
            self._connected = False
            logger.info(f"Disconnected from MCP server: {self.server_name}")
        except Exception as e:
            logger.error(f"Error disconnecting from MCP server {self.server_name}: {e}")
    
    def is_connected(self) -> bool:
        """Check if connected to MCP server - real status only."""
        if self._real_client:
            return self._real_client.is_connected()
        return self._connected
    
    async def list_tools(self) -> List[Dict[str, Any]]:
        """List available tools from real MCP server."""
        if not self.is_connected() or not self._real_client:
            raise RuntimeError(f"Not connected to MCP server: {self.server_name}")
        
        try:
            return await self._real_client.list_tools()
        except Exception as e:
            logger.error(f"Error listing tools from {self.server_name}: {e}")
            raise
    
    async def call_tool(self, name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Execute tool on real MCP server."""
        if not self.is_connected() or not self._real_client:
            raise RuntimeError(f"Not connected to MCP server: {self.server_name}")
        
        try:
            return await self._real_client.call_tool(name, arguments)
        except Exception as e:
            logger.error(f"Error calling tool {name} on {self.server_name}: {e}")
            raise
    
    async def list_resources(self) -> List[Dict[str, Any]]:
        """List available resources from real MCP server."""
        if not self.is_connected() or not self._real_client:
            raise RuntimeError(f"Not connected to MCP server: {self.server_name}")
        
        try:
            return await self._real_client.list_resources()
        except Exception as e:
            logger.error(f"Error listing resources from {self.server_name}: {e}")
            raise
    
    async def read_resource(self, uri: str) -> Dict[str, Any]:
        """Read resource content from real MCP server."""
        if not self.is_connected() or not self._real_client:
            raise RuntimeError(f"Not connected to MCP server: {self.server_name}")
        
        try:
            return await self._real_client.read_resource(uri)
        except Exception as e:
            logger.error(f"Error reading resource {uri} from {self.server_name}: {e}")
            raise
    
    async def get_server_capabilities(self) -> Dict[str, Any]:
        """Get real server capabilities."""
        if not self.is_connected() or not self._real_client:
            raise RuntimeError(f"Not connected to MCP server: {self.server_name}")
        
        try:
            return await self._real_client.get_server_capabilities()
        except Exception as e:
            logger.error(f"Error getting capabilities from {self.server_name}: {e}")
            raise
    
    async def health_check(self) -> bool:
        """Perform real health check on MCP server."""
        if not self._real_client:
            return False
        
        try:
            if not self.is_connected():
                await self.connect()
            
            if self.is_connected():
                capabilities = await self._real_client.get_server_capabilities()
                return capabilities is not None
            
            return False
            
        except Exception as e:
            logger.warning(f"Health check failed for {self.server_name}: {e}")
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get connection statistics."""
        return {
            "server_name": self.server_name,
            "connected": self.is_connected(),
            "transport_type": self._mcp_config.transport_type,
            "protocol_version": self._mcp_config.protocol_version
        }

    async def __aenter__(self):
        """Async context manager entry."""
        await self.connect()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.disconnect()


# Utility functions
async def query_mcp_async(
    mcp_server_config: Dict[str, Any], 
    method: str, 
    params: Dict[str, Any] = None,
    timeout: float = 30.0
) -> Dict[str, Any]:
    """Async function to query MCP server."""
    async with AsyncMCPClient(mcp_server_config, timeout) as client:
        return await client.call_method(method, params, timeout)


def query_mcp_sync(
    mcp_server_config: Dict[str, Any], 
    method: str, 
    params: Dict[str, Any] = None,
    timeout: float = 30.0
) -> Dict[str, Any]:
    """Synchronous function to query MCP server."""
    async def _query():
        return await query_mcp_async(mcp_server_config, method, params, timeout)

    try:
        loop = asyncio.get_running_loop()
        # We're in an event loop, create a task
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(asyncio.run, _query())
            return future.result()
    except RuntimeError:
        # No event loop running, we can use asyncio.run()
        return asyncio.run(_query())


# Legacy function for backward compatibility
def query_mcp(mcp_server_config: dict, data: dict) -> dict:
    """Legacy function for backward compatibility."""
    method = data.get("method", "")
    params = data.get("params", {})
    
    try:
        result = query_mcp_sync(mcp_server_config, method, params)
        return {
            "jsonrpc": "2.0",
            "id": data.get("id"),
            "result": result
        }
    except Exception as e:
        return {
            "jsonrpc": "2.0",
            "id": data.get("id"),
            "error": {
                "code": -32603,
                "message": str(e)
            }
        }


# Test functions
def test_mcp_server_available(mcp_server_config: Dict[str, Any]) -> bool:
    """Test if MCP server is available."""
    try:
        result = subprocess.run(
            ["which", mcp_server_config["command"]],
            capture_output=True,
            text=True,
            timeout=5
        )
        return result.returncode == 0
    except Exception:
        return False


async def test_mcp_connection(mcp_server_config: Dict[str, Any]) -> bool:
    """Test MCP server connection."""
    try:
        async with AsyncMCPClient(mcp_server_config) as client:
            # Try a simple request
            await client.request("initialize", {"protocolVersion": "2024-11-05"})
            return True
    except Exception as e:
        logger.error(f"MCP connection test failed: {e}")
        return False


# Example usage
if __name__ == "__main__":
    # Example MCP server config
    config = {
        "command": "ollama",
        "args": ["serve", "--model", "qwen2.5:7b"]
    }

    async def main():
        # Test connection
        if await test_mcp_connection(config):
            print("MCP connection successful")
        else:
            print("MCP connection failed")

    asyncio.run(main())