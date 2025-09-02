"""
MCP Client Configuration.

Configuration management for KWE CLI's MCP client connections,
supporting multiple server types with authentication and transport options.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from enum import Enum

import yaml


class TransportType(str, Enum):
    """MCP transport types."""
    HTTP = "http"
    STDIO = "stdio"
    WEBSOCKET = "websocket"


class AuthType(str, Enum):
    """Authentication types for MCP servers."""
    NONE = "none"
    API_KEY = "api_key"
    OAUTH2 = "oauth2"
    BEARER_TOKEN = "bearer_token"


@dataclass
class OAuth2Config:
    """OAuth 2.1 configuration for MCP server authentication."""
    client_id: str
    client_secret: Optional[str] = None
    authorization_url: Optional[str] = None
    token_url: Optional[str] = None
    scopes: List[str] = field(default_factory=list)
    redirect_uri: str = "http://localhost:3000/callback"
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "OAuth2Config":
        """Create OAuth config from dictionary."""
        return cls(
            client_id=data["client_id"],
            client_secret=data.get("client_secret"),
            authorization_url=data.get("authorization_url"),
            token_url=data.get("token_url"),
            scopes=data.get("scopes", []),
            redirect_uri=data.get("redirect_uri", "http://localhost:3000/callback")
        )


@dataclass
class MCPServerConfig:
    """Configuration for individual MCP server."""
    name: str
    transport: TransportType
    
    # HTTP transport settings
    url: Optional[str] = None
    
    # STDIO transport settings  
    command: Optional[str] = None
    args: List[str] = field(default_factory=list)
    env: Dict[str, str] = field(default_factory=dict)
    working_directory: Optional[str] = None
    
    # Authentication settings
    auth_type: AuthType = AuthType.NONE
    api_key: Optional[str] = None
    bearer_token: Optional[str] = None
    oauth2: Optional[OAuth2Config] = None
    
    # Connection settings
    timeout: int = 30
    retry_attempts: int = 3
    retry_delay: float = 1.0
    retry_backoff_factor: float = 2.0
    max_retry_delay: float = 60.0
    
    # Health check settings
    health_check_enabled: bool = True
    health_check_interval: int = 60
    
    # Tool settings
    tool_timeout: int = 120
    max_concurrent_tools: int = 5
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        self._validate_configuration()
    
    def _validate_configuration(self):
        """Validate server configuration."""
        if self.transport == TransportType.HTTP and not self.url:
            raise ValueError(f"HTTP transport requires 'url' for server '{self.name}'")
            
        if self.transport == TransportType.STDIO and not self.command:
            raise ValueError(f"STDIO transport requires 'command' for server '{self.name}'")
            
        if self.auth_type == AuthType.API_KEY and not self.api_key:
            raise ValueError(f"API key authentication requires 'api_key' for server '{self.name}'")
            
        if self.auth_type == AuthType.BEARER_TOKEN and not self.bearer_token:
            raise ValueError(f"Bearer token authentication requires 'bearer_token' for server '{self.name}'")
            
        if self.auth_type == AuthType.OAUTH2 and not self.oauth2:
            raise ValueError(f"OAuth2 authentication requires 'oauth2' config for server '{self.name}'")
    
    @classmethod
    def from_dict(cls, name: str, data: Dict[str, Any]) -> "MCPServerConfig":
        """Create server config from dictionary."""
        # Handle OAuth2 config first
        oauth2_config = None
        auth_type = AuthType(data.get("auth_type", "none"))
        if auth_type == AuthType.OAUTH2 and "oauth2" in data:
            oauth2_config = OAuth2Config.from_dict(data["oauth2"])
        
        config = cls(
            name=name,
            transport=TransportType(data.get("transport", "http")),
            url=data.get("url"),
            command=data.get("command"),
            args=data.get("args", []),
            env=data.get("env", {}),
            working_directory=data.get("working_directory"),
            auth_type=auth_type,
            api_key=data.get("api_key"),
            bearer_token=data.get("bearer_token"),
            oauth2=oauth2_config,
            timeout=data.get("timeout", 30),
            retry_attempts=data.get("retry_attempts", 3),
            retry_delay=data.get("retry_delay", 1.0),
            retry_backoff_factor=data.get("retry_backoff_factor", 2.0),
            max_retry_delay=data.get("max_retry_delay", 60.0),
            health_check_enabled=data.get("health_check_enabled", True),
            health_check_interval=data.get("health_check_interval", 60),
            tool_timeout=data.get("tool_timeout", 120),
            max_concurrent_tools=data.get("max_concurrent_tools", 5)
        )
            
        return config
    
    def resolve_environment_variables(self):
        """Resolve environment variables in configuration values."""
        if self.api_key and self.api_key.startswith("${") and self.api_key.endswith("}"):
            env_var = self.api_key[2:-1]
            self.api_key = os.getenv(env_var)
            
        if self.bearer_token and self.bearer_token.startswith("${") and self.bearer_token.endswith("}"):
            env_var = self.bearer_token[2:-1]
            self.bearer_token = os.getenv(env_var)
            
        # Resolve environment variables in env dict
        for key, value in self.env.items():
            if isinstance(value, str) and value.startswith("${") and value.endswith("}"):
                env_var = value[2:-1]
                self.env[key] = os.getenv(env_var, value)


@dataclass
class ConnectionPoolConfig:
    """Configuration for MCP connection pooling."""
    max_connections: int = 10
    max_idle_connections: int = 5
    max_idle_time_seconds: int = 300
    health_check_interval_seconds: int = 60
    connection_timeout_seconds: int = 30


@dataclass
class MonitoringConfig:
    """Configuration for MCP client monitoring."""
    metrics_enabled: bool = True
    prometheus_enabled: bool = False
    prometheus_port: int = 8080
    log_level: str = "INFO"
    log_requests: bool = False
    log_responses: bool = False
    trace_enabled: bool = False
    
    def should_log_requests(self) -> bool:
        """Check if request logging is enabled."""
        return self.log_requests and self.log_level in ["DEBUG", "TRACE"]
    
    def should_log_responses(self) -> bool:
        """Check if response logging is enabled."""
        return self.log_responses and self.log_level in ["DEBUG", "TRACE"]


@dataclass
class SecurityConfig:
    """Security configuration for MCP client."""
    verify_ssl: bool = True
    ca_cert_file: Optional[str] = None
    client_cert_file: Optional[str] = None
    client_key_file: Optional[str] = None
    allowed_hosts: List[str] = field(default_factory=list)
    token_storage_path: str = "~/.kwe/mcp_tokens"
    token_encryption_enabled: bool = True


@dataclass
class MCPClientConfig:
    """Main configuration for KWE CLI MCP client."""
    servers: Dict[str, MCPServerConfig] = field(default_factory=dict)
    connection_pool: ConnectionPoolConfig = field(default_factory=ConnectionPoolConfig)
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)
    security: SecurityConfig = field(default_factory=SecurityConfig)
    
    # Client information
    client_name: str = "KWE CLI"
    client_version: str = "1.0.0"
    
    # Global defaults
    default_timeout: int = 30
    default_retry_attempts: int = 3
    
    @classmethod
    def from_file(cls, config_file: Union[str, Path]) -> "MCPClientConfig":
        """Load configuration from YAML file."""
        config_path = Path(config_file).expanduser()
        
        if not config_path.exists():
            raise FileNotFoundError(f"MCP configuration file not found: {config_path}")
        
        try:
            with open(config_path, 'r') as f:
                data = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML in configuration file: {e}")
        
        return cls.from_dict(data)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MCPClientConfig":
        """Create configuration from dictionary."""
        config = cls(
            client_name=data.get("client_name", "KWE CLI"),
            client_version=data.get("client_version", "1.0.0"),
            default_timeout=data.get("default_timeout", 30),
            default_retry_attempts=data.get("default_retry_attempts", 3)
        )
        
        # Parse server configurations
        servers_data = data.get("servers", {})
        for server_name, server_data in servers_data.items():
            server_config = MCPServerConfig.from_dict(server_name, server_data)
            server_config.resolve_environment_variables()
            config.servers[server_name] = server_config
        
        # Parse connection pool configuration
        if "connection_pool" in data:
            pool_data = data["connection_pool"]
            config.connection_pool = ConnectionPoolConfig(
                max_connections=pool_data.get("max_connections", 10),
                max_idle_connections=pool_data.get("max_idle_connections", 5),
                max_idle_time_seconds=pool_data.get("max_idle_time_seconds", 300),
                health_check_interval_seconds=pool_data.get("health_check_interval_seconds", 60),
                connection_timeout_seconds=pool_data.get("connection_timeout_seconds", 30)
            )
        
        # Parse monitoring configuration
        if "monitoring" in data:
            monitoring_data = data["monitoring"]
            config.monitoring = MonitoringConfig(
                metrics_enabled=monitoring_data.get("metrics_enabled", True),
                prometheus_enabled=monitoring_data.get("prometheus_enabled", False),
                prometheus_port=monitoring_data.get("prometheus_port", 8080),
                log_level=monitoring_data.get("log_level", "INFO"),
                log_requests=monitoring_data.get("log_requests", False),
                log_responses=monitoring_data.get("log_responses", False),
                trace_enabled=monitoring_data.get("trace_enabled", False)
            )
        
        # Parse security configuration
        if "security" in data:
            security_data = data["security"]
            config.security = SecurityConfig(
                verify_ssl=security_data.get("verify_ssl", True),
                ca_cert_file=security_data.get("ca_cert_file"),
                client_cert_file=security_data.get("client_cert_file"),
                client_key_file=security_data.get("client_key_file"),
                allowed_hosts=security_data.get("allowed_hosts", []),
                token_storage_path=security_data.get("token_storage_path", "~/.kwe/mcp_tokens"),
                token_encryption_enabled=security_data.get("token_encryption_enabled", True)
            )
        
        return config
    
    def validate(self):
        """Validate entire configuration."""
        if not self.servers:
            raise ValueError("At least one MCP server must be configured")
        
        # Validate each server configuration
        for server_name, server_config in self.servers.items():
            try:
                server_config._validate_configuration()
            except ValueError as e:
                raise ValueError(f"Server '{server_name}' configuration error: {e}")
        
        # Validate security settings
        if self.security.client_cert_file and not self.security.client_key_file:
            raise ValueError("Client key file required when client certificate is specified")
        
        if self.security.client_key_file and not self.security.client_cert_file:
            raise ValueError("Client certificate file required when client key is specified")
    
    def get_server_config(self, server_name: str) -> Optional[MCPServerConfig]:
        """Get configuration for specific server."""
        return self.servers.get(server_name)
    
    def list_server_names(self) -> List[str]:
        """Get list of configured server names."""
        return list(self.servers.keys())
    
    def get_servers_by_transport(self, transport: TransportType) -> List[MCPServerConfig]:
        """Get servers using specific transport type."""
        return [config for config in self.servers.values() if config.transport == transport]
    
    def get_http_servers(self) -> List[MCPServerConfig]:
        """Get HTTP-based servers."""
        return self.get_servers_by_transport(TransportType.HTTP)
    
    def get_stdio_servers(self) -> List[MCPServerConfig]:
        """Get STDIO-based servers."""
        return self.get_servers_by_transport(TransportType.STDIO)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        data = {
            "client_name": self.client_name,
            "client_version": self.client_version,
            "default_timeout": self.default_timeout,
            "default_retry_attempts": self.default_retry_attempts,
            "servers": {},
            "connection_pool": {
                "max_connections": self.connection_pool.max_connections,
                "max_idle_connections": self.connection_pool.max_idle_connections,
                "max_idle_time_seconds": self.connection_pool.max_idle_time_seconds,
                "health_check_interval_seconds": self.connection_pool.health_check_interval_seconds,
                "connection_timeout_seconds": self.connection_pool.connection_timeout_seconds
            },
            "monitoring": {
                "metrics_enabled": self.monitoring.metrics_enabled,
                "log_level": self.monitoring.log_level,
                "log_requests": self.monitoring.log_requests,
                "log_responses": self.monitoring.log_responses
            },
            "security": {
                "verify_ssl": self.security.verify_ssl,
                "token_encryption_enabled": self.security.token_encryption_enabled
            }
        }
        
        # Convert server configs (simplified - sensitive data should be redacted)
        for name, server_config in self.servers.items():
            server_data = {
                "transport": server_config.transport.value,
                "timeout": server_config.timeout,
                "retry_attempts": server_config.retry_attempts,
                "auth_type": server_config.auth_type.value
            }
            
            if server_config.transport == TransportType.HTTP:
                server_data["url"] = server_config.url
            elif server_config.transport == TransportType.STDIO:
                server_data["command"] = server_config.command
                server_data["args"] = server_config.args
            
            # Redact sensitive authentication data
            if server_config.auth_type in [AuthType.API_KEY, AuthType.BEARER_TOKEN]:
                server_data["auth_configured"] = True
            
            data["servers"][name] = server_data
        
        return data
    
    def save_to_file(self, config_file: Union[str, Path]):
        """Save configuration to YAML file."""
        config_path = Path(config_file).expanduser()
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        data = self.to_dict()
        
        with open(config_path, 'w') as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=True)


def load_default_config() -> MCPClientConfig:
    """Load default MCP client configuration."""
    default_config_paths = [
        Path.home() / ".kwe" / "mcp_config.yaml",
        Path.home() / ".config" / "kwe" / "mcp_config.yaml",
        Path("./mcp_config.yaml"),
        Path("./config/mcp_config.yaml")
    ]
    
    for config_path in default_config_paths:
        if config_path.exists():
            return MCPClientConfig.from_file(config_path)
    
    # Return minimal default configuration
    return MCPClientConfig()


def create_example_config(config_file: Union[str, Path]):
    """Create an example MCP configuration file."""
    example_config = {
        "client_name": "KWE CLI",
        "client_version": "1.0.0",
        "default_timeout": 30,
        "default_retry_attempts": 3,
        
        "servers": {
            "claude_code": {
                "transport": "http",
                "url": "https://api.anthropic.com/mcp",
                "auth_type": "oauth2",
                "oauth2": {
                    "client_id": "${CLAUDE_CLIENT_ID}",
                    "client_secret": "${CLAUDE_CLIENT_SECRET}",
                    "scopes": ["user", "tools", "resources"]
                },
                "timeout": 60,
                "retry_attempts": 3
            },
            
            "local_filesystem": {
                "transport": "stdio",
                "command": "python",
                "args": ["-m", "mcp_server_filesystem", "/workspace"],
                "timeout": 30
            }
        },
        
        "connection_pool": {
            "max_connections": 10,
            "max_idle_connections": 5,
            "max_idle_time_seconds": 300
        },
        
        "monitoring": {
            "metrics_enabled": True,
            "log_level": "INFO",
            "log_requests": False,
            "log_responses": False
        },
        
        "security": {
            "verify_ssl": True,
            "token_encryption_enabled": True
        }
    }
    
    config_path = Path(config_file).expanduser()
    config_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(config_path, 'w') as f:
        yaml.dump(example_config, f, default_flow_style=False, sort_keys=False)