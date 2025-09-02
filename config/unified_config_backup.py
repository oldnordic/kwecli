#!/usr/bin/env python3
"""
KWE CLI Unified Configuration System with MCP Support

Complete MCP server management with environment variable overrides,
real command validation, and seamless integration.
"""

import yaml
import os
import json
import subprocess
import re
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)

@dataclass
class KWEConfiguration:
    """Unified configuration for KWE CLI with ACP bridge integration"""
    
    # === EXISTING config.yaml COMPATIBILITY ===
    openai_api_key: str = ""
    ollama_model: str = "hf.co/unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF:Q4_K_M"
    use_gpt: bool = False
    use_ollama: bool = True
    
    # === ENHANCED AI CONFIGURATION ===
    ollama_base_url: str = "http://localhost:11434"
    default_model_timeout: int = 120
    fallback_model: str = "qwen3-coder"
    model_temperature: float = 0.7
    max_tokens: int = 4000
    
    # === ACP BRIDGE CONFIGURATION ===
    acp_enabled: bool = False
    acp_port: int = 8001
    acp_protocol: str = "grpc"  # "grpc" or "zmq"
    acp_discovery_port: int = 8002
    acp_timeout: int = 30
    
    # === BACKEND SERVER CONFIGURATION ===
    backend_host: str = "127.0.0.1"  # Security: Changed from 0.0.0.0
    backend_port: int = 8000
    cors_origins: List[str] = field(default_factory=lambda: [
        "http://localhost:*",
        "http://127.0.0.1:*"
    ])
    cors_methods: List[str] = field(default_factory=lambda: ["GET", "POST"])
    api_prefix: str = "/api"
    
    # === AGENT SYSTEM CONFIGURATION ===
    agent_timeout: int = 120
    max_concurrent_agents: int = 10
    quality_gates_enabled: bool = True
    agent_retry_count: int = 3
    agent_retry_delay: float = 1.0
    
    # === LOGGING AND DEBUG ===
    log_level: str = "INFO"
    debug_mode: bool = False
    log_file: Optional[str] = None
    
    # === SECURITY SETTINGS ===
    enable_cors: bool = True
    require_api_key: bool = False
    
    # === MCP SERVER CONFIGURATION ===
    mcp_enabled: bool = True
    mcp_config_path: str = ".claude/mcp.json"
    mcp_servers: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    mcp_timeout: int = 30
    mcp_max_retries: int = 3
    
    @classmethod
    def load_from_files(cls, config_path: str = "config.yaml", 
                       dev_config_path: str = "config/development.yaml") -> 'KWEConfiguration':
        """
        Load configuration from YAML files with environment variable overrides.
        
        Priority order:
        1. Environment variables (highest priority)
        2. Development config file (if exists)
        3. Main config file
        4. Default values (lowest priority)
        """
        config_data = {}
        
        # Load from main config.yaml (backward compatibility)
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    config_data = yaml.safe_load(f) or {}
                logger.info(f"Loaded configuration from {config_path}")
            except Exception as e:
                logger.warning(f"Failed to load {config_path}: {e}")
        
        # Load development overrides if they exist
        if os.path.exists(dev_config_path):
            try:
                with open(dev_config_path, 'r') as f:
                    dev_config = yaml.safe_load(f) or {}
                config_data.update(dev_config)
                logger.info(f"Applied development overrides from {dev_config_path}")
            except Exception as e:
                logger.warning(f"Failed to load development config {dev_config_path}: {e}")
        
        # Environment variable overrides (highest priority)
        env_overrides = cls._get_env_overrides()
        config_data.update(env_overrides)
        
        if env_overrides:
            logger.info(f"Applied {len(env_overrides)} environment variable overrides")
        
        # Create configuration instance
        config = cls(**config_data)
        
        # Load MCP servers if MCP is enabled
        if config.mcp_enabled:
            config.load_mcp_servers()
        
        return config
    
    @classmethod
    def _get_env_overrides(cls) -> Dict[str, Any]:
        """Extract configuration overrides from environment variables."""
        env_mapping = {
            # AI Configuration
            'KWE_OLLAMA_MODEL': 'ollama_model',
            'KWE_OLLAMA_URL': 'ollama_base_url',
            'KWE_FALLBACK_MODEL': 'fallback_model',
            'KWE_MODEL_TEMPERATURE': ('model_temperature', float),
            'KWE_MAX_TOKENS': ('max_tokens', int),
            
            # ACP Configuration
            'KWE_ACP_ENABLED': ('acp_enabled', lambda x: x.lower() == 'true'),
            'KWE_ACP_PORT': ('acp_port', int),
            'KWE_ACP_PROTOCOL': 'acp_protocol',
            'KWE_ACP_TIMEOUT': ('acp_timeout', int),
            
            # Backend Configuration
            'KWE_BACKEND_HOST': 'backend_host',
            'KWE_BACKEND_PORT': ('backend_port', int),
            'KWE_API_PREFIX': 'api_prefix',
            
            # Agent Configuration
            'KWE_AGENT_TIMEOUT': ('agent_timeout', int),
            'KWE_MAX_CONCURRENT_AGENTS': ('max_concurrent_agents', int),
            'KWE_QUALITY_GATES': ('quality_gates_enabled', lambda x: x.lower() == 'true'),
            'KWE_AGENT_RETRY_COUNT': ('agent_retry_count', int),
            
            # Logging and Debug
            'KWE_LOG_LEVEL': 'log_level',
            'KWE_DEBUG': ('debug_mode', lambda x: x.lower() == 'true'),
            'KWE_LOG_FILE': 'log_file',
            
            # Security & MCP Configuration
            'KWE_REQUIRE_API_KEY': ('require_api_key', lambda x: x.lower() == 'true'),
            'KWE_MCP_ENABLED': ('mcp_enabled', lambda x: x.lower() == 'true'),
            'KWE_MCP_CONFIG_PATH': 'mcp_config_path',
            'KWE_MCP_TIMEOUT': ('mcp_timeout', int),
            'KWE_MCP_MAX_RETRIES': ('mcp_max_retries', int),
        }
        
        overrides = {}
        for env_var, config_field in env_mapping.items():
            env_value = os.getenv(env_var)
            if env_value is not None:
                if isinstance(config_field, tuple):
                    field_name, converter = config_field
                    try:
                        overrides[field_name] = converter(env_value)
                    except (ValueError, TypeError) as e:
                        logger.warning(f"Invalid value for {env_var}: {env_value} ({e})")
                else:
                    overrides[config_field] = env_value
        
        # Special handling for CORS origins (comma-separated list)
        cors_origins_env = os.getenv('KWE_CORS_ORIGINS')
        if cors_origins_env:
            overrides['cors_origins'] = [origin.strip() for origin in cors_origins_env.split(',')]
        
        return overrides
    
    def save_to_file(self, config_path: str = "config.yaml") -> bool:
        """Save core configuration to YAML file for backward compatibility."""
        try:
            config_dict = {
                'openai_api_key': self.openai_api_key, 'ollama_model': self.ollama_model,
                'use_gpt': self.use_gpt, 'use_ollama': self.use_ollama,
                'ollama_base_url': self.ollama_base_url, 'fallback_model': self.fallback_model,
                'backend_host': self.backend_host, 'backend_port': self.backend_port,
                'quality_gates_enabled': self.quality_gates_enabled
            }
            with open(config_path, 'w') as f:
                yaml.dump(config_dict, f, default_flow_style=False, sort_keys=True)
            logger.info(f"Configuration saved to {config_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to save configuration: {e}")
            return False
    
    def validate(self) -> List[str]:
        """Validate configuration and return list of issues."""
        issues = []
        
        # Validate ports
        if not (1000 <= self.backend_port <= 65535):
            issues.append(f"Invalid backend_port: {self.backend_port} (must be 1000-65535)")
        
        if not (1000 <= self.acp_port <= 65535):
            issues.append(f"Invalid acp_port: {self.acp_port} (must be 1000-65535)")
        
        # Validate protocols
        if self.acp_protocol not in ['grpc', 'zmq']:
            issues.append(f"Invalid acp_protocol: {self.acp_protocol} (must be 'grpc' or 'zmq')")
        
        # Validate timeouts
        if self.agent_timeout <= 0:
            issues.append(f"Invalid agent_timeout: {self.agent_timeout} (must be positive)")
        
        # Validate model configuration
        if not self.ollama_model and not self.fallback_model:
            issues.append("At least one of ollama_model or fallback_model must be specified")
        
        # Validate host binding for security
        if self.backend_host == "0.0.0.0":
            issues.append("WARNING: backend_host 0.0.0.0 allows external connections (security risk)")
        
        # Validate MCP configuration
        if self.mcp_enabled:
            mcp_issues = self.validate_mcp_servers()
            issues.extend(mcp_issues)
        
        return issues
    
    def get_effective_model(self) -> str:
        """Get the model name that will actually be used."""
        return self.ollama_model or self.fallback_model
    
    def is_development_mode(self) -> bool:
        """Check if running in development mode."""
        return self.debug_mode or self.log_level == "DEBUG"
    
    def get_cors_config(self) -> Dict[str, Any]:
        """Get CORS configuration for FastAPI."""
        return {
            "allow_origins": self.cors_origins,
            "allow_methods": self.cors_methods,
            "allow_headers": ["*"]
        } if self.enable_cors else {}
    
    def load_mcp_servers(self) -> bool:
        """
        Load MCP server configurations from the MCP config file.
        
        Returns:
            bool: True if loading was successful, False otherwise
        """
        try:
            if not os.path.exists(self.mcp_config_path):
                logger.warning(f"MCP config file not found: {self.mcp_config_path}")
                return False
            
            with open(self.mcp_config_path, 'r') as f:
                mcp_config = json.load(f)
            
            # Extract server configurations
            servers = mcp_config.get('mcpServers', {})
            
            # Process environment variable substitution
            for server_name, server_config in servers.items():
                if 'env' in server_config:
                    processed_env = {}
                    for key, value in server_config['env'].items():
                        processed_env[key] = self._substitute_env_variables(value)
                    server_config['env'] = processed_env
            
            self.mcp_servers = servers
            logger.info(f"Loaded {len(servers)} MCP server configurations")
            return True
            
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in MCP config file {self.mcp_config_path}: {e}")
            return False
        except PermissionError as e:
            logger.error(f"Permission denied reading MCP config file {self.mcp_config_path}: {e}")
            return False
        except Exception as e:
            logger.error(f"Failed to load MCP servers from {self.mcp_config_path}: {e}")
            return False
    
    def _substitute_env_variables(self, value: str) -> str:
        """
        Substitute environment variables in string values.
        
        Args:
            value: String potentially containing ${VAR_NAME} patterns
            
        Returns:
            String with environment variables substituted
        """
        if not isinstance(value, str):
            return value
        
        # Pattern to match ${VAR_NAME} or $VAR_NAME
        pattern = r'\${([^}]+)}|\$([A-Za-z_][A-Za-z0-9_]*)'
        
        def replace_var(match):
            var_name = match.group(1) or match.group(2)
            return os.getenv(var_name, match.group(0))  # Return original if not found
        
        return re.sub(pattern, replace_var, value)
    
    def get_mcp_server_config(self, server_name: str) -> Optional[Dict[str, Any]]:
        """
        Get configuration for a specific MCP server.
        
        Args:
            server_name: Name of the MCP server
            
        Returns:
            Server configuration dict or None if not found
        """
        return self.mcp_servers.get(server_name)
    
    def validate_mcp_servers(self) -> List[str]:
        """
        Validate MCP server configurations.
        
        Returns:
            List of validation issues
        """
        issues = []
        
        for server_name, server_config in self.mcp_servers.items():
            try:
                # Check required fields
                if 'command' not in server_config:
                    issues.append(f"MCP server '{server_name}': Missing command field")
                    continue
                
                command = server_config['command']
                
                # Validate command exists
                try:
                    result = subprocess.run(
                        ['which', command],
                        capture_output=True,
                        text=True,
                        timeout=5
                    )
                    
                    if result.returncode != 0:
                        issues.append(f"MCP server '{server_name}': Command '{command}' not found")
                        
                except subprocess.TimeoutExpired:
                    issues.append(f"MCP server '{server_name}': Command validation timeout")
                except subprocess.CalledProcessError:
                    issues.append(f"MCP server '{server_name}': Command '{command}' not found")
                    
            except Exception as e:
                issues.append(f"MCP server '{server_name}': Validation error - {e}")
        
        return issues
    
    def get_available_mcp_servers(self) -> List[str]:
        """
        Get list of available MCP server names.
        
        Returns:
            List of server names
        """
        return list(self.mcp_servers.keys())
    
    def is_mcp_server_available(self, server_name: str) -> bool:
        """
        Check if a specific MCP server is available and configured.
        
        Args:
            server_name: Name of the MCP server
            
        Returns:
            True if server is available and command exists
        """
        if server_name not in self.mcp_servers:
            return False
        
        server_config = self.mcp_servers[server_name]
        command = server_config.get('command')
        
        if not command:
            return False
        
        try:
            result = subprocess.run(
                ['which', command],
                capture_output=True,
                text=True,
                timeout=5
            )
            return result.returncode == 0
        except Exception:
            return False
    
    def get_mcp_connection_config(self, server_name: str) -> Optional[Dict[str, Any]]:
        """
        Get connection configuration for MCP client.
        
        Args:
            server_name: Name of the MCP server
            
        Returns:
            Connection configuration with timeout and retry settings
        """
        server_config = self.get_mcp_server_config(server_name)
        if not server_config:
            return None
        
        # Add global timeout and retry settings
        connection_config = server_config.copy()
        connection_config['timeout'] = self.mcp_timeout
        connection_config['max_retries'] = self.mcp_max_retries
        
        return connection_config


# === GLOBAL CONFIGURATION INSTANCE ===
_config_instance: Optional[KWEConfiguration] = None

def get_config(reload: bool = False) -> KWEConfiguration:
    """
    Get singleton configuration instance.
    
    Args:
        reload: If True, reload configuration from files
    
    Returns:
        KWEConfiguration instance
    """
    global _config_instance
    if _config_instance is None or reload:
        _config_instance = KWEConfiguration.load_from_files()
        
        # Validate configuration
        issues = _config_instance.validate()
        if issues:
            for issue in issues:
                logger.warning(f"Configuration issue: {issue}")
    
    return _config_instance

def reload_config() -> KWEConfiguration:
    """Reload configuration from files and return new instance."""
    return get_config(reload=True)

def get_env_template() -> str:
    """Generate .env template with all configuration variables."""
    return """# KWE CLI Configuration Environment Variables
# AI Configuration
KWE_OLLAMA_MODEL=qwen3-coder
KWE_OLLAMA_URL=http://localhost:11434
KWE_FALLBACK_MODEL=qwen3-coder
# Backend Configuration  
KWE_BACKEND_HOST=127.0.0.1
KWE_BACKEND_PORT=8000
# Agent Configuration
KWE_AGENT_TIMEOUT=120
KWE_QUALITY_GATES=true
# MCP Configuration
KWE_MCP_ENABLED=true
KWE_MCP_CONFIG_PATH=.claude/mcp.json
KWE_MCP_TIMEOUT=30
KWE_MCP_MAX_RETRIES=3
# Debug
KWE_DEBUG=false
"""


if __name__ == "__main__":
    config = get_config()
    print("=== KWE CLI Configuration ===")
    print(f"Model: {config.get_effective_model()}")
    print(f"Backend: {config.backend_host}:{config.backend_port}")
    print(f"MCP: {config.mcp_enabled} ({len(config.mcp_servers)} servers)")
    issues = config.validate()
    print(f"Status: {'✅ Valid' if not issues else '❌ Issues found'}")
    for issue in issues: print(f"  - {issue}")