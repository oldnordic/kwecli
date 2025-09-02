#!/usr/bin/env python3
"""
KWE CLI Unified Configuration with Complete MCP Support
Enhanced with environment overrides and real command validation.
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
    """Unified configuration with complete MCP integration."""
    
    openai_api_key: str = ""
    ollama_model: str = "hf.co/unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF:Q4_K_M"
    use_gpt: bool = False
    use_ollama: bool = True
    ollama_base_url: str = "http://localhost:11434"
    default_model_timeout: int = 360  # Increased for complex code generation tasks
    fallback_model: str = "qwen3-coder"
    model_temperature: float = 0.7
    max_tokens: int = 4000

    # Quality policy (mandatory)
    max_file_lines: int = 300
    modularization_required: bool = True
    disallow_placeholders: bool = True  # no mocks, stubs, placeholders, or 'pass' scaffolds
    tdd_required: bool = True
    offline_docs_required: bool = True  # research/download and store local docs; no online dependency at runtime
    # RAG/Docs cache
    rag_enabled: bool = False
    docs_cache_path: str = "docs"
    
    acp_enabled: bool = False
    acp_host: str = "127.0.0.1"
    acp_port: int = 8001
    acp_protocol: str = "grpc"
    acp_discovery_port: int = 8002
    acp_timeout: int = 30
    acp_max_retries: int = 3
    acp_retry_delay: float = 1.0
    
    backend_host: str = "127.0.0.1"
    backend_port: int = 8000
    cors_origins: List[str] = field(default_factory=lambda: ["http://localhost:*", "http://127.0.0.1:*"])
    cors_methods: List[str] = field(default_factory=lambda: ["GET", "POST"])
    api_prefix: str = "/api"
    
    agent_timeout: int = 120
    max_concurrent_agents: int = 10
    quality_gates_enabled: bool = True
    agent_retry_count: int = 3
    agent_retry_delay: float = 1.0
    
    log_level: str = "INFO"
    debug_mode: bool = False
    log_file: Optional[str] = None
    
    enable_cors: bool = True
    require_api_key: bool = False
    api_keys: List[str] = field(default_factory=list)

    # Security limits
    rate_limit_per_minute: int = 120
    max_request_size_bytes: int = 2_000_000

    # Offline policy
    offline_mode: bool = True
    
    mcp_enabled: bool = True
    mcp_config_path: str = ".claude/mcp.json"
    mcp_servers: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    mcp_timeout: int = 30
    mcp_max_retries: int = 3

    @classmethod
    def load_from_files(cls, config_path: str = "config.yaml", 
                       dev_config_path: str = "config/development.yaml") -> 'KWEConfiguration':
        """Load configuration with environment overrides and MCP integration."""
        config_data = {}
        
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    config_data = yaml.safe_load(f) or {}
                logger.info(f"Loaded configuration from {config_path}")
            except Exception as e:
                logger.warning(f"Failed to load {config_path}: {e}")
        
        if os.path.exists(dev_config_path):
            try:
                with open(dev_config_path, 'r') as f:
                    dev_config = yaml.safe_load(f) or {}
                config_data.update(dev_config)
                logger.info(f"Applied development overrides from {dev_config_path}")
            except Exception as e:
                logger.warning(f"Failed to load development config {dev_config_path}: {e}")
        
        env_overrides = cls._get_env_overrides()
        config_data.update(env_overrides)
        
        if env_overrides:
            logger.info(f"Applied {len(env_overrides)} environment variable overrides")
        
        config = cls(**config_data)
        
        if config.mcp_enabled:
            config.load_mcp_servers()
        
        return config
    
    @classmethod
    def _get_env_overrides(cls) -> Dict[str, Any]:
        """Extract configuration overrides from environment variables."""
        env_mapping = {
            'KWE_OLLAMA_MODEL': 'ollama_model',
            'KWE_OLLAMA_URL': 'ollama_base_url',
            'KWE_FALLBACK_MODEL': 'fallback_model',
            'KWE_MODEL_TEMPERATURE': ('model_temperature', float),
            'KWE_MAX_TOKENS': ('max_tokens', int),
            'KWE_ACP_ENABLED': ('acp_enabled', lambda x: x.lower() == 'true'),
            'KWE_ACP_HOST': 'acp_host',
            'KWE_ACP_PORT': ('acp_port', int),
            'KWE_ACP_PROTOCOL': 'acp_protocol',
            'KWE_ACP_TIMEOUT': ('acp_timeout', int),
            'KWE_ACP_MAX_RETRIES': ('acp_max_retries', int),
            'KWE_ACP_RETRY_DELAY': ('acp_retry_delay', float),
            'KWE_BACKEND_HOST': 'backend_host',
            'KWE_BACKEND_PORT': ('backend_port', int),
            'KWE_API_PREFIX': 'api_prefix',
            'KWE_AGENT_TIMEOUT': ('agent_timeout', int),
            'KWE_MAX_CONCURRENT_AGENTS': ('max_concurrent_agents', int),
            'KWE_QUALITY_GATES': ('quality_gates_enabled', lambda x: x.lower() == 'true'),
            'KWE_AGENT_RETRY_COUNT': ('agent_retry_count', int),
            'KWE_LOG_LEVEL': 'log_level',
            'KWE_DEBUG': ('debug_mode', lambda x: x.lower() == 'true'),
            'KWE_LOG_FILE': 'log_file',
            'KWE_REQUIRE_API_KEY': ('require_api_key', lambda x: x.lower() == 'true'),
            'KWE_API_KEYS': 'api_keys',
            'KWE_RATE_LIMIT_PER_MINUTE': ('rate_limit_per_minute', int),
            'KWE_MAX_REQUEST_SIZE_BYTES': ('max_request_size_bytes', int),
            'KWE_OFFLINE_MODE': ('offline_mode', lambda x: x.lower() == 'true'),
            'KWE_MCP_ENABLED': ('mcp_enabled', lambda x: x.lower() == 'true'),
            'KWE_MCP_CONFIG_PATH': 'mcp_config_path',
            'KWE_MCP_TIMEOUT': ('mcp_timeout', int),
            'KWE_MCP_MAX_RETRIES': ('mcp_max_retries', int),
            # Quality policy
            'KWE_MAX_FILE_LINES': ('max_file_lines', int),
            'KWE_MODULARIZATION_REQUIRED': ('modularization_required', lambda x: x.lower() == 'true'),
            'KWE_DISALLOW_PLACEHOLDERS': ('disallow_placeholders', lambda x: x.lower() == 'true'),
            'KWE_TDD_REQUIRED': ('tdd_required', lambda x: x.lower() == 'true'),
            'KWE_OFFLINE_DOCS_REQUIRED': ('offline_docs_required', lambda x: x.lower() == 'true'),
            'KWE_RAG_ENABLED': ('rag_enabled', lambda x: x.lower() == 'true'),
            'KWE_DOCS_CACHE_PATH': 'docs_cache_path',
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
        
        cors_origins_env = os.getenv('KWE_CORS_ORIGINS')
        if cors_origins_env:
            overrides['cors_origins'] = [origin.strip() for origin in cors_origins_env.split(',')]
        
        api_keys_env = os.getenv('KWE_API_KEYS')
        if api_keys_env:
            overrides['api_keys'] = [key.strip() for key in api_keys_env.split(',')]
        
        return overrides
    
    def load_mcp_servers(self) -> bool:
        """Load MCP server configurations from the MCP config file."""
        try:
            if not os.path.exists(self.mcp_config_path):
                logger.warning(f"MCP config file not found: {self.mcp_config_path}")
                return False
            
            with open(self.mcp_config_path, 'r') as f:
                mcp_config = json.load(f)
            
            servers = mcp_config.get('mcpServers', {})
            
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
        """Substitute environment variables in string values."""
        if not isinstance(value, str):
            return value
        
        pattern = r'\${([^}]+)}|\$([A-Za-z_][A-Za-z0-9_]*)'
        
        def replace_var(match):
            var_name = match.group(1) or match.group(2)
            return os.getenv(var_name, match.group(0))
        
        return re.sub(pattern, replace_var, value)
    
    def get_mcp_server_config(self, server_name: str) -> Optional[Dict[str, Any]]:
        """Get configuration for a specific MCP server."""
        return self.mcp_servers.get(server_name)
    
    def validate_mcp_servers(self) -> List[str]:
        """Validate MCP server configurations."""
        issues = []
        
        for server_name, server_config in self.mcp_servers.items():
            try:
                if 'command' not in server_config:
                    issues.append(f"MCP server '{server_name}': Missing command field")
                    continue
                
                command = server_config['command']
                
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
        """Get list of available MCP server names."""
        return list(self.mcp_servers.keys())
    
    def is_mcp_server_available(self, server_name: str) -> bool:
        """Check if a specific MCP server is available and configured."""
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
        """Get connection configuration for MCP client."""
        server_config = self.get_mcp_server_config(server_name)
        if not server_config:
            return None
        
        connection_config = server_config.copy()
        connection_config['timeout'] = self.mcp_timeout
        connection_config['max_retries'] = self.mcp_max_retries
        
        return connection_config
    
    def validate(self) -> List[str]:
        """Validate configuration and return list of issues."""
        issues = []
        
        if not (1000 <= self.backend_port <= 65535):
            issues.append(f"Invalid backend_port: {self.backend_port} (must be 1000-65535)")
        
        if not (1000 <= self.acp_port <= 65535):
            issues.append(f"Invalid acp_port: {self.acp_port} (must be 1000-65535)")
        
        if self.acp_protocol not in ['grpc', 'zmq', 'stdio']:
            issues.append(f"Invalid acp_protocol: {self.acp_protocol} (must be 'grpc', 'zmq', or 'stdio')")
        
        if self.acp_host == "0.0.0.0" and self.acp_enabled:
            issues.append("WARNING: acp_host 0.0.0.0 allows external ACP connections (security risk)")
        
        if self.acp_enabled and not (1000 <= self.acp_discovery_port <= 65535):
            issues.append(f"Invalid acp_discovery_port: {self.acp_discovery_port} (must be 1000-65535)")
        
        if self.agent_timeout <= 0:
            issues.append(f"Invalid agent_timeout: {self.agent_timeout} (must be positive)")
        
        if not self.ollama_model and not self.fallback_model:
            issues.append("At least one of ollama_model or fallback_model must be specified")
        
        if self.require_api_key and not self.api_keys:
            issues.append("API key required but no valid API keys configured")
        
        if self.backend_host == "0.0.0.0":
            issues.append("WARNING: backend_host 0.0.0.0 allows external connections (security risk)")
        
        if self.mcp_enabled:
            mcp_issues = self.validate_mcp_servers()
            issues.extend(mcp_issues)

        # Quality policy validations (advisory; enforcement occurs in tools)
        if self.max_file_lines <= 0:
            issues.append("max_file_lines must be > 0")
        if not self.modularization_required:
            issues.append("WARNING: modularization_required is disabled; large files may slip through")
        if not self.tdd_required:
            issues.append("WARNING: tdd_required is disabled; tests may be skipped")
        if not self.offline_docs_required:
            issues.append("WARNING: offline_docs_required is disabled; online lookups may occur")
        if self.rag_enabled and not self.docs_cache_path:
            issues.append("rag_enabled requires docs_cache_path")
        
        return issues
    
    def get_effective_model(self) -> str:
        """Get the model name that will actually be used."""
        return self.ollama_model or self.fallback_model
    
    def get_backend_url(self) -> str:
        """Get the complete backend URL."""
        return f"http://{self.backend_host}:{self.backend_port}"
    
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
    
    def is_valid_api_key(self, api_key: str) -> bool:
        """Check if provided API key is valid."""
        if not self.require_api_key:
            return True  # No API key required
        
        if not api_key:
            return False  # API key required but not provided
        
        return api_key in self.api_keys


_config_instance: Optional[KWEConfiguration] = None

def get_config(reload: bool = False) -> KWEConfiguration:
    """Get singleton configuration instance."""
    global _config_instance
    if _config_instance is None or reload:
        _config_instance = KWEConfiguration.load_from_files()
        
        issues = _config_instance.validate()
        if issues:
            for issue in issues:
                logger.warning(f"Configuration issue: {issue}")
    
    return _config_instance

def reload_config() -> KWEConfiguration:
    """Reload configuration from files and return new instance."""
    return get_config(reload=True)

def get_env_template() -> str:
    """Generate a .env template with all available environment variables."""
    return """# KWE CLI Configuration Environment Variables
# AI Model Configuration
KWE_OLLAMA_MODEL=qwen3-coder
KWE_OLLAMA_URL=http://localhost:11434
KWE_FALLBACK_MODEL=qwen3-coder
KWE_MODEL_TEMPERATURE=0.7
KWE_MAX_TOKENS=4000

# Backend Configuration
KWE_BACKEND_HOST=127.0.0.1
KWE_BACKEND_PORT=8000
KWE_API_PREFIX=/api
KWE_CORS_ORIGINS=http://localhost:*,http://127.0.0.1:*

# Agent System Configuration
KWE_AGENT_TIMEOUT=120
KWE_MAX_CONCURRENT_AGENTS=10
KWE_QUALITY_GATES=true
KWE_AGENT_RETRY_COUNT=3

# MCP Server Configuration
KWE_MCP_ENABLED=true
KWE_MCP_CONFIG_PATH=.claude/mcp.json
KWE_MCP_TIMEOUT=30
KWE_MCP_MAX_RETRIES=3

# Debug and Logging
KWE_DEBUG=false
KWE_LOG_LEVEL=INFO

# API Keys (comma-separated)
KWE_API_KEYS=your-api-key-1,your-api-key-2

# Quality Policy (mandatory)
KWE_MAX_FILE_LINES=300
KWE_MODULARIZATION_REQUIRED=true
KWE_DISALLOW_PLACEHOLDERS=true
KWE_TDD_REQUIRED=true
KWE_OFFLINE_DOCS_REQUIRED=true
KWE_RAG_ENABLED=false
KWE_DOCS_CACHE_PATH=docs
"""


if __name__ == "__main__":
    config = get_config()
    print("=== KWE CLI Configuration Test ===")
    print(f"Model: {config.get_effective_model()}")
    print(f"Backend: {config.backend_host}:{config.backend_port}")
    print(f"MCP: {config.mcp_enabled} ({len(config.mcp_servers)} servers)")
    issues = config.validate()
    print(f"Status: {'✅ Valid' if not issues else '❌ Issues found'}")
    for issue in issues: print(f"  - {issue}")
