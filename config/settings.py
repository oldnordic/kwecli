from pathlib import Path
import os
from typing import List
try:
    from pydantic_settings import BaseSettings
    from pydantic import Field, field_validator
except ImportError:
    # Fallback for older pydantic versions
    from pydantic import BaseSettings, Field
    field_validator = lambda *args, **kwargs: lambda func: func
import yaml

class Settings(BaseSettings):
    # Config file
    config_file: Path = Field(default=Path(__file__).parent.parent / 'config.yaml', env='KWECLI_CONFIG_FILE')
    # Backend server
    backend_host: str = Field('localhost', env='KWECLI_BACKEND_HOST')
    backend_port: int = Field(8000, env='KWECLI_BACKEND_PORT')
    # Ollama settings
    ollama_base_url: str = Field('http://localhost:11434', env='KWECLI_OLLAMA_BASE_URL')
    model_chat: str = Field('llama2-mini', env='KWECLI_MODEL_CHAT')
    model_embed: str = Field('llama2-embeddings', env='KWECLI_MODEL_EMBED')
    # ACP protocol
    acp_enabled: bool = Field(False, env='KWECLI_ACP_ENABLED')
    acp_host: str = Field('127.0.0.1', env='KWECLI_ACP_HOST')
    acp_port: int = Field(8001, env='KWECLI_ACP_PORT')
    # Rate limits
    require_api_key: bool = Field(False, env='KWECLI_REQUIRE_API_KEY')
    api_keys: List[str] = Field(default_factory=list, env='KWECLI_API_KEYS')
    # Logging
    log_level: str = Field('INFO', env='KWECLI_LOG_LEVEL')
    # MCP (temporarily disabled)
    mcp_enabled: bool = Field(False, env='KWECLI_MCP_ENABLED')
    # Quality gates
    quality_gates_enabled: bool = Field(True, env='KWECLI_QUALITY_GATES_ENABLED')

    model_config = {
        'env_prefix': '',
        'case_sensitive': False,
        'extra': 'ignore'
    }

    @field_validator('api_keys', mode='before')
    @classmethod
    def split_api_keys(cls, v):
        if isinstance(v, str):
            return [k.strip() for k in v.split(',') if k.strip()]
        return v

    def validate(self) -> list[str]:
        """Validate critical settings and return list of issues."""
        issues: list[str] = []
        # Example validations
        if self.backend_port <= 0 or self.backend_port > 65535:
            issues.append(f"Invalid backend_port: {self.backend_port}")
        return issues

    def is_development_mode(self) -> bool:
        """Return True if running in development mode (debug)."""
        return getattr(self, 'debug_mode', False)

    def get_cors_config(self) -> dict[str, list[str]] | None:
        """Return CORS configuration or None if disabled."""
        origins = os.getenv('KWECLI_CORS_ORIGINS')
        if origins:
            allow_origins = [o.strip() for o in origins.split(',') if o.strip()]
        else:
            allow_origins = []
        methods = ["GET", "POST"]
        headers = ["*"]
        if allow_origins:
            return {"allow_origins": allow_origins, "allow_methods": methods, "allow_headers": headers}
        return None

    def load_yaml(self):
        try:
            data = yaml.safe_load(self.config_file.read_text()) or {}
            return data
        except Exception:
            return {}

    @classmethod
    def create(cls):
        # Load env and defaults, then override with YAML file
        base = cls()
        yaml_data = base.load_yaml()
        return cls(**{**yaml_data, **base.model_dump()})

# Singleton settings
settings = Settings.create()
