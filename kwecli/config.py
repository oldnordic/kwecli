#!/usr/bin/env python3
"""
KWECLI Configuration Module
===========================

Modular configuration management using focused loader modules.
Main coordination interface following ≤300 lines constraint.

Features:
- Coordinated configuration loading from multiple sources
- Configuration validation and caching
- LTMC integration for configuration tracking
- SQLite persistence support

File: kwecli/config.py
Purpose: Configuration management coordinator
"""

import sys
import logging
from typing import Dict, Any, Optional, List
from pathlib import Path
from dataclasses import dataclass
from datetime import datetime

# Add project root for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from bridge.ltmc_native import get_ltmc_native
from .config_loaders import ConfigLoaders

logger = logging.getLogger(__name__)


@dataclass
class KWECLIConfig:
    """KWECLI Configuration dataclass with sensible defaults."""
    
    # Project settings
    project_path: str = "."
    project_name: str = "kwecli_project"
    
    # LTMC settings
    ltmc_data_dir: str = "/home/feanor/Projects/Data"
    ltmc_enabled: bool = True
    
    # AI/Model settings
    ollama_model: str = "qwen3-coder"
    ollama_base_url: str = "http://localhost:11434"
    model_temperature: float = 0.7
    max_tokens: int = 4000
    
    # Quality policy (mandatory per architecture)
    max_file_lines: int = 300
    modularization_required: bool = True
    disallow_placeholders: bool = True
    tdd_required: bool = True
    
    # TUI settings
    tui_enabled: bool = True
    tui_theme: str = "dark"
    
    # CLI settings
    cli_fallback_enabled: bool = True
    verbose_logging: bool = False
    
    # Logging
    log_level: str = "INFO"
    log_file: Optional[str] = None
    
    # Database settings (for LTMC fallback)
    config_db_path: str = "kwecli_config.db"
    
    # Session settings
    session_timeout: int = 3600  # 1 hour
    auto_save_session: bool = True


class ConfigManager:
    """Configuration manager coordinating multiple loading sources."""
    
    def __init__(self, project_path: str = "."):
        """Initialize configuration manager."""
        self.project_path = Path(project_path).resolve()
        self.config_cache = {}
        self.ltmc = None
        
        # Initialize loaders
        self.loaders = ConfigLoaders(self.project_path)
        
        # Initialize LTMC connection
        self._init_ltmc()
    
    def _init_ltmc(self):
        """Initialize LTMC connection for configuration storage."""
        try:
            self.ltmc = get_ltmc_native()
            logger.info("LTMC connected for configuration management")
        except Exception as e:
            logger.warning(f"LTMC not available for config: {e}")
            self.ltmc = None
    
    def load_config(self) -> KWECLIConfig:
        """Load configuration from all sources with priority order."""
        try:
            # Start with defaults
            config_data = {}
            
            # 1. Load from YAML files (lowest priority)
            yaml_data = self.loaders.load_yaml_config()
            config_data.update(yaml_data)
            
            # 2. Load from .env file (higher priority)
            env_data = self.loaders.load_env_config()
            config_data.update(env_data)
            
            # 3. Load from environment variables (highest priority)
            env_vars = self.loaders.load_environment_variables()
            config_data.update(env_vars)
            
            # 4. Load from SQLite fallback if needed
            db_data = self.loaders.load_db_config()
            for key, value in db_data.items():
                if key not in config_data:  # Only use DB values if not set elsewhere
                    config_data[key] = value
            
            # Create config object
            config = KWECLIConfig(**{k: v for k, v in config_data.items() 
                                   if hasattr(KWECLIConfig, k)})
            
            # Save successful config to LTMC
            if self.ltmc:
                self._save_config_to_ltmc(config_data)
            
            # Cache the config
            self.config_cache = config_data
            
            return config
            
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            return KWECLIConfig()  # Return defaults on error
    
    
    def _save_config_to_ltmc(self, config_data: Dict[str, Any]):
        """Save configuration to LTMC for tracking."""
        try:
            self.ltmc.save_thought(
                kind="config",
                content=f"KWECLI configuration loaded: {len(config_data)} settings",
                metadata={
                    "project_path": str(self.project_path),
                    "config_sources": ["yaml", "env", "environment", "sqlite"],
                    "settings_count": len(config_data),
                    "timestamp": datetime.now().isoformat()
                }
            )
        except Exception as e:
            logger.debug(f"Failed to save config to LTMC: {e}")
    
    def save_config(self, config: KWECLIConfig) -> bool:
        """Save configuration using loaders module."""
        return self.loaders.save_db_config(config.__dict__)
    
    def validate_config(self, config: KWECLIConfig) -> List[str]:
        """Validate configuration and return list of issues."""
        issues = []
        
        # Required settings validation
        if not config.project_path:
            issues.append("project_path is required")
        
        if config.max_file_lines <= 0:
            issues.append("max_file_lines must be positive")
        
        if config.max_file_lines > 1000:
            issues.append(f"max_file_lines ({config.max_file_lines}) exceeds recommended maximum (1000)")
        
        # Path validation
        project_path = Path(config.project_path)
        if not project_path.exists():
            issues.append(f"project_path does not exist: {config.project_path}")
        
        ltmc_data_dir = Path(config.ltmc_data_dir)
        if config.ltmc_enabled and not ltmc_data_dir.exists():
            issues.append(f"LTMC data directory does not exist: {config.ltmc_data_dir}")
        
        # Model validation
        if not config.ollama_model:
            issues.append("ollama_model is required")
        
        # Log level validation
        valid_log_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        if config.log_level not in valid_log_levels:
            issues.append(f"Invalid log_level: {config.log_level} (must be one of {valid_log_levels})")
        
        return issues


# Global configuration manager instance
_config_manager = None

def get_config_manager(project_path: str = ".") -> ConfigManager:
    """Get or create global configuration manager instance."""
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager(project_path)
    return _config_manager

def load_config(project_path: str = ".") -> KWECLIConfig:
    """Load KWECLI configuration from all sources."""
    manager = get_config_manager(project_path)
    return manager.load_config()


# Test functionality if run directly
if __name__ == "__main__":
    print("🧪 Testing KWECLI Configuration Manager...")
    
    # Test configuration loading
    config = load_config(".")
    print(f"✅ Configuration loaded:")
    print(f"  Project: {config.project_name} at {config.project_path}")
    print(f"  LTMC: {'enabled' if config.ltmc_enabled else 'disabled'} (data: {config.ltmc_data_dir})")
    print(f"  Model: {config.ollama_model}")
    print(f"  Quality: max_file_lines={config.max_file_lines}, TDD={config.tdd_required}")
    
    # Test validation
    manager = get_config_manager(".")
    issues = manager.validate_config(config)
    print(f"✅ Validation: {'passed' if not issues else 'failed'}")
    for issue in issues:
        print(f"  ⚠️  {issue}")
    
    # Test save
    save_ok = manager.save_config(config)
    print(f"✅ Save to SQLite: {'success' if save_ok else 'failed'}")
    
    print("✅ KWECLI Configuration Manager test complete")