"""
Configuration Manager for KWE CLI Security Policies.

This module provides comprehensive configuration management with environment-specific
overrides, validation, and secure policy loading following enterprise patterns.
"""

import os
import yaml
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass
from datetime import datetime

from .models import SecurityPolicyConfig, PolicyMetadata, GlobalSettings, EnforcementMode
from .policy_validator import SecurityPolicyValidator


@dataclass
class ConfigPath:
    """Configuration file path specification."""
    path: str
    required: bool = False
    environment_specific: bool = False
    description: Optional[str] = None


class ConfigurationManager:
    """
    Enterprise-grade configuration manager for security policies.
    
    Handles loading, validation, merging, and environment-specific overrides
    of security policy configurations with proper error handling and logging.
    """
    
    def __init__(self, base_config_dir: Optional[str] = None):
        """Initialize configuration manager with base directory."""
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.base_config_dir = Path(base_config_dir or self._get_default_config_dir())
        self.validator = SecurityPolicyValidator()
        self.loaded_configs: Dict[str, SecurityPolicyConfig] = {}
        
        # Ensure config directory exists
        self.base_config_dir.mkdir(parents=True, exist_ok=True)
    
    def _get_default_config_dir(self) -> str:
        """Get default configuration directory."""
        # Use XDG config directory or fallback
        xdg_config = os.environ.get('XDG_CONFIG_HOME')
        if xdg_config:
            return os.path.join(xdg_config, 'kwe-cli', 'security')
        
        home = os.path.expanduser('~')
        return os.path.join(home, '.config', 'kwe-cli', 'security')
    
    def get_config_paths(self, environment: str = "development") -> List[ConfigPath]:
        """Get ordered list of configuration file paths to check."""
        paths = [
            ConfigPath(
                path=str(self.base_config_dir / "base.yaml"),
                required=False,
                description="Base security policy configuration"
            ),
            ConfigPath(
                path=str(self.base_config_dir / f"{environment}.yaml"),
                required=False,
                environment_specific=True,
                description=f"Environment-specific configuration for {environment}"
            ),
            ConfigPath(
                path=str(self.base_config_dir / "local.yaml"),
                required=False,
                description="Local development overrides (not version controlled)"
            )
        ]
        
        # Add environment variable override path
        env_config_path = os.environ.get('KWE_SECURITY_CONFIG_PATH')
        if env_config_path:
            paths.append(ConfigPath(
                path=env_config_path,
                required=True,
                description="Environment variable specified configuration"
            ))
        
        return paths
    
    def load_configuration(self, environment: str = "development") -> SecurityPolicyConfig:
        """
        Load and merge configuration from multiple sources.
        
        Args:
            environment: Target environment (development, staging, production)
            
        Returns:
            SecurityPolicyConfig: Merged and validated configuration
        """
        self.logger.info(f"Loading security configuration for environment: {environment}")
        
        # Check cache first
        cache_key = f"{environment}_{self._get_config_hash()}"
        if cache_key in self.loaded_configs:
            self.logger.debug(f"Using cached configuration for {environment}")
            return self.loaded_configs[cache_key]
        
        # Load and merge configurations
        merged_config = self._load_and_merge_configs(environment)
        
        # Validate final configuration
        validation_errors = self.validator.validate_policy_dict(merged_config)
        if validation_errors:
            error_msg = f"Configuration validation failed: {validation_errors}"
            self.logger.error(error_msg)
            raise ValueError(error_msg)
        
        # Create policy configuration object
        policy_config = self._create_policy_config(merged_config)
        
        # Cache the configuration
        self.loaded_configs[cache_key] = policy_config
        
        self.logger.info(f"Successfully loaded security configuration for {environment}")
        return policy_config
    
    def _load_and_merge_configs(self, environment: str) -> Dict[str, Any]:
        """Load and merge configurations from multiple sources."""
        merged_config = {}
        config_paths = self.get_config_paths(environment)
        
        for config_path in config_paths:
            path = Path(config_path.path)
            
            if not path.exists():
                if config_path.required:
                    raise FileNotFoundError(f"Required configuration file not found: {path}")
                self.logger.debug(f"Optional configuration file not found: {path}")
                continue
            
            try:
                with open(path, 'r') as f:
                    config_data = yaml.safe_load(f)
                
                if config_data:
                    self.logger.debug(f"Loading configuration from: {path}")
                    merged_config = self._deep_merge_dict(merged_config, config_data)
                
            except Exception as e:
                error_msg = f"Failed to load configuration from {path}: {e}"
                if config_path.required:
                    raise ValueError(error_msg)
                self.logger.warning(error_msg)
        
        # Apply environment variable overrides
        merged_config = self._apply_env_overrides(merged_config, environment)
        
        # Set default values if not present
        merged_config = self._set_defaults(merged_config, environment)
        
        return merged_config
    
    def _deep_merge_dict(self, base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """Deep merge two dictionaries."""
        merged = base.copy()
        
        for key, value in override.items():
            if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
                merged[key] = self._deep_merge_dict(merged[key], value)
            else:
                merged[key] = value
        
        return merged
    
    def _apply_env_overrides(self, config: Dict[str, Any], environment: str) -> Dict[str, Any]:
        """Apply environment variable overrides."""
        overrides = {}
        
        # Global settings overrides
        if 'KWE_ENFORCEMENT_MODE' in os.environ:
            overrides.setdefault('global_settings', {})
            overrides['global_settings']['enforcement_mode'] = os.environ['KWE_ENFORCEMENT_MODE']
        
        if 'KWE_FAIL_FAST' in os.environ:
            overrides.setdefault('global_settings', {})
            overrides['global_settings']['fail_fast'] = os.environ['KWE_FAIL_FAST'].lower() == 'true'
        
        if 'KWE_MAX_CONCURRENT_SCANS' in os.environ:
            try:
                overrides.setdefault('global_settings', {})
                overrides['global_settings']['max_concurrent_scans'] = int(os.environ['KWE_MAX_CONCURRENT_SCANS'])
            except ValueError:
                self.logger.warning("Invalid KWE_MAX_CONCURRENT_SCANS value, ignoring")
        
        # Vulnerability scanning overrides
        if 'KWE_VULN_SCAN_ENABLED' in os.environ:
            overrides.setdefault('vulnerability_scanning', {})
            overrides['vulnerability_scanning']['enabled'] = os.environ['KWE_VULN_SCAN_ENABLED'].lower() == 'true'
        
        if 'KWE_VULN_SCAN_FREQUENCY' in os.environ:
            overrides.setdefault('vulnerability_scanning', {})
            overrides['vulnerability_scanning']['frequency'] = os.environ['KWE_VULN_SCAN_FREQUENCY']
        
        # Typosquatting detection overrides
        if 'KWE_TYPO_DETECTION_ENABLED' in os.environ:
            overrides.setdefault('typosquatting_detection', {})
            overrides['typosquatting_detection']['enabled'] = os.environ['KWE_TYPO_DETECTION_ENABLED'].lower() == 'true'
        
        # Package reputation overrides
        if 'KWE_REPUTATION_ENABLED' in os.environ:
            overrides.setdefault('package_reputation', {})
            overrides['package_reputation']['enabled'] = os.environ['KWE_REPUTATION_ENABLED'].lower() == 'true'
        
        if overrides:
            self.logger.info("Applying environment variable overrides")
            config = self._deep_merge_dict(config, overrides)
        
        return config
    
    def _set_defaults(self, config: Dict[str, Any], environment: str) -> Dict[str, Any]:
        """Set default values for missing configuration sections."""
        # Set metadata defaults
        if 'metadata' not in config:
            config['metadata'] = {}
        
        metadata = config['metadata']
        metadata.setdefault('version', '1.0.0')
        metadata.setdefault('environment', environment)
        metadata.setdefault('created_by', f'kwe-cli-config-manager')
        metadata.setdefault('last_modified', datetime.now().isoformat())
        metadata.setdefault('schema_version', '1.0.0')
        metadata.setdefault('description', f'Auto-generated security policy for {environment}')
        
        # Set global settings defaults
        if 'global_settings' not in config:
            config['global_settings'] = {}
        
        global_settings = config['global_settings']
        
        # Environment-specific enforcement mode defaults
        if environment == 'production':
            global_settings.setdefault('enforcement_mode', 'enforce')
            global_settings.setdefault('fail_fast', True)
        elif environment == 'staging':
            global_settings.setdefault('enforcement_mode', 'audit')
            global_settings.setdefault('fail_fast', False)
        else:  # development
            global_settings.setdefault('enforcement_mode', 'audit')
            global_settings.setdefault('fail_fast', False)
        
        global_settings.setdefault('notification_channels', ['email'])
        global_settings.setdefault('default_timeout', '1h')
        global_settings.setdefault('max_concurrent_scans', 5)
        
        # Set default CVSS thresholds if not present
        if 'cvss_thresholds' not in config:
            config['cvss_thresholds'] = self._get_default_cvss_thresholds(environment)
        
        # Set scanning defaults
        if 'vulnerability_scanning' not in config:
            config['vulnerability_scanning'] = {
                'enabled': True,
                'frequency': 'daily',
                'timeout': '30m',
                'parallel_scans': 3,
                'retry_attempts': 2
            }
        
        # Set typosquatting defaults
        if 'typosquatting_detection' not in config:
            config['typosquatting_detection'] = {
                'enabled': environment == 'production'  # Enable by default in production
            }
        
        # Set package reputation defaults
        if 'package_reputation' not in config:
            config['package_reputation'] = {
                'enabled': True,
                'scoring_weights': {
                    'author_reputation': 0.25,
                    'package_age': 0.20,
                    'download_count': 0.15,
                    'github_metrics': 0.20,
                    'security_history': 0.20
                }
            }
        
        # Set security integrations defaults
        if 'security_integrations' not in config:
            config['security_integrations'] = {
                'osv_dev': {'enabled': True},
                'nvd': {'enabled': True},
                'github_advisory': {'enabled': True}
            }
        
        return config
    
    def _get_default_cvss_thresholds(self, environment: str) -> Dict[str, Any]:
        """Get default CVSS thresholds based on environment."""
        if environment == 'production':
            # Stricter thresholds for production
            return {
                'critical': {
                    'min_score': 9.0,
                    'max_score': 10.0,
                    'severity': 'critical',
                    'actions': [
                        {'type': 'block_build', 'immediate': True},
                        {'type': 'create_incident', 'priority': 'P0'}
                    ]
                },
                'high': {
                    'min_score': 7.0,
                    'max_score': 8.9,
                    'severity': 'high',
                    'actions': [
                        {'type': 'require_approval', 'assignee': 'security-team', 'timeout': '4h'}
                    ]
                },
                'medium': {
                    'min_score': 4.0,
                    'max_score': 6.9,
                    'severity': 'medium',
                    'actions': [
                        {'type': 'create_ticket', 'priority': 'P2', 'assignee': 'security-team'}
                    ]
                }
            }
        else:
            # More permissive for development/staging
            return {
                'critical': {
                    'min_score': 9.0,
                    'max_score': 10.0,
                    'severity': 'critical',
                    'actions': [
                        {'type': 'notify', 'channels': ['email']},
                        {'type': 'log_only'}
                    ]
                },
                'high': {
                    'min_score': 7.0,
                    'max_score': 8.9,
                    'severity': 'high',
                    'actions': [
                        {'type': 'notify', 'channels': ['email']}
                    ]
                }
            }
    
    def _create_policy_config(self, config_dict: Dict[str, Any]) -> SecurityPolicyConfig:
        """Create SecurityPolicyConfig object from dictionary."""
        # This would normally use a proper factory or builder pattern
        # For now, we'll create a simplified version that focuses on the key components
        
        from decimal import Decimal
        from .models import (
            CVSSThreshold, PolicyAction, ActionType, SeverityLevel,
            ScanningConfig, TyposquattingConfig, ReputationConfig, IntegrationConfig
        )
        
        # Create metadata
        metadata_dict = config_dict['metadata']
        metadata = PolicyMetadata(
            version=metadata_dict['version'],
            environment=metadata_dict['environment'],
            created_by=metadata_dict['created_by'],
            last_modified=datetime.fromisoformat(metadata_dict['last_modified'].replace('Z', '+00:00')),
            schema_version=metadata_dict.get('schema_version', '1.0.0'),
            description=metadata_dict.get('description')
        )
        
        # Create global settings
        global_dict = config_dict['global_settings']
        global_settings = GlobalSettings(
            enforcement_mode=EnforcementMode(global_dict['enforcement_mode']),
            fail_fast=global_dict.get('fail_fast', False),
            notification_channels=global_dict.get('notification_channels', []),
            default_timeout=global_dict.get('default_timeout', '1h'),
            max_concurrent_scans=global_dict.get('max_concurrent_scans', 5)
        )
        
        # Create CVSS thresholds
        cvss_thresholds = {}
        for severity, threshold_dict in config_dict.get('cvss_thresholds', {}).items():
            actions = []
            for action_dict in threshold_dict.get('actions', []):
                action = PolicyAction(
                    type=ActionType(action_dict['type']),
                    immediate=action_dict.get('immediate', False),
                    priority=action_dict.get('priority'),
                    assignee=action_dict.get('assignee'),
                    timeout=action_dict.get('timeout'),
                    channels=action_dict.get('channels', [])
                )
                actions.append(action)
            
            threshold = CVSSThreshold(
                min_score=Decimal(str(threshold_dict['min_score'])),
                max_score=Decimal(str(threshold_dict['max_score'])),
                actions=actions,
                severity=SeverityLevel(threshold_dict['severity'])
            )
            cvss_thresholds[severity] = threshold
        
        # Create other components with defaults
        vuln_scan_dict = config_dict.get('vulnerability_scanning', {})
        vulnerability_scanning = ScanningConfig(
            enabled=vuln_scan_dict.get('enabled', True),
            frequency=vuln_scan_dict.get('frequency', 'daily'),
            timeout=vuln_scan_dict.get('timeout', '30m'),
            parallel_scans=vuln_scan_dict.get('parallel_scans', 3),
            retry_attempts=vuln_scan_dict.get('retry_attempts', 2)
        )
        
        typo_dict = config_dict.get('typosquatting_detection', {})
        typosquatting_detection = TyposquattingConfig(
            enabled=typo_dict.get('enabled', True)
        )
        
        repo_dict = config_dict.get('package_reputation', {})
        package_reputation = ReputationConfig(
            enabled=repo_dict.get('enabled', True),
            scoring_weights=repo_dict.get('scoring_weights', {'default': 1.0})
        )
        
        security_integrations = IntegrationConfig()
        
        return SecurityPolicyConfig(
            metadata=metadata,
            global_settings=global_settings,
            cvss_thresholds=cvss_thresholds,
            vulnerability_scanning=vulnerability_scanning,
            typosquatting_detection=typosquatting_detection,
            package_reputation=package_reputation,
            security_integrations=security_integrations
        )
    
    def _get_config_hash(self) -> str:
        """Get hash of current configuration state for caching."""
        # Simple implementation - in production would hash file contents and timestamps
        import hashlib
        config_info = f"{self.base_config_dir}_{datetime.now().strftime('%Y%m%d%H')}"
        return hashlib.md5(config_info.encode()).hexdigest()[:8]
    
    def validate_configuration(self, config_path: str) -> List[str]:
        """Validate a configuration file."""
        return self.validator.validate_policy_file(config_path)
    
    def create_sample_config(self, environment: str = "development", output_path: Optional[str] = None) -> str:
        """Create a sample configuration file."""
        if not output_path:
            output_path = str(self.base_config_dir / f"sample-{environment}.yaml")
        
        sample_config = {
            'metadata': {
                'version': '1.0.0',
                'environment': environment,
                'created_by': 'kwe-cli-config-manager',
                'last_modified': datetime.now().isoformat(),
                'description': f'Sample security policy configuration for {environment}'
            },
            'global_settings': {
                'enforcement_mode': 'enforce' if environment == 'production' else 'audit',
                'fail_fast': environment == 'production',
                'notification_channels': ['email'],
                'max_concurrent_scans': 5
            },
            'cvss_thresholds': self._get_default_cvss_thresholds(environment),
            'vulnerability_scanning': {
                'enabled': True,
                'frequency': 'daily',
                'timeout': '30m'
            },
            'typosquatting_detection': {
                'enabled': environment == 'production'
            },
            'package_reputation': {
                'enabled': True,
                'scoring_weights': {
                    'author_reputation': 0.25,
                    'package_age': 0.20,
                    'download_count': 0.15,
                    'github_metrics': 0.20,
                    'security_history': 0.20
                }
            }
        }
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            yaml.dump(sample_config, f, default_flow_style=False, sort_keys=False)
        
        self.logger.info(f"Sample configuration created: {output_path}")
        return output_path
    
    def clear_cache(self):
        """Clear the configuration cache."""
        self.loaded_configs.clear()
        self.logger.info("Configuration cache cleared")