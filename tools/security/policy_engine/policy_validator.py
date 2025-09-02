"""
Security Policy Validator for YAML-based policy configuration.

Provides comprehensive validation for security policy configurations
using JSON Schema validation with enterprise security patterns.
"""

import yaml
import jsonschema
import logging
from typing import Dict, Any, List, Optional, Union
from pathlib import Path
from decimal import Decimal

from .models import (
    SecurityPolicyConfig,
    PolicyMetadata,
    GlobalSettings,
    CVSSThreshold,
    ActionType,
    SeverityLevel,
    EnforcementMode,
    NotificationChannel
)


class ValidationError(Exception):
    """Raised when policy validation fails."""
    pass


class SecurityPolicyValidator:
    """
    Validates security policy configurations against schema and business rules.
    
    Provides both JSON Schema validation and custom business logic validation
    following enterprise security patterns and NIST CSF 2.0 guidelines.
    """
    
    def __init__(self, schema_path: Optional[str] = None):
        """Initialize validator with optional custom schema."""
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.schema = self._load_default_schema() if schema_path is None else self._load_schema(schema_path)
    
    def _load_default_schema(self) -> Dict[str, Any]:
        """Load default JSON Schema for security policy validation."""
        return {
            "$schema": "http://json-schema.org/draft-07/schema#",
            "title": "KWE CLI Security Policy Configuration",
            "type": "object",
            "properties": {
                "metadata": {
                    "type": "object",
                    "properties": {
                        "version": {
                            "type": "string",
                            "pattern": "^[0-9]+\\.[0-9]+\\.[0-9]+$"
                        },
                        "environment": {
                            "type": "string",
                            "enum": ["development", "staging", "production"]
                        },
                        "created_by": {"type": "string"},
                        "last_modified": {"type": "string", "format": "date-time"},
                        "schema_version": {"type": "string", "default": "1.0.0"},
                        "description": {"type": "string"}
                    },
                    "required": ["version", "environment", "created_by", "last_modified"]
                },
                "global_settings": {
                    "type": "object",
                    "properties": {
                        "enforcement_mode": {
                            "type": "string",
                            "enum": ["audit", "enforce", "disabled"]
                        },
                        "fail_fast": {"type": "boolean"},
                        "notification_channels": {
                            "type": "array",
                            "items": {"type": "string"}
                        },
                        "default_timeout": {"type": "string"},
                        "max_concurrent_scans": {
                            "type": "integer",
                            "minimum": 1,
                            "maximum": 50
                        }
                    },
                    "required": ["enforcement_mode"]
                },
                "cvss_thresholds": {
                    "type": "object",
                    "patternProperties": {
                        "^(critical|high|medium|low|none)$": {
                            "type": "object",
                            "properties": {
                                "min_score": {
                                    "type": "number",
                                    "minimum": 0.0,
                                    "maximum": 10.0
                                },
                                "max_score": {
                                    "type": "number",
                                    "minimum": 0.0,
                                    "maximum": 10.0
                                },
                                "severity": {
                                    "type": "string",
                                    "enum": ["critical", "high", "medium", "low", "none"]
                                },
                                "actions": {
                                    "type": "array",
                                    "items": {
                                        "type": "object",
                                        "properties": {
                                            "type": {
                                                "type": "string",
                                                "enum": ["block_build", "require_approval", "create_ticket", "create_incident", "notify", "log_only", "auto_resolve"]
                                            },
                                            "immediate": {"type": "boolean"},
                                            "priority": {"type": "string"},
                                            "assignee": {"type": "string"},
                                            "timeout": {"type": "string"},
                                            "channels": {
                                                "type": "array",
                                                "items": {
                                                    "type": "string",
                                                    "enum": ["email", "slack", "teams", "webhook", "pagerduty"]
                                                }
                                            },
                                            "metadata": {"type": "object"}
                                        },
                                        "required": ["type"]
                                    }
                                }
                            },
                            "required": ["min_score", "max_score", "severity", "actions"]
                        }
                    }
                },
                "vulnerability_scanning": {
                    "type": "object",
                    "properties": {
                        "enabled": {"type": "boolean"},
                        "frequency": {"type": "string"},
                        "timeout": {"type": "string"},
                        "parallel_scans": {"type": "integer", "minimum": 1},
                        "retry_attempts": {"type": "integer", "minimum": 0},
                        "ecosystems": {"type": "object"},
                        "scanners": {"type": "object"},
                        "exclusions": {"type": "object"}
                    }
                },
                "typosquatting_detection": {
                    "type": "object",
                    "properties": {
                        "enabled": {"type": "boolean"},
                        "algorithms": {"type": "object"},
                        "ecosystem_rules": {"type": "object"},
                        "thresholds": {"type": "object"},
                        "allowlists": {"type": "object"},
                        "blocklists": {"type": "object"}
                    }
                },
                "package_reputation": {
                    "type": "object",
                    "properties": {
                        "enabled": {"type": "boolean"},
                        "scoring_weights": {
                            "type": "object",
                            "patternProperties": {
                                "^[a-zA-Z_]+$": {
                                    "type": "number",
                                    "minimum": 0.0,
                                    "maximum": 1.0
                                }
                            }
                        },
                        "thresholds": {"type": "object"},
                        "author_metrics": {"type": "object"},
                        "package_metrics": {"type": "object"},
                        "github_metrics": {"type": "object"},
                        "security_analysis": {"type": "object"}
                    }
                },
                "security_integrations": {
                    "type": "object",
                    "properties": {
                        "osv_dev": {"type": "object"},
                        "nvd": {"type": "object"},
                        "github_advisory": {"type": "object"},
                        "custom_feeds": {"type": "array"},
                        "sync_settings": {"type": "object"}
                    }
                }
            },
            "required": ["metadata", "global_settings"]
        }
    
    def _load_schema(self, schema_path: str) -> Dict[str, Any]:
        """Load JSON Schema from file."""
        try:
            with open(schema_path, 'r') as f:
                if schema_path.endswith('.yaml') or schema_path.endswith('.yml'):
                    return yaml.safe_load(f)
                else:
                    return yaml.safe_load(f)  # Assuming YAML format
        except Exception as e:
            raise ValidationError(f"Failed to load schema from {schema_path}: {e}")
    
    def validate_policy_file(self, policy_path: str) -> List[str]:
        """
        Validate security policy from YAML file.
        
        Args:
            policy_path: Path to YAML policy file
            
        Returns:
            List of validation error messages (empty if valid)
        """
        try:
            with open(policy_path, 'r') as f:
                policy_data = yaml.safe_load(f)
        except Exception as e:
            return [f"Failed to load policy file {policy_path}: {e}"]
        
        return self.validate_policy_dict(policy_data)
    
    def validate_policy_dict(self, policy_data: Dict[str, Any]) -> List[str]:
        """
        Validate security policy from dictionary.
        
        Args:
            policy_data: Policy configuration as dictionary
            
        Returns:
            List of validation error messages (empty if valid)
        """
        errors = []
        
        # JSON Schema validation
        try:
            jsonschema.validate(policy_data, self.schema)
        except jsonschema.ValidationError as e:
            errors.append(f"Schema validation error: {e.message}")
        except jsonschema.SchemaError as e:
            errors.append(f"Schema error: {e.message}")
        
        # Custom business logic validation
        errors.extend(self._validate_business_rules(policy_data))
        
        return errors
    
    def validate_policy_object(self, policy_config: SecurityPolicyConfig) -> List[str]:
        """
        Validate SecurityPolicyConfig object.
        
        Args:
            policy_config: SecurityPolicyConfig instance
            
        Returns:
            List of validation error messages (empty if valid)
        """
        errors = []
        
        # Use the built-in validation from the config object
        errors.extend(policy_config.validate())
        
        # Additional validation specific to the validator
        errors.extend(self._validate_policy_object_rules(policy_config))
        
        return errors
    
    def _validate_business_rules(self, policy_data: Dict[str, Any]) -> List[str]:
        """
        Validate custom business rules for policy configuration.
        
        Args:
            policy_data: Policy configuration dictionary
            
        Returns:
            List of validation error messages
        """
        errors = []
        
        # Validate CVSS threshold ordering
        if 'cvss_thresholds' in policy_data:
            errors.extend(self._validate_cvss_thresholds(policy_data['cvss_thresholds']))
        
        # Validate package reputation scoring weights
        if 'package_reputation' in policy_data:
            errors.extend(self._validate_reputation_config(policy_data['package_reputation']))
        
        # Validate scanning configuration
        if 'vulnerability_scanning' in policy_data:
            errors.extend(self._validate_scanning_config(policy_data['vulnerability_scanning']))
        
        # Validate global settings consistency
        if 'global_settings' in policy_data:
            errors.extend(self._validate_global_settings(policy_data['global_settings']))
        
        return errors
    
    def _validate_cvss_thresholds(self, thresholds: Dict[str, Any]) -> List[str]:
        """Validate CVSS threshold configuration."""
        errors = []
        
        # Handle None thresholds
        if not thresholds or not isinstance(thresholds, dict):
            return errors
        
        # Extract threshold values for ordering validation
        threshold_values = {}
        for severity, config in thresholds.items():
            if 'min_score' in config:
                threshold_values[severity] = float(config['min_score'])
        
        # Validate threshold ordering (critical > high > medium > low)
        severity_order = ['critical', 'high', 'medium', 'low', 'none']
        
        for i in range(len(severity_order) - 1):
            current = severity_order[i]
            next_severity = severity_order[i + 1]
            
            if current in threshold_values and next_severity in threshold_values:
                if threshold_values[current] <= threshold_values[next_severity]:
                    errors.append(f"{current.title()} threshold ({threshold_values[current]}) must be higher than {next_severity} threshold ({threshold_values[next_severity]})")
        
        # Validate threshold ranges don't overlap inappropriately
        for severity, config in thresholds.items():
            if 'min_score' in config and 'max_score' in config:
                min_score = float(config['min_score'])
                max_score = float(config['max_score'])
                
                if min_score > max_score:
                    errors.append(f"{severity} threshold min_score ({min_score}) cannot be greater than max_score ({max_score})")
        
        # Validate action configurations
        for severity, config in thresholds.items():
            if 'actions' in config:
                for i, action in enumerate(config['actions']):
                    if 'type' not in action:
                        errors.append(f"{severity} threshold action {i}: 'type' is required")
                    elif action['type'] not in [t.value for t in ActionType]:
                        errors.append(f"{severity} threshold action {i}: invalid action type '{action['type']}'")
        
        return errors
    
    def _validate_reputation_config(self, reputation_config: Dict[str, Any]) -> List[str]:
        """Validate package reputation configuration."""
        errors = []
        
        # Handle None reputation_config
        if not reputation_config or not isinstance(reputation_config, dict):
            return errors
        
        if 'scoring_weights' in reputation_config:
            weights = reputation_config['scoring_weights']
            
            # Validate weight values
            for metric, weight in weights.items():
                if not isinstance(weight, (int, float)):
                    errors.append(f"Reputation scoring weight '{metric}' must be a number, got {type(weight)}")
                elif weight < 0 or weight > 1:
                    errors.append(f"Reputation scoring weight '{metric}' must be between 0 and 1, got {weight}")
            
            # Validate total weight sum
            if isinstance(weights, dict):
                total_weight = sum(float(w) for w in weights.values() if isinstance(w, (int, float)))
                if abs(total_weight - 1.0) > 0.001:
                    errors.append(f"Reputation scoring weights must sum to 1.0, got {total_weight}")
        
        return errors
    
    def _validate_scanning_config(self, scanning_config: Dict[str, Any]) -> List[str]:
        """Validate vulnerability scanning configuration."""
        errors = []
        
        # Handle None scanning_config
        if not scanning_config or not isinstance(scanning_config, dict):
            return errors
        
        # Validate parallel_scans value
        if 'parallel_scans' in scanning_config:
            parallel_scans = scanning_config['parallel_scans']
            if not isinstance(parallel_scans, int) or parallel_scans < 1:
                errors.append(f"parallel_scans must be a positive integer, got {parallel_scans}")
            elif parallel_scans > 20:  # Reasonable upper limit
                errors.append(f"parallel_scans should not exceed 20 for performance reasons, got {parallel_scans}")
        
        # Validate timeout format
        if 'timeout' in scanning_config:
            timeout = scanning_config['timeout']
            if not self._is_valid_timeout_format(timeout):
                errors.append(f"Invalid timeout format '{timeout}'. Use format like '30m', '1h', '90s'")
        
        # Validate frequency format
        if 'frequency' in scanning_config:
            frequency = scanning_config['frequency']
            if not self._is_valid_frequency_format(frequency):
                errors.append(f"Invalid frequency format '{frequency}'. Use 'daily', 'weekly', 'monthly', or cron format")
        
        return errors
    
    def _validate_global_settings(self, global_settings: Dict[str, Any]) -> List[str]:
        """Validate global settings configuration."""
        errors = []
        
        # Handle None global_settings
        if not global_settings or not isinstance(global_settings, dict):
            return errors
        
        # Validate enforcement_mode
        if 'enforcement_mode' in global_settings:
            mode = global_settings['enforcement_mode']
            valid_modes = [m.value for m in EnforcementMode]
            if mode not in valid_modes:
                errors.append(f"Invalid enforcement_mode '{mode}'. Valid options: {valid_modes}")
        
        # Validate notification_channels
        if 'notification_channels' in global_settings:
            channels = global_settings['notification_channels']
            if isinstance(channels, list):
                valid_channels = [c.value for c in NotificationChannel]
                for channel in channels:
                    if channel not in valid_channels:
                        errors.append(f"Invalid notification channel '{channel}'. Valid options: {valid_channels}")
        
        # Validate max_concurrent_scans
        if 'max_concurrent_scans' in global_settings:
            max_scans = global_settings['max_concurrent_scans']
            if not isinstance(max_scans, int) or max_scans < 1:
                errors.append("max_concurrent_scans must be a positive integer")
            elif max_scans > 50:  # Enterprise reasonable limit
                errors.append("max_concurrent_scans should not exceed 50 for performance and resource management")
        
        return errors
    
    def _validate_policy_object_rules(self, policy_config: SecurityPolicyConfig) -> List[str]:
        """Validate additional rules for SecurityPolicyConfig objects."""
        errors = []
        
        # Validate environment-specific rules
        environment = policy_config.metadata.environment
        
        if environment == "production":
            # Production-specific validation
            if policy_config.global_settings.enforcement_mode == EnforcementMode.DISABLED:
                errors.append("Production environment cannot have enforcement_mode set to 'disabled'")
            
            # Ensure critical actions are properly configured
            if 'critical' in policy_config.cvss_thresholds:
                critical_threshold = policy_config.cvss_thresholds['critical']
                has_blocking_action = any(
                    action.type in [ActionType.BLOCK_BUILD, ActionType.REQUIRE_APPROVAL]
                    for action in critical_threshold.actions
                )
                if not has_blocking_action:
                    errors.append("Production environment must have blocking actions for critical vulnerabilities")
        
        elif environment == "development":
            # Development-specific recommendations (warnings, not errors)
            self.logger.info("Development environment detected - using permissive validation")
        
        return errors
    
    def _is_valid_timeout_format(self, timeout: str) -> bool:
        """Validate timeout string format (e.g., '30m', '1h', '90s')."""
        import re
        pattern = r'^\d+[smhd]$'
        return bool(re.match(pattern, timeout))
    
    def _is_valid_frequency_format(self, frequency: str) -> bool:
        """Validate frequency string format."""
        if not isinstance(frequency, str):
            return False
            
        valid_frequencies = ['daily', 'weekly', 'monthly', 'hourly']
        if frequency in valid_frequencies:
            return True
        
        # Check for cron format (basic validation)
        import re
        # More flexible cron pattern that allows */5 format
        cron_pattern = r'^(\*|\d+|\*/\d+|\d+-\d+)\s+(\*|\d+|\*/\d+|\d+-\d+)\s+(\*|\d+|\*/\d+|\d+-\d+)\s+(\*|\d+|\*/\d+|\d+-\d+)\s+(\*|\d+|\*/\d+|\d+-\d+)$'
        return bool(re.match(cron_pattern, frequency))
    
    def generate_validation_report(self, policy_path: str) -> Dict[str, Any]:
        """
        Generate comprehensive validation report for policy file.
        
        Args:
            policy_path: Path to policy file
            
        Returns:
            Validation report with errors, warnings, and metadata
        """
        report = {
            'policy_file': policy_path,
            'validation_timestamp': yaml.safe_load('2024-01-01T00:00:00Z'),  # Would use datetime.now() in real implementation
            'is_valid': False,
            'errors': [],
            'warnings': [],
            'metadata': {}
        }
        
        try:
            # Load policy file
            with open(policy_path, 'r') as f:
                policy_data = yaml.safe_load(f)
            
            # Extract metadata
            if 'metadata' in policy_data:
                report['metadata'] = policy_data['metadata']
            
            # Validate policy
            errors = self.validate_policy_dict(policy_data)
            report['errors'] = errors
            report['is_valid'] = len(errors) == 0
            
            # Generate warnings for best practices
            warnings = self._generate_best_practice_warnings(policy_data)
            report['warnings'] = warnings
            
        except Exception as e:
            report['errors'] = [f"Failed to process policy file: {e}"]
        
        return report
    
    def _generate_best_practice_warnings(self, policy_data: Dict[str, Any]) -> List[str]:
        """Generate warnings for policy best practices."""
        warnings = []
        
        # Handle None policy_data
        if not policy_data or not isinstance(policy_data, dict):
            return warnings
        
        # Check for missing recommended configurations
        if 'typosquatting_detection' not in policy_data:
            warnings.append("Typosquatting detection is not configured - recommended for comprehensive security")
        
        if 'package_reputation' not in policy_data:
            warnings.append("Package reputation scoring is not configured - recommended for supply chain security")
        
        # Check for overly permissive settings in production
        metadata = policy_data.get('metadata', {})
        global_settings = policy_data.get('global_settings', {})
        
        if isinstance(metadata, dict) and metadata.get('environment') == 'production':
            if isinstance(global_settings, dict) and global_settings.get('enforcement_mode') == 'audit':
                warnings.append("Production environment using 'audit' mode - consider 'enforce' for stronger security")
        
        return warnings