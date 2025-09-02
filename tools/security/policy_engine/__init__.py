"""
Security Policy Engine for KWE CLI - Enterprise-grade YAML-based security governance.

This module provides comprehensive security policy configuration and enforcement
for vulnerability assessment workflows, following NIST CSF 2.0 and ISO 27001 patterns.
"""

__version__ = "1.0.0"
__author__ = "KWE CLI Security Team"

from .models import (
    ActionType,
    SeverityLevel,
    EnforcementMode,
    NotificationChannel,
    PolicyAction,
    CVSSThreshold,
    TyposquattingConfig,
    ReputationConfig,
    ScanningConfig,
    IntegrationConfig,
    PolicyViolation,
    PolicyMetadata,
    GlobalSettings,
    SecurityPolicyConfig
)

from .policy_enforcement import (
    PolicyEnforcementEngine,
    ViolationContext,
    EnforcementResult
)

from .config_manager import (
    ConfigurationManager,
    ConfigPath
)

__all__ = [
    "ActionType",
    "SeverityLevel", 
    "EnforcementMode",
    "NotificationChannel",
    "PolicyAction",
    "CVSSThreshold",
    "TyposquattingConfig",
    "ReputationConfig",
    "ScanningConfig",
    "IntegrationConfig",
    "PolicyViolation",
    "PolicyMetadata",
    "GlobalSettings",
    "SecurityPolicyConfig",
    "PolicyEnforcementEngine",
    "ViolationContext",
    "EnforcementResult",
    "ConfigurationManager",
    "ConfigPath"
]