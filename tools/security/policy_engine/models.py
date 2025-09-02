"""
Data models for security policy configuration system.

Defines the core data structures for representing security policies,
violations, and configuration following enterprise security patterns.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, Any, List, Optional, Union
from decimal import Decimal


class ActionType(Enum):
    """Security policy action types for violation response."""
    BLOCK_BUILD = "block_build"
    REQUIRE_APPROVAL = "require_approval"
    CREATE_TICKET = "create_ticket"
    CREATE_INCIDENT = "create_incident"
    NOTIFY = "notify"
    LOG_ONLY = "log_only"
    AUTO_RESOLVE = "auto_resolve"


class SeverityLevel(Enum):
    """Standard severity levels for security policies."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    NONE = "none"


class EnforcementMode(Enum):
    """Policy enforcement modes."""
    AUDIT = "audit"          # Log violations, don't block
    ENFORCE = "enforce"      # Block on violations
    DISABLED = "disabled"    # Policy checking disabled


class NotificationChannel(Enum):
    """Available notification channels."""
    EMAIL = "email"
    SLACK = "slack"
    TEAMS = "teams"
    WEBHOOK = "webhook"
    PAGERDUTY = "pagerduty"


@dataclass
class PolicyAction:
    """Represents a single policy enforcement action."""
    type: ActionType
    immediate: bool = False
    priority: Optional[str] = None
    assignee: Optional[str] = None
    timeout: Optional[str] = None
    channels: List[NotificationChannel] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CVSSThreshold:
    """CVSS score threshold configuration."""
    min_score: Decimal
    max_score: Decimal
    actions: List[PolicyAction]
    severity: SeverityLevel
    
    def __post_init__(self):
        """Validate threshold configuration."""
        if self.min_score < 0 or self.min_score > 10:
            raise ValueError(f"Invalid min_score: {self.min_score}")
        if self.max_score < 0 or self.max_score > 10:
            raise ValueError(f"Invalid max_score: {self.max_score}")
        if self.min_score > self.max_score:
            raise ValueError(f"min_score ({self.min_score}) > max_score ({self.max_score})")


@dataclass
class TyposquattingConfig:
    """Typosquatting detection configuration."""
    enabled: bool = True
    algorithms: Dict[str, Dict[str, Union[bool, float, int]]] = field(default_factory=dict)
    ecosystem_rules: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    thresholds: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    allowlists: Dict[str, List[str]] = field(default_factory=dict)
    blocklists: Dict[str, List[str]] = field(default_factory=dict)


@dataclass
class ReputationConfig:
    """Package reputation scoring configuration."""
    enabled: bool = True
    scoring_weights: Dict[str, float] = field(default_factory=dict)
    thresholds: Dict[str, float] = field(default_factory=dict)
    author_metrics: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    package_metrics: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    github_metrics: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    security_analysis: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate reputation configuration."""
        if self.scoring_weights:
            total_weight = sum(self.scoring_weights.values())
            if abs(total_weight - 1.0) > 0.001:
                raise ValueError(f"Scoring weights must sum to 1.0, got {total_weight}")


@dataclass
class ScanningConfig:
    """Vulnerability scanning configuration."""
    enabled: bool = True
    frequency: str = "daily"
    timeout: str = "30m"
    parallel_scans: int = 3
    retry_attempts: int = 2
    ecosystems: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    scanners: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    exclusions: Dict[str, List[str]] = field(default_factory=dict)


@dataclass
class IntegrationConfig:
    """Security tool integration configuration."""
    osv_dev: Dict[str, Any] = field(default_factory=dict)
    nvd: Dict[str, Any] = field(default_factory=dict)
    github_advisory: Dict[str, Any] = field(default_factory=dict)
    custom_feeds: List[Dict[str, Any]] = field(default_factory=list)
    sync_settings: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PolicyViolation:
    """Represents a security policy violation."""
    rule_name: str
    severity: SeverityLevel
    description: str
    actions: List[PolicyAction]
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    violation_id: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert violation to dictionary format."""
        return {
            'rule_name': self.rule_name,
            'severity': self.severity.value,
            'description': self.description,
            'actions': [self._action_to_dict(action) for action in self.actions],
            'metadata': self.metadata,
            'timestamp': self.timestamp.isoformat(),
            'violation_id': self.violation_id
        }
    
    def _action_to_dict(self, action: PolicyAction) -> Dict[str, Any]:
        """Convert PolicyAction to dictionary format."""
        return {
            'type': action.type.value,
            'immediate': action.immediate,
            'priority': action.priority,
            'assignee': action.assignee,
            'timeout': action.timeout,
            'channels': [channel.value if hasattr(channel, 'value') else str(channel) for channel in action.channels],
            'metadata': action.metadata
        }


@dataclass
class PolicyMetadata:
    """Policy configuration metadata."""
    version: str
    environment: str
    created_by: str
    last_modified: datetime
    schema_version: str = "1.0.0"
    description: Optional[str] = None
    
    def __post_init__(self):
        """Validate metadata."""
        if not self.version or not self.environment:
            raise ValueError("Version and environment are required")


@dataclass
class GlobalSettings:
    """Global security policy settings."""
    enforcement_mode: EnforcementMode = EnforcementMode.AUDIT
    fail_fast: bool = False
    notification_channels: List[str] = field(default_factory=list)
    default_timeout: str = "1h"
    max_concurrent_scans: int = 5
    
    def __post_init__(self):
        """Validate global settings."""
        if self.max_concurrent_scans < 1:
            raise ValueError("max_concurrent_scans must be >= 1")


@dataclass
class SecurityPolicyConfig:
    """Complete security policy configuration."""
    metadata: PolicyMetadata
    global_settings: GlobalSettings
    cvss_thresholds: Dict[str, CVSSThreshold]
    vulnerability_scanning: ScanningConfig
    typosquatting_detection: TyposquattingConfig
    package_reputation: ReputationConfig
    security_integrations: IntegrationConfig
    
    def validate(self) -> List[str]:
        """Validate complete policy configuration."""
        errors = []
        
        # Validate CVSS threshold ordering
        if self.cvss_thresholds:
            critical = self.cvss_thresholds.get('critical')
            high = self.cvss_thresholds.get('high')
            medium = self.cvss_thresholds.get('medium')
            low = self.cvss_thresholds.get('low')
            
            if critical and high and critical.min_score <= high.min_score:
                errors.append("Critical threshold must be higher than high threshold")
            if high and medium and high.min_score <= medium.min_score:
                errors.append("High threshold must be higher than medium threshold")
            if medium and low and medium.min_score <= low.min_score:
                errors.append("Medium threshold must be higher than low threshold")
        
        return errors
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert policy configuration to dictionary format."""
        return {
            'metadata': {
                'version': self.metadata.version,
                'environment': self.metadata.environment,
                'created_by': self.metadata.created_by,
                'last_modified': self.metadata.last_modified.isoformat(),
                'schema_version': self.metadata.schema_version,
                'description': self.metadata.description
            },
            'global_settings': {
                'enforcement_mode': self.global_settings.enforcement_mode.value,
                'fail_fast': self.global_settings.fail_fast,
                'notification_channels': self.global_settings.notification_channels,
                'default_timeout': self.global_settings.default_timeout,
                'max_concurrent_scans': self.global_settings.max_concurrent_scans
            },
            'cvss_thresholds': {
                name: {
                    'min_score': float(threshold.min_score),
                    'max_score': float(threshold.max_score),
                    'severity': threshold.severity.value,
                    'actions': [self._action_to_dict_static(action) for action in threshold.actions]
                }
                for name, threshold in self.cvss_thresholds.items()
            },
            'vulnerability_scanning': self.vulnerability_scanning.__dict__,
            'typosquatting_detection': self.typosquatting_detection.__dict__,
            'package_reputation': self.package_reputation.__dict__,
            'security_integrations': self.security_integrations.__dict__
        }
    
    def _action_to_dict_static(self, action: PolicyAction) -> Dict[str, Any]:
        """Convert PolicyAction to dictionary format (static version)."""
        return {
            'type': action.type.value,
            'immediate': action.immediate,
            'priority': action.priority,
            'assignee': action.assignee,
            'timeout': action.timeout,
            'channels': [channel.value if hasattr(channel, 'value') else str(channel) for channel in action.channels],
            'metadata': action.metadata
        }