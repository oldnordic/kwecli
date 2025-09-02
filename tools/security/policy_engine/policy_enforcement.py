"""
Policy Enforcement Engine for KWE CLI Security Policies.

This module provides comprehensive policy enforcement with violation detection,
action execution, and audit trails following enterprise security patterns.
"""

import logging
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from decimal import Decimal

from .models import (
    SecurityPolicyConfig,
    PolicyViolation,
    CVSSThreshold,
    PolicyAction,
    ActionType,
    SeverityLevel,
    EnforcementMode
)


@dataclass
class ViolationContext:
    """Context information for policy violation evaluation."""
    package_name: str
    package_version: str
    ecosystem: str
    vulnerability_data: Optional[Dict[str, Any]] = None
    cvss_score: Optional[Decimal] = None
    typosquatting_score: Optional[float] = None
    reputation_score: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EnforcementResult:
    """Result of policy enforcement operation."""
    allowed: bool
    violations: List[PolicyViolation] = field(default_factory=list)
    actions_executed: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def has_blocking_violations(self) -> bool:
        """Check if any violations have blocking actions."""
        blocking_actions = {ActionType.BLOCK_BUILD, ActionType.REQUIRE_APPROVAL}
        for violation in self.violations:
            for action in violation.actions:
                if action.type in blocking_actions:
                    return True
        return False


class PolicyEnforcementEngine:
    """
    Enterprise-grade policy enforcement engine.
    
    Evaluates security policies against package and vulnerability data,
    detects violations, and executes configured actions with audit trails.
    """
    
    def __init__(self, policy_config: SecurityPolicyConfig):
        """Initialize enforcement engine with policy configuration."""
        self.policy_config = policy_config
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.audit_trail: List[Dict[str, Any]] = []
        
        # Validate policy configuration
        validation_errors = self.policy_config.validate()
        if validation_errors:
            raise ValueError(f"Invalid policy configuration: {validation_errors}")
    
    def enforce_policy(self, context: ViolationContext) -> EnforcementResult:
        """
        Enforce security policies against the provided context.
        
        Args:
            context: Violation context with package and security data
            
        Returns:
            EnforcementResult with violations, actions, and decisions
        """
        self.logger.info(f"Enforcing policy for {context.package_name}@{context.package_version}")
        
        result = EnforcementResult(allowed=True)
        
        try:
            # Check if enforcement is disabled
            if self.policy_config.global_settings.enforcement_mode == EnforcementMode.DISABLED:
                self.logger.info("Policy enforcement is disabled")
                result.warnings.append("Policy enforcement is disabled")
                return result
            
            # Evaluate all policy rules
            violations = []
            
            # CVSS threshold violations
            if context.cvss_score is not None:
                cvss_violations = self._evaluate_cvss_thresholds(context)
                violations.extend(cvss_violations)
            
            # Typosquatting violations
            if context.typosquatting_score is not None:
                typo_violations = self._evaluate_typosquatting_policy(context)
                violations.extend(typo_violations)
            
            # Package reputation violations
            if context.reputation_score is not None:
                reputation_violations = self._evaluate_reputation_policy(context)
                violations.extend(reputation_violations)
            
            # Store violations in result
            result.violations = violations
            
            # Execute actions for violations
            if violations:
                actions_executed = self._execute_violation_actions(violations)
                result.actions_executed = actions_executed
                
                # Determine if package should be blocked
                if self.policy_config.global_settings.enforcement_mode == EnforcementMode.ENFORCE:
                    result.allowed = not result.has_blocking_violations()
                else:
                    # Audit mode - log violations but allow
                    result.allowed = True
                    result.warnings.append(f"Found {len(violations)} policy violations (audit mode)")
            
            # Add to audit trail
            self._add_to_audit_trail(context, result)
            
        except Exception as e:
            self.logger.error(f"Policy enforcement error for {context.package_name}: {e}")
            result.warnings.append(f"Policy enforcement error: {e}")
            
            # Fail-fast behavior
            if self.policy_config.global_settings.fail_fast:
                result.allowed = False
        
        return result
    
    def _evaluate_cvss_thresholds(self, context: ViolationContext) -> List[PolicyViolation]:
        """Evaluate CVSS score against configured thresholds."""
        violations = []
        cvss_score = context.cvss_score
        
        if not cvss_score:
            return violations
        
        # Check each threshold configuration
        for severity_name, threshold in self.policy_config.cvss_thresholds.items():
            if threshold.min_score <= cvss_score <= threshold.max_score:
                violation = PolicyViolation(
                    rule_name=f"cvss_{severity_name}_threshold",
                    severity=threshold.severity,
                    description=f"CVSS score {cvss_score} exceeds {severity_name} threshold ({threshold.min_score}-{threshold.max_score})",
                    actions=threshold.actions.copy(),
                    metadata={
                        "cvss_score": float(cvss_score),
                        "threshold_min": float(threshold.min_score),
                        "threshold_max": float(threshold.max_score),
                        "package_name": context.package_name,
                        "package_version": context.package_version,
                        "ecosystem": context.ecosystem
                    }
                )
                violations.append(violation)
                
                self.logger.warning(
                    f"CVSS threshold violation: {context.package_name}@{context.package_version} "
                    f"score {cvss_score} exceeds {severity_name} threshold"
                )
        
        return violations
    
    def _evaluate_typosquatting_policy(self, context: ViolationContext) -> List[PolicyViolation]:
        """Evaluate typosquatting detection results against policy."""
        violations = []
        typo_score = context.typosquatting_score
        
        if not self.policy_config.typosquatting_detection.enabled:
            return violations
        
        if typo_score is None:
            return violations
        
        # Check if score exceeds high-risk threshold
        high_risk_threshold = 0.8  # Default threshold
        if hasattr(self.policy_config.typosquatting_detection, 'thresholds'):
            thresholds = getattr(self.policy_config.typosquatting_detection.thresholds, 'high_risk', {})
            if isinstance(thresholds, dict):
                high_risk_threshold = thresholds.get('score', 0.8)
        
        if typo_score >= high_risk_threshold:
            actions = [
                PolicyAction(type=ActionType.BLOCK_BUILD, immediate=True),
                PolicyAction(type=ActionType.CREATE_INCIDENT, priority="P1")
            ]
            
            violation = PolicyViolation(
                rule_name="typosquatting_detection",
                severity=SeverityLevel.CRITICAL,
                description=f"High typosquatting risk detected for {context.package_name} (score: {typo_score})",
                actions=actions,
                metadata={
                    "typosquatting_score": typo_score,
                    "threshold": high_risk_threshold,
                    "package_name": context.package_name,
                    "package_version": context.package_version,
                    "ecosystem": context.ecosystem
                }
            )
            violations.append(violation)
        
        return violations
    
    def _evaluate_reputation_policy(self, context: ViolationContext) -> List[PolicyViolation]:
        """Evaluate package reputation against policy thresholds."""
        violations = []
        reputation_score = context.reputation_score
        
        if not self.policy_config.package_reputation.enabled:
            return violations
        
        if reputation_score is None:
            return violations
        
        # Check if score is below minimum threshold
        min_threshold = 0.6  # Default minimum reputation score
        if hasattr(self.policy_config.package_reputation, 'thresholds'):
            thresholds = getattr(self.policy_config.package_reputation.thresholds, 'minimum_score', 0.6)
            if isinstance(thresholds, (int, float)):
                min_threshold = float(thresholds)
        
        if reputation_score < min_threshold:
            # Severity based on how far below threshold
            if reputation_score < 0.3:
                severity = SeverityLevel.CRITICAL
                actions = [PolicyAction(type=ActionType.BLOCK_BUILD)]
            elif reputation_score < 0.5:
                severity = SeverityLevel.HIGH
                actions = [PolicyAction(type=ActionType.REQUIRE_APPROVAL)]
            else:
                severity = SeverityLevel.MEDIUM
                actions = [PolicyAction(type=ActionType.NOTIFY)]
            
            violation = PolicyViolation(
                rule_name="package_reputation_threshold",
                severity=severity,
                description=f"Package {context.package_name} reputation score {reputation_score} below threshold {min_threshold}",
                actions=actions,
                metadata={
                    "reputation_score": reputation_score,
                    "threshold": min_threshold,
                    "package_name": context.package_name,
                    "package_version": context.package_version,
                    "ecosystem": context.ecosystem
                }
            )
            violations.append(violation)
        
        return violations
    
    def _execute_violation_actions(self, violations: List[PolicyViolation]) -> List[str]:
        """Execute actions for policy violations."""
        executed_actions = []
        
        for violation in violations:
            for action in violation.actions:
                try:
                    action_result = self._execute_single_action(action, violation)
                    executed_actions.append(action_result)
                except Exception as e:
                    self.logger.error(f"Failed to execute action {action.type.value}: {e}")
                    executed_actions.append(f"FAILED: {action.type.value} - {e}")
        
        return executed_actions
    
    def _execute_single_action(self, action: PolicyAction, violation: PolicyViolation) -> str:
        """Execute a single policy action."""
        action_type = action.type
        
        if action_type == ActionType.BLOCK_BUILD:
            return self._execute_block_build_action(action, violation)
        elif action_type == ActionType.REQUIRE_APPROVAL:
            return self._execute_require_approval_action(action, violation)
        elif action_type == ActionType.CREATE_TICKET:
            return self._execute_create_ticket_action(action, violation)
        elif action_type == ActionType.CREATE_INCIDENT:
            return self._execute_create_incident_action(action, violation)
        elif action_type == ActionType.NOTIFY:
            return self._execute_notify_action(action, violation)
        elif action_type == ActionType.LOG_ONLY:
            return self._execute_log_only_action(action, violation)
        elif action_type == ActionType.AUTO_RESOLVE:
            return self._execute_auto_resolve_action(action, violation)
        else:
            raise ValueError(f"Unknown action type: {action_type}")
    
    def _execute_block_build_action(self, action: PolicyAction, violation: PolicyViolation) -> str:
        """Execute block build action."""
        self.logger.critical(f"BUILD BLOCKED: {violation.description}")
        return f"BLOCKED: {violation.rule_name} - {violation.description}"
    
    def _execute_require_approval_action(self, action: PolicyAction, violation: PolicyViolation) -> str:
        """Execute require approval action."""
        assignee = action.assignee or "security-team"
        timeout = action.timeout or "24h"
        
        self.logger.warning(f"APPROVAL REQUIRED: {violation.description} (assignee: {assignee}, timeout: {timeout})")
        return f"APPROVAL_REQUIRED: {violation.rule_name} - assignee: {assignee}, timeout: {timeout}"
    
    def _execute_create_ticket_action(self, action: PolicyAction, violation: PolicyViolation) -> str:
        """Execute create ticket action."""
        priority = action.priority or "P2"
        assignee = action.assignee or "security-team"
        
        self.logger.info(f"TICKET CREATED: {violation.description} (priority: {priority}, assignee: {assignee})")
        return f"TICKET_CREATED: {violation.rule_name} - priority: {priority}, assignee: {assignee}"
    
    def _execute_create_incident_action(self, action: PolicyAction, violation: PolicyViolation) -> str:
        """Execute create incident action."""
        priority = action.priority or "P1"
        
        self.logger.critical(f"INCIDENT CREATED: {violation.description} (priority: {priority})")
        return f"INCIDENT_CREATED: {violation.rule_name} - priority: {priority}"
    
    def _execute_notify_action(self, action: PolicyAction, violation: PolicyViolation) -> str:
        """Execute notification action."""
        channels = action.channels if action.channels else ["email"]
        
        self.logger.info(f"NOTIFICATION SENT: {violation.description} (channels: {channels})")
        return f"NOTIFIED: {violation.rule_name} - channels: {channels}"
    
    def _execute_log_only_action(self, action: PolicyAction, violation: PolicyViolation) -> str:
        """Execute log only action."""
        self.logger.info(f"POLICY VIOLATION LOGGED: {violation.description}")
        return f"LOGGED: {violation.rule_name} - {violation.description}"
    
    def _execute_auto_resolve_action(self, action: PolicyAction, violation: PolicyViolation) -> str:
        """Execute auto resolve action."""
        self.logger.info(f"AUTO RESOLVE ATTEMPTED: {violation.description}")
        return f"AUTO_RESOLVE: {violation.rule_name} - attempted automatic resolution"
    
    def _add_to_audit_trail(self, context: ViolationContext, result: EnforcementResult):
        """Add enforcement result to audit trail."""
        audit_entry = {
            "timestamp": datetime.now().isoformat(),
            "package": f"{context.package_name}@{context.package_version}",
            "ecosystem": context.ecosystem,
            "enforcement_mode": self.policy_config.global_settings.enforcement_mode.value,
            "allowed": result.allowed,
            "violations_count": len(result.violations),
            "violations": [v.to_dict() for v in result.violations],
            "actions_executed": result.actions_executed,
            "warnings": result.warnings,
            "metadata": {
                "cvss_score": float(context.cvss_score) if context.cvss_score else None,
                "typosquatting_score": context.typosquatting_score,
                "reputation_score": context.reputation_score,
                **context.metadata
            }
        }
        
        self.audit_trail.append(audit_entry)
        
        # Limit audit trail size
        if len(self.audit_trail) > 1000:
            self.audit_trail = self.audit_trail[-1000:]
    
    def get_audit_trail(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get audit trail entries."""
        if limit:
            return self.audit_trail[-limit:]
        return self.audit_trail.copy()
    
    def get_violation_summary(self) -> Dict[str, Any]:
        """Get summary statistics of policy violations."""
        if not self.audit_trail:
            return {"total_packages": 0, "violations": 0, "blocked": 0, "by_severity": {}}
        
        total_packages = len(self.audit_trail)
        total_violations = sum(entry["violations_count"] for entry in self.audit_trail)
        blocked_packages = sum(1 for entry in self.audit_trail if not entry["allowed"])
        
        severity_counts = {}
        for entry in self.audit_trail:
            for violation in entry["violations"]:
                severity = violation["severity"]
                severity_counts[severity] = severity_counts.get(severity, 0) + 1
        
        return {
            "total_packages": total_packages,
            "total_violations": total_violations,
            "blocked_packages": blocked_packages,
            "allowed_packages": total_packages - blocked_packages,
            "by_severity": severity_counts,
            "enforcement_mode": self.policy_config.global_settings.enforcement_mode.value
        }
    
    def clear_audit_trail(self):
        """Clear the audit trail (for testing or maintenance)."""
        self.audit_trail.clear()
        self.logger.info("Audit trail cleared")