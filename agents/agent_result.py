#!/usr/bin/env python3
"""
Agent Result and Status Infrastructure

This module provides the result data structures and status enums used by all agents.
Includes AgentResult class for standardized outputs and status/expertise enums.
"""

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Any, Optional


class AgentStatus(Enum):
    """Status of an agent."""
    IDLE = "idle"
    BUSY = "busy"
    ERROR = "error"
    OFFLINE = "offline"
    FAILED = "failed"
    COMPLETED = "completed"
    INITIALIZING = "initializing"
    SHUTDOWN = "shutdown"


class AgentExpertise(Enum):
    """Areas of expertise for agents."""
    GENERAL = "general"
    AI_ML = "ai_ml"
    AI_ENGINEERING = "ai_engineering"
    CODE_GENERATION = "code_generation"
    CODE_ANALYSIS = "code_analysis"
    BACKEND_ARCHITECTURE = "backend_architecture"
    FRONTEND_DEVELOPMENT = "frontend_development"
    DEVOPS = "devops"
    MOBILE_DEVELOPMENT = "mobile_development"
    TESTING = "testing"
    UX_RESEARCH = "ux_research"
    UI_DESIGN = "ui_design"
    PRODUCT_MANAGEMENT = "product_management"
    MARKETING = "marketing"
    CONTENT_CREATION = "content_creation"
    ANALYTICS = "analytics"
    INFRASTRUCTURE = "infrastructure"
    SUPPORT = "support"
    SYSTEM_INTEGRATION = "system_integration"
    FILE_OPERATIONS = "file_operations"


@dataclass
class AgentResult:
    """Standardized result format for all agent outputs."""
    success: bool
    output: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    error_message: Optional[str] = None
    quality_score: int = 0
    recommendations: List[str] = field(default_factory=list)
    execution_time: Optional[float] = None


# Convenience functions for creating agent results
def create_agent_result(
    success: bool,
    output: str,
    metadata: Optional[Dict[str, Any]] = None,
    error_message: Optional[str] = None,
    quality_score: int = 0,
    recommendations: Optional[List[str]] = None
) -> AgentResult:
    """Create an AgentResult with default values.
    
    Args:
        success: Whether the task was successful
        output: Output content
        metadata: Additional metadata
        error_message: Error message if failed
        quality_score: Quality score (0-100)
        recommendations: List of recommendations
        
    Returns:
        AgentResult instance
    """
    return AgentResult(
        success=success,
        output=output,
        metadata=metadata or {},
        error_message=error_message,
        quality_score=quality_score,
        recommendations=recommendations or []
    )


def create_success_result(
    output: str,
    quality_score: int = 100,
    metadata: Optional[Dict[str, Any]] = None,
    recommendations: Optional[List[str]] = None
) -> AgentResult:
    """Create a successful AgentResult.
    
    Args:
        output: Output content
        quality_score: Quality score (default 100)
        metadata: Additional metadata
        recommendations: List of recommendations
        
    Returns:
        Successful AgentResult instance
    """
    return create_agent_result(
        success=True,
        output=output,
        quality_score=quality_score,
        metadata=metadata,
        recommendations=recommendations
    )


def create_error_result(
    error_message: str,
    output: str = "",
    metadata: Optional[Dict[str, Any]] = None
) -> AgentResult:
    """Create a failed AgentResult.
    
    Args:
        error_message: Error description
        output: Any partial output (default empty)
        metadata: Additional metadata
        
    Returns:
        Failed AgentResult instance
    """
    return create_agent_result(
        success=False,
        output=output,
        error_message=error_message,
        quality_score=0,
        metadata=metadata
    )